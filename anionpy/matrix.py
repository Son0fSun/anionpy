"""anionpy.matrix -- composition wrapper for numpy's `matrix` (a deprecated,
always-2-D `ndarray` subclass).

SCOPE. `anionpy.ndarray` is a PyO3-backed Rust type and is NOT subclassable
from Python (`class Foo(anionpy.ndarray): ...` raises `TypeError: type
'anionpy.ndarray' is not an acceptable base type` -- measured live). Real
numpy's `matrix` gets its always-2-D, matmul-is-`*` behaviour entirely
through `ndarray` subclassing machinery (`__array_finalize__`,
`__array_priority__`, ...) that anionpy has no equivalent of and this file
may not add (no `.rs` edits in scope). So, exactly like `anionpy.ma.
MaskedArray` (see `anionpy/ma/core.py`), `matrix` here is a plain Python
class that HOLDS an `anionpy.ndarray` (`self._data`, always 2-D) rather than
IS one. This is a real, load-bearing difference from real numpy's `matrix`
(e.g. `isinstance(m, anionpy.ndarray)` is `False` here, `True` there) but it
is the only implementation strategy available under this task's edit scope,
and it is sufficient to reproduce matrix's VALUE/SHAPE/DTYPE contract, which
is what every differential case in `tests/differential/matrix_cases.py`
actually checks.

`matrix.__new__`'s `copy=False` path is DELIBERATELY NOT declared exact --
see `KNOWN-DIFFERENCES.md`'s 2026-08-07 entry. Real numpy's `copy=False`
construction from an existing `ndarray` shares the SAME underlying buffer
(mutating the source mutates the matrix); `anionpy.array()`/`anionpy.ndarray`
have no buffer-sharing construction path at all (verified live: mutating an
`anionpy.ndarray` built from a numpy array never touches the numpy source,
and there is no `frombuffer`/view-construction entry point in `anionpy` to
build one that does) -- so `copy=False` here always copies, silently
diverging from numpy's aliasing contract. Every OTHER item in this module is
tested and declared using ONLY `copy=True` (the default)-constructed
matrices, which sidesteps the gap entirely: A/A1/T/H/I/getA/.../__getitem__/
tolist/shape/dtype/ndim only look at a matrix's CURRENT data, not at how
it came to be aliased, so they are unaffected by which construction path
built it.

No numerical loops here: `__mul__`/`.I` dispatch to `anionpy`'s own
already-declared-exact `@`/`linalg.inv`/`linalg.pinv`; everything else is
attribute/shape plumbing over an already-constructed `anionpy.ndarray`.
"""
from __future__ import annotations

import ast as _ast
import warnings as _warnings

import anionpy as _ap

__all__ = ["matrix", "asmatrix"]

_DEPRECATION_MSG = (
    "the matrix subclass is not the recommended way to "
    "represent matrices or deal with linear algebra (see "
    "https://docs.scipy.org/doc/numpy/user/"
    "numpy-for-matlab-users.html). "
    "Please adjust your code to use regular ndarray."
)


def _convert_from_string(data: str):
    """Mirrors `numpy.matrixlib.defmatrix._convert_from_string` exactly:
    '[' / ']' stripped, ';' splits rows, ',' or whitespace splits columns,
    each scalar parsed with `ast.literal_eval`. Raises the same
    "Rows not the same size." `ValueError` numpy does on a ragged string.
    """
    for char in "[]":
        data = data.replace(char, "")
    rows = data.split(";")
    newdata = []
    ncols = None
    for count, row in enumerate(rows):
        trow = row.split(",")
        newrow = []
        for col in trow:
            for tok in col.split():
                newrow.append(_ast.literal_eval(tok))
        if count == 0:
            ncols = len(newrow)
        elif len(newrow) != ncols:
            raise ValueError("Rows not the same size.")
        newdata.append(newrow)
    return newdata


class matrix:
    """Composition-based stand-in for `numpy.matrix`. See module docstring
    for why this cannot be a real `anionpy.ndarray` subclass.
    """

    __slots__ = ("_data",)

    def __new__(cls, data, dtype=None, copy=True):
        _warnings.warn(_DEPRECATION_MSG, PendingDeprecationWarning, stacklevel=2)

        if isinstance(data, matrix):
            src = data._data
            if dtype is not None and str(src.dtype) != str(dtype):
                arr = src.astype(dtype)
            else:
                arr = src.copy() if copy else src
        elif isinstance(data, _ap.ndarray):
            if dtype is not None and str(data.dtype) != str(dtype):
                arr = data.astype(dtype)
            else:
                # `copy=False` SHOULD alias `data`'s buffer here (real numpy
                # does, via `data.view(cls)`); anionpy has no such
                # construction path (see module docstring) so this always
                # copies -- a known, documented divergence, never declared.
                arr = data.copy()
        else:
            if isinstance(data, str):
                data = _convert_from_string(data)
            arr = _ap.array(data, dtype=dtype)

        ndim = arr.ndim
        if ndim > 2:
            raise ValueError("matrix must be 2-dimensional")
        elif ndim == 0:
            arr = arr.reshape((1, 1))
        elif ndim == 1:
            arr = arr.reshape((1, arr.shape[0]))

        self = object.__new__(cls)
        self._data = arr
        return self

    def __init__(self, data, dtype=None, copy=True):
        # All real work happens in __new__ (mirrors numpy: `ndarray`
        # subclasses do their construction in __new__, __init__ is a no-op
        # for an already-built instance).
        pass

    # -- plain read-through delegation -----------------------------------
    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    def __array__(self, dtype=None, *, copy=None):
        """Hand this matrix's data to numpy as a plain `ndarray`.

        ADDED 2026-08-08 (task #66). Without this, `np.asarray(matrix(...))`
        did not merely return the wrong thing -- it RAISED, on the bare
        constructor, before any method was called:

            >>> np.asarray(anionpy.matrix([[1., 2.], [3., 4.]]))
            ValueError: setting an array element with a sequence. The
            requested array would exceed the maximum number of dimension
            of 64.

        Cause: this class is composition-based (see the module docstring),
        so `anionpy.ndarray` is not in its MRO and, lacking `__array__`,
        numpy fell back to the sequence protocol. That protocol never
        bottoms out here, because `m[0]` and `m[0][0]` are both `matrix`
        of shape `(1, 2)` -- forever. The infinite nesting is NOT a bug and
        must not be "fixed": real `np.matrix` indexes exactly the same way.
        It survives there only because it IS an `ndarray` subclass with a
        real buffer, so numpy reads the buffer and never walks the
        sequence protocol at all. Composition is what removed that escape
        hatch, and `__array__` is what puts it back.

        This was originally reported (`docs/EMPTY-AND-ORDER-AUDIT.md`,
        "family E") as a defect in `matrix.T`/`.H`/`.mT`/`.transpose`/
        `.getT`/`.getH`/`.swapaxes`/`.I`/`.getI`, later widened to ~29 more
        `matrix.*` methods. Those methods were never at fault; they were
        simply what the audit harness happened to call. One defect, in the
        base class, inherited by everything.

        Delegates to `anionpy.ndarray.__array__` (task #29), which already
        implements numpy 2.5.1's `(dtype=None, /, *, copy=None)` contract
        including the `copy=False`-needs-a-cast `ValueError`. No arithmetic
        and no conversion logic lives here; any cast happens in Rust via
        that method's `cast_to`.
        """
        return self._data.__array__(dtype, copy=copy)

    # -- matrix-specific derived views/copies -----------------------------
    @property
    def A(self):
        """`self` as a plain `anionpy.ndarray`. Equivalent to `asarray(self)`."""
        return self._data

    @property
    def A1(self):
        """`self`, raveled, as a plain `anionpy.ndarray`."""
        return self._data.ravel()

    @property
    def T(self):
        """Non-conjugated transpose, still a `matrix`."""
        out = object.__new__(matrix)
        out._data = self._data.transpose()
        return out

    @property
    def H(self):
        """Complex-conjugate transpose. Equal to `.T` for real dtypes."""
        out = object.__new__(matrix)
        d = self._data
        if str(d.dtype).startswith("complex"):
            out._data = d.transpose().conjugate()
        else:
            out._data = d.transpose()
        return out

    @property
    def I(self):  # noqa: E743 -- matches numpy's own (deprecated) name
        """Multiplicative inverse (square) / pseudo-inverse (non-square)."""
        m, n = self._data.shape
        func = _ap.linalg.inv if m == n else _ap.linalg.pinv
        out = object.__new__(matrix)
        out._data = func(self._data)
        return out

    # `getX` names are kept for numpy compatibility -- identical to the
    # property, called as a method.
    def getA(self):
        return self.A

    def getA1(self):
        return self.A1

    def getT(self):
        return self.T

    def getH(self):
        return self.H

    def getI(self):
        return self.I

    # -- indexing: always returns 2-D (row/col vectors stay matrices) -----
    def __getitem__(self, index):
        out = self._data[index]
        if not isinstance(out, _ap.ndarray):
            # scalar element (anionpy's own ndarray.__getitem__ contract,
            # not this class's to change)
            return out
        if out.ndim == 0:
            return out[()]
        if out.ndim == 1:
            sh = out.shape[0]
            try:
                n = len(index)
            except TypeError:
                n = 0
            if n > 1 and _is_scalar_index(index[1]):
                out = out.reshape((sh, 1))
            else:
                out = out.reshape((1, sh))
        wrapped = object.__new__(matrix)
        wrapped._data = out
        return wrapped

    # -- matrix multiplication (NOT elementwise) ---------------------------
    # A scalar `other` is dispatched to elementwise `*` (matches real numpy:
    # `matrix.__mul__` calls `N.dot(self, other)`, and `dot` against a
    # scalar operand IS elementwise multiply, not `matmul` -- confirmed
    # live, `anionpy`'s own `@` correctly REJECTS a 0-d/scalar operand with
    # `ValueError` where `numpy.dot` would not, so `@` cannot be used
    # unconditionally here).
    @staticmethod
    def _promote_2d(x):
        arr = x if isinstance(x, _ap.ndarray) else _ap.array(x)
        if arr.ndim == 0:
            return arr.reshape((1, 1))
        if arr.ndim == 1:
            return arr.reshape((1, arr.shape[0]))
        return arr

    @staticmethod
    def _wrap(result):
        out = object.__new__(matrix)
        if result.ndim == 2:
            out._data = result
        elif result.ndim == 1:
            out._data = result.reshape((1, result.shape[0]))
        else:
            out._data = result.reshape((1, 1))
        return out

    # numpy's `matrix.__mul__` routes through `N.dot`, NOT through `@`, and
    # the two raise DIFFERENT text on an inner-dimension mismatch. Measured
    # live 2026-08-07 on (2,1) reversed against (2,2):
    #
    #   numpy   matrix * matrix -> "shapes (2,1) and (2,2) not aligned:
    #                               1 (dim 1) != 2 (dim 0)"        (dot)
    #   numpy   ndarray @ ndarray -> "matmul: Input operand 1 has a mismatch
    #                                 in its core dimension 0, ..."  (matmul)
    #
    # anionpy's bare `ndarray @ ndarray` message is BYTE-IDENTICAL to numpy's
    # matmul message (also measured) -- so this is not a matmul defect. It is
    # that `matrix` must speak `dot`'s dialect while delegating to `@`.
    # anionpy has no `dot` yet (task #54), so the alignment check is raised
    # here, in the wrapper, before dispatch. No arithmetic happens in this
    # method: it compares two integers and then hands the multiply to Rust.
    @staticmethod
    def _check_aligned(lhs, rhs):
        if lhs.shape[1] != rhs.shape[0]:
            raise ValueError(
                "shapes ({},{}) and ({},{}) not aligned: "
                "{} (dim 1) != {} (dim 0)".format(
                    lhs.shape[0], lhs.shape[1], rhs.shape[0], rhs.shape[1],
                    lhs.shape[1], rhs.shape[0],
                )
            )

    def __mul__(self, other):
        if _ap.isscalar(other):
            return matrix._wrap(self._data * other)
        if isinstance(other, (matrix, _ap.ndarray, list, tuple)):
            rhs = other._data if isinstance(other, matrix) else matrix._promote_2d(other)
            matrix._check_aligned(self._data, rhs)
            return matrix._wrap(self._data @ rhs)
        return NotImplemented

    def __rmul__(self, other):
        if _ap.isscalar(other):
            return matrix._wrap(other * self._data)
        if isinstance(other, (matrix, _ap.ndarray, list, tuple)):
            lhs = other._data if isinstance(other, matrix) else matrix._promote_2d(other)
            matrix._check_aligned(lhs, self._data)
            return matrix._wrap(lhs @ self._data)
        return NotImplemented

    # -- plain conversions --------------------------------------------------
    def tolist(self):
        return self._data.tolist()

    def __repr__(self):
        return f"matrix({self._data.tolist()!r})"

    # -----------------------------------------------------------------
    # Extended inherited surface (2026-08-07 addition). Every method below
    # follows the SAME "unwrap operand -> call the already-proven
    # `anionpy.ndarray` primitive -> re-wrap the result as `matrix`" shape
    # already established above for `__mul__`/`T`/`H`/`I`/etc, and is
    # differential-tested (with matrix operands, checking `type(out)`, not
    # just values) in `tests/differential/matrix_cases.py`. Each is here
    # because real numpy's `matrix` behaves EXACTLY like the underlying
    # `ndarray` op for these -- elementwise arithmetic/comparison, unary
    # ops, and copy/shape ops that already return 2-D and just need
    # matrix-wrapping. Reductions are the one genuinely DIFFERENT case:
    # `axis=None` (numpy's default) returns a bare scalar (verified live,
    # `np.matrix([[1,2],[3,4]]).sum()` -> `numpy.int64(10)`, not a
    # 1x1 matrix) while `axis=0`/`axis=1` returns a matrix with the
    # reduced axis kept as a size-1 dimension (`keepdims=True`) -- both
    # branches are implemented explicitly below, not merely the
    # `axis=None` case.
    # -----------------------------------------------------------------

    # -- attribute passthroughs: these values are plain Python types
    #    (int/tuple), never array-wrapped, so no matrix-ness to preserve --
    @property
    def itemsize(self):
        return self._data.itemsize

    @property
    def nbytes(self):
        return self._data.nbytes

    @property
    def strides(self):
        return self._data.strides

    @property
    def size(self):
        return self._data.size

    @property
    def mT(self):
        """Matrix transpose -- identical to `.T` for a (necessarily 2-D)
        matrix (verified live against real numpy 2.5.1)."""
        return self.T

    # -- elementwise comparison dunders: return a `matrix` of `bool_`,
    #    NOT elementwise `*`/`**` semantics (only `__mul__`/`__pow__` are
    #    special-cased to matrix-multiply/matrix-power on real `matrix`) --
    def __eq__(self, other):
        return matrix._wrap(self._data == matrix._other_operand(other))

    def __ne__(self, other):
        return matrix._wrap(self._data != matrix._other_operand(other))

    def __lt__(self, other):
        return matrix._wrap(self._data < matrix._other_operand(other))

    def __le__(self, other):
        return matrix._wrap(self._data <= matrix._other_operand(other))

    def __gt__(self, other):
        return matrix._wrap(self._data > matrix._other_operand(other))

    def __ge__(self, other):
        return matrix._wrap(self._data >= matrix._other_operand(other))

    # -- elementwise arithmetic (NOT matrix-multiply -- only `*`/`**` are) --
    def __add__(self, other):
        return matrix._wrap(self._data + matrix._other_operand(other))

    def __radd__(self, other):
        return matrix._wrap(matrix._other_operand(other) + self._data)

    def __sub__(self, other):
        return matrix._wrap(self._data - matrix._other_operand(other))

    def __rsub__(self, other):
        return matrix._wrap(matrix._other_operand(other) - self._data)

    # -- unary --
    def __neg__(self):
        return matrix._wrap(-self._data)

    def __pos__(self):
        return matrix._wrap(+self._data)

    def __abs__(self):
        return matrix._wrap(abs(self._data))

    def __invert__(self):
        return matrix._wrap(~self._data)

    @staticmethod
    def _other_operand(other):
        """Unwrap a binary-op RHS the same way `__mul__`/`__rmul__` above
        already do: a `matrix` operand contributes its raw `_data`, a
        list/tuple is promoted to 2-D exactly like matrix construction
        does, anything else (a bare `anionpy.ndarray`, a Python/numpy
        scalar) is passed through unchanged -- `anionpy.ndarray`'s own
        binary dunders already accept scalars/ndarrays directly.
        """
        if isinstance(other, matrix):
            return other._data
        if isinstance(other, (list, tuple)):
            return matrix._promote_2d(other)
        return other

    # -- copy / shape ops: already 2-D (or reduce to 2-D per `_wrap`'s
    #    ndim-0/1 promotion rule), just need the result re-wrapped --
    def copy(self):
        return matrix._wrap(self._data.copy())

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy()

    def conj(self):
        return matrix._wrap(self._data.conj())

    def conjugate(self):
        return matrix._wrap(self._data.conjugate())

    def flatten(self, order="C"):
        return matrix._wrap(self._data.flatten(order))

    def ravel(self, order="C"):
        return matrix._wrap(self._data.ravel(order))

    def transpose(self, *axes):
        return matrix._wrap(self._data.transpose(*axes))

    # -- reductions: `axis=None` -> bare scalar (numpy scalar type, exactly
    #    what the underlying `anionpy.ndarray` reduction already returns);
    #    `axis=0`/`axis=1` -> matrix, reduced axis kept via `keepdims=True`
    #    (verified live against real numpy 2.5.1 for every op below) --
    def sum(self, axis=None, dtype=None, out=None):
        if axis is None:
            return self._data.sum(dtype=dtype)
        return matrix._wrap(self._data.sum(axis=axis, dtype=dtype, keepdims=True))

    def mean(self, axis=None, dtype=None, out=None):
        if axis is None:
            return self._data.mean(dtype=dtype)
        return matrix._wrap(self._data.mean(axis=axis, dtype=dtype, keepdims=True))

    def min(self, axis=None, out=None):
        if axis is None:
            return self._data.min()
        return matrix._wrap(self._data.min(axis=axis, keepdims=True))

    def max(self, axis=None, out=None):
        if axis is None:
            return self._data.max()
        return matrix._wrap(self._data.max(axis=axis, keepdims=True))

    def all(self, axis=None, out=None):
        if axis is None:
            return self._data.all()
        return matrix._wrap(self._data.all(axis=axis, keepdims=True))

    def any(self, axis=None, out=None):
        if axis is None:
            return self._data.any()
        return matrix._wrap(self._data.any(axis=axis, keepdims=True))

    # -- misc --
    def trace(self):
        """Unlike plain `ndarray.trace()` (bare scalar), real numpy's
        `matrix.trace()` returns a 1x1 `matrix` -- verified live
        (`type(np.matrix([[1,2],[3,4]]).trace()).__name__ == 'matrix'`)."""
        return matrix._wrap(matrix._promote_2d(self._data.trace()))

    def nonzero(self):
        return self._data.nonzero()

    def tobytes(self):
        return self._data.tobytes()

    def item(self):
        return self._data.item()

    # -----------------------------------------------------------------
    # 2026-08-07 (second) batch. Same unwrap/dispatch/rewrap discipline.
    # -----------------------------------------------------------------

    # -- elementwise arithmetic/bitwise dunders + reflected forms: numpy's
    #    `matrix` does NOT override any of these (only `__mul__`/`__pow__`
    #    get matrix-multiply/matrix-power semantics -- verified by reading
    #    `numpy/matrixlib/defmatrix.py`'s full `def __` listing), so they
    #    behave exactly like the underlying elementwise `ndarray` op,
    #    subclass-preserved -- measured live against real numpy 2.5.1 for
    #    every op below (matrix-matrix and matrix-scalar operands both
    #    return `matrix`). No numerical loops: each is one dispatch to
    #    anionpy's own already-declared-exact `ndarray` dunder. --
    def __floordiv__(self, other):
        return matrix._wrap(self._data // matrix._other_operand(other))

    def __rfloordiv__(self, other):
        return matrix._wrap(matrix._other_operand(other) // self._data)

    def __mod__(self, other):
        return matrix._wrap(self._data % matrix._other_operand(other))

    def __rmod__(self, other):
        return matrix._wrap(matrix._other_operand(other) % self._data)

    def __and__(self, other):
        return matrix._wrap(self._data & matrix._other_operand(other))

    def __rand__(self, other):
        return matrix._wrap(matrix._other_operand(other) & self._data)

    def __or__(self, other):
        return matrix._wrap(self._data | matrix._other_operand(other))

    def __ror__(self, other):
        return matrix._wrap(matrix._other_operand(other) | self._data)

    def __xor__(self, other):
        return matrix._wrap(self._data ^ matrix._other_operand(other))

    def __rxor__(self, other):
        return matrix._wrap(matrix._other_operand(other) ^ self._data)

    def __lshift__(self, other):
        return matrix._wrap(self._data << matrix._other_operand(other))

    def __rlshift__(self, other):
        return matrix._wrap(matrix._other_operand(other) << self._data)

    def __rshift__(self, other):
        return matrix._wrap(self._data >> matrix._other_operand(other))

    def __rrshift__(self, other):
        return matrix._wrap(matrix._other_operand(other) >> self._data)

    # -- scalar-conversion dunders: a `matrix` can NEVER be 0-d (construction
    #    always promotes to at least (1,1) -- see `__new__`), so these
    #    ALWAYS raise, exactly like real numpy's `matrix` does even for a
    #    (1,1) receiver (`int(np.matrix([[5]]))` raises the SAME
    #    "only 0-dimensional arrays can be converted to Python scalars"
    #    `TypeError` numpy raises for any 2-D array -- verified live, and
    #    verified that anionpy's own `ndarray.__int__`/`__float__`/
    #    `__complex__`/`__index__` (already declared exact) raise
    #    byte-identical text on a 2-D size-1 `anionpy.ndarray`, so a plain
    #    delegation reproduces it in every case, not just the ones tested) --
    def __int__(self):
        return self._data.__int__()

    def __float__(self):
        return self._data.__float__()

    def __complex__(self):
        return self._data.__complex__()

    def __index__(self):
        return self._data.__index__()

    # -- __bool__: depends on SIZE not ndim (already true of the underlying
    #    `ndarray.__bool__`, declared exact for arbitrary shapes), so a
    #    size-1 matrix of any shape is truthy/falsy per its single element
    #    and a multi-element matrix raises the same ambiguous-truth-value
    #    `ValueError` -- plain delegation reproduces both branches --
    def __bool__(self):
        return self._data.__bool__()

    # -- __len__: numpy's `len(matrix)` is `matrix.shape[0]`, same as any
    #    `ndarray` -- plain delegation (`ndarray.__len__` already exact) --
    def __len__(self):
        return self._data.__len__()

    # -- __iter__: yields matrix-wrapped ROWS (verified live: `type(row)
    #    for row in np.matrix(...)` is `matrix`, shape `(1, ncols)`), which
    #    is exactly what `self[i]` (this class's own `__getitem__`) already
    #    produces for an integer row index -- reuse it rather than
    #    reimplementing the row-promotion rule a second time. The `range`
    #    loop here is a control-flow/dispatch loop (building an iterator of
    #    already-computed rows), not a numerical one. --
    def __iter__(self):
        for i in range(self._data.shape[0]):
            yield self[i]

    # -- __contains__: real numpy's `ndarray`/`matrix` implement this as
    #    `(self == value).any()` (flattened), NOT the default per-element
    #    `__eq__`-or-`is` sequence-protocol fallback Python would otherwise
    #    synthesize from `__iter__` -- verified this distinction matters:
    #    `anionpy.ndarray` has no `__contains__` of its own, so `2 in
    #    anionpy_2d_array` hits Python's default and raises numpy's own
    #    ambiguous-truth-value `ValueError` (measured live), where real
    #    numpy succeeds. Implemented explicitly here rather than relying on
    #    inherited default. --
    def __contains__(self, value):
        return bool((self._data == matrix._other_operand(value)).any())

    # -- __setitem__: NOT declared exact -- measured divergence, kept
    #    implemented anyway because it is correct on 4 of 5 measured cases
    #    (scalar-at-origin, scalar-at-negative-index, row-slice, whole-int-
    #    row). `numpy.matrix.__setitem__` itself is not overridden in
    #    `defmatrix.py`, but real numpy's C-level `ndarray.__setitem__`
    #    resolves the assignment's broadcast-destination shape through the
    #    RECEIVER's own (possibly overridden) `__getitem__` -- so for
    #    `matrix`, a `(:, k)` column index promotes to a genuine `(n,1)`
    #    destination, and assigning a flat `(n,)` RHS to it raises
    #    `ValueError` on real numpy. This composition wrapper's plain
    #    passthrough validates against `anionpy.ndarray`'s natural flat
    #    `(n,)` destination for the same index instead, and silently
    #    succeeds. See `anionpy/_state/matrix.py`'s DECLINED section and
    #    `tests/differential/matrix_cases.py`'s `_SETITEM_CASES` comment
    #    for the measured `col_slice` case. Do not declare without fixing
    #    the broadcast-shape derivation.
    def __setitem__(self, index, value):
        self._data[index] = matrix._other_operand(value)

    # -- reductions matrix overrides beyond sum/mean/min/max/all/any
    #    (verified live against real numpy 2.5.1's `defmatrix.py` source,
    #    which implements each via `N.ndarray.<op>(self, ..., keepdims=True)
    #    ._collapse(axis)` -- axis=None collapses to a bare scalar exactly
    #    like the six already-declared reductions, axis=0/1 keeps the
    #    reduced axis as size-1 via `keepdims=True`) --
    def std(self, axis=None, dtype=None, out=None, ddof=0):
        if axis is None:
            return self._data.std(dtype=dtype, ddof=ddof)
        return matrix._wrap(self._data.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=True))

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        if axis is None:
            return self._data.var(dtype=dtype, ddof=ddof)
        return matrix._wrap(self._data.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=True))

    def prod(self, axis=None, dtype=None, out=None):
        if axis is None:
            return self._data.prod(dtype=dtype)
        return matrix._wrap(self._data.prod(axis=axis, dtype=dtype, keepdims=True))

    # -- argmax/argmin/ptp: DIFFERENT orientation rule than sum/mean/etc.
    #    Real numpy computes these WITHOUT `keepdims` (a 1-D result of
    #    length = the size of the OTHER axis) and then reorients via its
    #    own `_align(axis)` helper: `axis=0` keeps the 1-D result as a ROW
    #    (matches this class's own `_wrap`'s default 1-D->(1,n) promotion),
    #    `axis=1` reorients it as a COLUMN instead (`_wrap` alone would
    #    wrongly make it a row too, since `_wrap` cannot see which axis
    #    produced the data) -- verified live for all three ops (docstring
    #    example matches: `x.argmax(1)` is a COLUMN, not a row). `axis=None`
    #    collapses to a bare scalar, same as the `_collapse` reductions. --
    def argmax(self, axis=None, out=None):
        if axis is None:
            return self._data.argmax()
        r = self._data.argmax(axis=axis)
        if axis == 0:
            return matrix._wrap(r.reshape((1, r.shape[0])))
        return matrix._wrap(r.reshape((r.shape[0], 1)))

    def argmin(self, axis=None, out=None):
        if axis is None:
            return self._data.argmin()
        r = self._data.argmin(axis=axis)
        if axis == 0:
            return matrix._wrap(r.reshape((1, r.shape[0])))
        return matrix._wrap(r.reshape((r.shape[0], 1)))

    def ptp(self, axis=None, out=None):
        # `anionpy.ndarray` has no `.ptp()` method of its own (measured
        # live: `AttributeError`), but the top-level `anionpy.ptp` function
        # exists and is already declared exact -- max-minus-min, dispatched
        # entirely to Rust, not a Python-level numerical loop.
        if axis is None:
            return _ap.ptp(self._data)
        r = _ap.ptp(self._data, axis=axis)
        if axis == 0:
            return matrix._wrap(r.reshape((1, r.shape[0])))
        return matrix._wrap(r.reshape((r.shape[0], 1)))

    # -- squeeze: NOT overridden by real numpy `matrix` beyond the
    #    docstring (`N.ndarray.squeeze(self, axis=axis)`, subclass-preserved)
    #    -- and this class's own `_wrap` ALREADY implements the exact
    #    promotion rule matrix's squeeze needs (0-d -> (1,1), 1-d -> (1,n)),
    #    verified live case-by-case against real numpy: a (2,1)/(1,2)/(3,1)
    #    squeeze all collapse to a `(1, n)` matrix (never truly 1-D), and a
    #    (1,1) squeeze collapses to `(1,1)`, matching `_wrap`'s rule exactly. --
    def squeeze(self, axis=None):
        return matrix._wrap(self._data.squeeze(axis=axis))

    # -- shape/data ops that are already 2-D-in/2-D-out (or reduce to 2-D
    #    per `_wrap`'s rule) and, verified live, stay `matrix`-typed on
    #    real numpy because `matrix` does not override any of them --
    def reshape(self, *shape, **kwargs):
        return matrix._wrap(self._data.reshape(*shape, **kwargs))

    def swapaxes(self, axis1, axis2):
        return matrix._wrap(self._data.swapaxes(axis1, axis2))

    def repeat(self, repeats, axis=None):
        return matrix._wrap(self._data.repeat(repeats, axis=axis))

    def argsort(self, axis=-1, kind=None, order=None):
        return matrix._wrap(self._data.argsort(axis=axis, kind=kind))

    def cumsum(self, axis=None, dtype=None, out=None):
        return matrix._wrap(self._data.cumsum(axis=axis, dtype=dtype))

    def cumprod(self, axis=None, dtype=None, out=None):
        return matrix._wrap(self._data.cumprod(axis=axis, dtype=dtype))

    def clip(self, a_min=None, a_max=None, out=None):
        return matrix._wrap(self._data.clip(a_min, a_max))

    # -- fill/sort: in-place mutation, return None (matches real numpy) --
    def fill(self, value):
        self._data.fill(value)

    def sort(self, axis=-1, kind=None, order=None):
        self._data.sort(axis=axis, kind=kind)


def asmatrix(data, dtype=None):
    """`matrix(data, dtype=dtype, copy=False)` -- see `matrix.__new__`'s
    docstring for why `copy=False` here still copies (documented, not
    declared exact).
    """
    return matrix(data, dtype=dtype, copy=False)


def _is_scalar_index(x) -> bool:
    return isinstance(x, (int, bool)) or hasattr(x, "__index__")
