"""anionpy.memmap -- composition wrapper for numpy's `memmap` (a file-backed
`ndarray` subclass).

SCOPE. Exactly like `anionpy.matrix` (see `anionpy/matrix.py`'s module
docstring) and `anionpy.ma.MaskedArray`: `anionpy.ndarray` is a PyO3-backed
Rust type and cannot be subclassed from Python (`class Foo(anionpy.ndarray):
...` raises `TypeError: type 'anionpy.ndarray' is not an acceptable base
type` -- measured live, same as for `matrix`). So `memmap` here is a plain
Python class that HOLDS an `anionpy.ndarray` (`self._data`) rather than IS
one. `isinstance(m, anionpy.ndarray)` is `False` here, `True` on real numpy
-- the same accepted, documented divergence `matrix` already carries.

A PRIOR PASS (`anionpy/_state/matrix.py`, ticket #66/2026-08-07) measured
`memmap` "100% structurally blocked" and declared/attempted zero items,
reasoning that `anionpy.array()` always copies and there is no
buffer-protocol/`frombuffer` construction path, so a real file-backed
*shared* buffer cannot be built. That reasoning is correct as far as it
goes -- genuine buffer ALIASING (two objects mutating the same underlying
memory, or a write silently reappearing after reopening the file) is
unreproducible here, exactly like `matrix`'s already-accepted `copy=False`
gap -- but it does not follow that NOTHING about `memmap` is reachable.
`matrix` has the identical missing-aliasing problem (see its module
docstring) yet still shipped ~90 declared items by reproducing memmap's
VALUE/SHAPE/DTYPE/TYPE contract on top of a copied buffer. The same
strategy applies here: this module uses Python's stdlib `mmap` (NOT numpy)
to get a REAL OS-backed mapping of the file, decodes its bytes with
`struct` (stdlib, not numpy) into a flat Python list, and hands that list
to the already-existing, already-declared-exact `anionpy.array()` list
constructor. No numpy call ever produces a value; no new Rust is needed
because `anionpy.array()` already accepts nested lists and `.reshape()`
already implements C/F order (both Rust-backed, no arithmetic happens in
this file). This closes the specific gap the prior pass hit (there is no
buffer *sharing* path) without needing one: value-level behaviour never
required buffer sharing to begin with.

What this CANNOT reproduce, and is never declared here: two independent
`memmap` objects over the same file observing each other's writes; a
mode='r+'/'w+' write surviving without an explicit `.flush()` call (real
numpy's OS-backed `mmap` writes through continuously; this wrapper only
serializes `self._data` back to the file inside `.flush()` itself);
`mode='c'` (copy-on-write) diverging at the OS page-fault level under
concurrent access. `.flush()` itself, and single-object read-then-write-
then-flush-then-reopen round trips, ARE reproduced (verified live, see
`tests/differential/memmap_cases.py`).

TYPE-PRESERVING RULE (measured live against real numpy 2.5.1, not
guessed): `memmap.__array_priority__ == -100.0`, deliberately lower than
plain `ndarray`'s default (0.0). Consequence, confirmed for every op
below: ufunc/arithmetic/comparison/reduction results downgrade to a PLAIN
`ndarray` (`m + 1`, `m == m`, `m.sum(axis=0)` are all `numpy.ndarray`, not
`numpy.memmap`), while VIEW-shaped ops (`.T`, `.reshape`, `.ravel`,
`.flatten`, `.swapaxes`, `.squeeze`, `.transpose`, `__getitem__`) preserve
`memmap`-ness AND carry `filename`/`mode`/`offset` through from the
parent (measured live), whereas `.copy()`/`.astype()` reset those three
to `None` (also measured live -- `.copy()` goes through a fresh `empty()`
template with no `_mmap`, so `__array_finalize__` cannot find one to copy
from). This wrapper reproduces both branches explicitly rather than
uniformly wrapping or uniformly unwrapping.

No numerical loops: every arithmetic/reduction/comparison method below is
one dispatch to the already-built `self._data`'s own (Rust-backed)
operator/method. The only "loop" in this file is `_decode`/`_encode`,
which walks raw bytes through `struct.unpack`/`struct.pack` -- a
byte-format decode, not a numerical computation.
"""
from __future__ import annotations

import mmap as _mmap
import os as _os
import struct as _struct

import anionpy as _ap

__all__ = ["memmap"]

# numpy dtype spelling -> (struct format char, itemsize, is_complex)
_DTYPE_FMT: dict[str, tuple[str, int, bool]] = {
    "bool": ("?", 1, False),
    "int8": ("b", 1, False),
    "uint8": ("B", 1, False),
    "int16": ("h", 2, False),
    "uint16": ("H", 2, False),
    "int32": ("i", 4, False),
    "uint32": ("I", 4, False),
    "int64": ("q", 8, False),
    "uint64": ("Q", 8, False),
    "float16": ("e", 2, False),
    "float32": ("f", 4, False),
    "float64": ("d", 8, False),
    "complex64": ("f", 4, True),   # pair of float32
    "complex128": ("d", 8, True),  # pair of float64
}


def _itemsize(dtype: str) -> int:
    fmt, size, is_complex = _DTYPE_FMT[dtype]
    return size * 2 if is_complex else size


def _decode(raw: bytes, dtype: str, count: int) -> list:
    """Turn `count` items' worth of raw bytes into a flat Python list,
    matching `dtype`'s on-disk layout. Pure stdlib `struct` decode -- no
    numpy, no arithmetic, one `unpack` call.
    """
    fmt, _size, is_complex = _DTYPE_FMT[dtype]
    if is_complex:
        flat = _struct.unpack(f"={2 * count}{fmt}", raw)
        return [complex(flat[2 * i], flat[2 * i + 1]) for i in range(count)]
    vals = _struct.unpack(f"={count}{fmt}", raw)
    return list(vals)


def _encode(values: list, dtype: str) -> bytes:
    """Inverse of `_decode`: flat Python scalars -> raw bytes in `dtype`'s
    on-disk layout. Used only by `.flush()`.
    """
    fmt, _size, is_complex = _DTYPE_FMT[dtype]
    if is_complex:
        flat = []
        for v in values:
            c = complex(v)
            flat.append(c.real)
            flat.append(c.imag)
        return _struct.pack(f"={len(flat)}{fmt}", *flat)
    return _struct.pack(f"={len(values)}{fmt}", *values)


def _prod(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


# numpy long-form mode alias -> short form (`_core/memmap.py`; measured live
# against numpy 2.5.1: constructing with a long form reads back `.mode` as
# the corresponding short form, not the spelling passed in).
_MODE_ALIASES: dict[str, str] = {
    "readonly": "r",
    "copyonwrite": "c",
    "readwrite": "r+",
    "write": "w+",
}
_MODE_CHOICES = ("r", "c", "r+", "w+", "readonly", "copyonwrite", "readwrite",
                  "write")


class memmap:
    """Composition-based stand-in for `numpy.memmap`. See module docstring
    for why this cannot be a real `anionpy.ndarray` subclass, and for the
    exact type-preservation rule reproduced by the methods below.
    """

    __slots__ = (
        "_data", "filename", "mode", "offset", "_order", "_base", "_mm", "_fh",
        "_array_offset",
    )

    # matches real numpy's own class attribute exactly (measured live:
    # `np.memmap.__array_priority__ == -100.0`) -- a plain constant, no
    # behaviour of its own, but it IS what drives every "downgrade to
    # plain ndarray" rule documented above and reproduced explicitly by
    # this wrapper's arithmetic/reduction methods.
    __array_priority__ = -100.0

    def __new__(cls, filename, dtype="uint8", mode="r+", offset=0,
                shape=None, order="C"):
        # DEFECT (ticket #77, 2026-08-08): this used to accept only the 4
        # short mode forms. Real numpy also accepts 4 long aliases and
        # normalizes `.mode` to the short form afterward (measured live,
        # numpy 2.5.1) -- both the acceptance and the exact wording of the
        # rejection message (numpy's own enumeration, all 8 forms) are
        # reproduced here.
        if mode not in _MODE_CHOICES:
            raise ValueError(
                "mode must be one of ['r', 'c', 'r+', 'w+', 'readonly', "
                f"'copyonwrite', 'readwrite', 'write'] (got {mode!r})"
            )
        mode = _MODE_ALIASES.get(mode, mode)

        # DEFECT (ticket #77, 2026-08-08): this used to require `dtype` be
        # spelled as one of the 14 canonical name strings that key
        # `_DTYPE_FMT`, rejecting every other spelling numpy accepts
        # (`np.dtype(...)` instances, builtins, numpy scalar types, char
        # codes, `anionpy.dtype(...)` objects). `anionpy.dtype()` is already
        # declared exact and already normalizes every spelling it can parse
        # to the same canonical `.name` real numpy uses (and raises the same
        # `TypeError` for genuinely unparseable input) -- route through it
        # instead of matching against raw spellings by hand.
        dtype = _ap.dtype(dtype).name
        if dtype not in _DTYPE_FMT:
            raise ValueError(f"unsupported dtype for anionpy.memmap: {dtype!r}")
        itemsize = _itemsize(dtype)

        path = _os.fspath(filename)
        if isinstance(shape, int):
            shape = (shape,)

        if mode == "w+" and shape is None:
            raise ValueError("shape must be given if mode == 'w+'")

        fh = open(path, ("r" if mode == "c" else mode) + "b")

        # Port of real numpy's own `_core/memmap.py` byte-accounting
        # algorithm (read via `inspect.getsourcefile`, verified live against
        # numpy 2.5.1) rather than a re-derived approximation: an earlier
        # pass here unconditionally raised on a zero-byte file, which is
        # WRONG for 'w+'/'r+' -- numpy pads a 1-byte hole (`gh-27723`) to
        # make an empty memmap constructible, verified live to succeed, not
        # raise. Only 'r'/'c' on a genuinely-empty file with no padding
        # ability still hit the OS's own zero-length mmap rejection.
        fh.seek(0, 2)
        flen = fh.tell()
        if shape is None:
            remaining = flen - offset
            if remaining % itemsize != 0:
                raise ValueError(
                    "Size of available data is not a multiple of the "
                    "data-type size."
                )
            count = remaining // itemsize
            shape = (count,)
        else:
            count = _prod(shape)

        nbytes = offset + count * itemsize
        if mode in ("w+", "r+"):
            nbytes = max(nbytes, 1)
            if flen < nbytes:
                fh.seek(nbytes - 1, 0)
                fh.write(b"\x00")
                fh.flush()

        access = {
            "r": _mmap.ACCESS_READ,
            "r+": _mmap.ACCESS_WRITE,
            "w+": _mmap.ACCESS_WRITE,
            "c": _mmap.ACCESS_COPY,
        }[mode]

        start = offset - offset % _mmap.ALLOCATIONGRANULARITY
        map_bytes = nbytes - start
        if map_bytes == 0 and start > 0:
            map_bytes += _mmap.ALLOCATIONGRANULARITY
            start -= _mmap.ALLOCATIONGRANULARITY
        array_offset = offset - start
        # `mmap.mmap` itself still refuses `length=0` on a genuinely-empty
        # underlying file (mode 'r'/'c', nothing above padded it) -- this
        # is real numpy's own failure mode too (it makes the identical
        # `mmap.mmap(..., 0, ...)` call and lets the OS raise), not a
        # divergence to paper over.
        mm = _mmap.mmap(fh.fileno(), map_bytes, access=access, offset=start)

        raw = mm[array_offset:array_offset + count * itemsize]
        flat = _decode(raw, dtype, count)
        arr = _ap.array(flat, dtype=dtype)
        if len(shape) != 1 or shape[0] != count:
            arr = arr.reshape(shape, order=order)

        self = object.__new__(cls)
        self._data = arr
        self.filename = _os.path.abspath(path)
        self.mode = mode
        self.offset = offset
        self._order = order
        self._base = None
        self._mm = mm
        self._fh = fh
        self._array_offset = array_offset
        return self

    def __init__(self, filename, dtype="uint8", mode="r+", offset=0,
                 shape=None, order="C"):
        # All real work happens in __new__, matching numpy's own ndarray
        # subclass convention (and `anionpy.matrix`'s existing pattern).
        pass

    # -- internal: build a memmap wrapper around an already-computed
    #    `anionpy.ndarray`, either preserving this object's file identity
    #    (view-shaped ops: T/reshape/ravel/.../getitem) or resetting it
    #    (copy-shaped ops: copy/astype) -- mirrors real numpy's
    #    `__array_finalize__` branching (measured live, see module
    #    docstring's TYPE-PRESERVING RULE paragraph). --
    def _as_view(self, data):
        out = object.__new__(memmap)
        out._data = data
        out.filename = self.filename
        out.mode = self.mode
        out.offset = self.offset
        out._order = self._order
        out._base = self
        out._mm = None
        out._fh = None
        out._array_offset = None  # never read: flush() walks to the true root
        return out

    def _as_copy(self, data):
        out = object.__new__(memmap)
        out._data = data
        out.filename = None
        out.mode = None
        out.offset = None
        out._order = self._order
        out._base = None
        out._mm = None
        out._fh = None
        out._array_offset = None  # never read: a copy is never a flush root
        return out

    # -- plain attribute passthroughs (already-computed on `self._data`,
    #    no wrapping question -- these are never array-valued) --
    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

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
    def base(self):
        """Real numpy: a root `memmap` (opened straight from a file)
        reports its own `mmap.mmap` object as `.base`; a VIEW (T/reshape/
        .../getitem result) reports its parent `memmap`; `.copy()`/
        `.astype()` report `None` (verified live for all three cases,
        see module docstring). Reproduced with a genuine `mmap.mmap`
        instance for roots (this class always opens one via stdlib
        `mmap`, not a numpy fabrication) and the parent object for views.
        """
        if self._base is not None:
            return self._base
        return self._mm

    # -- construction/dispatch dunders --
    def __repr__(self):
        # numpy's repr indentation is `len(class_name) + 2` spaces before
        # continuation lines (the width of `"memmap(["`/`"array(["`) --
        # reproduced by re-indenting `self._data`'s own already-exact
        # `array(...)` repr rather than re-deriving numpy's formatting
        # logic from scratch.
        inner_repr = repr(self._data)
        prefix_old = "array("
        prefix_new = "memmap("
        assert inner_repr.startswith(prefix_old) and inner_repr.endswith(")")
        body = inner_repr[len(prefix_old):-1]
        lines = body.split("\n")
        if len(lines) > 1:
            # Only pad non-blank continuation lines: numpy's own >=3-D repr
            # inserts genuinely EMPTY lines between top-level "groups" (see
            # e.g. a (2,3,4) repr) -- measured live, those blank separator
            # lines carry no leading padding in real numpy's memmap repr
            # either, so padding them here would insert trailing whitespace
            # real numpy never emits (a real, corpus-caught divergence).
            pad = " " * (len(prefix_new) - len(prefix_old))
            lines = [lines[0]] + [(pad + ln) if ln else ln for ln in lines[1:]]
            body = "\n".join(lines)
        return prefix_new + body + ")"

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for i in range(len(self._data)):
            yield self[i]

    def __bool__(self):
        return bool(self._data)

    # matches real numpy: defining `__eq__` makes a class unhashable by
    # default unless `__hash__` is explicitly restored; `memmap` (like
    # any ndarray) is unhashable -- verified live, `hash(np.memmap(...))`
    # raises `TypeError: unhashable type: 'memmap'`. Setting the class
    # attribute to `None` directly (not a method returning `None`) is
    # what makes Python's own `hash()` builtin raise `TypeError` without
    # ever calling into this class -- exactly how real numpy's own
    # (identically `__eq__`-overriding) `ndarray`/`memmap` behave.
    __hash__ = None  # type: ignore[assignment]

    # -- indexing: full scalar index -> bare scalar (numpy scalar type,
    #    exactly what `anionpy.ndarray.__getitem__` already returns);
    #    anything array-shaped -> memmap, filename/mode/offset preserved
    #    (measured live: `m[0]`, `m[0:2]` are both `numpy.memmap`) --
    def __getitem__(self, index):
        out = self._data[index]
        if isinstance(out, _ap.ndarray):
            return self._as_view(out)
        return out

    def __setitem__(self, index, value):
        if self.mode == "r":
            # byte-identical to real numpy's message on a read-only
            # memmap -- verified live.
            raise ValueError("assignment destination is read-only")
        self._data[index] = value

    def fill(self, value):
        if self.mode == "r":
            raise ValueError("assignment destination is read-only")
        self._data.fill(value)

    def flush(self):
        """Serialize `self._data` back to the backing file, matching real
        numpy's write-through for 'r+'/'w+' (verified live: reopening the
        file after `.flush()` sees the new values); a no-op for 'c'
        (copy-on-write never persists, matches real numpy's MAP_PRIVATE
        semantics -- verified live) and 'r' (no writes are ever possible).
        Only reaches the file for a memmap that OWNS a real mmap/file
        handle (a root object, not a view) -- a view's `.flush()` walks
        to its root via `._base`, mirroring how real numpy's `.filename`/
        `.offset` chain back to the same underlying `_mmap`.
        """
        root = self
        while root._base is not None:
            root = root._base
        if root.mode not in ("r+", "w+") or root._mm is None:
            return
        flat = root._data.reshape((-1,), order=root._order).tolist()
        raw = _encode(flat, str(root._data.dtype))
        n = len(raw)
        # `root._mm` was opened at `mmap.mmap(..., offset=start)` (rounded
        # down to `ALLOCATIONGRANULARITY`, per real numpy's own algorithm
        # ported into `__new__`) -- writes into it must be indexed by
        # `_array_offset` (the position of the array's OWN first byte
        # inside that already-shifted mapping), not the raw constructor
        # `offset=` the caller passed, which is `_array_offset`'s pre-
        # rounding source and can differ from it whenever `offset` is not
        # itself a multiple of `ALLOCATIONGRANULARITY`.
        root._mm[root._array_offset:root._array_offset + n] = raw
        root._mm.flush()

    # -- view-shaped structural ops: memmap-preserving, filename/mode/
    #    offset carried through (measured live for every op below) --
    @property
    def T(self):
        return self._as_view(self._data.T)

    def transpose(self, *axes):
        return self._as_view(self._data.transpose(*axes))

    def _needs_copy_for_c_order(self):
        # numpy's reshape/ravel (default order='C') return a genuine VIEW
        # only when the source is already C-contiguous; otherwise numpy
        # must copy, and a memmap that has been copied loses its
        # filename/mode/offset/base ties (same rule as `.copy()`/
        # `.astype()` -- measured live: an F-order-constructed memmap
        # reshaped/raveled to a flat C-order shape comes back with
        # `.filename is None`). `flags` is queried on `self._data`
        # directly, not a converted copy, per this task's instrument
        # discipline.
        flags = self._data.flags
        return not flags["C_CONTIGUOUS"]

    def reshape(self, *shape, **kwargs):
        out = self._data.reshape(*shape, **kwargs)
        if self._needs_copy_for_c_order():
            # Measured live: unlike `.copy()`/`.astype()`/`.ravel()`/
            # `.flatten()` (all of which reset `.base` to None on a copy,
            # see their own comments), a copying `.reshape()` still sets
            # `.base` to the SOURCE memmap object -- `numpy.memmap`'s
            # reshape goes through a distinct "new-from-template"
            # construction path (`PyArray_Newshape`) from ravel's, which
            # tags provenance even when the buffer itself was copied. A
            # narrow, specifically-measured divergence from the sibling
            # methods, not a general "reshape never copies" rule.
            wrapped = self._as_copy(out)
            wrapped._base = self
            return wrapped
        return self._as_view(out)

    def ravel(self, order="C"):
        out = self._data.ravel(order)
        if self._needs_copy_for_c_order():
            return self._as_copy(out)
        return self._as_view(out)

    def flatten(self, order="C"):
        # `ndarray.flatten()` is documented to ALWAYS return a copy,
        # unconditionally (unlike `ravel`, which returns a view when it
        # can) -- so `memmap.flatten()` always resets filename/mode/offset
        # to None, regardless of the source's contiguity. Verified live.
        return self._as_copy(self._data.flatten(order))

    def swapaxes(self, axis1, axis2):
        return self._as_view(self._data.swapaxes(axis1, axis2))

    def squeeze(self, axis=None):
        return self._as_view(self._data.squeeze(axis=axis))

    # -- copy-shaped ops: memmap-preserving TYPE, but filename/mode/offset
    #    RESET to None (measured live: `.copy()`/`.astype()` go through a
    #    fresh `empty()` template with no backing `_mmap`) --
    def copy(self):
        return self._as_copy(self._data.copy())

    def astype(self, dtype):
        return self._as_copy(self._data.astype(dtype))

    # -- plain conversions (never memmap-wrapped) --
    def tolist(self):
        return self._data.tolist()

    def tobytes(self):
        return self._data.tobytes()

    def item(self):
        return self._data.item()

    # -- reductions: `__array_priority__=-100.0` downgrades EVERY result
    #    to a plain value/ndarray, never memmap -- axis=None already
    #    returns a bare numpy scalar from `anionpy.ndarray` itself
    #    (pre-existing `numpy_scalar_from_0d` behaviour), axis=k returns
    #    a plain `anionpy.ndarray` (measured live for every op below) --
    def sum(self, axis=None, dtype=None, out=None):
        return self._data.sum(axis=axis, dtype=dtype)

    def mean(self, axis=None, dtype=None, out=None):
        return self._data.mean(axis=axis, dtype=dtype)

    def min(self, axis=None, out=None):
        return self._data.min(axis=axis) if axis is not None else self._data.min()

    def max(self, axis=None, out=None):
        return self._data.max(axis=axis) if axis is not None else self._data.max()

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        return self._data.std(axis=axis, dtype=dtype, ddof=ddof)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        return self._data.var(axis=axis, dtype=dtype, ddof=ddof)

    def prod(self, axis=None, dtype=None, out=None):
        return self._data.prod(axis=axis, dtype=dtype)

    def all(self, axis=None, out=None):
        return self._data.all(axis=axis) if axis is not None else self._data.all()

    def any(self, axis=None, out=None):
        return self._data.any(axis=axis) if axis is not None else self._data.any()

    # -- elementwise arithmetic/bitwise dunders + reflected forms: plain
    #    `ndarray` result (same `__array_priority__` downgrade as
    #    reductions) -- one dispatch to `self._data`'s own already-exact
    #    dunder, no wrapping --
    @staticmethod
    def _unwrap(other):
        return other._data if isinstance(other, memmap) else other

    def __add__(self, other):
        return self._data + memmap._unwrap(other)

    def __radd__(self, other):
        return memmap._unwrap(other) + self._data

    def __sub__(self, other):
        return self._data - memmap._unwrap(other)

    def __rsub__(self, other):
        return memmap._unwrap(other) - self._data

    def __mul__(self, other):
        return self._data * memmap._unwrap(other)

    def __rmul__(self, other):
        return memmap._unwrap(other) * self._data

    def __mod__(self, other):
        return self._data % memmap._unwrap(other)

    def __rmod__(self, other):
        return memmap._unwrap(other) % self._data

    def __floordiv__(self, other):
        return self._data // memmap._unwrap(other)

    def __rfloordiv__(self, other):
        return memmap._unwrap(other) // self._data

    def __and__(self, other):
        return self._data & memmap._unwrap(other)

    def __rand__(self, other):
        return memmap._unwrap(other) & self._data

    def __or__(self, other):
        return self._data | memmap._unwrap(other)

    def __ror__(self, other):
        return memmap._unwrap(other) | self._data

    def __xor__(self, other):
        return self._data ^ memmap._unwrap(other)

    def __rxor__(self, other):
        return memmap._unwrap(other) ^ self._data

    def __lshift__(self, other):
        return self._data << memmap._unwrap(other)

    def __rlshift__(self, other):
        return memmap._unwrap(other) << self._data

    def __rshift__(self, other):
        return self._data >> memmap._unwrap(other)

    def __rrshift__(self, other):
        return memmap._unwrap(other) >> self._data

    def __neg__(self):
        return -self._data

    def __pos__(self):
        return +self._data

    def __abs__(self):
        return abs(self._data)

    def __invert__(self):
        return ~self._data

    # -- comparisons: also plain `ndarray` result --
    def __eq__(self, other):
        return self._data == memmap._unwrap(other)

    def __ne__(self, other):
        return self._data != memmap._unwrap(other)

    def __lt__(self, other):
        return self._data < memmap._unwrap(other)

    def __le__(self, other):
        return self._data <= memmap._unwrap(other)

    def __gt__(self, other):
        return self._data > memmap._unwrap(other)

    def __ge__(self, other):
        return self._data >= memmap._unwrap(other)
