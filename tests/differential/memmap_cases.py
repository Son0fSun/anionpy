"""Differential registry entries for `anionpy.memmap` (numpy's file-backed
`ndarray` subclass).

WIRED into `registry.py` (ticket #77, 2026-08-08) via a full,
collision-checked `REGISTRY.update(MEMMAP_SPECS)` merge -- see that file's
"memmap (ticket MEMMAP, 2026-08-08 / #77, 2026-08-08)" section. Originally
shipped unwired: `registry.py` was being actively edited by a concurrent
agent at the time this file was first written (same constraint
`poly1d_legacy_cases.py` documents), so touching it risked a
last-writer-wins collision with that work; it was free by the time ticket
#77 landed.

CORRECTS A PRIOR RULING. `matrix_cases.py`'s own module docstring (and
`anionpy/_state/matrix.py`, committed 2026-08-07) records `memmap` as
"100% structurally blocked: no buffer-sharing construction path at all".
That measurement was correct about the missing path (`anionpy.array()`
always copies; there is no `frombuffer`/buffer-aliasing constructor) but
overbroad in its conclusion: genuine cross-process buffer ALIASING is not
actually required to reproduce `memmap`'s VALUE/SHAPE/DTYPE/TYPE contract,
only to reproduce live cross-object write visibility without an explicit
`flush()`/reopen -- and `matrix` shipped ~90 declared items under the exact
same missing-buffer-sharing limitation by matching that contract on a
freshly-built (not aliased) array. `anionpy/memmap.py` sidesteps the gap
entirely using Python's stdlib `mmap` + `struct` modules (real OS-level
file mapping, genuine byte encode/decode, zero calls into real numpy, zero
new Rust) to build a composition wrapper around `anionpy.ndarray` -- same
pattern as `anionpy.matrix` (ticket #66) and `anionpy.ma.MaskedArray`,
required because `anionpy.ndarray` cannot be subclassed from Python at all.
See `anionpy/memmap.py`'s own module docstring for the full construction
mechanics and the measured TYPE-PRESERVING rule (`__array_priority__ ==
-100.0` downgrades every arithmetic/reduction result to plain `ndarray`;
view-shaped ops -- `T`/`reshape`/`ravel`/`flatten`/`swapaxes`/`squeeze`/
`__getitem__` -- preserve `memmap`-ness and propagate `filename`/`mode`/
`offset`/`.base`; `copy`/`astype` reset those three to `None`).

SCOPE. Every item here is `kind="custom"` with explicit `numpy_adapter`/
`ionp_adapter` functions and `scalar_like=True`, exactly like
`matrix_cases.py`/`ma_cases.py` use for the same reason: the object's TYPE
(memmap vs plain ndarray vs bare scalar), not just its values, is part of
the question. `_np_snapshot`/`_ionp_snapshot` below encode that distinction
directly into the compared value (a literal `"memmap"`/`"ndarray"`/
`"scalar"` tag, plus shape/dtype/data/filename-is-set/mode), so a
wrong-type return is a real, visible mismatch under
`harness._compare_scalar_like`'s `==`, not silently coerced away.

FIXTURE CONSTRUCTION. Each case carries a plain-data CONTENT DESCRIPTOR
(flat values already in on-disk/flat order, dtype string, shape, order,
mode, offset) through `custom_cases()`'s `args` tuple -- never a shared
file path or a shared file handle. `harness.run_case`'s `_freshen` step
deep-copies `args`/`kwargs` before each side's call, but that only isolates
the plain-data descriptor, not any file a shared path might point at,
so each of `_np_open`/`_ionp_open` independently calls `_make_fixture` to
write ITS OWN fresh temp file with byte-identical content (built via real
numpy's `tofile()` on a flat 1-D array -- ground-truth fixture generation,
the same use of `import numpy as np` every other `*_cases.py` in this
package already makes; not a call in `anionpy`'s own code path). This
avoids any cross-talk where one side's mutation (`__setitem__`/`fill`/
`flush`) could pollute what the other side reads. All fixture files live
under a single `mkdtemp` directory inside `/private/tmp`, removed via
`atexit`.

WHAT IS NOT COVERED (and why -- see `docs/TICKET-MEMMAP-2026-08-08.md` for
the full split):
  - `view`, `dot`, `diagonal`, `sort`, `argsort`, `__pow__`/`__rpow__`,
    `__truediv__`/`__rtruediv__`, `__matmul__`/`__rmatmul__`,
    `__divmod__`/`__rdivmod__`: `anionpy.ndarray` itself has no base-level
    counterpart to inherit from (`view` raises `AttributeError` outright;
    the arithmetic ones use numpy's true-division/power/matmul semantics
    which are separately-tracked base gaps, not memmap-specific). Declaring
    these on memmap without a working base would be exactly the
    fabricated-class-dict-entry failure mode this task was warned against.
  - `__array_interface__`, `__array_struct__`, `__array_finalize__`,
    `__array_wrap__`, `__array_priority__` is covered (own-defined, see
    below) but the rest of the `__array_*` protocol family, `__buffer__`,
    `__dlpack__`/`__dlpack_device__`: numpy-internal ufunc/buffer-protocol
    machinery `anionpy.ndarray` does not implement at all; invoking these
    directly is not how any real caller uses memmap.
  - `__reduce__`, `__setstate__`, `__getstate__` and friends (pickling):
    unimplemented, no measured contract to pin.
  - `__class__`, `__module__`, `__dir__`, `__static_attributes__`,
    `__firstlineno__`, and similar pure Python-mechanical introspection
    names: not meaningfully "memmap behavior", already excluded by every
    other `*_cases.py` in this package.
  - `memmap.__repr__` for a dtype spelling that is `==` but not `is`
    numpy's canonical dtype object (e.g. `'q'` vs `'int64'`): numpy's own
    repr shows an explicit `dtype=int64` suffix in that case, anionpy's
    does not (pre-existing, memmap-independent `anionpy.array` repr gap,
    surfaced by ticket #77's dtype-spelling-variation cases). WITHDRAWN
    from `anionpy/_state/memmap.py`, kept in the corpus for the record.
  - `memmap.__new__/unsupported_dtype` (ticket #77, 2026-08-08): dtypes
    real numpy's `np.dtype()` CAN parse (string/void/datetime64/object
    kinds) but `anionpy.memmap` has no `_DTYPE_FMT` support for at all --
    real numpy succeeds constructing these, `anionpy.memmap` cannot back
    them structurally. Pre-existing, permanent, narrower-than-numpy gap,
    unrelated to the dtype/mode SPELLING-parsing defect this ticket fixed.
    Kept in the corpus for the record, NOT declared.

  TICKET #77 (2026-08-08) FIXED, no longer excluded:
  - Invalid `mode=` value message text now matches numpy's exact wording
    (`mode` accepts all 8 forms; the `ValueError` enumerates all 8).
    `memmap.__new__/invalid_mode` moved from "case only" to declared.
"""
from __future__ import annotations

import atexit
import os
import shutil
import struct as _struct
import tempfile

import _bootstrap  # noqa: F401

import numpy as np

import anionpy as ap
from anionpy.memmap import memmap as ap_memmap
from registry import ItemSpec

MEMMAP_SPECS: dict[str, ItemSpec] = {}

# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------
_TMPDIR = tempfile.mkdtemp(prefix="memmap_cases_", dir="/private/tmp")
atexit.register(shutil.rmtree, _TMPDIR, ignore_errors=True)
_counter = [0]


def _make_fixture(flat_values, dtype, pad_before=0):
    """Write a fresh temp file: `pad_before` zero bytes, then `flat_values`
    encoded as `dtype`, in that flat (on-disk) order. Built via real numpy's
    `tofile()` on a 1-D array -- ground-truth fixture generation only, not
    part of anionpy's own call path (see module docstring)."""
    _counter[0] += 1
    path = os.path.join(_TMPDIR, f"fix_{_counter[0]}.bin")
    with open(path, "wb") as f:
        f.write(b"\x00" * pad_before)
        np.array(flat_values, dtype=dtype).tofile(f)
    return path


def _round_nested(v, ndigits=9):
    if isinstance(v, list):
        return [_round_nested(e, ndigits) for e in v]
    if isinstance(v, complex):
        return complex(round(v.real, ndigits), round(v.imag, ndigits))
    if isinstance(v, float):
        return round(v, ndigits)
    return v


def _np_snapshot(x):
    if isinstance(x, np.memmap):
        return ("memmap", tuple(x.shape), str(x.dtype), x.filename is not None,
                x.mode, _round_nested(x.tolist()))
    if isinstance(x, np.ndarray):
        return ("ndarray", tuple(x.shape), str(x.dtype), _round_nested(x.tolist()))
    return ("scalar", type(x).__name__, _round_nested(x) if isinstance(x, (float, complex)) else x)


def _ionp_snapshot(x):
    if isinstance(x, ap_memmap):
        return ("memmap", tuple(x.shape), str(x.dtype), x.filename is not None,
                x.mode, _round_nested(x.tolist()))
    if isinstance(x, ap.ndarray):
        return ("ndarray", tuple(x.shape), str(x.dtype), _round_nested(x.tolist()))
    return ("scalar", type(x).__name__, _round_nested(x) if isinstance(x, (float, complex)) else x)


# ---------------------------------------------------------------------------
# Base construction corpus: (label, dtype, flat_values, shape, order, mode, offset)
# Deliberately crosses dtype family (bool/every int+uint width/every float
# width/both complex widths), dimensionality (1-D/2-D/3-D), order ('C'/'F'),
# mode ('r'/'r+'/'c'; 'w+' exercised separately since it does not read an
# existing file), and a nonzero offset case.
# ---------------------------------------------------------------------------
_ALL_DTYPES = [
    "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "float32", "float64",
    "complex64", "complex128",
]


def _family(dtype) -> str:
    """Canonical family name for ANY dtype spelling numpy accepts (used only
    to pick fixture VALUES -- ground-truth `np.dtype()` resolution, never
    part of anionpy's own call path). `dtype` itself, whatever spelling it
    was given as, is still what gets passed to `np.memmap`/`ap_memmap`."""
    return str(np.dtype(dtype))


def _fixture_values(dtype, n):
    fam = _family(dtype)
    if fam == "bool":
        return [bool(i % 2) for i in range(n)]
    if fam.startswith("uint"):
        return [i for i in range(n)]
    if fam.startswith("int"):
        return [i - n // 2 for i in range(n)]
    if fam.startswith("float"):
        return [i - n / 2 + 0.25 for i in range(n)]
    if fam.startswith("complex"):
        return [complex(i - n / 2 + 0.25, -(i) + 0.5) for i in range(n)]
    raise ValueError(dtype)


def _dtype_cross_cases():
    cases = []
    for dtype in _ALL_DTYPES:
        cases.append((f"dtype/{dtype}", (dtype, 6, (6,), "C", "r+", 0), {}))
    return cases


# ---------------------------------------------------------------------------
# TICKET #77 (2026-08-08): the corpus above varies dtype/mode IDENTITY
# (which of the 14 families / 4 short forms) but never SPELLING -- every
# case passes `dtype` and `mode` as the exact canonical strings
# `_DTYPE_FMT`/the old short-form check expected. That made 918/918 green
# blind to `memmap.__new__` only accepting those 14 canonical name strings
# and 4 short mode forms, rejecting every other spelling real numpy accepts
# (dtype instances, builtins, numpy scalar types, char codes, long mode
# aliases). These cases vary SPELLING, holding the resolved family/mode
# fixed, to close that gap.
# ---------------------------------------------------------------------------
_DTYPE_SPELLINGS: dict[str, list] = {
    "float64": [np.dtype("float64"), float, np.float64, "f8"],
    "int32": [np.dtype("int32"), np.int32, "i4"],
    "int64": [int, "q", np.dtype("int64")],
    "bool": [bool, "?", np.dtype("bool")],
    "uint8": [np.uint8, "B", np.dtype("uint8")],
    "float16": ["e", np.float16],
    "complex128": [complex, "c16", np.dtype("complex128")],
    "complex64": [np.complex64],
}


def _dtype_spelling_cases():
    cases = []
    for family, spellings in _DTYPE_SPELLINGS.items():
        for spelling in spellings:
            cases.append(
                (f"dtype_spelling/{family}/{spelling!r}",
                 (spelling, 6, (6,), "C", "r+", 0), {})
            )
    return cases


_MODE_SPELLINGS = ["readonly", "copyonwrite", "readwrite", "write"]


def _mode_spelling_cases():
    return [
        (f"mode_spelling/{mode}", ("int32", 6, (6,), "C", mode, 0), {})
        for mode in _MODE_SPELLINGS
    ]


def _shape_order_cases():
    return [
        ("shape/1d", ("int32", 6, (6,), "C", "r+", 0), {}),
        ("shape/2d_C", ("int32", 6, (2, 3), "C", "r+", 0), {}),
        ("shape/2d_F", ("int32", 6, (2, 3), "F", "r+", 0), {}),
        ("shape/3d_C", ("float64", 24, (2, 3, 4), "C", "r+", 0), {}),
        ("shape/0d", ("int32", 1, (), "C", "r+", 0), {}),
        ("shape/offset", ("int16", 6, (6,), "C", "r+", 4), {}),
        ("shape/mode_r", ("int32", 6, (6,), "C", "r", 0), {}),
        ("shape/mode_c", ("int32", 6, (6,), "C", "c", 0), {}),
    ]


def _build_open(dtype, n, shape, order, mode, offset):
    values = _fixture_values(dtype, n)
    return values, dtype, shape, order, mode, offset


def _np_open(dtype, n, shape, order, mode, offset):
    values = _fixture_values(dtype, n)
    path = _make_fixture(values, dtype, pad_before=offset)
    return np.memmap(path, dtype=dtype, mode=mode, offset=offset,
                      shape=shape if shape else None, order=order)


def _ionp_open(dtype, n, shape, order, mode, offset):
    values = _fixture_values(dtype, n)
    path = _make_fixture(values, dtype, pad_before=offset)
    return ap_memmap(path, dtype=dtype, mode=mode, offset=offset,
                      shape=shape if shape else None, order=order)


def _plain_prop_cases():
    return (_dtype_cross_cases() + _shape_order_cases()
            + _dtype_spelling_cases() + _mode_spelling_cases())


def _make_prop_spec(name, extractor, scalar_like=True):
    def _numpy(dtype, n, shape, order, mode, offset):
        m = _np_open(dtype, n, shape, order, mode, offset)
        return extractor(m, _np_snapshot)

    def _ionp(dtype, n, shape, order, mode, offset):
        m = _ionp_open(dtype, n, shape, order, mode, offset)
        return extractor(m, _ionp_snapshot)

    MEMMAP_SPECS[name] = ItemSpec(
        name=name, kind="custom", scalar_like=scalar_like,
        numpy_adapter=_numpy, ionp_adapter=_ionp,
        custom_cases=_plain_prop_cases,
    )


_make_prop_spec("memmap.__new__", lambda m, snap: snap(m))
_make_prop_spec("memmap.dtype", lambda m, snap: str(m.dtype))
_make_prop_spec("memmap.shape", lambda m, snap: tuple(m.shape))
_make_prop_spec("memmap.ndim", lambda m, snap: m.ndim)
_make_prop_spec("memmap.size", lambda m, snap: m.size)
_make_prop_spec("memmap.itemsize", lambda m, snap: m.itemsize)
_make_prop_spec("memmap.nbytes", lambda m, snap: m.nbytes)
_make_prop_spec("memmap.__len__", lambda m, snap: len(m) if m.ndim else "0-d: no len")
_make_prop_spec("memmap.__repr__", lambda m, snap: repr(m))
_make_prop_spec("memmap.__bool__", lambda m, snap: (
    bool(m) if m.size == 1 else f"raises: {_bool_raises(m)}"))
_make_prop_spec("memmap.tolist", lambda m, snap: _round_nested(m.tolist()))
_make_prop_spec("memmap.tobytes", lambda m, snap: m.tobytes())
_make_prop_spec("memmap.__array_priority__", lambda m, snap: m.__array_priority__)


def _bool_raises(m):
    try:
        bool(m)
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}"


# ---------------------------------------------------------------------------
# View-preserving structural ops: must return `memmap`, propagate
# filename/mode/offset/base.
# ---------------------------------------------------------------------------
def _view_op_spec(name, op):
    def _numpy(dtype, n, shape, order, mode, offset):
        m = _np_open(dtype, n, shape, order, mode, offset)
        out = op(m)
        return (_np_snapshot(out), isinstance(out, np.memmap),
                getattr(out, "base", None) is not None)

    def _ionp(dtype, n, shape, order, mode, offset):
        m = _ionp_open(dtype, n, shape, order, mode, offset)
        out = op(m)
        return (_ionp_snapshot(out), isinstance(out, ap_memmap),
                getattr(out, "base", None) is not None)

    MEMMAP_SPECS[name] = ItemSpec(
        name=name, kind="custom", scalar_like=True,
        numpy_adapter=_numpy, ionp_adapter=_ionp,
        custom_cases=lambda: [
            (label, args, kwargs) for label, args, kwargs in _shape_order_cases()
            if "0d" not in label  # structural ops on a 0-d memmap are a separate, narrower question
        ],
    )


_view_op_spec("memmap.T", lambda m: m.T)
_view_op_spec("memmap.transpose", lambda m: m.transpose())
_view_op_spec("memmap.reshape", lambda m: m.reshape((-1,)))
_view_op_spec("memmap.ravel", lambda m: m.ravel())
_view_op_spec("memmap.flatten", lambda m: m.flatten())
_view_op_spec("memmap.squeeze", lambda m: m.squeeze())
_view_op_spec("memmap.swapaxes", lambda m: m.swapaxes(0, -1))
_view_op_spec("memmap.__getitem__", lambda m: m[0:2] if m.shape and m.shape[0] > 1 else m[...])
_view_op_spec("memmap.base", lambda m: m[0:2].base if m.shape and m.shape[0] > 1 else m.base)


# ---------------------------------------------------------------------------
# Copy-reset ops: must return `memmap` (per numpy: `.copy()`/`.astype()`
# still return a `memmap` instance) but with filename/mode/offset reset to
# None (measured live against real numpy 2.5.1).
# ---------------------------------------------------------------------------
def _copy_op_spec(name, op):
    def _numpy(dtype, n, shape, order, mode, offset):
        m = _np_open(dtype, n, shape, order, mode, offset)
        out = op(m)
        return (_np_snapshot(out), isinstance(out, np.memmap), out.filename)

    def _ionp(dtype, n, shape, order, mode, offset):
        m = _ionp_open(dtype, n, shape, order, mode, offset)
        out = op(m)
        return (_ionp_snapshot(out), isinstance(out, ap_memmap), out.filename)

    MEMMAP_SPECS[name] = ItemSpec(
        name=name, kind="custom", scalar_like=True,
        numpy_adapter=_numpy, ionp_adapter=_ionp,
        custom_cases=_shape_order_cases,
    )


_copy_op_spec("memmap.copy", lambda m: m.copy())
_copy_op_spec("memmap.astype", lambda m: m.astype("float64" if str(m.dtype) != "float64" else "int64"))


# ---------------------------------------------------------------------------
# Reductions/arithmetic/comparisons: `__array_priority__ == -100.0` downgrades
# every one of these to a plain ndarray result (measured live). scalar_like
# reductions (sum/mean/... on a 1-d/0-d array can return a bare scalar) are
# routed through the same snapshot tagging.
# ---------------------------------------------------------------------------
def _reduce_spec(name, op):
    def _numpy(dtype, n, shape, order, mode, offset):
        m = _np_open(dtype, n, shape, order, mode, offset)
        return _np_snapshot(op(m))

    def _ionp(dtype, n, shape, order, mode, offset):
        m = _ionp_open(dtype, n, shape, order, mode, offset)
        return _ionp_snapshot(op(m))

    MEMMAP_SPECS[name] = ItemSpec(
        name=name, kind="custom", scalar_like=True,
        numpy_adapter=_numpy, ionp_adapter=_ionp,
        custom_cases=lambda: [
            (label, args, kwargs) for label, args, kwargs in _dtype_cross_cases()
            if "bool" not in label or name in (
                "memmap.all", "memmap.any", "memmap.sum", "memmap.min", "memmap.max")
        ] + _shape_order_cases(),
    )


_reduce_spec("memmap.sum", lambda m: m.sum())
_reduce_spec("memmap.mean", lambda m: m.mean())
_reduce_spec("memmap.min", lambda m: m.min())
_reduce_spec("memmap.max", lambda m: m.max())
_reduce_spec("memmap.std", lambda m: m.std())
_reduce_spec("memmap.var", lambda m: m.var())
_reduce_spec("memmap.prod", lambda m: m.prod())
_reduce_spec("memmap.all", lambda m: m.all())
_reduce_spec("memmap.any", lambda m: m.any())


def _binop_spec(name, op):
    def _numpy(dtype, n, shape, order, mode, offset):
        m = _np_open(dtype, n, shape, order, mode, offset)
        out = op(m, 2)
        return (_np_snapshot(out), isinstance(out, np.memmap))

    def _ionp(dtype, n, shape, order, mode, offset):
        m = _ionp_open(dtype, n, shape, order, mode, offset)
        out = op(m, 2)
        return (_ionp_snapshot(out), isinstance(out, ap_memmap))

    numeric_only = [
        (label, args, kwargs) for label, args, kwargs in _dtype_cross_cases()
        if "complex" not in label or name in (
            "memmap.__add__", "memmap.__radd__", "memmap.__sub__", "memmap.__rsub__",
            "memmap.__mul__", "memmap.__rmul__", "memmap.__eq__", "memmap.__ne__",
            "memmap.__neg__", "memmap.__pos__", "memmap.__abs__")
    ]
    MEMMAP_SPECS[name] = ItemSpec(
        name=name, kind="custom", scalar_like=True,
        numpy_adapter=_numpy, ionp_adapter=_ionp,
        custom_cases=lambda: numeric_only,
    )


_binop_spec("memmap.__add__", lambda m, v: m + v)
_binop_spec("memmap.__radd__", lambda m, v: v + m)
_binop_spec("memmap.__sub__", lambda m, v: m - v)
_binop_spec("memmap.__rsub__", lambda m, v: v - m)
_binop_spec("memmap.__mul__", lambda m, v: m * v)
_binop_spec("memmap.__rmul__", lambda m, v: v * m)
_binop_spec("memmap.__mod__", lambda m, v: m % v)
_binop_spec("memmap.__rmod__", lambda m, v: v % m)
_binop_spec("memmap.__floordiv__", lambda m, v: m // v)
_binop_spec("memmap.__rfloordiv__", lambda m, v: v // m)
_binop_spec("memmap.__eq__", lambda m, v: m == v)
_binop_spec("memmap.__ne__", lambda m, v: m != v)
_binop_spec("memmap.__lt__", lambda m, v: m < v)
_binop_spec("memmap.__le__", lambda m, v: m <= v)
_binop_spec("memmap.__gt__", lambda m, v: m > v)
_binop_spec("memmap.__ge__", lambda m, v: m >= v)


def _int_only_cases():
    return [
        (label, args, kwargs) for label, args, kwargs in _dtype_cross_cases()
        if "float" not in label and "complex" not in label
    ]


def _bitop_spec(name, op):
    def _numpy(dtype, n, shape, order, mode, offset):
        m = _np_open(dtype, n, shape, order, mode, offset)
        out = op(m, 1)
        return (_np_snapshot(out), isinstance(out, np.memmap))

    def _ionp(dtype, n, shape, order, mode, offset):
        m = _ionp_open(dtype, n, shape, order, mode, offset)
        out = op(m, 1)
        return (_ionp_snapshot(out), isinstance(out, ap_memmap))

    MEMMAP_SPECS[name] = ItemSpec(
        name=name, kind="custom", scalar_like=True,
        numpy_adapter=_numpy, ionp_adapter=_ionp,
        custom_cases=_int_only_cases,
    )


_bitop_spec("memmap.__and__", lambda m, v: m & v)
_bitop_spec("memmap.__rand__", lambda m, v: v & m)
_bitop_spec("memmap.__or__", lambda m, v: m | v)
_bitop_spec("memmap.__ror__", lambda m, v: v | m)
_bitop_spec("memmap.__xor__", lambda m, v: m ^ v)
_bitop_spec("memmap.__rxor__", lambda m, v: v ^ m)
_bitop_spec("memmap.__lshift__", lambda m, v: m << v)
_bitop_spec("memmap.__rlshift__", lambda m, v: v << m)
_bitop_spec("memmap.__rshift__", lambda m, v: m >> v)
_bitop_spec("memmap.__rrshift__", lambda m, v: v >> m)


def _unary_spec(name, op, int_only=False):
    def _numpy(dtype, n, shape, order, mode, offset):
        m = _np_open(dtype, n, shape, order, mode, offset)
        out = op(m)
        return (_np_snapshot(out), isinstance(out, np.memmap))

    def _ionp(dtype, n, shape, order, mode, offset):
        m = _ionp_open(dtype, n, shape, order, mode, offset)
        out = op(m)
        return (_ionp_snapshot(out), isinstance(out, ap_memmap))

    MEMMAP_SPECS[name] = ItemSpec(
        name=name, kind="custom", scalar_like=True,
        numpy_adapter=_numpy, ionp_adapter=_ionp,
        custom_cases=(_int_only_cases if int_only else lambda: [
            (l, a, k) for l, a, k in _dtype_cross_cases() if "bool" not in l
        ]),
    )


_unary_spec("memmap.__neg__", lambda m: -m)
_unary_spec("memmap.__pos__", lambda m: +m)
_unary_spec("memmap.__abs__", lambda m: abs(m))
_unary_spec("memmap.__invert__", lambda m: ~m, int_only=True)


def _hash_cases():
    return [("hash/int32", ("int32", 6, (6,), "C", "r+", 0), {})]


def _hash_numpy(dtype, n, shape, order, mode, offset):
    m = _np_open(dtype, n, shape, order, mode, offset)
    try:
        hash(m)
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


def _hash_ionp(dtype, n, shape, order, mode, offset):
    m = _ionp_open(dtype, n, shape, order, mode, offset)
    try:
        hash(m)
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


MEMMAP_SPECS["memmap.__hash__"] = ItemSpec(
    name="memmap.__hash__", kind="custom", scalar_like=True,
    numpy_adapter=_hash_numpy, ionp_adapter=_hash_ionp,
    custom_cases=_hash_cases,
)


def _iter_cases():
    return [("iter/1d", ("int32", 6, (6,), "C", "r+", 0), {})]


def _iter_numpy(dtype, n, shape, order, mode, offset):
    m = _np_open(dtype, n, shape, order, mode, offset)
    return [_np_snapshot(x) for x in m]


def _iter_ionp(dtype, n, shape, order, mode, offset):
    m = _ionp_open(dtype, n, shape, order, mode, offset)
    return [_ionp_snapshot(x) for x in m]


MEMMAP_SPECS["memmap.__iter__"] = ItemSpec(
    name="memmap.__iter__", kind="custom", scalar_like=True,
    numpy_adapter=_iter_numpy, ionp_adapter=_iter_ionp,
    custom_cases=_iter_cases,
)


def _item_cases():
    return [("item/scalar", ("int32", 1, (), "C", "r+", 0), {})]


def _item_numpy(dtype, n, shape, order, mode, offset):
    m = _np_open(dtype, n, shape, order, mode, offset)
    return m.item()


def _item_ionp(dtype, n, shape, order, mode, offset):
    m = _ionp_open(dtype, n, shape, order, mode, offset)
    return m.item()


MEMMAP_SPECS["memmap.item"] = ItemSpec(
    name="memmap.item", kind="custom", scalar_like=True,
    numpy_adapter=_item_numpy, ionp_adapter=_item_ionp,
    custom_cases=_item_cases,
)


# ---------------------------------------------------------------------------
# Mutating ops: __setitem__, fill, flush. Each side gets its OWN freshly
# built file (see module docstring), so a mutation on one side can never
# leak into the other's read.
# ---------------------------------------------------------------------------
def _mutate_cases():
    return [
        ("setitem/r+", ("int32", 6, (6,), "C", "r+", 0), {}),
        ("setitem/w+", ("float64", 6, (6,), "C", "w+", 0), {}),
    ]


def _setitem_numpy(dtype, n, shape, order, mode, offset):
    m = _np_open(dtype, n, shape, order, mode, offset)
    m[0] = 111
    m[-1] = 222
    return _np_snapshot(m)


def _setitem_ionp(dtype, n, shape, order, mode, offset):
    m = _ionp_open(dtype, n, shape, order, mode, offset)
    m[0] = 111
    m[-1] = 222
    return _ionp_snapshot(m)


MEMMAP_SPECS["memmap.__setitem__"] = ItemSpec(
    name="memmap.__setitem__", kind="custom", scalar_like=True,
    numpy_adapter=_setitem_numpy, ionp_adapter=_setitem_ionp,
    custom_cases=_mutate_cases,
)


def _setitem_readonly_cases():
    return [("setitem/mode_r", ("int32", 6, (6,), "C", "r", 0), {})]


def _setitem_ro_numpy(dtype, n, shape, order, mode, offset):
    m = _np_open(dtype, n, shape, order, mode, offset)
    try:
        m[0] = 5
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


def _setitem_ro_ionp(dtype, n, shape, order, mode, offset):
    m = _ionp_open(dtype, n, shape, order, mode, offset)
    try:
        m[0] = 5
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


MEMMAP_SPECS["memmap.__setitem__/readonly"] = ItemSpec(
    name="memmap.__setitem__/readonly", kind="custom", scalar_like=True,
    numpy_adapter=_setitem_ro_numpy, ionp_adapter=_setitem_ro_ionp,
    custom_cases=_setitem_readonly_cases,
)


def _fill_numpy(dtype, n, shape, order, mode, offset):
    m = _np_open(dtype, n, shape, order, mode, offset)
    m.fill(9)
    return _np_snapshot(m)


def _fill_ionp(dtype, n, shape, order, mode, offset):
    m = _ionp_open(dtype, n, shape, order, mode, offset)
    m.fill(9)
    return _ionp_snapshot(m)


MEMMAP_SPECS["memmap.fill"] = ItemSpec(
    name="memmap.fill", kind="custom", scalar_like=True,
    numpy_adapter=_fill_numpy, ionp_adapter=_fill_ionp,
    custom_cases=_mutate_cases,
)


def _flush_numpy(dtype, n, shape, order, mode, offset):
    m = _np_open(dtype, n, shape, order, mode, offset)
    m[0] = 111
    m.flush()
    path = m.filename
    del m
    reopened = np.memmap(path, dtype=dtype, mode="r", offset=offset,
                          shape=shape if shape else None, order=order)
    return _np_snapshot(reopened)


def _flush_ionp(dtype, n, shape, order, mode, offset):
    m = _ionp_open(dtype, n, shape, order, mode, offset)
    m[0] = 111
    m.flush()
    path = m.filename
    del m
    reopened = ap_memmap(path, dtype=dtype, mode="r", offset=offset,
                          shape=shape if shape else None, order=order)
    return _ionp_snapshot(reopened)


MEMMAP_SPECS["memmap.flush"] = ItemSpec(
    name="memmap.flush", kind="custom", scalar_like=True,
    numpy_adapter=_flush_numpy, ionp_adapter=_flush_ionp,
    custom_cases=_mutate_cases,
)


def _flush_c_numpy(dtype, n, shape, order, mode, offset):
    m = _np_open(dtype, n, shape, order, mode, offset)
    m[0] = 999
    m.flush()
    path = m.filename
    del m
    reopened = np.memmap(path, dtype=dtype, mode="r", offset=offset,
                          shape=shape if shape else None, order=order)
    return _np_snapshot(reopened)


def _flush_c_ionp(dtype, n, shape, order, mode, offset):
    m = _ionp_open(dtype, n, shape, order, mode, offset)
    m[0] = 999
    m.flush()
    path = m.filename
    del m
    reopened = ap_memmap(path, dtype=dtype, mode="r", offset=offset,
                          shape=shape if shape else None, order=order)
    return _ionp_snapshot(reopened)


MEMMAP_SPECS["memmap.flush/mode_c_no_persist"] = ItemSpec(
    name="memmap.flush/mode_c_no_persist", kind="custom", scalar_like=True,
    numpy_adapter=_flush_c_numpy, ionp_adapter=_flush_c_ionp,
    custom_cases=lambda: [("flush/mode_c", ("int32", 6, (6,), "C", "c", 0), {})],
)


# ---------------------------------------------------------------------------
# Negative / boundary controls -- included for the record even where not
# declared exact (see module docstring's "NOT COVERED" section).
# ---------------------------------------------------------------------------
def _empty_file_cases():
    return [("empty_file", ("int32", 0, (0,), "C", "r+", 0), {})]


def _empty_file_numpy(dtype, n, shape, order, mode, offset):
    path = _make_fixture([], dtype, pad_before=0)
    try:
        np.memmap(path, dtype=dtype, mode="r+", shape=(0,))
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__, str(exc))


def _empty_file_ionp(dtype, n, shape, order, mode, offset):
    path = _make_fixture([], dtype, pad_before=0)
    try:
        ap_memmap(path, dtype=dtype, mode="r+", shape=(0,))
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__, str(exc))


MEMMAP_SPECS["memmap.__new__/empty_file"] = ItemSpec(
    name="memmap.__new__/empty_file", kind="custom", scalar_like=True,
    numpy_adapter=_empty_file_numpy, ionp_adapter=_empty_file_ionp,
    custom_cases=_empty_file_cases,
)


def _invalid_mode_cases():
    return [("invalid_mode", ("int32", 6, (6,), "C", "bogus", 0), {})]


def _invalid_mode_numpy(dtype, n, shape, order, mode, offset):
    values = _fixture_values(dtype, n)
    path = _make_fixture(values, dtype)
    try:
        np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__, str(exc))


def _invalid_mode_ionp(dtype, n, shape, order, mode, offset):
    values = _fixture_values(dtype, n)
    path = _make_fixture(values, dtype)
    try:
        ap_memmap(path, dtype=dtype, mode=mode, shape=shape)
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__, str(exc))


# TICKET #77 (2026-08-08): message-text divergence FIXED -- `__new__` now
# raises numpy's own exact wording (all 8 accepted mode forms enumerated,
# same format). Declared exact in anionpy/_state/memmap.py.
MEMMAP_SPECS["memmap.__new__/invalid_mode"] = ItemSpec(
    name="memmap.__new__/invalid_mode", kind="custom", scalar_like=True,
    numpy_adapter=_invalid_mode_numpy, ionp_adapter=_invalid_mode_ionp,
    custom_cases=_invalid_mode_cases,
)


# ---------------------------------------------------------------------------
# TICKET #77 (2026-08-08): `dtype=` values real numpy's own `np.dtype()`
# cannot parse at all (as opposed to structurally-unsupported-by-memmap
# values it CAN parse, e.g. 'S10'/'V4' -- not covered here, see module
# docstring). `anionpy.dtype()` is already declared exact and already
# raises the matching `TypeError`/message for these; `__new__` now routes
# through it, so this should match by TYPE and MESSAGE TEXT both.
# ---------------------------------------------------------------------------
def _invalid_dtype_cases():
    return [
        ("invalid_dtype/not_a_dtype", ("not_a_dtype", 6, (6,), "C", "r+", 0), {}),
        ("invalid_dtype/int_12345", (12345, 6, (6,), "C", "r+", 0), {}),
        ("invalid_dtype/float_3_14", (3.14, 6, (6,), "C", "r+", 0), {}),
    ]


def _invalid_dtype_numpy(dtype, n, shape, order, mode, offset):
    path = _make_fixture([0] * n, "uint8")
    try:
        np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__, str(exc))


def _invalid_dtype_ionp(dtype, n, shape, order, mode, offset):
    path = _make_fixture([0] * n, "uint8")
    try:
        ap_memmap(path, dtype=dtype, mode=mode, shape=shape)
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__, str(exc))


MEMMAP_SPECS["memmap.__new__/invalid_dtype"] = ItemSpec(
    name="memmap.__new__/invalid_dtype", kind="custom", scalar_like=True,
    numpy_adapter=_invalid_dtype_numpy, ionp_adapter=_invalid_dtype_ionp,
    custom_cases=_invalid_dtype_cases,
)


# ---------------------------------------------------------------------------
# TICKET #77 (2026-08-08): `dtype=` values real numpy's `np.dtype()` CAN
# parse but `anionpy.memmap` has no `_DTYPE_FMT`/`_decode`/`_encode` support
# for at all (structured/string/datetime/object kinds) -- real numpy
# SUCCEEDS constructing these (no validation beyond `np.dtype()` itself);
# `anionpy.memmap` cannot back them (pre-existing, permanent, narrower-than-
# numpy gap, unrelated to the parsing defect this ticket fixes). Kept in
# the corpus for the record; deliberately NOT declared in
# anionpy/_state/memmap.py -- this is expected to MISMATCH.
# ---------------------------------------------------------------------------
def _unsupported_dtype_cases():
    return [
        ("unsupported_dtype/S10", ("S10", 6, (6,), "C", "r+", 0), {}),
        ("unsupported_dtype/U5", ("U5", 6, (6,), "C", "r+", 0), {}),
        ("unsupported_dtype/V4", ("V4", 6, (6,), "C", "r+", 0), {}),
        ("unsupported_dtype/datetime64", ("datetime64[D]", 6, (6,), "C", "r+", 0), {}),
        ("unsupported_dtype/object", (object, 6, (6,), "C", "r+", 0), {}),
    ]


def _unsupported_dtype_numpy(dtype, n, shape, order, mode, offset):
    path = _make_fixture([0] * 256, "uint8")  # oversized, zero-padded
    try:
        np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__,)


def _unsupported_dtype_ionp(dtype, n, shape, order, mode, offset):
    path = _make_fixture(b"\x00" * 256, "uint8")
    try:
        ap_memmap(path, dtype=dtype, mode=mode, shape=shape)
        return "no-raise"
    except Exception as exc:  # noqa: BLE001
        return (type(exc).__name__,)


MEMMAP_SPECS["memmap.__new__/unsupported_dtype"] = ItemSpec(
    name="memmap.__new__/unsupported_dtype", kind="custom", scalar_like=True,
    numpy_adapter=_unsupported_dtype_numpy, ionp_adapter=_unsupported_dtype_ionp,
    custom_cases=_unsupported_dtype_cases,
)
