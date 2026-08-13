"""Differential ItemSpecs for TICKET #38 (object dtype, marker-only scope).

`ionp_core::DType` has no `Object` variant at all -- deliberately: it is
matched exhaustively (no wildcard arm) at dozens of sites across
`ufunc.rs`/`matmul.rs`, files this ticket does not own and is forbidden from
editing (owned by two other concurrently-running agents). Adding a variant
there would force edits into those files. Instead `np.dtype('O')` is tracked
as a SEPARATE sentinel on `PyDType`'s EXISTING `spelling: Option<char>`
field (`spelling == Some('O')`) -- see `ionp-py/src/lib.rs`'s
`PyDType::new` doc comment for the full design rationale, including why an
earlier draft (`is_object: bool`, a genuinely new struct field) had to be
abandoned: it broke the forbidden file `ionp-py/src/array_protocol.rs`'s
own 2-field `PyDType { inner, spelling: None }` literal.

SCOPE, stated plainly (this is the line drawn, not an oversight): the
free-standing `anionpy.dtype('O')`/`anionpy.dtype(object)`/
`anionpy.dtype('object')`/`anionpy.dtype('object_')` OBJECT ITSELF --
construction, `.name`/`.kind`/`.char`/`.itemsize`/`.alignment`/`.num`/
`.str`/`.byteorder`/`.hasobject`, `repr`/`str`, `__eq__`/`__hash__`,
`__lt__`/`__le__`/`__gt__`/`__ge__` safe-cast ordering (including against a
GENUINE `numpy.dtype('O')` instance, not just against another anionpy
`dtype`) -- is in scope and verified live against real numpy 2.5.1 below.
Actually STORING an object-dtype value anywhere (array construction,
`astype`, ufuncs, `min_scalar_type`'s own return value being usable as a
real dtype for buffer allocation) is explicitly NOT in scope: `zeros`/
`array`/`astype` unconditionally reject `dtype='O'`/`dtype=object` with
`TypeError`, unchanged by this ticket, and `object_storage_gap_cases` below
pins that as an intentional, visible FAIL rather than hiding it by
omission. Per the task brief's own bar ("declining to declare a suspect
item is a SUCCESS, not a shortfall"), `anionpy.zeros(3, dtype=object)` etc.
staying unsupported is the correct, honest outcome of this scope line, not
a bug.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from registry import ItemSpec

OBJECT_DTYPE_SPECS: dict[str, ItemSpec] = {}


# ---------------------------------------------------------------------------
# dtype/object: construction + full property surface, across every spelling
# real numpy accepts for object dtype (verified live: 'O', 'object',
# 'object_' strings, and the builtin `object` type by identity -- NOT
# `np.object_`, which anionpy has no scalar type for at all, unrelated gap;
# NOT the numpy-2.0-removed 'object0' alias, which real numpy itself now
# rejects).
# ---------------------------------------------------------------------------

def _np_object_probe(spelling):
    d = np.dtype(spelling)
    return (d.name, d.kind, d.char, d.itemsize, d.alignment, d.num, d.str,
            d.hasobject, d.byteorder, repr(d), str(d))


def _ionp_object_probe(spelling):
    import anionpy
    d = anionpy.dtype(spelling)
    return (d.name, d.kind, d.char, d.itemsize, d.alignment, d.num, d.str,
            d.hasobject, d.byteorder, repr(d), str(d))


def object_dtype_cases():
    return [
        ("spelling_O", ("O",), {}),
        ("spelling_object", ("object",), {}),
        ("spelling_object_", ("object_",), {}),
        ("builtin_object_type", (object,), {}),
    ]


OBJECT_DTYPE_SPECS["dtype/object"] = ItemSpec(
    name="dtype/object", kind="custom", custom_cases=object_dtype_cases,
    numpy_adapter=_np_object_probe, ionp_adapter=_ionp_object_probe,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# dtype/object_eq: equality + safe-cast ordering, both directions, against
# BOTH an anionpy dtype and a genuine numpy dtype instance. The
# numpy-instance direction is not a hypothetical: a live probe during this
# ticket found `anionpy.dtype('O') == np.dtype('O')` returning `False`
# (should be `True`) because `PyDType.__eq__`'s object-dtype branch only
# recognized anionpy's own spellings/identity, never a real numpy dtype
# object handed to it as an operand -- fixed in `lib.rs`'s
# `other_is_object` helper, this case is the regression guard.
# ---------------------------------------------------------------------------

def _np_object_eq_probe(op, other):
    d = np.dtype("O")
    if op == "eq":
        return bool(d == other)
    if op == "eq_rev":
        return bool(other == d)
    if op == "lt":
        return bool(d < other)
    if op == "gt":
        return bool(other < d)
    raise ValueError(op)


def _ionp_object_eq_probe(op, other):
    import anionpy
    d = anionpy.dtype("O")
    if op == "eq":
        return bool(d == other)
    if op == "eq_rev":
        return bool(other == d)
    if op == "lt":
        return bool(d < other)
    if op == "gt":
        return bool(other < d)
    raise ValueError(op)


def object_dtype_eq_cases():
    return [
        ("eq_vs_ionp_object", ("eq", np.dtype("O")), {}),
        ("eq_vs_numpy_object", ("eq_rev", np.dtype("O")), {}),
        ("eq_vs_numpy_int64", ("eq", np.dtype("int64")), {}),
        ("neq_vs_numpy_int64_rev", ("eq_rev", np.dtype("int64")), {}),
        ("lt_object_vs_int64", ("lt", np.dtype("int64")), {}),  # object < int64 is False (nothing widens OUT of object)
        ("gt_int64_vs_object", ("gt", np.dtype("int64")), {}),  # int64 < object is True (reflected: object > int64)
    ]


OBJECT_DTYPE_SPECS["dtype/object_eq"] = ItemSpec(
    name="dtype/object_eq", kind="custom", custom_cases=object_dtype_eq_cases,
    numpy_adapter=_np_object_eq_probe, ionp_adapter=_ionp_object_eq_probe,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# min_scalar_type's own out-of-i64/u64-range gap (dtypeinfo_cases.py's
# `min_scalar_type_cases` already exercises these two values as part of its
# main corpus -- this item independently pins the exact object-dtype
# construction path they route through, `PyDType { spelling: Some('O') }`
# built in `ionp-py/src/dtypeinfo.rs`'s `min_scalar_type`, so a regression
# there shows up here even if it were ever removed from the main sweep).
# ---------------------------------------------------------------------------

def _np_min_scalar_overflow_probe(v):
    d = np.min_scalar_type(v)
    return (d.name, d.kind)


def _ionp_min_scalar_overflow_probe(v):
    import anionpy
    d = anionpy.min_scalar_type(v)
    return (d.name, d.kind)


def min_scalar_type_overflow_cases():
    return [
        ("below_i64_min", (-9223372036854775809,), {}),
        ("above_u64_max", (18446744073709551616,), {}),
    ]


OBJECT_DTYPE_SPECS["dtype/object_via_min_scalar_type"] = ItemSpec(
    name="dtype/object_via_min_scalar_type", kind="custom",
    custom_cases=min_scalar_type_overflow_cases,
    numpy_adapter=_np_min_scalar_overflow_probe,
    ionp_adapter=_ionp_min_scalar_overflow_probe,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# object_storage_gap: documented, PERMANENT, out-of-scope divergence.
# Included so it shows as an honest, visible FAIL rather than being hidden
# by omission -- see this file's module docstring. A "fail" verdict here is
# the expected, correct outcome of the marker-only scope, not a defect.
# ---------------------------------------------------------------------------

def _np_object_storage_probe(kind):
    if kind == "zeros":
        return np.zeros(3, dtype=object).dtype.name
    if kind == "array":
        return np.array([1, "a", 3.0], dtype=object).dtype.name
    if kind == "astype":
        return np.array([1, 2, 3]).astype(object).dtype.name
    raise ValueError(kind)


def _ionp_object_storage_probe(kind):
    import anionpy
    if kind == "zeros":
        return anionpy.zeros(3, dtype=object).dtype.name
    if kind == "array":
        return anionpy.array([1, "a", 3.0], dtype=object).dtype.name
    if kind == "astype":
        return anionpy.array([1, 2, 3]).astype(object).dtype.name
    raise ValueError(kind)


def object_storage_gap_cases():
    return [
        ("zeros_object_gap", ("zeros",), {}),
        ("array_object_gap", ("array",), {}),
        ("astype_object_gap", ("astype",), {}),
    ]


OBJECT_DTYPE_SPECS["dtype/object_storage_gap"] = ItemSpec(
    name="dtype/object_storage_gap", kind="custom",
    custom_cases=object_storage_gap_cases,
    numpy_adapter=_np_object_storage_probe,
    ionp_adapter=_ionp_object_storage_probe,
    scalar_like=True,
)
