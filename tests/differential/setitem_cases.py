"""Differential corpus for `ndarray.__setitem__`.

WHY THIS FILE EXISTS. `__setitem__` landed on 2026-08-02 and had no corpus of
its own at all -- it was exercised only incidentally, as the write-through
step of `view_semantics_cases.py`. An item with no test is an unwritten pass,
so it stayed undeclared. This file is its corpus.

WHAT IS COMPARED. Not the return value -- `__setitem__` returns `None` on
both sides and always will. What is compared is a DESCRIPTOR built the same
way on each side:

    ret | root values | root dtype | root shape          (on success)
    raised:<ExcType>:<message>                           (on failure)

The ROOT owner's values, not the indexed view's. For the view kinds below
that is the whole point: an assignment through `a.T` or `a[::2]` that writes
into a private copy instead of the shared buffer leaves the view looking
correct and the root untouched.

Errors are compared with their FULL message and exception type, because that
is where every defect this corpus was built from actually lived. A sweep of
4,320 (dtype x layout x key x value) combinations found 186 divergences and
NOT ONE of them was a wrong stored value -- 132 wrong messages, 42 wrong
exception types, and 12 cases where one side raised and the other did not.
Grading on values alone would have called that implementation finished.

THE FOUR RULES THIS CORPUS PINS. Each was measured, and none is derivable
from the others:

1. SCALAR ASSIGNMENT FAST PATH. A key made only of integers, exactly `ndim`
   of them, goes to the dtype's own scalar coercion on the RAW Python
   object -- no array conversion. Every other key spelling converts and
   broadcasts. On a bool array these disagree about the stored VALUE:

       a[0]      = [0]   ->  True    (bool([0]) is True: truthiness)
       a[(0,)]   = [0]   ->  True
       a[..., 0] = [0]   ->  ValueError (... maximum number of dimension of 0.)
       a[0:1]    = [0]   ->  False   (conversion: asarray([0], bool))
       a[[0]]    = [0]   ->  False

   Rows 1 and 4 address the same element with the same object and store
   opposite values. anionpy raised on all three sequence rows before this.

2. SEQUENCE VALUES AND ARRAY VALUES OBEY DIFFERENT RANK RULES. numpy caps
   the rank of the array it builds from a sequence at the destination's
   rank; an already-built array is only broadcast, so leading length-1 axes
   are droppable:

       a = zeros((3, 4))
       a[0] = ones((1, 4))       OK
       a[0] = ones((1, 1, 4))    OK
       a[0] = [[1., 1., 1., 1.]] ValueError (... maximum number of dimension of 1.)

   Same rank, same contents, different answer. anionpy rejected all three.

3. THE LONE FULL-RANK BOOLEAN MASK HAS ITS OWN TWO ERRORS. `a[mask]` and
   `a[mask,]` report "NumPy boolean array indexing assignment ..."; a mask
   covering only some axes, or one reached through `a[..., mask]`, reports
   the generic "shape mismatch". The rank complaint is a TypeError and comes
   FIRST -- `a[mask] = [[1, 2]]` is rejected even though (1,2) would
   broadcast onto the two selected elements perfectly well.

4. VALUE VALIDATION PRECEDES INDEX BOUNDS CHECKING.

       np.zeros(3)[[9]] = [1., 2.]
         -> ValueError: shape mismatch: value array of shape (2,) could not
            be broadcast to indexing result of shape (1,)

   Both that and `IndexError: index 9 is out of bounds` are true statements
   about the call; numpy just reaches the value one first. anionpy reached the
   index one first. (The first characterisation of this said "empty arrays",
   because every instance in the sweep that raised it happened to have a
   0-length axis. The repro above has no empty axis. It was the sweep's
   input shapes that were narrow, not the rule.)

INPUTS ARE BUILT NATIVELY ON EACH SIDE, never ingested across the boundary:
`mod.array(<nested list>)` and then the side's OWN slicing/transpose for the
view kinds. Handing an already-strided numpy array to `anionpy.array()` silently
re-contiguises it, so a foreign view can never reach the strided write path
that these cases exist to test.

DELIBERATELY NOT COVERED: `object` dtype and string dtypes (anionpy has no
ingestion for either -- `anionpy.array(['x'])` raises), and `range` as a value
(anionpy routes it to the scalar path where numpy converts it as a sequence).
Both are pre-existing absences recorded elsewhere, not properties of
`__setitem__`, and a case that asserted the current behaviour of either would
be pinning a gap in place.
"""
from __future__ import annotations

import numpy as np

from registry import REGISTRY, ItemSpec


def _nested(shape, dt="float64"):
    n = int(np.prod(shape)) if shape else 1
    return np.arange(n, dtype=dt).reshape(shape).tolist()


# (label) -> (nested payload, view op applied on each side's own array)
_KINDS = {
    "1d": (_nested((6,)), None),
    "2d": (_nested((3, 4)), None),
    "3d": (_nested((2, 3, 4)), None),
    "1x5": (_nested((1, 5)), None),
    "5x1": (_nested((5, 1)), None),
    # Non-contiguous destinations. An assignment that quietly writes into a
    # compacted copy is invisible unless the ROOT is read back afterwards,
    # which is what the descriptor does.
    "1d_step": (_nested((6,)), "step"),
    "1d_rev": (_nested((6,)), "rev"),
    "2d_T": (_nested((3, 4)), "T"),
    "3d_sl": (_nested((2, 3, 4)), "slice"),
}


def _build(mod, kind, dtype):
    payload, op = _KINDS[kind]
    root = mod.array(payload, dtype=dtype)
    if op is None:
        return root, root
    if op == "T":
        return root.T, root
    if op == "step":
        return root[::2], root
    if op == "rev":
        return root[::-1], root
    if op == "slice":
        return root[:, ::2], root
    raise AssertionError(op)


# Keys. Each is built on the side under test, so an array-valued key is that
# side's own array -- never a numpy array handed to anionpy.
_KEYS = {
    # -- rule 1: the fast path and its near neighbours, which must NOT take it
    "int": lambda m, a: 0,
    "int_neg": lambda m, a: -1,
    "int_tuple": lambda m, a: tuple(0 for _ in range(a.ndim)),
    "ell_int": lambda m, a: (Ellipsis, 0),
    "slice_1": lambda m, a: slice(0, 1),
    "fancy_1": lambda m, a: m.array([0], dtype="int64"),
    # -- basic keys
    "all": lambda m, a: slice(None),
    "step2": lambda m, a: slice(None, None, 2),
    "rev": lambda m, a: slice(None, None, -1),
    "empty": lambda m, a: slice(2, 2),
    "oob_slice": lambda m, a: slice(0, 99),
    "newaxis": lambda m, a: (Ellipsis, None),
    "int_slice": lambda m, a: (0, slice(None)) if a.ndim >= 2 else 0,
    # -- advanced keys
    "fancy": lambda m, a: m.array([0, 1], dtype="int64"),
    "fancy_dup": lambda m, a: m.array([0, 0, 1], dtype="int64"),
    "fancy_neg": lambda m, a: m.array([-1, 0], dtype="int64"),
    "fancy_oob": lambda m, a: m.array([99], dtype="int64"),
    "fancy_2d": lambda m, a: m.array([[0, 0], [0, 0]], dtype="int64"),
    "fancy_slice": (
        lambda m, a: (m.array([0, 1], dtype="int64"), slice(None))
        if a.ndim >= 2
        else m.array([0, 1], dtype="int64")
    ),
    # -- rule 3: a mask over EVERY axis vs one over some
    "mask_full": lambda m, a: m.array(
        (np.arange(a.size).reshape(a.shape) % 2 == 0).tolist(), dtype="bool"
    ),
    "mask_axis0": lambda m, a: m.array(
        [i % 2 == 0 for i in range(a.shape[0])], dtype="bool"
    ),
    "mask_ell": lambda m, a: (
        Ellipsis,
        m.array([i % 2 == 0 for i in range(a.shape[-1])], dtype="bool"),
    ),
    # -- rejected keys
    "float_key": lambda m, a: 1.5,
    "str_key": lambda m, a: "x",
    "too_many": lambda m, a: (0,) * (a.ndim + 2),
    "oob_int": lambda m, a: 99,
}

# Values. The list/tuple entries and the array entries are deliberately
# PAIRED at the same shapes -- that pairing is what makes rule 2 visible.
_VALUES = {
    "scalar_int": lambda m, a: 7,
    "scalar_float": lambda m, a: -2.5,
    "scalar_bool": lambda m, a: True,
    "scalar_complex": lambda m, a: 1 + 2j,
    "none": lambda m, a: None,
    "big": lambda m, a: 300,
    "list_1": lambda m, a: [1.5],
    # THE DISAMBIGUATOR. `[1.5]` above cannot witness rule 1 on a bool
    # destination: the fast path (truthiness, `bool([1.5])`) and the general
    # path (conversion, `asarray([1.5], bool)`) both say True, so a corpus
    # holding only truthy one-element sequences documents the split without
    # being able to detect it. `[0]` separates them -- truthiness says True
    # (a non-empty list), conversion says False (the element is zero):
    #     a[0]   = [0]  ->  True    (fast path)
    #     a[0:1] = [0]  ->  False   (general path)
    # Added after a mutation run showed the fast path could be disabled
    # outright with a smaller corpus response than the rule deserved.
    "list_zero": lambda m, a: [0],
    "list_zero_f": lambda m, a: [0.0],
    "tuple_zero": lambda m, a: (0,),
    "list_3": lambda m, a: [1.0, 2.0, 3.0],
    "list_nested": lambda m, a: [[1.0, 2.0]],
    "list_empty": lambda m, a: [],
    "tuple_1": lambda m, a: (1.5,),
    "arr_0d": lambda m, a: m.array(9.0, dtype="float64"),
    "arr_1": lambda m, a: m.array([1.5], dtype="float64"),
    "arr_3": lambda m, a: m.array([1.0, 2.0, 3.0], dtype="float64"),
    "arr_lead1": lambda m, a: m.array([[[7.0]]], dtype="float64"),
    "arr_1xN": lambda m, a: m.array([[1.0, 2.0, 3.0, 4.0]], dtype="float64"),
    "arr_int": lambda m, a: m.array([1, 2, 3], dtype="int64"),
    "arr_bool": lambda m, a: m.array([True, False], dtype="bool"),
    # Self-assignment: the destination and the value share a buffer, so an
    # implementation that writes element by element without a copy corrupts
    # its own source partway through.
    "self": lambda m, a: a,
    "self_rev": lambda m, a: a[::-1] if a.ndim == 1 else a.T,
    # Objects that are NOT numbers and NOT sequences. numpy's float setitem
    # substitutes a sequence message for list/tuple but lets float()'s own
    # TypeError out for these -- the substitution is keyed on sequence-ness,
    # not on "conversion failed".
    "dict": lambda m, a: {1: 2},
    "object": lambda m, a: object(),
    "str": lambda m, a: "abc",
}

_DTYPES = ("float64", "int64", "complex128", "bool", "uint8", "float32")


class SetProbe:
    """Sentinel: "build this input on your own side, run this assignment,
    and describe what happened." Carried through the adapters the same way
    `view_semantics_cases.py`'s `ViewProbe` is."""

    __slots__ = ("kind", "dtype", "key", "value")

    def __init__(self, kind, dtype, key, value):
        self.kind = kind
        self.dtype = dtype
        self.key = key
        self.value = value

    def __repr__(self):
        return (
            f"SetProbe({self.kind!r}, {self.dtype!r}, {self.key!r}, {self.value!r})"
        )


def _flat(v):
    out = v.tolist()
    while isinstance(out, list) and out and isinstance(out[0], list):
        out = [x for sub in out for x in sub]
    return out if isinstance(out, list) else [out]


def _probe(mod, p):
    try:
        a, root = _build(mod, p.kind, p.dtype)
        key = _KEYS[p.key](mod, a)
        value = _VALUES[p.value](mod, a)
    except BaseException as exc:  # pragma: no cover - a broken case, not a result
        return f"setup:{type(exc).__name__}:{exc}"
    try:
        ret = a.__setitem__(key, value)
    except BaseException as exc:
        # `BaseException`, not `Exception`: PyO3 raises `PanicException`,
        # which derives from BaseException, and a bare `except Exception`
        # would let a panic escape as a harness error instead of a FAIL.
        return f"raised:{type(exc).__name__}:{exc}"
    # The ROOT, not the view that was written through.
    return f"ret={ret!r}|root={_flat(root)}|dt={root.dtype.name}|shape={tuple(root.shape)}"


def _numpy_adapter(arg, *rest, **kwargs):
    assert isinstance(arg, SetProbe), arg
    return _probe(np, arg)


def _ionp_adapter(arg, *rest, **kwargs):
    import anionpy

    assert isinstance(arg, SetProbe), arg
    return _probe(anionpy, arg)


# Which (key, value) pairs are worth crossing with which dtypes. A full
# 9x6x27x24 cross product is 35,000 cases and would swamp the suite for no
# extra signal; these groups are chosen so that every rule in the module
# docstring is exercised on every dtype that can express it.
_FAST_PATH_VALUES = (
    "list_1", "list_zero", "list_zero_f", "tuple_zero",
    "list_3", "tuple_1", "list_empty", "arr_0d", "arr_1", "arr_3",
    "scalar_int", "scalar_float", "scalar_bool", "scalar_complex", "none",
    "dict", "object", "str", "big",
)
_FAST_PATH_KEYS = ("int", "int_neg", "int_tuple", "ell_int", "slice_1", "fancy_1")

_RANK_VALUES = ("arr_lead1", "arr_1xN", "list_nested", "arr_1", "list_1", "arr_3",
                "list_3", "self", "self_rev")
_RANK_KEYS = ("int", "all", "int_slice", "step2", "rev", "newaxis", "ell_int")

_ADV_KEYS = ("fancy", "fancy_dup", "fancy_neg", "fancy_oob", "fancy_2d",
             "fancy_slice", "mask_full", "mask_axis0", "mask_ell")
_ADV_VALUES = ("scalar_int", "list_3", "arr_3", "arr_1", "list_nested",
               "list_empty", "self", "arr_int", "arr_bool")

_REJECT_KEYS = ("float_key", "str_key", "too_many", "oob_int", "oob_slice", "empty")
_REJECT_VALUES = ("scalar_int", "list_3", "arr_3")


def _cases():
    out = []
    seen = set()

    def add(kind, dtype, key, value):
        label = f"set/{kind}/{dtype}/{key}/{value}"
        if label in seen:
            return
        seen.add(label)
        out.append((label, (SetProbe(kind, dtype, key, value),), {}))

    # Rule 1: every dtype, because the fast path's answer is dtype-dispatched
    # (bool coerces by truthiness, int/complex let their constructor's
    # TypeError out, float substitutes a sequence message).
    for dtype in _DTYPES:
        for key in _FAST_PATH_KEYS:
            for value in _FAST_PATH_VALUES:
                add("1d", dtype, key, value)
        for value in _FAST_PATH_VALUES:
            add("2d", dtype, "int_tuple", value)

    # Rule 2: rank rules, on contiguous and non-contiguous destinations alike.
    for kind in ("2d", "3d", "2d_T", "3d_sl", "1x5", "5x1"):
        for key in _RANK_KEYS:
            for value in _RANK_VALUES:
                add(kind, "float64", key, value)
    for kind in ("1d", "1d_step", "1d_rev"):
        for key in ("all", "step2", "rev", "empty", "oob_slice"):
            for value in _RANK_VALUES:
                add(kind, "float64", key, value)

    # Rules 3 and 4: advanced keys. Two dtypes, because the mask messages
    # quote element COUNTS and the casting behaviour differs.
    for kind in ("1d", "2d", "3d", "1d_step", "2d_T", "3d_sl"):
        for dtype in ("float64", "int64"):
            for key in _ADV_KEYS:
                for value in _ADV_VALUES:
                    add(kind, dtype, key, value)

    # Rejected keys.
    for kind in ("1d", "2d", "2d_T"):
        for key in _REJECT_KEYS:
            for value in _REJECT_VALUES:
                add(kind, "float64", key, value)

    # Casting, on the dtypes where an assignment can lose or wrap
    # information. `big` into uint8 and `scalar_complex` into a real dtype
    # are the two that must raise rather than silently truncate.
    for dtype in ("uint8", "int64", "float32", "bool", "complex128"):
        for key in ("all", "int", "fancy", "mask_axis0"):
            for value in ("big", "scalar_complex", "scalar_float", "arr_int",
                          "scalar_bool", "none", "arr_3"):
                add("1d", dtype, key, value)

    return out


def _install():
    REGISTRY["ndarray.__setitem__"] = ItemSpec(
        name="ndarray.__setitem__",
        kind="custom",
        numpy_adapter=_numpy_adapter,
        ionp_adapter=_ionp_adapter,
        scalar_like=True,
        custom_cases=_cases,
    )


_install()
