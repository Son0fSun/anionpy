"""anionpy.polynomial.{Chebyshev,Legendre,Hermite,HermiteE,Laguerre,Polynomial}
-- the 17 generic `object`/`abc.ABC`-mechanics dunder names ticket #34's
brief measured as ALREADY MATCHING real numpy, plus `__init_subclass__`
(18 names x 6 classes = 108 items), verified out of corpus in this task's
own probe (`/private/tmp/probe34_full.py`) before a single line of this
corpus was written -- see docs/TICKET-34-DUNDERS-CORRECTION-2026-08-08.md
for the full measurement writeup and the correction to `195107b`'s "120
items structurally unmatchable" classification (it was ~12, not ~120).

NEW FILE (permitted, same "one dict[str, ItemSpec] merged into
registry.REGISTRY" pattern as every other `*_cases.py`). Deliberately
separate from `abc_poly_class_cases.py` (covers only the 5 non-Polynomial
classes) and `polynomial_cases.py` (owns `Polynomial`'s arithmetic
dunders): this file is the one place that needs simultaneous access to
ALL SIX numpy/anionpy class pairs at once (cross-basis ordering-dunder
cases: `Chebyshev([1]) < Legendre([1])`), so it is generic over the class
list rather than being bolted onto either existing per-basis-family file.

No `.py` numeric loop anywhere below: every item here is pure Python
object-protocol probing (attribute get/set/delete, pickling-tuple shape,
`dir()` membership, ordering-exception message text) -- there is no
arithmetic to route through anionpy at all, matching this ticket's
"string formatting/object protocol is not arithmetic" carve-out.

WHAT IS DECLARED, and what is deliberately EXCLUDED (see also the
per-item comments below):

  - __lt__/__le__/__gt__/__ge__: `ABCPolyBase` defines none of these, so
    Python's default `object` behaviour applies on both sides -- always
    raises `TypeError` with a message built from `type(a).__name__` and
    `type(b).__name__` (NOT the module-qualified name), so the message is
    bit-exact even though `__module__` differs. Corpus varies the RHS
    operand across: int, float, str, None, a bare `object()`, and -- the
    case a fixed-data corpus would never produce -- an instance of a
    DIFFERENT basis class (`Chebyshev([1]) < Legendre([1])`), checked to
    confirm the raised message names the two RIGHT types, not just "a
    TypeError happened".
    EXCLUDED: comparison against a raw `ndarray` operand
    (`Chebyshev([1]) < anionpy.array([1,2])`). Measured divergence, NOT
    declared, NOT worked around (out of this ticket's edit scope --
    `anionpy.ndarray`'s own comparison dunders, not `anionpy/polynomial/`):
    real numpy raises `TypeError: '<' not supported between instances of
    'Chebyshev' and 'numpy.ndarray'` (both sides return `NotImplemented`,
    Python's default fallback fires) but anionpy raises
    `TypeError: unsupported operand type(s): anionpy.ndarray comparison
    requires another anionpy.ndarray, a numpy scalar/array, or a Python
    bool/int/float/complex (got Chebyshev)` -- `anionpy.ndarray`'s own
    reflected comparison dunder raises instead of returning
    `NotImplemented` for an incompatible type, so Python's normal
    NotImplemented-fallback-to-default-TypeError protocol never gets a
    chance to fire. This is a real, reproducible bug in
    `anionpy.ndarray`'s comparison dunders (core Rust/PyO3 surface, not
    polynomial code), found BY varying the request per this ticket's own
    method, and is exactly the kind of finding that method exists to
    catch. Filed for whoever owns `anionpy.ndarray`'s comparison dunders
    next; not fixed here (out of this ticket's edit scope, and touching
    `ionp-py/src/` is expressly forbidden this session -- ticket #70 has
    in-flight edits there).
  - __setattr__ (existing attr rebind + brand-new attr) / __delattr__
    (present attr + absent attr): `ABCPolyBase` has no `__slots__` and no
    property-based attribute protection, so both operations succeed
    identically on both sides. `AttributeError` on deleting an absent
    attribute is checked for message-text parity too (it embeds the
    attribute name, not any type/module info, so it matches bit-exact).
  - __new__: called directly (bypassing `__init__`) both with and without
    the class as sole argument -- confirms both sides accept the bare
    `cls.__new__(cls)` calling convention.
  - __dir__: MEMBERSHIP, not full-list equality (the brief's own
    instruction -- the full list legitimately differs, since the two
    packages have different internal helper attributes). Checked: `dir()`
    returns a `list` on both sides, and a fixed set of real public API
    names (`coef`, `domain`, `window`, `degree`, `copy`, `deriv`, `integ`,
    `roots`, `trim`, `convert`) that MUST appear in both.
  - __reduce__ / __reduce_ex__ (protocols 0 through 5): the returned
    tuple's SHAPE (reconstructor function identity by name, embedded
    class by `__name__`, arg-count, and -- for protocol >= 2 -- the full
    state dict, which is plain coef/domain/window/symbol data with no
    module-path content) is checked directly, not just "did it raise".
    Measured (see this ticket's report): the returned tuples are IDENTICAL
    between numpy and anionpy at every protocol 0-5 except for the
    embedded class object itself (`numpy.polynomial.chebyshev.Chebyshev`
    vs `anionpy.polynomial.chebyshev.Chebyshev` -- different objects by
    construction, same `__module__` divergence as the Part C items), so
    the corpus below compares by NAME/SHAPE (the matchable property) and
    records the module-path piece as excluded, rather than asserting
    tuple `==` (which would spuriously fail on class identity alone).
  - __subclasshook__: both sides return `NotImplemented` when called
    directly on the class (default `abc.ABC`/`object` behaviour, never
    overridden by `ABCPolyBase`) -- checked for both classes AND cross-
    class (`Chebyshev.__subclasshook__(Legendre)`).
  - __getattribute__: called directly (not merely via `getattr`) to
    confirm the raw dunder itself is invocable and returns the right
    value, per the brief's "read the attribute off the object under test"
    instruction.
  - __sizeof__: measured BIT-EXACT (not merely "both positive ints") --
    both sides report exactly 16 bytes for every coefficient array tested
    (int/float/complex/single-element), because `ABCPolyBase` instances on
    both sides have no `__slots__`/C-level fixed fields contributing extra
    base size beyond the default `object` header; the `__dict__`/coef
    array itself is NOT counted by `object.__sizeof__` (that is
    `sys.getsizeof`'s job, which additionally adds `__dict__` overhead --
    not used here, deliberately: `__sizeof__` is the raw dunder under
    test, and its own contract already matches without needing
    `getsizeof`'s dict-inclusion behaviour).
  - __weakref__: both sides support `weakref.ref(instance)` without
    raising (neither class defines `__slots__` without `__weakref__`).
  - __slots__ / __abstractmethods__ / __static_attributes__: class-level
    attributes checked for VALUE equality (not merely presence) -- all
    three are bit-identical empty collections (`()`, `frozenset()`, `()`)
    on both sides, verified directly against the type object (not an
    instance), per the brief's specific claim.
  - __init_subclass__: NOT in the brief's original 17-name list, but
    investigated per this ticket's Part A/C boundary-drawing instruction
    (the brief flagged it as "differs only by object address" and asked
    for it to be checked). Measured: calling it directly
    (`Chebyshev.__init_subclass__()`) returns `None` identically on both
    sides, and calling it with an unexpected positional argument raises
    `TypeError` with a BIT-EXACT message on both sides (the message uses
    `type.__qualname__` scoped to the immediate class, e.g.
    "Chebyshev.__init_subclass__() takes no arguments (1 given)" -- no
    module path embedded at all). Declared as its own item below.

See docs/TICKET-34-DUNDERS-CORRECTION-2026-08-08.md for the full
per-name measurement table and the correction to `195107b`'s "120 items
structurally unmatchable" claim.
"""
from __future__ import annotations

import weakref

import numpy as np
import numpy.polynomial as npoly

import anionpy as _anionpy
import anionpy.polynomial as apoly

from registry import ItemSpec

_NAMES = ["Chebyshev", "Legendre", "Hermite", "HermiteE", "Laguerre", "Polynomial"]
_NP = {n: getattr(npoly, n) for n in _NAMES}
_AP = {n: getattr(apoly, n) for n in _NAMES}

FLOAT_C = [1.0, 2.0, 3.0]


def _std(pairs):
    return list(pairs)


def _n(cls_name, item):
    return f"polynomial.{cls_name}.{item}"


# ---------------------------------------------------------------------------
# __lt__ / __le__ / __gt__ / __ge__
# ---------------------------------------------------------------------------

def _other_operand(kind, side):
    """`side` is 'np' or 'ap' -- picks which package's class to build a
    cross-basis operand from. Non-poly kinds are side-independent."""
    classes = _NP if side == "np" else _AP
    if kind == "int":
        return 1
    if kind == "float":
        return 1.0
    if kind == "str":
        return "x"
    if kind == "none":
        return None
    if kind == "object":
        return object()
    return classes[kind]([9.0])  # a basis-class name, e.g. "Legendre"


def _mk_ordering_adapter(op_name, cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]
    op = {
        "__lt__": lambda a, b: a < b,
        "__le__": lambda a, b: a <= b,
        "__gt__": lambda a, b: a > b,
        "__ge__": lambda a, b: a >= b,
    }[op_name]

    def adapter(other_kind):
        p = cls(FLOAT_C)
        other = _other_operand(other_kind, side)
        return bool(op(p, other))  # never actually reached -- always raises
    return adapter


def _ordering_cases():
    other_basis = {n: ("Legendre" if n != "Legendre" else "Chebyshev") for n in _NAMES}
    kinds = ["int", "float", "str", "none", "object"]

    def build_for(cls_name):
        ks = kinds + [other_basis[cls_name]]
        return _std([(k, (k,), {}) for k in ks])
    return build_for


_ordering_case_builder = _ordering_cases()


# ---------------------------------------------------------------------------
# __setattr__ / __delattr__
# ---------------------------------------------------------------------------

def _mk_setattr_existing(cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]
    arr_mod = np if side == "np" else _anionpy

    def adapter():
        p = cls(FLOAT_C)
        p.coef = arr_mod.array([9.0, 9.0])
        return p.coef.tolist()
    return adapter


def _mk_setattr_new(cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter():
        p = cls(FLOAT_C)
        p.brand_new_attr_xyz = 42
        return p.brand_new_attr_xyz
    return adapter


def _mk_delattr(cls_name, side, present):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter():
        p = cls(FLOAT_C)
        if present:
            p.brand_new_attr_xyz = 1
            del p.brand_new_attr_xyz
            return "deleted"
        del p.totally_absent_attr_xyz  # always raises AttributeError
        return "unreachable"
    return adapter


# ---------------------------------------------------------------------------
# __new__
# ---------------------------------------------------------------------------

def _mk_new(cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter():
        o = cls.__new__(cls)
        return type(o).__name__
    return adapter


# ---------------------------------------------------------------------------
# __dir__ (membership, not full-list equality)
# ---------------------------------------------------------------------------

_EXPECTED_MEMBERS = ["coef", "domain", "window", "degree", "copy", "deriv",
                      "integ", "roots", "trim", "convert"]


def _mk_dir(cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter():
        p = cls(FLOAT_C)
        d = dir(p)
        assert isinstance(d, list), f"dir() returned {type(d)}, not list"
        return sorted(m for m in _EXPECTED_MEMBERS if m in d)
    return adapter


# ---------------------------------------------------------------------------
# __reduce__ / __reduce_ex__ -- shape comparison (name/arity), not identity
# ---------------------------------------------------------------------------

def _reduce_shape(result):
    """(reconstructor __name__, embedded-class __name__, arg count,
    sorted state-dict items with array values -> lists) -- the matchable
    projection of a __reduce__/__reduce_ex__ tuple, deliberately excluding
    the embedded class OBJECT itself (module-qualified, differs by
    construction -- see module docstring)."""
    fn, fnargs = result[0], result[1]
    state = result[2] if len(result) > 2 else None
    cls_in_args = fnargs[0].__name__ if fnargs else None
    if isinstance(state, dict):
        norm_state = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in sorted(state.items())
        }
    else:
        norm_state = state
    tail = result[3:] if len(result) > 3 else ()
    return (fn.__name__, cls_in_args, len(fnargs), norm_state, tail)


def _mk_reduce_ex(cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter(proto):
        p = cls(FLOAT_C)
        return _reduce_shape(p.__reduce_ex__(proto))
    return adapter


def _mk_reduce(cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter():
        p = cls(FLOAT_C)
        return _reduce_shape(p.__reduce__())
    return adapter


# ---------------------------------------------------------------------------
# __subclasshook__
# ---------------------------------------------------------------------------

def _mk_subclasshook(cls_name, side, other_cls_name):
    classes = _NP if side == "np" else _AP
    cls = classes[cls_name]
    other = classes[other_cls_name]

    def adapter():
        r = cls.__subclasshook__(other)
        return repr(r)
    return adapter


# ---------------------------------------------------------------------------
# __getattribute__ (direct invocation, not plain getattr)
# ---------------------------------------------------------------------------

def _mk_getattribute(cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter():
        p = cls(FLOAT_C)
        return p.__getattribute__("degree")()
    return adapter


# ---------------------------------------------------------------------------
# __sizeof__ -- bit-exact value across dtype/degree variation
# ---------------------------------------------------------------------------

_SIZEOF_COEFS = [
    ("float_multi", [1.0, 2.0, 3.0]),
    ("int_multi", [1, 2, 3]),
    ("complex_single", [1.0 + 2.0j]),
    ("single", [5.0]),
]


def _mk_sizeof(cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter(coefs):
        p = cls(coefs)
        return p.__sizeof__()
    return adapter


# ---------------------------------------------------------------------------
# __weakref__ -- support, not value
# ---------------------------------------------------------------------------

def _mk_weakref(cls_name, side):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter():
        p = cls(FLOAT_C)
        weakref.ref(p)
        return "ok"
    return adapter


# ---------------------------------------------------------------------------
# __slots__ / __abstractmethods__ / __static_attributes__ -- class-level
# value equality
# ---------------------------------------------------------------------------

def _mk_class_attr(cls_name, side, attr):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter():
        v = getattr(cls, attr)
        if isinstance(v, frozenset):
            return sorted(v)
        return list(v) if isinstance(v, tuple) else v
    return adapter


# ---------------------------------------------------------------------------
# __init_subclass__ -- call directly (returns None) and with a bad arg
# (raises TypeError, message scoped to __qualname__ only -- no module path)
# ---------------------------------------------------------------------------

def _mk_init_subclass(cls_name, side, bad_arg):
    cls = (_NP if side == "np" else _AP)[cls_name]

    def adapter():
        if bad_arg:
            cls.__init_subclass__(1)  # always raises TypeError
            return "unreachable"
        r = cls.__init_subclass__()
        return repr(r)
    return adapter


def _build_generic_dunder_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    for cls_name in _NAMES:
        for op in ("__lt__", "__le__", "__gt__", "__ge__"):
            specs[_n(cls_name, op)] = ItemSpec(
                name=_n(cls_name, op), kind="custom",
                custom_cases=lambda cn=cls_name: _ordering_case_builder(cn),
                numpy_adapter=_mk_ordering_adapter(op, cls_name, "np"),
                ionp_adapter=_mk_ordering_adapter(op, cls_name, "ap"),
                scalar_like=True,
            )

        specs[_n(cls_name, "__setattr__")] = ItemSpec(
            name=_n(cls_name, "__setattr__"), kind="custom",
            custom_cases=lambda: _std([
                ("existing", ("existing",), {}), ("new", ("new",), {}),
            ]),
            numpy_adapter=lambda case, cn=cls_name: (
                _mk_setattr_existing(cn, "np")() if case == "existing"
                else _mk_setattr_new(cn, "np")()
            ),
            ionp_adapter=lambda case, cn=cls_name: (
                _mk_setattr_existing(cn, "ap")() if case == "existing"
                else _mk_setattr_new(cn, "ap")()
            ),
            scalar_like=True,
        )

        specs[_n(cls_name, "__delattr__")] = ItemSpec(
            name=_n(cls_name, "__delattr__"), kind="custom",
            custom_cases=lambda: _std([
                ("present", ("present",), {}), ("absent", ("absent",), {}),
            ]),
            numpy_adapter=lambda case, cn=cls_name: _mk_delattr(cn, "np", case == "present")(),
            ionp_adapter=lambda case, cn=cls_name: _mk_delattr(cn, "ap", case == "present")(),
            scalar_like=True,
        )

        specs[_n(cls_name, "__new__")] = ItemSpec(
            name=_n(cls_name, "__new__"), kind="custom",
            custom_cases=lambda: _std([("no_args", (), {})]),
            numpy_adapter=_mk_new(cls_name, "np"),
            ionp_adapter=_mk_new(cls_name, "ap"),
            scalar_like=True,
        )

        specs[_n(cls_name, "__dir__")] = ItemSpec(
            name=_n(cls_name, "__dir__"), kind="custom",
            custom_cases=lambda: _std([("members", (), {})]),
            numpy_adapter=_mk_dir(cls_name, "np"),
            ionp_adapter=_mk_dir(cls_name, "ap"),
            scalar_like=True,
        )

        specs[_n(cls_name, "__reduce_ex__")] = ItemSpec(
            name=_n(cls_name, "__reduce_ex__"), kind="custom",
            custom_cases=lambda: _std([(f"proto{p}", (p,), {}) for p in range(6)]),
            numpy_adapter=_mk_reduce_ex(cls_name, "np"),
            ionp_adapter=_mk_reduce_ex(cls_name, "ap"),
            scalar_like=True,
        )

        specs[_n(cls_name, "__reduce__")] = ItemSpec(
            name=_n(cls_name, "__reduce__"), kind="custom",
            custom_cases=lambda: _std([("basic", (), {})]),
            numpy_adapter=_mk_reduce(cls_name, "np"),
            ionp_adapter=_mk_reduce(cls_name, "ap"),
            scalar_like=True,
        )

        other_basis = "Legendre" if cls_name != "Legendre" else "Chebyshev"
        specs[_n(cls_name, "__subclasshook__")] = ItemSpec(
            name=_n(cls_name, "__subclasshook__"), kind="custom",
            custom_cases=lambda: _std([("self", ("self",), {}), ("other_basis", ("other_basis",), {})]),
            numpy_adapter=lambda case, cn=cls_name, ob=other_basis: (
                _mk_subclasshook(cn, "np", cn)() if case == "self"
                else _mk_subclasshook(cn, "np", ob)()
            ),
            ionp_adapter=lambda case, cn=cls_name, ob=other_basis: (
                _mk_subclasshook(cn, "ap", cn)() if case == "self"
                else _mk_subclasshook(cn, "ap", ob)()
            ),
            scalar_like=True,
        )

        specs[_n(cls_name, "__getattribute__")] = ItemSpec(
            name=_n(cls_name, "__getattribute__"), kind="custom",
            custom_cases=lambda: _std([("degree", (), {})]),
            numpy_adapter=_mk_getattribute(cls_name, "np"),
            ionp_adapter=_mk_getattribute(cls_name, "ap"),
            scalar_like=True,
        )

        specs[_n(cls_name, "__sizeof__")] = ItemSpec(
            name=_n(cls_name, "__sizeof__"), kind="custom",
            custom_cases=lambda: _std([(lbl, (c,), {}) for lbl, c in _SIZEOF_COEFS]),
            numpy_adapter=_mk_sizeof(cls_name, "np"),
            ionp_adapter=_mk_sizeof(cls_name, "ap"),
            scalar_like=True,
        )

        specs[_n(cls_name, "__weakref__")] = ItemSpec(
            name=_n(cls_name, "__weakref__"), kind="custom",
            custom_cases=lambda: _std([("basic", (), {})]),
            numpy_adapter=_mk_weakref(cls_name, "np"),
            ionp_adapter=_mk_weakref(cls_name, "ap"),
            scalar_like=True,
        )

        for attr in ("__slots__", "__abstractmethods__", "__static_attributes__"):
            specs[_n(cls_name, attr)] = ItemSpec(
                name=_n(cls_name, attr), kind="custom",
                custom_cases=lambda: _std([("value", (), {})]),
                numpy_adapter=_mk_class_attr(cls_name, "np", attr),
                ionp_adapter=_mk_class_attr(cls_name, "ap", attr),
                scalar_like=True,
            )

        specs[_n(cls_name, "__init_subclass__")] = ItemSpec(
            name=_n(cls_name, "__init_subclass__"), kind="custom",
            custom_cases=lambda: _std([("no_args", ("no_args",), {}), ("bad_arg", ("bad_arg",), {})]),
            numpy_adapter=lambda case, cn=cls_name: _mk_init_subclass(cn, "np", case == "bad_arg")(),
            ionp_adapter=lambda case, cn=cls_name: _mk_init_subclass(cn, "ap", case == "bad_arg")(),
            scalar_like=True,
        )

    return specs


GENERIC_POLY_DUNDER_SPECS = _build_generic_dunder_specs()
