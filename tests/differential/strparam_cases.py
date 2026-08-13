"""NEW test-cases file (owned by this task, corpus-only): crosses every
string-VALUED parameter this task's `/tmp/bytesgap.py` sweep found (75
cases, 23 class mismatches) with the CAPABILITY axis that produced that
finding -- not with "the values a caller is expected to pass". See that
probe script (still present at /tmp/bytesgap.py, extended for this task's
item-4 investigation below) for the original measurement.

THE DEFECT THIS FILE MAKES VISIBLE
---------------------------------------------------------------------------
numpy duck-types its string-valued keyword parameters: it calls a string
method on the value, or formats it straight into an error message, or
decodes bytes as ASCII before comparing. anionpy instead extracts a Rust `str`
up front and raises `TypeError` the moment that extraction fails. The two
diverge on the TYPE of a string parameter -- an axis no other file in this
corpus varies (every existing `order=`/`kind=`/`side=`/`casting=`/`mode=`/
`ord=` case anywhere else in tests/differential/*_cases.py passes a literal
`str`, never anything else).

Three distinct sub-shapes, all real, all present below:

  SHAPE 1 -- numpy ACCEPTS bytes (decodes ASCII), anionpy rejects on type.
      reshape/ravel/flatten/copy/astype order=, array order=, sort kind=,
      searchsorted side=, zeros dtype=.
  SHAPE 2 -- numpy raises ValueError (value formatted into the message),
      anionpy raises TypeError (type rejected before the value is even
      looked at). linalg.qr mode=, linalg.norm ord=.
  SHAPE 3 -- THE DANGEROUS ONE: anionpy is MORE PERMISSIVE than numpy. numpy
      raises `TypeError: casting must be str, not NoneType` for
      `casting=None`; anionpy silently ACCEPTS it. add casting=,
      astype casting=, concatenate casting=. A drop-in replacement that
      accepts what numpy rejects silently changes the meaning of working
      code -- this is the one that matters most.

ENUMERATING BY CAPABILITY, NOT BY EXPECTED VALUES
---------------------------------------------------------------------------
This is the rule that produced the finding above, and the rule that shapes
every case list built here (`_capability_values`):

  correct type,   valid value      'C'                  (anti-vacuity)
  correct type,   invalid value    'Z'
  has str methods, wrong type      b'C', bytearray(b'C')
  lacks str methods                1.5, ['C'], None, object()

...plus two EXTRA classes this task's brief asked for explicitly (item 4:
"check whether numpy accepts a str SUBCLASS, and whether it accepts numpy's
own str_ scalar type -- I did not test either" -- not covered by the
original /tmp/bytesgap.py sweep at all):

  correct type,   str SUBCLASS     _StrSub('C')  (isinstance(x, str) True,
                                                   type(x) is str False)
  correct type,   np.str_ scalar   np.str_('C')

The valid-value case is included for EVERY (function, parameter) pair on
purpose: a check built only from rejections would pass a version of anionpy
that rejects everything, which is the vacuous form of this test. See
harness.py's `evaluate()` -- a "pass" here requires the valid-value case to
ALSO produce a matching non-exception result, not just matching exceptions
on the invalid ones.

NUMPY IS ORACLE ONLY. This file never calls real numpy to compute anionpy's
expected answer or to borrow a numpy-derived message string -- every case
below is just a (label, args, kwargs) tuple; `harness.run_case` (via
`registry.evaluate`) makes the live numpy call and the live anionpy call
itself and compares them, exactly like every other item in this registry.

WHY THIS FILE'S KEYS ARE `"strparam/<param>/<real-item-name>"`, NOT THE REAL
"reshape"/"ravel"/"copy"/"sort"/"searchsorted"/"zeros"/"add"/"linalg.qr"/
"ndarray.flatten"/"ndarray.astype"/"array"/"concatenate"/"linalg.norm" KEYS
THEMSELVES -- AND WHY THAT STILL COUNTS AS "ATTRIBUTED TO", NOT DODGING
---------------------------------------------------------------------------
[2026-08-02 RE-ATTRIBUTION, see this task's report] This file originally
used a fully synthetic `"strparam/<fn>_<param>"` namespace (e.g.
`"strparam/reshape_order"`) with an explicit collision guard, and that
guard's own docstring argued the synthetic name was *required* because
`registry.REGISTRY.update()`'s tail-merge "refuses to redeclare an existing
key". That is true and is still exactly why every key below is still not
literally `"reshape"` -- `reshape`/`ravel`/`copy`/`sort`/`searchsorted`/
`zeros`/`add`/`linalg.qr`/`linalg.norm`/`ndarray.flatten`/`ndarray.astype`/
`array`/`concatenate` are declared elsewhere (registry.py's own body,
creation_cases.py, sort_cases.py, ufunc_registry.py, linalg_cases.py,
ndarray_attrs_cases.py, manip_cases.py) and a second `ItemSpec` under any of
those exact names would raise this file's own collision assertion at import
time (measured, not assumed -- see the report). But collision-avoidance
was being used to justify a namespace ("strparam/...") that ALSO defeated
`tools/coverage.py`'s own `fold_axis_failures()` -- which exists
specifically to attribute a namespaced axis-corpus case back to the real
surface item it grades, by matching the FINAL "/"-segment of the report key
against the surface item list (see that function's docstring in
coverage.py, and its two prior uses for exactly this shape of problem:
`ufunc_order_cases.py`'s `"order/ufunc/<name>"` and `ufunc_dtype_cases.py`'s
`"dtype/ufunc/<name>"`, both of which predate this file and both of which
this file's *original* version failed to follow). Every key below is
therefore `"strparam/<param>/<real-item-name>"` -- unique in
`registry.REGISTRY` (no collision, no overwrite, no redeclaration) AND
tail-matching a real surface name, so `fold_axis_failures` correctly OR-s
each item's own failures into the real item's ledger verdict: the real item
is credited "failing" if EITHER its own dedicated corpus fails OR any
`strparam/*` case attributed to it fails, and stays whatever its own
dedicated corpus says otherwise. Proven live (not assumed): see this task's
report for the temporary-deliberately-failing-case experiment that measured
this fold behavior before any real case was moved.

Before this rename, THIRTEEN of the items this file's cases route through
were declared "exact" in the ledger while measurably diverging on this
axis -- `reshape`, `ravel`, `copy`, `sort`, `searchsorted`, `zeros`, `add`,
`linalg.qr`, `linalg.norm`, `ndarray.flatten`, `ndarray.astype`, `array`,
`concatenate` -- because the original synthetic namespace made
`tools/coverage.py --tests` blind to every one of them (`failing 0` while 9
of those names were independently true divergences). This file makes the
ledger look; the item declarations are not this task's to touch (see this
task's brief -- no `anionpy/_state/` edits, no `.rs` edits).
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
import registry
from registry import ItemSpec
# `linalg.qr`'s real Q/R output has an inherent sign/phase ambiguity (numpy
# and any independent LAPACK-based implementation can both be correct and
# still disagree element-wise), which is why the ALREADY-DECLARED
# "linalg.qr" item in linalg_cases.py never compares Q/R directly -- it
# checks reconstruction/orthonormality/upper-triangular INVARIANTS instead
# (`_qr_probe`/`_mk_pair`). Reusing that exact probe here (read-only import,
# linalg_cases.py itself is not edited by this task) is required for the
# same reason: a naive raw `np.linalg.qr(...) == anionpy.linalg.qr(...)`
# comparison fails on ITS OWN, for a reason that has nothing to do with the
# mode= string-type axis under test (measured: numpy's `mode='reduced'`
# returns a `QRResult` NAMEDTUPLE, anionpy's returns a plain `tuple` -- a type
# mismatch that would falsely fail even the anti-vacuity
# `correct_type_valid` case if graded by plain equality).
from linalg_cases import _mk_pair, _qr_probe


class _StrSub(str):
    """A genuine `str` subclass: `isinstance(x, str)` is True, `type(x) is
    str` is False. Used to probe whether numpy's/anionpy's string-extraction
    accepts any `str` instance or requires the concrete builtin type
    exactly -- this task's brief flagged it as untested by the original
    /tmp/bytesgap.py sweep (item 4)."""


def _capability_values(valid: str, invalid: str) -> dict[str, object]:
    """The full capability-class grid for one string-valued parameter,
    built from its one KNOWN-VALID string and one KNOWN-INVALID string.
    Order is insertion order -- kept deliberately human-readable (valid
    first, then progressively "more wrong") when a case list prints.
    """
    assert isinstance(valid, str) and isinstance(invalid, str)
    return {
        "correct_type_valid":              valid,
        "correct_type_invalid":            invalid,
        "correct_type_str_subclass":       _StrSub(valid),
        "correct_type_np_str_scalar":      np.str_(valid),
        "wrong_type_bytes":                valid.encode("ascii"),
        "wrong_type_bytearray":            bytearray(valid.encode("ascii")),
        "lacks_str_methods_float":         1.5,
        "lacks_str_methods_list":          [valid],
        "lacks_str_methods_none":          None,
        "lacks_str_methods_object":        object(),
    }


def _param_cases(prefix: str, base_args: tuple, kwarg_name: str,
                  valid: str, invalid: str, fixed_kwargs: dict | None = None
                  ) -> list[tuple[str, tuple, dict]]:
    """One (label, args, kwargs) triple per capability class for a single
    (function, parameter) pair. `base_args` is reused, unmutated, across
    every case -- safe because `harness.run_case`'s `_freshen` makes an
    independent copy of every ndarray argument immediately before EACH of
    the two (numpy, anionpy) calls, including in-place operations like
    `sort`'s -- see `_freshen_array`'s docstring in harness.py.
    """
    fixed_kwargs = dict(fixed_kwargs or {})
    cases = []
    for label, value in _capability_values(valid, invalid).items():
        kwargs = dict(fixed_kwargs)
        kwargs[kwarg_name] = value
        cases.append((f"{prefix}/{label}", base_args, kwargs))
    return cases


def _spec(key: str, numpy_path: str, ionp_path: str, cases: list) -> ItemSpec:
    return ItemSpec(
        name=key, kind="custom",
        numpy_path=numpy_path, ionp_path=ionp_path,
        custom_cases=lambda: cases,
        atol=0.0, rtol=0.0,
    )


# ---------------------------------------------------------------------------
# Shared source arrays -- deliberately independent of corpus.py's own seeded
# RNG stream (this file's grid is a fixed, small, hand-picked set of shapes,
# not a swept one), and independent of /tmp/bytesgap.py's module-global `a2`
# / `a1`/ `ai` (this file must not import a /tmp script).
# ---------------------------------------------------------------------------
_A2 = np.array([[1.0, 2.0], [3.0, 4.0]])          # 2x2, for reshape/ravel/
                                                    # flatten/copy/astype/
                                                    # array/qr/norm
_A1_UNSORTED = np.array([3.0, 1.0, 2.0])           # for sort_kind
_A1_SORTED = np.array([1.0, 2.0, 3.0])             # for searchsorted
_AI = np.array([1, 2, 3], dtype="int64")           # for add (int ufunc)


# ---------------------------------------------------------------------------
# SHAPE 1 items -- numpy decodes ASCII bytes, anionpy rejects on type.
# ---------------------------------------------------------------------------

_RESHAPE_ORDER = _param_cases(
    "strparam/order/reshape", (_A2, (4,)), "order", "C", "Z",
)
_RAVEL_ORDER = _param_cases(
    "strparam/order/ravel", (_A2,), "order", "C", "Z",
)
_FLATTEN_ORDER = _param_cases(
    "strparam/order/ndarray.flatten", (_A2,), "order", "C", "Z",
)
_COPY_ORDER = _param_cases(
    "strparam/order/copy", (_A2,), "order", "C", "Z",
)
_ASTYPE_ORDER = _param_cases(
    "strparam/order/ndarray.astype", (_A2, "float32"), "order", "C", "Z",
)
_ARRAY_ORDER = _param_cases(
    "strparam/order/array", (_A2,), "order", "C", "Z",
)
_SORT_KIND = _param_cases(
    "strparam/kind/sort", (_A1_UNSORTED,), "kind", "quicksort", "Z",
)
_SEARCHSORTED_SIDE = _param_cases(
    "strparam/side/searchsorted", (_A1_SORTED, 2.0), "side", "left", "Z",
)
_ZEROS_DTYPE = _param_cases(
    "strparam/dtype/zeros", (2,), "dtype", "float32", "not_a_real_dtype",
)

# order= is also a real, independent ufunc kwarg (controls output memory
# layout) -- crossed here too since it is a distinct str-extraction site
# from casting= in ionp-py's ufunc call path, not a duplicate of it.
_ADD_ORDER = _param_cases(
    "strparam/order/add", (_AI, _AI), "order", "C", "Z",
)

# ---------------------------------------------------------------------------
# SHAPE 2 items -- numpy raises ValueError with the (rejected) value
# formatted into the message; anionpy raises TypeError before ever looking at
# the value. mode= diverges for bytes, bytearray, 1.5 AND None all four,
# per this task's brief -- all captured by the shared capability grid
# above, no special-casing needed.
# ---------------------------------------------------------------------------

_QR_MODE = _param_cases(
    "strparam/mode/linalg.qr", (_A2,), "mode", "reduced", "Z",
)
_QR_NUMPY_ADAPTER, _QR_IONP_ADAPTER = _mk_pair(_qr_probe)
# ord= is genuinely polymorphic in real numpy (str like 'fro'/'nuc' OR a
# real number OR None -- unlike every other parameter in this file, which
# is str-or-nothing) -- that IS the point of measuring it here, since
# anionpy's own captured divergence ("must be real number, not bytes") is
# exactly anionpy treating ord= as number-only where numpy treats it as
# string-or-number-or-None. The capability grid's "lacks_str_methods_none"
# case is therefore expected to be a LEGITIMATE ACCEPT on numpy's side for
# a 2-D array (Frobenius default), not a rejection -- reported as data, not
# forced into a shape it doesn't have.
_NORM_ORD = _param_cases(
    "strparam/ord/linalg.norm", (_A2,), "ord", "fro", "Z",
)

# `linalg.matrix_norm` (the standalone Array-API function, not the
# `linalg.norm` dispatcher above) has its own independent `ord=`
# extraction site in anionpy (`parse_matrix_ord` in `ionp-py/src/linalg.rs`)
# -- a separate function from `linalg.norm`'s, even though both currently
# share the same behavior (verified directly: numpy's real `matrix_norm`
# raises the identical `ValueError: Invalid norm order for matrices.` --
# note, no value interpolated -- for every capability class here except
# the two valid-string/fro-equivalent ones and `None`, which is numpy's
# uniform-across-type domain-validation behavior for a MATRIX ord,
# unlike a vector ord's type-dependent generic-computation behavior;
# see `anionpy/_state/linalg.py`'s `linalg.matrix_norm` entry for the full
# writeup and why `linalg.vector_norm` does NOT get an equivalent grid
# here -- its real behavior for this same grid is NOT uniform (some
# values compute successfully via numpy's generic ufunc machinery, some
# raise a completely different, value/type-dependent message), so a
# capability-grid case for it would be a corpus case known to fail on
# `wrong_type_bytes`/`wrong_type_bytearray`/`lacks_str_methods_list`/
# `lacks_str_methods_object`, which this task's brief forbids adding).
_MATRIX_NORM_ORD = _param_cases(
    "strparam/ord/linalg.matrix_norm", (_A2,), "ord", "fro", "Z",
)

# ---------------------------------------------------------------------------
# SHAPE 3 items -- THE DANGEROUS ONE: anionpy is MORE PERMISSIVE than numpy.
# ---------------------------------------------------------------------------

_ASTYPE_CASTING = _param_cases(
    "strparam/casting/ndarray.astype", (_A2, "float32"), "casting", "same_kind", "Z",
)
_ADD_CASTING = _param_cases(
    "strparam/casting/add", (_AI, _AI), "casting", "same_kind", "Z",
)
_CONCATENATE_CASTING = _param_cases(
    "strparam/casting/concatenate", ([_A1_UNSORTED, _A1_UNSORTED],),
    "casting", "same_kind", "Z",
)


# ---------------------------------------------------------------------------
# `order_typename` items -- [2026-08-03] this task's deliverable 4: the
# `"{param} must be str, not {type}"` TypeError text itself divergently
# formatted the type's name (`anionpy.copy(a, order=np.int64(0))` said "not
# int64", real numpy says "not numpy.int64" -- see `python_type_display_name`
# in `ionp-py/src/lib.rs` for the fix and the measured formatting rule). The
# existing `_capability_values` grid above already covers `bytes`/
# `bytearray`/`float`/`list`/`None`/`object()` for `order=` (all EXCEPT
# `object()` share the exact same "not <builtin type name>" shape numpy and
# anionpy already agreed on even before this task, since `object`'s module is
# `builtins`) -- but it never included a numpy scalar type or a genuine
# user-defined class, which are exactly the two shapes whose formatting used
# to diverge (a static numpy C type needs its module prefixed; a class
# defined via an ordinary Python `class` statement does not, regardless of
# which module it lives in -- both verified live, see the doc comment cited
# above). This is therefore a SEPARATE small grid, not an addition to
# `_capability_values` (which also feeds `casting=`/`kind=`/`mode=`/`ord=`/
# `dtype=` cases this task was not asked to touch and has not verified for
# these particular extra values).
#
# `None` and `b'C'`/`bytearray(b'C')` are deliberately NOT included here even
# though they are "not a string" too: MEASURED live against real numpy 2.5.1,
# `order=None` and `order=b'C'` are both silently ACCEPTED (treated as
# omitted / decoded to `'C'` respectively) by every one of these five
# callers, not rejected -- so they are not members of the "wrong type"
# capability class at all for `order=` specifically (unlike `casting=`,
# where `None` genuinely raises `TypeError: casting must be str, not
# NoneType`, already covered by the existing grid's `lacks_str_methods_none`
# case). Treating them as "invalid" here would just document a pre-existing,
# separate order=None/bytes ACCEPT-vs-ACCEPT non-issue as a false failure.
class _ArbitraryProbeClass:
    """A genuine user-defined class (not a numpy/builtin type), used to
    confirm numpy's type-name formatting leaves ordinary Python-defined
    classes BARE (no module prefix) regardless of which module they live
    in -- see `python_type_display_name`'s doc comment in `ionp-py/src/
    lib.rs` for the live measurement this mirrors."""


def _order_typename_values() -> dict[str, object]:
    import numpy as _np
    return {
        "wrong_type_numpy_int64":  _np.int64(0),
        "wrong_type_numpy_float32": _np.float32(0),
        "wrong_type_py_int":       0,
        "wrong_type_py_bool":      True,
        "wrong_type_py_float":     3.5,
        "wrong_type_py_list":      ["C"],
        "wrong_type_py_dict":      {"order": "C"},
        "wrong_type_user_class":   _ArbitraryProbeClass(),
    }


def _order_typename_cases(prefix: str, base_args: tuple,
                           fixed_kwargs: dict | None = None
                           ) -> list[tuple[str, tuple, dict]]:
    fixed_kwargs = dict(fixed_kwargs or {})
    cases = []
    for label, value in _order_typename_values().items():
        kwargs = dict(fixed_kwargs)
        kwargs["order"] = value
        cases.append((f"{prefix}/{label}", base_args, kwargs))
    return cases


_RESHAPE_ORDER_TYPENAME = _order_typename_cases(
    "strparam/order_typename/reshape", (_A2, (4,)),
)
_RAVEL_ORDER_TYPENAME = _order_typename_cases(
    "strparam/order_typename/ravel", (_A2,),
)
_COPY_ORDER_TYPENAME = _order_typename_cases(
    "strparam/order_typename/copy", (_A2,),
)
_ARRAY_ORDER_TYPENAME = _order_typename_cases(
    "strparam/order_typename/array", (_A2,),
)
_ADD_ORDER_TYPENAME = _order_typename_cases(
    "strparam/order_typename/add", (_AI, _AI),
)


# ---------------------------------------------------------------------------
# `side_wording` items -- [2026-08-03] this task's deliverable 4: the
# `ValueError` TEXT for an invalid `side=` (not its type -- that's the
# `strparam/side/searchsorted` item above) used to always say `"search side
# must be 'left' or 'right' (got '{s}')"`; real numpy's actual
# `PyArray_SearchsideConverter` only uses that "(got ...)" form when the
# invalid string's FIRST character (case-insensitively) is neither 'l' nor
# 'r' -- anything starting with 'l'/'L'/'r'/'R', however nonsensical past
# that first letter (`"LEFT"`, `"Left"`, `"LR"`, `"rrrrr"`, ...), gets the
# terser `"search side must be one of 'left' or 'right'"` with NO value
# interpolated at all. Measured live against real numpy 2.5.1 across ~25
# probe strings (see `search_side_value_error`'s doc comment in
# `ionp-py/src/lib.rs`) -- this grid exercises both message shapes on BOTH
# the free function and the `ndarray.searchsorted` method, since each has
# its own independent extraction/validation site in ionp-py
# (`reductions.rs`/`ndarray_attrs.rs`).
_SIDE_WORDING_VALUES = {
    "terse_upper_left":   "LEFT",
    "terse_mixed_left":   "Left",
    "terse_single_l":     "l",
    "terse_nonsense_lr":  "LR",
    "terse_repeated_r":   "rrrrr",
    "got_clause_bogus":   "bogus",
    "got_clause_empty":   "",
    "got_clause_xl":      "xl",
}


def _side_wording_cases(prefix: str, base_args: tuple) -> list[tuple[str, tuple, dict]]:
    return [
        (f"{prefix}/{label}", base_args, {"side": value})
        for label, value in _SIDE_WORDING_VALUES.items()
    ]


_SEARCHSORTED_SIDE_WORDING = _side_wording_cases(
    "strparam/side_wording/searchsorted", (_A1_SORTED, 2.0),
)
_SEARCHSORTED_METHOD_SIDE_WORDING = _side_wording_cases(
    "strparam/side_wording/ndarray.searchsorted", (_A1_SORTED, 2.0),
)


# ---------------------------------------------------------------------------
# Registry wiring. `ionp_path`/`numpy_path` mirror how each real item is
# already resolved elsewhere in this registry: bare top-level name for a
# plain function or ufunc called directly (`resolve_numpy`/`resolve_ionp`
# default `path` to `self.name` when `numpy_path`/`ionp_path` is None, then
# `_resolve_dotted` on that -- note every entry below sets `numpy_path`/
# `ionp_path` EXPLICITLY, so the "strparam/<param>/<real-name>" `name=` key
# never leaks into path resolution; a ufunc object resolved this way, e.g.
# `np.add`, is still directly callable with `casting=`/`order=` kwargs --
# kind="custom" never routes it through the reduce/accumulate/at
# dispatcher, which is correct here since none of those forms are under
# test), "ndarray.<method>" for a real bound method (flatten, astype) so
# `resolve_ionp()`'s "ndarray."-prefix branch converts the numpy.ndarray
# receiver to a real anionpy.ndarray before dispatch, and "linalg.<fn>" for the
# two linalg items.
#
# Each key is `"strparam/<param>/<real-surface-item-name>"` -- unique in
# `registry.REGISTRY` (the collision guard below still applies, and still
# passes: nothing here literally equals a real item name), and its final
# "/"-segment is the EXACT real surface name so
# `tools/coverage.py`'s `fold_axis_failures()` attributes this item's
# failures back onto the real item (see module docstring's RE-ATTRIBUTION
# section and this task's report for the measured proof). `astype_order`/
# `astype_casting` and `add_order`/`add_casting` both fold onto one real
# item (`ndarray.astype`, `add` respectively) from two distinct keys --
# that is intentional: two independent str-extraction sites in the same
# real item, both real, neither a duplicate of the other (see the SHAPE-1/
# SHAPE-3 section comments above), and `fold_axis_failures` ORs any number
# of namespaced failures onto the same real item without needing them
# unified into one registry key.
# ---------------------------------------------------------------------------

STRPARAM_SPECS: dict[str, ItemSpec] = {
    "strparam/order/reshape": _spec(
        "strparam/order/reshape", "reshape", "reshape", _RESHAPE_ORDER),
    "strparam/order/ravel": _spec(
        "strparam/order/ravel", "ravel", "ravel", _RAVEL_ORDER),
    "strparam/order/ndarray.flatten": _spec(
        "strparam/order/ndarray.flatten", "ndarray.flatten", "ndarray.flatten", _FLATTEN_ORDER),
    "strparam/order/copy": _spec(
        "strparam/order/copy", "copy", "copy", _COPY_ORDER),
    "strparam/order/ndarray.astype": _spec(
        "strparam/order/ndarray.astype", "ndarray.astype", "ndarray.astype", _ASTYPE_ORDER),
    "strparam/casting/ndarray.astype": _spec(
        "strparam/casting/ndarray.astype", "ndarray.astype", "ndarray.astype", _ASTYPE_CASTING),
    "strparam/order/array": _spec(
        "strparam/order/array", "array", "array", _ARRAY_ORDER),
    "strparam/kind/sort": _spec(
        "strparam/kind/sort", "sort", "sort", _SORT_KIND),
    "strparam/side/searchsorted": _spec(
        "strparam/side/searchsorted", "searchsorted", "searchsorted", _SEARCHSORTED_SIDE),
    "strparam/dtype/zeros": _spec(
        "strparam/dtype/zeros", "zeros", "zeros", _ZEROS_DTYPE),
    "strparam/casting/add": _spec(
        "strparam/casting/add", "add", "add", _ADD_CASTING),
    "strparam/order/add": _spec(
        "strparam/order/add", "add", "add", _ADD_ORDER),
    "strparam/casting/concatenate": _spec(
        "strparam/casting/concatenate", "concatenate", "concatenate", _CONCATENATE_CASTING),
    "strparam/mode/linalg.qr": ItemSpec(
        name="strparam/mode/linalg.qr", kind="custom",
        custom_cases=lambda: _QR_MODE,
        numpy_adapter=_QR_NUMPY_ADAPTER, ionp_adapter=_QR_IONP_ADAPTER,
        # atol=0.0/rtol=0.0 -- exact, per registry.py's own rule (the
        # legacy nonzero atol/rtol path is now structurally forbidden; a
        # real tolerance would require its own >=20000-sample
        # epsilon_sweep, which is the DEDICATED "linalg.qr" item's job,
        # not this string-type-axis item's -- see this dict's other
        # comments). Measured directly (not assumed): this item's fixed
        # 2x2 float64 matrix's QR invariant residual is bit-exact 0.0 on
        # both sides for the only capability classes that ever reach this
        # comparison at all (`correct_type_valid`/
        # `correct_type_str_subclass`/`correct_type_np_str_scalar` -- every
        # other class raises on one side or the other and is graded by the
        # exception-comparison branch above, never this tolerance).
        atol=0.0, rtol=0.0,
    ),
    "strparam/ord/linalg.norm": _spec(
        "strparam/ord/linalg.norm", "linalg.norm", "linalg.norm", _NORM_ORD),
    "strparam/ord/linalg.matrix_norm": _spec(
        "strparam/ord/linalg.matrix_norm", "linalg.matrix_norm", "linalg.matrix_norm", _MATRIX_NORM_ORD),
    "strparam/order_typename/reshape": _spec(
        "strparam/order_typename/reshape", "reshape", "reshape", _RESHAPE_ORDER_TYPENAME),
    "strparam/order_typename/ravel": _spec(
        "strparam/order_typename/ravel", "ravel", "ravel", _RAVEL_ORDER_TYPENAME),
    "strparam/order_typename/copy": _spec(
        "strparam/order_typename/copy", "copy", "copy", _COPY_ORDER_TYPENAME),
    "strparam/order_typename/array": _spec(
        "strparam/order_typename/array", "array", "array", _ARRAY_ORDER_TYPENAME),
    "strparam/order_typename/add": _spec(
        "strparam/order_typename/add", "add", "add", _ADD_ORDER_TYPENAME),
    "strparam/side_wording/searchsorted": _spec(
        "strparam/side_wording/searchsorted", "searchsorted", "searchsorted",
        _SEARCHSORTED_SIDE_WORDING),
    "strparam/side_wording/ndarray.searchsorted": ItemSpec(
        name="strparam/side_wording/ndarray.searchsorted", kind="custom",
        custom_cases=lambda: _SEARCHSORTED_METHOD_SIDE_WORDING,
        numpy_adapter=lambda a, v, side="left": a.searchsorted(v, side=side),
        ionp_adapter=lambda a, v, side="left": a.searchsorted(v, side=side),
        atol=0.0, rtol=0.0,
    ),
}

_collisions = set(STRPARAM_SPECS) & set(registry.REGISTRY)
if _collisions:
    raise AssertionError(
        f"strparam_cases.py: {sorted(_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
registry.REGISTRY.update(STRPARAM_SPECS)
