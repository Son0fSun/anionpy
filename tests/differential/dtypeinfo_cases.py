"""Differential ItemSpecs for the dtype introspection/promotion block
(`can_cast`, `promote_types`, `result_type`, `min_scalar_type`, `iinfo`,
`typename`, `mintypecode`, `isdtype`, `finfo`), backed by
`ionp-py/src/dtypeinfo.rs` + the `finfo_for`/`iinfo_for`/`min_scalar_type_*`
helpers in `ionp-core/src/dtype.rs`. See `anionpy/_state/toplevel.py` for
which of these are actually DECLARED and the per-item evidence -- this file
intentionally also registers cases for the items that are NOT declared
(`min_scalar_type`'s out-of-i64/u64-range values, `isdtype`'s scalar-type-
class `kind` form, `finfo` entirely), so those known, permanent gaps show
up as visible fails in the differential report rather than being hidden by
omission. A "fail" verdict on those specific items is expected and is the
evidence, not a regression.

All items here are `kind="custom"`: none is a plain array-in/array-out
elementwise function. Dtype-returning items (`promote_types`,
`result_type`, `min_scalar_type`) are called directly with `numpy_path`/
`ionp_path` pointing at the real top-level functions and `scalar_like=True`
-- `harness._compare_scalar_like` already special-cases `np.dtype` results
by coercing the anionpy side via `np.dtype(ionp_out)` (anionpy's `dtype` objects
are numpy-dtype-interoperable via `.name`, verified directly), so no
adapter is needed for those. `can_cast` (bool) and `isdtype` (bool) use the
same direct-call + `scalar_like=True` path. `iinfo`/`finfo` return the
`IInfo`/`finfo` wrapper classes (see `anionpy/_dtypeinfo.py`; `finfo` was
renamed from `FInfo` 2026-08-06, task #28), which are not
directly comparable to numpy's own `iinfo`/`finfo` instances, so they use
adapters that reduce each side to a plain tuple of public fields -- same
pattern as `lib_cases.py`'s `NumpyVersion`/`Arrayterator` adapters.
`typename`/`mintypecode` are plain str in/str out and need no adapter.
"""
from __future__ import annotations

import itertools

import _bootstrap  # noqa: F401

import numpy as np

from registry import ItemSpec

DTYPEINFO_SPECS: dict[str, ItemSpec] = {}

ALL_DTYPE_NAMES = [
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "complex64", "complex128",
]
CASTING_RULES = ["no", "equiv", "safe", "same_kind", "unsafe"]


# ---------------------------------------------------------------------------
# can_cast
# ---------------------------------------------------------------------------

def can_cast_cases():
    cases = []
    for frm, to, rule in itertools.product(ALL_DTYPE_NAMES, ALL_DTYPE_NAMES, CASTING_RULES):
        cases.append((f"{frm}->{to}/{rule}", (frm, to, rule), {}))
    # error paths
    cases.append(("none_from", (None, "int8"), {}))
    cases.append(("none_to", ("int8", None), {}))
    cases.append(("bad_dtype_string", ("not_a_dtype", "int8"), {}))
    cases.append(("bare_int_scalar", (3, "int8"), {}))
    cases.append(("bare_float_scalar", (3.5, "int8"), {}))
    cases.append(("bare_complex_scalar", (3 + 1j, "int8"), {}))
    cases.append(("bare_bool_scalar", (True, "int8"), {}))
    cases.append(("bad_casting_string", ("int8", "int16", "bogus"), {}))
    # documented, permanent gaps (see toplevel.py's can_cast non-
    # declaration comment): numpy's PyArray_DescrConverter accepts an
    # empty list/dict as a zero-field void/structured dtype spec; anionpy has
    # no void dtype at all and raises instead of returning False. Included
    # so these show as honest FAILs, not hidden by omission.
    cases.append(("empty_list_as_dtype_gap", ([], "int8"), {}))
    cases.append(("empty_dict_as_dtype_gap", ({}, "int8"), {}))
    return cases


DTYPEINFO_SPECS["can_cast"] = ItemSpec(
    name="can_cast", kind="custom", custom_cases=can_cast_cases,
    numpy_path="can_cast", ionp_path="can_cast", scalar_like=True,
)


# ---------------------------------------------------------------------------
# promote_types
# ---------------------------------------------------------------------------

def promote_types_cases():
    cases = [
        (f"{a}/{b}", (a, b), {})
        for a, b in itertools.product(ALL_DTYPE_NAMES, ALL_DTYPE_NAMES)
    ]
    cases.append(("bad_a", ("not_a_dtype", "int8"), {}))
    cases.append(("bad_b", ("int8", "not_a_dtype"), {}))
    cases.append(("none_a", (None, "int8"), {}))
    cases.append(("none_b", ("int8", None), {}))
    cases.append(("none_none", (None, None), {}))
    # documented, permanent gaps (see toplevel.py's promote_types non-
    # declaration comment): void/structured-dtype-spec fallback ([]/{}),
    # (base_dtype, shape) 2-tuple parsing (()), fixed-width-string-from-
    # bytes coercion (b'x'), and array-as-dtype-spec ("Cannot construct a
    # dtype from an array") -- none of which anionpy implements. Included so
    # these show as honest FAILs.
    cases.append(("empty_list_gap", ([], "int8"), {}))
    cases.append(("empty_dict_gap", ({}, "int8"), {}))
    cases.append(("empty_tuple_gap", ((), "int8"), {}))
    cases.append(("bytes_gap", (b"x", "int8"), {}))
    return cases


DTYPEINFO_SPECS["promote_types"] = ItemSpec(
    name="promote_types", kind="custom", custom_cases=promote_types_cases,
    numpy_path="promote_types", ionp_path="promote_types", scalar_like=True,
)


# ---------------------------------------------------------------------------
# result_type
# ---------------------------------------------------------------------------

def result_type_cases():
    cases = [
        ("single_dtype", ("int8",), {}),
        ("two_dtypes", ("int8", "float32"), {}),
        ("three_dtypes", ("bool", "int16", "complex64"), {}),
        ("dtype_and_weak_int", ("int8", 300), {}),
        ("dtype_and_weak_negative_int", ("uint8", -1), {}),
        ("dtype_and_weak_float", ("int32", 1.5), {}),
        ("dtype_and_weak_complex", ("float32", 1 + 2j), {}),
        ("dtype_and_weak_bool", ("int8", True), {}),
        ("weak_only_int_float", (3, 1.5), {}),
        ("weak_only_int_complex", (3, 1 + 2j), {}),
        ("weak_only_bool_int", (True, 3), {}),
        ("many_mixed", ("int8", "float16", 3, 1.5, True), {}),
        ("empty_error", (), {}),
        ("garbage_arg", ("not_a_dtype",), {}),
        ("none_arg", (None,), {}),
        ("none_and_dtype", (None, "int8"), {}),
        # documented, permanent gap shared with promote_types (same
        # coerce_dtype_like path) -- see toplevel.py's non-declaration
        # comment.
        ("empty_list_gap", ([],), {}),
    ]
    return cases


DTYPEINFO_SPECS["result_type"] = ItemSpec(
    name="result_type", kind="custom", custom_cases=result_type_cases,
    numpy_path="result_type", ionp_path="result_type", scalar_like=True,
)


# ---------------------------------------------------------------------------
# min_scalar_type
# ---------------------------------------------------------------------------

def min_scalar_type_cases():
    values = [
        0, 1, -1, 127, 128, -128, -129, 255, 256, 32767, 32768, -32768, -32769,
        65535, 65536, 2147483647, 2147483648, -2147483648, -2147483649,
        4294967295, 4294967296, 9223372036854775807, -9223372036854775808,
        18446744073709551615,
        0.0, 1.5, -1.5, 65000.0, 65000.1, 65504.0,
        3.4e38, 3.4000001e38, -3.4e38,
        1 + 2j, (3.4e38 + 0j), True, False,
        # documented, permanent gap: outside i64/u64 range -- real numpy
        # silently returns dtype('O'); anionpy has no object dtype and raises
        # OverflowError instead. Included so this shows up as an honest
        # FAIL in the differential report (see this file's module
        # docstring and toplevel.py's min_scalar_type non-declaration
        # comment) rather than being hidden by omission.
        -9223372036854775809, 18446744073709551616,
    ]
    return [(repr(v), (v,), {}) for v in values]


DTYPEINFO_SPECS["min_scalar_type"] = ItemSpec(
    name="min_scalar_type", kind="custom", custom_cases=min_scalar_type_cases,
    numpy_path="min_scalar_type", ionp_path="min_scalar_type", scalar_like=True,
)


# ---------------------------------------------------------------------------
# iinfo
# ---------------------------------------------------------------------------

def _np_iinfo_probe(dtype):
    i = np.iinfo(dtype)
    return (i.min, i.max, i.bits, str(i.dtype), repr(i))


def _ionp_iinfo_probe(dtype):
    import anionpy
    i = anionpy.iinfo(dtype)
    return (i.min, i.max, i.bits, i.dtype.name, repr(i))


def iinfo_cases():
    int_dtypes = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
    cases = [(d, (d,), {}) for d in int_dtypes]
    for bad in ["bool", "float16", "float32", "float64", "complex64", "complex128"]:
        cases.append((f"error_{bad}", (bad,), {}))
    cases.append(("builtin_int", (int,), {}))
    cases.append(("builtin_bool", (bool,), {}))
    # CLOSED 2026-08-05 (was a documented gap): `np.iinfo` resolves its
    # argument the same way `np.dtype()` does -- try `np.dtype(x)`
    # directly, and if THAT raises, fall back to `np.dtype(type(x))`,
    # which itself never raises (an unrecognized type resolves to numpy's
    # generic `object` dtype, kind `'O'`). A bare ndarray fails the first
    # step (`np.dtype(some_array)` -> "Cannot construct a dtype from an
    # array", live-verified) and its TYPE is not a recognized dtype name
    # either, so it lands on `'O'` via the second step -- same path as a
    # plain list or a bare `object()` below, not a special "reject array"
    # case. See `ionp-py/src/dtypeinfo.rs`'s `resolve_iinfo_dtype`.
    cases.append(("array_0d_int32", (np.array(3, dtype=np.int32),), {}))
    cases.append(("array_1d_int32", (np.array([3], dtype=np.int32),), {}))
    cases.append(("array_2d_int8", (np.array([[1, 2], [3, 4]], dtype=np.int8),), {}))
    cases.append(("array_1d_float64", (np.array([3.0]),), {}))
    cases.append(("array_1d_bool", (np.array([True, False]),), {}))
    cases.append(("array_empty_int8", (np.array([], dtype=np.int8),), {}))
    cases.append(("array_0d_bool", (np.array(True),), {}))
    # bare Python `int`/`float`/`bool` VALUES (not types): `np.dtype(3)`
    # raises (a value is not a type-like spec), so these fall to
    # `np.dtype(type(x))` -- `int` -> int64, `float`/`bool` -> their own
    # (rejecting) kind. `anionpy.iinfo(3)` used to raise `TypeError: 3 is not
    # a supported integer type`; numpy accepts it as int64 (defect 1).
    cases.append(("bare_int_value", (3,), {}))
    cases.append(("bare_int_value_negative", (-7,), {}))
    cases.append(("bare_float_value", (3.5,), {}))
    cases.append(("bare_bool_value_true", (True,), {}))
    cases.append(("bare_bool_value_false", (False,), {}))
    cases.append(("bare_complex_value", (1 + 2j,), {}))
    cases.append(("none_value", (None,), {}))
    cases.append(("list_of_ints", ([1, 2],), {}))
    cases.append(("object_instance", (object(),), {}))
    cases.append(("ellipsis_value", (...,), {}))
    # numpy scalar VALUES (not types): `np.dtype(np.int8(3))` succeeds
    # directly (first step), resolving to the scalar's own dtype.
    cases.append(("numpy_scalar_int8", (np.int8(3),), {}))
    cases.append(("numpy_scalar_uint8", (np.uint8(5),), {}))
    cases.append(("numpy_scalar_bool_true", (np.bool_(True),), {}))
    cases.append(("numpy_scalar_float32", (np.float32(1.5),), {}))
    # a real `np.dtype` instance directly (first step, trivial identity).
    cases.append(("dtype_instance_int32", (np.dtype("int32"),), {}))
    cases.append(("dtype_instance_bool", (np.dtype("bool"),), {}))
    return cases


DTYPEINFO_SPECS["iinfo"] = ItemSpec(
    name="iinfo", kind="custom", custom_cases=iinfo_cases,
    numpy_adapter=_np_iinfo_probe, ionp_adapter=_ionp_iinfo_probe,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# finfo -- NOT declared (see toplevel.py), registered anyway so the value-
# level correctness is continuously checked and any regression is visible.
# ---------------------------------------------------------------------------

_FINFO_FIELDS = (
    "eps", "epsneg", "max", "min", "tiny", "smallest_normal",
    "smallest_subnormal", "resolution", "precision", "bits",
    "iexp", "nexp", "nmant", "machep", "negep", "minexp", "maxexp",
)


def _np_finfo_probe(dtype):
    # STRENGTHENED 2026-08-06 (Monday). This used to wrap every float field
    # in `float(...)` and every int field in `int(...)`, which LAUNDERED the
    # exact defect that the sibling `finfo.<attr>` items were failing on:
    # anionpy returned plain Python floats where numpy returns dtype-typed
    # scalars, and casting both sides to `float` made that invisible. The
    # `finfo` item therefore reported `pass` for as long as the divergence
    # existed. Returning the attributes untouched means this item now also
    # witnesses attribute TYPE, not just value.
    f = np.finfo(dtype)
    return tuple(getattr(f, name) for name in _FINFO_FIELDS) + (str(f.dtype),)


def _ionp_finfo_probe(dtype):
    import anionpy
    f = anionpy.finfo(dtype)
    return tuple(getattr(f, name) for name in _FINFO_FIELDS) + (f.dtype.name,)


def finfo_cases():
    return [(d, (d,), {}) for d in ["float16", "float32", "float64", "complex64", "complex128"]] + [
        ("error_int8", ("int8",), {}),
        ("error_bool", ("bool",), {}),
        ("builtin_float", (float,), {}),
        # REGRESSION GUARD added 2026-08-06 (Monday). numpy special-cases
        # None in `finfo` and ONLY in `finfo`:
        #     np.finfo(None) -> TypeError: dtype must not be None
        #     np.iinfo(None) -> ValueError: Invalid integer data type 'f'.
        # anionpy used to raise `TypeError: data type 'None' not inexact` here
        # -- right class, wrong message, so the harness's exact-message
        # comparison would have caught it if anything had ever asked. The
        # corpus never passed None, so it stayed invisible through the whole
        # finfo cluster going green. Found out-of-corpus, AFTER the suite
        # said pass.
        #
        # Bite MEASURED, not asserted: with the `dtype.is_none()` branch in
        # dtypeinfo.rs replaced by a comment and the .so rebuilt, this item
        # runs 9 cases / 1 failure, and the one failure is `error_none`:
        #     numpy (22 chars): 'dtype must not be None'
        #     anionpy  (28 chars): "data type 'None' not inexact"
        # Restoring the branch and rebuilding returns it to 9/0.
        #
        # That proof initially came back GREEN-WITH-GUARD-DISABLED, which
        # would have been a false all-clear. The fault was the scratch
        # runner, not the guard: `harness.run_case` returns a plain 4-tuple
        # `(ok, message, tolerant, max_ulp)`, and the runner was reading a
        # non-existent `.ok`/`.mismatch` attribute off it, so its pass check
        # was `True` unconditionally. Anyone writing a targeted runner must
        # unpack that tuple positionally -- and must sanity-check that their
        # instrument can report RED at all before trusting it to report green.
        ("error_none", (None,), {}),
        # ADDED 2026-08-06 (Monday, task #28): dtype-resolution + error-path
        # divergences, re-measured against numpy 2.5.1's actual
        # `finfo.__new__` (`numpy/_core/getlimits.py`, read directly): try
        # `np.dtype(dtype)`, and on failure fall back to
        # `np.dtype(type(dtype))` -- the SAME two-step resolution `iinfo`
        # already uses (`resolve_dtype_or_object` in dtypeinfo.rs, now
        # shared by both). Mirrors `iinfo_cases()`'s equivalent rows above
        # so both items exercise the same resolution shape.
        #
        # The dangerous direction (anionpy MORE PERMISSIVE than numpy): a bare
        # ndarray used AS the dtype argument. Real numpy's first resolution
        # step explicitly REJECTS an array ("Cannot construct a dtype from
        # an array"), then its type-fallback step lands on the generic
        # object dtype, so `np.finfo(some_array)` raises. A prior version of
        # `anionpy.finfo` instead read `.dtype` straight off the array (the
        # SAME coercion `iinfo`/`can_cast`/`promote_types` correctly use for
        # SCALARS-with-a-`.dtype`-attribute, wrong here) and SUCCEEDED.
        ("array_0d_float32", (np.array(1.5, dtype=np.float32),), {}),
        ("array_1d_float32", (np.array([1.5, 2.5], dtype=np.float32),), {}),
        ("array_2d_float64", (np.array([[1.0, 2.0], [3.0, 4.0]]),), {}),
        ("array_empty_float32", (np.array([], dtype=np.float32),), {}),
        ("array_0d_complex64", (np.array(1 + 2j, dtype=np.complex64),), {}),
        # Wrong exception CLASS/TEXT for everything `coerce_dtype_like`
        # can't resolve at all: numpy's type-fallback step still finds a
        # real (if wrong-kind) dtype and raises `ValueError` naming it;
        # anionpy used to raise a generic `TypeError` with an invented message
        # instead of resolving further. `iinfo_cases()` above already
        # covers the identical inputs; mirrored here for `finfo`.
        ("builtin_int", (int,), {}),
        ("builtin_bool", (bool,), {}),
        ("builtin_complex", (complex,), {}),
        ("bare_int_value", (3,), {}),
        ("bare_int_value_negative", (-7,), {}),
        ("bare_bool_value_true", (True,), {}),
        ("bare_bool_value_false", (False,), {}),
        ("list_of_floats", ([1.5, 2.5],), {}),
        ("object_instance", (object(),), {}),
        # numpy scalar / dtype-instance VALUES: first resolution step
        # succeeds directly (own dtype), already worked before this fix and
        # kept here as a non-regression guard now that the resolution path
        # changed underneath.
        ("numpy_scalar_float32", (np.float32(1.5),), {}),
        ("numpy_scalar_int8", (np.int8(3),), {}),
        ("dtype_instance_float32", (np.dtype("float32"),), {}),
        ("dtype_instance_int16", (np.dtype("int16"),), {}),
    ]


DTYPEINFO_SPECS["finfo"] = ItemSpec(
    name="finfo", kind="custom", custom_cases=finfo_cases,
    numpy_adapter=_np_finfo_probe, ionp_adapter=_ionp_finfo_probe,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# finfo -- instance caching / identity / `__eq__` / class name divergences
# (task #28's other 3 characterized classes; the 4th, error-path, is folded
# into `finfo_cases()` above since it shares that item's per-dtype-input
# shape). numpy's `finfo.__new__` caches ONE instance per RESOLVED dtype
# (`np.finfo('f4') is np.finfo('f4')` True), does NOT define its own
# `__eq__`/`__hash__` (confirmed: `'__eq__' not in np.finfo.__dict__`), so
# `==` on two DIFFERENT instances falls back to identity -- the cache is
# what makes two calls for the SAME resolved dtype compare equal, not a
# comparison method -- and `type(x).__name__` is the literal string
# `"finfo"`, not `"FInfo"`.
# ---------------------------------------------------------------------------

def _np_finfo_identity_probe(dtype):
    a = np.finfo(dtype)
    b = np.finfo(dtype)
    return (a is b, a == b, type(a).__name__)


def _ionp_finfo_identity_probe(dtype):
    import anionpy
    a = anionpy.finfo(dtype)
    b = anionpy.finfo(dtype)
    return (a is b, a == b, type(a).__name__)


def finfo_identity_cases():
    return [(d, (d,), {}) for d in ["float16", "float32", "float64", "complex64", "complex128"]]


DTYPEINFO_SPECS["finfo_identity"] = ItemSpec(
    name="finfo_identity", kind="custom", custom_cases=finfo_identity_cases,
    numpy_adapter=_np_finfo_identity_probe, ionp_adapter=_ionp_finfo_identity_probe,
    scalar_like=True,
)


def _np_finfo_cross_probe(d1, d2):
    a = np.finfo(d1)
    b = np.finfo(d2)
    return (a is b, a == b)


def _ionp_finfo_cross_probe(d1, d2):
    import anionpy
    a = anionpy.finfo(d1)
    b = anionpy.finfo(d2)
    return (a is b, a == b)


def finfo_cross_spelling_cases():
    return [
        # Different SPELLINGS of the identical concrete dtype must all
        # cache to the SAME instance -- measured live against numpy 2.5.1,
        # not assumed from the class name alone.
        ("f4_vs_float32", ("f4", "float32"), {}),
        ("float32_vs_type", ("float32", np.float32), {}),
        ("float32_vs_dtype_instance", ("float32", np.dtype("float32")), {}),
        # A complex dtype's finfo IS its component float's cached instance
        # (`np.finfo('complex64') is np.finfo('float32')`, live-verified) --
        # not merely equal-valued, the SAME object.
        ("complex64_vs_float32_redirect", ("complex64", "float32"), {}),
        ("complex128_vs_float64_redirect", ("complex128", "float64"), {}),
        # Genuinely distinct dtypes must NOT share an instance.
        ("float32_vs_float64_distinct", ("float32", "float64"), {}),
    ]


DTYPEINFO_SPECS["finfo_cross_spelling"] = ItemSpec(
    name="finfo_cross_spelling", kind="custom", custom_cases=finfo_cross_spelling_cases,
    numpy_adapter=_np_finfo_cross_probe, ionp_adapter=_ionp_finfo_cross_probe,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# typename
# ---------------------------------------------------------------------------

def typename_cases():
    chars = ["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q",
             "e", "f", "d", "F", "D", "S1", "U1", "O", "V"]
    cases = [(c, (c,), {}) for c in chars]
    cases.append(("unknown", ("zz",), {}))
    return cases


DTYPEINFO_SPECS["typename"] = ItemSpec(
    name="typename", kind="custom", custom_cases=typename_cases,
    numpy_path="typename", ionp_path="typename", scalar_like=True,
)


# ---------------------------------------------------------------------------
# mintypecode
# ---------------------------------------------------------------------------

def mintypecode_cases():
    variants = ["f", "d", "F", "D", "i", "S"]
    cases = [(v, (v,), {}) for v in variants]
    cases.append(("list_fd", (["f", "d"],), {}))
    cases.append(("list_fF", (["f", "F"],), {}))
    cases.append(("custom_typeset", ("f", "GDF", "d"), {}))
    return cases


DTYPEINFO_SPECS["mintypecode"] = ItemSpec(
    name="mintypecode", kind="custom", custom_cases=mintypecode_cases,
    numpy_path="mintypecode", ionp_path="mintypecode", scalar_like=True,
)


# ---------------------------------------------------------------------------
# isdtype -- NOT declared (see toplevel.py). `dtype` must be a REAL dtype
# object on both sides -- numpy's own `isdtype` rejects a bare dtype-name
# string (confirmed, fixed to match this session), so calling the harness
# with a shared string argument would never actually exercise the
# string/tuple `kind`-matching logic once that rejection matches. Adapters
# instead build each side's OWN dtype object from the same name (real
# `np.dtype(name)` for numpy, `anionpy.promote_types(name, name)` as a self-
# promotion name-lookup for anionpy -- there is no public string constructor
# for `anionpy.dtype`) before calling `isdtype`, so the string/tuple `kind`
# cases actually PASS and demonstrate the working logic, while the
# trailing scalar-type-class case is still expected to FAIL (documented,
# permanent gap: no scalar type hierarchy yet, owned by another agent).
# ---------------------------------------------------------------------------

def _isdtype_resolve(v, mod):
    """Map one `isdtype` argument onto the module under test.

    Three spellings, so the SAME case tuple exercises both libraries with
    their OWN objects where that is the honest comparison:
      * a plain ``str`` is a dtype NAME -> that module's dtype object;
      * ``"cls:<name>"`` is a scalar type CLASS -> that module's class,
        which is the operand-parity form (anionpy is asked about `anionpy.int8`,
        numpy about `np.int8`) and the only way to test that anionpy answers
        this question about its OWN type hierarchy and not merely about
        numpy's;
      * anything else passes through UNCHANGED and is therefore shared by
        both sides -- used for the foreign-numpy-class and the rejection
        rows, where sharing the object is the point.
    A tuple is resolved element-wise so mixed `kind` tuples work.
    """
    if isinstance(v, tuple):
        return tuple(_isdtype_resolve(x, mod) for x in v)
    if isinstance(v, str) and v.startswith("cls:"):
        name = v[4:]
        # numpy 2 spells the bool scalar class `np.bool_` (its `__name__`
        # is the bare "bool"); anionpy mirrors that spelling.
        alt = "bool_" if name == "bool" else name
        for cand in (name, alt):
            if hasattr(mod, cand):
                return getattr(mod, cand)
        # anionpy genuinely has no class for this name (`longdouble`,
        # `datetime64`, `void`, ... -- dtypes anionpy does not implement), so
        # the FOREIGN numpy class is the only spelling that exists. That is
        # not a cheat: numpy accepts these as a `kind`/`dtype` and answers
        # False rather than raising, and anionpy must do the same for the
        # numpy object a real caller would actually hand it. The rows where
        # anionpy DOES own the class still resolve to anionpy's own above, so
        # operand parity is preserved wherever parity is possible.
        return getattr(np, alt if hasattr(np, alt) else name)
    return v


def _np_isdtype_probe(dtype_name, kind):
    d = _isdtype_resolve(dtype_name, np)
    if isinstance(d, str):
        d = np.dtype(d)
    return np.isdtype(d, _isdtype_resolve(kind, np))


def _ionp_isdtype_probe(dtype_name, kind):
    import anionpy
    d = _isdtype_resolve(dtype_name, anionpy)
    if isinstance(d, str):
        d = anionpy.promote_types(d, d)
    return anionpy.isdtype(d, _isdtype_resolve(kind, anionpy))


def isdtype_cases():
    cases = [
        ("bool_kind", ("bool", "bool"), {}),
        ("signed_integer", ("int8", "signed integer"), {}),
        ("unsigned_integer", ("uint8", "unsigned integer"), {}),
        ("integral", ("int8", "integral"), {}),
        ("real_floating", ("float32", "real floating"), {}),
        ("complex_floating", ("complex64", "complex floating"), {}),
        ("numeric", ("int8", "numeric"), {}),
        ("tuple_kind", ("int8", ("integral", "real floating")), {}),
        ("bad_kind_string", ("int8", "bogus_kind"), {}),
        # WAS a documented gap ("numpy scalar-type CLASS kind is out of
        # scope"), CLOSED 2026-08-04: numpy ACCEPTS a scalar type class as
        # `kind` and simply reports no match for an abstract one, so the
        # old TypeError was a divergence, not a scope boundary. Uses the
        # adapter (real dtype objects on both sides) so the `dtype`
        # argument itself isn't rejected first, which would mask this the
        # way a bare string does.
        ("scalar_type_class_abstract", ("int8", np.integer), {}),
    ]
    cases += _isdtype_scalar_class_cases()
    return cases


_ISDTYPE_CONCRETE = [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float16", "float32", "float64", "complex64", "complex128",
]

# Every name in numpy's `allTypes`, i.e. exactly what `_preprocess_dtype`
# accepts. The ABSTRACT ones and the ones anionpy has no dtype for are the
# load-bearing half: numpy returns False for them, it does NOT raise, so an
# implementation that only knows anionpy's own 14 concrete dtypes turns a
# `False` into a `TypeError` on every one of these rows.
_ISDTYPE_ALL_SCALAR_NAMES = [
    "bool", "bytes_", "character", "clongdouble", "complex128", "complex64",
    "complexfloating", "datetime64", "flexible", "float16", "float32",
    "float64", "floating", "generic", "inexact", "int16", "int32", "int64",
    "int8", "integer", "longdouble", "longlong", "number", "object_",
    "signedinteger", "str_", "timedelta64", "uint16", "uint32", "uint64",
    "uint8", "ulonglong", "unsignedinteger", "void",
]


def _isdtype_scalar_class_cases():
    """Scalar-type-class `kind`/`dtype` spellings (see `_isdtype_resolve`).

    Three axes, each of which the string-only implementation got wrong:
      1. `kind` as a class -- concrete (`np.int8`, must MATCH the same
         dtype and only that one) and abstract (`np.integer`, must NOT
         match anything and must NOT raise);
      2. `dtype` as a class, which numpy's own docstring uses
         (`np.isdtype(np.float32, "real floating")` -> True), including
         ABSTRACT classes as the `dtype` argument, which are legal and
         match nothing;
      3. MIXED tuples. The old `Vec<String>` extraction failed on the whole
         tuple the moment one element was not a string, so `("integral",
         np.integer)` -- True in numpy -- discarded even the string half.
    The `cls:` rows ask each library about its OWN class object; the bare
    `np.*` rows hand anionpy a foreign numpy class, which is the form real
    interop code uses. Both must agree.
    """
    cases = []
    for name in _ISDTYPE_CONCRETE:
        for kname in _ISDTYPE_ALL_SCALAR_NAMES:
            cases.append((f"kindcls/{name}/{kname}", (name, f"cls:{kname}"), {}))
        cases.append((f"dtypecls/{name}", (f"cls:{name}", "numeric"), {}))
        cases.append((f"dtypecls_self/{name}", (f"cls:{name}", f"cls:{name}"), {}))
    for aname in ["integer", "floating", "number", "generic", "signedinteger",
                  "inexact", "object_", "str_", "void"]:
        cases.append((f"dtypeabs/{aname}", (f"cls:{aname}", "numeric"), {}))
        cases.append((f"dtypeabs_self/{aname}", (f"cls:{aname}", f"cls:{aname}"), {}))
    cases += [
        ("tuple_mixed", ("int8", ("integral", np.integer)), {}),
        ("tuple_mixed_cls", ("int8", ("integral", "cls:integer")), {}),
        ("tuple_classes", ("int8", ("cls:int8", "cls:float64")), {}),
        ("tuple_classes_nomatch", ("int8", ("cls:uint8", "cls:float64")), {}),
        ("tuple_empty", ("int8", ()), {}),
        # ORDERING: numpy walks the tuple in order and raises at the first
        # bad element, so the unknown-kind ValueError must beat the later
        # non-dtype TypeError -- pre-validating the tuple inverts these.
        ("tuple_badstr_then_badtype", ("int8", ("nope", int)), {}),
        ("tuple_badtype_then_badstr", ("int8", (int, "nope")), {}),
        ("kind_python_int_rejected", ("int8", int), {}),
        ("kind_python_str_rejected", ("int8", str), {}),
        ("kind_none_rejected", ("int8", None), {}),
        ("kind_number_rejected", ("int8", 1), {}),
        ("kind_ndarray_rejected", ("int8", np.ndarray), {}),
        ("kind_list_rejected", ("int8", [np.integer]), {}),
        ("dtype_python_type_rejected", (int, "numeric"), {}),
        ("dtype_none_rejected", (None, "numeric"), {}),
    ]
    return cases


DTYPEINFO_SPECS["isdtype"] = ItemSpec(
    name="isdtype", kind="custom", custom_cases=isdtype_cases,
    numpy_adapter=_np_isdtype_probe, ionp_adapter=_ionp_isdtype_probe,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# isdtype -- bare-string `dtype` argument rejection. numpy and anionpy both
# reject a bare dtype-name string for the `dtype` argument (confirmed
# byte-identical TypeError text this session); this is registered as its
# own item using the ORIGINAL direct-call form (shared string argument on
# both sides) specifically to keep that rejection path under differential
# coverage, since the adapter-based `isdtype` item above never passes a
# bare string through.
# ---------------------------------------------------------------------------

def isdtype_bare_string_rejection_cases():
    return [
        ("bare_string_dtype_rejected", ("int8", "signed integer"), {}),
        ("bare_string_dtype_rejected_bool", ("bool", "bool"), {}),
        # a bare ndarray used as the `dtype` argument's value is rejected
        # the same as any other non-dtype object -- included for
        # completeness, expected to match.
        ("array_as_dtype_rejected", (np.array([1, 2]), "signed integer"), {}),
    ]


# convert_ionp_args=False here is a CORRECTION, not a relaxation, and the
# distinction matters enough to spell out. This item was failing, and the
# failure was manufactured by the harness itself:
#
#     np.isdtype(np.array([1,2]), "signed integer")
#       -> TypeError: dtype argument must be a NumPy dtype, but it is a
#                     <class 'numpy.ndarray'>.
#     anionpy.isdtype(np.array([1,2]), "signed integer")        # raw, as numpy got it
#       -> TypeError: dtype argument must be a NumPy dtype, but it is a
#                     <class 'numpy.ndarray'>.               # IDENTICAL
#     anionpy.isdtype(anionpy.array([1,2]), "signed integer")      # what the harness sent
#       -> TypeError: dtype argument must be a NumPy dtype, but it is a
#                     <class 'anionpy.ndarray'>.                # differs
#
# The default `convert_ionp_args=True` rewrote the numpy array into an
# anionpy.ndarray before anionpy ever saw it, so the two sides were not run on the
# same input, and the only difference in the output was the type name anionpy
# correctly reported for the different object it was handed. Both behaviours
# are right; the harness simply was not comparing like with like.
#
# Turning the flag off makes the comparison STRICTER, not looser: anionpy now
# has to accept a foreign numpy.ndarray and reproduce numpy's message
# verbatim, which it does (verified live, /tmp/mg_isdt.py, 2026-08-02). The
# two bare-string cases in this item are unaffected -- strings were never
# converted.
#
# Note for anyone auditing the ledger: this was a false FAIL, which is the
# harmless direction. The same mechanism was deliberately searched for false
# PASSES (a conversion that normalises away a real divergence) across ~15
# adversarial cases on 6 functions; none was demonstrated. That is a genuine
# negative result, and it is recorded here rather than in a report because
# the next person to read this flag will want to know it was asked.
DTYPEINFO_SPECS["isdtype_bare_string_rejection"] = ItemSpec(
    name="isdtype_bare_string_rejection", kind="custom",
    custom_cases=isdtype_bare_string_rejection_cases,
    numpy_path="isdtype", ionp_path="isdtype", scalar_like=True,
    convert_ionp_args=False,
)
