"""Differential ItemSpecs for the comparison/predicate/introspection block:
`ndim, shape, size, isscalar, iterable, isfortran, iscomplexobj, isrealobj,
array_equal, array_equiv, allclose` -- all pure-Python wrappers defined in
`anionpy/__init__.py` (no Rust counterpart of their own; each is a thin shim
over existing Rust-backed ndarray attributes/ufuncs, exactly like real
numpy's own `_core/fromnumeric.py`/`_core/numeric.py`/
`lib/_function_base_impl.py`/`lib/_type_check_impl.py` sources).

Deliberately NOT registered here, with reasons (see also
`anionpy/_state/toplevel.py`'s block comment for this family):

  - `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`:
    already registered/tested elsewhere as ufuncs (`ufunc_registry.py`);
    not part of this file's scope, and not declared -- see toplevel.py's
    2026-08-02 correction comment (object/string-dtype operand gap).
  - `isclose`: real numpy's contract is dual-typed on the input -- scalar
    input returns a `numpy.bool`, array input a `numpy.ndarray` (verified
    against real numpy 2.5.1). Not implemented, not registered, not
    declared -- for `isclose` specifically, which has its own `rtol`/`atol`/
    `equal_nan` surface still to be built, NOT for the dual-typing, which
    is handled (see next paragraph).

    CORRECTION 2026-08-03: this paragraph used to cover `iscomplex`,
    `isreal`, `isneginf` and `isposinf` too, and gave as the reason that
    "`numpy.bool_` is a specific class belonging to real numpy; no
    ionp-side object can ever be `is`-identical to it without anionpy
    importing and calling into real numpy at runtime, which is out of
    bounds." That was reasoned from, never measured, and it is FALSE:
    anionpy's extension module already constructs canonical numpy scalar
    objects for 0-d results (`numpy_scalar_from_0d` in `ionp-py`, via
    rust-numpy's type objects) -- the same way any C extension does, with
    anionpy's Python side never calling numpy for an answer. All four, plus
    `real` and `imag`, are now implemented, registered and tested at the
    bottom of this file. The stale reason is left visible here rather than
    deleted, because the failure mode worth remembering is not the missing
    functions, it is that a plausible architectural excuse sat in a
    docstring for a day and nobody pointed a probe at it.
  - `isin`: already registered in `setops_cases.py` (different source
    file, backed by Rust `ionp-core::setops::isin`, not a pure-Python
    wrapper).

The eleven items in the ORIGINAL block below return a plain Python
`bool`/`int`/`tuple` -- never
an `anionpy.ndarray` -- by construction (each wrapper unwraps any internal 0-d
`anionpy.ndarray` intermediate via `.item()` before returning). `scalar_like=
True` is therefore the HONEST choice for every entry below, not a workaround:
harness.py's own docstring for that flag says "do NOT set this for anything
that returns an array," and none of these ever do -- verified per-item via
`type(numpy_result) is type(ionp_result)` probes before this file was
written (2026-08-02 session). That claim is scoped to those eleven; the six
dual-typed items added 2026-08-03 use a typed projection instead, for the
reasons given in their own block comment.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)

import numpy as np

import corpus
from registry import ItemSpec

PREDICATE_SPECS: dict[str, ItemSpec] = {}


# ---------------------------------------------------------------------------
# ndim / shape / size -- single-array-or-array_like introspection.
# ---------------------------------------------------------------------------

def _unary_array_cases():
    return [(c.label, (c.value,), {}) for c in corpus.unary_corpus()]


def _ndim_shape_cases():
    out = _unary_array_cases()
    # array_like (non-ndarray) inputs -- these never reach corpus.py, which
    # only hands out real numpy.ndarray instances.
    out += [
        ("python_int", (5,), {}),
        ("python_float", (5.5,), {}),
        ("python_list_1d", ([1, 2, 3],), {}),
        ("python_list_2d", ([[1, 2], [3, 4]],), {}),
        ("python_nested_ragged_free", ([[1, 2, 3], [4, 5, 6]],), {}),
    ]
    return out


PREDICATE_SPECS["ndim"] = ItemSpec(
    name="ndim", kind="custom", custom_cases=_ndim_shape_cases,
    numpy_path="ndim", ionp_path="ndim", scalar_like=True,
)
PREDICATE_SPECS["shape"] = ItemSpec(
    name="shape", kind="custom", custom_cases=_ndim_shape_cases,
    numpy_path="shape", ionp_path="shape", scalar_like=True,
)


def _size_cases():
    out = [(c.label, (c.value,), {}) for c in corpus.unary_corpus()
           if c.value.ndim >= 1]
    for c in corpus.unary_corpus():
        if c.value.ndim >= 2:
            out.append((f"{c.label}/axis0", (c.value,), {"axis": 0}))
            out.append((f"{c.label}/axis_last", (c.value,), {"axis": c.value.ndim - 1}))
            out.append((f"{c.label}/axis_neg1", (c.value,), {"axis": -1}))
            out.append((f"{c.label}/axis_all", (c.value,), {"axis": tuple(range(c.value.ndim))}))
    out += [
        ("python_list_1d", ([1, 2, 3],), {}),
        ("python_list_2d", ([[1, 2], [3, 4]],), {}),
        ("python_int", (5,), {}),
    ]
    # error paths: out-of-range axis, duplicate axis
    base2d = np.zeros((2, 3))
    out.append(("axis_out_of_range", (base2d,), {"axis": 5}))
    out.append(("axis_out_of_range_neg", (base2d,), {"axis": -5}))
    out.append(("axis_duplicate", (base2d,), {"axis": (0, 0)}))
    return out


PREDICATE_SPECS["size"] = ItemSpec(
    name="size", kind="custom", custom_cases=_size_cases,
    numpy_path="size", ionp_path="size", scalar_like=True,
)


# ---------------------------------------------------------------------------
# isscalar / iterable -- accept literally anything, including non-array
# Python objects; corpus.py's arrays cover the ndarray branch, a manual
# list covers every scalar/non-scalar Python type real numpy's own
# `isscalar` source (`isinstance(x, generic) or type(x) in ScalarType or
# isinstance(x, numbers.Number)`) branches on.
# ---------------------------------------------------------------------------

def _isscalar_cases():
    out = _unary_array_cases()
    out += [
        ("py_int", (5,), {}),
        ("py_float", (5.5,), {}),
        ("py_complex", (5j,), {}),
        ("py_bool", (True,), {}),
        ("py_str", ("numpy",), {}),
        ("py_bytes", (b"bytes",), {}),
        ("py_memoryview", (memoryview(b"abc"),), {}),
        ("py_list", ([1, 2],), {}),
        ("py_tuple", ((1, 2),), {}),
        ("py_none", (None,), {}),
        ("py_object", (object(),), {}),
        ("py_dict", ({"a": 1},), {}),
        # Deliberately NOT included: `np.int64(5)`/`np.bool_(True)`/etc as a
        # shared arg. `ItemSpec.kind="custom"` passes the SAME args object
        # to both `resolve_numpy()` and `resolve_ionp()`
        # (`make_ionp_array_converter` only converts `numpy.ndarray`, never
        # a bare numpy SCALAR -- see its own docstring, "converting those
        # would defeat the scalar/varargs call-form coverage this suite
        # exists to provide"), so a real numpy scalar reaches the anionpy side
        # completely unconverted. `anionpy.isscalar(np.int64(5))` correctly
        # returning False is not a bug -- anionpy has no way to recognize a
        # foreign library's own scalar class, exactly as real numpy's own
        # `isscalar` would not recognize an `anionpy.int64` either (symmetric,
        # verified directly). The genuinely comparable question --
        # "does anionpy.isscalar(anionpy.int64(5)) match np.isscalar(np.int64(5))"
        # -- was verified directly instead, see toplevel.py's `isscalar`
        # declaration comment; the shared-args harness has no clean way to
        # express "pass each side its OWN library's scalar" for a single
        # generic ItemSpec, so it is recorded as evidence there rather than
        # forced into a misleading registry case here.
    ]
    return out


PREDICATE_SPECS["isscalar"] = ItemSpec(
    name="isscalar", kind="custom", custom_cases=_isscalar_cases,
    numpy_path="isscalar", ionp_path="isscalar", scalar_like=True,
)


def _iterable_cases():
    out = _unary_array_cases()
    out += [
        ("py_list", ([1, 2, 3],), {}),
        ("py_int", (2,), {}),
        ("py_str", ("abc",), {}),
        ("py_iterator", (iter([1, 2]),), {}),
        ("py_none", (None,), {}),
        ("py_dict", ({"a": 1},), {}),
        ("py_generator", ((x for x in range(3)),), {}),
    ]
    return out


PREDICATE_SPECS["iterable"] = ItemSpec(
    name="iterable", kind="custom", custom_cases=_iterable_cases,
    numpy_path="iterable", ionp_path="iterable", scalar_like=True,
)


# ---------------------------------------------------------------------------
# isfortran -- needs real contiguity variety: C-contig, F-contig, a
# transposed C-contig array (which becomes F-contig-not-C-contig), a 1-D
# array (both C and F simultaneously -> False either way), and a
# non-contiguous view (neither).
# ---------------------------------------------------------------------------

def _isfortran_cases():
    out = _unary_array_cases()
    c2d = np.array([[1, 2, 3], [4, 5, 6]], order="C")
    f2d = np.asfortranarray(c2d)
    out += [
        ("c_contig_2d", (c2d,), {}),
        ("f_contig_2d", (f2d,), {}),
        ("transposed_c_is_f", (c2d.T,), {}),
        ("transposed_f_is_c", (f2d.T,), {}),
        ("1d", (np.array([1, 2, 3]),), {}),
        ("noncontig_view", (c2d[:, ::-1],), {}),
        ("0d", (np.array(5),), {}),
    ]
    return out


PREDICATE_SPECS["isfortran"] = ItemSpec(
    name="isfortran", kind="custom", custom_cases=_isfortran_cases,
    numpy_path="isfortran", ionp_path="isfortran", scalar_like=True,
)


# ---------------------------------------------------------------------------
# iscomplexobj / isrealobj -- dtype-family check only, values irrelevant.
# ---------------------------------------------------------------------------

def _complexobj_cases():
    out = _unary_array_cases()
    out += [
        ("py_int", (5,), {}),
        ("py_complex", (5 + 2j,), {}),
        ("py_float", (5.0,), {}),
        ("py_bool", (True,), {}),
    ]
    return out


PREDICATE_SPECS["iscomplexobj"] = ItemSpec(
    name="iscomplexobj", kind="custom", custom_cases=_complexobj_cases,
    numpy_path="iscomplexobj", ionp_path="iscomplexobj", scalar_like=True,
)
PREDICATE_SPECS["isrealobj"] = ItemSpec(
    name="isrealobj", kind="custom", custom_cases=_complexobj_cases,
    numpy_path="isrealobj", ionp_path="isrealobj", scalar_like=True,
)


# ---------------------------------------------------------------------------
# array_equal / array_equiv -- pairs, including shape-mismatched,
# non-broadcastable, NaN-bearing, dtype-mixed, and non-array-convertible
# (`object()`) operands.
# ---------------------------------------------------------------------------

def _equal_pair_cases():
    out = [(p.label, (p.a, p.b), {}) for p in corpus.binary_corpus()]
    nan_f = np.array([1.0, np.nan, 3.0])
    out += [
        ("same_values", (np.array([1, 2, 3]), np.array([1, 2, 3])), {}),
        ("diff_values", (np.array([1, 2, 3]), np.array([1, 2, 4])), {}),
        ("shape_mismatch", (np.array([1, 2]), np.array([1, 2, 3])), {}),
        ("nan_default", (nan_f, nan_f.copy()), {}),
        ("nan_equal_nan_true", (nan_f, nan_f.copy()), {"equal_nan": True}),
        ("nan_int_no_nan_possible", (np.array([1, 2]), np.array([1, 2])),
         {"equal_nan": True}),
        ("not_convertible", (object(), [1, 2]), {}),
        ("mixed_dtype_equal_values", (np.array([1, 2], dtype=np.int32),
                                       np.array([1.0, 2.0], dtype=np.float64)), {}),
    ]
    return out


PREDICATE_SPECS["array_equal"] = ItemSpec(
    name="array_equal", kind="custom", custom_cases=_equal_pair_cases,
    numpy_path="array_equal", ionp_path="array_equal", scalar_like=True,
)


def _equiv_pair_cases():
    out = [(p.label, (p.a, p.b), {}) for p in corpus.binary_corpus()]
    out += [
        ("same_values", ([1, 2, 3], [1, 2, 3]), {}),
        ("broadcastable_row", ([[1, 2, 3]], [1, 2, 3]), {}),
        ("broadcastable_2row", (np.array([[1, 2, 3], [1, 2, 3]]), [1, 2, 3]), {}),
        ("not_broadcastable", ([1, 2], [1, 2, 3]), {}),
        ("not_convertible", (object(), [1, 2]), {}),
    ]
    return out


PREDICATE_SPECS["array_equiv"] = ItemSpec(
    name="array_equiv", kind="custom", custom_cases=_equiv_pair_cases,
    numpy_path="array_equiv", ionp_path="array_equiv", scalar_like=True,
)


# ---------------------------------------------------------------------------
# allclose -- only meaningful on numeric dtypes; corpus.binary_corpus()
# already includes int/float/complex/bool pairs, broadcasting, and mixed
# dtypes, plus explicit NaN cases here.
# ---------------------------------------------------------------------------

def _allclose_cases():
    out = [(p.label, (p.a, p.b), {}) for p in corpus.binary_corpus()]
    nan_f = np.array([1.0, np.nan])
    out += [
        ("close_default_tol", ([1e10, 1e-7], [1.00001e10, 1e-8]), {}),
        ("close_loose_tol", ([1e10, 1e-8], [1.00001e10, 1e-9]), {}),
        ("not_close", ([1e10, 1e-8], [1.0001e10, 1e-9]), {}),
        ("nan_default", (nan_f, nan_f.copy()), {}),
        ("nan_equal_nan_true", (nan_f, nan_f.copy()), {"equal_nan": True}),
        ("custom_rtol_atol", ([1.0, 2.0], [1.01, 2.01]), {"rtol": 0.02, "atol": 0.0}),
        ("broadcast_scalar", ([1.0, 2.0, 3.0], 1.0), {}),
    ]
    return out


PREDICATE_SPECS["allclose"] = ItemSpec(
    name="allclose", kind="custom", custom_cases=_allclose_cases,
    numpy_path="allclose", ionp_path="allclose", scalar_like=True,
)


# ---------------------------------------------------------------------------
# 0-d x explicit-axis boundary for `size` (2026-08-03).
#
# `size` sits in this file rather than reduction_cases.py, so the boundary
# sweep added there did not reach it -- and `_size_cases` above excludes a
# 0-d operand TWICE over: `ndim >= 1` gates even the no-argument form, and
# `ndim >= 2` gates every axis form. `np.size(np.array(3.0), axis=0)` was
# therefore never once compared, while the item stood declared "exact".
#
# Added via `extra_cases` (additive, never shadows an existing case) rather
# than by loosening those guards, which would silently change the meaning of
# the cases already there. Twelve axis forms x five operands: numpy's 0-d
# rule here is the same deliberate inconsistency swept elsewhere (scalar
# 0/-1 tolerated, 1/-2 not, () tolerated, (0,0) rejected), and reproducing
# it is the contract. `size` takes no `keepdims`.
# ---------------------------------------------------------------------------

def _size_zero_d_axis_cases():
    cases = []
    for dlabel, arr in [("float64", np.array(3.0)),
                        ("float64_nan", np.array(np.nan)),
                        ("float64_zero", np.array(0.0)),
                        ("int", np.array(7)),
                        ("bool", np.array(True))]:
        cases.append((f"zero_d/{dlabel}/axis_omitted", (arr,), {}))
        for alabel, ax in [("axis_none", None), ("axis_0", 0), ("axis_neg1", -1),
                           ("axis_1", 1), ("axis_neg2", -2), ("axis_empty_tuple", ()),
                           ("axis_tuple_0", (0,)), ("axis_tuple_neg1", (-1,)),
                           ("axis_tuple_1", (1,)), ("axis_tuple_dup", (0, 0)),
                           ("axis_tuple_dup_neg", (0, -1))]:
            cases.append((f"zero_d/{dlabel}/{alabel}", (arr,), {"axis": ax}))
    return cases


def _size_append_zero_d_axis_cases():
    spec = PREDICATE_SPECS.get("size")
    assert spec is not None, "0-d boundary: no PREDICATE_SPECS entry for 'size'"
    prev = spec.extra_cases

    def wrapped(prev=prev):
        base = list(prev()) if prev is not None else []
        return base + _size_zero_d_axis_cases()

    spec.extra_cases = wrapped


_size_append_zero_d_axis_cases()


# ---------------------------------------------------------------------------
# real / imag / iscomplex / isreal / isposinf / isneginf  (added 2026-08-03)
#
# THE RECORDED BLOCKER FOR FOUR OF THESE WAS MEASURED FALSE. This file's
# header used to say (of `iscomplex`/`isreal`/`isneginf`/`isposinf`):
# "`numpy.bool_` is a specific class belonging to real numpy; no ionp-side
# object can ever be `is`-identical to it without anionpy importing and calling
# into real numpy at runtime, which is out of bounds." That was reasoned
# from, not measured. It is wrong: anionpy's Rust layer already hands back real
# numpy scalar objects for 0-d results (`ionp-py`'s `numpy_scalar_from_0d`,
# via rust-numpy's own type objects) -- anionpy's *Python* never imports numpy
# for an answer, the extension module simply constructs the canonical scalar
# type the same way any C extension does. A live 6-function x 6-input probe
# (2026-08-03) found every scalar-returning case agreeing in BOTH exact type
# and repr; the only diffs were `numpy.ndarray` vs `anionpy.ndarray` on the
# array-returning cases, which is the universal cross-library difference
# every item in this suite has, not a defect. The header has been corrected.
#
# WHY THESE SIX NEED A PROJECTION AND NOT `scalar_like=True`:
# they are dual-typed on the input. `isreal(3.0)` is a Python `bool`,
# `isreal(np.array(3.0))` is a `numpy.bool`, `isreal(np.array([3.0]))` is an
# ndarray. `scalar_like=True` demands `type(np_out) is type(ionp_out)`,
# which is correct for the first two and structurally impossible for the
# third. Dropping to the array path instead would stop checking the exact
# scalar type -- and the scalar type is precisely the thing the (now
# falsified) blocker claimed we could not get right, so it is the last
# thing to stop checking. The projection below keeps BOTH: it reports the
# fully-qualified result type, normalising only the one pair that can never
# match (`numpy.ndarray`/`anionpy.ndarray` -> `"ARRAY"`), and reports dtype,
# shape and full values alongside it. A `builtins.bool` returned where
# numpy returns a `numpy.bool` is a mismatch here, as it should be.
#
# The raise-capture is `BaseException`, not `Exception`, because PyO3
# panics surface as `PanicException`, which derives from `BaseException` --
# an `except Exception` here would let a Rust panic escape and be scored as
# a harness error instead of the mismatch it is.
# ---------------------------------------------------------------------------

_DUAL_FNS = ("real", "imag", "iscomplex", "isreal", "isposinf", "isneginf")


def _typed_projection(fn, x):
    """(type-name, dtype, shape, values) -- or the exception identity."""
    try:
        r = fn(x)
    except BaseException as exc:  # noqa: BLE001 - see block comment above
        return ("raised", type(exc).__name__, str(exc))
    tn = f"{type(r).__module__}.{type(r).__name__}"
    if tn in ("numpy.ndarray", "anionpy.ndarray"):
        return ("ARRAY", str(r.dtype), tuple(r.shape), repr(r.tolist()))
    # A 0-d ndarray would have been caught above; anything here is a scalar,
    # and its exact class is part of the contract (`builtins.float` vs
    # `numpy.float64` is a real, user-visible difference).
    return ("SCALAR", tn, repr(r))


def _dual_adapter(mod, name):
    def adapter(x, _mod=mod, _name=name):
        return _typed_projection(getattr(_mod, _name), x)
    return adapter


def _dual_cases():
    """Inputs chosen to separate the six functions from each other and from
    the plausible-but-wrong implementations of each. Complex-with-zero-imag
    vs complex-with-negative-zero-imag separates a value test from a sign
    test; `nan`-imaginary separates `imag == 0` from `not iscomplex`; the
    bool/int/float16 arrays exercise the branch that never inspects values
    at all; the Python scalars exercise the `except AttributeError` fallback
    in `real`/`imag`; and complex input is the one case where `isposinf`/
    `isneginf` must RAISE (numpy propagates `signbit`'s TypeError).
    """
    cases = [
        ("c128_mixed", np.array([1 + 2j, 3 + 0j, complex(4, -0.0),
                                 complex(5, np.nan), np.inf + 0j])),
        ("c64_mixed", np.array([1 + 2j, 3 + 0j], dtype=np.complex64)),
        ("f64_inf", np.array([1.0, np.inf, -np.inf, np.nan, -0.0, 0.0])),
        ("f32_inf", np.array([np.inf, -np.inf, 1.0], dtype=np.float32)),
        ("f16_inf", np.array([np.inf, -np.inf, 1.0], dtype=np.float16)),
        ("i64", np.array([-1, 0, 1])),
        ("u8", np.array([0, 1, 255], dtype=np.uint8)),
        ("bool", np.array([True, False])),
        ("empty_f64", np.array([], dtype=np.float64)),
        ("empty_c128", np.array([], dtype=np.complex128)),
        ("0d_f64", np.array(1.0)),
        ("0d_inf", np.array(np.inf)),
        ("0d_neginf", np.array(-np.inf)),
        ("0d_c128", np.array(1 + 2j)),
        ("0d_c128_realonly", np.array(3 + 0j)),
        ("0d_bool", np.array(True)),
        ("2d_f64", np.array([[1.0, np.inf], [-np.inf, np.nan]])),
        ("2d_c128", np.array([[1 + 0j, 2 + 1j]])),
        ("py_list_float", [1.0, np.inf, -np.inf]),
        ("py_list_complex", [1 + 1j, 2 + 0j]),
        ("py_float", 3.0),
        ("py_inf", float("inf")),
        ("py_neginf", float("-inf")),
        ("py_int", 5),
        ("py_bool", True),
        ("py_complex", 1 + 2j),
        ("py_complex_realonly", 3 + 0j),
    ]
    return [(label, (val,), {}) for label, val in cases]


def _register_dual_specs():
    import anionpy as _anionpy

    for _name in _DUAL_FNS:
        PREDICATE_SPECS[_name] = ItemSpec(
            name=_name, kind="custom", custom_cases=_dual_cases,
            numpy_path=_name, ionp_path=_name,
            numpy_adapter=_dual_adapter(np, _name),
            ionp_adapter=_dual_adapter(_anionpy, _name),
            scalar_like=True,
        )


_register_dual_specs()
