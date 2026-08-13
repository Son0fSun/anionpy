"""Differential registry entries for `anionpy.matrix` (numpy's deprecated
`matrix` ndarray-subclass).

SCOPE. `anionpy.matrix` is a composition wrapper, not a real
`anionpy.ndarray` subclass (`anionpy.ndarray` cannot be subclassed from
Python at all -- see `anionpy/matrix.py`'s module docstring). Every item
here is `kind="custom"` with explicit `numpy_adapter`/`ionp_adapter`
functions and `scalar_like=True`, exactly like `ma_cases.py` uses for the
same reason (`MaskedArray` is also not a real `ndarray`): the default
array-comparison path in `harness.compare_values` expects a `.dtype`-bearing
value whose TYPE is not itself part of the question, but here the type
(matrix vs plain ndarray vs bare scalar) genuinely IS part of the question
-- `.T`/`.A`/`.H`/... must each preserve or shed "matrix-ness" in exactly
the pattern real numpy does. `_snapshot()` below encodes that distinction
directly into the compared value (a literal `"matrix"`/`"ndarray"`/`"scalar"`
tag, plus shape/dtype/data), so a wrong-type return is a real, visible
mismatch under `_compare_scalar_like`'s `==`, not silently coerced away.

Every construction goes through `warnings.catch_warnings()` /
`simplefilter("ignore")` to suppress the (correctly-emitted, but not what
these particular cases are checking) `PendingDeprecationWarning` --
`matrix.__new__`'s own warning-fidelity contract is verified separately,
out of corpus, and is NOT declared here (see `KNOWN-DIFFERENCES.md`'s
2026-08-07 entry: `copy=False` construction cannot alias its source buffer
the way real numpy does, since `anionpy.array()`/`anionpy.ndarray` have no
buffer-sharing construction path at all -- verified live). Every case here
therefore constructs with the SAFE default `copy=True`, which sidesteps
that gap entirely: none of the items declared in this file depend on how
the matrix was aliased, only on its current data.
"""
from __future__ import annotations

import warnings

import numpy as np

import _bootstrap  # noqa: F401
from registry import ItemSpec

import anionpy as ap
from anionpy.matrix import matrix as ap_matrix


# ---------------------------------------------------------------------------
# snapshot: canonical (kind, shape, dtype, data) tuple, side-agnostic.
# ---------------------------------------------------------------------------
def _round_nested(v, ndigits):
    if isinstance(v, list):
        return [_round_nested(e, ndigits) for e in v]
    if isinstance(v, complex):
        return complex(round(v.real, ndigits), round(v.imag, ndigits))
    if isinstance(v, float):
        return round(v, ndigits)
    return v


def _nan_canon(v):
    """Replace IEEE NaN leaves with a sentinel string before the snapshot
    tuple is compared with a raw `==` (`harness._compare_scalar_like`'s
    fallback path for non-scalar-typed results, which every `_snapshot`
    return value routes through since it is itself a plain tuple): `nan !=
    nan` would otherwise flag a genuinely-matching NaN-bearing result
    (e.g. `matrix.mean()` of an empty-along-the-reduced-axis matrix) as a
    false mismatch. Mirrors the same IEEE-754 escape `_compare_scalar_like`
    already applies for its own bare-float/complex path (see harness.py) --
    this is that same fix, applied one level up, at the point the value
    gets folded into a tuple that the raw path can no longer see into. A
    genuine NaN-vs-non-NaN divergence (or NaN in the wrong position/shape)
    still fails normally, since only actual NaN leaves are touched.
    """
    if isinstance(v, list):
        return [_nan_canon(e) for e in v]
    if isinstance(v, complex):
        real = "NaN" if v.real != v.real else v.real
        imag = "NaN" if v.imag != v.imag else v.imag
        return (real, imag)
    if isinstance(v, float) and v != v:
        return "NaN"
    return v


def _snapshot(x, round_digits=None):
    """`round_digits`, when given, rounds float/complex leaves before
    comparison -- used ONLY for `.I`/`getI()` (see their dedicated
    adapters below), which route through `anionpy.linalg.inv`/`pinv`.
    Those are themselves declared "exact" only under a tightly-justified
    non-zero epsilon (see `anionpy/_state/linalg.py`'s own comment: LAPACK
    non-associativity noise between two independent call paths, atol=
    rtol=1e-9), not bit-exact -- so anything built on top of them
    inherits that same LAPACK-noise floor, not a new gap. 9 decimal
    digits is comfortably looser than the observed noise (~1e-15/1e-16,
    i.e. ULP-level) and comfortably tighter than the declared 1e-9
    epsilon itself.
    """
    tname = type(x).__name__
    if tname == "matrix":
        data = x.tolist()
        if round_digits is not None:
            data = _round_nested(data, round_digits)
        return ("matrix", tuple(x.shape), str(x.dtype), _nan_canon(data))
    if hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "tolist"):
        data = x.tolist()
        if round_digits is not None:
            data = _round_nested(data, round_digits)
        return ("ndarray", tuple(x.shape), str(x.dtype), _nan_canon(data))
    return ("scalar", tname, _nan_canon(x))


# ---------------------------------------------------------------------------
# shared construction corpus: (label, data, dtype_kwarg_or_None)
# Provenance varied (nested list / nested tuple / string / python scalar /
# already-2-D / 1-D-promoted / non-square / complex / bool), not just kwargs.
# ---------------------------------------------------------------------------
_CONSTRUCT_CASES = [
    ("2x2_int_list", [[1, 2], [3, 4]], None),
    ("2x3_float_list", [[1.5, -2.5, 0.0], [3.25, 4.0, -1.0]], None),
    ("3x2_float_nested_tuple", ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)), None),
    ("1x1_from_0d_scalar", 7, None),
    ("1x1_from_0d_float", 3.5, None),
    ("row_from_1d_list", [1, 2, 3, 4], None),
    ("string_form", "1 2; 3 4", None),
    ("string_form_neg", "1, -2; -3, 4", None),
    ("complex_2x2", [[1 + 2j, 3 - 4j], [-5j, 6 + 0j]], None),
    ("bool_2x2", [[True, False], [False, True]], None),
    ("dtype_override_float32", [[1, 2], [3, 4]], "float32"),
    ("dtype_override_complex128", [[1, 2], [3, 4]], "complex128"),
    ("negative_and_zero", [[0, -1], [-2, 0]], None),
    ("single_row_2x1_via_nested", [[1], [2], [3]], None),
]


def _make_pair(data, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np_m = np.matrix(data, dtype=dtype)
        ap_m = ap_matrix(data, dtype=dtype)
    return np_m, ap_m


def _read_only_cases_for(attr_kind: str):
    """attr_kind: 'property' name (A, A1, T, H, I) or a `getX()` method name."""
    def build():
        cases = []
        for label, data, dtype in _CONSTRUCT_CASES:
            if attr_kind == "I":
                # `.I`/`getI()` needs a genuinely invertible (or at least
                # non-singular-shaped) receiver; skip the two known-singular
                # constructions in this shared corpus (all-same-row 1x1 is
                # trivially invertible, fine -- nothing here is singular by
                # construction) and skip bool/complex-degenerate combos that
                # would need a dedicated non-singular-only sub-corpus.
                if label == "bool_2x2":
                    continue
            cases.append((f"{attr_kind}/{label}", (data, dtype), {}))
        return cases
    return build


def _adapters_for_property(prop: str):
    # `.I` rounds: see `_snapshot`'s docstring -- `linalg.inv`/`pinv` are
    # themselves declared exact only under a non-zero epsilon (LAPACK
    # non-associativity), so `.I` built on top of them inherits that same
    # noise floor rather than a new gap.
    rd = 9 if prop == "I" else None

    def numpy_adapter(data, dtype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.matrix(data, dtype=dtype)
            return _snapshot(getattr(m, prop), round_digits=rd)

    def ionp_adapter(data, dtype):
        m = ap_matrix(data, dtype=dtype)
        return _snapshot(getattr(m, prop), round_digits=rd)

    return numpy_adapter, ionp_adapter


def _adapters_for_method(meth: str):
    rd = 9 if meth == "getI" else None

    def numpy_adapter(data, dtype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.matrix(data, dtype=dtype)
            return _snapshot(getattr(m, meth)(), round_digits=rd)

    def ionp_adapter(data, dtype):
        m = ap_matrix(data, dtype=dtype)
        return _snapshot(getattr(m, meth)(), round_digits=rd)

    return numpy_adapter, ionp_adapter


MATRIX_SPECS: dict[str, ItemSpec] = {}

for _prop, _kind in (
    ("A", "A"), ("A1", "A1"), ("T", "T"), ("H", "H"), ("I", "I"),
):
    _np_a, _ionp_a = _adapters_for_property(_prop)
    MATRIX_SPECS[f"matrix.{_prop}"] = ItemSpec(
        name=f"matrix.{_prop}",
        kind="custom",
        scalar_like=True,
        numpy_adapter=_np_a,
        ionp_adapter=_ionp_a,
        custom_cases=_read_only_cases_for(_kind),
    )

_METHOD_TO_KIND = {"getA": "A", "getA1": "A1", "getT": "T", "getH": "H", "getI": "I"}
for _meth in ("getA", "getA1", "getT", "getH", "getI"):
    _kind = _METHOD_TO_KIND[_meth]
    _np_a, _ionp_a = _adapters_for_method(_meth)
    MATRIX_SPECS[f"matrix.{_meth}"] = ItemSpec(
        name=f"matrix.{_meth}",
        kind="custom",
        scalar_like=True,
        numpy_adapter=_np_a,
        ionp_adapter=_ionp_a,
        custom_cases=_read_only_cases_for(_kind),
    )


# -- shape / dtype / ndim / size: trivial delegating reads, still verified
#    against the SAME construction contract (0-d -> (1,1), 1-d -> (1,n)) ----
def _scalar_attr_adapters(attr: str):
    # NOTE on the "tname" field below: for `dtype` specifically we use a
    # FIXED literal ("dtype") rather than `type(v).__name__`. numpy 2.x's
    # `.dtype` returns instances of per-scalar-kind internal subclasses
    # (`Int64DType`, `Float64DType`, ...) while anionpy's `.dtype` always
    # returns its one generic `dtype` class -- that class-name difference
    # is a pre-existing, already-accepted divergence everywhere else numpy
    # dtypes are compared in this codebase (str/name is the contract, not
    # the exact internal Python type), never numpy's own runtime contract
    # for matrix construction itself. Comparing `type(v).__name__` here
    # was a bug in THIS test's own snapshot function, not a real
    # `anionpy.matrix` divergence -- fixed by comparing the dtype's
    # string spelling only, same as every other dtype comparison in this
    # suite.
    def numpy_adapter(data, dtype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.matrix(data, dtype=dtype)
            v = getattr(m, attr)
            tname = "dtype" if attr == "dtype" else type(v).__name__
            return ("scalar", tname, tuple(v) if attr == "shape" else v if attr != "dtype" else str(v))

    def ionp_adapter(data, dtype):
        m = ap_matrix(data, dtype=dtype)
        v = getattr(m, attr)
        tname = "dtype" if attr == "dtype" else type(v).__name__
        return ("scalar", tname, tuple(v) if attr == "shape" else v if attr != "dtype" else str(v))

    return numpy_adapter, ionp_adapter


for _attr in ("shape", "dtype", "ndim"):
    _np_a, _ionp_a = _scalar_attr_adapters(_attr)
    MATRIX_SPECS[f"matrix.{_attr}"] = ItemSpec(
        name=f"matrix.{_attr}",
        kind="custom",
        scalar_like=True,
        numpy_adapter=_np_a,
        ionp_adapter=_ionp_a,
        custom_cases=(lambda: [
            (f"{_attr}/{label}", (data, dtype), {})
            for label, data, dtype in _CONSTRUCT_CASES
        ]),
    )


# -- tolist -------------------------------------------------------------
def _tolist_cases():
    return [(f"tolist/{label}", (data, dtype), {}) for label, data, dtype in _CONSTRUCT_CASES]


def _tolist_numpy(data, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data, dtype=dtype)
        return ("scalar", "list", m.tolist())


def _tolist_ionp(data, dtype):
    m = ap_matrix(data, dtype=dtype)
    return ("scalar", "list", m.tolist())


MATRIX_SPECS["matrix.tolist"] = ItemSpec(
    name="matrix.tolist", kind="custom", scalar_like=True,
    numpy_adapter=_tolist_numpy, ionp_adapter=_tolist_ionp,
    custom_cases=_tolist_cases,
)


# -- __getitem__: always 2-D, row/col-vector promotion rules -------------
_GETITEM_CASES = [
    ("row0", [[1, 2, 3], [4, 5, 6]], 0),
    ("row1", [[1, 2, 3], [4, 5, 6]], 1),
    ("scalar_00", [[1, 2], [3, 4]], (0, 0)),
    ("scalar_11", [[1, 2], [3, 4]], (1, 1)),
    ("col_via_tuple", [[1, 2, 3], [4, 5, 6]], (slice(None), 1)),
    ("row_via_tuple", [[1, 2, 3], [4, 5, 6]], (0, slice(None))),
    ("full_slice", [[1, 2], [3, 4]], slice(None)),
    ("negative_row", [[1, 2], [3, 4], [5, 6]], -1),
    ("float_scalar", [[1.5, 2.5], [3.5, 4.5]], (0, 1)),
]


def _getitem_cases():
    return [(f"getitem/{label}", (data, idx), {}) for label, data, idx in _GETITEM_CASES]


def _getitem_numpy(data, idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        return _snapshot(m[idx])


def _getitem_ionp(data, idx):
    m = ap_matrix(data)
    return _snapshot(m[idx])


MATRIX_SPECS["matrix.__getitem__"] = ItemSpec(
    name="matrix.__getitem__", kind="custom", scalar_like=True,
    numpy_adapter=_getitem_numpy, ionp_adapter=_getitem_ionp,
    custom_cases=_getitem_cases,
)


# -- __mul__: matrix multiply (matrix*matrix / matrix*array-like / *scalar) -
_MUL_CASES = [
    ("mat_mat_2x2", [[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    ("mat_mat_nonsquare", [[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1], [1, 1]]),
    ("mat_list_col", [[1, 2], [3, 4]], [[1], [1]]),
    ("mat_tuple_col", [[1, 2], [3, 4]], ((2,), (3,))),
    ("mat_scalar_int", [[1, 2], [3, 4]], 3),
    ("mat_scalar_float", [[1.5, 2.5], [3.5, 4.5]], -2.0),
    ("mat_complex", [[1 + 1j, 2], [3, 4 - 1j]], [[1, 0], [0, 1]]),
    # (1,3)*(2,1): misaligned FORWARD (so `__mul__` exercises the error
    # path its corpus otherwise never reached), and validly (2,1)*(1,3) ->
    # (2,3) REVERSED, so `__rmul__` gets a success out of the same entry.
    ("mat_misaligned_fwd", [[1, 2, 3]], [[1], [2]]),
]


def _mul_cases():
    return [(f"mul/{label}", (a, b), {}) for label, a, b in _MUL_CASES]


def _mul_numpy(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ma = np.matrix(a)
        bb = np.matrix(b) if isinstance(b, (list, tuple)) and not isinstance(b, (int, float, complex)) else b
        return _snapshot(ma * bb)


def _mul_ionp(a, b):
    ma = ap_matrix(a)
    bb = ap_matrix(b) if isinstance(b, (list, tuple)) else b
    return _snapshot(ma * bb)


MATRIX_SPECS["matrix.__mul__"] = ItemSpec(
    name="matrix.__mul__", kind="custom", scalar_like=True,
    numpy_adapter=_mul_numpy, ionp_adapter=_mul_ionp,
    custom_cases=_mul_cases,
)


# ---------------------------------------------------------------------------
# 2026-08-07 batch: extended inherited surface. Same "unwrap -> call the
# already-declared primitive -> re-wrap" contract as everything above;
# every item verified live against real numpy 2.5.1 before being declared
# (see `anionpy/matrix.py`'s new-code docstring for the specific findings:
# `trace()` returns a (1,1) matrix -- NOT a bare scalar like plain
# `ndarray.trace()` -- reductions with `axis=None` return a bare numpy
# scalar while `axis=0`/`axis=1` return a matrix with the reduced axis
# kept via `keepdims=True`).
# ---------------------------------------------------------------------------

# -- itemsize / nbytes / size: plain-value attrs, same contract as the
#    shape/dtype/ndim loop above. `strides` is DELIBERATELY EXCLUDED here
#    (declined, not silently dropped): real numpy's `matrix.__new__`
#    selects `order='F'` internally whenever the source array's own
#    contiguity is ambiguous (`arr.flags.fortran` is True for any 2-D
#    array with a size-1 dimension, e.g. shape (3,1) or (1,4) -- both
#    C- and F-contiguous simultaneously), producing genuinely different
#    strides than naive C-order (measured live: `np.matrix([[1],[2],[3]])
#    .strides == (8, 24)`, not the `(8, 8)` a plain C-contiguous (3,1)
#    array -- and this composition wrapper -- would report). This is a
#    real difference in memory-layout CONSTRUCTION algorithm, not a
#    superficial gap: replicating it would require reimplementing numpy's
#    exact ambiguous-contiguity order-selection rule in the Rust core
#    purely for this one property, disproportionate for what it buys.
#    Declined for the whole item (not just the one construction case that
#    happens to trigger it), since any single-dim-1 2-D construction is
#    exposed to this, not only `single_row_2x1_via_nested`.
for _attr in ("itemsize", "nbytes", "size"):
    _np_a, _ionp_a = _scalar_attr_adapters(_attr)
    MATRIX_SPECS[f"matrix.{_attr}"] = ItemSpec(
        name=f"matrix.{_attr}",
        kind="custom",
        scalar_like=True,
        numpy_adapter=_np_a,
        ionp_adapter=_ionp_a,
        custom_cases=(lambda _attr=_attr: [
            (f"{_attr}/{label}", (data, dtype), {})
            for label, data, dtype in _CONSTRUCT_CASES
        ]),
    )

# -- mT: matrix-returning property, reuse the A/A1/T/H/I property machinery --
_np_a, _ionp_a = _adapters_for_property("mT")
MATRIX_SPECS["matrix.mT"] = ItemSpec(
    name="matrix.mT", kind="custom", scalar_like=True,
    numpy_adapter=_np_a, ionp_adapter=_ionp_a,
    custom_cases=_read_only_cases_for("mT"),
)

# -- zero-arg matrix-returning methods: reuse the getA/getA1/.../getI
#    method machinery (identical shape: `getattr(m, meth)()`, snapshot) --
for _meth in ("flatten", "ravel", "transpose", "copy", "conj", "conjugate", "trace"):
    _np_a, _ionp_a = _adapters_for_method(_meth)
    MATRIX_SPECS[f"matrix.{_meth}"] = ItemSpec(
        name=f"matrix.{_meth}", kind="custom", scalar_like=True,
        numpy_adapter=_np_a, ionp_adapter=_ionp_a,
        custom_cases=_read_only_cases_for(_meth),
    )


# -- __copy__ / __deepcopy__: exercised via `copy.copy`/`copy.deepcopy`,
#    not direct dunder calls (matches how real code invokes them) --
import copy as _copy_mod  # noqa: E402


def _copy_dunder_cases():
    return [(f"copydunder/{label}", (data, dtype), {}) for label, data, dtype in _CONSTRUCT_CASES]


def _copy_dunder_numpy(data, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data, dtype=dtype)
        return _snapshot(_copy_mod.copy(m))


def _copy_dunder_ionp(data, dtype):
    m = ap_matrix(data, dtype=dtype)
    return _snapshot(_copy_mod.copy(m))


MATRIX_SPECS["matrix.__copy__"] = ItemSpec(
    name="matrix.__copy__", kind="custom", scalar_like=True,
    numpy_adapter=_copy_dunder_numpy, ionp_adapter=_copy_dunder_ionp,
    custom_cases=_copy_dunder_cases,
)


def _deepcopy_dunder_numpy(data, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data, dtype=dtype)
        return _snapshot(_copy_mod.deepcopy(m))


def _deepcopy_dunder_ionp(data, dtype):
    m = ap_matrix(data, dtype=dtype)
    return _snapshot(_copy_mod.deepcopy(m))


MATRIX_SPECS["matrix.__deepcopy__"] = ItemSpec(
    name="matrix.__deepcopy__", kind="custom", scalar_like=True,
    numpy_adapter=_deepcopy_dunder_numpy, ionp_adapter=_deepcopy_dunder_ionp,
    custom_cases=_copy_dunder_cases,
)


# -- nonzero: returns a plain tuple of 1-D `ndarray` (NOT matrix-wrapped)
#    on real numpy matrix -- verified live --
def _nonzero_cases():
    data = [
        ("2x2_mixed", [[1, 0], [0, 4]]),
        ("all_zero", [[0, 0], [0, 0]]),
        ("all_nonzero", [[1, 2], [3, 4]]),
        ("nonsquare", [[0, 1, 0], [2, 0, 3]]),
        ("1x1_nonzero", [[5]]),
        ("1x1_zero", [[0]]),
    ]
    return [(f"nonzero/{label}", (d,), {}) for label, d in data]


def _nonzero_numpy(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        r = m.nonzero()
        return ("scalar", "nonzero", tuple(tuple(x.tolist()) for x in r))


def _nonzero_ionp(data):
    m = ap_matrix(data)
    r = m.nonzero()
    return ("scalar", "nonzero", tuple(tuple(x.tolist()) for x in r))


MATRIX_SPECS["matrix.nonzero"] = ItemSpec(
    name="matrix.nonzero", kind="custom", scalar_like=True,
    numpy_adapter=_nonzero_numpy, ionp_adapter=_nonzero_ionp,
    custom_cases=_nonzero_cases,
)


# -- tobytes: raw bytes, same construction corpus --
def _tobytes_cases():
    return [(f"tobytes/{label}", (data, dtype), {}) for label, data, dtype in _CONSTRUCT_CASES]


def _tobytes_numpy(data, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data, dtype=dtype)
        return ("scalar", "bytes", m.tobytes())


def _tobytes_ionp(data, dtype):
    m = ap_matrix(data, dtype=dtype)
    return ("scalar", "bytes", m.tobytes())


MATRIX_SPECS["matrix.tobytes"] = ItemSpec(
    name="matrix.tobytes", kind="custom", scalar_like=True,
    numpy_adapter=_tobytes_numpy, ionp_adapter=_tobytes_ionp,
    custom_cases=_tobytes_cases,
)


# -- item: only defined (without error) on size-1 matrices --
_ITEM_CASES = [
    ("1x1_int", [[7]], None),
    ("1x1_float", [[3.5]], None),
    ("1x1_from_scalar", 9, None),
    ("1x1_complex", [[1 + 2j]], None),
]


def _item_cases():
    return [(f"item/{label}", (data, dtype), {}) for label, data, dtype in _ITEM_CASES]


def _item_numpy(data, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data, dtype=dtype)
        return _snapshot(m.item())


def _item_ionp(data, dtype):
    m = ap_matrix(data, dtype=dtype)
    return _snapshot(m.item())


MATRIX_SPECS["matrix.item"] = ItemSpec(
    name="matrix.item", kind="custom", scalar_like=True,
    numpy_adapter=_item_numpy, ionp_adapter=_item_ionp,
    custom_cases=_item_cases,
)


# -- comparisons + arithmetic: binary ops, matrix vs {matrix, list, scalar}
_BINOP_CASES = [
    ("mat_mat_2x2", [[1, 2], [3, 4]], [[4, 3], [2, 1]]),
    ("mat_mat_nonsquare", [[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
    ("mat_scalar_int", [[1, 2], [3, 4]], 2),
    ("mat_scalar_float", [[1.5, -2.5], [3.5, 4.5]], -1.5),
    ("mat_list", [[1, 2], [3, 4]], [[1, 1], [1, 1]]),
    ("mat_1x1", [[5]], [[3]]),
    ("mat_row", [[1, 2, 3]], [[3, 2, 1]]),
    ("mat_col", [[1], [2], [3]], [[3], [2], [1]]),
    ("mat_complex", [[1 + 1j, 2], [3, 4 - 1j]], [[1, 0], [0, 1]]),
]


def _binop_is_scalar(b):
    return isinstance(b, (int, float, complex)) and not isinstance(b, bool)


def _make_binop_spec(name, np_op, ionp_op):
    def cases():
        return [(f"{name}/{label}", (a, b), {}) for label, a, b in _BINOP_CASES]

    def numpy_adapter(a, b):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ma = np.matrix(a)
            bb = b if _binop_is_scalar(b) else np.matrix(b)
            return _snapshot(np_op(ma, bb))

    def ionp_adapter(a, b):
        ma = ap_matrix(a)
        bb = b if _binop_is_scalar(b) else ap_matrix(b)
        return _snapshot(ionp_op(ma, bb))

    MATRIX_SPECS[f"matrix.{name}"] = ItemSpec(
        name=f"matrix.{name}", kind="custom", scalar_like=True,
        numpy_adapter=numpy_adapter, ionp_adapter=ionp_adapter,
        custom_cases=cases,
    )


_make_binop_spec("__eq__", lambda a, b: a == b, lambda a, b: a == b)
_make_binop_spec("__ne__", lambda a, b: a != b, lambda a, b: a != b)
_make_binop_spec("__lt__", lambda a, b: a < b, lambda a, b: a < b)
_make_binop_spec("__le__", lambda a, b: a <= b, lambda a, b: a <= b)
_make_binop_spec("__gt__", lambda a, b: a > b, lambda a, b: a > b)
_make_binop_spec("__ge__", lambda a, b: a >= b, lambda a, b: a >= b)
_make_binop_spec("__add__", lambda a, b: a + b, lambda a, b: a + b)
_make_binop_spec("__sub__", lambda a, b: a - b, lambda a, b: a - b)
_make_binop_spec("__radd__", lambda a, b: b + a, lambda a, b: b + a)
_make_binop_spec("__rsub__", lambda a, b: b - a, lambda a, b: b - a)


# -- __rmul__: reuse the `__mul__` corpus, reversed operand order.
#    `mat_list_col`/`mat_tuple_col` are (2,2)*(2,1) valid FORWARD, but
#    reversed become (2,1)*(2,2), whose inner dimensions do not match: a
#    genuine ValueError on both sides. They are INCLUDED, deliberately --
#    they are the only error path this file exercises.
#
#    An earlier revision excluded them, on the stated ground that anionpy's
#    matmul error text already diverged from numpy's on a bare `ndarray @
#    ndarray` with no `matrix` involved. That diagnosis was re-measured
#    2026-08-07 and is FALSE: bare `ndarray @ ndarray` produces BYTE-
#    IDENTICAL text in both libraries. The real cause is that numpy's
#    `matrix.__mul__` routes through `N.dot`, whose "shapes (2,1) and (2,2)
#    not aligned: 1 (dim 1) != 2 (dim 0)" wording differs from `matmul`'s
#    "Input operand 1 has a mismatch in its core dimension 0, ..." -- a
#    wrapper-level divergence, fixable at the wrapper, and now fixed there
#    (`matrix._check_aligned`, anionpy/matrix.py). The exclusion is
#    therefore withdrawn rather than kept: it was hiding a real, in-scope
#    defect behind a narrowed corpus.
def _rmul_cases():
    return [(f"mul/{label}", (a, b), {}) for label, a, b in _MUL_CASES]


def _rmul_numpy(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ma = np.matrix(a)
        bb = np.matrix(b) if isinstance(b, (list, tuple)) and not isinstance(b, (int, float, complex)) else b
        return _snapshot(bb * ma)


def _rmul_ionp(a, b):
    ma = ap_matrix(a)
    bb = ap_matrix(b) if isinstance(b, (list, tuple)) else b
    return _snapshot(bb * ma)


MATRIX_SPECS["matrix.__rmul__"] = ItemSpec(
    name="matrix.__rmul__", kind="custom", scalar_like=True,
    numpy_adapter=_rmul_numpy, ionp_adapter=_rmul_ionp,
    custom_cases=_rmul_cases,
)


# -- unary ops --
_UNARY_NUMERIC_CASES = [
    ("2x2_mixed_sign", [[1, -2], [-3, 4]], None),
    ("float_mixed", [[1.5, -2.5], [-3.5, 4.5]], None),
    ("1x1", [[-7]], None),
    ("complex", [[1 + 2j, -1 - 2j]], None),
    ("nonsquare", [[1, -2, 3], [-4, 5, -6]], None),
]

_UNARY_INT_CASES = [
    ("int_2x2", [[1, 2], [3, 4]], "int64"),
    ("int_neg", [[-1, 2], [3, -4]], "int64"),
    ("bool_2x2", [[True, False], [False, True]], "bool"),
]


def _make_unary_spec(name, np_op, ionp_op, cases_list):
    def cases():
        return [(f"{name}/{label}", (data, dtype), {}) for label, data, dtype in cases_list]

    def numpy_adapter(data, dtype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.matrix(data, dtype=dtype)
            return _snapshot(np_op(m))

    def ionp_adapter(data, dtype):
        m = ap_matrix(data, dtype=dtype)
        return _snapshot(ionp_op(m))

    MATRIX_SPECS[f"matrix.{name}"] = ItemSpec(
        name=f"matrix.{name}", kind="custom", scalar_like=True,
        numpy_adapter=numpy_adapter, ionp_adapter=ionp_adapter,
        custom_cases=cases,
    )


_make_unary_spec("__neg__", lambda m: -m, lambda m: -m, _UNARY_NUMERIC_CASES)
_make_unary_spec("__pos__", lambda m: +m, lambda m: +m, _UNARY_NUMERIC_CASES)
_make_unary_spec("__abs__", lambda m: abs(m), lambda m: abs(m), _UNARY_NUMERIC_CASES)
_make_unary_spec("__invert__", lambda m: ~m, lambda m: ~m, _UNARY_INT_CASES)


# -- reductions: axis=None -> bare scalar, axis=0/1 -> matrix (keepdims) --
_REDUCTION_CASES = [
    ("2x2_int", [[1, 2], [3, 4]], "int64"),
    ("2x3_float", [[1.5, -2.5, 0.0], [3.25, 4.0, -1.0]], None),
    ("1x1", [[5]], None),
    ("row_1x4", [[1, 2, 3, 4]], None),
    ("col_4x1", [[1], [2], [3], [4]], None),
    ("bool_2x2", [[True, False], [True, True]], None),
    ("empty_0x3", [[], [], []], None),  # will be reshaped below for axis case
]


def _reduction_matrix_case_data():
    # Empty case needs an explicit 2-D empty shape; numpy's `matrix`
    # constructor collapses `[[],[],[]]` oddly, so build empty cases via
    # `.reshape` on both sides inside the adapter instead of raw data.
    return _REDUCTION_CASES


def _make_reduction_spec(name, np_op, ionp_op):
    def cases():
        out = []
        for label, data, dtype in _REDUCTION_CASES:
            for axis in (None, 0, 1):
                out.append((f"{name}/{label}/axis={axis}", (data, dtype, axis), {}))
        return out

    def numpy_adapter(data, dtype, axis):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if data == [[], [], []]:
                m = np.matrix(np.empty((3, 0)), dtype=dtype)
            else:
                m = np.matrix(data, dtype=dtype)
            return _snapshot(np_op(m, axis))

    def ionp_adapter(data, dtype, axis):
        if data == [[], [], []]:
            m = ap_matrix(ap.array([]).reshape((3, 0)), dtype=dtype)
        else:
            m = ap_matrix(data, dtype=dtype)
        return _snapshot(ionp_op(m, axis))

    MATRIX_SPECS[f"matrix.{name}"] = ItemSpec(
        name=f"matrix.{name}", kind="custom", scalar_like=True,
        numpy_adapter=numpy_adapter, ionp_adapter=ionp_adapter,
        custom_cases=cases,
    )


_make_reduction_spec("sum", lambda m, ax: m.sum(axis=ax), lambda m, ax: m.sum(axis=ax))
_make_reduction_spec("mean", lambda m, ax: m.mean(axis=ax), lambda m, ax: m.mean(axis=ax))
_make_reduction_spec("min", lambda m, ax: m.min(axis=ax), lambda m, ax: m.min(axis=ax))
_make_reduction_spec("max", lambda m, ax: m.max(axis=ax), lambda m, ax: m.max(axis=ax))
_make_reduction_spec("all", lambda m, ax: m.all(axis=ax), lambda m, ax: m.all(axis=ax))
_make_reduction_spec("any", lambda m, ax: m.any(axis=ax), lambda m, ax: m.any(axis=ax))


# ---------------------------------------------------------------------------
# 2026-08-07 second batch: elementwise binary dunders not overridden by real
# numpy `matrix` (verified against `numpy/matrixlib/defmatrix.py`'s full
# `def __` listing -- only `__mul__`/`__pow__`/their in-place/reflected
# forms get matrix semantics), the scalar-conversion/protocol dunders, and
# the remaining matrix-specific reduction overrides (std/var/prod/argmax/
# argmin/ptp/squeeze) plus a handful of already-2-D-preserving shape/data
# ops (reshape/swapaxes/repeat/argsort/cumsum/cumprod/clip/fill/sort).
# ---------------------------------------------------------------------------

_INT_BINOP_CASES = [
    ("mat_mat_2x2", [[1, 2], [3, 4]], [[5, 6], [7, 0]]),
    ("mat_mat_nonsquare", [[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
    ("mat_scalar", [[1, 2], [3, 4]], 2),
    ("mat_scalar_neg", [[1, -2], [3, -4]], -3),
    ("mat_list", [[1, 2], [3, 4]], [[1, 1], [1, 1]]),
    ("mat_1x1", [[5]], [[3]]),
    ("mat_bool", [[True, False], [False, True]], [[True, True], [False, False]]),
]

for _name, _np_op, _ionp_op in (
    ("__floordiv__", lambda a, b: a // b, lambda a, b: a // b),
    ("__rfloordiv__", lambda a, b: b // a, lambda a, b: b // a),
    ("__mod__", lambda a, b: a % b, lambda a, b: a % b),
    ("__rmod__", lambda a, b: b % a, lambda a, b: b % a),
    ("__and__", lambda a, b: a & b, lambda a, b: a & b),
    ("__rand__", lambda a, b: b & a, lambda a, b: b & a),
    ("__or__", lambda a, b: a | b, lambda a, b: a | b),
    ("__ror__", lambda a, b: b | a, lambda a, b: b | a),
    ("__xor__", lambda a, b: a ^ b, lambda a, b: a ^ b),
    ("__rxor__", lambda a, b: b ^ a, lambda a, b: b ^ a),
    ("__lshift__", lambda a, b: a << b, lambda a, b: a << b),
    ("__rlshift__", lambda a, b: b << a, lambda a, b: b << a),
    ("__rshift__", lambda a, b: a >> b, lambda a, b: a >> b),
    ("__rrshift__", lambda a, b: b >> a, lambda a, b: b >> a),
):
    def _cases(_cases_list=_INT_BINOP_CASES):
        return [(f"{_name}/{label}", (a, b), {}) for label, a, b in _cases_list]

    def _numpy_adapter(a, b, _f=_np_op):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ma = np.matrix(a)
            bb = b if _binop_is_scalar(b) else np.matrix(b)
            return _snapshot(_f(ma, bb))

    def _ionp_adapter(a, b, _f=_ionp_op):
        ma = ap_matrix(a)
        bb = b if _binop_is_scalar(b) else ap_matrix(b)
        return _snapshot(_f(ma, bb))

    MATRIX_SPECS[f"matrix.{_name}"] = ItemSpec(
        name=f"matrix.{_name}", kind="custom", scalar_like=True,
        numpy_adapter=_numpy_adapter, ionp_adapter=_ionp_adapter,
        custom_cases=_cases,
    )


# -- scalar-conversion dunders: ALWAYS raise (a `matrix` can never be 0-d) --
_SCALAR_DUNDER_CASES = [
    ("1x1_int", [[5]]),
    ("1x1_float", [[3.5]]),
    ("2x2", [[1, 2], [3, 4]]),
    ("row", [[1, 2, 3]]),
]


def _make_scalar_dunder_spec(name):
    def cases():
        return [(f"{name}/{label}", (data,), {}) for label, data in _SCALAR_DUNDER_CASES]

    def numpy_adapter(data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.matrix(data)
            try:
                r = getattr(m, name)()
                return ("scalar", type(r).__name__, r)
            except Exception as e:
                return ("error", type(e).__name__, str(e))

    def ionp_adapter(data):
        m = ap_matrix(data)
        try:
            r = getattr(m, name)()
            return ("scalar", type(r).__name__, r)
        except Exception as e:
            return ("error", type(e).__name__, str(e))

    MATRIX_SPECS[f"matrix.{name}"] = ItemSpec(
        name=f"matrix.{name}", kind="custom", scalar_like=True,
        numpy_adapter=numpy_adapter, ionp_adapter=ionp_adapter,
        custom_cases=cases,
    )


for _name in ("__int__", "__float__", "__complex__", "__index__"):
    _make_scalar_dunder_spec(_name)


# -- __bool__: size-1 -> truthiness of the element, else ValueError --
_BOOL_CASES = [
    ("truthy_1x1", [[5]]),
    ("falsy_1x1", [[0]]),
    ("truthy_1x1_neg", [[-1]]),
    ("multi", [[1, 2], [3, 4]]),
    ("multi_zero", [[0, 0]]),
]


def _bool_cases():
    return [(f"__bool__/{label}", (data,), {}) for label, data in _BOOL_CASES]


def _bool_numpy(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        try:
            return ("scalar", "bool", bool(m))
        except Exception as e:
            return ("error", type(e).__name__, str(e))


def _bool_ionp(data):
    m = ap_matrix(data)
    try:
        return ("scalar", "bool", bool(m))
    except Exception as e:
        return ("error", type(e).__name__, str(e))


MATRIX_SPECS["matrix.__bool__"] = ItemSpec(
    name="matrix.__bool__", kind="custom", scalar_like=True,
    numpy_adapter=_bool_numpy, ionp_adapter=_bool_ionp,
    custom_cases=_bool_cases,
)


# -- __len__ --
def _len_cases():
    return [(f"__len__/{label}", (data,), {}) for label, data, _ in _CONSTRUCT_CASES]


def _len_numpy(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        return ("scalar", "int", len(m))


def _len_ionp(data):
    m = ap_matrix(data)
    return ("scalar", "int", len(m))


MATRIX_SPECS["matrix.__len__"] = ItemSpec(
    name="matrix.__len__", kind="custom", scalar_like=True,
    numpy_adapter=_len_numpy, ionp_adapter=_len_ionp,
    custom_cases=_len_cases,
)


# -- __hash__: always unhashable (matrix defines __eq__) --
def _hash_cases():
    return [("hash/basic", ([[1, 2], [3, 4]],), {})]


def _hash_numpy(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        try:
            return ("scalar", "int", hash(m))
        except Exception as e:
            return ("error", type(e).__name__, str(e))


def _hash_ionp(data):
    m = ap_matrix(data)
    try:
        return ("scalar", "int", hash(m))
    except Exception as e:
        return ("error", type(e).__name__, str(e))


MATRIX_SPECS["matrix.__hash__"] = ItemSpec(
    name="matrix.__hash__", kind="custom", scalar_like=True,
    numpy_adapter=_hash_numpy, ionp_adapter=_hash_ionp,
    custom_cases=_hash_cases,
)


# -- __iter__: snapshot the sequence of yielded rows --
def _iter_cases():
    data_list = [
        [[1, 2, 3], [4, 5, 6]],
        [[1.5, -2.5]],
        [[7]],
        [[1, 2], [3, 4], [5, 6]],
    ]
    return [(f"__iter__/{i}", (d,), {}) for i, d in enumerate(data_list)]


def _iter_numpy(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        return ("scalar", "rows", tuple(_snapshot(row) for row in m))


def _iter_ionp(data):
    m = ap_matrix(data)
    return ("scalar", "rows", tuple(_snapshot(row) for row in m))


MATRIX_SPECS["matrix.__iter__"] = ItemSpec(
    name="matrix.__iter__", kind="custom", scalar_like=True,
    numpy_adapter=_iter_numpy, ionp_adapter=_iter_ionp,
    custom_cases=_iter_cases,
)


# -- __contains__ --
_CONTAINS_CASES = [
    ("found", [[1, 2], [3, 4]], 3),
    ("not_found", [[1, 2], [3, 4]], 99),
    ("found_float", [[1.5, 2.5]], 2.5),
    ("found_bool", [[True, False]], False),
]


def _contains_cases():
    return [(f"__contains__/{label}", (data, v), {}) for label, data, v in _CONTAINS_CASES]


def _contains_numpy(data, v):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        return ("scalar", "bool", v in m)


def _contains_ionp(data, v):
    m = ap_matrix(data)
    return ("scalar", "bool", v in m)


MATRIX_SPECS["matrix.__contains__"] = ItemSpec(
    name="matrix.__contains__", kind="custom", scalar_like=True,
    numpy_adapter=_contains_numpy, ionp_adapter=_contains_ionp,
    custom_cases=_contains_cases,
)


# -- __setitem__: registered (and 4/5 cases pass) but NOT declared -- see
#    `anionpy/_state/matrix.py`'s comment on the measured `col_slice`
#    divergence: real numpy's `ndarray.__setitem__`, for a subclass that
#    overrides `__getitem__` (as `matrix` does), validates the assignment's
#    broadcast target against the SUBCLASS-promoted destination shape (a
#    genuine column `(n,1)` for a `(:, k)` index), not the plain flat
#    `(n,)` shape a bare `ndarray` would use for the same index -- so
#    `m[:, 1] = [70, 80]` raises `ValueError` on real numpy (can't
#    broadcast `(2,)` into `(2,1)`) but silently succeeds here, since this
#    composition wrapper's plain `self._data[index] = value` passthrough
#    validates against the flat shape instead.
_SETITEM_CASES = [
    ("scalar_at_00", [[1, 2], [3, 4]], (0, 0), 99),
    ("scalar_at_neg", [[1, 2], [3, 4]], (-1, -1), 42),
    ("row_slice", [[1, 2, 3], [4, 5, 6]], (0, slice(None)), [7, 8, 9]),
    ("col_slice", [[1, 2, 3], [4, 5, 6]], (slice(None), 1), [70, 80]),
    ("int_row", [[1, 2], [3, 4]], 0, [9, 9]),
]


def _setitem_cases():
    return [(f"__setitem__/{label}", (data, idx, val), {}) for label, data, idx, val in _SETITEM_CASES]


def _setitem_numpy(data, idx, val):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        m[idx] = val
        return _snapshot(m)


def _setitem_ionp(data, idx, val):
    m = ap_matrix(data)
    m[idx] = val
    return _snapshot(m)


MATRIX_SPECS["matrix.__setitem__"] = ItemSpec(
    name="matrix.__setitem__", kind="custom", scalar_like=True,
    numpy_adapter=_setitem_numpy, ionp_adapter=_setitem_ionp,
    custom_cases=_setitem_cases,
)


# -- std/var/prod: same axis=None-scalar/axis=0,1-matrix(keepdims) pattern
#    as sum/mean/min/max/all/any, reuse `_make_reduction_spec` --
_make_reduction_spec("std", lambda m, ax: m.std(axis=ax), lambda m, ax: m.std(axis=ax))
_make_reduction_spec("var", lambda m, ax: m.var(axis=ax), lambda m, ax: m.var(axis=ax))
_make_reduction_spec("prod", lambda m, ax: m.prod(axis=ax), lambda m, ax: m.prod(axis=ax))

# -- argmax/argmin/ptp: axis=None -> scalar, axis=0 -> row, axis=1 -> column
#    (a DIFFERENT orientation rule than sum/mean/etc's keepdims form -- see
#    `anionpy/matrix.py`'s comment on `argmax`/`ptp`) --
_make_reduction_spec("argmax", lambda m, ax: m.argmax(axis=ax), lambda m, ax: m.argmax(axis=ax))
_make_reduction_spec("argmin", lambda m, ax: m.argmin(axis=ax), lambda m, ax: m.argmin(axis=ax))
_make_reduction_spec("ptp", lambda m, ax: m.ptp(axis=ax), lambda m, ax: m.ptp(axis=ax))


# -- squeeze --
_SQUEEZE_CASES = [
    ("col_2x1", [[1], [2]], None),
    ("row_1x2", [[1, 2]], None),
    ("1x1", [[5]], None),
    ("2x2_noop", [[1, 2], [3, 4]], None),
    ("col_3x1", [[1], [2], [3]], None),
]


def _squeeze_cases():
    return [(f"squeeze/{label}", (data,), {}) for label, data, _ in _SQUEEZE_CASES]


def _squeeze_numpy(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        return _snapshot(m.squeeze())


def _squeeze_ionp(data):
    m = ap_matrix(data)
    return _snapshot(m.squeeze())


MATRIX_SPECS["matrix.squeeze"] = ItemSpec(
    name="matrix.squeeze", kind="custom", scalar_like=True,
    numpy_adapter=_squeeze_numpy, ionp_adapter=_squeeze_ionp,
    custom_cases=_squeeze_cases,
)


# -- reshape/swapaxes/repeat/argsort/cumsum/cumprod/clip: already-2-D-in/
#    2-D-out (or `_wrap`-promoted), matrix-preserving per real numpy
#    (not overridden by `defmatrix.py`, verified live) --
_SHAPE_OP_CASES = [
    ("2x3", [[1, 2, 3], [4, 5, 6]]),
    ("3x2", [[1, 2], [3, 4], [5, 6]]),
    ("1x4", [[4, 1, 3, 2]]),
    ("neg", [[-1, 2, -3], [4, -5, 6]]),
]


def _reshape_cases():
    shapes = [(3, 2), (6, 1), (1, 6), (2, 3)]
    out = []
    for label, data in _SHAPE_OP_CASES:
        for shape in shapes:
            out.append((f"reshape/{label}/{shape}", (data, shape), {}))
    return out


def _reshape_numpy(data, shape):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        try:
            return _snapshot(m.reshape(shape))
        except Exception as e:
            return ("error", type(e).__name__, str(e))


def _reshape_ionp(data, shape):
    m = ap_matrix(data)
    try:
        return _snapshot(m.reshape(shape))
    except Exception as e:
        return ("error", type(e).__name__, str(e))


MATRIX_SPECS["matrix.reshape"] = ItemSpec(
    name="matrix.reshape", kind="custom", scalar_like=True,
    numpy_adapter=_reshape_numpy, ionp_adapter=_reshape_ionp,
    custom_cases=_reshape_cases,
)


def _make_simple_shape_spec(name, np_call, ionp_call):
    def cases():
        return [(f"{name}/{label}", (data,), {}) for label, data in _SHAPE_OP_CASES]

    def numpy_adapter(data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.matrix(data)
            return _snapshot(np_call(m))

    def ionp_adapter(data):
        m = ap_matrix(data)
        return _snapshot(ionp_call(m))

    MATRIX_SPECS[f"matrix.{name}"] = ItemSpec(
        name=f"matrix.{name}", kind="custom", scalar_like=True,
        numpy_adapter=numpy_adapter, ionp_adapter=ionp_adapter,
        custom_cases=cases,
    )


_make_simple_shape_spec("swapaxes", lambda m: m.swapaxes(0, 1), lambda m: m.swapaxes(0, 1))
_make_simple_shape_spec("repeat", lambda m: m.repeat(2, axis=0), lambda m: m.repeat(2, axis=0))
_make_simple_shape_spec("argsort", lambda m: m.argsort(), lambda m: m.argsort())
_make_simple_shape_spec("cumsum", lambda m: m.cumsum(), lambda m: m.cumsum())
_make_simple_shape_spec("cumprod", lambda m: m.cumprod(), lambda m: m.cumprod())
_make_simple_shape_spec("clip", lambda m: m.clip(-2, 3), lambda m: m.clip(-2, 3))
# `matrix.cumsum`'s corpus above covers axis=None only; the axis=0/1 forms
# are verified separately, out of corpus, before declaring (see this
# task's report) rather than added as additional registry items here.


# -- fill / sort: in-place mutation, snapshot the mutated receiver --
_FILL_SORT_CASES = [
    ("2x3", [[3, 1, 2], [6, 4, 5]]),
    ("1x4", [[4, 1, 3, 2]]),
    ("neg", [[-1, 2, -3], [4, -5, 6]]),
]


def _fill_cases():
    return [(f"fill/{label}", (data,), {}) for label, data in _FILL_SORT_CASES]


def _fill_numpy(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        r = m.fill(7)
        return ("scalar", "fill", (r, _snapshot(m)))


def _fill_ionp(data):
    m = ap_matrix(data)
    r = m.fill(7)
    return ("scalar", "fill", (r, _snapshot(m)))


MATRIX_SPECS["matrix.fill"] = ItemSpec(
    name="matrix.fill", kind="custom", scalar_like=True,
    numpy_adapter=_fill_numpy, ionp_adapter=_fill_ionp,
    custom_cases=_fill_cases,
)


def _sort_cases():
    return [(f"sort/{label}", (data,), {}) for label, data in _FILL_SORT_CASES]


def _sort_numpy(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.matrix(data)
        r = m.sort()
        return ("scalar", "sort", (r, _snapshot(m)))


def _sort_ionp(data):
    m = ap_matrix(data)
    r = m.sort()
    return ("scalar", "sort", (r, _snapshot(m)))


MATRIX_SPECS["matrix.sort"] = ItemSpec(
    name="matrix.sort", kind="custom", scalar_like=True,
    numpy_adapter=_sort_numpy, ionp_adapter=_sort_ionp,
    custom_cases=_sort_cases,
)


import registry  # noqa: E402

_collisions = set(MATRIX_SPECS) & set(registry.REGISTRY)
if _collisions:
    raise AssertionError(f"matrix_cases.py: REGISTRY collision(s): {sorted(_collisions)}")
registry.REGISTRY.update(MATRIX_SPECS)
