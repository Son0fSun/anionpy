"""NEW test-cases file (owned by this task): order-semantics differential
coverage crossing order in {'C','F','A','K'} x memory-layout class (C-
contiguous, F-contiguous, transposed view, sliced/non-contiguous view,
row-reversed view, 0-d, empty) x all 14 dtypes, for ravel/flatten/copy/
astype/zeros_like/ones_like/empty_like/full_like/reshape/ndarray.array-
ingestion.

WHY THIS IS A SEPARATE FILE, NOT MORE ENTRIES IN THE EXISTING
"ravel"/"ndarray.ravel"/"copy"/etc. items
---------------------------------------------------------------------------
Those item NAMES already exist in the registry (declared by
creation_cases.py / ndarray_attrs_cases.py / registry.py's own body) and the
tail-merge pattern this file also uses REFUSES a name collision (see the
merge block appended to registry.py for this file). This file therefore
registers every case under a FRESH key (the "order/..." prefix below) and
points `numpy_path`/`ionp_path` at the REAL surface item (e.g.
`ionp_path="ndarray.ravel"`), so `ItemSpec.resolve_numpy()`/`resolve_ionp()`
dispatch to the actual `arr.ravel(...)` call on both sides -- these are not
synthetic/fake comparisons, they exercise the identical Rust code path the
already-declared "ndarray.ravel" item does, just against a purpose-built
corpus this task's three bugs actually needed (measured absence: the
existing corpus only ever passed order='F' to ravel/flatten, and NOTHING
anywhere passed order='A' or order='K' to copy/astype/_like/reshape before
this file). A key not present in tools/numpy_surface.json contributes zero
to the coverage ledger (coverage.py only ever looks up the fixed surface
list, see build_ledger) -- these items exist purely as evidence for this
task's report, not to inflate the percentage.

WHY THIS ACTUALLY EXERCISES THE ndarray_from_numpy INGESTION FIX
---------------------------------------------------------------------------
Every case below starts from a genuine numpy array (real strides, real
layout, produced by real numpy operations like `.T`, `np.asfortranarray`,
slicing). `resolve_ionp()`'s conversion path
(`registry.make_ionp_array_converter`) turns that into an anionpy array via
`anionpy.array(x)` -- literally the function this task patched
(`ndarray_from_numpy` in ionp-py/src/lib.rs) to preserve F-vs-C layout
instead of always forcing C order. So an order='A'/'K' ravel/flatten/copy/
astype/_like case against an F-contiguous or transposed source is a direct,
end-to-end regression test for that fix, not a synthetic one.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec
from corpus import _fill, ALL_DTYPES

# Distinct, pinned seed for this file's own array construction -- keeps this
# file's corpus independent of (and reproducible without depending on the
# exact call sequence of) corpus.py's own SEED-derived RNG stream.
_SEED = 20260801

# corpus.ALL_DTYPES (bool + int8-64 + uint8-64 + float32/64 + complex64/128)
# is 13 dtypes -- it deliberately omits float16 (see corpus.py's own
# SWEEP_DTYPES comment: it's a representative subset for most items). This
# task's brief explicitly asks for "all 14 dtypes", and anionpy DOES support
# float16 (ndarray_from_numpy's half::f16 branch), so it is added back here
# rather than silently inherited-missing.
ORDER_DTYPES = ALL_DTYPES + [np.float16]

ORDERS = ["C", "F", "A", "K"]


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _layouts(rng: np.random.Generator, dtype) -> dict[str, np.ndarray]:
    """One array per memory-layout class this task's bugs care about."""
    base = _fill(rng, (3, 4), dtype)
    return {
        "c_contig_2d": np.ascontiguousarray(base),
        "f_contig_2d": np.asfortranarray(base),
        # base is C-contiguous, so its transpose is genuinely F-contiguous
        # AND NOT C-contiguous -- exactly the case bug (a) got wrong.
        "transposed_view": base.T,
        "sliced_noncontig_2d": base[:, ::2],
        "row_reversed_2d": base[::-1, :],
        "scalar_0d": _fill(rng, (), dtype),
        "empty_1d": _fill(rng, (0,), dtype),
        "empty_2d": _fill(rng, (0, 3), dtype),
    }


def _order_matrix(label_prefix: str, extra_args=(), extra_kwargs_fn=None) -> list[tuple[str, tuple, dict]]:
    """(label, args, kwargs) triples for every (dtype, layout, order) combo.

    `extra_args`/`extra_kwargs_fn(arr)` let callers (astype, reshape) inject
    a dtype/shape positional argument ahead of the order kwarg.
    """
    out = []
    rng = _rng()
    for dtype in ORDER_DTYPES:
        dtname = np.dtype(dtype).name
        for layout_name, arr in _layouts(rng, dtype).items():
            for order in ORDERS:
                kwargs = {"order": order}
                if extra_kwargs_fn is not None:
                    kwargs.update(extra_kwargs_fn(arr))
                out.append((
                    f"{label_prefix}/{dtname}/{layout_name}/order_{order}",
                    (arr, *extra_args),
                    kwargs,
                ))
    return out


# ---------------------------------------------------------------------------
# ravel / flatten -- bug (a) and (b) directly: order='A' was silently wrong
# for an ingested F-contiguous/transposed source (root cause:
# ndarray_from_numpy always forcing Order::C on ingestion, now fixed), and
# order='K' was rejected outright everywhere (now implemented via
# NdArray::ravel_order's axis_perm_for_order/gather_by_perm).
# ---------------------------------------------------------------------------

def ravel_order_cases():
    return _order_matrix("order/ravel")


def flatten_order_cases():
    return _order_matrix("order/flatten")


# ---------------------------------------------------------------------------
# copy / astype -- both default to order='K' in real numpy (not 'C'), and
# both now route through NdArray::to_contiguous_order (array.rs) for all
# four order letters.
# ---------------------------------------------------------------------------

def copy_order_cases():
    return _order_matrix("order/copy")


def _astype_same_dtype_kwargs(arr):
    return {}


def astype_order_cases():
    rng = _rng()
    out = []
    for dtype in ORDER_DTYPES:
        dtname = np.dtype(dtype).name
        for layout_name, arr in _layouts(rng, dtype).items():
            for order in ORDERS:
                # Same-dtype target isolates the order/layout effect from
                # dtype-cast value changes -- astype's *casting* correctness
                # is already covered by the existing "ndarray.astype" item;
                # this item is specifically about what layout the OUTPUT
                # ends up with, which is exactly what order= controls.
                out.append((
                    f"order/astype/{dtname}/{layout_name}/order_{order}",
                    (arr, dtype),
                    {"order": order},
                ))
    return out


# ---------------------------------------------------------------------------
# zeros_like / ones_like / full_like -- order= is resolved against the
# PROTOTYPE array's own layout (see ionp-py/src/creation.rs's
# like_strides()), so the prototype's layout variety here is the entire
# point of this matrix (a same-shape-as-prototype `_like` call with
# order='K' on an F-contiguous prototype must come back F-contiguous, not
# C). empty_like is deliberately excluded -- its contents are uninitialized
# garbage on both sides and not a valid basis for comparison (same rule
# creation_cases.py's own empty/empty_like items already follow); shape/
# dtype/layout-affecting behavior for empty_like is not order-specific
# beyond what zeros_like/ones_like already prove here.
# ---------------------------------------------------------------------------

def zeros_like_order_cases():
    return _order_matrix("order/zeros_like")


def ones_like_order_cases():
    return _order_matrix("order/ones_like")


def full_like_order_cases():
    rng = _rng()
    out = []
    for dtype in ORDER_DTYPES:
        dtname = np.dtype(dtype).name
        fill_value = True if np.dtype(dtype).kind == "b" else 3
        for layout_name, arr in _layouts(rng, dtype).items():
            for order in ORDERS:
                out.append((
                    f"order/full_like/{dtname}/{layout_name}/order_{order}",
                    (arr, fill_value),
                    {"order": order},
                ))
    return out


# ---------------------------------------------------------------------------
# reshape -- order='K' is refused with a DISTINCT numpy message ("order 'K'
# is not permitted for reshaping") even though 'K' is a recognized letter
# everywhere else; 'C'/'F'/'A' must still reshape correctly across the same
# layout matrix. Two separate case builders so the "must raise" ones don't
# get diluted into (and can't silently mask a value regression in) the
# value-correctness ones.
# ---------------------------------------------------------------------------

def reshape_order_value_cases():
    out = []
    rng = _rng()
    for dtype in ORDER_DTYPES:
        dtname = np.dtype(dtype).name
        for layout_name, arr in _layouts(rng, dtype).items():
            n = int(np.asarray(arr).size)
            for order in ("C", "F", "A"):
                out.append((
                    f"order/reshape/{dtname}/{layout_name}/order_{order}",
                    (arr, (n,)),
                    {"order": order},
                ))
    return out


def reshape_order_k_must_raise_cases():
    out = []
    rng = _rng()
    for dtype in ORDER_DTYPES:
        dtname = np.dtype(dtype).name
        arr = _fill(rng, (3, 4), dtype)
        out.append((
            f"order/reshape_k_must_raise/{dtname}",
            (arr, (12,)),
            {"order": "K"},
        ))
    return out


ORDER_SPECS: dict[str, ItemSpec] = {
    "order/ravel": ItemSpec(
        name="order/ravel", kind="custom",
        ionp_path="ndarray.ravel", numpy_path="ndarray.ravel",
        custom_cases=ravel_order_cases,
        atol=0.0, rtol=0.0,
    ),
    "order/flatten": ItemSpec(
        name="order/flatten", kind="custom",
        ionp_path="ndarray.flatten", numpy_path="ndarray.flatten",
        custom_cases=flatten_order_cases,
        atol=0.0, rtol=0.0,
    ),
    "order/copy": ItemSpec(
        name="order/copy", kind="custom",
        ionp_path="ndarray.copy", numpy_path="ndarray.copy",
        custom_cases=copy_order_cases,
        atol=0.0, rtol=0.0,
    ),
    "order/astype": ItemSpec(
        name="order/astype", kind="custom",
        ionp_path="ndarray.astype", numpy_path="ndarray.astype",
        custom_cases=astype_order_cases,
        atol=0.0, rtol=0.0,
    ),
    "order/zeros_like": ItemSpec(
        name="order/zeros_like", kind="custom",
        ionp_path="zeros_like", numpy_path="zeros_like",
        custom_cases=zeros_like_order_cases,
        atol=0.0, rtol=0.0,
    ),
    "order/ones_like": ItemSpec(
        name="order/ones_like", kind="custom",
        ionp_path="ones_like", numpy_path="ones_like",
        custom_cases=ones_like_order_cases,
        atol=0.0, rtol=0.0,
    ),
    "order/full_like": ItemSpec(
        name="order/full_like", kind="custom",
        ionp_path="full_like", numpy_path="full_like",
        custom_cases=full_like_order_cases,
        atol=0.0, rtol=0.0,
    ),
    "order/reshape_values": ItemSpec(
        name="order/reshape_values", kind="custom",
        ionp_path="ndarray.reshape", numpy_path="ndarray.reshape",
        custom_cases=reshape_order_value_cases,
        atol=0.0, rtol=0.0,
    ),
    "order/reshape_k_must_raise": ItemSpec(
        name="order/reshape_k_must_raise", kind="custom",
        ionp_path="ndarray.reshape", numpy_path="ndarray.reshape",
        custom_cases=reshape_order_k_must_raise_cases,
        atol=0.0, rtol=0.0,
    ),
}
