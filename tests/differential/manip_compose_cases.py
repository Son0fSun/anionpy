"""Differential coverage for `anionpy/_manip_compose.py`: `asanyarray`,
module-level `astype`, and `unstack`. NEW FILE (same collision-checked-merge
pattern as manip_cases.py/order_cases.py/reduction_cases.py -- builds a dict
of ItemSpecs, merged into registry.REGISTRY from the bottom of registry.py).

All three items are Python-only wrappers over already-existing/already-
verified anionpy primitives (`asarray`, `ndarray.astype`, `moveaxis`) -- no
Rust changes. See `_manip_compose.py`'s module docstring for the exact
declared-vs-excluded scope of each (the open identity/layout-contract
question: `copy=False` on `asanyarray`/`astype`, and non-`{None,"K"}`
`order=` on `asanyarray`, are deliberately NOT exercised here -- see that
docstring for why).

`unstack` returns a tuple of arrays from BOTH real numpy and anionpy (verified:
`type(np.unstack(x))` is `tuple`) -- registered `multi_output=True`, no
adapter needed (unlike split/array_split/h/v/dsplit's own list->tuple
adapters in manip_cases.py, which exist only because THOSE return list, not
tuple, natively).
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec
from corpus import unary_corpus, SWEEP_DTYPES

_CORPUS = unary_corpus()


def _corpus_cases(fn_kwargs=None, label_suffix: str = "") -> list[tuple[str, tuple, dict]]:
    kw = fn_kwargs or {}
    return [(c.label + label_suffix, (c.value,), dict(kw)) for c in _CORPUS]


# ---------------------------------------------------------------------------
# asanyarray
# ---------------------------------------------------------------------------

def asanyarray_cases():
    out = []
    # Full unary corpus (shape sweep, special floats, integer boundaries,
    # views/non-contiguous/negative-stride/F-order, random) at the default
    # call form -- dtype=None, order=None, device=None, copy=None, like=None.
    out += _corpus_cases(label_suffix="/default")

    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    # dtype= override, same-dtype no-op, and every SWEEP_DTYPES target
    # (narrowing AND widening conversions).
    out.append(("dtype_none_explicit", (a,), {"dtype": None}))
    out.append(("dtype_same", (a,), {"dtype": np.int32}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype_target/{np.dtype(dtype).name}", (a,), {"dtype": dtype}))
    # dtype as a string spec (a.dtype's own __eq__ accepts strings directly,
    # verified live -- see _manip_compose.py's normalization note).
    out.append(("dtype_string_same", (a,), {"dtype": "int32"}))
    out.append(("dtype_string_diff", (a,), {"dtype": "float64"}))

    # order in the declared scope: None and "K" only (see module docstring
    # for why "C"/"F"/"A" are excluded).
    out.append(("order_none", (a,), {"order": None}))
    out.append(("order_K", (a,), {"order": "K"}))

    # device=None (the only value real numpy accepts as a no-op besides
    # "cpu") and device="cpu" explicitly, both must be accepted silently.
    out.append(("device_none", (a,), {"device": None}))
    out.append(("device_cpu", (a,), {"device": "cpu"}))
    # device="gpu": real numpy raises ValueError with an exact message this
    # function replicates verbatim -- boundary case, not a value comparison.
    out.append(("device_bad_raises", (a,), {"device": "gpu"}))

    # copy=True: forces the asarray-copy path regardless of input type --
    # values must still be correct (this is the declared/tested copy=
    # scope; copy=False is deliberately excluded, see module docstring).
    out.append(("copy_true", (a,), {"copy": True}))
    out.append(("copy_none_explicit", (a,), {"copy": None}))

    # 0-d and empty-array forms.
    out.append(("scalar_0d", (np.array(5.0),), {}))
    out.append(("empty_1d", (np.array([], dtype=np.float64),), {}))
    out.append(("empty_2d_zero_rows", (np.empty((0, 3), dtype=np.int64),), {}))

    # foreign (non-ndarray) input forms -- list/tuple/range/plain scalar,
    # all of which must fall through to the asarray-conversion path.
    out.append(("form_list", ([1, 2, 3],), {}))
    out.append(("form_nested_list", ([[1, 2], [3, 4]],), {}))
    out.append(("form_tuple", ((1.0, 2.0, 3.0),), {}))
    out.append(("form_range", (range(5),), {}))
    out.append(("form_python_scalar", (5,), {}))
    out.append(("form_python_float", (5.5,), {}))

    return out


# ---------------------------------------------------------------------------
# astype
# ---------------------------------------------------------------------------

def astype_cases():
    out = []
    a = np.arange(12, dtype=np.int32).reshape(3, 4)

    # Every SWEEP_DTYPES target (widening, narrowing, float<->int,
    # int<->complex where legal, bool round-trips), default copy=True.
    for src_dtype in SWEEP_DTYPES:
        src = a.astype(src_dtype) if np.dtype(src_dtype).kind != "c" else \
            (a + 1j * a).astype(src_dtype)
        for dst_dtype in SWEEP_DTYPES:
            out.append((
                f"{np.dtype(src_dtype).name}_to_{np.dtype(dst_dtype).name}",
                (src, dst_dtype), {},
            ))

    # copy=True explicit (same as default -- must still copy, never
    # identity, verified live against real numpy's own copy=True
    # never-passthrough behavior).
    out.append(("copy_true_same_dtype", (a, np.int32), {"copy": True}))

    # device=None / "cpu" accepted; device="gpu" raises the exact ValueError
    # real numpy raises (verified live), replicated verbatim.
    out.append(("device_none", (a, np.float64), {"device": None}))
    out.append(("device_cpu", (a, np.float64), {"device": "cpu"}))
    out.append(("device_bad_raises", (a, np.float64), {"device": "gpu"}))

    # Non-ndarray input: real numpy's own dedicated TypeError
    # ("Input should be a NumPy array or scalar. It is a <class 'list'>
    # instead.", verified live, not paraphrased) -- boundary/error-path
    # case, not a value comparison.
    out.append(("list_input_raises", ([1, 2, 3], np.float64), {}))

    # 0-d, empty, and view (non-contiguous/negative-stride/F-order) inputs
    # via the full unary corpus, converting every case to float64.
    for c in _CORPUS:
        out.append((f"corpus/{c.label}/to_f64", (c.value, np.float64), {}))

    return out


# ---------------------------------------------------------------------------
# unstack
# ---------------------------------------------------------------------------

def unstack_cases():
    out = []

    m = np.arange(12).reshape(3, 4)
    out.append(("axis0_default", (m,), {}))
    out.append(("axis0_explicit", (m,), {"axis": 0}))
    out.append(("axis1", (m,), {"axis": 1}))
    out.append(("axis_neg1", (m,), {"axis": -1}))
    out.append(("axis_neg2", (m,), {"axis": -2}))

    t3 = np.arange(24).reshape(2, 3, 4)
    out.append(("3d_axis0", (t3,), {}))
    out.append(("3d_axis1", (t3,), {"axis": 1}))
    out.append(("3d_axis2", (t3,), {"axis": 2}))
    out.append(("3d_axis_neg1", (t3,), {"axis": -1}))

    out.append(("1d", (np.arange(5),), {}))
    out.append(("1d_single_element", (np.array([7]),), {}))

    # 0-d raises ValueError ("Input array must be at least 1-d.", verified
    # live, exact text).
    out.append(("0d_raises", (np.array(5.0),), {}))

    # shape[axis] == 0 -> empty tuple, no split/division involved at all.
    out.append(("empty_axis0", (np.empty((0, 3)),), {}))
    out.append(("empty_axis1", (np.empty((3, 0)),), {"axis": 1}))

    # axis out of range -> AxisError (verified live: anionpy's own moveaxis
    # already raises numpy's actual AxisError class here, see this file's
    # accompanying report for that finding).
    out.append(("axis_oob_raises", (m,), {"axis": 5}))
    out.append(("axis_oob_neg_raises", (m,), {"axis": -5}))

    # Non-ndarray input: real numpy's own body reads `x.ndim` unguarded, so
    # a plain list raises AttributeError -- verified live, matched exactly
    # (see _manip_compose.py's docstring for why no asarray-coercion
    # happens here).
    out.append(("list_input_raises", ([1, 2, 3],), {}))

    # dtype sweep + views (non-contiguous/negative-stride/F-order) via the
    # full unary corpus, restricted to ndim>=1 (0-d already covered above
    # as its own explicit raises-case) and axis=0.
    for c in _CORPUS:
        if c.value.ndim == 0:
            continue
        out.append((f"corpus/{c.label}", (c.value,), {}))

    return out


def _build_manip_compose_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}
    specs["asanyarray"] = ItemSpec(name="asanyarray", kind="custom", custom_cases=asanyarray_cases)
    specs["astype"] = ItemSpec(name="astype", kind="custom", custom_cases=astype_cases)
    specs["unstack"] = ItemSpec(
        name="unstack", kind="custom", custom_cases=unstack_cases, multi_output=True,
    )
    return specs


MANIP_COMPOSE_SPECS = _build_manip_compose_specs()
