"""NEW test-cases file: N-D `.reduceat`/`.accumulate` corpus, closing the
SILENT wrong-shape/wrong-value bug both methods had on any `ndim > 1`
input before their 2026-08-06 (Monday) fix.

THE BUG THIS CORPUS TARGETS
----------------------------------------------------------------------------
`ionp-py/src/lib.rs`'s `.reduceat()` and `.accumulate()` ufunc-method
bindings used to accept an `axis` argument and then throw it away entirely
(`let _ = (axis, dtype);`), unconditionally delegating to `ionp-core`'s
1-D-only `reduceat_binary`/`reduceat_math_binary`/`accumulate_binary`/
`accumulate_math_binary`, all four of which walk the input via
`shape[0]`/`strides[0]` no matter what `axis` was or how many dimensions
the array actually had.

Measured effect (real numpy 2.5.1 vs anionpy, pre-fix):
  - `.reduceat` on any `ndim > 1` array: WRONG SHAPE (every dimension
    except axis 0 is silently dropped -- `np.add.reduceat(np.arange(24.)
    .reshape(4,6), [0,2], axis=0)` is shape `(2,6)` in real numpy, anionpy
    silently returned shape `(2,)`, keeping only column 0) AND WRONG
    VALUES for every axis other than 0 (bound-checked against the WRONG
    axis's length when `axis != 0`, and folds only the axis-0 fiber).
  - `.accumulate` on `ndim > 1`: usually a Rust-buffer-length `ValueError`
    (loud, not silent) -- EXCEPT when every non-zeroth dimension happens
    to be size 1 (`product(shape) == shape[0]`), in which case the output
    buffer is coincidentally the right SIZE and the wrong-axis
    accumulation is returned with no error at all.

Fixed by `reduceat_axis_generic`/`reduceat_binary_axis`/
`reduceat_math_binary_axis` (new) and `accumulate_axis`/
`accumulate_math_binary_axis` (the `BinaryOp` one, `accumulate_axis`, was
pre-existing infrastructure already used by `cumsum`/`cumprod`; only its
`MathBinaryOp` sibling `accumulate_math_binary_axis` is new) in
`ionp-core/src/ufunc.rs`, wired through a shared
`normalize_ufunc_method_axis` courtesy-ndim helper in `ionp-py/src/lib.rs`.

WHY THIS CANNOT BE `ufunc_cases.py`'s EXISTING GENERIC `.reduceat` CASES
----------------------------------------------------------------------------
`ufunc_cases.py`'s `_reduceat_indices(n)` generator and the whole
`kind="ufunc"` call-form corpus it feeds are built exclusively from
`corpus.unary_corpus()` 1-D arrays with `axis` always defaulting to `0`
(`sample[0]`/`arr.shape[0]` throughout) -- no case in that corpus, however
thorough on dtype/index axes, is even capable of constructing an `ndim > 1`
call with a non-zero `axis`. This is a fresh, additive namespace
(`"ndim/reduceat/<op>"` / `"ndim/accumulate/<op>"`), not more entries on
the real `"add"`/`"multiply"`/... items, for the same registry-key-
collision reason `floordiv_crossing_cases.py`/`power_view_cases.py` give
for their own separate files.

CORPUS SHAPE
----------------------------------------------------------------------------
Six ops crossed with N-D shapes/axes/dtypes/edge-case indices:
  - `add`, `multiply`, `maximum`, `minimum` (plain `BinaryOp`, float64 +
    int64 + complex128 as applicable)
  - `logical_or` (plain `BinaryOp`, bool -- the logical family)
  - `power` (`MathBinaryOp` -- also exercises the per-fiber negative-
    integer-exponent segment check generalized to N-D)
Shapes: 2-D `(4,6)` and 3-D `(2,3,4)`, axis in `{0, 1, -1}` (and axis `2`
for the 3-D shape). Edge-case indices: empty, unsorted, out-of-range,
repeated, descending, `index == axis_len`. Plus a dedicated 0-d
`TypeError`/`AxisError` boundary case for both methods (courtesy-ndim
axis validation, `normalize_ufunc_method_axis`'s own doc comment).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import registry
from registry import ItemSpec


def _mk_pair(probe):
    """Mirrors linalg_cases.py/power_view_cases.py's `_mk_pair`: splits one
    mod-parameterized probe into (numpy_adapter, ionp_adapter), lazily
    importing anionpy."""

    def numpy_side(*args, **kwargs):
        import numpy as np

        return probe(np, *args, **kwargs)

    def ionp_side(*args, **kwargs):
        import anionpy

        return probe(anionpy, *args, **kwargs)

    return numpy_side, ionp_side


def _reduceat_probe_for(opname):
    def probe(mod, data, dtype_name, indices, axis):
        arr = mod.asarray(data, dtype=getattr(mod, dtype_name))
        op = getattr(mod, opname)
        return op.reduceat(arr, indices, axis=axis)

    return _mk_pair(probe)


def _accumulate_probe_for(opname):
    def probe(mod, data, dtype_name, axis):
        arr = mod.asarray(data, dtype=getattr(mod, dtype_name))
        op = getattr(mod, opname)
        return op.accumulate(arr, axis=axis)

    return _mk_pair(probe)


def _reduceat_scalar_axis_probe(mod, opname, axis):
    arr = mod.asarray(5.0)
    op = getattr(mod, opname)
    return op.reduceat(arr, [0], axis=axis)


def _accumulate_scalar_axis_probe(mod, opname, axis):
    arr = mod.asarray(5.0)
    op = getattr(mod, opname)
    return op.accumulate(arr, axis=axis)


# ---------------------------------------------------------------------------
# Base data.
# ---------------------------------------------------------------------------

_DATA_2D = [[float(4 * r + c) + 1.0 for c in range(6)] for r in range(4)]  # (4, 6), 1..24
_DATA_3D = [[[float(12 * a + 4 * b + c) + 1.0 for c in range(4)] for b in range(3)] for a in range(2)]  # (2, 3, 4)
_DATA_2D_INT = [[(4 * r + c) % 7 - 3 for c in range(6)] for r in range(4)]  # includes negatives, mod cycles
_DATA_2D_BOOL = [[bool((r + c) % 2) for c in range(6)] for r in range(4)]
_DATA_2D_COMPLEX = [[complex(r, c) for c in range(6)] for r in range(4)]
# Small positive-int base for power (avoid negative-base surprises; the
# negative-EXPONENT segment check gets its own dedicated cases below).
_DATA_2D_POW = [[1 + (r + c) % 3 for c in range(6)] for r in range(4)]


def _reduceat_index_edge_cases_2d():
    """Edge-case indices crossed with both axes of the (4, 6) shape.
    axis=0 has length 4, axis=1 has length 6."""
    cases = []
    # axis=0 (length 4)
    for label, idx in [
        ("empty", []),
        ("unsorted", [2, 0]),
        ("repeated", [1, 1, 1]),
        ("descending", [3, 2, 1, 0]),
        ("index_eq_len_passthrough", [0, 4]),
        ("single_full_span", [0]),
        ("all_singletons", [0, 1, 2, 3]),
    ]:
        cases.append((f"axis0/{label}", (_DATA_2D, "float64", idx, 0), {}))
    # axis=1 (length 6)
    for label, idx in [
        ("empty", []),
        ("unsorted", [4, 1, 3]),
        ("repeated", [2, 2, 5]),
        ("descending", [5, 3, 0]),
        ("index_eq_len_passthrough", [0, 3, 6]),
        ("single_full_span", [0]),
        ("all_singletons", [0, 1, 2, 3, 4, 5]),
    ]:
        cases.append((f"axis1/{label}", (_DATA_2D, "float64", idx, 1), {}))
    # negative axis (numpy: axis=-1 == axis=1 here)
    cases.append(("axis_neg1/mixed", (_DATA_2D, "float64", [0, 2, 5], -1), {}))
    # out-of-range index must still raise IndexError, now bound-checked
    # against the CORRECT (target) axis's length, not axis 0's.
    cases.append(("axis1/out_of_range_against_axis1_len", (_DATA_2D, "float64", [0, 9], 1), {}))
    cases.append(("axis0/out_of_range_against_axis0_len", (_DATA_2D, "float64", [0, 9], 0), {}))
    # negative index: never valid for .reduceat (no wraparound).
    cases.append(("axis1/negative_index_invalid", (_DATA_2D, "float64", [0, -1], 1), {}))
    return cases


def _reduceat_index_edge_cases_3d():
    cases = []
    for axis, idx_variants in [
        (0, [("mid", [0, 1]), ("descending", [1, 0]), ("empty", [])]),
        (1, [("mid", [0, 2]), ("unsorted", [2, 0, 1]), ("repeated", [1, 1])]),
        (2, [("mid", [0, 2]), ("index_eq_len_passthrough", [0, 4]), ("descending", [3, 1, 0])]),
        (-1, [("mixed", [1, 3])]),
    ]:
        for label, idx in idx_variants:
            cases.append((f"axis{axis}/{label}", (_DATA_3D, "float64", idx, axis), {}))
    return cases


def _reduceat_dtype_cases():
    return [
        ("int64/axis0", (_DATA_2D_INT, "int64", [0, 2], 0), {}),
        ("int64/axis1", (_DATA_2D_INT, "int64", [0, 2, 4], 1), {}),
        ("complex128/axis0", (_DATA_2D_COMPLEX, "complex128", [0, 2], 0), {}),
        ("complex128/axis1", (_DATA_2D_COMPLEX, "complex128", [0, 3], 1), {}),
    ]


def _accumulate_edge_cases_2d():
    return [
        ("axis0", (_DATA_2D, "float64", 0), {}),
        ("axis1", (_DATA_2D, "float64", 1), {}),
        ("axis_neg1", (_DATA_2D, "float64", -1), {}),
        ("axis_neg2", (_DATA_2D, "float64", -2), {}),
        ("int64/axis0", (_DATA_2D_INT, "int64", 0), {}),
        ("int64/axis1", (_DATA_2D_INT, "int64", 1), {}),
        ("complex128/axis0", (_DATA_2D_COMPLEX, "complex128", 0), {}),
        ("complex128/axis1", (_DATA_2D_COMPLEX, "complex128", 1), {}),
    ]


def _accumulate_edge_cases_3d():
    return [
        ("axis0", (_DATA_3D, "float64", 0), {}),
        ("axis1", (_DATA_3D, "float64", 1), {}),
        ("axis2", (_DATA_3D, "float64", 2), {}),
        ("axis_neg1", (_DATA_3D, "float64", -1), {}),
    ]


def _reduceat_cases_for(opname, dtype_cases_enabled=True):
    def _cases():
        out = list(_reduceat_index_edge_cases_2d())
        out += list(_reduceat_index_edge_cases_3d())
        if dtype_cases_enabled:
            out += list(_reduceat_dtype_cases())
        return out

    return _cases


def _accumulate_cases_for(opname):
    def _cases():
        return list(_accumulate_edge_cases_2d()) + list(_accumulate_edge_cases_3d())

    return _cases


def _logical_or_reduceat_cases():
    return [
        ("axis0/basic", (_DATA_2D_BOOL, "bool_", [0, 2], 0), {}),
        ("axis1/basic", (_DATA_2D_BOOL, "bool_", [0, 3], 1), {}),
        ("axis1/unsorted", (_DATA_2D_BOOL, "bool_", [4, 1], 1), {}),
    ]


def _logical_or_accumulate_cases():
    return [
        ("axis0", (_DATA_2D_BOOL, "bool_", 0), {}),
        ("axis1", (_DATA_2D_BOOL, "bool_", 1), {}),
    ]


def _power_reduceat_cases():
    # Positive-base cases (values/shape correctness).
    cases = [
        ("axis0/positive", (_DATA_2D_POW, "int64", [0, 2], 0), {}),
        ("axis1/positive", (_DATA_2D_POW, "int64", [0, 2, 4], 1), {}),
    ]
    return cases


_power_negexp_np, _power_negexp_ionp = _mk_pair(
    lambda mod, data, indices, axis: mod.power.reduceat(mod.asarray(data, dtype=mod.int64), indices, axis=axis)
)


def _power_reduceat_negexp_case_list():
    data = [[2, 3, 4, 5, 6, 7], [2, 3, -4, 5, -6, 7]]
    return [
        # Segment starts (col 0, col 3) coincide with the negative
        # positions in a DIFFERENT case below to prove the exemption is
        # genuinely per-segment (exempt), not a blanket "index 0 of the
        # fiber" rule.
        ("axis1/negexp_mid_segment_raises", (data, [0, 3], 1), {}),
        # Negative values ARE at segment starts here (indices=[2, 4]) --
        # must be exempt (no raise), proving numpy's own per-segment base
        # exemption is honored, not just "first column of the array".
        ("axis1/negexp_at_segment_start_ok", (data, [2, 4], 1), {}),
    ]


def _power_accumulate_negexp_cases():
    data = [[2, 3, 4, 5, 6, 7], [2, 3, -4, 5, -6, 7]]
    return [
        # Accumulate along axis=0 (length 2): each COLUMN is its own
        # fiber; column 2 and column 4 have their negative at fiber
        # position 1 (not position 0), so must raise. Column 0's fiber
        # ([2, 2]) has no negative at all and must not raise.
        ("axis0/negexp_mid_fiber_raises", (data, 0), {}),
    ]


REDUCEAT_NDIM_SPECS: dict[str, ItemSpec] = {}
ACCUMULATE_NDIM_SPECS: dict[str, ItemSpec] = {}

for _op in ["add", "multiply", "maximum", "minimum"]:
    _rq_np, _rq_ionp = _reduceat_probe_for(_op)
    REDUCEAT_NDIM_SPECS[f"ndim/reduceat/{_op}"] = ItemSpec(
        name=f"ndim/reduceat/{_op}",
        kind="custom",
        numpy_path=_op,
        ionp_path=_op,
        atol=0.0,
        rtol=0.0,
        numpy_adapter=_rq_np,
        ionp_adapter=_rq_ionp,
        custom_cases=_reduceat_cases_for(_op),
    )
    _ac_np, _ac_ionp = _accumulate_probe_for(_op)
    ACCUMULATE_NDIM_SPECS[f"ndim/accumulate/{_op}"] = ItemSpec(
        name=f"ndim/accumulate/{_op}",
        kind="custom",
        numpy_path=_op,
        ionp_path=_op,
        atol=0.0,
        rtol=0.0,
        numpy_adapter=_ac_np,
        ionp_adapter=_ac_ionp,
        custom_cases=_accumulate_cases_for(_op),
    )

_lor_rq_np, _lor_rq_ionp = _reduceat_probe_for("logical_or")
REDUCEAT_NDIM_SPECS["ndim/reduceat/logical_or"] = ItemSpec(
    name="ndim/reduceat/logical_or",
    kind="custom",
    numpy_path="logical_or",
    ionp_path="logical_or",
    numpy_adapter=_lor_rq_np,
    ionp_adapter=_lor_rq_ionp,
    custom_cases=_logical_or_reduceat_cases,
)
_lor_ac_np, _lor_ac_ionp = _accumulate_probe_for("logical_or")
ACCUMULATE_NDIM_SPECS["ndim/accumulate/logical_or"] = ItemSpec(
    name="ndim/accumulate/logical_or",
    kind="custom",
    numpy_path="logical_or",
    ionp_path="logical_or",
    numpy_adapter=_lor_ac_np,
    ionp_adapter=_lor_ac_ionp,
    custom_cases=_logical_or_accumulate_cases,
)

_pow_rq_np, _pow_rq_ionp = _reduceat_probe_for("power")
REDUCEAT_NDIM_SPECS["ndim/reduceat/power"] = ItemSpec(
    name="ndim/reduceat/power",
    kind="custom",
    numpy_path="power",
    ionp_path="power",
    atol=0.0,
    rtol=0.0,
    numpy_adapter=_pow_rq_np,
    ionp_adapter=_pow_rq_ionp,
    custom_cases=_power_reduceat_cases,
)
REDUCEAT_NDIM_SPECS["ndim/reduceat/power_negexp_segment"] = ItemSpec(
    name="ndim/reduceat/power_negexp_segment",
    kind="custom",
    numpy_path="power",
    ionp_path="power",
    numpy_adapter=_power_negexp_np,
    ionp_adapter=_power_negexp_ionp,
    custom_cases=_power_reduceat_negexp_case_list,
)
_pow_ac_np, _pow_ac_ionp = _accumulate_probe_for("power")
ACCUMULATE_NDIM_SPECS["ndim/accumulate/power"] = ItemSpec(
    name="ndim/accumulate/power",
    kind="custom",
    numpy_path="power",
    ionp_path="power",
    atol=0.0,
    rtol=0.0,
    numpy_adapter=_pow_ac_np,
    ionp_adapter=_pow_ac_ionp,
    custom_cases=lambda: [("axis0/positive", (_DATA_2D_POW, "int64", 0), {})],
)
_power_accneg_np, _power_accneg_ionp = _mk_pair(
    lambda mod, data, axis: mod.power.accumulate(mod.asarray(data, dtype=mod.int64), axis=axis)
)
ACCUMULATE_NDIM_SPECS["ndim/accumulate/power_negexp_fiber"] = ItemSpec(
    name="ndim/accumulate/power_negexp_fiber",
    kind="custom",
    numpy_path="power",
    ionp_path="power",
    numpy_adapter=_power_accneg_np,
    ionp_adapter=_power_accneg_ionp,
    custom_cases=_power_accumulate_negexp_cases,
)

# 0-d scalar boundary: courtesy-ndim axis validation must match numpy's
# exact TypeError/AxisError split (see normalize_ufunc_method_axis's doc
# comment in ionp-py/src/lib.rs for the measured rule).
_reduceat_scalar_np, _reduceat_scalar_ionp = _mk_pair(_reduceat_scalar_axis_probe)
_accumulate_scalar_np, _accumulate_scalar_ionp = _mk_pair(_accumulate_scalar_axis_probe)
REDUCEAT_NDIM_SPECS["ndim/reduceat/scalar_axis_boundary"] = ItemSpec(
    name="ndim/reduceat/scalar_axis_boundary",
    kind="custom",
    numpy_path="add",
    ionp_path="add",
    numpy_adapter=_reduceat_scalar_np,
    ionp_adapter=_reduceat_scalar_ionp,
    custom_cases=lambda: [
        ("axis0_courtesy_typeerror", ("add", 0), {}),
        ("axis_neg1_courtesy_typeerror", ("add", -1), {}),
        ("axis5_out_of_range_axiserror", ("add", 5), {}),
    ],
)
ACCUMULATE_NDIM_SPECS["ndim/accumulate/scalar_axis_boundary"] = ItemSpec(
    name="ndim/accumulate/scalar_axis_boundary",
    kind="custom",
    numpy_path="add",
    ionp_path="add",
    numpy_adapter=_accumulate_scalar_np,
    ionp_adapter=_accumulate_scalar_ionp,
    custom_cases=lambda: [
        ("axis0_courtesy_typeerror", ("add", 0), {}),
        ("axis_neg1_courtesy_typeerror", ("add", -1), {}),
        ("axis5_out_of_range_axiserror", ("add", 5), {}),
    ],
)

_ALL_NDIM_SPECS = {**REDUCEAT_NDIM_SPECS, **ACCUMULATE_NDIM_SPECS}
_ndim_collisions = set(_ALL_NDIM_SPECS) & set(registry.REGISTRY)
if _ndim_collisions:
    raise AssertionError(
        f"reduceat_ndim_cases.py: {sorted(_ndim_collisions)} already present "
        f"in registry.REGISTRY -- refusing to silently overwrite an existing "
        f"item"
    )
registry.REGISTRY.update(_ALL_NDIM_SPECS)
