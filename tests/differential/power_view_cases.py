"""NEW test-cases file: offset-view corpus for `power`/`power.outer`/
`power.at`'s integer-negative-exponent guard, closing the exact gap
`check_int_pow_no_negative_self` (ionp-core/src/ufunc.rs) had before its
2026-08-06 fix.

THE BUG THIS CORPUS TARGETS
----------------------------------------------------------------------------
`check_int_pow_no_negative_self` used to scan `operand_of!(a_cast,
$variant)`'s raw backing buffer (`v.as_slice()`, the WHOLE `Vec` behind the
array, not just the elements the array's shape/strides/offset actually
describe) instead of walking the array's LOGICAL elements. `NdArray::
cast_to`'s fast paths preserve a nonzero `offset` over a buffer sized to
the ORIGINAL allocation (only the `to_contiguous()` fallback rebases
offset to 0), so any ionp-native VIEW into a larger backing array (e.g.
`bf[2:]` where `bf = anionpy.asarray([-3,2,3,4,5])`) carries slack: elements
before the view's logical start (here, `bf`'s own `-3`) are still present
in the raw buffer the buggy scan walked, even though they are not part of
the view `bf[2:]` actually represents (`[3,4,5]`, no negatives). Effect:
false-positive `ValueError: Integers to negative integer powers are not
allowed.` on sliced/offset views that logically contain no negative
exponent -- anionpy.power(2, bf[2:]) raised, when it should return
`[8, 16, 32]` like real numpy.

WHY THIS CANNOT BE A `_power_edge_cases()` ENTRY IN ufunc_cases.py
----------------------------------------------------------------------------
Every existing power-edge case there constructs its ionp-side operand by
handing a NUMPY array (already possibly sliced) to the differential
harness, which converts it to anionpy via `ndarray_from_numpy`/
`ingest_array_preserving_layout` (ionp-py/src/lib.rs) -- and THAT
conversion builds a MINIMAL buffer sized exactly to the view's own extent
(`len = max_off - min_off + 1`), so no buffer slack survives the crossing
from numpy into anionpy. `anionpy.asarray` of a numpy slice therefore always
HIDES this bug -- the offset-blind scan has nothing stale left to read.
The only way to reproduce it is to build the FULL array as a real
`anionpy.ndarray` first, and then perform the slicing/offsetting using
anionpy's OWN `__getitem__`, exactly as `view_semantics_cases.py`'s
established idiom already does for other view-identity bugs. Hence this
file uses `_mk_pair`-style mod-parameterized probes (see linalg_cases.py's
`_mk_pair`): each probe receives `mod` (either the `numpy` module or the
`anionpy` module) and PLAIN PYTHON DATA (not numpy arrays), and performs
`mod.asarray(...)` + `mod`'s own slicing itself, so both sides build their
view using their own native machinery.

Plain Python lists/tuples are used for all base data (never numpy arrays)
specifically so `registry.py`'s `_wrap_custom_conversion`/
`make_ionp_array_converter` (`convert_ionp_args=True` default) has nothing
to convert -- only `np.ndarray` instances are converted, and there are
none in these cases' args, so this mechanism is a no-op here by
construction, not by having disabled it.

WHY A SEPARATE FILE/NAMESPACE, NOT MORE ENTRIES ON THE REAL "power" ITEM
----------------------------------------------------------------------------
Same reason `floordiv_crossing_cases.py` gives for its own namespace:
"power"/"power.outer"/"power.at" are already declared by
`ufunc_registry.py`. Fresh `"crossing/ufunc/power_view_*"` keys,
`kind="custom"` ItemSpecs, merged into `registry.REGISTRY` the same
tail-merge way, imported by run.py for the side effect.

MUST-NOT-REGRESS CASES INCLUDED
----------------------------------------------------------------------------
`bf[::2]` (logical `[-3,3,5]`, a GENUINE negative survives the stride)
must still raise -- both sides' probes assert this. `bf[1:][::-1]`
(reversed offset view, logical `[5,4,3,2]`, no negatives) must still
return the correct powers -- included as a passing case.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import registry
from registry import ItemSpec


def _mk_pair(probe):
    """Mirrors linalg_cases.py's `_mk_pair`: splits one mod-parameterized
    probe into (numpy_adapter, ionp_adapter), lazily importing anionpy."""

    def numpy_side(*args, **kwargs):
        import numpy as np

        return probe(np, *args, **kwargs)

    def ionp_side(*args, **kwargs):
        import anionpy

        return probe(anionpy, *args, **kwargs)

    return numpy_side, ionp_side


def _power_call_probe(mod, base_data, sl, exponent):
    """Build `mod.asarray(base_data)`, slice it with `mod`'s OWN
    `__getitem__` using `sl`, then compute `mod.power(exponent, view)`.
    `exponent` may itself be a python scalar/list (built via
    `mod.asarray` internally by `mod.power` for the scalar case, or
    explicitly asarray'd here for array bases -- kept as a plain list so
    both sides construct it identically)."""
    full = mod.asarray(base_data)
    view = full[sl]
    if isinstance(exponent, list):
        exponent = mod.asarray(exponent)
    return mod.power(exponent, view)


def _power_outer_probe(mod, base_data, sl, exponent_data):
    full = mod.asarray(base_data)
    view = full[sl]
    exp_arr = mod.asarray(exponent_data)
    return mod.power.outer(exp_arr, view)


def _power_at_probe(mod, base_data, sl, idx, out_seed):
    """`power.at(out, idx, view)`: mutates `out` in place, then RETURNS it
    (rather than None) so the harness's normal return-value comparison
    catches a wrong result without needing a separate descriptor-compare
    mechanism -- same trick copyto_cases.py's module docstring discusses
    for other in-place ops, applied minimally here since `.at`'s own
    output dtype/shape never varies (base identity), only its VALUES do."""
    full = mod.asarray(base_data)
    view = full[sl]
    out = mod.asarray(list(out_seed))
    mod.power.at(out, mod.asarray(list(idx)), view)
    return out


power_call_np, power_call_ionp = _mk_pair(_power_call_probe)
power_outer_np, power_outer_ionp = _mk_pair(_power_outer_probe)
power_at_np, power_at_ionp = _mk_pair(_power_at_probe)


# ---------------------------------------------------------------------------
# Case data.
# ---------------------------------------------------------------------------

_BASE_1D = [-3, 2, 3, 4, 5]  # bf in the task's own repro


def _extra_probes_and_specs():
    """Builds the probes/ItemSpecs that need slicing shapes
    `_power_call_probe` doesn't cover (two-step slice chains, 2-D,
    negative-stride-over-offset, broadcast) -- each gets its own tiny
    probe rather than over-generalizing `_power_call_probe`'s signature."""

    # bf[1:][::-1]: two-step slice chain -- needs its own probe variant
    # since `_power_call_probe` only slices once. Compose it directly here
    # instead of generalizing the probe for a single case.
    def _reversed_offset_probe(mod, base_data, exponent):
        full = mod.asarray(base_data)
        view = full[1:][::-1]
        return mod.power(exponent, view)

    rev_np, rev_ionp = _mk_pair(_reversed_offset_probe)

    # 2-D offset view, no negatives: base rows [-8..-5, -4..-1, 0..3, 4..7]
    # (4x4), view = base[2:, 2:] == [[2,3],[6,7]].
    base_2d = [[-8 + 4 * r + c for c in range(4)] for r in range(4)]

    def _power_call_probe_2d(mod, base_data, sl0, sl1, exponent):
        full = mod.asarray(base_data)
        view = full[sl0:, sl1:]
        return mod.power(exponent, view)

    c2d_np, c2d_ionp = _mk_pair(_power_call_probe_2d)

    # 2-D offset view WITH a negative-stride reversal layered on top, no
    # negative values: base[2:, 2:][::-1, ::-1] == [[7,6],[3,2]].
    def _power_call_probe_2d_revstride(mod, base_data, exponent):
        full = mod.asarray(base_data)
        view = full[2:, 2:][::-1, ::-1]
        return mod.power(exponent, view)

    c2drev_np, c2drev_ionp = _mk_pair(_power_call_probe_2d_revstride)

    # Broadcast: a 1-D offset view broadcast against a (2,1) exponent.
    base_bcast = [-3, 2, 3, 4, 5, 6]

    def _power_call_probe_broadcast(mod, base_data, exponent):
        full = mod.asarray(base_data)
        view = full[3:]  # [4,5,6], no negatives
        return mod.power(exponent, view)

    cb_np, cb_ionp = _mk_pair(_power_call_probe_broadcast)

    return {
        "extra_specs": {
            "crossing/ufunc/power_view_reversed_offset": ItemSpec(
                name="crossing/ufunc/power_view_reversed_offset",
                kind="custom",
                numpy_path="power",
                ionp_path="power",
                atol=0.0,
                rtol=0.0,
                numpy_adapter=rev_np,
                ionp_adapter=rev_ionp,
                custom_cases=lambda: [("view/call/reversed_offset_no_neg", (_BASE_1D, 2), {})],
            ),
            "crossing/ufunc/power_view_2d_offset": ItemSpec(
                name="crossing/ufunc/power_view_2d_offset",
                kind="custom",
                numpy_path="power",
                ionp_path="power",
                atol=0.0,
                rtol=0.0,
                numpy_adapter=c2d_np,
                ionp_adapter=c2d_ionp,
                custom_cases=lambda: [("view/call/2d_offset_no_neg", (base_2d, 2, 2, 2), {})],
            ),
            "crossing/ufunc/power_view_2d_offset_negstride": ItemSpec(
                name="crossing/ufunc/power_view_2d_offset_negstride",
                kind="custom",
                numpy_path="power",
                ionp_path="power",
                atol=0.0,
                rtol=0.0,
                numpy_adapter=c2drev_np,
                ionp_adapter=c2drev_ionp,
                custom_cases=lambda: [("view/call/2d_offset_negstride_no_neg", (base_2d, 2), {})],
            ),
            "crossing/ufunc/power_view_broadcast": ItemSpec(
                name="crossing/ufunc/power_view_broadcast",
                kind="custom",
                numpy_path="power",
                ionp_path="power",
                atol=0.0,
                rtol=0.0,
                numpy_adapter=cb_np,
                ionp_adapter=cb_ionp,
                custom_cases=lambda: [
                    ("view/call/broadcast_offset_no_neg", (base_bcast, [[1], [2]]), {}),
                ],
            ),
        },
    }


_extra = _extra_probes_and_specs()


def _outer_cases():
    return [
        # power.outer's own offset-view repro from the task brief:
        # power.outer([2], bf[2:]) -> numpy [[8,16,32]].
        ("view/outer/forward_slice_no_neg", (_BASE_1D, slice(2, None), [2]), {}),
        ("view/outer/strided_has_neg", (_BASE_1D, slice(None, None, 2), [2]), {}),
    ]


def _at_cases():
    return [
        ("view/at/forward_slice_no_neg",
         (_BASE_1D, slice(2, None), [0, 1, 2], [10, 10, 10]), {}),
        ("view/at/strided_has_neg",
         (_BASE_1D, slice(None, None, 2), [0, 1, 2], [10, 10, 10]), {}),
    ]


POWER_VIEW_SPECS: dict[str, ItemSpec] = {
    "crossing/ufunc/power_view_call": ItemSpec(
        name="crossing/ufunc/power_view_call",
        kind="custom",
        numpy_path="power",
        ionp_path="power",
        atol=0.0,
        rtol=0.0,
        numpy_adapter=power_call_np,
        ionp_adapter=power_call_ionp,
        custom_cases=lambda: [
            ("view/call/forward_slice_no_neg", (_BASE_1D, slice(2, None), 2), {}),
            ("view/call/strided_has_neg", (_BASE_1D, slice(None, None, 2), 2), {}),
        ],
    ),
    "crossing/ufunc/power_view_outer": ItemSpec(
        name="crossing/ufunc/power_view_outer",
        kind="custom",
        numpy_path="power",
        ionp_path="power",
        atol=0.0,
        rtol=0.0,
        numpy_adapter=power_outer_np,
        ionp_adapter=power_outer_ionp,
        custom_cases=_outer_cases,
    ),
    "crossing/ufunc/power_view_at": ItemSpec(
        name="crossing/ufunc/power_view_at",
        kind="custom",
        numpy_path="power",
        ionp_path="power",
        atol=0.0,
        rtol=0.0,
        numpy_adapter=power_at_np,
        ionp_adapter=power_at_ionp,
        custom_cases=_at_cases,
    ),
    **_extra["extra_specs"],
}

_power_view_collisions = set(POWER_VIEW_SPECS) & set(registry.REGISTRY)
if _power_view_collisions:
    raise AssertionError(
        f"power_view_cases.py: {sorted(_power_view_collisions)} already "
        f"present in registry.REGISTRY -- refusing to silently overwrite an "
        f"existing item"
    )
registry.REGISTRY.update(POWER_VIEW_SPECS)
