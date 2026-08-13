"""NEW test-cases file (owned by the ufunc `order=` task): rigorous,
STRIDES-CHECKED differential coverage for the `order=` kwarg on every
applicable `kind="ufunc"` item's plain `__call__` form, crossing order in
{'C','F','A','K'} x {unary, binary-same-layout, binary-mixed-layout,
binary-broadcast} x {2-D, 3-D shapes}.

WHY THIS IS A SEPARATE FILE/NAMESPACE, NOT MORE ENTRIES ON THE REAL
"add"/"negative"/etc. ITEMS IN ufunc_registry.py
---------------------------------------------------------------------------
Two independent reasons, both following the exact precedent order_cases.py
already set for ravel/copy/astype/_like's own order= coverage (see that
file's module docstring for the original reasoning this one mirrors):

1. Registry-key collision rules: the tail-merge pattern this file uses
   (like order_cases.py's) refuses to redeclare an existing name, and
   "add"/"negative"/... are already declared by ufunc_registry.py.

2. STRIDES CHECKING NEEDS TO BE OPT-IN AND NARROWLY SCOPED. The task's
   correctness bar for order= requires comparing the OUTPUT's `.strides`,
   not just values/dtype/shape (see `registry.ItemSpec.check_strides`,
   added by this same task). Turning that check on for the REAL "add" /
   "negative" / ... items would apply it to their ENTIRE existing corpus
   (corpus.py's full unary/binary sweep, including `_views()`-derived
   sliced/transposed operands) -- and doing so immediately surfaces a
   confirmed, PRE-EXISTING, OUT-OF-SCOPE bug: `ndarray_from_numpy`
   (ionp-py/src/lib.rs) only special-cases a source that is "genuinely
   F-contiguous, not also C" (see that function's own doc comment); any
   OTHER non-C/non-F layout (e.g. a 3-axis `np.transpose`, or a
   `np.swapaxes` on a non-square 3-D array) silently collapses to a fresh
   C-contiguous copy on ingestion. Values still come out correct (the
   gather is still logically row-major-correct), but the resulting
   anionpy.ndarray's OWN strides no longer match the numpy source's -- so
   ANY downstream order='K' call (or, for that matter, this task's honest
   default-order fix) can only ever reproduce what anionpy itself believes
   its input's layout is, not the numpy source's true layout. Reproduction
   (verified live, not guessed):

       base = np.arange(60, dtype=np.int32).reshape(5, 3, 4)
       a = np.transpose(base, (2, 0, 1))       # numpy strides (4, 48, 16)
       anionpy.array(a).strides                    # -> (60, 12, 4), i.e. C order

   vs. building the SAME transposed view via anionpy's own `.transpose()` on
   an already-ingested array (no re-ingestion of a pre-transposed numpy
   array involved) reproduces numpy's strides exactly -- proving the K-order
   *ufunc* logic this task implemented is correct, and the defect is
   entirely upstream, in ingestion. This is a real, reportable finding
   (see this task's final report) but is NOT `Ufunc.__call__`, so it is
   explicitly out of scope to fix here per the task brief.

   Consequence for this file's corpus: operand layouts below are
   DELIBERATELY restricted to 'c' (plain C, the trivial ingestion case)
   and 'f' (the one non-C layout `ndarray_from_numpy` already special-cases
   correctly, per its own doc comment and the F-vs-C fix note in
   order_cases.py) -- i.e. exactly the two layouts ingestion is ALREADY
   proven to preserve. This keeps `check_strides=True` here from
   conflating "order= is broken" with "ingestion silently re-lays-out my
   test input before order= ever sees it", while still rigorously proving
   order='K' vs order='A' (their divergence needs nothing more than a
   mixed C/F operand pair -- see the module docstring reasoning in
   ufunc_cases.py's `_order_call_cases`, which this file's shape/layout
   matrix mirrors) and order='C'/'F' (unconditional target layouts,
   trivially insensitive to the ingestion gap either way).

A key not present in tools/numpy_surface.json contributes zero to the
coverage ledger (coverage.py only ever looks up the fixed surface list, see
build_ledger) -- these items exist purely as rigorous, strides-checked
evidence for this task's report and per-item declarations, not to inflate
the percentage.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
import registry
import ufunc_introspect
from registry import ItemSpec

ORDERS = ["C", "F", "A", "K"]
# `(4, 1, 2)` added 2026-08-08 (ticket #7): a length-1 axis is the
# discriminator for the defect this file's `check_strides=True` corpus
# exists to catch (see docs/TICKET-64-REMEASURE-2026-08-08.md's FOURTH
# CORRECTION and docs/TICKET-7-2026-08-08.md). Neither of the original two
# shapes has a size-1 axis, so this whole file's `order/ufunc/*` corpus
# was previously blind to the exact repro table
# (`np.abs(zeros((4,1,2)).copy('F'))` losing F strides) that #7 measured
# 118 items against.
#
# CORRECTION (2026-08-08, same day): an earlier version of this comment
# claimed reverting the fix "turns order/ufunc/abs (and ~20 others) from
# PASS to FAIL" -- that claim was never actually run to completion and is
# FALSE. Measured directly: reverting `apply_ufunc_order`'s `'K'` arm
# (`ionp-py/src/lib.rs`) to its pre-fix form, rebuilding, and rerunning the
# full differential suite with this shape present reproduces the EXACT
# SAME 29-item baseline failure set, byte-for-byte, with ZERO new
# failures. Root cause: `harness.py::_strides_match` deliberately masks
# stride comparisons at any axis position whose extent is <= 1 ("CLASS B"
# divergence, documented there, added 2026-08-04) -- and every one of
# this ticket's repro shapes diverges EXACTLY at that masked position.
# This shape stays in SHAPES because it still adds real, non-masked
# coverage at every OTHER axis position across this file's full
# order/layout/ufunc matrix, but it is NOT a regression guard for ticket
# #7's own defect. The actual, PROVEN-to-bite regression guard for that
# defect is a set of Rust `#[test]`s in `ionp-core/src/array.rs` (see
# `k_order_unary_f_contiguous_len1_*`) -- proven by reverting their
# `k_order_perm` fast path in isolation, rebuilding, observing the
# expected FAILs, then restoring byte-identical and reconfirming PASS.
SHAPES = [(2, 3), (2, 2, 3), (4, 1, 2)]


def _reshape_like(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    n = 1
    for d in shape:
        n *= d
    flat = np.asarray(arr).reshape(-1)
    if flat.size == 0:
        return np.zeros(shape, dtype=arr.dtype)
    reps = -(-n // flat.size)
    return np.tile(flat, reps)[:n].reshape(shape)


def _cf(arr: np.ndarray, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """'c' and 'f' layouts ONLY -- see module docstring for why 'swapped'
    (a genuine partial-transpose, neither-C-nor-F layout) is excluded here:
    it cannot be honestly round-tripped through `anionpy.array()` today."""
    base = _reshape_like(arr, shape)
    return {"c": np.ascontiguousarray(base), "f": np.asfortranarray(base)}


def _cases_for_ufunc(name: str) -> list[tuple]:
    info = ufunc_introspect.describe(name)
    if info.signature is not None or info.is_string or info.nout != 1:
        return []
    sample = info.typed_sample
    if sample is None:
        return []

    cases: list[tuple] = []

    if info.nin == 1:
        for shape in SHAPES:
            for layout_name, arr in _cf(sample[0], shape).items():
                for order in ORDERS:
                    cases.append((
                        f"{name}/unary_{layout_name}/{shape}/order_{order}",
                        (arr,), {"order": order},
                    ))
    elif info.nin == 2:
        a_arr, b_arr = sample[0], sample[1]
        for shape in SHAPES:
            a_layouts = _cf(a_arr, shape)
            b_layouts = _cf(b_arr, shape)
            for layout_name in ("c", "f"):
                a, b = a_layouts[layout_name], b_layouts[layout_name]
                for order in ORDERS:
                    cases.append((
                        f"{name}/binary_same_{layout_name}/{shape}/order_{order}",
                        (a, b), {"order": order},
                    ))
            for an, bn in (("f", "c"), ("c", "f")):
                a, b = a_layouts[an], b_layouts[bn]
                for order in ORDERS:
                    cases.append((
                        f"{name}/binary_mixed_{an}_{bn}/{shape}/order_{order}",
                        (a, b), {"order": order},
                    ))
            b_broadcast = _cf(b_arr, shape[1:])
            for layout_name in ("c", "f"):
                a, b = a_layouts[layout_name], b_broadcast[layout_name]
                for order in ORDERS:
                    cases.append((
                        f"{name}/binary_broadcast_{layout_name}/{shape}/order_{order}",
                        (a, b), {"order": order},
                    ))
    return cases


def _build_order_specs() -> dict[str, ItemSpec]:
    import ufunc_registry  # local import: avoid a module-load-order cycle
    specs: dict[str, ItemSpec] = {}
    for name in ufunc_introspect.UFUNC_NAMES:
        base_spec = ufunc_registry.UFUNC_SPECS.get(name)
        if base_spec is None:
            continue
        cases = _cases_for_ufunc(name)
        if not cases:
            continue
        key = f"order/ufunc/{name}"
        specs[key] = ItemSpec(
            name=key,
            kind="custom",
            numpy_path=name,
            ionp_path=name,
            atol=base_spec.atol,
            rtol=base_spec.rtol,
            ulp_tolerance=base_spec.ulp_tolerance,
            ulp_justification=base_spec.ulp_justification,
            ulp_sweep=base_spec.ulp_sweep,
            custom_cases=(lambda n=name: _cases_for_ufunc(n)),
            check_strides=True,
        )
    return specs


ORDER_UFUNC_SPECS: dict[str, ItemSpec] = _build_order_specs()

registry.REGISTRY.update(ORDER_UFUNC_SPECS)
