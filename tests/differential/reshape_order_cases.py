"""Ticket #7: `reshape`'s missing identity-shape short-circuit (numpy
`_reshape_with_copy_arg`'s `PyArray_View(array, NULL, NULL)` fast path,
`shape.c` ~237-247) and its `order='A'`/`'K'` resolution
(`resolve_reshape_order`, see `docs/NUMPY-STRIDE-ORDER-SPEC.md` §3a and the
`## CORRECTION -- 2026-08-08` section).

WHY THIS FILE EXISTS. `creation_cases.py`'s pre-existing `reshape`/
`ndarray.reshape` corpus, and `view_semantics_cases.py`'s `_AUGMENT`
descriptor wrapping of both, already compare `.strides` -- but neither
ever calls `.reshape(shape)` with `shape == a.shape` (an identity
reshape), nor sweeps `order='A'`/`'K'` on a non-C/non-F-contiguous base.
That is precisely the gap #7 lived in: `NdArray::reshape` recomputed
strides via `attempt_nocopy_reshape` even when the requested shape never
changed, which numpy's own C never lets happen (it returns a bare view of
the SOURCE before any order-specific algorithm runs at all). Confirmed
live before the fix, `ap.zeros((4,1,2)).copy('F').reshape((4,1,2))
.strides == (8, 64, 32)` vs numpy's `(8, 32, 32)`.

INPUTS ARE BUILT NATIVELY ON EACH SIDE, never ingested cross-library --
same reason `view_semantics_cases.py`/`empty_stride_cases.py` do it:
`mod.zeros(shape)`/`mod.arange(...)` is called against `np`/`anionpy`
independently inside `_probe()`, then reshaped/copied/sliced with each
module's OWN ops, so ingestion's separate, already-documented layout-loss
gap (#69) never enters this corpus -- see `ufunc_order_cases.py`'s module
docstring for that gap's own live repro, deliberately not re-litigated
here.

Compares a DESCRIPTOR STRING (shape/strides/dtype/C_CONTIGUOUS/
F_CONTIGUOUS/OWNDATA/`base is None`/values), not the raw array, following
`view_semantics_cases.py`'s `ViewProbe` mechanism exactly -- a sentinel
argument recognized by a wrapping numpy_adapter/ionp_adapter pair, any
non-sentinel argument falling straight through to the item's pre-existing
resolved callable so every case that already existed for `reshape`/
`ndarray.reshape` keeps running unchanged. MUST be imported after
`view_semantics_cases` in `run.py` (it wraps `REGISTRY["reshape"]`/
`REGISTRY["ndarray.reshape"]`'s CURRENT adapters, i.e. the ones
`view_semantics_cases.py` already installed, and falls through to those --
not to the raw pre-`view_semantics_cases` callables -- so both layers of
cases keep running).

Grid: dtype x shape x layout x order, per the task brief's explicit
"vary dtype/shape/layout/provenance, not just kwargs" instruction:

- dtypes: float64, float32, int32, complex128, bool (5) -- spans real,
  complex, and single-byte-distinct-from-multi-byte itemsize.
- shapes: 0-d `()`, 1-d `(5,)`, length-1-axis `(4,1,2)`, zero-extent
  `(0,3)`/(2,0,4)`, ordinary 3-d `(2,3,4)`.
- layouts: `c` (plain), `f` (`.copy('F')`), `t` (full transpose/reverse
  axes -- neither C nor F contiguous for ndim>=2 non-square), `neg`
  (`[::-1]` on axis 0 -- negative stride), `step2` (`[::2]` on axis 0 --
  non-unit-step, non-contiguous). Layouts that are structurally
  impossible for a given ndim (e.g. `t`/`neg`/`step2` on a 0-d array) are
  skipped, not faked.
- orders: `'C'`, `'F'`, `'A'` for the identity reshape (`shape == a.shape`,
  the #7 bug's exact trigger) AND for a genuine shape-changing reshape
  (`(n,) -> (n,)` flatten-and-back via a DIFFERENT grouping when possible,
  else skipped) so the fix is checked to not have disturbed the
  non-identity path; `'K'` is checked SEPARATELY as a required-ValueError
  case (every dtype/shape/layout/base), since numpy rejects it
  unconditionally, even for an identity reshape -- see the spec's citation
  that order resolution (including the K-rejection) runs before the
  identity short-circuit.
"""
from __future__ import annotations

import numpy as np

from registry import REGISTRY, ItemSpec

# ---------------------------------------------------------------------------
# Probe inputs -- built natively on each side, never ingested cross-library.
# ---------------------------------------------------------------------------

_DTYPES = ("float64", "float32", "int32", "complex128", "bool")

_SHAPES = {
    "0d": (),
    "1d": (5,),
    "len1ax": (4, 1, 2),
    "zero_a": (0, 3),
    "zero_b": (2, 0, 4),
    "3d": (2, 3, 4),
}

_LAYOUTS = ("c", "f", "t", "neg", "step2")


def _layout_applies(shape, layout):
    ndim = len(shape)
    if layout == "c" or layout == "f":
        return True
    if layout == "t":
        return ndim >= 2
    if layout in ("neg", "step2"):
        # Needs at least one non-empty axis to slice meaningfully.
        return ndim >= 1 and shape[0] not in (0, 1)
    raise AssertionError(layout)


def _build(mod, shape, dtype, layout):
    """Native construction, entirely on `mod`'s own ops."""
    n = 1
    for d in shape:
        n *= d
    base = mod.arange(n, dtype=dtype).reshape(shape) if n > 0 else mod.zeros(shape, dtype=dtype)
    if layout == "c":
        return base
    if layout == "f":
        return base.copy("F")
    if layout == "t":
        return base.T if len(shape) == 2 else mod.transpose(base)
    if layout == "neg":
        return base[::-1]
    if layout == "step2":
        return base[::2]
    raise AssertionError(layout)


class ReshapeOrderProbe:
    """Sentinel: "build (dtype, shape, layout) natively on your own side,
    reshape it to `target_shape` with `order`, and describe what came
    back." `target_shape=None` means "reshape to my own current shape" --
    the exact identity case #7's bug lived in."""

    __slots__ = ("shape_label", "dtype", "layout", "order", "target_shape")

    def __init__(self, shape_label, dtype, layout, order, target_shape=None):
        self.shape_label = shape_label
        self.dtype = dtype
        self.layout = layout
        self.order = order
        self.target_shape = target_shape

    def __repr__(self):
        return (
            f"ReshapeOrderProbe({self.shape_label!r}, {self.dtype!r}, "
            f"{self.layout!r}, order={self.order!r}, target={self.target_shape!r})"
        )


def _flat_values(v):
    out = v.tolist()
    while isinstance(out, list) and out and isinstance(out[0], list):
        out = [x for sub in out for x in sub]
    return out if isinstance(out, list) else [out]


def _descriptor(v):
    return "|".join(
        [
            f"shape={tuple(v.shape)}",
            f"strides={tuple(v.strides)}",
            f"dtype={str(v.dtype)}",
            f"c_contig={bool(v.flags['C_CONTIGUOUS'])}",
            f"f_contig={bool(v.flags['F_CONTIGUOUS'])}",
            f"owndata={bool(v.flags['OWNDATA'])}",
            f"base_is_none={v.base is None}",
            f"vals={_flat_values(v)}",
        ]
    )


def _probe(mod, p: "ReshapeOrderProbe"):
    """Via the free function (`np.reshape`/`anionpy.reshape`) -- exercises
    `creation.rs::reshape`, a distinct entry point from the bound method
    below (`lib.rs::PyArray::reshape`), both independently routed through
    `NdArray::reshape_with_order`."""
    shape = _SHAPES[p.shape_label]
    a = _build(mod, shape, p.dtype, p.layout)
    target = shape if p.target_shape is None else p.target_shape
    try:
        v = mod.reshape(a, target, order=p.order)
    except BaseException as exc:
        return f"raised:{type(exc).__name__}:{exc}"
    return _descriptor(v)


def _method_probe(mod, p: "ReshapeOrderProbe"):
    """Same probe, via `ndarray.reshape` (the bound method) rather than the
    free function -- both are declared items and both route through
    `NdArray::reshape_with_order`, but independently, per `lib.rs`'s own
    `PyArray::reshape` vs `creation.rs::reshape` -- see array.rs's
    `reshape`/`reshape_with_order`, both of which got the #3a fix."""
    shape = _SHAPES[p.shape_label]
    a = _build(mod, shape, p.dtype, p.layout)
    target = shape if p.target_shape is None else p.target_shape
    try:
        v = a.reshape(target, order=p.order)
    except BaseException as exc:
        return f"raised:{type(exc).__name__}:{exc}"
    return _descriptor(v)


# ---------------------------------------------------------------------------
# The grid.
# ---------------------------------------------------------------------------

def _identity_reshape_cases():
    """order in {'C','F','A'} on shape==a.shape -- the #7 bug's own
    trigger. numpy: unchanged view, source strides untouched, regardless
    of which order was requested."""
    out = []
    for shape_label, shape in _SHAPES.items():
        for dtype in _DTYPES:
            for layout in _LAYOUTS:
                if not _layout_applies(shape, layout):
                    continue
                for order in ("C", "F", "A"):
                    out.append(
                        ReshapeOrderProbe(shape_label, dtype, layout, order, target_shape=None)
                    )
    return out


def _k_rejection_cases():
    """order='K' must raise ValueError unconditionally -- including on an
    identity reshape, since numpy resolves/rejects order BEFORE the
    identity short-circuit (spec's `## CORRECTION` section, cited in this
    file's module docstring)."""
    out = []
    for shape_label, shape in _SHAPES.items():
        for dtype in ("float64", "complex128"):
            for layout in _LAYOUTS:
                if not _layout_applies(shape, layout):
                    continue
                out.append(ReshapeOrderProbe(shape_label, dtype, layout, "K", target_shape=None))
    return out


def _nonidentity_order_a_cases():
    """order='A' resolution on a genuine shape CHANGE (not identity), to
    confirm the fix didn't disturb the already-correct §3b nocopy/copy
    dispatch that only the identity short-circuit was supposed to
    preempt. Every non-0d/non-empty shape gets flattened to 1-d and back
    to a different grouping when its size allows a nontrivial regroup."""
    out = []
    _REGROUP = {
        "len1ax": (2, 4),  # (4,1,2) size 8 -> (2,4)
        "zero_b": (4, 0, 6),  # (2,0,4) size 0 -> (4,0,6), still size 0
        "3d": (4, 6),  # (2,3,4) size 24 -> (4,6)
    }
    for shape_label, target in _REGROUP.items():
        shape = _SHAPES[shape_label]
        for dtype in _DTYPES:
            for layout in _LAYOUTS:
                if not _layout_applies(shape, layout):
                    continue
                for order in ("C", "F", "A"):
                    out.append(
                        ReshapeOrderProbe(shape_label, dtype, layout, order, target_shape=target)
                    )
    return out


def _all_probes():
    return _identity_reshape_cases() + _k_rejection_cases() + _nonidentity_order_a_cases()


def _label(p: ReshapeOrderProbe) -> str:
    kind = "identity" if p.target_shape is None else "regroup"
    return f"{kind}/{p.shape_label}/{p.dtype}/{p.layout}/order_{p.order}"


def make_adapters(probe_fn, orig_numpy, orig_ionp):
    def numpy_adapter(arg, *rest, **kwargs):
        if isinstance(arg, ReshapeOrderProbe):
            return probe_fn(np, arg)
        assert orig_numpy is not None
        return orig_numpy(arg, *rest, **kwargs)

    def ionp_adapter(arg, *rest, **kwargs):
        import anionpy

        if isinstance(arg, ReshapeOrderProbe):
            return probe_fn(anionpy, arg)
        assert orig_ionp is not None
        return orig_ionp(arg, *rest, **kwargs)

    return numpy_adapter, ionp_adapter


def _cases(probes):
    return [(f"reshape_order/{_label(p)}", (p,), {}) for p in probes]


def _appender(prev, extra):
    def wrapped():
        base = list(prev()) if prev is not None else []
        return base + extra

    return wrapped


def _install():
    probes = _all_probes()

    for name, probe_fn in (("reshape", _probe), ("ndarray.reshape", _method_probe)):
        spec = REGISTRY[name]
        orig_numpy = spec.resolve_numpy()
        orig_ionp = spec.resolve_ionp()
        np_ad, ionp_ad = make_adapters(probe_fn, orig_numpy, orig_ionp)
        spec.numpy_adapter = np_ad
        spec.ionp_adapter = ionp_ad
        spec.extra_cases = _appender(spec.extra_cases, _cases(probes))


_install()
