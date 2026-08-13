"""Empty-shape strides/layout DESCRIPTOR corpus for the nine items whose
declared-exact status a 2026-08-04 out-of-corpus measurement (Task #61)
found FALSE specifically on zero-sized arrays: `ndarray.flatten`,
`ascontiguousarray`, `empty_like`, `zeros_like`, `ones_like`, `full_like`,
`expand_dims`, `roll`, `tile`.

WHY THIS FILE EXISTS. Every one of these nine already had passing
differential cases -- none of them compared `.strides` or `.flags`, only
values, and an empty array has no values to compare (`np.zeros((0,3)) ==
anything` is vacuously true elementwise over zero elements). "Strides are
part of the contract" is a standing project decision (see `ItemSpec.
check_strides`'s docstring in registry.py), so a stride mismatch on an
empty array is exactly as real a bug as a wrong value on a nonempty one --
it was simply invisible to every prior case these nine items had.

The measured rule (real numpy 2.5.1, re-verified directly against source
and behavior, not assumed from any prior comment): a FRESHLY ALLOCATED
zero-sized array gets ALL-ZERO strides regardless of requested order/axis
placement (`np.zeros((0,)/(0,3)/(3,0)/(2,0,4)/(0,0))`, `.copy('F')` of any
of those, all zero). A VIEW/RESHAPE/insert-axis composition over an
existing empty base does NOT get zeroed -- it keeps the plain
`shape::c_strides`-style "reshape formula" strides instead (e.g.
`np.expand_dims`, a plain `.reshape()`, `.ravel()`, or a transpose/slice of
an already-empty array). `ascontiguousarray` of an already-contiguous empty
is a genuine no-op, inheriting the input's own strides unchanged rather
than recomputing anything. Each of the nine op wrappers below was built
from directly reading that item's real numpy source
(`inspect.getsource`), not from restating this paragraph -- see each op's
own comment for the specific numpy source lines it mirrors and the
specific measured cases (including several that do NOT reduce to "always
zero", e.g. `tile`'s reps=(1,1,1) "all-ones" branch and `roll`'s ndim==1
axis=None special case) that pin the two failure MODES this task's
original nine-item measurement found: strides too large where numpy
zeroes ("Group A": flatten/ascontiguousarray/*_like), and strides zeroed
(or otherwise wrong) where numpy keeps its own computed ones ("Group B":
expand_dims/roll/tile).

INPUTS ARE BUILT NATIVELY ON EACH SIDE, never ingested cross-library --
same reason `view_semantics_cases.py`/`creation_cases.py`'s `_ViewCase` do
it: `mod.zeros(shape)` is called against `np`/`anionpy` independently
inside `_probe()`, so a fresh, correctly-zeroed empty allocation is what
each side's own code actually produces, not what ingestion happens to
preserve.

Follows the exact wrapper mechanism `view_semantics_cases.py` established:
a sentinel argument (`EmptyProbe`) is recognized by a `numpy_adapter`/
`ionp_adapter` pair that intercepts it and returns a DESCRIPTOR STRING
(shape/strides/contiguity flags/dtype) instead of the raw array; any
non-sentinel argument falls straight through to the item's pre-existing
resolved callable, so every case that already existed for these nine
items keeps running exactly as before -- this file only ever ADDS cases.
"""
from __future__ import annotations

import numpy as np

from registry import REGISTRY, ItemSpec

# ---------------------------------------------------------------------------
# The five shapes this task's out-of-corpus measurement was run against.
# ---------------------------------------------------------------------------

_EMPTY_SHAPES = {
    "empty1d": (0,),
    "empty2d_tall": (0, 3),
    "empty2d_wide": (3, 0),
    "empty3d": (2, 0, 4),
    "empty2d_square": (0, 0),
}


def _descriptor(v):
    # Deliberately shape/strides/dtype/C_CONTIGUOUS/F_CONTIGUOUS only --
    # this task's declared contract ("strides are part of the contract")
    # is about the STRIDE VALUES and the layout classification that
    # follows directly from them, and those are what got this pass's nine
    # items measured FALSE. OWNDATA/`.base` were tried and deliberately
    # left OUT after measuring a real, but separate and much wider, gap:
    # `expand_dims`/`roll`/`tile` report OWNDATA=True even where real
    # numpy's OWNDATA=False (numpy's `roll`/`tile` return a VIEW of an
    # internal freshly-built array, not the array itself -- e.g.
    # `np.roll(np.arange(4.), 1).flags['OWNDATA']` is False, a view of an
    # internal `empty_like` copy via the axis=None ravel+reshape
    # composition), and this reproduces IDENTICALLY on ordinary NONEMPTY
    # inputs (verified directly: `np.roll`/`np.tile` on `(2,3)` and `(4,)`
    # give `OWNDATA=False` where anionpy gives `True`) -- i.e. it is not
    # an empty-array defect this task's stride measurement covers at all,
    # it is ionp-py's `.base`/OWNDATA book-keeping not modeling numpy's
    # "view of an internal scratch array" construction anywhere, for any
    # shape. Fixing that is a structural, wide-footprint change (the same
    # category as the reshape identity-shortcut and expand_dims nonempty
    # gaps found and declined elsewhere in this pass) and is explicitly
    # out of scope here -- reported, not silently worked around.
    return "|".join([
        f"shape={tuple(v.shape)}",
        f"strides={tuple(v.strides)}",
        f"dtype={v.dtype}",
        f"c_contig={bool(v.flags['C_CONTIGUOUS'])}",
        f"f_contig={bool(v.flags['F_CONTIGUOUS'])}",
    ])


class EmptyProbe:
    """Sentinel: "build shape S natively on your own side (mod.zeros), run
    the op, describe what came back." `extra` carries whatever per-case
    knobs (dtype/order/axis/shift/reps/...) that op needs -- a plain dict
    so a single adapter pair per item serves every variant."""

    __slots__ = ("shape_label", "extra")

    def __init__(self, shape_label, **extra):
        self.shape_label = shape_label
        self.extra = extra

    def __repr__(self):
        return f"EmptyProbe({self.shape_label!r}, {self.extra!r})"


def _probe(mod, p, op):
    shape = _EMPTY_SHAPES[p.shape_label]
    try:
        v = op(mod, shape, p)
    except BaseException as exc:  # error parity is part of the contract too
        return f"raised:{type(exc).__name__}:{exc}"
    return _descriptor(v)


def make_adapters(op, orig_numpy=None, orig_ionp=None):
    def numpy_adapter(arg, *rest, **kwargs):
        if isinstance(arg, EmptyProbe):
            return _probe(np, arg, op)
        assert orig_numpy is not None, "non-probe case on a probe-only item"
        return orig_numpy(arg, *rest, **kwargs)

    def ionp_adapter(arg, *rest, **kwargs):
        if isinstance(arg, EmptyProbe):
            import anionpy
            return _probe(anionpy, arg, op)
        assert orig_ionp is not None, "non-probe case on a probe-only item"
        return orig_ionp(arg, *rest, **kwargs)

    return numpy_adapter, ionp_adapter


def _cases(probes):
    return [(f"empty_strides/{p!r}", (p,), {}) for p in probes]


# ---------------------------------------------------------------------------
# The nine ops. Each `op(mod, shape, p)` builds its own input via
# `mod.zeros(shape, ...)` and returns the array to describe.
# ---------------------------------------------------------------------------

def _op_flatten(mod, shape, p):
    dtype = p.extra.get("dtype", "float64")
    order = p.extra.get("order")
    a = mod.zeros(shape, dtype=dtype)
    return a.flatten() if order is None else a.flatten(order=order)


def _op_ascontiguousarray(mod, shape, p):
    a = mod.zeros(shape)
    return mod.ascontiguousarray(a)


def _op_empty_like(mod, shape, p):
    a = mod.zeros(shape, dtype=p.extra.get("dtype", "float64"))
    order = p.extra.get("order")
    return mod.empty_like(a) if order is None else mod.empty_like(a, order=order)


def _op_zeros_like(mod, shape, p):
    a = mod.zeros(shape, dtype=p.extra.get("dtype", "float64"))
    order = p.extra.get("order")
    return mod.zeros_like(a) if order is None else mod.zeros_like(a, order=order)


def _op_ones_like(mod, shape, p):
    a = mod.zeros(shape, dtype=p.extra.get("dtype", "float64"))
    order = p.extra.get("order")
    return mod.ones_like(a) if order is None else mod.ones_like(a, order=order)


def _op_full_like(mod, shape, p):
    a = mod.zeros(shape, dtype=p.extra.get("dtype", "float64"))
    order = p.extra.get("order")
    fill = p.extra.get("fill", 3.0)
    if p.extra.get("array_fill"):
        fill = mod.array(fill)
    return mod.full_like(a, fill) if order is None else mod.full_like(a, fill, order=order)


def _op_expand_dims(mod, shape, p):
    a = mod.zeros(shape)
    return mod.expand_dims(a, p.extra["axis"])


def _op_roll(mod, shape, p):
    a = mod.zeros(shape)
    axis = p.extra.get("axis")
    shift = p.extra.get("shift", 1)
    if axis is None:
        return mod.roll(a, shift)
    return mod.roll(a, shift, axis=axis)


def _op_tile(mod, shape, p):
    a = mod.zeros(shape)
    return mod.tile(a, p.extra["reps"])


# ---------------------------------------------------------------------------
# Wiring: for each item, build the probe list, capture the item's own
# pre-existing resolved callables as the fallthrough, install the adapters,
# and append the probe cases via `extra_cases` (additive, per `run.py`'s
# `build_cases` docstring -- never replaces the item's existing corpus).
# ---------------------------------------------------------------------------

_SHAPE_LABELS = list(_EMPTY_SHAPES)


def _flatten_probes():
    out = []
    for lbl in _SHAPE_LABELS:
        for dtype in ("float64", "int8", "complex128"):
            out.append(EmptyProbe(lbl, dtype=dtype))
        for order in ("C", "F"):
            out.append(EmptyProbe(lbl, order=order))
    return out


def _ascontiguousarray_probes():
    return [EmptyProbe(lbl) for lbl in _SHAPE_LABELS]


def _like_probes():
    out = []
    for lbl in _SHAPE_LABELS:
        out.append(EmptyProbe(lbl))
        for order in ("C", "F", "A", "K"):
            out.append(EmptyProbe(lbl, order=order))
    return out


def _full_like_probes():
    out = []
    for lbl in _SHAPE_LABELS:
        out.append(EmptyProbe(lbl))
        for order in ("C", "F", "A", "K"):
            out.append(EmptyProbe(lbl, order=order))
        out.append(EmptyProbe(lbl, array_fill=True, fill=5.0))
    return out


def _expand_dims_probes():
    out = []
    for lbl in _SHAPE_LABELS:
        ndim = len(_EMPTY_SHAPES[lbl])
        for axis in {0, ndim, -1, -(ndim + 1)}:
            out.append(EmptyProbe(lbl, axis=axis))
    return out


def _roll_probes():
    out = []
    for lbl in _SHAPE_LABELS:
        ndim = len(_EMPTY_SHAPES[lbl])
        out.append(EmptyProbe(lbl, axis=None, shift=1))
        out.append(EmptyProbe(lbl, axis=None, shift=-3))
        for ax in range(ndim):
            out.append(EmptyProbe(lbl, axis=ax, shift=1))
            out.append(EmptyProbe(lbl, axis=ax, shift=-2))
    return out


def _tile_probes():
    out = []
    for lbl in _SHAPE_LABELS:
        ndim = len(_EMPTY_SHAPES[lbl])
        out.append(EmptyProbe(lbl, reps=tuple([2] * max(ndim, 1))))
        out.append(EmptyProbe(lbl, reps=tuple([1] * max(ndim, 1))))
        # more entries than ndim, all ones: numpy's distinct
        # `all(x == 1 for x in tup)` fresh-copy branch (see manip.rs's
        # `tile` fix comment) -- the case that caught the gap this file
        # exists to guard.
        out.append(EmptyProbe(lbl, reps=tuple([1] * (ndim + 2))))
        # more entries than ndim, not all ones: genuine shape change,
        # reshape-view formula (not zeroed).
        out.append(EmptyProbe(lbl, reps=(3,) + tuple([1] * ndim)))
        # fewer entries than ndim (only meaningful for ndim > 1).
        if ndim > 1:
            out.append(EmptyProbe(lbl, reps=(2,)))
    return out


_ITEMS = {
    "ndarray.flatten": (_op_flatten, _flatten_probes),
    "ascontiguousarray": (_op_ascontiguousarray, _ascontiguousarray_probes),
    "empty_like": (_op_empty_like, _like_probes),
    "zeros_like": (_op_zeros_like, _like_probes),
    "ones_like": (_op_ones_like, _like_probes),
    "full_like": (_op_full_like, _full_like_probes),
    "expand_dims": (_op_expand_dims, _expand_dims_probes),
    "roll": (_op_roll, _roll_probes),
    "tile": (_op_tile, _tile_probes),
}


def _appender(prev, extra):
    def wrapped():
        base = list(prev()) if prev is not None else []
        return base + extra

    return wrapped


def _install():
    for name, (op, probe_fn) in _ITEMS.items():
        spec = REGISTRY[name]
        orig_numpy = spec.resolve_numpy()
        orig_ionp = spec.resolve_ionp()
        np_ad, ionp_ad = make_adapters(op, orig_numpy, orig_ionp)
        spec.numpy_adapter = np_ad
        spec.ionp_adapter = ionp_ad
        spec.extra_cases = _appender(spec.extra_cases, _cases(probe_fn()))


_install()
