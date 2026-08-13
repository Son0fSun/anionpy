"""STRIDE-comparing corpus for the two length-1-axis-insertion CONVENTIONS
numpy uses (see `ionp-core/src/creation.rs`'s `insert_newaxis`/`expand_dims`
doc comments): `expand_dims`, `atleast_2d`, `atleast_3d`, `stack`.

WHY THIS FILE EXISTS. `expand_dims`'s pre-existing differential corpus
(`creation_cases.py::expand_dims_cases`) never compared `.strides`, only
values -- and a length-1 axis has no values of its own to disagree on, so
a wrong stride there is invisible to a values-only comparison no matter
how large the corpus is. That gap is exactly how `expand_dims` ended up
DECLARED EXACT in `anionpy/_state/toplevel.py` while diverging from real
numpy on 22 of 30 probed (shape, order, axis) combinations on NONEMPTY
input (Task #newaxis-split, 2026-08-08): the function had implemented
numpy's stride-0 NEWAXIS-INDEXING rule (`a[None]`) for every caller,
including `expand_dims` itself, when real numpy's `expand_dims` is
`a.reshape(new_shape)` (`_core/shape_base.py`) and gets a DIFFERENT,
reshape-computed stride whenever `a` is non-degenerate.

`atleast_2d`/`atleast_3d` already got a stride-comparing corpus
(`view_semantics_cases.py`, 2026-08-05) that caught and fixed the same
class of bug for THEM specifically -- but its input set (`_KINDS`) has no
explicit F-order (only C-contiguous-then-`.T`, which is F-order only for
2-D) and no zero-extent shape. `stack` never got one at all. This file
covers exactly those gaps for all four items, using the same
sentinel-descriptor mechanism `empty_stride_cases.py` established: inputs
are built NATIVELY on each side (`mod.zeros(...)`, `mod.arange(...)`,
never ingested cross-library -- ingesting a foreign-layout numpy array
through `anionpy.array()` silently re-contiguises it, so a genuinely
F-order or transposed anionpy operand could never be produced that way),
and the returned array is reduced to a shape/strides/dtype/contiguity
descriptor STRING before comparison, so a stride mismatch on an all-same
value (a length-1 axis, or an empty array) is not invisible the way plain
value comparison would make it.

Required by the task brief: C-order, F-order, transposed views,
non-contiguous slices, length-1 axes, zero-extent, and 0-d, each verified
identically constructed on both sides before the op runs (`_LAYOUTS`
below builds every base with the SAME sequence of native ops on both
`np`/`anionpy`, and `_probe` asserts the two bases' own strides agree
before invoking the op under test -- otherwise a mismatch would be
measuring the constructor, not the operation, per the task brief's own
warning about `np.asfortranarray`/`.copy('F')` disagreeing on 0-d and
`arange().reshape()`'s own empty divergence, which is exactly why every
layout below is built from `zeros`/`arange` + `.copy('F')`/slicing/
transpose, never `asfortranarray`).
"""
from __future__ import annotations

import numpy as np

from registry import REGISTRY

# ---------------------------------------------------------------------------
# Layouts. Each entry is a NATIVE-construction recipe: `builder(mod)` calls
# only `mod`'s own zeros/arange/copy/T/slicing, so the same recipe run
# against `np` and against `anionpy` produces independently-constructed,
# equal-strides bases (asserted in `_probe` below) rather than an ingested
# copy of one side's array.
# ---------------------------------------------------------------------------

def _c_2x3(mod):
    return mod.zeros((2, 3))


def _f_2x3(mod):
    return mod.zeros((2, 3)).copy('F')


def _c_len1_4x1x2(mod):
    return mod.zeros((4, 1, 2))


def _f_len1_4x1x2(mod):
    return mod.zeros((4, 1, 2)).copy('F')


def _transposed_2x3(mod):
    # zeros((3,2)).T has shape (2,3) but F-ish (non-C-contiguous) strides,
    # the "genuine transpose view" layout distinct from an explicit F-order
    # allocation (`_f_2x3` above) -- both are non-C-contiguous, but a
    # transpose's strides are the SOURCE array's reversed, not `f_strides`
    # of its own shape (they coincide for a plain 2-D transpose, but the
    # provenance -- and therefore what a naive "is it F-contiguous" branch
    # in the op under test might do differently -- is not, which is why
    # both are kept as separate cases here).
    return mod.zeros((3, 2)).T


def _noncontig_4x5(mod):
    return mod.zeros((4, 10))[:, ::2]


def _zero_0x3(mod):
    return mod.zeros((0, 3))


def _zero_3x0(mod):
    return mod.zeros((3, 0))


def _zero_0d_extent(mod):
    # a genuinely zero-SIZE 3-D shape with a length-1 axis mixed in, to
    # catch a size-0-but-not-every-dim-is-0 case distinct from `_zero_0x3`/
    # `_zero_3x0` (both of which happen to also have every strides-bearing
    # axis size >1 except the zero one).
    return mod.zeros((2, 0, 1))


def _0d(mod):
    return mod.zeros(())


_LAYOUTS = {
    "c_2x3": _c_2x3,
    "f_2x3": _f_2x3,
    "c_len1_4x1x2": _c_len1_4x1x2,
    "f_len1_4x1x2": _f_len1_4x1x2,
    "transposed_2x3": _transposed_2x3,
    "noncontig_4x5": _noncontig_4x5,
    "zero_0x3": _zero_0x3,
    "zero_3x0": _zero_3x0,
    "zero_2x0x1": _zero_0d_extent,
    "0d": _0d,
}


def _descriptor(v):
    return "|".join([
        f"shape={tuple(v.shape)}",
        f"strides={tuple(v.strides)}",
        f"dtype={v.dtype}",
        f"c_contig={bool(v.flags['C_CONTIGUOUS'])}",
        f"f_contig={bool(v.flags['F_CONTIGUOUS'])}",
    ])


class LayoutProbe:
    """Sentinel: "build layout L natively via `mod`, run the op, describe
    what came back." `extra` carries per-case knobs (axis, etc)."""

    __slots__ = ("layout_label", "extra")

    def __init__(self, layout_label, **extra):
        self.layout_label = layout_label
        self.extra = extra

    def __repr__(self):
        return f"LayoutProbe({self.layout_label!r}, {self.extra!r})"


def _base(mod, label):
    return _LAYOUTS[label](mod)


def _probe(mod, other_mod, p, op):
    a = _base(mod, p.layout_label)
    a_other = _base(other_mod, p.layout_label)
    # Constructor sanity check (task brief requirement): the two
    # independently-built bases must themselves already agree on strides,
    # or a mismatch below would be measuring the constructor, not the op.
    assert tuple(a.strides) == tuple(a_other.strides), (
        f"base construction diverged for {p.layout_label!r} before the op "
        f"even ran: {tuple(a.strides)} vs {tuple(a_other.strides)}"
    )
    try:
        v = op(mod, a, p)
    except BaseException as exc:
        return f"raised:{type(exc).__name__}:{exc}"
    return _descriptor(v)


def make_adapters(op, orig_numpy=None, orig_ionp=None):
    def numpy_adapter(arg, *rest, **kwargs):
        if isinstance(arg, LayoutProbe):
            import anionpy
            return _probe(np, anionpy, arg, op)
        assert orig_numpy is not None, "non-probe case on a probe-only item"
        return orig_numpy(arg, *rest, **kwargs)

    def ionp_adapter(arg, *rest, **kwargs):
        if isinstance(arg, LayoutProbe):
            import anionpy
            return _probe(anionpy, np, arg, op)
        assert orig_ionp is not None, "non-probe case on a probe-only item"
        return orig_ionp(arg, *rest, **kwargs)

    return numpy_adapter, ionp_adapter


def _cases(probes):
    return [(f"newaxis_strides/{p!r}", (p,), {}) for p in probes]


# ---------------------------------------------------------------------------
# The four ops.
# ---------------------------------------------------------------------------

def _op_expand_dims(mod, a, p):
    return mod.expand_dims(a, p.extra["axis"])


def _op_atleast_2d(mod, a, p):
    return mod.atleast_2d(a)


def _op_atleast_3d(mod, a, p):
    return mod.atleast_3d(a)


def _op_stack(mod, a, p):
    # A second, independently-built same-layout operand -- `stack`
    # requires identical shapes, and reusing `a` twice would not exercise
    # anything `expand_dims` on a single array doesn't already.
    b = _base(mod, p.layout_label)
    return mod.stack([a, b], axis=p.extra["axis"])


def _expand_dims_probes():
    out = []
    for lbl in _LAYOUTS:
        ndim = len(_base(np, lbl).shape)
        for axis in sorted({0, ndim, -1, -(ndim + 1)}):
            out.append(LayoutProbe(lbl, axis=axis))
    return out


def _atleast_probes():
    return [LayoutProbe(lbl) for lbl in _LAYOUTS]


def _stack_probes():
    out = []
    for lbl in _LAYOUTS:
        ndim = len(_base(np, lbl).shape)
        for axis in range(ndim + 1):
            out.append(LayoutProbe(lbl, axis=axis))
    return out


_ITEMS = {
    "expand_dims": (_op_expand_dims, _expand_dims_probes),
    "atleast_2d": (_op_atleast_2d, _atleast_probes),
    "atleast_3d": (_op_atleast_3d, _atleast_probes),
    "stack": (_op_stack, _stack_probes),
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
