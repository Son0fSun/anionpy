"""View/identity semantics for the shape ops: `.base`, `OWNDATA`, `is`,
and write-through aliasing.

WHY THIS FILE EXISTS. Fourteen shape items (`ravel`, `flip`, `fliplr`,
`flipud`, `atleast_1d`, `broadcast_arrays`, `matrix_transpose`,
`linalg.matrix_transpose`, and the `ndarray.mT/ravel/reshape/squeeze/
swapaxes/transpose` methods) all had passing differential tests and were
still not declared, because every existing case compares VALUES ONLY. A
shape op that returns a fresh owning copy where numpy returns a view
produces byte-identical values forever and is still wrong: the difference
shows up as `v.base is not a`, as `OWNDATA=True`, and -- since
`ndarray.__setitem__` exists -- as a write through `v` that never reaches
`a`. That last one is not metadata. It is a wrong VALUE, arrived at one
statement later than the tests were looking.

So the instrument here does not compare the returned array. It compares a
DESCRIPTOR of the returned array, built identically on both sides:

    shape | strides | base_is_input | base_is_none | is_input | owndata |
    writeable | writethru | vals

`is_input` is load-bearing on its own. Measured on numpy 2.5.1, three of
these ops hand back the ARGUMENT OBJECT ITSELF rather than a view of it
when they have nothing to do -- `np.squeeze(a) is a`, `np.atleast_1d(a)
is a` (ndim >= 1), `np.broadcast_arrays(a)[0] is a` -- while
`a.reshape(a.shape) is a` and `a.transpose() is a` are both False even
though those are equally no-ops. There is no principle to derive that
from; it was measured, and it is checked here rather than reasoned about.

INPUTS ARE DATA-OWNING. Every probe array is built with
`mod.array(<nested list>)`, never `mod.arange(n).reshape(shape)`. Both
numpy and anionpy collapse `.base` to the ROOT owner, so if `a` is itself a
view then `v.base is a` is False on BOTH sides for reasons that have
nothing to do with the op under test -- the probe then reports agreement
it never actually tested. An earlier version of this measurement did
exactly that and produced a self-contradictory reading (`mT` agreeing on
`base_is_a` while disagreeing on `OWNDATA`), which is what caught it.

The non-contiguous inputs (`strided`, `transposed`, `2d_T`) are built by
applying anionpy's / numpy's OWN slicing and transpose to a data-owning
array on each side independently, for the same reason
`creation_cases.py`'s `_ViewCase` does it: ingesting an already-strided
numpy array through `anionpy.array()` silently re-contiguises it, so a
foreign view can never reach the code path being tested. For those three
kinds `base_is_input`/`is_input` are compared against the intermediate
view, not the root, and `base_is_none` carries the root-collapse fact.

`ravel` is swept over all four order letters because its view-vs-copy
answer depends on the pair (input layout, order) and on nothing else:

    order:            C      F      A      K
    C-contiguous:   view   copy   view   view
    F-contiguous:   copy   view   view   view
    neither:        copy   copy   copy   copy

A single order would have graded the C-contiguous row and called it a
sweep.

`atleast_2d`/`atleast_3d`: COVERED as of 2026-08-05 (Monday). They used to
be deliberately excluded here -- their trailing/leading length-1-axis
STRIDE convention diverged from numpy (`atleast_3d` on a 2-D transposed
input: numpy gives a 0 stride where anionpy gave an itemsize-derived one,
via a `reshape` that recomputed a fresh C-contiguous layout instead of
inserting a stride-0 newaxis). Fixed in `ionp-core/src/manip.rs` by
routing the ndim==1/ndim==2 branches through `creation::expand_dims`
(the same newaxis-insertion idiom `stats.rs` already used), leaving only
the true ndim==0 case as an actual reshape. Now included in `_AUGMENT`
below like every other view-contract item; the `strides` field this
docstring used to warn against skipping is exercised for real.
"""
from __future__ import annotations

import numpy as np

from registry import REGISTRY, ItemSpec


# ---------------------------------------------------------------------------
# Probe inputs -- built natively on each side, never ingested cross-library.
# ---------------------------------------------------------------------------

def _nested(shape):
    return np.arange(int(np.prod(shape)), dtype="float64").reshape(shape).tolist()


# (kind label) -> (nested list for the OWNING array, op producing the probe
# input from it). `None` means "the owning array is the probe input".
_KINDS = {
    "0d": (3.0, None),
    "1d": (_nested((4,)), None),
    "2d": (_nested((3, 4)), None),
    "3d": (_nested((2, 3, 4)), None),
    "1x3": (_nested((1, 3)), None),
    "3x1": (_nested((3, 1)), None),
    "1x1": (_nested((1, 1)), None),
    "2x1x3": (_nested((2, 1, 3)), None),
    # 1-D NON-CONTIGUOUS. These two exist because of a real defect they
    # caught: `ravel`'s view-vs-copy decision was implemented as "did
    # `reshape_with_order` succeed?", and reshaping a strided 1-D array to
    # its own 1-D shape is a trivially successful no-op restride -- so
    # every 1-D input got a strided VIEW where numpy copies. Every kind
    # above is either contiguous or at least 2-D, so none of them could
    # see it.
    "1d_step": (_nested((6,)), "step"),
    "1d_rev": (_nested((6,)), "rev"),
    "transposed": (_nested((2, 3, 4)), "T"),
    "strided": (_nested((2, 3, 4)), "slice"),
    "2d_T": (_nested((3, 4)), "T"),
}


def _build(mod, kind):
    """Returns (probe_input, owning_root). They are the same object for the
    contiguous kinds and differ for the three view kinds."""
    payload, op = _KINDS[kind]
    root = mod.array(payload, dtype="float64")
    if op is None:
        return root, root
    if op == "T":
        return root.T, root
    if op == "slice":
        return root[:, ::2], root
    if op == "step":
        return root[::2], root
    if op == "rev":
        return root[::-1], root
    raise AssertionError(op)


class ViewProbe:
    """Sentinel argument: "build kind K natively on your own side, apply the
    op, and describe what came back." Carried through the adapters below the
    same way `creation_cases.py`'s `_ViewCase` is. `order` is carried on the
    probe rather than passed as a kwarg so that a single adapter pair per
    item serves the whole order sweep -- the harness calls the adapter with
    the case's args verbatim, and an `order=` kwarg would have to be
    forwarded to the fallthrough path too, where the item's own existing
    cases already use it for something else."""

    __slots__ = ("kind", "order")

    def __init__(self, kind, order=None):
        self.kind = kind
        self.order = order

    def __repr__(self):
        return f"ViewProbe({self.kind!r}, order={self.order!r})"


def _flat_values(v):
    """Values of a result, as a plain Python list, without asking numpy to
    interpret an anionpy array. `tolist()` exists on both sides and is already
    a declared item; going through `np.asarray(ionp_result)` here would
    route the comparison through the ingestion boundary rather than the op."""
    out = v.tolist()
    while isinstance(out, list) and out and isinstance(out[0], list):
        out = [x for sub in out for x in sub]
    return out if isinstance(out, list) else [out]


def _describe(v, probe_input, root):
    """The compared output. A string, so the harness's scalar_like path
    grades it by exact equality -- any single field diverging is a FAIL with
    both descriptors printed side by side, which is what makes the failure
    readable without a debugger."""
    fields = [
        f"shape={tuple(v.shape)}",
        f"strides={tuple(v.strides)}",
        f"ndim={v.ndim}",
        f"is_input={v is probe_input}",
        f"base_is_input={v.base is probe_input}",
        f"base_is_root={v.base is root}",
        f"base_is_none={v.base is None}",
        f"owndata={bool(v.flags['OWNDATA'])}",
        # The BASE's own shape and OWNDATA, not just whether a base exists.
        # numpy's must-copy `reshape` returns a view of a hidden
        # intermediate whose shape is the INPUT's -- without these two
        # fields an implementation could satisfy `base is not None` by
        # pointing at any array at all and still be wrong.
        f"base_shape={None if v.base is None else tuple(v.base.shape)}",
        f"base_owndata={None if v.base is None else bool(v.base.flags['OWNDATA'])}",
        f"writeable={bool(v.flags['WRITEABLE'])}",
    ]
    # Write-through. The single most direct statement of "is this a view":
    # mutate the result, then read the ROOT owner back. A copy leaves the
    # root untouched; a view does not. Assigning into a read-only result
    # (broadcast_to's, if it ever reaches here) raises, and the exception
    # type is recorded rather than swallowed.
    try:
        before = _flat_values(root)
        v[(0,) * v.ndim] = -7.5
        after = _flat_values(root)
        fields.append(f"writethru={before != after}")
    except BaseException as exc:  # PanicException is not an Exception
        fields.append(f"writethru=raised:{type(exc).__name__}")
    fields.append(f"vals={_flat_values(v)}")
    return "|".join(fields)


def _probe(mod, probe, op, multi=False):
    """`multi` wraps the descriptor in a 1-tuple for an item whose spec is
    `multi_output=True` (`broadcast_arrays`). The harness walks such a
    result as a sequence and compares it element by element -- a bare
    descriptor STRING would be walked character by character, which is how
    this was caught: 11/11 probe cases failing while a direct side-by-side
    comparison of the two descriptors showed them byte-identical."""
    probe_input, root = _build(mod, probe.kind)
    try:
        v = op(mod, probe_input, probe)
    except BaseException as exc:
        # Error PARITY is part of the contract too: `mT` on a 1-D input and
        # `fliplr` on a 1-D input both raise, and a descriptor that silently
        # dropped those kinds would be a smaller test wearing the same name.
        out = f"raised:{type(exc).__name__}:{exc}"
        return (out,) if multi else out
    out = _describe(v, probe_input, root)
    return (out,) if multi else out


def make_adapters(op, orig_numpy=None, orig_ionp=None, multi=False):
    """Builds the (numpy_adapter, ionp_adapter) pair for one op.

    `orig_*` are the item's pre-existing resolved callables. When they are
    supplied, a non-`ViewProbe` argument falls straight through to them, so
    attaching these adapters to an item that already has cases leaves every
    one of those cases running exactly as before -- this file only ever
    ADDS assertions."""

    def numpy_adapter(arg, *rest, **kwargs):
        if isinstance(arg, ViewProbe):
            return _probe(np, arg, op, multi)
        assert orig_numpy is not None, "non-probe case on a probe-only item"
        return orig_numpy(arg, *rest, **kwargs)

    def ionp_adapter(arg, *rest, **kwargs):
        import anionpy

        if isinstance(arg, ViewProbe):
            return _probe(anionpy, arg, op, multi)
        assert orig_ionp is not None, "non-probe case on a probe-only item"
        return orig_ionp(arg, *rest, **kwargs)

    return numpy_adapter, ionp_adapter


def probe_cases(orders=None):
    """One case per input kind; per (kind, order) when the op has an order
    axis."""
    if orders is None:
        return [(f"view/{k}", (ViewProbe(k),), {}) for k in _KINDS]
    return [
        (f"view/{k}/order_{o}", (ViewProbe(k, o),), {})
        for k in _KINDS
        for o in orders
    ]


# ---------------------------------------------------------------------------
# The ops.
# ---------------------------------------------------------------------------

_ORDERS = ("C", "F", "A", "K")

# Items that ALREADY have specs (and passing value-only cases): keep every
# existing case, add the descriptor cases on top. `orders` is non-None only
# for the two ravel spellings, whose view-vs-copy answer is a function of
# (input layout, order) -- see the module docstring's table.
_AUGMENT = {
    "ravel": (lambda mod, a, p: mod.ravel(a, order=p.order or "C"), _ORDERS, False),
    "flip": (lambda mod, a, p: mod.flip(a), None, False),
    "fliplr": (lambda mod, a, p: mod.fliplr(a), None, False),
    "flipud": (lambda mod, a, p: mod.flipud(a), None, False),
    "atleast_1d": (lambda mod, a, p: mod.atleast_1d(a), None, False),
    "atleast_2d": (lambda mod, a, p: mod.atleast_2d(a), None, False),
    "atleast_3d": (lambda mod, a, p: mod.atleast_3d(a), None, False),
    # Wrapped in a 1-tuple, not returned bare: `broadcast_arrays`'s spec is
    # `multi_output=True`, so the harness walks the result as a sequence and
    # compares it element by element. A bare descriptor STRING would be
    # walked character by character -- which is how this was caught (11/11
    # probe cases failing while a direct side-by-side comparison of the two
    # descriptors showed them identical). The single element keeps the
    # multi-output shape the harness expects.
    "broadcast_arrays": (lambda mod, a, p: mod.broadcast_arrays(a)[0], None, True),
    "linalg.matrix_transpose": (lambda mod, a, p: mod.linalg.matrix_transpose(a), None, False),
    # Already DECLARED items. They are here not to earn a declaration but to
    # keep one honest: `squeeze`'s identity rule and `reshape`/`transpose`/
    # `swapaxes`/`moveaxis`'s view semantics had no guard at all, so the
    # value-only cases that declared them would have gone on passing if the
    # returned view silently became a copy. A mutation run confirmed the
    # gap was real: breaking `squeeze`'s identity rule left the free
    # `squeeze` item GREEN while `ndarray.squeeze` (which has these cases)
    # went red.
    "squeeze": (lambda mod, a, p: mod.squeeze(a), None, False),
    "reshape": (lambda mod, a, p: mod.reshape(a, -1), None, False),
    "transpose": (lambda mod, a, p: mod.transpose(a), None, False),
    "swapaxes": (lambda mod, a, p: mod.swapaxes(a, 0, -1), None, False),
    "moveaxis": (lambda mod, a, p: mod.moveaxis(a, 0, -1), None, False),
}

# Items with NO spec at all: this file is their whole differential corpus,
# so they get the descriptor cases only.
_NEW = {
    "matrix_transpose": (lambda mod, a, p: mod.matrix_transpose(a), None),
    "ndarray.mT": (lambda mod, a, p: a.mT, None),
    "ndarray.transpose": (lambda mod, a, p: a.transpose(), None),
    "ndarray.reshape": (lambda mod, a, p: a.reshape(-1), None),
    "ndarray.squeeze": (lambda mod, a, p: a.squeeze(), None),
    "ndarray.swapaxes": (lambda mod, a, p: a.swapaxes(0, -1), None),
    "ndarray.ravel": (lambda mod, a, p: a.ravel(order=p.order or "C"), _ORDERS),
}


def _appender(prev, extra):
    def wrapped():
        base = list(prev()) if prev is not None else []
        return base + extra

    return wrapped


def _install():
    for name, (op, orders, multi) in _AUGMENT.items():
        spec = REGISTRY[name]
        orig_numpy = spec.resolve_numpy()
        orig_ionp = spec.resolve_ionp()
        np_ad, ionp_ad = make_adapters(op, orig_numpy, orig_ionp, multi)
        spec.numpy_adapter = np_ad
        spec.ionp_adapter = ionp_ad
        spec.extra_cases = _appender(spec.extra_cases, probe_cases(orders))

    for name, (op, orders) in _NEW.items():
        np_ad, ionp_ad = make_adapters(op)
        REGISTRY[name] = ItemSpec(
            name=name,
            kind="custom",
            numpy_adapter=np_ad,
            ionp_adapter=ionp_ad,
            scalar_like=True,
            custom_cases=(lambda orders=orders: probe_cases(orders)),
        )


_install()
