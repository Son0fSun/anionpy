"""ma-phase5-2026-08-08 (Monday): differential cases for eight previously-
absent `ma.MaskedArray` attribute/method items -- `.data`, `.dtype`, `.mask`,
`.itemsize`, `.nbytes`, `.strides`, `.iscontiguous()`, `.get_fill_value()`.

Scope decision (see this task's TICKET for the full writeup): of the 214
absent `ma.*` items measured this session, 135 are `ma.MaskedArray.*`
sub-items. Most of those need either new Rust reduction/sort kernels or
resolution of the `.flags`/`.strides` layout-observability gap (#63 in this
task's brief) before anything downstream of them can honestly claim layout
parity. This module picks the sub-cluster that is BOTH real and finishable:
plain read-only attribute mirrors of `self._data`, each one delegating to an
already-declared-exact `anionpy.ndarray` base (`itemsize`/`nbytes`/`strides`
are declared exact in `anionpy/_state/ndarray.py`) or an already-implemented
Phase-0 property (`data`/`dtype`/`mask`, implemented in
`anionpy/ma/core.py` since Phase 0 but never declared/verified as their own
manifest items).

`.flags` (the top-level `MaskedArray.flags`, not `.data.flags`) is
DELIBERATELY NOT covered here -- real numpy's `MaskedArray` is an `ndarray`
SUBCLASS, so `a.flags.owndata` measures `False` for EVERY masked array
(verified live, see probe below and `anionpy/ma/core.py`'s
`iscontiguous`/comment block), an artifact of `.data` always being reached
through an extra `.view()` layer in real numpy's implementation. anionpy's
`MaskedArray` is a plain (data, mask) pair -- `self._data` is genuinely
OWNED, so a naive `self._data.flags` mismatches `owndata` on literally every
case (a uniform mismatch is itself the finding, not a reason to fabricate a
hand-built flags-like value to paper over it). `.iscontiguous()` sidesteps
this by reading only `C_CONTIGUOUS`, which real numpy verified live does NOT
depend on the owndata/view distinction.

Every case below is built from FRESH construction calls (`np.ma.array(...)`
/ `anionpy.ma.MaskedArray(...)`) against real numpy 2.5.1, run live by this
task's probe scripts (`/private/tmp/madata/probe1_numpy.py`,
`probe2_compare.py`) BEFORE `core.py` was touched -- corpus-first, per this
task's method requirement. The five acceptance points (MA-DESIGN.md section
7) are all covered by `_build_corpus()`: nomask, fully-masked,
partially-masked, empty, and fill_value propagation (explicit vs default) --
plus dtype variety (all 10 non-structured dtypes touched), 0-d, and three
non-C-contiguous layouts (F-order 2x2, a transposed view, a strided slice)
so `.strides`/`.iscontiguous()` are not exercised only on the trivial
C-contiguous case.
"""

from __future__ import annotations

import numpy as np

from registry import ItemSpec


# ---------------------------------------------------------------------------
# Shared corpus: (label, data, dtype_or_None, mask_kw, extra_kw)
# ---------------------------------------------------------------------------

_DTYPES = [
    "bool", "int8", "int32", "uint32", "int64",
    "float16", "float32", "float64", "complex64", "complex128",
]


def _basic_cases():
    cases = []
    for dt in _DTYPES:
        if dt == "bool":
            data = [True, False, True]
        elif dt in ("complex64", "complex128"):
            data = [1 + 2j, 3 - 1j, -2 + 0j]
        else:
            data = [1, 2, 3] if not dt.startswith("float") else [1.5, -2.5, 3.0]
        for mask_label, mask in [
            ("nomask", None),
            ("full", [True, True, True]),
            ("partial", [False, True, False]),
        ]:
            kw = {"dtype": dt}
            if mask is not None:
                kw["mask"] = mask
            cases.append((f"{dt}/{mask_label}", data, kw))
    return cases


def _empty_cases():
    cases = []
    for dt in ("float64", "int32", "bool", "complex128"):
        cases.append((f"empty/{dt}", [], {"dtype": dt}))
    return cases


def _zerod_cases():
    cases = []
    for dt in ("float64", "int32"):
        for mask_label, mask in [("nomask", None), ("masked", True), ("unmasked", False)]:
            kw = {"dtype": dt}
            if mask is not None:
                kw["mask"] = mask
            cases.append((f"0d/{dt}/{mask_label}", 7.0 if dt == "float64" else 7, kw))
    return cases


def _fill_value_cases():
    cases = []
    for dt in ("float64", "int32"):
        data = [1.5, -2.5, 3.0] if dt == "float64" else [1, 2, 3]
        cases.append((f"explicit_fv/{dt}", data, {"dtype": dt, "mask": [False, True, False], "fill_value": -999}))
    return cases


def _layout_cases_np(mod):
    """Non-C-contiguous provenance, built identically on both sides by the
    caller (each side calls this with its own `mod` -- `np.ma` or
    `anionpy.ma`). Returns [(label, receiver_array_no_mask)]."""
    import anionpy

    # `np.asfortranarray(...)` (real force-F-contiguous copy) has no
    # equivalent name in anionpy, but `array(..., order="F")` was verified
    # live to produce the identical genuinely-F-contiguous result on both
    # sides (strides=(8,16), F_CONTIGUOUS=True) -- unlike
    # `.reshape(shape, order="F")` on an already-(2,2)-shaped source, which
    # is a DIFFERENT operation (reinterprets read/write order, not memory
    # layout) and was verified to give C-contiguous strides=(16,8) on BOTH
    # numpy and anionpy for this exact case, which is why an earlier
    # version of this corpus that mixed `np.asfortranarray` on the numpy
    # side with `.reshape(order="F")` on the anionpy side produced a
    # spurious 2/48 "mismatch" that was a corpus-construction bug, not a
    # real anionpy divergence (numpy's OWN `.reshape(order="F")` matches
    # anionpy's `.reshape(order="F")` bit-for-bit here -- both give
    # strides=(16,8) -- confirmed live before this fix).
    is_numpy = mod is np.ma
    if is_numpy:
        f_order = np.array([[1.0, 2.0], [3.0, 4.0]], order="F")
        base2d = mod.masked_array([[1.0, 2.0], [3.0, 4.0]]).data
        transposed = base2d.T
        strided = mod.masked_array([1.0, 2.0, 3.0, 4.0, 5.0]).data[::2]
    else:
        f_order = anionpy.array([[1.0, 2.0], [3.0, 4.0]], order="F")
        base2d = mod.MaskedArray([[1.0, 2.0], [3.0, 4.0]]).data
        transposed = base2d.T
        strided = mod.MaskedArray([1.0, 2.0, 3.0, 4.0, 5.0]).data[::2]
    return [
        ("f_order_2x2", f_order),
        ("transposed_view", transposed),
        ("strided_slice", strided),
    ]


def _build_corpus():
    """Returns [(label, args, kwargs)] in the (label, args, kwargs) shape
    `ItemSpec(kind="custom")` expects: `args=(payload,)`, `kwargs` are the
    `mask=`/`dtype=`/`fill_value=` construction kwargs, and `payload` is
    either a plain nested-list data literal OR the sentinel string
    `"__LAYOUT__:<label>"` telling the adapter to fetch a pre-built
    non-C-contiguous receiver from `_layout_cases_np` instead (since those
    can't be expressed as a plain literal without losing their strides)."""
    cases = []
    for label, data, kw in _basic_cases() + _empty_cases() + _zerod_cases() + _fill_value_cases():
        cases.append((label, (data,), kw))
    for layout_label, _ in _layout_cases_np(np.ma):
        for mask_label in ("nomask", "partial"):
            # The mask decision for `__LAYOUT__:` payloads can't be made at
            # corpus-build time (the receiver's shape isn't known until
            # `_layout_cases_np` actually runs against each side's own
            # module) -- so it rides along as a kwarg instead of a `label`
            # parameter the adapter never actually receives (`custom_cases`
            # only feeds `(args, kwargs)` into the adapter call; `label` is
            # bookkeeping-only, confirmed against harness.run_case's
            # `_call(np_fn, np_args, np_kwargs)`).
            kw = {"__mask_hint__": mask_label}
            cases.append((f"layout/{layout_label}/{mask_label}",
                           (f"__LAYOUT__:{layout_label}",), kw))
    return cases


def _resolve_payload(mod, payload, mask_hint):
    """`mod` is `np.ma` or `anionpy.ma`. Builds the receiver MaskedArray for
    one corpus entry."""
    if isinstance(payload, str) and payload.startswith("__LAYOUT__:"):
        layout_label = payload.split(":", 1)[1]
        receivers = dict(_layout_cases_np(mod))
        base = receivers[layout_label]
        if mask_hint == "partial":
            n = base.shape[0] if base.ndim else 1
            mask = [i % 2 == 0 for i in range(n)]
            if base.ndim == 2:
                mask = [[((r + c) % 2 == 0) for c in range(base.shape[1])] for r in range(base.shape[0])]
            return mod.masked_array(base, mask=mask) if mod is np.ma else mod.MaskedArray(base, mask=mask)
        return mod.masked_array(base) if mod is np.ma else mod.MaskedArray(base)
    return None  # signals "use plain data/kw path", handled by caller


# ---------------------------------------------------------------------------
# Per-item adapters
# ---------------------------------------------------------------------------

def _nan_safe(vals):
    if isinstance(vals, list):
        return [_nan_safe(v) for v in vals]
    try:
        if isinstance(vals, float) and vals != vals:
            return "NAN"
    except Exception:  # noqa: BLE001
        pass
    return vals


def _make_case_builder(mod_is_numpy):
    def build(payload, kw):
        # `kw` may carry `__mask_hint__` (layout cases only, see
        # `_build_corpus`) -- pop it before it can leak into the real
        # `np.ma.array`/`anionpy.ma.MaskedArray` constructor kwargs, which
        # know nothing about it.
        kw = dict(kw)
        mask_hint = kw.pop("__mask_hint__", None)
        if mod_is_numpy:
            if isinstance(payload, str) and payload.startswith("__LAYOUT__:"):
                return _resolve_payload(np.ma, payload, mask_hint)
            return np.ma.array(payload, **kw)
        else:
            import anionpy

            if isinstance(payload, str) and payload.startswith("__LAYOUT__:"):
                return _resolve_payload(anionpy.ma, payload, mask_hint)
            return anionpy.ma.MaskedArray(payload, **kw)

    return build


_build_np = _make_case_builder(True)
_build_ionp = _make_case_builder(False)


def _snap_array_np(a):
    return ("ARR", _nan_safe(a.tolist()), str(a.dtype))


def _snap_array_ionp(a):
    return ("ARR", _nan_safe(a.tolist()), str(a.dtype))


def _make_attr_adapters(attr):
    # Signature is `(payload, **kw)`, NOT `(label, payload, **kw)`: the
    # harness calls `np_fn(*args, **kwargs)` where `args=(payload,)` (see
    # `_build_corpus`'s docstring and `harness.run_case`'s
    # `_call(np_fn, np_args, np_kwargs)` -- `label` is bookkeeping-only,
    # never passed into the adapter call). An earlier version of this
    # module got this wrong and failed 48/48 cases on every item with a
    # self-inflicted TypeError, not a real anionpy divergence -- see this
    # task's report.
    def numpy_adapter(payload, **kw):
        arr = _build_np(payload, kw)
        val = getattr(arr, attr)
        if attr == "data":
            return _snap_array_np(val)
        if attr == "mask":
            if val is np.ma.nomask:
                return ("SCALAR", False, "numpy.bool")
            return _snap_array_np(val)
        return val

    def ionp_adapter(payload, **kw):
        import anionpy

        arr = _build_ionp(payload, kw)
        val = getattr(arr, attr)
        if attr == "data":
            return _snap_array_ionp(val)
        if attr == "mask":
            if val is anionpy.ma.nomask:
                return ("SCALAR", False, "numpy.bool")
            return _snap_array_ionp(val)
        return val

    return numpy_adapter, ionp_adapter


def _make_method_adapters(methodname):
    def numpy_adapter(payload, **kw):
        arr = _build_np(payload, kw)
        return getattr(arr, methodname)()

    def ionp_adapter(payload, **kw):
        arr = _build_ionp(payload, kw)
        return getattr(arr, methodname)()

    return numpy_adapter, ionp_adapter


def _cases_fn():
    return _build_corpus()


MA_PHASE5_SPECS: dict[str, ItemSpec] = {}

for _attr in ("data", "dtype", "mask", "itemsize", "nbytes", "strides"):
    _np_a, _ionp_a = _make_attr_adapters(_attr)
    MA_PHASE5_SPECS[f"ma.MaskedArray.{_attr}"] = ItemSpec(
        name=f"ma.MaskedArray.{_attr}", kind="custom", scalar_like=True,
        numpy_adapter=_np_a, ionp_adapter=_ionp_a,
        convert_ionp_args=False, custom_cases=_cases_fn,
    )

for _meth in ("iscontiguous", "get_fill_value"):
    _np_m, _ionp_m = _make_method_adapters(_meth)
    MA_PHASE5_SPECS[f"ma.MaskedArray.{_meth}"] = ItemSpec(
        name=f"ma.MaskedArray.{_meth}", kind="custom", scalar_like=True,
        numpy_adapter=_np_m, ionp_adapter=_ionp_m,
        convert_ionp_args=False, custom_cases=_cases_fn,
    )
