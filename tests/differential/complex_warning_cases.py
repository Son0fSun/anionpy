"""Differential corpus for the `ComplexWarning`-on-real-cast diagnostic gap.

WHY THIS FILE EXISTS. Measured 2026-08-06 against numpy 2.5.1: casting a
complex array/scalar down to a real dtype emits `numpy.exceptions.
ComplexWarning` ("Casting complex values to real discards the imaginary
part") on real numpy but was completely silent on anionpy, at every site that
performs the cast -- `astype`, `ndarray.__array__(dtype=...)` (which
`np.asarray(x, dtype=real)` and `np.array(x, dtype=real)` both funnel
through for an already-materialized array), `anionpy.array`/`anionpy.asarray`'s
own `dtype=` kwarg on non-array sources, and `a[...] = complex_value`
broadcast assignment into a real-dtype destination. The VALUES were already
correct on both sides (real-part-only, matching numpy bit-for-bit) -- this
is a pure lost-diagnostic gap: code that runs
`warnings.simplefilter('error')` to catch an accidental complex-to-real
truncation raised on numpy and passed silently on anionpy.

Fixed 2026-08-06 by `errors::warn_complex_cast` (`ionp-py/src/errors.rs`),
wired into the four `cast_to` call sites named above (`ionp-py/src/lib.rs`:
`PyArray::astype`, `PyArray::__array__`, `array_impl` [shared by
`array`/`asarray`], `coerce_setitem_value` [all three of its source
branches: an anionpy array, an external `numpy.ndarray`, and a list/tuple]).
anionpy's `ComplexWarning` is built the same way `LinAlgError`/`AxisError`
already are (`errors.rs`'s established pattern): it SUBCLASSES numpy's real
`numpy.exceptions.ComplexWarning` when numpy is importable (so `except
numpy.exceptions.ComplexWarning:` still catches it), and falls back to
subclassing the builtin `RuntimeWarning` when numpy is absent (numpy's own
`ComplexWarning.__mro__` was measured live to be exactly
`(ComplexWarning, RuntimeWarning, Warning, Exception, BaseException,
object)` -- a plain `class ComplexWarning(RuntimeWarning): pass` with no
custom `__init__`/`__str__` to reimplement, unlike `AxisError`).

WHAT IS COMPARED. Each case wraps a real call in
`warnings.catch_warnings(record=True)` + `simplefilter('always')` and
records `[(w.category.__name__, str(w.message)), ...]` -- the
CLASS NAME, the exact TEXT, and (via list equality) the COUNT, not merely
"something fired". That list is folded into the same descriptor string
alongside the call's own outcome (post-cast dtype/shape/values, or the
raised-exception type+message for the handful of sites where numpy raises
outright rather than warning) -- so a case that got the warning right but
broke the underlying cast value, or vice versa, still fails.

Warning CLASS is compared by `__name__` only (`"ComplexWarning"`), not by
identity/module: real numpy's category lives in `numpy.exceptions`, anionpy's
in `anionpy` (mirroring the `LinAlgError`/`AxisError` compat-subclass pattern
`harness.py`'s `_is_ionp_compat_subclass` already documents for exceptions
-- a warning category is exactly the same kind of user-facing hierarchy).
The identity relationship itself (anionpy's class really does subclass
numpy's, `except numpy.exceptions.ComplexWarning:` really does catch it) is
checked separately, live, in this task's report -- not by this corpus,
which only proves the two sides fire the SAME family with the SAME text.

NEGATIVE CONTROLS. Roughly a third of the cases assert NO warning on
EITHER side: complex128->complex64 (narrowing within complex is not a
real-part-discarding cast), real->real casts (including overflowing ones,
e.g. float64->int8), and int/float/bool -> complex (widening). A corpus
that only ever fires the positive case cannot tell "warns correctly" apart
from "warns on everything" -- these pin the boundary.

WHAT IS DELIBERATELY NOT COVERED (see KNOWN-DIFFERENCES.md for the dated
entry). `ufunc(..., dtype=real)` / `ufunc(..., out=real_array)` fed a
complex input: real numpy does not warn there either -- it raises
`UFuncTypeError` outright at 'same_kind' casting (measured: both numpy and
anionpy already raise, matching -- confirmed a NEGATIVE, not a gap). `.fill()`
is a separate, worse bug (anionpy silently SUCCEEDS storing the real part
where numpy raises `TypeError`; not a missing-warning case, a missing-
exception one) and is recorded separately, not folded into this corpus's
verdict. `np.putmask`/`np.place` are simply unimplemented in anionpy
(`AttributeError`), unrelated to this task. The whole separate
`RuntimeWarning` family (overflow/invalid-value/divide-by-zero in ufuncs,
overflow-in-cast) remains the pre-existing, already-recorded library-wide
gap -- `nanmax`'s all-NaN-slice `RuntimeWarning` is the one site that
already worked before this task and is unchanged by it.
"""

import warnings

from registry import REGISTRY, ItemSpec

# NOTE (2026-08-06, false-green fix): cases used to bundle (site, src,
# dst_dtype) into a single `WarnProbe` namedtuple and pass ONE such object
# per case. That was silently unsound: `harness.run_case()` calls
# `_freshen()` on every argument before either side is invoked, and
# `_freshen()` (like `registry.py`'s `make_ionp_array_converter`, for the
# same "protect against shared mutable buffers" reason) treats ANY `tuple`
# -- including a namedtuple subclass -- as a plain container and rebuilds it
# with `tuple(_freshen(v) for v in x)`, which silently drops the namedtuple
# subclass and hands both adapters a bare 3-tuple. Both `_numpy_adapter` and
# `_ionp_adapter` asserted `isinstance(arg, WarnProbe)`, so EVERY case raised
# `AssertionError` identically on BOTH sides -- same type, same message
# (since both got the same mangled tuple) -- which `run_case()` grades a
# PASS (matching exception type + text), never exercising astype/array/
# asarray/__array__/setitem at all. 354/354 "passed" while the compiled
# extension was reverted to fully silent, confirmed by direct probing.
# Fixed by passing `site`/`src`/`dst_dtype` as three independent plain
# `str` positional arguments instead of one container -- `_freshen` and
# `make_ionp_array_converter` both pass a bare `str` through unchanged, so
# there is nothing left for either to mangle.

# ---------------------------------------------------------------------------
# Source builders: name -> callable(module) -> array/list/tuple, spanning
# both complex widths, both "should warn" and "should NOT warn" source
# dtypes, and a few provenance variations (0-d, empty, transposed/F-order,
# external-numpy-array source for the setitem sites' dedicated branch).
# ---------------------------------------------------------------------------

def _ext_numpy(vals, dtype):
    """A REAL external `numpy.ndarray` -- deliberately NOT built via the
    module-under-test. Feeding this into `anionpy`'s `coerce_setitem_value`
    exercises its dedicated "foreign numpy.ndarray" branch (the middle of
    its three source branches); feeding it into real numpy is simply an
    ordinary numpy array. This is numpy supplying INPUT BYTES, never an
    ANSWER -- the value itself is trivial (`np.array(vals, dtype=dtype)`),
    and neither side's warning/result is read off this object, only off
    the module-under-test's own call.
    """
    import numpy as _np
    return _np.array(vals, dtype=dtype)


_SRCS = {
    "c128_1d": lambda m: m.array([1 + 2j, 3 - 4j, -5 + 6j], dtype="complex128"),
    "c64_1d": lambda m: m.array([1 + 2j, 3 - 4j], dtype="complex64"),
    "c128_0d": lambda m: m.array(1 + 2j, dtype="complex128"),
    "c128_empty": lambda m: m.array([], dtype="complex128"),
    "c128_2d": lambda m: m.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype="complex128"),
    "c128_zero_imag": lambda m: m.array([1 + 0j, 2 + 0j], dtype="complex128"),
    "c128_list": lambda m: [1 + 2j, 3 - 4j, -5 + 6j],
    "c128_tuple": lambda m: (1 + 2j, 3 - 4j),
    "c128_scalar": lambda m: 1 + 2j,
    "c128_extnumpy": lambda m: _ext_numpy([1 + 2j, 3 - 4j, 5 + 6j], "complex128"),
    "c64_extnumpy": lambda m: _ext_numpy([1 + 2j, 3 - 4j], "complex64"),
    # Negative-control sources: casting these to any real dtype must NOT
    # fire ComplexWarning on either side.
    "c128_to_c64_src": lambda m: m.array([1 + 2j, 3 - 4j], dtype="complex128"),
    "f64_1d": lambda m: m.array([1.5, 2.5, -3.5], dtype="float64"),
    "i32_1d": lambda m: m.array([1, 2, 300], dtype="int32"),
    "bool_1d": lambda m: m.array([True, False, True]),
}


def _record(fn):
    """Run `fn()` (already bound to the module under test) inside a FRESH,
    isolated `catch_warnings` block on `simplefilter('always')` (so a
    warning fires every time regardless of the ambient filter/registry
    state, and so this probe can never observe a warning some OTHER case
    already triggered -- see this task's report for why leaking filter
    state between checks was flagged as a specific trap to guard against).
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            result = fn()
            exc = None
        except BaseException as e:  # noqa: BLE001 -- PyO3 panics derive BaseException
            result = None
            exc = e
    # `w.category` is already the warning CLASS object (that's what
    # `warnings` records it as), not an instance -- `type(w.category)`
    # therefore returns the METACLASS (`type`) on every warning regardless
    # of its real class, which made this comparison vacuously match on the
    # literal string `"type"` for both sides. Fixed 2026-08-06: read the
    # class's own `__name__` directly.
    warns = [(w.category.__name__, str(w.message)) for w in rec]
    return warns, result, exc


def _result_descr(result):
    if hasattr(result, "dtype") and hasattr(result, "shape"):
        return f"dtype={result.dtype.name}|shape={tuple(result.shape)}|values={result.tolist()!r}"
    return f"pyval={result!r}"


def _descriptor(warns, result, exc):
    if exc is not None:
        body = f"raised:{type(exc).__name__}:{exc}"
    else:
        body = _result_descr(result)
    return f"warns={warns!r}|{body}"


# ---------------------------------------------------------------------------
# Site actions: site name -> callable(module, src_builder, dst_dtype) -> descriptor
# ---------------------------------------------------------------------------

def _site_astype(m, src_key, dst_dtype):
    def fn():
        a = _SRCS[src_key](m)
        return a.astype(dst_dtype)
    return _record(fn)


def _site_array_dtype(m, src_key, dst_dtype):
    def fn():
        src = _SRCS[src_key](m)
        return m.array(src, dtype=dst_dtype)
    return _record(fn)


def _site_asarray_dtype(m, src_key, dst_dtype):
    def fn():
        src = _SRCS[src_key](m)
        return m.asarray(src, dtype=dst_dtype)
    return _record(fn)


def _site_array_dunder(m, src_key, dst_dtype):
    def fn():
        a = _SRCS[src_key](m)
        if not hasattr(a, "__array__"):
            # Plain list/tuple/scalar sources have no `__array__` of their
            # own on EITHER side -- skip by raising the same sentinel both
            # sides will produce identically (folded into the descriptor,
            # not a real divergence).
            raise TypeError("no __array__ on this source (expected)")
        return a.__array__(dtype=dst_dtype)
    return _record(fn)


def _site_setitem(m, src_key, dst_dtype):
    def fn():
        src = _SRCS[src_key](m)
        n = len(src) if hasattr(src, "__len__") else 1
        n = max(n, 1)
        a = m.zeros(n, dtype=dst_dtype)
        if hasattr(src, "shape") and src.shape == ():
            a = m.zeros(1, dtype=dst_dtype)
            a[0] = src if not hasattr(src, "item") else src
            return a
        a[:] = src
        return a
    return _record(fn)


_SITES = {
    "astype": _site_astype,
    "array_dtype": _site_array_dtype,
    "asarray_dtype": _site_asarray_dtype,
    "array_dunder": _site_array_dunder,
    "setitem": _site_setitem,
}


def _numpy_adapter(site, src, dst_dtype, **kwargs):
    import numpy as np

    warns, result, exc = _SITES[site](np, src, dst_dtype)
    return _descriptor(warns, result, exc)


def _ionp_adapter(site, src, dst_dtype, **kwargs):
    import anionpy

    warns, result, exc = _SITES[site](anionpy, src, dst_dtype)
    return _descriptor(warns, result, exc)


def _scope_excluded(site, src, dst_dtype):
    """True for a (site, src, dst_dtype) combination that exercises a
    DIFFERENT, PRE-EXISTING anionpy/numpy divergence than the one this corpus
    exists to test -- see this task's report and the dated
    KNOWN-DIFFERENCES.md entry it adds.

    MEASURED 2026-08-06: for a raw Python `list`/`tuple`/scalar source (not
    yet an ndarray on either side) fed into `array(dtype=real)`/
    `asarray(dtype=real)`/setitem with a non-bool real destination, real
    numpy does NOT build a complex array and then cast it down -- it
    attempts `float(x)`/`int(x)` element-by-element and that raises
    `TypeError` immediately (`float() argument must be ... not 'complex'`).
    anionpy instead auto-detects the complex dtype, builds a complex128 array,
    THEN casts to the target -- succeeding (now with `ComplexWarning`)
    where numpy raises outright. This is an anionpy list/tuple/scalar
    ingestion gap, unrelated to warnings, and was already there before this
    task's fix (the fix only changed whether the pre-existing silent
    success also warns) -- fixing container-ingestion semantics is out of
    scope for the lost-diagnostic bug this corpus targets. `bool` is exempt
    from this exclusion because it is not part of the affected sweep
    (`array_impl`'s dtype-detection-then-cast path is what diverges here,
    independent of the bool-exemption in `warn_complex_cast` itself).
    """
    if src not in ("c128_list", "c128_tuple", "c128_scalar"):
        return False
    return site in ("array_dtype", "asarray_dtype", "setitem") and dst_dtype != "bool"


def _cases():
    out = []

    def add(site, src, dst_dtype):
        if _scope_excluded(site, src, dst_dtype):
            return
        label = f"{site}|{src}|{dst_dtype}"
        out.append((label, (site, src, dst_dtype), {}))

    # --- POSITIVE: complex source -> real destination, every fixed site,
    # both complex widths, several destination dtypes/kinds.
    complex_srcs = (
        "c128_1d", "c64_1d", "c128_0d", "c128_2d", "c128_zero_imag",
        "c128_list", "c128_tuple", "c128_extnumpy", "c64_extnumpy",
    )
    real_dsts = ("float64", "float32", "float16", "int32", "int8", "uint8", "bool")
    for site in ("astype", "array_dtype", "asarray_dtype", "array_dunder", "setitem"):
        for src in complex_srcs:
            for dst in real_dsts:
                add(site, src, dst)

    # `c128_empty`/`c128_scalar`: shape/provenance edges, a narrower dtype
    # sweep (the positive-case count above already covers the dtype axis).
    for site in ("astype", "array_dtype", "asarray_dtype", "array_dunder"):
        for dst in ("float64", "int32"):
            add(site, "c128_empty", dst)
    for site in ("array_dtype", "asarray_dtype", "setitem"):
        for dst in ("float64", "int32"):
            add(site, "c128_scalar", dst)

    # --- NEGATIVE CONTROLS: must NOT warn on either side.
    for site in ("astype", "array_dtype", "asarray_dtype", "array_dunder", "setitem"):
        # complex -> complex (narrower width): no real part discarded.
        add(site, "c128_to_c64_src", "complex64")
        # real -> real, including an overflowing narrow int cast.
        add(site, "f64_1d", "int8")
        add(site, "i32_1d", "float64")
        # widening into complex: never discards anything.
        add(site, "f64_1d", "complex128")
        add(site, "bool_1d", "complex64")

    return out


def _install():
    REGISTRY["complex_warning_on_real_cast"] = ItemSpec(
        name="complex_warning_on_real_cast",
        kind="custom",
        numpy_adapter=_numpy_adapter,
        ionp_adapter=_ionp_adapter,
        scalar_like=True,
        custom_cases=_cases,
    )


_install()
