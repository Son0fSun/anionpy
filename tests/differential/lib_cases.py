"""Differential ItemSpecs for the four `lib.*`/`ctypeslib.*` items this task
implements (`lib.NumpyVersion`, `lib.Arrayterator`, `ctypeslib.load_library`,
`ctypeslib.c_intp`). See `anionpy/lib.py`, `anionpy/ctypeslib.py`, and
`anionpy/_state/misc_namespaces.py` for what these are and why the other eight
`lib.*`/`ctypeslib.*` surface items are NOT here (feasibility assessment
recorded in this task's report, not restated here).

All four items are `kind="custom"`: none of them is a plain
array-in/array-out elementwise function, so none fits `unary`/`binary`/
`binary_op`/`method`. `NumpyVersion` and `Arrayterator` construct a CLASS
INSTANCE, not an array or a scalar -- `harness.py`'s `scalar_like` path
requires `type(np_out) is type(ionp_out)` (see `_compare_scalar_like`'s
docstring), which a `numpy.lib.NumpyVersion` vs `anionpy.lib.NumpyVersion`
instance can never satisfy (different classes by construction, on purpose
-- anionpy does not subclass numpy's types). Each therefore uses a
`numpy_adapter`/`ionp_adapter` pair that constructs the real object on its
own side and reduces it to a plain, type-matching Python value (a tuple of
its public attributes for `NumpyVersion`; a tuple of each yielded block's
`.tolist()` for `Arrayterator`) before handing off to the ordinary
`scalar_like` tuple-equality comparison -- this compares the OBJECTS'
observable behavior/state, not some unrelated proxy value.

`load_library`/`c_intp`-adjacent note: `load_library` is `kind="custom"`
for the same class-of-reasons (it returns a live `ctypes.CDLL` handle,
not a value `compare_values` has any notion of comparing) -- the adapter
here doesn't compare the returned handle at all, it compares whether both
sides agree on SUCCESS vs the SAME exception type for a given
(libname, loader_path) pair, which is the only well-defined cross-
implementation invariant a dynamically loaded library handle has.
`c_intp` is not registered here at all: it is a module-level constant
with no arguments and no call semantics, so there is nothing for a
`kind="custom"` case to vary -- it is compared once, directly, by this
task's own verification (see the report), not through the differential
harness (which has no "compare this bare attribute" kind).
"""
from __future__ import annotations

import os

import _bootstrap  # noqa: F401

import numpy as np

from registry import ItemSpec

LIB_SPECS: dict[str, ItemSpec] = {}


# ---------------------------------------------------------------------------
# lib.NumpyVersion
# ---------------------------------------------------------------------------

def _numpy_version_probe(vstring):
    v = np.lib.NumpyVersion(vstring)
    return (v.vstring, v.version, v.major, v.minor, v.bugfix,
            v.pre_release, v.is_devversion)


def _ionp_version_probe(vstring):
    import anionpy
    v = anionpy.lib.NumpyVersion(vstring)
    return (v.vstring, v.version, v.major, v.minor, v.bugfix,
            v.pre_release, v.is_devversion)


def numpyversion_cases():
    return [
        ("released", ("1.8.0",), {}),
        ("released_high_numbers", ("12.34.56",), {}),
        ("alpha", ("1.8.0a1",), {}),
        ("alpha_high", ("1.8.0a12",), {}),
        ("beta", ("1.8.0b2",), {}),
        ("rc", ("1.8.0rc1",), {}),
        ("dev_no_prerelease", ("1.8.0.dev-f1234afa",), {}),
        ("dev_unknown", ("1.8.0.dev-Unknown",), {}),
        ("dev_after_alpha", ("1.8.0a1.dev-f1234afa",), {}),
        ("dev_after_beta", ("1.8.1b2.dev-f1234afa",), {}),
        ("dev_after_rc", ("1.8.1rc1.dev-f1234afa",), {}),
        ("current_installed", (np.__version__,), {}),
        ("too_short_raises", ("1.7",), {}),
        ("empty_raises", ("",), {}),
        ("non_numeric_raises", ("abc",), {}),
    ]


LIB_SPECS["lib.NumpyVersion"] = ItemSpec(
    name="lib.NumpyVersion", kind="custom", custom_cases=numpyversion_cases,
    numpy_adapter=_numpy_version_probe, ionp_adapter=_ionp_version_probe,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# lib.Arrayterator
# ---------------------------------------------------------------------------

def _numpy_arrayterator_probe(var, buf_size=None):
    it = np.lib.Arrayterator(var, buf_size)
    return (it.shape, tuple(np.asarray(block).tolist() for block in it))


def _ionp_arrayterator_probe(var, buf_size=None):
    import anionpy
    it = anionpy.lib.Arrayterator(var, buf_size)
    return (it.shape, tuple(block.tolist() for block in it))


def arrayterator_cases():
    a1 = np.arange(24).reshape(2, 3, 4)
    a2 = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    a3 = np.arange(10)
    return [
        ("3d_buf_none", (a1, None), {}),
        ("3d_buf_2", (a1, 2), {}),
        ("3d_buf_7", (a1, 7), {}),
        ("4d_buf_2", (a2, 2), {}),
        ("4d_buf_50", (a2, 50), {}),
        ("1d_buf_3", (a3, 3), {}),
        ("1d_buf_none", (a3, None), {}),
    ]


LIB_SPECS["lib.Arrayterator"] = ItemSpec(
    name="lib.Arrayterator", kind="custom", custom_cases=arrayterator_cases,
    numpy_adapter=_numpy_arrayterator_probe, ionp_adapter=_ionp_arrayterator_probe,
    scalar_like=True,
    # convert_ionp_args left at its True default deliberately: this is what
    # makes `_ionp_arrayterator_probe`'s `var` a REAL `anionpy.ndarray`
    # (`_wrap_custom_conversion` converts the numpy array `arrayterator_cases()`
    # hands it before calling the adapter) -- exercising `anionpy.ndarray`'s own
    # `.shape`/`.ndim`/tuple-of-slice `__getitem__` inside `Arrayterator`,
    # not numpy's. Without this, the "anionpy" probe would silently wrap a real
    # numpy array in anionpy's Arrayterator class and never touch anionpy.ndarray
    # at all -- a tautology this task's own module docstring explicitly
    # rules out for other items (see registry.py's `_wrap_custom_conversion`
    # docstring).
)


# ---------------------------------------------------------------------------
# ctypeslib.load_library
# ---------------------------------------------------------------------------

def _numpy_load_library_probe(libname, loader_path):
    try:
        np.ctypeslib.load_library(libname, loader_path)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - reporting the outcome class, not the handle
        return f"{type(exc).__name__}"


def _ionp_load_library_probe(libname, loader_path):
    import anionpy
    try:
        anionpy.ctypeslib.load_library(libname, loader_path)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - reporting the outcome class, not the handle
        return f"{type(exc).__name__}"


def load_library_cases():
    return [
        ("missing_lib_raises", ("definitely_not_a_real_library_xyz", "/usr/lib"), {}),
        ("missing_dir_raises", ("definitely_not_a_real_library_xyz", "/no/such/dir/at/all"), {}),
        ("libc_by_full_name", ("libc.dylib" if os.uname().sysname == "Darwin" else "libc.so.6", "/usr/lib"), {}),
    ]


LIB_SPECS["ctypeslib.load_library"] = ItemSpec(
    name="ctypeslib.load_library", kind="custom", custom_cases=load_library_cases,
    numpy_adapter=_numpy_load_library_probe, ionp_adapter=_ionp_load_library_probe,
    scalar_like=True, convert_ionp_args=False,
)


# ---------------------------------------------------------------------------
# ctypeslib.c_intp
# ---------------------------------------------------------------------------
# Added 2026-08-03 (Wave 1, "connect what already exists"). This module's own
# docstring previously said `c_intp` "is not registered here at all ... there
# is nothing for a kind="custom" case to vary". That was true for VARYING
# input, but a module-level constant is still comparable once, directly --
# the same shape of check `scalar_alias_identity`/the per-alias items in
# scalar_cases.py already use for other bare constants. `c_intp` is numpy's
# own `ctypes` type alias for `np.intp` (verified live on this platform, arm64
# Darwin/LP64: both `numpy.ctypeslib.c_intp` and `anionpy.ctypeslib.c_intp` are
# literally `ctypes.c_long`) -- there is exactly one observable property, the
# class identity, and it is checked directly rather than varied across cases.

def _numpy_c_intp_probe():
    return np.ctypeslib.c_intp


def _ionp_c_intp_probe():
    import anionpy
    return anionpy.ctypeslib.c_intp


LIB_SPECS["ctypeslib.c_intp"] = ItemSpec(
    name="ctypeslib.c_intp", kind="custom",
    custom_cases=(lambda: [("value", (), {})]),
    numpy_adapter=_numpy_c_intp_probe, ionp_adapter=_ionp_c_intp_probe,
    scalar_like=True, convert_ionp_args=False,
)
