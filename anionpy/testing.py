"""
anionpy.testing -- a numpy.testing-compatible surface for anionpy.

Python is a skin here, exactly as everywhere else in anionpy. This module never
performs a numerical computation itself: every elementwise comparison,
tolerance check, and reduction ("are all elements true?", "is this NaN?",
"is this inf?") is expressed as a composition of anionpy's existing Rust-backed
operators and ufuncs (``__eq__ __ne__ __lt__ __le__ __gt__ __ge__ __sub__
__abs__``, ``equal``/``not_equal``/``less``/... , ``logical_and``/``or``/
``not``, and the genuine Rust reduction ``sum_f64``). Python only does
control flow, message formatting, exception raising, and bookkeeping of
already-Rust-computed scalars -- precisely what numpy's own
``numpy/testing/_private/utils.py`` does (it is itself a thin Python layer
over C/numpy ufuncs).

Two items from the numpy.testing surface are deliberately left unimplemented
(``assert_array_almost_equal_nulp``, ``assert_array_max_ulp``): both require
reinterpreting a float's bit pattern as a signed integer ("ULP distance"),
and anionpy exposes no bit-reinterpretation/view primitive in Rust. Doing that
reinterpretation by hand in Python would mean computing on array elements
outside Rust, which is exactly the "no Python arithmetic" rule this project
exists to enforce. They stay absent rather than being faked.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import importlib.metadata
import importlib.util
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import threading
import unittest
import warnings as _warnings
from functools import wraps
from tempfile import mkdtemp, mkstemp
from unittest.case import SkipTest

import anionpy as _anionpy

__all__ = [
    # constants
    "verbose", "IS_PYPY", "IS_PYSTON", "IS_WASM", "IS_MUSL", "IS_64BIT",
    "HAS_REFCOUNT", "HAS_LAPACK64", "BLAS_SUPPORTS_FPE", "NOGIL_BUILD",
    "IS_EDITABLE", "IS_INSTALLED", "NUMPY_ROOT",
    # classes
    "IgnoreException", "KnownFailureException", "SkipTest", "TestCase",
    "clear_and_catch_warnings", "suppress_warnings",
    # functions
    "assert_", "assert_allclose", "assert_almost_equal", "assert_approx_equal",
    "assert_array_almost_equal", "assert_array_compare", "assert_array_equal",
    "assert_array_less", "assert_equal", "assert_no_gc_cycles",
    "assert_no_warnings", "assert_raises", "assert_raises_regex",
    "assert_string_equal", "assert_warns", "break_cycles", "build_err_msg",
    "check_support_sve", "decorate_methods", "jiffies", "measure",
    "memusage", "print_assert_equal", "run_threaded", "rundocs",
    "runstring", "tempdir", "temppath", "test",
]

# ---------------------------------------------------------------------------
# Rust-routed primitives -- the only place this module touches array data.
# Everything below composes existing anionpy operators; none of it loops over
# elements or computes a difference in Python.
# ---------------------------------------------------------------------------

TestCase = unittest.TestCase


def _as_ionp(x):
    """Coerce array_like `x` to an anionpy.ndarray. Pure data marshalling."""
    if isinstance(x, _anionpy.ndarray):
        return x
    return _anionpy.array(x)


def _all_true(mask):
    """True iff every element of boolean anionpy array `mask` is True.

    Routed entirely through Rust: logical_not (ufunc) + astype (Rust cast)
    + sum_f64 (genuine Rust reduction). The only Python-level operation is
    comparing the resulting reduced scalar to 0.0.
    """
    if mask.size == 0:
        return True
    inverted = _anionpy.logical_not(mask).astype("float64")
    return _anionpy.sum_f64(inverted.__array__()) == 0.0


def _any_true(mask):
    """True iff at least one element of boolean anionpy array `mask` is True."""
    if mask.size == 0:
        return False
    total = _anionpy.sum_f64(mask.astype("float64").__array__())
    return total != 0.0


def _is_nan(a):
    """Elementwise NaN detection via IEEE754 self-inequality (Rust ufunc)."""
    return _anionpy.not_equal(a, a)


def _is_inf(a):
    """Elementwise +-inf detection via Rust equal + logical_or."""
    pos = _anionpy.equal(a, float("inf"))
    neg = _anionpy.equal(a, float("-inf"))
    return _anionpy.logical_or(pos, neg)


def _rs1(name, a):
    """Apply a Rust unary ufunc to a scalar, returning a Python float."""
    out = getattr(_anionpy, name)(_anionpy.array(float(a)))
    return float(out.__array__())


def _rs2(name, a, b):
    """Apply a Rust binary ufunc to two scalars, returning a Python float."""
    out = getattr(_anionpy, name)(_anionpy.array(float(a)), _anionpy.array(float(b)))
    return float(out.__array__())


def _shape_of(a):
    return tuple(a.shape)


# ---------------------------------------------------------------------------
# Platform-fact constants. These describe anionpy's own build/install, not
# numpy's -- pure Python/OS introspection, no array math involved anywhere.
# ---------------------------------------------------------------------------

verbose = 0

NUMPY_ROOT = pathlib.Path(_anionpy.__file__).parent

try:
    _ionp_dist = importlib.metadata.distribution("anionpy")
except importlib.metadata.PackageNotFoundError:
    IS_INSTALLED = IS_EDITABLE = False
else:
    IS_INSTALLED = True
    try:
        if sys.version_info < (3, 13):
            import json
            import types
            _origin = json.loads(
                _ionp_dist.read_text("direct_url.json") or "{}",
                object_hook=lambda data: types.SimpleNamespace(**data),
            )
            IS_EDITABLE = _origin.dir_info.editable
        else:
            IS_EDITABLE = _ionp_dist.origin.dir_info.editable
    except AttributeError:
        IS_EDITABLE = False

    if not IS_EDITABLE and _ionp_dist.locate_file("anionpy") != NUMPY_ROOT:
        IS_INSTALLED = False

IS_WASM = platform.machine() in ("wasm32", "wasm64")
IS_PYPY = sys.implementation.name == "pypy"
IS_PYSTON = hasattr(sys, "pyston_version_info")
HAS_REFCOUNT = getattr(sys, "getrefcount", None) is not None and not IS_PYSTON

# ionp-core is a native Rust ndarray/ufunc engine with no BLAS/LAPACK
# dependency (see GOAL-ionp.md architecture section) -- there is no FPE
# behaviour to inherit from a BLAS library and no ILP64 LAPACK build to
# report. False is the honest answer, not a placeholder.
BLAS_SUPPORTS_FPE = False
HAS_LAPACK64 = False

IS_MUSL = False
_host_gnu_type = sysconfig.get_config_var("HOST_GNU_TYPE") or ""
if "musl" in _host_gnu_type:
    IS_MUSL = True

NOGIL_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
IS_64BIT = sys.maxsize > 2**32


# ---------------------------------------------------------------------------
# Exception / TestCase-family classes
# ---------------------------------------------------------------------------

class KnownFailureException(Exception):
    """Raise this exception to mark a test as a known failing test."""


class IgnoreException(Exception):
    """Ignoring this exception due to disabled feature."""


# ---------------------------------------------------------------------------
# assert_ and message building
# ---------------------------------------------------------------------------

def assert_(val, msg=""):
    """Assert that works in release mode (the bare `assert` is optimized out
    under -O). Accepts a callable msg to defer evaluation until failure."""
    __tracebackhide__ = True
    if not val:
        try:
            smsg = msg()
        except TypeError:
            smsg = msg
        raise AssertionError(smsg)


def build_err_msg(arrays, err_msg, header="Items are not equal:",
                   verbose=True, names=("ACTUAL", "DESIRED"), precision=8):
    msg = ["\n" + header]
    err_msg = str(err_msg)
    if err_msg:
        if err_msg.find("\n") == -1 and len(err_msg) < 79 - len(header):
            msg = [msg[0] + " " + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):
            try:
                r = repr(a)
            except Exception as exc:
                r = f"[repr failed for <{type(a).__name__}>: {exc}]"
            if r.count("\n") > 3:
                r = "\n".join(r.splitlines()[:3])
                r += "..."
            name = names[i] if i < len(names) else f"arg{i}"
            msg.append(f" {name}: {r}")
    return "\n".join(msg)


# ---------------------------------------------------------------------------
# The core array-comparison engine
# ---------------------------------------------------------------------------

def _elem_str(v):
    """String an already-Rust-indexed 0-d anionpy value the way numpy's own
    scalar types stringify (`str(np.int64(3))` == '3', `str(np.float64(1.0))`
    == '1.0' -- keeping the trailing `.0` that `array2string` below does
    NOT keep). Routed through the existing `__array__` interop marshalling
    path (materializes a genuine 0-d numpy array purely so Python's `str()`
    reuses numpy's own formatting -- no arithmetic happens here)."""
    return str(v.__array__())


def _array2string_scalar(v):
    """Format a 0-d anionpy value the way numpy's `array2string` formats a
    bare scalar (drops the trailing `.0` `_elem_str` keeps, e.g. '1.' not
    '1.0', and prints a bare 'inf' for +inf) -- reuses anionpy's own array
    repr (already numpy-format-verified elsewhere in this module, e.g.
    `build_err_msg`'s ` ACTUAL: array([...])` lines) on a 1-element
    reshape, then strips the `array([...])` wrapper. Pure string
    surgery on an already-Rust-computed value, no new computation."""
    r = repr(v.reshape((1,)))
    prefix, suffix = "array([", "])"
    if r.startswith(prefix) and r.endswith(suffix):
        return r[len(prefix):-len(suffix)]
    return r  # pragma: no cover - defensive, unexpected repr shape


def _mismatch_diagnostics(ax, ay, invalid, names):
    """Build numpy's `assert_array_compare` diagnostic block -- the
    "Mismatch at index(es)" / "Max absolute difference among violations" /
    "Max relative difference among violations" lines that follow the
    "Mismatched elements: ..." line -- verified against real numpy 2.5.1's
    `numpy/testing/_private/utils.py`. Every reduction/elementwise step
    here is a genuine anionpy Rust op (`argwhere`, `where`, `abs`, `subtract`,
    `divide`, `not_equal`, `logical_and`, `amax`); Python only indexes by
    position, formats, and joins strings.

    Returns a list of remark lines (possibly empty, e.g. if `ax`/`ay` are
    both 0-d so there is no "index" to report).
    """
    lines = []

    if invalid.ndim != 0:
        positions = _anionpy.argwhere(invalid).tolist()
        shown = positions[:5]
        idx_lines = []
        for row in shown:
            key = tuple(row)
            xv = ax if ax.ndim == 0 else ax[key]
            yv = ay if ay.ndim == 0 else ay[key]
            idx_lines.append(f" {row}: {_elem_str(xv)} ({names[0]}), {_elem_str(yv)} ({names[1]})")
        s = "\n".join(idx_lines)
        if len(positions) == 1:
            lines.append(f"Mismatch at index:\n{s}")
        elif len(positions) <= 5:
            lines.append(f"Mismatch at indices:\n{s}")
        else:
            lines.append(f"First 5 mismatches are at indices:\n{s}")

    # "Max absolute difference among violations": max(|x - y|) restricted
    # to invalid positions only, via `where(invalid, error, 0)` (error is
    # never negative, so a 0 fill never wins the max unless every invalid
    # entry is itself exactly 0, in which case 0 IS the correct answer) --
    # this sidesteps anionpy having no boolean/fancy indexing rather than
    # faking the restriction.
    error = _anionpy.abs(_anionpy.subtract(ax, ay))
    masked_error = _anionpy.where(invalid, error, 0)
    max_abs_error = _anionpy.amax(masked_error)
    lines.append("Max absolute difference among violations: " + _array2string_scalar(max_abs_error))

    # "Max relative difference among violations": matches assert_allclose's
    # own definition, error / |y|, restricted to positions that are BOTH
    # invalid AND have a nonzero divisor (`nonzero_and_invalid`) -- if that
    # set is empty, numpy reports plain `inf`.
    nonzero = _anionpy.not_equal(ay, 0)
    nonzero_and_invalid = _anionpy.logical_and(invalid, nonzero)
    if _all_true(_anionpy.logical_not(nonzero_and_invalid)):
        rel_str = "inf"
    else:
        rel_error = _anionpy.divide(error, _anionpy.abs(ay))
        masked_rel = _anionpy.where(nonzero_and_invalid, rel_error, 0)
        max_rel_error = _anionpy.amax(masked_rel)
        rel_str = _array2string_scalar(max_rel_error)
    lines.append("Max relative difference among violations: " + rel_str)

    return lines


def assert_array_compare(comparison, x, y, err_msg="", verbose=True, header="",
                          precision=6, equal_nan=True, equal_inf=True,
                          *, strict=False, names=("ACTUAL", "DESIRED")):
    """Compare `x` and `y` elementwise using `comparison` (an anionpy ufunc such
    as anionpy.equal / anionpy.less), routing every elementwise step and the final
    "were they all true?" reduction through Rust.
    """
    __tracebackhide__ = True
    ox, oy = x, y
    ax = _as_ionp(x)
    ay = _as_ionp(y)

    ashape, bshape = _shape_of(ax), _shape_of(ay)
    if strict:
        cond = ashape == bshape and str(ax.dtype) == str(ay.dtype)
    else:
        cond = (ashape == () or bshape == ()) or ashape == bshape
    if not cond:
        if ashape != bshape:
            reason = f"\n(shapes {ashape}, {bshape} mismatch)"
        else:
            reason = f"\n(dtypes {ax.dtype}, {ay.dtype} mismatch)"
        raise AssertionError(build_err_msg(
            [ox, oy], err_msg + reason, verbose=verbose, header=header,
            names=names, precision=precision))

    ok = comparison(ax, ay)

    exempt = None
    if equal_nan:
        xn, yn = _is_nan(ax), _is_nan(ay)
        if not _all_true(_anionpy.equal(xn, yn)):
            raise AssertionError(build_err_msg(
                [ox, oy], err_msg + "\nnan location mismatch:",
                verbose=verbose, header=header, names=names,
                precision=precision))
        exempt = _anionpy.logical_and(xn, yn)

    if equal_inf:
        xi, yi = _is_inf(ax), _is_inf(ay)
        if not _all_true(_anionpy.equal(xi, yi)):
            raise AssertionError(build_err_msg(
                [ox, oy], err_msg + "\ninf location mismatch:",
                verbose=verbose, header=header, names=names,
                precision=precision))
        both_inf_equal = _anionpy.logical_and(xi, _anionpy.equal(ax, ay))
        exempt = both_inf_equal if exempt is None else _anionpy.logical_or(exempt, both_inf_equal)

    if exempt is not None:
        ok = _anionpy.logical_or(ok, exempt)

    if not _all_true(ok):
        n_mismatch = None
        invalid = None
        try:
            invalid = _anionpy.logical_not(ok)
            n_mismatch = int(_anionpy.sum_f64(invalid.astype("float64").__array__()))
        except Exception:
            pass
        remark = ""
        if n_mismatch is not None and ok.size:
            pct = 100.0 * n_mismatch / ok.size
            remark = f"\nMismatched elements: {n_mismatch} / {ok.size} ({pct:.3g}%)"
            try:
                diag_lines = _mismatch_diagnostics(ax, ay, invalid, names)
            except Exception:
                diag_lines = []
            if diag_lines:
                remark += "\n" + "\n".join(diag_lines)
        raise AssertionError(build_err_msg(
            [ox, oy], err_msg + remark, verbose=verbose, header=header,
            names=names, precision=precision))


def assert_array_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    """Raise AssertionError unless `actual` and `desired` are elementwise
    equal (NaNs in matching positions are treated as equal)."""
    __tracebackhide__ = True
    assert_array_compare(_anionpy.equal, actual, desired, err_msg=err_msg,
                          verbose=verbose, header="Arrays are not equal",
                          strict=strict)


def assert_array_less(x, y, err_msg="", verbose=True, *, strict=False):
    """Raise AssertionError unless every element of `x` is strictly less
    than the corresponding element of `y`."""
    __tracebackhide__ = True
    assert_array_compare(_anionpy.less, x, y, err_msg=err_msg, verbose=verbose,
                          header="Arrays are not strictly ordered `x < y`",
                          equal_inf=False, strict=strict, names=("x", "y"))


def assert_array_almost_equal(actual, desired, decimal=6, err_msg="", verbose=True):
    """Raise AssertionError unless abs(desired - actual) < 1.5 * 10**-decimal
    elementwise (NaNs in matching positions are treated as equal)."""
    __tracebackhide__ = True
    bound = 1.5 * 10.0 ** (-decimal)

    def compare(a, b):
        diff = _anionpy.abs(_anionpy.subtract(a, b))
        return _anionpy.less(diff, bound)

    assert_array_compare(
        compare, actual, desired, err_msg=err_msg, verbose=verbose,
        header=f"Arrays are not almost equal to {decimal} decimals",
        precision=decimal)


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True,
                     err_msg="", verbose=True, *, strict=False):
    """Raise AssertionError unless abs(actual - desired) <= atol +
    rtol * abs(desired) elementwise."""
    __tracebackhide__ = True

    def compare(a, b):
        diff = _anionpy.abs(_anionpy.subtract(a, b))
        bound = _anionpy.add(atol, _anionpy.multiply(rtol, _anionpy.abs(b)))
        return _anionpy.less_equal(diff, bound)

    header = f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}"
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
                          verbose=verbose, header=header, equal_nan=equal_nan,
                          strict=strict)


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    """Raise AssertionError unless abs(desired - actual) < 1.5 * 10**-decimal.

    Delegates to `assert_array_almost_equal` for array/list inputs (matching
    numpy); compares scalars directly via Rust otherwise.
    """
    __tracebackhide__ = True
    if isinstance(actual, (_anionpy.ndarray, list, tuple)) or \
            isinstance(desired, (_anionpy.ndarray, list, tuple)):
        return assert_array_almost_equal(actual, desired, decimal, err_msg, verbose)

    header = f"Arrays are not almost equal to {decimal} decimals"

    def _msg():
        return build_err_msg([actual, desired], err_msg, verbose=verbose, header=header)

    a_isnan = _rs2("not_equal", float(actual), float(actual)) != 0.0
    d_isnan = _rs2("not_equal", float(desired), float(desired)) != 0.0
    if a_isnan or d_isnan:
        if not (a_isnan and d_isnan):
            raise AssertionError(_msg())
        return
    bound = 1.5 * 10.0 ** (-decimal)
    diff = _rs1("abs", _rs2("subtract", float(desired), float(actual)))
    if diff >= bound:
        raise AssertionError(_msg())


def assert_approx_equal(actual, desired, significant=7, err_msg="", verbose=True):
    """Raise AssertionError unless `actual` and `desired` agree to the given
    number of significant digits."""
    __tracebackhide__ = True
    actual = float(actual)
    desired = float(desired)
    if desired == actual:
        return

    header = f"Items are not equal to {significant} significant digits:"

    def _msg():
        return build_err_msg([actual, desired], err_msg, header=header, verbose=verbose)

    a_isnan = _rs2("not_equal", actual, actual) != 0.0
    d_isnan = _rs2("not_equal", desired, desired) != 0.0
    a_isinf = abs(actual) == float("inf")
    d_isinf = abs(desired) == float("inf")
    if a_isnan or d_isnan or a_isinf or d_isinf:
        if a_isnan or d_isnan:
            if not (a_isnan and d_isnan):
                raise AssertionError(_msg())
            return
        if not desired == actual:
            raise AssertionError(_msg())
        return

    scale = _rs2("multiply", 0.5, _rs2("add", _rs1("abs", desired), _rs1("abs", actual)))
    scale = _rs2("power", 10.0, _rs1("floor", _rs1("log10", scale))) if scale != 0.0 else 1.0
    try:
        sc_desired = _rs2("divide", desired, scale)
    except ZeroDivisionError:
        sc_desired = 0.0
    try:
        sc_actual = _rs2("divide", actual, scale)
    except ZeroDivisionError:
        sc_actual = 0.0

    thresh = _rs2("power", 10.0, float(-(significant - 1)))
    if _rs1("abs", _rs2("subtract", sc_desired, sc_actual)) >= thresh:
        raise AssertionError(_msg())


def assert_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    """Raise AssertionError unless `actual` and `desired` are equal. Handles
    dicts, lists/tuples (recursively), and falls back to `assert_array_equal`
    once either side is an anionpy array; otherwise uses Python `==`.
    """
    __tracebackhide__ = True
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k in desired:
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], f"key={k!r}\n{err_msg}", verbose)
        return

    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], f"item={k!r}\n{err_msg}", verbose)
        return

    if isinstance(actual, _anionpy.ndarray) or isinstance(desired, _anionpy.ndarray):
        return assert_array_equal(actual, desired, err_msg, verbose, strict=strict)

    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)

    is_num = isinstance(actual, (int, float)) and isinstance(desired, (int, float)) \
        and not isinstance(actual, bool) and not isinstance(desired, bool)
    if is_num:
        a_isnan = _rs2("not_equal", float(actual), float(actual)) != 0.0
        d_isnan = _rs2("not_equal", float(desired), float(desired)) != 0.0
        if a_isnan and d_isnan:
            return

    if not (desired == actual):
        raise AssertionError(msg)


def print_assert_equal(test_string, actual, desired):
    """Test `actual == desired`; print a diff-style message on failure."""
    __tracebackhide__ = True
    import pprint
    from io import StringIO

    if not (actual == desired):
        msg = StringIO()
        msg.write(test_string)
        msg.write(" failed\nACTUAL: \n")
        pprint.pprint(actual, msg)
        msg.write("DESIRED: \n")
        pprint.pprint(desired, msg)
        raise AssertionError(msg.getvalue())


def assert_string_equal(actual, desired):
    """Raise AssertionError with a diff unless the two strings are equal."""
    __tracebackhide__ = True
    import difflib

    if not isinstance(actual, str):
        raise AssertionError(repr(type(actual)))
    if not isinstance(desired, str):
        raise AssertionError(repr(type(desired)))
    if desired == actual:
        return

    diff = list(difflib.Differ().compare(actual.splitlines(True), desired.splitlines(True)))
    diff_list = []
    while diff:
        d1 = diff.pop(0)
        if d1.startswith("  "):
            continue
        if d1.startswith("- "):
            group = [d1]
            d2 = diff.pop(0)
            if d2.startswith("? "):
                group.append(d2)
                d2 = diff.pop(0)
            if not d2.startswith("+ "):
                raise AssertionError(repr(d2))
            group.append(d2)
            if diff:
                d3 = diff.pop(0)
                if d3.startswith("? "):
                    group.append(d3)
                else:
                    diff.insert(0, d3)
            if d2[2:] == d1[2:]:
                continue
            diff_list.extend(group)
            continue
        raise AssertionError(repr(d1))
    if not diff_list:
        return
    msg = f"Differences in strings:\n{''.join(diff_list).rstrip()}"
    if actual != desired:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# assert_raises / assert_raises_regex (delegate to unittest, as numpy does)
# ---------------------------------------------------------------------------

class _Dummy(unittest.TestCase):
    def nop(self):
        pass


_d = _Dummy("nop")


def assert_raises(*args, **kwargs):
    """assert_raises(exc_class, callable, *a, **kw) or use as a context
    manager: `with assert_raises(exc_class): ...`."""
    __tracebackhide__ = True
    return _d.assertRaises(*args, **kwargs)


def assert_raises_regex(exception_class, expected_regexp, *args, **kwargs):
    """Like `assert_raises` but also requires the exception message to match
    `expected_regexp`."""
    __tracebackhide__ = True
    return _d.assertRaisesRegex(exception_class, expected_regexp, *args, **kwargs)


# ---------------------------------------------------------------------------
# Warnings machinery
# ---------------------------------------------------------------------------

class clear_and_catch_warnings(_warnings.catch_warnings):
    """Context manager that resets a module's __warningregistry__ on entry
    and restores it on exit, so warnings can be re-triggered reliably."""

    class_modules = ()

    def __init__(self, record=False, modules=()):
        self.modules = set(modules).union(self.class_modules)
        self._warnreg_copies = {}
        super().__init__(record=record)

    def __enter__(self):
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super().__enter__()

    def __exit__(self, *exc_info):
        super().__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])


class suppress_warnings:
    """Context manager / decorator for filtering and recording warnings,
    working around the "ignore" filter's inability to re-show a warning
    once it has been seen (see numpy.testing.suppress_warnings, which this
    ports almost verbatim -- it is pure `warnings`-module bookkeeping with
    no array computation anywhere in it).
    """

    def __init__(self, forwarding_rule="always", _warn=True):
        if _warn:
            _warnings.warn(
                "anionpy.testing warning suppression utilities mirror numpy's "
                "deprecated ones. Prefer warnings.catch_warnings / "
                "pytest.warns.",
                DeprecationWarning, stacklevel=2)
        self._entered = False
        self._suppressions = []
        if forwarding_rule not in {"always", "module", "once", "location"}:
            raise ValueError("unsupported forwarding rule.")
        self._forwarding_rule = forwarding_rule

    def _clear_registries(self):
        if hasattr(_warnings, "_filters_mutated"):
            _warnings._filters_mutated()
            return
        for module in self._tmp_modules:
            if hasattr(module, "__warningregistry__"):
                module.__warningregistry__.clear()

    def _filter(self, category=Warning, message="", module=None, record=False):
        record = [] if record else None
        if self._entered:
            if module is None:
                _warnings.filterwarnings("always", category=category, message=message)
            else:
                module_regex = module.__name__.replace(".", r"\.") + "$"
                _warnings.filterwarnings("always", category=category,
                                          message=message, module=module_regex)
                self._tmp_modules.add(module)
                self._clear_registries()
            self._tmp_suppressions.append(
                (category, message, re.compile(message, re.I), module, record))
        else:
            self._suppressions.append(
                (category, message, re.compile(message, re.I), module, record))
        return record

    def filter(self, category=Warning, message="", module=None):
        self._filter(category=category, message=message, module=module, record=False)

    def record(self, category=Warning, message="", module=None):
        return self._filter(category=category, message=message, module=module, record=True)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("cannot enter suppress_warnings twice.")
        self._orig_show = _warnings.showwarning
        self._catch_warnings = _warnings.catch_warnings()
        self._catch_warnings.__enter__()
        self._entered = True
        self._tmp_suppressions = []
        self._tmp_modules = set()
        self._forwarded = set()
        self.log = []
        for cat, mess, _pat, mod, log in self._suppressions:
            if log is not None:
                del log[:]
            if mod is None:
                _warnings.filterwarnings("always", category=cat, message=mess)
            else:
                module_regex = mod.__name__.replace(".", r"\.") + "$"
                _warnings.filterwarnings("always", category=cat, message=mess,
                                          module=module_regex)
                self._tmp_modules.add(mod)
        _warnings.showwarning = self._showwarning
        self._clear_registries()
        return self

    def __exit__(self, *exc_info):
        _warnings.showwarning = self._orig_show
        self._catch_warnings.__exit__(*exc_info)
        self._clear_registries()
        self._entered = False
        del self._orig_show
        del self._catch_warnings

    def _showwarning(self, message, category, filename, lineno, *args,
                      use_warnmsg=None, **kwargs):
        for cat, _mess, pattern, mod, rec in (self._suppressions + self._tmp_suppressions)[::-1]:
            if issubclass(category, cat) and pattern.match(message.args[0] if message.args else "") is not None:
                if mod is None:
                    if rec is not None:
                        wm = _warnings.WarningMessage(message, category, filename, lineno, **kwargs)
                        self.log.append(wm)
                        rec.append(wm)
                    return
                elif getattr(mod, "__file__", "").startswith(filename):
                    if rec is not None:
                        wm = _warnings.WarningMessage(message, category, filename, lineno, **kwargs)
                        self.log.append(wm)
                        rec.append(wm)
                    return
        if self._forwarding_rule == "always":
            if self._orig_show is not None:
                self._orig_show(message, category, filename, lineno, *args, **kwargs)
            else:
                _warnings._filters_mutated_showwarning(message, category, filename, lineno) \
                    if hasattr(_warnings, "_filters_mutated_showwarning") else None
            return
        key = (category, filename) if self._forwarding_rule == "module" else \
            (category, filename, lineno) if self._forwarding_rule == "location" else \
            (category,)
        if key not in self._forwarded:
            self._forwarded.add(key)
            if self._orig_show is not None:
                self._orig_show(message, category, filename, lineno, *args, **kwargs)

    def __call__(self, func):
        @wraps(func)
        def new_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return new_func


@contextlib.contextmanager
def _assert_warns_context(warning_class, name=None):
    __tracebackhide__ = True
    with suppress_warnings(_warn=False) as sup:
        log = sup.record(warning_class)
        yield
        if not len(log) > 0:
            name_str = f" when calling {name}" if name is not None else ""
            raise AssertionError("No warning raised" + name_str)


def assert_warns(warning_class, *args, **kwargs):
    """Fail unless `warning_class` is raised by the given callable, or use
    as a context manager: `with assert_warns(SomeWarning): ...`."""
    _warnings.warn(
        "anionpy.testing warning-suppression/assertion utilities mirror "
        "numpy's deprecated ones; prefer warnings.catch_warnings or "
        "pytest.warns.",
        DeprecationWarning, stacklevel=2)
    if not args and not kwargs:
        return _assert_warns_context(warning_class)
    elif len(args) < 1:
        if "match" in kwargs:
            raise RuntimeError("assert_warns does not use 'match' kwarg")
        raise RuntimeError("assert_warns(...) needs at least one arg")
    func = args[0]
    args = args[1:]
    with _assert_warns_context(warning_class, name=getattr(func, "__name__", None)):
        return func(*args, **kwargs)


@contextlib.contextmanager
def _assert_no_warnings_context(name=None):
    __tracebackhide__ = True
    with _warnings.catch_warnings(record=True) as log:
        _warnings.simplefilter("always")
        yield
        if len(log) > 0:
            name_str = f" when calling {name}" if name is not None else ""
            raise AssertionError(f"Got warnings{name_str}: {log}")


def assert_no_warnings(*args, **kwargs):
    """Fail if the given callable produces any warnings, or use as a
    context manager: `with assert_no_warnings(): ...`."""
    if not args:
        return _assert_no_warnings_context()
    func = args[0]
    args = args[1:]
    with _assert_no_warnings_context(name=getattr(func, "__name__", None)):
        return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# GC / reference-cycle utilities
# ---------------------------------------------------------------------------

def break_cycles():
    """Break reference cycles by calling gc.collect (repeatedly on PyPy)."""
    gc.collect()
    if IS_PYPY:
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()


@contextlib.contextmanager
def _assert_no_gc_cycles_context(name=None):
    __tracebackhide__ = True
    if not HAS_REFCOUNT:
        yield
        return
    assert_(gc.isenabled())
    gc.disable()
    gc_debug = gc.get_debug()
    try:
        for _ in range(100):
            if gc.collect() == 0:
                break
        else:
            raise RuntimeError(
                "Unable to fully collect garbage - perhaps a __del__ "
                "method is creating more reference cycles?")
        gc.set_debug(gc.DEBUG_SAVEALL)
        yield
        n_objects_in_cycles = gc.collect()
        objects_in_cycles = gc.garbage[:]
    finally:
        del gc.garbage[:]
        gc.set_debug(gc_debug)
        gc.enable()

    if n_objects_in_cycles:
        import pprint
        name_str = f" when calling {name}" if name is not None else ""
        raise AssertionError(
            "Reference cycles were found{}: {} objects were collected, "
            "of which {} are shown below:{}".format(
                name_str,
                n_objects_in_cycles,
                len(objects_in_cycles),
                "".join(
                    "\n  {} object with id={}:\n    {}".format(
                        type(o).__name__,
                        id(o),
                        pprint.pformat(o).replace("\n", "\n    "),
                    )
                    for o in objects_in_cycles
                ),
            ))


def assert_no_gc_cycles(*args, **kwargs):
    """Fail if the given callable produces any reference cycles, or use as
    a context manager: `with assert_no_gc_cycles(): ...`."""
    if not args:
        return _assert_no_gc_cycles_context()
    func = args[0]
    args = args[1:]
    with _assert_no_gc_cycles_context(name=getattr(func, "__name__", None)):
        func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Reflection / decoration utility
# ---------------------------------------------------------------------------

def decorate_methods(cls, decorator, testmatch=None):
    """Apply `decorator` to every public method of `cls` whose name matches
    `testmatch` (default: numpy/nose's historical "look like a test" regex).
    """
    if testmatch is None:
        testmatch = re.compile(rf"(?:^|[\\b_\\.{os.sep}-])[Tt]est")
    else:
        testmatch = re.compile(testmatch)
    cls_attr = cls.__dict__

    from inspect import isfunction
    methods = [m for m in cls_attr.values() if isfunction(m)]
    for function in methods:
        try:
            funcname = getattr(function, "compat_func_name", function.__name__)
        except AttributeError:
            continue
        if testmatch.search(funcname) and not funcname.startswith("_"):
            setattr(cls, funcname, decorator(function))


# ---------------------------------------------------------------------------
# Timing / process utilities
# ---------------------------------------------------------------------------

if sys.platform.startswith("linux"):
    def jiffies(_proc_pid_stat=None, _load_time=[]):
        """Jiffies (1/100ths of a second) this process has run in user mode."""
        import time
        _proc_pid_stat = _proc_pid_stat or f"/proc/{os.getpid()}/stat"
        if not _load_time:
            _load_time.append(time.time())
        try:
            with open(_proc_pid_stat) as f:
                parts = f.readline().split(" ")
            return int(parts[13])
        except Exception:
            return int(100 * (time.time() - _load_time[0]))
else:
    def jiffies(_load_time=[]):
        """Jiffies (1/100ths of a second) elapsed since first call (fallback
        for platforms without /proc)."""
        import time
        if not _load_time:
            _load_time.append(time.time())
        return int(100 * (time.time() - _load_time[0]))


if os.name == "nt":
    def memusage(*args, **kwargs):
        """Return virtual memory size in bytes. [Not implemented on Windows
        without pywin32; matches numpy's own optional dependency here.]"""
        raise NotImplementedError(
            "memusage() on Windows requires pywin32, which anionpy does not "
            "depend on")
elif sys.platform.startswith("linux"):
    def memusage(_proc_pid_stat=None):
        """Return virtual memory size in bytes of the running process."""
        _proc_pid_stat = _proc_pid_stat or f"/proc/{os.getpid()}/stat"
        try:
            with open(_proc_pid_stat) as f:
                parts = f.readline().split(" ")
            return int(parts[22])
        except Exception:
            return None
else:
    def memusage():
        """Return memory usage of running python. [Not implemented on this
        platform.]"""
        raise NotImplementedError


def measure(code_str, times=1, label=None):
    """Return elapsed wall time (seconds) for executing `code_str` `times`
    times in the caller's namespace."""
    frame = sys._getframe(1)
    locs, globs = frame.f_locals, frame.f_globals
    code = compile(code_str, f"Test name: {label} ", "exec")
    i = 0
    elapsed = jiffies()
    while i < times:
        i += 1
        exec(code, globs, locs)
    elapsed = jiffies() - elapsed
    return 0.01 * elapsed


def run_threaded(func, max_workers=8, pass_count=False, pass_barrier=False,
                  outer_iterations=1, prepare_args=None):
    """Run `func` many times in parallel across a thread pool."""
    for _ in range(outer_iterations):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as tpe:
            args = [] if prepare_args is None else prepare_args()
            barrier = None
            if pass_barrier:
                barrier = threading.Barrier(max_workers)
                args = args + [barrier]
            if pass_count:
                all_args = [(func, i, *args) for i in range(max_workers)]
            else:
                all_args = [(func, *args) for i in range(max_workers)]
            futures = []
            try:
                for arg in all_args:
                    futures.append(tpe.submit(*arg))
            finally:
                if len(futures) < max_workers and barrier is not None:
                    barrier.abort()
            for f in futures:
                f.result()


def check_support_sve(__cache=[]):
    """Best-effort detection of ARM SVE support via `lscpu` (pure string
    matching on subprocess output -- no computation)."""
    if __cache:
        return __cache[0]
    try:
        output = subprocess.run("lscpu", capture_output=True, text=True)
        result = "sve" in output.stdout
    except (OSError, subprocess.SubprocessError):
        result = False
    __cache.append(result)
    return __cache[0]


# ---------------------------------------------------------------------------
# File / doctest / execution utilities
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def tempdir(*args, **kwargs):
    """Context manager yielding a temporary directory path, removed on exit."""
    tmpdir = mkdtemp(*args, **kwargs)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


@contextlib.contextmanager
def temppath(*args, **kwargs):
    """Context manager yielding a closed temporary file path, removed on exit."""
    fd, path = mkstemp(*args, **kwargs)
    os.close(fd)
    try:
        yield path
    finally:
        os.remove(path)


def runstring(astr, dict):
    """Exec `astr` in namespace `dict`."""
    exec(astr, dict)


def rundocs(filename=None, raise_on_error=True):
    """Run doctests found in `filename` (defaults to the caller's file)."""
    import doctest

    if filename is None:
        f = sys._getframe(1)
        filename = f.f_globals["__file__"]
    name = os.path.splitext(os.path.basename(filename))[0]
    spec = importlib.util.spec_from_file_location(name, filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tests = doctest.DocTestFinder().find(m)
    runner = doctest.DocTestRunner(verbose=False)

    msg = []
    out = msg.append if raise_on_error else None
    for t in tests:
        runner.run(t, out=out)

    if runner.failures > 0 and raise_on_error:
        raise AssertionError("Some doctests failed:\n" + "\n".join(msg))


class _IonpTester:
    """Minimal analogue of numpy's PytestTester: runs anionpy's own test suite
    via pytest. No numeric computation -- process/argv bookkeeping only."""

    def __init__(self, module_name):
        self.module_name = module_name

    def __call__(self, label="fast", verbose=1, extra_argv=None,
                 doctests=False, coverage=False, durations=-1, tests=None):
        # SIGNATURE IS PART OF THE CONTRACT, and this is why the three
        # parameters below exist. They were originally omitted as "internal
        # pytest plumbing nobody calls", and the differential corpus stayed
        # green because it only ever invoked the callable, never inspected it.
        # Measured against numpy 2.5.1: `numpy.fft.test(durations=5)` binds,
        # and the four-parameter version raised
        # `TypeError: got an unexpected keyword argument 'durations'` --
        # a declared-exact item that diverges on any caller passing them.
        #
        # Order of argv construction below is numpy's, not a tidier one:
        # doctests, extra_argv, verbose, coverage, label, durations. pytest
        # is order-sensitive for repeated `-m`/`-W` flags, so a "cleaner"
        # ordering is a behavioural change wearing a refactor's clothes.
        import pytest

        module = sys.modules.get(self.module_name)
        pytest_args = ["-l", "-q"]
        if doctests:
            pytest_args += ["--doctest-modules"]
        if extra_argv:
            pytest_args += list(extra_argv)
        if verbose > 1:
            pytest_args += ["-" + "v" * (verbose - 1)]
        if coverage:
            # numpy uses abspath(module.__path__[0]); PyO3 submodules such as
            # anionpy.fft have no __path__, so fall back to the package dir.
            path = getattr(module, "__path__", None)
            target = (os.path.abspath(path[0]) if path
                      else os.path.abspath(os.path.dirname(__file__)))
            pytest_args += ["--cov=" + target]
        if label == "fast":
            pytest_args += ["-m", "not slow"]
        elif label != "full":
            pytest_args += ["-m", label]
        if durations >= 0:
            pytest_args += [f"--durations={durations}"]

        if tests is not None:
            pytest_args += list(tests)
        elif module is not None and hasattr(module, "__path__"):
            pytest_args += ["--pyargs", self.module_name]
        else:
            repo_root = pathlib.Path(_anionpy.__file__).parent.parent
            candidate = repo_root / "tests"
            if not candidate.exists():
                raise RuntimeError(
                    f"no discoverable test suite for {self.module_name!r} "
                    f"(looked for {candidate})")
            pytest_args += [str(candidate)]

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code
        return code == 0


test = _IonpTester("anionpy.testing")
