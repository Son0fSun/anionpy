"""Differential ItemSpecs for the `anionpy.testing` block (50 numpy_surface.json
items under the `testing.` prefix).

Owned exclusively by this task (see the task brief): this file, anionpy/testing.py,
and any new ionp-py/src/ helper. Mirrors linalg_cases.py's/inplace_cases.py's
pattern exactly -- a `TESTING_SPECS: dict[str, ItemSpec]` merged into
registry.REGISTRY by registry.py itself (registry.py is NOT owned by this task;
see the bottom-of-file comment for the exact merge block that needs to land
there, matching the LINALG_SPECS/INPLACE_SPECS collision-checked merge already
present).

Every ItemSpec here is `kind="custom"`: numpy.testing's surface is not
ufunc/method/binary_op shaped (see registry.py's ItemSpec.kind docstring), so
each item supplies its own (label, args, kwargs) corpus via `custom_cases`,
run against both `numpy.testing.X(*args, **kwargs)` and
`anionpy.testing.X(*args, **kwargs)` by harness.run_case -- which already does
the exception-type-match comparison this task's verification requirement
asks for ("both a passing case and a case that must raise, confirming
exception type matches numpy's"): if one side raises and the other doesn't,
or they raise different exception types, that IS a failing case; no special
casing needed here for that.

All 48 implemented items are declared bit-exact (no atol/rtol at all, since
every one of these is a boolean-outcome/exception-outcome assertion or a
platform-introspection value, never a tolerance-graded numeric result --
ItemSpec.__post_init__ structurally refuses non-zero atol/rtol regardless).

Two items (`assert_array_almost_equal_nulp`, `assert_array_max_ulp`) are
absent from `anionpy.testing` (see that module's docstring: both require
float-bit reinterpretation with no Rust primitive) and therefore have NO
ItemSpec here -- run.py's build_cases would have nothing to call in either
case, and the honest state is "missing from the registry", not a fabricated
entry that immediately fails.

Design notes on what's NOT registered as a numpy-vs-ionp array/value-equality
ItemSpec, and why (each of these is honest handling, not a gap swept under
the rug -- see this task's final report for how each was actually verified):

  - `testing.assert_array_compare` -- its first argument is a *comparison
    callable* (e.g. `anionpy.equal`); passed as a plain positional argument, it
    is NOT converted by the custom-kind ndarray converter (only
    numpy.ndarray *arguments* are converted, not callables), so a single
    comparison function can't be handed identically to both numpy's engine
    (which calls it with numpy arrays) and anionpy's engine (which calls it
    with anionpy arrays) without either failing structurally or silently
    testing the wrong thing. It is exercised transitively (every
    `assert_array_equal`/`assert_array_less`/`assert_array_almost_equal`
    case below calls straight through it) and was directly, exhaustively
    verified against real numpy earlier in this task (26+ manual probes:
    NaN-position matching, inf-sign matching, shape/dtype mismatch,
    broadcasting).
  - `testing.jiffies` / `testing.memusage` -- process-clock/OS-memory
    introspection; numpy's own and anionpy's own values are two independent
    reads of a live, monotonically-changing counter taken at different
    wall-clock instants, so equality is not the right assertion (it would
    be either flaky-false or a tautological same-process check). Registered
    below with a *type/contract* adapter (returns an int, or raises
    NotImplementedError on non-Linux/non-nt, matching numpy's own fallback
    on this darwin box) rather than a value-equality one.
  - `testing.check_support_sve` -- both sides call the same real `lscpu`
    subprocess; registered with a boolean-result adapter (not raw
    passthrough) for the same "two independent OS probes" reason as jiffies.
  - Platform-fact constants -- 2026-08-01 falsifiability rework (see the
    falsifiability audit in this task's final report): `verbose`, `IS_PYPY`,
    `IS_PYSTON`, `IS_WASM`, `IS_MUSL`, `IS_64BIT`, `HAS_REFCOUNT`,
    `NOGIL_BUILD` are genuinely shared same-machine/same-interpreter facts,
    so they are now graded by EXACT VALUE AND TYPE equality against real
    `numpy.testing.<NAME>` (`_direct_value_probe`), not the isinstance-only
    check that shipped originally (which passed for any value of the right
    type). `BLAS_SUPPORTS_FPE` / `HAS_LAPACK64` are graded against a fixed,
    architecturally-justified expected constant (`False`) instead of numpy's
    live value, because ionp-core has zero BLAS/LAPACK dependency by design
    while numpy in this venv genuinely links a real ILP64 LAPACK -- equality
    to numpy's value would force the correct anionpy implementation to fail.
    `IS_EDITABLE` / `IS_INSTALLED` / `NUMPY_ROOT` describe *anionpy's own*
    build/install (anionpy/testing.py's own docstring/comments establish this),
    not copies of numpy's values, so they are graded against an
    INDEPENDENTLY RECOMPUTED ground truth (`_recompute_ionp_install_facts`,
    a separate importlib.metadata-based code path, not a second read of the
    same live attribute) rather than either numpy's differing value or a
    circular re-read of `anionpy.testing` itself.
  - `testing.SkipTest` / `testing.TestCase` -- both numpy's and anionpy's are
    literally `unittest.case.SkipTest` / `unittest.TestCase` (verified via
    `is`-identity below): re-exports of the same stdlib object, not
    reimplementations, so the correct differential assertion is object
    identity to the stdlib original, not cross-module `==`.
  - `testing.IgnoreException` / `testing.KnownFailureException` -- plain
    marker `Exception` subclasses with no behaviour beyond existing;
    registered with a construct-and-introspect adapter (returns
    `(cls.__name__, is Exception subclass, str(instance), caught as itself)`,
    all plain non-array values) rather than raw instances (which would
    spuriously fail cross-module `type(...) is type(...)` in
    harness.compare_values's scalar_like path, since
    `anionpy.testing.IgnoreException` and `numpy.testing.IgnoreException` are,
    correctly, two distinct classes) or a bare isinstance/str check (2026-
    08-01 falsifiability fix: an unrelated `class _Unrelated(Exception)`
    swapped in passed both of those trivially; `cls.__name__` is the
    discriminator that actually catches it).
  - `testing.clear_and_catch_warnings` / `testing.suppress_warnings` --
    stateful context-manager classes; registered with a behavioural adapter
    (does entering/exiting/filtering actually suppress a warning?) rather
    than an object-identity or attribute-diff comparison.
  - `testing.test` -- a `PytestTester`-alike callable object, not a pure
    function; registered with a type/callability adapter.
  - `testing.build_err_msg` -- registered directly (kind="custom",
    numpy_path/ionp_path), but ONLY with plain Python list/str arguments
    (never numpy.ndarray), because `arrays` elements are formatted via
    `repr()`: a numpy.ndarray argument would be auto-converted to an
    anionpy.ndarray by the custom-kind converter before anionpy's side ever sees
    it, so its `repr()` would legitimately differ from numpy's own
    ndarray repr for reasons that have nothing to do with a real bug. Lists
    and strings pass through the converter unchanged (see
    make_ionp_array_converter's docstring), so their `repr()` is
    byte-for-byte identical input on both sides -- a real, meaningful
    differential test, not a weakened one.
"""
from __future__ import annotations

import re

import numpy as np

from registry import ItemSpec

# 2026-08-01: registry.ItemSpec.resolve_ionp()'s dotted-path branch (used by
# every plain `_reg(...)` item below -- assert_*, build_err_msg) resolves
# via `getattr(anionpy, "testing")` then `getattr(that, "<name>")`. `anionpy`
# itself does NOT import the `testing` submodule at package init (and
# anionpy/__init__.py is not owned by this task, so that can't be changed
# here) -- Python only registers a submodule as an attribute of its parent
# package once *something* has done `import anionpy.testing` somewhere in the
# process. Every item-specific probe function below does that import
# lazily, inside itself, which is fine for adapter-based items but is too
# late for the plain dotted-path items if THEY happen to be the first
# testing.* spec evaluated in a fresh process -- resolve_ionp() would see
# no `testing` attribute yet and report a false "absent" (confirmed live:
# in a fresh interpreter, `hasattr(anionpy, "testing")` is False until this
# import runs; a plain `_reg`-based item evaluated first then reports
# "not implemented on anionpy (absent)" for a function that plainly exists).
# Importing it here, at load time of the one file that owns every
# testing.* ItemSpec, guarantees it happens before any of them can be
# evaluated, regardless of dict/registration order.
try:
    import anionpy.testing  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Small helpers used as case payloads (must be plain, picklable-shaped
# module-level callables -- assert_raises/assert_warns/... call them by
# reference from both the numpy side and the anionpy side).
# ---------------------------------------------------------------------------

def _raise_valueerror():
    raise ValueError("boom")


def _raise_valueerror_boom():
    raise ValueError("boom text")


def _raise_nothing():
    return 42


def _warn_user():
    import warnings
    warnings.warn("heads up", UserWarning)


def _do_nothing():
    return 1


# ---------------------------------------------------------------------------
# assert_* items: plain kind="custom" resolution against `testing.<name>`
# on both sides (numpy_path/ionp_path default to `name`, which already IS
# "testing.<name>" for every item below since ItemSpec.name is the full
# numpy_surface.json key).
# ---------------------------------------------------------------------------

TESTING_SPECS: dict[str, ItemSpec] = {}


def _reg(name, cases_fn, **kwargs):
    TESTING_SPECS[name] = ItemSpec(
        name=name, kind="custom", custom_cases=cases_fn, **kwargs
    )


_reg("testing.assert_", lambda: [
    ("truthy_passes", (True,), {}),
    ("truthy_nonbool_passes", (1,), {}),
    ("falsy_raises", (False, "boom"), {}),
], scalar_like=True)

_reg("testing.assert_array_equal", lambda: [
    ("equal_int_passes", (np.array([1, 2, 3]), np.array([1, 2, 3])), {}),
    ("nan_same_position_passes",
     (np.array([1.0, np.nan]), np.array([1.0, np.nan])), {}),
    ("nan_position_differs_raises",
     (np.array([np.nan, 1.0]), np.array([1.0, 1.0])), {}),
    ("mismatch_raises",
     (np.array([1, 2, 3]), np.array([1, 2, 4])), {}),
    ("shape_mismatch_raises",
     (np.array([1, 2]), np.array([1, 2, 3])), {}),
], scalar_like=True)

_reg("testing.assert_array_less", lambda: [
    ("strictly_less_passes", (np.array([1.0, 2.0]), np.array([2.0, 3.0])), {}),
    ("not_less_raises", (np.array([2.0, 2.0]), np.array([1.0, 3.0])), {}),
], scalar_like=True)

_reg("testing.assert_array_almost_equal", lambda: [
    ("close_within_decimal_passes",
     (np.array([1.0, 2.0]), np.array([1.0000001, 2.0])), {}),
    ("far_raises", (np.array([1.0, 2.0]), np.array([1.1, 2.0])), {}),
], scalar_like=True)

_reg("testing.assert_allclose", lambda: [
    ("close_passes", (np.array([1.0, 2.0]), np.array([1.0 + 1e-9, 2.0])), {}),
    ("far_raises", (np.array([1.0]), np.array([2.0])), {}),
], scalar_like=True)

_reg("testing.assert_almost_equal", lambda: [
    ("close_scalar_passes", (1.0, 1.0000001), {}),
    ("far_scalar_raises", (1.0, 1.5), {}),
], scalar_like=True)

_reg("testing.assert_approx_equal", lambda: [
    ("close_scalar_passes", (1.0, 1.0000001), {}),
    ("far_scalar_raises", (1.0, 2.0), {}),
], scalar_like=True)

_reg("testing.assert_equal", lambda: [
    ("equal_list_passes", ([1, 2, 3], [1, 2, 3]), {}),
    ("unequal_list_raises", ([1, 2, 3], [1, 2, 4]), {}),
    ("equal_scalar_passes", (3, 3), {}),
    ("unequal_scalar_raises", (3, 4), {}),
    ("none_passes", (None, None), {}),
], scalar_like=True)

_reg("testing.assert_raises", lambda: [
    ("catches_expected_passes", (ValueError, _raise_valueerror), {}),
    ("no_raise_raises", (ValueError, _raise_nothing), {}),
], scalar_like=True)

_reg("testing.assert_raises_regex", lambda: [
    ("regex_matches_passes", (ValueError, "boom", _raise_valueerror_boom), {}),
    ("regex_mismatch_raises", (ValueError, "xyz", _raise_valueerror_boom), {}),
], scalar_like=True)

_reg("testing.assert_warns", lambda: [
    ("warns_passes", (UserWarning, _warn_user), {}),
    ("no_warning_raises", (UserWarning, _do_nothing), {}),
], scalar_like=True)

_reg("testing.assert_no_warnings", lambda: [
    ("silent_passes", (_do_nothing,), {}),
    ("warns_raises", (_warn_user,), {}),
], scalar_like=True)

_reg("testing.assert_string_equal", lambda: [
    ("equal_strings_pass", ("abc", "abc"), {}),
    ("unequal_strings_raise", ("abc", "abd"), {}),
], scalar_like=True)

_reg("testing.print_assert_equal", lambda: [
    ("equal_passes", ("msg", [0, 1], [0, 1]), {}),
    ("unequal_raises", ("msg", [0, 1], [0, 2]), {}),
], scalar_like=True)

_reg("testing.build_err_msg", lambda: [
    ("plain_lists_no_err_msg",
     ([[1, 2], [1, 3]], ""), {}),
    ("plain_lists_with_err_msg_and_header",
     ([[1, 2], [1, 3]], "custom failure text"),
     {"header": "Items are not equal:", "verbose": True,
      "names": ("ACTUAL", "DESIRED"), "precision": 8}),
], scalar_like=True)

# 2026-08-01 falsifiability fix: the previous single case
# (`no_cycles_passes`) only exercised the non-raising branch, so a no-op
# stand-in (which also never raises) was indistinguishable from the real
# cycle-detecting implementation. Add a second case that builds a REAL
# reference cycle inside the probed callable -- assert_no_gc_cycles must
# raise AssertionError there on both sides, graded via the harness's
# exception-type-match path.
def _gc_cycles_body(make_cycle):
    if make_cycle:
        class _Node:
            pass
        a = _Node()
        b = _Node()
        a.other = b
        b.other = a


def _probe_assert_no_gc_cycles_numpy(make_cycle):
    import numpy.testing as npt
    npt.assert_no_gc_cycles(lambda: _gc_cycles_body(make_cycle))
    return "ok"


def _probe_assert_no_gc_cycles_ionp(make_cycle):
    import anionpy.testing as it
    it.assert_no_gc_cycles(lambda: _gc_cycles_body(make_cycle))
    return "ok"


TESTING_SPECS["testing.assert_no_gc_cycles"] = ItemSpec(
    name="testing.assert_no_gc_cycles", kind="custom", scalar_like=True,
    custom_cases=lambda: [
        ("no_cycle_passes", (False,), {}),
        ("real_cycle_raises", (True,), {}),
    ],
    numpy_adapter=lambda make_cycle: _probe_assert_no_gc_cycles_numpy(make_cycle),
    ionp_adapter=lambda make_cycle: _probe_assert_no_gc_cycles_ionp(make_cycle),
)


# 2026-08-01 falsifiability fix: "runs_without_raising" alone can't tell a
# real gc.collect() from a no-op that also never raises. Build a real
# reference cycle, hold a weakref to it with the cyclic garbage collector
# disabled (so nothing collects it automatically), then confirm the weakref
# still resolves BEFORE break_cycles() and no longer resolves AFTER --  a
# no-op leaves it alive in both snapshots.
class _CycleNode:
    pass


def _make_cycle_and_weakref():
    import weakref
    a = _CycleNode()
    b = _CycleNode()
    a.other = b
    b.other = a
    wa = weakref.ref(a)
    del a, b
    return wa


def _probe_break_cycles(mod):
    import gc
    gc.disable()
    try:
        wa = _make_cycle_and_weakref()
        before = wa() is not None
        mod.break_cycles()
        after = wa() is not None
        return (before, after)
    finally:
        gc.enable()


def _probe_break_cycles_numpy():
    import numpy.testing as npt
    return _probe_break_cycles(npt)


def _probe_break_cycles_ionp():
    import anionpy.testing as it
    return _probe_break_cycles(it)


TESTING_SPECS["testing.break_cycles"] = ItemSpec(
    name="testing.break_cycles", kind="custom", scalar_like=True,
    custom_cases=lambda: [("collects_a_real_reference_cycle", (), {})],
    numpy_adapter=lambda: _probe_break_cycles_numpy(),
    ionp_adapter=lambda: _probe_break_cycles_ionp(),
)


# 2026-08-01 falsifiability fix: the previous case never inspected the
# namespace dict runstring is contractually supposed to exec into -- only
# the (always-None) return value was graded, so a no-op that never called
# exec() at all was indistinguishable from the real implementation. Build
# the namespace locally inside the probe and check it actually gained the
# `x = 1 + 1` binding.
def _probe_runstring(mod):
    ns = {}
    ret = mod.runstring("x = 1 + 1", ns)
    return (ret, ns.get("x"))


def _probe_runstring_numpy():
    import numpy.testing as npt
    return _probe_runstring(npt)


def _probe_runstring_ionp():
    import anionpy.testing as it
    return _probe_runstring(it)


TESTING_SPECS["testing.runstring"] = ItemSpec(
    name="testing.runstring", kind="custom", scalar_like=True,
    custom_cases=lambda: [("execs_and_binds_in_namespace", (), {})],
    numpy_adapter=lambda: _probe_runstring_numpy(),
    ionp_adapter=lambda: _probe_runstring_ionp(),
)


# ---------------------------------------------------------------------------
# `tempdir`/`temppath` are context managers, not plain callables: calling
# `testing.tempdir()` just constructs the `@contextlib.contextmanager`
# generator-object wrapper on both sides without entering it, which IS a
# meaningful, side-effect-free equality-shaped probe (both must be entered
# as `with ... as p:` to actually create anything) -- but comparing two
# generator objects for equality is meaningless. Use adapters that actually
# drive the `with` protocol and report only plain (bool, bool) facts: did a
# path come back, does it exist inside the block, is it gone after.
# ---------------------------------------------------------------------------

def _probe_tempdir_numpy():
    import os
    import numpy.testing as npt
    with npt.tempdir() as d:
        existed_inside = os.path.isdir(d)
    return (existed_inside, os.path.isdir(d))


def _probe_tempdir_ionp():
    import os
    import anionpy.testing as it
    with it.tempdir() as d:
        existed_inside = os.path.isdir(d)
    return (existed_inside, os.path.isdir(d))


TESTING_SPECS["testing.tempdir"] = ItemSpec(
    name="testing.tempdir", kind="custom", scalar_like=True,
    custom_cases=lambda: [("enter_exit_lifecycle", (), {})],
    numpy_adapter=lambda: _probe_tempdir_numpy(),
    ionp_adapter=lambda: _probe_tempdir_ionp(),
)


def _probe_temppath_numpy():
    import os
    import numpy.testing as npt
    with npt.temppath() as p:
        existed_inside = os.path.isfile(p)
    return (existed_inside, os.path.isfile(p))


def _probe_temppath_ionp():
    import os
    import anionpy.testing as it
    with it.temppath() as p:
        existed_inside = os.path.isfile(p)
    return (existed_inside, os.path.isfile(p))


TESTING_SPECS["testing.temppath"] = ItemSpec(
    name="testing.temppath", kind="custom", scalar_like=True,
    custom_cases=lambda: [("enter_exit_lifecycle", (), {})],
    numpy_adapter=lambda: _probe_temppath_numpy(),
    ionp_adapter=lambda: _probe_temppath_ionp(),
)


# ---------------------------------------------------------------------------
# jiffies / memusage / check_support_sve -- independent OS/process probes;
# graded on TYPE/CONTRACT, not value equality (see module docstring).
# ---------------------------------------------------------------------------

def _probe_jiffies_numpy():
    import numpy.testing as npt
    return isinstance(npt.jiffies(), int)


def _probe_jiffies_ionp():
    import anionpy.testing as it
    return isinstance(it.jiffies(), int)


TESTING_SPECS["testing.jiffies"] = ItemSpec(
    name="testing.jiffies", kind="custom", scalar_like=True,
    custom_cases=lambda: [("returns_int", (), {})],
    numpy_adapter=lambda: _probe_jiffies_numpy(),
    ionp_adapter=lambda: _probe_jiffies_ionp(),
)


def _probe_memusage_numpy():
    import numpy.testing as npt
    try:
        npt.memusage()
        return "ok"
    except NotImplementedError:
        return "not_implemented"


def _probe_memusage_ionp():
    import anionpy.testing as it
    try:
        it.memusage()
        return "ok"
    except NotImplementedError:
        return "not_implemented"


TESTING_SPECS["testing.memusage"] = ItemSpec(
    name="testing.memusage", kind="custom", scalar_like=True,
    custom_cases=lambda: [("same_platform_contract", (), {})],
    numpy_adapter=lambda: _probe_memusage_numpy(),
    ionp_adapter=lambda: _probe_memusage_ionp(),
)


def _probe_check_support_sve_numpy():
    import numpy.testing as npt
    return isinstance(npt.check_support_sve(), bool)


def _probe_check_support_sve_ionp():
    import anionpy.testing as it
    return isinstance(it.check_support_sve(), bool)


TESTING_SPECS["testing.check_support_sve"] = ItemSpec(
    name="testing.check_support_sve", kind="custom", scalar_like=True,
    custom_cases=lambda: [("returns_bool", (), {})],
    numpy_adapter=lambda: _probe_check_support_sve_numpy(),
    ionp_adapter=lambda: _probe_check_support_sve_ionp(),
)


# ---------------------------------------------------------------------------
# measure / run_threaded / decorate_methods / rundocs -- exercised for
# "does it run to completion without raising, using the observable stdlib
# side effects" rather than a return-value equality (measure returns a
# wall-clock float, run_threaded/decorate_methods/rundocs return None on
# success in both numpy and anionpy).
# ---------------------------------------------------------------------------

def _probe_measure_numpy():
    import numpy.testing as npt
    t = npt.measure("1 + 1", times=2)
    return isinstance(t, float) and t >= 0.0


def _probe_measure_ionp():
    import anionpy.testing as it
    t = it.measure("1 + 1", times=2)
    return isinstance(t, float) and t >= 0.0


TESTING_SPECS["testing.measure"] = ItemSpec(
    name="testing.measure", kind="custom", scalar_like=True,
    custom_cases=lambda: [("returns_nonneg_float", (), {})],
    numpy_adapter=lambda: _probe_measure_numpy(),
    ionp_adapter=lambda: _probe_measure_ionp(),
)


# 2026-08-01 falsifiability fix: the previous adapter returned an
# unconditional "ok" after calling run_threaded, so a no-op stand-in that
# never invoked the worker at all still passed. This probe counts real
# invocations (thread-safe) and, via pass_count=True, records exactly which
# indices 0..max_workers-1 the workers were called with -- a no-op yields
# (0, []) instead of (4, [0, 1, 2, 3]), which the harness now catches.
def _probe_run_threaded(mod):
    import threading
    lock = threading.Lock()
    counter = {"n": 0, "seen": set()}

    def worker(i):
        with lock:
            counter["n"] += 1
            counter["seen"].add(i)

    mod.run_threaded(worker, max_workers=4, pass_count=True)
    return (counter["n"], sorted(counter["seen"]))


def _probe_run_threaded_numpy():
    import numpy.testing as npt
    return _probe_run_threaded(npt)


def _probe_run_threaded_ionp():
    import anionpy.testing as it
    return _probe_run_threaded(it)


TESTING_SPECS["testing.run_threaded"] = ItemSpec(
    name="testing.run_threaded", kind="custom", scalar_like=True,
    custom_cases=lambda: [("invokes_worker_max_workers_times_with_indices", (), {})],
    numpy_adapter=lambda: _probe_run_threaded_numpy(),
    ionp_adapter=lambda: _probe_run_threaded_ionp(),
)


# 2026-08-01 falsifiability fix: the original fixture subclassed
# _DecorateTarget with an EMPTY body (`type("T", (_DecorateTarget,), {})`),
# so `test_one`/`helper` lived on the BASE class, not in the new class's own
# `__dict__`. decorate_methods iterates `cls.__dict__.values()` only (see
# both numpy's and anionpy's real source) -- with an empty subclass dict there
# is nothing to iterate on EITHER side, so both numpy's real implementation
# and a no-op stand-in produced the identical (False, False) result. Build
# the methods directly into the new class's own dict instead, so there is
# something for decorate_methods to actually find and decorate.
def _mark(fn):
    fn._marked = True
    return fn


def _make_decorate_target():
    def test_one(self):
        return 1

    def helper(self):
        return 2

    return type("T", (object,), {"test_one": test_one, "helper": helper})


def _probe_decorate_methods_numpy():
    import numpy.testing as npt
    cls = _make_decorate_target()
    npt.decorate_methods(cls, _mark, testmatch=re.compile(r"^test_"))
    return (getattr(cls.test_one, "_marked", False),
            getattr(cls.helper, "_marked", False))


def _probe_decorate_methods_ionp():
    import anionpy.testing as it
    cls = _make_decorate_target()
    it.decorate_methods(cls, _mark, testmatch=re.compile(r"^test_"))
    return (getattr(cls.test_one, "_marked", False),
            getattr(cls.helper, "_marked", False))


TESTING_SPECS["testing.decorate_methods"] = ItemSpec(
    name="testing.decorate_methods", kind="custom", scalar_like=True,
    custom_cases=lambda: [("marks_matching_methods_only", (), {})],
    numpy_adapter=lambda: _probe_decorate_methods_numpy(),
    ionp_adapter=lambda: _probe_decorate_methods_ionp(),
)


# rundocs: build a tiny module-like temp file with a passing doctest and a
# failing doctest; both must raise AssertionError for the failing one and
# return None for the passing one.
_RUNDOCS_PASS_SRC = '''
def add_one(x):
    """
    >>> add_one(1)
    2
    """
    return x + 1
'''

_RUNDOCS_FAIL_SRC = '''
def add_one(x):
    """
    >>> add_one(1)
    999
    """
    return x + 1
'''


def _write_and_rundocs(mod, src):
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        mod.rundocs(path)
        return "ok"
    finally:
        os.remove(path)


# 2026-08-01 falsifiability fix: `_RUNDOCS_FAIL_SRC` was defined but never
# wired into a case -- only the passing branch was ever exercised, so a
# no-op rundocs (which also never raises) was indistinguishable from the
# real one. Parameterize the adapter on which source to run and register
# both branches: the failing doctest must raise AssertionError on both
# sides, exercised via the harness's exception-type-match path.
def _probe_rundocs_numpy(should_fail):
    import numpy.testing as npt
    src = _RUNDOCS_FAIL_SRC if should_fail else _RUNDOCS_PASS_SRC
    return _write_and_rundocs(npt, src)


def _probe_rundocs_ionp(should_fail):
    import anionpy.testing as it
    src = _RUNDOCS_FAIL_SRC if should_fail else _RUNDOCS_PASS_SRC
    return _write_and_rundocs(it, src)


TESTING_SPECS["testing.rundocs"] = ItemSpec(
    name="testing.rundocs", kind="custom", scalar_like=True,
    custom_cases=lambda: [
        ("passing_doctest_ok", (False,), {}),
        ("failing_doctest_raises", (True,), {}),
    ],
    numpy_adapter=lambda should_fail: _probe_rundocs_numpy(should_fail),
    ionp_adapter=lambda should_fail: _probe_rundocs_ionp(should_fail),
)


# ---------------------------------------------------------------------------
# Classes / constants / `test` -- behavioural or type/identity adapters,
# never raw cross-module object equality (see module docstring for why).
# ---------------------------------------------------------------------------

# 2026-08-01 falsifiability fix: `isinstance(ie, Exception)` is true for
# nearly any exception instance, and `str(ie) == "x"` is true for nearly
# any single-arg Exception subclass -- a coordinator probe swapping in an
# unrelated `class _Unrelated(Exception): pass` passed both checks
# trivially. Add the class's own `__name__` (an unrelated class's name
# differs) and an explicit issubclass(..., Exception) + raise/catch
# round-trip against the class itself.
def _exception_facts(cls, msg):
    try:
        raise cls(msg)
    except cls as e:
        caught_as_self = True
        s = str(e)
    except Exception:
        caught_as_self = False
        s = None
    return (cls.__name__, issubclass(cls, Exception), s, caught_as_self)


def _probe_ignore_exception_numpy():
    import numpy.testing as npt
    return _exception_facts(npt.IgnoreException, "x")


def _probe_ignore_exception_ionp():
    import anionpy.testing as it
    return _exception_facts(it.IgnoreException, "x")


TESTING_SPECS["testing.IgnoreException"] = ItemSpec(
    name="testing.IgnoreException", kind="custom", scalar_like=True,
    custom_cases=lambda: [("name_subclass_message_catchable", (), {})],
    numpy_adapter=lambda: _probe_ignore_exception_numpy(),
    ionp_adapter=lambda: _probe_ignore_exception_ionp(),
)


def _probe_known_failure_exception_numpy():
    import numpy.testing as npt
    return _exception_facts(npt.KnownFailureException, "y")


def _probe_known_failure_exception_ionp():
    import anionpy.testing as it
    return _exception_facts(it.KnownFailureException, "y")


TESTING_SPECS["testing.KnownFailureException"] = ItemSpec(
    name="testing.KnownFailureException", kind="custom", scalar_like=True,
    custom_cases=lambda: [("name_subclass_message_catchable", (), {})],
    numpy_adapter=lambda: _probe_known_failure_exception_numpy(),
    ionp_adapter=lambda: _probe_known_failure_exception_ionp(),
)


def _probe_skiptest_is_stdlib_numpy():
    import unittest
    import numpy.testing as npt
    return npt.SkipTest is unittest.SkipTest


def _probe_skiptest_is_stdlib_ionp():
    import unittest
    import anionpy.testing as it
    return it.SkipTest is unittest.SkipTest


TESTING_SPECS["testing.SkipTest"] = ItemSpec(
    name="testing.SkipTest", kind="custom", scalar_like=True,
    custom_cases=lambda: [("is_the_stdlib_class", (), {})],
    numpy_adapter=lambda: _probe_skiptest_is_stdlib_numpy(),
    ionp_adapter=lambda: _probe_skiptest_is_stdlib_ionp(),
)


def _probe_testcase_is_stdlib_numpy():
    import unittest
    import numpy.testing as npt
    return npt.TestCase is unittest.TestCase


def _probe_testcase_is_stdlib_ionp():
    import unittest
    import anionpy.testing as it
    return it.TestCase is unittest.TestCase


TESTING_SPECS["testing.TestCase"] = ItemSpec(
    name="testing.TestCase", kind="custom", scalar_like=True,
    custom_cases=lambda: [("is_the_stdlib_class", (), {})],
    numpy_adapter=lambda: _probe_testcase_is_stdlib_numpy(),
    ionp_adapter=lambda: _probe_testcase_is_stdlib_ionp(),
)


def _probe_ccw_numpy():
    import warnings
    import numpy.testing as npt
    before = len(warnings.filters)
    with npt.clear_and_catch_warnings():
        warnings.filterwarnings("ignore")
        inside = len(warnings.filters)
    after = len(warnings.filters)
    return (before == after, inside >= before)


def _probe_ccw_ionp():
    import warnings
    import anionpy.testing as it
    before = len(warnings.filters)
    with it.clear_and_catch_warnings():
        warnings.filterwarnings("ignore")
        inside = len(warnings.filters)
    after = len(warnings.filters)
    return (before == after, inside >= before)


TESTING_SPECS["testing.clear_and_catch_warnings"] = ItemSpec(
    name="testing.clear_and_catch_warnings", kind="custom", scalar_like=True,
    custom_cases=lambda: [("restores_filters_on_exit", (), {})],
    numpy_adapter=lambda: _probe_ccw_numpy(),
    ionp_adapter=lambda: _probe_ccw_ionp(),
)


def _probe_suppress_warnings_numpy():
    import warnings
    import numpy.testing as npt
    with npt.suppress_warnings() as sup:
        sup.filter(UserWarning)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            warnings.warn("hidden", UserWarning)
            saw_it_leak_through = len(rec) > 0
    return not saw_it_leak_through


def _probe_suppress_warnings_ionp():
    import warnings
    import anionpy.testing as it
    with it.suppress_warnings() as sup:
        sup.filter(UserWarning)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            warnings.warn("hidden", UserWarning)
            saw_it_leak_through = len(rec) > 0
    return not saw_it_leak_through


TESTING_SPECS["testing.suppress_warnings"] = ItemSpec(
    name="testing.suppress_warnings", kind="custom", scalar_like=True,
    custom_cases=lambda: [("actually_suppresses_filtered_warning", (), {})],
    numpy_adapter=lambda: _probe_suppress_warnings_numpy(),
    ionp_adapter=lambda: _probe_suppress_warnings_ionp(),
)


# 2026-08-01 falsifiability fix: `callable(x) and hasattr(x, "__call__")`
# is true for nearly anything, including a no-op `lambda *a, **k: None`
# swapped in for `test`. Both numpy's PytestTester and anionpy's _IonpTester
# call `pytest.main(pytest_args)` internally (verified: identical
# `import pytest; ...; pytest.main(pytest_args)` shape in both real
# sources) -- mock `pytest.main` and check it actually got invoked with a
# real, non-empty argv containing "-q" (both testers build this), and that
# the tester's own return value reflects pytest.main's return code. A
# no-op never touches pytest.main at all, which the mock now catches.
def _probe_test_call(mod):
    # Delegates to the ONE shared implementation. This used to be a private
    # copy that compared invocation only and never the signature; see
    # harness.probe_pytest_tester for the divergence that slipped through
    # all three copies.
    from harness import probe_pytest_tester
    return probe_pytest_tester(mod)


def _probe_test_numpy():
    import numpy.testing as npt
    return _probe_test_call(npt)


def _probe_test_ionp():
    import anionpy.testing as it
    return _probe_test_call(it)


TESTING_SPECS["testing.test"] = ItemSpec(
    name="testing.test", kind="custom", scalar_like=True,
    custom_cases=lambda: [("invokes_pytest_main_with_constructed_argv", (), {})],
    numpy_adapter=lambda: _probe_test_numpy(),
    ionp_adapter=lambda: _probe_test_ionp(),
)


# 2026-08-01 falsifiability fix (coordinator audit, confirmed live on this
# machine via faithful `anionpy.testing.<attr>` substitution -- see the
# falsifiability report in the final commit message for the exact repro):
# `isinstance(getattr(it, name), expected_type)` is true for ANY value of
# the right type, so flipping IS_64BIT/IS_PYPY/etc. to their wrong-but-
# still-bool value, or bumping `verbose` to a wrong-but-still-int value,
# passed silently. anionpy runs in the same interpreter, on the same machine,
# as the numpy compared against -- for these 8 constants that is a genuine
# shared fact, so grade exact value AND type equality against real
# numpy.testing.<NAME> directly (no isinstance wrapper at all).
def _direct_value_probe(name):
    def numpy_probe():
        import numpy.testing as npt
        return getattr(npt, name)

    def ionp_probe():
        import anionpy.testing as it
        return getattr(it, name)

    return numpy_probe, ionp_probe


_SAME_MACHINE_CONSTS = (
    "verbose", "IS_PYPY", "IS_PYSTON", "IS_WASM", "IS_MUSL",
    "IS_64BIT", "HAS_REFCOUNT", "NOGIL_BUILD",
)

for _const_name in _SAME_MACHINE_CONSTS:
    _item_name = f"testing.{_const_name}"
    _np_probe, _ionp_probe = _direct_value_probe(_const_name)
    TESTING_SPECS[_item_name] = ItemSpec(
        name=_item_name, kind="custom", scalar_like=True,
        custom_cases=lambda: [("exact_value_and_type_matches_numpy", (), {})],
        numpy_adapter=_np_probe,
        ionp_adapter=_ionp_probe,
    )


# `BLAS_SUPPORTS_FPE` / `HAS_LAPACK64` are NOT safe to grade by equality to
# real numpy despite being "same machine, same interpreter" facts: ionp-core
# has zero BLAS/LAPACK dependency by architecture (see GOAL-ionp.md and
# anionpy/testing.py's own comment at these two constants), so anionpy's correct
# value is unconditionally False on every machine, while numpy's real value
# in THIS venv is HAS_LAPACK64=True (a real ILP64 LAPACK is actually linked)
# -- asserting equality to numpy's value here would force the CORRECT anionpy
# implementation to FAIL. Grade against a fixed, architecturally-justified
# expected constant instead: still falsifiable (a sabotaged True fails the
# comparison), but never asserts a falsehood in either direction.
def _probe_blas_supports_fpe_ionp():
    import anionpy.testing as it
    return it.BLAS_SUPPORTS_FPE


TESTING_SPECS["testing.BLAS_SUPPORTS_FPE"] = ItemSpec(
    name="testing.BLAS_SUPPORTS_FPE", kind="custom", scalar_like=True,
    custom_cases=lambda: [("architecturally_always_false_no_blas_dependency", (), {})],
    numpy_adapter=lambda: False,
    ionp_adapter=lambda: _probe_blas_supports_fpe_ionp(),
)


def _probe_has_lapack64_ionp():
    import anionpy.testing as it
    return it.HAS_LAPACK64


TESTING_SPECS["testing.HAS_LAPACK64"] = ItemSpec(
    name="testing.HAS_LAPACK64", kind="custom", scalar_like=True,
    custom_cases=lambda: [("architecturally_always_false_no_lapack_dependency", (), {})],
    numpy_adapter=lambda: False,
    ionp_adapter=lambda: _probe_has_lapack64_ionp(),
)


# `IS_EDITABLE` / `IS_INSTALLED` / `NUMPY_ROOT` describe *anionpy's own*
# install characteristics (per anionpy/testing.py's docstring), not numpy's --
# anionpy is genuinely editable-installed in this venv while numpy is not, so
# comparing to numpy's live value would assert a falsehood. But merely
# re-reading `anionpy.testing.<NAME>` on both "sides" would be circular: a
# sabotaged module attribute would poison both reads identically. Instead,
# INDEPENDENTLY RECOMPUTE the same fact here via a separate code path
# (mirroring, but not calling, anionpy/testing.py's own importlib.metadata
# logic) and compare that fresh computation against the live attribute --
# a swapped-in wrong constant now disagrees with the independent recompute.
def _recompute_ionp_install_facts():
    import importlib.metadata
    import json
    import pathlib
    import sys
    import types
    import anionpy
    root = pathlib.Path(anionpy.__file__).parent
    try:
        dist = importlib.metadata.distribution("anionpy")
    except importlib.metadata.PackageNotFoundError:
        return (False, False, root)
    installed = True
    try:
        if sys.version_info < (3, 13):
            origin = json.loads(
                dist.read_text("direct_url.json") or "{}",
                object_hook=lambda data: types.SimpleNamespace(**data),
            )
            editable = origin.dir_info.editable
        else:
            editable = dist.origin.dir_info.editable
    except AttributeError:
        editable = False
    if not editable and dist.locate_file("anionpy") != root:
        installed = False
    return (installed, editable, root)


def _probe_is_installed_ionp():
    import anionpy.testing as it
    recomputed_installed, _recomputed_editable, _root = _recompute_ionp_install_facts()
    return (recomputed_installed, it.IS_INSTALLED)


TESTING_SPECS["testing.IS_INSTALLED"] = ItemSpec(
    name="testing.IS_INSTALLED", kind="custom", scalar_like=True,
    custom_cases=lambda: [("matches_independently_recomputed_install_fact", (), {})],
    numpy_adapter=lambda: (True, True),
    ionp_adapter=lambda: _probe_is_installed_ionp(),
)


def _probe_is_editable_ionp():
    import anionpy.testing as it
    _installed, recomputed_editable, _root = _recompute_ionp_install_facts()
    return (recomputed_editable, it.IS_EDITABLE)


TESTING_SPECS["testing.IS_EDITABLE"] = ItemSpec(
    name="testing.IS_EDITABLE", kind="custom", scalar_like=True,
    custom_cases=lambda: [("matches_independently_recomputed_install_fact", (), {})],
    numpy_adapter=lambda: (True, True),
    ionp_adapter=lambda: _probe_is_editable_ionp(),
)


def _probe_numpy_root_ionp():
    import anionpy.testing as it
    _installed, _editable, recomputed_root = _recompute_ionp_install_facts()
    return (recomputed_root == it.NUMPY_ROOT, True)


TESTING_SPECS["testing.NUMPY_ROOT"] = ItemSpec(
    name="testing.NUMPY_ROOT", kind="custom", scalar_like=True,
    custom_cases=lambda: [("matches_independently_recomputed_root_path", (), {})],
    numpy_adapter=lambda: (True, True),
    ionp_adapter=lambda: _probe_numpy_root_ionp(),
)


# ---------------------------------------------------------------------------
# Merge block for registry.py (NOT executed here -- registry.py is not owned
# by this task; this mirrors LINALG_SPECS'/INPLACE_SPECS' merge exactly and
# is the block that needs to be added to the bottom of registry.py):
#
#   from testing_cases import TESTING_SPECS  # noqa: E402
#   _testing_collisions = set(TESTING_SPECS) & set(REGISTRY)
#   if _testing_collisions:
#       raise AssertionError(
#           f"testing_cases.py: {sorted(_testing_collisions)} already present "
#           f"in registry.REGISTRY -- refusing to silently overwrite an "
#           f"existing item"
#       )
#   REGISTRY.update(TESTING_SPECS)
# ---------------------------------------------------------------------------
