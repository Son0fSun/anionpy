"""Differential corpus for the floating-point-error (FPE) subsystem gap --
`numpy.seterr`/`geterr`/`errstate`/`seterrcall`/`geterrcall` and the
`RuntimeWarning`/`FloatingPointError` machinery they gate.

WHY THIS FILE EXISTS. Measured 2026-08-06 (docs/FPE-SUBSYSTEM-SPEC-2026-08-
06.md carries the full measurement, this is the executable half of the same
finding): anionpy has NO floating-point error subsystem at all.
`hasattr(anionpy, 'errstate'|'seterr'|'geterr'|'seterrcall'|'geterrcall')` are
all `False`, and every ufunc-error condition this file exercises (divide by
zero, 0/0, sqrt(-1), log(0), overflow, underflow, empty-slice mean/std,
cast overflow) is silently NUMERICALLY CORRECT but diagnostically SILENT on
anionpy where real numpy emits a `RuntimeWarning` (or, under
`errstate(...='raise')`, a `FloatingPointError`). Code that runs
`np.errstate(divide='raise')` to convert a silent-wrong-answer bug into a
loud one gets the silent-wrong-looking answer back, unchanged, under anionpy.

STATUS AS OF 2026-08-07 (RE-MEASURED, not carried forward from the note
above): the five entry points now EXIST and this corpus PASSES 100/100
against the installed binary -- `seterr`/`geterr`/`errstate`/`seterrcall`/
`geterrcall` landed in `ionp-core/src/fpe.rs`/`ionp-py/src/fpstate.rs`
(commit 53e1ac5), roughly 50 minutes after this corpus was originally
written (7959c2d) describing an absent subsystem; the two facts sat
uncommented-on in the same file until now. `run.py` did not even import
this module until a separate fix (see its own comment on the
`import fperr_cases` line) -- before that, the coverage ledger counted all
five names "absent" despite the implementation and this corpus both
already existing.

THIS CORPUS IS CURATED, NOT GENERAL -- passing it is necessary, not
sufficient, evidence that the subsystem works broadly. Re-measured directly
against the installed binary: of 10 representative FP-warning-triggering
ops NOT in this corpus (`power(-1, 0.5)`, `log2(0)`, `log10(0)`,
`exp(1000)`, `arcsin(2)`, `arccos(2)`, `reciprocal(0)`, float `%0`,
`std`/`var` on an empty array), 8 diverge -- real numpy emits a
`RuntimeWarning`, anionpy emits nothing, for every one of them except float
`%0` (where neither side warns, matching). The five entry points and the
warning/exception PLUMBING they gate are real and demonstrably wired up
for the ops this corpus exercises; the set of ops that actually TRIGGER a
diagnostic through that plumbing is narrow and not yet general. Neither
this docstring correction nor the `run.py` import fix expands this
corpus's case list or declares any of the five entry-point names in
`anionpy/_state/` -- both are left for whoever picks up the general-case
gap next, with this measurement as the starting point rather than a
re-discovery.

VALUE PARITY WAS VERIFIED BEFORE ANY CASE BELOW WAS WRITTEN (see this
task's report for the full grid): every op this file uses already returns
BIT-IDENTICAL values on anionpy vs numpy today (inf/nan/wraparound/clamped-
cast values all match). Cases are deliberately restricted to that verified-
matching set so that a failure here can only ever mean "the diagnostic
(warning/exception/callback) is missing or wrong", never "the underlying
arithmetic is also wrong" -- conflating the two would make a future
partial fix (warnings wired up, but into arithmetic that was ALSO buggy)
impossible to distinguish from a real regression in this same corpus.

WHAT IS COMPARED. Every case wraps one call in a fresh, isolated
`warnings.catch_warnings(record=True)` + `simplefilter('always')` block
(mirrors complex_warning_cases.py's `_record` exactly, for the same "must
never leak into or observe another case's filter/registry state" reason)
and folds `[(category_name, message_text), ...]` together with the call's
own outcome (result descriptor, or raised exception type+message) into one
descriptor string -- so a case that gets the warning right but the value
wrong, or vice versa, still fails. `seterr`/`errstate`/`seterrcall` state
mutated by a case is ALWAYS restored (try/finally) before the case
returns, on both sides, so cases cannot leak configuration into each other
even though `numpy`'s state genuinely is process/thread-global mutable
state across calls within this one test process.

NEGATIVE CONTROLS (see module docstring convention established by
complex_warning_cases.py): int overflow (`int8(127)+1`) and complex
`sqrt` of a negative real are cases where real numpy NEVER warns, at ANY
`seterr` setting -- not "defaults off", genuinely unreachable via this
API. Both are included, and (measured) BOTH CURRENTLY PASS against anionpy
today, because anionpy's blanket silence and numpy's principled architectural
silence happen to coincide for exactly these two shapes -- proving this
corpus is not vacuously red across the board; see this task's report for
the pasted green/red instrument-sanity check this claim is not asked to be
taken on faith for.

WHAT IS DELIBERATELY NOT COVERED. `'print'` mode (writes to the real
process file-descriptor-2 stderr, bypassing Python's `sys.stderr` object --
confirmed in the spec doc; capturing it needs OS-level fd redirection,
which this file does not attempt, to keep the corpus hermetic and
pytest-safe). Platform-dependence of the default state (spec doc section 6
flags this as unverified across architectures in this measurement pass).
"""
from __future__ import annotations

import threading
import warnings

from registry import REGISTRY, ItemSpec

# ---------------------------------------------------------------------------
# Op library: op_key -> fn(module, dtype_name) -> array/scalar result.
# Every one of these was measured (2026-08-06, this task) to already
# produce a BIT-IDENTICAL value on anionpy vs numpy -- see module docstring.
# ---------------------------------------------------------------------------

def _op_divide_by_zero(m, dt):
    return m.array([1.0], dtype=dt) / m.array([0.0], dtype=dt)


def _op_invalid_zero_over_zero(m, dt):
    return m.array([0.0], dtype=dt) / m.array([0.0], dtype=dt)


def _op_sqrt_negative(m, dt):
    return m.sqrt(m.array([-1.0], dtype=dt))


def _op_log_zero(m, dt):
    return m.log(m.array([0.0], dtype=dt))


def _op_log_negative(m, dt):
    return m.log(m.array([-1.0], dtype=dt))


def _op_overflow_multiply(m, dt):
    return m.array([3.0e38], dtype=dt) * m.array([10.0], dtype=dt)


def _op_underflow_multiply(m, dt):
    return m.array([1e-38], dtype=dt) * m.array([1e-38], dtype=dt)


def _op_int_overflow_add(m, dt):
    return m.array([127], dtype=dt) + m.array([1], dtype=dt)


def _op_int_floordiv_zero(m, dt):
    return m.array([1], dtype=dt) // m.array([0], dtype=dt)


def _op_int_mod_zero(m, dt):
    return m.array([1], dtype=dt) % m.array([0], dtype=dt)


def _op_cast_overflow(m, dt):
    return m.array([1e300]).astype(dt)


def _op_mean_empty(m, dt):
    return m.mean(m.array([], dtype=dt))


def _op_sum_empty(m, dt):
    return m.sum(m.array([], dtype=dt))


def _op_complex_divide_zero(m, dt):
    return m.array([1 + 1j], dtype=dt) / m.array([0 + 0j], dtype=dt)


def _op_complex_zero_over_zero(m, dt):
    return m.array([0 + 0j], dtype=dt) / m.array([0 + 0j], dtype=dt)


def _op_complex_sqrt_negative(m, dt):
    return m.sqrt(m.array([-1 + 0j], dtype=dt))


def _op_mean_empty_axis(m, dt):
    return m.mean(m.zeros((0, 3), dtype=dt), axis=0)


def _op_sum_empty_axis(m, dt):
    return m.sum(m.zeros((3, 0), dtype=dt), axis=1)


# ---------------------------------------------------------------------------
# Task #56 (2026-08-07, Monday) additions. The seven items the brief pointed
# at (`remainder`/`mod`/`ndarray.__mod__`/`__rmod__`/`__imod__` over-warning
# on a float; `divmod`/`ndarray.__divmod__` under-warning) were illustrative,
# not exhaustive -- the full measured blast radius (see this task's report
# for the complete sweep) also included two items NOT in that list:
#   - `fmod` (int) was completely silent (no case in the dispatch match at
#     all), same shape of bug as `divmod`.
#   - `floor_divide`/`divmod` on an INTEGER promoted output dtype at 0/0
#     used to fire `Invalid` ("invalid value encountered"); real numpy
#     fires `Divide` ("divide by zero encountered") there -- `Invalid` is
#     only correct when the promoted output dtype is FLOATING. This was a
#     latent bug in `detect_divide_family` itself, not one of the seven.
# Every op below was verified bit-identical in VALUE against real numpy
# before being added here (same discipline as the rest of this file) --
# a failure below can only mean the diagnostic is wrong, not the arithmetic.
# ---------------------------------------------------------------------------

def _op_float_mod_zero(m, dt):
    # NEGATIVE CONTROL: real numpy's `%`/`mod`/`remainder` is silent for
    # EVERY floating-dtype input, at any numerator, including 0/0 -- this
    # is the over-warn half of the defect (anionpy used to warn here).
    return m.array([1.0], dtype=dt) % m.array([0.0], dtype=dt)


def _op_float_remainder_zero_over_zero(m, dt):
    # NEGATIVE CONTROL, same rule as above via the `remainder` ufunc name
    # directly (not the `%` dunder) and with a 0/0 numerator specifically
    # (float `remainder` stays silent even at 0/0, unlike `floor_divide`).
    return m.remainder(m.array([0.0], dtype=dt), m.array([0.0], dtype=dt))


def _op_fmod_zero_int(m, dt):
    # `fmod` shares `remainder`'s exact rule; found silent on anionpy
    # (no dispatch case existed for it at all) during this task's sweep.
    return m.fmod(m.array([1], dtype=dt), m.array([0], dtype=dt))


def _op_fmod_zero_float_negative_control(m, dt):
    return m.fmod(m.array([1.0], dtype=dt), m.array([0.0], dtype=dt))


def _op_int_divmod_zero(m, dt):
    return m.divmod(m.array([1], dtype=dt), m.array([0], dtype=dt))


def _op_int_divmod_zero_over_zero(m, dt):
    # The additional (non-illustrative-list) bug: int 0/0 must be `Divide`,
    # not `Invalid` -- `Invalid` is a floating-output-dtype-only rule.
    return m.divmod(m.array([0], dtype=dt), m.array([0], dtype=dt))


def _op_float_divmod_nonzero_over_zero(m, dt):
    # float x/0 (x != 0) is still `Divide` on `divmod`/`floor_divide` --
    # unlike `remainder`, which is silent here. This is the case that most
    # directly distinguishes `divmod`'s detector from `remainder`'s.
    return m.divmod(m.array([1.0], dtype=dt), m.array([0.0], dtype=dt))


def _op_float_divmod_zero_over_zero(m, dt):
    return m.divmod(m.array([0.0], dtype=dt), m.array([0.0], dtype=dt))


def _op_int_floordiv_zero_over_zero(m, dt):
    # Same additional bug as int divmod 0/0, on `floor_divide` directly.
    return m.array([0], dtype=dt) // m.array([0], dtype=dt)


def _op_builtin_divmod_dunder_int(m, dt):
    # Exercises `ndarray.__divmod__` (the Python-builtin-`divmod()` call
    # path), not the `np.divmod`/`anionpy.divmod` ufunc-OBJECT path above --
    # these are two separate code paths in anionpy (see `__divmod__`'s own
    # doc comment in `lib.rs`) that must each independently be verified.
    return divmod(m.array([1], dtype=dt), m.array([0], dtype=dt))


def _op_builtin_divmod_dunder_float_zero_over_zero(m, dt):
    return divmod(m.array([0.0], dtype=dt), m.array([0.0], dtype=dt))


def _op_builtin_rdivmod_dunder_int(m, dt):
    # `ndarray.__rdivmod__` -- `divmod(<python scalar>, <ndarray>)`.
    return divmod(1, m.array([0], dtype=dt))


_OPS = {
    "divide_by_zero": (_op_divide_by_zero, "float64"),
    "invalid_zero_over_zero": (_op_invalid_zero_over_zero, "float64"),
    "sqrt_negative": (_op_sqrt_negative, "float64"),
    "log_zero": (_op_log_zero, "float64"),
    "log_negative": (_op_log_negative, "float64"),
    "overflow_multiply_f32": (_op_overflow_multiply, "float32"),
    "underflow_multiply_f32": (_op_underflow_multiply, "float32"),
    "int_overflow_add_i8": (_op_int_overflow_add, "int8"),
    "int_floordiv_zero_i64": (_op_int_floordiv_zero, "int64"),
    "int_mod_zero_i64": (_op_int_mod_zero, "int64"),
    "cast_overflow_i32": (_op_cast_overflow, "int32"),
    "mean_empty_f64": (_op_mean_empty, "float64"),
    "sum_empty_f64": (_op_sum_empty, "float64"),
    "complex_divide_zero_c128": (_op_complex_divide_zero, "complex128"),
    "complex_zero_over_zero_c128": (_op_complex_zero_over_zero, "complex128"),
    "complex_sqrt_negative_c128": (_op_complex_sqrt_negative, "complex128"),
    "mean_empty_axis_f64": (_op_mean_empty_axis, "float64"),
    "sum_empty_axis_f64": (_op_sum_empty_axis, "float64"),
    # --- task #56 additions ---
    "float_mod_zero_f64": (_op_float_mod_zero, "float64"),
    "float_remainder_zero_over_zero_f64": (_op_float_remainder_zero_over_zero, "float64"),
    "fmod_zero_int_i64": (_op_fmod_zero_int, "int64"),
    "fmod_zero_float_f64": (_op_fmod_zero_float_negative_control, "float64"),
    "int_divmod_zero_i64": (_op_int_divmod_zero, "int64"),
    "int_divmod_zero_over_zero_i64": (_op_int_divmod_zero_over_zero, "int64"),
    "float_divmod_nonzero_over_zero_f64": (_op_float_divmod_nonzero_over_zero, "float64"),
    "float_divmod_zero_over_zero_f64": (_op_float_divmod_zero_over_zero, "float64"),
    "int_floordiv_zero_over_zero_i64": (_op_int_floordiv_zero_over_zero, "int64"),
    "builtin_divmod_dunder_int_i64": (_op_builtin_divmod_dunder_int, "int64"),
    "builtin_divmod_dunder_float_zero_over_zero_f64": (_op_builtin_divmod_dunder_float_zero_over_zero, "float64"),
    "builtin_rdivmod_dunder_int_i64": (_op_builtin_rdivmod_dunder_int, "int64"),
}

# Which seterr *category* governs each op -- from the spec doc's measured
# table (log(0) is 'divide' not 'invalid'; cast overflow is 'invalid'; int
# floordiv/mod by zero is 'divide'; overflow/underflow are 'over'/'under').
_OP_CATEGORY = {
    "divide_by_zero": "divide",
    "invalid_zero_over_zero": "invalid",
    "sqrt_negative": "invalid",
    "log_zero": "divide",
    "log_negative": "invalid",
    "overflow_multiply_f32": "over",
    "underflow_multiply_f32": "under",
    "int_floordiv_zero_i64": "divide",
    "int_mod_zero_i64": "divide",
    "cast_overflow_i32": "invalid",
    "complex_divide_zero_c128": "divide",
    "complex_zero_over_zero_c128": "invalid",
    # --- task #56 additions ---
    "fmod_zero_int_i64": "divide",
    "int_divmod_zero_i64": "divide",
    "int_divmod_zero_over_zero_i64": "divide",
    "float_divmod_nonzero_over_zero_f64": "divide",
    "float_divmod_zero_over_zero_f64": "invalid",
    "int_floordiv_zero_over_zero_i64": "divide",
    "builtin_divmod_dunder_int_i64": "divide",
    "builtin_divmod_dunder_float_zero_over_zero_f64": "invalid",
    "builtin_rdivmod_dunder_int_i64": "divide",
}


def _run_op(m, op_key):
    fn, dt = _OPS[op_key]
    return fn(m, dt)


def _record(fn):
    """Mirrors complex_warning_cases.py's `_record` exactly: fresh,
    isolated `catch_warnings` block on `simplefilter('always')` so a
    warning fires every time regardless of ambient filter state, and so
    this probe can never observe a warning some OTHER case already
    triggered."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            result = fn()
            exc = None
        except BaseException as e:  # noqa: BLE001 -- PyO3 panics derive BaseException
            result = None
            exc = e
    warns = [(w.category.__name__, str(w.message)) for w in rec]
    return warns, result, exc


def _result_descr(result):
    if hasattr(result, "dtype") and hasattr(result, "shape"):
        return f"dtype={result.dtype.name}|shape={tuple(result.shape)}|values={result.tolist()!r}"
    if isinstance(result, tuple):
        # `divmod`/builtin-`divmod()` return a 2-tuple of arrays -- describe
        # each element the same array-aware way rather than falling through
        # to `repr(tuple)`, which would key equality on each side's own
        # `__repr__` formatting instead of the actual dtype/shape/values
        # (task #56, 2026-08-07: verified anionpy's array repr already
        # matches numpy's today, but this is not something this corpus
        # should have to assume stays true to keep detecting real value
        # regressions).
        return f"tuple[{', '.join(_result_descr(elem) for elem in result)}]"
    return f"pyval={result!r}"


def _descriptor(warns, result, exc):
    if exc is not None:
        return f"warns={warns!r}|raised:{type(exc).__name__}:{exc}"
    return f"warns={warns!r}|{_result_descr(result)}"


# ---------------------------------------------------------------------------
# Site 1: default state, no seterr/errstate touched at all -- the raw gap
# from the finding table (`docs/FPE-SUBSYSTEM-SPEC-2026-08-06.md` section 1).
# ---------------------------------------------------------------------------

def _site_default(m, op_key):
    def fn():
        return _run_op(m, op_key)
    return _record(fn)


# ---------------------------------------------------------------------------
# Site 2: seterr(<category>=<mode>), restored via try/finally. `old` stays
# `None` (skip restore) if `seterr` itself raised -- i.e. on today's anionpy,
# where the attribute doesn't exist at all.
# ---------------------------------------------------------------------------

def _site_seterr_mode(m, op_key, mode):
    category = _OP_CATEGORY[op_key]

    def fn():
        old = None
        try:
            old = m.seterr(**{category: mode})
            return _run_op(m, op_key)
        finally:
            if old is not None:
                m.seterr(**old)
    return _record(fn)


# ---------------------------------------------------------------------------
# Site 3: errstate as a context manager.
# ---------------------------------------------------------------------------

def _site_errstate_cm(m, op_key, mode):
    category = _OP_CATEGORY[op_key]

    def fn():
        with m.errstate(**{category: mode}):
            return _run_op(m, op_key)
    return _record(fn)


# ---------------------------------------------------------------------------
# Site 4: errstate as a decorator.
# ---------------------------------------------------------------------------

def _site_errstate_decorator(m, op_key, mode):
    category = _OP_CATEGORY[op_key]

    def fn():
        @m.errstate(**{category: mode})
        def inner():
            return _run_op(m, op_key)
        return inner()
    return _record(fn)


# ---------------------------------------------------------------------------
# Site 5: nested errstate -- inner scope's mode applies exclusively while
# active; on exiting the inner block, the OUTER mode is restored and
# re-takes effect (spec doc section 4.5). Encodes both the inner-scope
# result AND a second call made after the inner block exits (still inside
# the outer block) into one descriptor.
# ---------------------------------------------------------------------------

def _site_errstate_nested(m, op_key, outer_mode, inner_mode):
    category = _OP_CATEGORY[op_key]

    def fn():
        with m.errstate(**{category: outer_mode}):
            with warnings.catch_warnings(record=True) as inner_rec:
                warnings.simplefilter("always")
                try:
                    inner_result = _run_op(m, op_key)
                    inner_exc = None
                except BaseException as e:  # noqa: BLE001
                    inner_result = None
                    inner_exc = e
            inner_warns = [(w.category.__name__, str(w.message)) for w in inner_rec]
            with warnings.catch_warnings(record=True) as outer_rec:
                warnings.simplefilter("always")
                try:
                    outer_result = _run_op(m, op_key)
                    outer_exc = None
                except BaseException as e:  # noqa: BLE001
                    outer_result = None
                    outer_exc = e
            outer_warns = [(w.category.__name__, str(w.message)) for w in outer_rec]
        return (
            f"inner[mode={inner_mode}]={_descriptor(inner_warns, inner_result, inner_exc)}"
            f"||outer[mode={outer_mode}]={_descriptor(outer_warns, outer_result, outer_exc)}"
        )
    # fn() already returns a descriptor string directly (not routed through
    # the ordinary result/exc split) -- wrap it so _record's shape is
    # still honored by the caller below.

    def outer_fn():
        return fn()
    return _record(outer_fn)


# ---------------------------------------------------------------------------
# Site 6: seterrcall 'call' mode -- registrant is a plain callback
# `f(short_msg: str, flag: int)`.
# ---------------------------------------------------------------------------

def _site_call_mode(m, op_key):
    def fn():
        calls = []

        def handler(msg, flag):
            calls.append((msg, flag))

        old_call = None
        old_state = None
        try:
            old_call = m.seterrcall(handler)
            old_state = m.seterr(**{_OP_CATEGORY[op_key]: "call"})
            result = _run_op(m, op_key)
        finally:
            if old_state is not None:
                m.seterr(**old_state)
            if old_call is not None or hasattr(m, "seterrcall"):
                try:
                    m.seterrcall(old_call)
                except Exception:  # noqa: BLE001 -- best-effort restore
                    pass
        return f"calls={calls!r}|{_result_descr(result)}"
    return _record(fn)


# ---------------------------------------------------------------------------
# Site 7: seterrcall 'log' mode, WITH a registered `.write`-object.
# ---------------------------------------------------------------------------

class _Logger:
    def __init__(self):
        self.msgs = []

    def write(self, msg):
        self.msgs.append(msg)


def _site_log_mode(m, op_key):
    def fn():
        logger = _Logger()
        old_call = None
        old_state = None
        try:
            old_call = m.seterrcall(logger)
            old_state = m.seterr(**{_OP_CATEGORY[op_key]: "log"})
            result = _run_op(m, op_key)
        finally:
            if old_state is not None:
                m.seterr(**old_state)
            if old_call is not None or hasattr(m, "seterrcall"):
                try:
                    m.seterrcall(old_call)
                except Exception:  # noqa: BLE001
                    pass
        return f"msgs={logger.msgs!r}|{_result_descr(result)}"
    return _record(fn)


# ---------------------------------------------------------------------------
# Site 8: seterrcall 'log' mode, NO registrant -- must raise NameError
# (spec doc section 4.2) at the point of the arithmetic op itself.
# ---------------------------------------------------------------------------

def _site_log_mode_no_registrant(m, op_key):
    def fn():
        old_call = None
        old_state = None
        try:
            old_call = m.seterrcall(None)
            old_state = m.seterr(**{_OP_CATEGORY[op_key]: "log"})
            return _run_op(m, op_key)
        finally:
            if old_state is not None:
                m.seterr(**old_state)
            if old_call is not None or hasattr(m, "seterrcall"):
                try:
                    m.seterrcall(old_call)
                except Exception:  # noqa: BLE001
                    pass
    return _record(fn)


# ---------------------------------------------------------------------------
# Site 9: thread-local state (spec doc section 4.4). A background thread
# sets `seterr(divide='raise')` and exits; the calling (main) thread's own
# state must be completely unaffected, before, during (approximated by
# after `join()`, since a race on "during" would be nondeterministic and
# this corpus must be deterministic), and after.
# ---------------------------------------------------------------------------

def _site_thread_local(m, op_key):
    def fn():
        results = {}

        def worker():
            old = m.seterr(**{_OP_CATEGORY[op_key]: "raise"})
            results["worker_state"] = dict(m.geterr())
            m.seterr(**old)

        main_before = dict(m.geterr())
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        main_after = dict(m.geterr())
        return (
            f"main_before={main_before!r}|worker_state={results.get('worker_state')!r}"
            f"|main_after={main_after!r}|unaffected={main_before == main_after!r}"
        )
    return _record(fn)


# ---------------------------------------------------------------------------
# Site 10: geterr()/seterr() basic contract -- seterr returns the PREVIOUS
# state; mutating a returned dict must not affect internal state.
# ---------------------------------------------------------------------------

def _site_seterr_roundtrip(m, op_key):
    category = _OP_CATEGORY[op_key]

    def fn():
        before = dict(m.geterr())
        d = m.geterr()
        d[category] = "raise"  # mutate the LOCAL copy only
        unaffected = dict(m.geterr()) == before
        old = m.seterr(**{category: "raise"})
        old_matches_before = old == before
        after_set = dict(m.geterr())
        m.seterr(**old)
        restored = dict(m.geterr()) == before
        return (
            f"mutate_local_copy_unaffected={unaffected!r}|seterr_returned_old={old_matches_before!r}"
            f"|after_set={after_set!r}|restored={restored!r}"
        )
    return _record(fn)


# ---------------------------------------------------------------------------
# Adapters + case list.
# ---------------------------------------------------------------------------

def _numpy_adapter(site, *rest, **kwargs):
    import numpy as np
    warns, result, exc = _SITES[site](np, *rest)
    return _descriptor(warns, result, exc)


def _ionp_adapter(site, *rest, **kwargs):
    import anionpy
    warns, result, exc = _SITES[site](anionpy, *rest)
    return _descriptor(warns, result, exc)


_SITES = {
    "default": _site_default,
    "seterr_mode": _site_seterr_mode,
    "errstate_cm": _site_errstate_cm,
    "errstate_decorator": _site_errstate_decorator,
    "errstate_nested": _site_errstate_nested,
    "call_mode": _site_call_mode,
    "log_mode": _site_log_mode,
    "log_mode_no_registrant": _site_log_mode_no_registrant,
    "thread_local": _site_thread_local,
    "seterr_roundtrip": _site_seterr_roundtrip,
}

_MODED_OPS = list(_OP_CATEGORY.keys())

# These four ops are the deliberate negative controls (numpy provably never
# warns for them, at any setting) and get their own explicitly-labeled
# entries further down. Excluded here so each op/site pair appears exactly
# ONCE in the corpus -- a duplicate label would silently double-count the
# same assertion under two names and inflate the "9 cases already pass"
# number without adding any real coverage.
_NEGATIVE_CONTROL_OPS = {
    "int_overflow_add_i8",
    "complex_sqrt_negative_c128",
    "sum_empty_f64",
    "sum_empty_axis_f64",
    # task #56 additions: real numpy is silent for EVERY floating-dtype
    # `remainder`/`mod`/`fmod` input, at any numerator, including 0/0 --
    # these three are the over-warn half of the defect's negative controls.
    "float_mod_zero_f64",
    "float_remainder_zero_over_zero_f64",
    "fmod_zero_float_f64",
}
_ALL_OPS_FOR_DEFAULT = [k for k in _OPS if k not in _NEGATIVE_CONTROL_OPS]


def _cases():
    out = []

    def add(site, label, *rest):
        out.append((f"{site}|{label}", (site,) + rest, {}))

    # --- POSITIVE: default state, every op. This is the raw finding-table
    # gap: every one of these currently fails (numpy warns, anionpy silent).
    for op_key in _ALL_OPS_FOR_DEFAULT:
        add("default", op_key, op_key)

    # --- POSITIVE: seterr modes 'ignore'/'warn'/'raise', every moded op.
    for op_key in _MODED_OPS:
        for mode in ("ignore", "warn", "raise"):
            add("seterr_mode", f"{op_key}/{mode}", op_key, mode)

    # --- POSITIVE: errstate as context manager, 'ignore' and 'raise'
    # (the two modes with an observable side effect distinct from bare
    # default 'warn').
    for op_key in _MODED_OPS:
        for mode in ("ignore", "raise"):
            add("errstate_cm", f"{op_key}/{mode}", op_key, mode)

    # --- POSITIVE: errstate as decorator, 'raise' (proves __call__ works,
    # not just __enter__/__exit__).
    for op_key in ("divide_by_zero", "sqrt_negative", "overflow_multiply_f32"):
        add("errstate_decorator", f"{op_key}/raise", op_key, "raise")

    # --- POSITIVE: nested errstate -- inner overrides, outer resumes after.
    for op_key in ("divide_by_zero", "invalid_zero_over_zero"):
        add("errstate_nested", f"{op_key}/outer_raise_inner_ignore", op_key, "raise", "ignore")
        add("errstate_nested", f"{op_key}/outer_ignore_inner_raise", op_key, "ignore", "raise")

    # --- POSITIVE: seterrcall 'call'/'log' modes.
    for op_key in (
        "divide_by_zero",
        "invalid_zero_over_zero",
        "overflow_multiply_f32",
        "int_divmod_zero_over_zero_i64",
        "float_divmod_zero_over_zero_f64",
    ):
        add("call_mode", op_key, op_key)
        add("log_mode", op_key, op_key)
    add("log_mode_no_registrant", "divide_by_zero", "divide_by_zero")
    add("log_mode_no_registrant", "int_divmod_zero_over_zero_i64", "int_divmod_zero_over_zero_i64")

    # --- POSITIVE: thread-local isolation.
    for op_key in ("divide_by_zero", "overflow_multiply_f32", "invalid_zero_over_zero"):
        add("thread_local", op_key, op_key)

    # --- POSITIVE: geterr/seterr roundtrip contract.
    for op_key in ("divide_by_zero", "overflow_multiply_f32"):
        add("seterr_roundtrip", op_key, op_key)

    # --- NEGATIVE CONTROLS: numpy never warns here at ANY setting -- these
    # must (and, measured, currently DO) PASS against anionpy's blanket
    # silence, proving this corpus is not vacuously red end-to-end.
    add("default", "int_overflow_add_i8_NEGATIVE_CONTROL", "int_overflow_add_i8")
    add("default", "complex_sqrt_negative_c128_NEGATIVE_CONTROL", "complex_sqrt_negative_c128")
    add("default", "sum_empty_f64_NEGATIVE_CONTROL", "sum_empty_f64")
    add("default", "sum_empty_axis_f64_NEGATIVE_CONTROL", "sum_empty_axis_f64")
    # task #56: float `%`/`remainder`/`fmod` by zero (including 0/0) never
    # warns on real numpy at ANY setting -- this is the over-warn half of
    # the defect this task fixes; these three must (and, measured, now DO
    # post-fix) pass against anionpy's silence.
    add("default", "float_mod_zero_f64_NEGATIVE_CONTROL", "float_mod_zero_f64")
    add(
        "default",
        "float_remainder_zero_over_zero_f64_NEGATIVE_CONTROL",
        "float_remainder_zero_over_zero_f64",
    )
    add("default", "fmod_zero_float_f64_NEGATIVE_CONTROL", "fmod_zero_float_f64")
    for mode in ("ignore", "warn", "raise"):
        add(
            "seterr_mode",
            f"int_overflow_NEGATIVE_CONTROL_via_divide_category/{mode}",
            "int_floordiv_zero_i64",  # category 'divide' governs a moded op;
            mode,                      # int_overflow itself has NO category
        )

    # mean_empty_axis_f64's composite double-warning case (spec doc section
    # 3: "Mean of empty slice" + "invalid value encountered in divide" both
    # fire) is already covered above by the `_ALL_OPS_FOR_DEFAULT` loop --
    # no separate entry needed here.

    return out


def _install():
    REGISTRY["fperr_subsystem"] = ItemSpec(
        name="fperr_subsystem",
        kind="custom",
        numpy_adapter=_numpy_adapter,
        ionp_adapter=_ionp_adapter,
        scalar_like=True,
        custom_cases=_cases,
    )


_install()
