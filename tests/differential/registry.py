"""The table-driven registry of API items under differential test.

Adding a new item is meant to be a few lines here, not a new test file. See
README.md's "adding an item" section for the recipe. This module declares
*what* to test; harness.py decides *whether it passed*; corpus.py supplies
*what inputs*.

Item names use the same dotted convention as tools/numpy_surface.json /
tools/coverage.py: bare names for top-level or submodule items ("sum",
"linalg.svd"), "ndarray.<attr>" for methods/properties, and
"ndarray.<dunder>" for operator-protocol methods. That is deliberate: the
registry key IS the coverage-ledger key, so a report produced from this
registry can be handed straight to `coverage.py --tests`.

---------------------------------------------------------------------------
CALL-FORM COVERAGE (read this before adding a "method" or "binary_op" item)
---------------------------------------------------------------------------
A differential test that only ever calls an item ONE way is not testing the
item, it is testing one overload of it. numpy's own methods routinely accept
several distinct argument shapes for "the same" operation --
`a.reshape((2, 2))`, `a.reshape(2, 2)`, and `a.reshape(-1)` are three
different call sites that a real implementation must all support. A harness
that only ever tries the tuple form can credit a PASS to code that only
implements that one overload -- which is exactly the bug this module fixes
(see the 2026-07-31 postmortem: `ndarray.reshape` and `ndarray.__add__` were
both marked "exact" while `reshape(-1)`, `reshape(2, 2)`, and `array + 1`
all raised TypeError).

`CallForm` is the declarative unit for one such argument shape. `ItemSpec`
for `kind="method"` takes `call_forms: list[CallForm]` (mandatory, non-empty
-- see `__post_init__`); each form is applied to every array produced by
`corpus.unary_corpus()`. `kind="binary_op"` is the dedicated case for the
arithmetic/comparison dunders: it pairs every corpus array against every
other array (via `corpus.binary_corpus()`) AND against every scalar-like
right-hand operand (via `corpus.scalar_operands()`), because `array + array`
and `array + 1` are different code paths in any real implementation
(different unboxing, different broadcast rules for a 0-d operand, etc).

`_require_varargs_form_if_numpy_supports_it()` is the introspection half of
this: for every "method" item, it inspects numpy's OWN bound method with
`inspect.signature`. If numpy's signature declares a `VAR_POSITIONAL`
parameter (e.g. `reshape(self, *shape, ...)`) but none of the item's
declared `CallForm`s are flagged `exercises_varargs=True`, the registry
fails to import with an assertion naming the gap. This is deliberately a
hard failure, not a warning: the entire point is that a human forgetting to
list the varargs form should not be able to silently under-test an item.
Where numpy's signature can't be introspected at all (common for raw
C-implemented callables with no `__text_signature__`), the check is a no-op
-- there is nothing trustworthy to check against -- and the *only* safety
net left is the pre-existing, unconditional requirement that
`call_forms` be non-empty and explicit. That is the "fall back to the
declared form list, and require it to be explicit" behavior called for in
the harness's design brief; it is enforced by `__post_init__` below, not
just documented.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import math
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import anionpy

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from harness import InvalidCase

MIN_ULP_SWEEP_N = 20000
"""Minimum sample size, per applicable dtype, an `ItemSpec.ulp_sweep` entry
must record before `ulp_tolerance` may be declared (2026-08-01, sample-size
defect fix). Below this size a discrepancy can simply not be in the sample
-- `absolute`/`abs`'s original 1.0 ULP bound, measured over the ~150-case
differential corpus, was already wrong at this size (true bound: 2.0 ULP,
found by a 20,000-value sweep; see
reports/ionp-ulp-tolerance-decision-2026-08-01.md's correction block). See
`ulp_sweep.py` for how the sweep evidence is produced.

2026-08-01 (per-dtype grading fix): the same postmortem also identified a
SECOND defect one layer up -- even after the sample-size fix, `ulp_tolerance`
was still a single scalar applied to every dtype an item's corpus touches.
`absolute`/`abs`'s own re-swept evidence is itself per-dtype (float32/
float64 bit-exact at 0.0 ULP, complex64/complex128 genuinely imprecise at
2.0 ULP) -- declaring one item-wide bound of 2.0 ULP graded the bit-exact
real dtypes with 2 ULP of unearned slack, which would let a real float32/
float64 regression pass silently. `ulp_tolerance` is therefore now
`Optional[dict[str, float]]`: `{dtype_name: bound}`, graded per dtype (see
harness.compare_values). `ItemSpec.__post_init__` below enforces that this
dict's key set is EXACTLY the `ulp_sweep` key set -- neither a tolerance
without matching sweep evidence nor swept-but-undeclared dtype is allowed,
so a dtype can never silently inherit another dtype's bound. A dtype with no
entry in either dict at grading time (i.e. not part of this item's declared,
evidenced set at all) is graded bit-exact -- see compare_values's `.get(...,
0.0)` -- which is the deliberate, enforced safe default, not an accident of
missing keys."""


def _numpy_exc_type(fn, *args, **kwargs):
    """Captures real numpy's own exception CLASS for a known-raising call,
    without hardcoding its (private, `numpy._core._exceptions`-internal,
    not-guaranteed-stable-across-versions) import path. Same pattern as
    `strings_cases.py`/`fft_cases.py`'s own `_numpy_exc_type` helpers,
    duplicated here (rather than imported from either) because this one
    backs a registry-wide DEFAULT for the ~134 auto-generated `kind="ufunc"`
    items built by `ufunc_registry.py`, not a single file's hand-declared
    items -- see `_DEFAULT_UFUNC_EXC_EQUIV` below."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - probing numpy's own exception type, not app code
        return type(exc)
    raise AssertionError(f"{fn} did not raise on {args!r}/{kwargs!r} while probing numpy's exception type")


# UPDATED 2026-08-02 (bug found and fixed this same day, see
# `ionp-py/src/lib.rs`/`fft.rs`/`reductions.rs` doc comments): the
# 2026-08-01 comment below described `anionpy` as raising a plain, Rust-native
# `TypeError` for these two shapes and declared THAT as the accepted
# substitution. That was itself the bug this default equivalence was
# quietly papering over -- numpy raises a private, unimportable subclass
# of `TypeError` for both shapes (`_UFuncNoLoopError`/
# `_UFuncOutputCastingError`, both displayed as `UFuncTypeError` via
# numpy's `@_display_as_base`), and `anionpy` was raising plain `TypeError`
# instead of its own already-existing `UFuncTypeError`
# (`_anionpy.UFuncTypeError`, see `ionp-py/src/lib.rs`'s
# `create_exception!(_anionpy, UFuncTypeError, PyTypeError)`) -- a real class
# divergence invisible to `isinstance`/`except TypeError` checks (since
# `UFuncTypeError` IS-A `TypeError`) and invisible in this harness's
# printed diagnostics too (both render as the string "UFuncTypeError" via
# `type(e).__name__`), only caught by comparing the exact type object.
# `ionp-py`'s raise sites are now fixed to raise `_anionpy.UFuncTypeError`
# for both shapes. Since numpy's own matching class is PRIVATE (no public
# `numpy.exceptions.UFuncTypeError` alias exists to import and compare
# against directly -- see this file's `_numpy_exc_type` docstring), exact
# type-object identity between the two implementations' exceptions is
# structurally impossible; this default equivalence is what makes the
# comparison "exact" in the only sense available: numpy's own captured
# private class is declared equivalent to anionpy's own real, stable, public
# `UFuncTypeError` -- never to plain `TypeError` anymore. This default
# declares that substitution to the harness for every `kind="ufunc"` item
# that doesn't already declare its own `exception_equivalences` -- the
# ~134 items in `ufunc_registry.py`, which is not this file and is not
# touched here; `ItemSpec.__post_init__` below applies this default at
# construction time regardless of which module builds the ItemSpec, so it
# reaches those items without editing that file. Both probes are real
# numpy calls verified live (not guessed): unary no-loop via `np.sign` on
# bool (matches lib.rs's `Sign`-on-bool and `positive`'s bool-rejection
# raise sites), output-casting via `np.add` with an incompatible `out=`
# dtype (matches lib.rs's in-place-op raise site).
#
# CORRECTED 2026-08-02, THIRD PASS (the sentence this replaces, AND the
# replacement that followed it for a few hours the same day, were each
# wrong in a different way -- read this fully before touching this area
# again): the comparison-reduce no-loop shape
# (`np.equal.reduce`/`.accumulate`/`.reduceat` on a non-bool dtype) does
# NOT reliably raise the same `_UFuncNoLoopError` class this file's
# `np.sign` probe captures -- but it is not reliably plain `TypeError`
# either. It is genuinely BOTH, depending on real numpy's own internal
# per-(ufunc, exact input dtype) type-resolution cache state: cold (no
# prior elementwise call for that exact op+dtype pair in this process) ->
# plain `builtins.TypeError`; warmed-for-that-exact-dtype (an elementwise
# `np.equal(bool_arr, that_dtype_arr)` has resolved earlier in the SAME
# process, for ANY reason, including an unrelated earlier item elsewhere
# in this ~1180-item corpus) -> the rich `_UFuncNoLoopError`
# (`UFuncTypeError`). Minimal repro, one process:
#     np.equal.reduce(np.array([1,2,3], dtype=np.int32))            # cold: plain TypeError
#     np.equal(np.array([False]), np.array([4], dtype=np.int32))    # warms ONLY the int32 pair
#     np.equal.reduce(np.array([1,2,3], dtype=np.int32))             # now: rich UFuncTypeError
# This means a live gate run shows mismatches in BOTH directions for the
# SAME op in the SAME process -- e.g. `equal`'s `int32`/`float64` reduce
# cases land warm (rich class expected) while its `uint8`/`uint32`/
# `float16`/`uint16` reduce cases land cold (plain class expected), purely
# because of which OTHER items in the corpus happened to touch each dtype
# pair first. `anionpy` cannot decide this per-call without mirroring numpy's
# exact internal cache state across the entire corpus's call history, which
# is not available to it (or to any sane implementation) at raise time.
# `ionp-py/src/lib.rs`'s `to_py_err_compare_reduce` deliberately keeps
# raising `UFuncTypeError` unconditionally -- measured, not assumed, to be
# the better of the two fixed choices against the real gate (9/382
# residual failures per comparison op vs. 18/382 for the "always plain
# TypeError" alternative, which was tried and reverted the same day). This
# default equivalence entry is therefore correct exactly when real numpy is
# in the warm state, and the residual mismatches when it is cold are a
# documented, understood INSTRUMENT ARTIFACT of this differential
# methodology's global mutable process state, not a fixable `anionpy` bug and
# not something this equivalence dict can paper over (numpy's cold-state
# class is plain `TypeError`, a genuinely different type object than either
# `_UFUNC_NO_LOOP_ERROR` or `anionpy._anionpy.UFuncTypeError`). A same-day probe
# script (`/tmp/cmpclass.py`) that always calls `.reduce` as the very first
# touch of that op+dtype pair in a fresh interpreter only ever samples the
# COLD state and so cannot be used alone to judge this shape -- always
# cross-check against the full gate (`tests/differential/run.py`), which
# carries the corpus's real warm state, before changing anything here.
_UFUNC_NO_LOOP_ERROR = _numpy_exc_type(np.sign, np.array([True, False]))
_UFUNC_OUTPUT_CASTING_ERROR = _numpy_exc_type(
    np.add,
    np.array([1, 2], dtype=np.int64),
    np.array([1, 2], dtype=np.int64),
    out=np.empty(2, dtype=np.bool_),
)
_DEFAULT_UFUNC_EXC_EQUIV = {
    _UFUNC_NO_LOOP_ERROR: {anionpy._anionpy.UFuncTypeError},
    _UFUNC_OUTPUT_CASTING_ERROR: {anionpy._anionpy.UFuncTypeError},
}

# ADDED 2026-08-02, closing PASS 3 above (read that comment first -- this
# is its continuation, not a competing theory): the 9/382-per-comparison-op
# residual documented above is a real cold/warm split in real numpy, but
# `run_case` (harness.py) ALREADY requires exact, byte-normalized MESSAGE
# equality on top of whatever the type-equivalence dict allows (see that
# function's `np_msg != ionp_msg` arm) -- and both the coordinator's and
# this agent's independent measurements found 0 message mismatches for
# every one of these cases, cold or warm. That means the cold-state plain
# `TypeError` and the warm-state rich `_UFuncNoLoopError` are, for this one
# shape, carrying IDENTICAL text -- i.e. numpy is describing the exact same
# failure through two different wrapper classes depending on unrelated
# prior process history, not two different failures. That is real evidence
# of sameness, not a coincidence to be suspicious of: the message text is
# generated deep in numpy's type-resolution code from the operand dtypes
# and ufunc name alone, before the wrapper-class decision is made, so both
# paths produce it identically.
#
# `_COMPARISON_REDUCE_EXC_EQUIV` below is what makes anionpy's (always-rich)
# `UFuncTypeError` accepted against BOTH of numpy's possible classes for
# this shape, WITHOUT loosening grading anywhere else:
#
#   sign(bool)                   cold UFuncTypeError / warm UFuncTypeError   STABLE, unaffected
#   add(int64,int64,out=bool_)   cold UFuncTypeError / warm UFuncTypeError   STABLE, unaffected
#   equal/greater/.../.reduce    cold TypeError       / warm UFuncTypeError   the shape this covers
#   equal/greater/.../.accumulate  same as above
#   equal/greater/.../.reduceat    same as above
#
# This is deliberately NOT folded into `_DEFAULT_UFUNC_EXC_EQUIV` (which
# `ItemSpec.__post_init__` applies to all ~134+ kind="ufunc"/hand-written
# items with no explicit `exception_equivalences` of their own). A global
# `TypeError: {anionpy._anionpy.UFuncTypeError}` entry would accept plain
# `TypeError` from numpy ANYWHERE anionpy raises `UFuncTypeError` -- including
# the two STABLE raise sites above, where numpy's class never varies. If
# `ionp-py/src/lib.rs` ever regressed and started raising plain `TypeError`
# for `sign` on bool (a real, structurally-different bug: it would mean
# anionpy had lost track of which exception class it means to be raising), a
# global rule would grade that regression a PASS, because "numpy said
# TypeError" would trivially match. Message equality alone is not a
# sufficient guard against that either, in principle: a coincidentally-
# matching message text on a genuinely different bug is not something this
# comment wants to rely on. So the equivalence is applied per-item, only to
# the six comparison ufuncs' `ItemSpec`s in `ufunc_registry.py`
# (`equal`/`not_equal`/`greater`/`greater_equal`/`less`/`less_equal`), which
# is the finest granularity `ItemSpec.exception_equivalences` supports
# (per-ufunc, not per-call-form) -- acceptable here because the no-loop
# shape this covers can only ever be reached through those six ufuncs'
# `.reduce`/`.accumulate`/`.reduceat` call forms in the first place: their
# `call`/`outer`/`at` forms never hit this no-loop path (comparison output
# is always bool and elementwise casting always succeeds), so scoping by
# ufunc name does not accidentally widen the shape. See
# `ufunc_registry.py`'s `_build_ufunc_specs` for where this is wired in.
_COMPARISON_REDUCE_EXC_EQUIV = dict(_DEFAULT_UFUNC_EXC_EQUIV)
_COMPARISON_REDUCE_EXC_EQUIV[TypeError] = {anionpy._anionpy.UFuncTypeError}

# CORRECTED AGAIN, same day (this replaces the "NEW FINDING" note that used
# to sit here, which claimed "numpy's message for this shape is the SHORT
# generic form in BOTH cold and warm state" -- that claim was FALSE, and was
# never actually observed directly: it was inferred from the gate log's
# printed failure details, which structurally can only ever show numpy's
# message for a FAILING case, i.e. one that landed COLD in that corpus
# position. Warm/matching cases pass silently and never print numpy's
# message, so "always short" was one observation point (the cold state)
# mistaken for a property of the whole axis -- read PASS 1/2/3 above; this
# is the same mistake shape, caught before it was implemented in `.rs` this
# time instead of after).
#
# Direct fresh-process measurement (30+ samples, both the empty and
# non-empty comparison-reduce shapes, several dtypes each, one process per
# sample) shows real numpy's CLASS and MESSAGE for this whole shape family
# are not two independent properties -- they are ONE coupled, two-state
# artifact of numpy's own per-(ufunc, exact dtype) type-resolution cache:
#
#   COLD or warmed-SAME-dtype (e.g. equal(int32, int32) doesn't warm the
#   bool x int32 pair):
#       plain `builtins.TypeError`, SHORT generic message:
#       "No loop matching the specified signature and casting was found
#        for ufunc {name}"
#   WARM-MIXED (an elementwise `np.equal(bool_arr, that_dtype_arr)` has
#   resolved earlier in this process, for ANY reason):
#       rich `_UFuncNoLoopError` (displays `UFuncTypeError`), LONG message:
#       "ufunc '{name}' did not contain a loop with signature matching
#        types (<class 'numpy.dtypes.BoolDType'>, <class
#        'numpy.dtypes.{Dtype}DType'>) -> None"
#
# No other combination -- (TypeError, LONG) or (UFuncTypeError, SHORT) --
# was ever observed, for either the empty or non-empty shape, across every
# sample. `ionp-core/src/ufunc.rs`'s `compare_reduce_error` was UNIFIED
# 2026-08-02 (see that function's doc) to always emit the WARM pair
# (`NoUfuncLoop` -> `UFuncTypeError` + LONG message) for both shapes,
# rather than hardcoding a different fixed guess per shape -- deliberate,
# not by omission: `anionpy` cannot replicate numpy's own process-history
# cache and must not try. That leaves exactly one gap: numpy's COLD state
# (TypeError + SHORT) against anionpy's fixed WARM pair. Since the two states
# are coupled, this cannot be closed by a class equivalence plus a loosened
# message check (that would also silently accept (UFuncTypeError, SHORT),
# a combination real numpy never produces) -- so it needs a predicate that
# validates the PAIR, not the class and message independently. See
# `_comparison_reduce_message_pair_ok` below, wired in via
# `ItemSpec.message_pair_ok` (harness.py's `run_case`), which is checked
# ONLY as a fallback after exact message equality has already failed.
_UFUNC_NO_LOOP_SHORT_TEMPLATE = (
    "No loop matching the specified signature and casting was found for ufunc {name}"
)
# TIGHTENED 2026-08-02 (coordinator review): the second dtype slot used to be
# the open wildcard `\w+DType`, which -- besides matching every real dtype
# class -- also matches any misspelling ("Uint8DType"), any non-dtype
# `\w+DType`-shaped garbage, and, structurally, "BoolDType" itself in that
# position (that specific swap is separately blocked by the first slot being
# pinned to the literal "BoolDType", not by anything about the second slot's
# pattern -- confirmed: `(BoolDType, UInt8DType)` matches, `(UInt8DType,
# BoolDType)` does not, because slot one requires the literal text
# "BoolDType" and "UInt8DType" != "BoolDType"; the wildcard was never what
# was preventing that particular swap). Replaced with an explicit
# alternation of the real non-bool `dtype_class_name` outputs
# (`ionp-py/src/lib.rs`), so a garbage or misspelled class name in that slot
# is now rejected outright, not just "happens not to occur in practice".
#
# KNOWN LIMITATION (TODO, tracked here so it cannot quietly become
# permanent) -- not closed by this tightening, flagged rather than hidden:
# this predicate is built ONCE per ufunc name
# (`_comparison_reduce_message_pair_ok` is called once in
# `ufunc_registry.py`'s `_build_ufunc_specs`, its closure shared across
# every dtype the corpus tests for that ufunc) and is handed only the two
# exception OBJECTS, not the actual input dtype of the specific call under
# test. So it can confirm "anionpy's message names A REAL non-bool dtype in
# the right slot" but not "anionpy's message names THE dtype this exact call
# actually used" -- e.g. it cannot catch anionpy reporting Float16DType for a
# call that was actually made with a uint8 array. Real numpy's own SHORT
# cold-state message (the only thing available on the numpy side for this
# predicate) carries no dtype information at all to check against.
#
# WORSE THAN IT SOUNDS (coordinator's sharper framing, 2026-08-02, recorded
# verbatim rather than restated weaker): this predicate only ever FIRES
# when real numpy is COLD -- a warm-numpy LONG message gets caught by the
# ordinary exact-match check above, before this predicate is ever
# consulted, so a wrong-dtype anionpy bug is still caught there in full. That
# means a hypothetical "names the wrong dtype" anionpy defect would be CAUGHT
# on warm-cache corpus positions and INVISIBLE on cold-cache ones for the
# exact same op/dtype pair -- i.e. whether this specific class of bug gets
# detected would itself depend on numpy's own process-history cache state,
# the identical nondeterminism this whole mechanism exists to neutralize
# on the PASS/FAIL side. Closing this needs the actual call's input dtype
# threaded into `message_pair_ok` from `harness.py`'s `run_case` -- in
# scope (harness.py is on the authorized edit list), but deliberately NOT
# done in this same change: coordinator-directed sequencing is build
# window -> four anti-vacuity checks -> full gate -> real numbers -> THEN
# this as its own increment with its own anti-vacuity check, specifically
# so an unmeasured Rust change, an unmeasured harness change, and a third
# unmeasured harness change don't all land before any of them has touched
# a real gate run. Do not let this comment go stale once that increment
# lands -- either the dtype-threading fix removes this whole limitation
# and this note should be deleted, or it's still open and this note should
# still say so.
_UFUNC_DTYPE_CLASS_NAMES = (
    "Int8DType", "Int16DType", "Int32DType", "Int64DType",
    "UInt8DType", "UInt16DType", "UInt32DType", "UInt64DType",
    "Float16DType", "Float32DType", "Float64DType",
    "Complex64DType", "Complex128DType",
)
_UFUNC_NO_LOOP_LONG_RE_TEMPLATE = (
    r"^ufunc '{name}' did not contain a loop with signature matching types "
    r"\(<class 'numpy\.dtypes\.BoolDType'>, <class 'numpy\.dtypes\.(?:"
    + "|".join(_UFUNC_DTYPE_CLASS_NAMES)
    + r")'>\) -> None$"
)


def _comparison_reduce_message_pair_ok(ufunc_name: str) -> Callable:
    """Factory: returns an `ItemSpec.message_pair_ok` predicate scoped to one
    comparison ufunc's name, for the coupled cold/warm pair documented
    above. Deliberately validates BOTH exceptions' class AND text as a
    single unit rather than treating "message roughly matches" as
    sufficient -- see the four anti-vacuity checks in the commit this
    landed with (break anionpy's message alone -> must fail; break anionpy's
    class alone -> must fail; corrupt the long-form template by one
    character -> must fail; break the unrelated stable sign-on-bool shape
    -> must still fail, unaffected by this predicate, since it is never
    even consulted there -- `sign` doesn't set `message_pair_ok`).
    """
    short_msg = _UFUNC_NO_LOOP_SHORT_TEMPLATE.format(name=ufunc_name)
    long_re = re.compile(_UFUNC_NO_LOOP_LONG_RE_TEMPLATE.format(name=re.escape(ufunc_name)))

    def _pair_ok(np_exc: BaseException, ionp_exc: BaseException) -> bool:
        if type(np_exc) is not TypeError:
            return False
        if type(ionp_exc) is not anionpy._anionpy.UFuncTypeError:
            return False
        if str(np_exc) != short_msg:
            return False
        return bool(long_re.match(str(ionp_exc)))

    return _pair_ok


def _resolve_dotted(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            raise AttributeError(f"{dotted!r}: no attribute {part!r} on {obj!r}")
        obj = getattr(obj, part)
    return obj


def make_ionp_array_converter(ionp_module, nd_cls):
    """Factored out of the "ndarray."-prefixed resolve_ionp() branch below
    (2026-07-31 fix) so kind="ufunc" items (added for the ufunc block, see
    ufunc_registry.py) get the identical protection: the corpus hands out
    real numpy.ndarray instances (numpy IS the reference implementation),
    and calling an anionpy function/method with them un-converted would either
    (a) crash on a type anionpy doesn't accept, masking the real question, or
    (b) for "ndarray."-prefixed lookups specifically, silently resolve the
    attribute on the numpy object itself -- a tautology that can never
    fail. Every numpy.ndarray operand is converted to a real anionpy.ndarray;
    anything else (python/numpy scalars, dtypes, index objects, ...) is
    passed through unconverted, since converting those would defeat the
    scalar/varargs call-form coverage this suite exists to provide.

    2026-08-02 addition: recurses into `tuple`/`list` (mirroring
    `harness._freshen`'s own tuple/list recursion, same reasoning) so a
    multi-output ufunc's `out=(buf0, buf1)` -- or a partial `out=(buf,
    None)` -- gets each ARRAY member converted individually, `None`
    members passed through unchanged. Without this, `_wrap_custom_conversion`
    (the only place kind="custom" items route numpy.ndarray kwargs through
    this converter) left the tuple itself untouched, so every element
    inside it reached the anionpy call as a raw numpy.ndarray -- exactly the
    type `write_into_out_ufunc`'s `.cast::<PyArray>()` rejects with "return
    arrays must be of ArrayType". Found via `ufunc_out_axis_cases.py`'s
    `out=` corpus for divmod/frexp/modf (the only ufuncs whose `out=` is
    ever a tuple): every "match"/"partial" case failed this way before this
    fix, for a reason that had nothing to do with the two engine defects
    that corpus was written to catch. Single ndarrays (float_power/ldexp's
    bare `out=`) were never affected -- they hit the `isinstance(x,
    np.ndarray)` branch above directly.
    """
    def _to_ionp(x):
        if isinstance(x, nd_cls):
            return x
        if isinstance(x, np.ndarray):
            return ionp_module.array(x)
        if isinstance(x, tuple):
            return tuple(_to_ionp(v) for v in x)
        if isinstance(x, list):
            return [_to_ionp(v) for v in x]
        return x
    return _to_ionp


def _load_exploded_prefixes() -> "frozenset[str]":
    """The full set of curated exploded-class prefixes ("ndarray", "dtype",
    "ma.MaskedArray", "random.Generator", "matrix", ...), read straight
    from `tools/numpy_surface.json`'s `"exploded"` keys -- the SAME source
    `tools/coverage.py`'s own `all_items()`/`resolve()` already read (see
    that file). Loaded once, at import time, so `resolve_numpy()`/
    `resolve_ionp()` below can recognize ANY curated class generically
    instead of the pre-existing code recognizing only the literal string
    "ndarray." -- which was the exact gap this module was asked to close:
    every other exploded class's items (`dtype.kind`, `ma.MaskedArray.
    filled`, `random.Generator.normal`, ...) fell through to
    `_resolve_dotted(np_or_ionp, path)`, which can never succeed for a
    "<class>.<method>" key (there is no top-level `np.dtype.kind` or
    `anionpy.dtype.kind` attribute -- `dtype` is a CLASS, `kind` is a
    per-INSTANCE property, reachable only via a constructed receiver, not
    via dotted attribute lookup against the package object), so every one
    of those items silently resolved to `None` and was misreported as
    unresolved by both sides equally.

    Read from the manifest rather than hardcoded here for the same reason
    `coverage.py`'s `resolve()` reads it that way (see that function's own
    docstring): a class added to/removed from `snapshot_surface.py`'s
    `CURATED_EXPLODE` would otherwise require a second, easy-to-forget
    edit here to stay in sync.

    Falls back to a hardcoded literal containing only "ndarray" if the
    manifest is missing -- this can only happen if someone runs this suite
    without ever having generated the manifest, in which case every
    exploded item was already unresolvable before this change (nothing
    regresses); it must NOT raise here, since raising at import time would
    take down the entire registry (every other, unrelated item) over a
    problem specific to the ~2500 exploded-class items alone.
    """
    surface_path = Path(__file__).resolve().parents[2] / "tools" / "numpy_surface.json"
    try:
        surface = json.loads(surface_path.read_text())
        return frozenset(surface["exploded"].keys())
    except (OSError, ValueError, KeyError):
        return frozenset({"ndarray"})


_EXPLODED_PREFIXES = _load_exploded_prefixes()


def _exploded_prefix_of(path: str):
    """If `path` is a "<curated-class>.<method-or-dunder>" item key, return
    the curated-class prefix it names (e.g. "ma.MaskedArray" for
    "ma.MaskedArray.filled", "dtype" for "dtype.kind"); else None.

    Longest-prefix-first is not needed here despite prefixes like "dtype"
    vs (hypothetically) "dtype.something_longer" both being in the set,
    because `_EXPLODED_PREFIXES` (see `_load_exploded_prefixes` above) is
    the curated CLASS NAME set, not a set of arbitrary path fragments --
    no two entries in it are themselves prefixes of one another (verified:
    "ma.MaskedArray" and "ma" are not both curated class names, only the
    former is), so at most one entry can ever match a given `path`.
    """
    for prefix in _EXPLODED_PREFIXES:
        if path.startswith(prefix + "."):
            return prefix
    return None


assert None is _exploded_prefix_of("dtype")  # bare class name, no dot -> not an item key


def _ionp_exploded_class(ionp_module, prefix: str):
    """Resolve the real anionpy class object a curated exploded-class prefix
    names (walking dotted prefixes like "ma.MaskedArray" component by
    component), or None if anionpy has no such class at all.

    None is a legitimate, CORRECT outcome for 12 of the 18 curated
    classes (`matrix`, `recarray`, `memmap`, `poly1d`, `char.chararray`,
    `random.RandomState`, and the 6 `polynomial.*` bases) -- anionpy simply
    does not implement them, and this function must never fabricate,
    stub, or alias one into existence to make an item "resolve"; a
    resolution failure here is data (correctly reported "absent"/"fail"
    by evaluate() in harness.py), not a defect in this function.
    """
    obj = ionp_module
    for part in prefix.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _convert_dtype_receiver(ionp_module, np_dtype):
    """numpy hands out a real `numpy.dtype` instance as the receiver for
    every `dtype.<attr>` item (see corpus builder in exploded_class_cases.py).

    TICKET #78 UPDATE (2026-08-08): this used to build the anionpy-side
    receiver by round-tripping through a throwaway array
    (`ionp_module.array(np.zeros(1, dtype=np_dtype)).dtype`) with a comment
    claiming `anionpy.dtype(...)` "cannot be constructed directly." That
    claim is now stale -- `anionpy.dtype(...)` has had a public `#[new]`
    constructor since before this ticket (see `PyDType::new` in
    ionp-py/src/lib.rs) -- and the array round-trip has an actual
    correctness cost this ticket's own fix exposed: `PyArray::dtype()`
    always returns a plain, untagged `PyDType` (`spelling: None`, see that
    getter), so for numpy's "duplicate" dtypes ('q'/'Q'/'g'/'G' and their
    C-name aliases -- distinct char-code spellings that canonicalize to the
    same DType as a different spelling but report a genuinely different
    `.char`/`.num` on the bare dtype object, verified live against numpy
    2.5.1) the array round-trip silently collapsed the receiver to its
    canonical twin (`np.dtype('q')` -> an anionpy dtype reporting `.char ==
    'l'`, not `'q'`) before `.char`/`.num` were ever read -- not a
    real anionpy bug, just an artifact of this converter's own construction
    path, that would have made this ticket's fix look broken by every
    "duplicate spelling" case added to the corpus below.
    Reconstructing directly via `ionp_module.dtype(np_dtype.char)` instead
    avoids that: `np_dtype.char` IS numpy's own already-spelling-aware
    single-character code (`np.dtype('q').char == 'q'`,
    `np.dtype('int64').char == 'l'`), and `anionpy.dtype(...)` accepts
    every one of those codes and tags the duplicate ones exactly the way
    `anionpy.dtype('q')` called directly would (see `duplicate_dtype_spelling`
    in ionp-py/src/lib.rs). Verified live to produce IDENTICAL
    `.name`/`.itemsize`/`.char`/`.num` to the old array-round-trip path for
    all 14 canonical dtypes (the only receivers this converter was ever
    exercised against before this ticket), so this is a pure widening, not
    a behavior change for any pre-existing passing item. Only S/U flexible
    dtypes would be mishandled by `.char` (`np.dtype('S5').char == 'S'`
    loses the width) -- not a regression, because no S/U dtype has ever
    been a `dtype.<attr>` receiver in this corpus (`_DTYPE_NAMES` in
    exploded_class_cases.py is the fixed-14 list only).
    """
    return ionp_module.dtype(np_dtype.char)


def _convert_finfo_receiver(ionp_module, np_finfo):
    """`anionpy.finfo(dtype)` is directly constructible (unlike anionpy.dtype)
    and accepts a real numpy dtype object directly -- verified directly
    (`anionpy.finfo(np.dtype('float64'))` succeeds and matches
    `anionpy.finfo('float64')`)."""
    return ionp_module.finfo(np_finfo.dtype)


def _convert_iinfo_receiver(ionp_module, np_iinfo):
    """Mirrors _convert_finfo_receiver -- `anionpy.iinfo(dtype)` also accepts
    a real numpy dtype object directly."""
    return ionp_module.iinfo(np_iinfo.dtype)


def _reconstruct_np_maskedarray(receiver):
    """MEASURED, load-bearing (2026-08-05): `harness.run_case()`
    unconditionally passes every case's args through `_freshen()` before
    either side is ever called (see that function in harness.py, out of
    edit scope for this task) -- and `_freshen()` special-cases
    `isinstance(x, np.ndarray)` by rebuilding a bare `np.ndarray` from the
    raw buffer via `_freshen_array()`. `numpy.ma.MaskedArray` IS an
    `np.ndarray` subclass, so a real MaskedArray handed in as a case
    argument is SILENTLY DOWNGRADED to a plain ndarray before this
    function (or resolve_ionp's matching branch) ever sees it -- confirmed
    directly: a receiver that entered `_freshen` as
    `type(x).__name__ == 'MaskedArray'` comes back `'ndarray'`, mask
    information gone. This is a genuine, narrow harness limitation
    (reported, not fixed -- harness.py is out of this task's edit scope),
    not a defect in this dispatch or in anionpy.

    The workaround, entirely within registry.py: `ma.MaskedArray`'s corpus
    (exploded_class_cases.py's `_maskedarray_receiver_cases`) hands the
    case a `(data_ndarray, mask_ndarray)` PLAIN TUPLE instead of a real
    MaskedArray -- `_freshen` recurses into a tuple element-wise (see its
    own docstring), freshening `data`/`mask` independently as ordinary
    ndarrays and leaving the 2-tuple structure intact, so BOTH survive
    `_freshen` untouched in shape. This function (numpy side) and
    `_convert_maskedarray_receiver` (anionpy side, below) each rebuild a real
    MaskedArray from that surviving tuple, on their OWN side, from data
    that has ALREADY been through `_freshen` -- so this is not bypassing
    the mutation-safety `_freshen` exists to provide, only working around
    its ndarray-subclass blind spot for this one class.
    """
    data, mask = receiver
    return np.ma.MaskedArray(data, mask=mask)


def _convert_maskedarray_receiver(ionp_module, receiver):
    """ionp-side counterpart of `_reconstruct_np_maskedarray` -- see that
    function's docstring for why the corpus hands a `(data, mask)` tuple
    rather than a real MaskedArray, and why rebuilding on each side from
    the post-`_freshen` tuple is correct, not a mutation-safety bypass.
    """
    data, mask = receiver
    return ionp_module.ma.MaskedArray(np.asarray(data), mask=np.asarray(mask))


_EXPLODED_RECEIVER_CONVERTERS = {
    # "ndarray" is deliberately NOT here: its own, older, more heavily
    # commented conversion path (make_ionp_array_converter, applied
    # inline in the "ndarray."-prefix branch of resolve_ionp() below) is
    # left completely untouched by this generalization -- it already
    # worked, handles the scalar-receiver tautology hazard with its own
    # narrower InvalidCase guard (see that branch's 2026-08-01 comment),
    # and every other kind="ufunc"/"custom" item in this file already
    # depends on its exact behavior. Duplicating or routing it through
    # this table would risk a behavior change to ~1300 pre-existing,
    # already-passing items for zero benefit.
    "dtype": _convert_dtype_receiver,
    "finfo": _convert_finfo_receiver,
    "iinfo": _convert_iinfo_receiver,
    "ma.MaskedArray": _convert_maskedarray_receiver,
    # "random.Generator" is deliberately NOT here: a live numpy Generator
    # cannot be "converted" to an anionpy one by this table's (ionp_module,
    # np_receiver) -> ionp_receiver shape -- doing so would require
    # reproducing the numpy Generator's internal PCG64 bit-generator
    # STATE, not just constructing something of the right class, and the
    # corpus has no way to hand this converter that state (a
    # numpy.random.Generator does not expose "the seed it was built
    # from," only its opaque current state). random.Generator.<method>
    # items are therefore wired up in exploded_class_cases.py via
    # kind="custom" + explicit numpy_adapter/ionp_adapter pairs that each
    # construct BOTH sides fresh from the SAME seed -- the same,
    # already-established pattern random_cases.py uses for the coarser
    # "random.Generator" item -- rather than through this generic
    # receiver-conversion table. The remaining 12 curated classes anionpy
    # does not implement at all are also, correctly, absent from this
    # table -- see _ionp_exploded_class's docstring.
}

_EXPLODED_NUMPY_TYPES = {
    # The real numpy type each convertible prefix's receiver must be an
    # instance of -- mirrors the "ndarray."-prefix branch's own
    # `isinstance(arr, (np.ndarray, nd))` tautology guard (see that
    # branch's 2026-08-01 comment for the full hazard this defends
    # against: a non-instance receiver silently resolving the attribute
    # on numpy's OWN object instead of exercising anionpy at all).
    "dtype": np.dtype,
    "finfo": np.finfo,
    "iinfo": np.iinfo,
    "ma.MaskedArray": np.ma.MaskedArray,
}


def build_ufunc_dispatcher(obj, to_ionp=None):
    """Turn a ufunc-like `obj` (a real numpy ufunc, a real anionpy ufunc once
    one exists, or -- in selftest.py -- a hand-built fake exposing the same
    __call__/.reduce/.accumulate/.outer/.reduceat/.at protocol) into the
    single callable ItemSpec.resolve_numpy()/resolve_ionp() hand back for
    kind="ufunc" items.

    Cases built by ufunc_cases.py always start with a call-form tag as
    their first positional argument ("call" | "reduce" | "accumulate" |
    "outer" | "reduceat" | "at"); this function is where that tag gets
    turned into the actual dispatch, for both the numpy side and the anionpy
    side, via the SAME code path -- so there is exactly one place that
    could get "reduce means call .reduce" wrong, not two.

    Two things this function is responsible for that a naive
    `getattr(obj, form)(*rest, **kwargs)` would get wrong:

    1. **Mutation safety.** `harness._call()` invokes the numpy-side
       callable and the ionp-side callable with the literal SAME `args`/
       `kwargs` tuple/dict objects (see run_case()). `.at()` mutates its
       first argument in place and `out=`/`where=`+`out=` write into the
       array passed as `out`. Without defensive copying here, the numpy
       call's mutation would leak into the array the anionpy call then reads
       -- corrupting the second call's input with the first call's output,
       which can either mask a real anionpy bug (if anionpy copies the already-
       correct numpy answer instead of computing its own) or fail a
       correct anionpy implementation (if it starts from already-mutated
       data). Both sides get a fresh `.copy()` of any mutable target
       immediately before use.
    2. **anionpy conversion.** `to_ionp`, when given (the anionpy side only),
       converts numpy.ndarray positional/`out`/`where` arguments to real
       anionpy.ndarray before dispatch -- see make_ionp_array_converter's
       docstring for why this is not optional.
    """
    conv = to_ionp or (lambda x: x)

    def fn(form, *rest, **kwargs):
        rest2 = tuple(conv(r) for r in rest)
        kw = {}
        for k, v in kwargs.items():
            if k in ("out", "where") and v is not None:
                v = conv(v)
            kw[k] = v
        if kw.get("out") is not None and hasattr(kw["out"], "copy"):
            kw["out"] = kw["out"].copy()

        if form == "at":
            target = rest2[0].copy() if hasattr(rest2[0], "copy") else rest2[0]
            obj.at(target, *rest2[1:], **kw)
            return target
        if form == "call":
            return obj(*rest2, **kw)
        method = getattr(obj, form, None)
        if method is None:
            raise AttributeError(form)
        return method(*rest2, **kw)

    return fn


@dataclasses.dataclass(frozen=True)
class CallForm:
    """One argument-shape variant of a method call.

    label             -- short, unique-within-item name, folded into the
                          case label as "<corpus-case>/<label>".
    build             -- fn(arr) -> (args_tuple, kwargs_dict). Called once
                          per corpus array; `arr` is the numpy (reference)
                          array being reused for both the numpy call and the
                          anionpy call, so it MUST NOT be mutated.
    applicable        -- fn(arr) -> bool. Some forms only make sense for
                          certain shapes (e.g. an axis-permutation form
                          needs ndim >= 1); forms that aren't applicable are
                          skipped for that corpus array rather than forced
                          to raise a shape error that isn't the thing under
                          test. Defaults to "always applicable".
    exercises_varargs -- mark True if this form calls the method with
                          `*args` expansion (e.g. `reshape(2, 2)` as opposed
                          to `reshape((2, 2))`). Read by
                          `_require_varargs_form_if_numpy_supports_it`.
    """

    label: str
    build: Callable[[object], tuple[tuple, dict]]
    applicable: Callable[[object], bool] = lambda arr: True
    exercises_varargs: bool = False


@dataclasses.dataclass
class ItemSpec:
    """One row of the registry.

    name            -- the numpy_surface.json / coverage.py item name.
    kind            -- "unary" | "binary" | "binary_op" | "method" | "custom".
                        Determines which corpus is used and how args are
                        assembled; see run.py:build_cases.
                          unary      f(array) over corpus.unary_corpus().
                          binary     f(a, b) over corpus.binary_corpus() only
                                     (array-vs-array; kept for non-dunder
                                     binary functions like np.add itself).
                          binary_op  arr.<dunder>(other) over BOTH
                                     corpus.binary_corpus() (array-vs-array)
                                     AND corpus.unary_corpus() x
                                     corpus.scalar_operands() (array-vs-every
                                     scalar call form). Use this for
                                     __add__/__radd__/__mul__/__rmul__ and
                                     friends -- see module docstring.
                          method     arr.<name>(*call_form) for every
                                     declared CallForm x every corpus array.
                          custom     anything else; custom_cases() supplies
                                     (label, args, kwargs) directly.
    numpy_path      -- dotted path resolved against `numpy` (defaults to
                        `name` if omitted). Ignored for kind="custom" (the
                        custom_cases callable is expected to call numpy
                        itself if it needs a reference value).
    ionp_path       -- dotted path resolved against `anionpy` (defaults to
                        `name`). Set this to the *true* numpy-surface name
                        so the registry entry lines up with what
                        coverage.py will eventually look for.
    ionp_adapter    -- explicit callable, bypassing ionp_path resolution.
                        Use this only for harness self-tests / bridging a
                        differently-named helper (e.g. sum_f64) during
                        early bring-up -- see README.md's honesty note.
    numpy_adapter   -- explicit callable, bypassing numpy_path resolution,
                        mirroring ionp_adapter above. Added 2026-08-01 for
                        invariant-based comparisons (see
                        linalg_cases.py's module docstring): an item whose
                        raw numpy output cannot be meaningfully compared to
                        anionpy's raw output (eigenvector sign/scale
                        ambiguity, singular-vector sign/phase ambiguity)
                        instead supplies a probe that independently
                        recomputes the decomposition on ITS OWN side and
                        returns a small array of "should be ~0" algebraic
                        invariants -- resolve_numpy() returns this callable
                        directly instead of resolving numpy_path, exactly
                        like ionp_adapter already does for resolve_ionp().
    atol / rtol     -- MANDATORY for any item whose result is float/complex.
                        Left as None for exact (integer/bool/exception-only)
                        items; harness.py will refuse to grade an inexact
                        result with no declared tolerance rather than guess.
    scalar_like     -- this item's return value is not an ndarray (e.g.
                        `.shape` -> tuple, `.ndim` -> int, `.dtype` ->
                        dtype, `__repr__` -> str, `__len__` -> int).
                        harness.compare_values uses a dedicated comparison
                        path for these instead of requiring a `.dtype`
                        attribute on the result (see harness.py). Do NOT set
                        this for anything that returns an array.
    exception_equivalences -- optional {numpy_exc_type: {acceptable anionpy
                        exc types}}, for the rare item where numpy's own
                        exception type isn't the natural Rust-side one
                        (e.g. numpy's IndexError vs a hypothetical
                        ionp-specific error). Empty by default: same type
                        or nothing.
    call_forms      -- kind="method" only, MANDATORY and non-empty (see
                        module docstring): list[CallForm] to try, each
                        against every array in corpus.unary_corpus().
    custom_cases    -- kind="custom" only: fn() -> list of
                        (label, args_tuple, kwargs_dict) run against both
                        spec.resolve_numpy() and spec.resolve_ionp().
    """

    name: str
    kind: str
    numpy_path: Optional[str] = None
    ionp_path: Optional[str] = None
    ionp_adapter: Optional[Callable] = None
    numpy_adapter: Optional[Callable] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    scalar_like: bool = False
    exception_equivalences: Optional[dict] = None
    message_pair_ok: Optional[Callable] = None
    """Optional fn(np_exc, ionp_exc) -> bool, checked ONLY as a fallback when
    the ordinary exact-message-equality check (harness.run_case()'s
    `np_msg != ionp_msg` arm) has already failed AND the class check already
    passed (directly or via `exception_equivalences`). Added 2026-08-02 for
    exactly one shape: the comparison-ufunc `.reduce`/`.accumulate`/
    `.reduceat` no-loop case, where real numpy's (exception class, message
    text) pair is a COUPLED, history-dependent artifact of its own internal
    per-(ufunc, exact dtype) type-resolution cache -- see
    `_comparison_reduce_message_pair_ok` below for the measured pair states
    and why a bare class-equivalence-plus-loose-message-check would be wrong
    (it could accept a (UFuncTypeError, SHORT-message) combination real
    numpy never actually produces). Do NOT use this for anything else: it is
    a narrow, measured exception to "message is part of the API," not a
    general escape hatch -- see this field's one call site in
    `ufunc_registry.py` for the exact scope."""
    call_forms: Optional[list] = None
    custom_cases: Optional[Callable] = None
    extra_cases: Optional[Callable] = None
    """ANY kind: fn() -> list of (label, args_tuple, kwargs_dict), appended
    to whatever cases that kind already builds. Added 2026-08-03 (Monday)
    for the 0-d x explicit-axis boundary.

    `custom_cases` could not carry it, because it is honoured for
    kind="custom" ONLY -- and the reductions that most needed the boundary
    (`prod`, `all`, `any`, `min`, `max`, `cumsum`, `cumprod`, `argmin`,
    `argmax`, `ptp`, `count_nonzero`, `nanmin`, `nanmax`) are all
    kind="method". Their cases are built as unary_corpus() x call_forms,
    and `corpus.unary_corpus()` contains NO 0-dimensional array at any
    rank -- so no call form, however thorough, could ever have produced a
    0-d operand for them. That is the structural reason a whole family of
    items was declared "exact" while panicking on `f(np.array(3.0),
    axis=0)`: the gap was in the corpus, not in anyone's diligence about
    kwargs.

    Deliberately additive and kind-agnostic: it cannot shadow or remove an
    existing case, only add. Prefer widening the corpus when a gap affects
    everything; use this when a boundary applies to a specific, named set
    of items and would be meaningless (or wrong) for the rest."""
    convert_ionp_args: bool = True
    """kind="custom" only (ignored otherwise). When True (the default),
    resolve_ionp() converts every numpy.ndarray positional/keyword argument
    to a real anionpy.ndarray before calling the resolved anionpy function/adapter
    -- see registry.py's `_wrap_custom_conversion` and
    `make_ionp_array_converter` -- closing the 2026-08-01 hole where every
    kind="custom" item (all 18 `anionpy.linalg.*` bindings) was handed a raw
    numpy.ndarray, which they reject, so the differential suite never
    actually exercised anionpy.ndarray as an input. Set False ONLY for fixtures
    whose `custom_cases`/`ionp_adapter` are deliberately built from real
    numpy semantics rather than standing in for a real anionpy call -- e.g.
    selftest.py's `ulp_sqrt_*` proof-of-detection fixtures, whose
    `ionp_adapter` is a hand-written numpy-based shim (not a real anionpy
    binding) used to prove the ULP-tolerance mechanism itself, and would
    break under conversion for reasons unrelated to any anionpy.ndarray defect."""
    multi_output: bool = False
    """kind="ufunc" only. True for nout>1 ufuncs (divmod/modf/frexp): the
    result is a tuple of arrays that must be compared element-wise against
    numpy's tuple, not treated as a single array-like result. See
    ufunc_registry.py (sets this from ufunc_introspect.UfuncInfo.nout) and
    harness.py's run_case()/evaluate(), which branch on it."""
    ulp_tolerance: Optional[dict] = None
    """Opt-in, PER-DTYPE ULP tolerance for float/complex results:
    `{dtype_name: bound}` (e.g. `{"float32": 0.0, "complex64": 2.0}`) -- see
    reports/ionp-ulp-tolerance-decision-2026-08-01.md and harness.py's
    ulp_distance()/max_ulp_distance()/compare_values(). None (the default)
    means the item is graded bit-exact against numpy with no tolerance of
    any kind -- this is orthogonal to and takes priority over atol/rtol when
    set. A dtype produced by this item's corpus that has NO entry in this
    dict is also graded bit-exact (compare_values looks it up via
    `.get(dtype_name, 0.0)`) -- an ungraded dtype never silently inherits
    another dtype's slack; see MIN_ULP_SWEEP_N's docstring for the
    2026-08-01 postmortem this closes (the original `absolute`/`abs` bound
    was a single scalar applied uniformly, which gave float32/float64 2 ULP
    of slack their own evidence showed they never needed). MUST NOT be set
    without a non-empty `ulp_justification` citing recorded evidence that
    numpy's own loop is not correctly rounded for this item (enforced below,
    in __post_init__ -- undocumented tolerance must be structurally
    impossible, not merely discouraged), and its key set MUST exactly match
    `ulp_sweep`'s (also enforced below)."""
    ulp_justification: Optional[str] = None
    """Required, non-empty companion to `ulp_tolerance`: a short string
    recording the evidence (measurement + where it's written down, e.g.
    'KNOWN-DIFFERENCES.md: np.abs(complex64) vs hypotf, 35.1% disagree by
    1 ULP') that numpy's own loop is not correctly rounded for this item.
    Ignored (may be left None) when ulp_tolerance is None."""

    ulp_sweep: Optional[dict] = None
    """Required, non-empty companion to `ulp_tolerance`: `{dtype_name:
    (sample_size, measured_max_ulp)}` recording the ADEQUATE randomized
    sweep this tolerance was measured against -- see `ulp_sweep.py`. Added
    2026-08-01 to close the sample-size defect: `ulp_justification` alone
    let a tolerance be declared and "documented" from a measurement over
    the ~150-case differential corpus, which is edge-case coverage, not
    volume, and can fail to contain the discrepancy a bound needs to cover
    (this is exactly how the original `absolute`/`abs` bound of 1.0 ULP
    was wrong -- see the ⛔ correction in
    reports/ionp-ulp-tolerance-decision-2026-08-01.md). `__post_init__`
    below refuses to construct an ItemSpec with a `ulp_tolerance` unless
    every entry here records `sample_size >= MIN_ULP_SWEEP_N` AND the
    declared `ulp_tolerance` equals (not merely covers) the max measured
    across all entries -- so the bound can be neither under-evidenced nor
    padded above what was actually measured. Ignored (may be left None)
    when ulp_tolerance is None."""

    epsilon_justification: Optional[str] = None
    """DEAD FIELD as of the 2026-08-01 "hardened door / open window" closure
    below: `ItemSpec.__post_init__` now refuses to construct ANY ItemSpec
    with a non-zero `atol`/`rtol`, unconditionally -- see that check for the
    full reasoning. This field (and the legacy scalar `atol`/`rtol` above)
    can therefore never actually be exercised via ItemSpec construction any
    more; `atol=0.0, rtol=0.0` (or leaving both at their None default) is
    the only value ItemSpec will ever accept, and that is exact, not a
    tolerance, so it needs no justification regardless.

    Left in place, unused, rather than deleted: `harness.compare_values()`
    still accepts `atol`/`rtol`/`epsilon_justification` as direct keyword
    arguments -- that is a separate, still-live mechanism (its own
    independent `_require_epsilon_justification` check, "defense in depth"
    per that function's docstring) used by selftest.py and
    test_ulp_distance.py, which call `compare_values()` directly and do not
    go through ItemSpec for this. Deleting these ItemSpec fields would break
    nothing at runtime (they are dataclass fields, not the enforcement
    itself) but would silently strip the docstring context a reader needs
    when they hit the construction-time refusal below and go looking for
    why these fields exist at all. See `epsilon_tolerance` below for the
    field a real registry item should use instead."""

    epsilon_tolerance: Optional[dict] = None
    """Opt-in, PER-DTYPE, EVIDENCE-GATED epsilon tolerance for float/complex
    results -- the fix for the "hardened door, open window" defect (see
    reports/ionp-hardened-door-open-window-2026-08-01.md): the legacy
    `atol`/`rtol` + `epsilon_justification` mechanism above has no sample-
    size floor, no per-dtype keying, and no code that ever checks a declared
    bound against a real measurement -- a non-empty string is the whole
    gate. This field closes all three holes, mirroring `ulp_tolerance`'s
    shape and `__post_init__` validation exactly (same MIN_ULP_SWEEP_N
    floor, same exact-equality-to-measured-evidence check, same per-dtype
    keying, same "no entry -> bit-exact" default), but grades by a
    magnitude-based distance instead of ULP distance, which is the right
    instrument for decomposition/factorization outputs (LAPACK backward
    error, sign/phase ambiguity resolved via invariants) where ULP distance
    is either not meaningful (compared factors are not numpy's own float
    loop output) or explodes for reasons that are metric artifacts, not
    defects (see this field's sweep evidence in linalg_cases.py for the
    worked example: complex128 `det`'s ULP distance reaches 1e18+ while its
    RELATIVE error stays at 1e-13, because the disagreement lives in a
    numerically-zero component).

    Shape: `{dtype_name: (metric, bound)}` where `metric` is the literal
    string `"abs"` or `"rel"`:
      - `"abs"`: graded by `harness.max_abs_distance` -- elementwise
        `|numpy_out - ionp_out|` (via `np.abs`, so this is already a
        complex-modulus distance for complex dtypes, not per-component),
        maximized over the output. Right instrument for outputs whose
        MAGNITUDE is bounded by the problem's own scale and does not vary
        multiplicatively with matrix size/condition number (cholesky
        factors, inv, solve, pinv, svdvals, eigvalsh, lstsq, and the
        eig/eigh/qr/svd invariant residuals below).
      - `"rel"`: graded by `harness.max_rel_distance` -- elementwise
        `|numpy_out - ionp_out| / max(|numpy_out|, 1.0)`, maximized over the
        output. Right instrument for outputs whose magnitude scales
        multiplicatively with problem size/condition number (`det`,
        `matrix_power`, `cond`; measured directly: an ABSOLUTE bound tight
        enough to be meaningful on a near-zero determinant fails on a
        large-magnitude one purely from scale, not from a defect -- see
        linalg_cases.py's module docstring). The `max(..., 1.0)` floor keeps
        the metric behaving like an absolute bound for genuinely
        small-magnitude outputs (never divides by a near-zero number) while
        being genuinely relative for large ones.

    A dtype produced by an item's corpus with NO entry in this dict is
    graded bit-exact -- same enforced-safe-default rule as `ulp_tolerance`
    (see that field's docstring); it never inherits another dtype's slack.
    Takes priority over the legacy `atol`/`rtol` path (checked first in
    `harness.compare_values`) but is itself subordinate to `ulp_tolerance`
    when both are somehow declared on the same item (neither actually is,
    in this registry, as of this fix). MUST NOT be set without a non-empty
    `epsilon_tolerance_justification` (enforced below in __post_init__) and
    a matching `epsilon_sweep` whose key set is EXACTLY this dict's key set
    and whose declared bound EQUALS (not merely covers) the measured value
    for that dtype -- neither under-evidenced nor padded, exactly the
    ulp_tolerance rule, for the exact same reason (see MIN_ULP_SWEEP_N's
    docstring: a bound that is not checked against real evidence is a
    property of whoever typed it, not of the implementation)."""

    epsilon_tolerance_justification: Optional[str] = None
    """Required, non-empty companion to `epsilon_tolerance`: a short string
    recording WHY this item's outputs need `abs` or `rel` grading rather
    than bit-exactness (e.g. floating-point non-associativity between two
    independent LAPACK call sequences, or genuine multiplicative scale in a
    determinant/product-of-eigenvalues quantity) -- mirrors
    `ulp_justification`'s role for the ULP mechanism. Ignored (may be left
    None) when epsilon_tolerance is None."""

    epsilon_sweep: Optional[dict] = None
    """Required, non-empty companion to `epsilon_tolerance`: `{dtype_name:
    (sample_size, measured_value)}` recording the ADEQUATE randomized sweep
    this tolerance was measured against, using the SAME metric (`"abs"` or
    `"rel"`) declared for that dtype in `epsilon_tolerance` -- mirrors
    `ulp_sweep`'s role and shape exactly, reusing the same
    `MIN_ULP_SWEEP_N` sample-size floor (see `ulp_sweep.py`, which this
    module's `sweep_matrix_item`/`sweep_matrix_pair`/`sweep_probe_pair`
    functions extend with the matrix-generation machinery this measurement
    needs, in addition to the simple-array sweeps the ULP mechanism already
    had). `__post_init__` below refuses to construct an ItemSpec with an
    `epsilon_tolerance` unless every entry here records `sample_size >=
    MIN_ULP_SWEEP_N` AND the declared `epsilon_tolerance` bound equals the
    measured maximum for that dtype -- so the bound can be neither
    under-evidenced nor padded above what was actually measured. Ignored
    (may be left None) when epsilon_tolerance is None."""

    signed_zero_tie_exempt: bool = False
    """Opt-in, item-wide (NOT per-dtype -- unlike ulp_tolerance/
    epsilon_tolerance, there is nothing to grade per dtype here: this is a
    binary exemption, not a magnitude bound) declaration that this item's
    comparison should treat a `-0.0` vs `+0.0` byte-level mismatch, AT A
    GIVEN OUTPUT POSITION, as equal -- provided BOTH sides' value at that
    position is genuinely `0.0` (see harness.py's `_bit_exact_equal`).
    Added 2026-08-01 (coordinator's harder probe) for exactly three items:
    `sort`, `ndarray.sort`, `sort_complex`. Root cause: `-0.0 == +0.0`
    under IEEE comparison, so their relative ARRANGEMENT within a tied run
    of a sort is decided by numpy's SIMD introsort kernel's internal
    partitioning, not by any comparator -- verified directly by the
    coordinator to differ by array length (n=32 vs n=200) on the identical
    tied input, i.e. it is not even a fixed per-numpy-version contract,
    let alone one Rust code could reproduce deliberately. This is the same
    fundamental class of problem as `argsort`'s already-scoped-out
    duplicate-value tie order (see this module's own top-of-file
    docstring) -- the correct response is a declared, ledger-visible
    exemption, never a silent pass and never an attempted reimplementation
    of SIMD-dependent internals.

    Deliberately item-wide rather than per-dtype: unlike ULP/epsilon slack
    (which varies by dtype's precision), "does -0.0/+0.0 arrangement need
    to be forgiven" is the same yes/no question for every float dtype this
    item touches (f16/f32/f64), and for complex it applies independently
    to the real and imaginary components (see `_bit_exact_equal`'s
    recursive per-component handling) -- there is no dtype-specific bound
    to record, so a `{dtype: bound}` shape would be pure ceremony.

    CRITICALLY NARROW: this exemption applies ONLY to a zero-vs-zero sign
    mismatch. It does NOT relax NaN payload canonicalization (sort.rs's
    `canonicalize_nans_*`, gated on `shape[axis] > 1`) even slightly -- a
    NaN bit pattern is never `0.0` under `==`, so `_bit_exact_equal`'s
    zero-only carve-out can never match one; NaN payload divergence still
    fails exactly as it did before this field existed. The two mechanisms
    are ORTHOGONAL: canonicalization decides WHAT bit pattern a moved NaN
    is rewritten to (never touches non-NaN signed zeros), this field
    decides whether a non-NaN, exactly-zero VALUE's sign bit is graded at
    all (never touches NaN bit patterns) -- they cannot interact because
    they operate on disjoint value classes (NaN vs. finite zero) by
    construction, not by coincidence.

    Mutually exclusive with `ulp_tolerance`/`epsilon_tolerance` by
    construction below (a magnitude-bound mechanism and a binary sign-of-
    zero exemption answering different questions on the same item would
    be a modeling error, not a real combination this registry needs)."""

    signed_zero_tie_exempt_justification: Optional[str] = None
    """Required, non-empty companion to `signed_zero_tie_exempt=True`,
    mirroring `ulp_justification`'s role: a short string recording WHY
    this item's -0.0/+0.0 arrangement is not a reproducible comparator
    contract (see `signed_zero_tie_exempt`'s own docstring for the full
    evidence). Ignored (may be left None) when the flag is False."""

    check_strides: bool = False
    """Opt-in, item-wide: when True, `harness.compare_values` additionally
    requires the anionpy result's `.strides` tuple to equal numpy's, on top of
    the existing dtype/shape/value checks -- a correct-VALUES,
    wrong-LAYOUT result (e.g. an `order='F'` call that returns C strides)
    is a real anionpy defect and must fail, not pass silently. Added
    2026-08-01 for the ufunc `order=` kwarg task: prior to that task,
    NOTHING in this harness ever compared output strides for any item --
    values/dtype/shape only (true of `order_cases.py`'s existing
    ravel/flatten/copy/astype/_like/reshape corpus too, not just the ufunc
    one -- a pre-existing, unrelated gap, reported not fixed).

    NOT set on the main `kind="ufunc"` items in `ufunc_registry.py`
    (`_build_ufunc_specs` leaves this at its False default for all ~134 of
    them). It was tried there first and reverted: turning it on for those
    items also applies it to their ENTIRE pre-existing corpus (including
    `corpus.py`'s `_views()`-derived general-transpose/sliced operands),
    which immediately surfaces a confirmed, PRE-EXISTING, OUT-OF-SCOPE bug
    in `ndarray_from_numpy` (ionp-py/src/lib.rs): it only special-cases a
    source that is "genuinely F-contiguous, not also C" on ingestion; any
    OTHER non-C/non-F source layout (a general 3-axis transpose, a partial
    axis swap, a non-contiguous slice) is silently ingested as if it were
    C-contiguous -- values survive, true layout does not. Blanket-enabling
    `check_strides` there would fail dozens of unrelated pre-existing
    corpus cases (e.g. `trunc`'s `view/3d_transpose_axes`) on an ingestion
    defect that has nothing to do with `Ufunc.__call__`'s order= logic,
    which is exactly the kind of scope creep this task's brief forbids.

    Instead used narrowly on the dedicated, purpose-built `kind="custom"`
    ItemSpecs in `ufunc_order_cases.py` (registered under fresh
    `"order/ufunc/<name>"` keys, not the real ufunc names), whose operand
    layouts are deliberately restricted to 'c' and 'f' -- the two layouts
    `ndarray_from_numpy` is already proven to ingest correctly -- so
    `check_strides=True` there tests order='C'/'F'/'A'/'K' honestly
    without also re-litigating the unrelated ingestion bug. See that
    file's module docstring for the full reasoning and a live
    reproduction of the ingestion defect.

    Left False everywhere else in the registry (the default): turning this
    on for an item whose anionpy binding was never verified to preserve
    layout would silently fail cases that were previously passing on
    values alone. Ignored for `scalar_like`/`multi_output` items (nothing
    with a single `.strides` tuple to compare) and for exception-only
    cases (no output exists).

    KNOWN HARNESS BUG, documented not fixed (2026-08-02, see
    `docs/stride-gap-classification.md`, commit 65dd475): if `check_strides`
    is ever flipped on broadly, `harness._strides_match` runs a 3rd check
    (`tobytes('A')` byte-for-byte equality) AFTER `compare_values` has
    already accepted the item under its declared `ulp_tolerance` /
    `epsilon_tolerance`. That numeric tolerance was reviewed and justified
    specifically to accept small floating-point noise on VALUES; the
    `tobytes('A')` byte check has no concept of tolerance at all and
    re-litigates that exact same noise at the raw-byte level, where it
    almost never survives (a value within 3.4e-14 relative is essentially
    never byte-identical). This is why, in the 2026-08-02 audit, 16 items
    with real epsilon-tolerant math (`linalg.cholesky`, `linalg.inv`, the
    `fft.*`/`hfft`/`irfft` family, etc.) appeared to fail a stride check
    even though their strides and bytes were never the actual problem --
    the check ran somewhere it structurally cannot pass, not because of a
    genuine layout defect. Do not "fix" this by weakening or removing the
    tolerance declarations; the fix (not made here) belongs in
    `_strides_match` itself, e.g. skipping or tolerance-aware-relaxing
    step 3 whenever the item already carries a numeric tolerance."""

    def __post_init__(self):
        if self.kind == "method":
            if not self.call_forms:
                raise ValueError(
                    f"{self.name}: kind='method' requires a non-empty, EXPLICIT "
                    f"call_forms list -- this is the harness's guard against "
                    f"exactly the bug it exists to catch (crediting a PASS for "
                    f"only one supported argument shape). See registry.py's "
                    f"module docstring."
                )
        if self.ulp_tolerance is not None:
            if not self.ulp_justification or not str(self.ulp_justification).strip():
                raise ValueError(
                    f"{self.name}: ulp_tolerance={self.ulp_tolerance!r} declared with "
                    f"no recorded ulp_justification -- refusing to construct this "
                    f"ItemSpec. Undocumented ULP tolerance is structurally forbidden: "
                    f"every declared tolerance must cite the recorded evidence that "
                    f"numpy's own loop is not correctly rounded for this item (see "
                    f"reports/ionp-ulp-tolerance-decision-2026-08-01.md and "
                    f"KNOWN-DIFFERENCES.md)."
                )
            if not isinstance(self.ulp_tolerance, dict) or not self.ulp_tolerance:
                raise ValueError(
                    f"{self.name}: ulp_tolerance must be a non-empty {{dtype_name: bound}} "
                    f"dict, graded per dtype -- got {self.ulp_tolerance!r}. A single "
                    f"item-wide scalar is exactly the defect this field's docstring "
                    f"(and MIN_ULP_SWEEP_N's) describes: it gives every dtype the "
                    f"worst dtype's slack, including dtypes whose own evidence shows "
                    f"they need none."
                )
            for dtype_name, tol in self.ulp_tolerance.items():
                if tol < 0:
                    raise ValueError(
                        f"{self.name}: ulp_tolerance[{dtype_name!r}] must be >= 0, "
                        f"got {tol!r}"
                    )
            # Sample-size (adequacy) gate, 2026-08-01: a ulp_justification string
            # alone is not evidence of adequate SAMPLE SIZE -- it can (and once
            # did, for absolute/abs's original 1.0 ULP bound) cite a measurement
            # over the small ~150-case differential corpus, which is a property
            # of the sample, not the implementation. See ulp_sweep.py and this
            # field's docstring above.
            if not self.ulp_sweep:
                raise ValueError(
                    f"{self.name}: ulp_tolerance={self.ulp_tolerance!r} declared with "
                    f"no recorded ulp_sweep evidence -- refusing to construct this "
                    f"ItemSpec. A ulp_tolerance must be measured over an adequate "
                    f"randomized sweep (>= {MIN_ULP_SWEEP_N} seeded values per "
                    f"applicable dtype, see ulp_sweep.py), not merely over the item's "
                    f"ordinary differential corpus -- a corpus too small to contain "
                    f"the discrepancy will under-evidence the bound (see the ⛔ "
                    f"correction in reports/ionp-ulp-tolerance-decision-2026-08-01.md, "
                    f"where this happened to absolute/abs's own original bound)."
                )
            for dtype_name, entry in self.ulp_sweep.items():
                try:
                    n, measured = entry
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{self.name}: ulp_sweep[{dtype_name!r}]={entry!r} is not a "
                        f"(sample_size, measured_max_ulp) pair"
                    ) from exc
                if n < MIN_ULP_SWEEP_N:
                    raise ValueError(
                        f"{self.name}: ulp_sweep[{dtype_name!r}] sample_size={n!r} is "
                        f"below the required minimum of {MIN_ULP_SWEEP_N} -- "
                        f"under-evidenced sweep, refusing to construct this ItemSpec."
                    )
                if not math.isfinite(measured):
                    raise ValueError(
                        f"{self.name}: ulp_sweep[{dtype_name!r}] measured_max_ulp="
                        f"{measured!r} is not finite -- a finite ulp_tolerance cannot "
                        f"be justified against a non-finite measurement."
                    )
            # Per-dtype grading (2026-08-01 fix): the declared tolerance dict's key
            # set must be EXACTLY the sweep's key set -- not a subset (that would
            # let a dtype be graded with tolerance despite no evidence for it) and
            # not a superset (that would let a swept-but-undeclared dtype silently
            # fall back to bit-exact via compare_values's .get(..., 0.0) without
            # that being a deliberate choice recorded here). See
            # MIN_ULP_SWEEP_N's docstring.
            if set(self.ulp_tolerance) != set(self.ulp_sweep):
                raise ValueError(
                    f"{self.name}: ulp_tolerance dtypes {sorted(self.ulp_tolerance)} do "
                    f"not match ulp_sweep dtypes {sorted(self.ulp_sweep)} -- every dtype "
                    f"with a declared tolerance must carry its own sweep evidence, and "
                    f"every swept dtype must have its own explicit declared tolerance "
                    f"(a dtype must never silently inherit another dtype's bound)."
                )
            for dtype_name, tol in self.ulp_tolerance.items():
                _n, measured = self.ulp_sweep[dtype_name]
                if tol != measured:
                    raise ValueError(
                        f"{self.name}: ulp_tolerance[{dtype_name!r}]={tol!r} does not "
                        f"equal the measured sweep max {measured!r} for that dtype -- "
                        f"declare exactly the measured PER-DTYPE bound: lower would "
                        f"under-evidence it, higher would pad beyond the measurement "
                        f"(both forbidden, see ulp_sweep.py's module docstring). "
                        f"Grading is per dtype now, not item-wide, so each dtype's "
                        f"declared bound is checked against its OWN measured max, not "
                        f"the max across all dtypes."
                    )
        atol_nonzero = self.atol is not None and self.atol != 0.0
        rtol_nonzero = self.rtol is not None and self.rtol != 0.0
        if atol_nonzero or rtol_nonzero:
            # 2026-08-01, "hardened door / open window" closure (see
            # reports/ionp-hardened-door-open-window-2026-08-01.md, Finding 2):
            # a non-empty epsilon_justification string used to be the ENTIRE
            # gate on this path -- no sample-size floor, no per-dtype keying,
            # no check that the declared bound matched anything ever measured.
            # That is exactly how 11 items acquired unearned tolerance (9 of
            # them justified by a single shared constant). The fix is not a
            # stronger string check; it's refusing the legacy path outright.
            # `epsilon_tolerance` (below) is the replacement: same shape,
            # but per-dtype, requires >=MIN_ULP_SWEEP_N samples per dtype, and
            # requires the declared bound to EQUAL (not merely cite) the
            # measured maximum. There is no justification string that makes
            # a non-zero legacy atol/rtol acceptable any more -- construct
            # with `epsilon_tolerance` + `epsilon_tolerance_justification` +
            # `epsilon_sweep` instead. `atol=0.0, rtol=0.0` (or omitting both)
            # remains valid and needs nothing further: it is not a tolerance,
            # it is a restatement of the default bit-exact bar.
            raise ValueError(
                f"{self.name}: atol={self.atol!r} rtol={self.rtol!r} declared "
                f"non-zero -- refusing to construct this ItemSpec. The legacy "
                f"atol/rtol (+ epsilon_justification) path is structurally "
                f"forbidden now, regardless of justification text: it had no "
                f"sample-size floor, no per-dtype keying, and no check that the "
                f"declared bound matched any real measurement, which is exactly "
                f"how unearned tolerance got in (see "
                f"reports/ionp-hardened-door-open-window-2026-08-01.md). Use "
                f"`epsilon_tolerance` + `epsilon_tolerance_justification` + "
                f"`epsilon_sweep` instead -- the evidence-gated per-dtype "
                f"replacement (>= {MIN_ULP_SWEEP_N} samples per dtype, declared "
                f"bound must equal the measured maximum). `atol=0.0, rtol=0.0` "
                f"(or omitting both) remains valid: that is not a tolerance, it "
                f"is exact."
            )

        if self.epsilon_tolerance is not None:
            if not self.epsilon_tolerance_justification or \
                    not str(self.epsilon_tolerance_justification).strip():
                raise ValueError(
                    f"{self.name}: epsilon_tolerance={self.epsilon_tolerance!r} declared "
                    f"with no recorded epsilon_tolerance_justification -- refusing to "
                    f"construct this ItemSpec. Undocumented epsilon tolerance is "
                    f"structurally forbidden, same as ulp_tolerance: every declared "
                    f"tolerance must cite the recorded evidence for why bit-exactness "
                    f"is not the right bar for this item (see "
                    f"reports/ionp-hardened-door-open-window-2026-08-01.md)."
                )
            if not isinstance(self.epsilon_tolerance, dict) or not self.epsilon_tolerance:
                raise ValueError(
                    f"{self.name}: epsilon_tolerance must be a non-empty "
                    f"{{dtype_name: (metric, bound)}} dict, graded per dtype -- got "
                    f"{self.epsilon_tolerance!r}. A single item-wide scalar is exactly "
                    f"the defect this field exists to close: it gives every dtype the "
                    f"worst dtype's slack, including dtypes whose own evidence shows "
                    f"they need none (see MIN_ULP_SWEEP_N's docstring for the ULP-side "
                    f"precedent of this exact bug)."
                )
            for dtype_name, entry in self.epsilon_tolerance.items():
                try:
                    metric, bound = entry
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{self.name}: epsilon_tolerance[{dtype_name!r}]={entry!r} is not "
                        f"a (metric, bound) pair"
                    ) from exc
                if metric not in ("abs", "rel"):
                    raise ValueError(
                        f"{self.name}: epsilon_tolerance[{dtype_name!r}] metric "
                        f"{metric!r} must be 'abs' or 'rel'"
                    )
                if bound < 0:
                    raise ValueError(
                        f"{self.name}: epsilon_tolerance[{dtype_name!r}] bound must be "
                        f">= 0, got {bound!r}"
                    )
            # Sample-size (adequacy) gate, mirroring ulp_tolerance's: a
            # epsilon_tolerance_justification string alone is not evidence of
            # adequate SAMPLE SIZE -- see ulp_sweep.py's module docstring for
            # why (the original absolute/abs ULP bound was wrong precisely
            # because it was measured on too small a sample). The exact same
            # failure mode was live on this epsilon path with NO sample-size
            # requirement at all (the recorded justification for the 14
            # linalg items this closes cited 50 seeded matrices; see
            # reports/ionp-hardened-door-open-window-2026-08-01.md).
            if not self.epsilon_sweep:
                raise ValueError(
                    f"{self.name}: epsilon_tolerance={self.epsilon_tolerance!r} declared "
                    f"with no recorded epsilon_sweep evidence -- refusing to construct "
                    f"this ItemSpec. An epsilon_tolerance must be measured over an "
                    f"adequate randomized sweep (>= {MIN_ULP_SWEEP_N} seeded matrices "
                    f"per applicable dtype, see ulp_sweep.py), not merely asserted in "
                    f"prose."
                )
            for dtype_name, entry in self.epsilon_sweep.items():
                try:
                    n, measured = entry
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{self.name}: epsilon_sweep[{dtype_name!r}]={entry!r} is not a "
                        f"(sample_size, measured_value) pair"
                    ) from exc
                if n < MIN_ULP_SWEEP_N:
                    raise ValueError(
                        f"{self.name}: epsilon_sweep[{dtype_name!r}] sample_size={n!r} "
                        f"is below the required minimum of {MIN_ULP_SWEEP_N} -- "
                        f"under-evidenced sweep, refusing to construct this ItemSpec."
                    )
                if not math.isfinite(measured) or measured < 0:
                    raise ValueError(
                        f"{self.name}: epsilon_sweep[{dtype_name!r}] measured_value="
                        f"{measured!r} must be a finite value >= 0."
                    )
            # Per-dtype grading, mirroring ulp_tolerance's key-set-equality
            # rule exactly (see that block's comment above for the full
            # rationale: neither a tolerance without matching sweep evidence
            # nor a swept-but-undeclared dtype is allowed).
            if set(self.epsilon_tolerance) != set(self.epsilon_sweep):
                raise ValueError(
                    f"{self.name}: epsilon_tolerance dtypes "
                    f"{sorted(self.epsilon_tolerance)} do not match epsilon_sweep "
                    f"dtypes {sorted(self.epsilon_sweep)} -- every dtype with a "
                    f"declared tolerance must carry its own sweep evidence, and every "
                    f"swept dtype must have its own explicit declared tolerance."
                )
            for dtype_name, (metric, bound) in self.epsilon_tolerance.items():
                _n, measured = self.epsilon_sweep[dtype_name]
                if bound != measured:
                    raise ValueError(
                        f"{self.name}: epsilon_tolerance[{dtype_name!r}]=({metric!r}, "
                        f"{bound!r}) does not equal the measured sweep max "
                        f"{measured!r} for that dtype -- declare exactly the measured "
                        f"PER-DTYPE bound: lower would under-evidence it, higher would "
                        f"pad beyond the measurement (both forbidden, same rule as "
                        f"ulp_tolerance -- see that field's docstring)."
                    )

        if self.signed_zero_tie_exempt:
            if not self.signed_zero_tie_exempt_justification or \
                    not str(self.signed_zero_tie_exempt_justification).strip():
                raise ValueError(
                    f"{self.name}: signed_zero_tie_exempt=True declared with no "
                    f"recorded signed_zero_tie_exempt_justification -- refusing to "
                    f"construct this ItemSpec. Undocumented comparison exemptions are "
                    f"structurally forbidden, same as ulp_tolerance/epsilon_tolerance: "
                    f"see this field's docstring above."
                )
            if self.ulp_tolerance is not None or self.epsilon_tolerance is not None:
                raise ValueError(
                    f"{self.name}: signed_zero_tie_exempt=True cannot be combined with "
                    f"ulp_tolerance/epsilon_tolerance on the same item -- these answer "
                    f"different questions (magnitude bound vs. binary sign-of-zero "
                    f"exemption) and combining them on one item would be a modeling "
                    f"error, not a real requirement (see the field's docstring)."
                )

        # 2026-08-01: default `exception_equivalences` covering numpy's
        # PRIVATE `_UFuncNoLoopError`/`_UFuncOutputCastingError` classes
        # (see `_DEFAULT_UFUNC_EXC_EQUIV` above for the full rationale).
        # Only fills the gap when the item hasn't declared its own mapping
        # -- never overrides an explicit one. Deliberately NOT gated on
        # `kind == "ufunc"`: the same two raise sites in `lib.rs`
        # (`no_ufunc_loop_err_dtypes` / `ufunc_output_casting_err`) are
        # reached both by the ~134 auto-generated `kind="ufunc"` items
        # AND by hand-declared `kind="unary"`/`kind="binary_op"` ndarray
        # dunder items in this same file (e.g. `ndarray.__pos__` calling
        # into `positive`'s bool-rejection; `ndarray.__imod__`/
        # `__ilshift__`/`__irshift__`'s in-place output-casting checks).
        # The mapping is keyed on numpy's own exact (private) exception
        # TYPE, not on this item's `kind`, so applying it unconditionally
        # cannot mask an unrelated bug in some other item -- it only ever
        # engages when numpy itself raises one of these two specific
        # classes, which happens only for genuine no-loop/output-casting
        # situations.
        if self.exception_equivalences is None:
            self.exception_equivalences = dict(_DEFAULT_UFUNC_EXC_EQUIV)

    def resolve_numpy(self):
        if self.numpy_adapter is not None:
            return self.numpy_adapter
        path = self.numpy_path if self.numpy_path is not None else self.name
        if self.kind == "ufunc":
            try:
                obj = np
                for part in path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                return None
            return build_ufunc_dispatcher(obj)
        exploded_prefix = _exploded_prefix_of(path)
        if exploded_prefix is not None:
            attr = path[len(exploded_prefix) + 1:]

            # Generic on purpose, and IDENTICAL in shape to the pre-existing
            # "ndarray."-only branch this replaces: numpy is the reference
            # implementation, so whatever real numpy instance the corpus
            # hands us as the receiver (a real numpy.dtype, numpy.finfo,
            # numpy.ma.MaskedArray, ...) is resolved against directly, no
            # conversion needed -- this side never had a tautology hazard
            # (see resolve_ionp() below for where that hazard lives and is
            # guarded against).
            def fn(receiver, *rest, **kwargs):
                # "ma.MaskedArray" is the one exploded prefix whose corpus
                # hands the case a `(data, mask)` tuple rather than a real
                # numpy instance -- see `_reconstruct_np_maskedarray`'s
                # docstring above for the `_freshen`-strips-the-subclass
                # reason why. Reassemble it here, on the numpy reference
                # side, before doing the actual getattr/call.
                if exploded_prefix == "ma.MaskedArray":
                    real_receiver = _reconstruct_np_maskedarray(receiver)
                else:
                    real_receiver = receiver
                target = getattr(real_receiver, attr)
                return target(*rest, **kwargs) if callable(target) else target

            return fn
        try:
            return _resolve_dotted(np, path)
        except AttributeError:
            return None

    def resolve_ionp(self):
        import anionpy  # local import: anionpy may not be importable at all (absent), which is data

        if self.ionp_adapter is not None:
            return self._wrap_custom_conversion(anionpy, self.ionp_adapter)

        path = self.ionp_path if self.ionp_path is not None else self.name

        if self.kind == "ufunc":
            # Resolved against the real `anionpy` package once one exists;
            # today anionpy exposes no ufuncs at all (see anionpy/__init__.py),
            # so this returns None for all 134 items and evaluate() short-
            # circuits to the correct "absent" verdict -- see harness.py.
            obj = anionpy
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                return None
            nd = getattr(anionpy, "ndarray", None)
            to_ionp = make_ionp_array_converter(anionpy, nd) if nd is not None else None
            return build_ufunc_dispatcher(obj, to_ionp=to_ionp)

        exploded_prefix = _exploded_prefix_of(path)

        if exploded_prefix is not None and exploded_prefix != "ndarray":
            # Generalized dispatch for every curated exploded class OTHER
            # than "ndarray" (dtype, finfo, iinfo, ma.MaskedArray, ...;
            # see _EXPLODED_RECEIVER_CONVERTERS above). The "ndarray."
            # branch keeps its own separate, pre-existing, untouched code
            # path immediately below -- this new branch does not alter its
            # behavior at all, deliberately (see
            # _EXPLODED_RECEIVER_CONVERTERS' "ndarray" comment for why).
            attr = path[len(exploded_prefix) + 1:]
            converter = _EXPLODED_RECEIVER_CONVERTERS.get(exploded_prefix)
            if converter is None:
                # Either one of the 12 curated classes anionpy does not
                # implement at all, or "random.Generator" (handled by its
                # own kind="custom" adapter items in
                # exploded_class_cases.py instead -- see that dict's
                # comment). Either way, correctly "absent": returning None
                # here is what makes evaluate() in harness.py grade this
                # item as "not implemented on anionpy (absent)", never a
                # fabricated pass.
                return None
            expected_numpy_type = _EXPLODED_NUMPY_TYPES[exploded_prefix]
            _sentinel = object()

            def fn(receiver, *rest, **kwargs):
                # Tautology guard, mirroring the "ndarray."-prefix branch's
                # own `isinstance(arr, (np.ndarray, nd))` check immediately
                # below: if `receiver` is not a real instance of the numpy
                # class this item was exploded from, converting it and
                # resolving `attr` on the result cannot exercise anionpy
                # meaningfully (either it isn't the right kind of object at
                # all, or -- the specific hazard this guards against -- it
                # is already an anionpy instance from a previous case's mutation
                # leaking through, in which case getattr would silently
                # resolve on anionpy's own object redundantly rather than
                # signal that the corpus fed this item something unexpected).
                if exploded_prefix == "ma.MaskedArray":
                    # "ma.MaskedArray" receivers travel through the corpus
                    # (and through harness.py's `_freshen()`) as a plain
                    # `(data, mask)` tuple, not a live MaskedArray -- see
                    # `_reconstruct_np_maskedarray`'s docstring above for
                    # why. The tautology guard here checks the tuple shape
                    # itself, since `isinstance(receiver, np.ma.MaskedArray)`
                    # would (incorrectly) reject every genuinely valid case.
                    valid_tuple = (
                        isinstance(receiver, tuple)
                        and len(receiver) == 2
                        and all(isinstance(part, np.ndarray) for part in receiver)
                    )
                    if not valid_tuple:
                        raise InvalidCase(
                            f"{exploded_prefix}.{attr}: receiver is "
                            f"{type(receiver).__name__} ({receiver!r}), not "
                            f"a (data, mask) ndarray pair -- this case "
                            f"cannot exercise anionpy at all"
                        )
                elif not isinstance(receiver, expected_numpy_type):
                    raise InvalidCase(
                        f"{exploded_prefix}.{attr}: receiver is "
                        f"{type(receiver).__name__} ({receiver!r}), not a "
                        f"real {expected_numpy_type.__name__} -- this case "
                        f"cannot exercise anionpy at all"
                    )
                ionp_receiver = converter(anionpy, receiver)
                target = getattr(ionp_receiver, attr, _sentinel)
                if target is _sentinel:
                    raise AttributeError(attr)
                if not callable(target):
                    return target
                nd = getattr(anionpy, "ndarray", None)
                to_ionp = make_ionp_array_converter(anionpy, nd) if nd is not None else (lambda x: x)
                ionp_rest = tuple(to_ionp(r) for r in rest)
                ionp_kwargs = {k: to_ionp(v) for k, v in kwargs.items()}
                return target(*ionp_rest, **ionp_kwargs)

            return fn

        if path.startswith("ndarray."):
            attr = path.split(".", 1)[1]
            nd = getattr(anionpy, "ndarray", None)
            if nd is None:
                return None

            # The corpus (corpus.py) hands out real numpy.ndarray instances
            # -- that is deliberate, numpy IS the reference implementation.
            # But `getattr(numpy_array, "reshape")` is a perfectly real,
            # perfectly WRONG method to call here: it's numpy's own
            # reshape, not anionpy's. Without this conversion, every
            # "ndarray."-prefixed item silently tests numpy against itself
            # and can never fail -- which is precisely how
            # `ndarray.reshape`/`ndarray.__add__` were credited PASS before
            # this fix, independent of and in addition to the call-form gap
            # this module otherwise fixes. See make_ionp_array_converter's
            # docstring for exactly what is and isn't converted.
            _to_ionp = make_ionp_array_converter(anionpy, nd)

            _sentinel = object()

            def fn(arr, *rest, **kwargs):
                # 2026-08-01, scalar-receiver-tautology fix: `arr` here is
                # the RECEIVER this dunder/method is resolved on. For
                # kind="method" items every `arr` is guaranteed a real
                # numpy.ndarray (it comes straight from
                # corpus.unary_corpus()), so this can never fire there --
                # it exists for kind="binary_op"'s array-vs-array corpus
                # (corpus.binary_corpus()), which has exactly one Pair
                # whose first operand is a numpy SCALAR, not an array
                # (`broadcast/scalar_array`, receiver `np.float64(2.5)`).
                # `_to_ionp` (make_ionp_array_converter) passes a non-array
                # through UNCONVERTED, by design -- correct for a scalar
                # used as the second operand, wrong here: `ionp_arr` would
                # be the exact same numpy scalar the numpy reference side
                # calls, so `getattr(ionp_arr, attr)` resolves the dunder
                # on numpy's OWN object -- a tautology that can never fail
                # (see make_ionp_array_converter's docstring, hazard (b)),
                # or for the handful of dunders numpy scalars lack
                # (__ifloordiv__ etc.), a loud but equally meaningless
                # AttributeError. Neither exercises anionpy. Raising
                # InvalidCase here (instead of silently proceeding) lets
                # harness.run_case()/evaluate() record this case as
                # explicitly skipped/invalid rather than crediting or
                # blaming anionpy for a call that never reached it --
                # legitimate because kind="binary_op"'s OTHER case source
                # (corpus.unary_corpus() x corpus.scalar_operands(), see
                # this module's docstring) already exercises a real
                # anionpy.ndarray receiver against every scalar call form,
                # including array-vs-scalar broadcasting, so nothing is
                # silently lost.
                if not isinstance(arr, (np.ndarray, nd)):
                    raise InvalidCase(
                        f"ndarray.{attr}: receiver is {type(arr).__name__} "
                        f"({arr!r}), not a numpy.ndarray/anionpy.ndarray -- this "
                        f"case cannot exercise anionpy at all (see "
                        f"make_ionp_array_converter's docstring, hazard (b), "
                        f"and InvalidCase's docstring in harness.py)"
                    )
                ionp_arr = _to_ionp(arr)
                ionp_rest = tuple(_to_ionp(r) for r in rest)
                # `getattr(..., None)` used to be the "missing" sentinel here,
                # but that conflates "attribute absent" with "attribute
                # legitimately equals None" -- e.g. `ndarray.__hash__` is
                # `None` on BOTH numpy and anionpy by design (numpy sets
                # `__hash__ = None` on purpose to make ndarray unhashable; an
                # anionpy class making the same explicit choice must compare
                # equal, not be miscounted as "absent"). A private object()
                # sentinel can never collide with a real attribute value.
                target = getattr(ionp_arr, attr, _sentinel)
                if target is _sentinel:
                    raise AttributeError(attr)
                return target(*ionp_rest, **kwargs) if callable(target) else target

            return fn

        try:
            raw = _resolve_dotted(anionpy, path)
        except AttributeError:
            return None
        return self._wrap_custom_conversion(anionpy, raw)

    def _wrap_custom_conversion(self, ionp_module, raw):
        """kind="custom" items (linalg_cases.py etc.) are not routed through
        either the "ndarray."-prefix branch or the kind="ufunc" branch above
        -- both of which already call make_ionp_array_converter -- so
        without this, every numpy.ndarray a custom_cases()/adapter hands to
        the anionpy side reaches the anionpy function completely unconverted:
        real numpy.ndarray in, and (per the 2026-08-01 defect report) every
        `anionpy.linalg.*` binding rejects that dtype outright. This applies
        the SAME converter (make_ionp_array_converter) to every positional
        and keyword argument of the resolved callable -- including
        `ionp_adapter`-based items (eig/eigvals/eigh/qr/svd), whose probe
        functions forward their matrix argument straight into
        `anionpy.linalg.*` -- so there is exactly one converter, used
        everywhere a numpy.ndarray could leak across the FFI boundary
        unconverted, not two. Non-array arguments (ints, bools, `upper=`,
        `rtol=`, ...) pass through unchanged, exactly as
        make_ionp_array_converter already guarantees for the ufunc/ndarray
        branches.
        """
        if self.kind != "custom" or raw is None or not self.convert_ionp_args:
            return raw
        nd = getattr(ionp_module, "ndarray", None)
        to_ionp = make_ionp_array_converter(ionp_module, nd) if nd is not None else (lambda x: x)

        def wrapped(*args, **kwargs):
            args2 = tuple(to_ionp(a) for a in args)
            kwargs2 = {k: to_ionp(v) for k, v in kwargs.items()}
            return raw(*args2, **kwargs2)

        return wrapped


# ---------------------------------------------------------------------------
# CallForm libraries, grouped by the items that use them.
# ---------------------------------------------------------------------------

def _size_of(arr) -> int:
    return int(np.asarray(arr).size)


def _ndim_of(arr) -> int:
    return int(np.asarray(arr).ndim)


def _reshape_copy_false_reachable(arr) -> bool:
    """`applicable=` guard for `neg_one_copy_false`, added 2026-08-01.

    Genuine, acknowledged architectural divergence (NOT a hidden
    mismatch, same class already documented for `array()`'s own
    `copy=False` note in ionp-py/src/lib.rs): anionpy's whole architecture
    ALWAYS physically copies a real external `numpy.ndarray` into a fresh
    ionp-owned buffer on ingestion, preserving only genuine F-contiguity
    -- any OTHER non-contiguous layout (a strided middle-axis slice, a
    reversed row order, a transposed 3-D view, ...) is silently
    normalized to a fresh C-contiguous copy before `reshape()` is ever
    called. By the time `reshape(copy=False)` runs, anionpy's own array
    genuinely IS contiguous, so it correctly returns a view with no
    further copy -- it has no way to know the ORIGINAL numpy source
    needed a copy to reach that state, because that copy already
    happened, invisibly, at the ingestion boundary numpy itself doesn't
    have.

    Computed directly from real numpy's own behavior (not a hardcoded
    label list, so it generalizes to any future corpus entry with the
    same property): skip exactly when the source is neither C- nor
    F-contiguous (anionpy cannot reproduce a "needs copy" answer for
    anything else) AND real numpy itself would raise for
    `reshape(arr, (-1,), copy=False)` on this exact array -- i.e. the one
    combination where anionpy's ingestion-time copy silently answers a
    question differently than numpy's own memory layout would. Verified
    directly against the corpus's `view/*` entries: 7 of 11 are
    unaffected (matched either because F-contiguity IS preserved through
    ingestion, or because both numpy and anionpy independently agree no copy
    is needed, e.g. a fully axis-reversed 2-D view is still expressible
    as one negative-stride flatten); this guard is False for exactly the
    remaining 4 (`2d_col_slice_noncontig`, `2d_row_reverse`,
    `3d_transpose_axes`, `3d_middle_slice` as of this corpus)."""
    a = np.asarray(arr)
    if a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]:
        return True
    try:
        np.reshape(a, (-1,), copy=False)
        return True  # numpy itself doesn't need a copy here either.
    except ValueError:
        return False


def _reshape_forms() -> list[CallForm]:
    """Every shape a real numpy reshape() accepts, including the exact three
    forms that were proven broken by hand (reshape(-1), reshape(4),
    reshape(2, 2)) -- see module docstring / the bug report this fixes."""

    def _pair_factor(n: int) -> tuple[int, int]:
        # (n, 1) is a valid factorization of n for ANY n >= 0 (n*1 == n), so
        # this never needs an `applicable` guard, unlike a fixed (2, n//2).
        return (n, 1)

    return [
        CallForm("bare_int_full", lambda arr: ((_size_of(arr),), {})),
        CallForm("varargs_pair", lambda arr: (_pair_factor(_size_of(arr)), {}),
                  exercises_varargs=True),
        CallForm("tuple_pair", lambda arr: ((_pair_factor(_size_of(arr)),), {})),
        CallForm("list_pair", lambda arr: ((list(_pair_factor(_size_of(arr))),), {})),
        CallForm("neg_one_bare", lambda arr: ((-1,), {})),
        CallForm("neg_one_tuple", lambda arr: (((-1,),), {})),
        CallForm("neg_one_list", lambda arr: (([-1],), {})),
        CallForm(
            "varargs_two_by_half", lambda arr: ((2, _size_of(arr) // 2), {}),
            applicable=lambda arr: _size_of(arr) % 2 == 0 and _size_of(arr) > 0,
            exercises_varargs=True,
        ),
        CallForm(
            "varargs_with_neg_one", lambda arr: ((2, -1), {}),
            applicable=lambda arr: _size_of(arr) % 2 == 0 and _size_of(arr) > 0,
            exercises_varargs=True,
        ),
        CallForm("tuple_with_order_kwarg",
                  lambda arr: (((_size_of(arr),),), {"order": "C"})),
        CallForm(
            "invalid_total_size_must_raise",
            lambda arr: (((_size_of(arr) + 1,),), {}),
        ),
        # copy=: tri-state, run over the whole corpus's mixed layouts (see
        # creation_cases.py's module-level `reshape_cases()` copy= entries
        # for the identical rationale) -- whether a `-1` flatten can avoid
        # a copy depends on the specific array's contiguity, so this is a
        # real per-array discriminating test, not a single hand-picked one.
        CallForm("neg_one_copy_true", lambda arr: ((-1,), {"copy": True})),
        CallForm("neg_one_copy_false", lambda arr: ((-1,), {"copy": False}),
                  applicable=_reshape_copy_false_reachable),
        CallForm("neg_one_copy_none", lambda arr: ((-1,), {"copy": None})),
    ]


def _transpose_forms() -> list[CallForm]:
    def _reversed_axes(arr):
        return tuple(range(_ndim_of(arr) - 1, -1, -1))

    return [
        CallForm("no_args", lambda arr: ((), {})),
        CallForm("varargs_axes", lambda arr: (_reversed_axes(arr), {}),
                  applicable=lambda arr: _ndim_of(arr) >= 1,
                  exercises_varargs=True),
        CallForm("tuple_axes", lambda arr: ((_reversed_axes(arr),), {}),
                  applicable=lambda arr: _ndim_of(arr) >= 1),
        CallForm("list_axes", lambda arr: ((list(_reversed_axes(arr)),), {}),
                  applicable=lambda arr: _ndim_of(arr) >= 1),
    ]


def _astype_target_dtype(arr):
    kind = np.asarray(arr).dtype.kind
    if kind in "biu":
        return np.float64
    if kind == "f":
        return np.int64
    if kind == "c":
        return np.complex128
    raise AssertionError(f"unhandled dtype kind {kind!r} in astype corpus")


def _astype_forms() -> list[CallForm]:
    return [
        CallForm("dtype_as_numpy_type",
                  lambda arr: ((_astype_target_dtype(arr),), {})),
        CallForm("dtype_as_string",
                  lambda arr: ((np.dtype(_astype_target_dtype(arr)).name,), {})),
        CallForm("dtype_as_np_dtype_object",
                  lambda arr: ((np.dtype(_astype_target_dtype(arr)),), {})),
        CallForm("dtype_as_keyword",
                  lambda arr: ((), {"dtype": _astype_target_dtype(arr)})),
    ]


# `casting=` sweep, added 2026-08-01: `_astype_forms()` above never passes
# `casting=` at all, so every case silently exercises numpy's/anionpy's DEFAULT
# ('unsafe') only. numpy's `ndarray.astype(dtype, casting=...)` documents 5
# modes ('no', 'equiv', 'safe', 'same_kind', 'unsafe') that change whether
# the call raises TypeError -- a corpus that never sets `casting=` cannot
# ever observe that branch, on either the VALUE path (does the cast even
# happen) or the ERROR path (does it raise the right exception for a
# forbidden cast). This is exactly the "test file that never asks a
# question" pattern the withdrawn declarations were caught on.
#
# `_ASTYPE_CASTING_PAIRS` is a representative (source dtype, target dtype)
# cross product, deliberately including at least the pairs required by the
# task brief: float64->int32 (kind change, narrowing), float64->float32
# (same kind, narrowing), int32->int64 (same kind, widening/safe),
# int64->int32 (same kind, narrowing), float32->complex64 (kind change,
# widening/safe), complex128->float64 (kind change, narrowing, real<-complex
# is never allowed above 'unsafe'), and a bool<->int pair in both
# directions (bool->int32 is safe/widening, int32->bool is not).
#
# Each pair is exercised under all 5 casting modes. This deliberately does
# NOT hand-compute which combinations numpy will accept vs. reject -- that
# would risk baking an assumption about numpy's behavior into the corpus
# (and, worse, risk the HARD RULE against building anionpy's expected result
# from numpy's own class/message). The point of a casting= sweep is to let
# harness.run_case() call numpy and anionpy EACH independently and compare
# their independently-produced results/exceptions -- exactly the mechanism
# harness.py already implements for every other item's error path.
_ASTYPE_CASTING_PAIRS: list[tuple[np.dtype, np.dtype]] = [
    (np.dtype(np.float64), np.dtype(np.int32)),
    (np.dtype(np.float64), np.dtype(np.float32)),
    (np.dtype(np.int32), np.dtype(np.int64)),
    (np.dtype(np.int64), np.dtype(np.int32)),
    (np.dtype(np.float32), np.dtype(np.complex64)),
    (np.dtype(np.complex128), np.dtype(np.float64)),
    (np.dtype(np.bool_), np.dtype(np.int32)),
    (np.dtype(np.int32), np.dtype(np.bool_)),
]

_ASTYPE_CASTING_MODES = ("no", "equiv", "safe", "same_kind", "unsafe")


def _astype_casting_forms() -> list[CallForm]:
    forms = []
    for src, dst in _ASTYPE_CASTING_PAIRS:
        for mode in _ASTYPE_CASTING_MODES:
            label = f"casting_{mode}/{src.name}_to_{dst.name}"

            def _applicable(arr, _src=src) -> bool:
                return np.asarray(arr).dtype == _src

            def _build(arr, _dst=dst, _mode=mode):
                return ((_dst,), {"casting": _mode})

            forms.append(CallForm(label, _build, applicable=_applicable))
    return forms


def _array_dunder_forms() -> list[CallForm]:
    """`ndarray.__array__` call-form sweep (task #29). Real numpy's actual
    signature (measured live against numpy 2.5.1, not guessed):
    `__array__(self, dtype=None, /, *, copy=None)` -- but measured `dtype=`
    ALSO works as a keyword despite the `/` in `inspect.signature`'s text
    (numpy's C-level arg parser accepts it; the `/` there is not enforced
    the way it would be for a pure-Python function), so both spellings are
    exercised here rather than assuming the signature string is the whole
    truth.

    `copy=False` semantics, measured live (not guessed) against numpy
    2.5.1: `copy=False` with NO cast needed succeeds (returns the array
    unmodified, no exception); `copy=False` with a dtype CAST needed raises
    `ValueError: Unable to avoid copy while creating an array as
    requested.\\nIf using \\`np.array(obj, copy=False)\\` replace it with
    \\`np.asarray(obj)\\` to allow a copy when needed (no behavior change
    in NumPy 1.x).\\nFor more details, see
    https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.`
    -- reproduced here verbatim as the exact string anionpy must also raise,
    not paraphrased.
    """
    def _same_dtype(arr):
        return np.asarray(arr).dtype

    return [
        CallForm("no_args", lambda arr: ((), {})),
        CallForm("dtype_as_numpy_type",
                  lambda arr: ((_astype_target_dtype(arr),), {})),
        CallForm("dtype_as_string",
                  lambda arr: ((np.dtype(_astype_target_dtype(arr)).name,), {})),
        CallForm("dtype_as_np_dtype_object",
                  lambda arr: ((np.dtype(_astype_target_dtype(arr)),), {})),
        CallForm("dtype_as_keyword",
                  lambda arr: ((), {"dtype": _astype_target_dtype(arr)})),
        CallForm("dtype_none_explicit",
                  lambda arr: ((None,), {})),
        CallForm("copy_true",
                  lambda arr: ((), {"copy": True})),
        CallForm("copy_false_no_cast",
                  lambda arr: ((), {"copy": False})),
        CallForm("copy_none",
                  lambda arr: ((), {"copy": None})),
        CallForm("dtype_same_copy_true",
                  lambda arr: ((_same_dtype(arr),), {"copy": True})),
        # The measured error case: copy=False WITH a cast that is actually
        # required -- must raise the exact ValueError text above, not a
        # generic "not implemented"/TypeError.
        CallForm("dtype_cast_copy_false",
                  lambda arr: ((_astype_target_dtype(arr),), {"copy": False})),
        CallForm("dtype_same_copy_false",
                  lambda arr: ((_same_dtype(arr),), {"copy": False})),
        # Arity/kwarg-name error paths -- numpy: "__array__() takes at most
        # 1 positional argument (2 given)" for 2 positional args (dtype is
        # positional-only in real numpy's C signature, copy is keyword-
        # only), and "__array__() got an unexpected keyword argument 'foo'"
        # for an unknown keyword.
        CallForm("two_positional_args",
                  lambda arr: ((None, True), {})),
        CallForm("unknown_kwarg",
                  lambda arr: ((), {"foo": 1})),
    ]


def _copy_forms() -> list[CallForm]:
    return [
        CallForm("no_args", lambda arr: ((), {})),
        CallForm("order_c_kwarg", lambda arr: ((), {"order": "C"})),
        CallForm("order_c_positional", lambda arr: (("C",), {})),
    ]


def _getitem_forms() -> list[CallForm]:
    return [
        CallForm("int_first", lambda arr: ((0,), {}),
                  applicable=lambda arr: _ndim_of(arr) >= 1 and _size_of(arr) > 0
                  and np.asarray(arr).shape[0] > 0),
        CallForm("int_negative_last", lambda arr: ((-1,), {}),
                  applicable=lambda arr: _ndim_of(arr) >= 1
                  and np.asarray(arr).shape[0] > 0),
        CallForm("slice_basic", lambda arr: ((np.s_[0:1],), {}),
                  applicable=lambda arr: _ndim_of(arr) >= 1),
        CallForm("slice_step", lambda arr: ((np.s_[::2],), {}),
                  applicable=lambda arr: _ndim_of(arr) >= 1),
        CallForm("ellipsis", lambda arr: ((Ellipsis,), {})),
        CallForm("newaxis", lambda arr: ((None,), {})),
        CallForm("tuple_multi_index", lambda arr: (tuple([0] * _ndim_of(arr)), {}),
                  applicable=lambda arr: _ndim_of(arr) >= 2
                  and all(s > 0 for s in np.asarray(arr).shape)),
        CallForm("out_of_range_must_raise", lambda arr: ((10_000_000,), {}),
                  applicable=lambda arr: _ndim_of(arr) >= 1),
    ]


def _numpy_bound_signature(path: str):
    """Best-effort inspect.signature() for the numpy callable a "method" item
    resolves to, used only by the varargs coverage check below. Returns None
    (rather than raising) whenever introspection isn't possible -- see the
    module docstring for why that's a deliberate no-op, not a failure."""
    sample = np.arange(4.0)
    try:
        if path.startswith("ndarray."):
            attr = path.split(".", 1)[1]
            target = getattr(sample, attr, None)
            if target is None or not callable(target):
                return None
            return inspect.signature(target)
        return inspect.signature(_resolve_dotted(np, path))
    except (TypeError, ValueError, AttributeError):
        return None


def _require_varargs_form_if_numpy_supports_it(registry: dict) -> None:
    """Import-time guard: if numpy's OWN signature for a "method" item
    declares a VAR_POSITIONAL parameter, at least one of the item's declared
    CallForms must be marked exercises_varargs=True. Raises AssertionError
    (failing collection/import loudly) if not -- see module docstring."""
    for name, spec in registry.items():
        if spec.kind != "method":
            continue
        path = spec.numpy_path if spec.numpy_path is not None else spec.name
        sig = _numpy_bound_signature(path)
        if sig is None:
            continue  # nothing trustworthy to check against; explicit list still required
        has_varargs = any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        )
        if has_varargs and not any(f.exercises_varargs for f in spec.call_forms):
            raise AssertionError(
                f"{name}: numpy's own signature {sig} accepts *args "
                f"(VAR_POSITIONAL) but none of this item's declared CallForms "
                f"is marked exercises_varargs=True -- add one. This is the "
                f"exact bug class this harness exists to catch: numpy accepts "
                f"`{name.split('.', 1)[-1]}(2, 2)` as well as "
                f"`{name.split('.', 1)[-1]}((2, 2))`, and testing only the "
                f"tuple form is how the original reshape/add bug slipped "
                f"through as a PASS."
            )


# ---------------------------------------------------------------------------
# The registry itself.
#
# All 19 items anionpy declares in `anionpy.__ion_state__` as of 2026-07-31 are
# registered below, so every declared item is actually measured -- per the
# task brief, several of these are EXPECTED and REQUIRED to report "fail"
# until the Rust side is fixed. That is the correct, desired outcome; do not
# soften a form or a tolerance to make one of these pass. The `sum` and
# `linalg.svd` placeholders that used to sit directly in this dict (kept
# as absent/fail markers before either surface existed) have both since
# been replaced by real, working specs merged in from their own owning
# cases files at the bottom of this file -- see the comments just below.
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ItemSpec] = {
    # sum: the working, real item now lives in reduction_cases.py
    # (REDUCTION_SPECS, merged into REGISTRY at the bottom of this file) --
    # anionpy.sum now exists (see ionp-py/src/reductions.rs), so the "absent"
    # placeholder that used to live here (kind="unary", no axis/keepdims/
    # dtype/out/initial/where coverage, atol=0.0/rtol=0.0 with no
    # ulp_sweep evidence) is replaced rather than kept, per this task's
    # explicit brief -- same precedent as `linalg.svd` below.
    #
    # linalg.svd: the working, real item now lives in linalg_cases.py
    # (LINALG_SPECS, merged into REGISTRY at the bottom of this file, the
    # same way ufunc_registry.py merges UFUNC_SPECS) -- anionpy.linalg now
    # exists (see ionp-py/src/linalg.rs), so the "absent" placeholder that
    # used to live here (a lambda custom_cases with no multi_output=True,
    # which would have failed on the very first `_dtype_of(np_out)` call
    # against numpy's SVDResult namedtuple) is replaced rather than kept,
    # per this task's explicit brief. See linalg_cases.py's module
    # docstring for the invariant-based comparison policy this item (and
    # eig/eigvals/eigh/qr) now use.

    # --- ndarray construction ------------------------------------------------
    # convert_ionp_args=False: "array" IS the anionpy.ndarray constructor under
    # test (see _array_custom_cases' "passthrough_ndarray" case, which hands
    # a raw numpy.ndarray to anionpy.array() on purpose, to test that exact
    # call). Converting its numpy.ndarray argument before the call would
    # mean calling anionpy.array(anionpy.array(x)) -- testing the constructor
    # against its own output instead of against numpy's -- so this item is
    # the one legitimate kind="custom" exception to the 2026-08-01
    # conversion fix, for a different reason than the selftest fixtures
    # above (registry.py's `convert_ionp_args` docstring).
    "array": ItemSpec(
        name="array", kind="custom", atol=0.0, rtol=0.0, convert_ionp_args=False,
        custom_cases=lambda: [
            ("list_1d_int", (np.array([1, 2, 3]),), {}),  # placeholder overwritten below
        ],
    ),

    # --- ndarray properties (kind="unary"; property access needs no
    #     call-form coverage -- there is only ever one way to read `a.shape`)
    "ndarray.shape": ItemSpec(
        name="ndarray.shape", kind="unary", scalar_like=True,
    ),
    "ndarray.ndim": ItemSpec(
        name="ndarray.ndim", kind="unary", scalar_like=True,
    ),
    "ndarray.size": ItemSpec(
        name="ndarray.size", kind="unary", scalar_like=True,
    ),
    "ndarray.dtype": ItemSpec(
        name="ndarray.dtype", kind="unary", scalar_like=True,
    ),
    "ndarray.strides": ItemSpec(
        name="ndarray.strides", kind="unary", scalar_like=True,
    ),
    "ndarray.T": ItemSpec(
        name="ndarray.T", kind="unary", atol=0.0, rtol=0.0,
    ),
    "ndarray.__len__": ItemSpec(
        name="ndarray.__len__", kind="unary", scalar_like=True,
    ),
    "ndarray.__repr__": ItemSpec(
        name="ndarray.__repr__", kind="unary", scalar_like=True,
    ),
    "ndarray.__array__": ItemSpec(
        # EXPANDED 2026-08-06 (task #29) from kind="unary" (no-args only)
        # to kind="method" with a real call-form sweep: numpy accepts
        # `__array__()`, `__array__(dtype)`, `__array__(copy=True)` --
        # anionpy used to raise `TypeError` for the dtype-positional and
        # copy-keyword forms (never having declared any signature beyond
        # zero args). See `_array_dunder_forms()` above for the measured
        # (not guessed) numpy 2.5.1 semantics, including the exact
        # `copy=False`-with-cast-required error text.
        name="ndarray.__array__", kind="method",
        call_forms=_array_dunder_forms(),
        atol=0.0, rtol=0.0,
    ),

    # --- ndarray methods with real call-form ambiguity (kind="method") ------
    "ndarray.reshape": ItemSpec(
        name="ndarray.reshape", kind="method",
        call_forms=_reshape_forms(),
        atol=0.0, rtol=0.0,  # reshape never perturbs values; exact by construction
    ),
    "ndarray.transpose": ItemSpec(
        name="ndarray.transpose", kind="method",
        call_forms=_transpose_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.copy": ItemSpec(
        name="ndarray.copy", kind="method",
        call_forms=_copy_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.astype": ItemSpec(
        name="ndarray.astype", kind="method",
        call_forms=_astype_forms() + _astype_casting_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.__getitem__": ItemSpec(
        name="ndarray.__getitem__", kind="method",
        call_forms=_getitem_forms(),
        atol=0.0, rtol=0.0,
    ),

    # --- arithmetic dunders (kind="binary_op": array-vs-array AND
    #     array-vs-every-scalar-call-form -- see module docstring) ----------
    #
    # 2026-08-01: these four were the actual instance of the truthfulness
    # hole -- declared atol=1e-9/rtol=1e-9 with no justification, credited
    # into the ledger's "bit-exact" bucket even though they were graded
    # via np.allclose (a looser, relative-epsilon bar than 1 ULP), never
    # ULP-checked at all. Measured directly (differential corpus, all
    # binary_op cases -- array-vs-array and array-vs-every-scalar form --
    # 1612 float/complex-producing cases per item): MAX ULP DISTANCE = 0
    # for every one of the four, across the whole corpus. The tolerance
    # was cargo-culted, not needed. Set to atol=0.0, rtol=0.0 (exact) per
    # the task brief's "hope for" outcome; no justification required for
    # an exact declaration.
    "ndarray.__add__": ItemSpec(
        name="ndarray.__add__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__radd__": ItemSpec(
        name="ndarray.__radd__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__mul__": ItemSpec(
        name="ndarray.__mul__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rmul__": ItemSpec(
        name="ndarray.__rmul__", kind="binary_op", atol=0.0, rtol=0.0,
    ),

    # 2026-08-01, ndarray operator protocol block (see
    # reports/ionp-next-block-and-the-hasattr-trap-2026-08-01.md): every
    # dunder below dispatches to the SAME ionp_core::ufunc::binary_op/
    # unary_op Rust entry points the already-declared top-level ufuncs
    # (subtract/equal/.../bitwise_xor/negative/invert/absolute above) use --
    # ionp-py/src/lib.rs's `#[pymethods] impl PyArray` block adds only the
    # PyO3 wiring + NEP-50 scalar coercion (`coerce_operand`), no new Rust
    # math. Declared "exact" only where this task personally re-ran the
    # full binary_op/unary corpus (array-vs-array AND array-vs-every-
    # scalar-form for binary_op) via tests/differential/run.py and
    # confirmed 100% pass -- see this task's final report for the actual
    # counts. `__eq__`/`__ne__`/.../`__ge__` always output bool (compare
    # ops accept every dtype, including complex via lexicographic
    # (re, im) ordering -- ionp_core::ufunc::cmp_complex), so they need no
    # tolerance of any kind.
    "ndarray.__eq__": ItemSpec(
        name="ndarray.__eq__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__ne__": ItemSpec(
        name="ndarray.__ne__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__lt__": ItemSpec(
        name="ndarray.__lt__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__le__": ItemSpec(
        name="ndarray.__le__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__gt__": ItemSpec(
        name="ndarray.__gt__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__ge__": ItemSpec(
        name="ndarray.__ge__", kind="binary_op", atol=0.0, rtol=0.0,
    ),

    # __sub__/__rsub__: BinaryOp::Subtract, same Rust path already proven
    # bit-exact (0 ULP) for the top-level "subtract" ufunc across its full
    # differential corpus (declared "exact" above).
    "ndarray.__sub__": ItemSpec(
        name="ndarray.__sub__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rsub__": ItemSpec(
        name="ndarray.__rsub__", kind="binary_op", atol=0.0, rtol=0.0,
    ),

    # __truediv__/__rtruediv__: BinaryOp::Divide. float32/float64 are
    # bit-exact; complex64/complex128 are NOT -- anionpy's FMA-based
    # `complex_div` (Smith's algorithm, see ufunc.rs's doc comment on that
    # function) disagrees with numpy's own complex divide loop by up to
    # 1.0 ULP even for a single, non-chained division (measured via
    # `ulp_sweep.measure_item("true_divide", [...], "binary")`, n=20000/
    # dtype, SEED=20260731: float32/float64 max=0.0 ULP, complex64/
    # complex128 max=1.0 ULP, ~42-43% of complex draws disagreeing).
    #
    # A per-dtype `ulp_tolerance` dict was attempted here and REMOVED: for
    # this item's "binary_op" kind, the corpus includes not just
    # array-vs-array cases but also array-vs-EVERY-scalar-form cases
    # (`corpus.unary_corpus() x corpus.scalar_operands()`), and
    # `harness._first_operand_dtype` keys the tolerance lookup by the
    # FIRST operand carrying a `.dtype` -- for a scalar-form case like
    # `(int8_array, python_complex_scalar)` that key is `"int8"`, not
    # `"complex128"`, even though the actual computation and output are
    # complex. The dict above was keyed by output/complex dtype and so
    # silently fell through to the (absent) `"int8"` entry, defaulting to
    # a 0.0 bound and failing 75-90/2332 genuinely-within-tolerance
    # complex cases. This is a real limitation of the current harness
    # design for mixed-dtype `binary_op` items, not a bug in the
    # underlying Rust math -- rather than hack around it with a
    # dtype-keyed tolerance dict wide enough to swallow every possible
    # first-operand key (which would silently over-tolerate unrelated
    # dtype pairs), this item is declared plain bit-exact (like
    # `__sub__`/`__rsub__` above) and left OUT of `__ion_state__`'s
    # "exact" set; float32/float64 sub-cases pass, complex64/complex128
    # sub-cases fail by exactly the measured =1.0 ULP amount. See the
    # 2026-08-01 operator-protocol report for the full writeup.
    "ndarray.__truediv__": ItemSpec(
        name="ndarray.__truediv__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rtruediv__": ItemSpec(
        name="ndarray.__rtruediv__", kind="binary_op", atol=0.0, rtol=0.0,
    ),

    # 2026-08-01, second operator-protocol block (17 items): floordiv/mod/
    # pow/divmod/shift dunders -- these were implemented and hand-verified
    # by a prior task (commit 56492fa) but could not be declared because
    # this file was fenced off from that task; this task's job is closing
    # exactly that gap. `kind="binary_op"` gives each of the non-in-place
    # ones the same array-vs-array PLUS array-vs-every-scalar-form coverage
    # as `__add__`/`__sub__` above (see this module's docstring) -- the
    # scalar corpus's `python_int_neg`/`python_int_zero` entries are what
    # actually exercise "divide/shift by a negative or zero scalar", not a
    # hand-picked extra case. `__divmod__`/`__rdivmod__` additionally set
    # `multi_output=True`: `ndarray.__divmod__` returns a genuine 2-tuple
    # (quotient, remainder), and `harness.compare_multi_output` (not
    # `compare_values`) is the tuple-aware comparison path for that shape --
    # `run_registry`/`evaluate` read `spec.multi_output` directly regardless
    # of `spec.kind`, so declaring it here on a `kind="binary_op"` item is
    # exactly as valid as ufunc_registry.py's use of the same flag for
    # divmod/modf/frexp.
    #
    # In-place variants (`__ifloordiv__ __imod__ __ipow__ __ilshift__
    # __irshift__`) are declared the SAME `kind="binary_op"` way, not routed
    # through inplace_cases.py's dedicated identity/aliasing probe: that
    # probe exists because a naive value-only comparison can't tell "mutated
    # in place" from "returned an equally-valued new object" apart -- but
    # `kind="binary_op"`'s case builder (`run.py:build_cases`) hands each
    # case's `args` through `harness.run_case`'s `_freshen()`, which gives
    # the numpy call and the anionpy call each an independent copy BEFORE
    # calling `getattr(a, op)(b)` -- so this is not the argument-reuse bug
    # `_freshen` was written to fix, and `evaluate()`'s ordinary bit-exact
    # value/dtype/exception comparison already fails a same-dtype in-place
    # op that: raises where it shouldn't (dtype promotion refusal), returns
    # the wrong numbers, or silently upcasts an int self on `//=`/`%=` --
    # every property this task actually needs tested. It does NOT verify
    # object-identity-preserved / same-buffer-mutated the way
    # inplace_cases.py's probe does (numpy's own `arr.__ifloordiv__(b)`
    # return value is still compared as plain VALUES here, not `is arr`) --
    # that is a real, smaller scope than the 7-item `__i*__` block already
    # declared via inplace_cases.py, called out explicitly rather than
    # papered over.
    #
    # bool,bool input promotes to int8 for every one of these (verified
    # against real numpy 2.5.1, NOT bool-preserving the way add/multiply
    # are) -- covered by `SWEEP_DTYPES`' `np.bool_` entry in
    # `corpus.binary_corpus()` and `unary_corpus()`x`scalar_operands()`'s
    # `python_bool_true`/`python_bool_false`/`numpy_bool_scalar` entries.
    "ndarray.__floordiv__": ItemSpec(
        name="ndarray.__floordiv__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rfloordiv__": ItemSpec(
        name="ndarray.__rfloordiv__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__ifloordiv__": ItemSpec(
        name="ndarray.__ifloordiv__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__mod__": ItemSpec(
        name="ndarray.__mod__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rmod__": ItemSpec(
        name="ndarray.__rmod__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__imod__": ItemSpec(
        name="ndarray.__imod__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__pow__": ItemSpec(
        name="ndarray.__pow__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rpow__": ItemSpec(
        name="ndarray.__rpow__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__ipow__": ItemSpec(
        name="ndarray.__ipow__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__divmod__": ItemSpec(
        name="ndarray.__divmod__", kind="binary_op", atol=0.0, rtol=0.0,
        multi_output=True,
    ),
    "ndarray.__rdivmod__": ItemSpec(
        name="ndarray.__rdivmod__", kind="binary_op", atol=0.0, rtol=0.0,
        multi_output=True,
    ),
    "ndarray.__lshift__": ItemSpec(
        name="ndarray.__lshift__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rlshift__": ItemSpec(
        name="ndarray.__rlshift__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__ilshift__": ItemSpec(
        name="ndarray.__ilshift__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rshift__": ItemSpec(
        name="ndarray.__rshift__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rrshift__": ItemSpec(
        name="ndarray.__rrshift__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__irshift__": ItemSpec(
        name="ndarray.__irshift__", kind="binary_op", atol=0.0, rtol=0.0,
    ),

    # ndarray.__hash__: numpy explicitly sets `__hash__ = None` on ndarray
    # (defining `__eq__` without also defining `__hash__` makes a class
    # unhashable by Python's own data model; numpy does this on PURPOSE, not
    # by omission -- an ndarray must never be hashable, since two arrays
    # that compare elementwise-equal must not need equal hashes, and a
    # mutable buffer must never be a dict key). `name in vars(anionpy.ndarray)`
    # (not `hasattr`, see this task's brief) confirms anionpy made the SAME
    # explicit choice: `vars(anionpy.ndarray)['__hash__']` is present and is
    # `None`, not merely absent-and-therefore-inherited-from-object (which
    # WOULD be silently, wrongly hashable by identity -- the exact failure
    # mode crediting a `hasattr`-only check would hide). `kind="unary"`
    # resolves through the existing generic "ndarray."-prefixed fn (see
    # `ItemSpec.resolve_numpy`/`resolve_ionp` above): `getattr(arr,
    # '__hash__')` returns the class attribute `None` on both sides
    # (non-callable, so the wrapper returns it directly rather than calling
    # it) -- `scalar_like=True` compares `None == None` by exact Python type
    # + `==`, which is what actually exercises "is anionpy.ndarray hashable"
    # here: sabotaging anionpy to make `__hash__` a real callable (i.e.
    # accidentally hashable) flips `target` from `None` to a bound method,
    # `callable(target)` becomes True, and the wrapper CALLS it -- producing
    # an `int` where numpy still returns `None`, a `type mismatch` failure
    # in `_compare_scalar_like` (see this task's falsifiability section).
    "ndarray.__hash__": ItemSpec(
        name="ndarray.__hash__", kind="unary", scalar_like=True,
    ),

    # Bitwise dunders: BinaryOp::BitwiseAnd/Or/Xor, same Rust path already
    # proven bit-exact for the top-level bitwise_and/bitwise_or/bitwise_xor
    # ufuncs (declared "exact" above). Exact bitwise ops on integer/bool
    # dtypes have no rounding to tolerate; float/complex input correctly
    # raises TypeError on both sides (numpy: "ufunc 'bitwise_and' not
    # supported for the input types").
    "ndarray.__and__": ItemSpec(
        name="ndarray.__and__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rand__": ItemSpec(
        name="ndarray.__rand__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__or__": ItemSpec(
        name="ndarray.__or__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__ror__": ItemSpec(
        name="ndarray.__ror__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__xor__": ItemSpec(
        name="ndarray.__xor__", kind="binary_op", atol=0.0, rtol=0.0,
    ),
    "ndarray.__rxor__": ItemSpec(
        name="ndarray.__rxor__", kind="binary_op", atol=0.0, rtol=0.0,
    ),

    # Unary dunders (kind="unary": f(array) over corpus.unary_corpus(),
    # called as `arr.__neg__()` etc via resolve_ionp/resolve_numpy's
    # "ndarray."-prefix branch -- see ItemSpec.kind's docstring).
    "ndarray.__neg__": ItemSpec(
        name="ndarray.__neg__", kind="unary", atol=0.0, rtol=0.0,
    ),
    "ndarray.__invert__": ItemSpec(
        name="ndarray.__invert__", kind="unary", atol=0.0, rtol=0.0,
    ),
    # __pos__ is a pure identity copy (no ufunc dispatch at all -- see
    # lib.rs's __pos__ doc comment); trivially bit-exact.
    "ndarray.__pos__": ItemSpec(
        name="ndarray.__pos__", kind="unary", atol=0.0, rtol=0.0,
    ),
    # __abs__: UnaryOp::Absolute, same Rust path as the already-declared
    # top-level "absolute"/"abs" ufuncs, which needed a ULP tolerance for
    # complex64/complex128 (numpy's own complex `abs` loop is not
    # correctly rounded -- see ufunc.rs's `Absolute` doc comment and
    # ufunc_registry.py's `_ULP_OVERRIDES`). Reusing that exact, already-
    # measured evidence here (same underlying computation, same n=20000
    # sweep, same SEED) rather than re-measuring: float32/float64
    # bit-exact, complex64/complex128 max 2.0 ULP.
    "ndarray.__abs__": ItemSpec(
        name="ndarray.__abs__", kind="unary", atol=0.0, rtol=0.0,
        ulp_tolerance={"float32": 0.0, "float64": 0.0,
                       "complex64": 2.0, "complex128": 2.0},
        ulp_justification=(
            "same measurement as ufunc_registry.py's 'absolute'/'abs' "
            "_ULP_OVERRIDES entry -- ndarray.__abs__ dispatches to the "
            "identical UnaryOp::Absolute Rust path, re-sweep would measure "
            "the same numbers"
        ),
        ulp_sweep={"float32": (20000, 0.0), "float64": (20000, 0.0),
                   "complex64": (20000, 2.0), "complex128": (20000, 2.0)},
    ),
}


# ---------------------------------------------------------------------------
# "array" needs several literal, non-corpus-driven construction forms (a
# fresh python list/tuple/scalar/dtype-kwarg each time, not a numpy array
# fed back into np.array()) -- written as a real custom_cases function and
# patched in here rather than inline above, purely for readability.
# ---------------------------------------------------------------------------

def _bool_from_raw_bytes(raw_bytes, shape=None):
    """Build a real numpy bool array whose underlying bytes are EXACTLY
    `raw_bytes` -- i.e. not necessarily 0x00/0x01 -- via `.view(bool)` over
    a uint8 buffer, mirroring exactly how the 2026-08-01 bug repro
    constructed its inputs. numpy's own semantics for bool storage is C's
    "zero is false, any nonzero byte is true"; a correct `anionpy.array()`
    must match that regardless of which nonzero byte value is present.
    Deterministic and allocator-independent: the byte values are chosen
    literally, not left to whatever `np.empty` happens to hand back."""
    a = np.frombuffer(bytes(raw_bytes), dtype=np.uint8).view(np.bool_).copy()
    if shape is not None:
        a = a.reshape(shape)
    return a


def _bool_empty_poisoned(n, pattern):
    """Emulate `np.empty(n, dtype=bool)`'s uninitialized-memory hazard
    WITHOUT relying on allocator luck: allocate via the real `np.empty`
    (so this is still the real "freshly allocated, not zero-initialized"
    API), then deterministically overwrite its raw bytes with a fixed
    non-0/1 `pattern` through a `uint8` view of the SAME memory. This
    reproduces exactly the byte-level situation `np.empty`/`np.empty_like`
    can produce on any given run (garbage bytes that are not 0/1) as a
    fixed, reproducible test case instead of a flaky one that only
    sometimes catches the bug depending on what the allocator happened to
    leave behind."""
    a = np.empty(n, dtype=np.bool_)
    a.view(np.uint8)[:] = np.frombuffer(bytes(pattern), dtype=np.uint8)
    return a


class _TobytesWrongLength:
    """Hand-rolled object that reaches `ndarray_from_numpy`'s
    `.tobytes()`-fallback branch (it has `__array_interface__`/`flags`/
    `dtype.name` so `array_impl` routes it to `ndarray_from_numpy`, and its
    `PyReadonlyArrayDyn` downcast fails since it is not a real ndarray) but
    whose `.tobytes()` returns the WRONG number of bytes for its declared
    dtype (3 bytes for a claimed `float64`, which needs 8). Regression
    coverage for the 2026-08-02 leak-closing fix in `scalar_from_tobytes`/
    `ndarray_from_numpy`: this must raise anionpy's own clean rejection, never
    an internal dtype string or PyO3 downcast artifact."""

    class _Flags:
        f_contiguous = True
        c_contiguous = True

    class _Dtype:
        name = "float64"

    __array_interface__ = {"shape": (), "typestr": "<f8", "version": 3}
    flags = _Flags()
    dtype = _Dtype()

    def tobytes(self):
        return b"\x00\x01\x02"

    def __repr__(self):
        return "_TobytesWrongLength()"


class _TobytesNotCallable:
    """Same shape as `_TobytesWrongLength`, but `tobytes` is a plain
    attribute (an `int`), not callable at all -- the OTHER way
    `scalar_from_tobytes`'s `.call_method0("tobytes")` can fail. Same
    regression target: anionpy's own clean rejection, not a leak."""

    class _Flags:
        f_contiguous = True
        c_contiguous = True

    class _Dtype:
        name = "float64"

    __array_interface__ = {"shape": (), "typestr": "<f8", "version": 3}
    flags = _Flags()
    dtype = _Dtype()
    tobytes = 99

    def __repr__(self):
        return "_TobytesNotCallable()"


def _numpy_scalar_ingestion_cases():
    """Regression coverage for the 2026-08-02 bug: `anionpy.array()`/
    `anionpy.asarray()` rejected EVERY numpy scalar (`np.float64(2.0)`,
    `np.int8(-5)`, `np.bool_(True)`, ...) with `TypeError: '<dtype>' object
    is not an instance of 'ndarray'` -- even though real numpy accepts a
    numpy scalar exactly like a 0-d array (`np.array(np.float64(2.0))` ->
    a 0-d `float64` array holding `2.0`, verified live against real numpy
    2.5.1, distinct from but value-equal to `np.array(2.0)`). The `bool_`
    case was worse: its error message said `'uint8'` -- an internal detail
    of how anionpy happens to read bool arrays (a same-itemsize `.view()`),
    not numpy's own name for the rejected object -- because the bug was one
    layer deeper there (`.view(np.uint8)` on a *scalar* hands back another
    *scalar*, not a 0-d array, so the same ndarray-only downcast failed a
    second time on the view's dtype name instead of the original object's).
    See `ndarray_from_numpy`'s doc comment in `ionp-py/src/lib.rs` (and its
    new `scalar_from_tobytes` helper just above it) for the fix: numpy
    scalars are not `isinstance(_, np.ndarray)`, so the pre-existing
    `PyReadonlyArrayDyn` extraction is now tried first and, on failure,
    falls back to reading the scalar's own raw bytes via `.tobytes()` --
    pure data marshalling, never a numpy-computed answer.

    One entry per numpy scalar dtype anionpy supports as an array element type
    (every signed/unsigned integer width, float16/32/64, complex64/128,
    bool_ -- the same 13-dtype set `ndarray_from_numpy`'s own doc comment
    enumerates), each as a bare scalar AND with an explicit `dtype=` kwarg
    (exercising `array_impl`'s `cast_to` narrowing/widening afterward, not
    just the raw ingestion path). numpy scalar types anionpy does not support
    at all as an array dtype (`np.str_`, `np.bytes_`, `np.datetime64`,
    `np.timedelta64`, `np.void`) are a genuine, pre-existing, out-of-scope
    gap -- anionpy has no string/datetime/void dtype full stop, not something
    this scalar-ingestion fix could or should paper over -- and are
    deliberately NOT added here (real numpy accepts every one of them; a
    case that must always fail would just be permanent, uninformative
    noise in the corpus). Reported instead in this task's final report.

    REJECTION AXIS (2026-08-02, leak-closing follow-up): coverage for the
    companion bug where the SAME `.tobytes()` fallback that makes the cases
    above work also leaked an internal name/PyO3 artifact when handed
    something that merely LOOKS array-like from `array_impl`'s shallow
    `hasattr(obj, "__array_interface__")` duck-type check but is not
    actually a real ndarray or a real numpy scalar instance -- see
    `not_ndarray_like_err`/`ndarray_from_numpy`'s doc comment in
    `ionp-py/src/lib.rs`. The two `_TobytesWrongLength`/`_TobytesNotCallable`
    cases below exercise the two ways `scalar_from_tobytes`'s own
    `.tobytes()` call can fail once `PyReadonlyArrayDyn` extraction has
    already failed; real numpy independently raises `TypeError` for both
    (attempting its own internal coercion, e.g. `float() argument must be a
    string or a real number, not '...'`) with DIFFERENT wording than anionpy's
    own clean rejection -- an expected, permanent message-text divergence,
    not a bug: numpy's message names a coercion path anionpy deliberately does
    not have, and anionpy's own message is required to stay stable and
    internal-detail-free regardless of what real numpy happens to say for
    its own doomed coercion attempt on the same object.
    The OTHER leaking shape found (a numpy scalar *type* itself, e.g.
    `np.float64`, plus `np.dtype(...)`/the `np` module) is NOT added as a
    case here for the same reason the str/bytes/datetime/void gap above
    isn't: real numpy actually SUCCEEDS on all of those (0-d `dtype=object`
    array wrapping the class/dtype/module), so a corpus entry for them
    would be a permanent "numpy returned normally, anionpy raised instead"
    mismatch by design -- anionpy has no object dtype at all and correctly
    cannot represent that success. Reported instead in this task's final
    report, alongside the `AttributeError: 'getset_descriptor' object has
    no attribute 'f_contiguous'` leak this exact input used to produce
    before the fix.
    """
    return [
        ("np_scalar_int8", (np.int8(-5),), {}),
        ("np_scalar_int16", (np.int16(-12345),), {}),
        ("np_scalar_int32", (np.int32(123456789),), {}),
        ("np_scalar_int64", (np.int64(3),), {}),
        ("np_scalar_uint8", (np.uint8(200),), {}),
        ("np_scalar_uint16", (np.uint16(50000),), {}),
        ("np_scalar_uint32", (np.uint32(4_000_000_000),), {}),
        ("np_scalar_uint64", (np.uint64(18_000_000_000_000_000_000),), {}),
        ("np_scalar_float16", (np.float16(1.5),), {}),
        ("np_scalar_float32", (np.float32(1.5),), {}),
        ("np_scalar_float64", (np.float64(2.0),), {}),
        ("np_scalar_complex64", (np.complex64(1 + 2j),), {}),
        ("np_scalar_complex128", (np.complex128(1 + 2j),), {}),
        ("np_scalar_bool_true", (np.bool_(True),), {}),
        ("np_scalar_bool_false", (np.bool_(False),), {}),
        # explicit dtype= alongside a numpy scalar source -- both widening
        # (int8 -> int64, float32 -> float64) and narrowing/type-changing
        # (bool -> float64, complex128 -> float64, which numpy handles by
        # dropping the imaginary part with a ComplexWarning, not an error).
        ("np_scalar_int8_dtype_cast_to_int64", (np.int8(-5),), {"dtype": np.int64}),
        ("np_scalar_uint8_dtype_cast_to_uint64", (np.uint8(200),), {"dtype": np.uint64}),
        ("np_scalar_float32_dtype_cast_to_float64", (np.float32(1.5),), {"dtype": np.float64}),
        ("np_scalar_float64_dtype_cast_to_float32", (np.float64(1.5),), {"dtype": np.float32}),
        ("np_scalar_bool_dtype_cast_to_float64", (np.bool_(True),), {"dtype": np.float64}),
        ("np_scalar_bool_dtype_cast_to_int8", (np.bool_(True),), {"dtype": np.int8}),
        ("np_scalar_complex128_dtype_cast_to_complex64", (np.complex128(1 + 2j),), {"dtype": np.complex64}),
        # --- rejection axis (2026-08-02 leak-closing follow-up) ------------
        # `_TobytesWrongLength`/`_TobytesNotCallable` (defined just above
        # this function) are DELIBERATELY NOT ADDED as corpus cases here,
        # for the same reason the str/bytes/datetime/void gap above isn't:
        # real numpy independently raises TypeError for both, but via its
        # own unrelated internal coercion path (`float() argument must be a
        # string or a real number, not '...'`) -- an implementation detail
        # of numpy's, not a contract anionpy should mimic. anionpy's own message
        # is required to stay its stable, internal-detail-free canonical
        # rejection (`not_ndarray_like_err` in `ionp-py/src/lib.rs`)
        # regardless of what real numpy happens to say for its own doomed
        # coercion attempt on the same object, so the message text can
        # never match and a corpus entry here would be a permanent,
        # unfixable [FAIL] -- not a regression guard, just ledger noise
        # that demotes `array`/`asarray` out of "exact". This axis IS
        # verified, just not through the differential corpus: a direct
        # script probe (`anionpy.array(_TobytesWrongLength())`,
        # `anionpy.array(_TobytesNotCallable())`) confirms both raise anionpy's
        # canonical `TypeError: anionpy.array() only supports ...` and never
        # leak an internal dtype string or PyO3 downcast artifact -- see
        # this fix's task report (commit c5c2771) for the pasted RED/GREEN
        # transcript. Do not re-add these as registry cases without first
        # finding a comparison mechanism other than "numpy and anionpy must
        # produce byte-identical exception text" -- that mechanism cannot
        # express "deliberately different on purpose."
    ]


def _array_custom_cases():
    return [
        *_numpy_scalar_ingestion_cases(),
        ("list_1d_int", ([1, 2, 3, 4],), {}),
        ("list_1d_float", ([1.0, 2.5, -3.25],), {}),
        ("list_1d_bool", ([True, False, True],), {}),
        ("list_mixed_int_float_promotes", ([1, 2.5, 3],), {}),
        ("nested_list_2d", ([[1, 2], [3, 4]],), {}),
        ("nested_list_3d", ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],), {}),
        ("tuple_1d", ((1, 2, 3),), {}),
        ("tuple_of_tuples_2d", (((1, 2), (3, 4)),), {}),
        ("empty_list", ([],), {}),
        ("scalar_int", (5,), {}),
        ("scalar_float", (5.5,), {}),
        ("passthrough_ndarray", (np.arange(6.0).reshape(2, 3),), {}),
        ("list_with_dtype_kwarg", ([1, 2, 3],), {"dtype": np.float32}),
        ("list_with_dtype_string_kwarg", ([1, 2, 3],), {"dtype": "int16"}),
        ("ragged_list_must_raise_or_object",
         ([[1, 2], [3, 4, 5]],), {}),
        # --- 2026-08-01 bug: bool array built from raw non-0/1 bytes -------
        # (regression coverage for ionp-py/src/lib.rs's ndarray_from_numpy
        # bool-dispatch soundness fix -- see that function's docstring).
        # Every one of these is a byte pattern taken from, or in the same
        # family as, the deterministic repro that found the bug: a bool
        # array is any byte where numpy's rule is "zero is false, nonzero
        # is true", but the buggy code briefly read those bytes straight
        # through Rust's `bool` (valid only at 0x00/0x01), which is
        # undefined behavior for every other byte value and previously
        # produced everything from silently wrong values to a `ValueError`
        # about a too-short buffer.
        ("bool_raw_bytes_100_217_2_0",
         (_bool_from_raw_bytes([100, 217, 2, 0]),), {}),
        ("bool_raw_bytes_1_0_200_7",
         (_bool_from_raw_bytes([1, 0, 200, 7]),), {}),
        ("bool_raw_bytes_2_0_0_0",
         (_bool_from_raw_bytes([2, 0, 0, 0]),), {}),
        ("bool_raw_bytes_all_zero",
         (_bool_from_raw_bytes([0, 0, 0, 0]),), {}),
        ("bool_raw_bytes_255_1_1_1",
         (_bool_from_raw_bytes([255, 1, 1, 1]),), {}),
        ("bool_raw_bytes_full_byte_range_0_255",
         (_bool_from_raw_bytes(list(range(256))),), {}),
        ("bool_raw_bytes_2d_shape",
         (_bool_from_raw_bytes([100, 217, 2, 0, 255, 7, 0, 1], shape=(2, 4)),), {}),
        ("bool_raw_bytes_single_garbage_byte",
         (_bool_from_raw_bytes([42]),), {}),
        # --- 2026-08-01 bug: np.empty(dtype=bool)-style uninitialized
        # memory, made deterministic (not allocator-luck-dependent) by
        # explicitly poisoning the freshly allocated buffer with a fixed
        # non-0/1 pattern before handing it to anionpy.array().
        ("bool_empty_poisoned_mixed",
         (_bool_empty_poisoned(6, [5, 9, 13, 0, 1, 250]),), {}),
        ("bool_empty_poisoned_all_nonzero_non_one",
         (_bool_empty_poisoned(5, [3, 7, 11, 250, 128]),), {}),
        ("bool_empty_poisoned_single",
         (_bool_empty_poisoned(1, [77]),), {}),
        # --- ndmin= / copy= / order= / subok= (2026-08-01 gap-fix task) ---
        # ndmin: pads leading size-1 axes until ndim >= ndmin; NOT tested
        # at explicit ndmin=0 or above the input's own ndim minus a small
        # margin here beyond what's below -- see linalg.rs/lib.rs's `array`
        # doc comment for why the "omitted" sentinel default is not a
        # stable, Python-visible contract (deliberately out of scope,
        # reported in the task's final report, not silently skipped).
        ("ndmin_pads_1d_to_3d", ([1, 2, 3],), {"ndmin": 3}),
        ("ndmin_below_actual_ndim_noop", ([[1, 2], [3, 4]],), {"ndmin": 1}),
        ("ndmin_equal_to_actual_ndim_noop", ([[1, 2], [3, 4]],), {"ndmin": 2}),
        ("ndmin_on_scalar", (5,), {"ndmin": 2}),
        # copy=: tri-state. False on a fresh Python list/tuple/scalar
        # ALWAYS raises in real numpy too (a list is never already a
        # matching ndarray, so a copy is never avoidable) -- this is the
        # one source kind where anionpy's `array(..., copy=False)` behavior
        # is NOT architecturally divergent from real numpy (see this
        # item's Rust doc comment for the one case that IS: an existing
        # numpy.ndarray source, deliberately not tested here, see the
        # task's final report).
        ("copy_true_list", ([1, 2, 3],), {"copy": True}),
        ("copy_false_list_always_raises", ([1, 2, 3],), {"copy": False}),
        ("copy_none_list", ([1, 2, 3],), {"copy": None}),
        # order=: 'C'/'F' on a nested (2-D+) list -- both sides start from
        # the same freshly-built row-major default, so 'F' genuinely
        # forces a re-layout while still producing the same logical
        # values, exercising `array()`'s `to_contiguous_order` call.
        ("order_F_2d", ([[1, 2, 3], [4, 5, 6]],), {"order": "F"}),
        ("order_C_2d", ([[1, 2, 3], [4, 5, 6]],), {"order": "C"}),
        # subok=: deliberately NOT asserted to raise for non-bool values
        # here (see this item's `check_subok` doc comment in lib.rs: numpy
        # itself is behaviorally inconsistent about this across different
        # C functions -- an implementation artifact, not a documented
        # contract -- so anionpy accepts any value unconditionally). Only the
        # accept-and-no-op contract for the two DOCUMENTED bool values is
        # asserted here.
        ("subok_false", ([1, 2, 3],), {"subok": False}),
        ("subok_true", ([1, 2, 3],), {"subok": True}),
        ("ndmin_and_order_and_copy_combined", ([1, 2, 3],), {"ndmin": 2, "order": "F", "copy": True}),
        # --- 2026-08-02 bug: full uint64/int64 range through the Python
        # list/tuple/scalar constructor path (ionp-py/src/lib.rs's
        # `flatten_nested`, the PyInt leaf arm) -- found while investigating
        # the reported `anionpy.array([2**64-1], dtype='uint64')` OverflowError.
        # ROOT CAUSE (verified live, NOT fixed here -- `lib.rs` is owned by
        # another in-flight task per this task's file-ownership rules):
        # `flatten_nested`'s PyInt arm does `obj.extract::<i64>()? as f64`
        # unconditionally for every int leaf, then `ndarray_from_pylist`
        # always builds an I64 (or, for LeafKind::Float, F64) `Buffer`
        # regardless of the eventual `dtype=` target, and `array_impl`
        # finally reaches the requested dtype via a generic, UNCHECKED
        # `cast_to` narrowing -- no numpy-style bounds validation at all on
        # this path. Three distinct, independently verified real bugs fall
        # out of that:
        #   1. Any Python int outside `i64`'s range (uint64's whole upper
        #      half, e.g. `2**64-1` or `2**63`) fails the `extract::<i64>`
        #      immediately and raises `OverflowError: Python int too large
        #      to convert to C long` even for values a `uint64` target holds
        #      exactly -- this is the originally reported symptom, and it
        #      also reproduces for a bare scalar (`anionpy.array(2**64-1,
        #      dtype='uint64')`) and a nested list, not just a flat 1-elem
        #      list, since a bare/nested scalar reaches the identical
        #      `flatten_nested` PyInt arm.
        #   2. Silent PRECISION LOSS for in-range `int64`/`uint64` values
        #      that need more than `f64`'s 53-bit mantissa: verified
        #      `anionpy.array([2**62 + 5], dtype='int64')` -> `4611686018427387904`
        #      where real numpy (and Python) give the true value
        #      `4611686018427387909` -- the round-trip through `f64` in
        #      `flatten_nested` corrupts it even though `i64` extraction
        #      itself succeeded and the value is nowhere near overflowing.
        #   3. `array_impl`'s final `cast_to` narrowing silently WRAPS
        #      instead of raising `OverflowError` for an in-`i64`-range int
        #      that doesn't fit a NARROWER target dtype: verified
        #      `anionpy.array([256], dtype='uint8')` -> `[0]` and
        #      `anionpy.array([-1], dtype='uint8')` -> `[255]`, where real numpy
        #      raises `OverflowError: Python integer 256 out of bounds for
        #      uint8` / `... -1 out of bounds for uint8` respectively. This
        #      is the same missing-bounds-check defect as #1, just visible
        #      at the OTHER end (values that overflow the narrow target but
        #      not `i64` itself, so they never even reach the `extract::
        #      <i64>` failure that exposes #1).
        # The correct, ALREADY-WORKING routine for all of this
        # (`weak_int_overflow_check`/`check_int_bounds`/`int_buffer_from_i128`
        # in lib.rs, extracting as `i128` and bounds-checking against the
        # real target range with numpy's exact two-stage message rule) is
        # reachable today only via the bare-scalar `weak_scalar_buffer` path
        # (`anionpy.full`, `anionpy.uint64(...)`, `.astype(...)`) -- `array`/
        # `asarray`'s LIST path never calls it. Every case below is expected
        # to currently FAIL against anionpy; they exist to pin down the exact
        # gap (verified against real numpy 2.5.1 by running it live, not
        # from memory) for whoever picks up the `lib.rs` fix.
        ("uint64_high_half_list", ([2**64 - 1],), {"dtype": "uint64"}),
        ("uint64_boundary_2pow63_list", ([2**63],), {"dtype": "uint64"}),
        ("uint64_high_half_scalar", (2**64 - 1,), {"dtype": "uint64"}),
        ("uint64_high_half_nested_list", ([[2**64 - 1]],), {"dtype": "uint64"}),
        # int64 min/max: PASS today (by coincidence -- see #2 above, an
        # exact power of two/`f64`-representable-adjacent value survives the
        # f64 round-trip, or Rust's saturating `f64 as i64` cast happens to
        # land back on the right boundary) -- kept here as a regression
        # guard, NOT as evidence the underlying path is sound.
        ("int64_min_list", ([-(2**63)],), {"dtype": "int64"}),
        ("int64_max_list", ([2**63 - 1],), {"dtype": "int64"}),
        # Bug #2 above: an in-range int64 value that needs more than 53 bits
        # of mantissa precision to round-trip through `flatten_nested`'s f64
        # intermediate.
        ("int64_precision_loss_regression", ([2**62 + 5],), {"dtype": "int64"}),
        # Bug #1/#3 above: numpy's exact two-stage OverflowError message
        # rule (verified live) -- out-of-range-for-i64 always gets "Python
        # int too large to convert to C long"; in-range-for-i64-but-not-
        # target gets "Python integer {v} out of bounds for {dtype}".
        ("uint64_overflow_2pow64_list", ([2**64],), {"dtype": "uint64"}),
        ("uint64_overflow_2pow65_list", ([2**65],), {"dtype": "uint64"}),
        ("uint64_negative_out_of_bounds_list", ([-1],), {"dtype": "uint64"}),
        ("uint8_overflow_256_list", ([256],), {"dtype": "uint8"}),
        ("uint8_negative_out_of_bounds_list", ([-1],), {"dtype": "uint8"}),
        ("int64_overflow_2pow63_list", ([2**63],), {"dtype": "int64"}),
        ("int64_overflow_neg_2pow63_minus_1_list", ([-(2**63) - 1],), {"dtype": "int64"}),
        # --- #37 ("Add S/U as real dtypes") ---------------------------
        # A homogeneous str/bytes Python list/tuple construction path was
        # added to `array_impl` (`ndarray_from_pylist_str` in lib.rs),
        # building real `Buffer::S`/`Buffer::U` directly with ZERO numpy
        # involvement. It was verified byte-for-byte against real numpy
        # 2.5.1 (dtype, shape, values, ragged-error text, truncation,
        # decline-error text) via out-of-corpus manual probing from
        # `/private/tmp` -- see `docs/TICKET-37-2026-08-08.md`.
        #
        # These 15 cases were originally drafted here and then removed
        # BEFORE COMMIT in 5bb6fc9, because the harness's generic
        # value-comparator materializes a real numpy array from any
        # array-typed result via `PyArray::__array___no_cast`, and that
        # function had no `Buffer::S`/`Buffer::U` arm at all -- so an S/U
        # RESULT could not be graded, only constructed. That gap is #70,
        # closed by `__array___no_cast` gaining an S/U arm (see this file's
        # `docs/TICKET-70-2026-08-08.md`): the flat, already-NUL-padded
        # record bytes `Buffer::S`/`Buffer::U` already store are built into
        # a raw byte buffer entirely in Rust, then handed to
        # `numpy.frombuffer`/`.copy()`/`.reshape()` purely to assemble a
        # container from bytes this crate already finished computing --
        # the same category of numpy call the grandfathered
        # `strings.rs::encode_string_array` already makes, not a NEW
        # value/answer-computing call site. Restored here now that the
        # comparator can actually grade them (re-verified byte-for-byte
        # against real numpy 2.5.1 from `/private/tmp` before restoring,
        # not assumed identical to the original draft).
        ("su_str_flat_list", (['a', 'bb', 'ccc'],), {}),
        ("su_str_all_empty", (['', '', ''],), {}),
        ("su_str_single_empty", ([''],), {}),
        ("su_str_multibyte", (['héllo', '日本語'],), {}),
        ("su_str_astral", (['😀x', 'y'],), {}),
        ("su_str_tuple", (('a', 'bb', 'ccc'),), {}),
        ("su_str_nested_2d", ([['a', 'bb'], ['ccc', 'd']],), {}),
        ("su_str_ragged_must_raise", ([['a'], ['b', 'c']],), {}),
        ("su_str_dtype_truncate", (['hello'],), {"dtype": "U3"}),
        ("su_str_dtype_widen", (['hi'],), {"dtype": "U5"}),
        ("su_bytes_flat_list", ([b'a', b'bb', b'ccc'],), {}),
        ("su_bytes_single_empty", ([b''],), {}),
        ("su_bytes_all_empty", ([b'', b'', b''],), {}),
        # Embedded NUL bytes are legal S data and must survive -- a
        # C-string round trip would silently truncate at the first one;
        # this case is the direct regression guard for that.
        ("su_bytes_embedded_nul", ([b'\x00abc', b'ab\x00c'],), {"dtype": "S4"}),
        ("su_bytes_dtype_truncate", ([b'hello'],), {"dtype": "S3"}),
        # --- #70 width-0-means-infer guard -----------------------------
        # Regression guard for 0546775 (S/U width-0 silent data loss):
        # width-0 dtype spellings must mean "infer width from the data",
        # never "truncate every element to nothing". `dtype=str`/`bytes`
        # (bare type objects, not dtype strings) exercise the SAME
        # width-0-means-infer path through a different kwarg spelling --
        # see this task's report for why "vary the request, not just the
        # data" is the point of these cases, not the flat-list ones above.
        ("su_width0_dtype_U", (['a', 'bb'],), {"dtype": "U"}),
        ("su_width0_dtype_U0", (['a', 'bb'],), {"dtype": "U0"}),
        ("su_width0_dtype_U0_explicit_endian", (['a', 'bb'],), {"dtype": "<U0"}),
        ("su_width0_dtype_str_type", (['a', 'bb'],), {"dtype": str}),
        ("su_width0_dtype_S", ([b'a', b'bb'],), {"dtype": "S"}),
        ("su_width0_dtype_S0", ([b'a', b'bb'],), {"dtype": "S0"}),
        ("su_width0_dtype_S0_pipe", ([b'a', b'bb'],), {"dtype": "|S0"}),
        # The empty-string floor: even an ALL-empty-string input under an
        # explicit width-0 target must floor to itemsize 1, not 0 -- see
        # ticket #52 and 0546775.
        ("su_width0_empty_string_floor", ([''],), {"dtype": "U0"}),
        # Sized targets alongside the width-0 ones above, so a regression
        # that broke ONLY the zero-width branch (or ONLY the sized one)
        # would be caught by whichever half it actually broke.
        ("su_width0_sized_U1", (['abc', 'de'],), {"dtype": "U1"}),
        ("su_width0_sized_U3", (['abc', 'de'],), {"dtype": "U3"}),
        ("su_width0_sized_U5", (['abc', 'de'],), {"dtype": "U5"}),
        ("su_width0_sized_S1", ([b'abc', b'de'],), {"dtype": "S1"}),
        ("su_width0_sized_S3", ([b'abc', b'de'],), {"dtype": "S3"}),
        ("su_width0_sized_S5", ([b'abc', b'de'],), {"dtype": "S5"}),
    ]


REGISTRY["array"].custom_cases = _array_custom_cases

# Import-time self-check: see _require_varargs_form_if_numpy_supports_it's
# docstring. Deliberately runs unconditionally on import (not just under
# pytest) so `run.py` catches a registry regression too, not only pytest.
_require_varargs_form_if_numpy_supports_it(REGISTRY)

# ---------------------------------------------------------------------------
# Merge linalg_cases.py's LINALG_SPECS in, the same way ufunc_registry.py
# merges UFUNC_SPECS into this same REGISTRY dict (collision-checked, loud
# failure rather than a silent overwrite either direction). Done here
# (inside registry.py, which this task owns) rather than via an extra
# import line in run.py, since run.py is not a file this task owns/may
# edit -- `from registry import REGISTRY` picks up the post-merge dict
# either way, since REGISTRY is mutated in place, not rebound.
# ---------------------------------------------------------------------------
from linalg_cases import LINALG_SPECS  # noqa: E402

_linalg_collisions = set(LINALG_SPECS) & set(REGISTRY)
if _linalg_collisions:
    raise AssertionError(
        f"linalg_cases.py: {sorted(_linalg_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(LINALG_SPECS)

# ---------------------------------------------------------------------------
# Merge inplace_cases.py's INPLACE_SPECS in -- the seven `ndarray.__i*__`
# in-place dunders (__iadd__/__isub__/__imul__/__itruediv__/__iand__/__ior__/
# __ixor__), declared 2026-08-01. Same collision-checked pattern as the
# LINALG_SPECS merge immediately above (loud failure, not a silent
# overwrite); done here for the same reason -- run.py is not owned by this
# task and REGISTRY is mutated in place, so `from registry import REGISTRY`
# picks up the post-merge dict regardless of import order elsewhere.
# ---------------------------------------------------------------------------
from inplace_cases import INPLACE_SPECS  # noqa: E402

_inplace_collisions = set(INPLACE_SPECS) & set(REGISTRY)
if _inplace_collisions:
    raise AssertionError(
        f"inplace_cases.py: {sorted(_inplace_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(INPLACE_SPECS)

# ---------------------------------------------------------------------------
# Merge testing_cases.py's TESTING_SPECS in -- the `numpy.testing` block.
#
# This merge was written, reviewed and then NOT applied for most of 2026-08-01,
# because registry.py was owned by a concurrent agent every time the testing
# block was ready. Consequence: 47 completed, falsifiability-hardened items sat
# disconnected from REGISTRY, invisible to both the ledger and the
# falsifiability audit -- `tools/falsifiability.py --prefix testing.` matched
# zero items and (correctly) refused rather than reporting a clean bill.
#
# Recorded because it is the concrete cost of one-owner-per-file partitioning
# on a shared declaration file, and the second time today that finished work
# went uncredited for that reason -- see
# reports/ionp-throughput-bottleneck-2026-08-01.md.
#
# Same collision-checked pattern as the LINALG_SPECS/INPLACE_SPECS merges above.
# ---------------------------------------------------------------------------
from testing_cases import TESTING_SPECS  # noqa: E402

_testing_collisions = set(TESTING_SPECS) & set(REGISTRY)
if _testing_collisions:
    raise AssertionError(
        f"testing_cases.py: {sorted(_testing_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(TESTING_SPECS)

# ---------------------------------------------------------------------------
# Merge creation_cases.py's CREATION_SPECS in -- the top-level array-creation
# + shape-manipulation block (zeros/ones/empty/full [+ _like], arange/
# linspace/eye/identity, asarray/copy/ascontiguousarray, and the top-level
# view wrappers reshape/ravel/transpose/swapaxes/moveaxis/squeeze/
# expand_dims/broadcast_to). Same collision-checked pattern as the merges
# above; see creation_cases.py's module docstring for the two real bugs
# found (and one fixed) while building this block's specs.
# ---------------------------------------------------------------------------
from creation_cases import CREATION_SPECS  # noqa: E402

_creation_collisions = set(CREATION_SPECS) & set(REGISTRY)
if _creation_collisions:
    raise AssertionError(
        f"creation_cases.py: {sorted(_creation_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(CREATION_SPECS)

# `asarray` shares `array`'s numpy-scalar-ingestion bug (both funnel a
# real-numpy-sourced object through the same `array_impl`/
# `ndarray_from_numpy` in lib.rs -- see `_numpy_scalar_ingestion_cases`'s
# doc comment above). `asarray`'s own case list lives in creation_cases.py
# (`asarray_cases`, merged in as part of `CREATION_SPECS` just above) --
# rather than duplicating that whole function here just to add a few more
# entries, wrap the callable creation_cases.py already registered and
# append the same numpy-scalar cases `array` gets, the same pattern
# `REGISTRY["array"].custom_cases = _array_custom_cases` above uses to
# attach a real case list to a `kind="custom"` item.
_base_asarray_custom_cases = REGISTRY["asarray"].custom_cases


def _asarray_custom_cases_with_numpy_scalars():
    return [*_base_asarray_custom_cases(), *_numpy_scalar_ingestion_cases()]


REGISTRY["asarray"].custom_cases = _asarray_custom_cases_with_numpy_scalars

# ---------------------------------------------------------------------------
# Merge ndarray_attrs_cases.py's NDARRAY_ATTRS_SPECS in -- the `ndarray`
# metadata-getter + method block backed by ionp-py/src/ndarray_attrs.rs
# (itemsize/nbytes/mT, ravel/flatten/squeeze/swapaxes, item/tolist,
# all/any/prod, conj/conjugate). Same collision-checked pattern as the
# merges above; see ndarray_attrs_cases.py's module docstring for the
# AxisError-vs-IndexError exception-type gap it scopes around and the three
# items (sum/min/max) it deliberately drops after measuring real failures
# against them (ULP drift from summation order, and two genuine
# ionp-core/src/ufunc.rs bugs in the Minimum/Maximum reduce path).
# ---------------------------------------------------------------------------
from ndarray_attrs_cases import NDARRAY_ATTRS_SPECS  # noqa: E402

_ndarray_attrs_collisions = set(NDARRAY_ATTRS_SPECS) & set(REGISTRY)
if _ndarray_attrs_collisions:
    raise AssertionError(
        f"ndarray_attrs_cases.py: {sorted(_ndarray_attrs_collisions)} already "
        f"present in registry.REGISTRY -- refusing to silently overwrite an "
        f"existing item"
    )
REGISTRY.update(NDARRAY_ATTRS_SPECS)

# ---------------------------------------------------------------------------
# Merge ndarray_float_cases.py's NDARRAY_FLOAT_SPECS in -- `ndarray.__float__`
# (`float(arr)`), 2026-08-03. Had zero coverage anywhere in this tree before
# this task (confirmed by grep); a fresh file+kind="custom" ItemSpec because
# it doesn't fit unary/binary_op/method (see that file's module docstring).
# Same collision-checked merge pattern as the merges above.
# ---------------------------------------------------------------------------
from ndarray_float_cases import NDARRAY_FLOAT_SPECS  # noqa: E402

_ndarray_float_collisions = set(NDARRAY_FLOAT_SPECS) & set(REGISTRY)
if _ndarray_float_collisions:
    raise AssertionError(
        f"ndarray_float_cases.py: {sorted(_ndarray_float_collisions)} already "
        f"present in registry.REGISTRY -- refusing to silently overwrite an "
        f"existing item"
    )
REGISTRY.update(NDARRAY_FLOAT_SPECS)

# ---------------------------------------------------------------------------
# Merge order_cases.py's ORDER_SPECS in -- dedicated order=('C'|'F'|'A'|'K')
# x dtype x memory-layout differential coverage for ravel/flatten/copy/
# astype/zeros_like/ones_like/full_like/reshape, added for the 2026-08-01
# order-semantics bug-fix task (order='A' silently wrong after numpy
# ingestion, order='K' rejected everywhere it should be accepted). Registered
# under fresh "order/..." keys (see that file's module docstring for why:
# the real item names it exercises -- "ndarray.ravel", "zeros_like", etc --
# already exist in this registry, so it points numpy_path/ionp_path at them
# instead of colliding on the dict key). Same collision-checked merge
# pattern as every block above.
# ---------------------------------------------------------------------------
from order_cases import ORDER_SPECS  # noqa: E402

_order_collisions = set(ORDER_SPECS) & set(REGISTRY)
if _order_collisions:
    raise AssertionError(
        f"order_cases.py: {sorted(_order_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(ORDER_SPECS)

# ---------------------------------------------------------------------------
# Merge reduction_cases.py's REDUCTION_SPECS in -- the top-level numpy
# reduction/statistics FUNCTION block (sum/prod/all/any/min/max/amin/amax,
# argmin/argmax, cumsum/cumprod, mean, ptp, count_nonzero) backed by
# ionp-py/src/reductions.rs, as distinct from the `ndarray` METHOD versions
# of some of the same names already covered by NDARRAY_ATTRS_SPECS above.
# This is also where the stale "sum" placeholder that used to sit directly
# in REGISTRY's dict body (see the comment at the top of that dict) is
# actually replaced -- REGISTRY["sum"] did not exist between that removal
# and this merge, and does now. Same collision-checked pattern as every
# merge above; see reduction_cases.py's module docstring for the three
# measured findings (sum/mean's partial pairwise-summation gap, the
# AxisError-vs-IndexError gap on argmin/argmax/cumsum/cumprod's axis
# validation, and the float16 accumulator gap deliberately surfaced rather
# than fixed here).
# ---------------------------------------------------------------------------
from reduction_cases import REDUCTION_SPECS  # noqa: E402

_reduction_collisions = set(REDUCTION_SPECS) & set(REGISTRY)
if _reduction_collisions:
    raise AssertionError(
        f"reduction_cases.py: {sorted(_reduction_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(REDUCTION_SPECS)

# ---------------------------------------------------------------------------
# Merge sort_cases.py's SORT_SPECS in -- the SORTING AND SEARCHING block
# (sort, sort_complex, lexsort, nonzero, flatnonzero, argwhere, extract,
# where (+ its 1-arg nonzero-alias form), searchsorted, nanargmax,
# nanargmin, and the ndarray.sort/argmax/argmin/nonzero/searchsorted method
# forms). argmin/argmax's own differential coverage already lives in
# reduction_cases.py's REDUCTION_SPECS (merged above) and is NOT duplicated
# here. Same collision-checked merge pattern as every block above; see
# sort_cases.py's module docstring for the bugs found+fixed this task
# (sort_complex's dtype-promotion table, the do_argext/do_nanargext 0-d+
# keepdims reshape asymmetry, and the nanargmax/nanargmin NaN-vs-inf
# tie-breaking algorithm).
# ---------------------------------------------------------------------------
from sort_cases import SORT_SPECS  # noqa: E402

_sort_collisions = set(SORT_SPECS) & set(REGISTRY)
if _sort_collisions:
    raise AssertionError(
        f"sort_cases.py: {sorted(_sort_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(SORT_SPECS)

# ---------------------------------------------------------------------------
# Merge manip_cases.py's MANIP_SPECS in -- the array-manipulation block
# (concatenate/stack/hstack/vstack/row_stack/dstack/column_stack,
# flip/fliplr/flipud, roll, tile, repeat, broadcast_shapes/
# broadcast_arrays, atleast_1d/2d/3d, diag/diagflat/diagonal/tril/triu/
# trace), backed by ionp-core/src/manip.rs + ionp-py/src/manip.rs. See
# manip_cases.py's module docstring for the deliberate ~24-of-~40-item
# scope decision and the atleast_1d/2d/3d multi-array-tuple-return
# harness-architecture caveat. Same collision-checked merge pattern as
# every block above.
# ---------------------------------------------------------------------------
from manip_cases import MANIP_SPECS  # noqa: E402

_manip_collisions = set(MANIP_SPECS) & set(REGISTRY)
if _manip_collisions:
    raise AssertionError(
        f"manip_cases.py: {sorted(_manip_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(MANIP_SPECS)
# Merge setops_cases.py's SETOPS_SPECS in -- the SET OPERATIONS / DIFF-
# UNIQUE / SORTING-REMAINDER block (unique, unique_counts, unique_inverse,
# unique_all, diff, ediff1d, trim_zeros, intersect1d, union1d, setdiff1d,
# setxor1d, isin). `unique_values` is deliberately NOT declared here -- see
# setops_cases.py's module docstring (numpy's own unspecified hash-order
# for `sorted=False`, verified via `inspect.getsource(np.unique_values)`).
# Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from setops_cases import SETOPS_SPECS  # noqa: E402

_setops_collisions = set(SETOPS_SPECS) & set(REGISTRY)
if _setops_collisions:
    raise AssertionError(
        f"setops_cases.py: {sorted(_setops_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(SETOPS_SPECS)

# ---------------------------------------------------------------------------
# Merge emath_cases.py's EMATH_SPECS in -- the `numpy.emath` block
# (sqrt/log/log2/log10/logn/power/arccos/arcsin/arctanh: the branch-cut-
# aware variants that promote real input to complex instead of returning
# NaN), backed by ionp-core/src/emath.rs + ionp-py/src/emath.rs. See
# emath_cases.py's module docstring for why 7 of the 9 items need no
# custom cases at all (the existing unary_corpus() already covers every
# branch-cut boundary these items trigger on). Same collision-checked
# merge pattern as every block above.
# ---------------------------------------------------------------------------
from emath_cases import EMATH_SPECS  # noqa: E402

_emath_collisions = set(EMATH_SPECS) & set(REGISTRY)
if _emath_collisions:
    raise AssertionError(
        f"emath_cases.py: {sorted(_emath_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(EMATH_SPECS)

# ---------------------------------------------------------------------------
# Merge fft_cases.py's FFT_SPECS in -- the `numpy.fft` block (fft/ifft/
# rfft/irfft/hfft/ihfft/fftn/ifftn/fft2/ifft2/rfftn/rfft2/irfftn/irfft2/
# fftshift/ifftshift/fftfreq/rfftfreq), backed by ionp-core/src/fft.rs +
# ionp-py/src/fft.rs. See fft_cases.py's module docstring for the measured
# per-dtype epsilon_tolerance evidence (rustfft vs pocketfft non-
# associativity), the loud rustfft-crate-reuse disclosure, and three
# genuine correctness bugs found and fixed while building this block
# (fftfreq/rfftfreq ZeroDivisionError, fft-namespace-specific axis-out-of-
# range IndexError messages, rfftfreq's negative-n empty-array-not-raise
# divergence from fftfreq). Same collision-checked merge pattern as every
# block above.
# ---------------------------------------------------------------------------
from fft_cases import FFT_SPECS  # noqa: E402

_fft_collisions = set(FFT_SPECS) & set(REGISTRY)
if _fft_collisions:
    raise AssertionError(
        f"fft_cases.py: {sorted(_fft_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(FFT_SPECS)

# ---------------------------------------------------------------------------
# Merge lib_cases.py's LIB_SPECS in -- `lib.NumpyVersion`, `lib.Arrayterator`,
# `ctypeslib.load_library` (see lib_cases.py's module docstring for why only
# these three of the twelve `lib.*`/`ctypeslib.*` items are registered; the
# rest are blocked or out of scope, documented in this task's report rather
# than faked here). Same collision-checked merge pattern as every block
# above.
# ---------------------------------------------------------------------------
from lib_cases import LIB_SPECS  # noqa: E402

_lib_collisions = set(LIB_SPECS) & set(REGISTRY)
if _lib_collisions:
    raise AssertionError(
        f"lib_cases.py: {sorted(_lib_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(LIB_SPECS)

# ---------------------------------------------------------------------------
# Merge stats_cases.py's STATS_SPECS in -- `median`, `nanmedian`,
# `percentile`, `quantile`, `nanpercentile`, `nanquantile`, backed by
# ionp-core/src/stats.rs + ionp-py/src/stats.rs. See stats_cases.py's module
# docstring for the two measured (and deliberately excluded, not hidden)
# narrow-float dtype-preservation gaps -- weak-scalar-`q` NEP 50 promotion,
# and the analogous all-NaN-slice short-circuit in `nanquantile`/
# `nanpercentile` -- plus the multi-dimensional-`q` and `weights=` gaps.
# Every INCLUDED call-form is bit-exact (atol=0.0, rtol=0.0), verified by a
# 129,320-call out-of-corpus probe (tests/differential/_probe_stats.py)
# showing 0 value mismatches and 0 error-type/message mismatches once the
# excluded forms are set aside. Same collision-checked merge pattern as
# every block above.
# ---------------------------------------------------------------------------
from stats_cases import STATS_SPECS  # noqa: E402

_stats_collisions = set(STATS_SPECS) & set(REGISTRY)
if _stats_collisions:
    raise AssertionError(
        f"stats_cases.py: {sorted(_stats_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(STATS_SPECS)

# ---------------------------------------------------------------------------
# Merge random_cases.py's RANDOM_SPECS in -- `random.SeedSequence`,
# `random.PCG64`, `random.PCG64DXSM`, `random.Generator`, `random.default_rng`.
# All bit-exact seeded cases (see random_cases.py's module docstring for why
# none of the 5 are declared "exact" in anionpy/_state/random.py despite every
# case here passing: ledger granularity for these items is whole-class, and
# each class implements only a documented subset of numpy's real method
# surface). Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from random_cases import RANDOM_SPECS  # noqa: E402

_random_collisions = set(RANDOM_SPECS) & set(REGISTRY)
if _random_collisions:
    raise AssertionError(
        f"random_cases.py: {sorted(_random_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(RANDOM_SPECS)
# Merge scalar_cases.py's SCALAR_SPECS in -- the scalar-type-hierarchy +
# module-constants block (bool_/int8../uint64/float16../complex128, the 10
# abstract bases, the 17 same-object aliases, and nan/inf/pi/e/euler_gamma/
# newaxis/little_endian/True_/False_), backed by ionp-py/src/scalars.rs. See
# scalar_cases.py's module docstring for the derived weak/strong
# construction rule, the `kind="custom"` + adapter pattern used throughout
# (mirroring lib_cases.py), and the two disclosed, permanent gaps
# (float64/complex128 cannot also subclass Python's builtin float/complex;
# every scalar constructor rejects list/tuple input rather than silently
# building an array). Same collision-checked merge pattern as every block
# above.
# ---------------------------------------------------------------------------
from scalar_cases import SCALAR_SPECS  # noqa: E402

_scalar_collisions = set(SCALAR_SPECS) & set(REGISTRY)
if _scalar_collisions:
    raise AssertionError(
        f"scalar_cases.py: {sorted(_scalar_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(SCALAR_SPECS)
# Merge dtypeinfo_cases.py's DTYPEINFO_SPECS in -- the dtype introspection/
# promotion toplevel block (`can_cast`, `promote_types`, `result_type`,
# `min_scalar_type`, `iinfo`, `finfo`, `typename`, `mintypecode`,
# `isdtype`), backed by `ionp-py/src/dtypeinfo.rs` +
# `ionp-core/src/dtype.rs`. Only `typename`/`mintypecode` are declared
# "exact" in toplevel.py -- the rest are registered here anyway (including
# cases that are EXPECTED to fail) so their real, permanent gaps stay
# continuously visible instead of being hidden by omission; see
# dtypeinfo_cases.py's module docstring and toplevel.py's per-item
# non-declaration comments for exactly which call forms diverge and why.
# Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from dtypeinfo_cases import DTYPEINFO_SPECS  # noqa: E402

_dtypeinfo_collisions = set(DTYPEINFO_SPECS) & set(REGISTRY)
if _dtypeinfo_collisions:
    raise AssertionError(
        f"dtypeinfo_cases.py: {sorted(_dtypeinfo_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(DTYPEINFO_SPECS)
# Merge predicate_cases.py's PREDICATE_SPECS in -- the comparison/predicate/
# introspection toplevel block (`ndim`, `shape`, `size`, `isscalar`,
# `iterable`, `isfortran`, `iscomplexobj`, `isrealobj`, `array_equal`,
# `array_equiv`, `allclose`), all pure-Python wrappers in `anionpy/__init__.py`
# with no Rust counterpart of their own. See predicate_cases.py's module
# docstring for exactly which sibling items in this family (`isclose`,
# `iscomplex`, `isreal`, `isneginf`, `isposinf`, the six comparison ufuncs)
# are deliberately NOT registered here and why. Same collision-checked merge
# pattern as every block above.
# ---------------------------------------------------------------------------
from predicate_cases import PREDICATE_SPECS  # noqa: E402

_predicate_collisions = set(PREDICATE_SPECS) & set(REGISTRY)
if _predicate_collisions:
    raise AssertionError(
        f"predicate_cases.py: {sorted(_predicate_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(PREDICATE_SPECS)

# ---------------------------------------------------------------------------
# Merge window_cases.py's WINDOW_SPECS in -- the window-function / near-free
# toplevel math block (`bartlett`, `blackman`, `hamming`, `hanning`,
# `kaiser`, `i0`, `angle`, `sinc`, `unwrap`), all pure-Python compositions
# of existing `_anionpy` primitives in `anionpy/_window_math.py` -- no new Rust.
# See window_cases.py's module docstring and anionpy/_window_math.py's module
# docstring for the out-of-corpus bit-exactness evidence. Same
# collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from window_cases import WINDOW_SPECS  # noqa: E402

_window_collisions = set(WINDOW_SPECS) & set(REGISTRY)
if _window_collisions:
    raise AssertionError(
        f"window_cases.py: {sorted(_window_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(WINDOW_SPECS)

# ---------------------------------------------------------------------------
# Merge manip_compose_cases.py's MANIP_COMPOSE_SPECS in -- `asanyarray`,
# module-level `astype`, `unstack`: array-API-alignment wrappers over
# existing anionpy primitives (`asarray`, `ndarray.astype`, `moveaxis`) in
# `anionpy/_manip_compose.py`, no new Rust. `unstack` is registered
# `multi_output=True` (returns a tuple natively, no adapter needed). See
# manip_compose_cases.py's and _manip_compose.py's module docstrings for the
# deliberately narrowed scope on each (order=/copy=False left unverified --
# the open identity/layout-contract question) and toplevel.py for the
# differential evidence behind each declaration. Same collision-checked
# merge pattern as every block above.
# ---------------------------------------------------------------------------
from manip_compose_cases import MANIP_COMPOSE_SPECS  # noqa: E402

_manip_compose_collisions = set(MANIP_COMPOSE_SPECS) & set(REGISTRY)
if _manip_compose_collisions:
    raise AssertionError(
        f"manip_compose_cases.py: {sorted(_manip_compose_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(MANIP_COMPOSE_SPECS)

# ---------------------------------------------------------------------------
# Merge compare_compose_cases.py's COMPARE_COMPOSE_SPECS in -- `isclose`: a
# Python-only wrapper over existing anionpy primitives (`asarray`, `multiply`,
# `subtract`, `absolute`, `add`, `less_equal`, `equal`, `isfinite`, `isnan`,
# `logical_and`, `logical_or`) in `anionpy/_compare_compose.py`, no new Rust.
# `numpy_path`/`ionp_path` both default to `spec.name` ("isclose"), resolving
# to `np.isclose`/`anionpy.isclose` respectively. See compare_compose_cases.py's
# and _compare_compose.py's module docstrings for the one deliberate
# substitution (multiply-by-1.0 standing in for the known-broken
# result_type/promote_types/can_cast trio) and toplevel.py for the
# differential evidence behind the declaration. Same collision-checked merge
# pattern as every block above.
# ---------------------------------------------------------------------------
from compare_compose_cases import COMPARE_COMPOSE_SPECS  # noqa: E402

_compare_compose_collisions = set(COMPARE_COMPOSE_SPECS) & set(REGISTRY)
if _compare_compose_collisions:
    raise AssertionError(
        f"compare_compose_cases.py: {sorted(_compare_compose_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(COMPARE_COMPOSE_SPECS)

# ---------------------------------------------------------------------------
# Merge arrayapi_cases.py's ARRAYAPI_SPECS in -- `permute_dims`, `clip`,
# `cumulative_sum`, `cumulative_prod`, `tri`: Python-only compositions in
# `anionpy/__init__.py` over existing Rust primitives (`transpose`,
# `ndarray.clip`, `cumsum`, `cumprod`, `arange`, `greater_equal`, `astype`,
# `concatenate`, `full_like`), no new Rust. See arrayapi_cases.py's module
# docstring for what is deliberately NOT exercised (`out=` on any of them,
# `tri(like=)`) and `__init__.py`'s block comment for the three siblings
# from the same numpy source files that are deliberately NOT implemented
# (`geomspace` -- needed __setitem__, which landed 2026-08-02 and was
# declared 2026-08-03, so this one is UNBLOCKED and merely not-yet-done;
# `logspace` -- built on a failing
# `power`; `concat` -- an alias of the undeclared `concatenate`). Same
# collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from arrayapi_cases import ARRAYAPI_SPECS  # noqa: E402

_arrayapi_collisions = set(ARRAYAPI_SPECS) & set(REGISTRY)
if _arrayapi_collisions:
    raise AssertionError(
        f"arrayapi_cases.py: {sorted(_arrayapi_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(ARRAYAPI_SPECS)

# ---------------------------------------------------------------------------
# Merge round_cases.py's ROUND_SPECS in -- `round`, `around`,
# `ndarray.round`: a Python driver (`anionpy/_round_compose.py`) transcribing
# numpy's C driver `PyArray_Round`, over anionpy's own already-verified
# `multiply`/`divide`/`rint` ufuncs, plus three private Rust primitives in
# `ionp-core/src/round.rs` (`PyArray_CopyInto`, the `.real=`/`.imag=`
# setters, and `power_of_ten`) for the three things no ufunc call can
# express. See round_cases.py's module docstring for the branch order that
# turned out to be observable, and for the two siblings from the same numpy
# source that are deliberately NOT implemented (`fix` -- needs a
# DeprecationWarning channel anionpy does not have, and diverges on bool and
# complex anyway; `copyto` -- `_copy_into` is only its private half, its
# `casting=`/`where=` surface is unmeasured). Same collision-checked merge
# pattern as every block above.
# ---------------------------------------------------------------------------
from round_cases import ROUND_SPECS  # noqa: E402

_round_collisions = set(ROUND_SPECS) & set(REGISTRY)
if _round_collisions:
    raise AssertionError(
        f"round_cases.py: {sorted(_round_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(ROUND_SPECS)

# ---------------------------------------------------------------------------
# Merge typecheck_cases.py's TYPECHECK_SPECS in -- `real_if_close`: a
# Python-only transcription of numpy's OWN Python implementation
# (numpy/lib/_type_check_impl.py) over existing anionpy primitives
# (`asanyarray`, `multiply`, `absolute`, `less`, `all`, `ndarray.real`,
# `ndarray.imag`) in `anionpy/_typecheck_compose.py`, no new Rust.
# `numpy_path`/`ionp_path` both default to `spec.name` ("real_if_close"),
# resolving to `np.real_if_close`/`anionpy.real_if_close` respectively. See
# typecheck_cases.py's module docstring for what the corpus deliberately
# targets (the eps*tol threshold seam, the eps-DTYPE promotion axis, the
# strict `>` gate, inherited bad-`tol` messages, array-valued `tol` through
# ndarray.__bool__, and the branch-ORDER fact that non-complex input returns
# BEFORE `tol` is ever looked at) and _typecheck_compose.py's for why
# anionpy.finfo is deliberately not used. Same collision-checked merge pattern
# as every block above.
# ---------------------------------------------------------------------------
from typecheck_cases import TYPECHECK_SPECS  # noqa: E402

_typecheck_collisions = set(TYPECHECK_SPECS) & set(REGISTRY)
if _typecheck_collisions:
    raise AssertionError(
        f"typecheck_cases.py: {sorted(_typecheck_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(TYPECHECK_SPECS)

# ---------------------------------------------------------------------------
# Merge shape_compose_cases.py's SHAPE_COMPOSE_SPECS in -- `kron`: a
# Python-only transcription of numpy's OWN Python implementation
# (numpy/lib/_shape_base_impl.py) over existing anionpy primitives
# (`asanyarray`, `array`, `reshape`, `expand_dims`, `multiply`) in
# `anionpy/_shape_compose.py`, no new Rust and no accumulation of any kind --
# every output element is exactly one a[i]*b[j] product, which is why
# bit-exactness is free here. `numpy_path`/`ionp_path` both default to
# `spec.name` ("kron"), resolving to `np.kron`/`anionpy.kron`. See
# shape_compose_cases.py's module docstring for why kind="custom" and not
# "binary" (the generic binary corpus broadcasts its operands against each
# other, which is exactly what kron does NOT do) and for the axes the corpus
# targets -- above all the rank ASYMMETRY (`ndmin=b.ndim` promotes `a` to
# `b`'s rank but never the reverse) and the odd/even expand_dims interleave,
# which a corpus of square symmetric operands would grade as a pass while
# swapped. Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from shape_compose_cases import SHAPE_COMPOSE_SPECS  # noqa: E402

_shape_compose_collisions = set(SHAPE_COMPOSE_SPECS) & set(REGISTRY)
if _shape_compose_collisions:
    raise AssertionError(
        f"shape_compose_cases.py: {sorted(_shape_compose_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(SHAPE_COMPOSE_SPECS)

# ---------------------------------------------------------------------------
# Merge exploded_class_cases.py's EXPLODED_CLASS_SPECS in -- Phase 2b: the
# generic curated-exploded-class dispatch this file's `_exploded_prefix_of`/
# `_EXPLODED_RECEIVER_CONVERTERS`/`_ionp_exploded_class` machinery (above)
# makes resolvable, for the 5 present-on-ionp classes OTHER than `ndarray`
# (which was already fully wired up and is untouched): `dtype.<attr>`,
# `finfo.<attr>`, `iinfo.<attr>`, `ma.MaskedArray.<method>`,
# `random.Generator.<method>`. See exploded_class_cases.py's module
# docstring for the exact per-class item list, what is deliberately
# excluded and why (an all-masked MaskedArray fixture; `cumsum`/`cumprod`;
# `data`/`mask`), and the measured `Generator.integers()` return-type
# finding. Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from exploded_class_cases import EXPLODED_CLASS_SPECS  # noqa: E402

_exploded_class_collisions = set(EXPLODED_CLASS_SPECS) & set(REGISTRY)
if _exploded_class_collisions:
    raise AssertionError(
        f"exploded_class_cases.py: {sorted(_exploded_class_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(EXPLODED_CLASS_SPECS)

# ---------------------------------------------------------------------------
# Merge polynomial_cases.py's POLYNOMIAL_SPECS in: the power-series basis
# (`numpy.polynomial.polynomial`) vertical slice -- 21 items, all
# kind="custom". See polynomial_cases.py's module docstring for corpus
# design and anionpy/polynomial/polynomial.py's module docstring for the
# exact, deliberate scope boundary (N-D composition helpers and the
# Polynomial/ABCPolyBase class hierarchy are not attempted). Same
# collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from polynomial_cases import POLYNOMIAL_SPECS  # noqa: E402

_polynomial_collisions = set(POLYNOMIAL_SPECS) & set(REGISTRY)
if _polynomial_collisions:
    raise AssertionError(
        f"polynomial_cases.py: {sorted(_polynomial_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(POLYNOMIAL_SPECS)

# ---------------------------------------------------------------------------
# Merge legendre_cases.py's LEGENDRE_SPECS in: the Legendre-series basis
# (`numpy.polynomial.legendre`) vertical slice -- 24 items, all
# kind="custom". See legendre_cases.py's module docstring for corpus design
# and anionpy/polynomial/legendre.py's module docstring for the exact,
# deliberate scope boundary (`Legendre(ABCPolyBase)` and the `*2d`/`*3d`/
# `*nd` composition helpers are not attempted). Same collision-checked merge
# pattern as every block above.
# ---------------------------------------------------------------------------
from legendre_cases import LEGENDRE_SPECS  # noqa: E402

_legendre_collisions = set(LEGENDRE_SPECS) & set(REGISTRY)
if _legendre_collisions:
    raise AssertionError(
        f"legendre_cases.py: {sorted(_legendre_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(LEGENDRE_SPECS)

# ---------------------------------------------------------------------------
# Merge laguerre_cases.py's LAGUERRE_SPECS in: the Laguerre-series basis
# (`numpy.polynomial.laguerre`) vertical slice, all kind="custom". See
# laguerre_cases.py's module docstring for corpus design (in particular the
# "looks like Legendre and is not" traps it pins explicitly) and
# anionpy/polynomial/laguerre.py's module docstring for the exact scope
# boundary. Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from laguerre_cases import LAGUERRE_SPECS  # noqa: E402

_laguerre_collisions = set(LAGUERRE_SPECS) & set(REGISTRY)
if _laguerre_collisions:
    raise AssertionError(
        f"laguerre_cases.py: {sorted(_laguerre_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(LAGUERRE_SPECS)

# ---------------------------------------------------------------------------
# Hermite (physicists') basis (`numpy.polynomial.hermite`), function-level
# only. See tests/differential/hermite_cases.py's module docstring for
# corpus design (in particular the herm2poly n==2 doubling trap and the
# hermder/hermint direct-assignment probes it pins explicitly) and
# anionpy/polynomial/hermite.py's module docstring for the exact scope
# boundary. Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from hermite_cases import HERMITE_SPECS  # noqa: E402

_hermite_collisions = set(HERMITE_SPECS) & set(REGISTRY)
if _hermite_collisions:
    raise AssertionError(
        f"hermite_cases.py: {sorted(_hermite_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(HERMITE_SPECS)

# ---------------------------------------------------------------------------
# HermiteE (probabilists') basis (`numpy.polynomial.hermite_e`),
# function-level only. See tests/differential/hermite_e_cases.py's module
# docstring for corpus design (in particular the herme2poly n==2
# no-doubling trap, the hermecompanion/hermeroots -0.5-free len==2
# branches, and the hermeroots always-complex-dtype-for-degree>=3 probe it
# pins explicitly) and anionpy/polynomial/hermite_e.py's module docstring
# for the exact scope boundary. Same collision-checked merge pattern as
# every block above.
# ---------------------------------------------------------------------------
from hermite_e_cases import HERMITE_E_SPECS  # noqa: E402

_hermite_e_collisions = set(HERMITE_E_SPECS) & set(REGISTRY)
if _hermite_e_collisions:
    raise AssertionError(
        f"hermite_e_cases.py: {sorted(_hermite_e_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(HERMITE_E_SPECS)

# ---------------------------------------------------------------------------
# Chebyshev (first-kind) basis (`numpy.polynomial.chebyshev`),
# function-level only. See tests/differential/chebyshev_cases.py's module
# docstring for corpus design (in particular the z-series-vs-direct-
# recurrence split, the chebval no-coefficient-on-c1 Clenshaw shape, and the
# chebmul/chebpow np.convolve non-bit-exactness finding it documents in
# detail) and anionpy/polynomial/chebyshev.py's module docstring for the
# exact scope boundary. Same collision-checked merge pattern as every block
# above.
# ---------------------------------------------------------------------------
from chebyshev_cases import CHEBYSHEV_SPECS  # noqa: E402

_chebyshev_collisions = set(CHEBYSHEV_SPECS) & set(REGISTRY)
if _chebyshev_collisions:
    raise AssertionError(
        f"chebyshev_cases.py: {sorted(_chebyshev_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(CHEBYSHEV_SPECS)

# ---------------------------------------------------------------------------
# The five new `ABCPolyBase` subclasses (`Legendre`, `Laguerre`, `Hermite`,
# `HermiteE`, `Chebyshev`), class-level items only (function-level items for
# each basis are already merged above via their own `*_cases.py`). See
# tests/differential/abc_poly_class_cases.py's module docstring for the
# full per-class scope (in particular: `convert`/`cast` undeclared for all
# five due to the shared `_compose_affine` bug; `__mul__`/`__rmul__`/
# `__pow__`/`fromroots` undeclared for Chebyshev only; Legendre's own
# complex128 `__mul__`/`__rmul__`/`__pow__`/`fromroots` epsilon-toleranced,
# the other three classes' fully bit-exact). Same collision-checked merge
# pattern as every block above.
# ---------------------------------------------------------------------------
from abc_poly_class_cases import ABC_POLY_CLASS_SPECS  # noqa: E402

_abc_poly_class_collisions = set(ABC_POLY_CLASS_SPECS) & set(REGISTRY)
if _abc_poly_class_collisions:
    raise AssertionError(
        f"abc_poly_class_cases.py: {sorted(_abc_poly_class_collisions)} "
        f"already present in registry.REGISTRY -- refusing to silently "
        f"overwrite an existing item"
    )
REGISTRY.update(ABC_POLY_CLASS_SPECS)

# ---------------------------------------------------------------------------
# generic_poly_dunder_cases.py's GENERIC_POLY_DUNDER_SPECS: the 17 generic
# `object`/`abc.ABC`-mechanics dunder names ticket #34 measured as ALREADY
# MATCHING real numpy (`__lt__`/`__le__`/`__gt__`/`__ge__`,
# `__setattr__`/`__delattr__`, `__new__`, `__dir__`, `__reduce__`/
# `__reduce_ex__`, `__subclasshook__`, `__getattribute__`, `__sizeof__`,
# `__weakref__`, `__slots__`/`__abstractmethods__`/`__static_attributes__`)
# plus `__init_subclass__`, across all SIX basis classes -- see
# tests/differential/generic_poly_dunder_cases.py's module docstring for
# the full per-item reasoning and the one measured, disclosed, NOT-declared
# exclusion (ordering dunders vs a raw ndarray operand). Same collision-
# checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from generic_poly_dunder_cases import GENERIC_POLY_DUNDER_SPECS  # noqa: E402

_generic_poly_dunder_collisions = set(GENERIC_POLY_DUNDER_SPECS) & set(REGISTRY)
if _generic_poly_dunder_collisions:
    raise AssertionError(
        f"generic_poly_dunder_cases.py: {sorted(_generic_poly_dunder_collisions)} "
        f"already present in registry.REGISTRY -- refusing to silently "
        f"overwrite an existing item"
    )
REGISTRY.update(GENERIC_POLY_DUNDER_SPECS)

# ---------------------------------------------------------------------------
# polyutils_cases.py's POLYUTILS_SPECS: `numpy.polynomial.polyutils`'s
# shared, basis-agnostic helper surface (`trimseq`, `as_series`, `trimcoef`,
# `getdomain`, `mapparms`, `mapdomain`) -- the ONE genuinely shared
# implementation backing all six polynomial bases above, not per-basis
# work. `format_float` is out of scope (dragon4-backed, see
# anionpy/polynomial/polyutils.py's module docstring) and has no entry.
# Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from polyutils_cases import POLYUTILS_SPECS  # noqa: E402

_polyutils_collisions = set(POLYUTILS_SPECS) & set(REGISTRY)
if _polyutils_collisions:
    raise AssertionError(
        f"polyutils_cases.py: {sorted(_polyutils_collisions)} "
        f"already present in registry.REGISTRY -- refusing to silently "
        f"overwrite an existing item"
    )
REGISTRY.update(POLYUTILS_SPECS)

# ---------------------------------------------------------------------------
# Merge poly1d_legacy_cases.py's LEGACY_POLY_SPECS in (ticket #72): numpy's
# LEGACY polynomial API (`polyval`, `polyadd`, `polysub`, `polyder`,
# `polyint`, `polydiv`, `poly`, `roots`, `polyfit`/`polyfit_full`/
# `polyfit_cov` -- 9 items, all kind="custom") -- the pre-1.4,
# highest-degree-first coefficient convention, UNRELATED to the modern
# `numpy.polynomial` package merged in above. `poly1d` (the class itself)
# has no items here: this corpus only covers the free functions. See
# poly1d_legacy_cases.py's module docstring for corpus design and the
# declined items (`polymul`, `convolve`, `correlate`, `array_str`,
# `poly1d.__mul__`/`__rmul__` non-scalar, `poly1d.__pow__` n>1 -- all need
# `np.convolve`'s unreproduced summation order, ticket #45) and
# docs/TICKET-72-POLY1D-2026-08-08.md for the full implementation writeup.
# Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from poly1d_legacy_cases import LEGACY_POLY_SPECS  # noqa: E402

_legacy_poly_collisions = set(LEGACY_POLY_SPECS) & set(REGISTRY)
if _legacy_poly_collisions:
    raise AssertionError(
        f"poly1d_legacy_cases.py: {sorted(_legacy_poly_collisions)} "
        f"already present in registry.REGISTRY -- refusing to silently "
        f"overwrite an existing item"
    )
REGISTRY.update(LEGACY_POLY_SPECS)


# ---------------------------------------------------------------------------
# finfo / iinfo (ticket #71) -- the machine-limits introspection surface.
# 17 items covering the STORED-vs-COMPUTED attribute split (numpy exposes
# its computed limits as class-level property/cached_property descriptors,
# not instance attributes; anionpy stored them as instance attributes, which
# made 9 correct items structurally invisible to coverage.py's
# `attr in vars(cls)` check), plus `__repr__`/`__str__` on both classes and
# the `__new__`/`__init__` construction paths.
#
# The 4 construction-path items are deliberately NOT declared exact: they hit
# the ticket #31 gap, where an invalid dtype argument produces an error
# message naming dtype('O') against numpy's dtype('<U') / dtype('V'). That is
# architectural (anionpy has no void/string DType variant to name) and stays
# open. Their cases are wired anyway so the divergence is MEASURED on every
# run rather than remembered.
#
# See docs/TICKET-71-FINFO-2026-08-08.md.
# Same collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
# MERGE IS A DELIBERATE SUBSET, and the reason is recorded here rather than
# left to be re-derived. Of this module's 17 specs, 11 COLLIDE with specs
# dtypeinfo_cases.py already merged above (`iinfo.min`/`.max`,
# `finfo.epsneg`/`.iexp`/`.machep`/`.negep`/`.nexp`/`.resolution`/`.tiny`,
# and both `__repr__`s). Those 11 were never uncovered -- they were
# `absent` because they did not RESOLVE (instance attributes, invisible to
# coverage.py's `attr in vars(cls)`), not because nothing tested them. The
# #71 property conversion fixed resolution, so the pre-existing
# dtypeinfo_cases specs now grade them, and importing a second corpus for
# the same 11 items would overwrite working coverage with untried cases.
#
# Only the 6 genuinely-new items are merged. This module's richer
# 5-spelling x 13-dtype matrix for the other 11 is NOT discarded -- it stays
# in finfo_iinfo_cases.py, and whether to migrate the 11 onto it is a
# separate, deliberate decision that must compare the two corpora on
# strength rather than on which one happened to be written second.
#
# The excluded-by-name assertion below is the control: if a name in
# _FINFO_DEFERRED ever STOPS being present in REGISTRY, this subset has
# silently become a coverage hole and the suite must fail loudly instead of
# quietly grading 11 fewer items.
# ---------------------------------------------------------------------------
from finfo_iinfo_cases import FINFO_IINFO_SPECS  # noqa: E402

_FINFO_NEW = (
    "finfo.__str__", "iinfo.__str__",
    "finfo.__new__", "iinfo.__new__",
    "finfo.__init__", "iinfo.__init__",
)
_FINFO_DEFERRED = sorted(set(FINFO_IINFO_SPECS) - set(_FINFO_NEW))

_missing_new = [n for n in _FINFO_NEW if n not in FINFO_IINFO_SPECS]
if _missing_new:
    raise AssertionError(
        f"finfo_iinfo_cases.py no longer defines {_missing_new} -- the "
        f"subset merge below would silently add nothing for them"
    )
_finfo_iinfo_collisions = set(_FINFO_NEW) & set(REGISTRY)
if _finfo_iinfo_collisions:
    raise AssertionError(
        f"finfo_iinfo_cases.py: {sorted(_finfo_iinfo_collisions)} "
        f"already present in registry.REGISTRY -- refusing to silently "
        f"overwrite an existing item"
    )
_finfo_uncovered = [n for n in _FINFO_DEFERRED if n not in REGISTRY]
if _finfo_uncovered:
    raise AssertionError(
        f"finfo_iinfo_cases.py: deferred items {_finfo_uncovered} are NOT "
        f"covered by dtypeinfo_cases.py either -- the subset merge would "
        f"leave them untested. Merge them explicitly or fix the deferral."
    )
REGISTRY.update({n: FINFO_IINFO_SPECS[n] for n in _FINFO_NEW})

# --- ma phase-5 layout/metadata mirrors (ticket MA-PHASE5, 2026-08-08) -------
# 8 read-only MaskedArray layout/metadata items, delegating to already-exact
# anionpy.ndarray properties. `ma.MaskedArray.flags` is deliberately NOT here:
# real numpy's MaskedArray is an ndarray subclass, so `.flags.owndata` reads
# False for every real MaskedArray (a view-semantics artifact) while anionpy's
# `_data` is genuinely owned -- a naive delegation would mismatch owndata on
# literally every case. `iscontiguous()` was implemented instead, after
# verifying C_CONTIGUOUS does not depend on the owndata/view distinction.
#
# Unlike the finfo/iinfo block above this is a FULL merge, not a subset, so the
# control assertion is the mirror image: every spec the module defines must be
# claimed here. A spec added to ma_phase5_cases.py later would otherwise be
# silently left untested -- the failure mode the finfo block's deferral check
# exists to prevent, arriving from the opposite direction.
from ma_phase5_cases import MA_PHASE5_SPECS  # noqa: E402

_MA_PHASE5_NEW = (
    "ma.MaskedArray.data", "ma.MaskedArray.dtype", "ma.MaskedArray.mask",
    "ma.MaskedArray.itemsize", "ma.MaskedArray.nbytes", "ma.MaskedArray.strides",
    "ma.MaskedArray.iscontiguous", "ma.MaskedArray.get_fill_value",
)

_ma5_missing = [n for n in _MA_PHASE5_NEW if n not in MA_PHASE5_SPECS]
if _ma5_missing:
    raise AssertionError(
        f"ma_phase5_cases.py no longer defines {_ma5_missing} -- the merge "
        f"below would silently add nothing for them"
    )
_ma5_unclaimed = sorted(set(MA_PHASE5_SPECS) - set(_MA_PHASE5_NEW))
if _ma5_unclaimed:
    raise AssertionError(
        f"ma_phase5_cases.py defines {_ma5_unclaimed}, which this merge block "
        f"does not claim -- they would be built but never tested. Add them to "
        f"_MA_PHASE5_NEW or explain the deferral here."
    )
_ma5_collisions = set(_MA_PHASE5_NEW) & set(REGISTRY)
if _ma5_collisions:
    raise AssertionError(
        f"ma_phase5_cases.py: {sorted(_ma5_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update({n: MA_PHASE5_SPECS[n] for n in _MA_PHASE5_NEW})

# --- memmap (ticket MEMMAP, 2026-08-08 / #77, 2026-08-08) -------------------
# `docs/TICKET-MEMMAP-2026-08-08.md`: `tests/differential/memmap_cases.py`
# shipped as a new, UNWIRED module (registry.py was owned by a concurrent
# agent at the time) with a paste-ready merge block for a human to apply once
# free -- it is free now (see `git log` for this file: nothing since the
# ma-phase5 merge above). Full merge, not a subset: every `MEMMAP_SPECS` item
# is claimed, matching the doc's own merge block verbatim.
#
# TICKET #77 (2026-08-08) then measured "memmap.__new__": "exact" FALSE --
# the corpus varied dtype/mode IDENTITY (which of 14 families / 4 short
# modes) but never SPELLING, so it never exercised `_DTYPE_FMT`'s narrow
# 14-name-string parser or the missing long-form mode aliases. Fixed in
# `anionpy/memmap.py` (dtype routed through the already-exact `ap.dtype()`
# normalizer; mode aliases + numpy's exact invalid-mode message added) and
# the corpus extended with dtype/mode spelling-variation cases (see that
# file's own TICKET #77 comments). `memmap.__repr__` was WITHDRAWN from
# `anionpy/_state/memmap.py` by the same ticket: the wider corpus surfaced a
# genuine, pre-existing, memmap-independent repr divergence for dtypes
# constructed via a char code that is `==` but not `is` real numpy's
# canonical dtype object (e.g. `'q'` vs `'int64'`) -- see that file's module
# docstring for the full measurement.
from memmap_cases import MEMMAP_SPECS  # noqa: E402

# RESTORED (#77, 2026-08-08): this guard was written for the original memmap
# merge, lost when that merge was rewritten, and is restored here deliberately.
# It enforces declared-subset-of-corpus: nothing may be declared in
# `anionpy/_state/memmap.py` without a differential spec backing it. Ticket #77
# is itself an instance of a false `exact` declaration surviving a green suite,
# so this invariant earns its keep. Direction matters: corpus-without-
# declaration is fine (a measured, deliberately-undeclared divergence, of which
# this block has two), but declaration-without-corpus is the failure mode the
# project's definition of done forbids outright.
from anionpy._state.memmap import MEMMAP_STATE as _MEMMAP_STATE  # noqa: E402

_memmap_undertested = sorted(set(_MEMMAP_STATE) - set(MEMMAP_SPECS))
if _memmap_undertested:
    raise AssertionError(
        f"anionpy/_state/memmap.py declares {_memmap_undertested} but "
        f"memmap_cases.py has no spec for them -- a declaration with no "
        f"differential test is exactly what this project's definition of done "
        f"forbids. Add the specs or withdraw the declarations."
    )

_memmap_collisions = set(MEMMAP_SPECS) & set(REGISTRY)
if _memmap_collisions:
    raise AssertionError(
        f"memmap_cases.py: {sorted(_memmap_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite an existing item"
    )
REGISTRY.update(MEMMAP_SPECS)

# ---------------------------------------------------------------------------
# Merge array_protocol_cases.py's ARRAY_PROTOCOL_SPECS in -- the
# array-protocol dunder block on `ndarray` (`__array_finalize__`,
# `__array_wrap__`, `__array_priority__`, `__array_namespace__`,
# `__array_function__`, `__array_ufunc__`, `__dlpack_device__`,
# `__array_interface__`, `__array_struct__`) backed by the new
# `ionp-py/src/array_protocol.rs` (same multi-file `#[pymethods]` pattern as
# `ndarray_attrs.rs`), 2026-08-13. All nine had zero differential coverage
# before this task. `__array_prepare__` is correctly absent (removed from
# numpy 2.x); `__buffer__`/full `__dlpack__` are not implemented (see that
# file's module docstring for why) and so have no spec here either. Same
# collision-checked merge pattern as every block above.
# ---------------------------------------------------------------------------
from array_protocol_cases import ARRAY_PROTOCOL_SPECS  # noqa: E402

_array_protocol_collisions = set(ARRAY_PROTOCOL_SPECS) & set(REGISTRY)
if _array_protocol_collisions:
    raise AssertionError(
        f"array_protocol_cases.py: {sorted(_array_protocol_collisions)} already "
        f"present in registry.REGISTRY -- refusing to silently overwrite an "
        f"existing item"
    )
REGISTRY.update(ARRAY_PROTOCOL_SPECS)

# ---------------------------------------------------------------------------
# Merge products_io_cases.py's PRODUCTS_IO_SPECS in -- two clusters landed
# together 2026-08-13: the `.npy`/`.npz` file-I/O toplevel functions
# (`save`, `load`, `savez`, `savez_compressed`, `frombuffer`, backed by new
# `ionp-py/src/io_ops.rs` + pre-existing `ionp-core/src/format.rs`) and
# `cross` from the contraction family (`ionp-core/src/products.rs`,
# `ionp-py/src/products_py.rs`). `dot`/`vdot`/`inner`/`tensordot` from the
# same file are deliberately NOT registered here -- see
# products_io_cases.py's own module docstring for the measured BLAS-order
# non-bit-exactness that keeps them undeclared in toplevel.py, reported
# rather than hidden behind a corpus that would either accept known
# failures or dodge the cases that expose them.
# ---------------------------------------------------------------------------
from products_io_cases import PRODUCTS_IO_SPECS  # noqa: E402

_products_io_collisions = set(PRODUCTS_IO_SPECS) & set(REGISTRY)
if _products_io_collisions:
    raise AssertionError(
        f"products_io_cases.py: {sorted(_products_io_collisions)} already "
        f"present in registry.REGISTRY -- refusing to silently overwrite an "
        f"existing item"
    )
REGISTRY.update(PRODUCTS_IO_SPECS)

# ---------------------------------------------------------------------------
# Merge object_dtype_cases.py's OBJECT_DTYPE_SPECS in -- TICKET #38's
# marker-only `np.dtype('O')` support (`spelling == Some('O')` sentinel on
# `PyDType`, `ionp-py/src/lib.rs`). Four items: `dtype/object` (construction
# + full property surface across every spelling numpy accepts), `dtype/
# object_eq` (equality/ordering, including against a genuine numpy dtype
# instance -- the exact gap a live probe found missing during this ticket),
# `dtype/object_via_min_scalar_type` (the out-of-i64/u64-range overflow path
# that used to raise `OverflowError` unconditionally, now returns the object
# marker matching real numpy), and `dtype/object_storage_gap` (documented,
# PERMANENT, out-of-scope: `zeros`/`array`/`astype` with `dtype=object` still
# raise `TypeError`, unchanged by this ticket -- included so that stays a
# visible FAIL, not a silent omission). Same collision-checked merge pattern
# as every block above.
# ---------------------------------------------------------------------------
from object_dtype_cases import OBJECT_DTYPE_SPECS  # noqa: E402

_object_dtype_collisions = set(OBJECT_DTYPE_SPECS) & set(REGISTRY)
if _object_dtype_collisions:
    raise AssertionError(
        f"object_dtype_cases.py: {sorted(_object_dtype_collisions)} already "
        f"present in registry.REGISTRY -- refusing to silently overwrite an "
        f"existing item"
    )
REGISTRY.update(OBJECT_DTYPE_SPECS)

from pickle_cases import PICKLE_SPECS  # noqa: E402
_pickle_collisions = set(PICKLE_SPECS) & set(REGISTRY)
if _pickle_collisions:
    raise AssertionError(f'pickle_cases.py collisions: {sorted(_pickle_collisions)}')
REGISTRY.update(PICKLE_SPECS)

# Import-time invariant, same idiom and same reason as
# _require_varargs_form_if_numpy_supports_it above: run UNCONDITIONALLY on
# import so `run.py` enforces it too, not only pytest.
#
# This one guards declarations rather than cases: no bit-generator class may
# be declared "exact" in RANDOM_STATE while its public surface is narrower
# than numpy's. It exists because `random.SFC64` was declared exact at
# 17b7bc3 with 1 of numpy's 8 public attributes, and the differential suite
# passed -- the suite grades the methods a corpus exercises, and has no
# opinion about the ones nobody wrote a case for.
#
# Wiring matters more than authorship here. The guard shipped as a pytest
# module, and the routine declaration workflow is `run.py`, not pytest -- a
# guard that only fires under a command nobody runs on the path where the
# mistake is made is one step from decorative. Hence this line.
from test_bitgen_surface import (  # noqa: E402
    test_no_bitgen_class_is_declared_exact_with_an_incomplete_surface
    as _require_declared_bitgens_match_numpys_surface,
)
_require_declared_bitgens_match_numpys_surface()

from buffer_cases import BUFFER_SPECS  # noqa: E402
_buffer_collisions = set(BUFFER_SPECS) & set(REGISTRY)
if _buffer_collisions:
    raise AssertionError(f'buffer_cases.py collisions: {sorted(_buffer_collisions)}')
REGISTRY.update(BUFFER_SPECS)
