"""Core differential-test engine.

One job: given a numpy callable and an anionpy callable, and a list of
(label, args, kwargs) cases, decide -- per case and then per item -- whether
anionpy is equivalent to numpy under the policy below. No numerical work
happens here beyond calling `np.allclose`/`np.array_equal` for comparison;
this module never computes an answer, only checks one.

Equivalence policy (see README.md for the full rationale):
  - exceptions: if numpy raises, anionpy must raise the same exception type
    (or a type in that item's declared equivalence set). If numpy does not
    raise, anionpy must not raise.
  - dtype: exact `np.dtype` equality. No "close enough" kind matching.
  - shape: exact tuple equality.
  - values: bool/int/uint dtypes compare exact (`np.array_equal`), always,
    no tolerance of any kind. float and complex dtypes compare TRUE
    byte-exact (`tobytes()`, `_bit_exact_equal` -- permanent standard as of
    2026-08-01, "keep the comparison harness byte-based": ULP-distance-0
    is not the same claim, since it collapses every NaN-vs-NaN pair
    regardless of payload) by default. A per-item, explicitly declared
    `ItemSpec.ulp_tolerance` (see registry.py) is the ONLY way to relax
    that on ordinary values, and it is graded PER DTYPE, not item-wide:
    `ulp_tolerance` is `{dtype_name: bound}`, and a result of dtype D is
    checked against D's
    own declared bound (or 0.0/bit-exact if D has no entry at all -- see
    `compare_values` below) -- never against another dtype's bound, even
    when they're declared on the same item. Comparison happens by real ULP
    distance (`ulp_distance`/`max_ulp_distance` below), never `np.isclose`/
    `np.allclose`. This exists because numpy's own float ufunc loops are
    demonstrably not correctly rounded (measured: `np.abs` on complex64
    disagrees with the platform's own scalar `hypotf` on 35% of seeded
    inputs, by up to 2 ULP -- see
    `reports/ionp-ulp-tolerance-decision-2026-08-01.md`), so demanding
    bit-exactness against numpy for those cases would mean reproducing
    numpy's SIMD approximation error, not correctness -- but that same
    measurement found `np.abs` on float32/float64 IS bit-exact (0 ULP), so
    grading those dtypes with the complex dtypes' slack would hide a real
    regression (the 2026-08-01 per-dtype-grading fix this module now
    implements). `ulp_tolerance` is opt-in, per item, structurally requires
    a recorded justification string (`ItemSpec.__post_init__` refuses to
    construct the item otherwise -- see registry.py), and defaults to
    nothing: an undeclared float/complex item, or a dtype absent from a
    declared item's tolerance dict, is graded bit-exact, full stop, no
    fallback tolerance of any kind.
  - the legacy `atol`/`rtol` (`np.allclose(..., equal_nan=True)`) path
    below is preserved ONLY for items that do not declare
    `ulp_tolerance`, to avoid re-grading (and possibly changing the
    verdict of) any currently-declared item as an unintended side effect
    of adding the ULP mechanism -- see the 2026-08-01 ULP-tolerance task
    brief. New items needing float/complex tolerance should use
    `ulp_tolerance`, not `atol`/`rtol`.

An item's verdict is "pass" only if every case in its corpus passes. One
case failing out of a thousand is "fail" -- there is no partial credit,
because coverage.py's ledger has no slot for "mostly right". A "pass" that
required ULP tolerance on at least one case is tracked separately
(`ItemResult.tolerant`) from one that was bit-exact on every case -- see
`evaluate()` below and `tools/coverage.py`'s ledger, which must never
conflate the two.
"""
from __future__ import annotations

import dataclasses
import math
import re
import struct

import numpy as np


# Substrings of an exception message that legitimately differ run-to-run on
# BOTH sides. These are scrubbed before comparison so that the rest of the
# message is still checked -- the alternative (exempting the whole item from
# message comparison) would throw away a real check to dodge one token.
#
# Only two corpus items currently need this, and neither is an anionpy defect:
#   testing.rundocs             -- embeds the OS temp file it just wrote
#   testing.assert_no_gc_cycles -- embeds live object memory addresses
#
# Keep this list SHORT and justified. It is not a place to hide mismatches:
# anything added here stops being graded. A pattern belongs here only if it
# is nondeterministic on numpy's own side too.
_NONDETERMINISTIC_SCRUBS = (
    # CPython object reprs / memory addresses: `<object at 0x104f3a2c0>`
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "<ADDR>"),
    # macOS and Linux temp directories, including the `/private` prefix that
    # macOS resolves them through.
    (re.compile(r"(?:/private)?/var/folders/[^\s'\"]+"), "<TMPPATH>"),
    (re.compile(r"/tmp/[^\s'\"]+"), "<TMPPATH>"),
    # Bare `tempfile`-generated basenames that appear without a directory.
    (re.compile(r"\btmp[0-9a-z_]{6,}\b"), "<TMPNAME>"),
    # `id()` values rendered in DECIMAL, which the `0x...` rule above cannot
    # reach. numpy's `assert_no_gc_cycles` failure report lists every
    # uncollectable object as `  <kind> object with id=4413572736:`. These are
    # CPython addresses: numpy and anionpy each build their own objects, so the
    # numbers can never agree, while everything around them (object count,
    # kind, and the full repr of each object) is real and must stay graded.
    #
    # Anchored on the literal `id=` that is part of numpy's own message
    # format, NOT on a bare digit-run like `\d{9,}`. A bare digit-run would
    # silently eat real numeric content out of other items' messages --
    # shapes, sizes, axis numbers, dtype itemsizes -- and turn a genuine
    # mismatch green. Measured before adding: zero occurrences of literal
    # `id=<digits>` anywhere in the corpus or in anionpy's own sources, so this
    # rule is reachable only from the gc-cycles report it was written for.
    (re.compile(r"\bid=\d+"), "id=<ID>"),
)


#: The exception classes anionpy builds at module-init in `ionp-py/src/errors.rs`
#: as SUBCLASSES of numpy's real ones. This is the whole reason `import anionpy`
#: no longer requires numpy: anionpy owns its exception hierarchy, and merely
#: re-parents it onto numpy's when numpy happens to be importable, so that
#: `except numpy.linalg.LinAlgError:` still catches an ionp-raised error --
#: which is the compat direction that actually matters to a caller.
_IONP_COMPAT_EXC_NAMES = frozenset({"LinAlgError", "AxisError"})


def _is_ionp_compat_subclass(ionp_type: type, np_type: type) -> bool:
    """True iff `ionp_type` is anionpy's own compat subclass OF `np_type` itself.

    Deliberately far narrower than a bare `issubclass(ionp_type, np_type)`.
    A plain isinstance-style relaxation would also forgive genuine
    divergences -- numpy raising `ValueError` while anionpy raises some
    unrelated `ValueError` subclass would sail straight through, and that IS
    a real difference in behaviour, not a compat shim. So all four of these
    must hold:

      1. anionpy's class lives in module "anionpy" (not numpy's, not a test's,
         not `__main__` -- the `__module__` stamping in `errors.rs`'s
         `build_class` is what makes this clause enforceable at all);
      2. it is one of the two names anionpy actually builds this way;
      3. numpy's raised class is a DIRECT base of anionpy's -- i.e. literally
         the class anionpy re-parented onto, not merely an ancestor; and
      4. that base really is numpy's, not a same-named lookalike.

    Under those constraints the only pairs this can admit are exactly
    (anionpy.LinAlgError, numpy.linalg.LinAlgError) and (anionpy.AxisError,
    numpy.exceptions.AxisError). The message equality check below still runs
    afterwards and is NOT relaxed, so an item cannot pass on type alone.
    """
    return (
        getattr(ionp_type, "__module__", None) == "anionpy"
        and ionp_type.__name__ in _IONP_COMPAT_EXC_NAMES
        and np_type in ionp_type.__bases__
        and getattr(np_type, "__module__", "").split(".")[0] == "numpy"
    )


def _normalize_exc_message(msg: str) -> str:
    """Scrub only provably nondeterministic spans; everything else is graded.

    Note this does NOT touch whitespace. Trailing and doubled spaces are
    real, load-bearing parts of numpy's message text (numpy is internally
    inconsistent about them -- its plural "with shapes (3,) (4,) " form
    carries a trailing space where the singular "with shape (3,) (4,)" form
    does not), and stripping them here would silently forgive that class of
    defect.
    """
    for pattern, replacement in _NONDETERMINISTIC_SCRUBS:
        msg = pattern.sub(replacement, msg)
    return msg

MAX_FAILURES_KEPT = 8


_PROBE_SIG_PATTERNS = (
    "got an unexpected keyword argument",
    "missing 1 required positional argument",
    "missing 2 required positional arguments",
    "takes 1 positional argument but",
    "takes 2 positional arguments but",
    "takes 3 positional arguments but",
)


def _corpus_defined_names():
    """Names of functions DEFINED in tests/differential/*_cases.py.

    Computed by reading the corpus sources rather than by importing them:
    this module is imported BY those files, so importing them back here
    would be circular.
    """
    global _CORPUS_NAMES
    if _CORPUS_NAMES is None:
        import re as _re
        from pathlib import Path as _Path
        names = set()
        for p in _Path(__file__).parent.glob("*_cases.py"):
            try:
                names.update(_re.findall(r"^def (\w+)", p.read_text(), _re.M))
            except OSError:
                continue
        _CORPUS_NAMES = names
    return _CORPUS_NAMES


_CORPUS_NAMES = None


def _is_probe_signature_error(exc) -> bool:
    """True iff `exc` is OUR OWN probe helper rejecting the call's signature.

    Deliberately narrow. `np.linalg.eigh(A, bogus=1)` raises a TypeError of
    exactly the same SHAPE ("eigh() got an unexpected keyword argument
    'bogus'"), and that is a legitimate differential case that must keep
    being graded normally -- so the message text alone cannot be the
    discriminator.

    Frame location cannot be the discriminator either: a signature TypeError
    is raised by the interpreter at the CALL site, so its innermost frame is
    the corpus file in BOTH the real-numpy case and the vacuous case. (A
    detector built on frame location was written first and rejected on
    measurement -- it flagged 9,310 signatures, nearly all of them genuine
    numpy C-level errors, which have no Python frame of their own.)

    What actually separates the two: WHICH function the message names. Our
    probe helpers are defined in tests/differential/*_cases.py; numpy's and
    anionpy's are not.
    """
    if not isinstance(exc, TypeError):
        return False
    msg = str(exc)
    if not any(pat in msg for pat in _PROBE_SIG_PATTERNS):
        return False
    fname = msg.split("(")[0].strip()
    return bool(fname) and fname in _corpus_defined_names()


class InvalidCase(Exception):
    """Raised by registry.py's "ndarray."-prefixed resolve_ionp() branch
    (2026-08-01, scalar-receiver-tautology fix) when a `kind="binary_op"`
    case's RECEIVER (the first operand a dunder is resolved on, i.e.
    `arr.__add__` etc.) is not a `numpy.ndarray`/`anionpy.ndarray` -- the one
    real instance of this today is corpus.py's `broadcast/scalar_array`
    Pair, whose first operand is `np.float64(2.5)`.

    `make_ionp_array_converter` converts only real `numpy.ndarray`
    instances and passes everything else through UNCONVERTED, by design
    (see that function's docstring, hazard (b)): that is exactly right for
    a scalar used as the second/right-hand operand (that's the whole point
    of the call-form coverage `corpus.scalar_operands()` exists for), but
    it is wrong for a scalar used as the RECEIVER, because the receiver is
    what `getattr(..., attr)` is resolved on. `getattr(np.float64(2.5),
    "__add__")` resolves numpy's OWN dunder on numpy's OWN object -- anionpy
    never runs. That is either a tautology that always passes (32 of the 37
    binary_op dunders: `np.float64` happens to define the same-named
    method, so both "sides" run identical numpy code) or a loud but
    equally meaningless AttributeError (the 5 in-place dunders `np.float64`
    lacks: `__ifloordiv__ __imod__ __ipow__ __ilshift__ __irshift__`).
    Neither outcome says anything about anionpy.

    run_case()/evaluate() special-case this exception: the case is recorded
    as explicitly INVALID (skipped, with a reason string) -- never credited
    as a pass, never blamed as a failure. This is legitimate specifically
    because `kind="binary_op"`'s OTHER case source
    (`corpus.unary_corpus() x corpus.scalar_operands()`, see registry.py's
    module docstring) already exercises a real anionpy.ndarray receiver
    against every scalar call form, including array-vs-scalar broadcasting
    -- so nothing is silently lost by refusing to grade the tautological
    case. See registry.py's resolve_ionp() for where this is raised."""


@dataclasses.dataclass
class ItemResult:
    name: str
    verdict: str  # "pass" | "fail"
    total: int
    failed: int
    failures: list  # list[str], first MAX_FAILURES_KEPT diagnostic lines
    reason: str
    tolerant: bool = False
    """True iff verdict == "pass" and at least one case ACTUALLY REQUIRED
    nonzero slack -- ULP slack (ItemSpec.ulp_tolerance) or epsilon slack
    (non-zero atol/rtol) -- to pass, i.e. this item is NOT bit-exact
    against numpy. False for an item that declares a tolerance mechanism
    but never needed it on this corpus (every case happened to be
    bit-identical) -- see `mechanism` for which mechanism was declared."""
    max_ulp_observed: float | None = None
    """Worst-case ULP distance seen across every case that had a
    meaningful one (float/complex-result cases only). None if no such case
    ran. Diagnostic only -- does not affect verdict, which is decided
    per-case against the declared tolerance. Computed even for items
    graded via the epsilon (atol/rtol) path -- see compare_values -- so
    "did this epsilon-graded item actually need the slack" is answerable
    the same way it is for ulp_tolerance items, not just assumed."""
    mechanism: str = "none"
    """Which tolerance/exemption mechanism this item's ItemSpec declares:
    "ulp" (non-None ulp_tolerance), "epsilon" (non-zero atol/rtol or
    declared epsilon_tolerance), "tie_exempt" (signed_zero_tie_exempt --
    added 2026-08-01 for `sort`/`ndarray.sort`/`sort_complex`'s -0.0/+0.0
    tied-run arrangement, see registry.py's field docstring), or "none"
    (none of the above -- byte-level bit-exact, including NaN payload, is
    the only way to pass). Static per item, independent of whether
    `tolerant` ended up True; the ledger (tools/coverage.py) buckets a
    passing item as bit-exact / ULP-tolerant / epsilon-tolerant / tie-
    exempt using `tolerant` (needed it) crossed with this field (which
    mechanism it would have needed)."""
    invalid: int = 0
    """Count of cases that raised `InvalidCase` (see that class's docstring
    above) -- a case whose receiver could never reach anionpy at all, so it is
    excluded from `total`/`failed` entirely rather than being credited a
    pass or blamed as a failure. 0 for every item that never hits this path
    (i.e. every item except a `kind="binary_op"` one whose corpus includes
    a non-ndarray-receiver Pair, as of 2026-08-01 only `corpus.py`'s
    `broadcast/scalar_array`). See the scalar-receiver-tautology fix
    report for the full accounting."""
    invalid_cases: list = dataclasses.field(default_factory=list)
    """Up to MAX_FAILURES_KEPT `"[label] reason"` strings, one per case that
    raised `InvalidCase`, mirroring `failures`' shape -- so a skipped case
    is exactly as auditable (by label and reason) as a failing one, per the
    "visible, not silently dropped" requirement this mechanism exists to
    satisfy."""


def _call(fn, args, kwargs):
    try:
        return fn(*args, **kwargs), None
    except BaseException as exc:  # noqa: BLE001 - this is a probe, not app code.
        # Deliberately BaseException, not Exception: a Rust panic crossing
        # the PyO3 boundary surfaces as `pyo3_runtime.PanicException`, which
        # derives from BaseException directly (NOT Exception -- verified:
        # `PanicException.__mro__ == (PanicException, BaseException,
        # object)`). An `except Exception` here would let a panic escape
        # this probe uncaught, blow through run_case()/evaluate() (which
        # only guard `except Exception` too) and abort the whole pytest
        # item as a collection-level error instead of being graded as an
        # ordinary "anionpy raised X, numpy raised Y" mismatch -- exactly the
        # failure mode that hid the setops.rs:935 diff panic from this
        # harness in the first place.
        return None, exc


# ---------------------------------------------------------------------------
# Argument freshening -- see the 2026-08-01 argument-reuse-contract fix.
#
# run_case() used to hand the SAME `args`/`kwargs` objects to both the numpy
# call and the anionpy call. For any operation that mutates its input (in-place
# dunders, `.at()`, `out=`) that is wrong twice over: (1) the numpy call runs
# first and, if it mutates, the anionpy call then receives numpy's ALREADY-
# MUTATED array as its own input -- comparing "anionpy(a+b)" against
# "numpy((a+=b)+b)" is not the same operation, and can spuriously fail OR
# (for symmetric ops) tautologically pass; (2) corpus.py's `_views()` derives
# eleven distinct cases from three SHARED base arrays -- a mutation in one
# case corrupts every later case sharing that base, so a failure can appear
# in a case that isn't the one that caused it.
#
# The fix is NOT `.copy()`/`np.empty_like(order='K')`: both of those hand
# back a freshly-allocated CONTIGUOUS, POSITIVE-STRIDE array regardless of
# the input's own layout, which would silently stop testing every
# non-contiguous/negative-stride/Fortran-order case corpus.py's `_views()`
# exists specifically to cover -- a coverage-quality regression disguised as
# a green suite. `_freshen()` instead reconstructs a new array with the
# IDENTICAL shape, dtype, strides, and byte offset as the original, backed
# by its own independent copy of the underlying buffer -- so a negative-
# stride reversed view stays a negative-stride reversed view, a
# non-contiguous column slice stays non-contiguous, Fortran order stays
# Fortran order, but the bytes live in memory nothing else can see or has
# already mutated.
# ---------------------------------------------------------------------------

def _freshen_array(arr: np.ndarray) -> np.ndarray:
    """Return an independent copy of `arr` with identical shape/dtype/
    strides/byte-offset, backed by a freshly-copied buffer. Walks the
    `.base` chain (a view-of-a-view, e.g. `np.diagonal(base2d[:4, :4])`, can
    be more than one hop) to find the ultimate owning array, copies THAT
    preserving its C/F contiguity, then re-derives this array's exact byte
    offset into the new buffer via the raw pointers in
    `__array_interface__` and rebuilds with `np.ndarray(..., buffer=,
    offset=, strides=)`. Every corpus base in corpus.py is documented as a
    freshly allocated C-contiguous array (`np.arange(...).reshape(...)`) or
    `np.asfortranarray(...)` (which owns a fresh F-contiguous buffer of its
    own), so this always succeeds for real corpus input; deliberately RAISES
    (does not silently fall back to a contiguous copy) if the root base is
    not itself C- or F-contiguous, or if the reconstruction fails to
    reproduce the original's shape/dtype/strides exactly -- a loud failure
    here is correct, a quiet contiguous-copy degradation would be the exact
    bug this function exists to avoid (see module docstring above)."""
    base = arr
    while base.base is not None:
        if not isinstance(base.base, np.ndarray):
            raise TypeError(
                f"_freshen_array: {arr!r} has a non-ndarray base "
                f"{type(base.base).__name__} in its .base chain -- cannot "
                f"faithfully reconstruct an independent copy"
            )
        base = base.base

    if not (base.flags["C_CONTIGUOUS"] or base.flags["F_CONTIGUOUS"]):
        raise TypeError(
            f"_freshen_array: root base of {arr!r} (shape={base.shape}, "
            f"strides={base.strides}) is neither C- nor F-contiguous -- "
            f"refusing to silently degrade to a contiguous copy, which "
            f"would stop exercising this array's actual layout"
        )
    order = "F" if (base.flags["F_CONTIGUOUS"] and not base.flags["C_CONTIGUOUS"]) else "C"
    new_base = base.copy(order=order)

    offset = (arr.__array_interface__["data"][0]
              - base.__array_interface__["data"][0])
    fresh = np.ndarray(shape=arr.shape, dtype=arr.dtype, buffer=new_base,
                        offset=offset, strides=arr.strides)

    if (fresh.shape != arr.shape or fresh.dtype != arr.dtype
            or fresh.strides != arr.strides
            or fresh.flags["C_CONTIGUOUS"] != arr.flags["C_CONTIGUOUS"]
            or fresh.flags["F_CONTIGUOUS"] != arr.flags["F_CONTIGUOUS"]):
        raise RuntimeError(
            f"_freshen_array: reconstruction did not faithfully preserve "
            f"{arr!r} -- got shape={fresh.shape} dtype={fresh.dtype} "
            f"strides={fresh.strides} (wanted shape={arr.shape} "
            f"dtype={arr.dtype} strides={arr.strides}) -- refusing to hand "
            f"back a silently-wrong copy"
        )
    return fresh


def _freshen(x):
    """Recursively rebuild `x` so the returned value shares no mutable
    buffer with `x`: ndarrays go through `_freshen_array` (layout-preserving
    independent copy), tuples/lists/dicts are rebuilt with every element
    freshened in turn, and anything else (Python/numpy scalars, dtype
    objects, strings, None, `out=`/call-form tag strings, ...) is returned
    unchanged -- there is nothing mutable-and-shared to protect there, and
    copying e.g. a dtype object would not change its identity semantics
    anyway. Used by run_case() to give the numpy call and the anionpy call each
    their own independent set of arguments, per case -- see module docstring
    above."""
    if isinstance(x, np.ndarray):
        return _freshen_array(x)
    if isinstance(x, tuple):
        return tuple(_freshen(v) for v in x)
    if isinstance(x, list):
        return [_freshen(v) for v in x]
    if isinstance(x, dict):
        return {k: _freshen(v) for k, v in x.items()}
    return x


def _dtype_of(x):
    d = getattr(x, "dtype", None)
    if d is None:
        # Plain Python scalars have no .dtype. Reporting None here (rather
        # than guessing a dtype for them) makes that visible as a real
        # mismatch against numpy, which always returns a typed scalar or
        # array -- see README.md "why sum_f64 fails the sum item" for a
        # concrete example.
        return None
    try:
        return np.dtype(d)
    except Exception as exc:  # noqa: BLE001 - reporting the failure, not swallowing it
        # #37 fix: `np.dtype(d)` on a FOREIGN (non-numpy) dtype-like object
        # duck-types through `.name` rather than `str(d)` -- fine for every
        # non-parameterized dtype (`.name` IS a valid construction string
        # there, e.g. "int64"), but genuinely broken for S/U: numpy's OWN
        # convention is that a parameterized string dtype's `.name` is a
        # DISPLAY-only form, not reconstructible (`np.dtype('<U3').name`
        # -> `'str96'`, and `np.dtype('str96')` raises `TypeError: data
        # type 'str96' not understood'` for a REAL numpy dtype too, not
        # just anionpy's). `str(d)` (`'<U3'`/`'|S3'`) IS numpy's
        # reconstructible typestring for these, so retry through it before
        # giving up -- this only ever WIDENS successful coercion to a real
        # `np.dtype` (never weakens the eventual value comparison, which
        # still requires exact equality against that real dtype).
        try:
            return np.dtype(str(d))
        except Exception:
            pass
        # `x.dtype` exists but is not numpy-dtype-interoperable (e.g. a
        # foreign dtype object np.dtype() refuses to coerce). That is a
        # real, reportable mismatch against numpy -- which always returns a
        # genuine np.dtype -- not a reason to let the whole case blow up
        # with an uncaught exception. Returning a distinct, never-equal-to-
        # a-real-np.dtype string here means compare_values' plain
        # `np_dtype != ionp_dtype` check reports it as a clean "dtype
        # mismatch" diagnostic instead of "harness raised while running
        # case", without weakening the check in any way (it still can only
        # ever compare unequal to a real np.dtype).
        return f"<not numpy-dtype-interoperable: {d!r} ({type(d).__name__}): {exc}>"


def _shape_of(x):
    return tuple(getattr(x, "shape", ()))


# ---------------------------------------------------------------------------
# ULP distance -- see reports/ionp-ulp-tolerance-decision-2026-08-01.md.
#
# Deliberately NOT np.isclose/np.allclose: epsilon/relative comparison is a
# different, sloppier bar that hides real defects near zero and doesn't
# correspond to "numpy's own loop isn't correctly rounded by more than a
# few representable steps", which is the actual, narrow thing a declared
# ulp_tolerance is allowed to excuse.
#
# Rules (stated, not just implemented, per the task brief):
#   - NaN: both-NaN is distance 0 (NaN-ness must match; payload/sign does
#     not matter). Exactly one NaN is distance +inf (always fails, at any
#     tolerance) -- NaN-ness is structural, never tolerated.
#   - +-inf: equal-signed infinities are distance 0. Anything else touching
#     an infinity (opposite-signed infinities, or one side finite) is
#     distance +inf.
#   - +0.0 vs -0.0: treated as distance +inf, i.e. NOT within any finite
#     ULP tolerance, even though the raw bit-pattern-ordering transform
#     below would otherwise put them 0 ULP apart. This is a deliberate
#     policy choice: sign of zero is a real, distinguishable IEEE754 bit
#     (branch cuts in complex sqrt/log/atan2 depend on it), and a harness
#     whose job is catching wrong special-case branches must not treat
#     "the sign bit is wrong" as "close enough". Verified by
#     `test_ulp_distance.py` and `selftest.ulp_sqrt_wrong_zero_sign`.
#   - otherwise: the standard IEEE754 "biased integer" total-order mapping
#     (Bruce Dawson's algorithm) -- reinterpret the bit pattern as a
#     monotonic integer, then take the absolute integer difference. Done in
#     arbitrary-precision Python ints throughout (never fixed-width numpy
#     ints) specifically so the ordering transform can never silently wrap.
# ---------------------------------------------------------------------------

def _float_bits(x: float, width: int) -> int:
    if width == 16:
        # IEEE754 binary16. struct's "e" format is exactly that, and it
        # round-to-nearest-evens on pack -- same rounding numpy applies -- so
        # a caller that has already narrowed via np.float16(...).item() gets
        # a bit-identical pattern back rather than a re-rounded one.
        return struct.unpack("<H", struct.pack("<e", x))[0]
    if width == 32:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    if width == 64:
        return struct.unpack("<Q", struct.pack("<d", x))[0]
    raise ValueError(f"ULP distance only defined for 16/32/64-bit floats, got width={width}")


def _ordered_int(bits: int, width: int) -> int:
    """Bruce Dawson's total-order integer mapping: reinterprets an unsigned
    IEEE754 bit pattern as a monotonic integer (increasing with the float's
    actual value), so `abs(ordered(a) - ordered(b))` is a true ULP count.
    +0.0 and -0.0 both map to 0 here -- the +-0.0-is-never-within-tolerance
    rule is enforced by the caller (`ulp_distance`), not here, so this
    function stays a pure, reusable bit-order transform."""
    mod = 1 << width
    half = 1 << (width - 1)
    signed = bits if bits < half else bits - mod
    if signed >= 0:
        return signed
    val = (-half - signed) % mod
    if val >= half:
        val -= mod
    return val


def ulp_distance(a: float, b: float, width: int) -> float:
    """ULP distance between two Python floats, both understood as
    representing a value of the given IEEE754 `width` (16, 32 or 64 bits) --
    i.e. callers must already have rounded `a`/`b` to that width (e.g. via
    `np.float32(...).item()`) before calling this. Returns `math.inf` for
    any case this module treats as never-within-tolerance (see module
    docstring above); otherwise a non-negative float ULP count."""
    a = float(a)
    b = float(b)
    a_nan, b_nan = math.isnan(a), math.isnan(b)
    if a_nan or b_nan:
        return 0.0 if (a_nan and b_nan) else math.inf
    a_inf, b_inf = math.isinf(a), math.isinf(b)
    if a_inf or b_inf:
        return 0.0 if a == b else math.inf
    if a == 0.0 and b == 0.0:
        return 0.0 if math.copysign(1.0, a) == math.copysign(1.0, b) else math.inf
    ai = _ordered_int(_float_bits(a, width), width)
    bi = _ordered_int(_float_bits(b, width), width)
    return float(abs(ai - bi))


def _bit_exact_equal(np_out, ionp_out, signed_zero_tie_exempt=False) -> tuple[bool, bool]:
    """TRUE byte-level equality (`tobytes()`), the permanent standard for
    the default bit-exact comparison path (2026-08-01, coordinator
    directive: "keep that comparison harness byte-based permanently").
    Returns (equal, zero_sign_exemption_used).

    Why this replaced the previous `max_ulp_distance(...) == 0` check:
    `ulp_distance` collapses every NaN-vs-NaN pair to distance 0
    REGARDLESS of bit payload -- so a NaN-payload-canonicalization
    regression in `sort` (see sort.rs's `canonicalize_nans_*`) would
    silently PASS under the old check, exactly the class of bug this
    harness is supposed to catch. It also had no way to express "same
    value, different sign of zero" other than the previous all-or-nothing
    inf/0 verdict. Byte comparison is the only mechanism structurally
    incapable of missing either.

    `signed_zero_tie_exempt`: opt-in (see `ItemSpec.signed_zero_tie_exempt`
    in registry.py), for `sort`/`ndarray.sort`/`sort_complex` ONLY. `-0.0`
    and `+0.0` compare equal under IEEE `==`, so their relative
    ARRANGEMENT within a tied run of a sort is decided by the sorting
    ALGORITHM's internal mechanics (numpy's SIMD introsort kernel: verified
    directly by the coordinator to differ by array length, e.g. n=32 vs
    n=200, on the exact same tied input) -- not by any comparator, the same
    fundamental class of non-reproducible-arrangement problem as
    `argsort`'s already-scoped-out duplicate-value tie order. When this
    flag is set, a byte-level mismatch is still tolerated ONLY when the
    two elements it came from are BOTH exactly `0.0` (opposite sign) --
    this is a per-element check, not "any float 0.0 anywhere in the
    array is fair game": every other byte-level divergence, including
    every NaN payload (payload canonicalization is orthogonal to sign of
    zero and stays strictly bit-exact even when this flag is set -- NaN
    bit patterns are never `0.0` so the zero-only carve-out below can
    never match one), still fails.
    """
    a = np.asarray(np_out)
    b = np.asarray(ionp_out)
    if a.shape != b.shape:
        return False, False
    if a.tobytes() == b.tobytes():
        return True, False
    if not signed_zero_tie_exempt:
        return False, False
    if a.dtype.kind == "c":
        ar, ai = np.ascontiguousarray(a.real), np.ascontiguousarray(a.imag)
        br, bi = np.ascontiguousarray(b.real), np.ascontiguousarray(b.imag)
        eq_r, used_r = _bit_exact_equal(ar, br, signed_zero_tie_exempt=True)
        eq_i, used_i = _bit_exact_equal(ai, bi, signed_zero_tie_exempt=True)
        return (eq_r and eq_i), (used_r or used_i)
    if a.dtype.kind != "f":
        return False, False
    uint_dtype = {2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
    au = np.ascontiguousarray(a).view(uint_dtype)
    bu = np.ascontiguousarray(b).view(uint_dtype)
    mismatch = au != bu
    if not mismatch.any():
        return True, False
    both_zero = (np.ascontiguousarray(a) == 0.0) & (np.ascontiguousarray(b) == 0.0)
    if bool(np.all(both_zero[mismatch])):
        return True, True
    return False, False


def _width_for_dtype(dtype: np.dtype) -> int:
    if dtype.kind == "f":
        return dtype.itemsize * 8
    if dtype.kind == "c":
        return (dtype.itemsize // 2) * 8  # complex64 -> f32 components, complex128 -> f64
    raise ValueError(f"ULP distance is only defined for float/complex dtypes, got {dtype}")


def max_ulp_distance(np_out, ionp_out, dtype: np.dtype) -> float:
    """Worst-case (max) elementwise ULP distance between two same-shape,
    same-dtype array-likes. Complex dtypes are compared per-component (real
    and imaginary parts each get their own ulp_distance call at that
    dtype's component width), and the element's distance is the max of the
    two -- matching the task brief's "per-component for complex". Returns
    `math.inf` the moment any element is never-within-tolerance (NaN-ness
    mismatch, inf mismatch, or a +-0.0 sign mismatch) -- short-circuits
    rather than computing the rest, since the verdict is already decided."""
    width = _width_for_dtype(dtype)
    a = np.asarray(np_out).ravel()
    b = np.asarray(ionp_out).ravel()
    is_complex = dtype.kind == "c"
    worst = 0.0
    for i in range(a.size):
        if is_complex:
            av, bv = a[i], b[i]
            d = max(
                ulp_distance(float(av.real), float(bv.real), width),
                ulp_distance(float(av.imag), float(bv.imag), width),
            )
        else:
            d = ulp_distance(float(a[i]), float(b[i]), width)
        if d == math.inf:
            return math.inf
        if d > worst:
            worst = d
    return worst


def max_abs_distance(np_out, ionp_out) -> float:
    """Elementwise `|numpy_out - ionp_out|`, maximized over the output.
    `np.abs` of a complex difference is already the complex modulus (not a
    per-component distance), so this is the right primitive for both real
    and complex dtypes without special-casing. Used by the `"abs"` metric
    of `ItemSpec.epsilon_tolerance` (registry.py) -- the right instrument
    for outputs whose magnitude is bounded by the problem's own scale
    (decomposition factors, solves, inverses) rather than varying
    multiplicatively with matrix size/condition number (see
    `max_rel_distance` for those)."""
    a = np.asarray(np_out)
    b = np.asarray(ionp_out)
    if a.size == 0:
        return 0.0
    d = np.abs(a - b)
    # NaN-vs-NaN and matching-signed-inf-vs-inf are agreement, not an
    # unmeasurable distance -- mirrors ulp_distance's existing
    # `0.0 if (a_nan and b_nan) else math.inf` / `0.0 if a == b else
    # math.inf` rules (harness.py line ~439-443) so the epsilon path and
    # the ULP/bit-exact path never disagree about what counts as "the
    # same". `a - b` is `nan` any time either side is NaN OR both sides are
    # infinite (inf - inf = nan even when they agree), which silently
    # poisoned this metric to `nan` (always "exceeds" any bound) the first
    # time an item legitimately produces matching NaN/inf output (var/std
    # family's ddof<=0 / all-NaN / N-ddof==0-division slices) -- discovered
    # 2026-08-02 wiring var/std/nanvar/nanstd/average epsilon_tolerance, see
    # reduction_cases.py's REDUCTION_SPECS comment.
    both_nan = np.isnan(a) & np.isnan(b)
    mismatched_nan = np.isnan(a) ^ np.isnan(b)
    if np.any(mismatched_nan):
        return math.inf
    # Only same-sign inf-vs-inf needs special handling (inf - inf = nan
    # even though they agree); any other inf/finite mismatch already
    # legitimately evaluates to +inf via plain subtraction, no help needed.
    both_inf_same_sign = np.isinf(a) & np.isinf(b) & (a == b)
    d = np.where(both_nan | both_inf_same_sign, 0.0, d)
    return float(np.max(d))


def max_rel_distance(np_out, ionp_out, floor: float = 1.0) -> float:
    """Elementwise `|numpy_out - ionp_out| / max(|numpy_out|, floor)`,
    maximized over the output. Used by the `"rel"` metric of
    `ItemSpec.epsilon_tolerance` (registry.py) -- the right instrument for
    outputs whose magnitude scales multiplicatively with problem size or
    condition number (`det`, `matrix_power`, `cond`, and the eigvals-probe
    invariant in linalg_cases.py): an absolute bound tight enough to mean
    something on a near-zero-magnitude case fails a large-magnitude one
    purely from scale, not from a defect (measured directly, 2026-08-01:
    complex128 `det`'s absolute error reached 3.46e-4 on large-determinant
    cases while its RELATIVE error stayed at ~1e-13 across the same sweep
    -- see linalg_cases.py's module docstring). `floor` (default 1.0, the
    same value used for every `"rel"`-graded item in this registry) keeps
    the metric behaving like an absolute bound for genuinely small-
    magnitude numpy outputs -- never dividing by a near-zero denominator --
    while being genuinely relative for large ones."""
    a = np.asarray(np_out)
    b = np.asarray(ionp_out)
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.abs(a), floor)
    d = np.abs(a - b) / denom
    # Same NaN-vs-NaN-is-agreement and same-sign-inf-vs-inf-is-agreement
    # rules as max_abs_distance above (see that function's comment) --
    # `a - b` is `nan` any time either side is NaN or both sides are
    # infinite, which otherwise poisons this metric to `nan` for any item
    # that legitimately produces matching NaN/inf output.
    both_nan = np.isnan(a) & np.isnan(b)
    mismatched_nan = np.isnan(a) ^ np.isnan(b)
    if np.any(mismatched_nan):
        return math.inf
    both_inf_same_sign = np.isinf(a) & np.isinf(b) & (a == b)
    d = np.where(both_nan | both_inf_same_sign, 0.0, d)
    return float(np.max(d))


def _compare_scalar_like(np_out, ionp_out) -> tuple[bool, str]:
    """Comparison path for items whose result is NOT an ndarray: `.shape`
    (tuple), `.ndim`/`.size`/`__len__` (int), `.dtype` (dtype object),
    `__repr__` (str). The dtype-presence path in compare_values() below is
    the wrong tool for these -- numpy itself never puts a `.dtype` on a
    tuple or an int, so requiring one would flag every correct
    implementation of `.shape` as a mismatch. This path is opted into
    per-item via ItemSpec.scalar_like=True; it is never used for anything
    that returns an array, so it does not weaken the array-comparison path
    below in any way.

    Still strict: `np.dtype` results require the anionpy value to be coercible
    via `np.dtype(...)` and equal (not just str()-equal -- a value that
    merely *prints* like a dtype but cannot be used as one anywhere numpy
    expects a real dtype is legitimately a failure, not a pass). Everything
    else requires exact Python type equality before comparing with `==`.
    """
    if isinstance(np_out, np.dtype):
        try:
            ionp_as_dtype = np.dtype(ionp_out)
        except Exception as exc:  # noqa: BLE001 - reporting the failure, not swallowing it
            return False, (
                f"numpy returned dtype {np_out!r}; anionpy returned "
                f"{ionp_out!r} (type {type(ionp_out).__name__}) which is not "
                f"numpy-dtype-interoperable: np.dtype(...) coercion raised "
                f"{type(exc).__name__}: {exc}"
            )
        ok = ionp_as_dtype == np_out
        return ok, ("" if ok else f"dtype mismatch: numpy={np_out} anionpy={ionp_as_dtype}")

    if type(np_out) is not type(ionp_out):
        return False, (
            f"type mismatch: numpy returned {type(np_out).__name__}({np_out!r}), "
            f"anionpy returned {type(ionp_out).__name__}({ionp_out!r})"
        )
    # `x == x` is False for a bare Python/numpy-scalar NaN even when both
    # sides are the exact same value (or even the exact same object --
    # `float.__eq__` never special-cases self-identity, unlike list/tuple
    # `__eq__`, which DOES check element identity first) -- IEEE 754, not a
    # bug. A blunt `==` here would flag a genuinely-correct NaN-preserving
    # result (e.g. `trim_zeros`'s ndim==0 identity path returning the
    # original NaN object unchanged) as a false mismatch. `math.isnan`
    # only accepts real numbers, so this is guarded to float/int/bool
    # (complex NaN already routes through the array-comparison path above
    # this function, not `_compare_scalar_like`, since it carries a
    # `.dtype`).
    if isinstance(np_out, (float, int)) and not isinstance(np_out, bool):
        try:
            if math.isnan(np_out) and math.isnan(ionp_out):
                return True, ""
        except (TypeError, ValueError):
            pass
        # Sign-of-zero guard (2026-08-13, #86). Plain Python `float` results
        # reaching the bottom `==` fallback below were previously graded
        # exactly like any other value -- but IEEE 754 defines `0.0 ==
        # -0.0` as True, so that fallback was structurally BLIND to a
        # sign-flipped zero, the same defect class `_bit_exact_equal`
        # (above) exists to police for ndarray results and the `complex`
        # branch below already polices via `copysign`. Measured directly:
        # `_compare_scalar_like(-0.0, 0.0)` returned `ok=True` before this
        # guard (see docs/ for the write-up) -- proof this was not a
        # hypothetical gap. `type(...) is float` (not `isinstance`) is
        # deliberate: `type(np_out) is not type(ionp_out)` already returned
        # False above when the types differ, so both sides are the SAME
        # type here; restricting to exactly `float` (not `int`, which has
        # no signed zero and for which `math.copysign` would silently
        # accept mixed-representation ints) keeps this addition from
        # touching any currently-passing int-returning item.
        if type(np_out) is float:
            if np_out == 0.0 and ionp_out == 0.0:
                ok = math.copysign(1.0, np_out) == math.copysign(1.0, ionp_out)
                return ok, ("" if ok else
                            f"value mismatch (sign of zero): numpy={np_out!r} "
                            f"anionpy={ionp_out!r}")
    # A builtin Python `complex` DOES reach here, contradicting the
    # parenthetical above -- `ndarray.__complex__` (declared 2026-08-03)
    # returns one, and a builtin complex carries no `.dtype`, so it never
    # routes through the array path. Seven of its 245 cases failed on
    # `(nan+0j) != (nan+0j)`: the same IEEE fact the float branch above
    # already exists to handle, arriving through a type that branch cannot
    # accept (`math.isnan` rejects complex).
    #
    # This branch is deliberately STRICTER than the `==` fallback it
    # replaces, not more tolerant. `==` on complex would also grade
    # `(-0+0j)` equal to `0j`, because `-0.0 == 0.0` -- so a comparison
    # that merely added a NaN escape would have quietly stopped
    # distinguishing signed zeros in the process. Each component is
    # therefore compared for NaN-ness first and then by its `copysign`
    # bit as well as its value, which pins the sign of both zeros. (The
    # float branch above WAS `==`-based and blind to signed zeros, exactly
    # this same gap, arriving through the sibling type this comparison
    # accepts; that was the "own measurement" ticket #86 delivered
    # 2026-08-13 -- see the `type(np_out) is float` guard directly above,
    # which closes it the same way this branch already does for complex.)
    if isinstance(np_out, complex) and not isinstance(np_out, bool):
        def _same_component(x: float, y: float) -> bool:
            x_nan, y_nan = math.isnan(x), math.isnan(y)
            if x_nan or y_nan:
                return x_nan and y_nan
            return x == y and math.copysign(1.0, x) == math.copysign(1.0, y)

        ok = (_same_component(np_out.real, ionp_out.real)
              and _same_component(np_out.imag, ionp_out.imag))
        return ok, ("" if ok else
                    f"value mismatch: numpy={np_out!r} anionpy={ionp_out!r}")
    ok = bool(np_out == ionp_out)
    return ok, ("" if ok else f"value mismatch: numpy={np_out!r} anionpy={ionp_out!r}")


def _require_ulp_justification(ulp_tolerance, ulp_justification, context: str) -> None:
    """Defense in depth for the provenance requirement. `ItemSpec.__post_init__`
    (registry.py) is the primary, structural guard -- it refuses to
    construct an ItemSpec with a `ulp_tolerance` and no recorded
    `ulp_justification` at all, so a real registry entry can never reach
    this point in that state. This is a second, independent check at the
    point of actual use, for any caller that builds/calls comparison
    machinery without going through ItemSpec (there is none in the real
    registry today, but the rule must hold structurally, not just by
    convention) -- and it raises, loudly, rather than silently comparing
    exact or silently comparing tolerant."""
    if ulp_tolerance is None:
        return
    if not ulp_justification or not str(ulp_justification).strip():
        raise RuntimeError(
            f"{context}: ulp_tolerance={ulp_tolerance!r} declared with no recorded "
            f"justification string -- refusing to grade under an undocumented "
            f"tolerance. Every ULP tolerance must cite the recorded evidence that "
            f"numpy's own loop is not correctly rounded for this item (see "
            f"reports/ionp-ulp-tolerance-decision-2026-08-01.md and "
            f"KNOWN-DIFFERENCES.md)."
        )


def _require_epsilon_justification(atol, rtol, epsilon_justification, context: str) -> None:
    """Defense in depth for the epsilon path, mirroring
    `_require_ulp_justification` above. `ItemSpec.__post_init__`
    (registry.py) is the primary, structural guard -- it refuses to
    construct an ItemSpec with a non-zero atol/rtol and no recorded
    `epsilon_justification` at all. This is a second, independent check at
    the point of actual use. `atol=0.0, rtol=0.0` (or both None) never
    reaches this function -- see compare_values, which only calls it once
    it has already determined the epsilon path is non-zero."""
    if not epsilon_justification or not str(epsilon_justification).strip():
        raise RuntimeError(
            f"{context}: atol={atol!r} rtol={rtol!r} declared non-zero with no "
            f"recorded epsilon_justification -- refusing to grade under an "
            f"undocumented tolerance. Every non-zero atol/rtol must cite the "
            f"recorded evidence that bit-exactness is not the right bar for this "
            f"item (see registry.py's ItemSpec.epsilon_justification docstring)."
        )


def _require_epsilon_tolerance_justification(epsilon_tolerance, justification, context: str) -> None:
    """Defense in depth for `ItemSpec.epsilon_tolerance`, mirroring
    `_require_ulp_justification`/`_require_epsilon_justification` above.
    `ItemSpec.__post_init__` (registry.py) is the primary, structural
    guard; this is a second, independent check at the point of actual use."""
    if epsilon_tolerance is None:
        return
    if not justification or not str(justification).strip():
        raise RuntimeError(
            f"{context}: epsilon_tolerance={epsilon_tolerance!r} declared with no "
            f"recorded epsilon_tolerance_justification -- refusing to grade under "
            f"an undocumented tolerance."
        )


def _first_operand_dtype(args) -> np.dtype | None:
    """The dtype an item's `ulp_sweep`/`ulp_tolerance` evidence is actually
    keyed by: the dtype of the OPERAND fed into the call, not necessarily
    the dtype of the result. These differ for `abs`/`absolute`: an input of
    dtype complex64 produces a float32 OUTPUT (numpy's own `abs` drops the
    imaginary part into a real magnitude), but the genuine ~2-ULP
    imprecision is a property of the complex64 *input* regime (hypotf-style
    magnitude computation), not of "float32" in general -- grading it by
    OUTPUT dtype would collide it with true float32-input `abs`, which is
    bit-exact, either wrongly forgiving a float32-input regression (if
    complex64's slack leaked onto float32) or wrongly failing genuine
    complex64-input imprecision (if graded at float32's 0.0). See
    `ulp_sweep.py`'s `_summarize`, which keys `SweepResult.dtype` by the
    swept INPUT dtype for exactly this reason -- this helper mirrors that at
    grading time by scanning `args` for the first operand carrying a
    `.dtype` (skipping non-array items like a ufunc call-form tag string,
    e.g. `"call"`/`"reduce"`, which have no `.dtype`). Returns None if no
    operand carries one, in which case the caller falls back to the
    result's own dtype (preserves prior behavior for items whose args are
    all non-array, e.g. scalar-only cases)."""
    for a in args:
        d = getattr(a, "dtype", None)
        if d is None:
            continue
        try:
            return np.dtype(d)
        except Exception:  # noqa: BLE001 - not a real numpy-interoperable dtype, keep scanning
            continue
    return None


def _explicit_dtype_override(kwargs) -> np.dtype | None:
    """The dtype the arithmetic ACTUALLY runs in when the caller passed an
    explicit `dtype=` to a ufunc -- which takes priority over
    `_first_operand_dtype` as the `ulp_tolerance` lookup key.

    Read `_first_operand_dtype`'s docstring first: it keys by INPUT dtype on
    purpose, because `abs`'s ~2-ULP slack is a property of the complex64
    *input* regime (hypotf-style magnitude), not of "float32" in general, and
    grading it by output dtype would collide those two regimes. That
    reasoning is correct and is deliberately preserved here -- but it is
    reasoning about an output dtype DERIVED from the input. An explicit
    `dtype=` is a different thing: it selects the loop, so it IS the
    computation regime, and the input is merely cast into it.

    Concretely, `sin(uint8_array, dtype=float32)` does float32 arithmetic and
    carries float32's declared 1-ULP slack. Keyed by the input, it looked up
    "uint8" -> 0.0 and failed a result that sits exactly inside what the
    declaration claims. Measured, the `dtype=` path is bit-identical to the
    native call it delegates to:

        sin  uint8 ->float32 : ulp(dtype=)=1  ulp(native float32 input)=1
        tanh float16->float64: ulp(dtype=)=1  ulp(native float64 input)=1
        cos  uint8 ->float32 : ulp(dtype=)=0  ulp(native float32 input)=0

    tanh's declared float64 bound is 1.0 and sin/cos's float32 bound is 1.0,
    so every one of those sits inside an already-recorded, already-justified
    tolerance -- verified against the registry rather than assumed.

    This is NOT a widening. When no `dtype=` is passed it returns None and
    nothing changes; when one is passed, the item still must have a declared,
    justified tolerance FOR THAT DTYPE or it is graded at 0.0 exactly as
    before. It moves the lookup key onto the regime the evidence was
    gathered in. Grading float32 arithmetic against uint8's 0.0 bound is the
    same error, mirrored, as this session's earlier one of grading a float32
    residue in float64 ULP units and reading a 1-ULP gap as 5.4e8: a check
    graded at a bar the real requirement never claimed. That error direction
    invents failures instead of hiding them, which is no better.

    Returns None for a `dtype=None` (numpy's own "no override" spelling) and
    for anything not interpretable as a dtype, so a bogus `dtype=` case still
    falls through to the operand key and gets graded normally.
    """
    # `kwargs` is NOT guaranteed to be a mapping. `kind="custom"` items route
    # their own case tuples through this slot (scalar_cases.py's
    # `scalar_hash_matrix` passes a raw int/bool there), so `.get` on it
    # raises AttributeError and the harness reports "harness raised while
    # running case" for 15 cases that are perfectly fine. Assuming a shape
    # the contract never promised is the same mistake this function's
    # docstring is about, so: ask, don't assume.
    if not isinstance(kwargs, dict):
        return None
    requested = kwargs.get("dtype")
    if requested is None:
        return None
    try:
        return np.dtype(requested)
    except Exception:  # noqa: BLE001 - a deliberately-invalid dtype= case; grade it by operand
        return None


def _dtype_key_candidates(source_dtype, np_dtype, dtype_override=None, operand_dtype=None):
    """Ordered lookup-key candidates for `ulp_tolerance`/`epsilon_tolerance`
    dicts, most-specific first. Fixes a structural defect (found FOUR
    separate times against `nanvar`/`nanstd`, see reduction_cases.py's
    module docstring and the 2026-08-02 key-split task): the single-name
    key `_explicit_dtype_override(kwargs) or _first_operand_dtype(args)`
    (`source_dtype`, unchanged below) collapses two populations with
    genuinely different achievable precision into ONE dict entry whenever
    an explicit `dtype=` override differs from the true operand dtype --
    e.g. `nanvar(float16_array, dtype='float64')` and a genuine float64
    array with no override both key to `"float64"`, even though numpy
    itself truncates the deviation to the INPUT dtype before squaring for
    the former (DEFECT 2, reductions.rs), so its real error scale is
    float16/float32-epsilon, not float64-epsilon. Grading both under one
    key forces the bound to cover the looser population, leaving the
    tighter one effectively unguarded (measured: nanvar's float64 bound
    was ~3.2e8x looser than genuine float64 needed).

    This does NOT change the DEFAULT key (`source_dtype.name`, i.e. the
    override if present, else the operand dtype, else the result dtype --
    exactly `_explicit_dtype_override`'s documented priority, unchanged):
    that stays correct for the common case an override IS the actual
    compute regime (e.g. `sin(uint8_array, dtype=float32)`, which really
    does run entirely in float32 -- see `_explicit_dtype_override`'s own
    docstring). It ADDS a MORE SPECIFIC candidate, tried first --
    `"{operand_dtype.name}->{dtype_override.name}"` -- for exactly the
    case an override and the operand dtype differ, so an item whose
    behavior genuinely depends on BOTH (like nanvar/nanstd's truncate-then-
    widen accumulation) can declare a dedicated entry for that composite
    regime without disturbing any dict that never adds one. A dict with no
    composite entry falls straight through to the old single-name key,
    byte-for-byte the prior behavior -- so this is backward compatible
    with every already-declared `ulp_tolerance`/`epsilon_tolerance` in the
    registry; only items that explicitly add a composite-keyed entry are
    affected, and adding an entry can only SHRINK the set of cases graded
    under the loose fallback key, never widen it.

    Returns `(candidate_names, grading_dtype)`: `candidate_names` is the
    ordered list of dict keys to try (composite first when applicable,
    then the fallback single name); `grading_dtype` is the fallback dtype
    object itself, kept for diagnostics/messages exactly as before."""
    grading_dtype = source_dtype if source_dtype is not None else np_dtype
    candidates = []
    if (dtype_override is not None and operand_dtype is not None
            and dtype_override != operand_dtype):
        candidates.append(f"{operand_dtype.name}->{dtype_override.name}")
    candidates.append(grading_dtype.name)
    return candidates, grading_dtype


def _lookup_dtype_tolerance(tolerance_dict, candidates, default=None):
    """First candidate key present in `tolerance_dict`, else `default`.
    Returns `(value, matched_key)` -- `matched_key` is the candidate that
    hit, or the LAST candidate (the fallback single-name key) if none did,
    so diagnostics can still report a meaningful key even on a miss."""
    for key in candidates:
        if key in tolerance_dict:
            return tolerance_dict[key], key
    return default, candidates[-1]


def _strides_match(np_out, ionp_out, tolerant: bool = False) -> tuple[bool, str]:
    """Byte-level layout equality between a real numpy result and a real
    anionpy result. Deliberately a POST-check applied by `run_case` (see
    `check_strides` below) rather than folded into `compare_values` itself:
    every branch of `compare_values` (ULP-tolerant, epsilon-tolerant,
    bit-exact, exact-integer) has its own early-return, and grafting a
    strides comparison into all of them in place would multiply the surface
    area for a mistake in an already-dense function; checking strides once,
    after `compare_values` already said the VALUES are correct, is
    equivalent and far less invasive. Added 2026-08-01 for the ufunc
    order= task -- see `ItemSpec.check_strides`'s docstring for why this
    did not already exist anywhere in the harness.

    CORRECTED 2026-08-02 after a coordinator audit proved the original
    version was a test that compares a component against its own claim and
    therefore cannot fail: it read `ionp_out.strides` -- anionpy's own
    self-reported attribute -- and compared it against numpy's real
    strides. anionpy's order='F' implementation was found to relabel the
    `.strides` metadata field WITHOUT ever reordering the underlying
    buffer, so the attribute said (8, 24) while the actual bytes were still
    C-ordered; the old check read the label, agreed with itself, and
    passed. Fixed to go through the buffer protocol on BOTH sides via
    `np.asarray(...)`, which is the only way to observe the real physical
    layout anionpy is handing back to Python -- `np.asarray` never trusts a
    reported `.strides` value, it walks the actual exported buffer. Checks
    three independent things, any one of which failing is a real defect:
    (1) `np.asarray(...).strides` -- the ACTUAL buffer strides, not the
    label; (2) C_CONTIGUOUS/F_CONTIGUOUS flags, since two different stride
    tuples can describe the same contiguity class for degenerate axes
    (size-1 or size-0 dims) and flag-only agreement there is legitimate;
    (3) `np.asarray(...).tobytes('A')` -- the literal physical byte
    sequence in memory order, which is the ground truth `.strides` and
    flags are only trying to describe. `tobytes()` WITHOUT 'A' is
    deliberately never used here: it walks LOGICAL (C) order and would
    return identical bytes for a C buffer and a correctly-F-ordered buffer
    holding the same logical values, which is exactly the kind of check
    that cannot fail on the bug this was written to catch.

    CORRECTED AGAIN 2026-08-02 (design: `docs/strides-check-tolerance-bug.md`).
    Check (3) had no concept of tolerance, while `compare_values` has four
    passing branches -- bit-exact, exact-integer, ULP-tolerant and
    epsilon-tolerant. For an item that passes TOLERANTLY the values are
    legitimately not bit-identical (that is the entire point of its declared
    `ulp_tolerance` / `epsilon_tolerance`, each carrying a written
    justification), so `tobytes('A')` differs EVEN WHEN THE LAYOUT IS
    PERFECT and the failure was reported as a "physical byte layout
    mismatch" -- naming layout as the defect when the real difference is a
    last-ulp value difference the item explicitly permits. An item's own
    declared tolerance was being silently overridden by a check running
    downstream of it. Measured blast radius: 16 items, every one a false
    positive. Same family as the signed-zero and default-off instrument
    bugs: a check whose failure mode is STRUCTURALLY unable to distinguish
    the thing it names from a different thing entirely.

    Two changes. First, a new value-INDEPENDENT check (0) that catches the
    exact relabel bug check (3) was written for -- anionpy's self-reported
    `.strides` attribute disagreeing with the strides its buffer protocol
    actually exports -- without touching a single value, so no tolerance can
    interact with it. It is also strictly sharper than (3): it localises the
    defect to anionpy rather than inferring it from a byte diff against numpy.
    Second, (3) now runs ONLY on the bit-exact branch (`tolerant=False`).
    Deliberately NOT done: giving (3) a tolerance of its own. Byte sequences
    in memory order are not a numeric space, "close bytes" is not a
    meaningful predicate, and building one would manufacture precisely the
    kind of blunt instrument that measures a gap instead of finding one.
    Disclosed cost: for tolerantly-passing items layout is validated by
    (0)+(1)+(2) only, which is a sound check, where (3) was an unsound one."""
    np_arr = np.asarray(np_out) if hasattr(np_out, "__array__") or isinstance(np_out, np.ndarray) else None
    ionp_arr = np.asarray(ionp_out) if hasattr(ionp_out, "__array__") or isinstance(ionp_out, np.ndarray) else None
    if np_arr is None or ionp_arr is None:
        # Neither side is a real ndarray (e.g. a 0-output-shape edge case
        # that fell through to a bare Python scalar on one side) -- nothing
        # to compare; compare_values already validated dtype/shape/value
        # equivalence for this case, so silently skipping here does not
        # hide a layout defect that could exist between two real arrays.
        return True, ""
    # (0) Value-independent self-consistency check on the anionpy side ALONE:
    # does anionpy's own reported `.strides` attribute agree with the strides
    # its buffer protocol actually exports? numpy is not consulted and no
    # values are read, so this cannot be perturbed by a declared tolerance.
    # A disagreement is the metadata-relabel defect: a `.strides` label
    # claiming a layout the underlying buffer does not have.
    reported = getattr(ionp_out, "strides", None)
    if reported is not None:
        try:
            reported = tuple(int(s) for s in reported)
        except (TypeError, ValueError):
            reported = None
    if reported is not None:
        observed = tuple(ionp_arr.strides)
        # MEASURED CORRECTION 2026-08-04: compare ONLY the positions whose
        # extent is > 1. A stride on an axis of extent 0 or 1 is
        # semantically unobservable (there is no second element to step
        # to), and the C buffer protocol's exporter fills those positions
        # with the contiguous value regardless of what the owning object
        # reports. numpy does EXACTLY the same thing, so the un-masked
        # form of this check calls numpy itself defective:
        #     r = np.quantile(np.array([True]), [0.1, 0.9],
        #                     keepdims=True, method='lower')
        #     r.strides            -> (1, 0)
        #     memoryview(r).strides -> (1, 1)
        #     np.median(np.array([True]), axis=0, keepdims=True)
        #         .strides -> (0,)   memoryview(...).strides -> (8,)
        # numpy never trips check (0) only because `np.asarray` on an
        # ndarray is the identity and re-reads the same label; anionpy's
        # result must cross the buffer protocol to reach np.asarray, so it
        # gets normalised and the two sides were being read by different
        # instruments. The relabel defect this check was written for
        # (order='F' claiming (8, 24) over C-ordered bytes) lives entirely
        # on extent>1 axes and is still caught.
        keep = _observable_stride_axes(ionp_arr.shape)
        if tuple(reported[i] for i in keep) != tuple(observed[i] for i in keep):
            return False, (
                f"anionpy strides self-report disagrees with its own exported "
                f"buffer -- metadata relabel without reordering the bytes: "
                f"anionpy.strides={reported} but buffer-protocol strides="
                f"{observed} (shape={ionp_arr.shape}, dtype={ionp_arr.dtype})"
            )
    np_strides = tuple(np_arr.strides)
    ionp_strides = tuple(ionp_arr.strides)
    # Same extent<=1 mask as check (0), for the same measured reason: the
    # numpy side of this comparison is `np.asarray(ndarray)` (identity, so
    # the RAW label, e.g. (1, 0)) while the anionpy side crossed the buffer
    # protocol (normalised, e.g. (1, 1)). Masking the unobservable
    # positions makes the two sides comparable; every position that can
    # actually be stepped along is still compared exactly.
    #
    # DISCLOSED COST OF THAT MASK, measured 2026-08-04 -- READ THIS BEFORE
    # TRUSTING `check_strides=True` AS FULL LAYOUT COVERAGE. The mask hides
    # a real divergence class: a length-1 axis whose LABEL differs while
    # the bytes are identical (CLASS B).
    #     b = np.arange(8, dtype='float16').reshape(2, 1, 4)
    #     np.median(b.T, axis=-1).strides                 -> (2, 2)
    #     anionpy.median(<same>, axis=-1).strides            -> (2, 8)
    # I tried to close this by reading the numpy side through `memoryview`
    # so both sides were normalised the same way. That is WRONG and was
    # backed out: `memoryview(np_result)` recomputes contiguous strides,
    # but anionpy does not export a C buffer at all (`memoryview(ionp_arr)`
    # raises TypeError) -- `np.asarray` reaches it through
    # `__array_interface__`, which normalises stride-0 but not much else.
    # So that variant swapped one instrument mismatch for another and lit
    # up 5,000+ cases across all six median/quantile items, the majority of
    # them the legitimate `keepdims` broadcast axis:
    #     np.median(np.zeros((2, 3), bool), axis=0, keepdims=True)
    #         .strides -> (0, 8)   memoryview -> (24, 8)
    #     anionpy same    .strides -> (0, 8)   np.asarray -> (8, 8)
    # The comparison that WOULD catch CLASS B cleanly is raw label vs raw
    # label (`np_out.strides` vs `ionp_out.strides`, no round trip at all --
    # (0, 8) == (0, 8) above, (2, 2) != (2, 8) for the f16 case). It is
    # deliberately NOT added here: CLASS B is a library-wide held class, and
    # switching this shared helper on for it would relabel a known,
    # separately-tracked gap as a fresh failure across every item that sets
    # `check_strides`. Add it WITH the CLASS B fix, not before.
    _keep = _observable_stride_axes(np_arr.shape)
    if tuple(np_strides[i] for i in _keep) != tuple(ionp_strides[i] for i in _keep):
        return False, (
            f"strides mismatch (via np.asarray, buffer-protocol-verified): "
            f"numpy={np_strides} anionpy={ionp_strides} "
            f"(shape={np_arr.shape}, dtype={np_arr.dtype})"
        )
    np_c, np_f = np_arr.flags["C_CONTIGUOUS"], np_arr.flags["F_CONTIGUOUS"]
    ionp_c, ionp_f = ionp_arr.flags["C_CONTIGUOUS"], ionp_arr.flags["F_CONTIGUOUS"]
    if (np_c, np_f) != (ionp_c, ionp_f):
        return False, (
            f"contiguity-flag mismatch: numpy C/F={(np_c, np_f)} "
            f"anionpy C/F={(ionp_c, ionp_f)} (shape={np_arr.shape}, "
            f"dtype={np_arr.dtype})"
        )
    if tolerant:
        # The item passed via a declared ULP/epsilon tolerance, so the values
        # are legitimately not bit-identical and tobytes('A') would report a
        # value difference as a layout defect. Checks (0)/(1)/(2) above stand
        # alone here -- see the docstring for why this is not given a
        # tolerance of its own instead.
        return True, ""
    np_bytes = np_arr.tobytes("A")
    ionp_bytes = ionp_arr.tobytes("A")
    if np_bytes != ionp_bytes:
        return False, (
            f"physical byte layout mismatch (tobytes('A'), the ground "
            f"truth strides/flags only describe): numpy first 24 bytes="
            f"{np_bytes[:24].hex()} anionpy first 24 bytes="
            f"{ionp_bytes[:24].hex()} (shape={np_arr.shape}, "
            f"dtype={np_arr.dtype}, numpy strides={np_strides})"
        )
    return True, ""


def _observable_stride_axes(shape) -> tuple:
    """Positions of `shape` whose stride is physically OBSERVABLE.

    A stride is only meaningful if some pair of adjacent elements along
    that axis actually exists to step between. Two cases where none does:

      * the array holds ZERO elements (any axis of extent 0) -- nothing is
        addressable at all, so no stride in the tuple can be exercised;
      * an axis of extent 1 -- there is no second element to step to.

    Both are numpy's own position, not a convenience. Measured against
    numpy 2.5.1 on 2026-08-04:

        a = np.zeros((2, 0, 3), dtype=bool)
        np.median(a, axis=0).shape    -> (0, 3)
        np.median(a, axis=0).strides  -> (0, 0)
        memoryview(np.median(a, axis=0)).strides -> (24, 8)
        np.zeros((0, 3)).strides      -> (0, 0)     # not (24, 8)

        r = np.quantile(np.array([True]), [0.1, 0.9],
                        keepdims=True, method='lower')
        r.strides -> (1, 0)   memoryview(r).strides -> (1, 1)

    i.e. numpy zeroes out unobservable strides on the object while the C
    buffer protocol's exporter fills them with contiguous values. numpy
    escapes both stride checks below only because `np.asarray(ndarray)` is
    the identity and re-reads the same label, whereas an anionpy result MUST
    cross the buffer protocol to reach `np.asarray` -- so the two sides
    were being read by two different instruments and the difference was
    being reported as an anionpy layout defect. Masking these positions makes
    the sides comparable; every position that can actually be stepped
    along is still compared exactly, and the order='F' relabel bug the
    checks were written for lives entirely on extent>1 axes of a non-empty
    array, so it is still caught.
    """
    if any(n == 0 for n in shape):
        return ()
    return tuple(i for i, n in enumerate(shape) if n > 1)


def compare_values(np_out, ionp_out, atol, rtol, scalar_like=False,
                    ulp_tolerance=None, ulp_justification=None,
                    epsilon_justification=None, source_dtype=None,
                    epsilon_tolerance=None,
                    epsilon_tolerance_justification=None,
                    signed_zero_tie_exempt=False,
                    dtype_override=None, operand_dtype=None) -> tuple[bool, str, bool, float | None]:
    """Returns (ok, detail, tolerant, max_ulp).

    `source_dtype`, when given, is the dtype of the call's OPERAND (see
    `_first_operand_dtype`) -- the per-dtype `ulp_tolerance` lookup below
    uses this key when available, falling back to the RESULT's dtype
    (`np_dtype`) only when no operand dtype could be determined. This
    matters whenever an item's operand dtype and result dtype can differ
    (e.g. `abs`/`absolute` on complex input).

    `dtype_override`/`operand_dtype`, when both given and different, add a
    MORE SPECIFIC composite lookup key (`"{operand}->{override}"`), tried
    before the plain `source_dtype`/result-dtype key -- see
    `_dtype_key_candidates`'s docstring for the defect this closes (an
    explicit `dtype=` override colliding a loose population with a tight
    one under one dict entry, found repeatedly on `nanvar`/`nanstd`). A
    dict with no composite entry is unaffected: lookup falls straight
    through to the exact same key/behavior as before this parameter
    existed.

    `tolerant` is True only when the comparison actually relied on nonzero
    slack to reach `ok=True` (an exact/bit-identical pass is
    `tolerant=False` even when a `ulp_tolerance` or non-zero atol/rtol was
    declared and available -- the ledger must be able to tell "didn't need
    it" from "used it", for EITHER mechanism, not just ULP). `max_ulp` is
    the observed ULP distance for float/complex comparisons (None for
    non-float/complex results, where ULP has no meaning), reported even on
    failure for diagnostics -- and computed for the epsilon (atol/rtol)
    path too, purely as a diagnostic/"was it needed" signal; the epsilon
    path's actual pass/fail decision still comes from np.allclose, never
    from this ULP number.

    Priority when a float/complex result is being graded (see body below):
    `ulp_tolerance` first if declared (takes priority over everything else,
    per the pre-existing 2026-08-01 ULP-tolerance decision); then
    `epsilon_tolerance` (the evidence-gated per-dtype abs/rel mechanism,
    see registry.py -- 2026-08-01 "hardened door / open window" fix); then
    the legacy non-zero atol/rtol path (the WEAK epsilon mechanism, kept
    only because selftest.py/test_ulp_distance.py depend on its exact
    behavior -- see ItemSpec.epsilon_justification's docstring); then the
    strict bit-exact default (ULP distance must be exactly 0) -- which is
    also what an EXPLICIT `atol=0.0, rtol=0.0` declaration routes to, since
    0.0/0.0 is not a tolerance, it is a restatement of bit-exact via the
    legacy call sites.
    """
    if scalar_like:
        ok, detail = _compare_scalar_like(np_out, ionp_out)
        return ok, detail, False, None

    np_dtype = _dtype_of(np_out)
    ionp_dtype = _dtype_of(ionp_out)

    if np_dtype is None or ionp_dtype is None:
        if np_dtype is None and ionp_dtype is None:
            # Both sides genuinely returned a non-array Python object. This
            # was previously an unconditional fail on the theory that numpy
            # never returns a bare scalar from an array op -- true for the
            # vast majority of items, but demonstrably FALSE for
            # `trim_zeros` (real numpy 2.5.1's own source,
            # `_function_base_impl.py`: 1-D non-ndarray input is trimmed via
            # `filt[sl[0]]`, Python's own slicing on the ORIGINAL object,
            # so list-in gives list-out, tuple-in gives tuple-out, and the
            # `axis=()` identity case returns `filt` itself verbatim --
            # see `ionp-py/src/setops.rs::trim_zeros`'s doc comment).
            # Falling through to the same type+value comparison
            # `scalar_like=True` items use (`_compare_scalar_like`) instead
            # of a blanket fail costs nothing for every OTHER item that
            # reaches this branch: it was already an automatic fail before,
            # so this can only turn a false fail into a correct pass (when
            # the values genuinely match) or leave a real mismatch failing
            # (with a more informative message) -- it can never turn a real
            # divergence into a false pass, since `_compare_scalar_like`
            # still requires exact type equality plus `==` value equality.
            ok, detail = _compare_scalar_like(np_out, ionp_out)
            return ok, detail, False, None
        return False, (
            f"dtype presence mismatch: numpy={np_dtype} (type {type(np_out).__name__}) "
            f"anionpy={ionp_dtype} (type {type(ionp_out).__name__})"
        ), False, None

    if np_dtype != ionp_dtype:
        return False, f"dtype mismatch: numpy={np_dtype} anionpy={ionp_dtype}", False, None

    np_shape, ionp_shape = _shape_of(np_out), _shape_of(ionp_out)
    if np_shape != ionp_shape:
        return False, f"shape mismatch: numpy={np_shape} anionpy={ionp_shape}", False, None

    # Return-TYPE check on the array path (2026-08-02, closes the
    # full-reduction 0-d-array-vs-numpy-scalar blindness -- see
    # docs/scalar-return-type-defect.md). Both np.generic (numpy scalar)
    # and np.ndarray carry a `.dtype` and a `.shape` (`()` for a scalar,
    # matching a 0-d array's shape), so the dtype/shape checks above this
    # point cannot distinguish `np.float64(6.0)` from `array(6.)` -- that is
    # exactly the blindness this closes. The rule is deliberately narrow
    # (per the doc's "Open question" section): it does NOT assert
    # `type(np_out) is type(ionp_out)` unconditionally (numpy legitimately
    # returns subclasses in other places, and anionpy arrays are legitimately
    # not `np.ndarray`) -- it only polices the one distinction that matters:
    # numpy-scalar-in must mean matching-numpy-scalar-out, numpy-ndarray-in
    # must mean non-numpy-scalar-out (an anionpy array). `scalar_like=True`
    # items never reach this line (they returned already, above); this is
    # the array path only.
    np_out_is_scalar = isinstance(np_out, np.generic)
    ionp_out_is_scalar = isinstance(ionp_out, np.generic)
    if np_out_is_scalar:
        if type(ionp_out) is not type(np_out):
            return False, (
                f"return-type mismatch: numpy returned numpy scalar "
                f"{type(np_out).__name__}({np_out!r}), anionpy returned "
                f"{type(ionp_out).__name__}({ionp_out!r}) -- a full reduction "
                f"must return the matching numpy scalar type, not an array"
            ), False, None
    elif ionp_out_is_scalar:
        return False, (
            f"return-type mismatch: numpy returned ndarray "
            f"{type(np_out).__name__}(dtype={np_dtype}, shape={np_shape}), anionpy "
            f"returned numpy scalar {type(ionp_out).__name__}({ionp_out!r}) -- "
            f"expected an anionpy array, not a numpy scalar"
        ), False, None

    if np_dtype.kind in "fc":
        _require_ulp_justification(ulp_tolerance, ulp_justification,
                                    context=f"compare_values(dtype={np_dtype})")

        if ulp_tolerance is not None:
            # Per-dtype grading (2026-08-01 fix): `ulp_tolerance` is
            # `{dtype_name: bound}`, not a single item-wide scalar -- a result
            # is graded against ITS OWN declared/evidenced bound, never
            # another dtype's. The grading key is the OPERAND's dtype
            # (`source_dtype`) when known, falling back to the RESULT's own
            # dtype otherwise -- these differ for e.g. `abs`/`absolute` on
            # complex input (complex64 in, float32 out): the evidence is
            # keyed by the swept INPUT dtype (see `_first_operand_dtype`'s
            # docstring and ulp_sweep.py's `_summarize`), so grading must use
            # that same key, not the output's dtype, or complex-input
            # imprecision would either wrongly bleed onto true float32-input
            # results or wrongly get rejected at float32's bit-exact bound.
            # A dtype key with no entry in the dict (not part of this item's
            # declared+evidenced set at all) defaults to 0.0, i.e. bit-exact
            # -- the deliberate, enforced safe default (see registry.py's
            # MIN_ULP_SWEEP_N docstring); it never inherits the bound of
            # whatever other dtype happens to be in the dict.
            #
            # 2026-08-02 input-vs-output-dtype key-split fix: `candidates`
            # tries a composite `"{operand}->{override}"` key FIRST when an
            # explicit `dtype=` override differs from the operand's own
            # dtype, before falling back to the single-name key exactly as
            # before -- see `_dtype_key_candidates`'s docstring.
            candidates, grading_dtype = _dtype_key_candidates(
                source_dtype, np_dtype, dtype_override, operand_dtype)
            per_dtype_tolerance, matched_key = _lookup_dtype_tolerance(
                ulp_tolerance, candidates, default=0.0)
            max_ulp = max_ulp_distance(np_out, ionp_out, np_dtype)
            ok = max_ulp <= per_dtype_tolerance
            tolerant = ok and max_ulp > 0
            if not ok:
                return False, (
                    f"ULP distance {max_ulp} exceeds declared tolerance for dtype "
                    f"{matched_key} ({per_dtype_tolerance}) (justification: "
                    f"{ulp_justification!r}): "
                    f"numpy={np.asarray(np_out).ravel()[:4]!r} "
                    f"anionpy={np.asarray(ionp_out).ravel()[:4]!r}"
                ), False, max_ulp
            return True, "", tolerant, max_ulp

        # epsilon_tolerance: the evidence-gated per-dtype abs/rel mechanism
        # (2026-08-01 "hardened door / open window" fix, see registry.py's
        # ItemSpec.epsilon_tolerance docstring). Same per-dtype grading-key
        # rule as ulp_tolerance directly above (operand dtype when known,
        # else result dtype; no entry for a dtype -> bit-exact default, it
        # never inherits another dtype's slack).
        _require_epsilon_tolerance_justification(
            epsilon_tolerance, epsilon_tolerance_justification,
            context=f"compare_values(dtype={np_dtype})",
        )
        if epsilon_tolerance is not None:
            # 2026-08-02 input-vs-output-dtype key-split fix: same
            # composite-key-first, single-name-fallback lookup as the
            # ulp_tolerance branch above -- see `_dtype_key_candidates`'s
            # docstring. A dict with no composite entry is looked up
            # exactly as before this change.
            candidates, grading_dtype = _dtype_key_candidates(
                source_dtype, np_dtype, dtype_override, operand_dtype)
            entry, matched_key = _lookup_dtype_tolerance(epsilon_tolerance, candidates)
            max_ulp = max_ulp_distance(np_out, ionp_out, np_dtype)
            if entry is not None:
                metric, bound = entry
                dist = (max_abs_distance(np_out, ionp_out) if metric == "abs"
                        else max_rel_distance(np_out, ionp_out))
                ok = dist <= bound
                tolerant = ok and dist > 0
                if not ok:
                    return False, (
                        f"{metric} distance {dist} exceeds declared epsilon_tolerance "
                        f"for dtype {matched_key} ({bound}) (justification: "
                        f"{epsilon_tolerance_justification!r}): "
                        f"numpy={np.asarray(np_out).ravel()[:4]!r} "
                        f"anionpy={np.asarray(ionp_out).ravel()[:4]!r}"
                    ), False, max_ulp
                return True, "", tolerant, max_ulp
            # No candidate key (composite or single-name) has an entry in
            # epsilon_tolerance: falls through to the strict bit-exact
            # default at the bottom of this function, deliberately -- same
            # enforced-safe-default rule as ulp_tolerance (see registry.py's
            # MIN_ULP_SWEEP_N docstring).
            ok = max_ulp == 0
            if not ok:
                return False, (
                    f"dtype {matched_key} has no epsilon_tolerance entry for "
                    f"this item -- graded bit-exact by default and FAILED at ULP "
                    f"distance {max_ulp}: "
                    f"numpy={np.asarray(np_out).ravel()[:4]!r} "
                    f"anionpy={np.asarray(ionp_out).ravel()[:4]!r}"
                ), False, max_ulp
            return True, "", False, 0.0

        # Non-zero atol/rtol: the legacy epsilon path -- np.allclose is a
        # looser, relative-epsilon bar than 1 ULP. Requires a recorded
        # epsilon_justification (structurally enforced by
        # ItemSpec.__post_init__; re-checked here in depth). An EXPLICIT
        # `atol=0.0, rtol=0.0` does NOT take this branch -- see the
        # `epsilon_active` check -- it falls through to the strict
        # bit-exact default below, since 0.0/0.0 is not a tolerance.
        epsilon_active = (
            atol is not None and rtol is not None
            and (atol != 0.0 or rtol != 0.0)
        )
        if epsilon_active:
            _require_epsilon_justification(
                atol, rtol, epsilon_justification,
                context=f"compare_values(dtype={np_dtype})",
            )
            ok = bool(np.allclose(np.asarray(np_out), np.asarray(ionp_out),
                                   atol=atol, rtol=rtol, equal_nan=True))
            if not ok:
                return False, (
                    f"values differ beyond atol={atol} rtol={rtol} "
                    f"(justification: {epsilon_justification!r}): "
                    f"numpy={np.asarray(np_out).ravel()[:4]!r} "
                    f"anionpy={np.asarray(ionp_out).ravel()[:4]!r}"
                ), False, None
            # Diagnostic only: real ULP distance, purely to answer "did
            # this pass actually need the epsilon slack" -- mirrors the
            # ulp_tolerance branch's tolerant = (ok and max_ulp > 0) rule,
            # so "bit-exact vs numpy" means the same thing for both
            # mechanisms in the ledger.
            max_ulp = max_ulp_distance(np_out, ionp_out, np_dtype)
            tolerant = max_ulp > 0
            return True, "", tolerant, max_ulp

        # Neither ulp_tolerance nor a non-zero atol/rtol is declared (this
        # also covers an explicit atol=0.0, rtol=0.0): bit-exact is the
        # only remaining option per the 2026-08-01 decision -- graded via
        # TRUE byte-level comparison (`_bit_exact_equal`, permanent
        # standard as of 2026-08-01, see its docstring), with the
        # `signed_zero_tie_exempt` carve-out available ONLY when the item
        # declares it (registry.py). `max_ulp` is still computed here as a
        # diagnostic (reported in failure details and the ledger's
        # max_ulp_observed), but no longer decides `ok` -- ULP distance
        # cannot see NaN payload divergence at all (see
        # `_bit_exact_equal`'s docstring), so using it as the pass/fail
        # gate was the exact defect this fixes.
        max_ulp = max_ulp_distance(np_out, ionp_out, np_dtype)
        ok, zero_exempt_used = _bit_exact_equal(np_out, ionp_out, signed_zero_tie_exempt=signed_zero_tie_exempt)
        if not ok:
            exempt_note = (
                " (signed_zero_tie_exempt is declared for this item, but the "
                "mismatch was not a zero-sign-only difference)"
                if signed_zero_tie_exempt else ""
            )
            return False, (
                f"no tolerance declared (neither ulp_tolerance nor atol/rtol) for a "
                f"float/complex-producing item -- graded bit-exact (byte-level) by "
                f"default and FAILED{exempt_note}, diagnostic ULP distance {max_ulp}: "
                f"numpy={np.asarray(np_out).ravel()[:4]!r} "
                f"anionpy={np.asarray(ionp_out).ravel()[:4]!r}"
            ), False, max_ulp
        return True, "", zero_exempt_used, 0.0

    ok = bool(np.array_equal(np.asarray(np_out), np.asarray(ionp_out)))
    if not ok:
        return False, f"exact value mismatch (dtype {np_dtype})", False, None
    return True, "", False, None


def compare_multi_output(np_out, ionp_out, atol, rtol, scalar_like=False,
                          ulp_tolerance=None, ulp_justification=None,
                          epsilon_justification=None, source_dtype=None,
                          epsilon_tolerance=None,
                          epsilon_tolerance_justification=None,
                          signed_zero_tie_exempt=False,
                          dtype_override=None, operand_dtype=None) -> tuple[bool, str, bool, float | None]:
    """Tuple-aware comparison for nout>1 ufuncs (divmod/modf/frexp). numpy
    returns an N-tuple of arrays for these; comparing that tuple directly
    with compare_values() would break on the very first `.dtype` lookup
    (tuples have none) and, worse, `np.array_equal`/`np.allclose` silently
    accept mismatched-arity tuples in ways that don't mean what they'd need
    to mean here. Compare arity first, then delegate to compare_values()
    element-wise so each output slot gets the exact same dtype/shape/value
    policy as every single-output item.

    `source_dtype` is threaded straight through to each `compare_values()`
    call unchanged -- it is the CALL's operand dtype (same for every output
    slot of a single call), not per-output-slot information.
    """
    if not isinstance(np_out, tuple):
        return False, f"expected numpy to return a tuple (multi_output item), got {type(np_out).__name__}", False, None
    if not isinstance(ionp_out, tuple):
        return False, (
            f"numpy returned a {len(np_out)}-tuple, anionpy returned "
            f"{type(ionp_out).__name__}({ionp_out!r}) instead of a tuple"
        ), False, None
    if len(np_out) != len(ionp_out):
        return False, f"tuple arity mismatch: numpy={len(np_out)} anionpy={len(ionp_out)}", False, None
    any_tolerant = False
    worst_ulp = None
    for i, (np_item, ionp_item) in enumerate(zip(np_out, ionp_out)):
        ok, detail, tolerant, max_ulp = compare_values(
            np_item, ionp_item, atol, rtol, scalar_like=scalar_like,
            ulp_tolerance=ulp_tolerance, ulp_justification=ulp_justification,
            epsilon_justification=epsilon_justification, source_dtype=source_dtype,
            epsilon_tolerance=epsilon_tolerance,
            epsilon_tolerance_justification=epsilon_tolerance_justification,
            signed_zero_tie_exempt=signed_zero_tie_exempt,
            dtype_override=dtype_override, operand_dtype=operand_dtype,
        )
        if max_ulp is not None:
            worst_ulp = max_ulp if worst_ulp is None else max(worst_ulp, max_ulp)
        if not ok:
            return False, f"output[{i}]: {detail}", False, worst_ulp
        any_tolerant = any_tolerant or tolerant
    return True, "", any_tolerant, worst_ulp


def run_case(np_fn, ionp_fn, args, kwargs, atol, rtol,
             exception_equivalences=None, scalar_like=False, multi_output=False,
             ulp_tolerance=None, ulp_justification=None,
             epsilon_justification=None, epsilon_tolerance=None,
             epsilon_tolerance_justification=None,
             signed_zero_tie_exempt=False,
             check_strides=False, message_pair_ok=None) -> tuple[bool | None, str, bool, float | None]:
    """Returns (ok, detail, tolerant, max_ulp) as documented on each branch
    below, with ONE exception: `ok is None` (rather than True/False) means
    this case is INVALID, not pass/fail -- the anionpy side raised
    `InvalidCase` (see that class's docstring), i.e. it could never have
    reached anionpy at all regardless of correctness. `detail` in that case is
    the human-readable reason, exactly like a failure's detail string, but
    `evaluate()` routes an `ok is None` result to its own `invalid` bucket
    instead of `failed`/`total` -- see evaluate()'s loop below."""
    dtype_override = _explicit_dtype_override(kwargs)
    operand_dtype = _first_operand_dtype(args)
    source_dtype = dtype_override or operand_dtype
    # Each side gets its OWN independent, layout-preserving copy of the
    # arguments -- see `_freshen`'s module docstring. Fixes the argument-
    # reuse contract defect: previously both calls below received the
    # literal SAME `args`/`kwargs` objects, so a mutating op (in-place
    # dunders, `.at()`, `out=`) run on the numpy side first would hand the
    # anionpy side its already-mutated input, and -- since corpus.py's
    # `_views()` derives multiple cases from shared base arrays -- could
    # corrupt sibling cases in the same run.
    np_args, np_kwargs = _freshen(args), _freshen(kwargs)
    ionp_args, ionp_kwargs = _freshen(args), _freshen(kwargs)
    np_out, np_exc = _call(np_fn, np_args, np_kwargs)
    ionp_out, ionp_exc = _call(ionp_fn, ionp_args, ionp_kwargs)

    if isinstance(ionp_exc, InvalidCase):
        # Checked BEFORE the ordinary exception-comparison logic below on
        # purpose: an InvalidCase means the anionpy call never ran the thing
        # under test at all (see that class's docstring), so it must never
        # fall into "numpy didn't raise but anionpy did" and get graded as an
        # ordinary failure -- that would blame anionpy for a case that was
        # never actually exercising it.
        return None, str(ionp_exc), False, None

    if _is_probe_signature_error(np_exc) or _is_probe_signature_error(ionp_exc):
        # VACUOUS-PASS GUARD (added 2026-08-02 after the linalg task).
        #
        # `_eigh_probe`/`_qr_probe` in linalg_cases.py did not declare
        # **kwargs. Every kwargs-bearing case for qr's `mode=` axis therefore
        # died with an IDENTICAL Python-level TypeError on BOTH sides,
        # raised by OUR OWN probe helper, before numpy or anionpy was ever
        # called. The arm below compares exception type + message and graded
        # that as a MATCHING PASS. The case counts looked healthy; the code
        # under test was never executed. A check that cannot fail.
        #
        # That specific instance is fixed, and a corpus-wide detector run on
        # 2026-08-02 found ZERO remaining occurrences. This guard exists
        # because "zero occurrences today" is not the same property as
        # "cannot recur": 24 probe helpers in tests/differential/*_cases.py
        # still lack **kwargs, and each becomes a silent vacuous pass the
        # moment someone adds a kwargs axis to it. The defect is invisible
        # precisely when it is doing the most damage, so it must be caught
        # structurally rather than by remembering to look.
        #
        # Graded InvalidCase, NOT pass and NOT fail: the case genuinely did
        # not exercise anionpy, so failing it would blame anionpy for a corpus
        # bug. InvalidCase is counted and surfaced by the runner, so the
        # case cannot quietly evaporate either -- which is the whole point.
        return None, (
            f"VACUOUS CASE (corpus bug, not an anionpy result): the probe helper "
            f"itself rejected this call before numpy/anionpy ran -- "
            f"{type(np_exc or ionp_exc).__name__}: {np_exc or ionp_exc}. "
            f"Give the probe helper **kwargs (see _svd_probe for the pattern)."
        ), False, None

    if np_exc is not None:
        if ionp_exc is None:
            return False, (
                f"numpy raised {type(np_exc).__name__}({np_exc}), "
                f"anionpy returned {ionp_out!r} instead of raising"
            ), False, None
        allowed = {type(np_exc)}
        if exception_equivalences:
            allowed |= set(exception_equivalences.get(type(np_exc), ()))
        if type(ionp_exc) not in allowed and not _is_ionp_compat_subclass(
            type(ionp_exc), type(np_exc)
        ):
            return False, (
                f"exception type mismatch: numpy raised {type(np_exc).__name__}, "
                f"anionpy raised {type(ionp_exc).__name__}({ionp_exc})"
            ), False, None
        # The message is part of the API. Until 2026-08-01 this arm returned
        # `True, ""` the moment the TYPE matched, so an item could be graded
        # "exact" while raising numpy's exception class with entirely
        # different text. That was corpus blind spot #4 (see PATH-TO-100.md).
        # Measured before enabling this check: 46,757 type-matched exception
        # cases across the corpus, 6,116 of them (13.1%) with diverging text,
        # touching 83 of the 280 then-"exact" items. Those 83 are real gaps,
        # not new breakage -- the ledger was simply not looking at them.
        np_msg = _normalize_exc_message(str(np_exc))
        ionp_msg = _normalize_exc_message(str(ionp_exc))
        if np_msg != ionp_msg:
            # COUPLED-PAIR FALLBACK (2026-08-02): checked ONLY here, after
            # exact message equality has already failed, and ONLY when the
            # spec opted in via `message_pair_ok` (see ItemSpec's docstring
            # for that field, and registry.py's
            # `_comparison_reduce_message_pair_ok` for the one real user).
            # This exists for a single measured shape where numpy's own
            # (exception class, message text) is a COUPLED, two-state,
            # history-dependent pair, not two independently-checkable
            # properties -- so the predicate is handed BOTH raw exception
            # objects and must validate the pair as a whole, not just
            # "close enough" text. It must remain narrower than a plain
            # message-equality relaxation: unscoped items (message_pair_ok
            # is None, the default for the other ~1170+ items) still fail
            # here exactly as before, byte-for-byte.
            if message_pair_ok is not None and message_pair_ok(np_exc, ionp_exc):
                # `tolerant=True`, not False: this pass relied on the
                # coupled cold/warm class+message equivalence, not a
                # byte-exact match to numpy -- coverage.py's build_ledger
                # buckets a passing item as bit-exact vs tolerant using
                # exactly this flag (see ItemResult.tolerant's docstring
                # above), so returning False here would file every one of
                # these passes as bit-exact, an invisible tolerance in the
                # one field whose job is to make tolerances visible (2026-
                # 08-02, coordinator review). `detail` is non-empty on
                # purpose even though this is a PASS, naming the mechanism
                # so a reader grepping failures/details for this shape
                # finds it instead of silence.
                return True, (
                    f"PASS via coupled cold/warm class+message equivalence "
                    f"(comparison-reduce no-loop shape, real numpy is COLD: "
                    f"{type(np_exc).__name__}({np_exc}); anionpy fixed-warm: "
                    f"{type(ionp_exc).__name__}({ionp_exc})) -- see "
                    f"registry.py's _comparison_reduce_message_pair_ok"
                ), True, None
            # repr(), deliberately: one of the first defects this check would
            # have caught was a lone TRAILING SPACE, which is invisible in
            # bare output -- the two messages printed identically and only a
            # byte-level diff distinguished them. Lengths are included for
            # the same reason.
            return False, (
                f"exception message mismatch ({type(np_exc).__name__}):\n"
                f"  numpy ({len(str(np_exc))} chars): {str(np_exc)!r}\n"
                f"  anionpy  ({len(str(ionp_exc))} chars): {str(ionp_exc)!r}"
            ), False, None
        return True, "", False, None

    if ionp_exc is not None:
        return False, (
            f"numpy returned {np_out!r} normally, "
            f"anionpy raised {type(ionp_exc).__name__}({ionp_exc}) instead"
        ), False, None

    if multi_output:
        # check_strides is deliberately NOT applied to multi-output items
        # here (divmod/modf/frexp): see ItemSpec.check_strides's docstring
        # -- out of scope for the order= task, left for a future pass.
        return compare_multi_output(np_out, ionp_out, atol, rtol, scalar_like=scalar_like,
                                     ulp_tolerance=ulp_tolerance, ulp_justification=ulp_justification,
                                     epsilon_justification=epsilon_justification, source_dtype=source_dtype,
                                     epsilon_tolerance=epsilon_tolerance,
                                     epsilon_tolerance_justification=epsilon_tolerance_justification,
                                     signed_zero_tie_exempt=signed_zero_tie_exempt,
                                     dtype_override=dtype_override, operand_dtype=operand_dtype)
    ok, detail, tolerant, max_ulp = compare_values(
        np_out, ionp_out, atol, rtol, scalar_like=scalar_like,
        ulp_tolerance=ulp_tolerance, ulp_justification=ulp_justification,
        epsilon_justification=epsilon_justification, source_dtype=source_dtype,
        epsilon_tolerance=epsilon_tolerance,
        epsilon_tolerance_justification=epsilon_tolerance_justification,
        signed_zero_tie_exempt=signed_zero_tie_exempt,
        dtype_override=dtype_override, operand_dtype=operand_dtype)
    if ok and check_strides and not scalar_like:
        strides_ok, strides_detail = _strides_match(np_out, ionp_out, tolerant=tolerant)
        if not strides_ok:
            return False, strides_detail, False, max_ulp
    return ok, detail, tolerant, max_ulp


def evaluate(spec, cases) -> ItemResult:
    """cases: iterable of (label, args, kwargs)."""
    np_fn = spec.resolve_numpy()
    if np_fn is None:
        return ItemResult(spec.name, "fail", 0, 0, [],
                           "could not resolve numpy reference -- registry bug, not an anionpy gap")

    ionp_fn = spec.resolve_ionp()
    if ionp_fn is None:
        return ItemResult(spec.name, "fail", 0, 0, [],
                           "not implemented on anionpy (absent) -- reported as fail, not a crash")

    ulp_tolerance = getattr(spec, "ulp_tolerance", None)
    ulp_justification = getattr(spec, "ulp_justification", None)
    epsilon_justification = getattr(spec, "epsilon_justification", None)
    epsilon_tolerance = getattr(spec, "epsilon_tolerance", None)
    epsilon_tolerance_justification = getattr(spec, "epsilon_tolerance_justification", None)
    signed_zero_tie_exempt = getattr(spec, "signed_zero_tie_exempt", False)
    check_strides = getattr(spec, "check_strides", False)

    spec_atol = getattr(spec, "atol", None)
    spec_rtol = getattr(spec, "rtol", None)
    if ulp_tolerance is not None:
        mechanism = "ulp"
    elif epsilon_tolerance is not None:
        # Reported as "epsilon" (not a new string) deliberately: this is
        # still the epsilon-family comparison from tools/coverage.py's
        # perspective (magnitude-based, not ULP-based) -- coverage.py
        # buckets on this exact string and is not part of this task's
        # allowed edits, so the mechanism label must stay one of the two
        # values it already understands. The evidence-gate improvement
        # this field represents is visible in the registry (epsilon_sweep)
        # and in this task's report, not by inventing a third bucket here.
        mechanism = "epsilon"
    elif (spec_atol is not None and spec_atol != 0.0) or (spec_rtol is not None and spec_rtol != 0.0):
        mechanism = "epsilon"
    elif signed_zero_tie_exempt:
        # New bucket (2026-08-01, coordinator's signed-zero tie-order
        # report): mutually exclusive with ulp/epsilon by construction
        # (`ItemSpec.__post_init__` refuses to combine them -- see
        # registry.py), reported distinctly so the ledger can tell "bit-
        # exact including NaN payload, EXCEPT tied +-0.0 arrangement" apart
        # from both plain bit-exact and the magnitude-tolerance mechanisms
        # -- coverage.py now understands this third string explicitly (see
        # its own edit).
        mechanism = "tie_exempt"
    elif getattr(spec, "message_pair_ok", None) is not None:
        # New bucket (2026-08-02, coordinator-directed): mirrors the
        # tie_exempt case above -- declared per-item from a static spec
        # field, mutually exclusive with the others in this chain by
        # construction (none of the six comparison ufuncs' ItemSpecs set
        # ulp_tolerance/epsilon_tolerance/atol/rtol/signed_zero_tie_exempt).
        # Only actually credited as tolerant at the ItemResult level if
        # `any_tolerant` ends up True below -- exactly like "ulp"/"epsilon"/
        # "tie_exempt", declaring the mechanism is not the same as having
        # needed it for every case. tools/coverage.py now understands this
        # string explicitly (see its own edit, same day) -- this used to
        # fall through to "none" here, which coverage.py's routing then
        # silently mislabeled as ULP-tolerant (max_ulp=None) whenever a
        # case actually passed via the coupled-pair fallback: a fail-open
        # default in the ledger's own routing, fixed same day in
        # coverage.py alongside this bucket.
        mechanism = "message_pair"
    else:
        mechanism = "none"

    total = 0
    failed = 0
    failures: list[str] = []
    any_tolerant = False
    worst_ulp = None
    invalid = 0
    invalid_cases: list[str] = []
    for label, args, kwargs in cases:
        try:
            ok, detail, tolerant, max_ulp = run_case(
                np_fn, ionp_fn, args, kwargs, spec.atol, spec.rtol,
                spec.exception_equivalences,
                scalar_like=getattr(spec, "scalar_like", False),
                multi_output=getattr(spec, "multi_output", False),
                ulp_tolerance=ulp_tolerance, ulp_justification=ulp_justification,
                epsilon_justification=epsilon_justification,
                epsilon_tolerance=epsilon_tolerance,
                epsilon_tolerance_justification=epsilon_tolerance_justification,
                signed_zero_tie_exempt=signed_zero_tie_exempt,
                message_pair_ok=getattr(spec, "message_pair_ok", None),
                check_strides=check_strides,
            )
        except BaseException as exc:  # noqa: BLE001
            # BaseException, not Exception: a Rust panic surfaces as
            # pyo3_runtime.PanicException, which derives from BaseException.
            # The probe site (see _call) was fixed for this; so must the
            # grading site, because a panic can also fire while COMPARING
            # results (e.g. materialising an anionpy array for the diff), not
            # only while calling. An `except Exception` here lets that panic
            # abort the whole run instead of grading the case as a failure --
            # which is how a declared item can panic and still look green.
            ok, detail, tolerant, max_ulp = False, f"harness raised while running case: {type(exc).__name__}: {exc}", False, None

        if ok is None:
            # InvalidCase (see run_case()'s docstring / InvalidCase's own
            # docstring): this case's receiver could never reach anionpy, so
            # it is excluded from `total`/`failed` entirely -- it must not
            # be credited a pass (nothing was tested) nor blamed as a
            # failure (anionpy never ran). Still recorded, not silently
            # dropped: `invalid_cases` mirrors `failures`' "[label] reason"
            # shape so this is exactly as auditable.
            invalid += 1
            if len(invalid_cases) < MAX_FAILURES_KEPT:
                invalid_cases.append(f"[{label}] {detail}")
            continue

        total += 1
        if max_ulp is not None:
            worst_ulp = max_ulp if worst_ulp is None else max(worst_ulp, max_ulp)
        if not ok:
            failed += 1
            if len(failures) < MAX_FAILURES_KEPT:
                failures.append(f"[{label}] {detail}")
        elif tolerant:
            any_tolerant = True

    verdict = "pass" if failed == 0 and total > 0 else "fail"
    invalid_note = (
        f" ({invalid} case(s) skipped as invalid -- receiver could not reach "
        f"anionpy, see invalid_cases)" if invalid else ""
    )
    if verdict == "pass":
        reason = "ok" + invalid_note
    elif total == 0:
        reason = (
            f"all {invalid} case(s) were invalid (skipped) -- no gradeable "
            f"cases remained" if invalid else "no applicable corpus cases -- registry bug"
        )
    else:
        reason = f"{failed}/{total} cases failed" + invalid_note
    return ItemResult(spec.name, verdict, total, failed, failures, reason,
                       tolerant=(verdict == "pass" and any_tolerant),
                       max_ulp_observed=worst_ulp,
                       mechanism=mechanism,
                       invalid=invalid,
                       invalid_cases=invalid_cases)


def probe_pytest_tester(mod):
    """Compare a module's `.test` attribute against numpy's PytestTester contract.

    THE SIGNATURE IS PART OF THE CONTRACT, AND THREE COPIES OF THIS PROBE
    MISSED IT. Each of `fft.test`, `linalg.test` and `testing.test` had its own
    hand-rolled version that only *invoked* the callable and checked that
    `pytest.main` received an argv containing "-q". All three stayed green
    while anionpy's `_IonpTester` was missing numpy's `doctests`, `coverage`
    and `durations` parameters outright: measured against numpy 2.5.1,
    `numpy.fft.test(durations=5)` bound and `anionpy.fft.test(durations=5)`
    raised `TypeError: got an unexpected keyword argument 'durations'` -- under
    an item declared exact. Comparing the rendered signature catches that whole
    class of divergence rather than the three parameters that happened to be
    missing.

    Full argv is deliberately NOT compared. numpy injects a long list of
    numpy-specific `-W` filters and a numpy-specific `--cov=` path, so argv
    equality would report divergences that are not divergences, and a probe
    that cries wolf gets weakened until it stops biting. What is compared is
    that each keyword reaches argv as the SAME FLAG on both sides.

    This lives in harness.py so there is exactly one copy. If you find yourself
    writing a fourth, put it here instead.
    """
    import inspect
    from unittest import mock

    sig = str(inspect.signature(mod.test.__call__))

    with mock.patch("pytest.main", return_value=0) as m:
        result = mod.test("fast")
    args = m.call_args.args[0] if m.call_args and m.call_args.args else None
    base_ok = bool(args and "-q" in args)

    with mock.patch("pytest.main", return_value=0) as m2:
        mod.test("fast", durations=0, doctests=True, coverage=True)
    kw = m2.call_args.args[0] if m2.call_args and m2.call_args.args else []
    kw_shape = (
        "--durations=0" in kw,
        "--doctest-modules" in kw,
        any(str(a).startswith("--cov=") for a in kw),
    )

    return (m.called, base_ok, result, sig, kw_shape)
