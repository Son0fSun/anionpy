"""Differential specs for the SORTING AND SEARCHING block: `sort,
sort_complex, lexsort, nonzero, flatnonzero, argwhere, extract, where,
searchsorted, nanargmax, nanargmin` (top-level functions) plus
`ndarray.sort, ndarray.argmax, ndarray.argmin, ndarray.nonzero,
ndarray.searchsorted` (ndarray methods) -- backed by `ionp-core/src/sort.rs`
and the corresponding thin wrappers in `ionp-py/src/reductions.rs` /
`ionp-py/src/ndarray_attrs.rs`. `argmin`/`argmax` (top-level) already have a
call-forms spec in `reduction_cases.py`'s `REDUCTION_SPECS["argmin"/"argmax"]`
(kind="method", default path = top-level function since it does not start
with "ndarray.") -- this file does not duplicate those, only declares the
NEW `ndarray.argmax`/`ndarray.argmin` method-path items reusing
`reduction_cases._ARGEXT_FORMS`, plus everything else in the block.

Deliberately OUT OF SCOPE this pass (per the task brief -- algorithm-internal
tie-order/pivot-selection items where numpy's own docs/behavior do not
fully pin down a bit-exact answer independent of the underlying algorithm,
so "matching numpy exactly" is not well-posed without replicating numpy's
introselect/introsort implementation line for line):
`argsort` / `ndarray.argsort`, `partition` / `ndarray.partition`,
`argpartition` / `ndarray.argpartition`. `msort` does not exist in numpy
2.5.1 (removed) -- not a gap, N/A.

SUPERSEDED IN PART (2026-08-03): top-level `argsort` is now IN scope and
declared -- numpy's introsort was replicated line for line, which is what
the paragraph above correctly said the job would take. See the `argsort`
section further down for why "that is what it would take" was recorded as
a reason to skip rather than as a work estimate, and for the two measured
numpy facts that made it tractable. `ndarray.argsort`, `partition` and
`argpartition` are still out of scope (introSELECT is a different
algorithm and has not been replicated).

Three real bugs were found and fixed while building this file's evidence
(all measured against real numpy 2.5.1 via direct interactive probing and
large randomized out-of-corpus sweeps, not assumed from docs):

1. `sort_complex`'s dtype-promotion table was wrong (per-source-width
   promotion, e.g. float32 -> complex64) -- numpy's real rule (its own
   source, `_function_base_impl.py`) is: sort, then if not already complex,
   cast to complex64 ONLY for int8/int16/uint8/uint16 source dtypes, else
   complex128 for every other non-complex source (including float32, which
   is NOT the same-width complex64 a naive width-doubling rule would give).
   Fixed in `ionp-core/src/sort.rs`'s `sort_complex`. Verified 0/4000 on a
   fresh randomized sweep across all 11 real+complex source dtypes.

2. `lexsort`'s `axis=` kwarg rejected ANY value other than the default -1
   with a generic `ValueError` -- real numpy accepts BOTH `axis=-1` and
   `axis=0` (each key is always effectively 1-D for validation, regardless
   of tuple-of-arrays vs single-2D-array call form) and raises the real
   `numpy.exceptions.AxisError` (message: "axis {n} is out of bounds for
   array of dimension 1") for anything else. Fixed in
   `ionp-py/src/reductions.rs`'s `lexsort`. Verified 0/4000.

3. `nanargmax`/`nanargmin` literally skipped NaN entries when picking the
   best candidate -- this LOOKS right but is not what real numpy does.
   numpy's actual source (`_nanfunctions_impl.py`) replaces every NaN with
   `-inf` (max) / `+inf` (min), then calls the ORDINARY `argmax`/`argmin`
   (first-occurrence tie-break), and only afterwards raises `ValueError:
   All-NaN slice encountered` if an entire reduced group was all-NaN. The
   two algorithms diverge whenever a genuine -inf/+inf sits in the same
   group as NaN: verified against real numpy 2.5.1,
   `np.nanargmax([nan, nan, -inf])` returns `0` (the filled-in `-inf` at
   index 0 ties with the real one at index 2; first-occurrence wins), NOT
   `2`, which is what "skip NaN, keep the best real value" (the previous
   anionpy implementation) wrongly returns. Fixed in `ionp-core/src/sort.rs`'s
   `nanargext`. Verified 0/4000 for each of nanargmax/nanargmin.

Also fixed alongside these (same bug CLASS, found while investigating #3):
the 0-d-input + `keepdims=True` shape bug that `reduction_cases.py`'s
module docstring already documents for `argmin`/`argmax` (`ndim > 1`
skipping the `ndim == 0` relabel case) turned out to have a near-identical
but NOT-identical twin in `do_nanargext` (`reductions.rs`): unlike
`argext`, whose raw flattened+keepdims result is always shape `[1]`
regardless of original ndim, `nanargext`'s raw flattened result is a
genuine 0-d scalar even for a 1-D original array, so the `ndim != 1` guard
that fixed `do_argext` would have LEFT `do_nanargext` broken for the
ndim==1 case specifically (verified: this exact regression was caught by
this file's own sweep before being fixed to an unconditional reshape). See
`ionp-py/src/reductions.rs`'s `do_nanargext` for the fix and full comment.

Same file-ownership-fence merge pattern as linalg_cases.py / inplace_cases.py
/ ndarray_attrs_cases.py / reduction_cases.py: a standalone `SORT_SPECS`
dict, merged into registry.REGISTRY at the tail of registry.py with a
collision check.

Three MORE real bugs were found and fixed on 2026-08-01 (coordinator's
harder 2,249-case probe, run against this block after the above was
declared):

4. `nanargmin`/`nanargmax` rejected complex dtypes outright (`TypeError`)
   where real numpy returns a valid index (`np.nanargmin(np.array(2+1j))`
   -> `0`). Fixed in `ionp-core/src/sort.rs`'s `nanargext`, reusing the
   already-existing `complex_cmp` total order plus the same NaN-to-
   sentinel-extreme substitution strategy already used for real floats
   (complex "NaN" = either component is NaN; sentinel = `Complex(-inf,
   -inf)` for max, `Complex(inf, inf)` for min -- verified these correctly
   reproduce numpy's first-occurrence tie-break against a genuine extreme
   value, same as bug #3 above).

5. `lexsort` refused ANY N-d key (`ValueError`) -- real numpy sorts each
   key of any dimensionality independently along `axis` (default -1),
   returning a result of the SAME shape as the keys (verified directly:
   `np.lexsort((a, b), axis=0)` for `(2, 3)`/`(1, 4)`/`(4, 1)`-shaped
   `a`/`b` genuinely works and is NOT equivalent to flattening). Fixed in
   `ionp-core/src/sort.rs`'s `lexsort` (now takes an explicit `axis`
   parameter and iterates every "keep axis" independently) and the
   `ionp-py/src/reductions.rs` wrapper (now validates `axis` against the
   keys' real ndim via the core function, not a hardcoded ndim=1 assumption
   left over from bug #2's original 1-D-only fix).

6. `sort`/`ndarray.sort`/`sort_complex`'s arrangement of `-0.0` vs `+0.0`
   within a TIED run, at larger array lengths (coordinator verified n=32,
   n=200 diverge from the smaller-n behavior my original sweep for bug #1
   above had checked), is decided by numpy's SIMD introsort kernel's
   internal partitioning, not by any comparator -- `-0.0 == +0.0` under
   IEEE comparison, so this is the same fundamental class of
   non-reproducible-arrangement problem as `argsort`'s already-scoped-out
   duplicate-value tie order (see this docstring's own "Deliberately OUT
   OF SCOPE" section above). NOT fixed by reimplementing numpy's SIMD
   internals (explicitly out of scope, same reasoning as `argsort`) --
   instead declared as an explicit, ledger-visible comparison exemption
   (`ItemSpec.signed_zero_tie_exempt`, see registry.py's field docstring
   and harness.py's `_bit_exact_equal`) on exactly these three items.
   Compared with `-0.0 == +0.0` PER ELEMENT (never a blanket pass) --
   NaN payload canonicalization (bug fixed earlier this task, gated on
   `shape[axis] > 1`, see sort.rs's `canonicalize_nans_*`) is completely
   ORTHOGONAL and untouched by this exemption: a NaN bit pattern is never
   `== 0.0`, so the exemption's zero-only carve-out can never apply to
   one, and every other item in this registry (audited 2026-08-01: none)
   declares no comparison exemption of any kind -- this is the ONLY one
   in the entire suite.

   NOTE on `ndarray.sort`: its probe (`_sort_method_probe` below) does NOT
   go through `compare_values`'s array-level `_bit_exact_equal` at all --
   it opts into `scalar_like=True` and instead returns a flat tuple graded
   by plain Python `==` (needed to sidestep bare-NaN's `NaN != NaN`
   problem for `_compare_scalar_like`). The same `-0.0`/`+0.0` exemption
   is reproduced there by normalizing the sign of every exact-zero element
   to `+0.0` independently on each side (numpy's output and anionpy's output
   each normalized on their own, before comparison) -- see
   `_zero_sign_normalize` below; this is semantically identical to
   `_bit_exact_equal`'s per-element carve-out and, being independent of
   the other side, is safe to apply the same way regardless of which side
   is being read. The probe was ALSO switched from `repr()`-per-element to
   `tobytes()` in the same edit -- `repr()` cannot see NaN PAYLOAD bits at
   all (`repr(float('nan'))` is always `"nan"`), which had silently left
   `ndarray.sort` unable to catch a NaN-payload-canonicalization
   regression even before this task; `tobytes()` closes that gap the same
   way the array-level `_bit_exact_equal` rewrite does for `sort` and
   `sort_complex`.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401

import corpus
from registry import CallForm, ItemSpec
from reduction_cases import _ARGEXT_FORMS

# ---------------------------------------------------------------------------
# sort / ndarray.sort -- shared call_forms. Real `np.sort` signature
# (verified): `(a, axis=-1, kind=None, order=None, *, stable=None)`.
# `ndarray.sort` is the same shape but in-place (`(axis=-1, kind=None,
# order=None, *, stable=None)`, no `a`) and returns `None` -- it needs its
# own custom probe (see `_sort_method_probe` below), NOT this shared list,
# since a plain value-equality check of `None == None` would trivially pass
# without ever checking the mutation happened at all (same reasoning
# inplace_cases.py's module docstring gives for the `__i*__` dunders).
#
# `order=` is rejected (`anionpy` has no structured-dtype support at all) --
# verified real numpy raises `TypeError: Cannot specify order when the
# array has no fields.` for a plain (non-structured) array too, i.e. this
# is not even a scope gap for the plain-dtype corpus this harness uses.
# ---------------------------------------------------------------------------

_SIGNED_ZERO_TIE_JUSTIFICATION = (
    "-0.0 == +0.0 under IEEE comparison, so their relative arrangement "
    "within a tied run of a sort is decided by numpy's SIMD introsort "
    "kernel's internal partitioning, not by any comparator -- verified "
    "directly (coordinator, 2026-08-01) to differ by array length (n=32 "
    "vs n=200) on the identical tied input, i.e. not even a fixed "
    "per-numpy-version contract. Same class as argsort's already-scoped-"
    "out duplicate-value tie order; compared with -0.0 == +0.0 per "
    "element, everything else (including NaN payload) stays bit-exact."
)

_SORT_FORMS = [
    CallForm("no_args", lambda arr: ((), {})),
    CallForm("axis0", lambda arr: ((), {"axis": 0}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_last", lambda arr: ((), {"axis": -1}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_none_flattens", lambda arr: ((), {"axis": None})),
    CallForm("kind_quicksort", lambda arr: ((), {"kind": "quicksort"})),
    CallForm("kind_stable", lambda arr: ((), {"kind": "stable"})),
    CallForm("kind_mergesort", lambda arr: ((), {"kind": "mergesort"})),
    CallForm("kind_heapsort", lambda arr: ((), {"kind": "heapsort"})),
    CallForm("stable_true", lambda arr: ((), {"stable": True})),
    CallForm("stable_false", lambda arr: ((), {"stable": False})),
    CallForm("stable_and_kind_conflict_must_raise",
             lambda arr: ((), {"kind": "quicksort", "stable": True})),
    CallForm("order_kwarg_must_raise", lambda arr: ((), {"order": "x"})),
    CallForm("axis_out_of_range_must_raise", lambda arr: ((), {"axis": 100})),
]


def _signed_zero_tie_arrays() -> list[tuple[str, np.ndarray]]:
    """Arrays whose sole purpose is to EXERCISE `signed_zero_tie_exempt`
    (2026-08-01, coordinator's harder probe, bug #6 in this file's module
    docstring): a mix of `-0.0`/`+0.0` (plus a few other tied duplicate
    values, so the zero run is not the ONLY tied run) at the exact lengths
    the coordinator reported diverging from the smaller-n behavior my
    original sweep had checked -- n=32 and n=200 -- across every float
    width (numpy's SIMD sort kernel choice is width-dependent) and complex,
    where the exemption also applies per-component. Without cases like
    these actually present in the corpus, the exemption would be declared
    but never exercised -- these make sure at least one case per relevant
    dtype genuinely has a tied run containing both signs of zero."""
    rng = np.random.default_rng(20260801)
    out: list[tuple[str, np.ndarray]] = []
    for n in (32, 200):
        for dtype in (np.float16, np.float32, np.float64):
            base = rng.choice(np.array([1.0, -1.0, 2.0, -2.0], dtype=dtype), size=n - 4)
            zeros = np.array([0.0, -0.0, 0.0, -0.0], dtype=dtype)
            arr = np.concatenate([base, zeros])
            rng.shuffle(arr)
            out.append((f"signed_zero_tie_n{n}_{np.dtype(dtype).name}", arr))
        for dtype in (np.complex64, np.complex128):
            real_base = rng.choice(np.array([1.0, -1.0, 2.0]), size=n - 4)
            imag_base = rng.choice(np.array([1.0, -1.0, 2.0]), size=n - 4)
            base = (real_base + 1j * imag_base).astype(dtype)
            zeros = np.array([0.0 + 0.0j, -0.0 + 0.0j, 0.0 - 0.0j, -0.0 - 0.0j], dtype=dtype)
            arr = np.concatenate([base, zeros])
            rng.shuffle(arr)
            out.append((f"signed_zero_tie_n{n}_{np.dtype(dtype).name}", arr))
    return out


def _sort_toplevel_cases() -> list:
    cases = []
    for c in corpus.unary_corpus():
        for form in _SORT_FORMS:
            if not form.applicable(c.value):
                continue
            args, kwargs = form.build(c.value)
            cases.append((f"{c.label}/{form.label}", (c.value, *args), kwargs))
    for label, value in _signed_zero_tie_arrays():
        for form in _SORT_FORMS:
            if not form.applicable(value):
                continue
            args, kwargs = form.build(value)
            cases.append((f"{label}/{form.label}", (value, *args), kwargs))
    return cases


def _zero_sign_normalize(out):
    """Canonicalize the sign of every element that is exactly `0.0` (either
    sign) to `+0.0`, leaving every other bit -- including every NaN's full
    payload -- untouched (`NaN == 0.0` is always False, so this can never
    touch a NaN). Applied INDEPENDENTLY to each side's own probe output
    (numpy's and anionpy's, each normalizing only its own array, with no
    knowledge of the other side's values) before taking a byte snapshot for
    comparison -- reproduces `signed_zero_tie_exempt`'s per-element `-0.0 ==
    +0.0` carve-out (see registry.py/harness.py's `_bit_exact_equal`) for
    `ndarray.sort`'s probe, which bypasses that array-level comparison path
    entirely via `scalar_like=True` (see this item's ItemSpec below)."""
    if np.iscomplexobj(out):
        re = np.where(out.real == 0.0, 0.0, out.real)
        im = np.where(out.imag == 0.0, 0.0, out.imag)
        return (re + 1j * im).astype(out.dtype)
    if out.dtype.kind == "f":
        return np.where(out == 0.0, np.zeros_like(out), out)
    return out


def _sort_method_probe(arr, **kwargs):
    """Adapter used as BOTH `numpy_adapter` and `ionp_adapter` for
    `ndarray.sort` -- generic/duck-typed (both `numpy.ndarray` and
    `anionpy.ndarray` expose an in-place `.sort(...)` with an identical
    kwarg surface), same shared-probe pattern `inplace_cases.py`'s
    `_probe` established for the `__i*__` dunders. Returns a flat tuple of
    plain Python values (never a raw ndarray) covering everything an
    in-place, `None`-returning method needs checked: the return value
    really is `None` (not a new array silently returned instead), the
    array's OWN buffer was actually mutated into sorted order (checked by
    reading `arr` back out after the call, not by trusting a separate
    return value), and dtype/shape are unchanged (in-place sort can never
    change either).

    `tobytes()` of a `_zero_sign_normalize`d copy, not raw `.tolist()`/
    `repr()` (2026-08-01 fix, see this file's module docstring bug #6):
    `scalar_like` comparison (harness.py's `_compare_scalar_like`) grades
    with plain Python `==`, which `bytes` objects support directly and
    exactly (no bare-NaN `NaN != NaN` problem the way raw float/complex
    tuple elements would have -- a `bytes` object compares by its byte
    content, not by IEEE float equality, so two independently-produced
    NaN payloads compare equal via `==` if and only if their PAYLOAD BITS
    actually match, which is exactly what bug #3's original NaN-vs-inf fix
    and the earlier NaN-payload-canonicalization fix both need verified).
    The previous `repr()`-per-element approach could not see NaN payload
    bits at all (`repr(float('nan'))` is always `"nan"`), which had
    silently left this item unable to catch a NaN-payload-canonicalization
    regression. `_zero_sign_normalize` is applied first, independently to
    THIS side's own output only, so the tied-run `-0.0`/`+0.0` arrangement
    exemption is reproduced without weakening the byte check for anything
    else."""
    ret = arr.sort(**kwargs)
    out = np.asarray(arr)
    normalized = _zero_sign_normalize(out) if out.dtype.kind in "fc" else out
    return (ret is None, normalized.tobytes(), str(out.dtype), out.shape)


def _sort_method_cases() -> list:
    cases = []
    for c in corpus.unary_corpus():
        for form in _SORT_FORMS:
            if not form.applicable(c.value):
                continue
            _args, kwargs = form.build(c.value)
            if _args:
                continue  # every _SORT_FORMS entry is kwargs-only
            cases.append((f"{c.label}/{form.label}", (c.value,), kwargs))
    for label, value in _signed_zero_tie_arrays():
        for form in _SORT_FORMS:
            if not form.applicable(value):
                continue
            _args, kwargs = form.build(value)
            if _args:
                continue
            cases.append((f"{label}/{form.label}", (value,), kwargs))
    return cases


# ---------------------------------------------------------------------------
# argsort
#
# CORRECTION (2026-08-03). This file's module docstring lists `argsort`
# under "Deliberately OUT OF SCOPE this pass", on the grounds that its
# tie order is "algorithm-internal ... where numpy's own docs/behavior do
# not fully pin down a bit-exact answer independent of the underlying
# algorithm, so 'matching numpy exactly' is not well-posed without
# replicating numpy's introselect/introsort implementation line for line".
#
# The DIAGNOSIS was exactly right and the CONCLUSION did not follow. The
# tie order is indeed algorithm-internal, and matching it does indeed
# require replicating numpy's introsort line for line. That is a
# description of the work, not an argument that the work is ill-posed --
# and unlike `sort`'s signed-zero tied runs (which the coordinator
# measured varying with array LENGTH on identical input, i.e. genuinely
# not a fixed contract), argsort's permutation was measured to be a
# deterministic function of the input alone. So it was replicated: see
# `ionp-core/src/sort.rs`'s `intro_argsort`, and the two non-obvious
# numpy facts recorded there (`kind='heapsort'` does not route to a
# heapsort; the heapsort code is still reached, as the depth-limit
# fallback).
#
# `partition`/`argpartition` remain out of scope -- introSELECT, a
# different algorithm, not covered by this work.
#
# Two things `argsort` does NOT inherit from `sort`, both measured:
#   * 0-d input SUCCEEDS (`np.argsort(np.array(5))` -> `array([0])`)
#     where `np.sort` raises `AxisError` -- hence `argsort_0d` below,
#     which `_SORT_FORMS` has no equivalent of.
#   * `descending=` is a comparator flip, not an output reversal; the
#     `descending_*` forms below are the regression guard for that (they
#     bite: the first implementation reversed the output slots and failed
#     these on every array with a tie or a NaN).
# ---------------------------------------------------------------------------

_ARGSORT_FORMS = _SORT_FORMS + [
    CallForm("descending_true", lambda arr: ((), {"descending": True})),
    CallForm("descending_false", lambda arr: ((), {"descending": False})),
    CallForm("descending_stable", lambda arr: ((), {"descending": True, "stable": True})),
    CallForm("descending_axis0", lambda arr: ((), {"descending": True, "axis": 0}),
             applicable=lambda arr: arr.ndim >= 1),
    CallForm("descending_and_kind_conflict_must_raise",
             lambda arr: ((), {"kind": "quicksort", "descending": True})),
    CallForm("kind_bytes", lambda arr: ((), {"kind": b"quicksort"})),
    CallForm("kind_bytearray_must_raise", lambda arr: ((), {"kind": bytearray(b"quicksort")})),
    CallForm("kind_bogus_must_raise", lambda arr: ((), {"kind": "bogus"})),
]


def _argsort_stress_arrays() -> list[tuple[str, np.ndarray]]:
    """Arrays built to REACH argsort's algorithm branches, which the shared
    `corpus.unary_corpus()` cannot: every array there is far below numpy's
    `SMALL_QUICKSORT` threshold of 15, so on the corpus alone the entire
    quicksort partition loop is dead code and only the insertion-sort tail
    ever runs. A permutation bug in the partition loop would pass a
    corpus-only spec unnoticed.

    So these deliberately straddle the threshold (14/15/16/17) and go well
    past it, with duplicate-heavy values so ties -- the only part of the
    permutation an ordering-only implementation gets wrong -- are dense.
    The `organ_pipe` entries exist for one specific branch: they are the
    only pattern found that drives the partition chain past
    `2 * floor(log2(n))` and actually fires the heapsort fallback (28
    ordinary arrays up to n=20000, including sorted/reversed/all-equal/
    sawtooth/random, never fired it once)."""
    rng = np.random.default_rng(20260803)
    out: list[tuple[str, np.ndarray]] = []
    for n in (14, 15, 16, 17, 40, 333):
        out.append((f"argsort_dup_int_n{n}",
                    rng.integers(0, max(2, n // 4), size=n).astype(np.int64)))
        out.append((f"argsort_dup_float_n{n}",
                    rng.choice(np.array([0.0, -0.0, 1.0, np.nan, np.inf, -np.inf, 2.0, 2.0]), size=n)))
    for n in (200, 1000):
        pipe = np.concatenate([np.arange(n // 2), np.arange(n - n // 2)[::-1]]).astype(np.float64)
        out.append((f"argsort_organ_pipe_n{n}", pipe))
    re = rng.choice(np.array([0.0, 1.0, np.nan, 2.0]), size=64)
    im = rng.choice(np.array([0.0, 1.0, np.nan, -0.0]), size=64)
    out.append(("argsort_dup_complex_n64", (re + 1j * im).astype(np.complex128)))
    out.append(("argsort_all_equal_n64", np.zeros(64)))
    out.append(("argsort_2d_dup", rng.integers(0, 3, size=(6, 7)).astype(np.int32)))
    return out


def _argsort_cases() -> list:
    cases = []
    for c in corpus.unary_corpus():
        for form in _ARGSORT_FORMS:
            if not form.applicable(c.value):
                continue
            args, kwargs = form.build(c.value)
            cases.append((f"{c.label}/{form.label}", (c.value, *args), kwargs))
    for label, value in _argsort_stress_arrays():
        for form in _ARGSORT_FORMS:
            if not form.applicable(value):
                continue
            args, kwargs = form.build(value)
            cases.append((f"{label}/{form.label}", (value, *args), kwargs))
    # 0-d: argsort accepts it, sort does not. Kept as explicit one-offs
    # rather than a corpus entry so the divergence from `sort` is visible.
    for label, kwargs in (("argsort_0d", {}), ("argsort_0d_axis0", {"axis": 0}),
                          ("argsort_0d_axis_none", {"axis": None}),
                          ("argsort_0d_axis1_must_raise", {"axis": 1})):
        cases.append((label, (np.array(5),), kwargs))
    return cases


def _argsort_method_probe(arr, **kwargs):
    """Adapter used as BOTH `numpy_adapter` and `ionp_adapter` for
    `ndarray.argsort` -- duck-typed, the same shared-probe pattern
    `_sort_method_probe` uses. Unlike that one it needs no byte snapshot
    or mutation check: `argsort` is not in-place and returns a real array,
    so the harness's ordinary array comparison applies directly."""
    return arr.argsort(**kwargs)


def _argsort_method_cases() -> list:
    """Same case set as the top-level function's, minus nothing -- the
    method's surface was measured to be identical (including `axis=None`,
    0-d acceptance, and every error message). Sharing the builder is the
    point: if the two ever diverge, this spec fails rather than quietly
    testing a weaker contract."""
    return _argsort_cases()


# ---------------------------------------------------------------------------
# sort_complex -- single positional arg, no other kwargs in real numpy's
# signature (verified: `sort_complex(a)`).
# ---------------------------------------------------------------------------

_SORT_COMPLEX_FORMS = [CallForm("no_args", lambda arr: ((), {}))]


def _sort_complex_cases() -> list:
    """`sort_complex` was `kind="method"` (cases sourced automatically from
    `corpus.unary_corpus()` x `_SORT_COMPLEX_FORMS`, see run.py's
    `build_cases`) until 2026-08-01, when it needed its own signed-zero
    tied-run cases (bug #6, this file's module docstring) to actually
    EXERCISE `signed_zero_tie_exempt` -- `corpus.unary_corpus()` is shared
    by every item in the registry, so extending it directly would have
    been a much wider blast radius than this one item's need. Switched to
    `kind="custom"`, reproducing the exact same corpus x call_forms
    iteration `run.py`'s `kind="method"` path did, plus the same
    `_signed_zero_tie_arrays()` used by `sort`/`ndarray.sort` above."""
    cases = []
    for c in corpus.unary_corpus():
        for form in _SORT_COMPLEX_FORMS:
            if not form.applicable(c.value):
                continue
            args, kwargs = form.build(c.value)
            cases.append((f"{c.label}/{form.label}", (c.value, *args), kwargs))
    for label, value in _signed_zero_tie_arrays():
        for form in _SORT_COMPLEX_FORMS:
            if not form.applicable(value):
                continue
            args, kwargs = form.build(value)
            cases.append((f"{label}/{form.label}", (value, *args), kwargs))
    return cases

# ---------------------------------------------------------------------------
# lexsort -- custom cases: both the tuple-of-1-D-arrays and single-2-D-array
# call forms, both valid `axis` values (-1 and 0, per finding #2 above,
# both behaving identically), and out-of-range axis as a must-raise form.
# Built from a small independent corpus (not corpus.unary_corpus() --
# lexsort needs MULTIPLE equal-length 1-D keys per case, which that corpus
# has no notion of), same reasoning inplace_cases.py gives for its own
# independent corpus.
# ---------------------------------------------------------------------------

_LEXSORT_SEED = 20260801


def _lexsort_custom_cases() -> list:
    rng = np.random.default_rng(_LEXSORT_SEED)
    cases = []
    dtypes = [np.int32, np.int64, np.float32, np.float64, np.uint8, np.bool_]
    lengths = [0, 1, 5, 20]
    for dt in dtypes:
        for n in lengths:
            for n_keys in (1, 2, 3):
                if dt == np.bool_:
                    keys = [rng.integers(0, 2, size=n).astype(dt) for _ in range(n_keys)]
                elif np.issubdtype(dt, np.integer):
                    # heavy duplicate density on purpose -- exercises the
                    # tie-break-by-earlier-key contract, not just distinct
                    # values.
                    keys = [rng.integers(-3, 3, size=n).astype(dt) for _ in range(n_keys)]
                else:
                    keys = [rng.choice([0.0, 1.0, -1.0, 2.5, np.nan], size=n).astype(dt)
                            for _ in range(n_keys)]
                label_base = f"{np.dtype(dt).name}/n{n}/k{n_keys}"
                # tuple-of-1-D-arrays call form, default axis
                cases.append((f"{label_base}/tuple_default_axis", (tuple(keys),), {}))
                # tuple-of-1-D-arrays call form, axis=0 and axis=-1 (both valid)
                cases.append((f"{label_base}/tuple_axis0", (tuple(keys),), {"axis": 0}))
                cases.append((f"{label_base}/tuple_axis_neg1", (tuple(keys),), {"axis": -1}))
                # out-of-range axis must raise the real AxisError
                cases.append((f"{label_base}/tuple_axis_bad", (tuple(keys),), {"axis": 5}))
                if n_keys >= 2 and n > 0:
                    # single-2-D-array call form (stack keys as rows, numpy's
                    # own documented equivalent input shape)
                    stacked = np.stack(keys, axis=0)
                    cases.append((f"{label_base}/stacked_default_axis", (stacked,), {}))
                    cases.append((f"{label_base}/stacked_axis0", (stacked,), {"axis": 0}))
    return cases


def _lexsort_np(keys, axis=-1):
    return np.lexsort(keys, axis=axis)


def _lexsort_ionp(keys, axis=-1):
    import anionpy
    if isinstance(keys, tuple):
        conv = tuple(anionpy.array(k) for k in keys)
    else:
        conv = anionpy.array(keys)
    return anionpy.lexsort(conv, axis=axis)


# ---------------------------------------------------------------------------
# nonzero / ndarray.nonzero -- returns a TUPLE of 1-D index arrays, which
# `harness.compare_values` cannot grade directly (a tuple has no `.dtype`).
# Adapter reduces the tuple to a flat, directly-comparable tuple of plain
# values: for each axis, the index array's dtype name + shape + value list
# -- covers dtype/shape/value exactly like the ordinary array path, just
# pre-flattened past the "which axis" tuple layer.
# ---------------------------------------------------------------------------


def _nonzero_probe_np(arr):
    cols = np.nonzero(arr)
    return tuple((str(c.dtype), c.shape, c.tolist()) for c in cols)


def _nonzero_probe_ionp(arr):
    import anionpy
    cols = anionpy.nonzero(arr)
    return tuple((str(np.asarray(c).dtype), np.asarray(c).shape, np.asarray(c).tolist()) for c in cols)


def _nonzero_method_probe_np(arr):
    cols = arr.nonzero()
    return tuple((str(c.dtype), c.shape, c.tolist()) for c in cols)


def _nonzero_method_probe_ionp(arr):
    cols = arr.nonzero()
    return tuple((str(np.asarray(c).dtype), np.asarray(c).shape, np.asarray(c).tolist()) for c in cols)


def _nonzero_cases() -> list:
    return [(c.label, (c.value,), {}) for c in corpus.unary_corpus() if c.value.ndim > 0 or True]


# ---------------------------------------------------------------------------
# flatnonzero / argwhere -- plain single-array-in, single-array-out, no
# extra kwargs in real numpy's signature.
# ---------------------------------------------------------------------------

_NO_ARGS_FORM = [CallForm("no_args", lambda arr: ((), {}))]

# ---------------------------------------------------------------------------
# extract(condition, arr) -- custom cases: condition is a same-shape boolean
# mask derived from each corpus array (not itself part of corpus.py, which
# has no notion of "a boolean array shaped like this other array").
# ---------------------------------------------------------------------------

_EXTRACT_SEED = 20260801


def _extract_cases() -> list:
    rng = np.random.default_rng(_EXTRACT_SEED)
    cases = []
    for c in corpus.unary_corpus():
        arr = c.value
        cond = rng.integers(0, 2, size=arr.shape).astype(bool)
        cases.append((f"{c.label}/random_mask", (cond, arr), {}))
        # all-true / all-false edge masks
        cases.append((f"{c.label}/all_true", (np.ones(arr.shape, dtype=bool), arr), {}))
        cases.append((f"{c.label}/all_false", (np.zeros(arr.shape, dtype=bool), arr), {}))
    return cases


# ---------------------------------------------------------------------------
# where -- both the 1-arg (alias for nonzero, tuple-returning -> needs the
# same tuple-flattening probe as nonzero) and 3-arg (elementwise ternary
# select, ordinary array-returning) forms.
# ---------------------------------------------------------------------------


def _where_1arg_np(cond):
    cols = np.where(cond)
    return tuple((str(c.dtype), c.shape, c.tolist()) for c in cols)


def _where_1arg_ionp(cond):
    import anionpy
    cols = anionpy.where(cond)
    return tuple((str(np.asarray(c).dtype), np.asarray(c).shape, np.asarray(c).tolist()) for c in cols)


def _where_cases() -> list:
    rng = np.random.default_rng(_EXTRACT_SEED)
    cases = []
    for c in corpus.unary_corpus():
        arr = c.value
        cond = rng.integers(0, 2, size=arr.shape).astype(bool)
        other = arr  # reuse same array as the "y" branch, dtype-compatible by construction
        cases.append((f"{c.label}/select_3arg", (cond, arr, other), {}))
    # 1-arg tuple-returning alias form gets its own item (see SORT_SPECS
    # below: "where_nonzero_alias") since it has a structurally different
    # return type (tuple vs array) than the 3-arg select form, same
    # reasoning `nonzero`'s own item needed a dedicated probe.
    return cases


def _where_1arg_cases() -> list:
    return [(c.label, (c.value,), {}) for c in corpus.unary_corpus()]


def _where_weak_scalar_cases() -> list:
    """`np.where(cond, x, y)` where exactly one of `x`/`y` is a BARE PYTHON
    scalar. This is the NEP 50 weak-scalar path, and until 2026-08-04 it was
    a hole in this corpus: `_where_cases` above only ever passes two arrays,
    so `where` sat declared "exact" while returning the wrong DTYPE for 14
    of 45 (dtype x scalar-kind) combinations -- values agreed, only the
    dtype was wrong, which is precisely why nothing caught it.

    Measured 2026-08-04, numpy 2.5.1, `cond = np.array([True, False, True])`:
        np.where(cond, 1,     int8_arr)      -> int8       (anionpy gave int64)
        np.where(cond, 1.0,   float32_arr)   -> float32    (anionpy gave float64)
        np.where(cond, 1e-20, complex64_arr) -> complex64  (anionpy gave complex128)
        np.where(cond, -7,    uint8_arr)     -> OverflowError: Python integer
                                                -7 out of bounds for uint8
        np.where(cond, 1e20,  float16_arr)   -> float16 [inf, ...] (no raise)
    Both operand POSITIONS are covered because the promotion is symmetric
    and a fix that only handled one side would still pass a one-sided grid.
    The out-of-range integer rows are deliberately included: they pin the
    STRICT (non-`relaxed`) overflow behaviour, which is what distinguishes
    this from the compare-op weak-scalar path that must NOT raise.
    """
    scalars = [
        ("int_1", 1), ("int_0", 0), ("int_neg7", -7), ("int_300", 300),
        ("int_2p40", 2 ** 40), ("float_1", 1.0), ("float_1p5", 1.5),
        ("float_tiny", 1e-20), ("float_huge", 1e20),
        ("bool_T", True), ("bool_F", False), ("complex", 1 + 2j),
    ]
    dtypes = ["bool", "int8", "uint8", "int16", "uint16", "int32", "int64",
              "uint64", "float16", "float32", "float64", "complex64",
              "complex128"]
    cond = np.array([True, False, True])
    cases = []
    for dt in dtypes:
        arr = np.array([1, 0, 2], dtype=dt)
        for lbl, sc in scalars:
            cases.append((f"weak/{dt}/{lbl}/x_scalar", (cond, sc, arr), {}))
            cases.append((f"weak/{dt}/{lbl}/y_scalar", (cond, arr, sc), {}))
    return cases


def _where_mismatched_xy_must_raise_cases() -> list:
    """`np.where(cond, x)` (exactly one of x/y given) must raise
    `ValueError` -- verified against real numpy 2.5.1: "either both or
    neither of x and y should be given"."""
    cases = []
    for c in corpus.unary_corpus()[:20]:
        cases.append((f"{c.label}/x_only_must_raise", (c.value, c.value), {}))
    return cases


# ---------------------------------------------------------------------------
# searchsorted / ndarray.searchsorted -- custom cases: independent small
# corpus of (sorted_or_unsorted_array, values, side, sorter) tuples --
# corpus.py's arrays are not "a base array plus a related values array",
# which this signature genuinely needs two of.
# ---------------------------------------------------------------------------

_SEARCHSORTED_SEED = 20260801


def _searchsorted_base_cases() -> list:
    rng = np.random.default_rng(_SEARCHSORTED_SEED)
    dtypes = [np.int32, np.int64, np.float32, np.float64, np.uint8, np.uint16]
    lengths = [0, 1, 5, 20]
    val_lengths = [0, 1, 3, 10]
    cases = []
    for dt in dtypes:
        for n in lengths:
            base_sorted = np.sort(rng.integers(-10, 10, size=n).astype(dt)) if np.issubdtype(dt, np.integer) \
                else np.sort(rng.choice([-2.5, -1, 0, 0.5, 1, 2.5, 3], size=n).astype(dt))
            for nv in val_lengths:
                values = (rng.integers(-12, 12, size=nv).astype(dt) if np.issubdtype(dt, np.integer)
                           else rng.choice([-3, -2.5, -1, 0, 0.5, 1, 2.5, 3, 4], size=nv).astype(dt))
                for side in ("left", "right"):
                    cases.append((f"{np.dtype(dt).name}/n{n}/v{nv}/{side}",
                                  (base_sorted, values), {"side": side}))
                # bogus side must raise
                cases.append((f"{np.dtype(dt).name}/n{n}/v{nv}/bad_side",
                              (base_sorted, values), {"side": "middle"}))
                # sorter= path: shuffle base, provide the sorter that
                # recovers sortedness
                if n > 0:
                    perm = rng.permutation(n)
                    unsorted = base_sorted[perm]
                    sorter = np.argsort(unsorted, kind="stable")
                    cases.append((f"{np.dtype(dt).name}/n{n}/v{nv}/sorter",
                                  (unsorted, values), {"side": "left", "sorter": sorter}))
    cases.extend(_searchsorted_promote_cases())
    cases.extend(_searchsorted_sorter_contract_cases())
    return cases


def _searchsorted_sorter_contract_cases() -> list:
    """Guard for the three `sorter=` defects fixed 2026-08-04.

    The `sorter` cases in the grid above only ever pass a VALID permutation
    (`np.argsort(unsorted, kind="stable")`), so every rejection path and the
    laziness of numpy's bounds check were entirely unexercised. What was
    actually wrong, measured against numpy 2.5.1:

      1. every argument-shape rejection collapsed into one home-grown
         message. numpy has three, with a measured precedence of
         ndim -> dtype -> size (a 2-D FLOAT sorter reports the ndim
         complaint, not the dtype one);
      2. an out-of-range entry raised `IndexError: index 5 is out of bounds
         for axis 0 with size 3` instead of numpy's
         `ValueError: Sorter index out of range.`;
      3. and it raised EAGERLY. numpy's check sits inside the search loop,
         so an out-of-range entry the bisection never dereferences is never
         reported at all. That is what the `oob/*` pairs below pin: the same
         `sorter=[5, 0, 1]` must RAISE against `[3, 1, 2]` (bisect touches
         position 0) and must SUCCEED, returning 3, against
         `[True, False, True]` (bisect touches positions 1 and 2 only).
         A correct-message-but-eager fix passes case (2) and fails this one,
         which is exactly why both spellings are here.

    Negative entries are included for the same reason: they used to be
    silently wrapped Python-style (`raw + n`), so `sorter=[-1, 0, 1]` quietly
    searched a different permutation instead of raising.
    """
    a = np.array([3, 1, 2], dtype=np.int64)
    a_bool = np.array([True, False, True])
    out = []
    for side in ("left", "right"):
        for lbl, sorter in [
            ("size_short", np.array([0, 1])),
            ("size_long", np.array([0, 1, 2, 0])),
            ("ndim2", np.array([[0, 1, 2]])),
            ("ndim2_float", np.array([[0.0, 1.0, 2.0]])),
            ("ndim2_wrongsize", np.array([[0, 1]])),
            ("float", np.array([0.0, 1.0, 2.0])),
            ("float_wrongsize", np.array([0.0, 1.0])),
            ("bool", np.array([True, False, True])),
            ("complex", np.array([0j, 1j, 2j])),
            ("negative", np.array([-1, 0, 1])),
            ("oob", np.array([5, 0, 1])),
            ("oob_high", np.array([1, 2, 99])),
            ("valid_u8", np.array([1, 2, 0], dtype=np.uint8)),
            ("valid_i32", np.array([1, 2, 0], dtype=np.int32)),
            # uint64 is REJECTED (not safely castable to intp), and with a
            # `ValueError` even though the ndim rejection uses the same
            # wording under `TypeError` -- see `ionp-core/src/sort.rs`.
            ("reject_u64", np.array([1, 2, 0], dtype=np.uint64)),
            ("reject_u64_2d", np.array([[1, 2, 0]], dtype=np.uint64)),
            ("reject_u64_wrongsize", np.array([1, 2], dtype=np.uint64)),
            ("valid_u16", np.array([1, 2, 0], dtype=np.uint16)),
            ("valid_u32", np.array([1, 2, 0], dtype=np.uint32)),
            ("valid_i8", np.array([1, 2, 0], dtype=np.int8)),
            ("valid_i16", np.array([1, 2, 0], dtype=np.int16)),
        ]:
            out.append((f"sorter_contract/{lbl}/{side}", (a, 2), {"side": side, "sorter": sorter}))
            # Same sorter, a base whose bisection touches DIFFERENT positions
            # -- this is the pair that distinguishes lazy from eager checking.
            out.append((f"sorter_contract/{lbl}/{side}/bool_base",
                        (a_bool, 2), {"side": side, "sorter": sorter}))
    # Empty `a` with an empty sorter is legal and must not trip the size or
    # bounds checks (the search loop never runs, so laziness makes the
    # bounds check vacuous here by construction).
    for side in ("left", "right"):
        out.append((f"sorter_contract/empty/{side}",
                    (np.array([], dtype=np.int64), 2),
                    {"side": side, "sorter": np.array([], dtype=np.int64)}))
        out.append((f"sorter_contract/empty_mismatch/{side}",
                    (np.array([], dtype=np.int64), 2),
                    {"side": side, "sorter": np.array([0, 1])}))
    return out


def _searchsorted_promote_cases() -> list:
    """Guard for the PHANTOM fixed 2026-08-04: `a` and `v` at DIFFERENT dtypes.

    The grid above builds `values` with `.astype(dt)` -- the SAME dtype as the
    base array, every single case. That is precisely why `searchsorted` could
    sit declared "exact" while raising `TypeError: searchsorted: value dtype
    mismatch` for 154 of 192 mixed-dtype cells: the corpus never once crossed
    the two dtypes. numpy has no matching rule at all, it promotes `a` and `v`
    to their common `result_type` and searches there.

    Three axes, deliberately kept separate because they are three different
    promotion paths in numpy and a fix for one does not imply the others:

      * bare Python scalars (`3`, `2.5`, `2+2j`, `True`) -- NEP 50 WEAK, take
        the array's dtype family;
      * numpy scalars (`np.float64(2.5)`, `np.int8(2)`, ...) -- STRONG, and
        note that `np.float64`/`np.int64`/`np.bool_` are genuine subclasses of
        Python `float`/`int`/`bool`, which is exactly the trap that produced a
        separate bug in `clip` on the same day;
      * 1-d and 0-d arrays -- STRONG, ordinary `promote_dtype`.

    The out-of-range integers (`300`, `256`, `-200`, `-7`, `2**63`) are here on
    purpose and are NOT decoration: they pin the RELAXED overflow behaviour,
    which is measurably different from `where`'s and `clip`'s. `where` and
    `clip` raise `OverflowError` for a weak int that does not fit; this one
    does not -- it compares by VALUE at a width that holds it. Measured
    2026-08-04 against numpy 2.5.1:

        np.searchsorted(np.array([1,2,3], np.int8),   300)  -> 3  (not 44)
        np.searchsorted(np.array([1,2,3], np.int8),   256)  -> 3  (not 0)
        np.searchsorted(np.array([1,2,3], np.int8),  -200)  -> 0
        np.searchsorted(np.array([1,2,3], np.uint8),  -7)   -> 0  (not 249)
        np.searchsorted(np.array([1,2,3], np.int64), 2**63) -> 3

    Each parenthesised number is what a NARROWING (strict) extraction would
    have produced, so these rows fail loudly if anyone swaps the `relaxed`
    flag in `reductions.rs::searchsorted` back to `false`.

    The mixed-STRONG rows matter separately from the weak ones: promoting the
    weak scalar against the array is not enough on its own, because two strong
    operands can still arrive at different dtypes and the kernel needs them
    equalised. `np.searchsorted(np.array([1,2,3], np.uint64), np.int64(-5))`
    is `0` -- both sides go to float64, i.e. plain `promote_dtype(U64, I64)`.
    """
    dtypes = ["bool", "int8", "uint8", "int16", "uint16", "int32", "int64",
              "uint64", "float16", "float32", "float64", "complex64",
              "complex128"]
    values = [
        ("py_bool", True), ("py_int", 3), ("py_int_big", 300),
        ("py_int_neg", -7), ("py_int_huge", 2**63), ("py_float", 2.5),
        ("py_float_big", 1e20), ("py_complex", 1 + 2j),
        ("np_bool", np.bool_(1)), ("np_int8", np.int8(2)),
        ("np_int64", np.int64(2)), ("np_uint8", np.uint8(2)),
        ("np_uint64", np.uint64(2)), ("np_float16", np.float16(2.5)),
        ("np_float32", np.float32(2.5)), ("np_float64", np.float64(2.5)),
        ("np_complex64", np.complex64(1 + 2j)),
        ("np_complex128", np.complex128(1 + 2j)),
        ("arr_int8", np.array([0, 2, 9], np.int8)),
        ("arr_float64", np.array([0.5, 2.5])),
        ("arr_complex64", np.array([1 + 2j, 0], np.complex64)),
        ("arr_bool", np.array([True, False])),
        ("arr0d_float64", np.array(2.5)),
        ("arr0d_int8", np.array(2, np.int8)),
    ]
    out = []
    for name in dtypes:
        base = (np.array([False, True, True]) if name == "bool"
                else np.array([1, 2, 3], dtype=name))
        for vlabel, v in values:
            for side in ("left", "right"):
                out.append((f"promote/{name}/{vlabel}/{side}",
                            (base, v), {"side": side}))
    return out


def _searchsorted_toplevel_cases() -> list:
    return [(label, (a, v), kw) for label, (a, v), kw in _searchsorted_base_cases()]


def _searchsorted_method_cases() -> list:
    # Method form: first positional becomes `self` (the sorted/base array),
    # `v` moves to the method's own first positional arg -- identical
    # arg/kwarg content, just resolved via `arr.searchsorted(v, ...)`
    # instead of `np.searchsorted(arr, v, ...)`.
    return [(label, (a, v), kw) for label, (a, v), kw in _searchsorted_base_cases()]


# ---------------------------------------------------------------------------
# nanargmax / nanargmin -- reuse the shape of _ARGEXT_FORMS (axis/keepdims/
# out-of-range-axis combos), applied over a corpus that actually contains
# NaN (corpus.unary_corpus()'s `_special_float_values()` arrays already
# include NaN -- see corpus.py) plus a dedicated all-NaN-slice corpus to
# exercise the `ValueError: All-NaN slice encountered` must-raise path,
# which the general corpus has no guaranteed hit for.
# ---------------------------------------------------------------------------

_NANARG_FORMS = [
    CallForm("no_args", lambda arr: ((), {})),
    CallForm("axis_none_explicit", lambda arr: ((), {"axis": None})),
    CallForm("axis0", lambda arr: ((), {"axis": 0}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_last", lambda arr: ((), {"axis": -1}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("keepdims_axis0", lambda arr: ((), {"axis": 0, "keepdims": True}),
             applicable=lambda arr: arr.ndim >= 1),
    CallForm("keepdims_axis_none", lambda arr: ((), {"axis": None, "keepdims": True})),
    CallForm("axis_out_of_range_must_raise", lambda arr: ((), {"axis": 100})),
]


def _nanarg_float_corpus() -> list:
    # Restrict to real (non-complex, non-bool) dtypes -- nanargmax/nanargmin
    # reject complex (`TypeError`, verified) and the general unary_corpus
    # includes complex/bool entries that would only exercise the rejection
    # path, not the NaN-handling logic this item is actually about. Bool
    # and every int dtype ARE kept (nanargmax/nanargmin accept them, and
    # int64 already has a dedicated regression case below for issue #3).
    out = []
    for c in corpus.unary_corpus():
        if c.value.dtype.kind == "c":
            continue
        out.append(c)
    return out


def _nanarg_custom_cases() -> list:
    cases = []
    for c in _nanarg_float_corpus():
        for form in _NANARG_FORMS:
            if not form.applicable(c.value):
                continue
            args, kwargs = form.build(c.value)
            cases.append((f"{c.label}/{form.label}", (c.value, *args), kwargs))
    # Dedicated all-NaN-slice regression cases (must raise ValueError).
    all_nan_1d = np.full((5,), np.nan, dtype=np.float64)
    all_nan_2d = np.full((3, 4), np.nan, dtype=np.float32)
    cases.append(("all_nan_1d/no_args", (all_nan_1d,), {}))
    cases.append(("all_nan_2d/axis0", (all_nan_2d,), {"axis": 0}))
    # Regression case for bug #3 (this file's module docstring): a NaN tied
    # against a genuine -inf/+inf in the same reduced group must resolve to
    # the FIRST occurrence, exactly like plain argmax/argmin's tie-break --
    # `np.nanargmax([nan, nan, -inf])` == 0, not 2.
    cases.append(("nan_ties_with_neg_inf/nanargmax_regression",
                  (np.array([np.nan, np.nan, -np.inf]),), {}))
    cases.append(("nan_ties_with_pos_inf/nanargmin_regression",
                  (np.array([np.nan, np.nan, np.inf]),), {}))
    cases.append(("neg_inf_first_then_nan/nanargmax_regression",
                  (np.array([-np.inf, np.nan, np.nan]),), {}))
    return cases


# ---------------------------------------------------------------------------
# Assembled specs
# ---------------------------------------------------------------------------

SORT_SPECS: dict[str, ItemSpec] = {
    "sort": ItemSpec(
        name="sort", kind="custom",
        custom_cases=_sort_toplevel_cases,
        atol=0.0, rtol=0.0,
        signed_zero_tie_exempt=True,
        signed_zero_tie_exempt_justification=_SIGNED_ZERO_TIE_JUSTIFICATION,
    ),
    # No `signed_zero_tie_exempt` here, deliberately. `sort`'s exemption
    # exists because the arrangement of a tied `-0.0`/`+0.0` run in a VALUE
    # output was measured to depend on array length inside numpy's SIMD
    # kernel. argsort does not go through those kernels, and its
    # permutation on the same tied input was measured to be a
    # deterministic function of the input -- so the exemption would be
    # unearned slack, and the corpus above carries `-0.0`/`+0.0` ties at
    # n up to 333 specifically so that claim is under test rather than
    # merely asserted.
    "argsort": ItemSpec(
        name="argsort", kind="custom",
        custom_cases=_argsort_cases,
        atol=0.0, rtol=0.0,
    ),
    "sort_complex": ItemSpec(
        name="sort_complex", kind="custom",
        custom_cases=_sort_complex_cases,
        atol=0.0, rtol=0.0,
        signed_zero_tie_exempt=True,
        signed_zero_tie_exempt_justification=_SIGNED_ZERO_TIE_JUSTIFICATION,
    ),
    "lexsort": ItemSpec(
        name="lexsort", kind="custom",
        custom_cases=_lexsort_custom_cases,
        numpy_adapter=lambda keys, axis=-1: _lexsort_np(keys, axis=axis),
        ionp_adapter=lambda keys, axis=-1: _lexsort_ionp(keys, axis=axis),
        convert_ionp_args=False,
        atol=0.0, rtol=0.0,
    ),
    "nonzero": ItemSpec(
        name="nonzero", kind="custom",
        custom_cases=_nonzero_cases,
        numpy_adapter=_nonzero_probe_np,
        ionp_adapter=_nonzero_probe_ionp,
        scalar_like=True,
        atol=0.0, rtol=0.0,
    ),
    "flatnonzero": ItemSpec(
        name="flatnonzero", kind="method",
        call_forms=_NO_ARGS_FORM,
        atol=0.0, rtol=0.0,
    ),
    "argwhere": ItemSpec(
        name="argwhere", kind="method",
        call_forms=_NO_ARGS_FORM,
        atol=0.0, rtol=0.0,
    ),
    "extract": ItemSpec(
        name="extract", kind="custom",
        custom_cases=_extract_cases,
        atol=0.0, rtol=0.0,
    ),
    "where": ItemSpec(
        name="where", kind="custom",
        custom_cases=lambda: (_where_cases() + _where_mismatched_xy_must_raise_cases()
                              + _where_weak_scalar_cases()),
        atol=0.0, rtol=0.0,
    ),
    "where_nonzero_alias": ItemSpec(
        name="where_nonzero_alias", kind="custom",
        custom_cases=_where_1arg_cases,
        numpy_adapter=_where_1arg_np,
        ionp_adapter=_where_1arg_ionp,
        scalar_like=True,
        atol=0.0, rtol=0.0,
    ),
    "searchsorted": ItemSpec(
        name="searchsorted", kind="custom",
        custom_cases=_searchsorted_toplevel_cases,
        atol=0.0, rtol=0.0,
    ),
    "nanargmax": ItemSpec(
        name="nanargmax", kind="custom",
        custom_cases=_nanarg_custom_cases,
        atol=0.0, rtol=0.0,
    ),
    "nanargmin": ItemSpec(
        name="nanargmin", kind="custom",
        custom_cases=_nanarg_custom_cases,
        atol=0.0, rtol=0.0,
    ),
    "ndarray.sort": ItemSpec(
        name="ndarray.sort", kind="custom",
        custom_cases=_sort_method_cases,
        numpy_adapter=_sort_method_probe,
        ionp_adapter=_sort_method_probe,
        scalar_like=True,
        atol=0.0, rtol=0.0,
        signed_zero_tie_exempt=True,
        signed_zero_tie_exempt_justification=_SIGNED_ZERO_TIE_JUSTIFICATION,
    ),
    "ndarray.argsort": ItemSpec(
        name="ndarray.argsort", kind="custom",
        custom_cases=_argsort_method_cases,
        numpy_adapter=_argsort_method_probe,
        ionp_adapter=_argsort_method_probe,
        atol=0.0, rtol=0.0,
    ),
    "ndarray.argmax": ItemSpec(
        name="ndarray.argmax", kind="method",
        numpy_path="ndarray.argmax", ionp_path="ndarray.argmax",
        call_forms=_ARGEXT_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "ndarray.argmin": ItemSpec(
        name="ndarray.argmin", kind="method",
        numpy_path="ndarray.argmin", ionp_path="ndarray.argmin",
        call_forms=_ARGEXT_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "ndarray.nonzero": ItemSpec(
        name="ndarray.nonzero", kind="custom",
        custom_cases=_nonzero_cases,
        numpy_adapter=_nonzero_method_probe_np,
        ionp_adapter=_nonzero_method_probe_ionp,
        scalar_like=True,
        atol=0.0, rtol=0.0,
    ),
    "ndarray.searchsorted": ItemSpec(
        name="ndarray.searchsorted", kind="custom",
        custom_cases=_searchsorted_method_cases,
        numpy_adapter=lambda a, v, side="left", sorter=None: a.searchsorted(v, side=side, sorter=sorter),
        ionp_adapter=lambda a, v, side="left", sorter=None: a.searchsorted(v, side=side, sorter=sorter),
        atol=0.0, rtol=0.0,
    ),
}


# ---------------------------------------------------------------------------
# AXIS-TYPE guards (2026-08-04). These items own one of numpy's five axis
# converters and were measurably wrong before cc5523f/cab03ab/db62b24; the
# sweep that found it now reads zero. This is what keeps it there.
#
# The case builder lives in reduction_cases.py (see `_axis_type_cases` for
# the converter map and for why shape () and dtype int64 are load-bearing).
# It is imported INSIDE the closure, i.e. at case-build time rather than at
# module import time, because reduction_cases.py and this module would
# otherwise form an import cycle through registry.py.
# ---------------------------------------------------------------------------

def _axistype_append_cases(specs, names):
    for name in names:
        spec = specs.get(name)
        assert spec is not None, f"axis-type: no spec entry for {name!r}"
        prev = spec.extra_cases

        def make(prev=prev, name=name):
            def wrapped():
                from reduction_cases import _axis_type_cases
                base = list(prev()) if prev is not None else []
                return base + _axis_type_cases(name)
            return wrapped

        spec.extra_cases = make()


_axistype_append_cases(SORT_SPECS, ["nanargmin", "nanargmax"])
