#!/usr/bin/env python3
"""Runner: executes the differential registry and emits {item: pass|fail}
JSON in exactly the format `tools/coverage.py --tests` consumes.

    python3 tests/differential/run.py --out report.json
    python3 tools/coverage.py --tests report.json

Also doubles as the self-test demo:

    python3 tests/differential/run.py --selftest --verbose

This module is the single source of truth for "how do we build the cases
for a given ItemSpec.kind" -- test_differential.py imports build_cases from
here so pytest and the standalone JSON runner can never quietly disagree
about what was actually tested.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import corpus
import ufunc_cases
import ufunc_registry  # noqa: F401  (import side effect: merges 134 ufunc items into REGISTRY)
import ufunc_order_cases  # noqa: F401  (import side effect: merges the "order/ufunc/<name>" strides-checked corpus into REGISTRY -- MUST come after ufunc_registry, whose UFUNC_SPECS it reads to build its cases; this import was missing entirely until 2026-08-02, so the corpus this file defines had never actually been executed by run.py -- see this task's report)
import ufunc_dtype_cases  # noqa: F401  (import side effect: merges the "dtype/ufunc/<name>" corpus into REGISTRY -- exercises the ufunc dtype= keyword's loop-selection + input-precasting axis for every currently-"exact"-declared ufunc; MUST come after ufunc_registry, whose UFUNC_SPECS it reads to build its cases. This axis was never previously exercised by any run.py-wired corpus -- see this task's report for genuine failures found on the sign/negative/subtract/bitwise_not/signbit/cos/sin/tanh items.)
import ufunc_out_axis_cases  # noqa: F401  (import side effect: merges the "out_axis/ufunc/<name>" corpus into REGISTRY -- exercises out= as a real crossed axis (ABSENT/match/partial-tuple/wrongdtype/wrongshape) for divmod/float_power/frexp/isnat/ldexp/modf, the six ufuncs a prior task implemented. MUST come after ufunc_registry, whose UFUNC_SPECS it reads to build its cases. A dedicated /tmp probe grid found 34/114 mismatches on this exact axis (missing per-output index in UFuncTypeError, out= buffers omitted from broadcast ValueError's shape list) before ionp-py/src/lib.rs's fix; this file is that same axis made ledger-visible so the regression class cannot silently return.)
import floordiv_crossing_cases  # noqa: F401  (import side effect: merges the "crossing/ufunc/floor_divide" and "crossing/ufunc/divmod" large-magnitude x fractional-divisor corpus into REGISTRY -- the mandatory crossing the floor_divide REVOKED comment in anionpy/_state/toplevel.py names as the bar for re-declaring; a fresh namespace, not more entries on the real "floor_divide"/"divmod" items, for the same registry-key-collision reason ufunc_order_cases.py/ufunc_dtype_cases.py already give.)
import complex_div_cases  # noqa: F401  (import side effect: merges the "crossing/ufunc/complex_divide_branch_coverage" corpus into REGISTRY -- permanent bit-exact coverage for complex_div's if/else branch split in ufunc.rs, including the else branch that had zero prior coverage; see complex_div_cases.py's module docstring for the measurement and the NaN-sign fix it guards.)
import strings_cases  # noqa: F401  (import side effect: overrides the 28 broken char.*/strings.* auto-derived ufunc entries with real kind="custom" ones -- see strings_cases.py's module docstring for why this must run after ufunc_registry)
import strparam_cases  # noqa: F401  (import side effect: merges the "strparam/<fn>_<param>" corpus into REGISTRY -- crosses every string-valued keyword parameter (order=/kind=/side=/casting=/mode=/ord=/dtype=) found by /tmp/bytesgap.py's 75-case sweep with the CAPABILITY axis that sweep exposed (correct-type valid/invalid, str subclass, np.str_ scalar, bytes/bytearray "has string methods but wrong type", and float/list/None/object "lacks string methods entirely") -- an axis no other file in this corpus varies. Surfaces that 9 currently-"exact" items (reshape, ravel, copy, sort, searchsorted, zeros, add, linalg.qr, ndarray.flatten) diverge from numpy on this axis, including cases where anionpy is MORE PERMISSIVE than numpy (casting=None silently accepted) -- see this task's report.)
import inplace_alias_cases  # noqa: F401  (import side effect: merges the "alias/self/<op>", "alias/view/<op>", "alias/nonaliased/<op>" corpus into REGISTRY -- the 2026-08-02 double-mutable-borrow fix's own verification corpus: true self-identity aliasing (`a //= a`, all 13 ndarray.__iXXX__ dunders) that inplace_cases.py's pre-existing 7-item corpus never constructs (its pairs are always two independently-filled arrays), plus a partial/overlapping-slice-aliasing probe that finds a real, separate, unfixed view-write-back gap -- see inplace_alias_cases.py's module docstring.)
import ma_cases  # noqa: F401  (import side effect: merges the "ma.<name>" corpus into REGISTRY -- Phases 0-2 of numpy.ma (MaskedArray core type, the _MaskedUnaryOperation/_MaskedBinaryOperation wrapper families, and already-exact-counterpart wrappers) -- see ma_cases.py's and anionpy/_state/ma.py's module docstrings for the full scope, the copyto-revert semantic fix, and the 7 candidate items deliberately excluded (tan, left_shift, right_shift, maximum, minimum, cos, tanh).)
import matrix_cases  # noqa: F401  (import side effect: merges the "matrix.<name>" corpus into REGISTRY -- the composition-wrapper `anionpy.matrix` (anionpy.ndarray cannot be subclassed, so this holds one rather than being one, same pattern as MaskedArray). 16 items: A/A1/T/H/I/getA/getA1/getT/getH/getI/shape/dtype/ndim/tolist/__getitem__/__mul__, each over the same 14-case provenance-varied construction corpus (+9 __getitem__-specific, +7 __mul__-specific cases). `.I`/`getI` use a 9-decimal-digit rounding snapshot matching the already-accepted atol=rtol=1e-9 LAPACK-noise epsilon their base functions (`linalg.inv`/`.pinv`) are declared exact under -- see matrix_cases.py's and anionpy/_state/matrix.py's module docstrings. `recarray`/`memmap`, the other two curated-explode classes surveyed alongside `matrix` this task, are NOT here: both measured 100% structurally blocked (no void/structured dtype kind in ionp-core for recarray; no buffer-sharing construction path at all for memmap) -- see anionpy/_state/matrix.py's module docstring for the full Phase-1 measurement on all three.)
import view_semantics_cases  # noqa: F401  (import side effect: attaches the view/identity DESCRIPTOR corpus -- shape/strides/`.base`/`OWNDATA`/`is`-the-input/write-through -- to the 14 shape items whose pre-existing cases compare VALUES ONLY, and registers the 7 of those that had no spec at all. MUST come after manip_cases/creation_cases/linalg_cases (via REGISTRY), whose specs it wraps rather than replaces: a non-probe argument falls straight through to the item's original callable, so every existing case still runs unchanged. A returned COPY where numpy returns a VIEW is byte-identical forever under a value-only comparison and still wrong -- see view_semantics_cases.py's module docstring.)
import setitem_cases  # noqa: F401  (import side effect: registers the "ndarray.__setitem__" DESCRIPTOR corpus in REGISTRY -- the item landed 2026-08-02 with no corpus of its own at all, exercised only incidentally as view_semantics_cases.py's write-through step. Compares a post-mutation descriptor of the ROOT OWNER (values + dtype + shape) plus the full exception type and message, because a 4,320-case sweep of the implementation found 186 divergences of which ZERO were a wrong stored value -- 132 wrong messages, 42 wrong exception types, 12 raise/no-raise splits. A value-only comparison would have graded that implementation finished. See setitem_cases.py's module docstring for the four measured rules it pins.)
import empty_stride_cases  # noqa: F401  (import side effect: attaches the empty-shape strides/flags DESCRIPTOR corpus -- shape/strides/dtype/C_CONTIGUOUS/F_CONTIGUOUS (deliberately NOT OWNDATA, see empty_stride_cases.py's module docstring) -- to the nine items Task #61 (2026-08-04) found declared-exact but FALSE on zero-sized arrays: ndarray.flatten, ascontiguousarray, empty_like, zeros_like, ones_like, full_like, expand_dims, roll, tile. Same wrapper mechanism as view_semantics_cases.py: a non-probe argument falls straight through to each item's pre-existing resolved callable, so every case that already existed keeps running unchanged. A value-only comparison is vacuously true on an empty array (there are no elements to differ), which is exactly how these nine stayed "passing" while wrong -- see empty_stride_cases.py's module docstring for the measured zero-vs-computed stride rule and the two failure modes (too-large where numpy zeroes; zeroed/wrong where numpy keeps its own computed strides) this corpus pins per item.)
import newaxis_stride_cases  # noqa: F401  (import side effect: attaches the C/F-order, transposed, non-contiguous, length-1-axis, zero-extent, and 0-d stride-DESCRIPTOR corpus to expand_dims/atleast_2d/atleast_3d/stack -- see newaxis_stride_cases.py's module docstring for why expand_dims's pre-existing corpus never compared strides and so never caught its 22/30 nonempty-input divergence from real numpy, now fixed by splitting `ionp-core/src/creation.rs::expand_dims` into `insert_newaxis` (stride-0, for atleast_2d/atleast_3d/stack/keepdims-reinsertion) and a reshape-routed `expand_dims` (Task #newaxis-split, 2026-08-08). MUST come after view_semantics_cases, whose atleast_2d/atleast_3d wrapping this file's REGISTRY-spec wrapping layers on top of (same non-probe-falls-through mechanism).)
import reshape_order_cases  # noqa: F401  (import side effect: attaches the identity-shape-short-circuit + order='A'/'K' resolution DESCRIPTOR corpus (Ticket #7) to reshape/ndarray.reshape -- dtype x shape (incl. 0-d/1-d/length-1-axis/zero-extent) x layout (c/f/t/neg/step2) x order grid, plus an unconditional order='K' ValueError check. MUST come after view_semantics_cases, whose reshape/ndarray.reshape adapter wrapping this file layers on top of (same non-probe-falls-through mechanism) -- see reshape_order_cases.py's module docstring.)
import copyto_cases  # noqa: F401  (import side effect: registers the "copyto" DESCRIPTOR corpus in REGISTRY -- the item was entirely ABSENT from anionpy until 2026-08-03 and is the keystone under the whole out= family plus nan_to_num/place/putmask. Compares a post-mutation descriptor of the ROOT OWNER (values + dtype + shape) plus full exception type and message, for the same reason setitem_cases.py does: a 1,687-case sweep of the implementation found divergences that were ALL wrong messages, wrong exception types, or raise/no-raise splits -- never a wrong stored value, so a value-only comparison would have graded it finished while it was still wrong in five ways. Pins six independently-measured rules, of which the seven-tier error ORDERING and the src-vs-where= asymmetric broadcast policy do not follow from any of the others. See copyto_cases.py's module docstring.)
import nanfuncs_cases  # noqa: F401  (import side effect: registers the "nan_to_num" DESCRIPTOR corpus in REGISTRY, same mechanism as copyto_cases above -- see that file for why nan_to_num cannot be graded by return value alone)
import power_view_cases  # noqa: F401  (import side effect: merges the "crossing/ufunc/power_view_*" corpus into REGISTRY -- offset/strided/2-D/broadcast ionp-native VIEW inputs to power/power.outer/power.at, the exact axis `check_int_pow_no_negative_self`'s pre-2026-08-06 offset-blind raw-buffer scan got wrong; each case builds its ionp-side operand via anionpy's OWN slicing on a real anionpy.ndarray, not via numpy-array ingestion, because ingestion always builds a slack-free minimal buffer that cannot reproduce this bug -- see power_view_cases.py's module docstring.)
import bitwise_count_casting_cases  # noqa: F401  (import side effect: merges the "casting/ufunc/bitwise_count" corpus into REGISTRY -- crosses casting= {None, 'no', 'equiv', 'safe', 'same_kind', 'unsafe'} x input dtype {bool, int8, uint8, int64, uint64} with dtype= ABSENT, the exact axis both the 2026-08-02 and 2026-08-06 bitwise_count declarations skipped (ee70a63's ufunc_dtype_cases.py always supplies dtype= alongside casting=, which cannot reach this bug). See toplevel.py's "RE-WITHDRAWN 2026-08-06" comment and bitwise_count_casting_cases.py's own module docstring for the full history.)
import reduceat_ndim_cases  # noqa: F401  (import side effect: merges the "ndim/reduceat/<op>" and "ndim/accumulate/<op>" corpus into REGISTRY -- N-D `.reduceat`/`.accumulate` (add/multiply/maximum/minimum/logical_or/power) across axes 0/1/2/-1/-2, 2-D and 3-D shapes, dtype float64/int64/complex128/bool, and edge-case indices (empty/unsorted/repeated/descending/out-of-range/index==axis_len), plus the 0-d scalar TypeError/AxisError boundary. Fresh namespace, not more entries on the real "add"/"multiply"/... items, for the same registry-key-collision reason floordiv_crossing_cases.py/power_view_cases.py give. See reduceat_ndim_cases.py's module docstring for the pre-2026-08-06 silent wrong-shape/wrong-value bug this corpus targets and guards against regressing.)
import complex_warning_cases  # noqa: F401  (import side effect: registers the "complex_warning_on_real_cast" DESCRIPTOR corpus in REGISTRY -- casting a complex array/scalar down to a real dtype must emit `ComplexWarning` (class name + exact text + count), matching real numpy, at astype/array(dtype=)/asarray(dtype=)/ndarray.__array__(dtype=)/`a[...] = complex_value`; anionpy was completely silent at all of them before this task's fix (`errors::warn_complex_cast`, `ionp-py/src/errors.rs`+`lib.rs`). See complex_warning_cases.py's module docstring for the negative controls and the explicitly out-of-scope sites (RuntimeWarning family, `.fill()`'s separate missing-exception bug, unimplemented putmask/place).)
import fperr_cases  # noqa: F401  (import side effect: registers the "fperr_subsystem" aggregate DESCRIPTOR item into REGISTRY -- the seterr/geterr/errstate/seterrcall/geterrcall FPE subsystem landed 2026-08-06 (`ionp-core/src/fpe.rs`, `ionp-py/src/fpstate.rs`) with a corpus committed at 7959c2d that was never actually imported here, so `run.py` never executed it and the coverage ledger still counted all five names `absent` despite the implementation existing and the corpus passing 100/100. See this task's report for which of the five real API names were separately declared off the back of this corpus, which were not, and why.)
import ma_warning_cases  # noqa: F401  (import side effect: registers the "ma_warning_domained_unary"/"ma_warning_domained_binary"/"ma_warning_dunder"/"ma_warning_plain_negative_controls" DESCRIPTOR corpus into REGISTRY -- `anionpy/ma/core.py`'s five masked-op factories now wrap their raw ufunc call in `anionpy.errstate(divide="ignore", invalid="ignore")` matching real numpy's own `_MaskedUnaryOperation`/`_MaskedBinaryOperation`/domained-family suppression (previously absent: anionpy emitted `RuntimeWarning`s real numpy does not, e.g. `ma.sqrt(MaskedArray([-4.0]))`, `MaskedArray([1.])/MaskedArray([0.])`) -- see ma_warning_cases.py's module docstring for the mandatory non-vacuous negative controls (overflow, genuinely-invalid plain-family data) that pin the suppression is SCOPED, not a blanket silencer.)
from harness import ItemResult, evaluate
from registry import REGISTRY
from selftest import EXPECTED_VERDICTS, SELFTEST_REGISTRY


def _build_cases_for_kind(spec):
    if spec.kind == "ufunc":
        return ufunc_cases.build_cases_for_item(spec)
    if spec.kind == "unary":
        # NOTE (2026-08-02, CLASS A / FORM-axis task): every kind="unary"
        # ItemSpec in the registry is "ndarray."-prefixed (properties/
        # dunders: .shape, .T, __neg__, __abs__, ...), which requires a
        # REAL ndarray RECEIVER -- resolve_ionp()'s "ndarray." branch
        # raises InvalidCase for anything else (see registry.py). A plain
        # Python list has no `.shape`/`.__neg__` of its own to test here,
        # so the array-like-FORM axis this task adds does not apply to
        # this kind at all; it is deliberately wired into kind="ufunc" and
        # the kind="custom" top-level-function items instead (see
        # ufunc_cases.py and reduction_cases.py/setops_cases.py/
        # stats_cases.py/manip_cases.py), which are the ones that actually
        # take an array-like as a plain positional argument.
        return [(c.label, (c.value,), {}) for c in corpus.unary_corpus()]
    if spec.kind == "binary":
        return [(label, (a, b), {}) for label, a, b in corpus.binary_corpus_as_tuples()]
    if spec.kind == "method":
        # spec.call_forms is a list[CallForm] (registry.py); ItemSpec's
        # __post_init__ already refuses to construct a "method" item with an
        # empty/missing call_forms, so this is never silently a single form.
        cases = []
        for c in corpus.unary_corpus():
            for form in spec.call_forms:
                if not form.applicable(c.value):
                    continue
                args, kwargs = form.build(c.value)
                cases.append((f"{c.label}/{form.label}", (c.value, *args), kwargs))
        # FORM axis (2026-08-02, CLASS A closure task): only meaningful for
        # kind="method" items that are actually top-level anionpy FUNCTIONS
        # under the hood (e.g. "prod"/"all"/"min"/"argmin"/"cumsum" --
        # resolve_ionp() falls through to a plain `anionpy.<name>(arr, ...)`
        # call for any path that does NOT start with "ndarray.", see that
        # method's docstring) -- a real "ndarray."-prefixed method needs an
        # actual ndarray RECEIVER (`arr.reshape(...)`), which a plain list
        # does not have, so those are excluded here exactly like the plain
        # kind="unary" branch above. Uses the item's own call_forms'
        # "no_args" baseline shape (args=(), kwargs={}) rather than every
        # form, since most other forms (axis=, keepdims=, ...) are built
        # from `arr.ndim`/`arr.shape`, which a raw list/tuple doesn't carry.
        #
        # Every REAL (non-selftest) registry item encodes the receiver
        # distinction directly in `name=` (verified by grep: "ndarray."-
        # prefixed items always set `name="ndarray.<method>"` with no
        # numpy_path/ionp_adapter override; top-level-function items always
        # set a bare `name="<func>"`, same story) -- but `selftest.py`'s
        # `reshape_all_call_forms`/`reshape_tuple_form_only` fixtures break
        # that pattern on purpose (`name="selftest.reshape_all_call_forms"`,
        # `numpy_path="ndarray.reshape"`, `ionp_adapter=...` bypassing path
        # resolution entirely) to prove the call-form engine -- appending
        # bare-list form cases there produced a genuine false FAIL (`selftest
        # --selftest` caught it: `ionp_adapter` still expects a real reshape
        # call, not a 0-arg bare call). Guarding on `numpy_path` too, not
        # just `ionp_path`/`name`, excludes exactly that pair without
        # touching any real registry item (none of which sets numpy_path to
        # an "ndarray."-prefixed path while leaving ionp_path/name bare).
        ionp_path = spec.ionp_path if spec.ionp_path is not None else spec.name
        numpy_path = spec.numpy_path or ""
        if not ionp_path.startswith("ndarray.") and not numpy_path.startswith("ndarray."):
            for c in corpus.form_axis_cases(sample_stride=8):
                cases.append((f"{c.label}/form", (c.value,), {}))
        return cases
    if spec.kind == "binary_op":
        # Arithmetic/comparison dunders: array-vs-array (broadcast corpus,
        # unchanged) PLUS array-vs-every-scalar-call-form (new). numpy
        # genuinely takes a different code path for `a + b` (both ndarray)
        # vs `a + 1` (python int) vs `a + np.float64(1)` (numpy scalar) vs
        # `a + np.array(1.0)` (0-d array) -- each is exercised here so an
        # implementation that only handles the array/array path is caught,
        # which is precisely how `ndarray.__add__` passed while `a + 1`
        # raised TypeError (see registry.py module docstring).
        cases = []
        for label, a, b in corpus.binary_corpus_as_tuples():
            cases.append((f"arr_arr/{label}", (a, b), {}))
        for c in corpus.unary_corpus():
            for op_label, other in corpus.scalar_operands():
                cases.append((f"{c.label}/scalar_{op_label}", (c.value, other), {}))
        return cases
    if spec.kind == "custom":
        return list(spec.custom_cases())
    raise ValueError(f"unknown ItemSpec.kind {spec.kind!r} for {spec.name}")


def build_cases(spec) -> list:
    """Cases for one item: whatever its kind builds, plus any `extra_cases`.

    `extra_cases` is appended for EVERY kind on purpose -- see its docstring
    in registry.py. It is additive only: it can never remove or shadow a
    case the kind itself produced, so wiring a boundary through it cannot
    shrink coverage.
    """
    cases = _build_cases_for_kind(spec)
    extra = getattr(spec, "extra_cases", None)
    if extra is not None:
        cases = list(cases) + list(extra())
    return cases


def run_registry(registry: dict) -> dict[str, ItemResult]:
    results: dict[str, ItemResult] = {}
    for name, spec in registry.items():
        try:
            cases = build_cases(spec)
            results[name] = evaluate(spec, cases)
        except Exception as exc:  # a broken registry entry is data, not a crash
            results[name] = ItemResult(
                name, "fail", 0, 0, [],
                f"harness raised while building cases: {type(exc).__name__}: {exc}",
            )
    return results


def _print_results(results: dict[str, ItemResult], verbose: bool) -> None:
    for name, r in sorted(results.items()):
        flag = "PASS" if r.verdict == "pass" else "FAIL"
        print(f"[{flag}] {name}  ({r.total - r.failed}/{r.total} cases)  -- {r.reason}")
        if verbose or r.verdict == "fail":
            for f in r.failures:
                print(f"    {f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write {item: pass|fail} JSON here")
    ap.add_argument("--selftest", action="store_true",
                     help="run the proof-of-detection registry instead of the real one")
    ap.add_argument("--verbose", action="store_true", help="print all case failures, not just a summary")
    args = ap.parse_args(argv)

    if args.selftest:
        results = run_registry(SELFTEST_REGISTRY)
        _print_results(results, args.verbose)
        mismatches = [
            name for name, expected in EXPECTED_VERDICTS.items()
            if results[name].verdict != expected
        ]
        print()
        if mismatches:
            print(f"SELF-TEST FAILED: harness disagreed with expected verdict for {mismatches}",
                  file=sys.stderr)
            return 1
        n_pass = sum(1 for v in EXPECTED_VERDICTS.values() if v == "pass")
        n_fail = sum(1 for v in EXPECTED_VERDICTS.values() if v == "fail")
        print(f"SELF-TEST OK: harness correctly passed the {n_pass} correct shim(s) and "
              f"failed all {n_fail} deliberately broken ones.")
        return 0

    results = run_registry(REGISTRY)
    _print_results(results, args.verbose)

    # Backward-compatible-by-shape report: every item is still gradeable as
    # a plain "pass"/"fail" string by any OLD consumer of this format
    # (tools/coverage.py's legacy branch), but a NEW consumer that wants to
    # tell a tolerant pass from an exact one gets that from the same file --
    # see tools/coverage.py's load_test_results(), which accepts both this
    # dict-of-dicts shape and the legacy dict-of-strings shape.
    report = {
        name: {
            "verdict": r.verdict,
            "tolerant": r.tolerant,
            "max_ulp": r.max_ulp_observed,
            "mechanism": r.mechanism,
        }
        for name, r in results.items()
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
