#!/usr/bin/env python3
"""THE SCOREBOARD for anionpy — 100% coverage NumPy-compatible.

This file answers exactly one question: *how much of numpy do we actually have?*

It exists because on 2026-07-31 I asserted a falsification from one cell of a
table, and twice before that restated a number without re-deriving it. A "100%
coverage" claim that a human types by hand is that same failure with better
lighting. So nobody types it. This tool emits it, every commit.

It is deliberately hostile to its own project. An item counts as covered only
if ALL of the following hold:

  1. it is DECLARED in anionpy.__ion_state__ with a state of 'ion' or 'exact'
  2. it is ACTUALLY PRESENT and resolvable on the anionpy package
  3. it has a PASSING differential test against real numpy

Declaring without implementing is caught (state 2). Implementing without
testing is caught (state 3). Both are reported as loudly as `absent`, because
both are ways of lying about the number.

Failing requirement 1 alone is NOT one uniform state: `absent` means the item
is not even resolvable on anionpy (requirement 2 also fails), while
`undeclared` means it resolves fine and may work perfectly -- nobody has
written the __ion_state__ line claiming it. Both are non-covered and both
still count against the target below; conflating them under one "not
implemented" label (ticket #82) mislabeled work that already exists as
missing. See docs/UNDECLARED-337.md for the standing triage.

No numerical work happens here. This is measurement tooling over numpy's own
introspection surface -- it is the one legitimate reason this file is Python.

Usage:
    python3 ionp/tools/coverage.py                 # human-readable ledger
    python3 ionp/tools/coverage.py --json          # machine-readable
    python3 ionp/tools/coverage.py --require 100   # exit 1 unless >= N% covered
    python3 ionp/tools/coverage.py --tests <junit.xml|report.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SURFACE = HERE / "numpy_surface.json"

VALID_STATES = ("ion", "exact")


def load_surface() -> dict:
    if not SURFACE.exists():
        sys.exit(
            f"FATAL: {SURFACE} missing.\n"
            "The surface manifest is the denominator. Regenerate it with "
            "tools/snapshot_surface.py before claiming any percentage."
        )
    return json.loads(SURFACE.read_text())


def all_items(surface: dict) -> list[str]:
    """Every numpy public API item we are on the hook for.

    `surface["exploded"]` maps a curated class name (e.g. "ndarray",
    "ma.MaskedArray", "random.Generator") to {"public": [...], "dunder":
    [...]} bare method/attr names; each becomes the item key
    "<class>.<name>", set-aware and generic -- no class is special-cased
    here (see tools/snapshot_surface.py's CURATED_EXPLODE for the set and
    why it is curated rather than "every class").

    Fails loudly on duplicate keys. A regression in snapshot_surface.py once
    emitted ndarray methods WITHOUT their "ndarray." prefix, so bare `sum`,
    `T`, `reshape` collided with the top-level functions of the same name:
    the denominator inflated by the collision count while every `ndarray.*`
    declaration matched no item and silently earned zero. The tell was that
    the per-state counts no longer summed to the reported total -- an
    arithmetic inconsistency I happened to notice by eye. Noticing by eye is
    not a control, so it is now an assertion. Building every exploded key by
    prefixing with its own class name here (rather than trusting a prefix
    baked into the manifest, as the old ndarray-only shape did) makes that
    specific regression structurally impossible for any curated class; the
    duplicate-key assertion below still catches the general case (e.g. two
    curated classes whose (prefix, method) pair somehow collided, or an
    exploded key colliding with a flat `names` entry).
    """
    names = list(surface["names"].keys())
    exploded = surface.get("exploded", {})

    exploded_items: list[str] = []
    for cls_name, methods in exploded.items():
        if not cls_name or not isinstance(methods, dict):
            sys.exit(
                f"FATAL: malformed 'exploded' entry {cls_name!r} in the "
                f"surface manifest. Regenerate with tools/snapshot_surface.py."
            )
        for kind in ("public", "dunder"):
            for n in methods.get(kind, []):
                exploded_items.append(f"{cls_name}.{n}")

    items = names + exploded_items
    dupes = {n for n in items if items.count(n) > 1}
    if dupes:
        sys.exit(
            f"FATAL: {len(dupes)} duplicate item keys in the surface manifest "
            f"(e.g. {sorted(dupes)[:5]}). The denominator would be inflated."
        )
    return sorted(items)


def _numpy_class(prefix: str):
    """Resolve a curated exploded-class prefix (e.g. 'polynomial.Chebyshev',
    'ma.MaskedArray') against REAL numpy, for tooling-only introspection.

    This file is measurement tooling over numpy's own introspection surface
    (see the module docstring), not shipped runtime -- unlike anionpy/**.py,
    it may import and inspect real numpy freely. Returns None if numpy is
    unavailable or the prefix doesn't resolve on it, so callers can fall
    back to the conservative (stricter) branch rather than guessing.
    """
    try:
        import numpy
    except Exception:
        return None
    obj = numpy
    for part in prefix.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


_NUMPY_CLASS_CACHE: dict[str, object] = {}


def _numpy_class_cached(prefix: str):
    if prefix not in _NUMPY_CLASS_CACHE:
        _NUMPY_CLASS_CACHE[prefix] = _numpy_class(prefix)
    return _NUMPY_CLASS_CACHE[prefix]


def resolve(root, dotted: str, exploded_classes: frozenset[str] = frozenset()):
    """Resolve 'linalg.svd' / 'ndarray.sum' / 'ma.MaskedArray.compressed'
    against the anionpy package.

    `exploded_classes` is `surface["exploded"].keys()` -- the set of
    curated class prefixes (see all_items/CURATED_EXPLODE). Generic and
    set-aware on purpose: an earlier version of this function hardcoded the
    "ndarray." special case, which meant every curated class added to the
    manifest needed its own hand-written branch here to ever resolve as
    anything but silently `absent`.

    Returns (found: bool, obj). Never raises -- a resolution failure is data,
    not an error.
    """
    for prefix in exploded_classes:
        lead = f"{prefix}."
        if not dotted.startswith(lead):
            continue
        attr = dotted[len(lead):]
        cls = root
        for part in prefix.split("."):
            if not hasattr(cls, part):
                return False, None
            cls = getattr(cls, part)
        # `hasattr(cls, attr)` alone is a presence test, not an
        # implementation claim: EVERY dunder inherited from `object`
        # (`__eq__`, `__lt__`, `__hash__`, `__str__`, ...) is `hasattr`-true
        # on any class without ever being defined by it, and those
        # inherited defaults are not merely empty -- `object.__eq__` does
        # identity comparison where numpy's `a == b` returns an elementwise
        # bool array, so crediting on `hasattr` evidence alone can declare
        # an item that silently returns WRONG answers instead of failing
        # loudly (THE HASATTR TRAP; see
        # reports/ionp-next-block-and-the-hasattr-trap-2026-08-01.md, the
        # near-miss that found this: 21 of 29 `hasattr`-true dunders were
        # inherited-only).
        #
        # But a flat `attr in vars(cls)` (own-`__dict__`-only) over-corrects
        # into THE MUTATION TRAP: it can be satisfied by writing the name
        # into the class `__dict__` with nothing behind it but the intent
        # to satisfy this exact check -- which is precisely what this
        # project's own `_COVERAGE_REBIND_*` blocks did (see ticket #75's
        # ruling). Behaviour matching is not evidence against this: the
        # rebinds worked by copying the ALREADY-CORRECT inherited
        # function/descriptor object onto the subclass's own `__dict__`,
        # so `attr in vars(cls)` went true while nothing about what the
        # method DOES changed -- a check that a class satisfies purely by
        # existing differently in a dict is not measuring behaviour at all.
        #
        # The fix is to phrase the requirement relative to numpy itself,
        # per curated class: if real numpy binds `attr` in THIS class's own
        # `__dict__` (e.g. `ndarray.__eq__`, which numpy defines directly
        # because it must return an elementwise array, not identity), then
        # numpy is telling us this name needs its own implementation here
        # too -- inheriting a generic default would silently diverge, so we
        # require the same own-`__dict__` binding of ours (shuts the
        # mutation trap: a rebind-only entry still fails, because the
        # REQUIREMENT itself is derived from numpy's dict, not ours).
        # If numpy itself merely INHERITS `attr` for this class (not in its
        # own `__dict__` -- e.g. `Chebyshev.__lt__`, `Chebyshev.basis`,
        # both actually implemented on `ABCPolyBase` and inherited by every
        # concrete basis, numpy's own included), then inheriting the same
        # default through our own base class is genuine architectural
        # parity with numpy, not a gap -- `hasattr` is sufficient and a
        # `_COVERAGE_REBIND_*`-style own-dict copy has nothing left to
        # prove (shuts the hasattr trap the other direction: this is not
        # "credit any hasattr", it's "credit hasattr only where numpy's own
        # structure says inheritance is the right answer").
        #
        # If numpy is unresolvable for this prefix (import failure, or the
        # prefix doesn't exist on the installed numpy), fall back to the
        # stricter own-`__dict__` check: without numpy to ask, we cannot
        # tell which case applies, and silently reopening the hasattr trap
        # in that situation would be the exact failure this file exists to
        # prevent.
        if not hasattr(cls, attr):
            return False, None
        numpy_cls = _numpy_class_cached(prefix)
        numpy_owns_it = numpy_cls is not None and attr in vars(numpy_cls)
        if numpy_owns_it or numpy_cls is None:
            # ticket #80: `attr in vars(cls)` alone (own-`__dict__`
            # membership) is NOT rebind-proof. It was introduced as the fix
            # for the mutation trap (see the big comment above), but it only
            # closes the trap when the copied-in value came from OUTSIDE
            # `cls`'s own inheritance chain -- a bare
            # `setattr(cls, attr, cls.basis_name)`-style copy of an object
            # `cls` would *already resolve to via its own MRO* still makes
            # `attr in vars(cls)` true, because dict-membership cannot tell
            # "I was given a genuinely new implementation" from "I was given
            # the exact same object my own base class already provides".
            # Repro (subclass real `Chebyshev` as `Stripped`, do nothing but
            # copy the inherited `basis_name` onto `Stripped`'s own dict):
            #   resolve(..., "polynomial.Chebyshev.basis_name", ...) is
            #   (False, None) before the copy, (True, 'T') after -- the
            #   bare copy alone flips the verdict with no new behaviour.
            #
            # PROVENANCE FIX: don't just ask "is `attr` in `cls`'s own
            # dict", ask "is the object bound there the SAME object
            # (identity, not equality -- two independently-written
            # implementations returning an equal value are not the same
            # object) that `cls`'s own MRO would already resolve `attr` to
            # if this own-dict entry didn't exist". If yes, this entry
            # cannot be evidence of new work: it is provably a copy of
            # something already reachable through inheritance, which is
            # exactly what a `_COVERAGE_REBIND_*`-style rebind does and
            # exactly what a genuine override does NOT do -- a real
            # reimplementation (a fresh `def`, a freshly computed value)
            # produces its own object, distinct by identity from whatever
            # object the base class happens to bind, even on the rare
            # occasion the VALUES coincide (e.g. two sibling classes both
            # legitimately choosing the same string or bool). This is
            # deliberately narrower than value-equality: rejecting on equal
            # VALUE would falsely fail a genuine override that happens to
            # match its base's default in some case, which the ticket rules
            # out as an unacceptable false negative. Rejecting on object
            # IDENTITY only catches the provably-copied case.
            #
            # This only protects against rebinds copied from within `cls`'s
            # OWN inheritance chain. A rebind that copies from an unrelated
            # class (not a base of `cls`) would bind a value/object `cls`
            # could never have resolved to on its own, so there is no
            # "what would this already resolve to" baseline to compare
            # against here -- that variant of the trap is NOT closed by this
            # check and would need a behavioural (differential-test) signal
            # instead of a structural one.
            own_dict = vars(cls)
            if attr not in own_dict:
                return False, None
            own_val = own_dict[attr]
            _MISSING = object()
            inherited_val = _MISSING
            for base in cls.__mro__[1:]:
                if attr in vars(base):
                    inherited_val = vars(base)[attr]
                    break
            if inherited_val is not _MISSING and own_val is inherited_val:
                # Provably a copy of what inheritance already provided --
                # the mutation trap, still open under a plain
                # `attr in vars(cls)` check.
                return False, None
            return True, own_val
        # numpy itself only inherits `attr` for this class -- hasattr
        # (already confirmed true above) is genuine parity.
        return True, getattr(cls, attr)
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return False, None
        obj = getattr(obj, part)
    return True, obj


def load_ionp():
    """Import anionpy if it exists. Absence is a legitimate, reportable state.

    The repo root (parent of tools/) is put on sys.path so the ledger can see
    the in-tree package regardless of cwd. Without this the tool reports 0%
    forever from any directory but one -- an under-report, which is the safe
    direction, but still a scoreboard that cannot see the game. Found by
    running the ledger and the test harness side by side and noticing only one
    of them could import the package.
    """
    root = str(HERE.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import anionpy  # noqa: F401

        return anionpy, None
    except Exception as exc:  # ImportError, or a broken native module
        return None, f"{type(exc).__name__}: {exc}"


# The complete set of mechanism strings this routing table knows how to
# bucket a tolerant pass into. Used to draw the KEY ABSENT vs KEY PRESENT-
# BUT-UNRECOGNIZED line in `load_test_results` below -- see that function's
# docstring, "THE FAIL-OPEN FIX" section, for why that line is load-bearing.
_RECOGNIZED_TOLERANT_MECHANISMS = ("ulp", "epsilon", "tie_exempt", "message_pair")


def load_test_results(
    path: str | None,
) -> tuple[
    set[str], set[str], dict[str, float | None], dict[str, float | None],
    dict[str, float | None], dict[str, float | None], str | None,
]:
    """Return (passed, failed, ulp_tolerant, epsilon_tolerant, tie_exempt,
    message_pair_tolerant, note).

    `ulp_tolerant` / `epsilon_tolerant` / `tie_exempt` / `message_pair_tolerant`
    each map item -> max_ulp_observed (None for tie_exempt and
    message_pair_tolerant, neither of which is a magnitude bound: tie_exempt
    is a binary sign-of-zero carve-out, message_pair_tolerant is a binary
    coupled-exception-pair equivalence -- see registry.py's
    `_comparison_reduce_message_pair_ok`) for every item whose PASS actually
    required nonzero slack under that specific mechanism (see harness.py's
    ItemResult.tolerant/.mechanism and
    reports/ionp-ulp-tolerance-decision-2026-08-01.md); an item in none of
    the four dicts either failed or passed bit-exact. An item can appear in
    at most one of the four dicts -- `mechanism` is static per ItemSpec (ULP,
    epsilon, tie_exempt, and message_pair are mutually exclusive
    declarations -- see registry.py's ItemSpec.__post_init__, which refuses
    to construct an item combining signed_zero_tie_exempt with
    ulp_tolerance/epsilon_tolerance -- and harness.compare_values/run_case).
    Bit-exact-vs-tolerant, and ULP-vs-epsilon-vs-tie_exempt-vs-message_pair,
    are distinctions the ledger must never lose: an item passing under ANY
    tolerance/exemption/equivalence is not identical to one passing exactly,
    and the scoreboard must not conflate any of the five states with each
    other (2026-08-01 truthfulness-hole fix: the epsilon/atol-rtol path used
    to be silently invisible here, credited straight into "bit-exact";
    `tie_exempt` added same day for sort/sort_complex/ndarray.sort's
    signed-zero tie-order exemption -- see sort_cases.py's
    `_SIGNED_ZERO_TIE_JUSTIFICATION`; `message_pair` added 2026-08-02 for the
    comparison-reduce no-loop coupled cold/warm exception-pair equivalence --
    see registry.py's `_comparison_reduce_message_pair_ok`).

    Three report shapes are accepted, so this stays backward compatible
    with every report ever produced before the ULP/epsilon/tie_exempt/
    message_pair mechanisms existed:
      - legacy:  {item: "pass"|"fail"}                          (no info)
      - ULP-era: {item: {"verdict": ..., "tolerant": bool,
                          "max_ulp": float|None}}                (no mechanism
                          -- treated as "ulp" for backward compatibility,
                          since that was the only mechanism this shape
                          could have meant)
      - current: {item: {"verdict": ..., "tolerant": bool,
                          "max_ulp": float|None, "mechanism":
                          "ulp"|"epsilon"|"tie_exempt"|"message_pair"|"none"}}

    THE FAIL-OPEN FIX (2026-08-02): earlier, any `tolerant=True` item whose
    `mechanism` value this routing table did not recognize -- including a
    genuinely NEW mechanism this file had simply never been taught about --
    fell through a bare `else` straight into `ulp_tolerant`. That is a
    fail-open default in the one instrument whose entire job is to make
    tolerances visible: it did not just mislabel one new mechanism
    (`message_pair`, on the day it was discovered -- an item legitimately
    passing via a coupled cold/warm exception-pair equivalence rendered in
    the ledger as "ok (ULP-tolerant, max_ulp=None)", a straightforwardly
    false statement about what kind of tolerance was used), it would
    silently relabel EVERY future unrecognized mechanism the same way, and
    always in the direction of looking more benign than reality (a bespoke
    equivalence check dressed up as a well-understood floating-point ULP
    bound). Adding a fourth named bucket without also closing this would
    only buy time until a fifth mechanism hit the exact same bug.

    The fix distinguishes two cases by the PRESENCE of the `mechanism` key,
    not by its value:
      - key ABSENT entirely -> genuinely a pre-mechanism report (nothing
        older than the mechanism field itself could have produced this
        shape) -> treated as "ulp", unchanged from before.
      - key PRESENT but not one of `_RECOGNIZED_TOLERANT_MECHANISMS` ->
        NOT legacy -- something newer than this routing table wrote this
        report, and that is exactly the situation where silence is most
        expensive. This raises loudly (`sys.exit`, matching this file's
        existing FATAL-on-integrity-violation convention -- see
        `all_items`/`build_ledger`'s consistency checks) rather than
        guessing a bucket, naming the offending item and mechanism string
        so the fix is "teach coverage.py this mechanism", not "silently
        keep scoring wrong".

    When no report is supplied, NOTHING counts as tested -- the
    conservative direction on purpose.
    """
    if not path:
        return set(), set(), {}, {}, {}, {}, "no test report supplied -- 0 items credited as tested"
    p = Path(path)
    if not p.exists():
        return set(), set(), {}, {}, {}, {}, f"test report {path} not found -- 0 items credited"
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        return set(), set(), {}, {}, {}, {}, f"test report unparseable ({exc}) -- 0 items credited"

    passed: set[str] = set()
    failed: set[str] = set()
    ulp_tolerant: dict[str, float | None] = {}
    epsilon_tolerant: dict[str, float | None] = {}
    tie_exempt: dict[str, float | None] = {}
    message_pair_tolerant: dict[str, float | None] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            verdict = v.get("verdict")
            if verdict == "pass":
                passed.add(k)
                if v.get("tolerant"):
                    if "mechanism" not in v:
                        # KEY ABSENT: genuinely a pre-mechanism report --
                        # ULP was the only mechanism this shape could ever
                        # have meant. See the docstring's "THE FAIL-OPEN
                        # FIX" section for why this is checked by key
                        # presence, not by value.
                        mechanism = "ulp"
                    else:
                        mechanism = v["mechanism"]
                    if mechanism == "epsilon":
                        epsilon_tolerant[k] = v.get("max_ulp")
                    elif mechanism == "tie_exempt":
                        tie_exempt[k] = v.get("max_ulp")
                    elif mechanism == "message_pair":
                        message_pair_tolerant[k] = v.get("max_ulp")
                    elif mechanism == "ulp":
                        ulp_tolerant[k] = v.get("max_ulp")
                    else:
                        # KEY PRESENT but unrecognized (this also catches
                        # mechanism="none" arriving together with
                        # tolerant=True, which is itself a contradiction --
                        # "none" documents an item that declared NO
                        # tolerance mechanism, so it must never co-occur
                        # with tolerant=True in a valid report). NOT legacy.
                        # Refuse to score it rather than guess a bucket.
                        sys.exit(
                            f"FATAL: item {k!r} reports tolerant=True with "
                            f"mechanism={mechanism!r}, which this coverage.py "
                            f"does not recognize (known: "
                            f"{', '.join(_RECOGNIZED_TOLERANT_MECHANISMS)}). "
                            f"The 'mechanism' key IS present, so this is NOT "
                            f"a legacy pre-mechanism report (those omit the "
                            f"key entirely and are still safely treated as "
                            f"ULP above) -- something newer than this "
                            f"routing table produced this report. Silently "
                            f"folding an unrecognized mechanism into "
                            f"ULP-tolerant is the exact fail-open bug fixed "
                            f"2026-08-02: it makes an unknown tolerance LOOK "
                            f"like a known, milder one instead of surfacing "
                            f"it. Add a bucket for {mechanism!r} in "
                            f"coverage.py's load_test_results/build_ledger "
                            f"(mirroring message_pair/tie_exempt) before "
                            f"this report can be scored."
                        )
            else:
                failed.add(k)
        else:
            # legacy shape: bare "pass"/"fail" string, no tolerance info possible
            if v == "pass":
                passed.add(k)
            else:
                failed.add(k)
    return passed, failed, ulp_tolerant, epsilon_tolerant, tie_exempt, message_pair_tolerant, None


def fold_axis_failures(failed: set[str], items) -> set[str]:
    """Attribute a failing AXIS-corpus case back to the item it grades.

    Several corpora test one keyword axis across many ufuncs and name their
    cases `"<axis>/<kind>/<item>"` -- `order/ufunc/right_shift`,
    `dtype/ufunc/negative`. Those keys are not surface item names, so
    without this every such failure landed in `failed_tests` and matched
    nothing, and a DECLARED item could fail its axis test while this ledger
    printed `failing 0`.

    That is not a rounding error, it is the exact lie this file exists to
    prevent. It was found the honest way: a new dtype= corpus caught four
    declared-exact ufuncs (bitwise_not, negative, subtract, signbit) truly
    diverging from numpy, and coverage still reported `failing 0` -- because
    `failing` only ever counted items whose OWN name appeared in the report.
    "Declared and wrong" is the worst state in this project and it was
    invisible.

    FAILURES ONLY, deliberately. A passing axis case is NOT folded into
    `passed`: being credited requires an item's own differential test, and
    letting `order/ufunc/foo` passing stand in for `foo` passing would
    manufacture coverage from a corpus that only ever probes one keyword.
    Failing is the conservative direction; passing is not.

    Only folds onto names that are real surface items, so an axis corpus
    covering something outside the tracked surface stays inert rather than
    inventing a row.
    """
    surface_items = set(items)
    folded = set(failed)
    for key in failed:
        if "/" not in key:
            continue
        candidate = key.rsplit("/", 1)[-1]
        if candidate in surface_items:
            folded.add(candidate)
    return folded


def build_ledger(tests_path: str | None) -> dict:
    surface = load_surface()
    items = all_items(surface)
    exploded_classes = frozenset(surface.get("exploded", {}).keys())
    root, import_err = load_ionp()
    (
        passed, failed_tests, ulp_tolerant, epsilon_tolerant, tie_exempt,
        message_pair_tolerant, test_note,
    ) = load_test_results(tests_path)
    failed_tests = fold_axis_failures(failed_tests, items)

    declared: dict[str, str] = {}
    if root is not None:
        declared = dict(getattr(root, "__ion_state__", {}) or {})

    rows = {}
    for item in items:
        state = declared.get(item)
        present, _obj = (
            (False, None) if root is None
            else resolve(root, item, exploded_classes)
        )

        if state not in VALID_STATES:
            if present:
                # Resolvable on anionpy right now (per `resolve()` above,
                # numpy-relative and mutation-trap-aware) but nobody wrote
                # a line in __ion_state__ claiming it. This is NOT the same
                # failure as "absent": the work may already exist and the
                # gap is bookkeeping, not implementation. Conflating the two
                # under one "absent"/"not implemented" label was ticket #82
                # -- 337 of the 1482 rows that printed "not implemented"
                # were, measurably, implemented. See docs/UNDECLARED-337.md
                # for the triage of what's behind each one; NONE of them are
                # declared here as part of that ticket -- declaring requires
                # its own differential-corpus verification per item, and
                # bulk-flipping this branch to "covered" would be exactly
                # the failure this split exists to prevent.
                verdict = "undeclared"
                reason = "resolvable on anionpy but not declared in __ion_state__"
            else:
                verdict = "absent"
                reason = "not declared in __ion_state__ and not resolvable on anionpy"
        elif not present:
            verdict = "phantom"
            reason = f"declared '{state}' but not resolvable on anionpy"
        elif item in failed_tests:
            verdict = "failing"
            reason = "differential test FAILED against numpy"
        elif item not in passed:
            verdict = "untested"
            reason = "no passing differential test"
        else:
            verdict = state
            reason = "ok"
        is_ulp_tolerant = item in ulp_tolerant
        is_epsilon_tolerant = item in epsilon_tolerant
        is_tie_exempt = item in tie_exempt
        is_message_pair_tolerant = item in message_pair_tolerant
        is_tolerant = (
            is_ulp_tolerant or is_epsilon_tolerant or is_tie_exempt
            or is_message_pair_tolerant
        )
        if is_ulp_tolerant:
            reason = f"ok (ULP-tolerant, max_ulp={ulp_tolerant[item]})"
        elif is_epsilon_tolerant:
            # NOTE 2026-08-03 (Monday): this deliberately does NOT read
            # "max_ulp=". Under an epsilon (atol/rtol) comparison, ULP is not
            # the acceptance metric -- it is an incidental statistic, and near
            # zero it explodes: two values 1e-18 apart on opposite sides of
            # zero are ~4.4e18 ULP apart while being absolutely identical to
            # 1e-18. Printing "epsilon-tolerant, max_ulp=4.4e18" reads as a
            # 4.4e18-ULP blanket tolerance, which would be a blindfold rather
            # than a bound. It is not one: the real gate is
            # `epsilon_tolerance` in registry.py, which requires a non-empty
            # justification AND an `epsilon_sweep` whose declared bound equals
            # the measured maximum over >= MIN_ULP_SWEEP_N samples.
            #
            # I misread my own ledger this way once. The number was honest and
            # the label was not, which is the same class of defect this file
            # exists to catch -- so the label is now explicit about which
            # metric actually gates the pass.
            reason = (
                f"ok (epsilon-tolerant via declared atol/rtol; incidental "
                f"max_ulp={epsilon_tolerant[item]} is NOT the bound -- see "
                f"epsilon_tolerance + epsilon_sweep in registry.py)"
            )
        elif is_tie_exempt:
            reason = (
                "ok (signed_zero_tie_exempt: -0.0 vs +0.0 arrangement within a "
                "tied run is algorithm-internal, compared with -0.0 == +0.0; "
                "see justification in registry.py)"
            )
        elif is_message_pair_tolerant:
            reason = (
                "ok (message_pair-tolerant: numpy's own exception class+message "
                "for this shape is a coupled, cache-dependent cold/warm pair, "
                "not independently checkable text -- see "
                "_comparison_reduce_message_pair_ok in registry.py)"
            )
        rows[item] = {
            "verdict": verdict,
            "declared": state,
            "reason": reason,
            "tolerant": is_tolerant,
            "tolerance_kind": (
                "ulp" if is_ulp_tolerant
                else ("epsilon" if is_epsilon_tolerant
                      else ("tie_exempt" if is_tie_exempt
                            else ("message_pair" if is_message_pair_tolerant else None)))
            ),
            "max_ulp": ulp_tolerant.get(
                item, epsilon_tolerant.get(
                    item, tie_exempt.get(item, message_pair_tolerant.get(item))
                )
            ),
        }

    counts: dict[str, int] = {}
    for r in rows.values():
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    # Tolerant-vs-exact is tracked as an ORTHOGONAL annotation, never a
    # separate verdict bucket -- an item is still "ion"/"exact" in the
    # primary ledger (that vocabulary is __ion_state__'s, not this
    # mechanism's to redefine), but a reader must be able to see at a
    # glance how many of those passes were bit-exact vs needed ULP slack
    # vs needed epsilon (atol/rtol) slack vs needed the signed-zero
    # tie-order exemption vs needed the message_pair coupled-exception
    # equivalence. Five mutually exclusive buckets (2026-08-01
    # truthfulness-hole fix, extended same day for tie_exempt, extended
    # again 2026-08-02 for message_pair): an item credited under ANY
    # tolerance/exemption/equivalence -- ULP, epsilon, tie_exempt, or
    # message_pair -- is never counted as bit-exact.
    ulp_tolerant_count = sum(
        1 for r in rows.values() if r["verdict"] in VALID_STATES and r["tolerance_kind"] == "ulp"
    )
    epsilon_tolerant_count = sum(
        1 for r in rows.values() if r["verdict"] in VALID_STATES and r["tolerance_kind"] == "epsilon"
    )
    tie_exempt_count = sum(
        1 for r in rows.values() if r["verdict"] in VALID_STATES and r["tolerance_kind"] == "tie_exempt"
    )
    message_pair_tolerant_count = sum(
        1 for r in rows.values() if r["verdict"] in VALID_STATES and r["tolerance_kind"] == "message_pair"
    )
    exact_pass_count = sum(
        1 for r in rows.values() if r["verdict"] in VALID_STATES and not r["tolerant"]
    )
    tolerant_count = (
        ulp_tolerant_count + epsilon_tolerant_count + tie_exempt_count
        + message_pair_tolerant_count
    )
    # Every covered item lands in exactly one of the five buckets; if not,
    # the buckets themselves are miscounting (same discipline as the
    # absent/phantom/failing/... state-sum assertion below).
    _covered_check = counts.get("ion", 0) + counts.get("exact", 0)
    _bucket_sum = (
        exact_pass_count + ulp_tolerant_count + epsilon_tolerant_count
        + tie_exempt_count + message_pair_tolerant_count
    )
    if _bucket_sum != _covered_check:
        sys.exit(
            f"FATAL: bit-exact ({exact_pass_count}) + ULP-tolerant "
            f"({ulp_tolerant_count}) + epsilon-tolerant ({epsilon_tolerant_count}) + "
            f"tie_exempt ({tie_exempt_count}) + message_pair ({message_pair_tolerant_count}) = "
            f"{_bucket_sum} but "
            f"{_covered_check} items are covered. The ledger is miscounting the "
            f"tolerance buckets; the percentage is void."
        )

    total = len(items)
    covered = counts.get("ion", 0) + counts.get("exact", 0)

    # Every item must land in exactly one state. If these disagree, the
    # ledger is miscounting and NO percentage it reports can be trusted.
    if sum(counts.values()) != total:
        sys.exit(
            f"FATAL: state counts sum to {sum(counts.values())} but there are "
            f"{total} items. The ledger is miscounting; the percentage is void."
        )

    return {
        "numpy_version": surface.get("numpy_version"),
        "total_items": total,
        "covered": covered,
        "coverage_pct": round(100.0 * covered / total, 3) if total else 0.0,
        "counts": counts,
        "import_error": import_err,
        "test_note": test_note,
        "rows": rows,
        "tolerant_count": tolerant_count,
        "exact_pass_count": exact_pass_count,
        "ulp_tolerant_count": ulp_tolerant_count,
        "epsilon_tolerant_count": epsilon_tolerant_count,
        "tie_exempt_count": tie_exempt_count,
        "message_pair_tolerant_count": message_pair_tolerant_count,
    }


def human(ledger: dict) -> str:
    c = ledger["counts"]
    out = []
    out.append("=" * 74)
    out.append("  anionpy COVERAGE LEDGER  --  target: 0 absent, 0 undeclared, 0 phantom, 0 failing")
    out.append("=" * 74)
    out.append(f"  numpy reference : {ledger['numpy_version']}")
    out.append(f"  surface items   : {ledger['total_items']}")
    out.append("")
    for state, blurb in (
        ("ion", "Rust core, structural fast path"),
        ("exact", "Rust core, correct, no asymptotic win claimed"),
        ("untested", "present but NOT credited -- no passing diff test"),
        ("phantom", "DECLARED BUT MISSING -- this is a lie in the making"),
        ("failing", "differential test FAILS vs numpy"),
        ("undeclared", "resolvable on anionpy but NOT declared in __ion_state__"),
        ("absent", "not resolvable on anionpy -- genuinely not implemented"),
    ):
        n = c.get(state, 0)
        flag = "  <-- BLOCKS DONE" if state in ("phantom", "failing") and n else ""
        out.append(f"  {state:<9} {n:>6}   {blurb}{flag}")
    out.append("")
    out.append(f"  COVERAGE: {ledger['coverage_pct']}%   ({ledger['covered']}/{ledger['total_items']})")
    out.append(
        f"    of which bit-exact vs numpy : {ledger['exact_pass_count']:>6}"
    )
    out.append(
        f"    of which ULP-tolerant       : {ledger['ulp_tolerant_count']:>6}"
        + ("  <-- see rows for max_ulp, justification is in registry.py"
           if ledger["ulp_tolerant_count"] else "")
    )
    out.append(
        f"    of which epsilon-tolerant   : {ledger['epsilon_tolerant_count']:>6}"
        + ("  <-- gated by declared atol/rtol, NOT by ULP; see "
           "epsilon_tolerance + epsilon_sweep in registry.py"
           if ledger["epsilon_tolerant_count"] else "")
    )
    out.append(
        f"    of which tie_exempt (0-sign): {ledger['tie_exempt_count']:>6}"
        + ("  <-- signed-zero tie-order only, justification is in registry.py"
           if ledger["tie_exempt_count"] else "")
    )
    out.append(
        f"    of which message_pair       : {ledger['message_pair_tolerant_count']:>6}"
        + ("  <-- coupled cold/warm exception-pair equivalence, see registry.py"
           if ledger["message_pair_tolerant_count"] else "")
    )
    if ledger["import_error"]:
        out.append(f"  NOTE: anionpy not importable -- {ledger['import_error']}")
    if ledger["test_note"]:
        out.append(f"  NOTE: {ledger['test_note']}")
    out.append("=" * 74)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit machine-readable ledger")
    ap.add_argument("--tests", help="JSON map {api_item: pass|fail} from the differential suite")
    ap.add_argument("--require", type=float, default=None, help="exit 1 below this coverage %%")
    ap.add_argument(
        "--list",
        help=(
            "print every item in this verdict state "
            "(ion|exact|untested|phantom|failing|undeclared|absent)"
        ),
    )
    args = ap.parse_args()

    ledger = build_ledger(args.tests)

    if args.list:
        for name, r in sorted(ledger["rows"].items()):
            if r["verdict"] == args.list:
                print(f"{name}\t{r['reason']}")
        return 0

    print(json.dumps(ledger, indent=2) if args.json else human(ledger))

    c = ledger["counts"]
    if c.get("phantom", 0) or c.get("failing", 0):
        print("\nFAIL: phantom or failing items present. These block DONE regardless of %.",
              file=sys.stderr)
        return 1
    if args.require is not None and ledger["coverage_pct"] < args.require:
        print(f"\nFAIL: coverage {ledger['coverage_pct']}% < required {args.require}%",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
