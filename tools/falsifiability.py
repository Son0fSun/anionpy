#!/usr/bin/env python3
"""Falsifiability audit: does each declared item's test have the ABILITY to fail?

*** STATUS: VALIDATED against a known-bad snapshot. Still not a clean bill. ***
Validated by reproducing, from HEAD, the result of a hand audit that predates
this file. Known-bad snapshot = commit `b784819` (the `numpy.testing` block
before its falsifiability fixes), where 21 of 47 items passed with a wrong
implementation installed:

    v1, module-attr patching only ....  could not reach 135/199 items
    v2, resolve_ionp override .......   0 of 21 detected
    v3, adapter-aware ...............   7 of 21 detected
    v4, real-module-attr basis ......  21 of 21 detected   <- current
    hand audit ......................  21 of 21  (the reference)

and, on the CURRENT fixed block, 47 audited / 47 caught / 0 unfalsifiable.

REPRODUCING THIS REQUIRES TWO THINGS, and both have burned someone already:
  * the project venv (`ionp/.venv/bin/python`). A bare `python3` has no pytest,
    so `testing.test`'s adapter raises identically on BOTH sides and the
    harness reads that as a pass. It also has a different numpy (2.4.2 vs
    2.5.1) and a stale installed `anionpy`.
  * the audited block actually merged into `registry.REGISTRY`. The
    `numpy.testing` block was orphaned for most of 2026-08-01, so this tool
    matched zero items -- and refused, rather than printing a clean bill for
    an empty set. Keep that refusal.

WHY THIS IS STILL NOT A CLEAN BILL OF HEALTH
A non-empty UNFALSIFIABLE list is a hard finding. An empty one means only that
these substitutions did not fool these tests. A "wrong value of the wrong
TYPE" escalation was tried and deliberately REJECTED: a realistically broken
`IS_64BIT` still returns a real bool, so a type-violating substitute proves
nothing and manufactures a false CAUGHT -- the same self-flattering failure
this file exists to catch, one layer down. See the comment above
`_verdict_of()`.

THE GAP THIS CLOSES
===================
The ledger already refuses two kinds of lie:

  * `phantom`  -- an item declared but not actually present
  * the tolerance gates -- a bound wider than the evidence measured for it

Neither of them asks the more basic question: **if the implementation were
replaced by the wrong answer, would the test notice?**

It has to be asked, because the project has now been bitten by "green tick that
exercises nothing" twice:

  1. `reports/ionp-hardened-door-open-window-2026-08-01.md` -- 18 linalg items
     passed while rejecting anionpy's own array type. The harness fed numpy arrays
     to both sides, so the anionpy side was never exercised. Coverage reverted
     59 -> 41.
  2. 2026-08-01, `numpy.testing` -- 47/47 reported passing; 21 of those tests
     still passed when the implementation under them was replaced with a
     deliberately wrong object.

Both were found by hand, late, after a number had already been reported. This
makes it a standing, machine-checked question instead.

METHOD
======
For each declared item:

  1. resolve the ionp-side object
  2. substitute a deliberately WRONG object for it
       bool     -> negated          int    -> +12345
       str      -> "MONDAY_WRONG"   class  -> an unrelated class
       callable -> lambda that ignores its arguments and returns None
  3. do this by BOTH substitution strategies (override `spec.resolve_ionp`,
     and patch the attribute on the real `anionpy` module), because each is blind
     exactly where the other sees -- see `audit()`. An item where no
     substitution took effect is reported UNAUDITED, never CAUGHT, never clean:
     a probe that silently fails to reach its target reports "everything is
     fine", which is the exact failure mode this file exists to catch.
  4. run the item's real differential cases
  5. the test SHOULD now fail. If it still passes, the test cannot fail, and
     the item's green tick means nothing.

WHAT A FAILURE HERE MEANS
=========================
`UNFALSIFIABLE` is not a style complaint. It means the item is credited on the
strength of a test that would grade a wrong implementation as correct, and it
should be treated exactly like `absent` until fixed -- per `GOAL-ionp.md`, an
item counts when it is "backed by a passing differential test against real
numpy", and a test that cannot fail is not one.

Deliberately NOT auto-wired into coverage.py's headline number: that file is
the scoreboard and changing what it counts mid-flight is how a denominator
quietly drifts. This reports alongside it.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tests", "differential"))


class _MondayWrongClass(Exception):
    """Substituted for class-valued items; unrelated to anything real."""


def wrongify(value):
    """Return something of the same broad shape that is definitely WRONG."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return "MONDAY_WRONG"
    if isinstance(value, int):
        return value + 12345
    if isinstance(value, float):
        return value + 12345.0
    if isinstance(value, type):
        return _MondayWrongClass
    if callable(value):
        return lambda *a, **k: None
    return "MONDAY_WRONG"


# REJECTED DIRECTION -- read before reaching for this again.
#
# The obvious-looking next move is: when `wrongify(value)` (same-shape wrong
# value) leaves an item's verdict unchanged, escalate to a substitute that
# violates the PROPERTY too -- swap a bool-valued constant for a plain
# object entirely, so `isinstance(x, bool)` itself goes False. It was built,
# and it "worked": IS_64BIT/IS_WASM/HAS_REFCOUNT/NOGIL_BUILD/verbose/etc. all
# flipped from CAUGHT-wrongly to CAUGHT-correctly... except they didn't --
# they flipped from (accidentally, for unrelated reasons -- see below)
# CAUGHT to CAUGHT again, past the item this tool exists to name:
# `isinstance(x, bool)` cannot EVER distinguish a wrong bool from a right
# one, for any value substitution, by construction. Forcing it to fail via a
# type it was never checking against doesn't demonstrate the adapter can
# catch a wrong IMPLEMENTATION -- a real broken IS_64BIT still returns a
# real bool, just the wrong one -- it demonstrates the adapter can catch an
# object that isn't a bool at all, which no real bug ever produces. That is
# a second, more subtle instance of the exact "manufactures its own
# positive result" trap `audit()`'s docstring already names for the
# `resolve` strategy: a CAUGHT verdict purchased through a channel the real
# grading logic doesn't exercise. Scoring these items CAUGHT on that basis
# would be the tool re-lying to itself. They stay UNFALSIFIABLE, correctly,
# and that IS the finding -- these `numpy.testing` constants are graded by
# an isinstance-only check with no way to ever fail on a same-type wrong
# value, full stop. (Class-shaped items are a different story: an
# `isinstance(x, Exception)` check genuinely CAN be tricked by a
# same-*shape* wrong class the same way a bool check can -- but there the
# realistic sabotage is "swap in some other class that also isn't
# Exception-derived", which is exactly what `wrongify()`'s `_MondayWrongClass`
# already is, no escalation needed; see its docstring.)
def _verdict_of(spec, build_cases, evaluate) -> str:
    try:
        return evaluate(spec, build_cases(spec)).verdict
    except Exception as exc:
        # A harness crash under sabotage still counts as the test noticing.
        return f"fail (harness raised {type(exc).__name__})"


def _sub_resolve(spec, wrong):
    """Strategy A: override the spec's own ionp-side resolution."""
    spec.resolve_ionp = lambda _w=wrong: _w  # type: ignore[method-assign]
    return lambda: delattr(spec, "resolve_ionp")


def _locate_module_attr(name, import_module):
    """Find (mod, attr) for a dotted item name on the real `anionpy` module, or
    None if it cannot be resolved. Split out of the old `_sub_module` so the
    audit loop can read the REAL current attribute value at that location --
    see `audit()` for why that matters -- separately from installing a
    substitute there.
    """
    head, _, tail = name.partition(".")
    if not tail:
        return import_module("anionpy"), head
    try:
        return import_module(f"anionpy.{head}"), tail
    except ModuleNotFoundError:
        return None


def _sub_module(mod, attr, wrong):
    """Strategy B: replace the attribute on the real anionpy module.

    Reaches specs that grade through a custom adapter and therefore never
    consult `resolve_ionp` -- which is most of the `numpy.testing` block.
    """
    if not hasattr(mod, attr):
        return None
    original = getattr(mod, attr)
    try:
        setattr(mod, attr, wrong)
    except (AttributeError, TypeError):
        return None  # PyO3 classes refuse attribute assignment
    return lambda: setattr(mod, attr, original)


def audit(registry, build_cases, evaluate):
    """Substitute a wrong object at the spec's OWN resolution boundary.

    An earlier version of this patched `module.attr` instead. That could not
    reach 135 of 199 items -- `ndarray.*` lives on a PyO3 class whose
    attributes cannot be monkeypatched at all, and absent items have no
    attribute to patch. It therefore reported "0 unfalsifiable" while auditing
    a third of the ledger, which is the same over-broad clean bill this tool
    exists to prevent, committed by the tool itself.

    Overriding `resolve_ionp` reaches every item by construction -- and
    detected **0 of the 21 known-unfalsifiable `numpy.testing` items** that
    motivated writing this file, because specs graded through a custom adapter
    never consult `resolve_ionp`.

    Each strategy is blind exactly where the other sees. Running one and
    generalising to "the ledger is falsifiable" is the same over-broad claim
    this tool exists to catch -- committed twice, by the tool itself, in one
    sitting. So both run, and an item counts as falsifiable only if some
    *effective* substitution made its test fail. If no substitution took
    effect, the item is UNAUDITED and reported as such -- never as clean.

    THE MODULE STRATEGY'S WRONG VALUE COMES FROM THE REAL ATTRIBUTE
    ------------------------------------------------------------------
    `wrongify()` needs to see the object actually living at
    `anionpy.testing.IS_64BIT` (a plain bool) to substitute a same-type wrong
    value (`not True`). But `spec.resolve_ionp()` does NOT hand that back
    for a `kind="custom"` item with an `ionp_adapter`: it always returns a
    wrapped CALLABLE (see `ItemSpec._wrap_custom_conversion`), regardless of
    whether the thing it wraps is itself a bool, a class, or a function.
    Basing `wrongify`'s input on that wrapper instead of the real attribute
    silently reintroduces the exact "manufactures a positive result" trap
    this function's docstring already warns about for the `resolve`
    strategy, just one layer down: `wrongify(<wrapped callable>)` always
    hits the `callable(value)` branch and returns a no-op lambda, so a bool
    constant gets replaced by a *function* -- which does happen to fail an
    `isinstance(x, bool)` check, but for a reason that has nothing to do
    with whether the check could ever catch a wrong BOOL. It is a coin flip
    dressed as a finding: it happens to work here only because a function
    is not a bool, not because the substitution was a deliberate
    wrong-value-of-the-right-type probe. See `_locate_module_attr` --
    reading the real attribute keeps `wrongify` doing what its name says,
    for every item, on purpose, rather than by whatever shape
    `resolve_ionp()`'s wrapping happens to produce.

    A TYPE-VIOLATING ESCALATION PASS WAS TRIED AND REJECTED
    -----------------------------------------------------------
    See the standalone comment above `_verdict_of` for the full account:
    forcing `isinstance(x, bool)` to fail by substituting a non-bool proves
    the check can be fooled by an object no real implementation bug would
    ever produce, not that it can catch a wrong boolean. That is the same
    manufactured-positive shape as the paragraph above, and it would have
    hidden the genuine finding (these constants ARE unfalsifiable under
    their current adapters) behind a false CAUGHT. No escalation pass runs;
    `wrongify()` alone, aimed at the right object, is the whole fix.
    """
    import importlib

    caught, unfalsifiable, skipped = [], [], []

    for name in sorted(registry):
        spec = registry[name]
        try:
            original = spec.resolve_ionp()
        except Exception as exc:
            skipped.append((name, f"resolve_ionp raised {type(exc).__name__}"))
            continue
        if original is None:
            # Not implemented. Already graded `absent`/fail by the harness;
            # there is no claim here to falsify.
            skipped.append((name, "absent -- nothing implemented to falsify"))
            continue

        # BASELINE FIRST. A test that fails unconditionally also fails under
        # sabotage, and would otherwise be scored CAUGHT. "This test can detect
        # a wrong implementation" is a meaningless thing to say about a test
        # that cannot pass a right one -- the sabotage verdict carries no
        # information unless the un-sabotaged verdict differs from it.
        #
        # Added 2026-08-01 while chasing a bug that turned out not to exist (a
        # misread report encoding made 47 passing items look like 47 failing
        # ones). No item was actually in this state -- which is the reason to
        # close it now rather than after one is, since the tool would have
        # reported those 47 as healthy either way.
        if not str(_verdict_of(spec, build_cases, evaluate)).startswith("pass"):
            skipped.append((name, "BASELINE FAILS -- sabotage proves nothing here"))
            continue

        located = _locate_module_attr(name, importlib.import_module)
        module_attr_present = located is not None and hasattr(*located)
        module_basis = getattr(*located) if module_attr_present else None

        noticed = False
        effective = []

        # ORDER AND ELIGIBILITY MATTER -- this cost a false clean bill once.
        #
        # Overriding `resolve_ionp` does NOT reach a spec that grades through
        # an `ionp_adapter`: the adapter is what the harness actually calls,
        # and it re-reads the real object itself. The override merely hands the
        # comparison a fabricated wrong value, which fails trivially and gets
        # scored CAUGHT -- while the spec's real grading path was never
        # exercised at all.
        #
        # That is exactly how this tool contradicted a correct hand audit on
        # 2026-08-01, reporting CAUGHT for 21 `numpy.testing` items that were
        # genuinely unfalsifiable (e.g. `IS_64BIT`'s adapter graded
        # `isinstance(..., bool)`, so flipping the value changed nothing).
        # A tool that manufactures its own positive result is worse than no
        # tool. So: module substitution is authoritative and runs FIRST, and
        # `resolve` is only eligible when there is no adapter to bypass.
        strategies = []
        if module_attr_present:
            mod, attr = located
            strategies.append((
                "module", module_basis,
                lambda w, _m=mod, _a=attr: _sub_module(_m, _a, w),
            ))
        if getattr(spec, "ionp_adapter", None) is None:
            strategies.append(("resolve", original, lambda w: _sub_resolve(spec, w)))

        for label, basis, install in strategies:
            restore = install(wrongify(basis))
            if restore is None:
                continue
            effective.append(label)
            try:
                if str(_verdict_of(spec, build_cases, evaluate)).startswith("fail"):
                    noticed = True
            finally:
                restore()
            if noticed:
                break

        if not effective:
            skipped.append((name, "no substitution strategy took effect -- UNAUDITED"))
        elif noticed:
            caught.append(name)
        else:
            unfalsifiable.append(name)

    return caught, unfalsifiable, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="",
                    help="only audit items whose name starts with this (e.g. 'testing.')")
    args = ap.parse_args(argv)

    import run as runner  # noqa: E402  (path set above)
    from harness import evaluate  # noqa: E402
    from registry import REGISTRY  # noqa: E402

    registry = {k: v for k, v in REGISTRY.items() if k.startswith(args.prefix)}
    caught, unfalsifiable, skipped = audit(registry, runner.build_cases, evaluate)

    total = len(registry)
    if total == 0:
        # An empty audit printing "0 unfalsifiable" is a clean bill of health
        # for nothing at all -- the same silent-pass shape this tool exists to
        # catch, turned on itself. Refuse loudly instead.
        print(f"REFUSING: no items matched prefix {args.prefix!r} -- an empty audit "
              f"is not a pass. Check the prefix, or that the block is merged "
              f"into registry.REGISTRY at all.", file=sys.stderr)
        return 2
    print(f"items audited        : {total}")
    print(f"CAUGHT (test can fail): {len(caught)}")
    print(f"UNFALSIFIABLE         : {len(unfalsifiable)}")
    print(f"skipped (absent etc.) : {len(skipped)}")

    if unfalsifiable:
        print("\nThese tests pass with a WRONG implementation installed:")
        for n in unfalsifiable:
            print(f"    {n}")
    if skipped:
        print("\nNot audited (NOT a clean bill for these):")
        for n, why in skipped:
            print(f"    {n:48s} {why}")

    # Non-zero exit on unfalsifiable so this can gate a commit later.
    return 1 if unfalsifiable else 0


if __name__ == "__main__":
    raise SystemExit(main())
