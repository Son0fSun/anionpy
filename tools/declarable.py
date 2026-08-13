#!/usr/bin/env python
"""Classify the `absent` surface into ACTIONABLE buckets.

`coverage.py` answers "how much is left" (729). It does not answer "left in what
WAY", and those are very different questions: an item nobody has written yet and
an item that is implemented, tested, and passing but simply never got a ledger
line are both "absent", and they cost wildly different amounts to close.

Four buckets, in ascending cost:

  DECLARABLE   implemented + has a corpus test + that test passes.
               Cheapest. But NOT auto-declarable: a passing corpus test is
               necessary, not sufficient. This session proved three separate
               times that a grid varying axes SEPARATELY returns clean and is
               worthless -- two such grids each falsified the other's
               declarations. Every entry here still owes a coordinator-written
               CROSSED grid (container x dtype x out= x casting= x order=
               x dtype=) before it earns a ledger line.

  BROKEN       implemented + has a corpus test + that test FAILS.
               Real engine bugs with a reproducer already written. Note these
               do NOT show up in coverage.py's `failing` counter, which only
               counts DECLARED items -- so `failing 0` coexists with dozens of
               genuinely failing tests. That gap is the point of this tool.

  UNTESTED     implemented, but no corpus test exists. Cannot be declared at
               all until someone writes the differential cases.

  MISSING      not implemented. The genuine build-it-from-scratch remainder.

Usage:
    .venv/bin/python tests/differential/run.py --out /tmp/r.json
    .venv/bin/python tools/declarable.py --tests /tmp/r.json [--bucket DECLARABLE]

(Always pass --tests. coverage.py silently reports 0.0% without it, and this
tool would silently call every item UNTESTED.)
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent


def absent_items(tests: str) -> list[str]:
    """Ask coverage.py for the absent set -- single source of truth.

    Deliberately shells out rather than re-deriving the surface: a second
    implementation of "what counts as absent" would be a second thing to keep
    in sync, and the two would drift silently.
    """
    out = subprocess.run(
        [sys.executable, str(HERE / "coverage.py"), "--tests", tests, "--list", "absent"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if out.returncode != 0:
        sys.exit(f"coverage.py failed:\n{out.stderr}")
    items = []
    for line in out.stdout.splitlines():
        name = line.split("\t")[0].strip()
        if name and not name.startswith("="):
            items.append(name)
    return items


def resolve(root, dotted: str) -> bool:
    """True if anionpy actually exposes this dotted path."""
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True, help="results JSON from tests/differential/run.py")
    ap.add_argument("--bucket", help="print only this bucket's items")
    args = ap.parse_args()

    import anionpy  # noqa: E402  -- after arg parsing so --help works without a build

    results = json.loads(pathlib.Path(args.tests).read_text())
    # The results schema is {item: {verdict, max_ulp, mechanism, tolerant}}.
    # The pass/fail field is `verdict` -- NOT `passed`, NOT `ok`. Guessing that
    # key silently yields an empty failure set and a falsely clean report.
    if results:
        sample = next(iter(results.values()))
        if "verdict" not in sample:
            sys.exit(f"unexpected results schema, fields={sorted(sample)!r}; expected 'verdict'")

    buckets: dict[str, list[str]] = collections.defaultdict(list)
    for item in absent_items(args.tests):
        if not resolve(anionpy, item):
            buckets["MISSING"].append(item)
        elif item not in results:
            buckets["UNTESTED"].append(item)
        elif results[item]["verdict"] == "pass":
            buckets["DECLARABLE"].append(item)
        else:
            buckets["BROKEN"].append(item)

    order = ["DECLARABLE", "BROKEN", "UNTESTED", "MISSING"]
    if args.bucket:
        for name in buckets[args.bucket.upper()]:
            print(name)
        return

    total = sum(len(v) for v in buckets.values())
    print(f"\n  ABSENT SURFACE BREAKDOWN  --  {total} items\n")
    blurb = {
        "DECLARABLE": "impl + test passes  -> owes a CROSSED out-of-corpus grid",
        "BROKEN": "impl + test FAILS   -> engine bug, reproducer exists",
        "UNTESTED": "impl, no test       -> write differential cases first",
        "MISSING": "not implemented     -> build it",
    }
    for name in order:
        print(f"  {name:<12}{len(buckets[name]):>5}   {blurb[name]}")

    print("\n  by namespace:")
    for name in order:
        if not buckets[name]:
            continue
        ns = collections.Counter(i.split(".")[0] if "." in i else "<top>" for i in buckets[name])
        print(f"    {name:<12}{', '.join(f'{k}:{v}' for k, v in ns.most_common(8))}")

    for name in ("DECLARABLE", "BROKEN"):
        print(f"\n  {name}:")
        for item in sorted(buckets[name]):
            print(f"    {item}")
    print()


if __name__ == "__main__":
    main()
