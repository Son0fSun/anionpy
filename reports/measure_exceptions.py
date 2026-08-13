#!/usr/bin/env python3
"""One-off measurement script (2026-08-01 blind-spot audit) -- NOT part of
the harness's real behavior. Runs the exact same full registry run.py does,
via the same build_cases()/evaluate() path, but additionally dumps the
message-comparison instrumentation added to harness.py (EXC_STATS /
EXC_DIVERGENCES) so we can measure how many declared "exact" items would
stop being exact if the harness's exception arm also compared message text.

Does not change any verdict computation. Read-only measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import harness
from registry import REGISTRY
from run import run_registry, build_cases  # noqa: F401 (build_cases used indirectly by run_registry via evaluate/build_cases)


def main():
    results = run_registry(REGISTRY)

    n_pass = sum(1 for r in results.values() if r.verdict == "pass")
    n_fail = sum(1 for r in results.values() if r.verdict == "fail")
    total_cases = sum(r.total for r in results.values())

    diverging_items = sorted({d["item"] for d in harness.EXC_DIVERGENCES})

    # one concrete example per diverging item (first occurrence)
    examples = {}
    for d in harness.EXC_DIVERGENCES:
        examples.setdefault(d["item"], d)

    out = {
        "total_items": len(results),
        "items_pass": n_pass,
        "items_fail": n_fail,
        "total_cases_run": total_cases,
        "exc_stats": harness.EXC_STATS,
        "n_diverging_items": len(diverging_items),
        "diverging_items": diverging_items,
        "examples": examples,
        "all_divergences": harness.EXC_DIVERGENCES,
    }
    Path("exc_measure.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"total_items={len(results)} pass={n_pass} fail={n_fail} total_cases={total_cases}")
    print(f"exc_stats={harness.EXC_STATS}")
    print(f"n_diverging_items={len(diverging_items)}")
    print("wrote exc_measure.json")


if __name__ == "__main__":
    main()
