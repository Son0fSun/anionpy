"""pytest entry point for the real differential suite (the one that feeds
tools/coverage.py). One parametrized test per REGISTRY item, all sharing
the exact same case-building and grading logic as run.py -- so `pytest
tests/differential/` and `python3 tests/differential/run.py --out r.json`
can never quietly disagree about which items pass.
"""
import _bootstrap  # noqa: F401
import pytest

import ufunc_registry  # noqa: F401  (import side effect: merges 134 ufunc items into REGISTRY)
from harness import evaluate
from registry import REGISTRY
from run import build_cases


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_item(name):
    spec = REGISTRY[name]
    cases = build_cases(spec)
    result = evaluate(spec, cases)
    assert result.verdict == "pass", (
        f"{name}: {result.reason}\n" + "\n".join(result.failures)
    )
