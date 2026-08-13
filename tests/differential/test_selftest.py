"""Proof that the harness catches real bugs, wired into pytest.

Each of these tests PASSES when the harness produces the EXPECTED verdict
for a fixture with a known, deliberate defect (or, for the control case,
no defect at all). If the harness were broken in the "always says pass"
direction, the three broken-shim tests here would fail loudly -- that is
the point of this file.
"""
import _bootstrap  # noqa: F401
import pytest

from harness import evaluate
from run import build_cases
from selftest import EXPECTED_VERDICTS, SELFTEST_REGISTRY


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_VERDICTS.items()))
def test_selftest_detection(name, expected):
    spec = SELFTEST_REGISTRY[name]
    cases = build_cases(spec)
    result = evaluate(spec, cases)
    assert result.verdict == expected, (
        f"harness self-test failed: {name} expected verdict {expected!r}, got "
        f"{result.verdict!r} ({result.reason})\n" + "\n".join(result.failures)
    )
    if expected == "fail":
        # Not just "it failed somehow" -- it must have failed for a reason,
        # with diagnostics, not silently.
        assert result.failures, f"{name} was marked fail but recorded no diagnostic detail"


def test_ulp_tolerant_pass_is_distinguished_from_exact_pass():
    """The ledger (tools/coverage.py) must be able to tell a PASS that
    needed ULP slack from one that was bit-exact -- see
    reports/ionp-ulp-tolerance-decision-2026-08-01.md point 3. This is the
    load-bearing assertion for that requirement: two items that both PASS,
    one of which is not ItemResult.tolerant (bit-exact) and one of which IS
    (needed exactly 1 declared ULP of slack)."""
    exact_spec = SELFTEST_REGISTRY["selftest.ulp_sqrt_correct"]
    exact_result = evaluate(exact_spec, build_cases(exact_spec))
    assert exact_result.verdict == "pass"
    assert exact_result.tolerant is False
    assert exact_result.max_ulp_observed == 0.0

    tolerant_spec = SELFTEST_REGISTRY["selftest.ulp_sqrt_one_ulp_off"]
    tolerant_result = evaluate(tolerant_spec, build_cases(tolerant_spec))
    assert tolerant_result.verdict == "pass"
    assert tolerant_result.tolerant is True
    assert tolerant_result.max_ulp_observed == 1.0
