"""Direct unit tests of harness.ulp_distance / harness.max_ulp_distance and
the ItemSpec provenance-refusal guard in registry.py.

These are the load-bearing edge-case tests for
reports/ionp-ulp-tolerance-decision-2026-08-01.md's stated (not just
implemented) rules:
  - NaN-ness must match exactly (both-NaN = 0, exactly-one-NaN = +inf).
  - +-inf: same-signed = 0, anything else touching an inf = +inf.
  - +0.0 vs -0.0 = +inf (deliberately NOT within any finite tolerance).
  - otherwise: real ULP count via the biased-integer total-order mapping.
  - undeclared tolerance is structurally impossible (ItemSpec refuses to
    construct).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import _bootstrap  # noqa: F401
from harness import max_ulp_distance, ulp_distance
from registry import ItemSpec


# ---------------------------------------------------------------------------
# ulp_distance: NaN
# ---------------------------------------------------------------------------

def test_both_nan_is_zero_distance():
    assert ulp_distance(float("nan"), float("nan"), 32) == 0.0
    assert ulp_distance(float("nan"), float("nan"), 64) == 0.0


def test_one_nan_is_infinite_distance():
    assert ulp_distance(float("nan"), 0.0, 32) == math.inf
    assert ulp_distance(1.0, float("nan"), 32) == math.inf


# ---------------------------------------------------------------------------
# ulp_distance: +-inf
# ---------------------------------------------------------------------------

def test_same_signed_infinities_are_zero_distance():
    assert ulp_distance(math.inf, math.inf, 32) == 0.0
    assert ulp_distance(-math.inf, -math.inf, 64) == 0.0


def test_opposite_signed_infinities_are_infinite_distance():
    assert ulp_distance(math.inf, -math.inf, 32) == math.inf


def test_inf_vs_finite_is_infinite_distance():
    assert ulp_distance(math.inf, 1e30, 32) == math.inf
    assert ulp_distance(1.0, math.inf, 64) == math.inf


# ---------------------------------------------------------------------------
# ulp_distance: +-0.0 (the deliberate policy exception)
# ---------------------------------------------------------------------------

def test_same_signed_zero_is_zero_distance():
    assert ulp_distance(0.0, 0.0, 32) == 0.0
    assert ulp_distance(-0.0, -0.0, 32) == 0.0


def test_opposite_signed_zero_is_infinite_distance():
    """The deliberate exception: a pure bit-order transform would put
    +0.0/-0.0 at distance 0 (they are adjacent in the biased-integer
    mapping), but this harness treats sign-of-zero as structurally
    meaningful (branch cuts in complex sqrt/log/atan2 depend on it) and
    refuses to ever tolerate it -- see harness.py's ulp_distance docstring."""
    assert ulp_distance(0.0, -0.0, 32) == math.inf
    assert ulp_distance(-0.0, 0.0, 64) == math.inf


# ---------------------------------------------------------------------------
# ulp_distance: ordinary finite values
# ---------------------------------------------------------------------------

def test_identical_finite_values_are_zero_distance():
    assert ulp_distance(1.5, 1.5, 32) == 0.0
    assert ulp_distance(3.14159, 3.14159, 64) == 0.0


def test_adjacent_float32_representable_values_are_one_ulp():
    a = np.float32(1.0)
    b = np.nextafter(a, np.float32(np.inf))
    assert ulp_distance(float(a), float(b), 32) == 1.0


def test_two_nextafter_steps_is_two_ulp():
    a = np.float32(1.0)
    b = np.nextafter(np.nextafter(a, np.float32(np.inf)), np.float32(np.inf))
    assert ulp_distance(float(a), float(b), 32) == 2.0


def test_negative_adjacent_values_are_one_ulp():
    a = np.float32(-1.0)
    b = np.nextafter(a, np.float32(-np.inf))
    assert ulp_distance(float(a), float(b), 32) == 1.0


def test_distance_crosses_zero_correctly():
    """The smallest positive and smallest negative subnormals are exactly 2
    ULP apart (one step each side of zero), not some huge integer -- this is
    exactly the case the biased-integer total-order mapping exists to get
    right, as opposed to a naive raw-bit-pattern subtraction (which would be
    wildly wrong across the sign change since IEEE754's sign-magnitude
    layout is not monotonic in raw bits)."""
    smallest_pos = np.nextafter(np.float32(0.0), np.float32(1.0))
    smallest_neg = np.nextafter(np.float32(0.0), np.float32(-1.0))
    assert ulp_distance(float(smallest_pos), float(smallest_neg), 32) == 2.0


def test_float64_width_is_respected():
    a = np.float64(1.0)
    b = np.nextafter(a, np.float64(np.inf))
    assert ulp_distance(float(a), float(b), 64) == 1.0
    # The same two float64 values interpreted at width=32 would not
    # generally be 1 ULP apart -- but more importantly, a huge float64 value
    # like 1e300 isn't even representable at width=32, so this asserts the
    # two widths are NOT interchangeable, guarding against a width
    # parameter being silently ignored.
    assert ulp_distance(1e300, float(np.nextafter(np.float64(1e300), np.float64(np.inf))), 64) == 1.0


# ---------------------------------------------------------------------------
# ulp_distance: float16 (IEEE754 binary16) -- 2026-08-01 width=16 support.
# Mirrors the float32/float64 sections above. Values are narrowed via
# `np.float16(x).item()` before being handed to `ulp_distance`, per the
# function's documented contract that callers pre-round to the target
# width.
# ---------------------------------------------------------------------------

def _f16(x) -> float:
    """Narrow a Python float/int to float16 precision and hand back a plain
    Python float, exactly as `ulp_distance`'s contract requires callers to
    do -- see harness.py's `ulp_distance` docstring."""
    return np.float16(x).item()


def test_adjacent_float16_representable_values_are_one_ulp():
    """Walk several magnitudes with np.nextafter and confirm each adjacent
    pair is exactly 1.0 ULP apart. Deliberately does NOT start a walk at
    6e4: float16's max finite value is 65504.0, and a walk starting that
    close to the top overflows to inf mid-walk -- overflow behavior is
    correct policy (see test_float16_nextafter_above_max_finite_is_inf
    below) but does not belong mixed into this adjacency test."""
    for magnitude in (0.5, 1.0, 2.0, 100.0, 1e-5, -1.0, -1e3):
        a = np.float16(magnitude)
        b = np.nextafter(a, np.float16(np.inf), dtype=np.float16)
        assert ulp_distance(_f16(a), _f16(b), 16) == 1.0, f"magnitude={magnitude}"


def test_two_nextafter_steps_is_two_ulp_float16():
    a = np.float16(1.0)
    b = np.nextafter(np.nextafter(a, np.float16(np.inf), dtype=np.float16),
                      np.float16(np.inf), dtype=np.float16)
    assert ulp_distance(_f16(a), _f16(b), 16) == 2.0


def test_identical_float16_values_are_zero_distance():
    assert ulp_distance(_f16(1.5), _f16(1.5), 16) == 0.0
    assert ulp_distance(_f16(-3.25), _f16(-3.25), 16) == 0.0


def test_float16_one_vs_one_point_five_is_exactly_512_ulp():
    """float16 has a 10-bit mantissa, so between 1.0 and 2.0 one ULP is
    2**-10; 0.5 / 2**-10 = 512 exactly."""
    assert ulp_distance(_f16(1.0), _f16(1.5), 16) == 512.0


def test_float16_width_is_actually_respected_not_silently_graded_as_float32():
    """The single most important test in this section: the SAME pair of
    Python floats (1.0, 1.5) must grade to 512.0 ULP at width=16 but
    4194304.0 ULP at width=32 -- proving `ulp_distance`'s width parameter
    genuinely dispatches to the binary16 bit-packing path instead of
    silently falling through to (or being conflated with) float32."""
    assert ulp_distance(1.0, 1.5, 16) == 512.0
    assert ulp_distance(1.0, 1.5, 32) == 4194304.0


def test_float16_special_cases_match_existing_policy():
    assert ulp_distance(0.0, -0.0, 16) == math.inf
    assert ulp_distance(float("nan"), float("nan"), 16) == 0.0
    assert ulp_distance(float("nan"), 0.0, 16) == math.inf
    assert ulp_distance(math.inf, _f16(65504.0), 16) == math.inf
    assert ulp_distance(math.inf, math.inf, 16) == 0.0
    assert ulp_distance(-math.inf, -math.inf, 16) == 0.0


def test_adjacent_float16_subnormals_are_one_ulp():
    """float16 subnormals live below ~6.1e-5 (2**-14). Adjacent subnormals
    must still be exactly 1 ULP apart -- the total-order integer mapping
    must not break down in the subnormal range."""
    smallest = np.nextafter(np.float16(0.0), np.float16(1.0), dtype=np.float16)
    next_smallest = np.nextafter(smallest, np.float16(1.0), dtype=np.float16)
    assert float(smallest) < 6.1e-5
    assert ulp_distance(_f16(smallest), _f16(next_smallest), 16) == 1.0


def test_float16_subnormal_normal_boundary_crossing_is_one_ulp():
    """The largest subnormal and the smallest normal float16 value are
    adjacent representable values and must be exactly 1 ULP apart, the same
    as any other adjacent pair -- this is exactly the boundary a naive
    exponent-only bit trick tends to get wrong."""
    smallest_normal = np.float16(2.0 ** -14)
    largest_subnormal = np.nextafter(smallest_normal, np.float16(0.0), dtype=np.float16)
    assert ulp_distance(_f16(largest_subnormal), _f16(smallest_normal), 16) == 1.0


def test_float16_nextafter_above_max_finite_is_inf():
    """float16 max finite is 65504.0; stepping one ULP further overflows to
    inf, and comparing against that inf must report math.inf (an
    inf-vs-finite case), not some huge-but-finite ULP count."""
    max_finite = np.float16(65504.0)
    assert float(max_finite) == 65504.0
    one_past = np.nextafter(max_finite, np.float16(np.inf), dtype=np.float16)
    assert np.isinf(one_past)
    assert ulp_distance(_f16(max_finite), float(one_past), 16) == math.inf


def test_width_for_dtype_float16():
    from harness import _width_for_dtype
    assert _width_for_dtype(np.dtype("float16")) == 16


def test_max_ulp_distance_float16_array_all_exact():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    b = a.copy()
    assert max_ulp_distance(a, b, a.dtype) == 0.0


def test_max_ulp_distance_float16_array_takes_the_worst_element():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    b = a.copy()
    b[1] = np.nextafter(
        np.nextafter(b[1], np.float16(np.inf), dtype=np.float16),
        np.float16(np.inf), dtype=np.float16,
    )
    assert max_ulp_distance(a, b, a.dtype) == 2.0


# ---------------------------------------------------------------------------
# max_ulp_distance: array + complex-component-wise behavior
# ---------------------------------------------------------------------------

def test_max_ulp_distance_float32_array_all_exact():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = a.copy()
    assert max_ulp_distance(a, b, a.dtype) == 0.0


def test_max_ulp_distance_takes_the_worst_element():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = a.copy()
    b[1] = np.nextafter(np.nextafter(b[1], np.float32(np.inf)), np.float32(np.inf))
    assert max_ulp_distance(a, b, a.dtype) == 2.0


def test_max_ulp_distance_short_circuits_to_inf_on_nan_mismatch():
    a = np.array([1.0, float("nan")], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    assert max_ulp_distance(a, b, a.dtype) == math.inf


def test_max_ulp_distance_complex_is_worst_of_real_and_imag_components():
    a = np.array([1.0 + 2.0j], dtype=np.complex64)
    b_real_off = np.array(
        [complex(np.nextafter(np.float32(1.0), np.float32(np.inf)), 2.0)],
        dtype=np.complex64,
    )
    assert max_ulp_distance(a, b_real_off, a.dtype) == 1.0

    b_imag_off = np.array(
        [complex(1.0, float(np.nextafter(np.nextafter(
            np.float32(2.0), np.float32(np.inf)), np.float32(np.inf))))],
        dtype=np.complex64,
    )
    assert max_ulp_distance(a, b_imag_off, a.dtype) == 2.0


def test_max_ulp_distance_complex_zero_sign_mismatch_is_infinite():
    a = np.array([0.0 + 0.0j], dtype=np.complex64)
    b = np.array([complex(-0.0, 0.0)], dtype=np.complex64)
    assert max_ulp_distance(a, b, a.dtype) == math.inf


# ---------------------------------------------------------------------------
# Provenance guard: undeclared tolerance must be structurally impossible.
# ---------------------------------------------------------------------------

def test_itemspec_refuses_ulp_tolerance_without_justification():
    with pytest.raises(ValueError, match="no recorded ulp_justification"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance=1)


def test_itemspec_refuses_ulp_tolerance_with_blank_justification():
    with pytest.raises(ValueError, match="no recorded ulp_justification"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance=1, ulp_justification="   ")


def test_itemspec_accepts_ulp_tolerance_with_real_justification():
    spec = ItemSpec(
        name="selftest.ok", kind="unary",
        ulp_tolerance={"float32": 1.0},
        ulp_justification="see KNOWN-DIFFERENCES.md: measured X",
        ulp_sweep={"float32": (20000, 1.0)},
    )
    assert spec.ulp_tolerance == {"float32": 1.0}


def test_itemspec_rejects_negative_ulp_tolerance():
    with pytest.raises(ValueError, match=r"ulp_tolerance\['float32'\] must be >= 0"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance={"float32": -1},
                 ulp_justification="x", ulp_sweep={"float32": (20000, 0.0)})


def test_itemspec_refuses_non_dict_ulp_tolerance():
    """A single item-wide scalar is exactly the defect this task fixes --
    ulp_tolerance must be a per-dtype {dtype_name: bound} dict."""
    with pytest.raises(ValueError, match="must be a non-empty"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance=1,
                 ulp_justification="x", ulp_sweep={"float32": (20000, 1.0)})


def test_itemspec_refuses_empty_dict_ulp_tolerance():
    with pytest.raises(ValueError, match="must be a non-empty"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance={},
                 ulp_justification="x", ulp_sweep={"float32": (20000, 1.0)})


def test_itemspec_default_has_no_ulp_tolerance():
    spec = ItemSpec(name="selftest.plain", kind="unary")
    assert spec.ulp_tolerance is None
    assert spec.ulp_justification is None


# ---------------------------------------------------------------------------
# Provenance guard, sample-size (adequacy) side -- 2026-08-01 sample-size-
# defect fix. A ulp_justification string alone is not evidence of an
# adequately-SIZED sample: absolute/abs's original 1.0 ULP bound had a real
# justification string attached and was still wrong (true max 2.0 ULP) because
# it was only measured over the ~150-case differential corpus. These tests
# are the anti-tautology proof that the new `ulp_sweep` gate actually fires,
# not just that it is documented.
# ---------------------------------------------------------------------------

def test_itemspec_refuses_ulp_tolerance_without_sweep_evidence():
    with pytest.raises(ValueError, match="no recorded ulp_sweep evidence"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance={"float32": 1.0},
                 ulp_justification="a real justification string, but no sweep")


def test_itemspec_refuses_ulp_tolerance_with_empty_sweep_dict():
    with pytest.raises(ValueError, match="no recorded ulp_sweep evidence"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance={"float32": 1.0},
                 ulp_justification="x", ulp_sweep={})


def test_itemspec_refuses_sweep_smaller_than_minimum_sample_size():
    with pytest.raises(ValueError, match="below the required minimum"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance={"float32": 1.0},
                 ulp_justification="x", ulp_sweep={"float32": (150, 1.0)})


def test_itemspec_refuses_declared_tolerance_above_measured_max():
    # Padding "to be safe" is exactly how the original 1.0 ULP figure went
    # unchecked -- the gate requires equality with the measured max, not >=.
    with pytest.raises(ValueError, match="does not equal"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance={"float32": 5.0},
                 ulp_justification="x", ulp_sweep={"float32": (20000, 2.0)})


def test_itemspec_refuses_declared_tolerance_below_measured_max():
    # Under-declaring is exactly the original defect (declared 1.0, true 2.0).
    with pytest.raises(ValueError, match="does not equal"):
        ItemSpec(name="selftest.bad", kind="unary", ulp_tolerance={"float32": 1.0},
                 ulp_justification="x", ulp_sweep={"float32": (20000, 2.0)})


def test_itemspec_accepts_sweep_matching_declared_tolerance_exactly():
    spec = ItemSpec(name="selftest.ok", kind="unary", ulp_tolerance={"float32": 2.0},
                     ulp_justification="x", ulp_sweep={"float32": (20000, 2.0)})
    assert spec.ulp_tolerance == {"float32": 2.0}


# ---------------------------------------------------------------------------
# Provenance guard, PER-DTYPE grading -- 2026-08-01 per-dtype-grading defect
# fix. The postmortem: `absolute`/`abs`'s own re-swept evidence was already
# per-dtype (float32/float64 bit-exact at 0.0 ULP, complex64/complex128
# genuinely imprecise at 2.0 ULP) but was declared as ONE item-wide scalar
# (2.0), which gave the bit-exact real dtypes 2 ULP of unearned slack -- a
# genuine 2-ULP regression on float32 `abs` would have passed silently.
# These are the anti-tautology proof that each dtype is now checked against
# ITS OWN measured max, not the worst dtype's.
# ---------------------------------------------------------------------------

def test_itemspec_checks_each_dtype_against_its_own_measured_max_not_the_worst():
    # This is exactly the shape of the original defect: declaring the ITEM's
    # worst-dtype figure (2.0, from complex64) as float32's own tolerance.
    # Per-dtype grading must reject it -- float32's own measured max is 0.0.
    with pytest.raises(ValueError, match="does not equal"):
        ItemSpec(name="selftest.bad", kind="unary",
                 ulp_tolerance={"float32": 2.0, "complex64": 2.0},
                 ulp_justification="x",
                 ulp_sweep={"float32": (20000, 0.0), "complex64": (20000, 2.0)})
    # The correct per-dtype declaration: each dtype gets its OWN measured
    # bound, so float32 is bit-exact (0.0) while complex64 gets its real 2.0.
    spec = ItemSpec(name="selftest.ok", kind="unary",
                     ulp_tolerance={"float32": 0.0, "complex64": 2.0},
                     ulp_justification="x",
                     ulp_sweep={"float32": (20000, 0.0), "complex64": (20000, 2.0)})
    assert spec.ulp_tolerance == {"float32": 0.0, "complex64": 2.0}


def test_itemspec_refuses_tolerance_dtype_with_no_matching_sweep_entry():
    # A dtype declared tolerant with no sweep evidence for THAT dtype (even
    # though a different dtype on the same item does have evidence) must be
    # refused -- a tolerance can't ride in on a sibling dtype's evidence.
    with pytest.raises(ValueError, match="do not match"):
        ItemSpec(name="selftest.bad", kind="unary",
                 ulp_tolerance={"float32": 0.0, "complex64": 2.0},
                 ulp_justification="x",
                 ulp_sweep={"float32": (20000, 0.0)})


def test_itemspec_refuses_swept_dtype_with_no_declared_tolerance():
    # The mirror image: a dtype that WAS swept but has no explicit declared
    # tolerance entry must also be refused -- it must not silently fall back
    # to bit-exact "by omission" at construction time; the omission must be
    # a loud, structural error, not an emergent default (the emergent
    # default only applies at GRADING time to a dtype outside the item's
    # declared set entirely -- see compare_values).
    with pytest.raises(ValueError, match="do not match"):
        ItemSpec(name="selftest.bad", kind="unary",
                 ulp_tolerance={"float32": 0.0},
                 ulp_justification="x",
                 ulp_sweep={"float32": (20000, 0.0), "complex64": (20000, 2.0)})


# ---------------------------------------------------------------------------
# Provenance guard, epsilon (atol/rtol) side -- 2026-08-01 truthfulness-hole
# fix, PLUS the 2026-08-01 "hardened door / open window" closure
# (reports/ionp-hardened-door-open-window-2026-08-01.md): the legacy
# atol/rtol path originally had no guard at all (any non-zero value was
# accepted silently, which is how ndarray.__add__/__radd__/__mul__/__rmul__
# were credited "bit-exact" while actually graded via a looser np.allclose
# comparison); a non-empty epsilon_justification string was then added as a
# gate, but a bare string was found to be the ENTIRE gate -- no sample-size
# floor, no per-dtype keying, no check that the declared bound matched any
# real measurement (exactly how 11 items later acquired unearned tolerance
# the same way). ItemSpec now refuses ANY non-zero atol/rtol outright,
# regardless of justification text -- `epsilon_tolerance` (below) is the
# evidence-gated replacement. These tests assert the current, final
# contract.
# ---------------------------------------------------------------------------

def test_itemspec_refuses_nonzero_atol_regardless_of_justification():
    with pytest.raises(ValueError, match="structurally forbidden"):
        ItemSpec(name="selftest.bad", kind="unary", atol=1e-9, rtol=1e-9)


def test_itemspec_refuses_nonzero_rtol_alone_regardless_of_justification():
    with pytest.raises(ValueError, match="structurally forbidden"):
        ItemSpec(name="selftest.bad", kind="unary", atol=0.0, rtol=1e-9)


def test_itemspec_refuses_nonzero_atol_with_blank_justification():
    with pytest.raises(ValueError, match="structurally forbidden"):
        ItemSpec(name="selftest.bad", kind="unary", atol=1e-9, rtol=1e-9,
                 epsilon_justification="   ")


def test_itemspec_refuses_nonzero_atol_even_with_real_justification():
    """The legacy path is not a weak gate any more -- it is closed. No
    justification string, however genuine, makes a non-zero legacy
    atol/rtol constructible. See epsilon_tolerance for the replacement."""
    with pytest.raises(ValueError, match="structurally forbidden"):
        ItemSpec(
            name="selftest.bad", kind="unary", atol=1e-9, rtol=1e-9,
            epsilon_justification="measured 2026-08-01: see coverage report X",
        )


def test_itemspec_accepts_explicit_zero_atol_rtol_without_justification():
    """atol=0.0, rtol=0.0 is not a tolerance -- it is a restatement of
    bit-exact via the legacy call sites, so it needs no justification."""
    spec = ItemSpec(name="selftest.exact", kind="unary", atol=0.0, rtol=0.0)
    assert spec.atol == 0.0 and spec.rtol == 0.0
    assert spec.epsilon_justification is None


def test_itemspec_default_has_no_atol_rtol():
    spec = ItemSpec(name="selftest.plain2", kind="unary")
    assert spec.atol is None and spec.rtol is None
    assert spec.epsilon_justification is None


# ---------------------------------------------------------------------------
# compare_values: the epsilon path actually grades looser than bit-exact,
# is correctly marked non-bit-exact ("tolerant") only when it was actually
# needed, and reports the real ULP distance as a diagnostic even though the
# pass/fail decision came from np.allclose, not ULP.
# ---------------------------------------------------------------------------

def test_compare_values_epsilon_path_requires_justification_even_bypassing_itemspec():
    """Defense in depth: harness.compare_values itself refuses to grade
    under an undocumented non-zero atol/rtol, independent of whether the
    caller went through ItemSpec.__post_init__ at all."""
    from harness import compare_values
    a = np.array([1.0], dtype=np.float64)
    b = np.array([1.0 + 1e-10], dtype=np.float64)
    with pytest.raises(RuntimeError, match="no recorded epsilon_justification"):
        compare_values(a, b, atol=1e-9, rtol=1e-9)


def test_compare_values_epsilon_path_bit_exact_input_is_not_marked_tolerant():
    from harness import compare_values
    a = np.array([1.0, 2.5], dtype=np.float64)
    b = np.array([1.0, 2.5], dtype=np.float64)
    ok, detail, tolerant, max_ulp = compare_values(
        a, b, atol=1e-9, rtol=1e-9, epsilon_justification="test fixture",
    )
    assert ok
    assert tolerant is False  # didn't need the slack -- genuinely bit-exact
    assert max_ulp == 0.0


def test_compare_values_epsilon_path_marks_tolerant_when_slack_was_used():
    from harness import compare_values
    a = np.array([1.0], dtype=np.float64)
    b = np.array([1.0 + 1e-10], dtype=np.float64)  # differs, but within atol/rtol
    ok, detail, tolerant, max_ulp = compare_values(
        a, b, atol=1e-9, rtol=1e-9, epsilon_justification="test fixture",
    )
    assert ok
    assert tolerant is True  # DID need the slack -- not bit-exact
    assert max_ulp > 0


def test_compare_values_explicit_zero_atol_rtol_is_graded_bit_exact_not_epsilon():
    """atol=0.0, rtol=0.0 must route to the strict ULP==0 default path, not
    np.allclose -- so it needs no epsilon_justification and any real
    mismatch fails, exactly like the undeclared-tolerance default."""
    from harness import compare_values
    a = np.array([1.0], dtype=np.float64)
    b = np.array([1.0 + 1e-15], dtype=np.float64)  # tiny but nonzero ULP distance
    ok, detail, tolerant, max_ulp = compare_values(a, b, atol=0.0, rtol=0.0)
    assert not ok
    assert "no tolerance declared" in detail


# ---------------------------------------------------------------------------
# compare_values: PER-DTYPE ULP grading (2026-08-01 fix). This is the direct
# anti-tautology proof for the task: a real `absolute`/`abs`-shaped
# ulp_tolerance dict ({"float32": 0.0, ..., "complex64": 2.0, ...}) must
# reject a float32 result that is only 1 ULP off (it would have PASSED under
# the old item-wide 2.0 scalar), while still accepting a genuinely
# 2-ULP-off complex64 result and still rejecting a 3-ULP-off one.
# ---------------------------------------------------------------------------

_ABS_SHAPED_TOLERANCE = {"float32": 0.0, "float64": 0.0,
                          "complex64": 2.0, "complex128": 2.0}


def _nudge_real_ulps(arr, n: int, dtype):
    """Move every element exactly n ULPs (n >= 0) toward +inf via repeated
    np.nextafter -- builds a result KNOWN to be exactly n ULPs off, mirroring
    selftest.py's `_nudge_ulps` helper (kept local here so this file's
    fixtures stay self-contained and don't depend on selftest.py's private
    helper)."""
    out = np.asarray(arr, dtype=dtype).copy()
    target = np.array(np.inf, dtype=dtype)
    for _ in range(n):
        out = np.nextafter(out, target).astype(dtype)
    return out


def _nudge_complex64_real_component_ulps(arr, n: int):
    """Move only the REAL component of every complex64 element exactly n
    ULPs toward +inf, leaving the imaginary component untouched -- gives a
    complex64 array with a known, exact per-element ULP distance (the max of
    real/imag component distances, per harness.max_ulp_distance's own
    definition), without needing to nudge both components."""
    real = _nudge_real_ulps(arr.real.astype(np.float32), n, np.float32)
    return (real + 1j * arr.imag.astype(np.float32)).astype(np.complex64)


def test_per_dtype_grading_float32_one_ulp_off_fails_under_bit_exact_entry():
    """The core anti-tautology check: under the real absolute/abs-shaped
    per-dtype tolerance dict, a float32 result 1 ULP off must FAIL, even
    though the SAME dict declares 2.0 ULP of slack for complex64/complex128
    on this exact same item. Under the OLD item-wide scalar (2.0, the
    dict's max), this would have PASSED -- proving the tightening actually
    bites."""
    from harness import compare_values
    a = np.array([1.0, 2.5, 3.0], dtype=np.float32)
    b_one_ulp_off = _nudge_real_ulps(a, 1, np.float32)
    ok, detail, tolerant, max_ulp = compare_values(
        a, b_one_ulp_off, atol=None, rtol=None,
        ulp_tolerance=_ABS_SHAPED_TOLERANCE,
        ulp_justification="test fixture: absolute/abs-shaped per-dtype tolerance",
    )
    assert not ok
    assert max_ulp == 1.0
    assert "float32" in detail


def test_per_dtype_grading_float32_bit_exact_still_passes():
    from harness import compare_values
    a = np.array([1.0, 2.5, 3.0], dtype=np.float32)
    ok, detail, tolerant, max_ulp = compare_values(
        a, a.copy(), atol=None, rtol=None,
        ulp_tolerance=_ABS_SHAPED_TOLERANCE,
        ulp_justification="test fixture: absolute/abs-shaped per-dtype tolerance",
    )
    assert ok
    assert tolerant is False
    assert max_ulp == 0.0


def test_per_dtype_grading_complex64_two_ulp_off_still_passes():
    """The real, evidenced numpy imprecision (measured: np.abs on complex64
    disagrees with hypotf by up to 2 ULP) must still be tolerated -- the
    tightening must not have collaterally broken the genuine complex64
    slack this same dict declares."""
    from harness import compare_values
    a = np.array([1.0 + 1.0j, 3.0 + 4.0j], dtype=np.complex64)
    b_two_ulp_off = _nudge_complex64_real_component_ulps(a, 2)
    ok, detail, tolerant, max_ulp = compare_values(
        a, b_two_ulp_off, atol=None, rtol=None,
        ulp_tolerance=_ABS_SHAPED_TOLERANCE,
        ulp_justification="test fixture: absolute/abs-shaped per-dtype tolerance",
    )
    assert ok
    assert tolerant is True
    assert max_ulp == 2.0


def test_per_dtype_grading_complex64_three_ulp_off_fails():
    from harness import compare_values
    a = np.array([1.0 + 1.0j, 3.0 + 4.0j], dtype=np.complex64)
    b_three_ulp_off = _nudge_complex64_real_component_ulps(a, 3)
    ok, detail, tolerant, max_ulp = compare_values(
        a, b_three_ulp_off, atol=None, rtol=None,
        ulp_tolerance=_ABS_SHAPED_TOLERANCE,
        ulp_justification="test fixture: absolute/abs-shaped per-dtype tolerance",
    )
    assert not ok
    assert max_ulp == 3.0


def test_per_dtype_grading_unlisted_dtype_defaults_to_bit_exact():
    """A dtype with NO entry in the declared ulp_tolerance dict at all
    (float16 is not part of anionpy's dtype set, but this proves the mechanism
    generically -- any dtype absent from the dict) must default to 0.0 ULP
    (bit-exact), never silently inherit another dtype's slack (e.g.
    complex64's 2.0). This is the deliberate, enforced safe default named in
    registry.py's MIN_ULP_SWEEP_N docstring."""
    from harness import compare_values
    tolerance_without_float64 = {"float32": 0.0, "complex64": 2.0}
    a = np.array([1.0, 2.5], dtype=np.float64)
    b_one_ulp_off = _nudge_real_ulps(a, 1, np.float64)
    ok, detail, tolerant, max_ulp = compare_values(
        a, b_one_ulp_off, atol=None, rtol=None,
        ulp_tolerance=tolerance_without_float64,
        ulp_justification="test fixture: dtype intentionally absent from the dict",
    )
    assert not ok
    assert max_ulp == 1.0


def test_per_dtype_grading_uses_operand_dtype_not_result_dtype_for_abs_shaped_item():
    """Reproduces the real abs/absolute shape directly: `np.abs` on a
    complex64 OPERAND produces a float32 RESULT (numpy drops the imaginary
    part into a real magnitude) -- but the 2.0-ULP slack this item declares
    belongs to the complex64 INPUT regime (see ulp_sweep.py's `_summarize`,
    which keys evidence by the swept input dtype), not to "float32" in
    general. Grading such a case must use `source_dtype` (the operand's
    dtype, complex64 here) to look up the 2.0-ULP bound -- even though the
    VALUES being compared are float32 arrays -- or it collides with true
    float32-input `abs`, which is bit-exact under this same dict.

    This is the actual bug found and fixed during implementation: an
    earlier version of compare_values() keyed the lookup by the RESULT's
    own dtype (float32) and would have wrongly rejected this genuine,
    evidenced 2-ULP complex64 imprecision at float32's 0.0 bound."""
    from harness import compare_values
    complex64_operand = np.array([1.0 + 1.0j, 3.0 + 4.0j], dtype=np.complex64)
    # np.abs's actual float32 output for this operand, then nudged 2 ULP --
    # exactly the magnitude of imprecision the real sweep measured.
    magnitudes = np.abs(complex64_operand)
    assert magnitudes.dtype == np.float32
    nudged = _nudge_real_ulps(magnitudes, 2, np.float32)

    # Correct: graded with source_dtype=complex64 (the real operand dtype)
    # -> hits the 2.0 ULP complex64 entry -> passes.
    ok, detail, tolerant, max_ulp = compare_values(
        magnitudes, nudged, atol=None, rtol=None,
        ulp_tolerance=_ABS_SHAPED_TOLERANCE,
        ulp_justification="test fixture: absolute/abs-shaped per-dtype tolerance",
        source_dtype=np.dtype(np.complex64),
    )
    assert ok
    assert tolerant is True
    assert max_ulp == 2.0

    # Proves the fix actually matters: WITHOUT source_dtype, the same
    # comparison falls back to the RESULT's dtype (float32, 0.0 ULP
    # declared) and must FAIL -- this is the bug that was found and fixed.
    ok_no_source, detail_no_source, _, max_ulp_no_source = compare_values(
        magnitudes, nudged, atol=None, rtol=None,
        ulp_tolerance=_ABS_SHAPED_TOLERANCE,
        ulp_justification="test fixture: absolute/abs-shaped per-dtype tolerance",
    )
    assert not ok_no_source
    assert max_ulp_no_source == 2.0
    assert "float32" in detail_no_source
