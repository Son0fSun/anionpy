"""Proof-of-detection fixtures: deliberately WRONG shim functions used to
demonstrate that the differential harness actually catches bugs, plus one
deliberately CORRECT shim as a control.

None of these names exist in tools/numpy_surface.json ("selftest.*" is not
a numpy item), so this registry is never fed to `coverage.py --tests` --
it exists purely to answer the question "does this harness actually work,
or does it just always print pass?"

Run it with:
    python3 tests/differential/run.py --selftest --verbose
or via pytest:
    pytest tests/differential/test_selftest.py -v
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
from registry import ItemSpec, _reshape_forms, build_ufunc_dispatcher


def correct_sum(arr):
    """Control case: behaves exactly like np.sum. If the harness can't mark
    THIS as pass, nothing above it can be trusted either."""
    return np.sum(arr)


def wrong_dtype_sum(arr):
    """Bug: always downcasts to float32, regardless of what numpy would
    return for this input's dtype (float64 stays float64, int64 stays
    int64, etc). A real bug class: silent precision loss."""
    result = np.sum(arr)
    return np.float32(result)


def off_by_tolerance_sum(arr):
    """Bug: adds a fixed 1e-3 offset to the true sum. Graded bit-exact
    (atol=rtol=0.0, see SELFTEST_REGISTRY below), so this must be caught
    even though 1e-3 looks "close" to a human skimming output."""
    result = np.sum(arr)
    return np.asarray(result, dtype=np.float64) + 1e-3


def no_raise_on_bad_broadcast_add(a, b):
    """Bug: numpy raises ValueError on incompatible shapes; this shim
    swallows that and silently truncates to the shorter length instead of
    raising -- a real class of bug (silent data corruption instead of a
    loud failure)."""
    try:
        return np.add(a, b)
    except ValueError:
        af, bf = np.asarray(a).ravel(), np.asarray(b).ravel()
        n = min(af.size, bf.size)
        return af[:n] + bf[:n]


def reshape_all_call_forms(arr, *args, **kwargs):
    """Control case: a correct reshape that supports every argument shape
    numpy does (bare int, varargs ints, tuple, list, -1 inference, order=
    kwarg, ...) by simply delegating to the real numpy ndarray METHOD (not
    the top-level np.reshape function -- that one only takes a single shape
    argument, no varargs; a.reshape(*shape) is method-only numpy behavior).
    If the new call-form engine can't mark THIS as pass, the engine itself
    is broken."""
    return np.asarray(arr).reshape(*args, **kwargs)


def reshape_tuple_form_only(arr, *args, **kwargs):
    """The bug this whole module exists to catch, reproduced as a minimal
    shim: a reshape that only implements the single-tuple call form
    (`a.reshape((2, 2))`) and raises TypeError for every other form numpy
    accepts (`a.reshape(2, 2)`, `a.reshape(-1)`, `a.reshape(4)`, list form,
    the `order=` keyword). This is, line for line, the shape of the real
    `anionpy.ndarray.reshape` bug reported 2026-07-31: 'ndarray.reshape' was
    marked PASS by the old harness because it only ever exercised the tuple
    form.

    The OLD harness (single args_provider, one call form) would have
    credited this shim a clean PASS: it is byte-for-byte correct on the one
    form it was tested with. The NEW harness -- this same shim run through
    registry._reshape_forms(), the identical CallForm library the real
    `ndarray.reshape` item uses -- MUST mark it FAIL, because most of that
    library's forms are not the tuple form. That is the capability this
    task added, demonstrated end to end without touching anionpy at all.
    """
    if len(args) == 1 and isinstance(args[0], tuple) and not kwargs:
        return np.reshape(arr, args[0])
    raise TypeError(
        "reshape_tuple_form_only: only a single tuple-shape positional "
        "argument is supported (no bare int, no varargs, no list, no "
        "order= keyword)"
    )


# --- proof of the ufunc-block call-form/dispatcher machinery (ufunc_cases.py
# / registry.build_ufunc_dispatcher / registry.resolve_*'s kind="ufunc"
# branches). These deliberately-wrong shims are run through the EXACT SAME
# case set the real "add"/"sqrt" REGISTRY items would get from
# ufunc_cases.build_cases_for_item -- ItemSpec below sets numpy_path="add"
# (or "sqrt") so run.py's build_cases() dispatches through the real
# introspection/applicability machinery, not a hand-picked easy subset. The
# ionp_adapter itself is the fake ufunc-like object wrapped through the SAME
# build_ufunc_dispatcher() the real (currently-absent) anionpy side will use
# once ufuncs exist, so this also proves the dispatcher's own call-form
# routing ("call"/"reduce"/"accumulate"/"outer"/"reduceat"/"at") is not a
# tautology.

class _FakeAddBase:
    """Forwards every ufunc-protocol entry point to the real np.add, so each
    subclass below is a single, isolated defect against a known-correct
    baseline -- not a from-scratch reimplementation that could hide multiple
    bugs at once."""

    def __call__(self, *args, **kwargs):
        return np.add(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        return np.add.reduce(*args, **kwargs)

    def accumulate(self, *args, **kwargs):
        return np.add.accumulate(*args, **kwargs)

    def outer(self, *args, **kwargs):
        return np.add.outer(*args, **kwargs)

    def reduceat(self, *args, **kwargs):
        return np.add.reduceat(*args, **kwargs)

    def at(self, *args, **kwargs):
        return np.add.at(*args, **kwargs)


class _FakeAddCorrect(_FakeAddBase):
    """Control: forwards everything, changes nothing. If the harness can't
    mark THIS pass, registry.build_ufunc_dispatcher itself is broken, not
    any hypothetical anionpy bug."""


class _FakeAddWrongReduce(_FakeAddBase):
    """Bug: .reduce always returns zero(s) of the correct shape/dtype
    regardless of input -- a real bug class (an accidentally-unconditional
    identity-element short-circuit). Only reduce/-derived cases should fail;
    plain call/accumulate/outer/reduceat/at cases still forward correctly,
    proving the harness catches a defect isolated to ONE call form rather
    than needing everything to be broken to notice."""

    def reduce(self, arr, *args, **kwargs):
        real = np.add.reduce(arr, *args, **kwargs)
        return np.zeros_like(real)


class _FakeAddSilentAt(_FakeAddBase):
    """Bug: .at() silently no-ops instead of mutating its target in place.
    numpy's real .at() communicates its effect ONLY via in-place mutation
    (it returns None) -- a broken .at that also returns None is
    indistinguishable from a correct one by return value alone; only
    inspecting the (un)mutated array catches it. This is exactly the
    mutation-safety scenario build_ufunc_dispatcher's docstring names: the
    dispatcher hands `.at()` a *copy* of the target and returns that copy,
    so this fixture's silent no-op shows up as "anionpy's returned array still
    equals the pre-.at input" instead of numpy's actually-incremented one."""

    def at(self, *args, **kwargs):
        return None


class _FakeSqrtWrongCall:
    """A second, differently-shaped ufunc (unary, nin=1) with a defect in
    plain __call__ itself rather than in a method -- np.sqrt(x) + 1 instead
    of np.sqrt(x). Proves the call-form/dispatcher machinery catches a
    defect in the most basic form ("call"), not just in the methods."""

    def __call__(self, *args, **kwargs):
        return np.sqrt(*args, **kwargs) + 1.0

    def at(self, target, indices, *args, **kwargs):
        return np.sqrt.at(target, indices, *args, **kwargs)


# ---------------------------------------------------------------------------
# ULP-tolerance self-check (reports/ionp-ulp-tolerance-decision-2026-08-01.md
# task brief): proof that a declared 1-ULP tolerance still fails a wrong
# formula, a wrong dtype promotion, a wrong special-case branch (NaN and
# +-0.0), and an off-by-2-ULP result -- i.e. the mechanism catches real
# defects and is not a backdoor to a looser bar. `selftest.ulp_sqrt_correct`
# and `selftest.ulp_sqrt_one_ulp_off` are the two POSITIVE controls: a
# bit-exact pass (tolerant=False) and a genuinely-1-ULP-off pass
# (tolerant=True) that the mechanism is supposed to accept.
#
# Fixed corpus (not the seeded random one) so every fixture's exact bit
# pattern is reproducible and inspectable by eye: includes a perfect square
# (bit-exact sqrt), a zero (for the +-0.0 fixture), and a negative value
# (for the NaN fixture) deliberately.
# ---------------------------------------------------------------------------

_ULP_SQRT_CORPUS = np.array([4.0, 1.0, 0.25, 0.0, -1.0], dtype=np.float32)

_ULP_JUSTIFICATION = (
    "selftest fixture, not a real registry item -- exists purely to prove "
    "the ulp_tolerance mechanism still fails wrong formula / wrong dtype / "
    "wrong NaN / wrong zero-sign / off-by-2-ULP, per the self-check required "
    "by reports/ionp-ulp-tolerance-decision-2026-08-01.md."
)

# Companion to _ULP_JUSTIFICATION for the adequate-sweep gate added
# 2026-08-01 (sample-size defect fix, see registry.py's
# `MIN_ULP_SWEEP_N`/`ItemSpec.ulp_sweep` and ulp_sweep.py). This is a
# selftest fixture over a deliberately tiny, fixed, hand-inspectable corpus
# (_ULP_SQRT_CORPUS, 5 values) -- it is not standing in for a real
# measurement, so the "n=20000" here is a fixture satisfying the gate's
# structural shape, not a claim that 20,000 sqrt calls were actually run.
# The real per-dtype adequacy requirement applies to REGISTRY items
# (ufunc_registry.py's _ULP_OVERRIDES), which do carry genuine ulp_sweep.py
# measurements -- see that module.
_ULP_SWEEP_FIXTURE = {"float32": (20000, 1.0)}


def _nudge_ulps(arr, n: int, dtype=np.float32):
    """Move every element of `arr` exactly `n` ULPs (n>=0) in the positive
    direction via repeated np.nextafter -- used to build a result that is
    KNOWN to be exactly n ULPs away from the true value, so the ULP-distance
    fixtures below are testing an exact, known quantity rather than
    "probably about n ULPs off"."""
    out = np.asarray(arr, dtype=dtype).copy()
    target = np.array(np.inf, dtype=dtype)
    for _ in range(n):
        out = np.nextafter(out, target).astype(dtype)
    return out


def ulp_sqrt_correct(arr):
    """Positive control: bit-exact. Must pass with tolerant=False -- if the
    harness can't tell "didn't need the tolerance" from "used it", the
    ledger's tolerant/exact distinction (tools/coverage.py) is a lie."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(arr)


def ulp_sqrt_one_ulp_off(arr):
    """Positive control: every element nudged exactly 1 ULP off the true
    value. Must pass (declared ulp_tolerance={"float32": 1.0}) with
    tolerant=True."""
    with np.errstate(invalid="ignore"):
        return _nudge_ulps(np.sqrt(arr), 1)


def ulp_sqrt_wrong_formula(arr):
    """Defect: a formula bug (multiplies by a constant close to 1) that
    misses by far more than 1 ULP for nonzero inputs -- must still FAIL
    under a 1-ULP tolerance, proving the tolerance doesn't paper over a
    genuinely wrong computation."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(arr) * np.float32(1.001)


def ulp_sqrt_wrong_dtype(arr):
    """Defect: correct value, wrong dtype (float64 instead of float32).
    Must FAIL -- dtype is a structural check that happens before ULP
    comparison is even reached, so a declared tolerance must never be able
    to launder a dtype-promotion bug."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(arr).astype(np.float64)


def ulp_sqrt_wrong_zero_sign(arr):
    """Defect: correct everywhere except sqrt(0.0) comes back as -0.0
    instead of 0.0 -- a wrong special-case branch. Must FAIL under a 1-ULP
    tolerance per this harness's deliberate policy that +-0.0 is NEVER
    within any finite ULP tolerance (see harness.py's ulp_distance
    docstring)."""
    with np.errstate(invalid="ignore"):
        result = np.sqrt(arr).copy()
        zero_mask = np.asarray(arr) == 0.0
        result[zero_mask] = -result[zero_mask]
        return result


def ulp_sqrt_wrong_nan(arr):
    """Defect: sqrt of a negative input should be NaN (numpy's real
    behavior); this shim substitutes 0.0 instead -- a wrong special-case
    branch. Must FAIL: this harness's ulp_distance treats "exactly one side
    NaN" as +inf, always outside any finite tolerance."""
    with np.errstate(invalid="ignore"):
        result = np.sqrt(arr).copy()
        neg_mask = np.asarray(arr) < 0.0
        result[neg_mask] = np.float32(0.0)
        return result


def ulp_sqrt_off_by_two(arr):
    """Defect: every result nudged exactly 2 ULPs off the true value --
    one ULP more than the declared tolerance=1 permits. Must FAIL. This is
    the fixture that most directly answers "is the bound actually 1, or did
    I round it up in practice?"."""
    with np.errstate(invalid="ignore"):
        return _nudge_ulps(np.sqrt(arr), 2)


SELFTEST_REGISTRY: dict[str, ItemSpec] = {
    # --- proof of the plain (non-tolerant) comparison path ------------------
    # 2026-08-01, "hardened door / open window" closure
    # (reports/ionp-hardened-door-open-window-2026-08-01.md): these four used
    # to declare a non-zero legacy atol/rtol=1e-9 specifically to exercise
    # compare_values' np.allclose branch through the full ItemSpec/run.py
    # pipeline. ItemSpec now refuses to construct with ANY non-zero
    # atol/rtol, unconditionally (see registry.py's __post_init__), so that
    # branch is no longer reachable from a registry entry at all -- by
    # design, that is the fix. Graded bit-exact (atol=rtol=0.0) instead:
    # every fixture below is either exactly right (must still PASS) or wrong
    # by far more than one ULP (must still FAIL), so bit-exact grading proves
    # the same detection with a strictly tighter bar, not a weaker one. The
    # legacy np.allclose branch itself is still directly exercised (with a
    # real epsilon_justification) by test_ulp_distance.py's
    # test_compare_values_epsilon_path_* tests, which call
    # harness.compare_values() directly, bypassing ItemSpec entirely.
    "selftest.correct_sum": ItemSpec(
        name="selftest.correct_sum", kind="unary", numpy_path="sum",
        ionp_adapter=correct_sum, atol=0.0, rtol=0.0,
    ),
    "selftest.wrong_dtype_sum": ItemSpec(
        name="selftest.wrong_dtype_sum", kind="unary", numpy_path="sum",
        ionp_adapter=wrong_dtype_sum, atol=0.0, rtol=0.0,
    ),
    "selftest.off_by_tolerance_sum": ItemSpec(
        name="selftest.off_by_tolerance_sum", kind="unary", numpy_path="sum",
        ionp_adapter=off_by_tolerance_sum, atol=0.0, rtol=0.0,
    ),
    "selftest.no_raise_on_bad_broadcast_add": ItemSpec(
        name="selftest.no_raise_on_bad_broadcast_add", kind="binary", numpy_path="add",
        ionp_adapter=no_raise_on_bad_broadcast_add, atol=0.0, rtol=0.0,
    ),

    # --- proof of the call-form-coverage capability added 2026-07-31 -------
    # numpy_path="ndarray.reshape" so build_cases (run.py) drives these
    # through corpus.unary_corpus() x registry._reshape_forms() -- the exact
    # same case-generation path the real `ndarray.reshape` REGISTRY item
    # uses, not a hand-picked easy subset.
    "selftest.reshape_all_call_forms": ItemSpec(
        name="selftest.reshape_all_call_forms", kind="method",
        numpy_path="ndarray.reshape", ionp_adapter=reshape_all_call_forms,
        call_forms=_reshape_forms(), atol=0.0, rtol=0.0,
    ),
    "selftest.reshape_tuple_form_only": ItemSpec(
        name="selftest.reshape_tuple_form_only", kind="method",
        numpy_path="ndarray.reshape", ionp_adapter=reshape_tuple_form_only,
        call_forms=_reshape_forms(), atol=0.0, rtol=0.0,
    ),

    # --- proof of the ufunc-block machinery added for the ufunc block -------
    # kind="ufunc" + numpy_path="add"/"sqrt" so run.py's build_cases()
    # drives these through ufunc_cases.build_cases_for_item -- the exact
    # same introspection-derived case set (plain call, dtype=/out=/where=,
    # reduce/accumulate/outer/reduceat/at) the real "add"/"sqrt" REGISTRY
    # items get once anionpy implements them.
    "selftest.ufunc_add_correct": ItemSpec(
        name="selftest.ufunc_add_correct", kind="ufunc", numpy_path="add",
        ionp_adapter=build_ufunc_dispatcher(_FakeAddCorrect()),
        atol=0.0, rtol=0.0,
    ),
    "selftest.ufunc_add_wrong_reduce": ItemSpec(
        name="selftest.ufunc_add_wrong_reduce", kind="ufunc", numpy_path="add",
        ionp_adapter=build_ufunc_dispatcher(_FakeAddWrongReduce()),
        atol=0.0, rtol=0.0,
    ),
    "selftest.ufunc_add_silent_at": ItemSpec(
        name="selftest.ufunc_add_silent_at", kind="ufunc", numpy_path="add",
        ionp_adapter=build_ufunc_dispatcher(_FakeAddSilentAt()),
        atol=0.0, rtol=0.0,
    ),
    "selftest.ufunc_sqrt_wrong_call": ItemSpec(
        name="selftest.ufunc_sqrt_wrong_call", kind="ufunc", numpy_path="sqrt",
        ionp_adapter=build_ufunc_dispatcher(_FakeSqrtWrongCall()),
        atol=0.0, rtol=0.0,
    ),

    # --- proof of the ULP-tolerance mechanism (2026-08-01 task) -------------
    # kind="custom" over a fixed, hand-inspectable corpus (_ULP_SQRT_CORPUS)
    # so every fixture's exact bit-level behavior is reproducible and
    # legible, not just "probably about right". numpy_path="sqrt" so
    # resolve_numpy() gives the real reference; ionp_adapter is the local
    # fake, never the real package -- see module docstring.
    "selftest.ulp_sqrt_correct": ItemSpec(
        name="selftest.ulp_sqrt_correct", kind="custom", numpy_path="sqrt",
        ionp_adapter=ulp_sqrt_correct,
        custom_cases=lambda: [("fixed", (_ULP_SQRT_CORPUS,), {})],
        ulp_tolerance={"float32": 1.0}, ulp_justification=_ULP_JUSTIFICATION,
        ulp_sweep=_ULP_SWEEP_FIXTURE,
        convert_ionp_args=False,
    ),
    "selftest.ulp_sqrt_one_ulp_off": ItemSpec(
        name="selftest.ulp_sqrt_one_ulp_off", kind="custom", numpy_path="sqrt",
        ionp_adapter=ulp_sqrt_one_ulp_off,
        custom_cases=lambda: [("fixed", (_ULP_SQRT_CORPUS,), {})],
        ulp_tolerance={"float32": 1.0}, ulp_justification=_ULP_JUSTIFICATION,
        ulp_sweep=_ULP_SWEEP_FIXTURE,
        convert_ionp_args=False,
    ),
    "selftest.ulp_sqrt_wrong_formula": ItemSpec(
        name="selftest.ulp_sqrt_wrong_formula", kind="custom", numpy_path="sqrt",
        ionp_adapter=ulp_sqrt_wrong_formula,
        custom_cases=lambda: [("fixed", (_ULP_SQRT_CORPUS,), {})],
        ulp_tolerance={"float32": 1.0}, ulp_justification=_ULP_JUSTIFICATION,
        ulp_sweep=_ULP_SWEEP_FIXTURE,
        convert_ionp_args=False,
    ),
    "selftest.ulp_sqrt_wrong_dtype": ItemSpec(
        name="selftest.ulp_sqrt_wrong_dtype", kind="custom", numpy_path="sqrt",
        ionp_adapter=ulp_sqrt_wrong_dtype,
        custom_cases=lambda: [("fixed", (_ULP_SQRT_CORPUS,), {})],
        ulp_tolerance={"float32": 1.0}, ulp_justification=_ULP_JUSTIFICATION,
        ulp_sweep=_ULP_SWEEP_FIXTURE,
        convert_ionp_args=False,
    ),
    "selftest.ulp_sqrt_wrong_zero_sign": ItemSpec(
        name="selftest.ulp_sqrt_wrong_zero_sign", kind="custom", numpy_path="sqrt",
        ionp_adapter=ulp_sqrt_wrong_zero_sign,
        custom_cases=lambda: [("fixed", (_ULP_SQRT_CORPUS,), {})],
        ulp_tolerance={"float32": 1.0}, ulp_justification=_ULP_JUSTIFICATION,
        ulp_sweep=_ULP_SWEEP_FIXTURE,
        convert_ionp_args=False,
    ),
    "selftest.ulp_sqrt_wrong_nan": ItemSpec(
        name="selftest.ulp_sqrt_wrong_nan", kind="custom", numpy_path="sqrt",
        ionp_adapter=ulp_sqrt_wrong_nan,
        custom_cases=lambda: [("fixed", (_ULP_SQRT_CORPUS,), {})],
        ulp_tolerance={"float32": 1.0}, ulp_justification=_ULP_JUSTIFICATION,
        ulp_sweep=_ULP_SWEEP_FIXTURE,
        convert_ionp_args=False,
    ),
    "selftest.ulp_sqrt_off_by_two": ItemSpec(
        name="selftest.ulp_sqrt_off_by_two", kind="custom", numpy_path="sqrt",
        ionp_adapter=ulp_sqrt_off_by_two,
        custom_cases=lambda: [("fixed", (_ULP_SQRT_CORPUS,), {})],
        ulp_tolerance={"float32": 1.0}, ulp_justification=_ULP_JUSTIFICATION,
        ulp_sweep=_ULP_SWEEP_FIXTURE,
        convert_ionp_args=False,
    ),
}

# Expected verdicts, used by test_selftest.py and printed by run.py --selftest.
EXPECTED_VERDICTS: dict[str, str] = {
    "selftest.correct_sum": "pass",
    "selftest.wrong_dtype_sum": "fail",
    "selftest.off_by_tolerance_sum": "fail",
    "selftest.no_raise_on_bad_broadcast_add": "fail",
    "selftest.reshape_all_call_forms": "pass",
    "selftest.reshape_tuple_form_only": "fail",
    "selftest.ufunc_add_correct": "pass",
    "selftest.ufunc_add_wrong_reduce": "fail",
    "selftest.ufunc_add_silent_at": "fail",
    "selftest.ufunc_sqrt_wrong_call": "fail",
    "selftest.ulp_sqrt_correct": "pass",
    "selftest.ulp_sqrt_one_ulp_off": "pass",
    "selftest.ulp_sqrt_wrong_formula": "fail",
    "selftest.ulp_sqrt_wrong_dtype": "fail",
    "selftest.ulp_sqrt_wrong_zero_sign": "fail",
    "selftest.ulp_sqrt_wrong_nan": "fail",
    "selftest.ulp_sqrt_off_by_two": "fail",
}
