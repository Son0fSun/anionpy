"""NEW test-cases file: rigorous differential coverage for the ufunc
`dtype=` keyword axis -- the axis the pre-existing differential corpus NEVER
exercised, for every ufunc currently declared "exact" in
`anionpy/_state/toplevel.py`.

WHY THIS FILE EXISTS
---------------------------------------------------------------------------
`anionpy/_state/toplevel.py` carries several "<- do not restore without input
precasting" withdrawal comments (grep for that string). They record: real
numpy resolves an explicit `ufunc(..., dtype=X)` call by finding the loop in
the ufunc's own declared type table whose OUTPUT matches `X`, then casting
the INPUT operand(s) into THAT loop's declared input dtype(s) before
computing -- not by computing on the operand's native dtype and casting the
OUTPUT afterward. The two models diverge on two documented axes: bool input
(`np.negative(bool_arr, dtype=np.int64)` succeeds in real numpy, historically
raised in anionpy) and complex<->real narrowing in either direction
(`np.absolute(complex128_arr, dtype=np.float32)` raises `UFuncTypeError` in
real numpy, historically silently succeeded with the wrong value in anionpy).
A 56-op x 9-in x 9-target sweep measured 63 divergent cases concentrated in
sign/abs/absolute/negative/subtract.

The withdrawal note is explicit about *why* the pre-existing corpus missed
this for as long as it did: "The existing differential tests never passed
`dtype=`, so they agreed with the declaration because neither looked --
which is why coverage still read `failing 0` while the declaration was
false." That is: a check graded at a bar the real requirement never
claimed. THIS file is the corpus that closes that hole permanently, for
every currently-declared-exact ufunc, not just the five the original
measurement happened to name (a prior investigation of this same defect
under-reported its size by only characterizing cases it happened to hit --
see `ufunc_dtype_grid_cases.py`'s own module docstring, which made the same
point about its own broader-than-5 scope).

This file's job is honest measurement, not declaration. It does not touch
`anionpy/_state/toplevel.py` and does not "fix" anything -- if a currently
declared-exact ufunc genuinely fails on this axis, that is real evidence for
whoever owns the ledger to act on, not something this file may weaken,
tolerate, or skip around.

WHY THIS IS A SEPARATE FILE/NAMESPACE, NOT MORE ENTRIES ON THE REAL
"add"/"negative"/etc. ITEMS IN ufunc_registry.py -- AND WHY IT DUPLICATES
NEITHER `ufunc_dtype_grid_cases.py` NOR `ufunc_order_cases.py`
---------------------------------------------------------------------------
Same registry-key-collision reason `ufunc_order_cases.py` already documented
for its own "order/ufunc/<name>" namespace: "add"/"negative"/... are already
declared by `ufunc_registry.py`, and the tail-merge pattern both files use
refuses to redeclare an existing name. This file follows that exact
precedent, one level up: fresh `"dtype/ufunc/<name>"` keys, `kind="custom"`
ItemSpecs, merged into `registry.REGISTRY` the same way.

`ufunc_dtype_grid_cases.py` (already in this directory, wired to the fix
that landed in `ionp-core/src/ufunc.rs`/`ionp-py/src/lib.rs`) already proves
this exact axis -- but as a standalone `pytest.mark.parametrize` module, NOT
as `kind="custom"` ItemSpecs feeding `registry.REGISTRY`. That means it
never produces entries in the `{item: verdict}` JSON `tests/differential/
run.py --out` emits, and therefore never reaches `tools/coverage.py
--tests` or shows up as a keyed, per-ufunc result a ledger owner can audit
one item at a time. That gap is the second half of what this task closes:
`ufunc_order_cases.py` itself was proof that a corpus module living in this
directory, uncommitted-to-run.py, "sat unimported for a long time and
therefore never ran" -- see run.py's own import comment. This file is
deliberately wired into run.py's REGISTRY-driven pipeline (see the bottom of
this file and run.py's import list) so that mistake cannot repeat here: its
item keys are asserted, by `test_differential.py`/CI, to actually appear as
keys in a real `run.py --out` JSON, not just in a separately-invoked pytest
module.

The two files are complementary, not redundant: `ufunc_dtype_grid_cases.py`
is the broader net (every ufunc it can build a sample for, catching class
AND exact message text via its own hand-rolled comparison) that FOUND and
evidenced the original defect; this file is the same axis re-expressed as
first-class, ledger-visible registry items, scoped specifically to the
ufuncs currently claiming "exact" in toplevel.py -- the set whose false
"failing 0" is the actual thing worth being unable to happen again.

SCOPE
---------------------------------------------------------------------------
Every ufunc in `anionpy/_state/toplevel.py`'s `TOPLEVEL_STATE` dict currently
mapped to `"exact"` AND shaped like a plain, single-output, non-gufunc,
non-string ufunc with a registered `ufunc_registry.UFUNC_SPECS` entry --
computed live against the CURRENT ledger state at import time (not a frozen
hand-typed list), so this corpus tracks whatever is declared "exact" right
now, including ufuncs declared after this file was written. As of this
writing that is 80 ufuncs (a superset of the ~58 the task brief estimated --
`acos`/`arccos`, `asin`/`arcsin`, `atan`/`arctan`, `atan2`/`arctan2`,
`atanh`/`arctanh`, `bitwise_not`/`invert`, `mod`/`remainder` are each counted
as two distinct declared names, plus `degrees`/`rad2deg` since 2026-08-02).
`abs`/`absolute`/`sign`/`negative`/`subtract` are correctly ABSENT: they are
currently withdrawn in toplevel.py specifically for this defect, so they are
not "currently declared exact" and this corpus does not claim to cover them
(see `ufunc_dtype_grid_cases.py` for their own standing evidence instead).

Crossed with all 11 dtypes named in the task brief -- bool, int8, int32,
int64, uint8, uint32, float16, float32, float64, complex64, complex128 --
as BOTH the input operand dtype and the explicit `dtype=` target, i.e. 121
(input, target) pairs per ufunc. Binary ufuncs (nin=2) use the SAME sampled
array for both operands (mirroring `ufunc_dtype_grid_cases.py`'s own
`(a, a)` convention) -- the defect under test is loop selection + input
precasting, which is per-operand-dtype, not an inter-operand interaction, so
one shared operand dtype per case is the right unit of coverage; every
input dtype still gets its own full row of 11 targets.

FURTHER crossed with `casting=` (Phase-2 Task B, see `CASTINGS` below): each
of the 121 (input, target) pairs is exercised once with no `casting=` kwarg
at all (numpy's default, 'same_kind' -- the original, unmodified case shape)
plus once each under 'no', 'equiv', and 'unsafe' -- the three casting rules
whose strictness differs from the default in a way `ionp-core/src/ufunc.rs`'s
`resolve_unary_output_dtype`/`resolve_binary_output_dtype` actually branch
on. That is 484 (input, target, casting) cases per ufunc. None of the
original 121-per-ufunc default-casting cases were removed, renamed in a way
that changes their registry-visible label's default-casting variant, or
weakened -- they are the `casting is None` branch in `_cases_for_ufunc`
below, byte-for-byte the same `(label, args, kwargs)` shape as before this
task. This crossing is exactly the shape of the coordinator's own probe
that found the 94-mismatch defect (7 ufuncs x 10 in x 10 target x 6
casting values) re-expressed as first-class, ledger-visible registry
items for every currently-declared-exact ufunc, the same reason this file
exists at all rather than relying solely on `ufunc_dtype_grid_cases.py`'s
standalone pytest sweep (see that section above).

This deliberately, explicitly includes bool input (the entire DTYPES tuple
starts with `"bool"`) and complex64/complex128 input crossed against every
real target dtype (`float16`/`float32`/`float64`/every int/uint/bool) --
the exact two divergence regions the withdrawal note names. It also
includes the reverse direction (real input, complex target), which the
original 56x9x9 sweep also covered and is included here for the same
reason: the note describes divergence "in BOTH directions".

RIGOR: every case here is a `kind="custom"` `ItemSpec`, which routes through
the SAME `harness.run_case`/`evaluate` machinery every other registry item
uses (see harness.py) -- not a hand-rolled comparison. That machinery
already compares, per case: exception CLASS (`type(exc)`, checked against
`exception_equivalences`, not `isinstance`), exception MESSAGE (`str(exc)`,
scrubbed only for provably nondeterministic spans -- see harness.py's
`_normalize_exc_message`), result dtype (exact `np.dtype` equality), result
shape (exact tuple equality), and result values (bit-exact by default,
`np.array_equal`/byte-for-byte `tobytes()` comparison, ULP-tolerant only if
this item's underlying `ufunc_registry.UFUNC_SPECS` entry already declares
an evidenced `ulp_tolerance` for the dtype in question -- reused verbatim
below, exactly as `ufunc_order_cases.py` reuses `base_spec.atol`/`rtol`/
`ulp_tolerance` for its own items). No component is ever compared against
its own self-reported metadata: `harness.compare_values`/`_bit_exact_equal`
round-trip every result through `np.asarray(...)` before comparing dtype,
shape, or bytes, on both sides.

NUMPY IS ORACLE ONLY. Every reference value/exception in this file's cases
comes from a real, live call to `np.<name>(..., dtype=...)` inside
`harness.run_case` -- this module never calls numpy to compute anionpy's
answer or borrows a numpy-derived string for anionpy's expected exception
text; it only supplies (label, args, kwargs) tuples for the harness to run
both sides itself.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
import anionpy
import registry
import ufunc_introspect
from registry import ItemSpec

# registry.py's own `_DEFAULT_UFUNC_EXC_EQUIV` (applied to every ItemSpec
# regardless of which module builds it, per registry.py's __post_init__)
# maps two of numpy's PRIVATE `UFuncTypeError` subclasses to anionpy's own
# public `UFuncTypeError`: `_UFuncNoLoopError` (probed via `np.sign` on
# bool) and `_UFuncOutputCastingError` (probed via `out=`). The `dtype=`
# axis this file exercises reaches a THIRD, distinct private subclass
# neither of those probes trips: `_UFuncInputCastingError` -- raised when
# the resolved loop's OWN input dtype cannot receive the operand under the
# active casting rule (e.g. `np.negative(int8_arr, dtype=np.uint8)`:
# 'same_kind' forbids int8->uint8). All three display identically as
# "UFuncTypeError" via numpy's `@_display_as_base` and are real, distinct
# classes under `type()` -- `harness.run_case` compares by exact type
# object, not `isinstance`/name, so a case that is otherwise a byte-for-
# byte-identical, fully correct anionpy answer (verified live: anionpy's message
# for this shape already matches numpy's, word for word) would spuriously
# register as an "exception type mismatch" without this. Probed live here,
# the same pattern `registry.py` itself uses for its two default entries --
# not guessed, not imported from numpy's private module path.
_UFUNC_INPUT_CASTING_ERROR = registry._numpy_exc_type(
    np.negative, np.array([1], dtype=np.int8), dtype=np.uint8,
)
_EXTRA_UFUNC_EXC_EQUIV = {
    _UFUNC_INPUT_CASTING_ERROR: {anionpy._anionpy.UFuncTypeError},
}

DTYPES: tuple[str, ...] = (
    "bool", "int8", "int32", "int64", "uint8", "uint32",
    "float16", "float32", "float64", "complex64", "complex128",
)

# Extends this corpus's original dtype=-only axis to CROSS it with
# `casting=`, per the coordinator's Phase-2 Task B brief: the
# `resolve_unary_output_dtype`/`resolve_binary_output_dtype` fix landed in
# `ionp-core/src/ufunc.rs` (commit "ufunc dtype=/casting= resolution...")
# changed casting-rule handling generically, not per-op -- and the
# coordinator's own crossed probe (7 ufuncs x 10 in x 10 target x 6
# casting values) is what found the 94-mismatch defect that fix closes.
# This file's pre-existing cases only ever call `ufunc(..., dtype=X)`
# with NO `casting=` kwarg, i.e. numpy's default ('same_kind') -- so they
# never touched the 'no'/'equiv' strictness path the fix changed at all
# (mirroring this same file's own docstring point about the ORIGINAL
# defect: "a check graded at a bar the real requirement never claimed").
#
# `None` here is a sentinel for "no casting= kwarg at all" -- i.e. the
# ALREADY-EXISTING case shape this file has always produced, kept
# unmodified (not weakened, not removed, not deduplicated away) below.
# 'safe' is deliberately omitted: numpy's default IS 'same_kind', a
# strictly WEAKER rule than 'safe', so a 'safe'-only case would be
# strictly implied by neither the default nor 'unsafe' cases and adds a
# fifth axis point without evidence it exercises new code (both resolvers
# treat 'safe'/'same_kind' identically in `candidate_input_reachable`'s
# `crate::dtype::can_cast` delegation) -- 'no'/'equiv'/'unsafe' are the
# three rules whose STRICTNESS differs from the default in a way the
# fixed code path actually branches on.
CASTINGS: tuple[str | None, ...] = (None, "no", "equiv", "unsafe")

# Two representative values per input dtype (positive and negative/second,
# where the dtype's domain allows it) -- enough to exercise real values/
# shape comparison (a 1-element sample can't distinguish "shape preserved"
# from "shape collapsed"), while staying small enough that 80 ufuncs x 11
# input dtypes x 11 targets (9,680 cases total) runs in reasonable time.
# Deliberately mixed-sign (not just "1"/"1.0"/"True") because the defect
# under test is loop selection + input CASTING, which the sign of the value
# does not change, but a mixed-sign sample still exercises negative-value
# casting into unsigned targets (wraps/truncates identically on numpy and
# anionpy only if anionpy's cast is genuinely the same operation, not a
# coincidence a same-signed sample could hide).
_VALUES: dict[str, list] = {
    "bool": [True, False],
    "int8": [3, -3],
    "int32": [7, -7],
    "int64": [11, -11],
    "uint8": [2, 5],
    "uint32": [4, 9],
    "float16": [1.5, -2.25],
    "float32": [1.5, -2.25],
    "float64": [1.5, -2.25],
    "complex64": [1 + 2j, -1.5 - 0.5j],
    "complex128": [1 + 2j, -1.5 - 0.5j],
}


def _sample(dtype_name: str) -> np.ndarray:
    return np.array(_VALUES[dtype_name], dtype=np.dtype(dtype_name))


def _cases_for_ufunc(name: str) -> list[tuple]:
    info = ufunc_introspect.describe(name)
    if info.signature is not None or info.is_string or info.nout != 1:
        return []
    if info.nin not in (1, 2):
        return []

    cases: list[tuple] = []
    for in_dtype in DTYPES:
        a = _sample(in_dtype)
        args = (a,) if info.nin == 1 else (a, a)
        for target in DTYPES:
            for casting in CASTINGS:
                if casting is None:
                    # Original, unmodified case shape -- kept exactly as
                    # before (same label, same kwargs) so no pre-existing
                    # case is weakened, renamed, or removed.
                    label = f"{name}/in_{in_dtype}/target_{target}"
                    kwargs = {"dtype": np.dtype(target)}
                else:
                    label = f"{name}/in_{in_dtype}/target_{target}/casting_{casting}"
                    kwargs = {"dtype": np.dtype(target), "casting": casting}
                cases.append((label, args, kwargs))
    return cases


def _currently_declared_exact_ufuncs() -> list[str]:
    """The live scope of this corpus: every name currently mapped to
    `"exact"` in `anionpy/_state/toplevel.py`'s `TOPLEVEL_STATE`, intersected
    with `ufunc_introspect.UFUNC_NAMES` (so a non-ufunc "exact" entry, e.g.
    a plain function or ndarray method sharing a name, is never mistaken
    for a ufunc) -- see this module's docstring for why this is computed
    live rather than hand-typed. `abs`/`absolute`/`sign`/`negative`/
    `subtract` are correctly excluded whenever they are withdrawn (as they
    are as of this writing): this corpus's whole point is to only claim
    coverage for what the ledger currently asserts, never to editorialize
    about what SHOULD be declared.
    """
    import importlib
    toplevel = importlib.import_module("anionpy._state.toplevel")
    exact_names = {k for k, v in toplevel.TOPLEVEL_STATE.items() if v == "exact"}
    ufunc_names = set(ufunc_introspect.UFUNC_NAMES)
    return sorted(exact_names & ufunc_names)


def _build_dtype_specs() -> dict[str, ItemSpec]:
    import ufunc_registry  # local import: avoid a module-load-order cycle
    specs: dict[str, ItemSpec] = {}
    for name in _currently_declared_exact_ufuncs():
        base_spec = ufunc_registry.UFUNC_SPECS.get(name)
        if base_spec is None:
            continue
        cases = _cases_for_ufunc(name)
        if not cases:
            continue
        key = f"dtype/ufunc/{name}"
        exc_equiv = dict(base_spec.exception_equivalences or {})
        for np_exc_type, ionp_types in _EXTRA_UFUNC_EXC_EQUIV.items():
            exc_equiv[np_exc_type] = exc_equiv.get(np_exc_type, set()) | ionp_types
        specs[key] = ItemSpec(
            name=key,
            kind="custom",
            numpy_path=name,
            ionp_path=name,
            atol=base_spec.atol,
            rtol=base_spec.rtol,
            ulp_tolerance=base_spec.ulp_tolerance,
            ulp_justification=base_spec.ulp_justification,
            ulp_sweep=base_spec.ulp_sweep,
            exception_equivalences=exc_equiv,
            custom_cases=(lambda n=name: _cases_for_ufunc(n)),
        )
    return specs


DTYPE_UFUNC_SPECS: dict[str, ItemSpec] = _build_dtype_specs()

registry.REGISTRY.update(DTYPE_UFUNC_SPECS)
