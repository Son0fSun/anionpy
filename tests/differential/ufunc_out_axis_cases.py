"""NEW test-cases file: adds `out=` as a genuinely CROSSED axis for the six
ufuncs `divmod`/`float_power`/`frexp`/`isnat`/`ldexp`/`modf` -- the ones a
prior task (commit 5f1f4c2a, "implement divmod/frexp/modf/float_power/
ldexp/isnat + fix floor_divide underflow") newly implemented.

WHY THIS FILE EXISTS
---------------------------------------------------------------------------
`ufunc_cases.py`'s `build_cases_for_ufunc` -- the SHARED, generic per-ufunc
case builder every `kind="ufunc"` item in `ufunc_registry.py` uses,
including these six -- only ever builds ONE `out=` case per applicable
ufunc: `out=np.empty_like(...)`, i.e. always the happy-path "buffer matches
the real result's dtype and shape" form. It never tries a WRONG dtype, a
WRONG shape, an ABSENT `out=`, or (for the two multi-output-capable-via-
tuple ufuncs among these six, `divmod`/`frexp`/`modf`) a PARTIAL tuple like
`out=(buf, None)`.

That gap is exactly how two real engine defects survived undetected in the
six newly-implemented ufuncs above: anionpy's `out=`-related `UFuncTypeError`
dropped numpy's per-output numeric index, and anionpy's `out=`-related
broadcast `ValueError` omitted `out=` buffers from the reported operand
shape list entirely (only showing "computed result shape" vs "out shape",
2 entries, instead of every input followed by every non-`None` out buffer).
A dedicated out=-crossing grid (not wired into `run.py`/`tools/coverage.py`
at all, so it never fed the ledger) found 34/114 mismatches across these
five call-forms before `ionp-py/src/lib.rs`'s `check_full_broadcast` /
`ufunc_output_casting_err`'s new `output_index` parameter fixed both. This
file is the corpus that makes that regression class impossible to miss
again: it is the SAME axis, re-expressed as first-class, ledger-visible
`kind="custom"` registry items (so it appears as real keys in a real
`run.py --out` JSON and in `tools/coverage.py --tests`, not just in a
standalone script under /tmp), following the exact precedent
`ufunc_dtype_cases.py`'s own module docstring set for this same kind of
gap ("a check graded at a bar the real requirement never claimed").

WHY THIS IS A SEPARATE FILE/NAMESPACE, NOT MORE ENTRIES ON THE REAL
"divmod"/"frexp"/... ITEMS IN ufunc_registry.py
---------------------------------------------------------------------------
Same registry-key-collision reason `ufunc_dtype_cases.py`/
`ufunc_order_cases.py` already documented for their own namespaces:
"divmod"/"frexp"/... are already declared by `ufunc_registry.py`, and the
tail-merge pattern all three of these files use (`registry.REGISTRY.update`
at import time) refuses to redeclare an existing name. This file follows
that exact precedent one level further: fresh `"out_axis/ufunc/<name>"`
keys, `kind="custom"` ItemSpecs, merged into `registry.REGISTRY` the same
way, reusing the base spec's `atol`/`rtol`/`ulp_tolerance`/
`ulp_justification`/`ulp_sweep`/`exception_equivalences`/`multi_output`.

SCOPE
---------------------------------------------------------------------------
Deliberately restricted to exactly the six ufuncs named in the task brief
(`divmod`, `float_power`, `frexp`, `isnat`, `ldexp`, `modf`) -- not every
currently-"exact" ufunc the way `ufunc_dtype_cases.py`/`ufunc_order_cases.py`
scope themselves. This is the axis that was missing specifically for these
six, and generalizing it to all 134 ufuncs is explicitly out of scope for
this task (a broader out=-axis sweep is a separate, larger undertaking with
its own operand-domain-per-ufunc design work this task was not asked to
do).

FORMS, PER THE TASK BRIEF, VERBATIM
---------------------------------------------------------------------------
Multi-output-via-tuple ufuncs (`divmod`, `frexp`, `modf`, each `nout == 2`):
ABSENT, full matching tuple, partial tuple (`(buf, None)` -- both slot
orderings), wrong-dtype member (both slot positions), wrong-shape member
(both slot positions).

Single-output ufuncs (`float_power`, `ldexp`, `isnat`, each `nout == 1`):
ABSENT, full matching tuple/bare buffer (`match`), wrong-dtype, wrong-shape.

Operand values mirror the coordinator's own out=-crossing probe (the one
that found the original 34/114) for `divmod`/`float_power`/`ldexp`/`frexp`/
`modf`, crossed over every real, distinct dtype loop each ufunc actually
has (float dtypes for all six; also int/uint dtypes for `divmod` and
`float_power`, which have real integer loops -- `frexp`/`modf`/`ldexp` are
float-only in real numpy, confirmed live: `np.frexp(np.array([1],
dtype='int32'))` raises `TypeError: ufunc 'frexp' not supported for the
input types`). `isnat` has no such dtype axis (its ONLY valid input kind is
`datetime64`/`timedelta64`), so it is instead crossed over a small set of
representative datetime64 UNITS (`D`, `ns`, `Y`) -- the analogous "does the
result out= interact correctly across the ufunc's real domain" coverage,
just keyed on time unit rather than numeric dtype since that is `isnat`'s
actual degree of freedom.

RIGOR: every case here is a `kind="custom"` ItemSpec, routed through the
SAME `harness.run_case`/`evaluate` machinery every other registry item
uses -- not a hand-rolled comparison. That machinery already compares, per
case: exception CLASS (`type(exc)`, checked against `exception_equivalences`,
not `isinstance`), exception MESSAGE (`str(exc)`, byte-exact via `repr()`
diffing on mismatch -- catches even a lone trailing-space difference), and
(for the `ABSENT`/`match` forms, which do not raise) result dtype/shape/
values via `compare_multi_output` (multi-output ufuncs) or `compare_values`
(single-output), reusing the base spec's declared tolerances -- these six
ufuncs are declared bit-exact (`atol=rtol=0.0`) so no slack is available to
hide a wrong value even on the non-exception forms.

NUMPY IS ORACLE ONLY. Every reference value/exception in this file's cases
comes from a real, live call to `np.<name>(..., out=...)` inside
`harness.run_case` -- this module never calls numpy to compute anionpy's
answer or borrows a numpy-derived string for anionpy's expected exception
text; it only supplies `(label, args, kwargs)` tuples for the harness to
run both sides itself.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
import registry
from registry import ItemSpec

DT_F = ("float16", "float32", "float64")
DT_I = ("int8", "int32", "int64", "uint8", "uint64")

# (nout, dtypes-to-cross) per ufunc.
_SPEC_BY_NAME = {
    "divmod": (2, DT_F + DT_I),
    "float_power": (1, DT_F + DT_I),
    "frexp": (2, DT_F),
    "ldexp": (1, DT_F),
    "modf": (2, DT_F),
    # isnat handled separately -- see _cases_for_isnat.
}


def _operands(name: str, dt: str) -> tuple:
    """Mirrors the operand construction the original coordinator probe (and
    this task's `/tmp/ufout.py` verification grid) used for these five
    ufuncs -- same values, same per-dtype sign handling for unsigned
    dtypes, kept identical here so this corpus is measuring the exact same
    shapes/values already proven, live, to round-trip correctly post-fix."""
    uns = dt.startswith("uint")
    if name == "divmod":
        x = [7, 3, 11, 5, 0, 9] if uns else [7, -7, 3, -3, 0, 11]
        y = [2, 2, 3, 4, 3, 4] if uns else [2, 2, -2, -2, 3, 4]
        return (np.array(x, dtype=dt), np.array(y, dtype=dt))
    if name == "float_power":
        x = [2, 3, 0, 1, 4, 5] if uns else [2, -2, 0, 3, 1, 4]
        y = [3, 2, 0, 5, 2, 1] if uns else [3, 2, 0, -1, 5, 2]
        return (np.array(x, dtype=dt), np.array(y, dtype=dt))
    if name == "ldexp":
        return (
            np.array([1.5, -1.5, 0.0, 3.25, 2.0, 7.0], dtype=dt),
            np.array([1, -2, 3, 0, 4, -1], dtype="int32"),
        )
    # frexp/modf: unary, float-only.
    v = [1.5, -1.5, 0.0, -0.0, 3.25, float("inf")]
    return (np.array(v, dtype=dt),)


def _out_dtype_for(name: str, ops: tuple) -> str | tuple[str, ...]:
    """The real result dtype(s), asked of live numpy -- never hand-guessed
    -- so "match"/"wrongdtype" out= buffers are built against ground truth
    (e.g. divmod's output dtype is the SAME as its integer input dtype, but
    frexp's second output is always int32 regardless of its float input
    dtype -- these differ per ufunc and must be measured, not assumed)."""
    res = getattr(np, name)(*ops)
    if isinstance(res, tuple):
        return tuple(str(np.asarray(r).dtype) for r in res)
    return str(np.asarray(res).dtype)


def _wrong_dtype(dt: str) -> str:
    return "int32" if not dt.startswith("int") else "float64"


def _mk(dt: str, shape: tuple) -> np.ndarray:
    return np.zeros(shape, dtype=dt)


def _single_out_forms(shape: tuple, out_dt: str) -> list[tuple[str, object]]:
    wrong_dt = _wrong_dtype(out_dt)
    return [
        ("ABSENT", None),
        ("match", _mk(out_dt, shape)),
        ("wrongdtype", _mk(wrong_dt, shape)),
        ("wrongshape", _mk(out_dt, shape + (2,))),
    ]


def _multi_out_forms(shape: tuple, out_dts: tuple[str, str]) -> list[tuple[str, object]]:
    d0, d1 = out_dts
    w0, w1 = _wrong_dtype(d0), _wrong_dtype(d1)
    return [
        ("ABSENT", None),
        ("match", (_mk(d0, shape), _mk(d1, shape))),
        ("partial_first_none", (None, _mk(d1, shape))),
        ("partial_second_none", (_mk(d0, shape), None)),
        ("wrongdtype_first", (_mk(w0, shape), _mk(d1, shape))),
        ("wrongdtype_second", (_mk(d0, shape), _mk(w1, shape))),
        ("wrongshape_first", (_mk(d0, shape + (2,)), _mk(d1, shape))),
        ("wrongshape_second", (_mk(d0, shape), _mk(d1, shape + (2,)))),
    ]


def _cases_for_ufunc(name: str) -> list[tuple]:
    nout, dtypes = _SPEC_BY_NAME[name]
    cases: list[tuple] = []
    for dt in dtypes:
        ops = _operands(name, dt)
        shape = ops[0].shape
        out_dt = _out_dtype_for(name, ops)
        if nout == 1:
            forms = _single_out_forms(shape, out_dt)
        else:
            forms = _multi_out_forms(shape, out_dt)
        for label, out_val in forms:
            kwargs = {} if out_val is None else {"out": out_val}
            cases.append((f"{name}/out_axis/{dt}/{label}", tuple(ops), kwargs))
    return cases


# isnat: no numeric dtype axis (its only valid domain is datetime64/
# timedelta64), so crossed over representative time UNITS instead. Single
# output, always bool.
_ISNAT_UNITS = ("D", "ns", "Y")


def _cases_for_isnat() -> list[tuple]:
    cases: list[tuple] = []
    for unit in _ISNAT_UNITS:
        vals = np.array(
            ["2020-01-01", "NaT", "2020-01-03", "1970-01-01"],
            dtype=f"datetime64[{unit}]",
        )
        shape = vals.shape
        for label, out_val in _single_out_forms(shape, "bool"):
            kwargs = {} if out_val is None else {"out": out_val}
            cases.append((f"isnat/out_axis/{unit}/{label}", (vals,), kwargs))
    return cases


_NAME_TO_CASE_BUILDER = {
    "divmod": lambda: _cases_for_ufunc("divmod"),
    "float_power": lambda: _cases_for_ufunc("float_power"),
    "frexp": lambda: _cases_for_ufunc("frexp"),
    "ldexp": lambda: _cases_for_ufunc("ldexp"),
    "modf": lambda: _cases_for_ufunc("modf"),
    "isnat": _cases_for_isnat,
}


def _build_out_axis_specs() -> dict[str, ItemSpec]:
    import ufunc_registry  # local import: avoid a module-load-order cycle

    specs: dict[str, ItemSpec] = {}
    for name, builder in _NAME_TO_CASE_BUILDER.items():
        base_spec = ufunc_registry.UFUNC_SPECS.get(name)
        if base_spec is None:
            continue
        key = f"out_axis/ufunc/{name}"
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
            exception_equivalences=base_spec.exception_equivalences,
            multi_output=base_spec.multi_output,
            custom_cases=builder,
        )
    return specs


OUT_AXIS_UFUNC_SPECS: dict[str, ItemSpec] = _build_out_axis_specs()

registry.REGISTRY.update(OUT_AXIS_UFUNC_SPECS)
