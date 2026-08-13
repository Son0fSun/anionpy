# dtype-metadata / printing / introspection cluster — measured (2026-08-02)

Read-only pre-verification of 36 items, against binary mtime **22:25:53**
(stable across the run). Nothing here is a declaration.

The headline: **only 5 of 36 resolve on `ionp` at all.** This cluster is
almost entirely greenfield, not broken-and-needing-repair. Confirmed by
attribute probe AND by grep across `ionp/` and `ionp-py/src/` — zero
definitions, zero registrations.

## The 5 that resolve

`can_cast`, `result_type`, `promote_types`, `min_scalar_type`, `isdtype`.

Measured: `can_cast` clean across a 14x14 dtype-pair x 5 casting-rule grid
(980 cases, 0 mismatches). `min_scalar_type` clean on an 8-value boundary
sweep. `promote_types(None,'int8')` and `result_type(None,int8)` match.

**`isdtype` is RED, and more wrong than its own source comment says.** When
`kind` is a scalar-type *class* (`np.integer`, `np.floating`, `np.int32`)
rather than a string or tuple, real numpy returns a genuine boolean —
`np.isdtype(int32, np.integer)` is `False`, `np.isdtype(int32, np.int32)` is
`True`. ionp raises `TypeError` for all such forms. The comment in
`dtypeinfo.rs` frames this as "falls through to a not-understood TypeError",
which undersells it: numpy's path here is a silent, correct answer, not an
error. That makes it a **correctness** gap, not a coverage gap.

None of the 5 is GREEN outright — each carries a disclosed architectural gap
(void/structured dtype spec, or object dtype for `min_scalar_type(2**65)`
where numpy returns `dtype('object')` and ionp raises `OverflowError`).

**Runtime-numpy check, and it is good news:** within this cluster's own
computation — the 5 functions plus the shared `coerce_dtype_like` /
`dtype_from_pyobj` helpers — there are **zero** runtime numpy calls.
`dtype_from_pyobj` now ends in ionp's own `TypeError` rather than a numpy
fallback, and `can_cast` was explicitly de-numpy'd. Two numpy imports remain
*adjacent* (`PyDType`'s `.dtype` interop getter and `__eq__` fallback at
`ionp-py/src/lib.rs` ~1463/~1480) but exist only so numpy can consume an ionp
dtype from its side; they feed none of the 5 answers.

## BLOCKED — not merely unwritten

- `issubdtype`, `common_type`, and the class-`kind` form of `isdtype` need the
  **scalar-type-class hierarchy** (`np.integer`/`np.floating` as first-class
  objects with a real subclass relation). Nothing in the tree has it.
- `typecodes`, `sctypeDict`, `ScalarType` need codes for datetime/void/object
  dtypes that `ionp_core::DType` structurally lacks (14 numeric variants,
  full stop). A faithful table cannot be built; the only options are shipping
  a knowingly-partial table or diverging from numpy's literal string. **This
  is a question for Mother, not a thing to decide in a build brief.**

## The cheap ones — genuinely trivial, no dependencies

`get_include` (return a path), `base_repr` and `binary_repr` (pure
integer→string, no dtype involvement), `getbufsize`/`setbufsize`,
`isfortran` (reads flags/strides), `iterable` (`try: iter(x)`).

`ndim`, `shape`, `size`, `isscalar` are thin wrappers but NOT free: numpy's
`isscalar` is idiosyncratic (Python scalars and numpy scalars count, 0-d
arrays do **not**), and the free-function `ndim`/`shape`/`size` must accept
lists and scalars, not just arrays. These four plus `iterable`/`isfortran`
are already inside the running comparison/predicate builder's scope.

## The one real project in here

`array2string` is the core formatter that `array_repr`, `array_str`,
`format_float_positional`, `format_float_scientific`, `printoptions`,
`get_printoptions` and `set_printoptions` all sit on top of — **8 items behind
one build.** It is also genuinely hard: byte-exact output demands numpy's
summarisation/threshold/edgeitems/precision logic and a shortest-round-trip
float formatter (Dragon4-class) that gets NaN, ±inf, -0.0 and subnormals
right, with no recourse to numpy's own formatter. That is one dedicated agent
on one well-scoped deliverable, and it should not be bundled with anything.

`printoptions`/`set_printoptions` are cheap plumbing but **pointless before
`array2string` exists** — there is nothing for the options to affect.

## `seterr`/`errstate` — a scoping trap worth naming

The state plumbing is trivial. Whether the flags actually *do* anything
depends on ionp's ufuncs having a hook to consult overflow/divide/invalid/
underflow state at compute time, and **nobody has verified such a hook
exists**. A `seterr` that doesn't affect computation is a much smaller and
differently-shaped task than a real one. Establish which it is before
scoping, or the item ships as a convincing no-op.

Related open defect already on file: float16/float32 don't raise numpy's
overflow `RuntimeWarning`. Same machinery.

## Not independently reproduced

The exotic-input divergences for `can_cast`/`promote_types`/`result_type`
(`[]`, `{}`, `()`, `b'x'`, bare-ndarray-as-dtype-spec) are **ledger-sourced,
read from source comments, not re-measured**. They are internally consistent
with what was measured, and that is all that can be said for them.

Also unprobed: whether any ufunc error-state hook exists at all;
`show_config`/`show_runtime` correctness (build-dependent, so "correct" needs
defining before it can be tested).
