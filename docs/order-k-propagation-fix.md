# order='K' output-layout propagation fix (2026-08-02)

Follow-up to `docs/stride-gap-classification.md` (commit c65e573 revoked 82
declarations for output-layout gaps). This fixes the largest mechanism from
that classification: elementwise/ufunc/binop/emath/clip output layout, plus
a genuinely separate bug in `ndarray.copy()`'s default found while probing
that mechanism. The reduction-output-layout mechanism (~24 items) was
explicitly out of scope and is NOT fixed here — see "Not fixed" below.

## Empirically-derived rule

Derived from live probes against numpy 2.5.1 (not trusted from the
classification doc's paraphrase alone):

**Out-of-place elementwise ops (dunders, `emath.*`, `clip`)**: numpy's
default when the caller can't spell `order=` at all (there is no way to
pass `order=` through `a + b`) is `order='K'`. K-order means: sort output
axes by decreasing absolute input stride, allocate contiguously in that
permuted order, then un-permute the strides back to the original axis
order. Negative input strides never produce negative output strides. With
multiple array operands (binops, `clip` with array min/max), numpy's real
multi-operand algorithm (`PyArray_CreateMultiSortedStridePerm`) is a stable
insertion sort over axis PAIRS starting from C order, with operands
right-aligned for broadcasting; size-1 axes never vote; disagreement
between operands' orderings falls back to C/identity order. ionp's existing
`multi_sorted_stride_perm`/`relayout_by_perm`/`apply_ufunc_order` machinery
already implemented this correctly for `Ufunc.__call__` — it just wasn't
wired into the dunders, in-place ops, `clip`, or `emath.*`.

**In-place ops** (`+=`, `//=`, `%=`, `<<=`, `>>=`, etc.): numpy NEVER
reallocates. The result always lands in the target's own existing buffer at
its own existing strides, unchanged, regardless of the other operand's
layout. Confirmed live: `v = b.T; v //= 1` leaves `v.strides` bit-identical
to before. Implemented by K-order-voting with only `&[&slf.inner]` (self
alone) as the operand list — `slf.inner`'s own strides sort against
themselves, so relaying the freshly computed value buffer out in that same
traversal order reconstructs `slf.inner`'s exact original strides.

**Hazard**: `cast_to` (core `array.rs`) always forces C-contiguous whenever
it performs a real copy (dtype change, or non-C-contiguous input). This
silently undoes any K-order relayout applied before it. Fix pattern for
in-place ops: always call `cast_to` FIRST, then `apply_ufunc_order` LAST.
`numpy.emath.*`'s internal complex-promotion cast has the identical hazard
(real numpy's own `astype(complex)` defaults to `order='K'` internally and
so preserves layout through promotion) — worked around by voting K-order
using the pre-promotion ORIGINAL operand(s)' shape/strides, not the
post-cast array's.

**`ndarray.clip`**: does full multi-operand K-order voting across `self`
plus whichever of `min`/`max` are array-valued (not bare Python scalars) —
confirmed empirically that e.g. `a.T.clip(b.T, None)` takes `b.T`'s layout
into account too, not just `self`'s.

**`ndarray.copy()` — a separate, unrelated bug found during this work**:
numpy's `ndarray.copy` METHOD defaults to `order='C'`. This is NOT the same
as the free function `np.copy(a, order=...)`, which defaults to
`order='K'`. This is a documented numpy footgun (numpy's own `ndarray.copy`
docstring calls it out). Verified live: `b = np.arange(24).reshape(4,6).T`
is F-contiguous with strides `(8, 48)`; `b.copy()` (no order arg at all) ->
C-contiguous `(32, 8)`; `np.copy(b)` -> stays `(8, 48)`. ionp's `.copy()`
wrongly defaulted to `'K'` like the free function. `__copy__`/`__deepcopy__`
(the `copy.copy()`/`copy.deepcopy()` protocol hooks) were already correct
with `'K'` from a prior session — those are a different call path from the
`.copy()` method and numpy does default THEM to K-preserving behavior.

## Verification method

All out-of-corpus, file-based probe scripts (never `python -c`, always
asserting `ionp.__file__` points at this repo, never calling into real
numpy from ionp source):
- `probe_dunders.py` — 26 dunder ops x 5 layouts (C/F/T/strided/negstride).
- `probe_inplace.py` — 4 in-place ops x 4 target/other layout combos.
- `probe_emath.py` — 7 `emath.*` functions x 5 layouts, value ranges chosen
  to trigger complex promotion where relevant.
- `probe_clip.py` — `clip` x 4 base layouts x scalar/array-min/array-max
  combinations (36 cases).
- `probe_copy2.py` — `ndarray.copy` x 5 layouts x 5 `order` values (25
  combinations).

Each probe checks BOTH `.strides` and `.tobytes()` against real numpy, on
inputs never used in the differential test corpus itself. All passed
("ALL OK") before any `_state` re-declaration was made.

## Ledger (measured, not inferred)

| Stage | exact | failing | phantom | untested |
|---|---|---|---|---|
| Baseline | 377 | 5 | 0 | 0 |
| After commit 1 (K-order propagation, 41 items) | 418 | 5 | 0 | 0 |
| After commit 2 (`ndarray.copy` default, 1 item) | 419 | 5 | 0 | 0 |

`failing` count unchanged from baseline in both steps (verified the 5
failing items — `add`, `copy`, `ndarray.flatten`, `searchsorted`, `zeros` —
are pre-existing free-function/other-mechanism failures, not new
regressions from this work; `ndarray.copy`/`ndarray.__add__`/etc. are
distinct ledger keys from the failing `copy`/`add` free functions).

## Items re-declared "exact" (42 total, all probe-backed)

`ionp/_state/ndarray.py` (35): `copy`, `__add__`, `__radd__`, `__mul__`,
`__rmul__`, `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`,
`__sub__`, `__rsub__`, `__and__`, `__rand__`, `__or__`, `__ror__`,
`__xor__`, `__rxor__`, `__neg__`, `__abs__`, `__invert__`, `__mod__`,
`__rmod__`, `__lshift__`, `__rlshift__`, `__rshift__`, `__rrshift__`,
`__ilshift__`, `__irshift__`, `__imod__`, `__rfloordiv__`, `clip`,
`__floordiv__`, `__ifloordiv__`.

`ionp/_state/misc_namespaces.py` (7): `emath.arccos`, `emath.arcsin`,
`emath.arctanh`, `emath.log`, `emath.log2`, `emath.log10`, `emath.sqrt`.

(`__rand__`/`__ror__`/`__rxor__` needed no direct Rust code change — they
already delegate to `__and__`/`__or__`/`__xor__` and inherited the fix.)

## Bonus, not required / not re-declared

`emath.logn` and `emath.power` also got the same `apply_k_order` fix in
`ionp-core/src/emath.rs` since they share the helper, but neither was
previously an `"exact"` declaration to revoke/re-declare, so no `_state`
change was made or needed for them.

## Not fixed: reduction-output-layout mechanism (~24 items)

Explicitly out of scope per task instructions — not attempted. From
reading `stride-gap-classification.md` and observing the ufunc/binop fix
process, the reduction case is a genuinely different rule with its own
edge cases (numpy's reduction `order=` handling depends on `keepdims`,
`axis`, and whether the reduction removes all/some/no axes — it is not a
simple "vote K-order across operands" problem since a reduction's *output*
has fewer dimensions than any operand, so there's no direct axis-to-axis
stride correspondence to sort by; numpy instead derives the reduced
output's contiguity from the *retained* axes of the input in their
original relative order). This would need its own empirical derivation
pass and probe suite before implementation; deferred entirely as
instructed, not attempted or partially implemented.
