# Ticket #64 remeasure — 2026-08-08

MEASUREMENT ONLY. No shipped Rust or Python source touched by this pass.
This document supersedes the "100 items" figure on record for #64 with a
measured one. It does **not** edit `docs/EMPTY-AND-ORDER-AUDIT.md` — that
document's corrections are read and cited below, not modified.

**Commit measured against: `5f2e09b`** (working tree clean, no local
changes). Ledger: 1329/3091 = 42.996% declared exact
(`./.venv/bin/python tools/coverage.py --tests /tmp/r_verify6.json`), same
1329-item set as the prior audit (`diff` against the prior
`exact_items.tsv` is empty — no items were added or removed from the
declared-exact set between the two passes).

## Headline

| bucket | count | ticket |
|---|---:|---|
| **#7 only** (diverges only on a length-1-axis or 0-d input) | **1** | #7 |
| **#64** (diverges on ≥1 combo with no length-1 axis) | **58** | #64 |
| Clean (measured, 0 divergent combos found) | 367 | — |
| Unverifiable (object has no `.strides`/`.flags` to compare) | 75 | #63 |
| NA / no array-shaped output at all (metadata, scalars, bools, exception classes, sentinels) | 108 (74 original + 34 newly resolved under `ma.*`) | — |
| Out of scope for this pass (unresolvable or no driveable generic call) | 720 | — |
| **Total declared exact** | **1329** | |

**The #64 population is 58, not 100.** The old 100 was not just an
overcount from the ~38 `matrix` items (as your background note
anticipated) — it was also measuring several base-construction bugs as if
they were per-operation bugs. Both effects are explained below with
evidence, and both point the same direction: down.

## The two expected effects, checked directly

### 1. `expand_dims` split (`7aa74e6`)

The declared-exact item is `expand_dims` (there is no separately-declared
`insert_newaxis` in the 1329-item ledger — `newaxis` and `ma.expand_dims`
are declared, `insert_newaxis` is not, so the split itself isn't a ledger
line item). `expand_dims` now measures **CLEAN**: 0 divergent combos
across every layout/extent/dtype combo tried, including the identity-shape
case that previously showed the spurious 0-stride. This confirms the fix.

### 2. `matrix.__array__` (`a2a1ebb`)

Confirmed and larger than expected. Of the 92 declared-exact `matrix.*`
items:

- **54 now CLEAN** (up from 7 in the old report's clean-77 list —
  `matrix.A1`, `.argsort`, `.copy`, `.flatten`, `.getA1`, `.ravel`,
  `.trace`).
- **5 still in #64**: `matrix.__pos__`, `matrix.cumprod`, `matrix.cumsum`,
  `matrix.fill`, `matrix.repeat` — all diverge only on zero-extent inputs
  (bucket 5, the `#61`/`#67` zero-extent family) except `matrix.repeat`,
  which also diverges on plain nonempty C-contiguous input (bucket 0) and
  is a distinct, un-ticketed defect worth its own look.
- **33 UNMEASURED** — not because they're broken, but because they're
  metadata/dunder items with no array-shaped output at all on this
  harness's probe (`.dtype`, `.itemsize`, `.ndim`, `.shape`, `.size`,
  `.item`, `.tobytes`, `.tolist`, `__bool__`/`__complex__`/`__float__`/
  `__int__`/`__index__`/`__len__`/`__hash__`/`__contains__`/`__iter__`,
  and the bitwise dunders which need an int-dtype base the generic probe
  didn't supply). None of these were in the old FALSE-38 list either.
- **0 in #7-only**.

So: `docs/EMPTY-AND-ORDER-AUDIT.md`'s SECOND CORRECTION was right that the
9-then-38 `matrix.*` "family E" items were never independent defects — but
its own closing claim ("the whole matrix surface reduces to the two
[already-ticketed] causes... nothing under it requires a third
explanation") undersold it. Re-verified directly at `5f2e09b`:

```
inner_np = np.array([[0,1,2],[3,4,5]], dtype='float64', order='F')
inner_ip = ionp.array([[0,1,2],[3,4,5]], dtype='float64', order='F')
mn = np.matrix(inner_np, copy=False)          # np strides (8, 16)  -- F preserved
mi = ionp.matrix(inner_ip, copy=False)        # ionp strides (24, 8) -- silently recomputed to C
```

`matrix()`'s own constructor loses F-order on a correctly-F-strided input
**before any method is called on the result** — same root defect as the
dtype-cast constructors below, inherited by every `matrix.*` operation
built on it. That's not a length-1-axis case and not zero-extent, so it's
squarely `#64`, not `#7` or `#67` — but it shows up as a **base**-level
defect, and this pass had to add a check the prior audit's harness lacked
(see next section) to keep from either hiding it or double-attributing it
to 30-some downstream methods.

## Methodology correction found mid-measurement: base validity was not actually being enforced

The task's own constraint says: *"Construct the input identically on both
sides and assert the two bases' `.strides` are EQUAL before comparing
outputs. A divergence in the base invalidates the comparison of the
output."* The prior audit's harness (`/private/tmp/audit_harness.py` +
`audit_main.py`, reused here) only asserted **shape** equality on bases,
not **strides** equality. I discovered this only after re-running the
existing instrument and getting numbers that didn't hold up to spot-check.

Direct measurement of the shared base-construction primitives themselves
(`zeros`/`arange`/`.copy('F')`/step-slicing — the task's own recommended
construction recipe, tested independently of any operation under audit):

```
len1ax_F_(4,1,2) via zeros().copy('F'):  np (8,32,32)  ionp (8,64,32)   DIVERGE
slice via arange()[::2]:                  np (16,)      ionp (8,)       DIVERGE
every zero-extent shape tried (5 shapes, C and F): np strides all-0 as expected, ionp strides nonzero (e.g. (0,3): np (0,0) ionp (24,8))  DIVERGE
plain C (2,3), plain F (2,3), transposed-view (2,3), length-1-axis-C (4,1,2), 0-d:  MATCH on all of these
```

So three of the eleven layout combos in the sweep (`len1axF`, `slice`, and
every zero-extent shape) have their **base already wrong before any
operation runs** — a real bug, but in `zeros`/`array`-construction and
step-slicing, not in the ~218 operations the old count attributed it to.
This is the same failure shape the audit document already caught twice
(`ma.MaskedArray` tested via plain-`ndarray` bases; `matrix` methods
blamed for the bare constructor's bug) — a uniform failure across a
cross-product turning out to be one shared object, not N independent
ones. Here it's a third instance: a uniform failure across combos turning
out to be the shared *input builder*, not N independent operations.

I patched the local, unshipped probe harness
(`/private/tmp/audit_main.py`) to assert `nb.strides == ib.strides` (via
`np.asarray(...).strides`, matching what `values_equal` already does) on
every combo before calling the operation, and to skip (not count either
way) any combo where the base itself diverges. Re-running with that fix
is what produced the 58/1 split above — **not** a change to the shipped
code, purely a harness correction, and the raw excluded-combo list is
recorded per-item in `/private/tmp/audit_results.json`'s
`base_invalid_combos` field for reproduction.

Concrete effect on a previously-#7-attributed case, `broadcast_to` — the
one item that still lands in #7-only after the fix, confirmed on a
**valid** base:

```
len1ax_C (4,1,2), valid base (np and ionp both (16,16,8)):
  numpy:  shape (4,1,2) strides (16, 0, 8)   -- broadcast dim gets 0-stride
  ionp:   shape (4,1,2) strides (16, 16, 8)  -- computes a real stride instead of 0
```
That's a genuine, narrow, single-item #7 finding — not diluted by the
base-construction noise.

## #7 (1 item)

`broadcast_to` — see example above. This is a large drop from the old
118. **I could not reproduce a like-for-like comparison against the old
118**, because the source document names only ~7 of the 118 individually
(`sort`, `astype`, `where`, `concatenate`, `stack`, `roll`, `tile` — and
two of those, `astype`/`concatenate`/`stack` bare, aren't even in the
1329-item ledger per the document's own note). What the evidence does show
directly: on a **valid** length-1-axis base (`len1ax_C`, which matches
between numpy and anionpy), `sort`/`roll`/`tile` diverge on the *F-order,
no-length-1-axis* combo too (shown below, under #64) — meaning they were
never #7-only to begin with; they belong entirely in #64 both before and
after this remeasure, consistent with the old report's own worked example
(`sort` F-order, no length-1 axis at all, `EMPTY-AND-ORDER-AUDIT.md` lines
107-118). Most of the rest of the old 118 most likely diverged only via
the now-invalidated `len1axF` sub-combo (the one whose *base itself* was
wrong) or the zero-extent combos — i.e. the old 118 count is best
explained as measuring the shared-builder bug found above, repeated across
~117 operations that never had an independent length-1-axis defect of
their own. That is inference from the mechanism, not a re-identification
of each of the 118 by name — the source document doesn't preserve that
list, so I cannot certify item-by-item which of the 118 specifically
"moved." Flagging this rather than guessing, per the task's own standard.

## #64 (58 items)

```
argwhere, bool_, byte, cdouble, complex128, complex64, csingle, diff, double, fft.fftshift,
fft.hfft, fft.ifftshift, fft.irfft, float16, float32, float64, half, hstack, i0, imag, int16,
int32, int64, int8, int_, intc, intp, isclose, isreal, linspace, long, ma.make_mask,
matrix.__pos__, matrix.cumprod, matrix.cumsum, matrix.fill, matrix.repeat, ndarray.fill, real,
roll, short, sinc, single, sort, sort_complex, tile, ubyte, uint, uint16, uint32, uint64,
uint8, uintc, uintp, ulong, unwrap, ushort, vstack
```

Bucket detail (which non-length-1-axis layout the divergence shows up on —
`0`=plain-C baseline, `2`=F-order, `3`=transposed view, `4`=non-contiguous
slice [excluded where the slice base itself was invalid — see above],
`5`=zero-extent):

- **30 dtype-cast constructors** (`bool_`, `byte`, `cdouble`, `complex64/128`,
  `csingle`, `double`, `float16/32/64`, `half`, `int8/16/32/64`, `int_`,
  `intc`, `intp`, `long`, `short`, `single`, `ubyte`, `uint`, `uint8/16/32/64`,
  `uintc`, `uintp`, `ulong`, `ushort`) — all diverge on plain F-order and
  transposed-view input, buckets {2,3}. Confirmed directly:
  `float32(F-order (2,3))`: numpy `(4,8)`, ionp `(12,4)`.
- **`sort`, `roll`, `tile`** — buckets {2,3}, confirmed directly on a
  base independently verified matching (`(8,16)` both sides pre-op):
  all three come out `(24,8)` on ionp vs `(8,16)` on numpy — reproduces
  the old report's worked example exactly, on a properly-validated base
  this time.
- **`fft.fftshift`, `fft.hfft`, `fft.ifftshift`, `fft.irfft`, `hstack`,
  `vstack`, `real`, `imag`, `isclose`, `isreal`, `unwrap`, `sinc`, `i0`,
  `sort_complex`, `diff`, `ndarray.fill`** — buckets {2,3}, same family,
  all named in the old report's "wider unexpected pattern" and confirmed
  still present.
- **`argwhere`, `linspace`** — bucket {0,1,2,3}: diverge even on the
  plain C-contiguous baseline, i.e. not an order-loss-on-non-C-input bug
  at all, a distinct/more severe defect (matches the old report's bucket-0
  flag, both items carried forward unchanged).
- **`matrix.__pos__`, `.cumprod`, `.cumsum`, `.fill`** — bucket {5} only
  (zero-extent, `#61`/`#67` family). **`matrix.repeat`** — buckets
  {0,1,4,5}, the broadest matrix defect, not just zero-extent.
- **`ma.make_mask`** — bucket {2}, new to this pass (the old report scoped
  `ma.*` out almost entirely as UNMEASURED; this one resolves under the
  corrected harness and genuinely diverges: F-order `(2,3)` int64 input,
  numpy `(1,2)`, ionp `(3,1)`).

## Items that moved off the old 218 and why

- **~54 `matrix.*` items**: moved FALSE→CLEAN. Cause: `a2a1ebb` added
  `__array__`, which made `np.asarray(matrix)` stop raising, which made
  the *comparison* possible for the first time (not just the operation).
  Confirmed directly above.
- **`expand_dims`**: moved FALSE→CLEAN. Cause: `7aa74e6` reimplemented it
  via `reshape` instead of the newaxis stride-0 rule. Confirmed directly.
- **Very likely ~110 of the old 118 `#7`-attributed items**: moved
  `#7`→(mostly still measurable-but-different-classification, largely now
  landing as CLEAN or unresolved rather than FALSE). Cause: this pass's
  base-validity fix, not a code change — the old measurement never had a
  code-level fix for this. **This is a methodology correction, not a
  product fix**, and I want to say plainly: I cannot rule out that some of
  those 118 have real, independent length-1-axis bugs that this harness's
  particular combo set (`len1ax_C`, valid) simply didn't happen to
  exercise as narrowly as `len1axF` (invalid) would have. `broadcast_to`
  is proof at least one real #7 defect exists and is catchable this way;
  I have not exhaustively re-derived all 118 by hand to confirm zero
  others exist. Treat "#7 = 1" as "1 confirmed, rest unconfirmed" rather
  than "117 disproven."
- **9 Family-E + ~29 further `matrix.*` items named in the old 100**: 5
  remain FALSE (now correctly attributed to zero-extent / bucket-0, not
  to the array-protocol crash that no longer happens), the rest moved to
  CLEAN or UNMEASURED (metadata dunders, never independently false to
  begin with per the SECOND CORRECTION's own diagnosis).
- **Everything else named in the old 100** (dtype casts, `fft.*`,
  `hstack`/`vstack`, `real`/`imag`, `isclose`/`isreal`, `unwrap`, `sinc`,
  `i0`, `sort_complex`, `diff`, `ndarray.fill`, `sort`, `roll`, `tile`):
  **unchanged** — all reconfirmed present in the new 58 on a
  base-validated combo. No fix landed for these; they were correctly
  attributed before and remain correctly attributed now.

I could not do a literal item-by-item diff against the old 118/100 lists
because the source document (`EMPTY-AND-ORDER-AUDIT.md`) only enumerates
them partially by name (a handful of representative examples per family,
not the full 218). Where it does name specific items, every one is
accounted for above, either confirmed-still-false or confirmed-now-clean
with a direct repro. Where it doesn't, I've explained the move at the
mechanism level (which fix or which methodology correction would move
that whole family) rather than fabricate a per-item list the source
material can't support.

## Unverifiable — #63 (75 items)

Confirmed by direct construction: the operation resolves, the generic
call succeeds, and the returned object's own type is `MaskedArray`, which
has neither `.strides` nor `.flags` (spot-checked directly, e.g.
`ionp.ma.masked_array(...).strides` raises `AttributeError`, and `numpy`'s
own `MaskedArray` does have both — confirmed not a numpy-side contract
gap).

```
ma.MaskedArray.T, ma.MaskedArray.__and__, ma.MaskedArray.__copy__, ma.MaskedArray.__deepcopy__,
ma.MaskedArray.__imod__, ma.MaskedArray.__invert__, ma.MaskedArray.__mod__,
ma.MaskedArray.__or__, ma.MaskedArray.__rand__, ma.MaskedArray.__rmod__,
ma.MaskedArray.__ror__, ma.MaskedArray.__rxor__, ma.MaskedArray.__xor__, ma.MaskedArray.conj,
ma.MaskedArray.conjugate, ma.MaskedArray.copy, ma.MaskedArray.mT, ma.MaskedArray.ravel,
ma.MaskedArray.reshape, ma.MaskedArray.squeeze, ma.MaskedArray.swapaxes,
ma.MaskedArray.transpose, ma.append, ma.arccos, ma.arcsin, ma.arctanh, ma.atleast_1d,
ma.atleast_2d, ma.atleast_3d, ma.bitwise_and, ma.bitwise_or, ma.bitwise_xor, ma.column_stack,
ma.compress, ma.copy, ma.cumprod, ma.cumsum, ma.diagflat, ma.dstack, ma.empty, ma.empty_like,
ma.expand_dims, ma.fix_invalid, ma.hstack, ma.identity, ma.masked_all, ma.masked_all_like,
ma.masked_array, ma.masked_equal, ma.masked_greater, ma.masked_greater_equal, ma.masked_inside,
ma.masked_invalid, ma.masked_less, ma.masked_less_equal, ma.masked_not_equal, ma.masked_object,
ma.masked_outside, ma.masked_values, ma.masked_where, ma.ones, ma.ones_like, ma.ravel,
ma.repeat, ma.reshape, ma.resize, ma.row_stack, ma.set_fill_value, ma.squeeze, ma.swapaxes,
ma.take, ma.transpose, ma.vstack, ma.zeros, ma.zeros_like
```

Kept fully out of both #7 and #64, and out of every percentage in this
document, per the task's instruction — neither passing nor failing.

**Note on undercount vs. the old report's 201:** the old document counted
201 `ma.*`/`ma.MaskedArray.*` items as structurally blocked (136 + 65),
larger than this pass's 75. The difference is not a disagreement about the
defect — it's that the old count included every `ma.*` item regardless of
whether the harness could actually drive a call on it, while this pass
only counts an item here once a real call is confirmed to return a
`MaskedArray` lacking `.strides`. A further 34 `ma.*` items resolved to
scalars/bools/tuples/exception classes/sentinels (no array-shaped output
at all, e.g. `ma.ndim`, `ma.is_masked`, `ma.MaskError`, `ma.nomask`) — NA
by the same logic as `dtype.*`/`testing.*`, not `#63`, listed separately
below. The remaining `ma.*` items sit in the general 720-item "out of
scope for this pass" pool (unresolved, wrong-dtype probe, or no
auto-derivable call signature), same as in the old report — not
reconfirmed as unverifiable, not reconfirmed as anything.

```
NA-additional (ma.*, no array-shaped output, not #63):
ma.MAError, ma.MaskError, ma.MaskType, ma.MaskedArray.__bool__, ma.MaskedArray.__complex__,
ma.MaskedArray.__contains__, ma.MaskedArray.__float__, ma.MaskedArray.__index__,
ma.MaskedArray.__int__, ma.MaskedArray.__iter__, ma.MaskedArray.__len__, ma.MaskedArray.count,
ma.MaskedArray.ndim, ma.MaskedArray.shape, ma.MaskedArray.size, ma.bool_, ma.count,
ma.default_fill_value, ma.getmask, ma.isMA, ma.isMaskedArray, ma.is_mask, ma.is_masked,
ma.isarray, ma.masked, ma.masked_print_option, ma.masked_singleton, ma.maximum_fill_value,
ma.minimum_fill_value, ma.ndim, ma.nomask, ma.nonzero, ma.shape, ma.size
```

## Clean (367 items)

Full list — grown from the old 77 mainly by ~54 newly-measurable
`matrix.*` items (the `__array__` fix) plus `expand_dims` (the reshape
fix) plus items that were previously measured on the invalid `len1axF`/
`slice`/zero-extent combos and are now correctly excluded from a false
FALSE verdict:

```
abs, absolute, acos, acosh, add, all, amax, amin, angle, any, append, arccos, arccosh, arcsin,
arcsinh, arctan, arctan2, arctanh, argmax, argmin, argsort, around, asanyarray, asarray,
asarray_chkfinite, ascontiguousarray, asin, asinh, atan, atan2, atanh, atleast_1d, atleast_2d,
atleast_3d, average, cbrt, ceil, clip, column_stack, conj, conjugate, copy, copysign, copyto,
cos, cosh, count_nonzero, cumprod, cumsum, cumulative_prod, cumulative_sum, deg2rad, degrees,
delete, diagflat, divide, dstack, ediff1d, emath.arccos, emath.arcsin, emath.arctanh,
emath.log, emath.log10, emath.log2, emath.logn, emath.sqrt, empty_like, exp, exp2, expand_dims,
extract, fabs, fill_diagonal, flatnonzero, flip, fliplr, flipud, floor, floor_divide, fmax,
fmin, fmod, full_like, heaviside, intersect1d, iscomplex, isfinite, isin, isinf, isnan,
isneginf, isposinf, kron, linalg.cond, linalg.matrix_norm, linalg.matrix_rank,
linalg.matrix_transpose, linalg.multi_dot, linalg.pinv, linalg.svdvals, linalg.tensordot, log,
log10, log1p, log2, logaddexp, logaddexp2, logical_and, logical_not, logical_or, logical_xor,
ma.MaskedArray.__abs__, ma.MaskedArray.__add__, ma.MaskedArray.__eq__,
ma.MaskedArray.__floordiv__, ma.MaskedArray.__ge__, ma.MaskedArray.__getitem__,
ma.MaskedArray.__gt__, ma.MaskedArray.__le__, ma.MaskedArray.__lt__, ma.MaskedArray.__mul__,
ma.MaskedArray.__ne__, ma.MaskedArray.__neg__, ma.MaskedArray.__pos__, ma.MaskedArray.__radd__,
ma.MaskedArray.__rfloordiv__, ma.MaskedArray.__rmul__, ma.MaskedArray.__rsub__,
ma.MaskedArray.__rtruediv__, ma.MaskedArray.__sub__, ma.MaskedArray.__truediv__,
ma.MaskedArray.all, ma.MaskedArray.any, ma.MaskedArray.argmax, ma.MaskedArray.argmin,
ma.MaskedArray.compressed, ma.MaskedArray.fill_value, ma.MaskedArray.filled,
ma.MaskedArray.max, ma.MaskedArray.min, ma.MaskedArray.ptp, ma.MaskedArray.sum, ma.abs,
ma.absolute, ma.add, ma.all, ma.allequal, ma.alltrue, ma.amax, ma.amin, ma.any, ma.arccosh,
ma.arcsinh, ma.arctan, ma.arctan2, ma.argmax, ma.argmin, ma.ceil, ma.common_fill_value,
ma.compressed, ma.conjugate, ma.cosh, ma.count_masked, ma.divide, ma.equal, ma.exp, ma.fabs,
ma.filled, ma.floor, ma.floor_divide, ma.fmod, ma.getdata, ma.getmaskarray, ma.greater,
ma.greater_equal, ma.hypot, ma.less, ma.less_equal, ma.log, ma.log10, ma.log2, ma.logical_and,
ma.logical_not, ma.logical_or, ma.logical_xor, ma.make_mask_none, ma.mask_or, ma.max, ma.min,
ma.mod, ma.multiply, ma.negative, ma.not_equal, ma.ptp, ma.remainder, ma.sin, ma.sinh,
ma.sometrue, ma.sqrt, ma.subtract, ma.sum, ma.true_divide, matrix.A, matrix.A1, matrix.H,
matrix.I, matrix.T, matrix.__abs__, matrix.__add__, matrix.__copy__, matrix.__eq__,
matrix.__floordiv__, matrix.__ge__, matrix.__getitem__, matrix.__gt__, matrix.__le__,
matrix.__lt__, matrix.__mod__, matrix.__ne__, matrix.__neg__, matrix.__radd__,
matrix.__rfloordiv__, matrix.__rmod__, matrix.__rsub__, matrix.__sub__, matrix.all, matrix.any,
matrix.argmax, matrix.argmin, matrix.argsort, matrix.clip, matrix.conj, matrix.conjugate,
matrix.copy, matrix.flatten, matrix.getA, matrix.getA1, matrix.getH, matrix.getI, matrix.getT,
matrix.mT, matrix.max, matrix.mean, matrix.min, matrix.prod, matrix.ptp, matrix.ravel,
matrix.reshape, matrix.sort, matrix.squeeze, matrix.std, matrix.sum, matrix.swapaxes,
matrix.trace, matrix.transpose, matrix.var, matrix_transpose, max, maximum, mean, min, minimum,
mod, multiply, nan_to_num, nanargmax, nanargmin, nancumprod, nancumsum, nanmax, nanmean,
nanmin, nanstd, nansum, nanvar, ndarray.__abs__, ndarray.__add__, ndarray.__copy__,
ndarray.__deepcopy__, ndarray.__eq__, ndarray.__floordiv__, ndarray.__ge__, ndarray.__gt__,
ndarray.__ifloordiv__, ndarray.__imod__, ndarray.__le__, ndarray.__lt__, ndarray.__mod__,
ndarray.__mul__, ndarray.__ne__, ndarray.__neg__, ndarray.__pos__, ndarray.__radd__,
ndarray.__rfloordiv__, ndarray.__rmod__, ndarray.__rmul__, ndarray.__rsub__,
ndarray.__setitem__, ndarray.__sub__, ndarray.all, ndarray.any, ndarray.argmax, ndarray.argmin,
ndarray.argsort, ndarray.clip, ndarray.conj, ndarray.conjugate, ndarray.copy, ndarray.cumprod,
ndarray.cumsum, ndarray.flatten, ndarray.mT, ndarray.max, ndarray.mean, ndarray.min,
ndarray.ravel, ndarray.reshape, ndarray.round, ndarray.sort, ndarray.squeeze, ndarray.std,
ndarray.swapaxes, ndarray.trace, ndarray.transpose, ndarray.var, negative, nextafter,
ones_like, outer, permute_dims, positive, ptp, rad2deg, radians, ravel, real_if_close,
reciprocal, remainder, reshape, resize, rint, round, setdiff1d, setxor1d, sign, signbit, sin,
sinh, spacing, sqrt, square, squeeze, std, subtract, sum, take, tan, tanh, transpose, tril,
triu, true_divide, trunc, union1d, unique, var, zeros_like
```

Same caveat the old report carried: some of these are clean by *contract*
(e.g. `ndarray.copy()` defaults `order='C'`, so there's nothing to
propagate) rather than by correctly handling a hard case — not
re-adjudicated here, carried forward as-is.

## What I could not determine

- **Item-by-item identity of the old 118/218.** The source document names
  only representative examples per family, not the full list. Explained
  at the mechanism level above; not fabricated at the item level.
- **Whether any of the 720 out-of-scope items (`polynomial.*`, `random.*`,
  `char.*`/`strings.*`, square-matrix `linalg.*`, and everything the
  generic call cascade couldn't drive) are clean or false.** Unmeasured,
  not counted as either — same scoping as the prior audit, not
  re-attempted here (time-boxed to re-measuring what changed).
- **Whether the length-1-axis-F / slice / zero-extent base-construction
  bugs found in "Methodology correction" above are themselves already
  covered by an open ticket.** They look like #7 (len1-axis) and
  #61/#67 (zero-extent) by symptom, and a genuinely new one for the
  step-slice case (`a[::2]`) which I did not find named anywhere in the
  existing docs — flagging, not filing, per the task's "measurement only"
  scope.

## Reproduction

Probe scripts (not part of the shipped tree, per the task's own
constraint): `/private/tmp/audit_harness.py`, `/private/tmp/audit_ops.py`,
`/private/tmp/audit_main.py` (this pass's only code change: added the
base-strides-equality check described above), `/private/tmp/audit_ma_check.py`
and `/private/tmp/audit_ma_detail.json` (manual follow-up on the 32 `ma.*`
items the generic cascade couldn't auto-drive), `/private/tmp/audit_analyze.py`,
`/private/tmp/audit_finalize2.py`. Raw results:
`/private/tmp/audit_results.json`, `/private/tmp/audit_partition.json`,
`/private/tmp/audit_final2.json`. Item ledger swept:
`/private/tmp/exact_items.tsv` (verified identical to the prior pass's
ledger snapshot). None of these are checked into the repo; this document
is the only new file.

---

## CORRECTION — 2026-08-08 (Monday, on review): the "base-construction bug" does not exist

**Everything above is left standing as written.** This section does not
rewrite it; it retracts one claim in it, and every number that depended on
that claim. Where the two disagree, this section is current.

### The retracted claim

The section *"Methodology correction found mid-measurement"* reports that
anionpy's own base-construction primitives diverge from numpy before any
operation runs, with three measurements:

| construction | claimed numpy | claimed anionpy |
|---|---|---|
| `zeros((4,1,2)).copy('F')` | `(8,32,32)` | `(8,64,32)` |
| `arange(12)[::2]` | `(16,)` | `(8,)` |
| `zeros((0,3))` | `(0,0)` | `(24,8)` |

Measured directly, on the same commit (`5f2e09b`), same interpreter, same
installed `.so`:

```
                     numpy.strides   anionpy.strides   np.asarray(anionpy).strides
(4,1,2).copy('F')    (8, 32, 32)     (8, 32, 32)       (8, 64, 32)
arange(12)[::2]      (16,)           (16,)             (8,)
zeros((0,3))         (0, 0)          (0, 0)            (24, 8)
```

**anionpy matches numpy on all three.** Every claimed divergent value is
reproduced exactly by the third column — `np.asarray(anionpy_arr).strides`.
The report states its check was implemented "via `np.asarray(...).strides`",
and that is the whole defect: `np.asarray` of an anionpy array
*rematerializes* it through the array protocol and hands back a fresh numpy
buffer with numpy's own C-contiguous strides. On the numpy side of the
comparison `np.asarray` is the identity; on the anionpy side it is a copy.
The instrument was asymmetric, so it reported the rematerialization as a
defect in the thing being measured.

### Why this invalidates the headline, not just a paragraph

The harness did not merely *report* those combos — it **skipped** them:
*"skip (not count either way) any combo where the base itself diverges."*
The skipped combos are `len1axF`, `slice`, and every zero-extent shape.
Those are not an arbitrary three-of-eleven. **They are #7's entire
population** — length-1-axis-with-F-order and zero-extent are the two
conditions #7 is defined by.

So `#7: 118 → 1` is not a ticket shrinking under two landed fixes. It is a
corpus narrowed to exclude exactly the shapes the ticket is about, on the
strength of an artifact. That is the failure mode this project names
explicitly as breaking the build — *narrowing a corpus to exclude a failing
shape*. The `#64: 58` figure rests on the same skip list and is equally
unsafe; it may be deflated, and nothing here shows by how much.

**Numbers that do NOT survive: 58, 1, 367, and the named lists derived from
them.** The `matrix` improvement is independently corroborated (I verified
`a2a1ebb` out of corpus when I closed #66) and the 75 `#63`-unverifiable
`ma.*` items do not depend on the skip, so those two survive.

### The finding that is real, and worth keeping

`np.asarray(anionpy_arr)` **does not preserve layout.** F-order, non-unit
step, and zero-extent inputs all come back C-rematerialized. That is a
genuine interop defect — numpy code handed an anionpy array silently gets a
different memory layout than it asked for — and it is a *new* one, adjacent
to but not covered by #39 (buffer protocol / no `.view`). Ticketed
separately. It is only not a base-construction bug.

### The pattern, stated plainly

This report correctly identifies that the audit lineage has now caught the
same class of error three times — `ma.MaskedArray` tested through
plain-`ndarray` bases, `matrix` methods blamed for the bare constructor,
and (it argued) operations blamed for the input builder. It is right about
the pattern and it is the **fourth** instance, not the third: the instrument
was measuring a conversion of the object rather than the object. Diagnosing
the class correctly is not the same as being outside it.

**Read the attribute off the object under test. Never off a conversion of
it.** A comparison in which one side passes through a copy and the other
does not is not a comparison.

---

## RE-MEASURE — 2026-08-08 (corrected instrument): #7 = 118, #64 = 64

**Everything above is left standing as written, including the CORRECTION
section.** This section does not edit or delete anything above it. It
supersedes the withdrawn numbers (58, 1, 367, and the lists built on them)
with a fresh derivation from a corrected instrument. Where this section
and an earlier one disagree, this section is current.

### `.so` provenance check, done first

Per the standing caution that the installed `.so` is shared mutable state
across concurrent sessions:

```
~/Monday/ionp/anionpy/_anionpy.abi3.so   mtime 2026-08-08 00:35:09
/private/tmp/audit_results.json (this run's output)  mtime 2026-08-08 01:19:08
/private/tmp/audit_final2.json  (this run's output)  mtime 2026-08-08 01:19:53
```

The `.so` predates the run that produced these numbers by ~44 minutes and
nothing wrote to it in between. I cannot rule out that it was rebuilt to
identical content by another session, but there is no evidence it changed
under me during this measurement.

### The instrument fix

`/private/tmp/audit_main.py`'s base-equality gate now reads
`nb.strides`/`ib.strides` directly off the constructed base objects, never
through `np.asarray(...)`. The rest of the harness (`describe()` in
`audit_harness.py`, which the earlier CORRECTION section confirmed was
already reading strides directly and was never the bug) is unchanged.

### Hand sanity-check, outside the harness, before trusting the rerun

Per the coordinator's instruction to verify at least three previously-
skipped combos now compare cleanly end-to-end:

```
len1axF base (4,1,2) + sort:
  base strides — numpy (8,32,32)  ionp (8,32,32)   MATCH (valid base)
  sort() output — numpy (8,32,32)  ionp (16,16,8)   DIVERGE
  -> genuine #7-relevant defect surfaces now that the base itself is
     admitted instead of being silently excluded.

slice base arange(12)[::2] + roll:
  base strides — numpy (16,)  ionp (16,)   MATCH
  roll() output — numpy (8,)  ionp (8,)    MATCH

zero-extent base zeros((0,3)) + zeros_like:
  base strides — numpy (0,0)  ionp (0,0)   MATCH
  zeros_like() output — numpy (0,0)  ionp (0,0)   MATCH
```

All three flow through to a real per-op comparison instead of being
dropped. The harness is trusted on this basis.

### Remaining base-invalid combos, reported by name (not skipped silently)

After the fix, exactly **74 items** still have any base-invalid combo, and
all 74 are `matrix.*`. The base-invalid combos themselves reduce to
**4 distinct (layout, dtype) pairs** — `matrix()`'s own constructor loses
F-order on construction, so every `matrix.*` item inherits it on
`F_2x3`/`Ftrans_2x3` at `float32`/`float64` (the only dtypes the probe
sweep uses these layouts with):

```
(F_2x3,     float32)  numpy (4,8)    ionp (12,4)
(F_2x3,     float64)  numpy (8,16)   ionp (24,8)
(Ftrans_2x3, float32) numpy (4,8)    ionp (12,4)
(Ftrans_2x3, float64) numpy (8,16)   ionp (24,8)
```

This is the same `matrix()`-constructor F-order-loss defect already
described in the original (non-retracted) part of this report under
"`matrix.__array__` (`a2a1ebb`)" — real, on `matrix`'s own constructor, not
on any of the 74 downstream methods individually. Those 4 combos are
excluded from each of the 74 items' per-op comparison (correctly — the
task's own method constraint requires this); every other combo for those
74 items (`C`, `len1ax_C`, `len1axF`, `slice`, `0d`, zero-extent, etc.) is
base-valid and was compared normally, which is why most of the 74 still
land in CLEAN or TICKET64 below rather than being dropped outright. No
non-`matrix` item has any base-invalid combo under the corrected
instrument — the asarray bug was the sole source of every other exclusion
in the previous (withdrawn) run.

### Corrected headline

| bucket | withdrawn (asarray-bug run) | corrected (this run) | ticket |
|---|---:|---:|---|
| **#7 only** | ~~1~~ | **118** | #7 |
| **#64** | ~~58~~ | **64** | #64 |
| Clean | ~~367~~ | **245** | — |
| Unverifiable (#63) | 75 | **75** (unchanged) | #63 |
| NA-additional `ma.*` (no array output) | 34 | **34** (unchanged) | — |
| NA (original, metadata/assertion items) | 74 | **74** (unchanged) | — |
| Out of scope for this pass | 720 | **719** | — |
| **Total declared exact** | 1329 | **1329** | |

`#63`/NA figures are unchanged because `ma.*`/`ma.MaskedArray.*` base
construction never went through `np.asarray()` on the anionpy side — that
code path was never exposed to the bug, confirmed by rerunning
`audit_ma_check.py` after the fix and getting an identical 79/32 raw split
to the pre-fix run.

**Do not reconcile these numbers against the withdrawn 58/1/367** — they
are a fresh derivation, though as it happens the arithmetic ties out
exactly against them (below), which is itself part of the evidence the fix
worked as intended rather than introducing a new, different distortion.

### #7 (118 items) — named list

```
abs, absolute, acos, acosh, add, angle, arccos, arccosh, arcsin, arcsinh, arctan, arctan2,
arctanh, around, asin, asinh, atan, atan2, atanh, broadcast_to, cbrt, ceil, clip, conj,
conjugate, copy, copysign, cos, cosh, deg2rad, degrees, divide, emath.arccos, emath.arcsin,
emath.arctanh, emath.log, emath.log10, emath.log2, emath.logn, emath.sqrt, empty_like, exp,
exp2, fabs, floor, floor_divide, fmax, fmin, fmod, full_like, heaviside, isfinite, isinf,
isnan, isneginf, isposinf, log, log10, log1p, log2, logaddexp, logaddexp2, logical_and,
logical_not, logical_or, logical_xor, maximum, minimum, mod, multiply, nan_to_num,
ndarray.__abs__, ndarray.__add__, ndarray.__copy__, ndarray.__deepcopy__, ndarray.__eq__,
ndarray.__floordiv__, ndarray.__ge__, ndarray.__gt__, ndarray.__le__, ndarray.__lt__,
ndarray.__mod__, ndarray.__mul__, ndarray.__ne__, ndarray.__neg__, ndarray.__radd__,
ndarray.__rfloordiv__, ndarray.__rmod__, ndarray.__rmul__, ndarray.__rsub__, ndarray.__sub__,
ndarray.clip, ndarray.round, negative, nextafter, ones_like, positive, rad2deg, radians,
reciprocal, remainder, rint, round, sign, signbit, sin, sinh, spacing, sqrt, square, subtract,
tan, tanh, tril, triu, true_divide, trunc, zeros_like
```

`broadcast_to` (the one item that survived in #7 under the withdrawn/
asarray-bug run) is still here, confirmed on a valid base as before
(`len1ax_C` numpy/ionp strides both `(16,16,8)` pre-op; numpy output
`(16,0,8)` [0-stride broadcast dim], ionp output `(16,16,8)` [materializes
a real stride instead of 0]). The other 117 are new to this pass in the
sense that the withdrawn run counted them as CLEAN — see delta section
below for the mechanism.

### #64 (64 items) — named list

```
argwhere, bool_, byte, cdouble, complex128, complex64, csingle, diff, double, fft.fftshift,
fft.hfft, fft.ifftshift, fft.irfft, flatnonzero, float16, float32, float64, half, hstack, i0,
imag, int16, int32, int64, int8, int_, intc, intp, isclose, isin, isreal, linspace, long,
ma.make_mask, matrix.__pos__, matrix.cumprod, matrix.cumsum, matrix.fill, matrix.repeat,
matrix.reshape, ndarray.__pos__, ndarray.fill, ndarray.reshape, real, reshape, roll, short,
sinc, single, sort, sort_complex, tile, ubyte, uint, uint16, uint32, uint64, uint8, uintc,
uintp, ulong, unwrap, ushort, vstack
```

### #64 delta against the withdrawn 58: exactly 6 items added, none removed

```
flatnonzero, isin, matrix.reshape, ndarray.__pos__, ndarray.reshape, reshape
```

All 6 were counted CLEAN under the withdrawn run. Mechanism: each has at
least one genuinely diverging combo whose *only* prior appearance was on a
now-restored base (`len1axF`/`slice`/zero-extent were the only combos the
asarray bug excluded), so with those combos correctly admitted, a real
non-length-1-axis divergence for these 6 also became visible on a
different, always-valid combo they share the failure mode with — i.e. this
is not "6 new bugs," it's 6 items whose #64-qualifying evidence was
previously hidden behind the same instrument bug, same as the 117 added to
#7. Every other one of the withdrawn 58 is unchanged and still present.

### CLEAN (245 items) — named list

```
all, amax, amin, any, append, argmax, argmin, argsort, asanyarray, asarray, asarray_chkfinite,
ascontiguousarray, atleast_1d, atleast_2d, atleast_3d, average, column_stack, copyto,
count_nonzero, cumprod, cumsum, cumulative_prod, cumulative_sum, delete, diagflat, dstack,
ediff1d, expand_dims, extract, fill_diagonal, flip, fliplr, flipud, indices, intersect1d,
iscomplex, kron, linalg.cond, linalg.matrix_norm, linalg.matrix_rank, linalg.matrix_transpose,
linalg.multi_dot, linalg.pinv, linalg.svdvals, linalg.tensordot, ma.MaskedArray.__abs__,
ma.MaskedArray.__add__, ma.MaskedArray.__eq__, ma.MaskedArray.__floordiv__,
ma.MaskedArray.__ge__, ma.MaskedArray.__getitem__, ma.MaskedArray.__gt__,
ma.MaskedArray.__le__, ma.MaskedArray.__lt__, ma.MaskedArray.__mul__, ma.MaskedArray.__ne__,
ma.MaskedArray.__neg__, ma.MaskedArray.__pos__, ma.MaskedArray.__radd__,
ma.MaskedArray.__rfloordiv__, ma.MaskedArray.__rmul__, ma.MaskedArray.__rsub__,
ma.MaskedArray.__rtruediv__, ma.MaskedArray.__sub__, ma.MaskedArray.__truediv__,
ma.MaskedArray.all, ma.MaskedArray.any, ma.MaskedArray.argmax, ma.MaskedArray.argmin,
ma.MaskedArray.compressed, ma.MaskedArray.fill_value, ma.MaskedArray.filled,
ma.MaskedArray.max, ma.MaskedArray.min, ma.MaskedArray.ptp, ma.MaskedArray.sum, ma.abs,
ma.absolute, ma.add, ma.all, ma.allequal, ma.alltrue, ma.amax, ma.amin, ma.any, ma.arccosh,
ma.arcsinh, ma.arctan, ma.arctan2, ma.argmax, ma.argmin, ma.ceil, ma.common_fill_value,
ma.compressed, ma.conjugate, ma.cosh, ma.count_masked, ma.divide, ma.equal, ma.exp, ma.fabs,
ma.filled, ma.floor, ma.floor_divide, ma.fmod, ma.getdata, ma.getmaskarray, ma.greater,
ma.greater_equal, ma.hypot, ma.less, ma.less_equal, ma.log, ma.log10, ma.log2, ma.logical_and,
ma.logical_not, ma.logical_or, ma.logical_xor, ma.make_mask_none, ma.mask_or, ma.max, ma.min,
ma.mod, ma.multiply, ma.negative, ma.not_equal, ma.ptp, ma.remainder, ma.sin, ma.sinh,
ma.sometrue, ma.sqrt, ma.subtract, ma.sum, ma.true_divide, matrix.A, matrix.A1, matrix.H,
matrix.I, matrix.T, matrix.__abs__, matrix.__add__, matrix.__copy__, matrix.__eq__,
matrix.__floordiv__, matrix.__ge__, matrix.__getitem__, matrix.__gt__, matrix.__le__,
matrix.__lt__, matrix.__mod__, matrix.__ne__, matrix.__neg__, matrix.__radd__,
matrix.__rfloordiv__, matrix.__rmod__, matrix.__rsub__, matrix.__sub__, matrix.all, matrix.any,
matrix.argmax, matrix.argmin, matrix.argsort, matrix.clip, matrix.conj, matrix.conjugate,
matrix.copy, matrix.flatten, matrix.getA, matrix.getA1, matrix.getH, matrix.getI, matrix.getT,
matrix.mT, matrix.max, matrix.mean, matrix.min, matrix.prod, matrix.ptp, matrix.ravel,
matrix.sort, matrix.squeeze, matrix.std, matrix.sum, matrix.swapaxes, matrix.trace,
matrix.transpose, matrix.var, matrix_transpose, max, mean, min, nanargmax, nanargmin,
nancumprod, nancumsum, nanmax, nanmean, nanmin, nanstd, nansum, nanvar, ndarray.__ifloordiv__,
ndarray.__imod__, ndarray.__setitem__, ndarray.all, ndarray.any, ndarray.argmax,
ndarray.argmin, ndarray.argsort, ndarray.conj, ndarray.conjugate, ndarray.copy,
ndarray.cumprod, ndarray.cumsum, ndarray.flatten, ndarray.mT, ndarray.max, ndarray.mean,
ndarray.min, ndarray.ravel, ndarray.sort, ndarray.squeeze, ndarray.std, ndarray.swapaxes,
ndarray.trace, ndarray.transpose, ndarray.var, outer, permute_dims, ptp, ravel, real_if_close,
resize, setdiff1d, setxor1d, squeeze, std, sum, take, transpose, union1d, unique, var
```

### CLEAN delta against the withdrawn 367: −122, arithmetic reconciled exactly

`367 − 245 = 122`. Diffed by name, not just by count:

- **117 items** moved CLEAN(withdrawn) → **#7**(corrected). This is the
  exact same 117-item set as "#7 items other than `broadcast_to`" above —
  every one of them was previously counted as having 0 divergent combos
  only because its real length-1-axis-or-0-d divergence lived on the
  excluded `len1axF`/zero-extent combos.
- **6 items** moved CLEAN(withdrawn) → **#64**(corrected): the same
  `flatnonzero, isin, matrix.reshape, ndarray.__pos__, ndarray.reshape,
  reshape` set named above.
- **+1 item**, separately: `ma.make_mask_none` was folded into CLEAN by
  the manual `ma.*` reclassification pass (`audit_finalize2.py`), same as
  in the withdrawn run — unaffected by the asarray fix, carried forward
  unchanged.

`367 − 117 − 6 + 1 = 245`. Exact.

### Unverifiable / NA — unchanged, re-confirmed by name

`UNVERIFIABLE` (75) and `NA-additional ma.*` (34) are **byte-identical** by
name to the withdrawn run's lists (diffed programmatically, zero items
different in either set). This is expected and was checked, not assumed:
`ma.*`/`ma.MaskedArray.*` base construction in `audit_harness.py`
(`build_masked_base`) never called `np.asarray()` on the anionpy side, so
that code path was never exposed to the bug the fix addresses. Both lists
are exactly as printed in the original (non-retracted) part of this
report above — not reproduced a second time here to avoid duplication.

### What moved off the old (pre-audit, `EMPTY-AND-ORDER-AUDIT.md`) 118/100 split — re-derived from scratch

Re-deriving directly against the *original* 118 (#7) / 100 (#64) split,
not against my own withdrawn numbers, per instruction #4:

- **#7: 118 → 118.** The corrected count lands on the same number as the
  original document's #7 figure. I want to be precise about what that
  does and doesn't establish: the original document does not enumerate
  the 118 by name (only ~7 representative examples), so I cannot confirm
  membership-for-membership that it's the *same* 118. What I can confirm:
  the corrected instrument, built independently and diffed against every
  item the original document does name for #7 (`sort`, `roll`, `tile`
  appear in the original's worked examples for the *wider* #64-adjacent
  pattern, not as #7-only — consistent with this pass, where all three
  land in #64, not #7, both times) shows no contradiction. Treat the
  numeric match as corroborating, not as proof of identical membership.
- **#64: 100 → 64.** A real, smaller-but-legitimate shrink, not an
  artifact this time — the asarray bug is fixed and confirmed not to be
  suppressing anything on the #64 side (no non-`matrix` item has any
  base-invalid combo left, per the disclosure above). Primary driver:
  the `matrix.__array__` fix (`a2a1ebb`) — of the ~38 `matrix.*` items
  the original audit's SECOND CORRECTION attributed to "family E," this
  pass measures 54 `matrix.*` items CLEAN, 5 still in #64
  (`matrix.__pos__`, `.cumprod`, `.cumsum`, `.fill`, `.repeat`, all
  zero-extent-only bar `.repeat`), 1 newly resolved into #64
  (`matrix.reshape`, via the mechanism in the delta section above), and
  74 with `F_2x3`/`Ftrans_2x3` combos correctly excluded as base-invalid
  (disclosed by name above) rather than silently miscounted either way.
  Secondary driver: `expand_dims` (`7aa74e6`) — measures CLEAN, 0
  divergent combos. Everything else named in the original document's
  #64 examples (dtype casts, `fft.*`, `hstack`/`vstack`, `real`/`imag`,
  `isclose`/`isreal`, `unwrap`, `sinc`, `i0`, `sort_complex`, `diff`,
  `sort`, `roll`, `tile`) reconfirms present in the corrected 64 on a
  base-validated combo — unchanged, no fix landed for these.
- As with the original pass, **I cannot do a literal item-by-item diff
  against the source document's 118/100** because it only names
  representative examples, not the full 218. This is stated plainly
  rather than filled in by inference this time, since inference is
  exactly the step that produced the retracted claim last time.

### Standing items, unaffected by this correction

- The `np.asarray(anionpy_arr)` layout-loss finding (ticket #69) stands as
  filed by the coordinator — out of scope to fix here, and the corrected
  instrument no longer routes through it anywhere.
- "What I could not determine" from the original pass (identity of the
  old 118/218 by name; whether the 719 out-of-scope items are clean or
  false; whether the length-1-axis-F/slice/zero-extent *base*-construction
  question is itself ticketed) is superseded by this section's more
  precise finding: **those base-construction primitives are not broken.**
  Direct, off-object measurement (both the withdrawn CORRECTION's evidence
  and this pass's independent 3-combo hand check) shows `zeros().copy('F')`,
  step-slicing, and zero-extent construction all match numpy's strides
  exactly when read off the object itself. There is no open question left
  there — it was the instrument.

### Reproduction (this section)

Same scratch files as the original Reproduction section, with
`/private/tmp/audit_main.py` now containing the corrected (direct-
attribute-read) base-equality gate rather than the asarray-based one.
Rerun order: `audit_main.py` → `audit_analyze.py` → `audit_ma_check.py`
(unchanged output, re-run for confirmation only) → `audit_finalize2.py`,
producing `/private/tmp/audit_final2.json` (final named lists) and
`/private/tmp/audit_results.json` (raw per-item results including
`base_invalid_combos`, used for the 4-combo `matrix` disclosure above).
None of these are checked into the repo.

---

## SECOND CORRECTION — 2026-08-08 (Monday, on review): the 74 excluded `matrix.*` bases are not invalid either

The corrected pass above is right about `#7 = 118` — I verified that
independently and it is a genuine finding on a valid base:

```
zeros((4,1,2)).copy('F'), bases identical at (8,32,32) on both sides:
  numpy   sort(axis=0).strides = (8, 32, 32)
  anionpy sort(axis=0).strides = (16, 16, 8)
```

Its other two hand-checks (`slice`+`roll`, zero-extent+`zeros_like`) also
reproduce clean. Those results stand.

**What does not stand is the remaining exclusion.** The pass reports 74
`matrix.*` items still base-invalid, "all reducing to the same 4
`(layout, dtype)` pairs (`F_2x3`/`Ftrans_2x3` × float32/float64) — the known
`matrix()`-constructor F-order-loss defect." Measured directly, reading
`.strides` off each object:

```
      layout        dtype     numpy         anionpy
  OK  F_2x3         float64   (24, 8)       (24, 8)
  OK  F_2x3         float32   (12, 4)       (12, 4)
  OK  Ftrans_2x3    float64   (24, 8)       (24, 8)
  OK  Ftrans_2x3    float32   (12, 4)       (12, 4)
```

All four agree. There is no `matrix()` F-order-loss defect to exclude on:
**numpy's own `np.matrix` discards F-order too** — `np.matrix` of an
F-ordered `(2,3)` returns C strides `(24,8)`, and anionpy matches it. Both
sides losing F-order identically is conformance, not a defect.

The likely mechanism is the same one as the first correction, one layer
further on: `anionpy.matrix` gained a real `.strides` only in `a2a1ebb`
(ticket #66, hours before this measurement). A gate that reaches for strides
by any other route — `np.asarray`, or the `_data` attribute, or a `hasattr`
fallback written before `a2a1ebb` landed — will still see a rematerialized
or mismatched tuple and mark the base invalid. `hasattr(anionpy.matrix(...),
'strides')` is now `True`; read it directly.

**Consequence: `#64 = 64` is still a floor, not a count.** 74 `matrix.*`
items were excluded without being measured, so the true figure is somewhere
in `[64, 138]`. `#7 = 118`, `UNVERIFIABLE = 75` and the 34 NA items are
unaffected — none of them depend on the matrix exclusion.

### Third time, same shape

First correction: strides read through `np.asarray`. Second: strides read
through something that is not the object's own `.strides`. Both times the
artifact did not merely mis-report — it drove an **exclusion**, and an
exclusion is invisible in the headline. A wrong number announces itself
eventually; a silently narrowed denominator does not.

Standing rule for any future pass on this ticket: **an exclusion list is a
finding and must be printed by name with both stride tuples pasted, every
time, even when it is believed to be a known defect.** "Known defect" was
the justification offered for all 74 of these, and it was wrong.

---

## THIRD PASS — 2026-08-08 (Monday): `#64 = 62`, and a genuinely new 8-combo exclusion

**Everything above is left standing as written, including both prior
corrections.** This section does not edit or delete anything above it.

### The speculated mechanism was checked and does not hold — the actual one does

The SECOND CORRECTION speculates the gate reaches strides "by another
route — `np.asarray`, or the `_data` attribute, or a `hasattr` fallback."
Checked directly, per instruction #1, before trusting anything further:

```
nb = build_matrix_base(np, (2,3), 'F', 'float64')
ib = build_matrix_base(ionp, (2,3), 'F', 'float64')
hasattr(ib, 'strides')          -> True
type(ib).strides.fget           -> <function matrix.strides at 0x...>
```

`type(ib).strides.fget` is `anionpy.matrix.strides`'s own `@property`
getter (`~/Monday/ionp/anionpy/matrix.py`, `return
self._data.strides`) — a real property on the matrix object itself. The
gate in `/private/tmp/audit_main.py` (lines ~111-118) calls `tuple(ib
.strides)` directly on this object, no `np.asarray`, no `_data` reached
around the object, no `hasattr` fallback branch. **The gate was already
reading the right attribute the right way.** So the F/Ftrans exclusion was
not a stale-route bug — it was a real, reproducible divergence, just one
that was out of the contract being tested. Reconciling that:

```
build_matrix_base's exact call: mod.matrix(inner, dtype=dtype, copy=False)

np.matrix(F-ordered (2,3) float64, copy=False):    strides (8, 16)  -- true
  zero-copy view, base IS the F-ordered input (mn.base is inner_np -> True)
anionpy.matrix(F-ordered (2,3) float64, copy=False): strides (24, 8) -- a
  full copy back to C order, despite copy=False being requested
```

This is not new — it is `~/Monday/ionp/anionpy/matrix.py`
lines 20-33 and `KNOWN-DIFFERENCES.md`'s 2026-08-07 "`matrix`/`recarray`/
`memmap`" entry, both already in the tree: `anionpy` has no buffer-
aliasing construction path at all, so `matrix(..., copy=False)` silently
always copies, and this is **documented as permanently DECLINED** — never
declared exact, for any of the 92 `matrix.*` items. The same source
states plainly: *"every OTHER item in this module is tested and declared
using ONLY `copy=True` (the default)-constructed matrices."* My harness's
`build_matrix_base` called `copy=False`. That is what produced the
divergence — a real one, reproducible, but on a construction path that
was never part of what any of these 92 items' declared-exact status
means. Testing it was testing outside the ledger's own contract, not
measuring the items.

### The fix

Changed `/private/tmp/audit_harness.py`'s `build_matrix_base` from
`mod.matrix(inner, dtype=dtype, copy=True)` — matching what "matrix" means
everywhere else these items are declared and tested — not a workaround, a
correction to match the actual declared contract. Verified before
trusting the rerun, per instruction #1:

```
      layout        dtype     numpy       anionpy      hasattr  path
  OK  F_2x3         float64   (24, 8)     (24, 8)       True    matrix.strides property (self._data.strides)
  OK  F_2x3         float32   (12, 4)     (12, 4)       True    matrix.strides property (self._data.strides)
  OK  Ftrans_2x3    float64   (24, 8)     (24, 8)       True    matrix.strides property (self._data.strides)
  OK  Ftrans_2x3    float32   (12, 4)     (12, 4)       True    matrix.strides property (self._data.strides)
```

All four now match, reproducing the coordinator's independently-measured
table exactly. No shipped code touched — `anionpy/matrix.py` was read,
not edited.

### Re-run: a different, genuinely new 8-combo exclusion appears

Re-running the full sweep with the fix, `base_invalid_combos` still shows
exactly 74 items, still all `matrix.*` — but the excluded combos changed
entirely. `F_2x3`/`Ftrans_2x3` no longer appear anywhere. In their place,
8 new (label, dtype) pairs, all on the two zero-extent shapes a `matrix`
can represent (`(0,3)` and `(0,0)`; `(3,0)` and 1-D `(0,)` were checked
directly and do NOT diverge):

```
empty_C_(0, 3)  float64   numpy (0, 0)   anionpy (8, 8)
empty_C_(0, 3)  float32   numpy (0, 0)   anionpy (4, 4)
empty_F_(0, 3)  float64   numpy (0, 0)   anionpy (8, 8)
empty_F_(0, 3)  float32   numpy (0, 0)   anionpy (4, 4)
empty_C_(0, 0)  float64   numpy (0, 0)   anionpy (8, 8)
empty_C_(0, 0)  float32   numpy (0, 0)   anionpy (4, 4)
empty_F_(0, 0)  float64   numpy (0, 0)   anionpy (8, 8)
empty_F_(0, 0)  float32   numpy (0, 0)   anionpy (4, 4)
```

**Per the standing rule, this is reported as a finding, not silently
skipped, even though the mechanism looks bounded and well-understood** —
that is exactly the posture that was wrong twice already. Isolated by
hand, outside the harness:

```
np.matrix(np.array([]), dtype='float64', copy=True).strides    -> (0, 0)
ionp.matrix(ionp.array([]), dtype='float64', copy=True).strides -> (8, 8)

For comparison, plain .reshape((1,0)) on the same empty 1-D arrays
agrees on BOTH sides -- (8, 8) numpy AND (8, 8) anionpy -- so this is not
a generic reshape-strides disagreement.
```

Mechanism: `anionpy.matrix.__new__` promotes a 1-D input to 2-D via a
literal `.reshape((1, arr.shape[0]))` call
(`~/Monday/ionp/anionpy/matrix.py` line 118), which for an
empty array agrees with `.reshape`'s own general convention on both sides
`(8,8)`/`(4,4)`, itemsize-based. Real numpy's `matrix()` constructor does
**not** go through a plain `.reshape` for this promotion internally, and
produces numpy's other general empty-array convention, `(0,0)`, instead.
So real numpy has two different empty-array stride conventions depending
on construction path (plain `.reshape` vs. `matrix()`'s own internal
promotion) and `anionpy.matrix` only reproduces one of them. **This is a
new, distinct finding** — not the `copy=False` gap (this reproduces at
`copy=True`, the declared-contract construction), and not mentioned
anywhere in `KNOWN-DIFFERENCES.md`'s existing `matrix` entry. Flagging it
plainly rather than filing it, per this task's measurement-only scope.

Separately, worth disclosing rather than folding in silently: `nested_list()`
(the harness's own input-data generator, `/private/tmp/audit_harness.py`)
collapses any shape with a size-0 leading dimension to a bare `[]`, so
`build_base`/`build_matrix_base` actually construct a 1-D `(0,)` array for
the `empty_*_(0, 3)` and `empty_*_(0, 0)` labels rather than literally a
`(0,3)`- or `(0,0)`-shaped one (`(2,0,4)` and `(3,0)`, where the leading
dim is non-zero, are unaffected). This is a pre-existing harness
limitation, not introduced this pass, and it does not manufacture the
divergence above out of nothing — the same divergence was independently
confirmed by hand on an explicitly-constructed empty 1-D array outside
the harness, in the isolated repro just above — but it does mean the
`empty_*_(0, 3)`/`empty_*_(0, 0)` labels are, for every item in this
entire 1329-item sweep (not just `matrix`), actually exercising a `(0,)`
input, not the literal shape the label names. Flagging as a scope
limitation of this measurement, not fixing it (out of this pass's time-box
and not requested).

### The 74, measured — not excluded, per instruction #2

None of the 74 `matrix.*` items were dropped. Every combo except the 8
above (out of 11 layout combos × 2 dtypes × however many zero-extent
shapes apply per item = up to 22 per item) was compared normally. Per-item
outcome for the 74:

| bucket | count | items |
|---|---:|---|
| CLEAN | 55 | (`matrix.A`, `.A1`, `.H`, `.I`, `.T`, dunders `__abs__/__add__/__copy__/__eq__/__floordiv__/__ge__/__getitem__/__gt__/__le__/__lt__/__mod__/__ne__/__neg__/__radd__/__rfloordiv__/__rmod__/__rsub__/__sub__`, `.all`, `.any`, `.argmax`, `.argmin`, `.argsort`, `.clip`, `.conj`, `.conjugate`, `.copy`, `.flatten`, `.getA`, `.getA1`, `.getH`, `.getI`, `.getT`, `.mT`, `.max`, `.mean`, `.min`, `.prod`, `.ptp`, `.ravel`, `.sort`, `.squeeze`, `.std`, `.sum`, `.swapaxes`, `.trace`, `.transpose`, `.var`) |
| #64 | 4 | `matrix.cumprod`, `matrix.cumsum`, `matrix.repeat`, `matrix.reshape` — all diverge on zero-extent bucket 5, on a base-**valid** combo (not the 8 excluded above; confirmed via `bucket_diverge` in `audit_results.json`, e.g. `{'5': 8}` occurrences tallied across the 8 valid empty-shape/dtype/layout combinations that remain after excluding the 2 invalid ones). `matrix.repeat` additionally diverges on buckets `{0,1,2,3,4}` — the broadest matrix defect, unchanged from earlier passes. |
| #7 | 0 | — |
| UNMEASURED (no array-shaped output; unrelated to base validity — metadata/dunders) | 15 | `matrix.__contains__`, `.__deepcopy__`, `.__iter__`, `.__len__`, `.__mul__`, `.__rmul__`, `.dtype`, `.itemsize`, `.nbytes`, `.ndim`, `.nonzero`, `.shape`, `.size`, `.tobytes`, `.tolist` |

`matrix.__pos__` and `matrix.fill` — both `#64` under the withdrawn
`copy=False` run — are now CLEAN: their earlier zero-extent divergence was
an artifact of the `copy=False` construction feeding a different base into
the op, not an independent defect. `matrix.cumprod`/`.cumsum`/`.repeat`/
`.reshape`'s zero-extent divergence, by contrast, reproduces on a
base-valid combo under `copy=True` and is real.

### Corrected headline

| bucket | prior pass (`copy=False`, retracted) | this pass (`copy=True`) | ticket |
|---|---:|---:|---|
| **#7 only** | 118 | **118** (unchanged) | #7 |
| **#64** | ~~64~~ | **62** | #64 |
| Clean | ~~245~~ | **247** | — |
| Unverifiable (#63) | 75 | **75** (unchanged) | #63 |
| NA-additional `ma.*` | 34 | **34** (unchanged) | — |
| NA (original) | 74 | **74** (unchanged) | — |
| Out of scope | 719 | **719** (unchanged) | — |
| **Total declared exact** | 1329 | **1329** | |

`#64: 64 → 62`, `CLEAN: 245 → 247` — exactly the 2 items (`matrix.__pos__`,
`matrix.fill`) that moved off `#64` once the construction matched the
declared contract. `#7` is untouched (no `matrix.*` item was ever in `#7`
in any pass). `#64 = 62` is no longer a floor: all 74 previously-excluded
items were measured on every combo except the 8 now individually named
above, and those 8 are disclosed, not absorbed into either bucket.

### #64 (62 items) — full named list

```
argwhere, bool_, byte, cdouble, complex128, complex64, csingle, diff, double, fft.fftshift,
fft.hfft, fft.ifftshift, fft.irfft, flatnonzero, float16, float32, float64, half, hstack, i0,
imag, int16, int32, int64, int8, int_, intc, intp, isclose, isin, isreal, linspace, long,
ma.make_mask, matrix.cumprod, matrix.cumsum, matrix.repeat, matrix.reshape, ndarray.__pos__,
ndarray.fill, ndarray.reshape, real, reshape, roll, short, sinc, single, sort, sort_complex,
tile, ubyte, uint, uint16, uint32, uint64, uint8, uintc, uintp, ulong, unwrap, ushort, vstack
```

### `.so` provenance check, repeated

```
~/Monday/ionp/anionpy/_anionpy.abi3.so   mtime 2026-08-08 00:35:09
```

unchanged from the prior section's check; still predates this pass's
output files. No source or `.so` rebuild was performed by me in this
pass — only `/private/tmp/audit_harness.py`'s `build_matrix_base` was
edited (a probe script, not shipped code).

### Reproduction

Same scratch files as before, with `/private/tmp/audit_harness.py`'s
`build_matrix_base` now using `copy=True`. Rerun order unchanged:
`audit_main.py` → `audit_analyze.py` → `audit_finalize2.py`. Raw per-item
results including the 8-combo `base_invalid_combos` for the 4 remaining
`#64` matrix items: `/private/tmp/audit_results.json`. Final named lists:
`/private/tmp/audit_final2.json`.

---

## THIRD CORRECTION — 2026-08-08 (Monday): my diagnosis was wrong, and 4 of the 62 are suspect

### I was wrong about the mechanism

My second correction guessed the 74-item exclusion came from a gate reaching
for `.strides` via `np.asarray`, `_data`, or a stale `hasattr` fallback. It
did not. The third pass verified the gate was already reading the matrix
object's real `@property`, and found the actual cause: the harness built
bases with `matrix(inner, copy=False)`, and `anionpy.matrix`'s `copy=False`
path is a **documented, permanently-declined divergence** — none of the 92
`matrix.*` ledger items were ever declared exact under that construction.
Confirmed directly:

```
                 copy=False        copy=True
  numpy          (8, 16)           (24, 8)
  anionpy        (24, 8)           (24, 8)
```

numpy's `copy=False` preserves the F-ordered base; anionpy's does not, by
prior decision. So the exclusion had a real cause — it was just the wrong
*contract*, not a broken instrument. My "second artifact" framing was
overstated: the correct criticism is that the harness tested a construction
none of the items are declared under, and that the exclusion was not printed
by name. Both of those stand. The mechanism I named does not.

### A real new divergence, independently confirmed

`matrix()` on zero-extent input:

```
  numpy    matrix(array([]), copy=True).strides  = (0, 0)
  anionpy  matrix(array([]), copy=True).strides  = (8, 8)
  numpy    array([]).reshape((1,0)).strides      = (8, 8)
  anionpy  array([]).reshape((1,0)).strides      = (8, 8)
```

Both sides agree on `reshape`; they disagree on the constructor. numpy applies
the fresh-allocation zero-stride rule inside `matrix.__new__`'s 1-D→2-D
promotion, anionpy does not. That is the **#67 family** — a fifth call site
where the rule belongs in the allocator rather than being hunted per caller.

### The 4 matrix items in #64 are suspect, not confirmed

The pass places `matrix.cumprod`, `.cumsum`, `.reshape` and `.repeat` in #64
"on a base-valid zero-extent combo." All four reproduce as divergent here —
but on a base that is itself divergent:

```
  base:    numpy (0,0)   anionpy (8,8)
  cumprod: numpy (0,0)   anionpy (8,8)
  cumsum:  numpy (0,0)   anionpy (8,8)
  reshape: numpy (0,0)   anionpy (8,8)
  repeat:  numpy (0,0)   anionpy (8,8)
```

Every method returns exactly the base's strides. Under this ticket's own rule
— *a divergence in the base invalidates the comparison of the output* — that
is the constructor defect propagating through four methods that touched it,
which is the `#66` lesson verbatim: **before attributing a defect to N
methods, call the constructor on its own and see whether N is really 1.**

I cannot close this either way. The pass reports these on a base-valid combo,
and it also discloses that `nested_list()` collapses any zero-leading-dim
shape to `(0,)`, so its labels may not name the construction it actually ran.
A claim I could not reproduce is not a claim I disproved — but I could not
reproduce a *valid* base for these four, and the burden is on the next pass
to paste one.

### Standing figure

**`#7 = 118`** — corroborated independently, unchanged across three passes.
**`#64 = 58 firm, 62 upper`** — 58 non-`matrix` items confirmed, plus 4
`matrix.*` items suspected of being one constructor defect (likely the
zero-extent divergence above, i.e. #67 rather than #64).
**`UNVERIFIABLE (#63) = 75`**, **NA = 34**, **CLEAN = 247 ± the same 4**.

Stopping the re-measure here. Three passes moved #64 from a stale 100 to a
bounded 58–62 and confirmed #7 at 118; a fourth pass would be arguing over
four items that most likely belong to a different ticket.

## FOURTH CORRECTION — 2026-08-08 (Monday): #7's 118 is NOT the reshape path, and my brief said it was

I briefed the #7 agent that the 118 items were caused by `NdArray::reshape`
missing numpy's identity-shape short-circuit, and I handed it this as the
downstream evidence:

```
base = zeros((4,1,2)).copy('F')      # strides identical on both sides: (8,32,32)
numpy   sort(axis=0).strides = (8, 32, 32)
anionpy sort(axis=0).strides = (16, 16, 8)
```

**That inference was wrong.** Two defects were visible on the same base and I
assumed the second was a consequence of the first. The agent found the reshape
bug was real, fixed it (`4f0b949`), and then reported — correctly — that the 118
were unaffected. Verified by me at `4f0b949`:

```
bases equal: True (8, 32, 32) (8, 32, 32)
reshape identity : (8, 32, 32)   (8, 32, 32)      <- FIXED
sort(axis=0)     : (8, 32, 32)   (16, 16, 8)      <- unchanged
abs              : (8, 32, 32)   (16, 16, 8)      <- unchanged
```

Ledger unchanged at 1335/3091 = 43.19% (measured off `/tmp/r7_v2_run2.json`),
suite failure set unchanged at 29 by name. Both are the expected result for a
fix to an edge case inside items already declared `exact` — and both are also
what you would see from a fix that did nothing, which is why the direct repro
above is the evidence and the ledger is not.

### What the 118 actually are

Characterized after the fact. Every base built the same way on both sides,
`zeros(shape).copy('F')`, bases asserted equal first:

| shape | base strides | numpy `abs` | anionpy `abs` | |
|---|---|---|---|---|
| (4,1,2) | (8,32,32) | (8,32,32) | (16,16,8) | FAIL |
| (1,4,2) | (8,8,32)  | (8,8,32)  | (64,8,32) | FAIL |
| (4,2,1) | (8,32,64) | (8,32,64) | (8,32,8)  | FAIL |
| (4,2)   | (8,32)    | (8,32)    | (8,32)    | ok |
| (3,4)   | (8,24)    | (8,24)    | (8,24)    | ok |
| (2,3,4) | (8,16,48) | (8,16,48) | (8,16,48) | ok |

**The discriminator is a length-1 axis, not F-order.** Every F-order base
without one preserves layout correctly. Every base with one loses it. `abs` is a
ufunc, so this is the ufunc/broadcast output-allocation path, distinct from
#64's non-ufunc `manip.rs`/`sort.rs`/`cast_to` cluster — though `sort` fails
identically, so the two paths may share a helper.

Note the (1,4,2) row: `(64,8,32)` is neither C nor F. Whatever picks the layout
is not merely defaulting to C — it is computing a wrong permutation. And note
that `multi_sorted_stride_perm` (`ionp-core/src/array.rs:1080`) already skips
size-1 axes at line 1099 exactly as numpy's `NpyIter` does, so either that
function is not on this path or something upstream of it is.

I am deliberately not diagnosing further from here. My record on this ticket
family is that reading three lines of an algorithm and naming the bug produces a
retraction — the negative-stride guess on #68, the stale-`hasattr` guess on #64.
The table above is a measurement; the cause is the next agent's to establish.

### The lesson, which is not the one I expected

Two defects sharing a reproduction case are not one defect. `reshape` and `sort`
both diverged on `zeros((4,1,2)).copy('F')` because a length-1 F-order base is
the shape that breaks *everything* in this family — that is what made it a good
repro and what made it a terrible piece of evidence for causation. A shape that
fails under many code paths tells you it is a hard shape. It does not tell you
those paths are the same path.

The agent's work stands and is a genuine fix — including a third bug it found
unprompted (`ravel_view_or_copy`'s synthesized-shape call). #7 stays OPEN at
118, re-scoped.
