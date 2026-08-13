# Task #62 — Empty/Order Audit: exhaustive layout sweep of declared-exact items

MEASUREMENT ONLY. No shipped source touched. This is the exhaustive follow-up
to `dfdfc99`'s illustrative sample (families A/B/C/D over `sort`, `where`,
`clip`, `*_like`, `reshape`, `expand_dims`, ...). That sample is confirmed
here — every one of its named cases reproduces exactly — and the sweep goes
on to cover the rest of the declared-exact ledger.

## Headline number

Of the **1329** items currently declared `exact` (via
`tools/coverage.py --tests /tmp/r_verify6.json --list exact`):

| bucket | count | meaning |
|---|---:|---|
| Declared exact, total | 1329 | the denominator |
| NA — no array-shaped output | 74 | `dtype.*` (metadata), `testing.*` (assert/bool, nothing to lay out) |
| UNMEASURED | 830 | see scope breakdown below — **not** counted as passing |
| MEASURED, scalar-output only | 130 | reduces to a 0-d scalar on every combo tried; shape/strides comparison is vacuous (`()`≡`()` always) — real but weak |
| **MEASURED, real array output** | **295** | genuinely exercises the strides/order contract |
| — of those, **CLEAN** | **77** | order preserved across every combo tried |
| — of those, **FALSE** | **218** | diverges from numpy on layout, value, or both on at least one combo |

**The number that matters: of the 295 declared-exact items this audit could
actually drive through an order/empty-sensitive comparison, 218 — 73.9% —
are false.** That is not a tail. That is the center of mass of the
measurable surface. The other 830 items are UNMEASURED, not clean — an
unmeasured item is not evidence of anything, and given a 74% false rate on
everything that *could* be checked, there is no basis to assume the
unmeasured 830 skew cleaner.

This was re-run twice after the instrument itself turned out to be wrong the
first time (see **Instrument validity**, below) — the 73.9% figure is from
the corrected run.

## Method

`/private/tmp/audit_harness.py` + `audit_ops.py` + `audit_main.py` (probe
scripts, not shipped, kept outside the repo per the task's constraints).

For each declared-exact item:

1. **Resolve** the item to a callable/attribute on both `numpy` and
   `anionpy` (top-level function, `ndarray.<method>`, `<submodule>.<func>`,
   `ma.MaskedArray.<method>`, `matrix.<method>`).
2. **Construct matching bases** on both sides using only `array(nested_python_list, dtype=..., order=...)`
   plus `.T` (transpose view) and `[::2]` (non-contiguous slice) — no
   `reshape`, `arange`, or `zeros`/`ones` in the construction path, because
   those are themselves declared items under test and using them to build
   inputs would measure the constructor instead of the operation (this was
   an actual mistake caught in the previous pass, see below). **The two
   bases' `.shape`/`.strides`/flags are asserted equal before any operation
   runs** — every combo below passed that check with zero exceptions.
3. Try a cascade of generic call signatures (`f(a)`, `f(a,a)`, `f([a,a])`,
   `f(a,axis=0)`, `f(a,a.shape)`, `f(a,a.dtype)`, ...) against a probe case;
   the first one that succeeds is applied uniformly across the full combo
   sweep. Two items (`roll`, `tile`) got an explicit override because the
   generic cascade landed on a technically-legal but degenerate call
   (`roll(a, shift=a)`, `tile(a, reps=0)`) that silently passed and would
   have under-reported divergence on exactly the two items the prior sample
   named as buggy — caught by eye, not by the instrument, which is itself a
   note for future passes.
4. Compare `.shape`, `.strides`, `(C_CONTIGUOUS, F_CONTIGUOUS)`, and values
   (`np.allclose`/`array_equal`, skipped for `empty`/`empty_like` since
   uninitialized memory has no defined value) across:
   - **layout**: C, F, a transposed view, a non-contiguous slice (`a[::2]`),
     a length-1-axis array (both C and F), 0-d
   - **extent**: nonempty, plus `(0,)`, `(0,3)`, `(3,0)`, `(2,0,4)`, `(0,0)`
     (each in both C and F)
   - **dtype**: `float64`, `float32`

That's 7 nonempty + 10 empty layout combos × 2 dtypes = up to 34 cases per
item; fewer where a combo doesn't apply to that item's required shape
(e.g. 0-d has no axis for `axis=0`).

## Instrument validity (why the first pass was thrown out)

The first run of this sweep reported 104 "measured, mostly clean"
`ma.MaskedArray.*` items and 51 "measured" `matrix.*` items. Both numbers
were **wrong in a way the task's own warning predicted**: I had wired
`ma.MaskedArray.<method>` and `matrix.<method>` items to run against a
**plain `ndarray` base**, not an actual `MaskedArray`/`matrix` instance.
Because `MaskedArray` and `matrix` share many method names with `ndarray`
(`reshape`, `sum`, `copy`, `transpose`, ...), the calls "succeeded" — but
they were testing `ndarray.reshape` a second time under a different label,
not the declared item. This is the same class of mistake the task
description called out for its own first pass (`asfortranarray` vs.
`.copy('F')` disagreeing on 0-d): the bases weren't of the same type, so the
result measured construction, not the operation.

Fixed by building real `ma.masked_array(...)` / `matrix(...)` bases per
item-prefix. That fix immediately surfaced two structural findings (below)
that the invalid first pass had been silently papering over.

## Families found

### A — order not preserved (the dominant family, 212 of 218 false items)

General case: numpy returns a result carrying the input's layout (F stays
F, a transpose stays transposed-strided); anionpy recomputes C-order
regardless. Confirmed on everything the prior sample named (`sort`, `clip`,
`roll`, `tile`, `add`, `copy`/`np.copy`, `*_like`) plus ~200 more —
essentially every ufunc, every stacking/broadcast function, every dtype
cast, most `ndarray` dunders (`__add__`, `__eq__`, `__neg__`, ...), and the
`matrix.*` analogues of all of them.

Concrete case (`sort`, F-order `(2,3)` float64 — reproduces the task's own
worked example exactly):
```
numpy:  shape=(2, 3) strides=(8, 16)  C=False F=True
anionpy: shape=(2, 3) strides=(24, 8) C=True  F=False
```
Concrete case (`roll`, same input, `roll(a, 1, axis=0)`):
```
numpy:  shape=(2, 3) strides=(8, 16)  C=False F=True
anionpy: shape=(2, 3) strides=(24, 8) C=True  F=False
```
`tile(a, 1)` — identical numbers, identical signature.

### B — length-1-axis stride miscomputation on `*_like`/`copy` (subset of the A-tagged items above)

Not mechanically split out by the classifier (it collapses into the "A"
bucket since both share the root cause: anionpy doesn't propagate input
strides), but every one of `empty_like`, `zeros_like`, `ones_like`,
`full_like`, `copy` reproduces the exact family-B signature on the F-order
length-1-axis combo:
```
input:   F-order (4,1,2) — numpy strides=(8,32,32)
numpy result:   shape=(4,1,2) strides=(8,32,32) C=False F=True
anionpy result: shape=(4,1,2) strides=(8,64,32) C=False F=True
```
(Flags happen to agree here — both call it F-contiguous — but the actual
byte stride on the size-1 axis is wrong: 64 where it should be 32. This is
the sharpest version of family A: layout metadata says "correct," the
number underneath isn't.)

### C — identity reshape (`reshape`, `ndarray.reshape`, `matrix.reshape`)

`a.reshape(a.shape)`: numpy returns a view carrying the input's strides;
anionpy recomputes C-order. 22/34 combos diverge for `reshape` and
`ndarray.reshape`; 12/26 for `matrix.reshape` (fewer applicable combos —
matrix is always ≥2-D). Same F-order length-1-axis example as above
reproduces here too (`(8,32,32)` vs `(8,64,32)`).

### D — `expand_dims` inserts a 0-stride (1 item, exactly as previously found)

Nonempty-input case, `(2,3)` C-order:
```
numpy:   shape=(1,2,3) strides=(48,24,8) C=True F=False
anionpy: shape=(1,2,3) strides=( 0,24,8) C=True F=False
```
14/34 combos diverge. No other item in the 1329-item ledger reproduces this
signature — it is specific to `expand_dims`.

### E — new: `matrix` transpose/inverse family lies about its own shape through the array protocol

Not named in the prior sample. Found on `matrix.T`, `.H`, `.mT`,
`.transpose`, `.getT`, `.getH`, `.swapaxes`, `.I`, `.getI` (9 items).

Two distinct symptoms, both on the *same* operations:

- **Nonempty input: `np.asarray()` on the anionpy result raises.**
  ```
  >>> b = ionp.matrix([[1,2,3],[4,5,6]], dtype='float64')
  >>> np.asarray(b.T)
  ValueError: setting an array element with a sequence. The requested array
  would exceed the maximum number of dimension of 64.
  ```
  The object's own `.shape`/`repr()` are fine (`matrix([[1.0, 4.0], ...])`)
  — only the array-protocol conversion explodes, as if the buffer/`__array__`
  path recurses into itself.

- **Empty input: `np.asarray()` succeeds but silently drops a dimension.**
  For an empty `(1,0)`-shaped matrix, `b.T.shape` (the object's own
  attribute) correctly reports `(0, 1)`; `np.asarray(b.T).shape` reports
  `(0,)`. The object disagrees with its own array-protocol view of itself.

This is a distinct defect class from A–D: A–D are all "the shape/strides
metadata is self-consistent but wrong (wrong stride, wrong flag)." E is
"the object's own metadata and what you get by actually materializing it
through the standard conversion path disagree with *each other*." It only
showed up because fixing the instrument-validity bug (above) forced real
`matrix` construction instead of accidentally testing `ndarray` a second
time.

16 of 24 measurable combos for `matrix.I`/`matrix.getI` register as `VALUE`
(numeric mismatch) rather than a raised exception specifically because
`values_equal()` catches the `np.asarray` exception internally and treats it
as "incomparable" (`None`, not counted as false) on the nonempty combos, but
the empty combos convert without raising and the resulting values compare
unequal because the shapes silently disagree post-conversion. **This means
the reported 218-false / 73.9% figure is an undercount for these 9 items**:
the nonempty crash cases are not in the diverge tally at all, only the empty
silently-wrong ones are.

## No SHAPE or ERROR-class output divergence found elsewhere

Across all ~9,900 individual (item × combo) comparisons that reached the
classifier, zero registered as a raw output-shape mismatch outside the
matrix-E cases above, and none registered as "both sides ran, results
incomparable" outside `matrix.I`/`getI`. The `(0,).reshape`,
`nonzero`/`where` `(0,)` vs `(8,)`, `meshgrid`, `atleast_2d`-of-empty loose
ends named in the prior sample are all on items either not currently
declared exact under these exact names, or filtered into the "ran, produced
no comparable array result" UNMEASURED bucket by this harness — not
re-confirmed or refuted here; flagging as still open rather than guessing.

## Scope / UNMEASURED breakdown (830 items — none counted as clean)

| reason | count |
|---|---:|
| Explicitly scoped out of the generic harness: `polynomial.*` (448, all six basis classes + module-level poly funcs — always 1-D coefficient arrays by construction, order concept doesn't apply the way it does to `ndarray`), `random.*` (well, mostly — see below), `finfo.*`/`iinfo.*`/`poly1d.*`/`recarray.*`/`memmap.*` | 448 |
| Generic call-signature cascade found no working invocation on the probe (needs args this harness can't derive generically: `linalg.{cholesky,det,eig,eigh,inv,solve,...}` need square/PD input; bitwise/shift dunders need integer dtype, probe used float; most `char.*`/`strings.*` need string-typed arrays, not implemented in this pass) | 233 |
| Ran successfully but never produced an array-shaped result to compare (attributes like `.itemsize`/`.nbytes`/`.dtype`/`.ndim`, `__len__`, `__bool__`, `tobytes`/`tolist` — legitimately have no strides concept) | 144 |
| Item did not resolve on `anionpy` via this harness's resolution logic | 5 |

None of these 830 are "probably fine." They are exactly what the task
called for: **not tested, therefore not counted as passing.** The
`char`/`strings` (82 items) and `linalg` square-matrix (12 items) buckets in
particular are plausible next passes if a fuller number is wanted — they
were excluded for time, not because they're expected to be clean; if
anything the 74% failure rate on the measured core is reason to expect
similar or worse there, not better.

## Items that came through CLEAN (77 of 295 real-array-output items)

`append, argsort, asanyarray, asarray, asarray_chkfinite, ascontiguousarray,
atleast_1d, column_stack, copyto, cumprod, cumsum, cumulative_prod,
cumulative_sum, delete, diagflat, dstack, ediff1d, extract, fill_diagonal,
flip, fliplr, flipud, indices, intersect1d, iscomplex, kron, linalg.cond,
linalg.matrix_norm, linalg.matrix_rank, linalg.matrix_transpose,
linalg.pinv, linalg.svdvals, ma.MaskedArray.compressed,
ma.MaskedArray.filled, ma.compressed, ma.filled, ma.getdata,
ma.getmaskarray, ma.mask_or, matrix.A1, matrix.argsort, matrix.copy,
matrix.flatten, matrix.getA1, matrix.ravel, matrix.trace, matrix_transpose,
nancumprod, nancumsum, ndarray.__ifloordiv__, ndarray.__imod__,
ndarray.__setitem__, ndarray.argsort, ndarray.conj, ndarray.conjugate,
ndarray.copy, ndarray.cumprod, ndarray.cumsum, ndarray.flatten, ndarray.mT,
ndarray.ravel, ndarray.sort, ndarray.squeeze, ndarray.swapaxes,
ndarray.trace, ndarray.transpose, outer, permute_dims, ravel,
real_if_close, resize, setdiff1d, setxor1d, squeeze, transpose, union1d,
unique`

Caveat worth flagging rather than burying: a handful of these are clean by
*contract*, not by anionpy correctly replicating a hard case — e.g.
`append` is documented to always flatten and return C-order regardless of
input layout, so there is no order-preservation behavior to get wrong.
`ndarray.copy()` defaults to `order='C'` (unlike `np.copy`, whose default is
`order='K'` — which is exactly why plain `copy`/`np.copy` is in the FALSE
list and `ndarray.copy` is in the CLEAN list; this is not a contradiction,
it's numpy's own two different defaults, both correctly measured). The
`ma.*`/`ma.MaskedArray.*` clean items are the ones whose *anionpy* return
type happens to be a plain `ndarray` (which has `.strides`) rather than a
`MaskedArray` (which, see below, has neither `.strides` nor `.flags` at
all) — narrow, but genuinely clean on what they returned.

## Structural finding: `anionpy.ma.MaskedArray` has no `.strides` or `.flags` at all

Not a divergence, and not one of A–E — a missing-attribute finding that
makes the whole `ma.MaskedArray.*` domain (65 declared-exact items) and
every `ma.*` module function that returns a `MaskedArray` (most of the
other 136 `ma.` items) structurally unable to be checked against the "Strides
ARE part of the contract" standing decision, because the object doesn't
expose one:

```
>>> a = ionp.ma.masked_array([[1,2,3],[4,5,6]], dtype='float64')
>>> a.strides
AttributeError: 'MaskedArray' object has no attribute 'strides'
>>> a.flags
AttributeError: 'MaskedArray' object has no attribute 'flags'
```
(Confirmed this is not the numpy-side contract too — `np.ma.masked_array(...)`
has both, being a genuine `ndarray` subclass.)

Consequence for this audit: most `ma.*`/`ma.MaskedArray.*` operations landed
in the "ran, no comparable output" UNMEASURED bucket, not because the
harness couldn't drive them, but because their outputs are categorically
inapplicable to a layout check. This should be read as its own line item,
separate from "218 false out of 295": **201 declared-exact items (136
`ma.*` + 65 `ma.MaskedArray.*`) are declared against a contract
(order/strides) that the returned object cannot express.** Whether that's a
declaration problem or a missing-attribute problem is a judgment call this
report is explicitly not making — flagging it, not fixing it.

## Decision-relevant summary

- **73.9% false** (218/295) on every declared-exact item this audit could
  drive through a real (non-scalar) array-layout comparison. That is not
  "one family, mass-fixable in one pass" — family A alone (212 items) shares
  one root cause (anionpy does not propagate input strides/order through
  most operations) and looks mass-fixable *if* the fix is architectural
  (make order propagation the default path rather than a per-op patch).
  Families C, D, and E are narrower and structurally distinct — C is
  specific to identity-shape reshape, D is specific to `expand_dims`'s
  stride-insertion logic, E is specific to `matrix`'s array-protocol
  conversion — and would need their own fixes even after an A-root-cause
  fix landed.
- 830 items are unmeasured, not clean. Given the 74% rate on what could be
  checked, treating "unmeasured" as "presumably fine" would be the same
  mistake the corpus blind spot already made twice this week.
- 201 `ma.*` items sit on a structural gap (no `.strides`/`.flags`) that a
  stride-propagation fix to `ndarray` will not touch by itself.
- This is a revocation-scale finding, not a patch-scale one: 218 confirmed
  false plus 201 structurally unverifiable against the stated contract, out
  of 1329 declared exact — whatever "exact" is meant to certify, it
  currently certifies it for well under a third of the ledger with any
  confidence, and the confirmed-false rate on the checkable third is 74%.

## Reproduction

Probe scripts (not part of the shipped tree): `/private/tmp/audit_harness.py`,
`/private/tmp/audit_ops.py`, `/private/tmp/audit_main.py`. Raw per-item
results: `/private/tmp/audit_results.json`. Item ledger this run swept:
`/private/tmp/exact_items.tsv`, produced by
`./.venv/bin/python tools/coverage.py --tests /tmp/r_verify6.json --list exact`.
None of these are checked into the repo; this document is the only new file.

---

## CORRECTION — 2026-08-08: Family A conflated ticket #7 with genuine order loss

**The section above is left standing, uncorrected, as originally written and
committed.** This section does not replace it — it replaces the *conclusion*
drawn from it. Where the two disagree, this section is the current finding.

### What was wrong

The original "Family A (212 items)" bucket was a single catch-all for every
non-C-baseline stride/flag divergence. A coordinator cross-check at `dfdfc99`
(bases asserted equal on both sides, same methodology) showed that on plain
F-order `(2,3)`, a transposed view, and a non-contiguous slice, most core
ufuncs (`add`, `sqrt`, `negative`, `equal`, `clip`, `abs`/`maximum`/`round`,
`copy`, `zeros_like`, `cumsum`) agree with numpy exactly. They only diverge
on the length-1-axis case, F-order `(4,1,2)` — which is **ticket #7**
("order='K' stride divergence on size-1-axis-with-meaningful-stride
operands"), open since Phase 3, not a newly-discovered defect. Family A as
originally reported collapsed that one already-known root cause together
with whatever independent order-loss bugs exist on plain F/transposed/sliced
inputs, so "212 items, one family" overstated both the novelty and the
blast radius of what the sweep actually found.

### Re-run: 5-bucket partition

Same harness (`/private/tmp/audit_main.py`), same 1329-item ledger, same
17-combo x 2-dtype cross product, rerun with each diverging combo tagged by
which layout produced it:

| bucket | meaning | attribution |
|---|---|---|
| 1 | input has a length-1 axis, or is 0-d | ticket #7 |
| 2 | F-order, no length-1 axis | genuine order loss |
| 3 | transposed view, no length-1 axis | genuine order loss |
| 4 | non-contiguous slice, no length-1 axis | genuine order loss |
| 5 | zero extent | ticket #61 family |
| 0 | plain C-contiguous, nonempty, no length-1 axis (baseline — should never diverge) | new/severe if present |

Of the 218 items previously classed as false (family A/B/C/D combined; E
tracked separately below), every one keeps at least one diverging bucket in
this rerun — the divergence count itself did not change, only its
attribution.

**Headline numbers:**

- **118 items are false ONLY in bucket 1** (length-1-axis/0-d). These
  attribute cleanly to ticket #7. If #7 is fixed and nothing else changes,
  these 118 are expected to go clean.
- **100 items are false in at least one of buckets {0, 2, 3, 4, 5}** — i.e.
  they have a diverging combo that is *not* a length-1-axis case, so they
  would **still be false after #7 is fixed**.

That 100 is longer than the coordinator's expected short list (`sort`,
`astype`, `where`, `concatenate`, `stack`, `roll`, `tile`). Two of those
names are not addressable from this ledger at all: `astype` and
`concatenate`/`stack` (bare) are **not present** in the 1329 declared-exact
items this harness swept — `grep` against `/private/tmp/exact_items.tsv`
confirms it. `where` *is* declared exact but this harness's call-signature
cascade could not drive it to a comparable array result on any combo
(status `UNMEASURED`, not counted in the 100). `sort`, `roll`, and `tile`
are confirmed and are in the 100.

Per-bucket breakdown of the 218 (an item can appear in more than one
bucket's tally):

| bucket | items with >=1 diverging combo in this bucket | items diverging ONLY in this bucket |
|---|---:|---:|
| 0 (C baseline) | 4 | 0 |
| 1 (#7) | 173 | 118 |
| 2 (F, no len-1) | 89 | 0 |
| 3 (transposed, no len-1) | 89 | 0 |
| 4 (slice, no len-1) | 11 | 1 |
| 5 (zero extent, #61) | 22 | 8 |

Every item that diverges in bucket 2 also diverges in bucket 3 (89 = 89,
confirmed pairwise) — F-order-loss and transposed-view-loss track together
in every case observed, which is consistent with both being symptoms of
the same "order not propagated" code path rather than two separate bugs.
The 100 survivors split as: 68 diverge in exactly {2,3} only, 11 in
{2,3,5}, 6 in {2,3,4}, 3 in {0,2,3,4}, 1 in {0,2,3,4,5}, 10 in {5} only,
1 in {4} only.

**Bucket 0 flag:** 4 items (`argwhere`, `expand_dims`, `linspace`,
`matrix.repeat`) diverge on plain C-contiguous nonempty input with *no*
order variation at all. That is not "order loss under F/transposed/sliced
input" — the coordinator's frame doesn't have a slot for it. `expand_dims`
divergence here is the previously-reported family D (spurious 0-stride on
the inserted axis, present even in the C-baseline case). `argwhere` and
`linspace` genuinely change output stride layout relative to numpy even
starting from a plain C input; `matrix.repeat` is part of family E (below).
These should not be filed under #7 or under "genuine order loss on
non-C-input" — they need their own look.

### Concrete examples the coordinator can re-run

Three of the 100 survivors, each shown with instrument-validated bases
(shapes agree before comparison) diverging on plain F-order `(2,3)` — no
length-1 axis anywhere in the combo, so not attributable to #7:

```
sort, F(2,3), float64:
  numpy stride  (8,16)   ionp stride  (24,8)   [values equal, order differs]

roll(a,1,axis=0), F(2,3), float64:
  numpy stride  (8,16)   ionp stride  (24,8)   [values equal, order differs]

float32(a) [dtype-cast constructor call], F(2,3), float64-built-then-cast:
  numpy stride  (4,8)    ionp stride  (12,4)   [values equal, order differs]
```

A wider and unexpected pattern in the 100: every dtype-cast constructor
swept (`bool_`, `byte`, `int8/16/32/64`, `uint*`, `float16/32/64`,
`complex64/128`, `cdouble`, `csingle`, `half`, `single`, `double`, `long`,
`short`, `intc`/`intp`/`int_`, `ulong`/`ushort`/`uintc`/`uintp` — roughly 30
items) loses order on plain F-order and transposed-view input, not just on
length-1-axis input. So does a cluster of `fft.*` items (`fftshift`,
`hfft`, `ifftshift`, `irfft`), `hstack`/`vstack`, `real`/`imag`,
`isclose`/`isreal`, `unwrap`, `sinc`, `i0`, `sort_complex`, `diff`,
`ndarray.fill`. This is a materially longer and more structural list than
"a few named ufuncs" — dtype casting in particular looks like it goes
through a different (order-blind) code path than the arithmetic ufuncs the
coordinator's own sample checked, and that path is not fixed by #7.

### Family E (`matrix` array-protocol) folded into the bucket count

The 9 `matrix` items from family E are present in the dirty set and are
folded into the bucket counts above (not double-counted, not omitted):

| item | buckets |
|---|---|
| `matrix.T` | 2, 3, 5 |
| `matrix.H` | 2, 3, 5 |
| `matrix.mT` | 2, 3, 5 |
| `matrix.transpose` | 2, 3, 5 |
| `matrix.getT` | 2, 3, 5 |
| `matrix.getH` | 2, 3, 5 |
| `matrix.swapaxes` | 2, 3, 5 |
| `matrix.I` | 5 |
| `matrix.getI` | 5 |

All 9 are in the "survives #7" list of 100 — none of them are length-1-axis
cases, so fixing #7 does nothing for this family. The original report's
undercount caveat still applies: `values_equal`'s internal exception catch
returns `None` (skip), not `False`, so the nonempty-input crash case
(`np.asarray()` raising "exceeds 64 dimensions") is invisible to the
diverge count and only the empty-input silent-wrong-shape case registers.
The true false rate for this family is higher than what these bucket counts
show.

While re-checking family E for this correction, 29 further `matrix.*`
methods surfaced in the dirty/survive set that were not called out
individually in the original report (`matrix.A`, `matrix.__abs__`,
`matrix.__add__`, arithmetic dunders, `matrix.clip`, `matrix.conj`,
`matrix.fill`, `matrix.repeat`, `matrix.reshape`, `matrix.sort`,
`matrix.squeeze`, and others) — all in buckets {2,3} or {2,3,5}, i.e. all
survive a #7 fix. These share family E's root cause (the `matrix` subclass
not propagating/reporting order correctly) rather than being 29 new
independent bugs, but they were not enumerated in the original E section
and are surfaced here for completeness.

### `ma.MaskedArray` structural blocker — restated as a separate line, not a bucket

Unchanged from the original report and **not** one of the five buckets
above: 201 declared-exact `ma.*`/`ma.MaskedArray.*` items cannot be checked
against the strides/flags contract at all, because `anionpy.ma.MaskedArray`
objects expose neither `.strides` nor `.flags`. This is a structural gap,
not an order-loss finding, and it is not folded into the 118/100 split or
into any percentage in this section — it sits outside the bucket
partition entirely, exactly as before.

### Revised decision framing

- **118 of the 218** previously-reported-false items attribute cleanly to
  ticket #7 and are plausible to resolve by fixing that one ticket. The
  coordinator's instinct — "fix #7 first, most of the 218 resolve" — is
  directionally correct for this subset: 118/218 = 54%.
- **100 of the 218** (46%, including all 9 Family E items plus ~29 further
  `matrix.*` methods surfaced above) have at least one diverging combo that
  is not a length-1-axis case, and will still be false after #7 lands.
  Roughly 30 of those 100 are dtype-cast constructors, a class the
  coordinator's own sample did not include and which appears to lose order
  through a different code path than the arithmetic ufuncs checked at
  `dfdfc99`.
- Net effect on the original headline: "73.9% false, consider mass
  revocation" was too coarse in blaming one root cause, but the corrected
  number is not small either — fixing #7 is necessary but leaves 100
  declared-exact items (including all of `matrix`'s transpose/inverse
  surface and the entire dtype-cast-constructor class) still false, on top
  of the 201 `ma.*` items that cannot even be checked. This is "fix #7 first,
  *then* re-audit," not "fix #7 and move on."

### Reproduction (unchanged)

Same probe scripts and same raw-results file as the original run:
`/private/tmp/audit_harness.py`, `/private/tmp/audit_ops.py`,
`/private/tmp/audit_main.py` (now with bucket tagging added),
`/private/tmp/audit_results.json`, `/private/tmp/exact_items.tsv`. None of
these are checked into the repo.

---

## SECOND CORRECTION — 2026-08-08: Family E was never a `matrix` *method* problem

Written by the coordinator after independently reproducing the family-E
symptom. The two sections above are left standing; this corrects their
attribution, not their observations. Family E's symptom was real and
reported accurately. Its cause was assigned to the wrong object.

### What the audit said

The original run named 9 items — `matrix.T`, `.H`, `.mT`, `.transpose`,
`.getT`, `.getH`, `.swapaxes`, `.I`, `.getI` — as raising

```
ValueError: setting an array element with a sequence. The requested array
would exceed the maximum number of dimension of 64.
```

under `np.asarray()`. The first correction widened this to "29 further
`matrix.*` methods sharing family E's root cause," for 38 in total.

### What is actually true

None of the 38 methods was at fault. The *constructor* was:

```
>>> np.asarray(anionpy.matrix([[1., 2.], [3., 4.]]))
ValueError: setting an array element with a sequence. The requested array
would exceed the maximum number of dimension of 64.
```

No transpose. No method call. The bare object would not convert.

Diagnosis, measured at `0566fca`:

```
type(a).__mro__          ['matrix', 'object']     # numpy's is matrix -> ndarray -> object
hasattr(a, '__array__')  False
a.shape                  (2, 2)                   correct
a.tolist()               [[1.0, 2.0], [3.0, 4.0]] correct
a[0]                     matrix([[1.0, 2.0]])  shape (1, 2)
a[0][0]                  matrix([[1.0, 2.0]])  shape (1, 2)   ... forever
```

`anionpy.matrix` is composition-based — it *holds* an `anionpy.ndarray`
rather than *being* one, because `anionpy.ndarray` is a PyO3 type and is
not subclassable from Python. So it is not in the MRO, and with no
`__array__`, numpy fell back to the sequence protocol. That protocol
never bottoms out here, and numpy walked it to the 64-dimension limit.

The infinite nesting is **numpy-faithful and must not be "fixed"**: real
`np.matrix` indexes exactly the same way. It survives there only because
it IS an ndarray subclass with a real buffer, so numpy reads the buffer
and never walks the sequence protocol at all. Composition is what removed
that escape hatch.

Fixed in `a2a1ebb` by adding `__array__` to `anionpy/matrix.py`,
delegating to `anionpy.ndarray.__array__`. Verified out of corpus across
four constructors x ten operations: all now match numpy.

### The residue, and why it matters to this document's thesis

After the fix, exactly two divergences remain anywhere under `matrix`,
and both are on the **base object's strides**:

```
matrix([[1., 2., 3.]])       (1,3)   numpy (8,8)   anionpy (24,8)   -> #7, length-1 axis
matrix(anionpy.zeros((0,1))) (0,1)   numpy (0,0)   anionpy (8,8)    -> #67, zero extent
```

That is independent confirmation of this document's own bucket split.
The whole `matrix` surface reduces to the two causes already ticketed;
nothing under it requires a third explanation.

### The lesson, which is this document's third instrument failure

The first pass tested plain `ndarray` bases for `ma.MaskedArray.*` and
`matrix.*` items and had to be thrown out. The second pass collapsed
ticket #7 into "family A" and had to be re-partitioned. This is the
third: a defect in a base class was reported as 38 defects in its
methods, because the methods were what the harness called.

The tell was there both times and was not read: **a uniform failure rate
across a whole class surface is a claim about the harness or the base
object, not about 38 independent implementations.** The repo already
states that a uniform failure rate across a cross-product is an
instrument bug. The same suspicion applies to a uniform failure rate
across a *class*. Before attributing a defect to N methods, call the
constructor on its own and see whether N is really 1.

### Consequence for the numbers above

The "100 of 218" figure includes all 9 family-E items and, per the first
correction, ~29 further `matrix.*` methods. Those ~38 are now fixed by a
single commit and should not be counted against #64. **Re-audit before
using the 100 figure to size #64.** The undercount caveat recorded for
family E in the original section is likewise discharged: `matrix` items
are now measurable, so a re-run will produce a real number for them for
the first time.
