# Triage of the 27 failing differential items — 2026-08-05

Measured against numpy 2.5.1, ionp at commit `02d8cda`, from three
byte-identical consecutive suite runs (`/tmp/det1.json`). Suite is
deterministic as of `02d8cda`; before that it was not, and any earlier
triage of these items should be distrusted for that reason alone.

The 27 items are NOT 27 problems. They are **8 causes**, and the two
largest clusters are both mis-described by their item names.

| # items | Cluster | Root cause | Tractable now? |
|--------:|---------|------------|----------------|
| 10 | pow family | complex `pow`, non-integral exponent path | YES — one algorithm |
| 5 | dtype introspection | ionp has no object dtype `dtype('O')` | architectural |
| 4 | sum/prod | float16 reduction ORDER on non-contiguous tuple-axis | YES |
| 3 | `isnat` | ionp has no datetime64/timedelta64 dtype | blocked on datetime |
| 1 | `ldexp` | `require_binary` misclassifies a binary ufunc | YES — needs core work |
| 1 | `hypot` | `.at` with repeated indices | YES — narrow |
| 1 | `expm1` | complex-only, ALREADY declined + documented | already declined |
| 1 | `__rtruediv__` | complex NaN bit patterns, ULP 0.0 | write-up queued |
| 1 | `scalar_comparison_unsupported_operand_gap` | needs `UFuncTypeError` | YES — small |

---

## Cluster 1 (10 items) — complex pow, and it is NOT the integer path

Items: `pow`, `power`, `float_power`, `emath.power`, `ndarray.__pow__`,
`ndarray.__rpow__`, `ndarray.__ipow__`, `alias/self/__ipow__`,
`alias/view/__ipow__`, `alias/nonaliased/__ipow__`.

The plan of record assumed this cluster was about *integer* powers. It is
not. **Integer exponents are already bit-exact.** Measured, 200 samples per
cell, both complex dtypes:

```
>>> base = (rng.normal(size=200) + 1j*rng.normal(size=200)).astype(np.complex64)
complex64    int exp 2      mismatching floats:    0/400   maxULP 0
complex64    int exp 3      mismatching floats:    0/400   maxULP 0
complex64    half exp 2.5   mismatching floats:  241/400   maxULP 9
complex64    complex exp    mismatching floats:  209/400   maxULP 94
complex128   int exp 2      mismatching floats:    0/400   maxULP 0
complex128   int exp 3      mismatching floats:    0/400   maxULP 0
complex128   half exp 2.5   mismatching floats:  184/400   maxULP 7
complex128   complex exp    mismatching floats:  116/400   maxULP 58
```

ionp already reproduces numpy's small-real-integral-exponent special case
(repeated multiplication). What diverges is the GENERAL branch, where numpy
does not compute `exp(b * log(a))` as a complex expression. `npy_cpow`
decomposes it into real primitives in a fixed order:

    r     = |a|,  theta = arg(a)
    rr    = pow(r, br) * exp(-bi * theta)
    angle = br * theta + bi * log(r)
    out   = rr * (cos(angle) + i*sin(angle))

Algebraically identical to `exp(b·log(a))`, numerically NOT — different
rounding at each step. Bit-exactness requires replicating the operation
ORDER, not the formula.

**Do not "fix" this by declaring a ULP tolerance.** The measured spread
reaches 94 ULP, which is not a rounding disagreement, it is a different
algorithm. A tolerance here would be hiding a mismatch, not declaring one.

## Cluster 2 (5 items) — no object dtype

`can_cast`, `promote_types`, `result_type`, `min_scalar_type`, `iinfo`.

Every failing case reduces to numpy producing or accepting `dtype('O')` or
the empty structured `dtype([])`:

- `can_cast([], ...)` -> numpy `False`; ionp raises `TypeError`.
- `promote_types([], ...)` -> numpy `DTypePromotionError`; ionp `TypeError`.
- `result_type([])` -> numpy `dtype([])`; ionp raises.
- `min_scalar_type(-2**63-1)` -> numpy `dtype('O')`; ionp `OverflowError`
  (whose message, to ionp's credit, already names exactly this gap).
- `iinfo(<any ndarray>)` -> numpy `ValueError("Invalid integer data type 'O'")`;
  ionp wrongly SUCCEEDS, returning that array's dtype info.

CORRECTION (same day, my first pass of this document was wrong on this
point, from reading the suite log instead of probing): the `iinfo` case is
NOT about object arrays. numpy's `iinfo(arr)` does not read `arr.dtype` at
all — it treats the ARRAY OBJECT as a dtype specifier, which resolves to
`dtype('O')`, hence the message. numpy therefore refuses EVERY ndarray
argument, `int64` included:

```
>>> np.iinfo(np.array([1, 2], dtype=np.int64))
ValueError: Invalid integer data type 'O'.
>>> ionp.iinfo(np.array([1, 2], dtype=np.int64))
iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)
```

ionp "helpfully" reads `.dtype` and answers. This is the only case in the
cluster where ionp is wrong in the DANGEROUS direction — a confident answer
where numpy refuses — and it is separable from object-dtype support:
refusing every ndarray argument does not require an object dtype to exist.

Note `iinfo(np.int64(5))` (a numpy SCALAR, not an array) is accepted by both
and must stay accepted; the fix is array-argument-specific, not a blanket
"reject anything that is not a dtype".

## Cluster 3 (4 items) — float16 reduction order

`ndarray.sum`, `ndarray.prod`, `prod`, `nanprod`. Every failure is
`float16/axis_tuple_noncontig`, 3 cases each, ULP 1–16.

REPRODUCED, but only with the EXACT call form. `axis_tuple_noncontig` is
`axis=(0, arr.ndim - 1)` — deliberately NON-ADJACENT axes, which cannot
coalesce into one contiguous run for ndim >= 3, forcing numpy's
sequential-outer / pairwise-inner fallback (the corpus comment at
`tests/differential/ndarray_attrs_cases.py:230` says so explicitly).

```
>>> a = rng.normal(size=(2,3,4,5,6)).astype(np.float16) * 3
>>> a.sum(axis=(0, 4))       # vs ionp
rank3 sum  axis=(0,2)  elems= 5  mismatch= 4
rank4 sum  axis=(0,3)  elems=20  mismatch= 4
rank5 sum  axis=(0,4)  elems=60  mismatch=18
rank4 prod axis=(0,3)  elems=20  mismatch= 3
rank5 prod axis=(0,4)  elems=60  mismatch=17
```

METHOD WARNING, recorded against myself: my first probe of this cluster
reported 0 mismatches across seven shapes and I nearly filed it as
"does not reproduce". It was operator error — I used `axis=(0,2)` on a
4-D array, where the corpus uses `(0, ndim-1)` = `(0,3)`. The path was
never entered. float32/float64 remain clean; only float16 is narrow enough
to expose the ordering difference.

This directly undercuts my confidence in the `hypot` non-reproduction below.
A failed reconstruction is evidence about MY PROBE at least as much as about
the code. Extract the corpus case; do not hand-build it.

## Cluster 4 (3 items) — isnat has nothing to be about

`isnat`, `order/ufunc/isnat`, `out_axis/ufunc/isnat` (16/16 and 12/12 total
failures). Not a broken ufunc — `ionp.array()` rejects the input dtype
outright: ionp supports bool, int8-64, uint8-64, float16/32/64,
complex64/128 and has no datetime64/timedelta64. `isnat` is definitionally
about NaT. These three items cannot move until datetime dtypes exist, and
should be tracked against that feature, not as three separate bugs.

## `ldexp` — a binary ufunc classified as not-binary

`ionp-py/src/lib.rs`'s `require_binary` groups `UfuncKind::Ldexp` and
`UfuncKind::FloatPower` with the UNARY kinds, so all four ufunc methods
raise `"requires a binary function"`. Both have `nin == 2`. They have their
own kinds only because their type resolution is special (`ldexp` is
float x int; `float_power` is always float64).

**The obvious fix is wrong.** numpy is not uniformly permissive here —
measured:

```
ldexp        outer      numpy OK        ionp TypeError(requires a binary function)
ldexp        reduce     numpy OK        ionp TypeError(...)
ldexp        accumulate numpy TypeError("the resolved dtypes are not compatible
                                         with ldexp.accumulate. Resolved ...")
ldexp        reduceat   numpy TypeError("... with ldexp.reduceat ...")
float_power  outer      numpy OK        ionp TypeError(...)
float_power  reduce     numpy OK        ionp TypeError(...)
float_power  accumulate numpy OK        ionp TypeError(...)
float_power  reduceat   numpy OK        ionp TypeError(...)
```

So: `float_power` must support all four. `ldexp` must support `outer` and
`reduce`, and must REJECT `accumulate`/`reduceat` — but with numpy's
dtype-resolution message, not with "requires a binary function". Allowing
all four for both kinds would replace one wrong answer with another.

## `hypot` (4 cases) — `.at` with repeated indices

All four failures are `at/repeated_index/...` across uint8, uint32, int32,
int64. Unbuffered accumulation semantics when the same index appears twice.

ROOT-CAUSED. `ufunc.at` is BUFFERED in ionp and UNBUFFERED in numpy.

My first hand-built probe (1-D array, scalar values) agreed with numpy and I
nearly filed this as "does not reproduce" — the same operator error as the
float16 cluster above. The corpus case is structurally different: `rep_values`
are ROWS of the array (`arr[[0,1]]`) applied to a 2-D/3-D target, and the
index is repeated so TWO folds hit the same slot.

numpy writes the result back into the TARGET dtype after EVERY fold, so the
second fold reads a truncated integer intermediate. ionp keeps the float
intermediate across both folds and casts once at the end, landing exactly one
higher wherever the discarded fraction mattered.

Both models reproduce their side bit-for-bit — this is not an inference:

```
>>> a    = np.array([[10,20,30,40,50]], dtype=np.uint8)
>>> vals = np.array([[3,4,5,6,7],[8,9,10,11,12]], dtype=np.uint8)
>>> np.hypot.at(a, [0,0], vals)
numpy .at               : [12 21 31 41 51]
ionp  .at               : [13 22 32 41 51]
model: UNBUFFERED       : [12 21 31 41 51]   <- truncate after EACH fold
model: BUFFERED (float) : [13 22 32 41 51]   <- truncate once at end
```

It is NOT a cast rounding-mode difference: on a SINGLE fold both truncate
toward zero and agree exactly. Only repeated indices expose it.

BLAST RADIUS, measured across 13 ufuncs with the same repeated-index probe:
only `hypot` and `nextafter` diverge. Everything computing in integer space
(`add`, `multiply`, `subtract`, `power`, `fmod`, `maximum`, ...) is
unaffected, because there is no float intermediate to lose. `nextafter` is
NOT in the failing 27 — its `.at` corpus evidently never hits a repeated
index — so fixing `hypot` alone would leave a known-identical bug live in a
currently-GREEN item. Fix the `.at` write-back, not `hypot`.

EXACT SITE — `ionp-core/src/ufunc.rs:8853`, `at_math_binary`'s promoted path:

```rust
if compute_dtype != dtype {
    let mut compute_target = target.cast_to(compute_dtype);
    at_math_binary(op, &mut compute_target, indices, values)?;  // ALL folds, compute dtype
    let result = compute_target.cast_to(dtype);                 // cast ONCE at the end
    target.buffer = result.buffer;
    return Ok(());
}
```

That is the buffered model written out literally: the whole target is
promoted, every fold runs in the wide dtype, and one cast happens at the
end. When `compute_dtype == dtype` there is no divergence, which is why only
integer-target/float-compute ufuncs show it.

IMPLEMENTATION NOTE for whoever fixes this: do NOT fix it by casting the
whole array back down once per applied index — that is O(applied_indices)
full-array casts and would be a serious performance regression on the
common path. The semantics numpy actually has are per-ELEMENT: read the
target element (target dtype), promote it, fold, demote, store. Each fold
then reads a value already round-tripped through the target dtype, which is
exactly the unbuffered model, at no extra allocation.

`at_binary` (line 8499) CHECKED and CLEAN: it never promotes. It casts
`values` DOWN to the target dtype and folds in target dtype directly, so
every fold already reads a target-dtype value — the unbuffered model by
construction. The defect is confined to `at_math_binary`'s promoted path.
`at_math_unary` (8724) and `at_unary_pure` (8807) CHECKED and CLEAN too —
and they are more than clean, they are the FIX TEMPLATE. Both compute one
element at a time, then:

    let mut computed = math_unary_op(op, &elem)?;   // may widen
    if computed.dtype() != dtype { computed = computed.cast_to(dtype); }
    write_scalar_in_place(buffer, pu, &computed);   // demoted, per element

That is precisely numpy's unbuffered model: read element, promote, fold,
demote, store, repeat. So the correct shape for the `at_math_binary` fix
already exists in this same file, twice. It does not need inventing, and
it does NOT need the O(n) whole-array cast per applied index that
`ed49f13` warned against — that warning stands, and the unary paths show
why it was never necessary.

One more thing found while checking, worth recording because it will
mislead the next reader: `at_math_binary`'s own doc comment (8845) says it
mirrors `at_binary` "plus its 'compute in the promoted dtype, cast back'
fallback." `at_binary` has NO such fallback — measured above. The comment
credits the defect to a function that does not contain it, which is how a
reader skimming for the bug gets sent to the wrong line. Fix the comment
in the same commit as the code.

## `scalar_comparison_unsupported_operand_gap` (1 case)

`int8` vs `str` comparison: numpy raises `UFuncTypeError`, ionp raises plain
`TypeError`. Needs a `UFuncTypeError` class. Note this must be ionp's OWN
class subclassing numpy's when numpy is present — the same pattern already
settled for `LinAlgError`/`AxisError` on 2026-08-05. Do not import numpy's.

---

## Ordering implied by this triage

1. `ldexp`/`float_power` `require_binary` — spec above is exact, but this is
   NOT a one-line `match` arm: `AnyBinaryOp` has only `Plain` and `Math`
   variants, so `FloatPower`/`Ldexp` are not representable at all and
   `ionp-core` needs matching reduce/accumulate/outer/reduceat support
   before the guard can be relaxed. Small-ish, not trivial.
2. `iinfo` over-acceptance — separable from object dtype, wrong direction.
3. `hypot.at` repeated index — narrow.
4. `scalar_comparison` `UFuncTypeError` — small, pattern already established.
5. float16 reduction order — 4 items, one cause.
6. complex pow general branch — 10 items, one algorithm, the big win.
7. object dtype — architectural, 5 items (4 after `iinfo` is separated).
8. datetime64/timedelta64 — architectural, 3 items.

Items 1-4 are ~4 items of coverage for small, well-specified work. Item 6 is
the single highest-yield change available anywhere in the suite.

---

## CORRECTION 2026-08-05 — "10 items, one cause" was too strong

An independent measurement pass (read-only, no build) re-derived the
cluster by calling `harness.run_case()` over every case in
`run.build_cases(spec)` for each of the 10 named items — the full set, not
`run.py`'s 8-per-item truncated sample, and not inferred from item labels.
Its count disagrees with mine and I accept the correction.

| item | failures | fixed by general branch alone | remainder cause |
|---|---|---|---|
| `ndarray.__rpow__` | 182 | 182 — fully green | — |
| `alias/self/__ipow__` | 1 | 1 — fully green | — |
| `alias/view/__ipow__` | 1 | 1 — fully green | — |
| `alias/nonaliased/__ipow__` | 2 | 2 — fully green | — |
| `emath.power` | 6 | ~3 | zero-base special case |
| `power` | 162 | 157 | empty-array negexp check |
| `pow` | 162 | 157 | same (alias of `power`) |
| `ndarray.__pow__` | 136 | 124 | empty-array negexp check |
| `ndarray.__ipow__` | 101 | 92 | reduce-family negexp check |
| `float_power` | 69 | **2** | `.at()` unimplemented + `require_binary` |

**Corrected: 4 items go fully green from the general-branch fix, 1 more
partially. The other 5 stay failing without three further independent
fixes.** `float_power` is the starkest — 67 of its 69 failures have
nothing to do with complex arithmetic, and my summary table folded it
wholly into this cluster while my own Ordering section listed it as
separate work. That was an internal contradiction in this document.

### The general-branch fix is NOT approved, and must not be written from a plausible transcription

This is now the **second** falsified attempt at replacing that branch.

The first is recorded in `ufunc.rs:2884-2903`: a `c_exp(y*c_log(x))`
replacement was tried and **measured worse** — 14 failing became 18 — and
introduced a new signed-zero regression class on zero-base inputs.

The second is the transcription proposed by this measurement pass. Its own
author measured it against numpy over N=2000 seeded complex inputs and got
**1269/2000 mismatches, maxULP 2469** — far worse than the 10–99 ULP the
shipping `num_complex::powc` produces — and could not explain why. Credit
where due: it reported that against its own recommendation rather than
burying it.

It would also have caused a concrete regression. Its proposal returns
`C128::new(0.0, 0.0)` for zero base with positive real exponent. Measured
live today:

```
>>> np.power(np.array(0j), np.array(3+0j))    # numpy
0j
>>> ionp.power(...)                            # ionp — ALREADY CORRECT
0j
```

That case passes today via `Complex::powc`'s own zero-base handling, and
is exactly the signed-zero class the *first* attempt broke. A hardcoded
`+0.0+0.0j` reintroduces the reverted bug.

**Verdict: the general branch stays as-is.** Closing it needs a
from-scratch Annex-G-correct `cpow` validated against numpy's actual
`npy_math_complex.c.src` line by line — which is not in the installed
wheel and must be pulled from an sdist or git. Not a transcription from a
formula in a task brief. Two attempts have now died at exactly that step.

### Three narrow fixes that ARE safe — all reproduced by me, first-hand

Independent of the general branch and of each other:

**(a) zero base, non-integer negative exponent**
```
>>> np.power(np.array(0j), np.array(-0.5+0j))     ->  (nan+nanj)
>>> ionp.power(...)                                ->  (inf+nanj)
```
Narrow: must fix `inf+nanj` without disturbing the `0**(3+0j)` signed-zero
case above, which is already right.

**(b) empty-array negative-integer-exponent validation** —
`ufunc.rs:5199-5222` validates unconditionally; numpy short-circuits when
there is nothing to compute.
```
>>> np.power(np.array([], dtype=np.int32), -7)    ->  array([], dtype=int32)
>>> ionp.power(...)  ->  ValueError: Integers to negative integer powers are not allowed.
```

**(c) length-1 reduce negative-base validation** —
`check_int_pow_no_negative_self`, `ufunc.rs:7829-7851`. numpy performs no
fold at all on a length-1 reduce, so it never validates.
```
>>> np.power.reduce(np.array([-3], dtype=np.int64))   ->  -3
>>> ionp.power.reduce(...)  ->  ValueError: Integers to negative integer powers are not allowed.
```

**(b) and (c) are the two sites I already flagged in this document as
raw-buffer-scan sites worth auditing** — `buf.iter().any(|&e| e < 0)` and
`check_int_pow_no_negative_self`. They are the same defect class as the
zero-size OOB read fixed in `02d8cda`: validation that walks data without
first asking whether the operation has any data to walk. Convergent
evidence from two independent directions, which is why I rate these three
high-confidence and the general branch untouchable.
