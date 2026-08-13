# Shape-manipulation cluster — measured ground truth (2026-08-02)

Pre-verification pass, done READ-ONLY against a stable binary
(`ionp/_ionp.abi3.so` mtime 21:45:45 unchanged across both the agent's run and
my audit). Purpose: let a build agent start from measured fact instead of
repeating the investigation. Nothing here is a declaration — every item below
is still `absent` in the ledger.

**Timestamp caveat.** Everything here was measured against the binary built at
**21:45:45**. A concurrent build agent working on the comparison/predicate
cluster rebuilt at **22:25:53**, so these findings predate that binary. Its
scope (`setops.rs`, comparison ufuncs, predicates) does not overlap the shape
functions, so I expect the table to still hold — but *expect* is not
*measured*, and any GREEN item here must be re-probed against the current
binary before it is declared, not just wired up to cases. A measurement's
timestamp is part of the measurement.

Method: differential probe vs numpy 2.5.1 checking **values, dtype, shape,
strides, and view/copy contract** — not values alone. Operand parity (shape,
strides, bytes) asserted on both sides before any output was compared.
Fixtures built with `ionp.array(nested_list, dtype=, order=)`, deliberately
NOT via `reshape`/`asarray` (one is under test, the other is on the
known-broken list), so fixture construction cannot measure the item under
test. `shares_memory` was not used as an oracle — it is known broken for
interleaved-disjoint views; `may_share_memory` was used instead.

## GREEN — verified correct on values, dtype, shape AND strides

`reshape`, `squeeze`, `transpose`, `swapaxes`, `moveaxis`, `atleast_1d`,
`atleast_2d`, `broadcast_to`, `broadcast_arrays`, `flip`, `fliplr`, `flipud`,
`stack`, `concatenate`

These need **differential cases only — no Rust work.** I independently
re-checked `reshape`/`transpose`/`ravel` on three layouts the agent never
varied (F-order, stepped `x[::2]`, negative-stride `x[::-1]`): 9/9 stride
agreement.

## RED — resolves, measurably wrong

Two defect families, and they are the same root cause wearing two hats.

### Family 1 — inserted-axis stride for size-1 axes

numpy is not internally consistent here, and ionp is wrong in *opposite
directions* on the two items, which is why this looked like a contradictory
report until measured:

| call | numpy | ionp |
|---|---|---|
| `expand_dims(x, 0)` on (3,4) | `(96, 32, 8)` | `(0, 32, 8)` |
| `expand_dims(x, 1)` | `(32, 32, 8)` | `(32, 0, 8)` |
| `expand_dims(x, 2)` / `-1` | `(32, 8, 8)` | `(32, 8, 0)` |
| `atleast_3d` on (3,4) | `(32, 8, 0)` | `(32, 8, 8)` |
| `atleast_3d` on (3,) | `(0, 8, 0)` | `(24, 8, 8)` |

Not arbitrary: numpy's `expand_dims` routes through `reshape`, which assigns a
contiguity-consistent stride; `atleast_3d` routes through `newaxis`
broadcasting, which assigns `0`. Both are valid descriptions of a size-1 axis
and the contiguity flags agree either way — but the contract is *match numpy*,
so both are defects. `atleast_3d` on a 0-d input is correct (`(8,8,8)`).

### Family 2 — no general strided-view constructor

`ravel`, `rollaxis`, `rot90`, `diagonal`, `diag` (2-d→1-d branch) all
**materialise a copy where numpy returns a view**, and the four that need
non-trivial strides also get the strides wrong:

| call on (3,4) | numpy strides | ionp strides | numpy view? | ionp view? |
|---|---|---|---|---|
| `ravel` (C-contig) | `(8,)` | `(8,)` | yes | **no** |
| `rollaxis(x, 1)` | `(8, 32)` | `(24, 8)` | yes | **no** |
| `rot90(x, 1)` | `(-8, 32)` | `(24, 8)` | yes | **no** |
| `rot90(x, 2)` | `(-32, -8)` | `(32, 8)` | yes | **no** |
| `diagonal(x)` | `(40,)` | `(8,)` | yes | **no** |
| `diag(x)` 2-d→1-d | `(40,)` | `(8,)` | yes | **no** |

The stride column tells the story: numpy's answers need **negative** strides
(`rot90`), **permuted** strides (`rollaxis`), and a **stride that is the sum of
two axis strides** (`diagonal` → 32+8=40). ionp's are in every case the
strides of a freshly-allocated C-contiguous buffer. These functions are not
computing a wrong view; they are not constructing a view at all — they fall
back to copying, and the C-contiguous strides are a *symptom* of that, not an
independent bug.

So this is ONE fix, not five: a general view constructor taking arbitrary
strides + offset, then routing these five through it. `diag`'s 1-d→2-d branch
builds a genuinely new array and is already correct — leave it alone.

Note `ravel` is the cheap member: its strides are already right, only the
view/copy contract is wrong, and only on C-contiguous input (the F-order,
negative-stride and stepped cases correctly copy, matching numpy).

## MISSING — does not resolve at all

`permute_dims`, `matrix_transpose`, `concat`, `block`, `unstack`, `require`,
`asfortranarray`, `asanyarray`

The first three are near-free: `permute_dims` is an alias of `transpose`,
`concat` of `concatenate`, `matrix_transpose` a last-two-axes swap — and all
three of their targets are GREEN.

## Not determined — do not assume these are clean

- dtypes narrower than int64/float64/complex128/bool were not probed
  (no uint, int32, float32, float16).
- `.base` object identity chains were not checked, only may-share truth values.
- `reshape(order='F'/'A')` variants untested; default order only.
- `stack`/`concatenate` untested with >2 arrays, mixed dtypes needing
  promotion, or `out=`.
- `diag`/`diagonal` untested with a nonzero `offset`.
- Error-message parity unprobed except one incidental broadcast `ValueError`.

Anything in that list is unmeasured, which means for declaration purposes it
is unknown, not clean.

---

# ADDENDUM 2026-08-02 — the 14 GREEN re-measured, and the view/copy contract

Measured against binary mtime **22:33:46** (stable across the whole run, checked
at start and end). The original classification above was against 21:45:45 /
22:25:53 and is superseded for these items.

## The headline correction

A re-probe reported "10 of 14 now RED — ionp silently materializes a copy where
numpy returns a view." That direction is right, the diagnosis was one bucket
where there are three, and the difference decides the fix.

**`a.T` returns a correct view. `transpose(a)` does not.** Same operation, two
spellings, opposite answers. So the view machinery EXISTS and works — this is
not "ionp has no views yet."

Separating "real copy" from "metadata lie" needs the actual data pointer, not
`.flags`: an array that shares a buffer but reports `OWNDATA=True` is
indistinguishable from a copy via flags alone, and the two need completely
different fixes.

| item | shares source data pointer? | `OWNDATA` | verdict |
|---|---|---|---|
| `a.T` (property) | yes | False | **correct view** |
| `transpose()` (free fn) | **yes** | True | **METADATA LIE** — right bytes, wrong labels |
| `squeeze` | no | True | **REAL COPY** where numpy views |
| `swapaxes` | no | True | **REAL COPY** |
| `moveaxis` | no | True | **REAL COPY** |
| `reshape` (free fn) | no | True | **REAL COPY** |
| `ravel` | no | True | **REAL COPY** |
| `flip` / `fliplr` / `flipud` | **UNDETERMINED** | True | see below |
| `broadcast_to` / `broadcast_arrays` | **UNDETERMINED** | True | plus `WRITEABLE` divergence |

## Three distinct defects, not one

1. **`transpose()` — metadata lie.** The buffer is already shared; only
   `OWNDATA`/`.base` are wrong. Cheapest fix in the cluster. Note the free
   function is wrong while the property is right, which is the signature of a
   free-function wrapper that rebuilds the array object instead of forwarding.
   This is the same shape as the already-known "`ionp.reshape()` free function
   doesn't attach `.base`" — suspect the whole free-function layer.

2. **`squeeze`/`swapaxes`/`moveaxis`/`reshape`/`ravel` — real copies.** These
   need the arbitrary-stride view constructor already identified as the missing
   primitive behind the 5 RED copy-not-view items (`ravel`, `rollaxis`, `rot90`,
   `diagonal`, `diag`). Same primitive, now unlocking **ten** items, not five.
   That makes it decisively the highest-leverage single fix in this cluster.

3. **`broadcast_to`/`broadcast_arrays` — plus a writeability divergence.**
   numpy's broadcast result is a **read-only** view (`WRITEABLE=False`); ionp
   reports `WRITEABLE=True`. Independent of the view question and still wrong
   even once views land.

## What I could NOT determine, and why — read this before trusting the table

My pointer test compares the result's data pointer to the source's for
**equality**. A legitimate view does not have to start at the same address:
`flip` views begin at the far end of the buffer with negative strides, and
`broadcast_to` uses 0-strides. For those four items pointer-equality returns
False for **real numpy too**, so the instrument cannot tell a view from a copy
there. They are marked UNDETERMINED, not "ok". Resolving them needs an
address-RANGE overlap test against the source's byte extent.

This is worth stating plainly because the naive reading of my own output was
"flip and broadcast_to agree with numpy" — they do not; my instrument is simply
blind to them. A blind spot reported as a pass is exactly the failure mode that
put 16 false declarations in the ledger this morning.

Two further caveats on the re-probe that produced the RED list:
- Its `concatenate` failure was its own fixture bug (a `(0,4)` nested list
  collapsed to a 1-d empty before construction), not an ionp divergence.
- My first version of the pointer probe was itself broken — it compared an
  ndarray to a string, so every numpy row came back as a `ValueError` and the
  table read as total disagreement. Fixture bugs in the measuring instrument are
  now a recurring theme; diff the instrument before believing the measurement.

## Still GREEN, confirmed

`transpose` (values/strides only), `stack`, `concatenate`, and `reshape` on all
non-empty cases are correct on values, dtype, shape and strides. The view
contract is the *only* axis on which the ten flipped — every one of them has
correct bytes and correct strides. Newly found: `reshape` of a size-0 1-d array
gives numpy strides `(0,)` and ionp `(itemsize,)`.

Also newly found and cluster-wide, not per-item: `ionp.ndarray.flags` has no
`"ALIGNED"` key at all (`ValueError: unsupported flag key`).

## Open question this forces — FOR MOTHER, not to be settled by me

Is the **view/copy contract part of what ionp promises**? Every one of these ten
returns correct values with correct strides. If views are Phase 2/3 work by
design, these items are shippable now and the contract should say so explicitly.
If view-ness is part of the contract, ten items are blocked on one primitive.
This is the same question as "is output memory layout part of the contract?" and
it should be answered once, for both.

---

# CORRECTION 2026-08-02 — the addendum above is WRONG, and how

Measured against binary **23:39:31** (stable across the run).

## The addendum's instrument was invalid

The addendum classified items by comparing the result's data pointer to the
source's, obtained via `np.asarray(ionp_arr).__array_interface__`.

**`np.asarray(ionp_arr)` allocates a fresh copy on EVERY call.** Measured: five
simultaneously-held `asarray()` results sit at five distinct addresses. So the
addendum compared the addresses of two short-lived temporaries. Worse, five
*consecutive* calls reused one address three times, because each temporary was
freed before the next was allocated — which is precisely how that probe
produced confident "shares pointer: True" answers. **The three-way split in the
addendum was allocator reuse, not measurement.**

Do not use `np.asarray()` to obtain an ionp array's address. There is no
`__array_interface__` or ctypes pointer on ionp arrays at all.

## The valid instrument, validated before use

`ionp.may_share_memory(a, b)` is backed by `Arc::ptr_eq` buffer identity plus
`memory_extent()` (`ionp-core/src/array.rs`), which handles negative and zero
strides. Validated against controls where the answer is known independently:

| control | expected | got |
|---|---|---|
| `x` vs `x` | True | True |
| `x` vs `copy.copy(x)` | False | False |
| `x` vs `copy.deepcopy(x)` | False | False |
| `x` vs `x.T` (known correct view) | True | True |
| `x` vs independently-built equal array | False | False |
| `x` vs `ionp.copy(x)` | False | False |

**6/6.** Validate the oracle, then use it — in that order.

## Corrected classification

| form | shares buffer | `OWNDATA` | `.base` | verdict |
|---|---|---|---|---|
| `a.T` | yes | False | set | **correct view** |
| `.transpose()` method | yes | False | set | **correct view** |
| `.reshape()` method | yes | False | set | **correct view** |
| `transpose()` free | yes | **True** | None | **metadata lie** |
| `reshape()` free | yes | **True** | None | **metadata lie** |
| `squeeze()` | yes | **True** | None | **metadata lie** |
| `swapaxes()` | yes | **True** | None | **metadata lie** |
| `moveaxis()` | yes | **True** | None | **metadata lie** |
| `flip()` | yes | **True** | None | **metadata lie** |
| `broadcast_to()` | yes | **True** | None | **metadata lie** |
| `ravel()` | **no** | True | None | **real copy** |
| `ionp.copy()` (control) | no | True | None | real copy, correct |

## What this changes — the fix got much cheaper

The addendum said five items were real copies needing an **arbitrary-stride view
constructor**. That was wrong. **Only `ravel` is a real copy.** Every other item
already shares the source buffer AND already has correct strides — the bytes and
the layout are right, and only the `OWNDATA`/`.base` labels are wrong.

So this is not a missing primitive. It is **one systematic defect in the
free-function layer**: it rebuilds the array object and drops the base
reference, where the method forms forward it correctly. Nine items, one fix, no
new numeric code. `ravel` alone needs real view support.

`broadcast_to` additionally reports `WRITEABLE=True` where numpy's broadcast
view is read-only — independent of the labelling bug and still wrong after it.

## The lesson, recorded because it cost real work

I published the addendum's split after checking that my *first* probe was
broken (it compared an ndarray to a string). Fixing the visible bug made the
output look plausible, and I stopped there. **The instrument was still invalid
for a completely different reason, and plausible output is not validation.**

The addendum even marked four items UNDETERMINED for fear of an instrument blind
spot — while the same instrument was silently wrong about the items it *did*
report. Caution aimed at the wrong axis.

The rule that would have caught it, and which the corrected measurement follows:
**validate the oracle against controls with independently known answers before
trusting a single one of its verdicts.** `x` vs `copy.copy(x)` must read False;
`x` vs a known view must read True. If those don't come out right, nothing
downstream means anything.
