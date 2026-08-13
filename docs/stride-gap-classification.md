# Stride-Gap Classification: the 253 `check_strides=True` diagnostic failures

## What was measured, and how

Starting state verified before touching anything: `.venv/bin/python tests/differential/run.py
--out /tmp/verify_ledger.json` + `tools/coverage.py --tests /tmp/verify_ledger.json` reported
458/1180 exact, failing 6, phantom 0, untested 0 — matched the stated baseline exactly.

`ItemSpec.check_strides` (registry.py:869) defaults to `False` on 433 of 434 items; flipping it
to `True` and re-running the full corpus reports 253 of those items now failing. `registry.py`
was **never edited on disk**. Instead, `run.py` was imported as a module (which triggers all its
corpus-merging side-effect imports), then `check_strides = True` was set directly on every
`ItemSpec` object already sitting in `run_module.REGISTRY` in Python memory, and
`run_module.run_registry(REGISTRY)` was called on that mutated in-memory registry. Since nothing
on disk was ever changed, there is nothing to revert — `git status`/`git diff` on `registry.py`
were re-confirmed clean at the end (see bottom of this doc).

`harness.MAX_FAILURES_KEPT` (normally 8) was also monkeypatched in-memory to 10,000 so every
failing case per item — not just the first 8 — was captured, and dumped to
`/tmp/mg_full_results.json` (`{item: {verdict, total, failed, failures: [...], reason}}` for the
entire registry). All classification below was built by parsing the *actual failure strings*
(concrete `numpy=(...) ionp=(...) shape=(...) dtype=...` tuples per corpus case), never by
trusting the item-name-only 253-line list.

`harness._strides_match` runs three checks in order: (1) buffer-protocol-verified `.strides`
equality, (2) `C_CONTIGUOUS`/`F_CONTIGUOUS` flag equality, (3) `tobytes('A')` physical-byte
equality. It runs **after** `compare_values` has already accepted the item on value grounds
(bit-exact, ULP-tolerant, or `epsilon_tolerance`-tolerant) — so it is blind to declared numeric
tolerance and will flag byte-level noise inside an already-accepted epsilon band.

## Bucket counts (sum to 253)

| Bucket | Count |
|---|---|
| **GENUINE** | **82** |
| **ARTIFACT** | **171** |
| **UNDETERMINED** | **0** |
| Total | 253 |

The 171 ARTIFACT items break down as: 150 pure zero-size-stride-convention, 16 pure
tolerance/byte-check-conflict, 4 items combining both artifact classes with no other signal
(`cos`, `linalg.pinv`, `sin`, `tanh`), and 1 non-reproducible anomaly (`full_like`).

## Taxonomy

### ARTIFACT — zero-size stride convention (150 items)

numpy 2.5.1's rule for **freshly allocated** zero-size arrays (`zeros(0)`, a fresh `.reshape`
result, etc.) is: strides are **always all-zero**, regardless of dtype/rank/`order=`. This is an
implementation quirk of the allocator, not a documented public contract, and it does not apply to
zero-length *views* into a real buffer (those keep the parent's real strides — see the GENUINE
`trim_zeros` case below, which is exactly this distinction cutting the other way). ionp instead
gives zero-size outputs plausible nonzero "as if C-contiguous" strides. Representative case
(`argwhere`, `sweep/empty1d/bool/no_args`): `numpy=(0, 0) ionp=(8, 8) (shape=(0, 1), dtype=int64)`.
Because the array has 0 elements, `tobytes()` is identical (`b''`) — there is no byte-level
divergence, only a metadata artifact on a buffer nobody can read. All 150 items in this bucket
were confirmed (via the bucketing script) to have **every** failing case fall into this exact
zero-size pattern, no exceptions.

### ARTIFACT — tolerance/byte-check conflict (16 items)

Items like `linalg.cholesky`, `linalg.inv`, and all `fft.*`/`hfft`/`irfft` family members declare
an explicit `epsilon_tolerance` (e.g. `fft`: `{"float64": ("rel", 3.412911368445222e-14)}`,
justified in-file as `_EPS_JUST_FFT`). `compare_values` already accepts these outputs as correct
under that declared, reviewed tolerance. `_strides_match`'s step 3 (`tobytes('A')`) then
independently re-litigates the same floating-point noise at the **raw byte** level, which the
epsilon tolerance was never designed to survive — a value passing "within 3.4e-14 relative"
almost never passes "byte-identical." This is a post-check applied somewhere it structurally
cannot pass, not a real ionp defect.

### ARTIFACT — combined, no residual signal (4 items: `cos`, `linalg.pinv`, `sin`, `tanh`)

Every failing case for these four decomposes into one of the two artifact classes above (some
cases zero-size, some tolerance-conflict) with zero cases left over showing a genuine non-trivial
stride divergence. Confirmed by the same case-by-case decomposition as the other buckets, not
inferred from the item name.

### ARTIFACT — non-reproducible (1 item: `full_like`)

`full_like` appears in the original 253-name list, but two independent re-runs of the exact same
methodology (same corpus, same in-memory `check_strides=True` flip, same built `.so`) show **0 of
86 cases failing** — it passes cleanly both times. I could not identify a mechanism (no
`check_strides=False` override anywhere in the registry for it, no order-dependent shared-state
effect found) that would explain the discrepancy. This is flagged explicitly rather than silently
dropped or silently counted: **the original 253-figure appears to contain at least one item that
does not currently reproduce as failing under the stated methodology.** This is a measured fact,
not a guess — I am not claiming to know *why* the original diagnostic listed it, only that I
cannot reproduce it failing.

### GENUINE — reduction output layout (24 items)

`all, amax, amin, any, count_nonzero, cumprod, cumsum, max, mean, median, min, nanmax, nanmean,
nanmedian, nanmin, nansum, ndarray.all, ndarray.any, ndarray.cumprod, ndarray.cumsum, ndarray.max,
ndarray.min, ptp, sum`. numpy propagates the input's memory order into a reduction's output layout
along non-reduced axes; ionp always emits a fresh C-contiguous buffer. Hand-verified
(`amax(a.T, axis=0)`, `a = arange(24).reshape(2,3,4).astype(f64)`): `numpy=(8, 24)
ionp=(16, 8)`, values equal, bytes differ. This is exactly the case the requester's own probe
(`/tmp/mg_verify253.py`) independently found — `np.amax(a.T, axis=0)` was one of their 2 confirmed
divergences.

### GENUINE — ufunc/binary-op `order='K'` propagation (41 items)

33 `ndarray.__dunder__` operators (`__add__`, `__mul__`, `__eq__`, `__invert__`, `__lshift__`,
etc. — full list in the summary below), the 7 `emath.*` functions, and `ndarray.clip`. numpy
ufuncs/binops default to `order='K'`, preserving a non-C-contiguous (e.g. transposed) input's
memory order in the output; ionp always emits C-contiguous. Hand-verified
(`b.T + b.T`, `b = arange(12).reshape(3,4).astype(f64)`): `numpy=(8, 32) ionp=(24, 8)`, values
equal, bytes differ. Also verified `ndarray.__eq__`, `ndarray.__invert__`, `emath.log`.

### GENUINE — view preservation (3 items: `asarray`, `ndarray.conj`, `ndarray.conjugate`)

> **CORRECTION 2026-08-03 (Monday): the `asarray` entry below is STALE — REFUTED
> against the current binary.** Re-run of this section's own stated trigger
> (`v = arange(10).astype(f64)[::-1]`) now gives `numpy.strides == ionp.strides
> == (-8,)` and `np.shares_memory == ionp.shares_memory == True`. ionp no longer
> forces a defensive contiguous copy; the aliasing consequence described below
> ("a downstream in-place write ... would not") no longer follows. The only
> residual divergence is `.base is None`, which is metadata infidelity and NOT
> memory unsafety — proven separately (7/7 results survive parent drop + gc;
> `Arc<Buffer>` keeps the storage alive). `asarray` was declared exact on this
> basis. **`conj`/`conjugate` were NOT re-tested and this correction says
> nothing about them.** Lesson #70: a recorded reason in the repo is not
> evidence about the current binary — reproduce it or refute it.

`ndarray.conj()`/`.conjugate()` on numpy return a **view**, not a copy — negative/non-contiguous
strides pass straight through. `asarray()` on an already-compatible ndarray is a no-op that must
preserve the exact input memory layout (it may not silently make a defensive copy). Hand-verified
directly on the *actual* trigger (a plain transposed view passes fine through ionp's `asarray` —
the divergence is specifically on non-contiguous slices/reversed views, not simple transposes):
`v = arange(10).astype(f64)[::-1]; np.asarray(v).strides == (-8,)` (numpy keeps the same buffer,
`r_np.base is v.base`) vs `ionp.asarray(iv).strides == (8,)` — ionp silently forces a fresh
contiguous copy instead of wrapping the existing buffer. Values equal, bytes equal (same logical
content, different physical arrangement and, critically, different aliasing semantics — a
downstream in-place write through the numpy result would mutate the original array; through the
ionp result it would not).

### GENUINE — `ndarray.copy` order-default bug (1 item)

numpy's `ndarray.copy(order=...)` **method** defaults to `order='C'` (unlike the *function*
`np.copy(a, order=...)`, which defaults to `'K'` — a well-known numpy footgun, confirmed by direct
side-by-side call). ionp's `ndarray.copy()` behaves as if it defaults to `'K'`-like propagation
instead. Hand-verified: `b.T.copy()` → `numpy=(24, 8)` (C-contiguous of the transposed *shape*,
per the method's `'C'` default) vs `ionp=(8, 32)` (propagated the transpose instead of
C-normalizing it).

### GENUINE — transpose/relayout implementation quirks (13 items)

`argwhere, delete, hstack, vstack, linalg.cross, linspace, sort, sort_complex, unique, trim_zeros,
fft.fft, fft.ifft, fft.rfft`. Each has a distinct, individually hand-confirmed mechanism:

- **`argwhere`**: numpy's implementation is essentially `transpose(nonzero(a))`; stacking numpy's
  per-axis 1-D `nonzero` outputs via transpose naturally yields an F-contiguous result. ionp
  returns C-contiguous. Verified on a concrete 4×3 bool array: `numpy=(8, 48) ionp=(16, 8)`.
- **`delete`**: verified on the *exact* corpus case (`m=arange(12).reshape(3,4)`,
  `delete(m, [0,2], axis=1)`): `numpy=(8, 24) ionp=(16, 8)`, values equal, bytes differ. (An
  earlier hand-check using a different shape happened to coincidentally match — re-testing the
  *actual* triggering corpus case, per the task's explicit warning against reduced repros, is what
  surfaced the real divergence.)
- **`hstack`/`vstack`**: when all inputs are Fortran-ordered, numpy's concatenate machinery
  preserves F-order in the output for these axis-0/axis-1-only stacks; ionp normalizes to
  C-order. Verified: `numpy=(1, 3) ionp=(8, 1)` on F-order bool inputs (shape (3,8)).
  (The corpus even labels these cases `grid/.../fortran/...` — the F-order-input dependence is
  by design in the corpus, not incidental.)
- **`linalg.cross`**: batched cross product over a non-default axis inherits input layout in
  numpy; ionp does not. Verified: `numpy=(48, 8, 24) ionp=(48, 16, 8)`, bytes equal (pure layout,
  not a value bug).
- **`linspace`**: with array `start`/`stop` and a non-default `axis`, numpy's output layout
  follows the axis-insertion order rather than plain C-order. Verified:
  `numpy=(16, 32, 8) ionp=(48, 16, 8)`, bytes equal.
- **`sort`**: sorting along the last axis of a transposed (F-order) view preserves F-order in
  numpy's output. Verified: `numpy=(8, 32) ionp=(40, 8)` on a transposed float64 view.
- **`sort_complex`**: same transpose-preservation mechanism as `sort`.
- **`unique`** (with `axis=`): numpy's `axis`-aware implementation internally transposes the data,
  computes uniqueness, and transposes back — a well-known internal detail that leaves F-order
  traces in the output. Verified on the exact corpus-style case (`unique(f, axis=1)` on a 3×4
  int64 array): `numpy=(8, 24) ionp=(32, 8)`, bytes differ.
- **`trim_zeros`** (N-D, `axis=None`): numpy's trim is implemented as a genuine **slice/view** of
  the original array — `r_np.base is a` confirmed True on the exact corpus case
  (`nd_basic = [[0,0,0],[0,1,0],[0,2,3],[0,0,0]]`, `trim='fb', axis=None`) — retaining the parent
  buffer's real row stride (`numpy=(24, 8)`, i.e. still 3-columns-wide underneath, only the row
  range narrowed) while ionp returns a freshly compacted copy (`ionp=(16, 8)`). Bytes happen to be
  equal here (same logical content) but the *aliasing* semantics genuinely differ, same as
  `asarray`/`conj` above.
- **`fft.fft`/`fft.ifft`/`fft.rfft`**: transforming along a non-last axis (`axis=0`) of a 2-D
  array — numpy returns a C-contiguous result, ionp returns F-contiguous (the reverse direction
  from most other patterns here, consistent with an internal transpose-compute-transpose-back
  implementation on ionp's side). Verified: `numpy=(64, 16) ionp=(16, 48)` on a 3×4 complex128
  array, bytes differ. (These three items *also* have additional failing cases from the
  tolerance/byte-check-conflict artifact — both an artifact-explainable subset *and* a genuine
  layout divergence coexist in the same item; the item is classified GENUINE because real signal
  exists, not because every one of its failures is genuine.)

## Hand-verification sample (24 checks, ≥15 required)

All performed via `import ionp` against the built `.so`, `.venv/bin/python`, `except BaseException`,
comparing both `.strides` and `tobytes('A')`, in the style of `/tmp/mg_verify253.py`. Script:
`/tmp/mg_handverify.py`.

| # | Case | numpy strides | ionp strides | Match? | Bytes eq? |
|---|---|---|---|---|---|
| 1 | `amax(a.T, axis=0)` | (8, 24) | (16, 8) | No | No |
| 2 | `sum(a.T, axis=0)` | (8, 24) | (16, 8) | No | No |
| 3 | `ndarray.max(a.T, axis=0)` | (8, 24) | (16, 8) | No | No |
| 4 | `cumsum(a.T, axis=0)` | (8, 32, 96) | (48, 16, 8) | No | No |
| 5 | `b.T + b.T` | (8, 32) | (24, 8) | No | No |
| 6 | `b.T * 2` | (8, 32) | (24, 8) | No | No |
| 7 | `emath.sqrt(abs(b.T))` | (8, 32) | (24, 8) | No | No |
| 8 | `ndarray.clip(b.T, 0, 5)` | (8, 32) | (24, 8) | No | No |
| 9 | `asarray(b.T)` (plain transpose) | (8, 32) | (8, 32) | **Yes** | Yes |
| 10 | `ndarray.conj(c.T)` | (16, 64) | (48, 16) | No | No |
| 11 | `b.T.copy()` | (24, 8) | (8, 32) | No | No |
| 12 | `argwhere(4x3 bool)` | (8, 48) | (16, 8) | No | No |
| 13 | `hstack([F-order, F-order])` | (1, 3) | (8, 1) | No | No |
| 14 | `vstack([F-order, F-order])` | (1, 6) | (4, 1) | No | No |
| 15 | `unique(3x4 int64, axis=1)` | (8, 24) | (32, 8) | No | No |
| 16 | `sort(transposed f64, axis=-1)` | (8, 32) | (40, 8) | No | No |
| 17 | `trim_zeros(nd_basic, 'fb', axis=None)` [exact corpus case] | (24, 8) | (16, 8) | No | Yes |
| 18 | `delete(3x4 arange, [0,2], axis=1)` [exact corpus case] | (8, 24) | (16, 8) | No | No |
| 19 | `linspace(array start/stop, axis=1)` | (16, 32, 8) | (48, 16, 8) | No | Yes |
| 20 | `linalg.cross(2x3x2, axis=1)` | (48, 8, 24) | (48, 16, 8) | No | Yes |
| 21 | `fft.fft(3x4 complex128, axis=0)` | (64, 16) | (16, 48) | No | No |
| 22 | `ndarray.__eq__(b.T, b.T)` (int32) | (1, 4) | (3, 1) | No | Yes |
| 23 | `ndarray.__invert__(b.T)` (int32) | (4, 16) | (12, 4) | No | No |
| 24 | `emath.log(abs(b.T)+1)` | (8, 32) | (24, 8) | No | No |

22 of 24 confirm a real, reproducible stride divergence; #9 confirms `asarray` is clean on plain
transposes specifically (its genuine divergence is on non-contiguous slices/reversed views, shown
separately above and reflected in the taxonomy, not in this table row); #22's byte-equality
despite differing strides shows some of these are pure-layout with no value/byte consequence,
which is expected for order='K' propagation gaps. No hand check contradicted a GENUINE
classification; none are silently omitted.

## True count and confidence

**82 of 253 declared items have a genuine, real, numpy-documented-or-consistently-reproducible
layout contract that ionp currently violates.** The remaining 171 are artifacts of how the
comparison is applied (mostly the zero-size-array stride convention, secondarily tolerance/byte-
check conflicts), not real ionp defects — plus one item (`full_like`) that does not reproduce as
failing at all under the exact stated methodology.

Confidence: **high** for the bucket boundaries (the zero-size and tolerance-conflict artifact
classes were confirmed by exhaustively decomposing every failing case per item, not sampling; the
82 GENUINE items were split into 5 named mechanisms covering 100% of the bucket with zero leftover
unclassified items, and 22 independent hand-checks against real numpy — 7 more than required —
confirmed the mechanism in every pattern, including reproducing the requester's own two prior
confirmed divergences, `amax`/`all` under Pattern A). Confidence is **lower** on two specific
points that would change the count if resolved differently: (1) the `fft.*` three items are
classified GENUINE on the strength of one non-last-axis case each — if that single mechanism
turns out itself to be an edge-case artifact of my synthetic 3×4 test array rather than a
general contract, those 3 items would move to ARTIFACT (78 → GENUINE would still hold for the
rest); (2) `full_like`'s non-reproducibility is reported honestly but its root cause (stale
original diagnostic vs. some subtler nondeterminism) was not tracked down — if it turns out there
*is* a real intermittent divergence, the true count could be 83 instead of 82.

What was **measured** (traced to a concrete corpus case, a concrete `numpy=(...) ionp=(...)` pair,
and in most cases an independent hand-check against real numpy): all 82 GENUINE items, the 150
pure-zero and 16 pure-tolerance-conflict ARTIFACT items (every failing case decomposed, not
sampled), the 4 mixed-artifact items, and the `full_like` non-reproduction. What was **inferred**
rather than independently re-derived from numpy source: the *named mechanism* behind each Pattern
(e.g. "numpy's `unique(axis=...)` transposes internally" is inferred from the observed strides
being consistent with that mechanism across many dtype/shape combinations, not from reading
numpy's C source) — the *existence and reproducibility* of every divergence is measured, the
*explanation for why numpy behaves that way* is the most parsimonious inference consistent with
every observed case, in a few instances (`unique`, `hstack`/`vstack`, `fft`) without independently
confirming against numpy's source code.

## Repository state confirmation

`git status --short` inside `~/Monday/ionp` at completion of this investigation shows
no changes to `registry.py` or any other tracked file under `ionp/` — the only pending changes in
the parent repo are an unrelated memory-log file and a pre-existing untracked script outside this
project's scope. `check_strides` was never set to `True` on disk; all 253-item enumeration and
classification above was produced entirely from in-memory monkeypatches of an imported `run`
module plus `/tmp`-only probe scripts.
