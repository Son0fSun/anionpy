# `multi_sorted_stride_perm` — the ambiguity guard is NOT the bug

**Date:** 2026-08-05
**Measured against:** numpy 2.5.1, live, on the installed `.so` (no rebuild).
**Status:** task #7's stated premise is **unsupported**. A real divergence
exists, but it is not where the task says it is.

## What task #7 claimed

"Fix `multi_sorted_stride_perm` ambiguity scan (Finding D)."

Two things are wrong with that title.

**(1) The "(Finding D)" tag is a mislabel.** Finding D in
`DENOMINATOR-AUDIT.md:179` is *"whole submodules never walked"* — the 41
missing `np.exceptions` (7) + `np.dtypes` (34) items. It has nothing to do
with strides. It is also **already done**: `tools/numpy_surface.json`
today contains exactly 7 `exceptions.*` and 34 `dtypes.*` entries, and the
2866 denominator matches the ledger. Nothing to fix there.

**(2) The ambiguity scan is not the defect.** Measured below.

## What was measured

`ionp-core/src/array.rs:953`. The scan skips an operand for a given axis
pair when either axis has **the operand's own** `shape == 1`, and stops
shifting when two operands disagree ("first decisive operand wins").
numpy's iterator instead skips on **`stride != 0`** and resolves conflicts
with "C-order wins" (`shouldswap` is latched to 0 by any operand voting
`<=`, and a later operand cannot re-raise it because `ambig` is already 0).

Those are different *mechanisms*, so the obvious suspicion is that they
give different *answers*. They do not — not on any case I could construct.

Probe: all ordered pairs from a 10-variant operand set (C, F, two
transposes, negative-stride, three size-1 slices, and two lower-ndim
operands), fed to `np.add(x, y, order='K')`, comparing resulting
`.strides`. 68 broadcastable pairs.

Real ionp vs real numpy: **2 stride mismatches / 27 comparable pairs.**

Then four candidate guards were simulated in Python and each scored
against numpy's actual output strides over all 68 pairs:

| guard | correct |
|---|---|
| `stride != 0` (numpy's iterator rule) | 66/68 |
| `shape != 1` on the **broadcast** shape | 62/68 |
| both conditions | 66/68 |
| either condition | 62/68 |

**The current behaviour and `stride != 0` score identically, and miss the
identical two cases.** Adopting numpy's literal guard would change
nothing. The broadcast-shape variant — which was my hypothesis going in —
is strictly *worse* than what ships today, 62 vs 66. I record that because
it was a real prediction that a measurement killed; had I implemented it
on the strength of the two-case reproduction alone, I would have traded 2
failures for 6.

## The residual divergence (real, reproducible, unexplained)

Both misses are the same operand pair:

```
>>> import numpy as np
>>> A  = np.arange(24, dtype=np.float64).reshape(2, 3, 4).copy()
>>> F  = np.asfortranarray(A)          # shape (2,3,4) strides (8, 16, 48)
>>> b1 = A[:, :1]                      # shape (2,1,4) strides (96, 32, 8)
>>> np.add(F, b1, order='K').strides
(96, 8, 24)                            # perm [0, 2, 1]
```
ionp yields `(32, 64, 8)` — perm `[1, 0, 2]`. Values are correct; this is
`.strides` only, the same class as the Phase 3 divergences.

What makes it hard: `b1` has a size-1 axis carrying a **meaningful**
stride (32), so "skip it" and "let it vote with stride 0" are both
defensible readings, and *neither* reproduces numpy here. All four guards
above get this case wrong in the same direction.

## Verdict

**Not fixed, and deliberately not guessed at.** The plan's own standing
note applies verbatim: *"deriving half the heuristic from one grid would
very likely be right here and wrong outside."* I have one failing shape
family and four falsified models; that is enough to know the current
explanation is wrong and nowhere near enough to write the right one.

A claim I could not reproduce is not the same as a claim I disproved —
and the inverse holds here too: a divergence I reproduced but cannot
model is not a fix I am entitled to write. The repro above is exact;
anyone re-opening this should start by pasting it, not by re-reading this
description.

## Not checked

- Whether the divergence widens beyond 3-D, beyond two operands, beyond
  float64, or beyond `np.add`. The grid is 3-D, 2-operand, float64,
  one ufunc. **ILLUSTRATIVE, NOT EXHAUSTIVE.**
- Whether numpy's axis *coalescing* (which runs after axis ordering)
  accounts for the residual. Untested; currently the leading suspect.
## Reachability — CHECKED, and the answer sets the priority

The open question above was resolved the same day, so it is answered here
rather than left as an unknown.

`order='K'` is well represented in the corpus — 36 references across
`order_cases.py`, `manip_compose_cases.py`, `compare_compose_cases.py`,
`ndarray_attrs_cases.py`, `linalg_cases.py`. But every one of them is a
**single-operand** operation: `copy`, `astype`, `ravel`, `flatten`,
`reshape`, `clip`, `*_like`. `compare_compose_cases.py:195`'s
`for order in (None,"C","F","A","K")` loop is `asarray_chkfinite`, also
single-operand.

**No corpus case exercises a two-operand ufunc under `order='K'` with
operands of conflicting memory layout.** That is the exact configuration
required to reach `multi_sorted_stride_perm`'s multi-operand conflict
path at all — with one operand the scan can never disagree with itself,
so the entire "first decisive operand wins" branch is dead code under
test.

This is why the divergence is not among the 27 known failures: nothing
looks for it. That makes it **worse** than a listed failure, not better.
A failing item is a debt you can see. This is a real, reproducible
`.strides` divergence in code that the suite reports as covered.

Consequence for sequencing: the corpus case must be written **before**
any fix, per the standing loop (corpus first, then implement, then verify
out of corpus, then declare). Writing it will move the suite baseline
**27 → 28**, and that delta is correct — it is the instrument starting to
measure something it was previously blind to, not a regression. Whoever
lands it must say so in the same commit, as the plan requires.
