# Reduction output layout fix (24 items re-declared)

Follow-up to `docs/order-k-propagation-fix.md` (commit range ending 6ca59ba),
which fixed order='K' memory-layout propagation for ELEMENTWISE ops and
deliberately left reductions out of scope. Commit c65e573 subsequently
revoked 24 reduction-family items for the analogous bug: ionp always
allocated C-contiguous output for reductions, where real numpy propagates
the input's memory layout into the output. This document covers the fix
for those 24 items.

## Items re-declared (24, exact set revoked in c65e573)

Top-level (18): `all`, `amax`, `amin`, `any`, `count_nonzero`, `cumprod`,
`cumsum`, `max`, `mean`, `median`, `min`, `nanmax`, `nanmean`, `nanmedian`,
`nanmin`, `nansum`, `ptp`, `sum`.

`ndarray.*` methods (6): `ndarray.all`, `ndarray.any`, `ndarray.cumprod`,
`ndarray.cumsum`, `ndarray.max`, `ndarray.min`.

All 24 are confirmed by dedicated out-of-corpus probes against real numpy
2.5.1 (0 mismatches each), not inferred from a paraphrase or from code-path
sharing alone:

* `/tmp/probe_reduce_verify.py` -- `sum`, `amax`, `amin`, `max`, `min`,
  `all`, `any`, `count_nonzero`, `nansum` (free-function forms). 1080 cases:
  shapes `(2,3,4)`, `(2,3,4,5)`, `(3,4)` x layouts C/F/fulltranspose/
  partialswap/permuted/steppedslice/negstride x axis None/0/last/multi-axis
  tuples x keepdims True/False. 0 mismatches.
* `/tmp/probe_reduce_verify2.py` -- `cumsum`, `cumprod`, `ptp`, `mean`,
  `nanmean`, `nanmin`, `nanmax`, `median`, `nanmedian`. 690 cases across the
  same shape/layout grid x axis None/0/last x keepdims True/False. 0
  mismatches.
* `/tmp/probe_ndarray_methods.py` -- the 6 `ndarray.*` method forms
  (`.min()`, `.max()`, `.all()`, `.any()`, `.cumsum()`, `.cumprod()`),
  dedicated probe since the free-function probes only exercise `np.sum(a)`
  style calls, not `a.sum()`. 420 cases, same layout/axis/keepdims grid. 0
  mismatches.
* `/tmp/probe_div_mech.py` / `/tmp/probe_div_mech2.py` -- direct real-numpy
  (no ionp) investigation isolating `mean`'s actual divide mechanism (see
  below). 15/15 + 2 masked (`where=`) matches, 0 exceptions.

Every probe asserts operand stride parity against numpy before comparing
outputs (views built via `.T`/slicing on a freshly-constructed array,
never `as_strided` for the final verification pass), and every probe
compares both VALUES (`np.allclose`, `equal_nan=True`) and STRIDES
(`.strides` in bytes) -- a mismatch in either counts as a failure.

## The rule: THREE distinct mechanisms, not one

Initial assumption was a single unified rule ("propagate input layout into
reduction output"). Empirical investigation disproved this -- there are
three genuinely different mechanisms, corresponding to three different
things real numpy's C implementation actually does:

### (a) Direct single reductions (unchanged from a prior pass, re-verified)

`sum`, `amax`, `amin`, `all`, `any`, `count_nonzero`, `nansum`, `max`,
`min`, `nanmin`, `nanmax`, and the `ndarray.*` equivalents:

> Pretend every reduced axis has size 1 (but keep its REAL original
> stride). Stable-sort ALL axes (reduced and retained) by strictly
> descending `abs(stride)`. Lay out C-contiguous strides for that permuted
> pretend-shape. Un-permute back onto the original axis positions. Without
> `keepdims`, drop the reduced-axis entries afterward.

Implemented as `NdArray::reduce_output_layout` /
`NdArray::relayout_for_reduction` / `NdArray::lift_reduction_keepdims` in
`ionp-core/src/array.rs` (unchanged this pass -- these were already correct
and are the foundation the other two mechanisms below reuse).

### (b) `cumsum` / `cumprod`

Follow the ELEMENTWISE K-order rule (`multi_sorted_stride_perm`), not the
reduction-squeeze rule -- confirmed empirically, 0 mismatches. These were
already correct going into this pass (fixed in the earlier elementwise-op
task, since a cumulative op's output has the same shape as its input, so
it never needed reduction-specific handling).

### (c) `ptp` (`max - min`)

A genuine two-operand elementwise composition of two REDUCTION results
(NOT `out=`-based). Real numpy's order='K' ufunc machinery, when a
`keepdims=True` reduction operand feeds a further elementwise op, checks
generalized C/F-contiguity across all operands FIRST and, if one is
decisively contiguous, uses freshly recomputed canonical strides for the
whole shape (discarding any synthetic size-1-axis stride); otherwise it
falls back to the general multi-operand stride vote.

Implemented as `NdArray::k_order_relayout_composed` in
`ionp-core/src/array.rs`. A real bug was found and fixed in this pass: the
original implementation checked ALL operands for C-contiguity, then ALL
operands for F-contiguity (two separate passes) -- this let a LATER
operand's trivial (0-d, always-both-C-and-F) contiguity wrongly pre-empt an
EARLIER operand's genuine, decisive F-contiguous layout. Fixed by checking
each operand fully (C then F) before moving to the next. After the fix,
`ptp` reaches 0/N mismatches with no regression elsewhere.

### (d) `mean` / `nanmean` (`sum / count`) -- the mechanism that looks like (c) but isn't

Despite being structurally identical to `ptp` ("two reduction operands
combined by a further elementwise op"), `mean`/`nanmean` do NOT go through
order='K' composition. Real numpy's actual `_mean` implementation
(`numpy/_core/_methods.py`) computes:

```python
um.true_divide(ret, rcount, out=ret, casting='unsafe')
```

The explicit `out=ret` argument writes the quotient's VALUES directly into
`ret`'s (the sum's) own pre-existing buffer/layout, bypassing order='K'
output-layout inference entirely. **`mean`'s result strides are simply,
unconditionally, `sum`'s own strides.**

Confirmed via direct real-numpy comparison (no ionp involved,
`/tmp/probe_div_mech2.py`): 15/15 matches across every layout
(C/F/fulltranspose/partialswap/permuted) x every axis, plus masked
(`where=`) C and F cases, all `keepdims=True`:
`np.sum(...).strides == np.mean(...).strides` exactly, every time.

Concrete disproof of the single-mechanism hypothesis: on a partialswap
array/axis where `sum`/`amax` both give `(32,32,8)` bytes (keepdims=True),
`np.ptp` gives `(64,32,8)` (fresh-recompute, mechanism (c)) while `np.mean`
gives `(32,32,8)` (sum's own, unrecomputed, preserved value, mechanism
(d)) -- same input, different mechanisms.

**Fix**: compose `sum_result`/`count_result` at `keepdims=false` (the
natural, retained-axis-only shape -- exactly what `relayout_for_reduction`
expects as input), do ALL dtype casting on that natural-shape result, then
call `raw_result.relayout_for_reduction(arr.shape(), arr.strides(), &axes,
keepdims)` once at the very end -- reusing the exact same function
`sum_result` itself is built with, guaranteeing structural identity by
construction rather than by a separate (and, for `mean`, wrong) K-order
vote. Implemented in `ionp-py/src/reductions.rs` (`mean`, `nanmean`).

Casting must happen BEFORE the final relayout, never after: `cast_to`
force-flattens to plain C whenever a real dtype change happens (see
`to_contiguous`), so casting after the layout-determining relayout would
silently discard it.

### (e) `median` / `nanmedian` -- a fourth code path, same rule as (a) plus a broadcast twist

`median`/`nanmedian` route through `ionp_core::stats::median_axis`
(`sort_rows`-based), a completely separate implementation from
`reduce_axis`, never touched by the earlier elementwise-op or (a)/(c)/(d)
work. Investigated fresh this pass.

`keepdims=False`: median's own output layout exactly matches a plain
`sum`/`mean` reduction over the same single axis -- confirmed via direct
real-numpy comparison (`/tmp/probe_median_mech2.py`):
`np.sum(...).strides == np.median(...).strides` for every layout x axis
combination tested. Fixed by calling `relayout_for_reduction` (mechanism
(a)'s own function) on the natural-shape result inside `median_axis`.

`keepdims=True`: UNLIKE `sum`/`mean` (which keep the array's REAL original
stride at the reduced axis's position -- `reduce_output_layout`'s
"pretend size-1 axis" trick), median's keepdims=True result carries a
genuine BROADCAST stride of 0 at that axis instead. Example: a
C-contiguous `(2,3,4)` array reduced on axis=0 gives
`np.sum(..., keepdims=True).strides == (96, 32, 8)` (real axis-0 stride
preserved) but `np.median(..., keepdims=True).strides == (0, 32, 8)`
(literal zero), same input. Fixed by using `ionp_core::creation::expand_dims`
(already implements the stride-0 broadcast convention) instead of
`lift_reduction_keepdims`.

A related bug in the SAME broadcast convention was found and fixed in the
Python-level `axis=None, keepdims=True` wrapper
(`expand_keepdims_none` in `ionp-py/src/stats.rs`, shared by
`median`/`nanmedian` and also `percentile`/`quantile`/`nanpercentile`/
`nanquantile`): it padded the flattened-to-scalar result's shape via plain
`reshape`, which relabels onto freshly-computed C-contiguous (nonzero)
strides -- not the literal 0-stride broadcast real numpy uses for every
inserted axis in this case (`np.median(a, axis=None, keepdims=True).strides
== (0, 0, 0)` for a `(2,3,4)` input, never nonzero). Fixed the same way,
with `expand_dims`. This helper is shared with `percentile`/`quantile`
(out of this task's scope), but the fix is a strict correctness
improvement there too (confirmed no regression via
`/tmp/probe_quantile_regression.py` -- the pre-existing `axis=0/2,
keepdims=True` layout gaps for `percentile`/`quantile` are unchanged,
unrelated to this fix, and were never in scope to fix here).

`percentile`/`quantile`/`nanpercentile`/`nanquantile` themselves
(`quantile_axis` in `ionp-core/src/stats.rs`) were NOT touched -- they were
never part of the 24-item target set and still have their own
keepdims=True layout gap for non-None axes (pre-existing, unrelated,
confirmed unchanged by the probe above).

## cumsum/cumprod: which rule?

Elementwise rule, not reduction rule -- confirmed via
`/tmp/probe_reduce_verify2.py` (0/N mismatches across the full layout x
axis grid) and unchanged from the prior elementwise-op task. This makes
sense structurally: a cumulative op's output has the SAME shape as its
input (nothing is actually reduced away), so numpy's ordinary
elementwise order='K' inference applies directly -- there is no
retained/reduced axis distinction for it to need reduction-specific
handling at all.

## Files changed

* `ionp-core/src/array.rs` -- fixed the operand-check-order bug in
  `NdArray::k_order_relayout_composed` (checks each operand fully, C then
  F, before the next, rather than two separate full passes); corrected its
  doc comment (previously overclaimed it also covered `mean`/`nanmean`);
  corrected `lift_reduction_keepdims`'s doc comment to stop describing a
  vote-based mechanism for `mean`/`nanmean` that no longer exists.
* `ionp-py/src/reductions.rs` -- `mean`, `nanmean`: replaced the (buggy for
  this case) `k_order_relayout_composed`-based composition with
  compose-at-keepdims=false + cast + `relayout_for_reduction`-once, per
  mechanism (d) above.
* `ionp-core/src/stats.rs` -- `median_axis`: added a `relayout_for_reduction`
  call for the natural (keepdims=false) layout, and an `expand_dims`-based
  broadcast insert for keepdims=True, per mechanism (e) above.
  `quantile_axis` was NOT touched (out of scope).
* `ionp-py/src/stats.rs` -- `expand_keepdims_none`: replaced the `reshape`
  based axis=None+keepdims=True padding with an `expand_dims`-based
  stride-0 broadcast insert, matching real numpy. Shared by
  `median`/`nanmedian` (in scope) and `percentile`/`quantile`/
  `nanpercentile`/`nanquantile` (out of scope, unaffected/improved as a
  side effect, verified no regression).
* `ionp/_state/toplevel.py`, `ionp/_state/ndarray.py` -- re-declared all 24
  target items `"exact"`, each with a `RE-DECLARED 2026-08-02` comment
  pointing at the specific probe evidence.

## Ledger

Measured directly, both steps (`run.py` then `coverage.py --tests`), by
this agent, in this session:

* Before any `_state/*.py` edits, after all code fixes: `exact 419, failing
  5, absent 756, phantom 0, untested 0` (1180 total) -- i.e. the code fixes
  alone did not move the ledger, since none of the 24 items were declared
  yet. `failing` unchanged from the documented clean-tree baseline at
  6ca59ba (419/5/756/0/0) -- no regression from the array.rs/reductions.rs/
  stats.rs edits.
* After re-declaring the 24 items: `exact 443, failing 5, absent 732,
  phantom 0, untested 0` (1180 total). `exact` moved by exactly +24 (the
  full target set), `failing` stayed at 5 (no regression), `phantom` and
  `untested` stayed 0.

`cargo test -p ionp-core --release`: 202 passed, 3 failed -- the same 3
pre-existing, unrelated failures already documented elsewhere
(`complex_input_is_rejected_for_math_unary_ops`,
`reciprocal_rejects_integer_input_documented_gap`,
`square_rejects_bool_documented_gap`, all unary-op dtype-rejection gaps,
untouched by this task). No new Rust test failures introduced.

## What was measured directly vs. carried over

Measured directly, this pass, via dedicated fresh probes: all 24 target
items (both value and stride correctness), the `k_order_relayout_composed`
operand-order bug and its fix, the `mean`/`nanmean` `out=ret` mechanism
(via direct real-numpy-only investigation), the `median`/`nanmedian`
natural-layout and broadcast-keepdims rules, the `expand_keepdims_none`
fix and its lack of regression on `percentile`/`quantile`.

Carried over from the prior elementwise-op task, not re-derived this pass:
the elementwise K-order rule itself (`multi_sorted_stride_perm`) and the
fact that `cumsum`/`cumprod` use it -- re-CONFIRMED (0 mismatches) but not
re-derived from scratch.

Not investigated at all: `percentile`/`quantile`/`nanpercentile`/
`nanquantile` (out of scope; their own keepdims=True, axis!=None layout
gap is pre-existing and untouched).
