# Contiguity flags + reshape `.base` fix (2026-08-02)

Baseline ledger (clean tree, HEAD=1a30706, measured before any change in
this task): **442 exact / 5 failing / 733 absent / 0 phantom / 0 untested**
out of 1180.

Final ledger (after this task's fix + declarations):
**442 exact / 5 failing / 730 absent / 0 phantom / 3 untested** out of 1180.

`exact` is unchanged (no regression). `untested` went from 0 to 3
(`ndarray.base`, `ndarray.flags`, `may_share_memory`) because these items
are now correctly declared but have no `tests/differential/registry.py`
entry backing them with a passing differential test -- `registry.py` is out
of this task's edit scope (explicitly forbidden), so there was no way to
move them further than `untested` through the normal coverage pipeline.
See "What was NOT done" below.

## Defect 1: contiguity flags were offset-sensitive; numpy's are not

### What the code actually did (confirmed, not just hypothesized)

`ionp-core/src/array.rs`, `NdArray::is_c_contiguous`/`is_f_contiguous`,
before this fix:

```rust
pub fn is_c_contiguous(&self) -> bool {
    self.size() == 0 || (self.offset == 0 && strides_match_ignoring_unit_axes(&self.shape, &self.strides, true))
}
pub fn is_f_contiguous(&self) -> bool {
    self.size() == 0 || (self.offset == 0 && strides_match_ignoring_unit_axes(&self.shape, &self.strides, false))
}
```

The task's hypothesis was confirmed by direct code reading: both getters
had a `self.offset == 0` conjunct alongside the real shape/strides check
(`strides_match_ignoring_unit_axes`, which already correctly implements
numpy's "ignore the stride of any length-<=1 axis" rule -- that part was
already right, added in an earlier pass the same day). The `offset == 0`
conjunct is what caused every nonzero-base-offset view with otherwise
canonical strides (`b[1:3]`, `b[1]`, `v[2:6]`, ...) to wrongly report
`C_CONTIGUOUS=False`/`F_CONTIGUOUS=False`.

### numpy's actual rule

Contiguity in real numpy is a pure function of **shape + strides +
itemsize**. The base offset into the underlying allocation plays no role
at all -- confirmed against real numpy 2.5.1 for every case in the task's
own table plus a much larger out-of-corpus probe (below).

### The fix

Dropped the `self.offset == 0 &&` conjunct from both getters. Nothing else
changed; `strides_match_ignoring_unit_axes` (unit-axis-tolerant, 0-size
early-return, 0-d trivially-true via the empty-shape loop) was already
correct and needed no changes.

## Defect 2: `.reshape()` did not attach `.base`

### What the code did

`ionp-py/src/lib.rs`'s `PyArray::reshape` built a plain
`PyArray { inner: out }` and returned it directly -- never calling the
pre-existing `attach_base` helper (already correctly wired into
`__getitem__`, `.T`, `.transpose()` by an earlier pass). So `.base` was
always `None` and `.flags['OWNDATA']` was always `True` for every
`.reshape()` result, view or not.

### An important empirical correction to this task's own brief

The task's brief hypothesized: "If a given reshape must copy... numpy
returns an array with `base is None` and `OWNDATA` True." **This is
false**, verified directly against real numpy 2.5.1:

```python
>>> t2 = np.arange(12., dtype='float64').reshape(3,4).T   # F-contig, (4,3)
>>> r2 = t2.reshape(3,4)                                   # cannot be a real view of t2
>>> r2.flags.owndata
False
>>> r2.base is None
False
>>> np.shares_memory(r2, t2)
False
>>> r2.base.shape, r2.base.flags.owndata, r2.base.base
((4, 3), True, None)
```

Real numpy's `PyArray_Newshape` never hands back an *owning* array
directly from `.reshape()`. When a direct view isn't possible, it first
makes a fresh contiguous copy that preserves the **original (pre-reshape)
shape** -- an otherwise-invisible array with `OWNDATA=True`, `base=None`
-- and then returns a genuine view of *that* at the new shape. So
`.reshape()`'s own result *always* has `base is not None` /
`OWNDATA=False`, whether or not a real copy happened underneath. This also
holds for `copy=True` forcing a copy on an input that *could* have stayed
a view (`orig.reshape(2,12,copy=True)`), and for `np.reshape()` the free
function, not just the `.reshape()` method.

### The fix

`PyArray::reshape` (`lib.rs`) now:

1. Computes the candidate reshape via the existing
   `NdArray::reshape_with_order`.
2. If it's a real view of `self` and `copy != True`: build the child and
   `attach_base(child, slf)` -- the existing chain-flattening helper,
   unchanged.
3. Otherwise (must-copy, or `copy=True` forced a copy): builds a "hidden
   owner" array via the pre-existing `NdArray::to_contiguous_order(order)`
   (unchanged; already returns a fresh, correctly-laid-out, original-shape,
   offset-0 buffer), reshapes a real view out of *that* owner via
   `reshape_with_order` again (now hits the owner's own fast contiguous
   path since the owner is genuinely contiguous), and `attach_base`s the
   child to a freshly-constructed `PyArray` wrapping the owner (which
   itself gets no `.base`, matching numpy's `base.base is None`).
4. `copy=False` on a must-copy candidate still raises
   `"Unable to avoid creating a copy while reshaping."`, matching prior
   behavior (moved earlier in the function, logic unchanged).

## Probes run

Two out-of-corpus probe scripts, written fresh for this task (not copies of
the task's own illustrative table), both asserting operand parity
(shape/dtype/strides equal between the numpy and ionp operand) before every
comparison:

- `probe_flags_base.py`: 25/25 `.flags` cases pass (nonzero-offset slices
  at 1d/2d/3d, length-1 axes in every position via both reshape and offset
  slicing, two 0-sized-array constructions, a 0-d array, transposes
  (plain and offset-then-transpose), negative-stride views (1d, and 2d on
  each axis and both), an F-contiguous input plus two offset slices of it,
  two view-of-view chains) and 16/16 `.base` cases pass (non-reshape view
  producers re-verified not regressed; `.copy()` and a 0-d literal
  correctly get no base; `.reshape()` view case, must-copy case, must-copy
  via `copy=True`, must-copy via a reversed 1d input, and an offset-slice
  view case).
- `probe_shares_memory.py`: 10 cases for `shares_memory`/
  `may_share_memory` together. `may_share_memory` matches numpy on all 10.
  `shares_memory` **diverges on 1 of 10** (see below).

Both scripts are saved under the task's scratchpad directory (not part of
the repo): `probe_flags_base.py`, `probe_shares_memory.py`.

## What was declared, and why

- `ndarray.flags`: declared `exact` in `ionp/_state/ndarray.py`. 25/25 own
  probe cases pass.
- `ndarray.base`: declared `exact` in `ionp/_state/ndarray.py`. 16/16 own
  probe cases pass, including the reshape must-copy hidden-owner case this
  task's own hypothesis got wrong.
- `may_share_memory`: declared `exact` in `ionp/_state/toplevel.py`. 10/10
  own probe cases pass. It was already implemented (bounds-only, matching
  numpy's own documented bounds-only contract for this specific function)
  before this task; this task only added the corpus-independent
  verification and the declaration.
- `shares_memory`: **NOT declared.** `ndarray_attrs.rs`'s implementation
  is literally the same bounds-only check as `may_share_memory` (its own
  pre-existing doc comment already discloses this), but real numpy's
  `shares_memory` does an *exact* overlap check. Found a genuine
  divergence: `ionp.shares_memory(v[0::2], v[1::2])` (interleaved strided
  views whose address bounds overlap but whose actual elements are
  disjoint) returns `True`; real numpy returns `False`. This is a
  pre-existing, disclosed gap, not something introduced by or in scope for
  this task's two assigned defects -- implementing numpy's real
  exact-overlap algorithm (a Diophantine/lattice-based solver in real
  numpy's C source) is nontrivial and left undone.

## What was NOT done / NOT verified

- **`ionp.reshape()` the free function** (`ionp-py/src/creation.rs`) is
  outside this task's edit scope and was left untouched. It still does not
  attach `.base` at all. Only the `.reshape()` *method* on `ndarray`
  (`lib.rs`) was fixed. Do not assume the free function matches.
- **`shares_memory`'s exact-overlap algorithm** was not fixed (see above);
  left undeclared with the divergence documented in
  `ionp/_state/toplevel.py`.
- **Registry coverage**: `tests/differential/registry.py` has no entries
  for `ndarray.base`, `ndarray.flags`, `shares_memory`, or
  `may_share_memory`. This task's rules forbid editing that file, so even
  the three items declared here show as `untested` in the coverage ledger
  (not `exact`) -- the declaration is truthful (the function/attribute
  exists and was independently verified) but is not registry-test-backed.
  This is expected, not an error; a follow-up task with `registry.py` in
  its edit scope would need to add real cases to move these three to
  `exact` on the ledger.
- **A found-but-out-of-scope divergence**: `b[1,2]` (full integer
  indexing on a 2d array) returns a numpy *scalar* (`np.float64`) in real
  numpy, whose `.flags` reports `OWNDATA=True, WRITEABLE=False` by
  numpy-scalar convention. `ionp`'s `__getitem__` instead returns a 0-d
  `ndarray`, whose (now-correct) `.flags` reports `OWNDATA=False,
  WRITEABLE=True`. This is a scalar-vs-0-d-array *return type* difference
  in `__getitem__`/`scalars.rs`, not a contiguity-flags computation bug --
  `.flags` itself is internally consistent and correct for whatever object
  it's called on. Out of this task's edit scope (`scalars.rs`,
  `__getitem__`'s scalar-collapse path) and not fixed here.
- **numpy's rarer flag keys** (`ALIGNED`, `WRITEBACKIFCOPY`, `FNC`,
  `FORC`, ...) were not touched; `ionp.ndarray.flags` still only exposes
  the four keys (`C_CONTIGUOUS`, `F_CONTIGUOUS`, `OWNDATA`, `WRITEABLE`)
  it exposed before.
- **`WRITEABLE`** was not made to match numpy bit-for-bit (ionp has no
  read-only-array concept and no `__setitem__` at all); it is still
  unconditionally `True`, which was already the documented, accepted
  behavior before this task.
- **Non-float64 dtypes**: the contiguity math itself is dtype-independent
  by construction (it only touches shape/strides/offset, never buffer
  contents), but the probe was not independently re-run per dtype to
  confirm no dtype-specific code path exists that could diverge.
- **3+ dimensional strided interleaving** beyond the specific 1d
  `v[0::2]`/`v[1::2]` case was not probed for `shares_memory`/
  `may_share_memory`; nor were cross-dtype/cross-itemsize comparisons.
