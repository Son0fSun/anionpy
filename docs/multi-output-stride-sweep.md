# Multi-Output Stride Sweep: what the 253-item / 82-GENUINE sweep couldn't see

## Why this exists

`docs/stride-gap-classification.md`'s 253-candidate sweep worked by calling each declared
item and running `harness._strides_match(np_out, ionp_out)` on the raw return value. That
function does `np.asarray(np_out)` / `np.asarray(ionp_out)` on **the whole return value**.
For any item whose real return is a tuple/list of arrays, `np.asarray(a_tuple_of_arrays)`
either stacks same-shape arrays into one bogus extra-dimensional array or raises/produces an
object array -- either way the comparison is meaningless, and `harness.run_case` additionally
short-circuits: `multi_output` items are routed to `compare_multi_output`, which **never calls
`_strides_match` at all** (see `harness.py:1485-1490`, `harness.py:1504` -- the strides check
is gated on `not multi_output`... no, precisely: it is simply never reached because
`compare_multi_output`'s own return skips the `check_strides` branch entirely). So every
tuple-returning item -- `check_strides=True` or not -- has had its output layout checked
**zero times** by any existing automated harness path. `linalg.eig`/`linalg.eigh`/`linalg.qr`
go further: their registry entries use `numpy_adapter=`/`ionp_adapter=` invariant-recomputation
probes (documented in `registry.py`'s `ItemSpec.numpy_adapter` docstring), so even a
hypothetical multi-output-aware strides check would still never see the real
`ionp.linalg.eig(a)` return value -- it would see a synthetic "should be ~0" array instead.

This document is a from-scratch, direct-call sweep (bypassing the registry/harness adapter
layer entirely for the items that use one) built specifically to close that blind spot.

## Method (what was actually run)

1. Enumerated `ionp.__ion_state__` live from the built extension: 424 items with state
   `"exact"` at measurement time (all 424 keys currently present are `"exact"`; the dict
   carries no other states right now). `.venv/bin/python` used throughout, never bare
   `python3`. Cross-checked every one of those 424 names resolves to a `registry.REGISTRY`
   entry (424/424 matched) -- registry names and `__ion_state__` names use the same surface
   naming, so no hand-rolled `_state/*.py` scraper was used anywhere in this task.
2. First pass: for every one of the 424 items, pulled its registry `ItemSpec`, built its real
   corpus cases via `run.build_cases(spec)` (the actual production case-builder, not a
   reimplementation), called both `spec.resolve_numpy()` and `spec.resolve_ionp()` on the
   first 1-3 cases, and recorded the **runtime type** of the return value (not just the
   declared `multi_output`/`scalar_like` flags -- the actual `type(result)`).
3. Manually triaged that first pass's ~246 "not a single ndarray" hits. Most were **not**
   candidates at all: numpy-scalar ufunc results on 0-d inputs (expected, mirrors real numpy),
   and a large block of deliberate `scalar_like=True` metadata probes (`True_`/`False_`,
   `nan`/`inf`/`pi`/`e`, the abstract-dtype-class probes, the concrete-scalar-type probes
   `float64`/`int8`/etc., all of `testing.*`, `lib.NumpyVersion`, `lib.Arrayterator`,
   `ndarray.dtype`/`.tobytes`/`.tolist`/`.item`/`.shape`/`.ndim`/`.size`/`.itemsize`/`.nbytes`)
   -- these return plain Python bools/floats/strs/ints/bytes/dtype objects/tuples-of-ints with
   **no ndarray anywhere in the structure**, so there is no strides concept to violate. These
   are correctly excluded, not swept under a rug: N/A, not GENUINE/ARTIFACT/UNDETERMINED.
4. That triage plus a direct audit of every registry entry with a `numpy_adapter=` override
   (to catch the eig/eigh/qr-style bypass) produced the real candidate population: **28 items**
   whose result structure genuinely contains one or more arrays inside a tuple/list, or whose
   registry entry hides the real array-bearing return behind an adapter.
5. For each of the 28, wrote direct calls against `ionp`/`numpy` (bypassing the harness
   entirely) across layouts: C 2-D, F 2-D (via `.reshape().T`), 3-D full transpose, 3-D
   `swapaxes(0,1)` partial swap, stepped slice (`[:, ::2]`), negative-step slice (`[:, ::-1]`),
   and a length-1 axis (`reshape(1,24)`); plus dedicated square-matrix C/F/transpose/sliced
   layouts for the `linalg.*` group.
6. **Operand parity was checked, not assumed.** `tuple(np_input.strides) ==
   tuple(np.asarray(ionp_input).strides)` was asserted for every layout before trusting any
   output comparison. This caught a real problem (see "Contamination" below) that would
   otherwise have produced false GENUINE findings.
7. Recursively walked every result structure (tuple/list/dict/namedtuple-`_fields`) comparing,
   at every array leaf: `.shape`, `.dtype`, `.strides` (via `np.asarray(...)`, never trusting a
   self-reported `.strides` label -- same buffer-protocol discipline as
   `harness._strides_match`'s 2026-08-02 fix), and `tobytes('A')` (physical bytes) plus a
   separate `np.array_equal` **logical**-value check to distinguish "layout differs, values
   agree" from "values are actually wrong."
8. numpy 2.5.1 was the oracle throughout, called only from `/tmp` probe scripts, never from
   ionp source. No file under the `ionp` package tree, no `_state/*.py`, no test, was edited.
   `.venv/bin/python` used for every invocation; no bare `python3`.

`ionp/_ionp.abi3.so` mtime was checked at the start and end of this sweep:
`1785725762` both times, current time `1785726782` at the final check -- **the extension did
not change under this measurement**, so results below are not contaminated by a concurrent
rebuild. This is stated as a measured fact, not an assumption.

## Population found (28 candidates)

`divmod`, `ndarray.__divmod__`, `ndarray.__rdivmod__`, `frexp`, `modf`, `array_split`,
`dsplit`, `hsplit`, `split`, `vsplit`, `diag_indices`, `diag_indices_from`, `ix_`, `indices`,
`tril_indices`, `triu_indices`, `unravel_index`, `intersect1d`, `unique_all`, `unique_counts`,
`unique_inverse`, `meshgrid`, `linalg.lstsq`, `linalg.slogdet`, `linalg.eig`, `linalg.eigh`,
`linalg.qr`, `strings.partition`, `strings.rpartition`, plus `nonzero`/`ndarray.nonzero`/
`where` (1-argument form) discovered while probing the index-tuple family (nonzero-shaped
items were not in my first-pass 28 because the corpus's default `where`/`nonzero` cases in the
registry always use the 3-argument/boolean-mask forms that return a single array; the 1-arg
tuple-returning form is a separate call shape on an already-"exact" item, so it's included
here). Final tested set: **31 call-shapes across 28 declared items**.

## Verdicts

### GENUINE (14 call-shapes, all with a stated mechanism, all values-correct/layout-only
unless noted)

| Item | Mechanism | Concrete evidence |
|---|---|---|
| `frexp` | order='K' not honored: numpy propagates the input's memory layout to both outputs (mantissa, exponent); ionp always allocates fresh C-contiguous buffers for both. | Input F-ordered (4,6), strides `(8,32)` on both sides (parity confirmed). `numpy` mantissa strides `(8,32)`, `ionp` mantissa strides `(48,8)`; exponent (int32) `numpy=(4,16)` `ionp=(24,4)`. `np.array_equal` **True** on both mantissa and exponent (hand-verified: `/tmp/probe_frexp_detail.py`) -- values correct, only physical layout differs. Same mechanism as the already-documented 24-item reduction-output-layout GENUINE bucket, just previously invisible because the result is a tuple. |
| `modf` | Same order='K' mechanism as `frexp`. | F_2d: `numpy=(8,32)` `ionp=(48,8)` on both outputs; transpose_3d and swapaxes_3d reproduce the same pattern. |
| `array_split`, `split`, `hsplit`, `vsplit`, `dsplit` | **Not primarily a layout bug -- a memory-aliasing contract violation.** numpy's split family returns VIEWS into the original array (`np.shares_memory(part, original) == True`); ionp returns independent COPIES (`np.shares_memory == False` for all 5, hand-verified: `/tmp/probe_split_share.py`). The stride numbers the sweep reported (e.g. `hsplit` C_2d: `numpy=(48,8)` view-into-original vs `ionp=(24,8)` fresh-packed-copy) are a *symptom* of this, not the defect itself. **Flagged loudly per the task's own escalation rule: this is worse than a layout cosmetic difference** -- code that mutates a split part expecting it to write back into the source array (a documented, common numpy idiom) silently does nothing under ionp. (Separately and not in scope here: `ionp.ndarray` currently has no `__setitem__` at all, so in-place mutation of *any* ionp array, split-derived or not, isn't yet possible -- this makes the aliasing gap latent today but not less real as a contract difference once `__setitem__` lands.) |
| `nonzero`, `ndarray.nonzero`, `where` (1-arg form) | numpy computes all axes' index arrays together into one shared `(N, ndim)` buffer and returns each axis as a strided VIEW into it (stride = `ndim * itemsize`, not the axis array's own itemsize); `rn[0].base is rn[1].base` is **True** in real numpy (hand-verified). ionp returns independently-packed, naturally-strided arrays per axis. | 2-D boolean input, 8 True elements: `numpy` per-axis strides `(16,)` (itemsize 8 x ndim 2); `ionp` `(8,)`. 3-D case: `numpy=(24,)` (8x3), `ionp=(8,)`. `np.array_equal` **True** on every axis in every layout tested (hand-verified: `/tmp/handverify.py`). Reproduced across all 5 clean-parity layouts (C_2d, F_2d, transpose_3d, swapaxes_3d, len1_axis). |
| `unravel_index` (array-index input form) | Same shared-buffer-view mechanism as `nonzero`: numpy's per-axis outputs are strided views into one combined buffer. `unravel_index(scalar, shape)` (the scalar-index form) does **not** hit this -- confirmed clean separately. | `numpy` per-axis strides `(16,)` for a 2-target-shape unravel, `ionp` `(8,)`. Values equal (hand-verified). |
| `strings.partition`, `strings.rpartition` | Same family of bug as the shared-buffer-view items above, applied to string dtype: numpy's 3 returned fields (before/sep/after) are views into ONE combined record buffer -- every field's stride equals the *combined record width* (sum of all 3 field byte-widths), not its own itemsize; `rn[0].base is rn[1].base` is **True** in real numpy (hand-verified). ionp allocates 3 independent, naturally-strided arrays. | `strings.partition(["a-b-c","d-e-f-g","noSep","x-"], "-")`: `numpy` all 3 fields stride `44`; `ionp` strides `20, 4, 20` (each field's own itemsize). Values equal on all 3 fields (hand-verified: `/tmp/handverify.py`). |
| `linalg.qr(a, mode="raw")` | Only surfaces under the **non-default** `mode=` keyword -- `mode="reduced"` (default), `"complete"`, and `"r"` are all clean (see below). `mode="raw"` returns LAPACK's native packed-Householder representation, which numpy keeps in its natural (Fortran-ish) layout; ionp forces it C-contiguous. | 5x4 input: `numpy` h-array strides `(8,32)`, `ionp` `(40,8)`; tau vector clean on both. `np.array_equal` **True** (hand-verified: `/tmp/probe_qr_modes.py`). This is exactly the task's called-out "divergence only appears under a non-default keyword" category. |

### ARTIFACT (zero-size-stride-convention -- same class as the already-documented 150-item
bucket: numpy's rule for freshly-allocated zero-element arrays is all-zero strides regardless
of dtype/order; ionp gives plausible nonzero strides; `tobytes()` is `b''` on both sides, no
byte-level divergence)

| Item | Detail |
|---|---|
| `diag_indices(0)`, `diag_indices(0, ndim=3)` | `numpy=(0,)` strides, `ionp=(8,)`, shape `(0,)`, both empty. |
| `triu_indices` (degenerate `k` offset) | **Self-correction, disclosed per this task's own honesty requirement:** my first probe called `np.triu_indices(*shp)` with `shp=(4,4)`, which unpacks to `triu_indices(n=4, k=4)` -- **not** `n=4, m=4` as I intended (the real signature is `triu_indices(n, k=0, m=None)`). `k=4` on a 4x4 puts the diagonal offset entirely outside the matrix, producing a genuinely empty result on *both* sides -- so the test was still valid, just not testing what I meant to test. It reproduces the same zero-size-convention artifact as the other zero-size cases, nothing more. `tril_indices` hit the identical mis-unpacking (`k=4` there means "everything," not "nothing") and happened to land on a non-degenerate, clean case by accident (see CLEAN table). I did not silently fix and re-run this as if I'd meant it originally -- flagging the corpus-construction mistake here, exactly as the task's own worked example warned it would happen. |
| `linalg.lstsq` residuals output | Square (non-overdetermined) 4x4 system: numpy's residuals array is always empty by definition for a square/underdetermined system (real numpy semantics, not a corpus artifact) -- `numpy=(0,)`, `ionp=(8,)`, both empty. The solution array (root[0]) is clean on every layout tested. |
| `intersect1d(..., return_indices=True)` | **See "Sample vs. aggregate contradiction" below** -- this item's automated-sweep "divergence" turned out to be 100% an artifact of my own corpus never producing a non-empty intersection; it is CLEAN with real data. Listed here only because the empty-corpus case itself is a valid (if accidental) zero-size-convention artifact instance, not because the item has a real defect. |

### CLEAN (measured directly across all tested layouts, no divergence, hand-verified
operand parity)

- **`linalg.eig`, `linalg.eigh`, `linalg.qr` (modes `reduced`/`complete`/`r`), `linalg.slogdet`**
  -- these had genuinely never been checked in their raw return form before (the registry
  routes them through `numpy_adapter=`/`ionp_adapter=` invariant probes for value comparison,
  which structurally cannot see the real tuple's layout). Direct measurement across
  C/F/true-transpose/sliced square-matrix layouts, with operand-parity independently confirmed
  for all 4, found **zero stride divergence**. This is a real, previously-nonexistent
  measurement, not an inference from the harness passing.
- `meshgrid` (both `"xy"` and `"ij"` indexing) -- clean on all 7 layouts.
- `unique_all`, `unique_counts`, `unique_inverse` -- clean on all 7 layouts.
- `ix_`, `diag_indices` (n=1,4,7 -- non-degenerate), `diag_indices_from`, `tril_indices`,
  `unravel_index` (scalar-index form) -- clean.
- `intersect1d(..., return_indices=True)` with a real non-empty intersection -- clean (see
  below).
- `divmod`, `ndarray.__divmod__`, `ndarray.__rdivmod__` -- **see next section; this is the
  item the task brief itself opens with as an already-confirmed defect, and I could not
  reproduce it.**

### The `divmod` non-reproduction (reported honestly, not reconciled away)

The task brief states: *"I just confirmed `ndarray.__divmod__` — declared `exact` — carries
the identical defect on 4 of 10 layouts (numpy `(8,32)` vs ionp `(24,8)`, bytes EQUAL)."*

I ran `divmod`/`ndarray.__divmod__`/`ndarray.__rdivmod__` across: int64 and float64 dtypes;
base/transpose/swapaxes-3D layouts; array-scalar and array-array forms; and a targeted
reproduction attempt at the exact shape the stride numbers imply (`(8,32)` vs `(24,8)` decode
as F-ordered-(4,3) vs C-ordered-(4,3) -- I built that exact input via `arange(12).reshape(3,4).T`
and tested it explicitly, both as float64 and int64, both scalar- and array-operand forms) --
**zero divergence found in every case**, `bytes_eq=True` and `strides` identical on both
sides throughout (`/tmp/probe_divmod.py`). The extension's mtime was unchanged (see above)
across my entire measurement window, so this isn't a rebuild racing under me *during* this
sweep. Two honest possibilities, and I cannot distinguish them from where I sit: (1) the
concurrent agent (who owns the build lock per this task's own setup) fixed this exact defect
in a build that landed before I started measuring, or (2) the originally-observed case used an
input/call shape I didn't reconstruct correctly. I am not claiming ionp's divmod is fixed as a
general fact -- only that every reconstruction I tried, including a deliberate attempt to hit
the exact reported strides, came back clean. This is stated as a measured contradiction of the
brief's own opening claim, not smoothed over.

## Contamination found and excluded (operand-parity check earning its keep)

The task's method section explicitly warned against measuring a divergent *input* instead of
the item. That happened here: `layouts_1d()`'s `stepped_slice` (`a[:, ::2]`) and
`neg_step_slice` (`a[:, ::-1]`) views, built with **ionp's own slicing**, do **not** carry the
same strides as the equivalent real-numpy view --
`stepped_slice`: numpy `(48,16)` vs ionp `(24,8)` (ionp appears to materialize a repacked copy
under stepped slicing rather than a true strided view);
`neg_step_slice`: numpy `(48,-8)` vs ionp `(48,8)` (ionp drops the negative stride entirely).
This is a real ionp defect in its own right, but it is **`__getitem__`'s** defect, not the
multi-output item being tested through it, and it is out of this task's declared scope
(read-only diagnostic of *declared-exact multi-output items*, not of slicing). Every finding
above that could have been touched by this was re-derived from a clean-parity layout instead
(`C_2d`, `F_2d`, `transpose_3d`, `swapaxes_3d`, `len1_axis` all had verified identical operand
strides on both sides); the `stepped_slice`/`neg_step_slice`-labeled rows for `array_split`/
`split`/`vsplit`/`hsplit` in the raw sweep output are excluded as evidence here, not cited as
GENUINE-supporting data anywhere above, even though several of those items are independently
GENUINE (copy-not-view) via the shares_memory test, which used a plain unsliced base array.

## Hand-verified sample (independent bespoke probes, not the automated sweep script) --
required minimum was 10, delivered 12

1. `frexp` (`/tmp/probe_frexp_detail.py`) -- confirmed layout-only divergence, values equal.
2. `hsplit`/`vsplit`/`split`/`array_split`/`dsplit` (`/tmp/probe_split_share.py`) -- confirmed
   copy-not-view via `np.shares_memory`, one shared mechanism check across all 5.
3. `triu_indices` (`/tmp/probe_triu.py`) -- confirmed zero-size artifact, caught my own
   `*shp`-unpacking mistake in the process.
4. `unravel_index` array-form (`/tmp/handverify.py`) -- confirmed shared-buffer-view mechanism.
5. `intersect1d` (`/tmp/handverify.py`, two sub-cases: empty and real-overlap) -- **contradicted
   the aggregate**, see below.
6. `strings.partition` (`/tmp/handverify.py`) -- confirmed shared-record-buffer mechanism.
7. `nonzero` (`/tmp/handverify.py`) -- confirmed shared-buffer-view mechanism, `.base is .base`
   check.
8. `linalg.eig` (`/tmp/multi_output_sweep.py` + parity probe) -- confirmed clean.
9. `linalg.eigh` -- confirmed clean.
10. `linalg.qr` (all 4 modes, `/tmp/probe_qr_modes.py`) -- confirmed clean except `mode="raw"`.
11. `linalg.slogdet` -- confirmed clean.
12. `divmod`/`__divmod__`/`__rdivmod__` (`/tmp/probe_divmod.py`) -- confirmed clean, contradicts
    the task brief's opening claim (reported above, not silently reconciled).

**Sample vs. aggregate:** the hand sample **contradicted** the raw aggregate for one item:
the automated sweep flagged `intersect1d/return_indices` as diverging in **all 7** tested
layouts. Hand-verification with a corpus that actually produces a non-empty intersection
(`intersect1d([1,3,5,7,9,11], [3,7,11,99])`) showed **zero divergence** -- shape `(3,)`,
strides `(8,)` on both sides, bytes equal. The aggregate's "7/7 layouts diverging" number was
never measuring 7 different real behaviors; it was measuring the *same* zero-size-artifact
case 7 times because my layout-sweep corpus for this item never happened to produce overlapping
values regardless of which of the 7 array *layouts* I fed it (the layout loop varies memory
order, not data content, and I reused the same non-overlapping values across all 7). Per this
task's own instruction, this contradiction is reported instead of a tidied-up total:
**`intersect1d` is CLEAN as a declared item; the only true positive in that row is the
zero-size-convention artifact, already counted separately and correctly under ARTIFACT.**

## What I measured directly vs. what I inferred

**Measured directly** (real `ionp`/`numpy` calls in `/tmp` probes, strides/bytes/values read
via `np.asarray(...)`, buffer-protocol-verified, not trusting any self-reported `.strides`
label): every row in the GENUINE, ARTIFACT, and CLEAN tables above, including the full
`linalg.eig`/`eigh`/`qr`/`slogdet` raw-structure check that no prior harness run had ever
performed, and the `divmod` non-reproduction.

**Inferred, not independently re-measured per-layout**: that `frexp`/`modf`'s mechanism
generalizes to *every* layout beyond the specific ones tested (reasonable given it is the same
already-documented order='K' allocator behavior as the existing 24-item GENUINE bucket, but I
did not exhaustively sweep every dtype x every layout combination for these two specifically).
That the split-family copy-not-view behavior is unconditional (I checked `shares_memory` on
one representative case per function, not every layout) -- the *stride numbers* differ by
layout (consistent with "always copies, copy layout depends on ionp's own allocation choice
for that shape"), but I did not re-run the `shares_memory` check on all 7 layouts for all 5
functions individually; I consider this low-risk to generalize since "does the returned object
own its data" is a structural property of the implementation, not a per-call one, but it is
flagged here as inference rather than exhaustive measurement.

**A whole category I did not get to**: axis=tuple keyword variants for reduction-adjacent
multi-output functions (none of the 28 candidates take a tuple `axis=` in a way that changes
their multi-output shape), and generator/iterator-returning items -- I found none among the 424
exact items (`nditer`, `broadcast`, `vectorize`, `apply_along_axis` etc. are all **not**
currently declared `"exact"`, confirmed by direct membership check against
`ionp.__ion_state__`), so this category is empty for the current declared surface, not
unexamined.

## Summary counts

- Declared-exact items examined: 424 (from live `ionp.__ion_state__`, not a `_state/*.py`
  scrape).
- Real multi-output/non-single-array candidates found after triage: 28 items / 31 call-shapes.
- GENUINE: 14 call-shapes (13 layout-only + 1 more-severe aliasing-contract class covering the
  5 split functions).
- ARTIFACT: 4 items, all zero-size-stride-convention, same class as the existing 150-item
  bucket.
- CLEAN (directly measured, including items never previously checked at all due to adapter
  bypass): 15 items/call-shapes, including the `divmod` family which the task's own opening
  claim expected to be GENUINE.
- UNDETERMINED: 0 -- a mechanism was identified for every divergence found.
