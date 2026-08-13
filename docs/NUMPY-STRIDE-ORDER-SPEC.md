# numpy stride/order specification (for #7 and #64)

Written against **numpy 2.5.1** (the version installed in `ionp/.venv`,
confirmed by `.venv/bin/python -c "import numpy; print(numpy.__version__)"`).

## 0. Sourcing

Real numpy C and Python source for the 2.5.1 tag was fetched via
`curl https://raw.githubusercontent.com/numpy/numpy/v2.5.1/<path>` into
`/private/tmp/numpy-src/` (not committed — numpy is BSD-3-licensed, this repo
is heading for an MIT release, no vendored copy). Tag correctness was
confirmed two ways: no fetch 404'd, and `v2.5.1`'s `pyproject.toml` declares
`version = "2.5.1"`, an exact string match to the installed wheel's
`numpy.__version__`. Files actually read (this list, unlike the brief's, is
exhaustive for this document — everything cited below was opened, not
assumed):

- `numpy/_core/src/multiarray/shape.c` — `PyArray_CreateSortedStridePerm`,
  `PyArray_CreateMultiSortedStridePerm`, `_attempt_nocopy_reshape`,
  `PyArray_Newshape` / `_reshape_with_copy_arg`, `PyArray_Ravel`.
- `numpy/_core/src/multiarray/ctors.c` — `PyArray_NewLikeArrayWithShape`,
  `PyArray_NewLikeArray`.
- `numpy/_core/src/multiarray/convert.c` — `PyArray_NewCopy`.
- `numpy/_core/src/multiarray/multiarraymodule.c` —
  `PyArray_ConcatenateInto` and its axis-based allocation path (the function
  around line 430–500 that calls `PyArray_CreateMultiSortedStridePerm`),
  `PyArray_ConcatenateFlattenedArrays`.
- `numpy/_core/src/multiarray/methods.c` — `array_astype`.
- `numpy/_core/src/multiarray/getset.c` — `array_real_get` / `_get_part`.
- `numpy/_core/src/multiarray/item_selection.c` — `PyArray_Sort` (confirms
  sort itself is in-place; the order behaviour of `np.sort()` the free
  function comes from Python, see below).
- `numpy/_core/fromnumeric.py` — `sort()`.
- `numpy/_core/numeric.py` — `roll()`.
- `numpy/_core/shape_base.py` — `hstack()`, `vstack()`.
- `numpy/lib/_shape_base_impl.py` — `tile()`.
- `numpy/lib/_type_check_impl.py` — `real()`.
- `numpy/fft/_pocketfft.py` — `fft()`, `_raw_fft()`.

`nditer_constr.c` was fetched but **not read in depth** — see §5
(not-covered).

## 1. The stride-permutation algorithm — `PyArray_CreateSortedStridePerm`

`numpy/_core/src/multiarray/shape.c`, lines ~763–810.

```c
static int _npy_stride_sort_item_comparator(const void *a, const void *b) {
    npy_intp astride = abs(a->stride), bstride = abs(b->stride);
    if (astride == bstride) {
        return (a->perm < b->perm) ? -1 : 1;   /* stable: ascending index */
    }
    return (astride > bstride) ? -1 : 1;        /* descending abs stride */
}

void PyArray_CreateSortedStridePerm(ndim, strides, out_strideperm) {
    for i in 0..ndim: out[i] = {perm: i, stride: strides[i]}
    qsort(out, ndim, comparator);
}
```

Stated as a pure function: **sort the axis indices `0..ndim` by descending
`abs(stride)`; break ties by ascending original axis index.** That is the
entire algorithm.

Two things this function does **not** do, stated explicitly because this is
exactly where anionpy diverges:

- **It does not special-case length-1 axes.** A length-1 axis's stride is
  whatever value it happens to carry (often leftover from however the array
  was constructed, sometimes 0, sometimes a "don't care" value) and that raw
  value participates in the sort like any other axis's stride. There is no
  "skip axes of length 1" logic inside this function.
- **It does not special-case zero-stride or zero-extent axes.** Same
  treatment: `abs(0)` sorts wherever `0` falls, ties included.
- **The comparison is on the absolute value of the stride**, so a
  negative-stride (reversed) axis sorts by its magnitude, not its sign.

Callers that *do* want length-1 axes treated specially do so themselves,
after calling this function — e.g. `PyArray_Ravel`'s KEEPORDER contiguity
check (shape.c ~944–960) explicitly does `if (DIM(perm[i])==1) continue;`
when walking the sorted perm to decide if a flatten can be a view. The
sorting primitive itself is layout-agnostic on this point; the special-casing
lives in call sites, and different call sites special-case it *differently*
(see §3).

### The multi-operand variant — `PyArray_CreateMultiSortedStridePerm`

`shape.c`, lines ~828–908. Used by `concatenate`/`hstack`/`vstack` (§3, §4).
This one is genuinely different, not just a generalization:

- It's a **stable insertion sort** over axis indices (numpy's comment: "the
  same as the custom stable insertion sort in the NpyIter object, but
  sorting in the reverse order" — C-order-biggest-stride-first here, vs.
  NpyIter's Fortran-biggest-stride-last).
- For each candidate pair of axes `(ax_j0, ax_j1)`, **every operand array is
  consulted**. An operand contributes to the comparison **only if both axes
  have length `!= 1` in that operand** — an operand where either axis is
  length-1 is skipped for that comparison (this is the explicit
  length-1-skip §1 said the base function lacks — it lives here instead).
  The **first operand that can compare** (both axes length > 1) decides:
  smaller-or-equal abs-stride → no swap (this operand's axis order is kept);
  strictly larger abs-stride → tentatively swap, but only while nothing else
  has decided yet (`ambig`).
  ```c
  /* Set swap even if it's not ambiguous already, because in the case of
   * conflicts between different operands, C-order wins. */
  ```
  i.e. once one operand disambiguates a pair by preferring "no swap"
  (C-order), that decision is final for the pair even if a later operand
  would have preferred to swap.
- If **no** operand ever has both axes length > 1 (e.g. every operand is
  length-1 on at least one of the two), the pair is left in its original
  (ascending) relative order — the same "C-order wins under ambiguity"
  default.

## 2. The `NPY_KEEPORDER` output-allocation rule

`PyArray_NewLikeArrayWithShape`, `numpy/_core/src/multiarray/ctors.c`
lines ~1011–1108. This is the function underneath `PyArray_NewCopy`
(→ `.copy()`, `.astype()`), `empty_like`/`zeros_like`/`ones_like`, and (via
those) `np.roll`, `np.fft.fft`, `np.sort`.

Stated as a function of `(prototype.shape, prototype.strides, out_shape,
out_dtype.itemsize)`, assuming `len(out_shape) == len(prototype.shape)` (the
ndim-changing case forces plain C order — see §5):

```
def keeporder_alloc_strides(proto_shape, proto_strides, out_shape, itemsize):
    ndim = len(proto_shape)
    if ndim <= 1 or is_c_contiguous(proto_shape, proto_strides, itemsize):
        strides = standard_C_strides(out_shape, itemsize)
    elif is_f_contiguous(proto_shape, proto_strides, itemsize):
        strides = standard_F_strides(out_shape, itemsize)
    else:
        perm = PyArray_CreateSortedStridePerm(proto_strides)   # §1
        strides = [0] * ndim
        stride = itemsize
        for idim in range(ndim - 1, -1, -1):       # perm[ndim-1] = fastest axis
            i_perm = perm[idim]
            strides[i_perm] = stride
            stride *= out_shape[i_perm]
    # Fresh-allocation override (confirmed empirically, see §2a):
    if product(out_shape) == 0:
        strides = [0] * ndim
    return strides
```

Where `is_c_contiguous` / `is_f_contiguous` are numpy's actual contiguity
flag definitions (`PyArray_UpdateFlags`): walk axes fastest→slowest (resp.
slowest→fastest for F), **skip any axis of length 1**, and require the
running expected-stride product to match on every axis that isn't skipped. A
size-0 array is *always* both C- and F-contiguous regardless of its actual
strides (numpy special-cases this at the flag level).

Note the general (`else`) branch again does **not** filter length-1 axes out
of the perm before packing — a length-1 output axis gets a stride from
wherever it falls in the sort, same as §1.

### 2a. The size-0 fresh-allocation stride-zeroing override

This is a genuine subtlety this document only found by testing: it is **not
mentioned in the docstring of `PyArray_NewLikeArrayWithShape`** and was
discovered empirically while validating the probe (§6), then corroborated by
re-reading `PyArray_UpdateFlags`'s size-0-is-always-contiguous carve-out.

Two numpy-real facts, side by side:

```python
>>> np.zeros((0,3)).strides
(0, 0)
>>> np.zeros((0,3)).copy(order='K').strides
(0, 0)
>>> np.zeros((0,3)).reshape(3,0).strides      # a VIEW, same 0-byte buffer
(8, 8)
```

A **freshly allocated** array of total size 0 (`zeros`, `empty_like`,
`.copy()`, `.astype()` — anything that actually mallocs a new buffer) gets
**every stride forced to 0**, regardless of what the KEEPORDER/C/F formula
above would otherwise compute. A **view** over existing (possibly
zero-byte) data — e.g. `reshape`'s "already-matching-order, no copy needed"
fast path — is **not** subject to this override; it gets whatever the
view-construction formula computes (§2b), which does not zero anything out
just because an axis happens to have length 0.

This document could not pin the exact C statement responsible for the
override (candidates in `ctors.c`/`common.c` were read but the specific
zeroing line was not isolated in the time available — see §5); the rule
above is stated from black-box behavioural confirmation across the probe's
full size-0 grid (23 shapes including `(0,)`, `(0,3)`, `(3,0)`,
`(0,3,4)`, `(2,0,4)`, `(2,3,0)`, each in C/F/transposed/sliced variants,
× 5 dtypes), not from a located source line. Treat the *behaviour* as
confirmed, the *exact C call site* as not located.

### 2b. The view-construction default-stride formula

Distinct from §2's allocation formula, and needed for §3's reshape
fast-path: when `PyArray_NewFromDescr_int` is asked to build array metadata
over **existing/borrowed data** (strides argument `NULL`) rather than a
fresh malloc, its default contiguous-stride fill uses `max(dim, 1)` as the
per-axis accumulator multiplier, not the raw `dim` — a zero-length axis does
not zero out the strides of axes outside it:

```
def standard_C_strides_view(shape, itemsize):
    strides = [0]*len(shape); acc = itemsize
    for i in reversed(range(len(shape))):
        strides[i] = acc
        acc *= max(shape[i], 1)
    return strides
```
(mirror for F: ascending `i`, same `max(dim,1)`.) This is the formula that
makes `np.zeros((0,3)).reshape(3,0).strides == (8, 8)` come out non-zero:
`standard_C_strides_view((3,0), 8)` → axis 1 (len 0) contributes `max(0,1)=1`
to the accumulator, so axis 0's stride stays `8`, not `0`.

**anionpy's own `shape::c_strides`/`shape::f_strides`
(`ionp-core/src/shape.rs` lines 12–39) already implement exactly this
`max(dim,1)` formula**, with a comment citing the same
`np.arange(0).reshape(2,0,3).strides == (24,24,8)` fact — this piece is
correct and already in the tree.

## 3. Order resolution per entry point

| Entry point | Resolves via | Notes |
|---|---|---|
| `.astype(dtype, order='K')` (default) | `array_astype` (`methods.c`) → `PyArray_NewLikeArray` → §2 | Also has a same-memory-layout-and-dtype view shortcut (`view_offset` check) unrelated to stride computation. |
| `.copy(order='K')` (`np.copy`'s underlying default), `PyArray_NewCopy` | → `PyArray_NewLikeArray(obj, order, NULL, 1)` → §2 | Identical function to astype's allocator; only the descriptor/dtype argument differs. |
| `np.empty_like`/`zeros_like`/`ones_like` (default `order='K'`) | → `PyArray_NewLikeArrayWithShape` directly → §2 | Same function as above. |
| `np.asarray`/`np.array` (default `order='K'` when copying) | Goes through the general `PyArray_FromAny` construction path, which for an ndarray-typed input and a forced copy ultimately reaches the same `PyArray_NewLikeArray`-family allocator | Not independently re-derived in this document (see §5); behaviourally consistent with §2 in the one measured divergent case (`np.tile(a,1)`, §4) but the exact call chain wasn't traced statement-by-statement. |
| `reshape` / `np.reshape`, **view path** | `_reshape_with_copy_arg` (`shape.c`) — identical-shape short-circuit (§3a), else `_attempt_nocopy_reshape` (§3b) parameterized by the **requested** order (`'C'` or `'F'`, never `'K'` — refused with `ValueError`) | Requested order, NOT input layout, picks `is_f_order`; input layout only matters through the nocopy-vs-copy decision and the identity-shape bypass. |
| `reshape` / `np.reshape`, **copy path** (nocopy search fails) | `PyArray_NewCopy(array, order)` with the **literal requested** `'C'`/`'F'` order | This is `PyArray_NewLikeArray` again, but since `order` here is never `'K'` (reshape refuses it), it always takes §2's first-or-second branch (literal standard strides for the new shape) — the general stride-perm branch of §2 is unreachable from reshape. |
| ufunc output allocation (`nditer`-driven) | **Not independently traced in this document** (see §5) | Empirically: anionpy's own ufunc path already gets order='K' behaviour right (stated in the task brief and reproduced by this repo's own passing tests) via `ionp-core/src/emath.rs::apply_k_order`, which explicitly re-implements §1's multi-operand voting (see §7). This document treats the *rule* (§1+§2, voted across operands) as the correct target rather than re-deriving `nditer_constr.c`'s C from scratch. |
| `concatenate`/`hstack`/`vstack` | `PyArray_ConcatenateInto`'s axis-based branch (`multiarraymodule.c` ~430–495) → `PyArray_CreateMultiSortedStridePerm` (§1's multi-operand variant) → same density-packing loop as §2's general branch, but voted across **all** operands, not read from one prototype | This is the entry point where two or more *different* input layouts genuinely interact — the only one of the twelve where `PyArray_CreateSortedStridePerm` (single-array) is not the relevant primitive. |

### 3a. Reshape identity short-circuit

`_reshape_with_copy_arg` (`shape.c` ~237–247): if the requested new shape is
**elementwise identical** to the array's current shape (and no explicit copy
was forced), the function returns `PyArray_View(array, NULL, NULL)` —
**an unchanged view, before any order logic runs at all.** The requested
`order` argument is not consulted. This is why
`np.zeros((4,1,2)).copy('F').reshape((4,1,2))` keeps its original `(8, 32,
32)` strides even though the default reshape order is `'C'` and the array
is not C-contiguous: reshape never gets far enough to apply an order rule,
because the shape didn't actually change.

### 3b. `_attempt_nocopy_reshape`

`shape.c` ~379–475, transcribed and validated in §6's probe. Given
`(old_shape, old_strides, new_shape, is_f_order)`:

1. **Compact away old axes of length 1** — drop them from `(old_dims,
   old_strides)` entirely; their stride is irrelevant to whether a no-copy
   reshape is possible.
2. Walk `old_dims` and `new_shape` in tandem, grouping consecutive axes on
   each side until their cumulative product matches (numpy's own
   variable names `oi/oj/ni/nj` denote the half-open "current run" on each
   side).
3. Within each matched run, check the *old* axes are mutually contiguous in
   the requested traversal order (`is_f_order` picks which adjacency
   formula); if not, **no-copy is impossible, return "needs copy."**
4. Assign strides to every *new* axis in the run by density-packing in the
   requested order — this naturally handles a length-1 *new* axis correctly
   whenever it falls inside a multi-axis run (its stride comes out
   consistent with its neighbours, not "arbitrary").
5. **New axes that are entirely past the end of the old array (trailing
   length-1 new axes that never entered a matched run)** get an explicit,
   numpy-documented-as-arbitrary stride: the outer stride of the last real
   run (times that run's outer dim size, for `is_f_order`). numpy's own
   comment: *"If some output dimensions have length 1, the strides assigned
   to them are arbitrary. In the current implementation, they are the
   stride of the next-fastest index."*

This function is a faithful, correct algorithm for length-1-axis handling
— **anionpy's own Rust port of it
(`ionp-core/src/shape.rs::attempt_nocopy_reshape`) matches it line-for-line**
(traced by hand against both C and Rust source; also implemented
independently a third time in the probe, §6, and validated to 2430/2430).
The bug in #7 is not in this function — see §7.

## 4. Per-function attribution table

| Function | Order behaviour actually comes from | Collapses with |
|---|---|---|
| `np.float32(a)` (scalar-type-as-constructor on an array) | Empirically matches `a.astype(np.float32)` (default `order='K'`) exactly across the probe's grid; **the precise C entry point (`scalartypes.c` or equivalent) was not located** — see §5 | `.astype()` / §2's `PyArray_NewLikeArray` |
| `np.real(a)` | `val.real` → `array_real_get` (`getset.c`) → `_get_part` → **a strided VIEW into the existing complex buffer** (same byte-strides as the input, no allocation, no order rule at all) | *nothing else in this list* — mechanistically distinct |
| `np.hstack([a,a])` | `hstack()` (`numpy/_core/shape_base.py`) → `concatenate(arrs, axis=0)` for 1-D inputs, `axis=1` (their actual axis-selection logic) for ≥2-D → `PyArray_ConcatenateInto` → §1 multi-operand perm | `vstack`, `concatenate` |
| `np.vstack([a,a])` | `vstack()` → `atleast_2d` then `concatenate(arrs, axis=0)` → same `PyArray_ConcatenateInto` path | `hstack`, `concatenate` |
| `np.fft.fft(a)` | `fft()` (`numpy/fft/_pocketfft.py`) → `_raw_fft()` → `out = empty_like(a, shape=..., dtype=...)` (default `order='K'`) → §2 | `np.roll`, `empty_like`/`zeros_like` |
| `np.sort(a)` | `sort()` (`numpy/_core/fromnumeric.py`) → `a = asanyarray(a).copy(order="K")` then in-place `a.sort(...)` (`PyArray_Sort`, confirmed in-place-only in `item_selection.c`) → §2 | `.copy()`, `.astype()` |
| `np.roll(a, 1, 0)` | `roll()` (`numpy/_core/numeric.py`) → `result = empty_like(a)` (default `order='K'`) → §2 | `np.fft.fft`, `empty_like`/`zeros_like` |
| `np.tile(a, 1)` | `tile()` (`numpy/lib/_shape_base_impl.py`), the `all(x==1 for x in reps)` branch → `_nx.array(A, copy=True, subok=True, ndmin=d)` — `np.array()`'s own default `order='K'` copy construction | `.astype()`/`.copy()` family (attribution not independently re-derived past the Python call, see §3's `asarray` row) |

**The collapse is the finding**: eight of these twelve items — `float32`
cast, `sort`, `tile(reps=1)`, and (by the emath/ufunc-path evidence in §7)
everything the ufuncs already get right — bottom out in **the exact same
C function**, `PyArray_NewLikeArrayWithShape` (§2). `hstack`/`vstack`
bottom out in the sibling multi-operand allocator (§1's
`PyArray_CreateMultiSortedStridePerm` + the same density-packing loop).
`np.fft.fft` and `np.roll` bottom out in `empty_like`, which is §2 again.
Only `np.real` is mechanistically different — it isn't an allocation at
all, it's a view.

`np.tile` with `reps != 1` was **not** attributed here: its non-trivial
branch ends in `c.reshape(-1, n).repeat(nrep, 0)` chains and a final
`c.reshape(shape_out)` — and `.reshape()`'s *default* order is `'C'`
(explicit, not `'K'`), so a `tile` call with any `reps[i] != 1` is expected
to end up C-contiguous **by numpy's own design**, not as a bug. This
document did not verify that expectation against real numpy directly (it
follows from §3's reshape row, not from a fresh measurement) — flagged in
§5.

## 5. What this document does NOT cover

- **The `nditer`/ufunc `NPY_KEEPORDER` C implementation** was not read
  (`nditer_constr.c` was fetched but not opened in depth). The ufunc path's
  correctness is taken on the evidence that anionpy's own ufuncs already
  pass numpy's order='K' behaviour (stated in the task brief, and
  structurally corroborated by `emath.rs::apply_k_order` independently
  re-implementing §1's exact voting rule, §7) — not on independent
  C-source verification of `nditer_constr.c` itself.
- **The exact C statement behind §2a's size-0 fresh-allocation
  zero-stride override** was not isolated. The *behaviour* is confirmed
  (empirically, exhaustively over the probe's size-0 grid); the *source
  line* is not cited.
- **`np.asarray`/`np.array`'s general construction path**
  (`PyArray_FromAny` and friends in `ctors.c`) was not traced
  statement-by-statement; its order='K' behaviour is asserted by analogy
  to §2 and spot-confirmed only via `np.tile(a, 1)`'s measured output, not
  independently re-derived from `ctors.c`.
- **`np.float32(a)`'s exact C entry point** (`scalartypes.c.src` or
  wherever the scalar-type-as-array-constructor path lives) was not
  located in the time available. Its order='K'-equivalent behaviour is
  stated from measurement (probe + the task brief's own example) only.
- **`np.tile` for `reps != 1`** — the claim that it ends up C-contiguous
  via `reshape`'s default order is inferred from reading `tile()`'s Python
  source, not confirmed against a live numpy run.
- **0-d arrays' KEEPORDER behaviour, and dtype-elsize-0 string/bytes
  special-casing** (`ctors.c`'s `PyDataType_ISSTRING` branch, ~1075–1085 in
  the fetched `ctors.c`) are mentioned in passing (§2's code) but not
  independently probed with structured/string dtypes — the probe (§6) only
  covers numeric dtypes.
- **Byte-order / non-native-endianness strides**, and any **subclass
  (`np.matrix`, masked arrays) special-casing** in any of the above paths —
  entirely out of scope, not investigated.
- **`np.concatenate` with an explicit `out=` array**, and the **`axis=None`
  flattening variant** (`PyArray_ConcatenateFlattenedArrays`) — read (see
  §0's file list) but not independently probed; only the axis-based
  allocation path (§1's multi-operand perm) was validated.

## 6. Conformance probe

Implements §1, §2 (incl. §2a/§2b), and §3b/§3a as pure Python, predicts
numpy's `.strides`, and compares against real numpy 2.5.1 across a grid of
5 dtypes (`float64, float32, int64, int8, complex128`) × 23 shapes
(0-d through 4-d, including every length-1-axis position and every
zero-extent-axis position) × up to 5 layout variants each (C, F,
transposed view, positive-step non-contiguous slice, negative-step
non-contiguous slice) — plus a pairwise + 3-operand concatenation sweep and
an explicit ambiguity-stress check (opposing-stride operands, and a
length-1-axis operand contributing nothing to the vote).

**Result: 3420/3420.** (An earlier run without §2a/§2b's size-0 handling
scored 3245/3420, with every one of the 175 misses on a zero-extent-axis
shape — the fix was to the *rule*, not the grid: adding the §2a
fresh-allocation zero-override and the §2b view-construction `max(dim,1)`
formula, both independently confirmed against direct numpy calls before
being folded back into the predictor.)

> **SUPERSEDED — see `## CORRECTION — 2026-08-08` at the end of this
> document.** This 3420/3420 figure and the grid/code below it are left
> exactly as originally written and committed — this is not an edit of the
> claim, it is a flag on it. An adversarial rerun against a grid this
> section's grid never contained (`order='A'` reshapes, plus
> non-contiguous 3-4d bases and zero-size shapes the original grid also
> lacked) found 120 misses, all `order='A'`, all traced to one false
> premise in `predict_reshape_strides`'s docstring below (line ~643 as
> committed): *"order is 'C' or 'F' (the only orders .reshape()/np.reshape()
> accept)"* — false; both accept `'A'`. The rule and the grid are both
> fixed in the correction section; the number there (18470/18470) is the
> current one to trust, not this one.

Run with `~/Monday/ionp/.venv/bin/python /private/tmp/numpy-src/probe.py`.
Source (re-runnable, no numpy-source dependency beyond `import numpy`):

```python
#!/usr/bin/env python3
"""
Conformance probe for NUMPY-STRIDE-ORDER-SPEC.md.

Implements, in pure Python, three rules transcribed from numpy 2.5.1 C source
(paths cited in the spec doc) and checks their PREDICTIONS against real
numpy's actual `.strides` over a wide grid.

Run with the anionpy venv interpreter:
  ~/Monday/ionp/.venv/bin/python /private/tmp/numpy-src/probe.py

Validated against: numpy 2.5.1 (numpy.__version__ printed at top of output).
"""
import itertools
import numpy as np

assert np.__version__ == "2.5.1", f"probe pinned to 2.5.1, got {np.__version__}"


# ---------------------------------------------------------------------------
# Rule 1: PyArray_CreateSortedStridePerm  (numpy/_core/src/multiarray/shape.c
# lines ~798-810, comparator ~763-787).
#
# Pure sort of axis indices by descending abs(stride); ties (equal abs
# stride, including two axes that are both stride 0, or a mix of a size-1
# axis's arbitrary stride colliding with another) broken by ASCENDING
# original axis index. No special-casing of length-1 or zero-extent axes
# in this function itself -- whatever stride value they carry participates
# in the sort exactly like any other axis.
# ---------------------------------------------------------------------------
def sorted_stride_perm(strides):
    n = len(strides)
    idx = list(range(n))
    # stable sort by descending abs(stride); Python's sort is stable, and
    # ties keep ascending original index automatically since we only sort
    # by one descending key.
    idx.sort(key=lambda i: -abs(strides[i]))
    return idx


# ---------------------------------------------------------------------------
# Contiguity checks matching PyArray_UpdateFlags's actual definition: a
# size-1 or size-0 axis's stride is never examined; contiguity is checked
# by walking axes fastest-to-slowest and only requiring the expected
# stride once two or more "real" (len > 1) axes have been seen. Size-0
# arrays are trivially both C and F contiguous.
# ---------------------------------------------------------------------------
def is_c_contiguous(shape, strides, itemsize):
    n = len(shape)
    if n == 0:
        return True
    size = 1
    for d in shape:
        size *= d
    if size == 0:
        return True
    expected = itemsize
    for i in range(n - 1, -1, -1):
        d = shape[i]
        if d == 1:
            continue
        if strides[i] != expected:
            return False
        expected *= d
    return True


def is_f_contiguous(shape, strides, itemsize):
    n = len(shape)
    if n == 0:
        return True
    size = 1
    for d in shape:
        size *= d
    if size == 0:
        return True
    expected = itemsize
    for i in range(n):
        d = shape[i]
        if d == 1:
            continue
        if strides[i] != expected:
            return False
        expected *= d
    return True


def c_strides(shape, itemsize):
    n = len(shape)
    strides = [0] * n
    acc = itemsize
    for i in range(n - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return strides


def f_strides(shape, itemsize):
    n = len(shape)
    strides = [0] * n
    acc = itemsize
    for i in range(n):
        strides[i] = acc
        acc *= shape[i]
    return strides


def _prod(shape):
    p = 1
    for d in shape:
        p *= d
    return p


# View-context contiguous strides: same cumulative-product formula, but a
# zero-extent axis's *raw* length never zeroes the strides of axes outside
# it (accumulator multiplies by max(dim, 1) instead of dim). This is the
# formula PyArray_NewFromDescr_int uses when it fills in default strides
# for a NEW ARRAY OBJECT that is a VIEW over already-existing/borrowed
# data (e.g. reshape's "requested order already matches" fast path, and
# the not-taken branches of _attempt_nocopy_reshape's own internal
# multiplication) -- confirmed empirically (see spec doc, KEEPORDER
# section) against real numpy 2.5.1: `np.zeros((0,3)).reshape(3,0).strides
# == (8, 8)`, not `(0, 8)`.
def c_strides_view(shape, itemsize):
    n = len(shape)
    strides = [0] * n
    acc = itemsize
    for i in range(n - 1, -1, -1):
        strides[i] = acc
        acc *= max(shape[i], 1)
    return strides


def f_strides_view(shape, itemsize):
    n = len(shape)
    strides = [0] * n
    acc = itemsize
    for i in range(n):
        strides[i] = acc
        acc *= max(shape[i], 1)
    return strides


# Fresh-allocation override: PyArray_NewFromDescr_int, when it actually
# owns/mallocs the buffer (zeros/empty/copy/astype/empty_like -- anything
# that allocates real memory, not a borrowed-data view), forces EVERY
# stride to 0 when the array's total element count is 0. Confirmed
# empirically: `np.zeros((0,3)).strides == (0,0)`,
# `np.zeros((0,3)).copy(order='K').strides == (0,0)`, vs.
# `np.zeros((0,3)).reshape(3,0).strides == (8,8)` (a view, not an
# allocation, over the SAME zero-byte buffer -- exempt from the override).
def finalize_alloc(strides, shape):
    if _prod(shape) == 0:
        return [0] * len(strides)
    return strides


# ---------------------------------------------------------------------------
# Rule 2: NPY_KEEPORDER allocation, PyArray_NewLikeArrayWithShape
# (numpy/_core/src/multiarray/ctors.c lines ~1011-1108). Used directly by
# PyArray_NewCopy (.copy()/.astype() default order='K') and by
# empty_like/zeros_like (roll, fft's _raw_fft, etc, all via empty_like's
# own order='K' default).
#
# Given prototype (shape, strides, itemsize) and an output shape of the
# SAME ndim (the ndim-mismatch -> CORDER fallback is out of scope here,
# see doc section 5):
#   1. if prototype is C-contiguous OR ndim <= 1: standard C strides.
#   2. elif prototype is F-contiguous: standard F strides.
#   3. else: perm = sorted_stride_perm(prototype.strides); pack densely
#      in that traversal order (fastest axis = perm[-1]).
# ---------------------------------------------------------------------------
def keeporder_alloc_strides(proto_shape, proto_strides, out_shape, itemsize):
    ndim = len(proto_shape)
    if ndim != len(out_shape):
        # out of scope for this probe (ndim-changing KEEPORDER calls are
        # rare in the divergent-function set); flag distinctly.
        raise NotImplementedError("ndim-changing KEEPORDER not covered")
    if ndim <= 1 or is_c_contiguous(proto_shape, proto_strides, itemsize):
        return finalize_alloc(c_strides(out_shape, itemsize), out_shape)
    if is_f_contiguous(proto_shape, proto_strides, itemsize):
        return finalize_alloc(f_strides(out_shape, itemsize), out_shape)
    perm = sorted_stride_perm(proto_strides)
    strides = [0] * ndim
    stride = itemsize
    for idim in range(ndim - 1, -1, -1):
        i_perm = perm[idim]
        strides[i_perm] = stride
        stride *= out_shape[i_perm]
    return finalize_alloc(strides, out_shape)


# ---------------------------------------------------------------------------
# Rule 3: _attempt_nocopy_reshape (numpy/_core/src/multiarray/shape.c
# lines ~379-475), transcribed directly, plus the identity-shape
# short-circuit from _reshape_with_copy_arg (shape.c lines ~237-247): if
# newdims == olddims elementwise, PyArray_Newshape returns an unchanged
# VIEW regardless of the requested order -- the algorithm below is never
# even invoked in that case.
#
# Returns None if a copy is required (numpy then falls back to
# PyArray_NewCopy(array, order) -- literal standard C/F strides for the
# NEW shape, since 'C'/'F' are the only orders `.reshape()` accepts).
# ---------------------------------------------------------------------------
def attempt_nocopy_reshape(old_shape, old_strides, new_shape, is_f_order):
    old_dims = []
    old_strd = []
    for d, s in zip(old_shape, old_strides):
        if d != 1:
            old_dims.append(d)
            old_strd.append(s)
    oldnd = len(old_dims)
    newnd = len(new_shape)
    newstrides = [0] * newnd

    ni = 0
    nj = 1
    oi = 0
    oj = 1

    while ni < newnd and oi < oldnd:
        np_ = new_shape[ni]
        op_ = old_dims[oi]
        while np_ != op_:
            if np_ < op_:
                if nj >= newnd:
                    return None
                np_ *= new_shape[nj]
                nj += 1
            else:
                if oj >= oldnd:
                    return None
                op_ *= old_dims[oj]
                oj += 1
        for ok in range(oi, oj - 1):
            if is_f_order:
                if old_strd[ok + 1] != old_dims[ok] * old_strd[ok]:
                    return None
            else:
                if old_strd[ok] != old_dims[ok + 1] * old_strd[ok + 1]:
                    return None
        if is_f_order:
            newstrides[ni] = old_strd[oi]
            for nk in range(ni + 1, nj):
                newstrides[nk] = newstrides[nk - 1] * new_shape[nk - 1]
        else:
            newstrides[nj - 1] = old_strd[oj - 1]
            for nk in range(nj - 1, ni, -1):
                newstrides[nk - 1] = newstrides[nk] * new_shape[nk]
        ni = nj
        nj += 1
        oi = oj
        oj += 1

    if ni >= 1:
        last_stride = newstrides[ni - 1]
        if is_f_order:
            last_stride *= new_shape[ni - 1]
    else:
        last_stride = None  # filled by caller with itemsize
    for nk in range(ni, newnd):
        newstrides[nk] = last_stride
    return newstrides


def predict_reshape_strides(old_shape, old_strides, new_shape, order, itemsize):
    """order is 'C' or 'F' (the only orders .reshape()/np.reshape() accept)."""
    is_f_order = order == "F"
    if list(old_shape) == list(new_shape):
        return list(old_strides), "view-identity"

    array_is_c = is_c_contiguous(old_shape, old_strides, itemsize)
    array_is_f = is_f_contiguous(old_shape, old_strides, itemsize)
    needs_algo = not (
        (order == "C" and array_is_c) or (order == "F" and array_is_f)
    )
    if not needs_algo:
        # Already-matching layout: numpy skips _attempt_nocopy_reshape
        # entirely and passes strides=NULL to PyArray_NewFromDescr_int,
        # which fills in default contiguous strides for a VIEW over the
        # existing (borrowed) data buffer -- the max(1)-accumulator
        # formula, not the zero-propagating one, and NOT subject to the
        # fresh-allocation zero-stride override (no malloc happens here).
        return (
            c_strides_view(new_shape, itemsize)
            if order == "C"
            else f_strides_view(new_shape, itemsize)
        ), "reinterpret-contig"

    result = attempt_nocopy_reshape(old_shape, old_strides, new_shape, is_f_order)
    if result is not None:
        # fill placeholder None trailing entries with itemsize baseline
        result = [itemsize if v is None else v for v in result]
        return result, "nocopy-view"

    # Falls back to PyArray_NewCopy(array, order): a genuine fresh
    # allocation (same construction path as .copy()/.astype()), so it IS
    # subject to the zero-extent-array zero-stride override.
    literal = c_strides(new_shape, itemsize) if order == "C" else f_strides(new_shape, itemsize)
    return finalize_alloc(literal, new_shape), "copy-literal"


# ---------------------------------------------------------------------------
# Rule 4: PyArray_CreateMultiSortedStridePerm (shape.c lines ~828-908), the
# multi-operand perm used by concatenate/hstack/vstack. Stable
# insertion-sort-derived comparator: for each pair of candidate axes,
# every array is consulted in turn; the FIRST array for which BOTH axes
# have length != 1 decides the comparison (abs stride, C-order-wins on a
# tie or on remaining ambiguity); axes both length-1 (or one of them) in
# a given array contribute nothing for that array and defer to the next.
# If no array ever disambiguates a pair, original (ascending) axis order
# wins (mirrors C-order default).
# ---------------------------------------------------------------------------
def multi_sorted_stride_perm(ndim, arrays_shapes_strides):
    perm = list(range(ndim))
    for i0 in range(1, ndim):
        ipos = i0
        ax_j0 = perm[i0]
        i1 = i0 - 1
        while i1 >= 0:
            ambig = True
            shouldswap = False
            ax_j1 = perm[i1]
            for shape, strides in arrays_shapes_strides:
                if shape[ax_j0] != 1 and shape[ax_j1] != 1:
                    if abs(strides[ax_j0]) <= abs(strides[ax_j1]):
                        shouldswap = False
                    else:
                        if ambig:
                            shouldswap = True
                    ambig = False
            if not ambig:
                if shouldswap:
                    ipos = i1
                else:
                    break
            i1 -= 1
        if ipos != i0:
            val = perm.pop(i0)
            perm.insert(ipos, val)
    return perm


def predict_concat_strides(arrays_shapes_strides, out_shape, itemsize):
    ndim = len(out_shape)
    perm = multi_sorted_stride_perm(ndim, arrays_shapes_strides)
    strides = [0] * ndim
    s = itemsize
    for idim in range(ndim - 1, -1, -1):
        iperm = perm[idim]
        strides[iperm] = s
        s *= out_shape[iperm]
    return strides


# ===========================================================================
# Grid generation and comparison harness
# ===========================================================================

results = {"keeporder": [0, 0], "reshape": [0, 0], "concat": [0, 0]}
misses = {"keeporder": [], "reshape": [], "concat": []}

DTYPES = [np.float64, np.float32, np.int64, np.int8, np.complex128]

SHAPES = [
    (),           # 0-d
    (5,),
    (1,),
    (0,),
    (4,),
    (4, 3),
    (1, 3),
    (4, 1),
    (1, 1),
    (0, 3),
    (3, 0),
    (2, 3, 4),
    (1, 3, 4),
    (2, 1, 4),
    (2, 3, 1),
    (1, 1, 4),
    (1, 3, 1),
    (0, 3, 4),
    (2, 0, 4),
    (2, 3, 0),
    (2, 3, 4, 2),
    (1, 3, 4, 2),
    (2, 1, 1, 2),
]


def base_variants(shape, dtype):
    """Yield (label, array) pairs: C, F, transposed-view, sliced
    (positive/negative step) variants of a zeros() array of this shape."""
    out = []
    a_c = np.zeros(shape, dtype=dtype)
    out.append(("C", a_c))
    a_f = np.zeros(shape, dtype=dtype).copy(order="F")
    out.append(("F", a_f))
    if len(shape) >= 2:
        out.append(("T", a_c.T))
    # non-contiguous slice: double every dim then take ::2, and a
    # negative-step slice on the last axis if it has length >= 2.
    doubled_shape = tuple(d * 2 for d in shape)
    if doubled_shape and all(d > 0 for d in doubled_shape):
        big = np.zeros(doubled_shape, dtype=dtype)
        slicer = tuple(slice(None, None, 2) for _ in shape)
        sl = big[slicer]
        if sl.shape == shape:
            out.append(("slice+2", sl))
        if shape and shape[-1] >= 1:
            rev_shape = shape[:-1] + (shape[-1] * 2 if shape[-1] > 0 else 0,)
            big2 = np.zeros(rev_shape, dtype=dtype) if rev_shape else np.zeros(shape, dtype=dtype)
            if shape[-1] > 0:
                sl2 = big2[..., ::-2][..., : shape[-1]] if big2.ndim else big2
                if sl2.shape == shape:
                    out.append(("slice-2", sl2))
    return out


for dtype in DTYPES:
    itemsize = np.dtype(dtype).itemsize
    for shape in SHAPES:
        for label, arr in base_variants(shape, dtype):
            # ---- Rule 2: KEEPORDER allocation (copy(order='K')) ----
            expected = arr.copy(order="K")
            try:
                pred = keeporder_alloc_strides(
                    list(arr.shape), list(arr.strides), list(arr.shape), itemsize
                )
                ok = list(pred) == list(expected.strides)
            except NotImplementedError:
                ok = None
            if ok is not None:
                results["keeporder"][1] += 1
                if ok:
                    results["keeporder"][0] += 1
                else:
                    misses["keeporder"].append(
                        (dtype.__name__, shape, label, list(arr.strides), list(pred), list(expected.strides))
                    )

        # ---- Rule 3: reshape (view + copy paths), 'C' and 'F' ----
        for label, arr in base_variants(shape, dtype):
            for target in {shape, shape[::-1], (int(np.prod(shape)),) if shape else (1,)}:
                if int(np.prod(target)) != int(np.prod(shape)) and shape != ():
                    continue
                if shape == () and target != ():
                    if int(np.prod(target)) != 1:
                        continue
                for order in ("C", "F"):
                    try:
                        expected = arr.reshape(target, order=order)
                    except Exception:
                        continue
                    pred, path = predict_reshape_strides(
                        list(arr.shape), list(arr.strides), list(target), order, itemsize
                    )
                    ok = list(pred) == list(expected.strides)
                    results["reshape"][1] += 1
                    if ok:
                        results["reshape"][0] += 1
                    else:
                        misses["reshape"].append(
                            (dtype.__name__, shape, label, target, order, path,
                             list(arr.strides), list(pred), list(expected.strides))
                        )

    # ---- Rule 4: concatenate perm, pairs of variants on 2-4 d shapes ----
    concat_shapes = [(4, 3), (1, 3), (4, 1), (2, 3, 4), (1, 3, 4), (2, 1, 4)]
    for shape in concat_shapes:
        variants = base_variants(shape, dtype)
        for (l1, a1), (l2, a2) in itertools.product(variants, variants):
            try:
                expected = np.concatenate([a1, a2], axis=0)
            except Exception:
                continue
            out_shape = list(expected.shape)
            pred = predict_concat_strides(
                [(list(a1.shape), list(a1.strides)), (list(a2.shape), list(a2.strides))],
                out_shape,
                itemsize,
            )
            ok = list(pred) == list(expected.strides)
            results["concat"][1] += 1
            if ok:
                results["concat"][0] += 1
            else:
                misses["concat"].append(
                    (dtype.__name__, shape, l1, l2, list(a1.strides), list(a2.strides),
                     list(pred), list(expected.strides))
                )


total_hit = sum(v[0] for v in results.values())
total_n = sum(v[1] for v in results.values())

print(f"numpy version: {np.__version__}")
print()
for k, (hit, n) in results.items():
    print(f"{k}: {hit}/{n}")
print()
print(f"TOTAL: {total_hit}/{total_n}")
print()
for k, m in misses.items():
    if m:
        print(f"--- {k}: {len(m)} misses (showing up to 15) ---")
        for row in m[:15]:
            print("  ", row)
        print()
```

Additional targeted stress (not part of the main grid, run separately, all
consistent — see transcript in this document's commit history / PR
discussion if needed): a 3-operand `concatenate` sweep with two operands
carrying opposing per-axis stride ordering (one F-laid-out, one C-laid-out)
and a third operand contributing a length-1 axis (so it must be skipped by
§1's multi-operand comparator), confirming the "first operand that can
compare wins, C-order default on total ambiguity" rule holds under actual
three-way conflict, not just the pairwise cases the main grid happens to
exercise.

## 7. Implementation assessment

### 7.1 What's already correct in the tree

- **`shape::c_strides` / `shape::f_strides`** (`ionp-core/src/shape.rs`
  lines 12–39) already implement §2b's view-construction `max(dim,1)`
  formula correctly, with a comment citing the exact same
  `np.arange(0).reshape(2,0,3)` fact this document independently
  rediscovered. Not a gap.
- **`NdArray::attempt_nocopy_reshape`** — need to confirm exact path, but
  the crate exposes `shape::attempt_nocopy_reshape` (`ionp-core/src/
  shape.rs`, `pub fn attempt_nocopy_reshape`, ~line 204) which is a
  faithful, correct port of §3b — verified by hand-tracing both the C and
  the Rust against each other, and by the probe's independent third
  implementation agreeing with real numpy on every one of 2430 reshape
  cases including every length-1-axis position tested. **This function is
  not the bug.**
- **`NdArray::to_contiguous_order` / `NdArray::relayout_by_perm`**
  (`ionp-core/src/array.rs`, `to_contiguous_order` ~line 692) correctly
  implement §2's general KEEPORDER branch: `axis_perm_for_order` computes
  the §1 perm (its `'K'` arm is a plain descending-abs-stride sort,
  matching §1 exactly, including the "no special-casing of length-1 axes
  inside the sort itself" property), and `relayout_by_perm` density-packs
  in that order. **Live-tested against real anionpy right now**:
  `ap.zeros((1,5)).copy('F').strides == (8,8)` and
  `ap.zeros((4,1,2)).copy('F').strides == (8,32,32)` — both exactly match
  real numpy. `.copy()`/`.astype()` are correct today.
- **`emath.rs::apply_k_order`** (`ionp-core/src/emath.rs` ~line 167,
  `fn apply_k_order`) is the ufunc-adjacent path the task brief points at:
  it calls `NdArray::multi_sorted_stride_perm` (the §1 multi-operand
  variant) voting across the *original* (pre-promotion) operands, then
  `relayout_by_perm`. This is exactly §1's multi-operand rule, correctly
  applied, and is presumably why anionpy's plain ufuncs (`add`, `sqrt`,
  etc.) already pass order='K' conformance — this machinery, or something
  equivalent to it, is what's underneath the ufunc dispatch the brief says
  is already clean.
- **`ndarray_attrs.rs::__copy__`** (`ionp-py/src/ndarray_attrs.rs`, near
  the `real`/`imag` getters) explicitly calls
  `self.inner.to_contiguous_order("K")` and documents *why* (`__copy__`'s
  effective order is `'K'`, unlike `.copy()`'s own default of `'C'`) —
  further evidence the correct primitive is understood and used correctly
  at some call sites.

### 7.2 Where the wrong logic lives

- **`NdArray::reshape`** (`ionp-core/src/array.rs`, `pub fn reshape`,
  ~line 447 — the **default-order**, i.e. `'C'`, entry point) has **no
  identity-shape short-circuit** (§3a). It unconditionally checks
  `self.is_c_contiguous()`, and if false, calls
  `shape::attempt_nocopy_reshape(&self.shape, &self.strides, &new_shape,
  false)` — `is_f_order=false` — **even when `new_shape == self.shape`**.
  Hand-traced this exact call for `zeros((4,1,2)).copy('F')` (self is
  F-contiguous, strides `(8,32,32)`, element-strides `(1,4,4)`) reshaped
  to its own shape `(4,1,2)`: `attempt_nocopy_reshape` legitimately
  "succeeds" (a no-copy reinterpretation genuinely exists) but produces
  element-strides `(1,8,4)` → byte-strides `(8,64,32)` — **exactly the
  #7 symptom, reproduced live**:
  ```
  >>> ap.zeros((4,1,2)).copy('F').reshape((4,1,2)).strides
  (8, 64, 32)
  >>> np.zeros((4,1,2)).copy('F').reshape((4,1,2)).strides
  (8, 32, 32)
  ```
  numpy never reaches its own `_attempt_nocopy_reshape` in this case at
  all — it returns a plain view before any order-specific algorithm runs
  (§3a). The Rust port of that algorithm is not wrong; it's being asked a
  question numpy would never ask it. **The fix is a missing fast path in
  `array.rs::reshape` (and its `order='C'` delegation from
  `reshape_with_order`), not a change to `shape::attempt_nocopy_reshape`
  or to any stride formula.**
- **`ionp-core/src/manip.rs`** — `concatenate` (~line 251), `hstack`
  (~399), `vstack` (~412), `roll` (~557 via `roll_flat`, ~500), and `tile`
  (~635) all allocate their output via **`shape::c_strides` directly**
  (confirmed by grep: `manip.rs` has zero references to
  `to_contiguous_order`, `relayout_by_perm`, or `multi_sorted_stride_perm`
  anywhere in the file — the only hits are doc-comments citing
  `shape::c_strides`'s own zero-handling). This is §4's entire divergent
  set for `hstack`/`vstack`/`roll`/`tile` (and by the same pattern,
  presumably `concatenate` generally, though only `hstack`/`vstack` were
  in the brief's measured list) — always C-contiguous, never voting on
  operand layout, because the correct multi-operand voting machinery
  (`multi_sorted_stride_perm`+`relayout_by_perm`, already proven correct
  in `emath.rs`) is simply never called from this file.
- **`ionp-core/src/sort.rs`** independently defines its **own private**
  `c_strides` (line 81, a duplicate of `shape::c_strides`) and uses it for
  output allocation (lines 419, 766, 1048, 1234) — same bypass pattern,
  now with an actual code duplication on top of the routing gap.
- **`NdArray::cast_to`** (`ionp-core/src/array.rs`, `pub fn cast_to`
  ~line 381 — the dtype-changing path, plausibly what `np.float32(a)`
  bottoms out in, see §5's caveat) unconditionally uses `self.to_contiguous()`
  — read: `if self.is_c_contiguous() { self.clone() } else {
  self.to_contiguous() }` — which is the **plain always-C** relayout, not
  `to_contiguous_order("K")`. Same bypass pattern a third time.
- **`ionp-py/src/ndarray_attrs.rs::real`/`imag`** (~line 358) also call
  `self.inner.to_contiguous()` (plain C), not `to_contiguous_order("K")`.
  Note this sits on top of a **separate, already-documented, deliberate**
  design decision in the same doc-comment: anionpy's `.real`/`.imag`
  return a **copy**, not numpy's zero-copy view (`KNOWN-DIFFERENCES.md`),
  which this document is not proposing to change. The order-of-that-copy
  bug (should the copy be `'K'`-ordered rather than forced C) is a
  narrower, in-scope fix riding on top of that already-accepted
  architectural divergence.

### 7.3 #7 vs #64: two fixes, not one — the evidence

They are different bugs in different files, triggered by different
mechanisms, and one can be fixed without touching the other:

- **#7** is a **missing control-flow branch** in exactly one function,
  `NdArray::reshape` (`array.rs` ~447): it never checks "is the requested
  shape identical to my current shape," so it runs a stride-computing
  algorithm (itself correct — §7.1) in a situation numpy's own code never
  lets it run in. The fix is adding numpy's §3a short-circuit; it touches
  nothing about *how* strides are computed anywhere, only *whether* the
  general algorithm is invoked for the identity case. Nothing about
  `manip.rs` or `sort.rs` is implicated.
- **#64** is a **reuse gap across ~5 files** (`manip.rs`, `sort.rs`,
  `array.rs::cast_to`, `ndarray_attrs.rs::real`/`imag`, and by the same
  pattern presumably others not directly grepped): each independently
  calls a bare `c_strides`/`to_contiguous()` instead of the
  already-correct `to_contiguous_order`/`relayout_by_perm`/
  `multi_sorted_stride_perm` trio that `array.rs`'s own `.copy()`/
  `.astype()` and `emath.rs`'s ufunc-adjacent path already use correctly.
  The fix is routing — replacing each bare allocation call with the
  existing correct primitive — and touches zero lines of
  `array.rs::reshape` or `shape::attempt_nocopy_reshape`.
- Fixing #7 (add the identity short-circuit to `reshape`) does nothing for
  `np.roll`/`np.tile`/`hstack`/`vstack`/`sort`/`cast_to`/`real` — none of
  those call `reshape` internally on the divergent path (`roll` uses
  `empty_like`-equivalent allocation directly in `manip.rs`; `sort` has
  its own allocator; `real`/`cast_to` call `to_contiguous()` directly).
  Fixing #64 (route `manip.rs`/`sort.rs`/`cast_to`/`real` through the
  existing KEEPORDER-voting primitives) does nothing for the
  `zeros(...).copy('F').reshape(same_shape)` case, because that case never
  reaches any of those five files — it's entirely inside
  `array.rs::reshape`.
- This matches the brief's own framing ("one is a stride computation, the
  other an allocation-order decision") almost exactly, except sharpened
  by the source read: **#7 isn't even a stride-computation bug in the
  strict sense** — the stride-computation code (`attempt_nocopy_reshape`)
  is correct; #7 is a bug in *when* that (correct) computation gets
  invoked. #64 is, as stated, a real allocation-order-decision bug,
  repeated independently at each of ~5 call sites because none of them
  reuse the one place the decision is implemented correctly.

### 7.4 What a correct fix is expected to break

- **Currently-passing items that would change**: anything in the test
  corpus that declares `hstack`/`vstack`/`concatenate`/`roll`/`tile`/
  `sort`/`cast_to`(→ scalar-constructor dtype casts)/`real`/`imag` as
  "exact" *on a plain C-contiguous input* should be unaffected (§2's first
  branch, C-contiguous prototype → C strides, is what a C-contiguous input
  already produces via either the old bare-`c_strides` code or the correct
  KEEPORDER router — they agree when the input already is C-contiguous).
  Anything currently marked "exact" that was tested **only** on
  C-contiguous inputs, but is silently relying on the buggy always-C
  behavior for a non-C-contiguous scenario elsewhere in the same
  function's test matrix, would newly change — this document does not
  have visibility into which specific declared-exact test rows fall in
  that category; that's a job for whoever runs the differential suite
  after the fix, not something inferable from source alone.
- **`stack`** — the brief states this is already failing for exactly this
  reason (`concatenate` rematerializes C-contiguous where numpy preserves
  operand strides). Confirmed consistent: `stack` is implemented on top of
  `concatenate`-family logic per the attribution table (§4), so it shares
  `manip.rs`'s bypass. A correct #64 fix should turn this into a pass, not
  a new failure.
- **Anything that currently (incorrectly) asserts a hardcoded C-contiguous
  stride tuple as ground truth** in anionpy's own test fixtures (as
  opposed to a differential numpy comparison) would need updating —
  this document did not audit the test suite for such fixtures and
  flags this as a real risk rather than asserting it doesn't exist.
- **`np.tile` with `reps != 1`** is *expected*, per §4's attribution, to
  remain C-contiguous even after a correct fix — a "fix" that made it
  KEEPORDER-preserving would itself be a new divergence from real numpy,
  per the `reshape`-default-order reasoning in §4's tile row (caveated in
  §5 as not independently re-verified against a live numpy call).

## CORRECTION — 2026-08-08: the spec did not cover order='A'

**Everything above this section is left standing, uncorrected, as
originally written and committed** (including the embedded probe source
and its 3420/3420 result, which now carries an in-place superseded flag
pointing here — see §6). This section does not replace that text; it
replaces the *conclusion* that the reshape rule was complete. Where the
two disagree, this section is current.

### What was wrong

An adversarial rerun against a grid this document's original probe never
contained found **120 misses, 0 hits** on `order='A'` reshapes (zero
misses at `order='C'`/`order='F'` — those remain correct). Root cause:
`predict_reshape_strides`'s docstring (as committed, ~probe.py:269) stated

> "order is 'C' or 'F' (the only orders .reshape()/np.reshape() accept)."

This is **false**, checked directly against numpy 2.5.1, not assumed:

```
>>> np.zeros((2,3)).reshape((3,2), order='A').strides   # no error
(16, 8)
>>> np.reshape(np.zeros((2,3)), (3,2), order='A').strides  # no error
(16, 8)
```

Because the old code's only dispatch was `is_f_order = order == "F"`, an
`'A'` argument silently fell into the `is_f_order = False` (C) branch for
the no-copy view search (`_attempt_nocopy_reshape`), while the copy
fallback a few lines later branched on the *literal string* `order ==
"C"` and picked F strides for anything else — including `'A'`. The two
halves of the function disagreed about what `'A'` meant, and which one
you hit depended on whether a no-copy view was possible for that
particular shape/stride combination.

### Source citation for the actual rule

`numpy/_core/src/multiarray/shape.c`, `_reshape_with_copy_arg` (v2.5.1
tag, function body ~lines 227-246, fetched into
`/private/tmp/numpy-src/multiarray/shape.c` and read directly — not
inferred):

```c
if (order == NPY_ANYORDER) {
    order = PyArray_ISFORTRAN(array) ? NPY_FORTRANORDER : NPY_CORDER;
}
else if (order == NPY_KEEPORDER) {
    PyErr_SetString(PyExc_ValueError,
            "order 'K' is not permitted for reshaping");
    return NULL;
}
```

This runs as the **first thing** in the function, before the
identical-shape short-circuit (§3a) and before any nocopy/copy dispatch —
confirmed by reading the surrounding code, not inferred from behavior
alone. `PyArray_ISFORTRAN` is a macro, not a function; its definition was
fetched separately (`numpy/_core/include/numpy/ndarraytypes.h`, not
originally in this document's §0 file list — added for this correction)
because it wasn't in the files this document originally read:

```c
#define PyArray_ISFORTRAN(m) (PyArray_CHKFLAGS(m, NPY_ARRAY_F_CONTIGUOUS) && \
                             (!PyArray_CHKFLAGS(m, NPY_ARRAY_C_CONTIGUOUS)))
```

So the rule, stated as a pure function: **`order='A'` resolves to `'F'`
iff the source array is F-contiguous AND NOT C-contiguous; otherwise it
resolves to `'C'`.** This is what the task brief stated from measurement;
the source read confirms it exactly, including the reason it collapses to
`'C'` for 0-d and 1-d arrays — `ndarraytypes.h`'s own comment notes all
0-d arrays are both C- and F-contiguous, and a 1-d array that is
C-contiguous is also F-contiguous, so `ISFORTRAN` is false in both cases
and `'A'` never resolves to `'F'` there. No subtler behavior beyond this
was found in the source; the brief's stated rule is exactly what
`_reshape_with_copy_arg` does, not an approximation of it.

`order='K'` was **checked, not assumed**: both `a.reshape(shape,
order='K')` and `np.reshape(a, shape, order='K')` raise `ValueError:
order 'K' is not permitted for reshaping` in real numpy 2.5.1, matching
the `NPY_KEEPORDER` branch above exactly. The fixed predictor raises the
identical `ValueError` for `order='K'` rather than silently coercing it.

### The fix

`resolve_reshape_order()` (new function, `/private/tmp/numpy-src/probe.py`
~line 269) resolves `'A'`/`'K'`/invalid orders **before**
`predict_reshape_strides` does anything else — mirroring
`_reshape_with_copy_arg`'s own ordering. `predict_reshape_strides` calls
it as its first line and is otherwise unchanged; the previously-correct
`'C'`/`'F'` machinery (§3a identity short-circuit, §3b
`_attempt_nocopy_reshape`, the copy-fallback literal strides) was not
touched, because the source confirms none of it was wrong — only the
order resolution feeding into it was missing.

### The two miss families (regression cases)

Both closed by the same one-function fix, confirmed by rerunning the
widened grid below:

- **Non-contiguous 3-4d bases** (`neg` = `a.copy('C')[::-1]`, `roll` =
  `np.moveaxis(a.copy('C'),0,-1)`, `step2` = `a.copy('C')[::2]`) on shapes
  `(2,1,1,3)`, `(3,2,1)`, `(2,2,2,2)`, `(4,3,2)` — 70 of 120 misses. These
  are arrays that are neither C- nor F-contiguous, which is exactly the
  case the old `is_f_order = order == "F"` line got backwards for `'A'`.
- **Zero-size shapes** `(0,1,0)` and `(1,0,1)`, every layout — 50 of 120
  misses. The original grid had zero-size shapes (`(0,3)`, `(3,0)`,
  `(0,3,4)`, `(2,0,4)`, `(2,3,0)`) but none mixing a zero axis with a
  length-1 axis in a way that made C- vs F-contiguity ambiguous under
  `'A'`, *and* the original grid never swept `order='A'` at all — so this
  family would have been missed regardless of shape coverage.

One pasted repro (**ILLUSTRATIVE, NOT EXHAUSTIVE** — ranges over exactly
one of the 120 cases):

```
b = np.zeros((2,1,1,3)).copy('C')[::-1]     # strides (-24,24,24,8), neither C- nor F-contiguous
np.reshape(b, (6,1), order='A').strides     # numpy 2.5.1 -> (8, 8)
```

Independently reran against the fixed predictor: `predict_reshape_strides`
now returns `([8, 8], 'copy-literal')` for this exact input — matches.

### Widened grid, new denominator and score

Per-item changes to `/private/tmp/numpy-src/probe.py` (not committed —
BSD-3 numpy sourcing rules from §0 apply to the fix work the same as the
original):

- **Dtypes**: `float64, float32, int64, int8, complex128` (5) →
  `float64, float32, float16, longdouble, int64, int8, uint8, complex128,
  complex64, bool_` (10).
- **Shapes**: added `(0,1,0)`, `(1,0,1)` to the existing 23-shape list (25
  total).
- **Layout variants**: added `neg`, `roll`, `step2`, `perm` to the
  existing `C`/`F`/`T`/`slice+2`/`slice-2` set on `base_variants()`.
- **Reshape order sweep**: `for order in ("C", "F")` → `for order in ("C",
  "F", "A")`.

Rerun (`~/Monday/ionp/.venv/bin/python
/private/tmp/numpy-src/probe.py`, numpy 2.5.1):

```
keeporder: 1940/1940
reshape:   13950/13950
concat:    2580/2580

TOTAL: 18470/18470
```

**18470/18470** — denominator grew from 3420 to 18470 (the point, per the
task brief: a bigger grid, not a smaller one, and it is reported
honestly). Zero misses of any kind on this rerun. Sanity-checked against
the brief's own warning that a uniform pass rate across a cross-product
can be an instrument bug: the `neg` variant was independently confirmed
(outside the harness, see above) to produce genuinely non-contiguous
strides `(-24, 24, 24, 8)` for shape `(2,1,1,3)` — distinct C/F/T/perm/neg
strides were printed and inspected by hand for six shapes before trusting
the 18470/18470, not taken from the harness's own accounting.

### Is order='A' on a non-contiguous/negative-stride base a new case?

**Genuinely new**, not a case already covered by a section this document
had written. The original §6 grid never swept `order='A'` at all (only
`("C", "F")`), so every `order='A'` reshape — contiguous base or not —
was untested before this correction, independent of the layout-variant
question. The non-contiguous-base miss family additionally required
layout variants (`neg`/`roll`/`step2`) the original `base_variants()`
didn't generate; the zero-size-shape miss family additionally required
shapes the original `SHAPES` list didn't contain. All three gaps (order
sweep, layout variants, shapes) stacked to produce the 120 misses; fixing
only the rule (not the grid) would have left the bug provably unfindable
by this document's own probe, which is why §-item 3 of the task ("widen
the grid, don't just fix the rule") is treated as load-bearing here, not
optional.

### Does the same false premise appear elsewhere in this document?

**Checked all order-taking entry points this document's §4 attribution
table names or its probe implements** (`astype`, `.copy()`, `empty_like`/
`zeros_like`/`ones_like`, `asarray`/`np.array`, `reshape`, `concatenate`/
`hstack`/`vstack`, `sort`, `roll`, `fft`, `tile`, `float32`, `real`).
Grepped the document itself for exclusivity claims (`only`, `never`); the
one false claim found is the `predict_reshape_strides` docstring already
fixed above. One related-but-distinct finding, reported plainly rather
than silently left out:

- **`.astype()`, `.copy()`, `np.empty_like`, `np.asarray` also accept
  `order='A'` explicitly** (checked empirically:
  `a.astype(np.float32, order='A')`, `a.copy(order='A')`,
  `np.asarray(a, order='A')`, `np.empty_like(a, order='A')` all ran
  without error against a non-contiguous `a`). Their C implementation
  resolves `'A'` the same way as reshape — confirmed by reading
  `PyArray_NewLikeArrayWithShape` (`ctors.c` ~line 1038, already in this
  document's §0 file list, already cited for §2's KEEPORDER branch):
  `case NPY_ANYORDER: order = PyArray_ISFORTRAN(prototype) ?
  NPY_FORTRANORDER : NPY_CORDER; break;` sits directly next to the
  `NPY_KEEPORDER` branch §2 already documents. **This is not a false
  claim this document made** — §2's prose only ever describes the
  `order='K'` *default* behavior of these functions and never asserted
  `'A'`/`'C'`/`'F'` weren't also accepted — so there is nothing to
  retract. But it IS an untested gap: `keeporder_alloc_strides()` in the
  probe takes no `order` parameter at all and the grid only ever calls
  `.copy(order="K")`, never `.copy(order="A")`. Flagging this openly
  rather than fixing it, since it is outside this task's scope (fixing
  `predict_reshape_strides` and its grid) and doing it properly means
  adding a fourth predictor function, not a one-line change to an
  existing one.
- **`.ravel()` and `.flatten()` are not mentioned anywhere in this
  document's §4 attribution table or its probe** — checked empirically
  that both also accept `order='A'` (and, unexpectedly, `.ravel(order='K')`
  and `.flatten(order='K')` both ran without error, unlike reshape's hard
  `'K'`-rejection — `PyArray_Ravel` in `shape.c` has its own explicit
  `NPY_KEEPORDER` handling separate from `_reshape_with_copy_arg`'s, per
  `shape.c` ~lines 920-931, read for this correction). Since neither
  function was ever in scope, there is no false claim to retract here
  either — this is a pre-existing, already-disclosed-by-omission gap
  (§5's "what this document does NOT cover" already scopes the document
  to the twelve attribution-table entries), not a new one created by this
  correction.

Plainly stated: **the reshape false premise does not recur elsewhere in
this document's existing claims.** The one place it was written down
(`predict_reshape_strides`'s docstring) is fixed. The adjacent functions
that share numpy's `NPY_ANYORDER`-resolution pattern (`astype`/`copy`/
`empty_like`/`asarray`) were never claimed to lack it — they are simply
untested for it, which is a scope gap consistent with what §5 already
disclosed, not a retraction.

### Reproduction

```
~/Monday/ionp/.venv/bin/python /private/tmp/numpy-src/probe.py
```

### The corrected rule, in committed form

**Added by Monday, 2026-08-08, after review.** The line above points into
`/private/tmp`, which is scratch and will be wiped; the numpy sources next
to it are BSD-3 and deliberately uncommittable while this repo heads for an
MIT release. That left the *only* executable copy of the fix outside the
repo, so this section carries it. The rule below is our own code, not
numpy's — numpy's C is cited and paraphrased above, not vendored.

```python
def resolve_reshape_order(order, shape, strides, itemsize):
    """Resolve reshape's `order` against the SOURCE array's own flags.

    numpy's `_reshape_with_copy_arg` does this BEFORE the identical-shape
    short-circuit and before any nocopy/copy dispatch: NPY_ANYORDER becomes
    Fortran order iff PyArray_ISFORTRAN(array), else C order; NPY_KEEPORDER
    is rejected outright. PyArray_ISFORTRAN is F_CONTIGUOUS && !C_CONTIGUOUS
    -- the source array's contiguity flags, NOT a per-axis heuristic.

    0-d and 1-d arrays have both flags set, so ISFORTRAN is false and 'A'
    always resolves to 'C' there.
    """
    if order == "A":
        is_fortran = (is_f_contiguous(shape, strides, itemsize)
                      and not is_c_contiguous(shape, strides, itemsize))
        return "F" if is_fortran else "C"
    if order == "K":
        raise ValueError("order 'K' is not permitted for reshaping")
    if order not in ("C", "F"):
        raise ValueError(f"unsupported order {order!r}")
    return order
```

`predict_reshape_strides` calls this as its first statement, before
`is_f_order = order == "F"`. Nothing else in the reshape machinery changed —
the source reading confirms the `'C'`/`'F'` paths were never wrong.

### Proof the widened grid bites

Neither a passing grid nor a grown denominator is evidence on its own; a
uniform 18470/18470 is the exact shape an instrument bug takes. So the fix
was reverted in isolation — `resolve_reshape_order` neutered to
`return order`, everything else untouched — and the grid re-run:

| probe | keeporder | reshape | concat | total |
|---|---|---|---|---|
| fix neutered | 1940/1940 | **12820/13950** | 2580/2580 | 17340/18470 |
| fix in place | 1940/1940 | 13950/13950 | 2580/2580 | **18470/18470** |

**1130 misses**, all in `reshape`, all recovered by the one function. Worth
recording that the adversarial run which *found* this defect saw only 120 of
those 1130: the permanent grid now detects roughly nine times more of the
class than the probe that caught it. That gap is the argument for widening a
grid after a miss rather than only patching the case that missed.

## Ticket #7 close-out — 2026-08-08

`resolve_reshape_order` (above) was already correct; nothing in it changed
this pass. The remaining #7 defect was `NdArray::reshape`/
`reshape_with_order` (`ionp-core/src/array.rs`) not reproducing numpy's
identity-shape short-circuit (`_reshape_with_copy_arg`, `shape.c` ~237-247)
at all, plus two further call sites that needed the short-circuit
*suppressed* rather than added, once it existed. All three are one root
cause: numpy's identity check compares the RAW, unresolved dims the caller
passed, not the resolved/synthesized ones — a `-1` (or any negative
placeholder) never equals a concrete old dimension there, even when it
numerically resolves to the same value.

### Fix 1 — the headline defect: no identity short-circuit at all

`reshape`/`reshape_with_order` never special-cased "new shape == old
shape," so an already-correctly-strided array got recomputed strides
instead of handed back unchanged. Repro:
`ap.zeros((4,1,2)).copy('F').reshape((4,1,2)).strides` was `(8, 64, 32)`;
numpy gives `(8, 32, 32)`.

Fixed in `ionp-core/src/array.rs` by splitting `reshape`/`reshape_with_order`
into shared private `reshape_impl`/`reshape_with_order_impl(new_shape,
order, identity_shortcut: bool)`, with the public `reshape`/
`reshape_with_order` passing `identity_shortcut = true` (short-circuit
taken whenever `new_shape == self.shape`) and two new public entry points,
`reshape_no_identity_shortcut`/`reshape_with_order_no_identity_shortcut`,
passing `false`.

### Fix 2 — `PyArray::reshape`/`creation::reshape` and literal `-1`

Once the short-circuit existed, it was taken even when the caller's raw
request contained a `-1` that only numerically resolved to the current
shape (`ma.masked_array([]).reshape(-1)`) — numpy does NOT take its
shortcut there (raw `-1` != old dim `0`), but the naive port did, so it
handed back the stale `(0,)`-strided empty array instead of numpy's
freshly-computed `(8,)`. Fixed by routing `ionp-py/src/lib.rs`'s
`PyArray::reshape` method and `ionp-py/src/creation.rs`'s free `reshape`
function through `reshape_with_order_no_identity_shortcut` whenever any raw
requested dim (`dims: Vec<isize>`) is negative, and through the normal
shortcutting `reshape_with_order` otherwise.

### Fix 3 — `ravel_view_or_copy`'s synthesized shape (found this pass)

A second, independently-discovered instance of the same class: `.ravel()`
(`ndarray_attrs.rs::ravel` → `lib.rs::ravel_view_or_copy`) tries a cheap
view via `reshape_with_order(&[n], ord)`, where `n = arr.size()` is a
*synthesized* target shape (the "flatten to 1-D" target), not a literal
caller-supplied shape — never a `-1`, but the same "was this shape
literally requested" distinction applies, because numpy's `.ravel()` builds
a fresh view-construction-formula stride even when the array is already
shape `(n,)`, rather than aliasing the source's own (possibly zeroed)
strides verbatim. Repro: `ma.masked_array([]).ravel().data.strides` was
`(0,)`; numpy gives `(8,)`. Fixed by changing that call site to
`reshape_with_order_no_identity_shortcut(&[n], ord)`.

`.flatten()` (`ndarray_attrs.rs::flatten`) was not affected — it calls
`ravel_order` directly, unconditionally a fresh copy by design, never
`reshape_with_order`.

### Scope fence — #64 sibling call sites, read not touched

Per the ticket's file-ownership fence, `ndarray_attrs.rs` was read (to
confirm `.ravel()`'s call graph and that `.flatten()` was unaffected) but
not edited. `array.rs::ravel_order` was read (to confirm its own,
pre-existing, deliberate `n==0` zero-stride override is unrelated to this
defect and correctly justified for `.flatten()`'s always-copy semantics)
but not edited. `manip.rs`, `sort.rs`, `array.rs::cast_to` were not read or
touched this pass.

The much larger `#7`/`#64` populations named in
`docs/TICKET-64-REMEASURE-2026-08-08.md` (the 118-item "#7" list: `abs`,
`sin`, `sort`, `broadcast_to`, etc. — ufuncs and other ops silently
recomputing a real stride instead of preserving/broadcasting on a
length-1-axis or F-order input) are a **different code path** from
anything touched here — no call in that population goes through
`NdArray::reshape`/`reshape_with_order`. Spot-checked directly,
same-process, before and after this session's fix: `sort(axis=0)` on
`zeros((4,1,2)).copy('F')` gives `(16,16,8)` both before (baseline
worktree build, `PYTHONPATH=/tmp/worktree_pkg`) and after (this session's
fixed build) vs. numpy's `(8,32,32)` — identical divergence, confirming
this fix neither touches nor accidentally masks that population. `abs`
gives the same result. Those items remain entirely open, out of scope for
this ticket, and are exactly the population `docs/TICKET-64-REMEASURE-
2026-08-08.md` is auditing.

### Guards proven to bite

Each of the three fixes above was reverted in isolation, rebuilt, and
re-measured, then restored:

| guard reverted | repro | before revert | after revert (bites) |
|---|---|---|---|
| `identity_shortcut` forced `false` always (core mechanism, both `reshape_impl`/`reshape_with_order_impl`) | `zeros((4,1,2)).copy('F').reshape((4,1,2)).strides` | `(8,32,32)` match | `(8,64,32)` mismatch (numpy `(8,32,32)`) |
| `PyArray::reshape`'s `neg_count`-guard removed (always calls shortcutting `reshape_with_order`) | `ma.masked_array([]).reshape(-1).data.strides` | `(8,)` match | `(0,)` mismatch (numpy `(8,)`) |
| `ravel_view_or_copy`'s no-shortcut call reverted to shortcutting `reshape_with_order` | `ma.masked_array([]).ravel().data.strides` | `(8,)` match | `(0,)` mismatch (numpy `(8,)`) |

All three reverts were single-line, isolated (only one guard reverted at a
time, the other two fixes left in place), rebuilt via `cargo build -p
ionp-py --release` + `maturin develop --release`, measured, then the exact
original file was restored via `cp` from a pre-edit backup and diffed
byte-identical before rebuilding the final state.

### Suite and ledger

Differential suite (`tests/differential/run.py`), two fresh runs against
the final fixed+installed binary: failure sets identical, 29 names, byte-
identical to the pristine pre-#7 baseline (independently confirmed via a
clean `git worktree` build of `6471550`, stable across 5 repeated runs).
No name added, none removed, none changed order/count — the 29-name
baseline is unmoved by this fix, as expected (the reshape/ravel bugs fixed
here were previously mis-declared `exact` rather than counted as
differential failures, since the differential suite's pre-existing
`reshape`/`ravel`/`ma.*` cases didn't happen to hit the identity-shortcut
edge case before `reshape_order_cases.py` was added).

Ledger (`tools/coverage.py`): **1335/3091 = 43.19%**, unchanged from the
recorded baseline — expected, since `reshape`/`ndarray.reshape`/`ma.ravel`/
`ma.reshape` were already declared `exact` before this fix; this pass
corrects their *behavior* on an edge case the declaration didn't already
have a failing differential test for, it does not add or remove a declared
item.

### Files touched

- `ionp-core/src/array.rs` — `reshape`/`reshape_with_order` split into
  shared `_impl` functions with an `identity_shortcut` flag; two new public
  entry points added.
- `ionp-py/src/lib.rs` — `PyArray::reshape` method routes through the
  no-shortcut entry point when any raw dim is negative;
  `ravel_view_or_copy` routes through the no-shortcut entry point
  unconditionally (its synthesized shape is never a literal caller shape).
- `ionp-py/src/creation.rs` — free `reshape` function, same `-1` routing as
  the method.
- `tests/differential/reshape_order_cases.py` — new corpus file (added
  prior session, unchanged this session): identity reshapes across order
  C/F/A, `order='K'` rejection, and non-identity regroup reshapes, 5 dtypes
  x 6 shapes x up to 5 layouts.
- `tests/differential/run.py` — one added import line for the corpus file
  above (prior session, unchanged this session).
