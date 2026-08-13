# `ionp.ma` — design

Status: Phases 0-4 IMPLEMENTED (77 of 225 `ma.*` manifest items, all declared
`exact`, 0 failing/phantom/untested -- measured 2026-08-03). Phases 5-6 (~28
items) remain design-only; see section 5a.

Original status when this document was written: DESIGN, not implemented.
`ionp.ma` did not exist (measured 2026-08-03 [sic, superseded same day]:
`hasattr(ionp, "ma") == False`, 0 of 225 manifest items present).

Author: Monday. Every number in this document was measured this session by
`/tmp/mg_ma_survey.py` and `/tmp/mg_ma_overlap.py`, not carried from notes.

UPDATE 2026-08-07 (Monday, fill_value-boxing session): the "225" denominator
above (and at lines 9/18/42) is the flat top-level count from
`tools/numpy_surface.json`'s `names` list -- accurate for what it measured,
but a LATER denominator audit exploded `ma.MaskedArray` (counted as 1 item
there) into its 191 individually-addressable dunder/public attributes
(`tools/numpy_surface.json`'s own `exploded["ma.MaskedArray"]` entry: 97
dunder + 94 public), which is where the "415" figure elsewhere in this
project's tracking comes from: `224 (current top-level ma.* count, drifts
slightly with numpy_surface.json regenerations) - 1 (the un-exploded
MaskedArray placeholder) + 191 (its exploded attrs) = 414`, matching within
rounding/regeneration drift of the audited "415". Both denominators are
real measurements of different things -- "225/224" answers "how many
top-level `ma.*` names", "415/414" answers "how many individually
addressable `ma.*` surface points including MaskedArray's own methods".
Neither supersedes the other; do not silently swap one number in for the
other in future edits without checking which question is being asked.

---

## 1. Why this document exists

`ma.*` is 225 of the 710 absent items — the largest single bucket, bigger than
the entire remaining top-level surface. I had it filed as *blocked on the buffer
representation decision*, because `np.ma` mutates in place:

```
a = np.ma.array([1., 2., 3.], mask=[0, 1, 0])
a[0] = np.ma.masked      # mutates
```

That filing was wrong, and the correction is the main point of this document.

## 2. The measurement that unblocks it

```
mask is its own ndarray: ndarray bool
data is its own ndarray: ndarray float64
mask shares memory with data: False
```

**A `MaskedArray` is a PAIR of ordinary arrays, not a special buffer.** numpy
does not encode maskedness inside the data buffer; it carries a parallel boolean
array (or the `nomask` singleton, which is literally `False`). Nothing about the
mask requires a new buffer representation.

Second measurement — which of the 225 items actually mutate:

| requires in-place write | count |
|---|---|
| `put`, `putmask`, `set_fill_value`, `soften_mask`, `harden_mask` | 5 |
| everything else | 220 |

`MaskedArray.__setitem__` and the `.mask` / `.fill_value` setters also mutate,
but the manifest counts `ma.MaskedArray` as **one** item, not as an expanded
method surface (`ma.MaskedArray.*` items in manifest: 1). So the mutability gate
is worth **~5 items**, and it is the SAME gate as the `out=` defect — not a
second one. It does not block the other 220.

**Conclusion: `ma.*` is not blocked. It was never blocked. I mis-sized it once
and then kept quoting my own mis-sizing** (lesson #66 — re-measure carried
beliefs before they size the work).

## 3. Surface shape

By kind (`np.ma`, measured):

| kind | count | note |
|---|---|---|
| plain function | 163 | |
| `_MaskedUnaryOperation` | 27 | mechanical |
| `_MaskedBinaryOperation` | 17 | mechanical |
| `_DomainedBinaryOperation` | 6 | mechanical |
| class | 7 | `MAError MaskError MaskType MaskedArray bool_ masked_array mvoid` |
| data singleton | 5 | `masked masked_singleton masked_print_option mr_ nomask` |

By overlap with what ionp already has:

| | count |
|---|---|
| same-named ionp toplevel already DECLARED exact | 92 |
| present on `ionp` but undeclared | 35 |
| no ionp counterpart at all | 98 |

The 50 `_Masked*Operation` items are **one generic wrapper**, not 50 pieces of
work. They are `ufunc + mask propagation`, and the three variants differ only in
where the mask comes from:

- `_MaskedUnaryOperation` — mask passes through.
- `_MaskedBinaryOperation` — mask is `mask_a | mask_b` broadcast.
- `_DomainedBinaryOperation` — same, PLUS a domain predicate that masks where
  the operation would be invalid (`divide` masks where `b == 0`), and the
  invalid lanes must be computed anyway or filled, not allowed to raise.

## 4. Architecture

```
ionp/ionp/ma/__init__.py     Python: the ma namespace, thin
ionp/ionp/ma/core.py         Python: MaskedArray class, wrappers, predicates
ionp-core/src/ma.rs          Rust:   mask kernels (propagate, domain, reduce-ignoring)
ionp-py/src/ma.rs            Rust:   bindings
```

This respects the standing rule — **Python where needed, Rust in the core, and
no arithmetic in a `.py` file.** The division is:

- **Rust does:** every elementwise mask combination, every domain predicate
  evaluation, every masked reduction (a reduction that skips masked lanes is a
  *kernel*, not a Python loop), `filled`, `compressed`, `count`.
- **Python does:** the class, attribute plumbing, dispatch, `__repr__`,
  and the ~40 one-line wrappers that are genuinely just naming
  (`masked_greater(x, v)` is `masked_where(greater(x, v), x)`).

**A Python `for` loop over elements is a bug in this subsystem too.** The
temptation is real here — mask logic *looks* like control flow. It is not; it is
elementwise boolean algebra and belongs in a kernel.

### `nomask`

numpy's `nomask` is the singleton `False`, not an array of zeros. This is a real
performance contract, not an implementation detail: an unmasked `MaskedArray`
must not allocate an N-element boolean array. Represent it as
`Option<NdArray>`/`None` internally and preserve the identity `a.mask is nomask`
where numpy preserves it. **This must be verified out of corpus** — it is
exactly the kind of thing that passes a value-equality test and still lies.

### `masked` singleton

`np.ma.masked` is a `MaskedConstant` — a 0-d MaskedArray with `mask=True`, and
it is a *singleton* (`np.ma.masked is np.ma.masked_singleton`). It has surprising
truthiness and printing behaviour. Model it explicitly; do not let it be an
ordinary instance that merely compares equal.

## 5. Sequencing

Ordered by items-per-unit-work, and deliberately front-loading the parts that
need no new decisions:

| phase | items | content |
|---|---|---|
| 0 | 0 | `MaskedArray` core: data+mask pair, `nomask`, `filled`, `getdata`, `getmask`, `getmaskarray`, `__repr__`. No declarations — this is the substrate. |
| 1 | ~50 | The three generic op wrappers. One kernel family, 50 items. |
| 2 | ~92 | The items whose ionp counterpart is already declared exact — mask-aware versions of already-proven functions. |
| 3 | ~30 | Mask construction: `masked_where` + the 10 thin comparators, `make_mask*`, `mask_or`, `is_mask`, `isMA`/`isMaskedArray`/`is_masked`. |
| 4 | ~25 | Fill-value machinery: `default_fill_value`, `maximum_fill_value`, `minimum_fill_value`, `common_fill_value`, `set_fill_value`, `masked_values`, `masked_object`, `fix_invalid`. |
| 5 | ~23 | Contiguity/structure utilities: `clump_masked`, `notmasked_edges`, `flatnotmasked_contiguous`, `mask_rowcols`, `compress_*`, `flatten_mask`. |
| 6 | ~5 | The mutating five. **Gated on the same buffer decision as `out=`.** |

Phases 1–5 = ~220 items and require **no** escalated decision.

## 5a. Phases 0-4 status (2026-08-03)

Phases 0-4 are implemented and declared (`ionp/ma/core.py`, `ionp/ma/__init__.py`,
`ionp/_state/ma.py`; differential cases in `ionp/tests/differential/ma_cases.py`) --
77 `ma.*` items, all bit-exact against real numpy 2.5.1, 0 failing/phantom/untested.
Phases 5 (contiguity/structure utilities) and 6 (the mutating five) remain out of
scope, as originally sequenced.

Three more corrections were found while building Phases 3-4, on top of the five
already recorded in `ionp/_state/ma.py`'s module docstring for Phases 0-2 -- see
section 6 below for the first two; the third is `masked_values`' data-overwrite
behavior, undocumented as a hazard anywhere in this file until now:

- **`masked_values` overwrites data at previously-masked positions.** Every other
  `masked_*`-family function in Phase 3 (`masked_where`, `masked_equal`, ...,
  `masked_invalid`) leaves the underlying data untouched and only changes the
  mask. `masked_values` is the one exception: real numpy's own source
  (`numpy/ma/core.py`) does `xnew = filled(x, value)` FIRST -- so a position `x`
  already had masked gets overwritten with `value` -- and only then computes the
  new mask against `xnew`, replacing the prior mask outright (no `mask_or` with
  the input's own mask). A first draft built `masked_values` on the same
  data-preserving `_masked_where_impl` every other comparator uses; the
  differential harness's `masked_values/*/fully_masked` and `.../partially_masked`
  cases caught the mismatch (numpy's `.data` was `[-999, -999, ...]`, ionp's was
  the original unmasked values). See `masked_values`'s docstring in
  `ionp/ma/core.py` for the fix.

## 5b. Phase 5 status (2026-08-03) -- and a scope-definition error in section 5

Section 5's Phase-5 row above (line 141) is **wrong as a description of what
"Phase 5" actually needed to cover**: it lists only the contiguity utilities
(`clump_masked`, `notmasked_edges`, `flatnotmasked_contiguous`,
`mask_rowcols`, `compress_*`, `flatten_mask`) and omits the entire
stack/join sub-family (`vstack`, `hstack`, `dstack`, `column_stack`,
`row_stack`, `append`, `diagflat`, `concatenate`, `stack`, `atleast_1d/2d/3d`,
`diag`) and the shape-transform sub-family (`transpose`, `swapaxes`,
`reshape`, `ravel`, `squeeze`, `expand_dims`, `resize`, `compress`,
`nonzero`) entirely -- despite this document's own section 3/4 surface
inventory (and the task brief that actually drove this phase) treating
"contiguity/structure/joining" as one combined ~23-item bucket. This is the
kind of family-membership error this document has had several of already
(see section 6) -- flagging it here rather than silently reinterpreting the
row.

What actually shipped for Phase 5 (`ionp/ma/core.py`, `ionp/ma/__init__.py`,
`ionp/_state/ma.py`; differential cases in
`ionp/tests/differential/ma_cases.py`): 17 items, all bit-exact against real
numpy 2.5.1, 0 failing/phantom/untested --
`vstack`/`hstack`/`dstack`/`column_stack`/`row_stack`/`append`/`diagflat`
(the stack/join sub-family) and
`transpose`/`swapaxes`/`reshape`/`ravel`/`squeeze`/`expand_dims`/`resize`/
`compress`/`compressed`/`nonzero` (the shape-transform sub-family, extending
Phase 2's `repeat`/`take` precedent).

Two genuinely different shrink/fill_value contracts were found live within
this set (see `ionp/ma/core.py`'s Phase-5 module comment and
`ionp/_state/ma.py`'s Phase-5 entries for the exact `inspect.getsource(np.ma.*)`
citations) -- do not assume either one is "the" Phase-5 rule:
  - the stack/join family (`vstack`/`hstack`/`dstack`/`column_stack`)
    ALWAYS materializes the mask to a real array and ALWAYS resets
    `fill_value` to the plain per-dtype default, even from all-`nomask`
    operands or matching custom fill_values on every side;
  - the shape-transform family preserves `nomask` and inherits `fill_value`
    from the input, except `resize`, which preserves `nomask` but still
    resets `fill_value` to the default (verified against real numpy's own
    `ma.resize` source, which rebuilds via `masked_array(...)` from scratch);
  - `append` is neither: real `numpy.ma.append` is literally
    `concatenate([a, b], axis)`, and `numpy.ma.concatenate`'s own source
    ALWAYS shrinks the combined mask (`data._mask = _shrink_mask(dm)`)
    whenever any operand carries a non-identity-`nomask` mask array (even an
    all-`False` one), while still defaulting `fill_value` like its stack
    siblings -- CAUGHT BY THIS TASK'S DIFFERENTIAL HARNESS: a first draft of
    `append` only checked `am.mask is nomask and bm.mask is nomask` by
    Python identity, which passed the "both truly nomask" cases but failed
    live on `a_none_b_nomask`/`both_all_false` cases (an operand carrying an
    explicit, real, all-`False` mask array) where real numpy still shrinks
    the result to `nomask` and the identity-only check did not -- fixed via
    `make_mask(combined, shrink=True)` on the concatenated mask, matching
    `numpy.ma.concatenate`'s always-shrink (modulo the all-identity-nomask
    fast path, which is an optimization producing the same observable
    result) contract.

The remaining Phase-5-candidate items (the contiguity utilities section 5's
row 141 actually names, plus `concatenate`/`stack`/`atleast_1d/2d/3d`/`diag`/
`diagonal`/`mask_rows`/`mask_cols`/`compress_rows`/`compress_cols`/
`compress_rowcols`/`compress_nd`/`flatten_mask`) remain out of scope --
`clump_masked`/`clump_unmasked`/`notmasked_edges`/`notmasked_contiguous`/
`flatnotmasked_contiguous`/`flatnotmasked_edges` inherently require a Python
loop over contiguous mask runs to build a list of `slice` objects (verified
against real numpy's own `numpy/ma/extras.py` source), which the project's
no-Python-loop rule forbids; `concatenate`/`stack`/`atleast_1d/2d/3d`/`diag`/
`diagonal` were declined because their required toplevel base function is
itself undeclared/param-blind in `ionp.__ion_state__`, not because of any
computational defect.

## 5c. Phase 6 status (2026-08-03) -- the DOMAINED unary/binary families

What shipped (`ionp/ma/core.py`, `ionp/ma/__init__.py`, `ionp/_state/ma.py`;
differential cases in `ionp/tests/differential/ma_cases.py`): 14 items, all
bit-exact against real numpy 2.5.1, 0 failing/phantom/untested --
`sqrt`/`log`/`log2`/`log10`/`arcsin`/`arccos`/`arccosh`/`arctanh` (domained
unary, real `type(np.ma.<name>) is _MaskedUnaryOperation` with a non-`None`
`.domain`) and `divide`/`true_divide`/`floor_divide`/`remainder`/`mod`/
`fmod` (`_DomainedBinaryOperation`).

CORRECTION: section 6 (and `ionp/_state/ma.py`'s pre-Phase-6 module
docstring) claimed the domained binary ops (`divide`, `power`, `hypot`) were
excluded from this document's scope because their toplevel base functions
were undeclared. That was accurate when written but went stale -- a
parallel toplevel-lane agent has since landed `divide`/`floor_divide`/
`fmod`/`mod`/`remainder`/`true_divide` as "exact" in `ionp.__ion_state__`.
Phase 6 builds on those six now-exact bases. `power` and `hypot` remain
genuinely excluded: `np.ma.power` is a plain `function`, not built on
`_DomainedBinaryOperation`/`_MaskedBinaryOperation` at all (an unreproduced,
different implementation); `np.ma.hypot` IS `_MaskedBinaryOperation` (the
PLAIN, non-domained family, so out of THIS phase's family scope regardless),
and its base `ionp.hypot` is separately still undeclared.

UPDATE 2026-08-03 (Monday, same-day Phase 0 follow-up): the fill_value
gaps documented in this section as "not fixed, out of this phase's scope"
turned out to have a real bug behind them, not just an asymmetry -- see
`ionp/_state/ma.py`'s Phase 6 comment block for the full incident writeup
(revoked, root-caused, and fixed the same day). Summary of what changed
since the paragraphs below were written:

  - `MaskedArray` (Phase 0) now DOES track "explicit vs default": a new
    `_fill_value_explicit` flag plus a `_fill_value` that stays `None`
    until a user explicitly sets one (constructor `fill_value=` or the
    setter), mirroring real numpy's own `_fill_value is None` sentinel
    exactly. `.fill_value` computes `_default_fill_value(dtype)` fresh on
    every access when not explicit, off the array's OWN dtype -- never a
    stale pre-promotion snapshot.
  - A new `MaskedArray._update_from(source)` (mirroring real numpy's own
    method of the same name) carries `_fill_value`/`_fill_value_explicit`
    verbatim, with no recast -- this is what every wrapper family below
    now uses instead of passing a computed `fill_value=` through the
    public constructor.
  - **0-d scalar collapse** (still a real, ACKNOWLEDGED-OPEN gap, genuinely
    untouched by the fill_value fix above): real numpy's
    `_MaskedUnaryOperation`/`_MaskedBinaryOperation.__call__` both end with
    `if not result.ndim: return masked if <mask bit> else <bare computed
    scalar>` -- a 0-d operand NEVER produces a `MaskedArray` back. Phase
    1's `make_masked_unary`/`make_masked_binary` still do not special-case
    this (verified live: `ionp.ma.sin(ionp.ma.MaskedArray(2.0)).__class__`
    is still `MaskedArray`, real numpy's is `numpy.float64`) -- a
    genuinely separate defect from fill_value propagation, left for a
    future phase. Phase 6's `make_masked_domained_unary`/
    `make_masked_domained_binary` implement the correct 0-d path for their
    own 14 items only, unchanged by this update.
  - **fill_value inheritance asymmetry**: FIXED. Real numpy's
    `_update_from`-style rule inherits `fill_value` from whichever
    ORIGINAL argument was already a `MaskedArray` (preferring `a`), not
    unconditionally from `a`'s coerced-to-masked form. `make_masked_binary`
    now uses a shared `_fv_source(a, b, am, bm)` helper (in
    `ionp/ma/core.py`, also used by `make_masked_domained_binary`) to pick
    the correct source, then `MaskedArray(...)._update_from(that source)`
    -- verified live: `np.ma.add([1.,2.,3.], np.ma.masked_array([1.,2.,3.],
    fill_value=-9.0)).fill_value` is `-9.0`, inherited from `b`; ionp now
    matches.
  - The Phase-6-specific dtype-promotion wrinkle previously worked around
    by a value-equality heuristic (`_domained_binary_fill_value`: "carry
    the source's fill_value across a dtype change UNLESS it exactly equals
    that source's own dtype's default") has been DELETED, not patched --
    that heuristic is provably wrong on its own defeat case (a user who
    explicitly sets a fill_value EQUAL to their array's own dtype default
    is indistinguishable from "never set" under value comparison, but real
    numpy still carries it through a promotion with its original dtype,
    verified live: explicit `fill_value=999999.0` on an int64 array through
    `ma.log` gives numpy `(999999.0, 'int64')`, not recast). The
    `_fv_source`/`_update_from` mechanism above replaces it structurally
    (tracking explicitness directly, the same way real numpy does) rather
    than approximating it by value, and has no such residual edge case.

The remaining Phase-6-candidate items considered and declined:
  - `ma.power`, `ma.hypot`: see the CORRECTION paragraph above.
  - `ma.tan`: real `type(np.ma.tan) is _MaskedUnaryOperation` with domain
    `_DomainTan`, genuinely domained -- but `ionp.tan` is not this phase's
    concern; excluded going into this phase already (see section 6/
    `ionp/_state/ma.py`'s existing `tan` note) for the same base-ULP-gap
    reasoning as `cos`/`tanh`.
  - `ma.equal`/`ma.not_equal`/`ma.greater`/etc. (comparison predicates):
    excluded, base still undeclared in `ionp.__ion_state__`.
  - `ma.angle`: NOT a domained item at all (real `.domain is None`) --
    belongs to the Phase 1 plain-unary family shape, not Phase 6; left for
    a future phase rather than mixed into this one's family-boundary
    discipline.
  - `ma.around`/`ma.clip`/`ma.argsort`/`ma.diff`/`ma.ediff1d` and the
    reduction family (`ma.all`/`ma.any`/`ma.count`/`ma.count_masked`/
    `ma.cumsum`/`ma.cumprod`/`ma.argmax`/`ma.argmin`/`ma.amax`/`ma.amin`/
    `ma.anom`/`ma.anomalies`/`ma.average`/`ma.allclose`/`ma.allequal`/
    `ma.alltrue`): none of these are `_MaskedUnaryOperation`/
    `_DomainedBinaryOperation` family members (several have undeclared
    toplevel bases per this task's own dependency warning; the rest are
    reduction-shaped, a structurally different problem than "apply a ufunc,
    carry the mask") -- out of this phase's "elementwise/ufunc-wrapper
    cluster" mandate, left for a future phase.

## 5d. Reduction-family PANIC, revoked then restored (documented late 2026-08-07, Monday)

Not previously written up in this file even though it happened within the
timeframe the sections above cover -- found while re-verifying this
document's own "15 of 32 revoked items restored" claim from a fill_value-
boxing session's task brief, and confirmed independently rather than taken
on faith: `anionpy/_state/ma.py` records 13 reduction items
(`all`/`alltrue`/`amax`/`amin`/`any`/`count`/`count_masked`/`max`/`min`/
`sometrue` and others in the same PANIC group -- see its own comment for
the exact 13) that originally hit a Rust-side
`PanicException "index out of bounds: the len is 0"` on a 0-d-with-explicit-
axis call shape, out of an originally-32-item revoked set (`anionpy/_state/
ma.py` line ~161's "Originally 32" note). `grep -c "RESTORED 2026-08-03"
anionpy/_state/ma.py` measures exactly 15 restorations (`alltrue`,
`sometrue`, `count_masked`, `amax`, `amin`, `argmax`, `argmin`, `cumsum`,
`cumprod`, `ptp`, and five more spread through the file's reduction/Phase-6
declarations) once the 0-d-x-explicit-axis defect underlying the PANIC was
root-caused and fixed, each bite-tested (neutering the fix makes the new
`ma_cases.py` differential cases -- 12 axis forms x 3 mask states x
keepdims per item -- fail). The remaining 17 of the original 32 are either
still genuinely excluded for unrelated reasons (undeclared toplevel base,
Python-loop requirement, etc. -- see sections 5b/5c above) or were never
in this reduction-PANIC group to begin with. This section exists so a
future reader does not have to re-derive the 15/32 figure from raw `grep`
the way this update did.

## 6. Known hazards

- **`ma.MaskedArray` is 1 manifest item but the largest piece of work.** Its
  items-per-effort ratio is the worst on the board and it is still first,
  because everything else needs it. Do not let the ledger number tempt a
  reordering.
- **`test`** is `np.ma.test` — a test-runner entry point, not numerics. Decide
  whether it is faithful to declare it at all.
- **`mvoid`, `bool_`, `frombuffer`, `fromflex`, `flatten_structured_array`,
  `make_mask_descr`** touch structured/void dtypes, which are outside ionp's 14
  numeric variants. These are **blocked on the DType enum decision**, not on ma.
  That is ~6 items and they should be declared absent with a pointer to the
  DType disclosure, not faked.
  **CORRECTED 2026-08-03 (Phase 3 implementation): `MaskType` does NOT belong in
  this bucket.** Verified live `np.ma.MaskType is np.bool_` -- it is plainly the
  scalar bool dtype used for every mask array in this whole module, the same
  type ionp already imports as `ionp.bool_`. It needs no DType-enum work and is
  declared `exact` (`ionp/_state/ma.py`).
- **`polyfit`, `vander`, `corrcoef`, `cov`, `convolve`, `correlate`, `dot`,
  `inner`, `innerproduct`, `outerproduct`** are the mask-aware halves of
  functions whose *unmasked* versions are themselves not all declared. Do not
  declare a masked version whose base is undeclared — that inverts the
  dependency and creates a lie with extra steps.
- **`alltrue`, `sometrue`, `product`, `round_`, `anomalies`, `isarray`** are
  numpy's own deprecated aliases. Reproduce them including any
  `DeprecationWarning`, or decline and say so. Do not silently make them clean.
  **CORRECTED 2026-08-03 (Phase 3 implementation): `isarray` raises no
  `DeprecationWarning` on real numpy 2.5.1.** Verified live with
  `warnings.catch_warnings(record=True)` around `np.ma.isarray(...)` and
  `np.ma.isMA(...)` -- both are plain aliases for `isMaskedArray`
  (`np.ma.isarray is np.ma.isMaskedArray`), no warning fires for either. Modeled
  as plain aliases with no warning machinery (`ionp/ma/core.py`); this doc's
  blanket claim did not hold for this specific name on this numpy version.
  (`alltrue`/`sometrue`/`product`/`round_`/`anomalies` are top-level `ionp.*`
  items, not `ma.*`, and were out of this task's scope -- not re-checked here.)

## 7. Definition of done for this subsystem

Same as everywhere: an item is `exact` only when it is verified OUT OF CORPUS
and carries differential cases. For `ma` specifically, every declaration must
have been checked on all five of:

1. `nomask` input (and `a.mask is nomask` identity preserved).
2. Fully-masked input.
3. Partially-masked input.
4. Empty input.
5. `fill_value` propagation through the operation.

A masked-array function that is right on partially-masked data and wrong on
`nomask` is the default failure mode here, not an edge case.
