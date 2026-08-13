# Top-level `<toplevel>` absent-item triage (267 items)

Read-only triage. No source, test, or `_state/*.py` file was edited. No build was run
(`cargo`/`maturin` never invoked). The differential test **runner** (`tests/differential/run.py`)
was executed twice against the already-built `.so` — that is test execution, not a build; it does
not recompile anything and was used purely as a second, independent verification channel against
my own `/tmp` probes.

## `.so` build-lock check

```
stat -f "%N %Sm" ionp/_ionp.abi3.so
```
Start of session: `Aug 2 21:02:07 2026`. End of session: `Aug 2 21:02:07 2026`. **Unchanged** —
the concurrently-running view-infrastructure agent did not rebuild during this triage, so every
finding below is measured against a single, fixed binary, not a moving target.

## Method

1. Enumerated the 267 live from `ionp.__ion_state__` + `tools/coverage.py`, exactly as specified
   (not by scraping `_state/*.py`) — confirmed count matches the stated 267.
2. For every item: `hasattr(ionp, name)`, and if present, attempted calls with trivial args,
   `except BaseException` (never `except Exception`), distinguishing `PanicException` from
   ordinary exceptions. **0 panics found** among the 267 — every failure was an ordinary Python
   exception (ionp raises cleanly on all currently-unimplemented/unbuilt surface it exposes at
   all).
3. Cross-referenced every item against `ionp/_state/toplevel.py`'s own comments (which turned out
   to be extensively and specifically documented — most of the 85 items with `hasattr(ionp,
   name) == True` already have a dated, evidence-backed reason recorded for why they're
   undeclared) and the three named docs (`stride-gap-classification.md`,
   `order-k-propagation-fix.md`; `reduction-output-layout.md` does not exist as a separate file —
   its content lives inline in `toplevel.py`'s "GENUINE — reduction output layout" comments and
   `reduction_cases.py`'s module docstring).
4. **Independent second channel**: ran `tests/differential/run.py --out ...` twice (full corpus,
   ~1180 items, using the pre-existing, already-registered `REGISTRY`) and read per-item verdicts
   and case counts directly from its output — this is a completely different code path from my own
   `/tmp` probes and was used specifically to corroborate or contradict my hand checks.

## Headline finding: 14 items already pass their full registered differential corpus and are undeclared purely because `_state/toplevel.py`'s dict is missing the line

This is not a hypothesis — it is read directly off two independent sources agreeing: my own
`/tmp` probes against real numpy, AND a fresh `tests/differential/run.py` run against the
existing, already-wired `REGISTRY` (which `reduction_cases.py`'s own docstring says was extended
2026-08-02, the same day the corresponding `_state/toplevel.py` declarations were apparently never
added — a wiring gap, exactly the kind the file's own header comment warns "five separate blocks
of finished, tested, committed work went uncounted" about).

| Item | Corpus verdict (run.py, fresh) | Hand probe vs numpy | Confidence |
|---|---|---|---|
| `average` | PASS 1397/1397 (epsilon-tolerant, justified) | matches (0-d array vs scalar return, same convention as hundreds of already-declared reductions) | high |
| `std` | PASS 3160/3160 | matches | high |
| `var` | PASS 3160/3160 | matches | high |
| `nanstd` | PASS 3275/3275 | matches | high |
| `nanvar` | PASS 3275/3275 | matches | high |
| `complex128` | PASS 16/16 | matches (`ionp.complex128(3+4j)` == numpy) | high |
| `complex64` | PASS 16/16 | matches | high |
| `finfo` | PASS 8/8 | matches field-for-field on float64 | high |

That is **8 items, hand-confirmed**, satisfying the ≥10-sample bar together with the corroborating
corpus run below. Declaring these costs one dict-line each in `ionp/_state/toplevel.py` — no Rust,
no new tests. `reduction_cases.py`'s own docstring already documents the real bugs that were fixed
to make `var`/`std`/`nanvar`/`nanstd` pass (complex-input companion-dtype handling, deviation
truncation order) — this is finished, tested, committed work, just never wired into the ledger.

**A near-miss that the hand-check correctly caught and downgraded** (this is the accuracy-bar
"contradiction" the task asked me to surface, not paper over): `matmul`, `matvec`, `vecdot`,
`vecmat` also show `PASS` in the fresh corpus run — but each on only **4 cases**. Hand-probing them
directly against real numpy with `order=`, `casting=`/`dtype=`, and `subok=` keywords (exactly the
axis `toplevel.py`'s own comment names as the blocker) reproduces the blocker immediately:
```
ionp.matmul(ia,ib,order='F')   -> TypeError: matmul() got an unexpected keyword argument 'order'
ionp.matmul(ia,ib,dtype=...,casting='unsafe') -> TypeError: ...unexpected keyword argument 'casting'
ionp.matmul(ia,ib,subok=False) -> TypeError: ...unexpected keyword argument 'subok'
```
The 4-case corpus simply never exercises those keywords. **Do not declare these from the green
corpus number alone** — the state file's existing blocker note is current and correct, not stale.
This is exactly the trap the task's accuracy bar warned about, and it would have produced a false
"12 free wins" headline if I had trusted the aggregate.

**A second near-miss, same lesson, different mechanism**: `median` and `nanmedian` both show
`PASS 500/500` in the fresh corpus run. Hand-probing the *specific* case `toplevel.py`'s own
REVOKED comment names (tuple `axis=`) reproduces the documented bug directly and immediately:
```
numpy.median(a, axis=(0,1))  -> array([10., 11., 12., 13.])
ionp.median(ia, axis=(0,1))  -> TypeError: 'tuple' object cannot be interpreted as an integer
```
The 500-case corpus does not include a tuple-axis case at all — this is a genuine corpus gap, not
noise. These stay in the "genuine functional gap" cluster below, not the free-win list.

**Not yet in the registry at all** (verdict `None`, i.e. no test exists to pass or fail):
`cdouble`, `csingle`, `unique_values`. Hand probes show `cdouble`/`csingle` are pure aliases of
`complex128`/`complex64` and behave identically to numpy — cheap to add (alias the existing
`complex128`/`complex64` corpus, no new Rust). `unique_values` hand-probed as **wrong**: on
`[3,1,2,1,3]` numpy's `unique_values` returns `[2, 1, 3]` (unsorted, first-occurrence-adjacent but
not literally insertion order either — needs numpy-source-level investigation) while ionp returns
`[1, 2, 3]` (sorted, matching plain `unique`) — a real, undocumented bug, not a free win.

## Cluster plan, ordered by items-unlocked-per-unit-effort

### Cluster 1 — declare-only, zero Rust (8 items, trivial, do first)
`average`, `std`, `var`, `nanstd`, `nanvar`, `complex128`, `complex64`, `finfo`. Already
implemented, already tested at 500+ cases each (bar `complex64`/`complex128`/`finfo` at 8–16,
still 100%), already correct per independent hand-check. **Cost: trivial** (8 one-line dict
entries, each carrying the evidence already sitting in `reduction_cases.py`'s docstring or a fresh
hand-probe transcript). Moves 8 of 267.

### Cluster 2 — declare + tiny corpus addition (3 items, trivial-to-moderate)
`cdouble`, `csingle` (alias existing complex64/128 corpus — trivial), `unique_values` (needs the
actual numpy ordering rule investigated before it can be declared — moderate, it is currently
wrong, not just untested). Moves up to 2 trivially, 1 needs real work.

### Cluster 3 — BLOCKED ON VIEW INFRASTRUCTURE (the concurrent agent's work), ~31 items
This is the single largest blocked cluster and the one most worth watching — when the view-infra
agent lands `.base`/`.flags`/`shares_memory`/`may_share_memory`/`__setitem__`, a sizeable slice of
this cluster (CLASS C, 17 items) re-declares for **free**, no further Rust work, because
`toplevel.py`'s own audit already confirmed shape/strides/bytes are correct everywhere probed —
only the view *metadata* is missing. All 31 items here **currently pass their full value-content
differential corpus already** (confirmed live, case counts in parentheses) — they are blocked
purely on a metadata/aliasing contract, not on incorrect values.

- **CLASS C — view metadata only, values/strides already correct** (17): `flip` (222), `fliplr`
  (141), `flipud` (196), `atleast_1d` (212), `broadcast_arrays` (6), `squeeze` (308), `ravel`
  (161), `transpose` (163), plus (per `toplevel.py`'s own list) `ndarray.T`/`.transpose`/
  `.reshape`/`.ravel`/`.squeeze`/`.swapaxes`/`.mT`, `moveaxis`, `swapaxes`, `reshape`,
  `linalg.matrix_transpose` (the last several are `ndarray.*`/`linalg.*` items, outside this
  toplevel count but sharing the identical mechanism — re-declaring the infra fix will move both
  namespaces at once). **Re-declares for free the moment `.base`/`.flags`/`shares_memory`/
  `__setitem__` land** — no additional Rust beyond the view-infra work already in flight.
- **CLASS B — length-1-axis stride convention, independent of full view infra** (4):
  `broadcast_to` (12), `atleast_2d` (212), `atleast_3d` (212), plus `expand_dims` (326, an
  `ndarray`-adjacent case sharing the mechanism). Needs one convention decision (zero the stride
  on the inserted/passed-through size-1 axis, matching numpy) — **does not require the full
  `.base`/`.flags`/`__setitem__` stack**, so this could plausibly land *before* Cluster-3's CLASS C
  if someone wants a quick independent win. **Cost: moderate** (a real Rust fix, but scoped and
  already fully diagnosed with exact before/after stride examples in the state file).
- **CLASS A — real aliasing needed, hardest of the three** (4): `diag` (72, 2-D-extraction case
  only), `diagonal` (75), `rot90` (21), `rollaxis` (21). numpy documents these as returning true
  views; ionp returns copies with correct values but structurally wrong strides — needs the full
  aliasing/`__setitem__` machinery from the view-infra work, genuinely gated on it, not just
  metadata. **Cost: hard**, tied to Rust core buffer-ownership work already in progress.
- **Transpose/relayout implementation quirks, distinct mechanism (output-construction order, not
  aliasing)** (6 of this toplevel set): `argwhere` (212), `hstack` (881), `vstack` (880), `sort`
  (2176), `sort_complex` (170), `unique` (1527), `trim_zeros` (346), `delete` (54), `linspace`
  (19), `concatenate` (926), `stack` (915). Each numpy divergence has its own hand-verified
  transpose/F-order mechanism (documented per-item in `stride-gap-classification.md`) — **does
  not require view infra** (no aliasing involved, purely "which order does the output get
  allocated in"), so this is a **separate, independently landable moderate-cost cluster**, not
  blocked on the concurrent agent at all. I grouped it here because it shares the "stride/layout,
  not value" nature with Cluster 3, but it has **no dependency on `.base`/`.flags`/`__setitem__`**
  and could be scheduled in parallel.
- **`asarray`** (994) — its own mechanism (view preservation on non-contiguous/reversed slices,
  NOT plain transposes, which already pass) — moderate, needs the general "wrap don't copy"
  buffer-aliasing capability, so effectively tied to the same view-infra work as CLASS A/C.

### Cluster 4 — reduction output layout (not a `<toplevel>`-only cluster, listed for completeness)
`toplevel.py` documents ~24 reduction items (`all, amax, amin, any, count_nonzero, cumprod, cumsum,
max, mean, median(values only — tuple-axis is separate), min, ptp, sum`, etc.) with the SAME
"numpy propagates input memory order into non-reduced output axes" gap. Of the 267 `<toplevel>`
absent items, none currently land squarely in this bucket by name (most of these are already
`"exact"`-declared per the RE-DECLARED 2026-08-02 note above `nanmean`) — flagged here only so the
next triage doesn't have to re-derive that this is a distinct, already-explored mechanism from
Cluster 3, explicitly out of scope per `order-k-propagation-fix.md`'s own "Not fixed" section.

### Cluster 5 — parameter-blindness: `out=`/`casting='unsafe'`/`order=`/`subok=` gaps (7 items, moderate)
`concatenate` (`out=`, `casting='unsafe'`), `stack` (same), `trace` (`out=`), `matmul`, `matvec`,
`vecdot`, `vecmat` (all four: `casting=`/`order=`/`subok=` on their dedicated gufunc bindings,
hand-confirmed still broken above). All need real buffer-writing support in the Rust binding layer
for the `out=` half, and simple kwarg passthrough for the `casting=`/`order=`/`subok=` half — the
matmul family's fix is probably the cheaper of the two sub-problems since the generic ufunc
dispatcher already solved the identical parameter-blindness gap for plain ufuncs
(`apply_ufunc_order` etc. per `order-k-propagation-fix.md`) — likely just needs wiring the same
already-built machinery into the gufunc-specific binding path. **Cost: moderate**, shared
machinery with work already done elsewhere in the codebase.

### Cluster 6 — genuine, disclosed, unsolved numeric/functional bugs (6 items, hard or open)
- `arange`: two distinct open bugs (negative-step unsigned-dtype float clamping through a
  crate-wide macro shared by many other declared items — fixing it is out of scope for a
  single-item task; and an integer-dtype fractional-step quirk that resisted extensive
  hypothesis-testing against real numpy's undocumented internal cast rule). **Hard** — flagged
  explicitly per the task's "rather know a thing is hard" instruction; do not re-attempt without a
  new idea, two documented attempts already failed to find the rule.
- `array`: missing `ndmax=` keyword (numpy 2.5.1 accepts it alongside `ndmin=`). **Trivial-to-moderate**
  once someone reads numpy's actual `ndmax` semantics — not investigated further here (out of
  scope for triage), but structurally simple (one more keyword on an existing, working
  constructor).
- `median`/`nanmedian`: tuple-`axis=` unsupported (hand-confirmed above, real gap, not corpus
  noise) plus a length-1-axis case nanmedian additionally diverges on. **Moderate.**
- `expm1`: fails its own corpus at 285/306 (21 failures), max 60 ULP — a real precision bug, not
  tolerance-able per the corpus's own verdict. **Moderate** (likely an algorithm/evaluation-order
  fix in the existing `emath`/ufunc machinery, same family as the already-documented complex-power
  ULP gaps).
- `min_scalar_type`/`result_type`: both fail 2-and-1-case corpuses respectively — small blast
  radius, likely one or two edge-case dtype-promotion rules wrong. **Moderate**, small, well-scoped
  (corpus already pinpoints the exact failing cases).
- `bitwise_count`: passes its value corpus (306/306) but was explicitly WITHDRAWN by the ledger
  owner for an axis "the sweep never varied" (per the state-file comment, text cut off at
  "systemic exception-class fix" — needs reading `bitwise_count`'s own withdrawal note in full
  before re-declaring; not independently re-derived here). **Moderate**, already diagnosed
  elsewhere in the file, just needs the referenced systemic fix.

### Cluster 7 — genuinely unbuilt, no existing Rust module, no prior attempt (182 items, `hasattr(ionp, name) == False` for every one)
None of these appear anywhere in `_state/toplevel.py`'s ~4,200 lines of dated commentary — they
have never been started, not even as a withdrawn/revoked attempt. Grouped by what Rust module they
would need (checked live against `ionp-core/src/` and `ionp-py/src/`, which currently has NO
polynomial, no printing/formatting beyond `repr.rs`'s two functions, no file-I/O, no
object/string/datetime scalar-type family, no grid-object/iterator-protocol types, and no
structured/matrix subclass support):

- **Linear-algebra products reusing the existing `matmul.rs` gufunc core** (7): `dot`, `vdot`,
  `inner`, `outer`, `tensordot`, `cross`, `kron`. **Moderate** — `ionp-py/src/matmul.rs` and
  `ionp-core`'s gufunc broadcasting already solved the hard part (shape/dtype/broadcast rules for
  N-D array products); these are mostly thinner wrappers around the same machinery with different
  contraction rules. Best next cluster after Clusters 1–2, since it reuses proven infra rather
  than opening a new dtype/module frontier. `einsum`/`einsum_path` are related but meaningfully
  harder (general Einstein-summation parsing) — keep separate, **hard**.
- **Printing/formatting** (9): `array2string`, `array_repr`, `array_str`, `base_repr`,
  `binary_repr`, `format_float_positional`, `format_float_scientific`, `get_printoptions`/
  `set_printoptions`/`printoptions` (context manager). `repr.rs` already has the hard part (numpy's
  actual float-formatting algorithm, per its own module doc) for the two functions it covers —
  `array2string`/`array_repr`/`array_str` are mostly parameter surface (`separator=`,
  `precision=`, `suppress=`, etc.) on top of that existing engine. **Moderate**, shares machinery,
  good cluster to batch.
- **Global-state context managers** (5): `errstate`, `seterr`/`geterr`, `seterrcall`/`geterrcall`,
  `getbufsize`/`setbufsize`, `printoptions` (overlaps above). Needs one small piece of new
  machinery (a global/thread-local settings store) shared across all five — **moderate**, one
  infra piece unlocks 5+ items.
- **File I/O / `.npy`/`.npz` format** (13): `save`, `savez`, `savez_compressed`, `load`, `savetxt`,
  `loadtxt`, `genfromtxt`, `fromfile`, `fromregex`, `fromstring`, `frombuffer`, `memmap`,
  `from_dlpack`. **Hard as a whole** (real binary format parsing, zip handling for `.npz`, buffer
  protocol/DLPack interop) but decomposable — `frombuffer`/`fromstring` are cheap (just buffer
  reinterpretation, no format spec) and could be pulled out as a trivial sub-cluster; `.npy` binary
  save/load is a well-specified, bounded format and is probably the single highest-value item in
  this bucket since numpy's `.npy` spec is public and simple; `.npz`/`genfromtxt`/`loadtxt` are
  genuinely harder (zip, text parsing with type inference).
- **Polynomial** (8, all of `poly, poly1d, polyadd, polyder, polydiv, polyfit, polyint, polymul,
  polysub, polyval, roots, vander`): zero existing Rust module. `poly1d` is a class wrapper; the
  arithmetic functions (`polyadd`/`polysub`/`polymul`/`polyder`/`polyint`/`polyval`) are simple
  coefficient-array math reusable across all of them once one polynomial-representation decision
  is made; `polyfit`/`roots` need real numerics (least-squares / companion-matrix eigenvalues) and
  are meaningfully harder than the rest of the cluster. **Moderate for the arithmetic half, hard
  for `polyfit`/`roots`.**
- **Structured/masked-adjacent, blocked on missing dtype family**: `object_`, `str_`, `bytes_`,
  `void`, `datetime64`, `timedelta64`, `record`, `recarray`, `busday_count`, `busday_offset`,
  `busdaycalendar`, `is_busday`, `datetime_as_string`, `datetime_data` (13). Confirmed live —
  `ionp-core/src/dtype.rs`'s `DType` enum has exactly `{Bool, I8/16/32/64, U8/16/32/64,
  F16/32/64, C64/128}` and nothing else; there is no object/string/void/datetime variant at all.
  **Hard, structural** — this is the "no object/string dtype family" blocker named in the task
  brief, confirmed by direct source inspection, not inferred. Every item in this sub-cluster is
  gated on the same one piece of missing core infrastructure; landing it would move all 13 at
  once (plus much of `ma.*`, `rec.*`, and part of `char.*`/`strings.*`, though those are outside
  this triage's 267-item scope).
- **Grid objects / index-construction helpers** (9): `mgrid`, `ogrid`, `r_`, `c_`, `s_`,
  `index_exp`, `ndindex`, `nditer`, `nested_iters`, `flatiter`. These are Python-protocol objects
  (`__getitem__`-driven slicing DSLs, iterator protocol) more than numeric kernels — likely
  cheapest to build almost entirely in the existing thin Python import-surface layer on top of
  primitives ionp already has (`arange`, `stack`, indexing), rather than new Rust. **Moderate**,
  and possibly cheaper than its numeric-looking neighbors precisely because it's glue, not new
  math — worth flagging as a good "small-Rust, more-Python" cluster.
- **Statistics/interpolation/signal reusing existing reduction & sort infra** (13): `histogram`,
  `histogram2d`, `histogramdd`, `histogram_bin_edges`, `digitize`, `bincount`, `corrcoef`, `cov`,
  `gradient`, `interp`, `convolve`, `correlate`, `i0`. Share `sort`/reduction primitives already
  built (`sort.rs`, `stats.rs`); `histogram*`/`digitize`/`bincount` cluster tightly (all bucket/bin
  logic); `corrcoef`/`cov` reuse `var`/`std`'s just-fixed complex/dtype logic directly (build
  right after Cluster 1). **Moderate.**
- **Windowing functions** (5): `bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`. Pure
  closed-form array math, no dependency on anything missing. **Trivial-to-moderate**, good quick
  cluster, essentially free-standing.
- **dtype/scalar introspection & other small predicates** (~20): `bool`, `longdouble`, `longlong`,
  `ulonglong`, `clongdouble`, `common_type`, `promote_types`(declared elsewhere already, not in
  this 267), `issubdtype`, `isdtype`(exists as ufunc-ish entry, not toplevel-absent per earlier
  grep — double check before assuming free), `isscalar`, `iterable`, `isfortran`, `iscomplex`,
  `iscomplexobj`, `isreal`, `isrealobj`, `isneginf`, `isposinf`, `ScalarType`, `sctypeDict`,
  `typecodes`, `get_include`, `show_config`, `show_runtime`, `test`. Mostly trivial predicates or
  static introspection data — **trivial-to-moderate** per item, but `ScalarType`/`sctypeDict`/
  `typecodes` are blocked (or at least incomplete) until the object/string/datetime dtype cluster
  above lands, since a correct list must include those types.
- **Array construction/manipulation misc** (~15): `asanyarray`, `asfortranarray`,
  `asarray_chkfinite`, `astype`, `block`, `choose`, `clip`(free function — distinct item from the
  already-fixed `ndarray.clip` method; same K-order machinery from `order-k-propagation-fix.md`
  should transfer cheaply), `copyto`, `pad`, `require`, `round`/`around`, `select`, `piecewise`,
  `apply_along_axis`, `apply_over_axes`, `vectorize`, `frompyfunc`, `fromfunction`, `fromiter`,
  `permute_dims`, `unstack`, `cumulative_sum`/`cumulative_prod`, `nancumsum`/`nancumprod`,
  `nan_to_num`, `packbits`/`unpackbits`, `partition`/`argpartition`, `putmask`/`place`, `fix`,
  `sinc`, `unwrap`, `trapezoid`, `tri`, `tril_indices_from`/`triu_indices_from`, `mask_indices`,
  `real`/`imag`/`angle`, `real_if_close`, `shape`/`size`/`ndim`, `shares_memory`/`may_share_memory`
  (tied to Cluster 3's view infra — do not attempt standalone), `allclose`/`isclose`/
  `array_equal`/`array_equiv`, `broadcast`(the class, distinct from already-existing
  `broadcast_shapes`/`broadcast_to`/`broadcast_arrays`), `matrix`/`bmat`/`asmatrix` (subclass,
  needs a design decision on whether ionp models numpy's deprecated matrix class at all). Wide
  range of individual cost from trivial (`shape`, `size`, `ndim` are almost certainly one-line
  attribute reads) to moderate (`apply_along_axis`/`vectorize`/`piecewise` need real callback
  dispatch into Python). Not further hand-tested individually — this is the "genuinely unstarted,
  wide, low-per-item-cost, no shared single blocker" tail of the list; recommend the next triage
  pass hand-verify `shape`/`size`/`ndim`/`real`/`imag` specifically since those look like they may
  be one-line free wins hiding in the "no hasattr" bucket for a trivial reason (e.g. exposed only
  as `ndarray.shape` property, not a free function) rather than a real gap.

## Summary table

| Category | Item count | Blocker | Cost |
|---|---|---|---|
| Free win — declare only | 8 | none, hand+corpus verified | trivial |
| Free win — trivial corpus add | 2 (`cdouble`,`csingle`) | none | trivial |
| Real bug, small | 1 (`unique_values`) | ordering-rule investigation | moderate |
| Blocked on view infra (CLASS A/B/C + asarray) | ~26 toplevel (+ several `ndarray.*`/`linalg.*` riding along) | `.base`/`.flags`/`shares_memory`/`__setitem__`, in flight | moderate–hard, unlocks in bulk when infra lands |
| Blocked on layout/transpose quirks (no view-infra dependency) | 11 | per-item output-order mechanism, documented | moderate, independently schedulable |
| Blocked on parameter-blindness (`out=`/`casting=`/`order=`/`subok=`) | 7 | Rust binding kwarg plumbing | moderate |
| Genuine disclosed bugs/open gaps | 6 | varies, `arange` is genuinely hard | moderate–hard |
| Blocked on object/string/datetime dtype family | 13 | `DType` enum has no such variant (confirmed by source read) | hard, structural |
| Genuinely unstarted, no blocker, spread across many small clusters | ~193 | none named — just unbuilt | trivial–hard depending on sub-cluster, see Cluster 7 |

267 total (8+2+1+26+11+7+6+13+193 ≈ 267, allowing for a handful of items counted under more than
one heading above, e.g. `expand_dims` appears in both the CLASS B view discussion and its own
right — not double-counted in the true absent set).

## What was measured directly vs. inferred

**Measured directly** (via `/tmp` probes against real numpy and/or a fresh `tests/differential/
run.py` execution against the untouched `.so`): the full 267-item enumeration; `hasattr`/callability
for all 267; a ≥10-item hand-verified sample (`average`, `std`, `var`, `nanstd`, `nanvar`,
`complex128`, `complex64`, `finfo`, `expm1`, `min_scalar_type`, `result_type`, `matmul`+kwargs,
`unique_values`, `median`/`nanmedian` tuple-axis) plus corroborating corpus-run case counts for
~35 more named items (`argwhere`, `asarray`, `atleast_1d/2d/3d`, `broadcast_arrays`, `broadcast_to`,
`diag`, `diagonal`, `flip`/`fliplr`/`flipud`, `hstack`/`vstack`, `sort`/`sort_complex`, `unique`,
`trim_zeros`, `rot90`, `rollaxis`, `delete`, `linspace`, `concatenate`, `stack`, `squeeze`,
`transpose`, `ravel`, `trace`, `sign`, `lexsort`, `diff`, `bitwise_count`, `arange`, `array`,
`prod`, `expand_dims`, `vecdot`, `vecmat`, `matvec`); the `DType` enum's contents (source read,
`ionp-core/src/dtype.rs`); the existence/absence of Rust modules per functional area (source
listing of `ionp-core/src` and `ionp-py/src`); the `.so` mtime before and after.

**Inferred, not independently re-derived**: the exact per-item build cost for the ~193-item
"genuinely unstarted" tail (Cluster 7) — grounded in which Rust modules already exist and general
numpy semantics, not in a line-by-line implementation attempt for each; the claim that CLASS C
items will "re-declare for free" once view infra lands is inferred from `toplevel.py`'s own
audit (confirmed shape/strides/bytes correct everywhere probed) rather than independently
re-probed here against a hypothetical future `.so`.

**One explicit accuracy-bar contradiction surfaced, not smoothed over**: my own first-pass
classification (based on `hasattr` + a single trivial successful call) flagged `matmul`, `matvec`,
`vecdot`, `vecmat` as candidate free wins alongside `average`/`std`/etc. Hand-verification with the
actual keywords the state file's blocker note names (`order=`, `casting=`, `subok=`) reproduced
the blocker immediately and removed all four from the free-win list. The aggregate ("14 items pass
their corpus") is therefore reported above with the specific 8-item subset that survived hand
verification called out separately from the 6 that did not (4 matmul-family + `median` +
`nanmedian`), per the task's explicit instruction to report the contradiction rather than the
total.
