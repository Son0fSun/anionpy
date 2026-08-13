# Path to 100% — anionpy coverage plan

**Status 2026-08-06: 670 / 3091 (21.676%).** phantom 0, failing 0, untested 0.
**2421 items absent.** Suite: 1262 items / 21 failing. Ledger measured at
`289d8e6`; the suite figure re-measured after `902827c` (the `ionp` → `anionpy`
rename), which moved neither number. The "26" this line carried for a day was
stale within hours of being written — see the note at the head of this file
about re-running `coverage.py` before quoting any figure here, including the
header itself.

> The denominator moved AGAIN, 2866 -> 3091, when the curated submodule descent
> landed (task #26): the Phase 2 explosion had covered container classes but
> skipped submodule descent, so ~225 items existed in numpy's surface and in no
> ledger. Same story as the 1180 -> 2866 move below — declarations were not lost,
> a bias was removed. `RUST-QUEUE.md`'s standard holds: *a coverage number that
> only ever rises is not being audited.* This one has now fallen twice, on
> purpose, and both falls were the number getting more honest.
>
> **The prior status line (`667 / 2866 (23.27%)`, 2026-08-05) was stale for a
> day.** Anyone quoting it would have quoted a denominator that no longer
> existed. That is the same failure mode this repo has hit seven times in other
> forms: *a recorded number in a document is not evidence about the current
> tree.* Re-run `tools/coverage.py --tests <results.json>` before quoting any
> figure on this page, including this one.

> ### THE DENOMINATOR CHANGED. THE COVERAGE DID NOT FALL.
>
> Same 667 declarations, before and after. **Zero were renamed, zero orphaned.**
> The number moved from 56.53% to 23.27% because the denominator went from 1180
> to 2866 — not because anything regressed. Nothing was lost; a bias was removed.
>
> **The bias, measured:** `snapshot_surface.py` exploded `np.ndarray` into 164
> per-method items via one hardcoded call, while each of ~110 other classes
> counted as exactly **1 opaque item**. `ndarray` also kept a flat entry on top
> of its explosion, so it carried 165 weight against `ma.MaskedArray`'s 1. The
> consequence was not cosmetic: an item could be graded **exact off its
> constructor alone** while its entire method surface went unmeasured. That is
> not hypothetical — `ma.MaskedArray` was revoked on 2026-08-03 for exactly it.
>
> **The fix:** 18 container/behaviour-bearing classes are now exploded, none of
> them double-counted (including `ndarray`, whose flat entry was dropped
> 2026-08-05 — keeping it would have left the denominator honest about 17
> classes and still rigged for the 18th). `np.exceptions` and `np.dtypes`, which
> were omitted from `SUBMODULES` entirely, are now scanned. Scalar types,
> exceptions and constants stay flat and opaque deliberately: exploding those is
> the only scenario that would force existing declarations to be renamed, and
> renaming declarations to chase a number is how a ledger starts lying.
>
> 1663 of the 1687 newly-counted items are `absent` because `registry.py` has no
> per-class operand dispatch for those 17 classes yet. `absent` is the honest
> verdict for them, not a placeholder — that dispatch is real work, tracked
> separately, and it is not done.
>
> `RUST-QUEUE.md`'s standard, applied to itself: *"A coverage number that only
> ever rises is not being audited."* This is the audit. Expect the number to be
> lower and true rather than higher and flattering.

*(Superseded header: 2026-08-03 end-of-session 571 / 1180 (48.39%), 609 absent —
and every percentage below this line that predates 2026-08-05 is against the old
1180 denominator. Do not compare them to the current figure directly.)*

This document exists because a plan was demanded and did not exist — the work so
far has been optimising the next seven items rather than the remaining thousand.

*(Superseded headers: 2026-08-03 mid-session 470 / 1180 (39.83%), 710 absent;
2026-08-01 168 / 1180 (14.24%), 1012 absent.)*

> **READ THE CORRECTIONS BEFORE THE PLAN.** Every projection in this file that was
> not measured live has so far proved optimistic, in the same direction, for the
> same reason: counting "present" or "declared" as "done." Wave 1 was overstated
> 4.5×; the family table understated `fft` 12× and `linalg` 11×. The measured
> sections are marked as such. Trust those; re-derive the rest before spending on it.

Definition of done, unchanged: **0 items `absent`, every non-absent item backed
by a passing differential test.** Not "0 failing" — an absent item is not a pass,
it is an unwritten one.

## THE PROJECTION (2026-08-03) — full reconciliation of all 710 absent items

Demanded and previously missing: this document counted items but never projected
a route. Every number below is measured, not estimated. Reproduce with
`/tmp/mg_split470.py` and `/tmp/mg_ndabsent2.py`.

Manifest = 1180 = `names` 1016 + `ndarray` 70 + `ndarray_dunder` 94.
Declared 470 (names 403, ndarray 28, dunder 39). **Absent 710.**

| bucket | names | ndarray | dunder | **total** | what it costs |
|---|---:|---:|---:|---:|---|
| present, tested, **passing**, undeclared | 81 | — | — | **81** | audit only |
| present, undeclared, **untested** | 8 | 16 | 34 | **58** | write cases, then audit |
| present, test **fails** | 16 | — | — | **16** | real defects |
| **genuinely missing** | 508 | 26 | 21 | **555** | build |
| | | | | **710** | |

**131 of the 710 are already built.** That is 18% of the remaining work already
sitting in the tree unwired — the single cheapest lane, and the third time this
condition has been observed (a 47-item `testing` block once moved the ledger
91 → 138 the day it was *connected*, having been built weeks earlier).

> **⚠️ FALSIFIED 2026-08-03, same day it was written. The "131 already built" claim
> above is WRONG and the Wave-1 line below is wrong with it. Real yield: 29.**
>
> Wave 1 ran to completion across three lanes. Actual declarations: 24 (names audit)
> + 3 (`csingle`, `cdouble`, `ctypeslib.c_intp`) + 2 (`ndarray.strides`,
> `ndarray.__float__`) = **29 of the projected 131 (22%)**.
>
> The error was in the *predicate*, not the count. "Present on the object graph AND
> has a passing test group" does NOT mean "correct and merely undeclared." An
> independent re-derivation found 62 items in that bucket and **all 62 had already
> been investigated and correctly DECLINED**, each with a load-bearing evidence
> comment in `_state/`. They are not unwired — they are **defective, and already
> known to be**. A passing test group only proves agreement on the axes that corpus
> happens to sample (lesson #34); out-of-corpus probing is what actually decides,
> and it had already said no.
>
> The 3 real wins came from a bucket this document did NOT emphasise: present,
> **no test group at all**, blocked by a stale doc premise rather than a defect.
>
> **Correction to the caveat below: the `object`-inherited dunder count is 13, not
> 11** — `__getstate__` and `__setattr__` were omitted. So ~45 real, not ~47.
>
> **Not fully closed:** 2 of the 8 no-group items were left unexamined beyond an
> initial `resolve()` check. Wave 1 is 29 confirmed, not 29 exhaustive.
>
> **Generalise this.** Every future wave estimate in this file is a projection from
> a predicate, and the predicate is the fragile part. Before budgeting a wave,
> state what its predicate assumes and how it could be false. "It resolves and its
> test passes" is the weakest predicate in this document and it overstated its
> wave by 4.5×.

Caveat on the 58 untested-but-present: `hasattr` is a weak test for dunders.
Roughly 11 of the 34 (`__delattr__`, `__dir__`, `__getattribute__`,
`__init_subclass__`, `__subclasshook__`, `__sizeof__`, `__format__`,
`__reduce__`, `__reduce_ex__`, `__new__`, `__init__`) are inherited from
`object` and are **not** anionpy implementations. Assume ~47 real, not 58.

### The 555 genuinely missing, by family

| family | count | notes |
|---|---:|---|
| `ma` | 225 | designed in `MA-DESIGN.md`; ~5 mutating items gated on buffer repr |
| toplevel | 168 | ~12 sub-families; windows + real/complex near-free; 13 are pure repr/printing with no numerics |
| `random` | 58 | must be **bitstream-identical** to numpy, not merely plausible |
| `ndarray` methods | 26 | |
| `char` | 24 | string dtypes — blocked on DType enum |
| dunders | 21 | incl. `__setitem__`, `__iter__`, `__buffer__`, dlpack |
| `rec` 9, `polynomial` 8, `ctypeslib` 4, `lib` 4, `strings` 4, `testing` 2, `fft` 1, `linalg` 1 | 33 | tail |

### MEASURED STATE 2026-08-03 (supersedes the asserted table above)

Re-derived live, reconciled (`502 declared + 514 absent == 1016` names; plus 95
absent across `ndarray`/`ndarray_dunder` = **609 absent of 1180**).

| family | absent | note |
|---|---:|---|
| toplevel | 223 | was asserted as 168 — **wrong**, and the gap is the Wave-1 items that turned out defective rather than merely undeclared |
| `ma` | 148 | 77 of 225 shipped (Phases 0–4) |
| `random` | 63 | was asserted 58 |
| `char` | 24 | DType-gated |
| `fft` | 12 | **was asserted 1 — badly wrong** |
| `linalg` | 11 | **was asserted 1 — badly wrong** |
| `rec` 9, `polynomial` 8, `ctypeslib` 4, `lib` 4, `strings` 4, `testing` 3, `emath` 1 | 33 | tail |

**The three asserted numbers I got wrong (toplevel 168→223, fft 1→12, linalg 1→11)
were all wrong in the same direction: they counted "declared or present" as "done."**
Same predicate error as the falsified Wave 1 above. Assume any number in this file
that was not measured live is optimistic.

### THE DTYPE GATE, MEASURED

anionpy constructs **14** dtypes: bool, int8/16/32/64, uint8/16/32/64, float16/32/64,
complex64/128.

It refuses **8**: `longdouble`, `clongdouble`, `datetime64`, `timedelta64`, `U` (unicode),
`S` (bytes), `object`, `V` (void/structured).

That gate directly blocks **37 absent items** (`char` 24 + `rec` 9 + `strings` 4) plus an
unmeasured share of `toplevel` (dtype introspection: `can_cast`, `promote_types`,
`result_type`, `min_scalar_type`, `isdtype` are all present-but-FAILING today, and
several fail specifically on `object`/`U`/`V` inputs). **37 is a floor, not the total.**

Related and measured today: **`anionpy.dtype` and `anionpy.ndarray` cannot be instantiated at
all** — every construction form raises `TypeError: cannot create instances`. Both names
resolve, so `hasattr` is happy, which is how it went unnoticed.

### HONEST STATEMENT ON REACHING 100%

100% is **not** reachable without two Rust efforts that no amount of Python-surface work
substitutes for:
1. **DType enum expansion** — 8 missing dtype categories, floor of 37 items.
2. **Buffer representation / interior mutability** — blocks `out=`, `ndarray.__setitem__`,
   the 5 mutating `ma.*`, and the in-place-operator partial-view aliasing defect
   (`a[1:4]` desyncs from `a` after `a.__iadd__(b)`; root cause is
   `slf.inner = out.cast_to(...)` full reassignment instead of a strided write through
   the shared buffer).

Everything else is volume, not risk. **There is no longer a cheap lane** — Wave 1 proved
the "already built, just connect" bucket was 22% real.

### Route to 100%, in dependency order

- **Wave 1 — connect what exists (~~131 items, 470 → ~600~~).** No new arithmetic.
  Every item needs an out-of-corpus probe before declaring; expect attrition, so
  budget ~600 not 601. **DONE 2026-08-03 — actual +29, not +131. See the falsified
  block above. "Expect attrition" was the right instinct at the wrong order of
  magnitude: attrition was 78%, not the ~1 item budgeted for.** The cheap lane was
  not cheap because it was not there. Everything from here is real building.
- **Wave 2 — `ma` Phases 0–2 (~145 of the 225).** A MaskedArray is a PAIR of
  ordinary ndarrays, so this is an import surface over existing Rust ufuncs, not
  new numerics. Phases 3–5 (~75) follow. *In flight 2026-08-03.*
- **Wave 3 — toplevel 168 + ndarray methods 26 + real dunders 21.** The bulk
  build. Sub-families are independently parallelisable.
- **Wave 4 — `random` 58.** Isolated but exacting: bit-reproducible PCG64.
- **Wave 5 — DType enum expansion (~22 items).** The gate. Unblocks `char` 24,
  `strings` 4, `rec` 9, structured/void, datetime/busday, and the
  `typecodes`/`sctypeDict`/`ScalarType` items that cannot be faithful without it.
- **Wave 6 — buffer representation / interior mutability** (~171 core sites).
  One gate, three symptoms: `out=`, the 5 mutating `ma.*` items,
  `ndarray.__setitem__`. Largest single Rust change on the board.
- **Wave 7 — the 16 present-but-failing defects, and the tail of 33.**

**Waves 1–4 are unblocked today and total ~518 items.** Waves 5 and 6 are the
only genuine architectural gates, and between them they hold ~80 items hostage.
Nothing on this list is blocked on an unanswered question except the ~5 mutating
`ma` items and the open *"is output memory layout part of the contract?"*
escalation, which threatens 4 currently-declared items (470 → 466 if layout is
contract) and gates ~13 undeclared ones.

## CORRECTION 2026-08-02: "absent" does not mean "unbuilt"

This plan was written assuming every absent item needs implementing. That is
false for roughly a fifth of them, and the ordering below is wrong to the extent
it assumes otherwise. Measured at 447/1180, of **725 absent** items:

| | count | meaning |
|---|---:|---|
| present on `anionpy`, differential verdict **pass**, undeclared | **81** | built, tested, green — never wired into the ledger |
| present on `anionpy`, differential verdict **fail** | 33 | real defects, correctly absent |
| present on `anionpy`, **no test at all** | 39 | need a harness before they can be judged |
| genuinely missing | 572 | the actual build queue |

So 153 of the 725 already exist in some form. This is the same failure the
`_state/` split was created to fix — see
`reports/ionp-throughput-bottleneck-2026-08-01.md`, where a 47-item `testing`
block moved the ledger 91 → 138 the day it was *connected*, having been built
weeks earlier. That was not an isolated incident; it was the first observation
of a standing condition.

**The cheapest coverage in the tree is auditing what is already green.** But
cheap is not free: a passing differential verdict only means "agrees with numpy
on the axes the corpus happens to sample" (lesson #34), so each of the 81 gets
an out-of-corpus probe before it is declared, and any that fails that probe
stays absent and becomes a defect report. Declaring all 81 unexamined would be
the exact sin this ledger exists to catch.

Standing consequence for planning: **before building anything, check whether it
is already built.** Re-run the present/passing/undeclared split above rather
than trusting an `absent` count as a build estimate.

## Where the 1022 actually are

| block | absent | total | done |
|---|---:|---:|---:|
| toplevel | 423 | 477 | 11.3% |
| `ma` | 225 | 225 | 0.0% |
| `ndarray` | 125 | 164 | 23.8% |
| `char` | 65 | 65 | 0.0% |
| `random` | 63 | 63 | 0.0% |
| `strings` | 46 | 46 | 0.0% |
| `fft` | 19 | 19 | 0.0% |
| `linalg` | 15 | 33 | 54.5% |
| `emath` | 9 | 9 | 0.0% |
| `rec` | 9 | 9 | 0.0% |
| `polynomial` | 8 | 8 | 0.0% |
| `ctypeslib` | 6 | 6 | 0.0% |
| `lib` | 6 | 6 | 0.0% |
| `testing` | 3 | 50 | 94.0% |

Two blocks are 55% of the remaining work: `toplevel` (423) and `ma` (225).

## Ordering, and the reasoning behind it

Ordered by **items per unit of new machinery**, not by item count. Anything that
forces a new dtype kind or a new storage model is expensive no matter how many
items sit behind it.

1. **toplevel creation + shape manipulation (~60)** — IN FLIGHT.
   `zeros`/`ones`/`empty`/`full` + `_like` variants share one implementation;
   reshape/ravel/transpose/broadcast are view mechanics on machinery that already
   exists. Best ratio on the board.
2. **`ndarray` attributes + methods (125)** — IN FLIGHT. Much of it is metadata
   already present in `NdArray` (ndim, size, itemsize, nbytes, strides, flags, T,
   real, imag, base). Also where the two retracted items live.
3. **toplevel reductions/statistics (~80)** — sum/prod/mean/std/var/min/max/
   argmin/argmax/cumsum/percentile/median. One axis-reduction kernel with an
   `axis`/`keepdims`/`where`/`initial` protocol lands most of it.
4. **toplevel sorting/searching/set ops (~50)** — sort/argsort/searchsorted/
   unique/isin/nonzero/where. Needs one comparison-sort kernel over strided views.
5. **`emath` (9), `lib` (6), `ctypeslib` (6), `rec` (9)** — small, mostly thin
   wrappers over things that will exist by then. Cheap tail.
6. **`fft` (19)** — self-contained. Needs a real FFT in Rust (Bluestein for
   non-power-of-2). Bounded, no dependency on the rest.
7. **`linalg` remainder (15)** — solve/inv/det/eig/svd/qr/cholesky. Accelerate
   BLAS/LAPACK is already linked; the work is protocol and error matching.
8. **`random` (63)** — requires bit-exact reproduction of numpy's PCG64/Philox
   streams and its exact ziggurat/Box-Muller paths. All-or-nothing: a generator
   that is statistically fine but stream-divergent is worth **zero** here,
   because the differential test compares values, not distributions. Do not start
   this until the cheap blocks are done.
9. **`char` (65) + `strings` (46)** — needs a genuine string dtype
   (`bytes_`/`str_`, itemsize-parameterised) in the core. One large piece of
   machinery, 111 items behind it. Highest fixed cost, second-best total payoff.
10. **`ma` (225) — LAST, deliberately.** Masked arrays are a parallel array model
    (data + mask + fill_value) that wraps nearly the entire rest of the API.
    Every `ma` item is cheap *if* the thing it mirrors already exists, and
    expensive otherwise. Building it first would mean building it twice.

## Rules that do not bend as throughput rises

The pressure from here is volume, and volume is exactly when the discipline gets
quietly dropped. It does not get dropped.

- **Rust does all arithmetic and data movement.** Python is the import/dispatch
  surface. Any numerical loop in a `.py` file is a bug. (Mother's standing rule;
  the Python exception for this project covers the *import surface only*.)
- **An item counts only with a passing differential test.** No test, no ledger
  entry — that is the `untested` state and it stays at 0.
- **A passing test is necessary, not sufficient.** Proven three times here:
  `matmul`; `ndarray.__repr__`/`__getitem__`; and the 2026-08-01 attribute block,
  where 10 of 14 reported-passing items were withheld on direct measurement.
  Every declaration needs out-of-corpus evidence too.
- **Test the SIGNATURE, not just the values.** This is the newest failure mode
  and the corpus is structurally blind to it: it exercises the default call form,
  so an item can be bit-exact on `a.sum()` while `a.sum(axis=0)` raises
  `TypeError: takes no keyword arguments`. Six reduction methods implement 1 of
  numpy's 5 documented parameters and read as green. **A method that is exact on
  the default form and rejects the documented ones is not a passing item, it is a
  passing test.** Every spec must cross the argument space, not just the dtypes.
- **Exception TYPE is part of the API.** `IndexError` where numpy raises
  `numpy.exceptions.AxisError` is not cosmetic — AxisError subclasses *both*
  ValueError and IndexError, so caller code written `except ValueError` catches
  numpy's and silently misses anionpy's.
- **`exception_equivalences` is not a repair.** It is sanctioned only for
  genuinely equivalent exceptions. Using it to retire a real type divergence is
  buying a green test with a tolerance — the exact thing
  `ItemSpec.__post_init__` exists to prevent.
- **A tolerance is a measurement, never a margin.** Declared bound must EQUAL the
  measured sweep max, ≥20,000 samples/dtype. `ItemSpec.__post_init__` enforces
  this in both directions and has been falsified against a positive control.
- **Skips are not passes.** Every probe prints `skipped` beside `compared`;
  nonzero skips invalidate the run until explained.
- **A fix that makes the failing case pass is not thereby correct.** An agent
  changed the float16 reduction accumulator from "round every step" to
  "accumulate in f32, round once," cited a real 1-D `prod` counter-example as
  evidence, turned 13 mismatches into 6, and reported the gap closed. Both
  models are wrong: numpy uses the wide register accumulator when the reduced
  axis is the innermost loop (every scalar-output reduce, contiguous `axis=-1`)
  and rounds into the float16 output buffer on every step when the reduced axis
  is an outer loop — measured at 3508 wide/0 narrow for 2-D `axis=1` against
  0 wide/920 narrow for `axis=0`, same array, same dtype. The old code and the
  fix were each right about half the matrix. **Mismatches going down is not
  evidence the model is right; it is evidence the overlap got bigger.** Before
  accepting a numerical fix, classify real numpy against BOTH candidate models
  over the discriminating cases and confirm one of them wins everywhere.
- **The ledger going DOWN because a false entry was removed is a win.** 143 → 141
  happened today and was the most valuable move of the morning.

## Known blockers carried forward

- `expm1` complex: subnormal dropout fixed (4c33d73), but complex `expm1` is
  still not bit-exact vs numpy (~36% of a 200k sweep). Failing, undeclared.
- `ndarray.__repr__` / `__getitem__`: `repr.rs`'s `flat_index`/`format_at` read
  buffer index 0 instead of the view's offset; `__getitem__` returns 0-d ndarray
  where numpy returns a scalar. **The corpus is structurally blind to both** — it
  compares via `np.asarray()`. Fixing them requires adding offset-view and
  scalar-return-type cases first, or the fix is unverifiable.
- numpy's `same_kind` output-casting refusal (`UFuncTypeError`) is unenforced.
  One fix in `lib.rs` lands the matmul family (4) plus the `__i*__` block (7).
- `divide`/`true_divide` can never carry an honest bound — reduce chains hit
  31,228 ULP. That is a defect to fix, not a tolerance to write.
- 3 stale `cargo test -p ionp-core` failures, pre-existing and documented.

## Throughput, which is the actual constraint

The measured bottleneck was never implementation — it was **serialisation on one
file**. Five separate blocks of finished, tested, committed work sat uncounted
because wiring them into a single 565-line dict was somebody else's file to
touch; the 47-item `testing` block moved the ledger 91 → 138 the moment it was
merely *connected*. Nothing was built that day.

Fixed structurally on 2026-08-01 by splitting `__ion_state__` into
`anionpy/_state/{toplevel,ndarray,linalg,testing}.py` with a loud import-time
collision check. Parallel agents now need per-block file ownership, and — the
lesson that cost real time today — **partitioning must be by RUNTIME DEPENDENCY,
not just edit ownership**: two agents with disjoint files still collide if both
rebuild the same `.so` or measure against it. Concurrent builds go through
`flock /tmp/ionp-build.lock`.

**The corpus cannot see a parameter it never passes.** The differential corpus
grades VALUES on the DEFAULT CALL FORM. An item that silently ignores half its
numpy signature still comes back `pass`. This is not hypothetical and it is not
rare: a signature audit of the 230 passing items on 2026-08-01 found 23 suspects
and confirmed 16 of them by probe -- `array(ndmin=/copy=/order=/subok=)`,
`asarray(order=/copy=)`, `reshape(copy=)`, `eye(order=)`, `linspace(axis=)`,
`zeros(like=)`, `zeros_like(subok=)`, `broadcast_to(subok=)`, `copy(subok=)`,
`linalg.svd(hermitian=)`, `linalg.trace(dtype=)` -- every one of them raising
`TypeError: unexpected keyword argument` while declared green. `tools/signature_audit.py`
now runs this check from the other side; run it before trusting a green ledger.
A green corpus means "the default call form produces the right values," which is
a strictly weaker claim than "this item is implemented."

**"Unreachable" must be proved with ionp-NATIVE constructions.** On 2026-08-01
an agent implemented `reshape(copy=)`, hit a divergence, declared it a
"structurally unreachable, deliberate architectural boundary," and added a
corpus guard that skipped the cases. It was reachable, and the divergence ran
the other way. The trap: anionpy NORMALIZES numpy arrays on ingestion, so a
non-contiguous view built in numpy and handed to `anionpy.array()` arrives
contiguous -- probing that way makes any strided-input bug look impossible. Build
the awkward input with anionpy's OWN slicing/transpose and it appears immediately:
`anionpy.array(a)[:, ::2]` keeps strides `(48,16)`, numpy reshapes that to 1-D as a
real stride-16 VIEW under `copy=False`, and anionpy raised `ValueError` because
`NdArray::reshape` only returns a view when `is_c_contiguous()` -- numpy runs the
full `_attempt_nocopy_reshape` stride-compatibility search. Rule: a reachability
claim is evidence only if the probe constructs its inputs through anionpy
operations. And a guard that makes cases stop running is never the fix -- it
converts an unknown into a green tick.

**The foreign-numpy-input path is a third blind spot.** anionpy physically copies any
external `numpy.ndarray` into a fresh ionp-owned buffer on ingestion, preserving
only genuine F-contiguity. Two consequences, both confirmed by probe on
2026-08-01: (a) `anionpy.array(x, copy=False)` raises where numpy succeeds, because
the ingestion copy is unavoidable, and any numpy array that is neither C- nor
F-contiguous silently arrives normalized -- so a layout-dependent bug is
invisible to any probe whose inputs come from numpy; (b) `subok=False` on
`zeros_like`/`broadcast_to`/`copy` raised `TypeError: 'ndarray' object is not an
instance of 'ndarray'` for a numpy argument while working perfectly for an anionpy
one -- an isinstance check against anionpy's own type. All four passed the corpus.
Rule: every parameter probe must be run twice, once with an anionpy array and once
with a real numpy array, and layout-sensitive probes must build their awkward
inputs with anionpy's own ops.

**Exception messages are a fourth blind spot.** `harness.py` (the exception arm of
`compare_call`) returns `True, ""` the instant the exception *type* matches the one
numpy raised. The message text is never examined. Confirmed 2026-08-01 by a post-hoc
probe of the 17 ufuncs merged in `9c762af`: 720 comparisons, **90 message-only
divergences, zero of which the corpus could see.** Two failure modes, both live:

- **Internal identifiers leaking into user-facing text.** `anionpy.left_shift` on a
  uint64 array raised ``ufunc 'LeftShift' not supported...`` -- the Rust enum
  variant, Debug-formatted -- where numpy says ``ufunc 'left_shift'``. Caused by
  `{op:?}` at six sites in `ufunc.rs`; `op.numpy_name()` already existed and was
  used correctly elsewhere in the same file.
- **Self-describing parentheticals that are false.** anionpy appended
  `(not yet implemented)` for complex `remainder`/`arctan2`/`copysign`. numpy has no
  complex loop for those either, so anionpy is at *parity* -- the message advertises a
  deficiency that does not exist. An error string that lies about the implementation
  is worse than one that merely differs.

Note the asymmetry this creates: a wrong message cannot fail a test, so it can only
be found by probing deliberately. Treat message text as part of the surface.

> **CLOSED 2026-08-01 (`f69aceb`).** The arm now compares messages byte-for-byte.
> The blast radius was measured *before* flipping it on, which is the only reason
> it was safe to flip in one commit: **172,952 corpus cases; 46,757 with a
> type-matched exception on both sides; 6,116 of those (13.1%) diverging in text;
> 83 of the then-280 "exact" items affected — 29.6% of the whole ledger.**
> Enabling it moved the ledger to 198 exact / 82 failing.
>
> That drop is not a regression, and the distinction matters: those items were
> already wrong, and the ledger's `failing 0` was purchased by not looking. A
> coverage number that only counts what it is willing to check is not a coverage
> number. Definition of Done requires `failing 0`, so the 82 are now visible work
> instead of invisible debt.
>
> Two implementation choices worth keeping:
> - Nondeterministic *spans* (memory addresses, OS temp paths) are **scrubbed**,
>   not whole items exempted. `testing.rundocs` and `testing.assert_no_gc_cycles`
>   embed a random tmp path and live object addresses respectively; exempting
>   either item outright would have discarded a real check to dodge one token.
> - Whitespace is deliberately **not** normalized. numpy's trailing spaces are
>   real and internally inconsistent (plural `"...with shapes (3,) (4,) "` has one,
>   singular `"...with shape (24,) (4,)"` does not), so normalizing would have
>   silently forgiven the exact trailing-space defect fixed in `1a37642`.
>
> Failure output prints `repr()` and character counts, because the first defect of
> this class was invisible in bare output — the two strings printed identically and
> differed only at 62 vs 63 characters.

**Alias identity is a fifth blind spot.** In numpy, `np.acos is np.arccos` is
`True` -- most alias pairs are the *same object*, and real code depends on it
(`if ufunc is np.add`, `__array_ufunc__` implementations comparing ufunc identity).
On 2026-08-01 all 13 aliases registered in `9c762af` were separate function objects:
`atan2 mod bitwise_left_shift bitwise_right_shift acos asin atan acosh asinh atanh
bitwise_invert pow true_divide`. Every one passed its differential test.

The trap, verified against numpy 2.5.1: **`deg2rad`/`radians` and `rad2deg`/`degrees`
are NOT aliases.** They are distinct ufunc objects, each reporting its own
`__name__`. Every other pair above is a true identity alias whose `__name__` is the
*canonical* name (`np.acos.__name__ == 'arccos'`). So "make the aliases aliases" is
right for 13 of 15 and wrong for 2 -- check identity per pair against numpy rather
than applying the rule uniformly.

---

## Exception identity: policy + runtime-numpy inventory (2026-08-01)

Two incompatible answers to "what class should anionpy raise?" landed in the tree
on the same day, so this fixes the policy before merge order does it for us.

**The question.** numpy raises private, version-unstable `TypeError` subclasses
(`numpy._core._exceptions._UFuncNoLoopError`, `_UFuncBinaryResolutionError`,
`_UFuncOutputCastingError`) plus public ones (`numpy.exceptions.AxisError`,
`numpy.linalg.LinAlgError`). The differential harness matches exceptions by
exact `type()` identity, so anionpy must either construct numpy's class or declare
an equivalence.

**Answer A (older, `lib.rs`/`linalg.rs`)** — import numpy at raise time and
construct its real class. Documented as a "differential-parity shim," and in
`lib.rs` it degrades: if numpy is not importable it falls back to a plain
`TypeError` with the same message. `linalg.rs` does NOT fall back.

**Answer B (newer, `strings.rs`)** — construct the exception entirely in Rust as
the nearest PUBLIC base (`TypeError`) with byte-identical message text, and
declare acceptance via `ItemSpec.exception_equivalences`.

**POLICY: B is preferred; A is legacy to be migrated opportunistically.**
Reasons, in order of weight:
1. A makes anionpy's error identity depend on `numpy._core._exceptions`, a private
   path numpy does not promise to keep stable. A numpy point release can break
   anionpy's error paths without anionpy changing at all.
2. B's equivalence mechanism is NARROW and does not weaken the bar: it is keyed
   on numpy's exact exception type, permits only the named stand-in, and the
   MESSAGE IS STILL GRADED BYTE-FOR-BYTE afterward (harness.py:950-956). It
   forgives class identity, nothing else. This was demonstrated the same day:
   the equivalence did not hide 8 genuinely-broken char/strings items — the
   message check caught them anyway.
3. The private classes are not public API. User code catches `TypeError`.

Caveat, stated honestly: B is a real, if narrow, divergence. `type(exc)` differs
from numpy's. Anything relying on `except numpy._core._exceptions...` would not
be caught. That is judged acceptable for private classes and NOT acceptable for
public ones — `AxisError` and `LinAlgError` are public API and must keep being
the real class.

### Runtime-numpy inventory (10 call sites, measured not assumed)

Measured claim, verified by blocking `numpy` in `sys.meta_path` and importing:
**anionpy's core imports and computes with numpy entirely absent** —
`anionpy.array([1,2,3]) + itself` returns `array([2, 4, 6])`. The core is
standalone. numpy is reached only on the paths below.

| site | kind | degrades if numpy absent? |
|---|---|---|
| `lib.rs` `no_ufunc_loop_err` (`numpy._core._exceptions`) | exception identity | yes, falls back to `TypeError` |
| `lib.rs` `ufunc_output_casting_err` (`numpy._core._exceptions`) | exception identity | yes, falls back |
| `lib.rs` AxisError x2 (`numpy.exceptions`) | exception identity, PUBLIC class | yes, falls back |
| `linalg.rs` LinAlgError x2 (`numpy.linalg`) | exception identity, PUBLIC class | NO — hard fail |
| `strings.rs` `load` (`numpy.asarray`) | input marshalling | NO — hard fail |
| `strings.rs` `encode_string_array` (`numpy.frombuffer`) | output buffer wrapping | NO — hard fail |

The two `strings.rs` sites are marshalling only and carry NO computation: the
`S`/`U` dtypes they handle are numpy's, and every value and every exception on
those paths is produced by anionpy. This was enforced, not assumed — an earlier
revision routed foreign-dtype input to real `numpy.strings.<name>`, which meant
`anionpy.strings.add` on numeric input returned `numpy.add`'s arithmetic and its
differential test compared numpy against numpy. Rejected and rewritten to
dispatch through anionpy's own `binary_op`.

**The standing line: numpy may hand us input bytes, never answers.**

The stale comment at `lib.rs:100` ("anionpy has no hard runtime dependency on
numpy outside this one differential-parity shim") is now inaccurate — there are
six such sites and three of them do not degrade. Left in place pending the
`NoUfuncLoop`-dtype work, which will touch that function anyway.

---

## Blind spot #1 (parameter-blindness) MEASURED — 2026-08-01

Measured the same way #4 was: read-only agent, worktree-isolated, probe every
(declared item, unexercised parameter) pair against live numpy.

**292 declared items. 785 (item, parameter) pairs. 357 never exercised by the
corpus. Of those: 236 RAISES, 99 OK, 22 NA.**

**Headline: 85 of 292 declared items (29.1%) are demonstrably not identical to
numpy under their full documented call surface.**

The one piece of good news, and it is real: **zero silent wrongness.** Every
one of the 85 fails loudly by raising. Nothing in the ledger was found quietly
returning a wrong answer for an unexercised keyword. The measurement also
survived its own scrutiny — 3 raw WRONG/MSG hits were traced to probe-harness
artifacts (an adapter collision on `linalg.svd`, mis-threaded adapter on
`ndarray.sort`, and a 1.67e-16 float64-noise false positive on `linalg.pinv`)
and hand-verified by direct bypass before being reclassified. 21 more pairs
were reclassified after finding the corpus supplies them POSITIONALLY with
varied values (`diag`'s `k`, `where`'s `x`/`y`, ...), which a kwarg-only scan
undercounts.

### The clusters

| cluster | items | fix cost |
|---|---|---|
| ufunc `casting`/`order`/`subok` rejected by the shared dispatcher | 65 | see below — NOT cheap |
| `concatenate`/`stack`/`hstack`/`vstack`/`trace` `out=`/`casting=` | ~5 | `out=` is a real buffer-writing feature |
| `linalg` `UPLO`/`hermitian`/`offset` algorithm paths | 4 | deep, self-documented `NotImplementedError` |

### REJECTED recommendation, recorded so it is not retried

The measurement recommended treating `casting`/`order`/`subok` as "meaningfully
NO-OPs" and fixing all 65 items at once by having the shared dispatcher "accept
and validate-or-ignore." **That is wrong and would make the tree worse.**
Verified directly against numpy 2.5.1:

```
order='F'  ->  strides (8, 16) and F_CONTIGUOUS=True
order='C'  ->  strides (24, 8)                          # observable; ndarray.strides is itself a ledger item
casting='no'     -> UFuncTypeError: Cannot cast ufunc 'add' input 1 from int64 to float64
casting='equiv'  -> UFuncTypeError  (same)
casting='safe'/'same_kind'/'unsafe' -> OK
```

So accepting-and-ignoring would turn 65 items that currently fail LOUDLY into
65 items that silently return answers numpy does not return — trading the one
good property this measurement found (no silent wrongness) for a bigger green
number. That is precisely the blind-spot-#1 rationalization this audit exists
to catch, and it is rejected.

The honest fix, in order of tractability:
1. `casting=` — implement real validation. numpy's rule is checkable and its
   message text is reproducible; this is genuine, matchable work.
2. `subok=` — probably a true no-op since anionpy has no ndarray subclasses, but
   CONFIRM against numpy before declaring it so.
3. `order=` — NOT satisfiable today. anionpy has one canonical memory layout, so
   `order='F'` cannot be honored without real strided-array support. Items
   whose only gap is `order=` stay UNDECLARED until that exists. An undeclared
   item is honest; a declared one that ignores `order=` is not.

### Ledger consequence

By the same standard already applied to `unique`/`trim_zeros`/`intersect1d`
(undeclared for raising on `axis=`/`sorted=`/`return_indices=`), these 85 items
assert a call surface they do not implement and must be undeclared. Deferred
only until the in-flight agents land, because several of them are editing the
same `_state/` declaration blocks and a simultaneous 85-item rewrite would risk
corrupting declarations during merge. This is a sequencing decision, not an
acceptance — the number to expect is roughly 273 exact down to ~190.

### Undeclaration pass: list finalized (2026-08-01)

Recomputed the blind-spot-#1 undeclare list directly from the probe data rather
than the prose summary, and corrected one of my own errors in the process.

My first pass counted 94 items by treating every non-`OK` probe class as
evidence of a defect. That was wrong: class `NA` means *the probe could not
construct a value for that parameter*, which is a limitation of the probe, not
a divergence in anionpy. All 22 `NA` rows are `testing.*` items with parameters the
probe has no construction rule for (`__cache`, `_load_time`, `testmatch`,
`forwarding_rule`, ...). Excluding them reproduces the report's original 85
exactly.

Final composition of the 85 (all currently declared `exact`):

    74  casting + order + subok   (the ufunc cluster)
     2  casting + out
     2  casting
     2  hermitian                 (matrix_rank, pinv)
     1  ndmax
     1  UPLO                      (eigvalsh)
     1  offset                    (linalg.trace)
     1  copy + dtype
     1  out

Every one of the 236 real divergences is class `RAISES` — anionpy raises where
numpy succeeds. There are ZERO `WRONG` and ZERO `MSG` rows. That is the single
best property this audit found and it is exactly what the rejected
"accept-and-ignore the parameter" recommendation would have destroyed: it would
have converted 85 loud failures into 85 silent wrong answers.

List persisted at `/tmp/undeclare_list.json`; the durable copy is
`reports/ionp-param-blindness-2026-08-01.md`.

### char/strings S-vs-U repair verified independently (2026-08-01)

Third attempt from this branch; the first two reports were false (both against a
stale `.so`). This one holds up under my own verification in the MERGED tree:
307 -> 315 exact, declared-failing 13 -> 5, all 28 char/strings items pass,
cargo unchanged at 141 + the 3 documented gaps.

Trust came from an out-of-corpus probe, not the branch's own numbers. I fed
shape/dtype pairs that appear nowhere in the corpus -- `S7`-vs-`U3`,
`S1`-vs-`int64`, `float64`-vs-`U2` -- across `strings.add`, `char.add`,
`strings.equal/less/greater_equal`. Every message matched numpy byte-for-byte,
including the two DIFFERENT shapes numpy uses for the same failure family:

    add:         ufunc 'add' cannot use operands with types dtype('S7') and dtype('<U3')
    add (mixed): ufunc 'add' did not contain a loop with signature matching types (dtype('S1'), dtype('int64')) -> None
    comparisons: ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StrDType'>) -> None

Note `add` renders operands as `dtype(...)` while the comparisons render them as
`<class 'numpy.dtypes....DType'>`. Matching both on unseen inputs is evidence the
rule was derived, not that corpus values were hardcoded.

One probe artifact of my own, recorded so it is not re-derived: my first probe
built inputs with `anionpy.array([b'abc'])`, which raises during CONSTRUCTION
before the function under test is ever called. It looked like 12 failures and
was entirely my error. String arrays must enter through the foreign path
(`np.array(...)` handed to `anionpy.strings.*`).

Exception-type policy check: anionpy raises `TypeError` where numpy raises
`UFuncTypeError`, forgiven by declared equivalence. Confirmed correct rather
than assumed -- `UFuncTypeError` is NOT public (absent from `numpy.exceptions`,
defined in the private `numpy._core._exceptions`), so the Rust-constructed
exception + narrow equivalence is the sanctioned "Answer B" pattern. It is a
real `TypeError` subclass, so `except TypeError` catches both and user-visible
behavior is unchanged. Had it been public, the real class would have been required.

## Blind spot #6: TAUTOLOGICAL RECEIVER (found 2026-08-01, NOT yet fixed)

The sixth corpus blind spot, and the first one that inflates the ledger rather
than merely limiting it. Found while auditing an agent's "this failure is the
harness's fault, not anionpy's" claim -- the alibi was true, and pulling on it
exposed something larger than the failure it excused.

`corpus.py:204` defines the binary-op pair

    Pair("broadcast/scalar_array", np.float64(2.5), _fill(rng, (3,4), np.float64))

whose FIRST operand is a numpy scalar. `make_ionp_array_converter` converts only
`numpy.ndarray` and passes everything else through by design. So for every
`ndarray.`-prefixed item, `resolve_ionp()` evaluates
`getattr(np.float64(2.5), "__add__")` -- on the identical numpy object the numpy
side uses. Verified directly: the converter returns the SAME OBJECT (`out is s`
-> True). No anionpy code executes for that case at all.

This is precisely hazard (b) that `make_ionp_array_converter`'s own docstring
warns about -- "silently resolve the attribute on the numpy object itself, a
tautology that can never fail" -- firing in the one place the guard does not
reach, because the guard protects against un-converted ARRAYS and this is a
SCALAR.

Measured blast radius across the 37 `ndarray.*` binary_op items:

    32  TAUTOLOGICALLY GREEN -- np.float64 has the dunder, both sides run the
        same numpy code, case passes and can never fail.
        __add__ __radd__ __mul__ __rmul__ __eq__ __ne__ __lt__ __le__ __gt__
        __ge__ __sub__ __rsub__ __truediv__ __rtruediv__ __floordiv__
        __rfloordiv__ __mod__ __rmod__ __pow__ __rpow__ __divmod__ __rdivmod__
        __lshift__ __rlshift__ __rshift__ __rrshift__ __and__ __rand__ __or__
        __ror__ __xor__ __rxor__

     5  LOUDLY RED -- np.float64 lacks the in-place dunder, so the harness's
        `raise AttributeError(attr)` emits the bare name while numpy emits
        "'numpy.float64' object has no attribute '__imod__'".
        __ifloordiv__ __imod__ __ipow__ __ilshift__ __irshift__

The 5 red ones are NOT more broken than the 32 green ones. They are the same
non-test; they merely fail noisily instead of passing silently. That asymmetry
is the only reason this was ever visible.

**The forbidden fix.** Making the AttributeError message match CPython's would
turn all 5 green and close the "failing" count to 0. That is the make-it-green
move this project forbids: it would convert the last loud evidence of a
32-item hole into silence. Explicitly rejected.

**Open design question** (deliberately not decided under time pressure): an
r-dunder's natural corpus shape IS (scalar, array), but the receiver of
`__radd__` should be the ARRAY with the scalar as argument. Whether the current
pair ordering is even meaningful for the 16 r-dunder/dunder pairs needs settling
before the fix, or the repair will encode the same confusion. Candidate fixes:
signal "not a test of anionpy" and record it as INVALID (never as pass), or fix the
operand ordering so the receiver is always the anionpy object.

Sequenced after the in-flight setops agent lands, since a `registry.py` edit
would collide with it. Expect the honest fix to REMOVE credit, not add it.

## Ledger correction (2026-08-01): 85 items undeclared for parameter blindness

A dedicated audit (`reports/ionp-param-blindness-2026-08-01.md`,
`reports/undeclare-list-2026-08-01.json`) probed every declared item's FULL
numpy call surface -- not just the parameters the differential corpus happens
to exercise -- against live numpy 2.5.1. Result: 85 of the 292 then-declared
`exact` items raise a `TypeError`/`ValueError`/`NotImplementedError` on at
least one keyword numpy accepts and succeeds on. All 85 are the same failure
class, `RAISES` (anionpy refuses where numpy succeeds) -- zero were found
silently returning a wrong value, and zero raised a different error message
for the same input. That "loud, not silent" property is the one piece of
good news; it does not change that the ledger's own definition of `exact`
("passes for its full documented numpy call surface, not just the corpus'
sample of it") was violated 85 times over.

**Independent re-verification, not a blind trust of the audit list.** Before
touching any declaration, every one of the 85 items was re-probed directly
against numpy 2.5.1 and the current anionpy build (script:
`/tmp/.../scratchpad/probe85.py`, not committed -- scratch). This matters
because several agents landed ufunc/linalg work in the days between the audit
and this pass, so some items could plausibly have been fixed already.
**Result: 85/85 still genuinely raise. 0 were found already fixed
("stale audit, still honest").** Every undeclaration below is backed by a
fresh, this-session TypeError/NotImplementedError string, not a citation of
the audit report.

**Composition of the 85** (by which parameter(s) fail; a few items appear
in more than one bucket count in the source report but each item is
undeclared exactly once here):
- 74 items -- the generic-ufunc-dispatcher cluster: `casting=`, `order=`,
  `subok=` all rejected with `TypeError: ufunc.__call__() got an unexpected
  keyword argument '...'` (or the gufunc-specific equivalent for
  `matmul`/`matvec`/`vecdot`/`vecmat`, which have their own binding).
- `concatenate`, `stack` -- `out=` and `casting='unsafe'`.
- `hstack`, `vstack` -- `casting=`.
- `trace` (top-level) -- `out=`.
- `linalg.trace` -- `offset=` (non-zero).
- `linalg.eigvalsh` -- `UPLO='U'`.
- `linalg.matrix_rank`, `linalg.pinv` -- `hermitian=True`.
- `array` -- `ndmax=` (numpy 2.5.1 added this keyword alongside the
  long-standing `ndmin=`; anionpy's `array()` has never accepted it).
- `ndarray.__array__` -- takes no keyword arguments at all, so both
  `dtype=` and `copy=` (the array-conversion-protocol's own keywords) raise.

Each undeclared entry in `anionpy/_state/toplevel.py`, `anionpy/_state/ndarray.py`,
and `anionpy/_state/linalg.py` carries its own comment naming the specific
missing parameter(s) and the exact exception anionpy raises, per the established
`unique`/`trim_zeros`/`intersect1d` precedent (those three were undeclared for
this identical reason -- raising on `axis=`/`sorted=`/`return_indices=` --
and only re-declared once those parameters were genuinely implemented). Note:
`linalg.py` is not among this task's originally-listed four `_state` files,
but 4 of the 85 items live there and the task cannot be done honestly without
touching it; it is the same block-split-era file as the other three (commit
`3f75f5f`) and showed no sign of concurrent edits, so it was included.

**Fix-class notes, so a future re-declare pass knows what "fixed" actually
requires:**
- `casting=` is genuinely matchable work: validate-or-widen at the shared
  ufunc-dispatch entry point (`casting='unsafe'` is a pure permission widen,
  not a behavior narrowing, since `'unsafe'` allows everything).
- `order=` is **not satisfiable** without real strided-array/memory-layout
  support -- anionpy has one canonical memory layout internally, so honoring a
  caller's `order='F'`/`'A'` request is not a validation shim, it is a
  from-scratch feature. These items must stay undeclared for `order=`
  specifically until that exists, even after `casting=`/`subok=` are fixed.
- `subok=` is probably a true no-op (anionpy has no `ndarray` subclassing to
  honor or reject) but this is **not yet confirmed** -- verify numpy's own
  `subok=False` semantics (does it force base-class output even when the
  input IS a subclass instance?) before wiring an accept-and-ignore path, so
  the "fix" does not itself become a new silent divergence.
- `out=` (`concatenate`/`stack`/`trace`) is moderate, real work: writing into
  a caller-supplied buffer touches the Rust binding layer, not just Python
  validation.
- `linalg.eigvalsh`/`matrix_rank`/`pinv`/`trace`'s gaps (`UPLO=`,
  `hermitian=`, `offset=`) are deep, already self-documented in anionpy's own
  `NotImplementedError` text as needing a genuinely different algorithm path
  (Hermitian-specialized LAPACK routing, non-zero-diagonal-offset trace) --
  not a mechanical widen.

**Ledger, measured before and after** (`tests/differential/run.py --out
r.json` then `tools/coverage.py --tests r.json`):

| | exact | absent | phantom | untested | failing |
|---|---|---|---|---|---|
| before | 320 | 857 | 0 | 0 | 3 |
| after  | 235 | 942 | 0 | 0 | 3 |

`exact` dropped by exactly 85 and `absent` rose by exactly 85, as expected --
this is a pure re-classification, no item's actual behavior changed.
`phantom` and `untested` are both 0 in both runs (undeclaring did not push
any item into either). `failing=3` is the pre-existing, unrelated,
already-documented gap (unchanged). Coverage: 27.119% -> 19.915%. Two
consecutive `run.py --out` runs after the change produced byte-identical
JSON (`shasum` match). `flock /tmp/ionp-build.lock cargo test --release`:
141 passed, 3 failed (same pre-existing 3 as baseline, 0 regressions -- this
pass touched only `anionpy/_state/*.py`, no Rust).

This is the ledger going down on purpose because a false entry was removed,
per this project's own stated rule (`anionpy/_state/__init__.py`'s module
docstring: "Never declare an item to protect a number -- the ledger going
down because a false entry was removed is a win"). The 85 items are not
gone; they are real, callable, mostly-correct Rust-backed functions that
handle their default call form -- they are simply no longer claiming a call
surface (`casting=`/`order=`/`subok=`/`out=`/`hermitian=`/`UPLO=`/
`offset=`/`ndmax=`/`__array__`'s kwargs) they do not yet implement.

## Route from 238/1180 — measured absent buckets (2026-08-01)

Derived from `tools/numpy_surface.json` (authoritative 1180-item surface) minus
`anionpy.__ion_state__`. 1008 absent (the 942 in the ledger plus items counted
elsewhere), by namespace:

    394  (toplevel)      largest and most heterogeneous
    225  ma              masked arrays -- deliberately LAST, needs a mask-aware
                         array type threaded through every op
    164  ndarray         methods/attributes on the array type itself
     63  random          needs bit-exact PRNG (PCG64/MT19937 streams) to match
                         numpy's generated values -- mechanical but exacting
     54  char
     29  strings
     19  fft
     19  linalg
      9  emath
      9  rec             likely blocked on record/structured dtypes
      8  polynomial      likely blocked on a class hierarchy anionpy lacks
      6  ctypeslib       partly out of scope for a Rust core
      6  lib
      3  testing

Wave dispatched against this (disjoint file ownership, four concurrent):
1. `casting=`/`subok=` ufunc parameters -- reclaims the largest block of the 85
   just undeclared. `order=` explicitly EXCLUDED: it changes real memory layout
   and `ndarray.strides` is an observable ledger item, so it is unsatisfiable
   without strided-array support and accepting-and-ignoring it would manufacture
   silent wrong answers.
2. `fft` (19) -- must handle arbitrary n (Bluestein/mixed-radix), not radix-2
   only. Instructed to MEASURE ULP divergence from pocketfft and leave items
   undeclared rather than paper over it with unjustified tolerance.
3. `char`/`strings` (83) -- with an explicit warning not to reintroduce the
   numpy delegation that once made `strings.add` compare numpy against numpy.
4. misc namespaces (41) -- `emath` first as the real win; `rec`/`polynomial`
   asked for a FEASIBILITY REPORT before any implementation, since both likely
   need dtype/class machinery that does not exist yet.

Standing instruction to all four: an honest `absent` beats a dishonest `exact`,
and `failing`/`phantom`/`untested` must all stay 0.

## Blind spot #5 (alias-identity) — MEASURED, CLEAN. Plus two of my own false alarms.

**Measurement (2026-08-01, against 254 declared):** 21 numpy alias groups exist
among declared items. 19 are the `char.*`/`strings.*` pairs (`upper`, `lower`,
`strip`, `isalpha`, `add`, ...) where numpy has literal object identity
(`np.char.upper is np.strings.upper`). **anionpy preserves identity in all 19.**
Blind spot #5 is closed for the current surface; re-run the check after any new
char/strings item lands, since a fresh independent binding is the easy mistake.

The remaining 2 "groups" were an artifact of my detection method: I grouped numpy
names by `id()`, and Python interns `True`/`False`, so the boolean `testing.*`
flags collapsed into two fake "alias groups." Grouping by id only implies aliasing
for non-interned objects.

### False alarm #1 — "two declared items diverge from numpy"
`testing.IS_EDITABLE` (numpy False / anionpy True) and `testing.HAS_LAPACK64`
(numpy True / anionpy False) really do differ, and both are declared `exact` while
the ledger reports `failing 0`. That looks exactly like a false green. It is not.
`testing_cases.py` handles both deliberately and documents why:

- `HAS_LAPACK64` / `BLAS_SUPPORTS_FPE`: ionp-core has **zero BLAS/LAPACK
  dependency by architecture**, so anionpy's correct value is unconditionally False
  on every machine, while this venv's numpy links a real ILP64 LAPACK. Grading by
  equality to numpy would force the CORRECT implementation to FAIL. They are
  graded against a fixed, architecturally-justified constant — still falsifiable
  (a sabotaged `True` fails), never asserting a falsehood in either direction.
- `IS_EDITABLE` / `IS_INSTALLED` / `NUMPY_ROOT` describe **anionpy's own** install,
  not numpy's. Naively re-reading `anionpy.testing.<NAME>` on both sides would be
  circular — a sabotaged attribute would poison both reads identically — so the
  fact is *independently recomputed* and compared against the attribute.

This is the correct treatment of an item whose right answer legitimately differs
from numpy's. Do not "fix" these to match numpy; that would make them lie.

### False alarm #2 — "237 of 254 items pass with mechanism='none'"
`mechanism` is the **tolerance** mechanism, not the comparison mechanism.
`none` = no tolerance applied = **bit-exact**. It is the strictest value, not the
absence of a test. Confirmed by cross-check: r.json shows ulp 15 / epsilon 15 /
tie_exempt 3 / none 367, and coverage independently reports bit-exact 240,
ULP-tolerant 1, epsilon-tolerant 11, tie_exempt 2.

**Standing lesson, third instance this session.** Both alarms had the same tell:
a large, uniform, suspiciously round set of "defects" appearing all at once. That
is the signature of a fault in MY probe, not N independent bugs in the code.
Verify before accusing — and when the code has a comment explaining itself,
read it before overriding it.

## Agent-work loss: "my work was silently wiped" is self-inflicted, three times over

Three agents today reported prior work vanishing. In every case the object was
never lost — the agent's own `git reset` / `git checkout` moved the branch ref
off its commit, leaving it dangling and unreachable by name.

Most recent instance: the lib/ctypeslib commit `b253965` reported as "sitting on
my branch." `git log worktree-agent-<id>` showed the branch one commit BEHIND it,
`git branch --contains b253965` returned nothing, and the worktree HEAD was on my
memory commit. The agent had run `git reset --mixed 9a008c7` to fix what it
believed was a stale branch pointer; that reset is what orphaned its own work.

**Recovery (the object survives — do not let an agent redo the work):**
```
git cat-file -t <hash>            # 'commit' => object intact, just unreferenced
git branch recover-<name> <hash>  # give it a ref back
git merge --no-ff recover-<name>
```
`git reflog` and `git fsck --lost-found` find the hash if it wasn't reported.

**Standing instruction now in every agent dispatch:** do not run `git reset`,
`git checkout <branch>`, or `git stash`; commit atomically; report the hash by
PASTING it. (An agent also misreported its own hash by one character today and
another misidentified its parent commit — retyping hashes is how a verification
claim ends up pointing at the wrong object.)

## Worktree isolation is not self-enforcing

Two agents were found mutating ONE directory — byte-identical untracked files in
both trees. Consequence: a commit carried `mod emath;` for a file tracked nowhere,
which compiled in the polluted tree and broke everywhere else. Its "verified"
claim was true-there, false-in-main.

**Countermeasure that works, adopted as standard:** when neighbours hold WIP on
disk, verify a commit by `git worktree add` at THAT COMMIT in a throwaway dir,
symlink the venv, build from source there, then rebuild from the home tree to
restore the editable install. One agent did this unprompted and it is the only
verification method immune to on-disk contamination.

**And the rule that catches it regardless:** a failed `maturin` build leaves a
stale `.so`, and `coverage.py` will happily report a confident number against it.
NEVER read a ledger number without checking the build printed a success line
immediately above it. Four occurrences on this project so far.

## Probe traps (things that make MY audit lie, not the code)

The recurring tell — *a large, uniform, suspiciously round block of defects surfacing
at once is a fault in the probe, not N independent bugs* — has now fired three times in
one day. Two concrete mechanisms behind it, both cheap to check first:

**1. `anionpy.asarray(np_buffer)` COPIES.** Passing `out=anionpy.asarray(o)` and then
inspecting `o` will always show an unmutated buffer, because the mutation landed on the
copy. To test any `out=` path, allocate natively: `anionpy.empty(shape, dtype=...)`, pass
that object, and assert `result is out`. Also note `anionpy.empty` does not accept dtype
shorthand (`"c16"`, `"f8"`) — use `"complex128"` / `"float64"`.

**2. Not every declared item is bit-exact — check before demanding it.** `coverage.py`
prints the split (bit-exact / ULP-tolerant / epsilon-tolerant / tie_exempt). The fft
transforms are legitimately epsilon-tolerant: rustfft and pocketfft are different
algorithms and round differently (measured 1e-14 to 7e-12 relative). A probe that
byte-compares them reports a wall of failures against correct code. Confirm the declared
grading standard in `registry.py` before writing the assertion.

Corollary that IS worth asserting: an `out=` result must be byte-identical to the same
call's no-`out=` result. That is an internal-consistency check requiring no numpy, and it
catches real bugs without inheriting the tolerance question.

## Runtime-numpy reach-in: the failure that makes tests unfalsifiable

Rejected commit `0d2d122` constructed anionpy's fft casting error by importing
`numpy._core._exceptions` and instantiating numpy's private `_UFuncOutputCastingError`
against numpy's real pocketfft ufunc object. It passed every test. It had to: both sides
of the differential comparison were then raising numpy's own exception object, so the
comparison was structurally incapable of failing.

Two independent reasons this is always wrong, worth stating separately because either
alone is disqualifying:
- **Dependency.** A replacement that needs the thing it replaces installed in order to
  raise its own error is not a replacement.
- **Circularity.** A test that cannot fail is worse than no test, because it reports green.

Settled policy stands: construct the exception in Rust; where numpy's class is private,
target a Rust-constructed builtin with byte-identical message text plus a documented
narrow equivalence in the `_state` file. Public classes stay the real class. When
auditing any exception work, grep the implementation for `py.import(` before reading the
numbers — the numbers will look fine.

---

# STATUS 2026-08-03: 372 / 1180 (31.525%)

phantom 0, untested 0, **failing 93**, absent 715.

## The number went DOWN before it went up, on purpose

457 → 441 → 314 → 372. Read that arc before reacting to it:

- **457 → 441.** Sixteen `exact` declarations REVOKED. They had been declared the
  same day on a 1080-case probe grid returning zero mismatches — but that grid
  measured LAYOUT and had no cell for TYPE. Full reductions returned a 0-d
  `ndarray` where numpy returns a numpy SCALAR; `float(r)`, `hash(r)`, and
  dict-key use all raised where numpy works. A clean grid is evidence about the
  cells it has.
- **441 → 314.** The harness gained a return-type check it had never had. 176
  groups went red in one step. That is EXPOSURE, not regression — verified by
  before/after verdict diff: 0 groups went pass→fail from any code change.
- **314 → 372.** The root cause fixed at the ufunc layer (358142c), 61 groups
  green, 0 regressions.

Net: 85 below the morning's headline, and every one of the 372 now survives a
check that did not exist that morning. **The ledger got smaller and more honest
in the same motion.** Do not "recover" the 85 by re-declaring; it was never real.

## The 93 failing and the 715 absent are DISJOINT populations

This is the structural fact that should drive sequencing, and it was not
understood when this plan was written:

- the **93 failing** = declared-but-wrong. A lie in the ledger. Fix FIRST.
- the **715 absent** = unwritten. Honest, just incomplete.

The four root causes tracked all day (F-order output layout, empty-array stride
convention, free-function OWNDATA/.base metadata, Phase-2 view write-through)
touch the ABSENT population almost exclusively. They do not intersect the 93.

Cluster breakdown of the 93 is in `docs/failing-93-clusters.md`. Headline:
~84 are ONE defect (0-d result must demote to a numpy scalar) with the fixing
helper already written and proven at 4+ call sites. **But 6 `char.*` comparison
items need the EXACT OPPOSITE** — numpy always returns an array there — so a
blanket rule fixes 84 and breaks 6. Plus 7 items (`slogdet`/`eig`/`eigh`/`qr`/
`svd`/`unique_counts`/`unique_all`) are a different defect entirely: right
elements, wrong CONTAINER (numpy named tuple vs plain tuple), and
`numpy_scalar_from_0d` will never fix them.

## Method lessons that changed the process (not just the code)

- **#57** — the harness's array path compared what a result CONTAINED, never
  what it WAS. `np.asarray()` maps a numpy scalar and a 0-d array onto identical
  dtype/shape/values, so the check could not fail on the defect.
- **#59** — I published a wrong conclusion because I fixed a VISIBLE bug in my
  probe, got plausible output, and stopped. The instrument was still invalid for
  an unrelated reason (`np.asarray(ionp_arr)` allocates a fresh copy every call,
  so pointer comparison was measuring allocator reuse). **Validate the oracle
  against controls with independently known answers BEFORE trusting one verdict.**
- **#60** — a classifier that reads failure MESSAGES groups defects by how they
  are DESCRIBED, not by what they ARE. That is how the named-tuple container bug
  hid inside the scalar-demotion bucket.
- **#61** — "record .so mtime at start and end" is INSUFFICIENT. An agent did
  exactly that, both matched, and was still measuring a stale binary: it had
  edited a source file 11 seconds after its last build. `git status` was clean
  and flagged nothing. **Clean tree proves source consistency, never
  source/binary consistency.** Rebuild before believing any number.

## Standing sequencing rule

1. Fix **declared-but-wrong** before building absent. A wrong declaration costs
   more than a missing one.
2. Within absent, **check whether it is already built** before building it —
   the free/near-free bucket (correct Rust reachable only under another name,
   e.g. `outer`/`tensordot`/`cross` living under `linalg` but absent at top
   level) is repeatedly the cheapest coverage in the tree.
3. **Never declare an item to move the number.** Every declaration needs an
   out-of-corpus probe, not just a green corpus verdict.

---

## CORRECTION 2026-08-03 (late): "the DType gate is the only route" WAS WRONG

Earlier today this document, and I in conversation, asserted that the DType enum
expansion and the buffer/interior-mutability work were **the only route to 100%**
and that "there is no longer a cheap lane." That was an assertion, not a
measurement. I measured it. It is false, and false in the same optimistic-
sounding-pessimistic direction as every other unmeasured claim in this file:
it made the remaining work sound more BLOCKED than it is, which is just as
inaccurate as making it sound easier.

### Measured, 2026-08-03, at 597/1180

488 items absent within the `names` partition. Classified by whether a BLOCKED
DTYPE is what stops them:

| gate | absent items |
|---|---:|
| `longdouble` / `clongdouble` | 2 |
| `datetime64` / `timedelta64` / busday / `isnat` | 9 |
| fixed-width str / bytes (`char.*`, `strings.*`, `str_`, `bytes_`) | 30 |
| `object_` / `rec.*` / `record` / `recarray` | 12 |
| **dtype-gated subtotal** | **53** |
| **NOT dtype-gated** | **435** |

**The DType gate blocks about 11% of the absent surface, not the bulk.**

The 435 non-dtype-gated items, by family:

| family | count |
|---|---:|
| toplevel | 198 |
| `ma` | 131 |
| `random` | 63 |
| `fft` | 12 |
| `linalg` | 11 |
| `polynomial` | 8 |
| `ctypeslib` | 4 |
| `lib` | 4 |
| `testing` | 3 |
| `emath` | 1 |

### What this changes

The two architectural gates are still REAL and still Rust-only. Nothing about
them softened:
- **DType enum expansion** — measured blast radius: 180 `DType::C128` sites and
  246 `DType::F64` sites across 19 files, every exhaustive `match` needing a new
  arm. This is why it has not been attempted casually.
- **Buffer representation / interior mutability** — still gates `out=`,
  `ndarray.__setitem__`, the mutating `ma.*` family, and the in-place
  partial-view aliasing defect.

What changed is their POSITION. They are not the bottleneck for the next several
hundred items; they are a wall at the far end. Python-surface composition over
already-exact Rust kernels remains the correct and available lane for the
majority of what is left, and abandoning it for the Rust gates would have been
the wrong sequencing.

### The recurring failure mode, stated plainly

Every wrong claim in this document has come from **reasoning about the shape of
the remaining work instead of counting it.** Wave 1 was overstated 4.5x. The
family table understated `fft` 12x and `linalg` 11x. And this one asserted a
blocker covered the field when it covers 11% of it. The direction of the error
varies; the cause does not. Count first.

One narrow measured note, kept because it is cheap and true: on this platform
(arm64 macOS) `longdouble` is bit-identical to `float64` (itemsize 8, nmant 52,
same eps) and `clongdouble` to `complex128` (itemsize 16). They differ only in
`.char` ('g'/'G'), `.name`, and repr. That makes those 2 items far cheaper here
than the general DType work suggests -- but it is PLATFORM-SPECIFIC (on x86-64
Linux `longdouble` is 80-bit extended), so anything built on it must be
conditional and documented as such.

---

## 2026-08-07 — The five architectural blockers, measured

Roughly **600 of the 2379 absent items cannot be reached by grinding.** They are
gated on five pieces of missing architecture, and no amount of wrapper work
touches them. Anyone planning a route to 100% should plan around these first,
because they are the only items whose cost does not scale with their count.

| Blocker | Gates | Measured evidence |
|---|---|---|
| **void / structured dtype** | `recarray` (most of 167), `rec` 9, and 2 live failures (`promote_types`, `result_type`) | `ionp-core::dtype::DType` has no void variant. `np.rec.array`, `np.recarray(...)`, `np.rec.fromarrays` all produce `dtype.kind == 'V'`. |
| **buffer-protocol / view construction** | `memmap` (all 167), the `matrix(copy=False)` decline, and the standing `real`/`imag`/`asarray` declines | `anionpy.ndarray` has **no `.view` attribute at all**, and `anionpy.array()` always copies — mutating a numpy source after ingestion never touches the result. `memmap`'s entire purpose is a real file-backed shared buffer. |
| **`S` / `U` dtypes** | 226 absent `char`, 6 `strings` — and makes the **84 already-credited** char items unconditional | `anionpy.array(['hello'])` raises. The whole char surface runs on borrowed `numpy.ndarray` containers. See `KNOWN-DIFFERENCES.md` 2026-08-07. |
| **object dtype** | `min_scalar_type` (1 live failure), the `finfo` `dtype('O')` message text | `anionpy.dtype('O')` cannot be constructed; `PyDType` is a bare newtype over `DType` with no intermediate layer. |
| **datetime64 / timedelta64** | `isnat` (2 live failures), `busday_*`, `datetime_as_string`, `datetime_data` — ~13 items | Genuinely unstarted; every source reference is a documented absence. |

### A correction worth making precisely

An earlier measurement in this pass concluded `recarray` is "100% structurally
blocked on void dtype." **That is very nearly true and wrong in a way that would
cost someone a week.** One construction path escapes it:

```
>>> np.array([1, 2, 3]).view(np.recarray).dtype.kind
'i'
```

So `recarray` is blocked on void dtype for its meaningful surface **and
separately on view construction for the path that avoids void dtype.** Building
void dtype alone would not unblock `recarray`; someone would finish that work and
find the class still unreachable. Two blockers, not one — state it that way.

### What this means for "100%"

`polynomial` (647) and `ma` (~51 attemptable) are grindable and are the near-term
route. But the last ~600 items are architecture, and three of the five blockers
above are load-bearing for correctness claims we already make, not just for new
coverage: view construction underpins the `real`/`imag`/`asarray` declines, and
`S`/`U` underpins 84 items already counted in the headline percentage. Those two
should be sequenced ahead of the kinds that only add new items.
