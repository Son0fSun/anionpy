# `UFuncTypeError` revocation comments — per-item re-measurement, 2026-08-06

**Read-only investigation. No source touched, no build run, no runtime timing
taken.** Measured entirely against the already-installed binary
(`~/Monday/ionp/ionp/_ionp.abi3.so`, mtime 2026-08-06 08:48, i.e.
whatever the other agent holding the build lane last produced) via read-only
Python probes run from `/private/tmp/ufunc_probe/` (neutral cwd — confirmed
`import ionp` there still resolves the editable-install source tree at
`~/Monday/ionp/ionp/__init__.py`, which is expected for a `pip
install -e` layout; the compiled extension loaded by that `__init__.py` is
the `.so` above, not a repo-tree stub).

- **numpy version measured against:** 2.5.1
  (`~/Monday/ionp/.venv/lib/python3.14/site-packages/numpy`).
- **Probe scripts:** `/private/tmp/ufunc_probe/probe1.py`,
  `/private/tmp/ufunc_probe/probe2.py`, plus several inline one-off `python
  -c` probes reproduced verbatim below. None of these are part of the repo;
  they are throwaway and were not committed.

This document supersedes nothing in
`docs/UFUNCTYPEERROR-DESIGN-2026-08-05.md` — it is the re-measurement that
document's own "Status" section (2026-08-06 correction) said had **not**
been done yet. Read that document first; this one assumes its §2/§4
background.

## Search scope

**Searched:** `grep -rn "UFuncTypeError" ~/Monday/ionp/ionp/_state/*.py`
— this covers exactly the five files in that glob:
`toplevel.py`, `fft.py`, `char_strings.py`, `linalg.py`, `ndarray.py`. 18 hits,
matching the design doc's table.

**NOT searched:**
- Any file outside `ionp/_state/*.py` (e.g. `ionp-core/src/*.rs`,
  `ionp-py/src/*.rs`, `tests/differential/*.py`, other `docs/*.md`). The
  design doc's §2a flags one stale Rust doc-comment; I did not re-search Rust
  source for others.
- `_state` subdirectories or files matching other name patterns (there were
  none at the time of this search — the glob covered the whole directory).
- Git history / blame for when each comment was written relative to the
  binary's current behavior. All conclusions below are "as of the current
  `.so`" only.

**Conclusion scope:** the classifications below apply only to the specific
scenarios I actually ran (listed per row). "STALE" for a row means the
*specific reproduction described in that comment* no longer reproduces — not
that every conceivable dtype/casting/order combination for that item is now
clean.

## Method note on the 18 → item mapping

The 18 comments do not map 1:1 to 18 blocked items. Some comments cite
`UFuncTypeError` for an item that is **already declared "exact"** elsewhere
in the same file (the comment is left-over historical narration, not an
active blocker). Some comments govern a whole batch of ufuncs at once (one
comment, ~10 dict keys). I resolved each hit by reading the surrounding
`_state` dict and checking whether the named key(s) currently appear as
live `"exact"` entries or as commented-out/absent entries. Where that was
genuinely ambiguous, I say so instead of guessing.

**Distinct items/groups identified: 9** (covering roughly 29 individual
dict-key-level items, several of which are batch-declared together).

## Per-item table

| # | Item(s) | Comment location | Comment's claimed blocker | Currently held back? | Measured class verdict | Measured message verdict | Classification |
|---|---|---|---|---|---|---|---|
| 1 | `strings.str_len`/`isalpha`/`isdigit`/`upper`/`isdecimal`/`isnumeric`/... (class-C unary predicates on numeric/bool/complex input, ~10 keys) | `char_strings.py:90,157` | ionp raises plain `TypeError` where numpy raises private `_UFuncNoLoopError`/`UFuncTypeError` | **No** — already declared `"exact"` via an explicit `exception_equivalences` entry in `tests/differential/strings_cases.py` (design choice, not a pending TODO) | **DIFFERS** (`TypeError` vs `UFuncTypeError`) | **MATCHES** verbatim | STILL BLOCKED, DIFFERENT REASON — by design. This is a permanent architectural choice (never construct numpy's private class), already accepted by the harness equivalence. Not something this task's message-text work would change. |
| 2 | fft `out=` dtype-casting family (`fft.fft`/`ifft`/`irfft`/`rfft_n_even`/`rfft_n_odd`, the "5 real pocketfft names") | `fft.py:125,200-201` | "`np.fft.fft(a, out=int64 array)` raises `UFuncTypeError`, ionp raises `TypeError`" | **Ambiguous** — see note below | **MATCHES** (`UFuncTypeError` both sides) | **MATCHES** verbatim | **STALE**, for the exact scenario cited. See mapping-ambiguity note below. |
| 3 | `linalg.vector_norm`, `linalg.norm` (vector branch) | `linalg.py:430` | "`object()` raises `UFuncTypeError`" (one of 3 named divergence axes) | **Yes** — both commented out / undeclared | **DIFFERS** (`TypeError` vs `UFuncTypeError`) | **DIFFERS** completely (different mechanism, not just text) | STILL BLOCKED, DIFFERENT REASON — deeper than message text; ionp doesn't even reach ufunc dispatch for this path (see message pair below). `UFuncTypeError` was never claimed as the sole blocker here. |
| 4 | `ndarray.__pos__` | `ndarray.py:402` | bool input needed rejection "to match numpy's `UFuncTypeError`" | **No** — declared `"exact"` (`ndarray.py`, `"ndarray.__pos__": "exact"`) | **MATCHES** | **MATCHES** verbatim | STALE / already resolved — this is historical narration of a fix already shipped, consistent with the current declaration. Not a currently-live blocker. |
| 5 | `ndarray.__imod__`, `ndarray.__ilshift__`, `ndarray.__irshift__` | `toplevel.py:64` (in-place `same_kind` casting-refusal note) | "ionp's in-place dispatch does not enforce numpy's output-casting-rule refusal ... must raise `UFuncTypeError`" | **No** — all three currently declared `"exact"` in `ndarray.py` (via a later, separate stride-gap revoke/re-declare cycle) | Not directly re-measured for these three specific keys | Not directly re-measured | **CANNOT DETERMINE** for these three specifically — I measured the sibling `__ipow__` instead (row 6) and it now matches; I did not run the identical same_kind scenario against `__imod__`/`__ilshift__`/`__irshift__` themselves before writing this up. Circumstantial evidence (row 6, and that they're already declared exact) suggests likely-fixed, but that is not a measurement. |
| 6 | `ndarray.__ipow__` | `toplevel.py:64,72` | Same in-place `same_kind` casting-refusal note, plus a separate complex64 ULP note | **Yes** — still undeclared (`ndarray.py:1886-1890`) | **MATCHES** (both `UFuncTypeError`) | **MATCHES** verbatim | **STALE** for the `UFuncTypeError` blocker specifically — see message pair below. Item stays undeclared overall, but now for the *unrelated* complex64 ULP reason named in the same comment, not this one. |
| 7 | `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal` | `toplevel.py:382` | str-operand case: "numpy → `UFuncTypeError(...)`, ionp → generic operand-type `TypeError`" (1 of 3 named divergence axes, alongside `object()`/`None` handling) | **Yes** — all six commented out / undeclared (`toplevel.py:344-349`) | **DIFFERS** | **DIFFERS** completely | STILL BLOCKED, DIFFERENT REASON, and **more fundamental than message text**: ionp cannot construct an `ionp.ndarray` from a string-dtype numpy array at all — it fails during input marshalling, never reaches ufunc dispatch. Confirmed with both a plain numpy-array operand and (per the harness rule against false negatives from copy-not-view) an **ionp view slice** of a string array — same result either way. |
| 8 | `matmul`, `vecdot`, `matvec`, `vecmat` | `toplevel.py:2228`, `5062-5065` | "`np.matmul(f64,f64,dtype=int64)` → `UFuncTypeError`; `ionp.matmul(...)` → silently returns truncated wrong answer" | **Yes** — all four commented out / undeclared | **RAISE-MISMATCH** (numpy raises, ionp does not — no class to compare) | N/A | STILL BLOCKED, DIFFERENT REASON, **and this is the worst class**: not a message mismatch at all — ionp silently computes and returns a wrong numeric result where numpy refuses. Confirmed reproducing today for all four gufuncs (see pairs below). |
| 9 | `spacing`, `nextafter`, `logaddexp`, `logaddexp2`, `heaviside`, `fmax`, `fmin`, `gcd`, `lcm`, `bitwise_count` (the "10 newly declared ufuncs" batch) | `toplevel.py:5831,5835,5843` | "792 [of 948] mismatches with byte-identical message text [class only], 156 also with diverging text (gcd/lcm under `dtype=`)"; `bitwise_count` additionally cites a `casting='no'` bool-input divergence | **No** — all 10 currently declared `"exact"` | 9 of 10 **MATCH**; `bitwise_count` **RAISE-MISMATCH** | 9 of 10 **MATCH**; `bitwise_count` N/A | **STALE** for `spacing`/`nextafter`/`logaddexp`/`logaddexp2`/`heaviside`/`fmax`/`fmin`/`gcd`/`lcm`. **STILL BLOCKED, DIFFERENT REASON — and currently mis-declared** for `bitwise_count`: see flag below, this is a real, reproducing divergence in an item the ledger currently marks `"exact"`. |

### Note on row 2's mapping ambiguity

The comment at `fft.py:196-204` sits, in file order, directly above three
commented-out entries (`fft.fft`, `fft.ifft`, `fft.rfft`) — but those three
are each individually annotated `REVOKED ... (stride-gap classification)`,
a **different, unrelated** reason (transpose/relayout quirk on non-last-axis
transforms), not this `UFuncTypeError` note. Immediately below those three,
`fft.irfft` and `fft.hfft` **are** currently declared `"exact"`, with no
comment re-litigating the `out=`-casting equivalence question for them. The
comment's own text says "10 ufuncs were just withdrawn for" the same defect
— that count matches row 9's batch of 10 toplevel ufuncs, not any FFT item,
suggesting this paragraph may be an out-of-place / copy-pasted reference to
that other withdrawal rather than a live blocker on any FFT dict key today.
I could not confirm which (if any) currently-absent FFT item this comment
is actually gating. What I *can* say is that the literal scenario it
describes (`np.fft.fft(a, out=int64 array)`) no longer reproduces the cited
divergence — see the message pair below.

## Verbatim message pairs

### Item 3 — `linalg.norm`/`vector_norm`, `ord=object()`

```
numpy: "Cannot cast ufunc 'power' output from dtype('O') to dtype('float64') with casting rule 'same_kind'"
ionp : "must be real number, not object"
```
(numpy's class: `UFuncTypeError` / `_UFuncOutputCastingError`. ionp's class:
plain `TypeError`, raised from what looks like a Python-level `**`/`pow()`
type check rather than any ufunc-casting logic — i.e. not a different
message on the same code path, but a structurally different code path.)

### Item 7 — comparison ufuncs, str operand (`less`, plain array; identical for `equal`, `greater`, and repeated with an ionp **view slice** operand)

```
numpy: "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.Int64DType'>) -> None"
ionp : "unsupported numpy dtype for ionp.array() (supported: bool, int8-64, uint8-64, float16/32/64, complex64/128)"
```

### Item 7 — comparison ufuncs, `object()` operand

```
np.equal(object(), 1):
  numpy: OK -> np.False_        (no exception on numpy's side at all)
  ionp : TypeError: 'ufunc operand must be an ionp.ndarray, a numpy scalar/array, or a Python bool/int/float/complex'

np.greater(object(), 1):
  numpy: TypeError: "'>' not supported between instances of 'object' and 'int'"
  ionp : TypeError: 'ufunc operand must be an ionp.ndarray, a numpy scalar/array, or a Python bool/int/float/complex'
```

### Item 8 — `matmul`/`vecdot`/`matvec`/`vecmat`, `dtype=int64` narrowing

All four raise on numpy's side and silently compute a truncated wrong
answer on ionp's side — no message pair exists because ionp never raises:

```
np.matmul(float64 2x2, float64 2x2, dtype=int64):
  numpy: UFuncTypeError "Cannot cast ufunc 'matmul' input 0 from dtype('float64') to dtype('int64') with casting rule 'same_kind'"
  ionp : OK -> array([[11, 15], [21, 29]])   (1.5 truncated to 1, silently)

np.vecdot(float64, float64, dtype=int64):
  numpy: UFuncTypeError "... ufunc 'vecdot' ... same_kind ..."
  ionp : OK -> np.int64(8)

np.matvec(float64, float64, dtype=int64):
  numpy: UFuncTypeError "... ufunc 'matvec' ... same_kind ..."
  ionp : OK -> array([8, 16])

np.vecmat(float64, float64, dtype=int64):
  numpy: UFuncTypeError "... ufunc 'vecmat' ... same_kind ..."
  ionp : OK -> array([11, 15])
```

### Item 9 — `bitwise_count`, `casting='no'`, bool input (the one member of the "10 newly declared ufuncs" batch that did NOT come clean)

```
numpy: UFuncTypeError "Cannot cast ufunc 'bitwise_count' input from dtype('bool') to dtype('int8') according to the rule 'no'"
ionp : OK -> array([1, 0], dtype=uint8)   -- performs the cast numpy refuses, silently
```
**This exactly reproduces the original withdrawal note's counterexample**,
against today's binary, even though `bitwise_count` is currently declared
`"exact"` in `toplevel.py` (line 6642) — re-declared 2026-08-02 for an
unrelated 0-d-scalar-return-type fix that never touched this casting path.
This is outside this task's literal scope (the item is not "held back"), but
it is directly load-bearing for anyone trusting that declaration, so it is
flagged prominently rather than filed only in a footnote.

## Items measured STALE, with matching pairs (brief, no divergence to show)

For completeness, all of the following now show `class match: True` AND
`msg match: True` against the exact scenario the withdrawing comment
described:

- `ndarray.__ipow__`, `bool_array **= int_array` (same_kind refusal):
  `UFuncTypeError "Cannot cast ufunc 'power' output from dtype('int64') to dtype('bool') with casting rule 'same_kind'"` — both sides identical.
- `ndarray.__ipow__`, `(3,1)_array **= (3,4)_array` (in-place broadcast shape,
  a `ValueError` not `UFuncTypeError`, tested because the same comment names
  it as the second structural gap): both sides
  `ValueError "non-broadcastable output operand with shape (3,1) doesn't match the broadcast shape (3,4)"` — identical.
- `fft.fft(a, out=int64_array)`: both sides
  `UFuncTypeError "Cannot cast ufunc 'fft' output from dtype('complex128') to dtype('int64') with casting rule 'same_kind'"`.
- `ndarray.__pos__`, `+bool_array`: both sides
  `UFuncTypeError "ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.BoolDType'> -> None"`.
- `gcd`/`lcm` under `dtype=` (both the no-loop case, `dtype=float64`, and the
  casting-refusal case, `dtype=int8, casting='safe'` with values that
  overflow int8): both sides match, both class and message, in both
  sub-cases.
- `spacing`, `nextafter`, `logaddexp`, `logaddexp2`, `heaviside`: representative
  dtype-narrowing probe (`dtype=int64` against a float64 operand pair, which
  for these five is a genuine "no loop" case, not a casting-refusal case) —
  both sides raise plain `TypeError "No loop matching the specified
  signature and casting was found for ufunc '<name>'"`, identical.
- `fmax`, `fmin`: `dtype=int64` narrowing (these DO have int64 loops, so this
  is a real casting-refusal case unlike the five above) — both sides
  `UFuncTypeError "Cannot cast ufunc '<name>' input 0 from dtype('float64') to dtype('int64') with casting rule 'same_kind'"`.
- Original bug-origin cases from the withdrawn `sign`/`abs`/`absolute`/
  `negative`/`subtract` dtype= note: `np.absolute(complex128, dtype=float32)`
  (input-casting refusal direction) now `UFuncTypeError`-matches on both
  sides; `np.negative(bool_array, dtype=int64)` (the silent-wrong-in-ionp
  direction) now succeeds identically on both sides too. Neither `sign` nor
  `negative`/`subtract`/`abs`/`absolute` is currently "held back" (all
  declared `"exact"`), so this is corroborating evidence for those
  declarations rather than a row of its own — but it is worth recording that
  the precasting-loop-selection fix those declarations' preconditions
  required does appear to have actually landed.

**ILLUSTRATIVE, NOT EXHAUSTIVE** for every item above: each is 1-2 probe
shapes/dtypes, not a sweep. In particular none of these re-runs varied
`order=`, `where=`, non-default `casting=` values beyond the one cited, 0-d
or empty-array inputs, or float16.

## The `<class '...'>` vs `dtype('...')` rendering split

**DETERMINED**, at the level of "which numpy subclass renders which way and
why its `__str__` produces that form" — via
`~/Monday/ionp/.venv/lib/python3.14/site-packages/numpy/_core/_exceptions.py`:

- `_UFuncNoLoopError.__str__` (lines 44-49) formats `self.dtypes`, a tuple
  populated (per its `__init__`, line 40-42) from whatever `dtypes` argument
  the raising code passed in. Empirically, across every no-loop case probed
  (`less`/`equal`/`greater` with a str operand, `positive` on bool,
  `isalpha` on int64, `spacing`/`nextafter`/`logaddexp`/`logaddexp2`/
  `heaviside` under an unreachable `dtype=`), that argument is always a
  tuple of **DType classes** (`<class 'numpy.dtypes.StrDType'>`,
  `<class 'numpy.dtypes.BoolDType'>`, ...), never dtype instances — the
  `repr()` numpy's own `!r` formatting then produces is `<class '...'>`.
- `_UFuncCastingError` and its `_UFuncInputCastingError`/
  `_UFuncOutputCastingError` subclasses (lines 68-105) format `self.from_`/
  `self.to`, populated from the `from_`/`to` constructor arguments. Every
  casting-refusal case probed (`add` output cast, `power`/`fmax`/`fmin`/
  `matmul`/`fft` input or output cast) passes **dtype instances**
  (`dtype('float64')`), whose `!r` is `dtype('...')`.

So the rule is: **which numpy exception subclass is raised** determines the
rendering, because the two subclasses store different kinds of objects in
the attributes their own `__str__` formats — `_UFuncNoLoopError` always
carries DType *classes* (there is no single concrete dtype instance to
report when no loop matches at all — the error is about which abstract type
category was tried), `_UFuncCastingError` always carries dtype *instances*
(a casting refusal is inherently about two concrete, already-resolved
dtypes). This is not a formatting inconsistency or a rendering bug in numpy
— it's a structural consequence of what each exception class represents.

**Caveat on the C-level provenance**: what I verified is the Python-level
`_exceptions.py` source (the `__str__` methods and the attribute names they
format) plus consistent empirical behavior across 12+ probes spanning both
subclasses. I did **not** trace the C call sites inside numpy's compiled
`_multiarray_umath` extension that actually construct these exception
objects and choose which object type to pass as `dtypes`/`from_`/`to` —
that C source isn't present as inspectable source in this venv (only the
compiled `.so`/`.pyd`, no `.c`/`.pyx` shipped in the numpy wheel). The rule
above is therefore established by (a) reading the Python formatting code
that is unambiguous about which stored objects get which repr, and (b)
confirming empirically, across every no-loop vs. casting-refusal case
probed in this task, that the two never mix. I did not find a single
counterexample. I am NOT separately claiming to know *why* numpy's C code
chooses to pass classes vs instances in the first place beyond "that's what
each subclass's contract requires" — that would be a design-intent claim
about numpy's own C source I have no visibility into.

## Not checked

- Any numpy version other than 2.5.1.
- `order=`, `where=`, non-default `casting=` values beyond the one cited per
  scenario above.
- 0-d, empty-array, and float16 operands for every scenario above (only
  spot-checked where a comment specifically named them).
- Structured/void/datetime/timedelta dtypes anywhere.
- `ndarray.__imod__`/`__ilshift__`/`__irshift__`'s own same_kind casting
  scenario directly (row 5 — deferred to the sibling `__ipow__` measurement,
  explicitly marked CANNOT DETERMINE rather than assumed).
- The full identity of the other ~7 unnamed members of the "10 newly
  declared ufuncs" batch beyond what I could recover by cross-referencing
  nearby declarations (`spacing`, `nextafter`, `logaddexp`, `logaddexp2`,
  `heaviside`, `fmax`, `fmin`, `gcd`, `lcm`, `bitwise_count` — this list of
  10 was reconstructed from context, not read off an explicit enumeration
  in the comment itself; I'm reasonably confident in it because it's
  exactly 10 names and each is a ufunc newly declared on 2026-08-02 per
  neighboring comments, but flagging the reconstruction as such).
- Whether `fft.py`'s row-2 ambiguity resolves to a real currently-gated item
  or is genuinely dead/orphaned text — I could not determine this from
  static reading alone and did not attempt to run the differential harness
  (out of scope: read-only, no build/test-run authorization implied by "run
  read-only Python probes").
- Every string-unary-predicate name individually (`str_len`, `isdigit`,
  `isalpha`, `upper`, `isdecimal`, `isnumeric`, and whatever else the "ten
  unary predicates" comprise) — only `isalpha` was directly probed; the
  comment's own text asserts the same Rust constructor
  (`numpy_no_loop_type_error()`) backs all of them uniformly, which I did
  not independently re-verify for each name.
- Whether the `tests/differential/*_cases.py` `exception_equivalences`
  entries that currently paper over class-only mismatches (strings, fft
  out=-casting) are still necessary now that several of the underlying
  scenarios (e.g. fft out=-casting itself) measure as fully matching —
  that's a harness-cleanup question, not something this task's mandate
  covers.
- Whether `bitwise_count`'s currently-live `"exact"` declaration should be
  revoked. I flagged the reproducing divergence; declaring it un-exact is
  not something this task is authorized to do, and wasn't asked for.

## Declaration authority

**No item in this document is being declared.** This is measurement only,
per the task's hard rules. Several rows above measure as STALE (the cited
`UFuncTypeError` blocker no longer reproduces), but STALE is not the same as
DECLARABLE — most of the STALE-classified items are STALE precisely because
they are *already* declared `"exact"` for other/later reasons (rows 1, 4, 6
partially, 9 mostly), not because this document is clearing them for
declaration. The two rows that are both currently-undeclared AND where I
found no remaining `UFuncTypeError`-specific divergence (`ndarray.__ipow__`,
and the ambiguous fft row) are explicitly still blocked or ambiguous for
independent reasons named in the table — nothing here is ready to flip.
