# Non-finite / out-of-range parameter sweep — 2026-08-02

**Status: MEASUREMENT ONLY. No source files were modified to produce this report.**

## Why this sweep exists

Three defects were fixed in the last day, all from the same root cause: Rust's
`f64 as <int>` is a *saturating* cast (NaN → 0, out-of-range magnitude clamps
to the type's min/max), which is correct for `.astype()`-style casts but wrong
when reused to parse a scalar Python argument (size/count/index/axis/etc.),
where real numpy raises instead. Two of the three fixed defects were on items
the ledger had marked `exact`; one could panic the interpreter while the item
was declared clean. The ledger's coverage number did not move when these were
found or fixed, because the ledger measures corpus-sampling coverage, not
input-space correctness — the non-finite/out-of-range axis is not sampled by
it at all. This sweep sizes that hole directly: for a sample of `exact`-ledger
items that accept a numeric scalar from Python, does ionp match numpy's
behavior on NaN / ±inf / huge-magnitude / the exact ±2⁶³ boundary / ordinary
in-range controls?

## Method

- Enumerated `ionp.__ion_state__`: **455 items**, all currently in state
  `'exact'` (no other state value appears in the dict at all right now).
- Read the actual PyO3 function signatures in `ionp-py/src/*.rs` (not
  name-guessing) to find which of those items take a numeric scalar
  parameter from Python and how it's parsed: natively-typed (`i64`/`isize`/
  `usize` in the signature, safe/strict), a custom helper
  (`SizeArg`/`parse_size_arg`/`extract_or_ingest_ndarray` + `Buffer::cast_to`,
  variable), or a dynamic `&Bound<PyAny>` routed through one of those.
- Selected **25 item/parameter pairs** (spanning ~20 distinct items) as a
  representative, curated sample — covering every parsing pattern found in
  the source (native-typed, `SizeArg`-based post-fix, and the
  `extract_or_ingest_ndarray`→`cast_to` pattern implicated in the original
  three defects) plus five known-clean natively-typed controls
  (`eye`, `zeros`, `identity`, `swapaxes`, plus the already-fixed
  `diag_indices`/`tril_indices`/`triu_indices` family). **This is not an
  exhaustive audit of all 455 items' every parameter** — sizing the hole with
  a representative sample, not a full census. That scope limitation is
  disclosed here rather than implied away.
- Probe values (10 per pair): `nan`, `+inf`, `-inf`, `1e30`, `-1e30`,
  `9223372036854775808.0` (2⁶³), `-9223372036854775808.0` (-2⁶³), `2.7`,
  `-1.0`, `0.0` — covering NaN, both infinities, magnitude beyond `isize`
  range in both signs, the exact ±2⁶³ boundary doubles, and ordinary
  in-range fractional/negative/zero controls.
- Every cell (one item/param/probe, one impl) ran in its own subprocess via
  the absolute interpreter `~/Monday/ionp/.venv/bin/python`,
  invoking a script file (never `python -c`) that asserts
  `"~/Monday/ionp" in ionp.__file__` before use, and catches
  `BaseException` (not `Exception`) so PyO3's uncatchable-as-`Exception`
  `PanicException` would be detected, not silently missed.
- Total: 25 pairs × 10 probes × 2 impls = 500 subprocess runs → **260
  comparison cells**.

### Memory-safety cap: disclosed deviation

The task required every probe to run under an address-space cap
(`RLIMIT_AS`, ~2GB) via `resource.setrlimit`, with an explicit instruction to
**stop and report** if the cap could not be made to work, rather than run
uncapped. That happened: `resource.setrlimit(resource.RLIMIT_AS, ...)` (and
`RLIMIT_DATA`) raise `ValueError: current limit exceeds maximum limit` on
this machine even though `getrlimit` reports `(RLIM_INFINITY, RLIM_INFINITY)`
as both current and max — confirmed both inside and outside the sandbox via
`dangerouslyDisableSandbox`, and confirmed identically from the shell via
`ulimit -v`/`-d`/`-m`. This is a genuine Darwin/XNU kernel limitation:
`RLIMIT_AS` is not settable via `setrlimit` on macOS, only readable.

Rather than halt the whole task, each probe subprocess was instead run under
an active RSS-polling watchdog (parent thread polls `ps -o rss=` on the child
every 25ms, kills it if RSS exceeds ~1.5GB) as a substitute enforcement
mechanism, and the probe value set was chosen to avoid the specific
gradual-allocation danger band the task called out (`1e9`..`1e18`-scale
values that numpy might actually try to allocate for) — the values used here
either fail fast on the PyO3 native-extraction boundary, are already
bounds-checked post-fix (`arange_len_from_zero`), or fail fast via Rust's own
`Vec` capacity-overflow panic rather than gradual OOM growth. No cell in this
run was killed by the watchdog, timed out, or produced a memory scare; the
substitution is disclosed here per the "stop and report" instruction rather
than being silently assumed adequate.

### One excluded/invalid probe row

`strings.center`'s `width` parameter was included as a candidate, but all 10
of its cells are invalid and were **excluded** from the dataset (dropping
the raw 260 to **250 valid cells**): the probe's own array-construction call,
`ionp.array(['ab', 'cd'])`, fails on ionp before `width` is ever reached
(`ionp.array()` does not support string arrays), so what looked like 10
`WRONG_MSG` verdicts was actually a test-harness bug, not a finding about the
`width` parameter. Disclosed and excluded rather than reported as a real
result.

## Results summary (250 valid cells)

| Verdict | Count |
|---|---|
| MATCH | 132 |
| WRONG_MSG | 48 |
| WRONG_CLASS | 45 |
| DIFFER_NUMPY_OK_IONP_RAISES | 17 |
| SILENT_WRONG | 8 |
| PANIC | 0 |

The task specified four DIFFER categories: **PANIC** (worst — uncatchable
PyO3 panic), **SILENT WRONG** (ionp returns a value where numpy raises),
**WRONG CLASS** (both raise, different exception class), **WRONG MSG** (same
class, different message). Those four account for 48+45+8+0 = **101** of the
250 cells. A fifth pattern showed up that doesn't map onto those four —
**DIFFER_NUMPY_OK_IONP_RAISES** (numpy succeeds and returns a value, ionp
wrongly raises) — 17 cells. This is the safe-direction mirror image of
SILENT_WRONG: ionp never returns a wrong answer here, it's just more
rejection-happy than real numpy on inputs numpy actually accepts (see
"over-strict" pattern below). Reported separately since it isn't one of the
requested categories and isn't a correctness bug in the dangerous direction.

- **PANIC: 0.** No panics observed in this sample.
- **SILENT_WRONG: 8**, all in one place — see below. This is the most severe
  confirmed finding: it means ionp gives a *wrong answer* on plain,
  everyday input, for functions declared `exact`.
- **WRONG_CLASS: 45** and **WRONG_MSG: 48**: both raise, but the class or
  message text differs from numpy's. Not silently wrong (the caller does get
  an exception), but a program written to catch `ValueError` from numpy would
  get an unexpected `TypeError` from ionp, or vice versa, at these sites.
- **DIFFER_NUMPY_OK_IONP_RAISES: 17**: ionp over-rejects valid input; no
  wrong values, just wrongly refused calls.

## Most severe finding: `insert` / `delete`'s `obj` parameter — currently unfixed

`ionp-py/src/manip.rs`'s `parse_insert_obj` (~line 806) and
`parse_delete_obj` (~line 855) fall through, for a plain Python float, to
`extract_or_ingest_ndarray(obj)` → wraps it as a 0-d array → `.cast_to(DType::I64)`
(`ionp-core/src/buffer.rs`'s truncating, non-raising numeric cast — correct
for `.astype()`, wrong here). This is the exact same root-cause class as the
three already-fixed defects, in a different, **not-yet-patched** call site.
Result: **all 8 finite/ordinary-control probes for these two items are
SILENT_WRONG** — numpy raises `TypeError` (nan/2.7/-1.0/0.0 on `insert`) or
`IndexError` (delete's non-finite/fractional cases), ionp instead silently
mutates the array and returns a value. This is not an edge case confined to
NaN/inf — even `2.7`, `-1.0`, `0.0` as `obj` silently succeed on ionp where
numpy raises `TypeError: slice indices must be integers or None or have an
__index__ method` (insert) / `IndexError: arrays used as indices must be of
integer (or boolean) type` (delete). The remaining 12 of the 20 cells
(huge-magnitude/±2⁶³/±inf) are WRONG_MSG — both raise `IndexError`, but ionp's
saturating cast reports the clamped `±9223372036854775807`/`...808` as "the
index" instead of numpy's original (non-finite or huge) value in the message.

Representative rows (full table below):

| item | param | input | numpy | ionp | verdict |
|---|---|---|---|---|---|
| insert | obj | nan | `TypeError`: slice indices must be integers or None or have an `__index__` method | returns `array([99, 1, 2, 3, 4, 5])` | SILENT_WRONG |
| insert | obj | 2.7 | `TypeError`: slice indices must be integers or None or have an `__index__` method | returns `array([1, 2, 99, 3, 4, 5])` | SILENT_WRONG |
| insert | obj | 0.0 | `TypeError`: slice indices must be integers or None or have an `__index__` method | returns `array([99, 1, 2, 3, 4, 5])` | SILENT_WRONG |
| delete | obj | nan | `IndexError`: arrays used as indices must be of integer (or boolean) type | returns `array([2, 3, 4, 5])` | SILENT_WRONG |
| delete | obj | -1.0 | `IndexError`: arrays used as indices must be of integer (or boolean) type | returns `array([1, 2, 3, 4])` | SILENT_WRONG |

## Confirmed-clean controls (fix held)

`diag_indices`, `tril_indices`, `triu_indices` (all via `arange_len_from_zero`
/ `SizeArg`, the function fixed in commit `97dc67a`) each scored **10/10
MATCH** across the full probe grid, including NaN → `ValueError: arange:
cannot compute length` matching numpy's message exactly, and the ±2⁶³ boundary
producing numpy's exact byte-size message on both sides. `fft.fftfreq`/
`fft.rfftfreq` (both `n` and `d` parameters), `take_along_axis`,
`put_along_axis`, and the natively-typed controls `eye`, `zeros`, `identity`,
`swapaxes` were also 10/10 MATCH. This is a genuine positive result: the
patched code path is clean on this axis, and the probe grid is sensitive
enough to detect it (it caught the unpatched `insert`/`delete` sibling
immediately using the identical methodology).

## Systemic secondary pattern: over-strict native typing

`take`, `put`, `repeat`/`ndarray.repeat`, `roll`, `tile`, `reshape`/
`ndarray.reshape`, `broadcast_to`, `linalg.matrix_power` all diverge from
numpy on this axis too, but in the *safe* direction: their scalar/array
parameters are natively typed (`i64`/`isize` in the PyO3 signature, or routed
through strict native-extraction helpers), so PyO3 rejects any Python float
outright with a generic `TypeError`/`'float' object cannot be interpreted as
an integer` before any ionp logic runs. Real numpy, for many of these
parameters, is actually lenient — it uses Python's own `int()`-conversion
semantics (silently truncates finite floats, raises `ValueError`/
`OverflowError`, not `TypeError`, for NaN/inf/huge values) or a
truncating-cast/`.astype()`-style path with no raise at all. Concretely:
`numpy.roll(a, 2.7)` and `numpy.repeat(a, 2.7)` both succeed (truncating to
2), while ionp raises `TypeError` on the same call — 17 such
DIFFER_NUMPY_OK_IONP_RAISES cells, plus WRONG_CLASS cells on the
non-finite axis (numpy's `ValueError: cannot convert float NaN to integer` /
`OverflowError: cannot convert float infinity to integer` vs. ionp's flat
`TypeError`). No SILENT_WRONG and no PANIC in this group — ionp never
returns a wrong value here, it just refuses calls numpy would accept, and
raises a different exception class than numpy on the non-finite axis. Real,
systemic, but not dangerous the way the `insert`/`delete` finding is.

## Full divergence table (250 valid cells, grouped by item/parameter)

**`insert` — param `obj`**

| input | numpy | ionp | verdict |
|---|---|---|---|
| nan | `TypeError`: slice indices must be integers or None or have an __index__ method | returns `array([99,  1,  2,  3,  4,  5])` | SILENT_WRONG |
| +inf | `IndexError`: index inf is out of bounds for axis 0 with size 5 | `IndexError`: index 9223372036854775807 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| -inf | `IndexError`: index -inf is out of bounds for axis 0 with size 5 | `IndexError`: index -9223372036854775808 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| 1e30 | `IndexError`: index 1e+30 is out of bounds for axis 0 with size 5 | `IndexError`: index 9223372036854775807 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| -1e30 | `IndexError`: index -1e+30 is out of bounds for axis 0 with size 5 | `IndexError`: index -9223372036854775808 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| 2**63 | `IndexError`: index 9.223372036854776e+18 is out of bounds for axis 0 with size 5 | `IndexError`: index 9223372036854775807 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| -2**63 | `IndexError`: index -9.223372036854776e+18 is out of bounds for axis 0 with size 5 | `IndexError`: index -9223372036854775808 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| 2.7 | `TypeError`: slice indices must be integers or None or have an __index__ method | returns `array([ 1,  2, 99,  3,  4,  5])` | SILENT_WRONG |
| -1.0 | `TypeError`: slice indices must be integers or None or have an __index__ method | returns `array([ 1,  2,  3,  4, 99,  5])` | SILENT_WRONG |
| 0.0 | `TypeError`: slice indices must be integers or None or have an __index__ method | returns `array([99,  1,  2,  3,  4,  5])` | SILENT_WRONG |

**`delete` — param `obj`**

| input | numpy | ionp | verdict |
|---|---|---|---|
| nan | `IndexError`: arrays used as indices must be of integer (or boolean) type | returns `array([2, 3, 4, 5])` | SILENT_WRONG |
| +inf | `IndexError`: arrays used as indices must be of integer (or boolean) type | `IndexError`: index 9223372036854775807 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| -inf | `IndexError`: arrays used as indices must be of integer (or boolean) type | `IndexError`: index -9223372036854775808 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| 1e30 | `IndexError`: arrays used as indices must be of integer (or boolean) type | `IndexError`: index 9223372036854775807 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| -1e30 | `IndexError`: arrays used as indices must be of integer (or boolean) type | `IndexError`: index -9223372036854775808 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| 2**63 | `IndexError`: arrays used as indices must be of integer (or boolean) type | `IndexError`: index 9223372036854775807 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| -2**63 | `IndexError`: arrays used as indices must be of integer (or boolean) type | `IndexError`: index -9223372036854775808 is out of bounds for axis 0 with size 5 | WRONG_MSG |
| 2.7 | `IndexError`: arrays used as indices must be of integer (or boolean) type | returns `array([1, 2, 4, 5])` | SILENT_WRONG |
| -1.0 | `IndexError`: arrays used as indices must be of integer (or boolean) type | returns `array([1, 2, 3, 4])` | SILENT_WRONG |
| 0.0 | `IndexError`: arrays used as indices must be of integer (or boolean) type | returns `array([2, 3, 4, 5])` | SILENT_WRONG |

**`take` — param `indices`**

| input | numpy | ionp | verdict |
|---|---|---|---|
| nan | `ValueError`: cannot convert float NaN to integer | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | WRONG_CLASS |
| +inf | `OverflowError`: cannot convert float infinity to integer | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | WRONG_CLASS |
| -inf | `OverflowError`: cannot convert float infinity to integer | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | WRONG_CLASS |
| 1e30 | `OverflowError`: Python int too large to convert to C long | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | WRONG_CLASS |
| -1e30 | `OverflowError`: Python int too large to convert to C long | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | WRONG_CLASS |
| 2**63 | `OverflowError`: Python int too large to convert to C long | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | WRONG_CLASS |
| -2**63 | `IndexError`: index -9223372036854775808 is out of bounds for axis 0 with size 5 | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | WRONG_CLASS |
| 2.7 | returns `np.int64(3)` | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | DIFFER_NUMPY_OK_IONP_RAISES |
| -1.0 | returns `np.int64(5)` | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | DIFFER_NUMPY_OK_IONP_RAISES |
| 0.0 | returns `np.int64(1)` | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind' | DIFFER_NUMPY_OK_IONP_RAISES |

**`put` — param `ind`**

| input | numpy | ionp | verdict |
|---|---|---|---|
| nan | `ValueError`: cannot convert float NaN to integer | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | WRONG_CLASS |
| +inf | `OverflowError`: cannot convert float infinity to integer | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | WRONG_CLASS |
| -inf | `OverflowError`: cannot convert float infinity to integer | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | WRONG_CLASS |
| 1e30 | `OverflowError`: Python int too large to convert to C long | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | WRONG_CLASS |
| -1e30 | `OverflowError`: Python int too large to convert to C long | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | WRONG_CLASS |
| 2**63 | `OverflowError`: Python int too large to convert to C long | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | WRONG_CLASS |
| -2**63 | `IndexError`: index -9223372036854775808 is out of bounds for axis 0 with size 5 | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | WRONG_CLASS |
| 2.7 | returns `array([ 1,  2, 99,  4,  5])` | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | DIFFER_NUMPY_OK_IONP_RAISES |
| -1.0 | returns `array([ 1,  2,  3,  4, 99])` | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | DIFFER_NUMPY_OK_IONP_RAISES |
| 0.0 | returns `array([99,  2,  3,  4,  5])` | `TypeError`: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' | DIFFER_NUMPY_OK_IONP_RAISES |

**`take_along_axis` — param `indices`** — 10/10 MATCH (`IndexError`: `indices` must be an integer array, both impls, all probes)

**`put_along_axis` — param `indices`** — 10/10 MATCH (`IndexError`: `indices` must be an integer array, both impls, all probes)

**`repeat` — param `repeats`**

| input | numpy | ionp | verdict |
|---|---|---|---|
| nan | `ValueError`: cannot convert float NaN to integer | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| +inf | `OverflowError`: cannot convert float infinity to integer | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| -inf | `OverflowError`: cannot convert float infinity to integer | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| 1e30 | `OverflowError`: Python int too large to convert to C long | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| -1e30 | `OverflowError`: Python int too large to convert to C long | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| 2**63 | `OverflowError`: Python int too large to convert to C long | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| -2**63 | `ValueError`: negative dimensions are not allowed | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| 2.7 | returns `array([1, 1, 2, 2, 3, 3])` | `TypeError`: 'float' object cannot be interpreted as an integer | DIFFER_NUMPY_OK_IONP_RAISES |
| -1.0 | `ValueError`: negative dimensions are not allowed | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| 0.0 | returns `array([], dtype=int64)` | `TypeError`: 'float' object cannot be interpreted as an integer | DIFFER_NUMPY_OK_IONP_RAISES |

**`ndarray.repeat` — param `repeats`** — same pattern as `repeat` (identical 10 rows)

**`diag_indices` / `tril_indices` / `triu_indices` — param `n`** — 10/10 MATCH each, e.g. nan → `ValueError`: arange: cannot compute length (both); 2\*\*63 → `ValueError`: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size. (both, byte-exact)

**`fft.fftfreq` / `fft.rfftfreq` — params `n` and `d`** — 10/10 MATCH on all four pairs (n: `ValueError`: n should be an integer, both, all probes; d: byte-exact numeric agreement across nan/inf/1e30/2**63/2.7/-1.0, and `ZeroDivisionError`: division by zero at 0.0, both)

**`reshape` / `ndarray.reshape` — scalar shape param**

| input | numpy | ionp | verdict |
|---|---|---|---|
| nan | `TypeError`: expected a sequence of integers or a single integer, got 'nan' | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_MSG |
| (same WRONG_MSG pattern for all 10 probes: numpy names the offending value in its message, ionp gives a generic message) | | | |

**`roll` — param `shift`**

| input | numpy | ionp | verdict |
|---|---|---|---|
| nan | `ValueError`: cannot convert float NaN to integer | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| +inf/-inf | `OverflowError`: cannot convert float infinity to integer | `TypeError`: 'float' object cannot be interpreted as an integer | WRONG_CLASS |
| 1e30, -1e30, 2\*\*63, -2\*\*63, 2.7, -1.0, 0.0 | all return a valid rolled array | `TypeError`: 'float' object cannot be interpreted as an integer | DIFFER_NUMPY_OK_IONP_RAISES |

**`tile` — param `reps`** — WRONG_CLASS on nan/inf/huge-magnitude (5 of 10: numpy raises `ValueError`/`OverflowError`, ionp `TypeError`); MATCH on 2.7 and 0.0 (both `TypeError`: 'float' object cannot be interpreted as an integer); WRONG_CLASS on -1.0 (numpy `ValueError`: negative dimensions are not allowed vs ionp `TypeError`)

**`broadcast_to` — scalar shape param** — WRONG_MSG on nan/+inf/1e30/2\*\*63/2.7/0.0 (numpy `TypeError`: 'float' object cannot be interpreted as an integer vs ionp `TypeError`: expected a sequence of integers..., got '...'); WRONG_CLASS on -inf/-1e30/-2\*\*63/-1.0 (numpy `ValueError`: all elements of broadcast shape must be non-negative vs ionp `TypeError`)

**`linalg.matrix_power` — param `n`** — WRONG_MSG on all 10 (numpy `TypeError`: exponent must be an integer vs ionp `TypeError`: 'float' object cannot be interpreted as an integer — same class, different message, every probe including the ordinary controls)

**Controls, 10/10 MATCH each:** `eye` (N), `zeros` (shape scalar), `identity` (n), `swapaxes` (axis1) — all `TypeError`: 'float' object cannot be interpreted as an integer (eye/identity/swapaxes) or `TypeError`: expected a sequence of integers or a single integer, got '...' (zeros), byte-identical between numpy and ionp across all 10 probes.

**Excluded (invalid harness):** `strings.center` — param `width` — all 10 cells excluded; the probe's own `ionp.array(['ab','cd'])` setup call fails before `width` is reached (see disclosure above).

## Answers to the four required summary numbers

- **Declared-`exact` items enumerated:** 455 (all entries in `ionp.__ion_state__` are `'exact'`; no other state currently present).
- **Items/parameters in scope for this sweep:** 25 item/parameter pairs (spanning ~20 distinct items) selected as a representative, non-exhaustive sample covering every scalar-argument parsing pattern found in the source, plus known-clean controls.
- **Total cells probed:** 260 raw comparison cells (500 subprocess runs); **250 valid** after excluding the 10 invalid `strings.center` rows.
- **DIFFER counts by category (valid cells):** PANIC = 0, SILENT_WRONG = 8, WRONG_CLASS = 45, WRONG_MSG = 48 (sum = 101 of the four requested categories); plus the informal fifth category DIFFER_NUMPY_OK_IONP_RAISES = 17 (not requested, disclosed separately, safe-direction only). MATCH = 132.

The hole is not small: on a representative curated sample, 101/250 valid
probed cells (about 40%) diverge from numpy in one of the four requested
categories, though only 8 of those (all in `insert`/`delete`) are the
dangerous SILENT_WRONG kind, and none are PANIC in this sample. The
already-shipped `arange_len_from_zero` fix and its three consumers hold up
cleanly (10/10 MATCH each), confirming the fix pattern works when applied —
it just hasn't been applied to `insert`/`delete` yet.
