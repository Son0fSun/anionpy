# ionp understands 15 dtype spellings. numpy accepts at least 65. Nine declarations are false.

**Date:** 2026-08-06
**Measured against:** numpy 2.5.1, installed `.so` at commit `e70514a`.
**Status:** **RESOLVED 2026-08-06 by `a151969`** — see "Resolution" at the bottom
before acting on anything above. Everything from here to that section describes
the state at `e70514a` and is preserved as the record of the finding, NOT as a
live claim about the current binary.
**Severity:** highest of the session. Silent, wide, and invisible to the entire suite.

## The finding

`dtype_name_to_dtype` (`ionp-py/src/lib.rs:2135`) is a flat match with **15
arms**, all canonical names (`"bool"`, `"bool_"`, `"int8"` … `"complex128"`).
Every other spelling numpy accepts falls through to
`TypeError: data type '{s}' not understood`.

Measured — 50 spellings tried, **50 rejected by ionp, all accepted by numpy**:

```
'?' 'b' 'B' 'h' 'H' 'i' 'I' 'l' 'L' 'q' 'Q' 'e' 'f' 'd' 'g' 'F' 'D' 'G'
'i1' 'i2' 'i4' 'i8' 'u1' 'u2' 'u4' 'u8' 'f2' 'f4' 'f8' 'c8' 'c16'
'<f4' '>f4' '=f8' '|b1'
'float' 'double' 'half' 'single' 'int' 'uint' 'intp' 'longlong'
'ubyte' 'short' 'csingle' 'cdouble' 'longdouble' 'str' 'bytes'
```

Three families: **single character codes**, **char+itemsize codes**,
**byte-order-prefixed codes**, plus **C-name aliases**.

```
>>> np.array([1.0], dtype='f4').dtype
dtype('float32')
>>> ionp.array([1.0], dtype='f4')
TypeError: data type 'f4' not understood
```

## Why the uniform 50/50 is a finding and not an instrument bug

The standing rule says a uniform rate across a cross-product is an instrument
bug. It was checked before this was believed. The instrument CAN report
success — canonical names, `np.dtype` objects, and numpy scalar classes all
resolve correctly:

```
'float32' -> float32   'int8' -> int8   'bool' -> bool   'complex128' -> complex128
np.dtype('f4')   -> float32      (dtype OBJECT path: works, and note it carries 'f4')
np.float32       -> float32      (class path: works)
```

Uniformity is admissible here because there is **one named shared mechanism**:
all 50 spellings are `str`, and every `str` goes through the one 15-arm table.
That is the only condition under which a uniform rate is not an instrument bug
— a single identified cause, not a coincidence.

Note the shape of the surviving paths: `np.dtype('f4')` works because the
*dtype object* self-describes as `'float32'` via `.name`. ionp resolves the
object correctly and the string incorrectly, for the same underlying dtype.

## The nine false declarations

Any function taking `dtype=` inherits the gap. Measured on the installed `.so`,
restricted to calls where **numpy accepts `'f4'` and ionp raises**:

| item | `__ion_state__` | suite verdict |
|---|---|---|
| `zeros` | **exact** | pass |
| `ones` | **exact** | pass |
| `empty` | **exact** | pass |
| `full` | **exact** | pass |
| `eye` | **exact** | pass |
| `identity` | **exact** | pass |
| `linspace` | **exact** | pass |
| `asarray` | **exact** | pass |
| `zeros_like` | **exact** | pass |
| `arange` | not declared | pass |
| `array` | not declared | pass |
| `can_cast` | not declared | fail |
| `promote_types` | not declared | fail |
| `result_type` | not declared | fail |

**ILLUSTRATIVE, NOT EXHAUSTIVE.** Fourteen entry points were probed because
they were convenient to script; the repo has ~40 `dtype=` call sites. The true
count is higher and has not been enumerated.

Per the standing rule already recorded in `_state/toplevel.py` — *a documented
gap inside a declared item is still a false declaration* — those nine are
false today. They pass their differential items because **the corpus spells
every dtype canonically.** Not one case anywhere passes `'f4'`.

## The part worth remembering

This gap was CREATED by a correct fix. `dtype_from_pyobj`'s doc comment records
that it used to end in `PyModule::import(py, "numpy")` + `np.dtype(obj).name`,
and that this was removed because *"a component that gets its answer from the
thing it is being compared against cannot fail the comparison."* That reasoning
is right and the removal was right.

But delegating to `np.dtype()` had also been silently supplying **numpy's
entire string grammar**. Removing the crutch narrowed the accepted input
surface by at least 50 spellings, the replacement table covered only the 15
canonical names, and the suite stayed green throughout because the corpus never
exercised the difference.

The lesson generalizes past dtypes: **when you remove a delegation to a
reference implementation, you inherit responsibility for everything it was
quietly doing for you — and your existing tests cannot tell you what that
was**, because they passed under the delegation. Removing a crutch demands an
input-surface audit, not just a green re-run.

## Scope note for whoever fixes this

Not every numpy code maps into ionp's 14 `DType`s. `S`/`U`/`V`/`O`/`M`/`m`
(bytes/str/void/object/datetime64/timedelta64) name dtypes ionp does not have
at all — those are the SEPARATE absent-dtype items (`isnat`, object dtype,
void/structured), and they must keep raising. Fixing this item means covering
the spellings that map onto dtypes ionp **already supports**, and nothing more.
Do not let this task quietly become the datetime64 task.

`'g'`/`'longdouble'` map to `float64` **on this platform only**. Encoding that
as a fixed alias would be wrong on a platform with true extended precision.
Whoever fixes it must decide deliberately and record the choice.

Byte-order prefixes (`<`, `>`, `=`, `|`) are accepted by numpy and ionp is
little-endian-only; `'>f4'` (big-endian) is NOT the same dtype as `'f4'` and
must not be silently aliased to it. numpy's own behaviour there needs measuring
before it is reproduced.

## Not checked

- The full ~40 `dtype=` call-site list.
- Structured/record spellings (`'i4,f8'`, `[('a','i4')]`), which numpy also accepts.
- Whether any currently-`absent` item is absent *because of* this gap rather
  than on its own merits.
- numpy versions other than 2.5.1.

---

## Resolution — 2026-08-06, commit `a151969`

`dtype_name_to_dtype` was widened from the 15-arm table to numpy's actual dtype
string grammar. Verified by me independently of the implementing agent's report,
against the freshly-built `.so`:

- **47 of 47 in-scope spellings now resolve correctly.** `still-broken in-scope
  spellings: 0 of 47`.
- **Zero leakage.** `S U V O M m str bytes object void datetime64 timedelta64
  >f4 i0 nonsense` all still raise `TypeError`. The scope note above was the
  stated main risk — the task did not quietly become the datetime64 task.
- **`'>f4'` deliberately still raises.** Measured justification:
  `np.dtype('>f4') == np.dtype('f4')` is `False`, byteorder `'>'` vs `'='`, and
  `tobytes()` differs. Silent aliasing would have corrupted values. Raising is
  the correct answer for a little-endian-only implementation.
- **`'g'`/`'longdouble'` alias to `float64`** — valid here because
  `np.dtype('g').itemsize == 8` on this platform, and recorded in the code as
  platform-dependent rather than universal.

Ledger after the fix: **1243 items / 30 failures**; `tools/coverage.py --tests`
→ **21.643% (669/3091)**, absent 2422, phantom/failing/untested 0. No suite
movement — as expected, since the corpus never spelled a dtype non-canonically,
which is the whole reason the gap survived.

**The nine declarations are now TRUE** (`zeros ones empty full eye identity
linspace asarray zeros_like`) — earned by the fix, not revoked by fiat.

### Still open after this resolution

Of the three "Not checked" items above, the **call-site list is now partly
closed** (2026-08-06, later the same day):

- **Static:** 66 dtype-accepting parameters across 8 files, 80 resolver call
  sites. Per-file counts do not match (creation 15/6, reductions 17/6, matmul
  5/2) — that is forwarding, not a second resolver: `sum`/`prod`/`nansum`/
  `cumsum`/`var` hand their `dtype` down to shared impls that resolve
  (`do_reduce_axis`, reductions.rs:921). `grep '"float64" =>'` across BOTH
  `ionp-py` and `ionp-core` matches **`lib.rs` only** — exactly one place in the
  workspace parses a dtype string.
- **Behavioural:** 19 entry points × 15 spellings = **285 ok / 0 fail**
  (`zeros ones empty full eye identity linspace arange array asarray zeros_like
  sum prod cumsum nansum mean var std cumprod`, spellings incl. `f4 f8 i4 u1 e
  d ? single double half intp <f4 =f8 |b1`). The probe was proven able to report
  RED first — miswiring numpy's side to `int16` gives 0 ok / 285 fail. Note that
  probing a nonsense spelling does **not** prove bite, because both sides reject
  it and agreement is the expected result.

**Still genuinely un-checked:** the other 47 dtype-accepting parameters
(`random`, `linalg`, `matmul`, `ndarray_attrs`) rest on the static argument
only, and static reachability is not behaviour. Structured/record spellings
(`'i4,f8'`, `[('a','i4')]`) are still unhandled, and no numpy version other than
2.5.1 has been tested. This section closes the *string-grammar* defect only. Do not read
it as a clean bill of health for dtype resolution generally — `finfo`'s
dtype-resolution and error-path divergences were re-measured after this fix and
**12 of 12 non-spelling cases still diverge**.
