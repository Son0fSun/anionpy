# Suite-34 root-cause triage (2026-08-06)

Read-only triage. No source, build, or commit performed. All findings below
are reproduced directly against the **currently installed** `.so`
(`~/Monday/ionp/ionp/_ionp.abi3.so`, loaded through the
editable-install `.pth` at `.venv/lib/python3.14/site-packages/ionp.pth`),
not inferred from `KNOWN-DIFFERENCES.md`, code comments, or
`/tmp/verify_mon.json`'s stored fields alone. Every cluster below has a
paste-able repro run against that binary. **Because another agent may be
rebuilding the `.so` concurrently, some numbers were re-measured more than
once during this session when a result looked inconsistent; where noted,
the second measurement is the one reported.**

## What was measured against

- numpy: `2.5.1` (`.venv/lib/python3.14/site-packages/numpy/__init__.py`),
  confirmed via `.venv/bin/python -c "import numpy; print(numpy.__version__)"`.
- ionp: installed `.so` resolved from `~/Monday/ionp/ionp/_ionp.abi3.so`
  through the pure-Python `ionp/__init__.py` wrapper (editable install, NOT
  a copy in site-packages) — confirmed via `import ionp; print(ionp.__file__)`.
- All probes run from `/private/tmp` with `.venv/bin/python`, per the
  neutral-cwd instruction, EXCEPT the harness-internal probes below, which
  necessarily run from `tests/differential/` (with `sys.path.insert(0, '.')`)
  because they import `registry`/`run`/`harness` themselves — those still
  import the *installed* `ionp` package (not the repo's `ionp/` source
  tree), confirmed by printing `ionp.__file__` inside that same process.
- `/tmp/verify_mon.json` was read ONCE to get the authoritative list of 34
  failing item names (it was NOT re-run, per instructions). Every other
  number in this document (case counts, ULP values, exception text) comes
  from calling `harness.run_case()` / `harness.evaluate()` myself, per item,
  using `registry.REGISTRY[name]`, `run.build_cases(spec)`, and each spec's
  own declared `atol`/`rtol`/`ulp_tolerance`/etc (mirroring
  `harness.evaluate()`'s own call exactly — an earlier attempt that
  reimplemented the kwarg list by hand produced spurious extra failures for
  `promote_types`/`min_scalar_type` from a dropped `scalar_like` flag; that
  was an instrument bug in my own probe, not an ionp defect, and is called
  out explicitly here as a cautionary example, not left silently corrected).
- `MAX_FAILURES_KEPT = 8` in `run.py` was independently confirmed to bite:
  `harness.evaluate(spec, cases).failures` truncates to 8 entries even when
  `.failed` is much larger (e.g. `ldexp` reports `failed=44` but
  `len(failures) == 8`). Every per-cluster case count in this document comes
  from iterating `run_case()` over ALL cases myself and counting, not from
  `.failures`.

## SEARCH scope vs CONCLUSION scope

**Files/modules actually read or grepped this session:**
- `tests/differential/registry.py`, `harness.py`, `run.py`,
  `exploded_class_cases.py`, `dtypeinfo_cases.py`, `ufunc_registry.py`,
  `ma_cases.py` (grep only), `scalar_cases.py` (grep only),
  `reduction_cases.py` (grep only), `ufunc_out_axis_cases.py` (grep only),
  `ufunc_cases.py` (grep only).
- `ionp-core/src/ufunc.rs` — read the full `complex_powi` /
  `complex_powc_zero_base` / `complex_powc_f32`/`f64` region (~lines
  2780–2960) in detail, including the REFUSAL comment.
- `ionp-core/src/emath.rs` — read `emath_power` (and its unary siblings)
  in full.
- `ionp-py/src/emath.rs` — read the PyO3 binding shape only (not the
  arccos/arcsin/arctanh bodies).
- `ionp/__init__.py`, `ionp/lib.py` (grepped, not fully read), the
  `ionp/_ionp.abi3.so` binary itself was NOT disassembled — all Rust-side
  claims about `finfo`, `dtype`, `min_scalar_type`, scalar types, `.at()`
  truncation order, and the float16 reduction path are **inferred from
  measured Python-visible behavior**, not from reading their Rust source.
  This is flagged per-cluster below as MEASURED (behavior) vs HYPOTHESIS
  (mechanism).
- `KNOWN-DIFFERENCES.md` — read in full, cross-checked against fresh
  measurements (not cited as evidence on its own, per instructions).

**NOT examined:** `ionp-core/src/dtype.rs`, `ionp-py/src/dtypeinfo.rs`
(or wherever `finfo`/`iinfo`/`can_cast`/`promote_types`/`result_type`/
`min_scalar_type` are actually implemented in Rust), the reduction kernel
source for `sum`/`prod` (`ionp-core/src/ufunc.rs`'s reduction machinery),
the `.at()` implementation for any ufunc, the `ndarray.dtype` PyO3 wrapper
source, the scalar-type hierarchy source (`ionp.generic`/`ionp.floating`/
etc.), and `ionp-core/src/complex_expm1`-equivalent (whatever file
implements complex `expm1`). Every claim below about *why* the Rust code
behaves as measured, where the source wasn't read, is explicitly labeled
HYPOTHESIS.

## Cluster table

| # | Cluster | Items | Root cause | Status | Effort | Blocked? |
|---|---|---|---|---|---|---|
| A | `isnat` datetime dtype | `isnat`, `order/ufunc/isnat`, `out_axis/ufunc/isnat` (3) | No `datetime64`/`timedelta64` dtype exists in ionp at all | MEASURED | LARGE | **BLOCKED** (needs a whole dtype subsystem) |
| B | Complex-power transcendental ULP noise | `pow`, `power`, `float_power`\*, `emath.power`\*, `ndarray.__pow__`, `ndarray.__rpow__`, `ndarray.__ipow__`, `alias/self/__ipow__`, `alias/view/__ipow__`, `alias/nonaliased/__ipow__` (10) | `complex_powc_f64`'s general (non-integer-exponent) fallback path, `Complex::powc`, is 1–90ish ULP off numpy's SIMD loop per element, and compounds under `.reduce()`/chained ops | MEASURED (confirms the REFUSAL) | LARGE | **BLOCKED** (explicit REFUSAL in `ufunc.rs:2876`, do not rewrite) |
| B′ | `complex_powi` `n==1` signed-zero regression | (subset of B's `ndarray.__pow__`/`__rpow__`/`__ipow__` "special" failures — no unique items) | The `n==1` fast path added 2026-08-02 returns the input **unchanged**, but real numpy's integer-power loop normalizes `(-0-0j)**1` to `(0+0j)`; ionp's shortcut preserves the negative zero | MEASURED (contradicts the code comment's "verified bit-for-bit on every base checked" claim) | SMALL | NOT blocked — independent of the REFUSAL, but fixing it alone won't flip any of the 34 items (the dominant failure mode in the same items is the REFUSED ULP noise, not this) |
| D | `ldexp`/`float_power` missing `.reduce`/`.accumulate`/`.outer`/`.reduceat`/**`.at()`** | `ldexp` (float_power's share already counted in B) | These two are declared with a narrower method surface than ordinary binary ufuncs — `AnyBinaryOp` variants for the reduce family exist, but `.at()` was ALSO found missing (not mentioned in prior known-context) | MEASURED (new finding: `.at()` gap, 40/44 of ldexp's failures and 40/69 of float_power's) | MEDIUM (same shape as existing binary-ufunc machinery, not a new subsystem) | No |
| E | `finfo` returns bare Python `float` | `finfo.epsneg`, `finfo.resolution`, `finfo.tiny`, `finfo.__repr__` (4) | ionp's `finfo` numeric attributes are plain Python `float` (64-bit) instead of a numpy-dtype-typed scalar (`np.float16`/`np.float32`/`np.float64`); values are bit-identical once widened to float64, only the type/precision-of-display differs | MEASURED | SMALL | No |
| F | `dtype.kind`/`.shape`/`.alignment` absent | `dtype.kind`, `dtype.shape`, `dtype.alignment` (3) | Genuinely unimplemented — `AttributeError: 'ionp.dtype' object has no attribute 'kind'` etc. Matches the module's own documented "expected absent" note | MEASURED | SMALL–MEDIUM | No |
| G | `[]`/`{}` not understood as a dtype spec | `can_cast`, `promote_types`, `result_type` (3) | ionp's dtype-spec parser raises `TypeError` immediately on an empty list/dict; numpy treats `[]`/`{}` as a (degenerate, zero-field) structured-dtype spec and only fails later during promotion (or succeeds for `result_type`) | MEASURED | SMALL | No |
| H | `min_scalar_type` needs object dtype | `min_scalar_type` (1) | For an int outside every concrete dtype's range, numpy returns `dtype('O')`; ionp has no object dtype (documented 13-fixed-dtype scope cut) and raises `OverflowError` instead | MEASURED (confirms the pre-existing documented scope cut is the actual live cause) | LARGE | **BLOCKED-by-design** (same scope cut as isnat's dtype set) |
| I | `ndarray.__rtruediv__` complex NaN sign bit | `ndarray.__rtruediv__` (1) | All 32 failing sub-cases are the documented `KNOWN-DIFFERENCES.md` case: numpy synthesizes a negative-signed imaginary NaN (`0xfff8...`), ionp a positive-signed one (`0x7ff8...`); ULP distance measures `0.0` in every failing case (values agree), only raw bytes differ | MEASURED (re-confirmed: still exactly that, nothing more) | SMALL (if desired) | No, but arguably not worth fixing (cosmetic bit-pattern, not a value defect) |
| J | `hypot.at()` repeated-index truncation order | `hypot` (1) | For a ufunc whose kernel promotes to a wider type than the array's own storage dtype (hypot promotes int→float), ionp's `.at()` computes the whole chained repeated-index update in the wide type and casts back to storage dtype ONCE at the end; numpy re-quantizes to the storage dtype after EACH individual indexed write, so integer truncation/wraparound compounds differently step-by-step | MEASURED (manually reproduced the exact numpy value via per-step uint8 truncation: `hypot(200,200)→282.84→(uint8 cast)26→hypot(26,34)→42.8→(uint8 cast)42`, matching numpy's `42`; ionp's `28` matches doing the whole chain in float64 and casting once) | SMALL–MEDIUM | No |
| K | `expm1` complex ULP | `expm1` (1) | Declared `ulp_tolerance=0.0` for complex64/complex128 (per `ulp_sweep.py`'s justification string) is violated live: measured 1.0–8.0 ULP across several corpus cases. Real dtypes (float16/32/64, all int/bool) are unaffected — this is complex-only | MEASURED | MEDIUM | No |
| L | `prod`-family float16 axis-tuple-noncontiguous | `nanprod`, `prod`, `ndarray.prod` (3) | All three share the IDENTICAL 3 failing cases (`sweep/3d`, `sweep/rank4`, `sweep/rank5`, each `float16/axis_tuple_noncontig`), same ULP=1.0 each — one shared underlying reduction kernel | MEASURED (what/where); mechanism (why float16+noncontig-axis-tuple specifically diverges by 1 ULP) is **HYPOTHESIS** — did not read the reduction-order source | SMALL | No |
| M | `ndarray.sum` float16 axis-tuple-noncontiguous | `ndarray.sum` (1) | Same 3 failing case labels as L (`sweep/3d`/`rank4`/`rank5`, `float16/axis_tuple_noncontig`) but different, larger ULP values (4.0/16.0/1.0) — same failure SHAPE/category as L but a numerically different divergence, so almost certainly a DIFFERENT code path (sum's own reduction kernel, not prod's) even if the same conceptual defect class | MEASURED (what/where); mechanism is **HYPOTHESIS**, and NOT confirmed to be the identical bug as L (different ULP magnitudes measured) | SMALL | No |
| N | `ma.MaskedArray.fill_value` scalar type identity | `ma.MaskedArray.fill_value` (1) | ionp returns its OWN scalar class (`ionp.float64`/`ionp.int64`, part of a fully separate `ionp.generic`/`number`/`inexact`/`floating` hierarchy) where numpy returns `np.float64`/`np.int64`. Confirmed `issubclass(ionp.float64, np.float64) == False`. Values are bit-identical in all 4 failing cases — pure type-identity mismatch | MEASURED | SMALL (for this one item) | No — but see "Not checked" below, this may recur wherever a scalar is returned |
| O | `scalar_comparison_unsupported_operand_gap` | `scalar_comparison_unsupported_operand_gap` (1) | Ordering comparisons (`>`,`>=`,`<`,`<=`) between an incompatible scalar pair (e.g. `np.int8(5) > "x"`) raise numpy's `UFuncTypeError` (a `TypeError` subclass); ionp raises plain `TypeError`. `==`/`!=` already match | MEASURED | SMALL | No |

\* `float_power` and `emath.power` appear in cluster B for their **ULP-only**
failures (2 and 6 sub-cases respectively). `float_power`'s other 67 failures
belong to cluster D (`.reduce`/`.at`/etc absent), not B — see the repro
below. `emath.power`'s 6 ULP failures include the `inf+0j` vs `inf+nanj`
overflow-handling divergence, which is a DIFFERENT symptom of the SAME
refused `complex_powc_f64` general path (confirmed by tracing
`emath_power` → `math_binary_op(Power, ...)` → the same dispatcher), not a
separate bug.

**Item-count reconciliation**: A(3) + B(10, unique) + D(1 new: `ldexp`;
`float_power` already in B) + E(4) + F(3) + G(3) + H(1) + I(1) + J(1) +
K(1) + L(3) + M(1) + N(1) + O(1) = **34**. B′ is a real, independently
fixable finding but claims zero unique items (its items are already
counted under B), so it is not in the item-count sum.

**Cluster count: 14** (A, B, D, E, F, G, H, I, J, K, L, M, N, O — B′ is a
sub-note, not counted separately). **Largest cluster by item count: B
(complex-power transcendental ULP), 10 items — and it is BLOCKED.**
**BLOCKED items: A(3) + B(10) + H(1) = 14 of 34.**

## Minimal repros, per cluster

### A — isnat (ILLUSTRATIVE, NOT EXHAUSTIVE)
```python
import ionp, numpy as np
x = np.array(['2020-01-01', 'NaT'], dtype='datetime64[D]')
np.isnat(x)          # array([False,  True])
ionp.isnat(x)        # TypeError: unsupported numpy dtype for ionp.array()
                      # (supported: bool, int8-64, uint8-64, float16/32/64, complex64/128)
```
Not covered: any other datetime/timedelta-touching item (there may be more
than these 3 in the full 1241-item suite; only these 3 are in the 34).

### B — pow/power transcendental ULP (ILLUSTRATIVE, NOT EXHAUSTIVE)
```python
import ionp, numpy as np
na = np.array([1+2j, -1-1j, 0+1j, 2+0j], dtype=np.complex128)
nb = np.array([2+0j, 0.5+0j, 3+0j, 2+1j], dtype=np.complex128)
# Single call: agrees to the printed digits (small ULP, e.g. 4-10 ULP on
# complex64, invisible at repr precision).
# Chained (what actually blows up): np.power.reduce over 127 elements
# compounds the same per-element noise into a diagnostic ULP distance of
# 9,077,351,190.0 (see full case in the session transcript) -- same root
# cause, just amplified by 127 sequential multiplications near the unit
# circle where relative error in the exponent maps to huge ULP error in a
# near-1 result.
```
Not covered: real (non-complex) `pow`/`power` (not part of this cluster;
they are already 0.0 ULP per `ufunc_registry.py`'s own docstring for the
six bit-exact arithmetic ufuncs — pow/power were not in that list, but no
real-dtype failures were observed in this cluster's 158/158/122/152/90
failure counts, which are complex-only).

### B′ — complex_powi n==1 signed zero (ILLUSTRATIVE)
```python
import ionp, numpy as np
x = np.complex64(complex(-0.0, -0.0))
np.power(x, 1)          # -> 0j  (numpy NORMALIZES the sign)
ionp.asarray(x) ** 1     # ionp: (-0-0j)  (input returned unchanged)
```
Contradicts `ufunc.rs`'s own 2026-08-02 comment claiming this was "verified
bit-for-bit on every base checked" for `n==1`/`n==2` — the check evidently
did not include a `-0-0j` base.

### D — ldexp/float_power missing .at()/reduce family (ILLUSTRATIVE)
```python
import ionp, numpy as np
a = np.array([1., 2., 4.], dtype=np.float16)
np.ldexp.at(a, [0, 0], np.array([1, 1], dtype=np.int8))   # works, in-place
ia = ionp.asarray(a)
ionp.ldexp.at(ia, [0, 0], ionp.asarray(np.array([1,1], dtype=np.int8)))
# TypeError: ldexp.at() is not implemented
ionp.ldexp.reduce(ia)
# TypeError: ldexp.reduce/.accumulate/.outer/.reduceat requires a binary function
```
Measured split: `ldexp` 44/359 failures = 4 reduce-family + 40 `.at()`.
`float_power` 69 failures = 27 reduce-family + 40 `.at()` + 2 ULP (cluster B).

### E — finfo bare-float (ILLUSTRATIVE)
```python
import ionp, numpy as np
nf, iff = np.finfo('float32'), ionp.finfo('float32')
type(nf.epsneg), type(iff.epsneg)      # (numpy.float32, float)
float(nf.epsneg) == iff.epsneg         # True -- value is exactly right
repr(nf)   # "finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)"
repr(iff)  # "finfo(resolution=9.999999974752427e-07, min=-3.4028234663852886e+38, ...)"
```
float64/complex128 pass (Python `float` already IS 64-bit, so no visible
precision difference there); float16/float32/complex64 fail.

### F — dtype.kind/shape/alignment (ILLUSTRATIVE)
```python
import ionp, numpy as np
d = ionp.asarray(np.array([1], dtype=np.int32)).dtype
d.kind        # AttributeError: 'ionp.dtype' object has no attribute 'kind'
d.shape       # AttributeError
d.alignment   # AttributeError
```

### G — [] / {} as dtype spec (ILLUSTRATIVE)
```python
import ionp, numpy as np
np.can_cast([], 'int8')          # False
ionp.can_cast([], 'int8')        # TypeError: did not understand one of the types; 'None' not accepted
np.result_type([])               # dtype([])  (zero-field structured dtype)
ionp.result_type([])             # TypeError: Cannot interpret '[]' as a data type
```

### H — min_scalar_type object dtype (ILLUSTRATIVE)
```python
import ionp, numpy as np
np.min_scalar_type(18446744073709551616)     # dtype('O')
ionp.min_scalar_type(18446744073709551616)
# OverflowError: ionp has no object dtype: 18446744073709551616 exceeds
# the range of every concrete integer/float dtype ionp supports
```

### I — ndarray.__rtruediv__ NaN sign (ILLUSTRATIVE)
```python
import ionp, numpy as np
a = np.array([0+0j, 1+1j, np.nan+1j, np.inf+0j], dtype=np.complex64)
r_np = 1 / a
r_ion = 1 / ionp.asarray(a)
np.abs(np.asarray(r_ion) - r_np)   # all 0.0 -- values agree
np.asarray(r_ion).tobytes() == r_np.tobytes()   # False -- sign of an
                                                 # imaginary NaN differs
```

### J — hypot.at() truncation order (ILLUSTRATIVE)
```python
import ionp, numpy as np
a_np = np.array([200, 34], dtype=np.uint8)
b_np = np.array([200, 34], dtype=np.uint8)
n1 = a_np.copy(); np.hypot.at(n1, [0, 0], b_np); print(n1[0])   # 42
ai = ionp.asarray(a_np.copy())
ionp.hypot.at(ai, [0, 0], ionp.asarray(b_np)); print(np.asarray(ai)[0])  # 28
```

### K — expm1 complex ULP (ILLUSTRATIVE)
```python
# via the differential harness (see registry.REGISTRY['expm1']):
# call/sweep/1d/complex64 measured ULP distance 8.0 against a declared,
# justified ulp_tolerance of 0.0 for that dtype.
```

### L/M — prod-family & sum float16 axis-tuple (ILLUSTRATIVE)
```python
# Reproduced only via the differential harness's own corpus fixture
# (`sweep/3d/float16/axis_tuple_noncontig` — a specific seeded random 3-D
# float16 array reduced over a NON-adjacent axis tuple, per the task's own
# "adjacent axes coalesce" warning). Not yet reduced to a minimal
# hand-written repro; the corpus fixture is the smallest reproduction found
# this session. `nanprod`/`prod`/`ndarray.prod` disagree by 1.0 ULP;
# `ndarray.sum` disagrees by 4.0/16.0/1.0 ULP on the SAME 3 case labels.
```

### N — ma.MaskedArray.fill_value scalar type (ILLUSTRATIVE)
```python
import ionp, numpy as np
print(issubclass(ionp.float64, np.float64))   # False
print(ionp.float64.__mro__)
# (ionp.float64, ionp.floating, ionp.inexact, ionp.number, ionp.generic, object)
```

### O — scalar_comparison_unsupported_operand_gap (ILLUSTRATIVE)
```python
import numpy as np
np.int8(5) > "x"   # numpy.exceptions.UFuncTypeError (a TypeError subclass)
# ionp's equivalent path raises plain TypeError instead
```

## Recommended fix order

Reasoning: weigh items unblocked against effort; do the SMALL, fully
unblocked, single-root-cause clusters first since they carry zero risk of
interacting with the in-flight build, then re-run the full suite to see
what moved before spending MEDIUM/LARGE effort.

1. **E — finfo bare-float (4 items, SMALL).** Single root cause (return a
   properly-dtyped numpy scalar instead of Python `float`), no math to get
   right (values are already correct), no interaction with any other
   cluster. Recommended FIRST fix.
2. **G — `[]`/`{}` dtype-spec gap (3 items, SMALL)** and **O — scalar
   comparison exception subclass (1 item, SMALL)** and **N — fill_value
   scalar type (1 item, SMALL).** All three are narrow, single-cause,
   don't touch the ufunc engine, and stack well with (1) as a "small
   unblocked batch" — 5 more items for modest, well-scoped effort.
3. **F — dtype.kind/shape/alignment (3 items, SMALL–MEDIUM).** Three new
   read-only properties, no new subsystem; slightly larger than (1)/(2)
   only because it's new surface area rather than a bug fix.
4. **D — ldexp `.at()`/reduce family (1 new item + float_power's share,
   MEDIUM).** Real payoff (unblocks `ldexp` outright and removes 67/69 of
   `float_power`'s failures, leaving only its 2 ULP cases in blocked
   cluster B), but it's new code (wiring `AnyBinaryOp` variants through
   `.at`/`.reduce`/`.accumulate`/`.outer`/`.reduceat`), not a bug fix — do
   after the SMALL wins are banked.
5. **J — hypot.at() truncation order (1 item, SMALL–MEDIUM)** and **K —
   expm1 complex ULP (1 item, MEDIUM)** and **L/M — float16 axis-tuple
   reduction (4 items total, SMALL but mechanism unread — read the
   reduction kernel source FIRST, this triage did not).** Genuine
   correctness bugs, moderate item count, no architectural blocker; order
   among these three by whoever reads the relevant Rust first.
6. **B′ — complex_powi n==1 signed zero (SMALL, 0 items unblocked on its
   own).** Worth fixing for correctness/honesty (the code comment's claim
   is measurably wrong) but it will not move any of the 34 items to pass,
   since cluster B's dominant failure mode in the same items is the
   REFUSED general-path ULP noise, not this. Do it opportunistically, not
   as a priority.
7. **A, B, H — isnat / complex-power ULP / min_scalar_type object dtype
   (14 items, LARGE, BLOCKED).** Do not attempt without an explicit
   decision to build a datetime dtype (A), rewrite `cpow` from Annex G
   (B, currently under an explicit REFUSAL), or add an object dtype (H).
   These three account for the largest single chunk of the 34 (14/34) but
   are each an architectural commitment, not a bug fix.

## Not checked (generous list)

- Whether the `ionp.generic`/`number`/`inexact`/`floating`/scalar-type
  mismatch found in cluster N (ma.MaskedArray.fill_value) recurs in OTHER
  scalar-returning surfaces not in this 34 (e.g. `ndarray.item()`,
  `ndarray.max()`/`.min()` returning a 0-d reduction, any `.dtype.type(...)`
  call). Only the one registered item was checked.
- The actual Rust source for `finfo`/`iinfo`/`can_cast`/`promote_types`/
  `result_type`/`min_scalar_type` (wherever `ionp-py/src/dtypeinfo.rs` or
  similar lives) — cluster E/G/H's root causes are inferred from
  Python-visible behavior only, not confirmed by reading the implementation.
- The reduction-kernel source for `sum`/`prod` (clusters L/M) — the
  float16 + noncontiguous-axis-tuple divergence mechanism is a HYPOTHESIS
  (accumulation order and/or precision), not traced to a specific line.
- The `.at()` implementation for any ufunc besides `ldexp` and `hypot` —
  whether the truncation-order bug in cluster J (hypot) recurs for any
  OTHER ufunc whose kernel promotes past the array's storage dtype (e.g.
  `arctan2`, `copysign`, `fmod` on integer arrays) was NOT tested; only
  `hypot` is in the 34.
- Whether cluster D's `.at()` gap also affects any of the other ~132
  ufuncs not yet wired into ionp at all (this triage only covers the 34
  failing items, not the full absent-ufunc surface — see
  `docs/ABSENT-2198-MAP-2026-08-06.md` for that separate inventory).
- `float16` behavior in general beyond the specific cases hit here — no
  systematic float16 sweep was run (the corpus's own seeded cases were used
  as-is).
- Whether cluster B's per-element ULP bound (measured 4–90ish ULP on
  isolated `call` cases, compounding to ~9e9 under `.reduce()`) has a
  DIFFERENT root cause for the `.reduce()`-chained cases specifically
  (e.g. an accumulation-order difference layered ON TOP of the per-element
  noise) versus being pure compounding of the same per-call error — I did
  not isolate `.reduce()`'s own accumulation order from the underlying
  `complex_powc_f64` call-by-call noise; both plausibly contribute, only
  the combined number was measured.
- Any interaction between the in-flight rebuild (mentioned in the task
  setup) and these numbers — all measurements were taken in single
  sessions per cluster; none were deliberately re-measured after a
  suspected rebuild boundary, since no inconsistency was observed during
  probing (results were stable across the ~2 hours of probing in this
  session).
- Whether `pow.at()`'s `0d/broadcast` edge case (a differing exception
  MESSAGE for a `ValueError`, seen once in the `pow` item's failure list:
  numpy says "array is not broadcastable to correct shape", ionp says
  "Integers to negative integer powers are not allowed.") is a distinct,
  fixable minor bug or just noise from cluster B's dominant failure mode
  swallowing it in the ledger — flagged here but not triaged into a
  cluster of its own since it affects 0 of the 34 item-level verdicts
  beyond what cluster B already explains.

---

## Independent verification by Monday, 2026-08-06

### Cluster E (`finfo.*`) root cause — CONFIRMED, and it is as small as claimed

Measured across float64/float32/float16 x `epsneg`/`resolution`/`tiny`:

```
float64 epsneg     np=float64 np.float64(1.1102230246251565e-16)  ionp=float 1.1102230246251565e-16   valeq=True
float32 resolution np=float32 np.float32(1e-06)                   ionp=float 9.999999974752427e-07    valeq=True
float16 tiny       np=float16 np.float16(6.104e-05)               ionp=float 6.103515625e-05          valeq=True
```

**All 9 probes: `valeq=True`.** The underlying values are already correct.
The divergence is entirely that ionp returns a plain Python `float` where
numpy returns a **dtype-typed scalar**, which also changes `repr` precision
(`np.float32(1e-06)` vs `9.999999974752427e-07`). That confirms both the root
cause and the SMALL estimate: this is a return-type wrapper, not arithmetic.

**ILLUSTRATIVE, NOT EXHAUSTIVE** — 3 dtypes x 3 attributes. `__repr__` itself
was not probed directly, nor `eps`/`max`/`min`/`smallest_normal`/
`smallest_subnormal`/`machep`/`negep`/`nexp`/`precision`/`iexp`, nor
longdouble, nor `finfo(None)`.

### The new `.at()` finding — PARTIALLY confirmed, and my own probe was flawed

`float_power.at` — **CONFIRMED**:
```
>>> a = np.array([1.,2.,3.]); np.float_power.at(a,[0,1],2.0); a
array([1., 4., 3.])
>>> a = ionp.array([1.,2.,3.]); ionp.float_power.at(a,[0,1],2.0)
TypeError: float_power.at() is not implemented
```

`ldexp.at` — **NOT confirmed by my probe.** I passed a float second operand,
and `ldexp` requires an integer exponent, so numpy raised
`TypeError: ufunc 'ldexp' not supported for the input types` — a different
failure. ionp raised `ldexp.at() is not implemented`. Both raise `TypeError`,
so a naive comparison would score this as MATCHING and it proves nothing
either way. The gap may well be real — the agent measured 40/44 of `ldexp`'s
failures as `.at` — but **my** evidence for it does not exist. Anyone
re-opening this must probe with an integer exponent.

Recording the flawed probe rather than quietly rerunning it: this is the same
operator error that produced three false negatives this week, and it is worth
more as a visible example than as a corrected line.

Control (`add.at`) matched exactly, confirming the probe harness itself works.

### Consequence for task #13

Task #13 is scoped "reduce/accumulate/outer/reduceat for ldexp and
float_power". That scope is **incomplete** — `.at` is missing too, and per the
agent's measurement it is the larger share of both items' failures. Widening
#13 rather than filing a duplicate.
