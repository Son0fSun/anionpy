# Ticket #34 — polynomial CLASS dunders — 2026-08-08

**Commit measured against (starting point):** `311502c` (working tree
clean before this pass's edits; concurrent ticket #37 had uncommitted
edits to `ionp-py/src/lib.rs`, `tests/differential/harness.py`, and
`tests/differential/registry.py` throughout — none of those three files
were touched by this ticket).

## 1. Re-measured population: 162, confirmed

```
cd ionp && ./.venv/bin/python tools/coverage.py --tests /tmp/r34_base.json --list absent
  | grep -c '^polynomial\.\(Chebyshev\|Legendre\|Hermite\|HermiteE\|Laguerre\|Polynomial\)\.'
```
gives **162**, matching the brief's own re-measured figure exactly (36 +
28 + 25 + 25 + 25 + 23), broken out per class:

| class | absent |
|---|---|
| Chebyshev | 36 |
| Legendre | 28 |
| Hermite | 25 |
| HermiteE | 25 |
| Laguerre | 25 |
| Polynomial | 23 |

Baseline suite (`/tmp/r34_base.json`, `wrote` line confirmed, 29
failures by name) and ledger (`1377/3091 = 44.549%`, absent 1714) were
both re-run fresh this session, not taken on faith from the brief.

## 2. The 36-vs-23 spread: legitimate, not a manifest artifact

Every absent item for each class was named-diffed against Polynomial's.
The spread decomposes into exactly two buckets, both already correctly
reasoned about pre-ticket in the code (`chebyshev.py`'s class docstring,
`abc_poly_class_cases.py`'s module docstring):

- **7 reflected/mul-family items** (`__mul__`, `__pow__`, `__radd__`,
  `__rdivmod__`, `__rfloordiv__`, `__rmod__`, `__rmul__`, `__rsub__`,
  `__rtruediv__` — 9 names, but see §3, 6 of these were a real gap fixed
  this ticket) plus `fromroots`: `chebmul`/`chebpow`/`chebfromroots`
  route through the same unreproduced `np.convolve` z-series summation
  order already REVOKED at the module-function level (ticket #45/#48
  territory) — correctly not declared for Chebyshev; Polynomial's own
  `polymul`/`polypow`/`polyfromroots` are power-series-native and don't
  have this problem, so they ARE declared. Same asymmetry, independently,
  makes Legendre's absent count 28 (not 23): `legmul`/`legpow` diverge on
  complex128 (ticket #48, open, unidentified root cause).
- **`cast`/`convert` (2 items)**: NOT declared for ANY of the five new
  orthogonal-basis classes (Chebyshev/Legendre/Hermite/HermiteE/Laguerre)
  — both route through `_polybase.py::_compose_affine`, a naive
  Horner-loop substitution that is mathematically WRONG (not merely
  imprecise) for any basis but the power series. Verified directly
  pre-ticket: `Legendre([1.,2.,3.]).convert(kind=Legendre)` (an identity
  conversion) returns `[2. 2. 2.]` instead of `[1. 2. 3.]`. Polynomial IS
  the power-series basis, so its `cast`/`convert` are valid and declared.
- **`interpolate` (1 item)**: genuinely unique to Chebyshev in real numpy
  itself — `numpy.polynomial.Polynomial` has no `interpolate` method at
  all, so it isn't even a surface item to compare against for the other
  five classes.

9 + 2 + 1 = 12... plus `__mul__`/`__pow__` counted once each above = the
full 13-item spread (36 − 23) resolves to: 9 reflected/mul-family dunder
names + 2 (cast, convert) + 1 (fromroots) + 1 (interpolate) = 13. Every
one of these has a specific, pre-existing, evidence-backed reason. **Not
a manifest artifact.**

## 3. Is N really 1?

**Yes, for the class scaffolding — confirmed by reading
`anionpy/polynomial/_polybase.py`.** `ABCPolyBase` implements every
dunder (`__add__`, `__radd__`, `__mul__`, `__truediv__`, `__floordiv__`,
`__mod__`, `__divmod__`, `__pow__`, `__eq__`, `__ne__`, `__len__`,
`__iter__`, `__call__`, `__neg__`, `__pos__`, `__repr__`, `__str__`,
`__format__`, etc.) exactly once, generically, in terms of 12 abstract
static methods (`_add _sub _mul _div _pow _val _int _der _fit _line
_roots _fromroots`) plus 3 abstract properties, each concrete subclass
supplying only its own basis-specific 1-D kernel functions
(`chebadd`/`chebmul`/...). Each per-basis file (`chebyshev.py`, etc.)
then rebinds the *same function objects* from `vars(ABCPolyBase)` onto
its own `__dict__` (`tools/coverage.py`'s `resolve()` only checks
`attr in vars(cls)`, own-`__dict__`, so this rebind is required for the
ledger to credit the subclass, not a second implementation).

Per-class DECLARATION status differs — not because the dunder logic
differs, but because each basis's own already-shipped, already-measured
1-D kernel (`chebmul` vs `legmul` vs `hermul`, etc.) has its own,
independently-discovered defect or lack thereof. The architecture is one
shared implementation; the divergence is entirely at the leaf kernel
level, already documented per-kernel before this ticket even started.

## 4. What this ticket found and fixed: a real, if narrow, gap

Reading `abc_poly_class_cases.py`'s corpus-generation code (not just its
declared-state comments) surfaced something the brief's own reasoning
in §2 doesn't cover: **6 of Chebyshev's 9 "reflected/mul-family" absent
items were not actually blocked by anything — they were bundled under
the wrong gate.**

`_ClassBinding.build()` built and declared `__radd__`, `__rsub__`,
`__rtruediv__`, `__rfloordiv__`, `__rmod__`, `__rdivmod__` **only inside
the `if self.declare_mul:` block**, alongside `__mul__`/`__rmul__`/
`__pow__`/`fromroots`. For Chebyshev, `declare_mul=False` (correctly, for
the mul-family), which silently also skipped the six add/sub/div-based
reflected ops — even though `ABCPolyBase.__radd__`/`__rsub__` call
`self._add`/`self._sub` directly, and `__rtruediv__`/`__rfloordiv__`/
`__rmod__`/`__rdivmod__` all reduce to `self._div` (see
`_polybase.py:454-499`). None of `_add`/`_sub`/`_div`
(`chebadd`/`chebsub`/`chebdiv`) has any known divergence — the ONLY known
defect in this family is `_mul`/`_pow` (`chebmul`/`chebpow`, the
`np.convolve` issue). These six dunders were **already correctly
implemented** in `chebyshev.py` (rebound straight from `ABCPolyBase`
onto `Chebyshev.__dict__`, visible in that file's
`_CHEB_COVERAGE_REBIND_DUNDER` tuple, which DOES include `__radd__`
through `__rtruediv__`) — just never split out of the mul-gated corpus
block and declared.

Also found: `Chebyshev.interpolate` (the classmethod, distinct from the
already-exact module-level `chebinterpolate` function) was fully
implemented in `chebyshev.py` (a direct port of numpy's own
`Chebyshev.interpolate`: remap `func` through `mapdomain` onto
`cls.window`, delegate to `chebinterpolate`) but had **zero** corpus
coverage anywhere and was not declared.

### Implementation

No `.py` numeric-loop or Rust changes — both items were pure
declaration/corpus work over already-shipped code:

- `tests/differential/abc_poly_class_cases.py`: split the six
  add/sub/div-based reflected-op `ItemSpec` entries out of the
  `if self.declare_mul:` block into their own always-built section
  (kept `__rmul__`/`__pow__`/`fromroots` correctly gated, since those DO
  route through `_mul`).
- `tests/differential/chebyshev_cases.py`: added
  `cheb_class_interpolate_cases()` (4 cases: 2 test functions × mixed
  degrees × default/custom domains, deliberately exercising a non-default
  `domain` since the `mapdomain` remap is a no-op at
  `domain == window == [-1, 1]`) and a new
  `polynomial.Chebyshev.interpolate` `ItemSpec`.
- `anionpy/_state/polynomial.py`: declared all 7 items exact for
  `Chebyshev`, with inline justification comments.

### Out-of-corpus verification (NaN-aware)

`/private/tmp/probe34_dunders.py`, run from `/private/tmp` against the
installed package:

- **Reflected ops**: 5 coefficient sets (float/int/complex/trailing-zero/
  single) × 8 scalar operand types (positive/negative/zero/int/complex/
  `nan`/`inf`) × 6 ops = **240/240 checks passed**, using an
  `equal_nan=True`-based comparator on both real and imaginary parts
  (avoids the `nan != nan` trap this ticket's brief calls out).
  `NotImplemented` return parity and exception type+message parity were
  checked explicitly (both branches present in the probe), not just
  values.
- Additional targeted probe for invalid/edge operand types (`str`,
  `None`, `dict`, 2-coefficient list, 3-tuple) against all 6 ops: **0
  mismatches** once compared by coefficient array rather than by
  cross-class `__eq__`/`__ne__` (a first pass showed spurious
  "mismatches" that were purely `isinstance(other, self.__class__)`
  returning `False` across the numpy-class/anionpy-class boundary in
  `__eq__` — a comparison-script artifact, not a real divergence; both
  sides *display* correctly, they're just never `==` to each other by
  construction, exactly as intended).
- **`interpolate`**: 2 test functions × 4 degrees × 3 domains (default +
  2 custom) = **24/24 checks passed**, coefficients AND domain both
  bit-exact.

**One real, disclosed, non-blocking divergence found and NOT fixed**:
for a degree-0 Chebyshev series divided by `inf`, real numpy's
`chebdiv`/`__rfloordiv__`/`__rmod__`/`__rdivmod__` path emits
`RuntimeWarning: invalid value encountered in multiply`; anionpy's
equivalent path returns the **bit-identical value** (`inf`/`nan` as
appropriate — confirmed by direct comparison, this is part of the 240
passing checks) but does not emit that warning. The differential harness
(`tests/differential/harness.py`, not touched by this ticket, owned by
ticket #37 in-flight) does not check warnings at all currently, so this
does not block declaration on the harness's own terms, but it is a real,
measured runtime-behavior gap and is recorded here per this ticket's own
"check RuntimeWarning parity" instruction rather than silently declaring
around it.

## 5. Declared this ticket: 7 items (all Chebyshev)

```
polynomial.Chebyshev.__radd__
polynomial.Chebyshev.__rsub__
polynomial.Chebyshev.__rtruediv__
polynomial.Chebyshev.__rfloordiv__
polynomial.Chebyshev.__rmod__
polynomial.Chebyshev.__rdivmod__
polynomial.Chebyshev.interpolate
```

All 7 show `"verdict": "pass"` in the full differential run
(`/tmp/r34d_final.json`), not just the standalone probe.

## 6. Ledger — before / after

```
before: ./.venv/bin/python tools/coverage.py --tests /tmp/r34_base.json
        COVERAGE: 44.549%  (1377/3091)
        exact 1377 / untested 0 / phantom 0 / failing 0 / absent 1714

after:  ./.venv/bin/python tools/coverage.py --tests /tmp/r34d_final.json
        COVERAGE: 44.775%  (1384/3091)
        exact 1384 / untested 0 / phantom 0 / failing 0 / absent 1707
```

Delta: **+7 exact, -7 absent**, matching this ticket's 7 declarations
exactly. `phantom=0`/`failing=0` confirm every declaration is backed by
a passing differential test.

Six-class absent count: **162 → 155** (verified by direct re-count
against `--list absent`, not inferred from the ledger delta alone).

## 7. Suite delta — zero, by name, verified with a captured baseline JSON

Unlike the prior `TICKET-34-2026-08-08.md` (nd/grid/vander family),
which flagged its own suite-delta claim as "not independently re-verified
against a byte-identical baseline JSON" — this session captured
`/tmp/r34_base.json` (baseline, `wrote` line + mtime confirmed) BEFORE
any edit, and diffed it against `/tmp/r34d_final.json` (after all edits)
by failure-set NAME:

```
baseline fails: 29
final fails:    29
added:   []
removed: []
```

Exact byte-for-byte set match against the stated 29-failure baseline.

## 8. Guard-bite proof

No Rust code was touched this ticket (pure-Python declaration/corpus
work), so there is no Rust regression guard to sabotage-and-revert in
the usual sense. What WAS provable, and was proven: that the corpus
split in `abc_poly_class_cases.py` is load-bearing for the 6 reflected-op
declarations, not decorative.

Procedure: restored `abc_poly_class_cases.py` to its pre-ticket `HEAD`
content (`git show HEAD:...` from the parent repo root) while leaving the
7 new `_state/polynomial.py` declarations in place, re-ran the full
suite, and listed `--list untested`:

```
polynomial.Chebyshev.__radd__       no passing differential test
polynomial.Chebyshev.__rdivmod__    no passing differential test
polynomial.Chebyshev.__rfloordiv__  no passing differential test
polynomial.Chebyshev.__rmod__       no passing differential test
polynomial.Chebyshev.__rsub__       no passing differential test
polynomial.Chebyshev.__rtruediv__   no passing differential test
```

All 6 flip to `untested` exactly as expected when the backing corpus is
removed while the declaration stays — proving the declaration is not
free-floating. Restored `abc_poly_class_cases.py` to the edited version
afterward and verified byte-identical via `md5`/`diff` against the saved
copy before re-running the full suite one more time to confirm the final
state (§6, §7 numbers above are from that final, restored-clean run).

## 9. Explicitly declined / not attempted, with reasons

**138 items (23 × 6 classes) — generic Python/`object`/`abc.ABC`
mechanics, out of this ticket's target dunder list.** Named:
`__abstractmethods__`, `__delattr__`, `__dir__`, `__firstlineno__`,
`__format__`, `__ge__`, `__getattribute__`, `__gt__`, `__init_subclass__`,
`__le__`, `__lt__`, `__module__`, `__new__`, `__reduce__`,
`__reduce_ex__`, `__repr__`, `__setattr__`, `__sizeof__`, `__slots__`,
`__static_attributes__`, `__str__`, `__subclasshook__`, `__weakref__`.
Split into two sub-reasons:

- `__repr__`/`__str__`/`__format__` (18 of the 138) — genuinely IN this
  ticket's target dunder list, and were investigated specifically. They
  are already implemented (functional, don't raise) but were already,
  correctly, and pre-ticket documented as NOT bit-exact-able against real
  numpy: `_polybase.py`'s module docstring records that numpy's own
  `__repr__`/`__str__`/`__format__` depend on `dragon4_positional`/
  `dragon4_scientific`, numpy's own internal shortest-round-trip float
  formatter, which this project's "never call real numpy to produce a
  value" rule forbids reimplementing by delegation, and reimplementing
  from scratch (Grisu/Dragon4-class algorithm) is a project unto itself,
  well outside this ticket's scope. Confirmed this reasoning still holds
  by inspection this session — not re-litigated, cited.
- The other 120 (20 × 6) — never in this ticket's target dunder list.
  Investigated anyway (since the ticket's own effort-budget language
  doesn't forbid closing genuinely free wins) and found to be
  **fundamentally unmatchable by construction**, not merely unattempted:
  `__module__` differs because the packages ARE different
  (`anionpy.polynomial.chebyshev` vs `numpy.polynomial.chebyshev`);
  `__firstlineno__`/`__static_attributes__` are CPython 3.13+ per-class
  source-line/attribute metadata that differs by definition between two
  different source files; `__sizeof__` reports actual object memory
  layout, which differs between the two implementations by design;
  `__dir__` enumerates the attribute set, which differs because the two
  classes don't have identical internal attribute names; `__lt__`/
  `__le__`/`__gt__`/`__ge__` (default `object` behaviour, both classes
  raise `TypeError` identically in *type*) fail this project's stated
  "match message" bar because the default `TypeError` message embeds the
  fully-qualified class name/repr, which differs by module path for the
  same reason as `__module__`. This matches the precedent already set in
  `anionpy/_state/matrix.py` (`__module__`/`__static_attributes__` "remain
  NOT ATTEMPTED (Python-mechanical attributes, not measured this pass)")
  — extended here with the fuller per-item reasoning for why they are not
  simply deprioritized but structurally out of reach for a from-scratch
  reimplementation, not a shortfall of effort.

**10 `cast`/`convert` items** (2 per class × 5 orthogonal bases) —
genuinely blocked, per this ticket's brief and independently re-confirmed
in §2 above (`_compose_affine` wrong-value bug, valid only for the
power-series basis). Left untouched.

**6 `mul`/`pow`/`fromroots`-family items**
(`Chebyshev.__mul__`/`__rmul__`/`__pow__`/`fromroots`,
`Legendre.__mul__`/`__pow__`; `Legendre.fromroots` was already separately
REVOKED pre-ticket) — genuinely blocked by tickets #45 (`np.convolve`
summation order, unreproduced) and #48 (`legmul`/`legpow` complex128
divergence, unidentified root cause). Disclosed, not worked around, per
this ticket's explicit instruction. Left untouched.

**A claim I could not independently re-verify**: the prior session's
`abc_poly_class_cases.py` module docstring states `Legendre.__mul__`/
`__pow__`/`fromroots` measure 22244/25000, 9006/25000, 12242/25000
mismatches on complex128 against a specific seeded sweep
(`seed=0x5EEDC1A55`, `/private/tmp/sweep_class.py`). That sweep script no
longer exists on disk (private `/tmp` scratch, not committed) and was not
re-run this session — this ticket's own scope was the Chebyshev
reflected-op/`interpolate` gap, not re-auditing #48. Cited as an existing
finding, not independently reproduced this session. Anyone re-opening
`Legendre.__mul__`/`__pow__`/`fromroots` should re-run a fresh sweep
rather than trust this description.

## 10. Files touched

- `anionpy/_state/polynomial.py` — 7 new `"exact"` declarations
  (Chebyshev's 6 reflected ops + `interpolate`) with inline justification
  comments; corrected/expanded the pre-existing per-class scope comment
  block.
- `tests/differential/abc_poly_class_cases.py` — split the 6 add/sub/
  div-based reflected-op `ItemSpec` entries out from under the
  `declare_mul` gate into their own always-built section (all 5 classes
  now get them; only Chebyshev's declaration status actually changed,
  since the other 4 already had `declare_mul=True`).
- `tests/differential/chebyshev_cases.py` — added
  `cheb_class_interpolate_cases()` and the
  `polynomial.Chebyshev.interpolate` `ItemSpec`; added the
  `anionpy.polynomial as _anionpy_poly` import it needs.
- `docs/TICKET-34-DUNDERS-2026-08-08.md` — this document.
- `/private/tmp/probe34_dunders.py` — out-of-corpus probe (scratch, not
  committed).

No `ionp-py/src/`, `harness.py`, or `registry.py` (dtype/char/strings
surface, owned by ticket #37's in-flight edits) were touched, read-edited,
or staged.

## 11. Honest scope statement

This ticket lands **7 of 162** absent items. That is a small fraction of
the cluster's raw count, by design: the method in force here ("write
corpus first, implement, verify out of corpus, declare only what you
verified") found that the overwhelming majority of the 162 — 138 by
generic-Python-mechanics, 10 by a pre-existing shared-helper bug, 6 by
two pre-existing open tickets — are not implementation gaps at all, but
either structurally unmatchable (different module identity, different
memory layout) or already correctly disclosed-and-blocked by earlier,
independent findings. The remaining 7 were the only items where reading
the actual code (not the declared-state comments describing it) revealed
a real, fixable gap: work that was already done in `chebyshev.py`
one session prior, simply never wired into the corpus/declaration layer.
A partial, honest declaration of exactly those 7 — each with an
out-of-corpus NaN-aware probe, a captured-and-diffed suite baseline, and
a guard-bite proof — is the deliverable, not a broad claim against the
other 155.
