# Denominator Audit — `tools/numpy_surface.json`

**Scope:** measurement only. No file other than this one was edited. Every
number below was produced by a script run against `.venv/bin/python` (numpy
2.5.1, matching the pinned `numpy_version` in `tools/numpy_surface.json`) —
none is reasoned about or estimated unless explicitly labeled `ESTIMATE`.

Live coverage read at the start of this audit (`tests/differential/run.py
--out /tmp/x.json` then `tools/coverage.py --tests /tmp/x.json`):

```
surface items   : 1180
ion              0
exact          612
COVERAGE: 51.864%   (612/1180)
```

`612` (the numerator) is held fixed throughout this document. Every
alternative denominator below is `612 / <denominator>`. This is explicitly an
illustration of magnitude, not a rebuilt ledger — see the caveat in §5.

---

## 1. How the current 1180 is composed

| source            | count | notes |
|---|---:|---|
| `names` (flat, one key = one item) | 1016 | walks `np` + 12 submodules, one dict entry per public non-underscore attribute |
| `ndarray` (exploded methods/attrs) | 70 | `dir(np.ndarray)`, non-dunder |
| `ndarray_dunder` (exploded dunders) | 94 | `dir(np.ndarray)`, dunder only |
| **TOTAL** | **1180** | matches `tools/coverage.py`'s reported `surface items` |

`names` broken down by kind:

| kind | count |
|---|---:|
| func | 735 |
| ufunc | 134 |
| class | 110 |
| const | 37 |

`names` broken down by family (submodule prefix):

| family | count |
|---|---:|
| `(top-level np)` | 477 |
| `ma.*` | 225 |
| `char.*` | 65 |
| `random.*` | 63 |
| `testing.*` | 50 |
| `strings.*` | 46 |
| `linalg.*` | 33 |
| `fft.*` | 19 |
| `emath.*` | 9 |
| `rec.*` | 9 |
| `polynomial.*` | 8 |
| `ctypeslib.*` | 6 |
| `lib.*` | 6 |

Of the 110 `class` entries, **exactly one** (`ndarray`) additionally gets its
member surface exploded into the two extra top-level keys `ndarray` /
`ndarray_dunder`. The other 109 — including `ma.MaskedArray`, `random.Generator`,
`dtype`, `finfo`, every scalar type, every polynomial class — are each a
single opaque dict entry, worth exactly 1 regardless of how many methods they
carry. **`ndarray` is also separately present in `names["ndarray"] = "class"`**,
so it is actually counted twice over: once as an opaque class entry (matching
everyone else) and once fully exploded (164 more). No other class receives
this treatment.

`snapshot_surface.py`'s notion of "public" is `dir(mod)` filtered to names not
starting with `_` (`surface()`, lines 26–44; same filter reused for `ndarray`
at lines 65–69). It does **not** consult `__all__`. Measured against numpy's
own `__all__`:

- top level: `len(np.__all__) == 496`, `len(dir(np) non-underscore) == 494` —
  close, 2-item gap is `__array_namespace_info__`/`__version__` present in
  `__all__` but absent from `dir()` (not a manifest bug, a numpy quirk).
- submodules diverge more: e.g. `np.char` has 53 items in `__all__` but 68 in
  `dir()` — a 15-item gap, investigated in §2/Finding C below.

So the manifest's rule ("`dir()` minus underscore") is applied identically
everywhere it's applied — the asymmetry is not in *which* names get admitted,
it's in whether an admitted **class** gets exploded into its members or not.

---

## 2. Every asymmetry found, with measured counts

### Finding A — the reported one, confirmed and generalized: ndarray is exploded, all other classes are not

Verified live: `ma.MaskedArray` has **94** public (non-dunder) attributes and
**97** dunders (191 total) in real numpy 2.5.1 — this matches the number
already independently measured and recorded in `ionp/_state/ma.py` (which
revoked `"ma.MaskedArray": "exact"` today, 2026-08-03, for the same reason).
It is worth **1** denominator item. `ndarray` — 70 public / 94 dunder — is
worth **165**.

The same asymmetry, generalized across **all 110 classes** in the manifest
(script: `dir(cls)` per class, live numpy):

| class | real public | real dunder | real total | manifest count |
|---|---:|---:|---:|---:|
| `ndarray` | 70 | 94 | 164 | **165** (double-counted, see above) |
| `ma.MaskedArray` | 94 | 97 | 191 | 1 |
| `ma.mvoid` | 94 | 98 | 192 | 1 |
| `matrix` | 80 | 96 | 176 | 1 |
| `memmap` / `rec.recarray` / `recarray` | 71 | 96 | 167 | 1 each |
| `char.chararray` | 108 | 96 | 204 | 1 |
| `str_` / `char.str_` | 112 | 70 | 182 | 1 each |
| `bytes_` / `char.bytes_` | 107 | 71 | 178 | 1 each |
| `random.Generator` | 45 | 23 | 68 | 1 |
| `random.RandomState` | 49 | 24 | 73 | 1 |
| `dtype` | 23 | 29 | 52 | 1 |
| `finfo` | 7 | 26 | 33 | 1 |
| `iinfo` | 2 | 26 | 28 | 1 |
| `poly1d` | 11 | 42 | 53 | 1 |
| `polynomial.Polynomial`/`Chebyshev`/`Hermite`/`HermiteE`/`Laguerre`/`Legendre` | 25–26 | 50 | 75–76 | 1 each |
| every scalar type (`float64`, `int64`, `complex128`, `bool_`, ...) | 65–112 | 66–74 | ~130–140 | 1 each |
| `ufunc` | 13 | 25 | 38 | 1 |
| `nditer` | 25 | 31 | 56 | 1 |

**All 109 non-`ndarray` classes are asymmetric.** Every one is either the
same order of magnitude as `ndarray` (many are *larger* — `char.chararray`
at 204 real members outweighs `ndarray`'s 164) or represents a genuinely
distinct type family (random bit generators, polynomial bases, dtype
introspection) that gets zero credit for its internal surface.

Sum across all 110 classes, raw (no dedup, no noise removal): **11,740** real
members vs **274** manifest-counted entries (110 flat + ndarray's extra 164).

### Finding B — reverse direction: 108 extra denominator entries from genuine re-export duplication

`id()`-identity check across every resolved item in `names` (with a filter to
exclude two false-positive groups caused by CPython's `True`/`False`
singleton caching — `little_endian`/`testing.HAS_LAPACK64`/... and
`testing.BLAS_SUPPORTS_FPE`/... — those are independently-defined booleans
that happen to share a cached value, not re-exports, confirmed by checking
`type(obj) is bool`):

**99 real duplicate-object groups, 108 extra denominator entries.** The same
underlying numpy object is reachable — and counted — under multiple dotted
keys. Examples:

- `bool`, `bool_`, `ma.MaskType`, `ma.bool_` → **same class object**, counted 4×
- `int64`, `int_`, `intp`, `long` → same class, counted 4×
- `uint`, `uint64`, `uintp`, `ulong` → same class, counted 4×
- `char.add`, `strings.add`, `add` → same ufunc, counted 3×
- `ma.MaskedArray`, `ma.masked_array` → same class, counted 2×
- `char.ndarray` / `ndarray`, `char.str_` / `str_`, `char.bytes_` / `bytes_`,
  `rec.recarray` / `recarray`, `rec.record` / `record` → same objects, 2× each
- ~50 more `char.*` / `strings.*` pairs (e.g. `char.upper` / `strings.upper`)
  that are literally the same function object exposed under two module paths
- `False_` / `ma.nomask` → same `numpy.bool` singleton object (verified with
  `is`, not just equality)

This inflates the denominator independent of the ndarray/MaskedArray
question. It's real, not an artifact — confirmed via `type()` and `is`, not
just `id()` coincidence.

### Finding C — non-API implementation leakage counted as if it were surface

21 `names` entries are not in their own module's `__all__`. Most of those are
Finding B duplicates or legitimately-undocumented-but-real functions
(`char.slice`, `random.get_bit_generator`/`set_bit_generator` — real,
callable, just missing from `__all__`). But **9 are not API at all**:

- `fft.test`, `lib.test`, `linalg.test`, `ma.test`, `polynomial.test`,
  `random.test`, `testing.test` — each is a `numpy._pytesttester.PytestTester`
  object, the legacy `np.<submodule>.test()` pytest-runner shim. It is not
  something ionp would ever "implement"; it runs numpy's own test suite.
- `char.array_function_dispatch`, `char.set_module` — internal decorator
  utilities (`functools.partial`/plain function) that leaked into the
  `np.char` namespace via `from ... import` and show up in `dir()` because
  they're not modules (the generator's `inspect.ismodule` guard doesn't catch
  them). Not part of `np.char.__all__`, not documented, not real char API.

9 items, verified live by inspecting the actual object each key resolves to.

### Finding D — whole submodules never walked; not undercounted, entirely absent

`snapshot_surface.py`'s `SUBMODULES` list is:
```python
SUBMODULES = ["linalg", "fft", "random", "ma", "polynomial",
              "strings", "rec", "char", "testing", "lib", "ctypeslib", "emath"]
```
Live `np` exposes these additional public submodules: `core`, `dtypes`,
`exceptions`, `f2py`, `typing`. `core` (deprecated/internal alias),
`f2py` (a Fortran-wrapping build tool, not array API) and `typing`
(type-hint definitions) are defensibly out of scope. **`np.exceptions` and
`np.dtypes` are not** — both are documented public numpy API surfaces (numpy
reference docs have dedicated pages for both) and neither has a top-level
`np.*` alias in this numpy version to fall back on:

- `np.exceptions`: 7 public members (`AxisError`, `ComplexWarning`,
  `DTypePromotionError`, `ModuleDeprecationWarning`, `RankWarning`,
  `TooHardError`, `VisibleDeprecationWarning`) — confirmed `hasattr(np,
  "AxisError")` is `False`, so these are not reachable via any other manifest
  entry either. **Zero representation in the denominator.**
- `np.dtypes`: 34 public members (`BoolDType`, `Float64DType`,
  `StringDType`, ... the whole DType-class family plus
  `register_dlpack_dtype`) — same story, zero representation.

**41 real, documented, unreachable-elsewhere items are simply missing.** This
is a different failure mode from Finding A: those items aren't undercounted,
they contribute exactly 0, not 1.

---

## 3. What the denominator should be

No single rule is uniquely correct; the project's own governance note in
`snapshot_surface.py` (docstring) treats changing the denominator as "a
separate, explicit, committed act" requiring a decision, not just a script
run. Two axes of correction are independent:

**Axis 1 — cleanup (uncontroversial, mechanical, small effect):**
remove the 9 non-API leak items (Finding C), collapse the 108 genuine
re-export duplicates to one canonical entry per underlying object
(Finding B), and add the 41 missing `exceptions`/`dtypes` items (Finding D).
Net effect on its own: **1180 → 1104** (−9 −108 +41), coverage
**51.864% → 55.435%** — it goes *up*, because the double-counted re-exports
currently in the manifest outweigh what `exceptions`/`dtypes` add. This
surprised me; I expected cleanup alone to also push it down. Flagging
per your instruction to say when the framing turns out incomplete: fixing
*only* the reverse-direction bugs makes the number look better, not worse,
which is exactly the kind of asymmetry a bad-faith fix would stop at.

**Axis 2 — the reported defect: extend ndarray's per-member explosion to
every class uniformly.** This is where the big, downward move lives, and it
has two defensible sub-rules depending on whether dunders count (both applied
on top of the Axis-1-cleaned, deduplicated base of 816 non-class items + 83
distinct classes after dedup — dedup collapses e.g. `bool`/`bool_`/
`ma.MaskType`/`ma.bool_` into 1 class, so 110 raw classes → 83 distinct
objects):

| rule | denominator | coverage (612/N) |
|---|---:|---:|
| **Rule "PUBLIC-ONLY"** — explode every class into its non-dunder public members only (matches the `ndarray` list of 70, drops dunders as Python-provided boilerplate) | **4,324** | **14.154%** |
| **Rule "FULL-PARITY"** — explode every class into public + dunder members, exactly the methodology already applied to `ndarray` (70+94) | **8,670** | **7.059%** |

Without the Axis-1 cleanup (i.e. naive extension of the current buggy
manifest, keeping the duplicates and leaks): PUBLIC-ONLY → 6,357 (9.628%),
FULL-PARITY → 12,646 (4.839%). I report the cleaned numbers as the headline
because leaving known duplication/leakage in place while also fixing the
main defect isn't a coherent rule — it's just not finishing the job.

**Headline range: the denominator should be somewhere in [4,324 – 8,670],
i.e. current coverage of 51.864% should be reported as somewhere in
[7.059% – 14.154%].** The low end (PUBLIC-ONLY) is the more defensible
default — dunders are near-universal boilerplate (`__repr__`, `__eq__`,
`__hash__`, ...) largely inherited from `object`/`generic` and not
meaningfully "coverage work" the way `.filled()` or `.sum()` is; counting
them the way `ndarray_dunder` currently does inflates with a lot of
near-identical entries across scalar-type aliases. But `ndarray_dunder`
already exists as a precedent (`__add__`, `__getitem__` etc. *are*
operator-protocol work ionp has to implement per type), so FULL-PARITY is
not unreasonable either — hence the range rather than a single number.

For completeness, the opposite direction — make `ndarray` match everyone
else instead of the reverse (drop its 164-item explosion, every class = 1
flat item) — is also internally consistent and is worth stating precisely
because it's the version of "fix" that would make the number look best:
Axis-1-cleaned + flattened = **899**, coverage **68.076%**. I don't recommend
this: `registry.py`'s own comment (lines 10–11) states the `ndarray.`
exploded-prefix convention is deliberate, and `ionp/_state/ma.py`'s
2026-08-03 revocation of `ma.MaskedArray` argues the opposite direction —
that opaque-class credit for a type that's 7/94ths implemented is the actual
lie, not that ndarray's granularity is excessive.

---

## 4. Coverage percentage under every candidate denominator

| scenario | denominator | coverage |
|---|---:|---:|
| current (shipped, defective) | 1,180 | 51.864% |
| Axis-1 cleanup only (dedup + remove leaks + add exceptions/dtypes) | 1,104 | 55.435% |
| noise removal only | 1,171 | 52.263% |
| dedup only | 1,072 | 57.090% |
| flatten `ndarray` to match everyone else, Axis-1-cleaned | 899 | 68.076% |
| flatten `ndarray` only, uncleaned | 1,016 | 60.236% |
| **explode all classes, PUBLIC-ONLY, Axis-1-cleaned** | **4,324** | **14.154%** |
| explode all classes, PUBLIC-ONLY, uncleaned | 6,357 | 9.628% |
| **explode all classes, FULL-PARITY, Axis-1-cleaned** | **8,670** | **7.059%** |
| explode all classes, FULL-PARITY, uncleaned | 12,646 | 4.839% |

The two bolded rows are the recommended range endpoints from §3.

Caveat on the numerator: `612` is held constant across every row above for
illustration of magnitude. A properly rebuilt denominator would also require
re-verifying, per newly-exploded item, whether it's actually declared+tested
under a matching dotted key (e.g. does anything declare `ma.MaskedArray.sum`
today? — no, nothing does, since that key doesn't exist yet). The true
post-fix numerator is very likely *lower* than 612, not equal to it, since
none of ionp's `_state/*.py` files currently declare per-method keys for any
type except `ndarray`. I did not attempt to recompute it — doing so means
writing new declaration keys, which is out of scope for a measurement-only
task and would require editing files under `ionp/`, prohibited here. Every
percentage in §3/§4 should be read as an upper bound on what coverage would
be after a real fix, not the exact post-fix number.

---

## 5. Recommendation (not applied — description only)

A real fix to `tools/snapshot_surface.py` would need to:

1. **Mechanical, low-risk (Axis 1):** add `"exceptions"` and `"dtypes"` to
   `SUBMODULES`; remove the `PytestTester`/decorator-leak items (either an
   explicit denylist, or filter by checking the object isn't a
   `numpy._pytesttester.PytestTester` and isn't defined in a different
   top-level module than the one being walked — `obj.__module__ !=
   mod.__name__` catches the `functools.partial`/leaked-decorator case);
   deduplicate by `id()` to one canonical key per underlying object (needs a
   policy for which key is canonical — shortest dotted path is what I used
   for measurement, but that's a choice, not a fact).

2. **Judgment call (Axis 2, the actual reported defect):** decide which
   classes get the `ndarray`-style per-member explosion. Doing it for
   literally all 110 (soon 108 post-dedup) classes means exploding ~40
   near-identical numeric scalar-type aliases into ~130 entries apiece —
   mechanically consistent but arguably absurd (thousands of near-duplicate
   `__eq__`/`__hash__`/`__add__` entries that differ only in which scalar
   dtype they belong to). A curated subset (container/behavior-bearing types:
   `ndarray`, `MaskedArray`, `matrix`, `recarray`, `chararray`, `memmap`,
   `poly1d` + the 6 polynomial bases, `Generator`, `RandomState`,
   `BitGenerator` + its 5 concrete subclasses, `dtype`, `finfo`, `iinfo`) is
   more defensible but is an editorial decision the file's own docstring
   says should be "a separate, explicit, committed act" — i.e. this needs a
   maintainer decision, not just a script change.

3. **What it breaks downstream:**
   - `tools/coverage.py`'s `all_items()` (lines 56–86) hardcodes the
     `"ndarray."` prefix in its dupe/prefix assertions (`bad = [n for n in nd
     + dunder if not n.startswith("ndarray.")]`). Generalizing to more
     exploded types means either generalizing this check to a set of known
     prefixes or restructuring `numpy_surface.json`'s schema away from the
     current `names`/`ndarray`/`ndarray_dunder` three-key shape into
     something like `{"names": {...}, "exploded": {"ndarray": [...],
     "MaskedArray": [...], ...}}`.
   - `tests/differential/registry.py`'s `resolve_ionp()` (line 1226) already
     special-cases the `"ndarray."` prefix to resolve against
     `ionp.ndarray` rather than top-level `ionp`. Every newly-exploded type
     needs the same dispatch branch added, keyed off its own prefix.
   - Every `_state/*.py` file is keyed by dotted item name. Newly-exploded
     types need new per-method declaration keys (e.g. `ma.MaskedArray.sum`)
     to replace what is currently either nothing (as of today,
     `ionp/_state/ma.py` already revoked the single `ma.MaskedArray` key
     rather than leave a false "exact") or a single opaque type-level key for
     every other type in the audit above. This is a large amount of new
     declaration work, not just a manifest regeneration — hundreds of new
     items to individually verify, matching the scale in §3.
   - Any currently-"exact"/"ion" declaration for a type that only tested the
     constructor + a handful of attributes (the exact failure mode
     `ionp/_state/ma.py`'s 2026-08-03 comment describes for
     `ma.MaskedArray`) would need to be re-examined for every other class in
     the audit, since the same "type marked exact because its constructor
     works" pattern is structurally possible for `matrix`, `recarray`,
     `poly1d`, and the random `Generator`/`RandomState` classes too — I did
     not check whether any of those are currently over-declared this way;
     that would be a second, separate audit against `_state/*.py`.

Bottom line: fixing the denominator honestly is bigger than a
`snapshot_surface.py` patch. It's a manifest schema change, a registry
dispatch change, and — the actual bulk of the work — hundreds of new
per-method declarations to individually verify before they can be credited.
That's consistent with the project's own stated policy that the denominator
doesn't move without a deliberate, committed decision.
