# UNDECLARED-337: triage of the "undeclared" verdict split (ticket #82)

## Background

`tools/coverage.py` used to fold two different failures under one verdict,
`absent`: items that are genuinely not implemented, and items that
`resolve()` (numpy-relative, mutation-trap-aware) finds are *actually
resolvable on anionpy right now*, but for which nobody wrote the
corresponding line in `__ion_state__`. Of the 1482 rows that used to print
"not implemented", 337 resolve `True` against a live anionpy import. Those
337 are the subject of this document. The other 1145 are unambiguous:
nothing resolves, there is nothing to triage.

**None of the 337 items are declared by this ticket.** Declaring requires a
per-item differential corpus plus out-of-corpus verification. This is a
classification of *why each one is still undeclared*, not a batch of new
declarations.

## Method

1. `tools/coverage.py --list undeclared` produced the ground-truth list of
   337 dotted names (`/tmp/undeclared337.txt`, regenerable any time from a
   fresh differential run).
2. For every item that belongs to one of the curated "exploded" classes
   (`ndarray`, `dtype`, `finfo`, `iinfo`, `matrix`, `memmap`, `ma.MaskedArray`,
   `poly1d`, the six polynomial bases, `random.Generator`, ...), a script
   reused `coverage.py`'s own `_numpy_class_cached()` to check whether real
   numpy's class **owns** the attribute in its own `__dict__` or only
   **inherits** it. This is the same distinction `resolve()`'s numpy-relative
   logic is built on, applied here for classification instead of scoring.
3. `grep` swept `anionpy/_state/*.py` and `KNOWN-DIFFERENCES.md` for a
   recorded reason per item, and each candidate citation was read in full
   context (not keyword-matched blindly) before being accepted.
4. Every citation used below was checked for staleness where cheap to do so.
   Two were found stale or false during this pass (see "Stale citations
   found" below) and are **not** used as bucket (a) evidence; the items they
   would have covered are filed under (b) instead, with the false citation
   named so it doesn't get silently reused later.

## Bucket counts

| Bucket | Count | Meaning |
|---|---|---|
| (a) DELIBERATE WITHHOLD | 180 | A recorded, contextually-verified reason exists |
| (b) NEVER TRIAGED | 21 | No recorded reason found anywhere, despite a real search |
| (c) STRUCTURALLY UNDECLARABLE | 136 | Resolves only via inheritance from `object` (or Python's automatic `__weakref__` slot), with no discretionary implementation on either side |
| **Total** | **337** | |

Bucket (c) is not "these are fine to ignore forever" — it means the item is
not a meaningful unit of work: real numpy's own class does not override the
attribute either, so there is nothing anionpy-specific to measure it
against. `resolve()`'s mutation-trap-aware ownership check is exactly the
mechanism that makes this distinction possible; this triage just reads that
signal for every undeclared item instead of only using it for scoring.

---

## Bucket (a): DELIBERATE WITHHOLD (180 items)

Grouped by identical or closely-related citation to keep this readable.
Each entry names the file (and line, where a stable one exists) the reason
was found in.

- **`arange`**: anionpy/_state/toplevel.py line 4506: still UNDECLARED after the 2026-08-01 reclaim -- widened sweep found two bug classes: (a) negative step + unsigned dtype= + float start/step silently clamps via cast_from_float! saturating a negative f64 to 0; (b) an exact negative-integer start combined with certain params. Fixing (a) means touching a crate-wide primitive relied on by many other declared items -- explicitly out of scope for the task that found it.

- **`array`**: anionpy/_state/toplevel.py line 18: 2026-08-01 parameter-blindness audit -- numpy 2.5.1's np.array() accepts ndmax= (in addition to ndmin=); anionpy raises TypeError for the unknown kwarg. Re-verified live against numpy 2.5.1.

- **`asmatrix`**: anionpy/_state/matrix.py lines ~63-105 + KNOWN-DIFFERENCES.md lines 2663-2691: copy=False construction path DECLINED permanently -- anionpy.matrix(x, copy=False)/asmatrix(x) does not alias the input buffer the way numpy's does; matrix.__new__'s copy=False path shares the identical gap.

- **`astype`**: anionpy/_state/toplevel.py (toplevel astype, not ndarray.astype): REVOKED 2026-08-03, the same day it was declared -- copy=False path was deliberately excluded from the tested corpus; the divergence is actually observable without .flags/.base, through plain object identity and shares_memory, and is real.

- **`concatenate`**: anionpy/_state/toplevel.py line 3744 + KNOWN-DIFFERENCES.md line 3439 (2026-08-08 entry): out= gap CONFIRMED CURRENT (anionpy raises 'out= is not supported'). The casting='unsafe' half of the original 2026-08-01 citation is STALE and must not be cited per the file's own later correction -- 2026-08-08 re-measurement found casting= now matches numpy across all five modes, values, dtype, and error text. Reported here with the stale half explicitly flagged, not silently inherited.

- **`diag`, `diagonal`, `rollaxis`, `rot90`, `trace`**: anionpy/_state/toplevel.py lines ~3675-4373 (toplevel forms) and anionpy/_state/linalg.py line ~325 (linalg.diagonal, 'CLASS A REVOKED 2026-08-02'): shared CLASS A pattern -- anionpy returns a COPY where numpy returns a VIEW with matching strides.

- **`equal`, `greater`, `greater_equal`, `less`, `less_equal`, `not_equal`**: anionpy/_state/toplevel.py lines 391-396: identical 'order= remains the SOLE blocker' citation for each of the 6 comparison ufuncs.

- **`errstate`, `geterr`, `geterrcall`, `seterr`, `seterrcall`**: KNOWN-DIFFERENCES.md line ~2280: errstate/seterr/geterr/seterrcall/geterrcall decline, with detailed divergence examples. CITED-BUT-UNVERIFIED (not re-run live this task; flagged only because this doc has at least one confirmed-stale entry elsewhere -- see concatenate/float_power notes -- so treat this citation as needing re-verification before being relied on for anything beyond this triage).

- **`expm1`**: anionpy/_state/toplevel.py lines ~1471-1489: expm1 citation.

- **12 items** (`fft.fft`, `fft.fft2`, `fft.fftn`, `fft.ifft`, `fft.ifft2`, `fft.ifftn`, ... +6 more): anionpy/_state/fft.py lines ~200-372: stride-gap classification, REVOKED 2026-08-02 -- anionpy's FFT uses a non-last-axis transpose-compute-transpose-back mechanism that numpy's pocketfft does not; layout/stride divergence on non-default-axis calls.

- **`finfo.__init__`, `finfo.__new__`, `iinfo.__init__`, `iinfo.__new__`**: anionpy/_state/finfo_iinfo.py lines ~92-99: 4/839 confirmed error-message divergence for unresolvable dtype-like inputs; withheld per the project's 'if ANY call form still diverges, do NOT declare it' rule. RE-VERIFIED (this task): claim is about __new__/__init__ specifically, current as far as re-read; the same comment block's broader claim that *all* other finfo/iinfo dunders are 'simply inherited from object' is IMPRECISE (see bucket (c) note on __weakref__) but does not affect this citation.

- **`hypot`**: anionpy/_state/toplevel.py lines ~1688-1699: hypot citation.

- **`linalg.cross`**: anionpy/_state/linalg.py line 388: REVOKED 2026-08-02 (stride-gap classification) -- batched cross product over a non-default axis inherits input layout in numpy; anionpy does not. Values/bytes equal, pure layout.

- **`linalg.diagonal`**: anionpy/_state/linalg.py line 325: CLASS A REVOKED 2026-08-02, same copy-vs-view pattern as diag/rot90/rollaxis/trace.

- **`linalg.matrix_power`**: anionpy/_state/linalg.py lines 207-249: RE-EARNED 2026-08-02 then REJECTED same day (main-session audit) -- negative-exponent path produces signed-zero divergences (-0.0 vs 0.0) invisible to tolerance-based comparators; confirmed a real, observable divergence (signbit propagates through 1/-0.0, division, complex arithmetic).

- **`linalg.norm`, `linalg.vector_norm`**: anionpy/_state/linalg.py line ~454/479: STAY UNDECLARED -- ord-domain validation defect found in a 2184-case sweep across vector_norm/matrix_norm/norm.

- **`linalg.trace`**: anionpy/_state/linalg.py lines 158-192: WITHDRAWN AGAIN 2026-08-02 -- linalg.trace(a, dtype=np.int32) silently returns int64 instead of int32; also raises NotImplementedError on float16 in one call form.

- **`linalg.vecdot`**: anionpy/_state/linalg.py lines ~305-312: REVOKED 2026-08-03 -- raises NotImplementedError ('only the default axis=-1 is implemented') for ANY explicit axis kwarg; the declaration appears never to have exercised the axis kwarg.

- **`ma.MaskedArray.flatten`**: anionpy/_state/ma.py lines ~1465-1475: empty-array stride divergence inherited from the ndarray layer.

- **`ma.MaskedArray.mean`**: anionpy/_state/ma.py lines ~1110-1119: float16 scalar-type-identity bug.

- **`matmul`, `matvec`, `vecdot`, `vecmat`**: anionpy/_state/toplevel.py lines 5191-5194: 2026-08-01 parameter-blindness audit -- same casting=/order=/subok= gap as the generic ufunc cluster, raised from each item's own dedicated gufunc binding. Re-verified live against numpy 2.5.1 (per the comment's own text).

- **`matrix.__new__`**: anionpy/_state/matrix.py / KNOWN-DIFFERENCES.md line ~2633: matrix's __new__/asmatrix copy=False DECLINE -- structural composition-wrapper limitation.

- **7 items** (`median`, `nanmedian`, `nanpercentile`, `nanprod`, `nanquantile`, `percentile`, ... +1 more): anionpy/_state/toplevel.py lines 3070-3260: median/nanmedian REVOKED 2026-08-02 (tuple-axis TypeError; nanmedian also has a length-1-axis stride divergence). percentile/quantile/nanpercentile/nanquantile WITHDRAWN 2026-08-02 same day as declared (weak-scalar-q dtype preservation gap, ndim>=2 q, quantile weights=). Re-audited 2026-08-04: weak-scalar dtype preservation now fixed, but ndim>=2 q / weights= / median length-1-axis stride label still open (measured, not inferred). nanprod: 3 remaining axis_tuple_noncontig failures, open unsolved traversal-order problem (brute-force search found no bit-exact model, best candidate ~67% match).

- **`min_scalar_type`, `promote_types`, `result_type`**: anionpy/_state/toplevel.py line ~438: same architectural gap as can_cast/isdtype -- missing object/string dtype family handling; 'nothing smaller closes any of them.'

- **`ndarray.T`**: anionpy/_state/ndarray.py: T citation (view/stride-contract note, CLASS A/B family).

- **`ndarray.__array__`**: anionpy/_state/ndarray.py: __array__ citation.

- **`ndarray.__divmod__`**: anionpy/_state/ndarray.py: __divmod__ citation.

- **`ndarray.__getitem__`**: anionpy/_state/ndarray.py: __getitem__ citation (fancy/advanced-indexing gap documented).

- **`ndarray.__iadd__`**: anionpy/_state/ndarray.py: __iadd__ family (in-place arithmetic dunders) citation -- documented alongside __ipow__/__imatmul__ as a shared in-place-casting gap.

- **`ndarray.__iand__`**: anionpy/_state/ndarray.py: __iadd__-family citation (in-place bitwise/arithmetic dunders share the same documented gap).

- **`ndarray.__imatmul__`, `ndarray.__matmul__`, `ndarray.__rmatmul__`**: anionpy/_state/ndarray.py: __matmul__/__rmatmul__/__imatmul__ citation.

- **`ndarray.__imul__`, `ndarray.__ior__`, `ndarray.__isub__`, `ndarray.__itruediv__`, `ndarray.__ixor__`**: anionpy/_state/ndarray.py: __iadd__-family citation.

- **`ndarray.__ipow__`**: anionpy/_state/ndarray.py: __ipow__ specific citation (in-place power casting gap).

- **`ndarray.__repr__`**: anionpy/_state/ndarray.py: __repr__ citation (formatting divergence from numpy's array2string).

- **`ndarray.__rtruediv__`**: anionpy/_state/ndarray.py: same __truediv__/__rtruediv__ citation.

- **`ndarray.__str__`**: anionpy/_state/ndarray.py: __str__ citation (same formatting-divergence family as __repr__).

- **`ndarray.__truediv__`**: anionpy/_state/ndarray.py: __truediv__/__rtruediv__ divergence citation (dtype/casting edge cases documented near the arithmetic-dunder block).

- **`ndarray.astype`**: anionpy/_state/ndarray.py: astype citation (casting-kwarg parameter-blindness gap).

- **`ndarray.compress`**: anionpy/_state/ndarray.py: compress/take citation (shared gap).

- **`ndarray.diagonal`**: anionpy/_state/ndarray.py line ~1431: CLASS A (copy vs numpy view+matching-strides) citation, same family as diag/rot90/rollaxis/trace/linalg.diagonal.

- **`ndarray.imag`, `ndarray.real`**: anionpy/_state/ndarray.py: real/imag citation.

- **`ndarray.prod`**: anionpy/_state/ndarray.py: prod/sum citation (dtype/reduction-layout gap).

- **`ndarray.sum`**: anionpy/_state/ndarray.py: prod/sum citation.

- **`ndarray.take`**: anionpy/_state/ndarray.py: compress/take citation.

- **`poly`, `polyfit`, `roots`**: anionpy/_state/poly_legacy.py (module docstring, lines ~3-55): declared in the registry with a real, evidence-gated epsilon_tolerance (not blind atol/rtol), same treatment as polynomial.polynomial.polyroots/polyfit. These pass under their declared tolerance but the measured slack itself is the reason they are not declared bit-exact -- 'state=exact with the epsilon caveat already declared in the registry entries' is explicitly distinguished from unconditional exactness in this file.

- **50 items** (`poly1d.__add__`, `poly1d.__array__`, `poly1d.__call__`, `poly1d.__delattr__`, `poly1d.__dir__`, `poly1d.__eq__`, ... +44 more): anionpy/_state/poly_legacy.py (module docstring): 'poly1d (the class itself) has NO items declared here: the committed corpus (poly1d_legacy_cases.py) covers only the 9 free functions above, not any poly1d.<method>/dunder -- there is nothing measured to declare a class-level item against. Declining is not a finding, just an absence of corpus for this pass to verify against.' CITED-BUT-UNVERIFIED beyond confirming the corpus file still only covers the 9 free functions (polydiv/polyadd/polysub/polyval/polyder/polyint + poly/roots/polyfit).

- **26 items** (`polynomial.Chebyshev.__mul__`, `polynomial.Chebyshev.__pow__`, `polynomial.Chebyshev.__rmul__`, `polynomial.Chebyshev.cast`, `polynomial.Chebyshev.convert`, `polynomial.Chebyshev.fromroots`, ... +20 more): KNOWN-DIFFERENCES.md 2026-08-07 entries (~lines 3039-3280): the underlying complex128 polymul/chebmul/legmul kernel is not bit-exact (polymul complex128 sweep, seed=31337, N=3000: 2388/3000 non-bit-exact, re-measured not assumed). fromroots/convert/cast/mul/pow across the affected bases build on this kernel and were given epsilon tolerances citing this exact divergence, then that epsilon tolerance was retained/unremoved rather than the item declared exact outright -- distinct from Laguerre/Hermite/HermiteE's own __mul__/__pow__/fromroots which are apparently already declared (not in the 337) per this same entry's 'four of the six bases already routed every complex multiply through complex_mul_fma' note. CITED-BUT-UNVERIFIED beyond reading the entry; did not re-run the N=3000 sweep this task.

- **`pow`, `power`**: anionpy/_state/toplevel.py line ~1731: pow/power citation.

- **`prod`**: anionpy/_state/toplevel.py line 4804: still WITHDRAWN -- 3 remaining failures are all axis_tuple_noncontig, same disclosed unsolved gapped-multi-axis reduction-order gap as nanprod.

- **`random.Generator.__new__`**: No dedicated __new__ citation found in anionpy/_state/random.py; however random.Generator (bare class item) has a citation that the class implements ~24/30 methods and is missing .choice()/.permutation()/.bit_generator -- __new__ constructing an incomplete class is covered by the same incompleteness, not a separate reason. Treated as CITED-BUT-UNVERIFIED / borderline (a)/(b): the class-level reason plausibly explains __new__ too but no comment names __new__ directly.

- **`random.PCG64`, `random.PCG64DXSM`, `random.SeedSequence`**: anionpy/_state/random.py: PCG64/PCG64DXSM missing .state/.advance()/.jumped(); SeedSequence missing .entropy/.spawn_key/.pool_size/.pool/.state/.spawn(). Self-correcting docstring dated 2026-08-07 (retracted a prior false premise about surface granularity) -- CITED-BUT-UNVERIFIED beyond confirming the corrected docstring is the current one in the file.

- **`stack`**: anionpy/_state/toplevel.py line 3792: 2026-08-01 parameter-blindness audit -- numpy accepts out= and casting='unsafe'; anionpy raises ValueError for both. Re-verified live against numpy 2.5.1.

- **`strings.partition`, `strings.rpartition`**: anionpy/_state/char_strings.py line 1048: REVOKED 2026-08-03. Real numpy truncates `sep` to the array's per-element itemsize before searching; anionpy searches with the untruncated sep, producing a silent wrong partition point (not a raise) when len(sep) > itemsize. 7/8 boundary cases diverge; measured live on this binary.

- **`trim_zeros`**: anionpy/_state/toplevel.py line 5427: REVOKED 2026-08-03 -- numpy's trim_zeros returns a view (base set, shares_memory True), anionpy returns a fresh copy (shares_memory False). Values/bytes equal, aliasing status differs.

---

## Bucket (b): NEVER TRIAGED (21 items)

No recorded reason was found for any of these, in `anionpy/_state/*.py`,
`KNOWN-DIFFERENCES.md`, or the source files themselves. This is a valid,
expected outcome for some fraction of 337 items — not every undeclared item
has to have a story. A few are flagged below as likely **declarable
candidates for a future ticket** because the evidence sitting right next to
them suggests the work may already be done; per the prohibition on this
ticket, none of them are declared here.

- **`float_power`** — has a citation (`KNOWN-DIFFERENCES.md` lines 831/916,
  "not implemented at all" / "deliberately scoped out"), but that citation
  is **FALSE, verified live this task**:
  `hasattr(anionpy, 'float_power')` is `True`, backed by a real
  `<anionpy.ufunc object>`. Deliberately NOT filed under (a) on the strength
  of a falsified reason — filed here instead, with the false citation named
  so nobody re-cites it. See "Stale/false citations found" below.
- `dtype.__ne__` — real numpy's `dtype` owns `__ne__` in its own class
  (confirmed via the same ownership check bucket (c) uses), so this is not
  structurally automatic; no citation found anywhere for why it specifically
  is undeclared while `dtype.__eq__`/`__lt__`/etc. are declared exact.
- `emath.power`, `isnat`, `ldexp` — no citation found in `toplevel.py` or
  `KNOWN-DIFFERENCES.md`.
- `lexsort` — declared as `"exact"` once at `anionpy/_state/toplevel.py`
  line 3507 (commented out / superseded); no reason recorded for why the
  live declaration was removed or never re-added.
- `ma.MaskedArray.__hash__`, `ma.MaskedArray.__repr__` — both are owned by
  real numpy's `MaskedArray` (not inherited from `object`), so bucket (c)
  does not apply; no citation found.
- **`ma.MaskedArray.cumsum`, `ma.MaskedArray.cumprod`** — DECLARABLE
  CANDIDATE. `anionpy/_state/ma.py` lines ~827-833 document that BOTH the
  module-function and method forms of `cumsum`/`cumprod` were tested
  together ("880 module-function cases + 760 method cases each... 0
  divergences"), yet only `ma.cumsum`/`ma.cumprod` (the module functions)
  got declared `"exact"`. The method forms on `MaskedArray` appear to have
  passing evidence sitting in the same comment block that was never turned
  into a declaration. Flagging, not declaring — this needs its own
  standalone verification run before it can be trusted, per this ticket's
  explicit prohibition on bulk-declaring from adjacent evidence.
- `matrix.__array__`, `matrix.__setitem__` — both real, non-inherited
  implementations in `anionpy/matrix.py`; no citation found for why
  undeclared.
- **`matrix.strides`, `memmap.strides`** — DECLARABLE CANDIDATE. Both
  `anionpy/matrix.py` (lines 371-372) and `anionpy/memmap.py` (lines
  337-338) contain a real, working `strides` property
  (`def strides(self): return self._data.strides`). No citation found
  anywhere for why either is undeclared. (Ownership-check note: real numpy's
  `matrix`/`memmap` classes inherit `strides` from `ndarray` rather than
  overriding it, which is why the classifier's owns=False signal alone does
  not put these in bucket (c) — unlike the `object`-protocol dunders,
  `strides` is a meaningful, already-implemented property on the anionpy
  side, not structural boilerplate on either side.)
- `ndarray.__delitem__`, `ndarray.__iter__`, `ndarray.__pow__`,
  `ndarray.__rpow__` — all four are owned by real numpy's `ndarray` (not
  inherited from `object`); `__rpow__` has a `registry.py` `ItemSpec` but no
  `_state` citation was found for any of the four.
- `shares_memory` — ironic given how often this function is *used* as
  ground truth elsewhere in `_state/*.py` (dozens of citations rely on its
  output to describe *other* items' aliasing bugs); no citation was found
  for why `shares_memory` itself is undeclared.
- `testing.assert_array_compare` — absent from `anionpy/_state/testing.py`
  entirely (the file declares 47 other `testing.*` items with a paired
  differential+falsifiability citation each); no mention of this one at all.
- `unique_values` — no citation found.

---

## Bucket (c): STRUCTURALLY UNDECLARABLE (136 items)

For each class below, the listed attributes resolve on anionpy (satisfying
`present=True`) purely because Python or the base class supplies them —
real numpy's own class does not override them either (confirmed live via
`attr in vars(numpy_cls)` → `False`, the same check `resolve()` uses to
avoid the "mutation trap"). There is nothing implementation-specific on
either side to measure.

**dtype** (11): `__delattr__`, `__dir__`, `__format__`, `__getattribute__`, `__getstate__`, `__init__`, `__init_subclass__`, `__reduce_ex__`, `__setattr__`, `__sizeof__`, `__subclasshook__`

**finfo** (19): `__delattr__`, `__dir__`, `__eq__`, `__format__`, `__ge__`, `__getattribute__`, `__getstate__`, `__gt__`, `__hash__`, `__init_subclass__`, `__le__`, `__lt__`, `__ne__`, `__reduce__`, `__reduce_ex__`, `__setattr__`, `__sizeof__`, `__subclasshook__`, `__weakref__`

**iinfo** (19): `__delattr__`, `__dir__`, `__eq__`, `__format__`, `__ge__`, `__getattribute__`, `__getstate__`, `__gt__`, `__hash__`, `__init_subclass__`, `__le__`, `__lt__`, `__ne__`, `__reduce__`, `__reduce_ex__`, `__setattr__`, `__sizeof__`, `__subclasshook__`, `__weakref__`

**ma.MaskedArray** (10): `__delattr__`, `__dir__`, `__format__`, `__getattribute__`, `__init__`, `__init_subclass__`, `__reduce_ex__`, `__setattr__`, `__sizeof__`, `__subclasshook__`

**matrix** (14): `__delattr__`, `__dir__`, `__format__`, `__getattribute__`, `__getstate__`, `__init__`, `__init_subclass__`, `__reduce__`, `__reduce_ex__`, `__repr__`, `__setattr__`, `__sizeof__`, `__str__`, `__subclasshook__`

**memmap** (14): `__delattr__`, `__dir__`, `__format__`, `__getattribute__`, `__getstate__`, `__init__`, `__init_subclass__`, `__reduce__`, `__reduce_ex__`, `__repr__`, `__setattr__`, `__sizeof__`, `__str__`, `__subclasshook__`

**ndarray** (8): `__delattr__`, `__dir__`, `__getattribute__`, `__getstate__`, `__init__`, `__init_subclass__`, `__setattr__`, `__subclasshook__`

**poly1d** (1): `__weakref__`

**polynomial.Chebyshev** (4): `__dir__`, `__format__`, `__repr__`, `__str__`

**polynomial.Hermite** (4): `__dir__`, `__format__`, `__repr__`, `__str__`

**polynomial.HermiteE** (4): `__dir__`, `__format__`, `__repr__`, `__str__`

**polynomial.Laguerre** (4): `__dir__`, `__format__`, `__repr__`, `__str__`

**polynomial.Legendre** (4): `__dir__`, `__format__`, `__repr__`, `__str__`

**polynomial.Polynomial** (4): `__dir__`, `__format__`, `__repr__`, `__str__`

**random** (16): `__delattr__`, `__dir__`, `__eq__`, `__format__`, `__ge__`, `__getattribute__`, `__gt__`, `__hash__`, `__init_subclass__`, `__le__`, `__lt__`, `__ne__`, `__reduce_ex__`, `__setattr__`, `__sizeof__`, `__subclasshook__`

**`__weakref__` special case (3 items: `finfo.__weakref__`,
`iinfo.__weakref__`, `poly1d.__weakref__`):** `anionpy/_state/finfo_iinfo.py`
line ~102 describes these (and the other inherited dunders) as "simply
inherit[ing] object's default." That mechanism claim is imprecise for
`__weakref__` specifically: verified live this task, real numpy's own
`finfo`/`iinfo`/`poly1d` classes **do** own `__weakref__` in their own
`__dict__` (`'__weakref__' in vars(np.finfo)` → `True`). This is not a
counterexample to bucket (c), though — it's normal Python class machinery:
any plain class without `__slots__` gets its own `__weakref__` descriptor
automatically, on both the numpy side and the anionpy side, with no
discretionary implementation behind it on either side. It is a different
*flavor* of structural-undeclarability than the other bucket (c) dunders
(automatic-on-both-sides vs. inherited-by-numpy-too), not the same
mechanism the finfo_iinfo.py comment names, so it is called out separately
here rather than silently folded into that citation.

---

## Stale/false citations found during this pass

Per the ticket's explicit warning that recorded reasons can themselves be
false or stale, three were checked closely enough to say something more
precise than "cited":

1. **`float_power`** — `KNOWN-DIFFERENCES.md` lines 831/916 say it is "not
   implemented at all" / "deliberately scoped out." This is **false**,
   verified live this task: `hasattr(anionpy, 'float_power')` is `True`,
   backed by a real `<anionpy.ufunc object>`. `float_power` IS one of the
   337 (it resolves but is undeclared) — it is filed under bucket (b)
   NEVER TRIAGED above, specifically *not* under bucket (a), because the
   only recorded reason for its absence is this falsified claim.
2. **`concatenate`** — `anionpy/_state/toplevel.py` line 3744's original
   2026-08-01 citation claims both `out=` and `casting='unsafe'` diverge.
   A later, dated correction in the same file (2026-08-08, immediately
   below the original comment) says the `casting=` half no longer
   reproduces and "must not be cited." The bucket (a) entry above for
   `concatenate` uses only the still-current `out=` half.
3. **`anionpy/_state/finfo_iinfo.py`**'s claim that finfo/iinfo's inherited
   dunders "simply inherit object's default" is imprecise for `__weakref__`
   — see the special-case note in bucket (c) above.

## Verification note

Coverage before/after this split: `undeclared 337` / `absent 1145`
(337 + 1145 = 1482, matches the pre-split `absent` count exactly), overall
coverage unchanged at `51.426% (1569/3051)`. This document does not change
that number — it is a labelling/triage exercise on the denominator side.
