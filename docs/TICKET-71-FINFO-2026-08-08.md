# TICKET-71: `finfo`/`iinfo` surface -- 2026-08-08

## Scope

`anionpy.iinfo`/`anionpy.finfo` -- the machine-limits classes for integer
and floating dtypes. Previously absent from the coverage ledger entirely
(no `__ion_state__` entries), despite both classes being implemented and,
per prior agents' work in `anionpy/_state/toplevel.py`, extensively
measured against real numpy already. This ticket's job was to find out
*why* the ledger showed them as structurally absent, fix what was fixable,
and declare what was genuinely verified.

No Rust was changed (`ionp-core/src/dtype.rs`'s `FInfo`/`IInfo` tables and
`ionp-py/src/dtypeinfo.rs`'s `iinfo`/`finfo` pyfunctions are untouched).
All changes are in `anionpy/_dtypeinfo.py` (Python dispatch layer only, no
numerical loops added).

## The central finding: STORED vs COMPUTED class shape

`tools/numpy_surface.json`'s `"exploded"` section curates
`finfo.epsneg`/`.iexp`/`.machep`/`.negep`/`.nexp`/`.resolution`/`.tiny` and
`iinfo.min`/`.max` as ledger items in their own right -- these are exactly
numpy's `@cached_property` (finfo, 7 of them) and `@property` (iinfo, 2 of
them) CLASS-level descriptors, confirmed by reading
`numpy/_core/getlimits.py` directly and by inspecting
`numpy._core.getlimits.{iinfo,finfo}.__dict__` live. `tools/coverage.py`'s
`resolve()` checks item presence with `attr in vars(cls)` -- i.e. the
attribute must be a genuine class-level descriptor, not merely an instance
attribute assigned in `__init__`, to even register as "present" on the
ledger, independent of whether its VALUE is correct.

anionpy's `iinfo`/`finfo` stored every one of these as a plain
`self.<name> = ...` instance attribute. Live-checked before this ticket's
fix:

```
'epsneg' in vars(anionpy.finfo)  -> False   (numpy: True)
'min'    in vars(anionpy.iinfo)  -> False   (numpy: True)
```

...for all 9 attributes. Regardless of correctness, these 9 items were
structurally `absent` from the ledger, not merely undeclared.

**Fix** (`anionpy/_dtypeinfo.py`): converted exactly these 9 attributes to
`@property` descriptors backed by private instance storage (`_min`, `_max`
on `iinfo`; `_epsneg`, `_iexp`, `_machep`, `_negep`, `_nexp`, `_resolution`
on `finfo`). `finfo.tiny` specifically is implemented as
`return self.smallest_normal` (a read-through alias with no private
storage of its own), matching numpy's actual implementation -- numpy's own
docstring literally calls it "alias of smallest_normal", it is not a
second stored copy of the same value. Every other attribute (`eps`, `max`,
`min` on finfo; `bits` etc.) stays a plain instance attribute, matching
numpy's own class shape exactly rather than converting everything
uniformly (which would have been wrong -- numpy itself does NOT make `eps`
or `max` class-level).

Verified post-fix, live, via `tools.coverage.resolve()` called directly
against the built package:

```
finfo.epsneg   (True, <property object ...>)
finfo.iexp     (True, <property object ...>)
finfo.machep   (True, <property object ...>)
finfo.negep    (True, <property object ...>)
finfo.nexp     (True, <property object ...>)
finfo.resolution (True, <property object ...>)
finfo.tiny     (True, <property object ...>)
iinfo.min      (True, <property object ...>)
iinfo.max      (True, <property object ...>)
```

All 9 flip from structurally absent to present, matching numpy.

## Other fixes in `anionpy/_dtypeinfo.py`

1. **Class rename**: `IInfo` -> `iinfo` (numpy's actual class name; the
   Python-facing symbol was exported under the wrong name via
   `anionpy/__init__.py`'s `from anionpy._dtypeinfo import IInfo as iinfo,
   finfo`, now `import iinfo, finfo` directly). One stale comment
   referencing the old name elsewhere in the codebase (`_state/toplevel.py`
   line ~5782) was left as-is -- historical note, not currently misleading
   since it's inside a preserved-for-history docstring block, not touched
   per this ticket's scope discipline (not the file I was measuring).
2. **`__str__` added to both classes** (previously absent on both --
   `str(x)` fell through to `object.__str__` -> `repr(x)`, diverging from
   numpy's real multi-line "Machine parameters for ..." block). Matched
   byte-for-byte against `numpy/_core/getlimits.py`'s own `get_str`/
   `__str__` source, including a real dead-code detail: `get_str(name,
   pad=None)` computes `s = str(val).ljust(pad)` but its `return` statement
   always returns `str(val)`, never `s` -- the `pad` argument is
   unreachable/inert in every numpy version checked. Reproduced faithfully
   (always plain unpadded `str()`), not "fixed" into something numpy itself
   doesn't do.

## Measurement

**Comprehensive out-of-corpus probe**
(`/private/tmp/finfo_probe/full_probe.py`, real numpy import, measurement
only, not shipped): 839 checks, covering every int dtype (int8/16/32/64,
uint8/16/32/64) for `iinfo` and every float + complex dtype
(float16/32/64, complex64/128) for `finfo`, each dtype specified 5 ways
(type object, dtype instance, name string, char code, numpy scalar
instance); every numeric attribute compared by VALUE AND TYPE, never by
`getattr` presence alone; `repr`/`str`/class-name; repeated-construction
identity semantics (`iinfo`: no caching, `is` -> `False`, matches numpy;
`finfo`: caches per resolved dtype including cross-spelling identity
`np.finfo('f4') is np.finfo(np.float32) is np.finfo(np.dtype('float32'))`
and complex->float redirection identity `np.finfo('complex64') is
np.finfo('float32')`, all `True`, matches numpy); 11 invalid inputs each to
`iinfo`/`finfo` (complex-to-iinfo, int-to-finfo, ndarray, nonsense string,
None, list, dict, plain object, bare int/float value, `str` type) compared
by exact exception TYPE and MESSAGE TEXT.

Run BEFORE the structural fix and again AFTER (to confirm the property
conversion introduced zero regressions): **identical result both times,
4/839 mismatches**, all the same pre-existing, already-documented gap:

```
iinfo(invalid:dict)          np=ValueError(Invalid integer data type 'V'.)
                              ionp=ValueError(Invalid integer data type 'O'.)
finfo(invalid:nonsense-str)  np=ValueError(data type dtype('<U') not compatible with finfo)
                              ionp=ValueError(data type dtype('<U0') not compatible with finfo)
finfo(invalid:dict)          np=ValueError(data type dtype([]) not compatible with finfo)
                              ionp=ValueError(data type dtype('O') not compatible with finfo)
finfo(invalid:str-type)      np=ValueError(data type dtype('<U') not compatible with finfo)
                              ionp=ValueError(data type dtype('<U0') not compatible with finfo)
```

anionpy has no void/structured/string dtype representation at all, so
unresolvable dtype-like inputs fall back to a `dtype('O')` object-dtype
error message where real numpy gives a genuine void (`'V'`) or
`<U`-string-typed fallback message. This is NOT new: it is the same
architectural gap already documented for `promote_types`/`result_type`/
`can_cast` and specifically for `finfo` under ticket #31. Re-measured this
session (not merely trusted from the prior record) and reconfirmed present,
unchanged, on the current binary. Fixing it cleanly would require adding a
new `DType` variant across `ionp-core` -- out of scope for this ticket, and
declining to force a fix here is consistent with the standing
"architectural gap, not a per-item bug" precedent.

**Corpus self-test**: ran every case in the new
`tests/differential/finfo_iinfo_cases.py` directly (bypassing `registry.py`
since it is off-limits, calling each `ItemSpec`'s `custom_cases`/
`numpy_adapter`/`ionp_adapter` by hand) -- 405 cases across the 17
registered items (13 declared + 4 intentionally-undeclared `__new__`/
`__init__` items), 8 mismatches, all 8 confined to the 4 undeclared items
and all 8 being the identical 4-way gap above (dict/nonsense-str/str-type
across both classes' construction paths). Zero mismatches in any of the 13
declared items.

## Declared (`anionpy/_state/finfo_iinfo.py`, `FINFO_IINFO_STATE`, 13 items, all "exact")

- `iinfo.min`, `iinfo.max`
- `finfo.epsneg`, `finfo.iexp`, `finfo.machep`, `finfo.negep`, `finfo.nexp`,
  `finfo.resolution`, `finfo.tiny`
- `iinfo.__repr__`, `iinfo.__str__`, `finfo.__repr__`, `finfo.__str__`

Reasoning for the split: `__repr__`/`__str__` are invoked only on an
already-successfully-constructed instance -- they never touch the
confirmed error-message divergence, so the one known gap does not taint
them. The 9 structural items passed with zero divergence across the full
dtype x spelling matrix on every valid input tested.

Verified NOT orphaned: `anionpy/_state/__init__.py` imports
`FINFO_IINFO_STATE` and includes it in `_BLOCKS`/`__all__`; live-checked
`anionpy.__ion_state__['finfo.epsneg'] == 'exact'` and
`anionpy.__ion_state__['iinfo.min'] == 'exact'` after import, and
`tools/coverage.py --json` shows all 13 items with verdict `untested`
(present + declared, but no passing differential test yet -- expected,
since the corpus is not wired into `registry.py`), NOT `phantom` (which
would mean declared-but-unresolvable) and NOT `absent` (undeclared).

## Declined

- **`iinfo.__init__`/`.__new__`, `finfo.__init__`/`.__new__`** (4 items):
  these ARE the methods that hit the 4/839 invalid-dtype-input divergence.
  Per this project's standing rule ("if ANY call form of an item still
  diverges from numpy -- any input, any error path -- do NOT declare it"),
  not declared, even though every valid-dtype call form passes. Genuine
  decline: registered in the new differential module anyway (for
  visibility / future regression detection), but explicitly excluded from
  `FINFO_IINFO_STATE`.
- **The `dtype('O')` vs `dtype('V')`/`dtype('<U')` fallback gap itself**:
  architectural, requires a new `DType` variant, out of scope (matches
  ticket #31's prior disposition, re-confirmed not re-solved here).
- **All remaining inherited-from-`object` dunders** (`__eq__`, `__hash__`,
  `__ne__`, ordering dunders, `__delattr__`, `__dir__`, `__format__`,
  `__getattribute__`, `__getstate__`, `__init_subclass__`, `__module__`,
  `__reduce__`/`__reduce_ex__`, `__setattr__`, `__sizeof__`,
  `__static_attributes__`, `__subclasshook__`, `__weakref__`,
  `__class_getitem__` -- 20 items x 2 classes worth of ledger entries):
  neither numpy's nor anionpy's classes override any of these. Spot-checked
  `__eq__`/`__hash__` (identity-based, matches numpy's cached-so-equal-
  instances-are-identical `finfo` behavior and non-cached-so-unequal
  `iinfo` behavior) but NOT declared -- this ticket's measurement effort
  covered the `finfo`/`iinfo`-SPECIFIC surface, not a full audit of
  object-inherited behavior, and a single spot-check per item is not
  enough rigor to declare 40 items. Left for a future ticket that actually
  audits that surface.

## Merge instruction (for whoever next has write access to `registry.py`)

```python
from finfo_iinfo_cases import FINFO_IINFO_SPECS
_check_no_collisions(FINFO_IINFO_SPECS)
REGISTRY.update(FINFO_IINFO_SPECS)
```

(Same pattern already used for `LINALG_SPECS`/`INPLACE_SPECS`/
`TESTING_SPECS`/`LEGACY_POLY_SPECS`.) Once merged and the differential
suite is run, `tools/coverage.py`'s verdict for the 13 declared items
should flip from `untested` to `exact` (all 13 have zero measured
mismatches); the 4 `__new__`/`__init__` items are registered but NOT in
`FINFO_IINFO_STATE`, so they will show `absent` regardless -- that is
intentional, not a bug in the merge.

## Build / verification trail

- `cargo build -p ionp-py --release` succeeded (no Rust changed; run as
  the required pre-commit gate anyway).
- No `maturin develop` rebuild performed or needed -- pure-Python edits
  under `anionpy/`, per the standing rule that those need no rebuild. The
  installed `.so` is unchanged by this ticket.
- `python -c "import anionpy"` succeeds with no `AssertionError` from
  `build_ion_state()`'s duplicate-key guard (confirms `finfo_iinfo.py`'s
  13 keys don't collide with any other block).
