# Fix: curated submodule descent closes the polynomial denominator hole

**Date:** 2026-08-06
**Fixes:** `docs/DENOMINATOR-SUBMODULE-HOLE-2026-08-06.md`
**Measured against:** numpy 2.5.1, `.venv/bin/python`, `/tmp/res_finfo.json`
(1243 items, 30 failures -- unchanged by this task; test results are not
this task's concern, only what they are scored against).

## What changed

`tools/snapshot_surface.py`: added `CURATED_SUBMODULE_DESCEND`, a fixed list
of dotted paths (relative to `np`) that `surface()` cannot see today because
`SUBMODULES` only scans one level deep and `surface()` skips module objects
outright. Each entry was individually adjudicated for re-export overlap
before being added -- see "Excluded" below for the ones that failed that
check. `tools/numpy_surface.json` was regenerated from this script (measurement
tooling only; no `maturin`/`cargo` build was run, `.venv` was not modified).

### Included (225 new items, all additive)

| namespace | new items | why included |
|---|---:|---|
| `polynomial.chebyshev` | 35 | per-basis free-function API (chebval, chebfit, chebroots, ...), zero overlap with top-level `np.polynomial` |
| `polynomial.hermite` | 31 | same shape, hermite basis |
| `polynomial.hermite_e` | 31 | same shape, hermite_e basis |
| `polynomial.laguerre` | 31 | same shape, laguerre basis |
| `polynomial.legendre` | 31 | same shape, legendre basis |
| `polynomial.polynomial` | 28 | same shape, plain-power basis |
| `polynomial.polyutils` | 7 | shared low-level helpers (as_series, format_float, getdomain, mapdomain, mapparms, trimcoef) + the module-level `ABCPolyBase` dedup landed here first |
| `lib.format` | 20 | npy on-disk format read/write (read_array, write_array, MAGIC_*, ...) -- zero bare-name overlap anywhere in the tracked surface |
| `lib.stride_tricks` | 2 | `as_strided`, `sliding_window_view` -- zero overlap |
| `lib.npyio` | 2 | `DataSource`, `NpzFile` (classes; tracked flat, not exploded -- same convention as other non-curated classes already in the manifest) -- zero overlap |
| `lib.array_utils` | 3 | `byte_bounds`, `normalize_axis_index`, `normalize_axis_tuple` -- zero overlap |
| `testing.overrides` | 4 | array_function/ufunc override introspection (`allows_array_function_override`, ...) -- zero overlap with `np.testing`'s own top-level names |
| **total** | **225** | |

Per-namespace counts are NOT uniform (35, 31×4, 28, 7, 20, 2, 2, 3, 4) --
each is the genuinely different real symbol count of that submodule, not an
artifact of the crawl. A uniform count across this list would have been the
instrument-bug tell from the HOLE doc; it is not what happened here.

### Class re-export handling (why 0 renames, and how the count stays honest)

Two things needed dedup so this stayed purely additive instead of
double-counting:

1. **The 6 polynomial basis classes** (`Chebyshev`, `Hermite`, `HermiteE`,
   `Laguerre`, `Legendre`, `Polynomial`) are the SAME object reachable both
   at `np.polynomial.<Name>` (already exploded into per-method items) and at
   `np.polynomial.<basis>.<Name>` (re-export, e.g.
   `np.polynomial.chebyshev.Chebyshev is np.polynomial.Chebyshev`). These
   are excluded from the new flat scan by object-identity dedup against the
   already-exploded class set.
2. **`ABCPolyBase`**, the shared abstract base, is reached via the literal
   same bare name in 6 of the 7 submodules
   (`np.polynomial.chebyshev.ABCPolyBase is np.polynomial.hermite.ABCPolyBase
   is ...`) -- one object, repeated by a module-import accident, not 6
   independent surfaces. Deduped the same way; it lands once, in
   `polynomial.chebyshev.ABCPolyBase` (first in processing order).

**Deliberately NOT deduped:** free functions/consts that happen to share an
underlying object under different names, e.g. `chebyshev.chebtrim`,
`legendre.legtrim`, and `polyutils.trimcoef` are all literally
`numpy.polynomial.polyutils.trimcoef` under the hood, but each is a real,
independently-documented public name (`from numpy.polynomial import
legendre; legendre.legtrim(...)` is genuine API). Deduping these by identity
would be inconsistent with how the manifest already treats `np.abs` and
`np.absolute` -- the same ufunc object, both already tracked as separate
top-level items before this change -- and would silently under-count real
per-basis surface. Verified: `np.abs is np.absolute` is `True`, and both
`"abs"` and `"absolute"` are keys in `tools/numpy_surface.json` today.

### Excluded (adjudicated, not silently dropped)

| namespace | overlap measured | verdict |
|---|---|---|
| `ma.core` | 179/220 public names already tracked under `ma.*` (81%); the 41 non-overlapping names are dominated by import aliases (`np`, `mu`, `ntypes`, `functools`, `inspect`, `operator`, `builtins`, `dt`) and internal helpers (`get_data`, `get_fill_value`, `min_val`, `max_val`, ...) that are public-by-accident, not documented API | EXCLUDED -- double-counting |
| `ma.extras` | 65/80 already tracked under `ma.*` (81%); remainder mostly import aliases (`functools`, `itertools`, `ma`, `np`, `warnings`) plus internal names (`AxisConcatenator`, `mr_class`) | EXCLUDED -- double-counting |
| `random.mtrand` | 53/57 already tracked under `random.*` (93%); remaining 4 (`Sequence`, `np`, `operator`, `warnings`) are import aliases | EXCLUDED -- double-counting |
| `lib.scimath` | `np.lib.scimath is np.emath` -- literally the same module object, and `emath.*` is already in `SUBMODULES` and fully tracked (9/9 items, confirmed by identity, not just name) | EXCLUDED -- 100% identical, not a new namespace at all |
| `numpy.typing` | not measured for overlap; excluded on a different ground | EXCLUDED -- `ArrayLike`/`DTypeLike`/`NBitBase`/`NDArray` are static-typing aliases with no runtime behavior for `ionp` to implement or for the differential harness to test against; adding them would create permanently-untestable denominator weight, a different failure mode than the one this fix targets. Not in `SUBMODULES` today either, so this would be a new top-level namespace, not a descent into an already-tracked flat one -- a bigger scope jump than this fix's mandate. Left for a human decision, not folded in silently. |

`poly1d` overlap with the polynomial submodule surface (flagged as "not
checked" in the HOLE doc) was also checked: `poly1d` is a completely
separate legacy class (`np.poly1d`), already curated-exploded under its own
`poly1d.*` prefix, and shares no object identity with anything added here.
No interaction.

## Measured before/after

Ledger command: `.venv/bin/python tools/coverage.py --tests /tmp/res_finfo.json`

**Before** (repo HEAD, `tools/numpy_surface.json` as committed):
```
surface items   : 2866
  exact        669
  absent      2197
COVERAGE: 23.343%   (669/2866)
  bit-exact vs numpy :    640
  ULP-tolerant       :      3
  epsilon-tolerant   :     24
  tie_exempt (0-sign):      2
  message_pair       :      0
```

**After** (this change):
```
surface items   : 3091
  exact        669
  absent      2422
COVERAGE: 21.643%   (669/3091)
  bit-exact vs numpy :    640
  ULP-tolerant       :      3
  epsilon-tolerant   :     24
  tie_exempt (0-sign):      2
  message_pair       :      0
```

Denominator: 2866 -> 3091 (+225). Numerator (`exact`): 669 -> 669
(unchanged). `absent`: 2197 -> 2422 (+225, exactly the new items -- none of
them are declared, so all 225 land in `absent`). `ion`, `untested`,
`phantom`, `failing`: 0 -> 0, unchanged. Coverage: 23.343% -> 21.643%.

**The denominator going up and coverage going down is the number getting
more honest, not the project regressing.** `RUST-QUEUE.md`'s standing rule:
*"A coverage number that only ever rises is not being audited."*

### Zero declarations renamed (mandatory verification)

`ionp.__ion_state__` was captured before this change (669 keys) and after
(669 keys); the symmetric difference between the two sorted key sets is
empty (`set()`). This is guaranteed by construction, not just observed: the
edit touched only `tools/snapshot_surface.py` (measurement tooling) and
regenerated `tools/numpy_surface.json`; `ionp/_state/*.py` and every other
file under `ionp/` were not touched, so `__ion_state__` could not have
changed. It was still re-measured directly rather than assumed, per the
HOLE doc's explicit instruction not to assume this property carries.

Separately, a set-diff of `tools/numpy_surface.json` before vs. after (every
flat name plus every exploded `<class>.<method>` key) shows 0 removed keys
and 225 added keys -- no existing item key was renamed, moved, or dropped.

```
before total 2866  after total 3091
removed (should be empty): 0
added: 225
```

## Task #26

No entry named "#26" exists in `RUST-QUEUE.md` or elsewhere in this repo as
of this commit; if it refers to an external tracker, that system should be
updated separately. Within this repo, the polynomial submodule hole
identified in `docs/DENOMINATOR-SUBMODULE-HOLE-2026-08-06.md` is CLOSED by
this change.
