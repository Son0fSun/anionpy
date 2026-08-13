# TICKET-MA-PHASE5: eight `ma.MaskedArray` layout/metadata attrs -- 2026-08-08

## Scope

Task brief: grind down `numpy.ma` coverage gaps (214 `ma.*` items measured
absent, the second-largest gap cluster in the repo). Chosen sub-cluster:
eight `ma.MaskedArray` instance attributes/methods that are plain read-only
mirrors of `self._data` -- `.data`, `.dtype`, `.mask`, `.itemsize`,
`.nbytes`, `.strides`, `.iscontiguous()`, `.get_fill_value()`.

Rationale for this sub-cluster over the other 127 `ma.MaskedArray.*`
absent items: most of those need either new Rust reduction/sort kernels,
or resolution of the `.flags`/`.strides` layout-observability gap before
anything downstream of them can honestly claim layout parity. This
sub-cluster is both real (adds genuine, previously-undeclared coverage)
and finishable in one pass without new Rust work -- `itemsize`/`nbytes`/
`strides` delegate to `anionpy.ndarray`'s own already-declared-exact
properties (`anionpy/_state/ndarray.py`); `data`/`dtype`/`mask` were
already-implemented Phase-0 properties in `anionpy/ma/core.py` that had
simply never been declared as their own manifest items.

## Method

Corpus-first, per this task's method requirement. Ground truth established
against real numpy 2.5.1 with two throwaway probe scripts run from
`/private/tmp/madata/` (`probe1_numpy.py`, real-numpy-only; `probe2_compare.py`,
both packages) across a wide provenance matrix: 10 dtypes x
{nomask, fully-masked, partially-masked}, empty, 0-d (masked/unmasked),
explicit `fill_value=`, F-order, transposed view, negative-stride slice.
Only after that matrix passed by hand did I touch `anionpy/ma/core.py`,
then wrote the permanent differential corpus, then re-verified through the
project's real harness, then declared.

## Implementation

`anionpy/ma/core.py`: eight new members inserted directly after the
existing `size` property. `itemsize`/`nbytes`/`strides` are one-line
delegations to `self._data.<x>` (verified live to depend ONLY on the data
array, never the mask array's own itemsize/nbytes -- checked against a
partially-masked int32 receiver where `.nbytes` reported the DATA-only
byte count). `data`/`dtype`/`mask` were already implemented. `iscontiguous()`
reads only `self._data.flags["C_CONTIGUOUS"]`. `get_fill_value()` is a
zero-arg alias for the already-exact `.fill_value` property (matches real
numpy's own `MaskedArray.get_fill_value` source: `return self.fill_value`,
nothing else). No Rust changed -- `cargo build -p ionp-py --release`
confirmed green before commit; `maturin develop --release` confirmed
`Installed anionpy-0.1.0`.

## `.flags` -- deliberately NOT implemented or declared

Real numpy's `MaskedArray` is an `ndarray` SUBCLASS, so `a.flags.owndata`
is `False` for literally every real MaskedArray regardless of provenance
(verified live: fresh construction, view construction, F-order all report
`owndata=False` uniformly) -- an artifact of `.data`/`.mask` always being
reached through an extra `.view()` layer in real numpy's implementation.
This is independently corroborated by pre-existing, uneditable reference
code: `tests/differential/ma_cases.py`'s `_layout_snapshot_np` function
documents the identical finding in its own comment. anionpy's
`MaskedArray` is a plain `(data, mask)` pair, not an `ndarray` subclass --
`self._data` is genuinely OWNED, so a naive `return self._data.flags`
would mismatch `owndata` on every single case with no honest fix short of
fabricating a hand-built flags-like object with a hardcoded
`owndata=False`, which this task's instrument-discipline rules forbid.
Declining to implement `.flags` is treated as a success, not a shortfall:
`iscontiguous()` is the safe substitute added instead, since
`C_CONTIGUOUS` was independently verified to NOT depend on the
owndata/view distinction.

## Differential corpus

New module (per file-ownership rules, `tests/differential/registry.py` and
`ma_cases.py` were not touched): `tests/differential/ma_phase5_cases.py`,
exporting `MA_PHASE5_SPECS: dict[str, ItemSpec]`. 48 cases per item (10
dtypes x 3 mask states, 4 empty-dtype, 6 0-d, 2 explicit-fill_value, 6
non-C-contiguous layout cases), run standalone through the project's real
`run.py:run_registry()` (not just ad-hoc probes). All 8 items now pass
48/48.

One genuine self-inflicted bug found and fixed along the way, in the test
harness wiring, not in `anionpy`: the first draft's adapter functions had
signature `(label, payload, **kw)`, but `harness.run_case` calls adapters
as `np_fn(*args, **kwargs)` where `args=(payload,)` only -- `label` is
bookkeeping and never passed into the adapter call (confirmed by reading
`run.py`'s `_build_cases_for_kind` kind="custom" branch and
`harness.run_case`'s `_call(np_fn, np_args, np_kwargs)`). This produced a
uniform `TypeError: ...adapter() missing 1 required positional argument`
on both sides for all 48 cases x 8 items -- an instance of the "uniform
result across a cross-product is a likely instrument artifact" rule from
this task's instrument-discipline requirements. Fixed by dropping `label`
from the adapter signatures and moving the layout corpus's mask-hint
signal into a `kw["__mask_hint__"]` key instead (popped before the real
`np.ma.array`/`anionpy.ma.MaskedArray` constructor call, which knows
nothing about it).

A second, genuine (not self-inflicted) bug turned up once the harness
wiring was fixed: 2/48 cases failed on `.iscontiguous`/`.strides` for the
"F-order 2x2" layout case, with anionpy reporting `C_CONTIGUOUS=True`/
strides `(16, 8)` against numpy's `False`/`(8, 16)`. Root cause was in the
corpus, not `anionpy`: the F-order receiver was built via
`np.asfortranarray(...)` on the numpy side but `anionpy.array(...).reshape(
(2,2), order="F")` on the anionpy side -- NOT equivalent constructions.
Live-checked: numpy's OWN `.reshape((2,2), order="F")` on an
already-(2,2)-shaped source gives C-contiguous strides `(16, 8)` (matches
what anionpy gave), because reshape-with-order only changes the
read/write traversal order, not memory layout, when the shape doesn't
change. `anionpy.array(..., order="F")` (construction-time order, not
reshape-time) was verified to give the genuinely F-contiguous
`strides=(8, 16)` matching `np.asfortranarray`. Fixed by using
`array(..., order="F")` on both sides in `_layout_cases_np`. Re-ran: 48/48
on all 8 items.

## REGISTRY collision check

Performed by importing the fully-wired `run` module (which imports every
`*_cases.py` side-effect module including `ma_cases.py`) and diffing
`MA_PHASE5_SPECS`'s 8 keys against the live 2058-entry `REGISTRY`. Zero
collisions. `registry.py` itself was not edited (off-limits, concurrent
agent). Merge block for the orchestrator:

```python
from ma_phase5_cases import MA_PHASE5_SPECS  # noqa: E402

_MA_PHASE5_NEW = (
    "ma.MaskedArray.data",
    "ma.MaskedArray.dtype",
    "ma.MaskedArray.mask",
    "ma.MaskedArray.itemsize",
    "ma.MaskedArray.nbytes",
    "ma.MaskedArray.strides",
    "ma.MaskedArray.iscontiguous",
    "ma.MaskedArray.get_fill_value",
)
_missing_new = [n for n in _MA_PHASE5_NEW if n not in MA_PHASE5_SPECS]
if _missing_new:
    raise AssertionError(f"ma_phase5_cases.py missing expected specs: {_missing_new}")
_ma_phase5_collisions = set(_MA_PHASE5_NEW) & set(REGISTRY)
if _ma_phase5_collisions:
    raise AssertionError(f"ma-phase5 name collision with existing REGISTRY: {_ma_phase5_collisions}")
REGISTRY.update({n: MA_PHASE5_SPECS[n] for n in _MA_PHASE5_NEW})
```

(No deferred/uncovered-item check is needed here -- unlike the
`finfo_iinfo_cases.py` example this pattern is drawn from,
`MA_PHASE5_SPECS` contains exactly the 8 items being merged, nothing
held back.)

## Declarations

`anionpy/_state/ma.py`'s `MA_STATE` dict, inserted after the existing
`"ma.MaskedArray.compressed": "exact"` entry (verified `MA_STATE` is
actually loaded: `anionpy/_state/__init__.py` line 46 imports it and
folds it into the state used by `anionpy.__ion_state__`):

```python
"ma.MaskedArray.data": "exact",
"ma.MaskedArray.dtype": "exact",
"ma.MaskedArray.mask": "exact",
"ma.MaskedArray.itemsize": "exact",
"ma.MaskedArray.nbytes": "exact",
"ma.MaskedArray.strides": "exact",
"ma.MaskedArray.iscontiguous": "exact",
"ma.MaskedArray.get_fill_value": "exact",
```

`tools/coverage.py --tests <report> --list absent | grep -c '^ *ma\.'`
confirmed to drop from 214 to 206 (exactly 8) after this change, with a
freshly generated report (`tests/differential/run.py --out report.json`,
2058 items, 33 non-`ma` pre-existing failures unrelated to this task
(`__ipow__`, `finfo`/`iinfo`, `matrix.__setitem__`, `hypot`,
`float_power`, etc. -- unaffected, unrelated concurrent-work baseline, not
introduced by this change), 0 `ma.*` failures/regressions).

## Declined items

- **`ma.MaskedArray.flags`** -- see above. Genuine, reproducible semantic
  gap (owndata-always-False in real numpy vs genuinely-owned data in
  anionpy), not thin evidence. Not implemented, not declared.

No other items in this batch were declined -- all 8 chosen items verified
cleanly both via ad-hoc probe and via the real differential harness.

## Contradictions found relative to the original brief

- **Doc path**: brief cited `docs/MA-DESIGN.md`; the actual file is at
  the repo root, `~/Monday/ionp/MA-DESIGN.md`.
- **"Phases 5-6 design-only" claim**: `MA-DESIGN.md` documents phases 0-6
  (~108 items) as ALREADY IMPLEMENTED and declared, not merely designed.
  The brief's premise that "phases 5-6 are already designed" undersold
  the actual state -- most of the design doc's scope is done; the 214
  absent count is real but is not phases-5-6-shaped, it's scattered
  across `ma.MaskedArray.*` sub-items (135 of the 214) plus module-level
  functions.
- **Ticket numbers #59/#60/#63**: the brief cited #59 (NEP-50 weak
  promotion for ma dunders), #60 (`__rand__` deferral TypeError), #63
  (MaskedArray missing `.strides`/`.flags`, layout-unverifiable) as
  specific, findable prior tickets. Searched `PATH-TO-100.md`,
  `docs/TICKET-64-REMEASURE-2026-08-08.md`, `docs/failing-93-clusters.md`
  and found DIFFERENT #59/#60/#61 entries in this repo (about
  instrument/probe validation, unrelated to ma/NEP-50/`__rand__`). Could
  not locate the brief's specific ma-related ticket text anywhere in the
  repo. However, the SUBSTANCE behind "#63" (the `.flags`/`.strides`
  layout-observability gap) is real and directly relevant -- independently
  re-derived via probing in this task and separately corroborated by
  pre-existing code in `ma_cases.py`. Flagging the ticket-number mismatch
  as a contradiction rather than silently absorbing it.

## Commit

`anionpy/ma/core.py`, `anionpy/_state/ma.py`,
`tests/differential/ma_phase5_cases.py`, this file -- committed by
explicit absolute path, `registry.py` untouched (off-limits).
