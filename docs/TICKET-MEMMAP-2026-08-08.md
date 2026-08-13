# TICKET-MEMMAP-2026-08-08: `anionpy.memmap`

## Task

Implement `numpy.memmap` (file-backed `ndarray` subclass) coverage.
`registry.py` was off-limits (a concurrent agent was editing it), so the
differential corpus lives in a NEW, unwired module
(`tests/differential/memmap_cases.py`) with a paste-ready merge block for a
human to apply once that file is free.

## Contradiction found and overridden

`anionpy/_state/matrix.py` (committed 2026-08-07) ruled `memmap` "100%
structurally blocked... there is no way to build an `anionpy` object that
shares a real file-backed mmap buffer, which is `memmap`'s entire reason to
exist... Zero items declared, zero items attempted."

That measurement (no `frombuffer`/buffer-aliasing constructor;
`anionpy.array()` always copies) is correct. The conclusion is overbroad:
genuine buffer ALIASING is only required to reproduce live cross-object
write visibility without an explicit `flush()`/reopen. It is NOT required
to reproduce `memmap`'s VALUE/SHAPE/DTYPE/TYPE contract, which is what
every declared item here actually measures — exactly the same situation
`anionpy.matrix` was already in (also no buffer-sharing path) when it
shipped ~90 declared items. `anionpy/memmap.py` sidesteps the gap using
Python's stdlib `mmap` + `struct` modules: real OS-level file mapping,
real byte encode/decode, zero calls into real numpy, zero new Rust.

This is reported as instructed: "I would rather hear a contradiction than
have you comply with a wrong instruction."

## Measurement (before implementing)

```
./.venv/bin/python tools/coverage.py --tests <report> --list absent \
  | grep '^ *memmap\.' | wc -l
```
-> **167 absent items**.

Own-vs-inherited split (`attr in vars(np.memmap)` vs `hasattr`-only):

- **8 own**: `__new__`, `__array_finalize__`, `__array_wrap__`,
  `__array_priority__`, `__getitem__`, `flush`, plus the
  `filename`/`offset`/`mode` property group.
- **159 inherited** from `ndarray`.
- 0 unresolvable.

Governing rule followed: mirror where NUMPY binds the name. Own-defined
items are bound directly in `vars(anionpy.memmap)`. Inherited items are
declared here only where `anionpy.memmap` genuinely implements the same
method itself via composition delegation (required because
`anionpy.ndarray` cannot be subclassed from Python — verified live,
`TypeError: type 'anionpy.ndarray' is not an acceptable base type`) — never
copied into the class dict purely to move the ledger.

## Implementation

`anionpy/memmap.py` (new, ~470 lines): composition wrapper holding
`self._data: anionpy.ndarray`, same pattern as `anionpy.matrix` (ticket
#66) and `anionpy.ma.MaskedArray`. `__new__` ports real numpy's own
`_core/memmap.py` byte-accounting algorithm (offset/shape/mode,
`ALLOCATIONGRANULARITY` rounding, the 1-byte pad numpy applies for empty
`'w+'`/`'r+'` memmaps per numpy's own `gh-27723` fix) directly from its
source, read via `inspect.getsourcefile(numpy.memmap)` — not a
re-derived approximation. `flush()` writes through the OS-backed `mmap.mmap`
object at `_array_offset` (the position inside the granularity-rounded
mapping), reads back with `struct`-based decode, and calls `mmap.flush()`.
No numerical loop exists anywhere in the file — `_decode`/`_encode` are
byte (de)serialization via `struct`, not computation; Rust core is
untouched (there is nothing arithmetic in this item — it is a
construction/dispatch wrapper).

Wired into `anionpy/__init__.py` (one import line, matching the
`matrix`/`asmatrix` line immediately above it — `memmap` is likewise not
added to `__all__`, consistent with that existing convention).

### Type-preserving rule (measured live against real numpy 2.5.1)

`__array_priority__ == -100.0` downgrades every arithmetic/comparison/
reduction result to a plain `ndarray`. View-shaped ops (`T`, `transpose`,
`swapaxes`, `squeeze`, `__getitem__`) preserve `memmap`-ness and propagate
`filename`/`mode`/`offset`/`.base`. `reshape`/`ravel` preserve `memmap`-ness
and `filename`/`mode`/`offset` only when achievable without a data copy
(source is C-contiguous for the default `order='C'` call); otherwise they
reset filename/mode/offset to `None` like `.copy()`/`.astype()` — **except**
`reshape` specifically still sets `.base` to the source object even when it
copied (measured live: real numpy's copying `reshape` goes through a
distinct "new-from-template" construction path from `ravel`'s own copy
path, and tags provenance even though the buffer was copied — a narrow,
specifically-measured divergence from `ravel`/`flatten`/`copy`/`astype`,
all four of which reset `.base` to `None` on a copy). `flatten` is
unconditionally a copy (matches numpy's documented contract).

## Corpus

`tests/differential/memmap_cases.py` (new, unwired): 69 `ItemSpec`s
(`kind="custom"`, `scalar_like=True`, explicit `numpy_adapter`/
`ionp_adapter`), **918 total cases**, run standalone via
`/private/tmp/run_memmap_cases.py` (calls `harness.evaluate` directly per
spec, bypassing `registry.py`/`run.py`'s REGISTRY-driven path entirely).
Crosses: all 14 dtypes (bool, every int/uint width, float16/32/64,
complex64/128), 1-D/2-D/3-D/0-D shapes, C/F order, modes `'r'`/`'r+'`/
`'w+'`/`'c'`, a nonzero offset, empty-file and invalid-mode negative
controls. Every case's two adapter functions build their OWN independent
temp file with byte-identical content (ground-truth bytes from real
numpy's `tofile()`, used only as a fixture generator — same convention
every other `*_cases.py` in this package already uses `import numpy as np`
for) — never a shared path or handle, so a mutating case (`__setitem__`/
`fill`/`flush`) on one side can never pollute what the other side reads.

**Merge block** (paste into `registry.py` once free):
```python
from memmap_cases import MEMMAP_SPECS
_collisions = set(MEMMAP_SPECS) & set(REGISTRY)
assert not _collisions, _collisions
REGISTRY.update(MEMMAP_SPECS)
```

## Bugs found and fixed by the corpus (real divergences, not tolerance widening)

1. **`__repr__`**: padded every continuation line of a >=3-D repr,
   including numpy's own genuinely-blank inter-group separator lines —
   inserted trailing whitespace real numpy's `memmap` repr never emits.
   Fixed: only pad non-empty lines.
2. **`reshape`/`ravel`/`flatten` on non-C-contiguous source**: always took
   the view-preserving path (kept filename/mode/offset), where real numpy
   resets them to `None` because a copy was actually required. Fixed via
   `_needs_copy_for_c_order()` (checks `self._data.flags['C_CONTIGUOUS']`
   directly, not a converted copy) plus the `reshape`-specific `.base`
   divergence documented above.
3. **`__new__` on an empty/zero-byte construction**: unconditionally raised
   `ValueError: cannot mmap an empty file`. Real numpy only raises this for
   `'r'`/`'c'` on a file that is STILL 0 bytes after construction; for
   `'w+'`/`'r+'` it pads a 1-byte hole (numpy's own `gh-27723` fix) so an
   empty memmap IS constructible. Root-caused by porting numpy's actual
   `_core/memmap.py` byte-accounting algorithm rather than patching the one
   observed case.
4. **`flush()`'s dtype key** (found before the corpus, during live
   probing): passed `root._data.dtype` (a dtype object) to `_encode`
   instead of `str(root._data.dtype)`, causing `KeyError`.

After all four fixes: **918/918 cases pass**, 0 failures, 0 invalid.

## Declared (69 items — see `anionpy/_state/memmap.py` for the full list
and per-bucket breakdown)

Own-defined (4): `__new__`, `__array_priority__`, `__getitem__`, `flush`.
Plain attributes (7): `dtype`, `shape`, `ndim`, `size`, `itemsize`,
`nbytes`, `base`. View-preserving structural (7): `T`, `transpose`,
`reshape`, `ravel`, `flatten`, `squeeze`, `swapaxes`. Copy-resetting (2):
`copy`, `astype`. Reductions (9): `sum`, `mean`, `min`, `max`, `std`,
`var`, `prod`, `all`, `any`. Arithmetic/comparison/bitwise dunders +
reflected forms (30). Misc (10): `__len__`, `__iter__`, `__bool__`,
`__hash__`, `__repr__`, `tolist`, `tobytes`, `item`, `fill`,
`__setitem__`.

Every one verified out of corpus (`918/918` standalone run) plus one
additional unaligned-large-offset roundtrip probe for `flush`
(`/private/tmp`, offset = 2×`ALLOCATIONGRANULARITY` + 37, not run through
the wired corpus but verified live).

## Declined (measured, not guessed)

- **`memmap.__new__` invalid-`mode=` message text**: real numpy's message
  enumerates more accepted aliases (`'readonly'`, `'copyonwrite'`,
  `'readwrite'`, `'write'`) in a different format. Both sides raise
  `ValueError` (class matches), but the text does not. Kept in the corpus
  (`memmap.__new__/invalid_mode`) for the record, not declared.
- `view`, `dot`, `diagonal`, `sort`, `argsort`, `searchsorted`,
  `__pow__`/`__rpow__`, `__truediv__`/`__rtruediv__`,
  `__matmul__`/`__rmatmul__`, `__divmod__`/`__rdivmod__`: no working
  `anionpy.ndarray` base to delegate to (`view` raises `AttributeError`
  outright; the arithmetic ones are independently-tracked base-level
  gaps, not memmap-specific). Not attempted — declaring these would be the
  fabricated-class-dict-entry failure this task warned against.
- `__array_interface__`, `__array_struct__`, `__array_finalize__`,
  `__array_wrap__`, `__buffer__`, `__dlpack__`/`__dlpack_device__`:
  numpy-internal protocol machinery `anionpy.ndarray` does not implement.
- Pickling family (`__reduce__`/`__getstate__`/`__setstate__`/...): not
  implemented, no measured contract.
- Pure Python-mechanical introspection names (`__class__`, `__module__`,
  `__dir__`, `__static_attributes__`, `__firstlineno__`, ...): excluded,
  same convention every other `*_cases.py` in this package follows.

Declining these is a success condition, not a shortfall.

## Ledger state (honest, by design)

`tools/coverage.py --tests <report>` shows the 69 declared items as
**"untested" (69), 0 phantom** — correctly NOT credited as `exact` yet,
because `memmap_cases.py` is not wired into `registry.py`/`run.py`'s live
suite (off-limits this task). `anionpy.__ion_state__` DOES contain all 69
keys (verified directly, `len([k for k in ap.__ion_state__ if
k.startswith('memmap.')]) == 69`) — the declarations are real and will
move the ledger to `exact` automatically the moment a human applies the
merge block above and re-runs `run.py`. This is the ledger's honesty gate
working as designed, not a bug.

## Verification performed

- `cargo build -p ionp-py --release` succeeds (pure-Python task; confirmed
  regardless per the hard constraint).
- Full differential suite (`tests/differential/run.py --out
  /tmp/rMemmap.json`) still passes for everything it already covered — 33
  pre-existing failures found, none `memmap.*`-prefixed (unrelated,
  concurrent-agent-owned areas: `__ipow__`/`__pow__`/`__rtruediv__`,
  `finfo`/`iinfo` `__new__`/`__init__`, `hypot`/`ldexp`/`isnat`/etc.).
- Standalone memmap corpus run: 918/918 cases pass.
- `anionpy.__ion_state__` confirmed to actually contain all 69 `memmap.*`
  keys after wiring.
