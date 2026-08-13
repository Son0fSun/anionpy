"""Coverage declarations for the `memmap` curated-explode block.

CORRECTS THE PRIOR RULING in `anionpy/_state/matrix.py` (2026-08-07), which
recorded `memmap` as "100% structurally blocked: no buffer-protocol /
frombuffer / view construction path at all... there is no way to build an
anionpy object that shares a real file-backed mmap buffer... zero items
declared, zero items attempted." That measurement of the missing
buffer-sharing path was correct; the conclusion drawn from it was
overbroad. Genuine cross-process buffer ALIASING is only required to
reproduce live cross-object write visibility without an explicit
`flush()`/reopen -- NOT to reproduce `memmap`'s VALUE/SHAPE/DTYPE/TYPE
contract, which is what every declaration below actually measures. `matrix`
shipped ~90 declared items under the identical missing-buffer-sharing
limitation by matching that same contract on a freshly-built (not aliased)
array; `memmap` does the same here, using Python's stdlib `mmap` + `struct`
modules for real OS-level file mapping and genuine byte encode/decode --
zero calls into real numpy, zero new Rust. See `anionpy/memmap.py`'s module
docstring for the full construction mechanics and `tests/differential/
memmap_cases.py` for the corpus. Originally 918 cases across 68 items, run
standalone since `registry.py` was off-limits that task; ticket #77
(2026-08-08) extended the corpus (dtype/mode spelling variation,
invalid_dtype) and wired it into `registry.py` via a full
`REGISTRY.update(MEMMAP_SPECS)` merge -- see the TICKET #77 section below.

MEASURED SPLIT (2026-08-08): coverage.py --list absent | grep '^ *memmap\\.'
-> 167 absent items. Own-vs-inherited (attr in vars(np.memmap) vs
hasattr-only): 8 own (`__new__`, `__array_finalize__`, `__array_wrap__`,
`__array_priority__`, `__getitem__`, `__setitem__` [via `_do_not_call_view`
guard machinery], `flush`, `_mmap`/`filename`/`offset`/`mode` property
group), 159 inherited from `ndarray`. Governing rule followed throughout:
mirror where NUMPY binds the name -- own-defined items are bound directly
in `vars(anionpy.memmap)` (true class-dict entries, not copied-down
inherited members); items numpy itself only inherits from `ndarray` are
declared here ONLY where `anionpy.memmap` genuinely implements the same
method itself (composition-wrapper delegation, since `anionpy.ndarray`
cannot be subclassed) -- never fabricated purely to move the ledger.

TYPE-PRESERVING RULE (measured live against real numpy 2.5.1, encoded in
`anionpy/memmap.py` and verified by every case below): `__array_priority__
== -100.0` downgrades every arithmetic/comparison/reduction result to a
plain `ndarray`. View-shaped ops (`T`, `transpose`, `swapaxes`, `squeeze`,
`__getitem__`) preserve `memmap`-ness and propagate `filename`/`mode`/
`offset`/`.base`. `reshape`/`ravel` preserve `memmap`-ness AND
`filename`/`mode`/`offset` only when the operation is achievable without a
data copy (source array is C-contiguous for the default order='C' call);
otherwise they behave like `.copy()`/`.astype()` (filename/mode/offset ->
None) -- EXCEPT `reshape` specifically still sets `.base` to the source
object even when it copied (a narrow, separately-measured divergence from
`ravel`/`flatten`/`copy`/`astype`, all four of which reset `.base` to None
on a copy -- see `anionpy/memmap.py::reshape`'s own comment). `flatten` is
unconditionally a copy (matches numpy's own documented contract), always
resetting filename/mode/offset/base.

TICKET #77 (2026-08-08): "memmap.__new__": "exact" was FALSE. The 918-case
corpus below varied dtype/mode IDENTITY (which of 14 families / 4 short
mode forms) but never SPELLING -- every case passed dtype/mode as the exact
canonical strings `__new__`'s own validation expected, so it could not see
that `__new__` rejected `np.dtype(...)` instances, builtins (`float`/`int`/
`bool`/`complex`), numpy scalar types, char codes (`'f8'`/`'q'`/...), and
`anionpy.dtype(...)` objects for `dtype=`, and rejected numpy's 4 long-form
mode aliases (`'readonly'`/`'copyonwrite'`/`'readwrite'`/`'write'`) for
`mode=`. Fixed in `anionpy/memmap.py`: `dtype` is now normalized via the
already-exact `ap.dtype(x).name` (matching numpy's own parse/reject
behavior, including TypeError text, for every spelling it can parse); `mode`
now accepts all 8 forms and normalizes long -> short, and the invalid-mode
ValueError now reproduces numpy's exact message (all 8 forms enumerated).
`tests/differential/memmap_cases.py` extended with `_dtype_spelling_cases`/
`_mode_spelling_cases` (dtype/mode SPELLING held-value-fixed variation) plus
two new negative-control items, `memmap.__new__/invalid_mode` (message text
now matches, moved from DECLINED to DECLARED) and
`memmap.__new__/invalid_dtype` (genuinely-unparseable dtypes, TypeError text
matches via `ap.dtype`). `registry.py` was free this task (no concurrent
owner since the ma-phase5 merge) -- `memmap_cases.py` is now wired in via a
full, collision-checked `REGISTRY.update(MEMMAP_SPECS)` merge, so these
declarations are live-graded by `tests/differential/run.py`, not just the
standalone corpus.

Extending the corpus to vary spelling also surfaced a genuine, unrelated
`__repr__` divergence -- see BUGS FOUND #4 and the DECLINED section below;
`memmap.__repr__` was WITHDRAWN rather than shipped false.

DECLARED (70 items: 69 pre-existing minus the withdrawn `__repr__` plus 2
new ticket-#77 items, every one verified by `tests/differential/
memmap_cases.py`'s corpus (918 base cases + spelling-variation and
invalid_dtype cases added 2026-08-08), now live via `registry.py`):

  Construction / own-defined (4): __new__, __array_priority__, __getitem__,
  flush. `__new__`'s byte-accounting (offset/shape/mode/ALLOCATIONGRANULARITY
  rounding, the 1-byte pad numpy applies for empty 'w+'/'r+' memmaps per
  numpy's own gh-27723 fix) is a direct port of real numpy's
  `_core/memmap.py` source (read via `inspect.getsourcefile`), not a
  re-derived approximation -- ported specifically because the corpus's
  `memmap.__new__/empty_file` case caught the wrong initial guess (see
  BUGS FOUND below). `flush`'s write targets `_array_offset` (the position
  inside the ALLOCATIONGRANULARITY-rounded `mmap.mmap` mapping), not the
  raw constructor `offset=`, verified against an unaligned large-offset
  round-trip case beyond the ones in the wired corpus
  (`/private/tmp` probe, offset = 2*ALLOCATIONGRANULARITY + 37).

  Plain attribute passthrough (7): dtype, shape, ndim, size, itemsize,
  nbytes, base.

  View-preserving structural (7): T, transpose, reshape, ravel, flatten,
  squeeze, swapaxes (__getitem__ is also view-preserving but is counted
  once, in the own-defined bucket above).

  Copy-resetting (2): copy, astype.

  Reductions (9): sum, mean, min, max, std, var, prod, all, any -- all
  downgrade to plain `ndarray`/scalar per the priority rule; grading target
  is the VALUE, which is unaffected by the downgrade.

  Arithmetic/comparison/bitwise dunders + reflected forms (30): __add__,
  __radd__, __sub__, __rsub__, __mul__, __rmul__, __mod__, __rmod__,
  __floordiv__, __rfloordiv__, __eq__, __ne__, __lt__, __le__, __gt__,
  __ge__, __and__, __rand__, __or__, __ror__, __xor__, __rxor__,
  __lshift__, __rlshift__, __rshift__, __rrshift__, __neg__, __pos__,
  __abs__, __invert__.

  Misc (9): __len__, __iter__, __bool__, __hash__ (unhashable, matches
  real numpy's `TypeError`), tolist, tobytes, item, fill, __setitem__.
  (__repr__ withdrawn 2026-08-08, see DECLINED.)

  Construction negative controls (2, added ticket #77 2026-08-08):
  __new__/invalid_mode, __new__/invalid_dtype.

  (4 + 7 + 7 + 2 + 9 + 30 + 9 + 2 = 70; __getitem__ is counted once, in the
  own-defined bucket, though it is also exercised by the view-preserving
  corpus.)

BUGS FOUND AND FIXED BY THE CORPUS (this is the corpus doing its job --
each was a real, measured divergence from real numpy 2.5.1, not a
tolerance-widening or a weakened test):
  1. `__repr__`: padded EVERY continuation line of a >=3-D array's repr,
     including numpy's own genuinely-blank inter-group separator lines --
     inserted trailing whitespace real numpy's `memmap` repr never has.
     Fixed: only pad non-empty lines.
  2. `reshape`/`ravel`/`flatten` on a non-C-contiguous source (e.g. an
     F-order-constructed memmap reshaped to a flat shape) always took the
     view-preserving path, keeping filename/mode/offset non-None where
     real numpy resets them to None (a copy was actually required). Fixed
     via `_needs_copy_for_c_order` (checks `self._data.flags
     ['C_CONTIGUOUS']`, not a converted copy) plus the `reshape`-specific
     `.base`-still-points-to-source narrow case documented above.
  3. `__new__` unconditionally raised `ValueError: cannot mmap an empty
     file` for ANY zero-byte-content construction. Real numpy raises this
     only when mode is 'r'/'c' AND the file is genuinely still 0 bytes
     after construction; for 'w+'/'r+' it pads a 1-byte hole (numpy's own
     gh-27723 fix) so an empty memmap IS constructible. The full
     `__new__` byte-accounting was ported from real numpy's source (see
     above) to fix this correctly rather than patching the single
     observed case.
  4. (ticket #77, 2026-08-08) `__new__` accepted only 14 canonical dtype
     name strings and 4 short mode forms, rejecting every other spelling
     real numpy accepts for `dtype=`/`mode=` -- see the TICKET #77 section
     above for the full description and fix. Discovered because the
     original 918-case corpus varied dtype/mode IDENTITY but never
     SPELLING, so it could not see a parser that only recognized one
     notation per value.

DECLINED (measured, not guessed):
  - `memmap.__repr__`: WITHDRAWN 2026-08-08 (was declared exact, now is
    not). The dtype-spelling-variation corpus added by ticket #77 caught a
    genuine, pre-existing, memmap-independent divergence: real numpy's
    `dtype('q') is not dtype('int64')` despite `==`-equality, and real
    numpy's own repr logic keys off that IDENTITY distinction -- an array
    constructed with `dtype='q'` reprs with an explicit `dtype=int64`
    suffix; one constructed with `dtype='int64'` omits it. `anionpy`'s
    array repr does not reproduce this (reproduces on plain
    `anionpy.array(..., dtype='q')` too, confirmed live -- not a
    `memmap.py` bug). Kept in the corpus (`memmap.__repr__`, still run over
    every dtype spelling) for the record; withdrawn rather than shipped
    false. Out of scope for this ticket to fix (base `anionpy.array` repr,
    not `memmap`).
  - `memmap.__new__/unsupported_dtype` (added ticket #77, 2026-08-08):
    dtypes real numpy's `np.dtype()` CAN parse but `anionpy.memmap` has no
    `_DTYPE_FMT`/`_decode`/`_encode` support for at all -- string (`S`/`U`),
    void (`V`), `datetime64`, and `object` kinds. Real numpy SUCCEEDS
    constructing these (memmap does no dtype validation beyond calling
    `np.dtype()`); `anionpy.memmap` cannot back them structurally. Kept in
    the corpus for the record (both sides always raise, so the corpus
    itself is green, but the raised TYPE is not compared since the
    divergence is expected); NOT declared -- pre-existing, permanent,
    narrower-than-numpy gap, unrelated to the parsing defect this ticket
    fixed.
  - `view`, `dot`, `diagonal`, `sort`, `argsort`, `searchsorted`,
    `__pow__`/`__rpow__`, `__truediv__`/`__rtruediv__`,
    `__matmul__`/`__rmatmul__`, `__divmod__`/`__rdivmod__`: no working
    `anionpy.ndarray` base to delegate to (`view` raises `AttributeError`
    outright on plain `anionpy.ndarray`; the arithmetic ones are
    independently-tracked base-level gaps). Declaring these on `memmap`
    without a working base is exactly the fabricated-class-dict-entry
    failure this task was warned against -- not attempted.
  - `__array_interface__`, `__array_struct__`, `__array_finalize__`,
    `__array_wrap__`, `__buffer__`, `__dlpack__`/`__dlpack_device__`:
    numpy-internal ufunc/buffer-protocol machinery `anionpy.ndarray` does
    not implement at all.
  - `__reduce__`/`__setstate__`/`__getstate__`/pickling family: not
    implemented, no measured contract to pin.
  - Pure Python-mechanical introspection names (`__class__`, `__module__`,
    `__dir__`, `__static_attributes__`, `__firstlineno__`, ...): excluded
    for the same reason every other `*_cases.py` in this package excludes
    them -- not "memmap behavior".
"""
from __future__ import annotations

MEMMAP_STATE: dict[str, str] = {
    # -- own-defined (mirrors `attr in vars(np.memmap)`) --
    "memmap.__new__": "exact",
    "memmap.__array_priority__": "exact",
    "memmap.__getitem__": "exact",
    "memmap.flush": "exact",

    # -- plain attribute passthrough (inherited from ndarray) --
    "memmap.dtype": "exact",
    "memmap.shape": "exact",
    "memmap.ndim": "exact",
    "memmap.size": "exact",
    "memmap.itemsize": "exact",
    "memmap.nbytes": "exact",
    "memmap.base": "exact",

    # -- view-preserving structural (inherited from ndarray) --
    "memmap.T": "exact",
    "memmap.transpose": "exact",
    "memmap.reshape": "exact",
    "memmap.ravel": "exact",
    "memmap.flatten": "exact",
    "memmap.squeeze": "exact",
    "memmap.swapaxes": "exact",

    # -- copy-resetting (inherited from ndarray) --
    "memmap.copy": "exact",
    "memmap.astype": "exact",

    # -- reductions (inherited from ndarray; all downgrade to plain
    #    ndarray/scalar per __array_priority__, value is the grading
    #    target and is unaffected) --
    "memmap.sum": "exact",
    "memmap.mean": "exact",
    "memmap.min": "exact",
    "memmap.max": "exact",
    "memmap.std": "exact",
    "memmap.var": "exact",
    "memmap.prod": "exact",
    "memmap.all": "exact",
    "memmap.any": "exact",

    # -- arithmetic / comparison / bitwise dunders + reflected forms
    #    (inherited from ndarray; every result downgrades to plain
    #    ndarray per __array_priority__) --
    "memmap.__add__": "exact",
    "memmap.__radd__": "exact",
    "memmap.__sub__": "exact",
    "memmap.__rsub__": "exact",
    "memmap.__mul__": "exact",
    "memmap.__rmul__": "exact",
    "memmap.__mod__": "exact",
    "memmap.__rmod__": "exact",
    "memmap.__floordiv__": "exact",
    "memmap.__rfloordiv__": "exact",
    "memmap.__eq__": "exact",
    "memmap.__ne__": "exact",
    "memmap.__lt__": "exact",
    "memmap.__le__": "exact",
    "memmap.__gt__": "exact",
    "memmap.__ge__": "exact",
    "memmap.__and__": "exact",
    "memmap.__rand__": "exact",
    "memmap.__or__": "exact",
    "memmap.__ror__": "exact",
    "memmap.__xor__": "exact",
    "memmap.__rxor__": "exact",
    "memmap.__lshift__": "exact",
    "memmap.__rlshift__": "exact",
    "memmap.__rshift__": "exact",
    "memmap.__rrshift__": "exact",
    "memmap.__neg__": "exact",
    "memmap.__pos__": "exact",
    "memmap.__abs__": "exact",
    "memmap.__invert__": "exact",

    # -- misc (inherited from ndarray) --
    "memmap.__len__": "exact",
    "memmap.__iter__": "exact",
    "memmap.__bool__": "exact",
    "memmap.__hash__": "exact",
    # memmap.__repr__: WITHDRAWN (ticket #77, 2026-08-08) -- see module
    # docstring DECLINED section. Real divergence for dtypes constructed via
    # a char code / other spelling that is `==` but not `is` numpy's
    # canonical dtype object (e.g. 'q' vs 'int64'): numpy's repr shows an
    # explicit `dtype=int64` in that case, anionpy's does not.
    "memmap.tolist": "exact",
    "memmap.tobytes": "exact",
    "memmap.item": "exact",
    "memmap.fill": "exact",
    "memmap.__setitem__": "exact",

    # -- construction negative controls (ticket #77, 2026-08-08) --
    "memmap.__new__/invalid_mode": "exact",
    "memmap.__new__/invalid_dtype": "exact",
}
