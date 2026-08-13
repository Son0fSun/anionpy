"""Coverage declarations for the `finfo`/`iinfo` block (ticket #71, 2026-08-08).

NEW FILE, owned by this task, per `anionpy/_state/__init__.py`'s split-file
collision-checked-merge convention (see that module's docstring). Backed by
`anionpy/_dtypeinfo.py` (`iinfo`/`finfo` classes) and
`ionp-py/src/dtypeinfo.rs` (`iinfo`/`finfo` pyfunctions, unchanged this
ticket) + `ionp-core/src/dtype.rs` (`FInfo`/`IInfo` constant tables,
unchanged this ticket). Differential coverage lives in the NEW, NOT-YET-
WIRED module `tests/differential/finfo_iinfo_cases.py` -- see that file's
header and this ticket's doc (`docs/TICKET-71-FINFO-2026-08-08.md`) for the
exact `registry.py` merge instruction, since `tests/differential/registry.py`
itself is owned by a concurrent agent and was not touched here.

WHAT THIS TICKET ACTUALLY CHANGED (all in `anionpy/_dtypeinfo.py`, no Rust):
  1. Renamed the `IInfo` class to `iinfo` (numpy's real class name; the
     Python-facing symbol was previously exported under the wrong name).
  2. Added `__str__` to both `iinfo` and `finfo` (previously absent on
     both, so `str(x)` fell through to `object.__str__` -> `repr(x)`,
     diverging from numpy's real multi-line "Machine parameters for ..."
     block). Byte-for-byte matched against numpy's own `get_str`/`__str__`
     source in `numpy/_core/getlimits.py`, including numpy's own dead-code
     detail that the `pad` argument to `get_str` is inert (`get_str`
     always `return str(val)`, never the padded form) -- reproduced
     faithfully rather than "improved".
  3. THE STRUCTURAL FIX THIS BLOCK IS ABOUT: numpy's `iinfo`/`finfo`
     classes store their COMPUTED attributes as class-level descriptors
     (`iinfo.min`/`iinfo.max` are `@property`; `finfo.epsneg`/`iexp`/
     `machep`/`negep`/`nexp`/`resolution`/`tiny` are `@cached_property`),
     confirmed by direct inspection of `numpy._core.getlimits.iinfo`'s and
     `.finfo`'s `__dict__`. `tools/coverage.py`'s ledger `resolve()`
     checks presence with `attr in vars(cls)` -- a genuine class-level
     descriptor, NOT merely an instance attribute set in `__init__`.
     anionpy previously stored ALL of these as plain `self.<name> = ...`
     instance attributes, so all 9 items (`iinfo.min`, `iinfo.max`,
     `finfo.epsneg`, `.iexp`, `.machep`, `.negep`, `.nexp`, `.resolution`,
     `.tiny`) were structurally `absent` from the ledger regardless of
     runtime correctness. Fixed by moving exactly those 9 to `@property`
     backed by private storage (`_min`/`_max`/`_epsneg`/`_iexp`/`_machep`/
     `_negep`/`_nexp`/`_resolution`), matching numpy's STORED-vs-COMPUTED
     class shape field-for-field. `finfo.tiny` specifically is implemented
     as `return self.smallest_normal` (a read-through alias, matching
     numpy's literal implementation -- numpy's own docstring says "alias
     of smallest_normal" -- not a separately-stored duplicate value).
     Verified post-fix via `tools.coverage.resolve()` called directly:
     all 9 items now resolve `present=True` against a live `anionpy`
     import, matching numpy's `True` (previously `False` for all 9).

MEASUREMENT (out-of-corpus, `tests/differential/registry.py` unavailable to
this task -- probe script only, real numpy, /private/tmp, not shipped):
`/private/tmp/finfo_probe/full_probe.py`, 839 checks, run against the CURRENT
built `.so` both before and after this ticket's edit (rerun after to confirm
zero regressions from the property-conversion refactor -- result unchanged
both times, 4 mismatches / 839, see below). Covered: all 8 int dtypes x 5
dtype-spelling forms (type object, dtype instance, name string, char code,
numpy scalar instance) for `iinfo`; all 3 float + 2 complex dtypes x the
same 5 spellings for `finfo`; every numeric attribute compared by VALUE AND
TYPE (never by `getattr`-presence alone, per this ticket's instrument rule);
`repr`/`str`/class-name comparison; repeated-construction identity
semantics (`iinfo`: no caching, matches numpy's `is` -> `False`; `finfo`:
caches per resolved dtype INCLUDING cross-spelling identity and
complex->float redirection identity, matches numpy's `is` -> `True` in all
tested cases); 11 invalid inputs each to `iinfo`/`finfo` (complex-to-iinfo,
int-to-finfo, ndarray, nonsense string, None, list, dict, plain object,
bare int/float value, `str` type) compared by exact exception TYPE and
MESSAGE TEXT.

RESULT: 4/839 mismatches, ALL the same pre-existing, already-documented,
declined architectural gap (see `TOPLEVEL_STATE`'s finfo/iinfo history and
ticket #31): anionpy has no void/structured/string dtype representation, so
unresolvable dtype-like inputs (`{}`, unrecognized strings, `str` type) fall
back to a `dtype('O')` object-dtype error message where real numpy gives a
genuine void (`dtype('V')`) or `<U`-string-typed fallback message. This is
NOT new -- re-measured and reconfirmed unchanged, not re-diagnosed as
something this ticket could cleanly fix (would require a new DType variant
across `ionp-core`, out of scope). Every valid-dtype-input check, across the
entire 5-spelling x 13-dtype matrix, passed with zero divergence.

WHAT IS DECLARED HERE AND WHY (out-of-corpus verified, per the rule above --
never by `getattr` alone, always invocation + value/type/text comparison):
  - The 9 structural-fix items (`iinfo.min`/`.max`, `finfo.epsneg`/`.iexp`/
    `.machep`/`.negep`/`.nexp`/`.resolution`/`.tiny`): every valid-input
    invocation across the full dtype x spelling matrix matched numpy
    exactly on both value and type. Declared "exact".
  - `iinfo.__repr__`/`.__str__`, `finfo.__repr__`/`.__str__`: these methods
    are invoked ONLY on an already-successfully-constructed instance --
    they do not participate in the invalid-input/error-message code path
    at all, so the one confirmed divergence class (construction-time error
    text for unresolvable dtypes) does not taint them. Matched byte-for-
    byte across every dtype tested. Declared "exact".

WHAT IS NOT DECLARED HERE AND WHY:
  - `iinfo.__init__`/`.__new__`, `finfo.__init__`/`.__new__`: these ARE the
    methods that hit the confirmed error-message divergence for
    unresolvable dtype-like inputs (4/839 above). Per this project's
    standing declaration rule (see `SCALARS_STATE`'s docstring: "if ANY
    call form of an item still diverges from numpy -- any input, any error
    path -- do NOT declare it"), these four are NOT declared, even though
    they pass on every valid-dtype input tested. This is a genuine decline,
    not a manufactured one -- the divergence is real, reproducible, and
    already independently documented (ticket #31) as an architectural gap,
    not a bug introduced or hidden by this ticket.
  - All remaining dunders on both classes (`__eq__`, `__hash__`, `__ne__`,
    `__lt__`/`__le__`/`__gt__`/`__ge__`, `__delattr__`, `__dir__`,
    `__format__`, `__getattribute__`, `__getstate__`, `__init_subclass__`,
    `__module__`, `__reduce__`/`__reduce_ex__`, `__setattr__`,
    `__sizeof__`, `__static_attributes__`, `__subclasshook__`,
    `__weakref__`, `__class_getitem__`): neither numpy's nor anionpy's
    `iinfo`/`finfo` override any of these -- both simply inherit `object`'s
    (or `type`'s, for `__class_getitem__`) default implementation. Spot-
    checked `__eq__`/`__hash__` (identity-based equality, matches numpy on
    both the cached-`finfo`-so-equal-instances-are-identical case and the
    non-cached-`iinfo`-so-separate-instances-are-unequal case) but NOT
    declared here: this ticket's measurement effort went into the
    `finfo`/`iinfo`-SPECIFIC surface (the 9 structural items + str/repr),
    not a full audit of inherited-from-`object` behavior, and declaring 20
    items on a single spot-check each would be exactly the kind of
    "declare to protect a number" the state-module convention warns
    against. Left undeclared; a future ticket doing a real audit of that
    surface can declare them properly.
"""
from __future__ import annotations

FINFO_IINFO_STATE: dict[str, str] = {
    "iinfo.min": "exact",
    "iinfo.max": "exact",
    "finfo.epsneg": "exact",
    "finfo.iexp": "exact",
    "finfo.machep": "exact",
    "finfo.negep": "exact",
    "finfo.nexp": "exact",
    "finfo.resolution": "exact",
    "finfo.tiny": "exact",
    "iinfo.__repr__": "exact",
    "iinfo.__str__": "exact",
    "finfo.__repr__": "exact",
    "finfo.__str__": "exact",
}
