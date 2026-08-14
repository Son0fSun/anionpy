"""Surface-completeness guard for `random.*` BitGenerator classes.

Added 2026-08-13 (sixth pass), directly in response to an external audit
finding: `random.SFC64` had been declared `"exact"` in
`anionpy/_state/random.py`'s `RANDOM_STATE` dict while implementing only 1
of numpy's 8 public `SFC64` attributes (`random_raw` only; missing
`capsule`, `cffi`, `ctypes`, `lock`, `seed_seq`, `spawn`, `state`), and
while rejecting a documented numpy constructor call form
(`SFC64([1, 2, 3])`, sequence-of-ints entropy) that numpy itself accepts.
That declaration self-contradicted this file's OWN existing precedent:
`anionpy/_state/random.py` already documents, in its "WHAT REMAINS
UNDECLARED" section, that `PCG64`/`PCG64DXSM` stay undeclared for the
IDENTICAL partiality (constructor + `.random_raw()` only). The SFC64
declaration cited that precedent for SCOPE but reached the opposite
VERDICT -- nothing mechanical stopped that self-contradiction from
happening once, so nothing mechanical would stop it from happening again
on the next bit generator (MT19937, Philox) without a real guard.

This module is that guard, and it is INTENTIONALLY NOT wired into
`registry.py`/`run.py`'s REGISTRY -- it does not touch the 2192-item
differential-suite baseline or the 37-name failure set at all. It is its
own pytest module, run directly (`pytest
tests/differential/test_bitgen_surface.py` or as part of a full `pytest
tests/differential/` run), independent of `tools/coverage.py`'s ledger.

`public_surface()` filters to attribute names not starting with `_` -- this
intentionally excludes dunders (`__reduce_cython__`, `__setstate__`, ...)
and cython/CPython implementation-detail names numpy itself prefixes with
an underscore (`_cffi`, `_ctypes`, `_seed_seq`, `_benchmark`), leaving only
the genuinely public, documented surface. Measured directly against a real
numpy 2.5.1 install (2026-08-13):

    >>> sorted(a for a in dir(np.random.SFC64(1)) if not a.startswith("_"))
    ['capsule', 'cffi', 'ctypes', 'lock', 'random_raw', 'seed_seq', 'spawn',
     'state']

8 attributes. anionpy's `SFC64` has exactly 1 (`random_raw`).
"""
import _bootstrap  # noqa: F401

import anionpy
import numpy as np

from anionpy._state.random import RANDOM_STATE


def public_surface(obj) -> set[str]:
    """The set of an object's non-dunder, non-private attribute names.

    Deliberately a simple `not name.startswith("_")` filter, not a
    `set(dir(x)) - set(dir(object))` filter -- the latter is what a first
    pass at this measurement used, and it silently let cython-internal
    names like `__pyx_vtable__`/`__reduce_cython__`/`_cffi`/`_ctypes`
    through (they aren't on `dir(object)`), overstating both sides' surface
    by the same wrong amount and making the two counts *coincidentally*
    look closer than they are. The `not name.startswith("_")` filter is
    the one that reproduces the audit's cited 8-vs-1 SFC64 figure exactly.
    """
    return {name for name in dir(obj) if not name.startswith("_")}


# Every numpy BitGenerator class this repo has ATTEMPTED (wired a
# constructor + random_raw for), regardless of whether `_state/random.py`
# currently declares it "exact". This list is intentionally broader than
# RANDOM_STATE's declared set: the point of this guard is to make it cheap
# to check "is this class's surface actually complete enough to declare",
# not merely to re-verify what's already declared.
BITGEN_CLASSES: dict[str, tuple[type, type]] = {
    "random.PCG64": (np.random.PCG64, anionpy.random.PCG64),
    "random.PCG64DXSM": (np.random.PCG64DXSM, anionpy.random.PCG64DXSM),
    "random.SFC64": (np.random.SFC64, anionpy.random.SFC64),
}


def bitgen_surface_gap(numpy_cls: type, ionp_cls: type) -> set[str]:
    """Attributes numpy's public surface has that anionpy's lacks.

    Constructs each side with a fixed integer seed (all three bit
    generators accept a bare int) purely to get an instance to call `dir()`
    on -- the actual VALUES are irrelevant here, only the attribute
    NAMES are being compared.
    """
    numpy_surface = public_surface(numpy_cls(1))
    ionp_surface = public_surface(ionp_cls(1))
    return numpy_surface - ionp_surface


def test_guard_bites_on_sfc64s_known_partial_surface():
    """Proves the guard actually rejects something, per the audit's
    explicit instruction: "prove the guard bites by constructing an
    input/case it rejects." SFC64's current 1-of-8 surface is exactly
    that rejected case -- if this test ever starts passing (gap becomes
    empty) without `_state/random.py` being updated to re-declare
    `random.SFC64` "exact", that is itself news worth capturing, not a
    silent green.
    """
    gap = bitgen_surface_gap(np.random.SFC64, anionpy.random.SFC64)
    assert gap == {
        "capsule",
        "cffi",
        "ctypes",
        "lock",
        "seed_seq",
        "spawn",
        "state",
    }, (
        f"expected SFC64's known 7-attribute gap (capsule/cffi/ctypes/"
        f"lock/seed_seq/spawn/state), got {sorted(gap)} -- either the "
        f"gap changed (update this test and _state/random.py together) "
        f"or the guard itself broke"
    )


def test_guard_bites_on_pcg64s_known_partial_surface():
    """Same guard, second known-partial class (PCG64), confirming this
    isn't an SFC64-specific special case -- PCG64/PCG64DXSM's own
    undeclared status in `_state/random.py` already documents this same
    gap in prose; this is that prose made mechanically checkable.
    """
    gap = bitgen_surface_gap(np.random.PCG64, anionpy.random.PCG64)
    assert "state" in gap
    assert "advance" in gap
    assert "jumped" in gap


def test_no_bitgen_class_is_declared_exact_with_an_incomplete_surface():
    """The actual regression guard: for every `random.<BitGenClass>` item
    this file knows how to construct on both sides, if `_state/random.py`
    ever declares it `"exact"` in `RANDOM_STATE`, its public surface MUST
    fully match numpy's. This is what would have failed loudly the moment
    `random.SFC64` was added to `RANDOM_STATE` as `"exact"` in the
    fifth pass, instead of that gap surviving to an external audit.

    Currently a vacuous pass (PCG64/PCG64DXSM/SFC64 are all correctly
    undeclared) -- that vacuousness is itself the point: this test starts
    doing real work the moment any of these three (or a future MT19937/
    Philox entry) gets declared, with no further code changes needed.
    """
    for item_name, (numpy_cls, ionp_cls) in BITGEN_CLASSES.items():
        if RANDOM_STATE.get(item_name) != "exact":
            continue
        gap = bitgen_surface_gap(numpy_cls, ionp_cls)
        assert not gap, (
            f"{item_name} is declared \"exact\" in RANDOM_STATE but is "
            f"missing {sorted(gap)} from numpy's public surface -- revoke "
            f"the declaration or implement the missing attributes before "
            f"re-declaring"
        )


def test_sfc64_rejects_documented_sequence_entropy_numpy_accepts():
    """The second half of the audit's Finding 2: not just a missing
    attribute, but a documented, numpy-accepted constructor call form
    (`SFC64([1, 2, 3])`, sequence-of-ints entropy) that anionpy's SFC64
    currently rejects with TypeError. Confirmed numpy-side success first
    (this assertion would be meaningless if numpy itself rejected the
    call), then confirms anionpy's rejection, so this test documents BOTH
    halves of the divergence, not just anionpy's failure in isolation.
    """
    np.random.SFC64([1, 2, 3])  # numpy: succeeds (must not raise)
    try:
        anionpy.random.SFC64([1, 2, 3])
    except TypeError:
        pass
    else:
        raise AssertionError(
            "anionpy.random.SFC64([1, 2, 3]) unexpectedly succeeded -- "
            "if sequence entropy is now supported, this test (and the "
            "surface-gap tests above, and _state/random.py) need updating "
            "together, not just this one assertion"
        )
