"""Assembles `anionpy.__ion_state__` from one declaration module per surface block.

WHY THIS EXISTS
---------------
Every coverage declaration used to live in a single 565-line dict inside
`anionpy/__init__.py`. That made one file a mandatory bottleneck: any two agents
declaring work had to be serialised through it, and any agent fenced off it
could not credit its own finished work. The measured cost was five separate
blocks of work that were implemented, tested, hardened and committed -- and
counted for nothing, because wiring them in was somebody else's file to touch.
The 47-item `numpy.testing` block alone moved the ledger 91 -> 138 the moment
it was connected; nothing was built that day, it was already built. See
`reports/ionp-throughput-bottleneck-2026-08-01.md`.

One module per block means two agents declaring in different blocks never touch
the same file.

THE COLLISION CHECK IS THE POINT
--------------------------------
Splitting a dict across files introduces a failure the single dict could not
have: the same key declared in two modules, where a plain `.update()` chain
would silently let the last writer win. Silent is the problem -- a declaration
that quietly overrides another is exactly the "the ledger and reality drift
apart in whichever direction nobody was checking" failure this whole ledger
exists to prevent. So a duplicate key is a hard, loud `AssertionError` at
import time. `anionpy` refusing to import at all is a far better outcome than
`anionpy` importing with a ledger that quietly lost a declaration.

This mirrors the merge already used by `tests/differential/registry.py` for
LINALG_SPECS / INPLACE_SPECS / TESTING_SPECS.

ADDING A DECLARATION
--------------------
Put it in the module for its block, WITH the measurement that justifies it.
The comments in these modules are the evidence, not decoration: an entry whose
reason is not written down cannot be audited later, and an unaudited "exact" is
indistinguishable from a guess. Never declare an item to protect a number --
the ledger going down because a false entry was removed is a win.
"""
from __future__ import annotations

from anionpy._state.char_strings import CHAR_STRINGS_STATE
from anionpy._state.fft import FFT_STATE
from anionpy._state.finfo_iinfo import FINFO_IINFO_STATE
from anionpy._state.linalg import LINALG_STATE
from anionpy._state.ma import MA_STATE
from anionpy._state.matrix import MATRIX_STATE
from anionpy._state.memmap import MEMMAP_STATE
from anionpy._state.misc_namespaces import MISC_NAMESPACES_STATE
from anionpy._state.ndarray import NDARRAY_STATE
from anionpy._state.poly_legacy import POLY_LEGACY_STATE
from anionpy._state.polynomial import POLYNOMIAL_STATE
from anionpy._state.random import RANDOM_STATE
from anionpy._state.scalars import SCALARS_STATE
from anionpy._state.testing import TESTING_STATE
from anionpy._state.toplevel import TOPLEVEL_STATE

_BLOCKS = (
    ("toplevel", TOPLEVEL_STATE),
    ("ndarray", NDARRAY_STATE),
    ("linalg", LINALG_STATE),
    ("ma", MA_STATE),
    ("matrix", MATRIX_STATE),
    ("memmap", MEMMAP_STATE),
    ("testing", TESTING_STATE),
    ("char_strings", CHAR_STRINGS_STATE),
    ("misc_namespaces", MISC_NAMESPACES_STATE),
    ("fft", FFT_STATE),
    ("random", RANDOM_STATE),
    ("scalars", SCALARS_STATE),
    ("polynomial", POLYNOMIAL_STATE),
    ("poly_legacy", POLY_LEGACY_STATE),
    ("finfo_iinfo", FINFO_IINFO_STATE),
)


def build_ion_state() -> dict[str, str]:
    """Merge every block's declarations, refusing loudly on any duplicate key."""
    merged: dict[str, str] = {}
    origin: dict[str, str] = {}
    for block_name, block in _BLOCKS:
        collisions = set(block) & set(merged)
        if collisions:
            detail = ", ".join(
                f"{key!r} (already declared in {origin[key]!r})"
                for key in sorted(collisions)
            )
            raise AssertionError(
                f"anionpy._state: duplicate coverage declaration(s) in "
                f"{block_name!r}: {detail}. Two modules declaring the same item "
                f"means one of them is silently ignored and the ledger is "
                f"counting something nobody verified. Fix the duplication -- do "
                f"not paper over it by deleting whichever entry looks wrong."
            )
        merged.update(block)
        origin.update({key: block_name for key in block})
    return merged


__all__ = ["build_ion_state", "LINALG_STATE", "NDARRAY_STATE",
           "TESTING_STATE", "TOPLEVEL_STATE", "MISC_NAMESPACES_STATE",
           "FFT_STATE", "RANDOM_STATE", "SCALARS_STATE", "MA_STATE",
           "MATRIX_STATE", "MEMMAP_STATE", "POLYNOMIAL_STATE",
           "POLY_LEGACY_STATE", "FINFO_IINFO_STATE"]
