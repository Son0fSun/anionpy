#!/usr/bin/env python3
"""Regenerate numpy_surface.json -- the DENOMINATOR of the coverage claim.

Run this only when deliberately re-pinning to a new numpy version. Bumping the
reference silently is how a coverage percentage gets to rise without any work
being done, so this is a separate, explicit, committed act.

Measurement tooling only -- no numerical work happens here.
"""

from __future__ import annotations

import collections
import inspect
import json
from pathlib import Path

import numpy as np

SUBMODULES = [
    "linalg", "fft", "random", "ma", "polynomial",
    "strings", "rec", "char", "testing", "lib", "ctypeslib", "emath",
    "exceptions", "dtypes",
]

# CURATED explosion set -- container/behavior-bearing types whose method
# surface is measured individually, the same way `ndarray` already is.
#
# Deliberately NOT "every class numpy exposes" (~110 of them). Exploding a
# SCALAR type (np.float64, np.int32, ...) would force existing declarations
# to be RENAMED: scalar item keys already exist today as bare top-level-ish
# names tied to the scalar's role elsewhere in the ledger, and colliding
# with a newly-exploded per-method key would mean choosing which one moves.
# The curated set below is chosen so explosion is purely ADDITIVE: none of
# these class names is currently declared anywhere in `anionpy.__ion_state__`
# as a bare flat key (verified by grep across `anionpy/_state/*.py` before
# this change), so nothing gets orphaned by removing a flat entry that
# nothing points at.
#
# Each entry is (item_key_prefix, class_object). `item_key_prefix` is the
# dotted path used to build each method's full ledger key
# ("<prefix>.<method>"), matching the SUBMODULES prefix convention already
# used for flat items (e.g. "ma.getdata", "random.default_rng").
# Curated submodule descent -- dotted paths (relative to `np`) into
# submodules whose public surface `surface()` cannot see today because
# `SUBMODULES` above only scans ONE level deep and `surface()` skips module
# objects outright (`if inspect.ismodule(o): continue`). Each entry here is
# a real, documented, commonly-imported public namespace, individually
# adjudicated -- see docs/DENOMINATOR-SUBMODULE-HOLE-2026-08-06.md and
# docs/DENOMINATOR-SUBMODULE-FIX-2026-08-06.md for the measured evidence.
#
# Deliberately NOT "recurse into every submodule numpy has": a blind
# recursive crawl across all of numpy returned a contaminated "3104
# untracked names" dominated by re-crawled `import numpy as np` aliases and
# f2py build internals. Every entry below was checked for re-export overlap
# against the already-tracked surface before being added; several sibling
# candidates were checked and EXCLUDED for failing that check (see the FIX
# doc's "Excluded" section) -- `ma.core` (179/220 names already tracked
# under `ma.*`), `ma.extras` (65/80 already under `ma.*`), `random.mtrand`
# (53/57 already under `random.*`, remaining 4 are import aliases), and
# `lib.scimath` (`np.lib.scimath is np.emath`, already tracked in full
# under `emath.*` -- 100% identical object, not merely name overlap).
CURATED_SUBMODULE_DESCEND = [
    # np.polynomial.* -- each is a distinct per-basis module exposing its
    # own free-function API (chebval/chebfit/chebroots/... for chebyshev,
    # legval/legfit/... for legendre, etc). Zero overlap with the top-level
    # `np.polynomial` namespace: none of these free functions is re-exported
    # there (only the 6 basis classes and 2 module-level helpers are).
    "polynomial.chebyshev", "polynomial.hermite", "polynomial.hermite_e",
    "polynomial.laguerre", "polynomial.legendre", "polynomial.polynomial",
    "polynomial.polyutils",
    # np.lib.* real submodules -- measured ZERO bare-name overlap against
    # every currently-tracked item (top-level `np.*` and every other
    # tracked prefix), unlike their sibling `lib.scimath` (excluded above).
    "lib.format",         # +20: npy on-disk format read/write (read_array,
                           # write_array, MAGIC_*, ...), no top-level alias.
    "lib.stride_tricks",  # +2: as_strided, sliding_window_view.
    "lib.npyio",           # +2: DataSource, NpzFile (classes; tracked flat,
                           # not exploded -- same convention as other
                           # non-curated classes already in `names`).
    "lib.array_utils",    # +3: byte_bounds, normalize_axis_index,
                           # normalize_axis_tuple.
    # np.testing.overrides -- real runtime functions (array_function/ufunc
    # override introspection), zero overlap with `np.testing`'s own
    # top-level names.
    "testing.overrides",
]

CURATED_EXPLODE = [
    ("ma.MaskedArray", lambda: np.ma.MaskedArray),
    ("random.Generator", lambda: np.random.Generator),
    ("random.RandomState", lambda: np.random.RandomState),
    ("dtype", lambda: np.dtype),
    ("matrix", lambda: np.matrix),
    ("recarray", lambda: np.recarray),
    ("memmap", lambda: np.memmap),
    ("char.chararray", lambda: np.char.chararray),
    ("poly1d", lambda: np.poly1d),
    ("finfo", lambda: np.finfo),
    ("iinfo", lambda: np.iinfo),
    ("polynomial.Chebyshev", lambda: np.polynomial.Chebyshev),
    ("polynomial.Hermite", lambda: np.polynomial.Hermite),
    ("polynomial.HermiteE", lambda: np.polynomial.HermiteE),
    ("polynomial.Laguerre", lambda: np.polynomial.Laguerre),
    ("polynomial.Legendre", lambda: np.polynomial.Legendre),
    ("polynomial.Polynomial", lambda: np.polynomial.Polynomial),
]


def surface(mod, prefix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for n in dir(mod):
        if n.startswith("_"):
            continue
        try:
            o = getattr(mod, n)
        except Exception:
            continue
        if inspect.ismodule(o):
            continue
        kind = (
            "ufunc" if isinstance(o, np.ufunc)
            else "class" if inspect.isclass(o)
            else "func" if callable(o)
            else "const"
        )
        out[f"{prefix}{n}"] = kind
    return out


# Ticket #81. Interpreter/metaclass machinery names that show up in
# `dir()` of an exploded class purely because the class is written in
# Python (or Python + ABCMeta), not because numpy chose to expose them as
# API. Each entry was verified to be "earned for free" by a trivial class
# that does NOTHING to opt in -- shown by inspecting `vars()` of a bare
# `class C: pass` (or, for `__abstractmethods__`, a plain concrete
# subclass of an `abc.ABC` base with no unimplemented methods):
#
#   >>> class Plain: pass
#   >>> sorted(vars(Plain))
#   ['__dict__', '__doc__', '__firstlineno__', '__module__',
#    '__static_attributes__', '__weakref__']
#
#   >>> import abc
#   >>> class Base(abc.ABC):
#   ...     @abc.abstractmethod
#   ...     def foo(self): ...
#   >>> class Concrete(Base):
#   ...     def foo(self): return 1
#   >>> vars(Concrete)['__abstractmethods__']
#   frozenset()
#
# `__module__` -- every class statement gets one from the enclosing
# module's globals, unconditionally. Cannot diverge: a reimplementation's
# class lives in a different file and so has a different (but equally
# "correct") value by construction, making equality meaningless as an API
# claim.
#
# `__firstlineno__` -- CPython 3.13+ compiler output: the source line
# number of the `class` statement. Not observable as behavior; not
# something an implementation "has" or "lacks", only a fact about which
# file it was typed into.
#
# `__static_attributes__` -- CPython 3.13+ compiler output: names the
# compiler saw assigned via `self.<name> = ...` anywhere in the class
# body. Derived mechanically from the method bodies already tracked
# individually elsewhere in this same manifest; tracking it AGAIN as a
# separate item double-counts the same source text.
#
# `__abstractmethods__` -- set by `ABCMeta.__new__` on every class using
# that metaclass, unconditionally, computed from which abstract methods
# remain unoverridden. For the CURATED_EXPLODE polynomial classes this is
# always `frozenset()` (they're concrete) -- there is no way to author a
# concrete subclass and have this attribute look any other way, so
# tracking it adds a name a correct implementation cannot get "wrong"
# independent of the method surface already tracked.
#
# `__weakref__` and `__slots__` are deliberately NOT on this list --
# see ticket #81. Both fail the "earned for free" test above: `__slots__`
# never appears unless a class body writes it explicitly, and
# `__weakref__` disappears the moment a class opts into `__slots__`
# without including it. Whether a numpy class supports weak references,
# and whether it restricts its instance `__dict__`, are real, observable,
# gettable-wrong properties of a reimplementation -- kept in the
# denominator.
_MACHINERY_DUNDER_DENYLIST = frozenset({
    "__module__",
    "__firstlineno__",
    "__static_attributes__",
    "__abstractmethods__",
})


def explode_class(cls) -> tuple[list[str], list[str]]:
    """Split a class's public dir() into (public, dunder) name lists.

    Bare method/attr names, NOT prefixed -- the full ledger item key
    ("<prefix>.<name>") is assembled by the caller from CURATED_EXPLODE's
    prefix, keeping this function reusable across every curated class
    instead of ndarray owning bespoke prefixing logic no other class gets.
    """
    public = sorted(n for n in dir(cls) if not n.startswith("_"))
    dunder = sorted(
        n for n in dir(cls)
        if n.startswith("__")
        and n not in ("__class__", "__doc__", "__dict__")
        and n not in _MACHINERY_DUNDER_DENYLIST
    )
    return public, dunder


def main() -> None:
    names: dict[str, str] = {}
    names.update(surface(np, ""))
    skipped = []
    for sub in SUBMODULES:
        try:
            names.update(surface(getattr(np, sub), f"{sub}."))
        except Exception as exc:
            skipped.append(f"{sub}: {exc}")

    exploded: dict[str, dict[str, list[str]]] = {}

    # ndarray is exploded exactly like every other curated class, and like
    # them it does NOT also keep a flat "ndarray": "class" entry.
    #
    # DECIDED 2026-08-05 (Monday). It previously kept both -- a flat entry
    # from the top-level surface() scan AND the full explosion -- which was
    # the double-count that gave one class 165 weight while ~110 others
    # carried 1 apiece. That asymmetry is precisely the bias this whole
    # change exists to remove, so leaving it in place "because it is
    # pre-existing" would ship a denominator that is honest about 17
    # classes and still rigged for the 18th.
    #
    # Costs zero declarations: `anionpy.__ion_state__` has no "ndarray" key
    # (checked directly against the live module, not by grep -- 667
    # declared, `st.get("ndarray")` is None), so nothing is orphaned and
    # nothing is renamed. It removes exactly one item, which was `absent`.
    nd_public, nd_dunder = explode_class(np.ndarray)
    exploded["ndarray"] = {"public": nd_public, "dunder": nd_dunder}
    names.pop("ndarray", None)

    exploded_class_ids: set[int] = set()
    for prefix, get_cls in CURATED_EXPLODE:
        try:
            cls = get_cls()
        except Exception as exc:
            skipped.append(f"{prefix}: {exc}")
            continue
        public, dunder = explode_class(cls)
        exploded[prefix] = {"public": public, "dunder": dunder}
        exploded_class_ids.add(id(cls))
        # Drop the flat single-count entry for this class if the scan above
        # picked one up (e.g. np.dtype, np.matrix, np.char.chararray). Left
        # in place it would double-count every curated class the way
        # ndarray already (knowingly) does; unlike ndarray this is new
        # behaviour, not a preserved quirk, and no declaration depends on
        # the flat key -- verified by grep: none of CURATED_EXPLODE's
        # prefixes appears as a bare key anywhere in anionpy/_state/*.py.
        names.pop(prefix, None)

    # Curated submodule descent (np.polynomial.chebyshev, .hermite, ...) --
    # see CURATED_SUBMODULE_DESCEND's comment and
    # docs/DENOMINATOR-SUBMODULE-HOLE-2026-08-06.md for why this exists and
    # why it is a fixed list rather than a recursive crawl.
    #
    # Re-export adjudication: each of the 6 basis classes (Chebyshev,
    # Hermite, HermiteE, Laguerre, Legendre, Polynomial) IS the identical
    # object already exploded above under "polynomial.<ClassName>" --
    # `np.polynomial.chebyshev.Chebyshev is np.polynomial.Chebyshev` holds.
    # Re-adding it here as a flat "class" entry under a new dotted path
    # would double-count something already in the ledger, so class-kind
    # entries are deduped by object identity against `exploded_class_ids`.
    # The shared abstract base `ABCPolyBase` is reached via the SAME bare
    # name in 6 of the 7 submodules (`np.polynomial.chebyshev.ABCPolyBase
    # is np.polynomial.hermite.ABCPolyBase is ...`) -- a re-export of one
    # object repeated by a module-import accident, not 6 independent public
    # surfaces, so it is deduped the same way and lands once (in whichever
    # submodule is processed first).
    #
    # Free functions/consts are NOT deduped by identity even when several
    # module-level names point at the same object -- e.g.
    # `chebyshev.chebtrim`, `legendre.legtrim`, and `polyutils.trimcoef`
    # are all literally `numpy.polynomial.polyutils.trimcoef` under the
    # hood, but each is a distinct, real, independently-documented public
    # name (`from numpy.polynomial import legendre; legendre.legtrim(...)`
    # is genuine numpy API, not an accident of import machinery). This
    # matches the manifest's existing convention elsewhere: `np.abs` and
    # `np.absolute` are the same ufunc object and both are already tracked
    # as separate top-level items today. Deduping these by identity would
    # be inconsistent with that precedent and would silently under-count
    # real per-basis API surface.
    submodule_class_ids = set(exploded_class_ids)
    for dotted in CURATED_SUBMODULE_DESCEND:
        mod = np
        try:
            for part in dotted.split("."):
                mod = getattr(mod, part)
        except Exception as exc:
            skipped.append(f"{dotted}: {exc}")
            continue
        for n, kind in surface(mod, "").items():
            o = getattr(mod, n)
            if kind == "class":
                if id(o) in submodule_class_ids:
                    continue
                submodule_class_ids.add(id(o))
            names[f"{dotted}.{n}"] = kind

    payload = {
        "numpy_version": np.__version__,
        "names": names,
        "exploded": exploded,
    }
    out = Path(__file__).resolve().parent / "numpy_surface.json"
    out.write_text(json.dumps(payload, indent=0, sort_keys=True))

    kinds = collections.Counter(names.values())
    exploded_total = sum(len(v["public"]) + len(v["dunder"]) for v in exploded.values())
    total = len(names) + exploded_total
    print(f"numpy {np.__version__}")
    print(f"  names   : {len(names)} {dict(kinds)}")
    for cls_name, v in sorted(exploded.items()):
        print(f"  exploded[{cls_name}] : {len(v['public'])} public, {len(v['dunder'])} dunder")
    print(f"  exploded total : {exploded_total}")
    print(f"  TOTAL   : {total}")
    if skipped:
        print("  skipped :", "; ".join(skipped))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
