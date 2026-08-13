"""Differential ItemSpecs for the `anionpy.emath` block (9 numpy_surface.json
items under the `emath.` prefix -- a.k.a. `numpy.lib.scimath`).

Owned exclusively by this task (see the task brief): this file,
ionp-core/src/emath.rs, ionp-py/src/emath.rs, and
anionpy/_state/misc_namespaces.py. Mirrors linalg_cases.py's/testing_cases.py's
pattern exactly -- an `EMATH_SPECS: dict[str, ItemSpec]` merged into
registry.REGISTRY at the bottom of registry.py itself (registry.py is NOT
owned by this task; the merge block there matches the LINALG_SPECS/
TESTING_SPECS collision-checked pattern already present).

Seven of the nine (`sqrt`/`log`/`log2`/`log10`/`arccos`/`arcsin`/`arctanh`)
are plain unary functions with the exact same call shape as any other
elementwise function in this suite, so `kind="unary"` (resolved generically
against `numpy.emath.<name>` / `anionpy.emath.<name>` by registry.py's own
fallback path -- no adapter needed) runs each one over the FULL
`corpus.unary_corpus()`. That corpus already includes -1.0/-0.0/0.0/1.0
explicitly (`corpus._special_float_values`'s `specials` list) plus random
sweeps over every signed dtype (which routinely produce negative values)
and the full signed-integer extremes (`corpus._integer_boundaries`) --
i.e. the branch-cut trigger condition (`x < 0` for
sqrt/log/log2/log10, `abs(x) > 1` for arccos/arcsin/arctanh) is exercised
by the existing corpus without any custom cases needed here.

The remaining two (`logn`, `power`) are two-argument; `kind="binary"` runs
each over `corpus.binary_corpus()` (array-vs-array, matching numpy's own
`logn(n, x)` / `power(x, p)` positional order -- registry.py passes the
corpus pair positionally in the same order to both the numpy and anionpy
sides, so argument order is preserved identically on both).

atol/rtol are 0.0 (bit-exact) for all nine: `ionp_core::emath`'s promotion
layer performs no numeric approximation of its own -- see
ionp-core/src/emath.rs's module doc comment -- it only decides whether to
cast to complex/float64 before handing off to `ufunc.rs`'s complex
Sqrt/Log/Log2/Log10/Arcsin/Arccos/Arctanh/Power loops.

Only 5 of the 9 registered items (sqrt/log/arccos/arcsin/arctanh) are
declared "exact" in anionpy/_state/misc_namespaces.py. The remaining 4
(log2/log10/logn/power) are registered here -- so the differential suite
runs and records them -- but are deliberately left UNDECLARED: they surface
real, pre-existing precision bugs in `ufunc.rs`'s own complex Log2/Log10/
Power/Divide loops (not owned by this task, not fixed by this task; see
misc_namespaces.py's module comment for the exact reproductions). Keeping
them registered-but-undeclared is what lets `tools/coverage.py` report them
correctly as "absent" rather than silently dropping the evidence.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from registry import ItemSpec

EMATH_SPECS: dict[str, ItemSpec] = {}


def _reg(name: str, kind: str) -> None:
    EMATH_SPECS[name] = ItemSpec(
        name=name,
        kind=kind,
        atol=0.0,
        rtol=0.0,
    )


_reg("emath.sqrt", "unary")
_reg("emath.log", "unary")
_reg("emath.log2", "unary")
_reg("emath.log10", "unary")
_reg("emath.arccos", "unary")
_reg("emath.arcsin", "unary")
_reg("emath.arctanh", "unary")
_reg("emath.logn", "binary")
_reg("emath.power", "binary")
