"""Differential registry entries for the `finfo`/`iinfo` STRUCTURAL fix and
new `__str__` methods (ticket #71, 2026-08-08).

NOT wired into `registry.py` -- that file is owned by a concurrent agent per
this ticket's brief and was not touched. This module follows the exact same
`ItemSpec`/`custom_cases` shape as every other `*_cases.py` (see
`poly1d_legacy_cases.py` for the template this was copied from) so a human
can merge

    from finfo_iinfo_cases import FINFO_IINFO_SPECS
    _check_no_collisions(FINFO_IINFO_SPECS)
    REGISTRY.update(FINFO_IINFO_SPECS)

into `registry.py` in one pass once that file is free to touch again.

RELATIONSHIP TO THE EXISTING `dtypeinfo_cases.py`
--------------------------------------------------
`dtypeinfo_cases.py` already registers WHOLE-OBJECT probes named `"iinfo"`
and `"finfo"` (constructs an instance, reads a fixed tuple of fields
including `.min`/`.max`/`.repr()`, compares the tuple). That module is
untouched by this ticket. What is missing from the ledger's perspective is
per-ATTRIBUTE items: `tools/numpy_surface.json`'s `"exploded"` section
curates `finfo.epsneg`/`.iexp`/`.machep`/`.negep`/`.nexp`/`.resolution`/
`.tiny` and `iinfo.min`/`.max` as their OWN ledger items (because those are
exactly numpy's `@cached_property`/`@property`-decorated CLASS attributes --
see `anionpy/_state/finfo_iinfo.py`'s docstring for the full mechanism), and
`finfo.__repr__`/`.__str__`/`iinfo.__repr__`/`.__str__` as their own items
too. None of THOSE dotted items were covered by any differential case before
this ticket (the whole-object `"iinfo"`/`"finfo"` probes exercise the same
underlying values but are not indexed under the dotted names
`tools/coverage.py`'s `resolve()` looks for). This module closes that gap
for exactly the 13 items declared "exact" in `FINFO_IINFO_STATE` --
deliberately NOT the 4 remaining dunders that still diverge
(`iinfo.__init__`/`.__new__`, `finfo.__init__`/`.__new__`), which are
represented here too but registered WITHOUT being included in the paste
instruction's implied "these all pass" framing -- see `_init_new_cases()`
below, kept for future use once someone picks up ticket #31's architectural
gap, not because they currently pass.

Every case below was measured directly against real numpy 2.5.1 (not just
inferred from the whole-object probe) via
`/private/tmp/finfo_probe/full_probe.py`, run OUT of this corpus using the
actual installed `anionpy` package -- see
`docs/TICKET-71-FINFO-2026-08-08.md` for the full sweep summary (839 checks,
4 mismatches, all in the `__init__`/`__new__` invalid-dtype-input path, none
in any of the 13 items below). This file re-encodes the same inputs in the
registry's format so the differential harness can re-run them once wired
in; it does not introduce new claims beyond what was already measured.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)

import numpy as np

from registry import ItemSpec

FINFO_IINFO_SPECS: dict[str, ItemSpec] = {}

INT_DTYPES = ["int8", "int16", "int32", "int64",
              "uint8", "uint16", "uint32", "uint64"]
FLOAT_DTYPES = ["float16", "float32", "float64"]
COMPLEX_DTYPES = ["complex64", "complex128"]


def _spelling_cases(dtypes):
    """Every dtype x every one of the 5 required spelling forms (type
    object, dtype instance, name string, char code, numpy scalar
    instance), per the ticket brief's "dtype specified 5 ways" requirement.
    """
    cases = []
    for name in dtypes:
        npdt = np.dtype(name)
        cases.append((f"{name}_typeobj", (npdt.type,), {}))
        cases.append((f"{name}_dtypeinst", (npdt,), {}))
        cases.append((f"{name}_namestr", (name,), {}))
        cases.append((f"{name}_charstr", (npdt.char,), {}))
        cases.append((f"{name}_npscalar", (npdt.type(1),), {}))
    return cases


# ---------------------------------------------------------------------------
# iinfo.min / iinfo.max  -- DECLARED "exact"
# ---------------------------------------------------------------------------

def _np_iinfo_min(dtype):
    return np.iinfo(dtype).min


def _ionp_iinfo_min(dtype):
    import anionpy
    return anionpy.iinfo(dtype).min


def _np_iinfo_max(dtype):
    return np.iinfo(dtype).max


def _ionp_iinfo_max(dtype):
    import anionpy
    return anionpy.iinfo(dtype).max


def _iinfo_attr_cases():
    return _spelling_cases(INT_DTYPES)


FINFO_IINFO_SPECS["iinfo.min"] = ItemSpec(
    name="iinfo.min", kind="custom", custom_cases=_iinfo_attr_cases,
    numpy_adapter=_np_iinfo_min, ionp_adapter=_ionp_iinfo_min,
    scalar_like=True,
)
FINFO_IINFO_SPECS["iinfo.max"] = ItemSpec(
    name="iinfo.max", kind="custom", custom_cases=_iinfo_attr_cases,
    numpy_adapter=_np_iinfo_max, ionp_adapter=_ionp_iinfo_max,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# iinfo.__repr__ / iinfo.__str__  -- DECLARED "exact"
# ---------------------------------------------------------------------------

def _np_iinfo_repr(dtype):
    return repr(np.iinfo(dtype))


def _ionp_iinfo_repr(dtype):
    import anionpy
    return repr(anionpy.iinfo(dtype))


def _np_iinfo_str(dtype):
    return str(np.iinfo(dtype))


def _ionp_iinfo_str(dtype):
    import anionpy
    return str(anionpy.iinfo(dtype))


FINFO_IINFO_SPECS["iinfo.__repr__"] = ItemSpec(
    name="iinfo.__repr__", kind="custom", custom_cases=_iinfo_attr_cases,
    numpy_adapter=_np_iinfo_repr, ionp_adapter=_ionp_iinfo_repr,
)
FINFO_IINFO_SPECS["iinfo.__str__"] = ItemSpec(
    name="iinfo.__str__", kind="custom", custom_cases=_iinfo_attr_cases,
    numpy_adapter=_np_iinfo_str, ionp_adapter=_ionp_iinfo_str,
)


# ---------------------------------------------------------------------------
# finfo.epsneg / .iexp / .machep / .negep / .nexp / .resolution / .tiny
#   -- DECLARED "exact"
# ---------------------------------------------------------------------------

_FINFO_DTYPES = FLOAT_DTYPES + COMPLEX_DTYPES


def _finfo_attr_cases():
    return _spelling_cases(_FINFO_DTYPES)


def _make_finfo_attr_adapters(attr):
    def np_adapter(dtype):
        v = getattr(np.finfo(dtype), attr)
        return (v, type(v).__name__)

    def ionp_adapter(dtype):
        import anionpy
        v = getattr(anionpy.finfo(dtype), attr)
        return (v, type(v).__name__)

    return np_adapter, ionp_adapter


for _attr in ("epsneg", "iexp", "machep", "negep", "nexp", "resolution", "tiny"):
    _np_adapter, _ionp_adapter = _make_finfo_attr_adapters(_attr)
    FINFO_IINFO_SPECS[f"finfo.{_attr}"] = ItemSpec(
        name=f"finfo.{_attr}", kind="custom", custom_cases=_finfo_attr_cases,
        numpy_adapter=_np_adapter, ionp_adapter=_ionp_adapter,
        scalar_like=True,
    )


# ---------------------------------------------------------------------------
# finfo.__repr__ / finfo.__str__  -- DECLARED "exact"
# ---------------------------------------------------------------------------

def _np_finfo_repr(dtype):
    return repr(np.finfo(dtype))


def _ionp_finfo_repr(dtype):
    import anionpy
    return repr(anionpy.finfo(dtype))


def _np_finfo_str(dtype):
    return str(np.finfo(dtype))


def _ionp_finfo_str(dtype):
    import anionpy
    return str(anionpy.finfo(dtype))


FINFO_IINFO_SPECS["finfo.__repr__"] = ItemSpec(
    name="finfo.__repr__", kind="custom", custom_cases=_finfo_attr_cases,
    numpy_adapter=_np_finfo_repr, ionp_adapter=_ionp_finfo_repr,
)
FINFO_IINFO_SPECS["finfo.__str__"] = ItemSpec(
    name="finfo.__str__", kind="custom", custom_cases=_finfo_attr_cases,
    numpy_adapter=_np_finfo_str, ionp_adapter=_ionp_finfo_str,
)


# ---------------------------------------------------------------------------
# iinfo.__init__/.__new__, finfo.__init__/.__new__ -- NOT declared.
#
# Registered for visibility (so a regression or a future fix of the
# void/string-dtype gap shows up as a status CHANGE rather than silence),
# but NOT part of the "these pass" claim: the 4 confirmed invalid-input
# mismatches (dict, nonsense string, `str` type -> object/void-dtype
# fallback error TEXT divergence; see this ticket's doc) live in exactly
# these two construction paths for each class. Included so the merge
# instruction is honest about what exists in this file, not just what was
# declared.
# ---------------------------------------------------------------------------

def _np_iinfo_new(dtype):
    try:
        return ("ok", repr(np.iinfo(dtype)))
    except Exception as e:  # noqa: BLE001 -- comparing exception identity
        return ("error", type(e).__name__, str(e))


def _ionp_iinfo_new(dtype):
    import anionpy
    try:
        return ("ok", repr(anionpy.iinfo(dtype)))
    except Exception as e:  # noqa: BLE001
        return ("error", type(e).__name__, str(e))


def _np_finfo_new(dtype):
    try:
        return ("ok", repr(np.finfo(dtype)))
    except Exception as e:  # noqa: BLE001
        return ("error", type(e).__name__, str(e))


def _ionp_finfo_new(dtype):
    import anionpy
    try:
        return ("ok", repr(anionpy.finfo(dtype)))
    except Exception as e:  # noqa: BLE001
        return ("error", type(e).__name__, str(e))


def _invalid_input_cases():
    invalid = [
        ("dict", {}),
        ("nonsense_str", "totally_bogus_dtype_xyz"),
        ("str_type", str),
        ("none_value", None),
        ("plain_object", object()),
    ]
    return [(label, (val,), {}) for label, val in invalid]


FINFO_IINFO_SPECS["iinfo.__new__"] = ItemSpec(
    name="iinfo.__new__", kind="custom", custom_cases=_invalid_input_cases,
    numpy_adapter=_np_iinfo_new, ionp_adapter=_ionp_iinfo_new,
)
FINFO_IINFO_SPECS["iinfo.__init__"] = ItemSpec(
    name="iinfo.__init__", kind="custom", custom_cases=_invalid_input_cases,
    numpy_adapter=_np_iinfo_new, ionp_adapter=_ionp_iinfo_new,
)
FINFO_IINFO_SPECS["finfo.__new__"] = ItemSpec(
    name="finfo.__new__", kind="custom", custom_cases=_invalid_input_cases,
    numpy_adapter=_np_finfo_new, ionp_adapter=_ionp_finfo_new,
)
FINFO_IINFO_SPECS["finfo.__init__"] = ItemSpec(
    name="finfo.__init__", kind="custom", custom_cases=_invalid_input_cases,
    numpy_adapter=_np_finfo_new, ionp_adapter=_ionp_finfo_new,
)
