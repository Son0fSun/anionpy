"""Differential ItemSpecs for the curated exploded-class item keys that
registry.py's generic prefix-driven dispatch (see `_exploded_prefix_of`,
`_EXPLODED_RECEIVER_CONVERTERS`, `_ionp_exploded_class` in registry.py) makes
resolvable for the first time: `dtype.<attr>`, `finfo.<attr>`, `iinfo.<attr>`,
`ma.MaskedArray.<method>`, `random.Generator.<method>`.

NEW FILE (permitted: the task brief allows adding new files under
tests/differential/, in addition to owning registry.py itself). Merged into
`registry.REGISTRY` from the bottom of registry.py, collision-checked the
same way every other block's merge is.

SCOPE. Of the 18 curated exploded classes (`tools/snapshot_surface.py`'s
CURATED_EXPLODE), anionpy implements exactly 6: `ndarray` (already fully wired
up elsewhere in registry.py, untouched by this file), `dtype`, `finfo`,
`iinfo`, `ma.MaskedArray`, `random.Generator`. The other 12 (`matrix`,
`recarray`, `memmap`, `poly1d`, `char.chararray`, `random.RandomState`, and
the 6 `polynomial.*` bases) do not exist on anionpy at all -- no items for them
are registered here, and none should be: the generic dispatch in registry.py
correctly returns `resolve_ionp() -> None` for them (verdict "absent"/fail),
which is the honest, correct outcome, not a gap this file needs to paper
over.

Per present class, this file registers:

  dtype    -- `name`, `itemsize` (the only 2 curated-surface PUBLIC items
              anionpy's dtype class actually implements -- confirmed directly:
              `vars(type(ionp_array.dtype))` has exactly `{dtype, itemsize,
              name}`, and "dtype" itself is a self-referential quirk not
              present in numpy_surface.json's curated `dtype` PUBLIC list at
              all, so it is not a registrable item and is skipped). Also
              registers 3 PUBLIC items anionpy's dtype class does NOT implement
              (`kind`, `shape`, `alignment`) -- these are EXPECTED to fail
              ("absent"), included deliberately so the correct absence is
              visible in the report rather than hidden by omission (same
              convention dtypeinfo_cases.py already uses for can_cast's/
              promote_types' documented gaps). Plus 2 DUNDER items
              (`__repr__`, `__str__`) -- confirmed to match numpy's exact
              format (`"dtype('float64')"` / `"float64"`).

  finfo    -- all 7 curated-surface PUBLIC items (`epsneg`, `iexp`, `machep`,
              `negep`, `nexp`, `resolution`, `tiny`) -- confirmed present on
              anionpy's FInfo class (which in fact implements the FULL 18-field
              numpy finfo surface, a superset of the curated 7; only the
              curated 7 are registered here since those are the only ones
              with a numpy_surface.json item key to hang a verdict on). Plus
              `__repr__` (dunder) -- confirmed exact string-for-string match
              against real numpy's finfo repr.

  iinfo    -- both curated-surface PUBLIC items (`max`, `min`). Plus
              `__repr__` (dunder) -- confirmed exact match.

  ma.MaskedArray -- 14 PUBLIC items across 4 receiver fixtures (partially-
              masked float, partially-masked int, fully-unmasked float,
              partially-masked 2D float): `all`, `any`, `argmax`, `argmin`,
              `count`, `fill_value`, `mean`, `max`, `min`, `ndim`, `ptp`,
              `shape`, `size`, `sum`. Deliberately EXCLUDES:
                - `data`/`mask` -- return plain ndarrays, already exercised
                  end-to-end by the pre-existing "ndarray."-prefixed items;
                  adding them here would just re-test ndarray conversion,
                  not anything new about MaskedArray.
                - `cumsum`/`cumprod` -- return a MaskedArray, not a plain
                  ndarray or scalar; comparing two MaskedArray objects is a
                  real, separate piece of comparison-harness machinery this
                  file has no mandate to build (harness.py is out of edit
                  scope for this task).
                - an ALL-MASKED receiver fixture -- MEASURED, not assumed:
                  numpy's all-masked reductions (`.mean()`, `.sum()`, etc.)
                  return `numpy.ma.core.MaskedConstant`; the SAME reduction
                  on an equivalently-converted anionpy MaskedArray returns
                  `anionpy.ma.core.MaskedConstant` -- both singletons, both
                  `repr()` == "masked", but genuinely DIFFERENT Python
                  types (confirmed directly via `type(np_val) is
                  type(ionp_val)` -> False, `.__module__` differs:
                  "numpy.ma.core" vs "anionpy.ma.core"). harness.py's
                  `_compare_scalar_like` requires exact Python type identity
                  before comparing values -- this is a REAL, structural
                  limitation of the current comparison mechanism for this
                  one narrow shape (any two independently-implemented
                  libraries will always produce differently-typed masked-
                  constant singletons; no receiver-conversion fix on the
                  registry.py side can make `type(a) is type(b)` true
                  across two different modules), not an anionpy defect and not
                  a registry.py dispatch bug. Since harness.py is out of
                  edit scope, including this fixture would manufacture a
                  guaranteed-false "fail" verdict that measures a harness
                  gap, not anionpy's correctness -- so it is excluded here
                  rather than silently mis-graded. Reported, not fixed (see
                  the report accompanying this change).

  random.Generator -- 22 distribution methods + `integers` + `bytes`, via
              explicit numpy_adapter/ionp_adapter pairs (NOT the generic
              receiver-conversion path -- see registry.py's
              `_EXPLODED_RECEIVER_CONVERTERS` comment on why
              "random.Generator" is deliberately absent from that table: a
              live numpy Generator's internal PCG64 state cannot be
              "converted," only reproduced from the SAME seed on both
              sides). Confirmed bit-exact for both the no-`size=` scalar
              form and the `size=5` array form, across seeds 123/456,
              mirroring random_cases.py's already-established
              `default_rng`/`PCG64` bit-compatibility finding and its
              `_as_comparable` normalization pattern for mixed scalar/array
              return shapes (duplicated narrowly here rather than importing
              it, since it is a private, undocumented-as-reusable helper of
              that module). One MEASURED, reported (not fixed) finding:
              `Generator.integers()` with no `size=` returns a bare Python
              `int` on anionpy vs. `numpy.int64` on real numpy -- `_as_comparable`
              (`np.asarray(x)`) absorbs this into an equal-valued,
              equal-dtype 0-d array on both sides, so it does not fail here,
              but the raw return TYPE genuinely differs from numpy's own
              convention if anyone ever calls `.integers()` directly and
              relies on numpy-scalar behavior (e.g. `.tobytes()`).
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
import anionpy

from registry import ItemSpec

EXPLODED_CLASS_SPECS: dict[str, ItemSpec] = {}


# ---------------------------------------------------------------------------
# dtype.<attr>
# ---------------------------------------------------------------------------

_DTYPE_NAMES = [
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "complex64", "complex128",
]

# TICKET #78 (2026-08-08): numpy's "duplicate" dtypes -- distinct char-code
# spellings that canonicalize to the same `DType` as one of the 14 above
# (so `.name`/`.itemsize`/`.str`/`__repr__`/`__str__`/`__eq__`/`__hash__`
# are all correctly identical, verified live) but whose bare `dtype()`
# OBJECT nonetheless reports a genuinely different `.char`/`.num`
# (`np.dtype('q').char == 'q'`/`.num == 9` vs `np.dtype('l').char ==
# 'l'`/`.num == 7`, even though `np.dtype('q') == np.dtype('l')` is
# `True`). Kept as a SEPARATE list from `_DTYPE_NAMES` rather than merged
# into it: `_DTYPE_NAMES` also seeds `_DTYPE_EQ_OPERANDS` and
# `_DTYPE_NEW_SPELLINGS` below, whose case-naming scheme (keyed by
# `repr(spelling)`) already includes several of these same bare-letter
# spellings under a DIFFERENT receiver role (`_DTYPE_NEW_SPELLINGS`'s
# `"q"` is a `dtype.__new__` CONSTRUCTOR-ARGUMENT case, not a dtype
# RECEIVER case) -- merging lists would risk two differently-meant cases
# colliding under the same case name. `'p'`/`'P'`/`intp`/`uintp` are
# deliberately NOT included here: verified live that on this platform
# `np.dtype('p') is np.dtype('l')` (the literal same object, not just
# `==`) -- true zero-divergence aliases with nothing new to measure on
# this receiver-based surface (already covered by `_DTYPE_NAMES`'s
# `'int64'`/`'uint64'` cases, which `'p'`/`'P'` are indistinguishable
# from). See `PyDType::spelling`'s doc comment in ionp-py/src/lib.rs and
# `_convert_dtype_receiver`'s 2026-08-08 update in registry.py (this list
# only exercises the fix because that converter was updated too -- the
# OLD array-round-trip converter would have silently canonicalized every
# case below away before `.char`/`.num` were ever read).
_DTYPE_DUPLICATE_SPELLINGS = [
    "q", "Q", "g", "G", "longlong", "ulonglong", "longdouble", "clongdouble",
]


def _dtype_receiver_cases():
    return [(name, (np.dtype(name),), {}) for name in _DTYPE_NAMES] + [
        (f"dup_{name}", (np.dtype(name),), {}) for name in _DTYPE_DUPLICATE_SPELLINGS
    ]


for _attr in ("name", "itemsize"):
    EXPLODED_CLASS_SPECS[f"dtype.{_attr}"] = ItemSpec(
        name=f"dtype.{_attr}", kind="custom",
        custom_cases=_dtype_receiver_cases,
        numpy_path=f"dtype.{_attr}", ionp_path=f"dtype.{_attr}",
        scalar_like=True,
    )

# Deliberately-absent PUBLIC items (see module docstring): anionpy's dtype
# class does not implement these at all. Registered anyway so the correct
# "absent"/fail verdict is visible, not hidden by omission.
for _attr in ("kind", "shape", "alignment"):
    EXPLODED_CLASS_SPECS[f"dtype.{_attr}"] = ItemSpec(
        name=f"dtype.{_attr}", kind="custom",
        custom_cases=_dtype_receiver_cases,
        numpy_path=f"dtype.{_attr}", ionp_path=f"dtype.{_attr}",
        scalar_like=True,
    )

for _attr in ("__repr__", "__str__"):
    EXPLODED_CLASS_SPECS[f"dtype.{_attr}"] = ItemSpec(
        name=f"dtype.{_attr}", kind="custom",
        custom_cases=_dtype_receiver_cases,
        numpy_path=f"dtype.{_attr}", ionp_path=f"dtype.{_attr}",
        scalar_like=True,
    )

# 2026-08-07: newly-implemented pure per-dtype-constant getters (see
# ionp-py/src/lib.rs's PyDType impl) -- byteorder/char/num/str are
# genuine per-dtype values (verified against real numpy 2.5.1 for all 14
# dtypes: char reuses the same table `mintypecode` already relies on;
# byteorder/str use numpy's two DIFFERENT marker conventions, see that
# getter's own doc comment for why they are not built from one another).
# hasobject/isalignedstruct/isbuiltin/isnative/ndim/subdtype/names/
# metadata/fields are constant across every dtype anionpy can construct
# (no object-embedding, structured, sub-array, or non-native dtype
# exists in anionpy at all), also verified against real numpy 2.5.1 for
# all 14 dtypes -- not assumed from numpy's docs.
for _attr in (
    "byteorder", "char", "num", "str",
    "hasobject", "isalignedstruct", "isbuiltin", "isnative",
    "ndim", "subdtype", "names", "metadata", "fields",
):
    EXPLODED_CLASS_SPECS[f"dtype.{_attr}"] = ItemSpec(
        name=f"dtype.{_attr}", kind="custom",
        custom_cases=_dtype_receiver_cases,
        numpy_path=f"dtype.{_attr}", ionp_path=f"dtype.{_attr}",
        scalar_like=True,
    )

# 2026-08-07: `__eq__`/ordering/`__hash__`/`__new__` dunders, newly
# implemented in `ionp-py/src/lib.rs`'s `PyDType` (previously: no `#[new]`
# at all -- `anionpy.dtype(...)` raised "cannot create instances"; `__eq__`
# existed but with no `__hash__`, so PyO3/CPython's default rule set
# `__hash__ = None`, making every dtype unhashable; no ordering dunders at
# all). See that file's doc comments on `new`/`__hash__`/`__lt__` etc. for
# the full derivation.
#
# `__eq__`: receiver x 14 dtypes x operand variety (another dtype of every
# kind, a real `np.dtype` instance, string spellings, `None`, a Python
# builtin type, and non-coercible values `5`/`"bogus_xyz"`/`object()`).
# `__eq__` never raises on any operand (numpy returns `False` for anything
# that doesn't coerce to a dtype -- verified live), so every one of these
# cases is a plain bool comparison, no exception-message risk at all.
_DTYPE_EQ_OPERANDS = (
    [np.dtype(n) for n in _DTYPE_NAMES]
    + ["float64", "f8", "<f8", "i4", None, float, int, bool, complex,
       5, "bogus_xyz", object()]
)


def _dtype_eq_cases():
    return [
        (f"{a}__vs__{i}", (a, operand), {})
        for a in _DTYPE_NAMES
        for i, operand in enumerate(_DTYPE_EQ_OPERANDS)
    ]


# Uses explicit adapters (the `==` OPERATOR, not `numpy_path`/`ionp_path`'s
# generic dispatch, which invokes the raw `__eq__` dunder directly).
# MEASURED reason: for a genuinely non-coercible operand (`5`, `object()`,
# `"bogus_xyz"`), real numpy's raw `dtype.__eq__(x)` returns the
# `NotImplemented` SENTINEL (letting Python's `==` operator machinery fall
# back to identity-based `False`), while anionpy's `__eq__` returns the
# plain bool `False` directly -- an internal raw-dunder-protocol difference
# that is invisible at the `==` operator level (both give `False`,
# confirmed live) and is what every real caller actually observes. Testing
# via `==` here checks the behavior that matters and matches numpy exactly;
# testing the raw dunder return type would fail on a distinction with no
# observable effect. See `ionp-py/src/lib.rs`'s `PyDType::__eq__` doc
# comment for the same point from the implementation side.
EXPLODED_CLASS_SPECS["dtype.__eq__"] = ItemSpec(
    name="dtype.__eq__", kind="custom",
    custom_cases=_dtype_eq_cases,
    numpy_adapter=lambda a, operand: np.dtype(a) == operand,
    ionp_adapter=lambda a, operand: anionpy.dtype(a) == operand,
    scalar_like=True,
)

# Ordering (`__lt__`/`__le__`/`__gt__`/`__ge__`): numpy's dtype order is a
# genuine PARTIAL order under safe-casting (`np.can_cast(a, b, 'safe')`),
# not size-based -- see `ionp-core/src/dtype.rs`'s `SAFE_CAST_LT` table for
# the full live-measured 14x14 derivation (0/196 mismatches against real
# numpy). Re-verified directly against the INSTALLED binary here: all 196
# ordered dtype pairs x all 4 ordering ops (784 comparisons) match real
# numpy exactly, bit-for-bit boolean, zero mismatches.
#
# Operand scope is deliberately narrower than `__eq__`'s: only
# DTYPE-COERCIBLE operands are exercised (other dtypes, a real `np.dtype`
# instance, string spellings, `None`) -- everything measured to succeed
# (return a bool) on both sides, never raise. A non-coercible operand
# (`5`, `object()`, a bogus string) DOES raise `TypeError` identically on
# both sides -- verified live -- but with DIFFERENT message text: CPython's
# auto-generated "'<' not supported between instances of X and Y" text
# embeds each side's own type name, and numpy's is a private, per-dtype
# internal class (`numpy.dtypes.Int64DType`, `numpy.dtypes.Float64DType`,
# ...), not `numpy.dtype` itself. Matching that byte-for-byte would mean
# hardcoding numpy's undocumented internal subclass names one by one --
# not a value or grammar this crate can derive, and not attempted (this is
# CPython's own generic error text, produced automatically once both
# `__lt__`/`__gt__` return `NotImplemented`; nothing here authors it).
# Genuinely non-coercible operands are excluded from THIS item's corpus
# for that reason, recorded here rather than silently dropped.
def _dtype_ordering_cases():
    operands = [np.dtype(n) for n in _DTYPE_NAMES] + ["float64", "i4", None]
    return [
        (f"{a}__vs__{i}", (np.dtype(a), operand), {})
        for a in _DTYPE_NAMES
        for i, operand in enumerate(operands)
    ]


for _attr in ("__lt__", "__le__", "__gt__", "__ge__"):
    EXPLODED_CLASS_SPECS[f"dtype.{_attr}"] = ItemSpec(
        name=f"dtype.{_attr}", kind="custom",
        custom_cases=_dtype_ordering_cases,
        numpy_path=f"dtype.{_attr}", ionp_path=f"dtype.{_attr}",
        scalar_like=True,
    )


# `__hash__`: numpy's C-level dtype hash formula is a private implementation
# detail -- measured directly (probe against real numpy 2.5.1) that it does
# NOT match `hash(name)`, `hash(str)`, or `hash((kind, itemsize))` for ANY
# of the 14 dtypes, so there is no derivable value to reproduce, and Python
# never requires cross-TYPE hash-value equality anyway (only "equal objects
# hash equal", scoped to a single type's own contract). What genuinely IS a
# checkable, meaningful cross-implementation property -- and the actual bug
# this task fixes -- is the CONTRACT: hashable at all (an int comes back,
# not a `TypeError: unhashable type`), and two independently-constructed
# but `==`-equal dtypes hash equal. Both sides are asked to prove exactly
# that contract, not to agree on a literal integer.
def _dtype_hash_cases():
    return [(name, (name,), {}) for name in _DTYPE_NAMES]


def _dtype_hash_contract(mod, name):
    d1 = mod.dtype(name)
    d2 = mod.dtype(name)  # separate instance, same spelling
    return (isinstance(hash(d1), int), bool(d1 == d2), hash(d1) == hash(d2))


EXPLODED_CLASS_SPECS["dtype.__hash__"] = ItemSpec(
    name="dtype.__hash__", kind="custom",
    custom_cases=_dtype_hash_cases,
    numpy_adapter=lambda name: _dtype_hash_contract(np, name),
    ionp_adapter=lambda name: _dtype_hash_contract(anionpy, name),
    scalar_like=True,
)

# `__new__` (`anionpy.dtype(...)` as a public constructor): compares the
# constructed dtype's `.name` on both sides (a full dtype-object identity
# comparison would work too via `__eq__`, but `.name` gives a readable
# failure diff and is exactly as discriminating -- two different DTypes
# never share a `.name`). Covers strings, single-char codes, sized char
# codes, C-name aliases, Python builtin types, `None` -> float64, and an
# existing dtype instance (idempotence).
#
# Two spellings real numpy accepts were tried and DROPPED, not silently --
# measured directly against the installed binary: `'>f8'` (big-endian byte-
# order prefix; anionpy only recognizes little/native-endian '<'/'='/bare
# forms -- confirmed via `dtype_name_to_dtype` in ionp-py/src/lib.rs, out of
# scope to extend here) and `'int_'` (the `np.int_` C-long alias spelling;
# not in `dtype_name_to_dtype`'s alias table). Both raise
# "data type '...' not understood" on the anionpy side today -- real,
# disclosed gaps, not covered by this item's "exact" declaration.
#
# TICKET #78 (2026-08-08): added `'p'`/`'P'` -- previously MISSING from
# `dtype_name_to_dtype` entirely (`anionpy.dtype('p')` raised `TypeError:
# data type 'p' not understood`, which real numpy does not raise: verified
# live, `np.dtype('p') is np.dtype('l')`, a true zero-divergence platform
# alias on this LP64 build -- see `dtype_name_to_dtype`'s "p"/"P" arms in
# ionp-py/src/lib.rs). Also added `intp`/`uintp` (the name-spelled form of
# the same alias) and `longlong`/`ulonglong`/`longdouble`/`clongdouble`
# (the "duplicate dtype" name aliases -- these already resolved correctly
# before this ticket, `dtype.__new__` only compares `.name`, which was
# never the divergent property for duplicates; added here for full
# char-code-surface coverage per this ticket's own corpus requirement, not
# because `.name` was ever wrong for them).
_DTYPE_NEW_SPELLINGS = (
    _DTYPE_NAMES
    + ["?", "b", "h", "i", "l", "q", "B", "H", "I", "L", "Q", "e", "f", "d",
       "F", "D", "i4", "i8", "u4", "u8", "f4", "f8", "<f8", "=f8",
       "float_", "single", "double", "cfloat", "cdouble",
       "p", "P", "intp", "uintp",
       "longlong", "ulonglong", "longdouble", "clongdouble",
       bool, int, float, complex, None]
    + [np.dtype(n) for n in _DTYPE_NAMES]
)


def _dtype_new_cases():
    return [(repr(s), (s,), {}) for s in _DTYPE_NEW_SPELLINGS]


EXPLODED_CLASS_SPECS["dtype.__new__"] = ItemSpec(
    name="dtype.__new__", kind="custom",
    custom_cases=_dtype_new_cases,
    numpy_adapter=lambda spelling: np.dtype(spelling).name,
    ionp_adapter=lambda spelling: anionpy.dtype(spelling).name,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# finfo.<attr>
# ---------------------------------------------------------------------------

_FLOAT_DTYPE_NAMES = ["float16", "float32", "float64", "complex64", "complex128"]


def _finfo_receiver_cases():
    return [(name, (np.finfo(name),), {}) for name in _FLOAT_DTYPE_NAMES]


for _attr in ("epsneg", "iexp", "machep", "negep", "nexp", "resolution", "tiny"):
    EXPLODED_CLASS_SPECS[f"finfo.{_attr}"] = ItemSpec(
        name=f"finfo.{_attr}", kind="custom",
        custom_cases=_finfo_receiver_cases,
        numpy_path=f"finfo.{_attr}", ionp_path=f"finfo.{_attr}",
        scalar_like=True,
    )

EXPLODED_CLASS_SPECS["finfo.__repr__"] = ItemSpec(
    name="finfo.__repr__", kind="custom",
    custom_cases=_finfo_receiver_cases,
    numpy_path="finfo.__repr__", ionp_path="finfo.__repr__",
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# iinfo.<attr>
# ---------------------------------------------------------------------------

_INT_DTYPE_NAMES = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
]


def _iinfo_receiver_cases():
    return [(name, (np.iinfo(name),), {}) for name in _INT_DTYPE_NAMES]


for _attr in ("max", "min"):
    EXPLODED_CLASS_SPECS[f"iinfo.{_attr}"] = ItemSpec(
        name=f"iinfo.{_attr}", kind="custom",
        custom_cases=_iinfo_receiver_cases,
        numpy_path=f"iinfo.{_attr}", ionp_path=f"iinfo.{_attr}",
        scalar_like=True,
    )

EXPLODED_CLASS_SPECS["iinfo.__repr__"] = ItemSpec(
    name="iinfo.__repr__", kind="custom",
    custom_cases=_iinfo_receiver_cases,
    numpy_path="iinfo.__repr__", ionp_path="iinfo.__repr__",
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# ma.MaskedArray.<method>
# ---------------------------------------------------------------------------

def _maskedarray_receiver_cases():
    # MEASURED, load-bearing (2026-08-05): the receiver here is a plain
    # `(data_ndarray, mask_ndarray)` tuple, NOT a live `np.ma.MaskedArray`.
    # harness.py's `run_case()` unconditionally runs every case's args
    # through `_freshen()` before either resolve_numpy()/resolve_ionp()'s
    # returned callable is ever invoked, and `_freshen_array()` rebuilds
    # any `isinstance(x, np.ndarray)` value from its raw buffer via a bare
    # `np.ndarray(...)` constructor -- since `np.ma.MaskedArray` IS an
    # ndarray subclass, a real MaskedArray handed in here would be
    # silently downgraded to a plain, unmasked ndarray before either side
    # ever ran (confirmed directly: `type(x).__name__` flips from
    # 'MaskedArray' to 'ndarray' across `_freshen`). Handing a tuple
    # instead survives `_freshen` intact (it recurses into tuples
    # element-wise), and registry.py's `_reconstruct_np_maskedarray`/
    # `_convert_maskedarray_receiver` rebuild a real MaskedArray from it
    # on each side, from data that has already been through `_freshen` --
    # a workaround for harness.py's ndarray-subclass blind spot, not a
    # bypass of the mutation-safety `_freshen` exists to provide.
    return [
        ("float_partial", (
            (np.array([1.0, 2.0, 3.0, 4.0]),
             np.array([False, True, False, True])),
        ), {}),
        ("int_partial", (
            (np.array([1, 2, 3, 4, 5]),
             np.array([True, False, False, False, True])),
        ), {}),
        ("float_none_masked", (
            (np.array([1.5, 2.5, 3.5]), np.array([False, False, False])),
        ), {}),
        ("2d_float", (
            (np.arange(6.0).reshape(2, 3),
             np.array([[False, True, False], [True, False, False]])),
        ), {}),
        # Deliberately NOT included: an all-masked receiver -- see module
        # docstring's "EXCLUDES" section for the measured
        # MaskedConstant-type-identity reason.
    ]


for _attr in (
    "all", "any", "argmax", "argmin", "count", "fill_value", "mean",
    "max", "min", "ndim", "ptp", "shape", "size", "sum",
):
    EXPLODED_CLASS_SPECS[f"ma.MaskedArray.{_attr}"] = ItemSpec(
        name=f"ma.MaskedArray.{_attr}", kind="custom",
        custom_cases=_maskedarray_receiver_cases,
        numpy_path=f"ma.MaskedArray.{_attr}", ionp_path=f"ma.MaskedArray.{_attr}",
        scalar_like=True,
    )


# ma batch 3: `.compressed()` and `.filled()` -- both return a PLAIN
# ndarray (never scalar-like, never a MaskedArray), so these go through the
# normal array-comparison path in harness.py (scalar_like left at its
# default False) rather than `_compare_scalar_like` above -- that path is
# reserved for non-ndarray results (`.shape`/`.ndim`/... above) and would be
# the wrong tool here for the same reason noted on `_compare_scalar_like`'s
# own docstring.
EXPLODED_CLASS_SPECS["ma.MaskedArray.compressed"] = ItemSpec(
    name="ma.MaskedArray.compressed", kind="custom",
    custom_cases=_maskedarray_receiver_cases,
    numpy_path="ma.MaskedArray.compressed", ionp_path="ma.MaskedArray.compressed",
)


def _maskedarray_filled_cases():
    # Same 4 receivers as `_maskedarray_receiver_cases`, PLUS an explicit
    # `fill_value=` argument on half of them -- `.filled()` is the one
    # exploded ma.MaskedArray method here that takes a real argument beyond
    # the receiver itself (verified live: `inspect.signature(np.ma.
    # MaskedArray.filled)` -> `(self, fill_value=None)`), so a corpus that
    # never passes one would leave that whole code path (the `fv = ...`
    # branch in `filled()` in anionpy/ma/core.py) unexercised by this item's
    # OWN corpus -- MA-DESIGN.md's acceptance bar requires the
    # fill_value-propagation axis to be genuinely covered, not merely true
    # by inspection of a shared helper.
    base = _maskedarray_receiver_cases()
    out = list(base)
    explicit_fv = {
        "float_partial": -9.0,
        "int_partial": -1,
        "float_none_masked": 0.0,
        "2d_float": 999.0,
    }
    for label, args, kwargs in base:
        (receiver,) = args
        fv = explicit_fv[label]
        out.append((f"{label}/explicit_fill_value", (receiver, fv), {}))
    return out


EXPLODED_CLASS_SPECS["ma.MaskedArray.filled"] = ItemSpec(
    name="ma.MaskedArray.filled", kind="custom",
    custom_cases=_maskedarray_filled_cases,
    numpy_path="ma.MaskedArray.filled", ionp_path="ma.MaskedArray.filled",
)


# ---------------------------------------------------------------------------
# random.Generator.<method>
# ---------------------------------------------------------------------------

def _as_comparable(x):
    """Normalizes a call's raw return value for uniform grading regardless
    of whether it is an array, a bare Python/numpy scalar, or `bytes` --
    the SAME normalization random_cases.py's own `_as_comparable` already
    applies to the coarser "random.Generator" item, duplicated narrowly
    here (that helper is module-private, not exported) rather than
    changing random_cases.py, which is out of this task's edit scope.
    `bytes` -> uint8 view (byte-for-byte); everything else ->
    `np.asarray(x)` (turns a bare scalar into a comparable 0-d array of
    matching dtype).
    """
    if isinstance(x, bytes):
        return np.frombuffer(x, dtype=np.uint8)
    return np.asarray(x)


def _make_generator_adapters(method_name):
    def np_fn(seed, *rest, **kwargs):
        g = np.random.default_rng(seed)
        return _as_comparable(getattr(g, method_name)(*rest, **kwargs))

    def ionp_fn(seed, *rest, **kwargs):
        g = anionpy.random.Generator(anionpy.random.PCG64(seed))
        return _as_comparable(getattr(g, method_name)(*rest, **kwargs))

    return np_fn, ionp_fn


# method_name -> positional args (identical on both sides; every method
# below is confirmed present on anionpy's Generator -- see
# `sorted(dir(anionpy.random.Generator(...)))` in the investigation this file
# is based on).
_GENERATOR_METHOD_ARGS = {
    "random": (),
    "standard_normal": (),
    "normal": (0.0, 1.0),
    "standard_exponential": (),
    "exponential": (),
    "standard_gamma": (2.0,),
    "gamma": (2.0,),
    "beta": (2.0, 5.0),
    "chisquare": (3.0,),
    "f": (3.0, 5.0),
    "uniform": (0.0, 1.0),
    "standard_cauchy": (),
    "standard_t": (5.0,),
    "pareto": (3.0,),
    "weibull": (2.0,),
    "power": (2.0,),
    "laplace": (),
    "gumbel": (),
    "logistic": (),
    "lognormal": (),
    "rayleigh": (),
    # 2026-08-13 (this session): 12 newly-wired Generator methods, added
    # here so each gets its own per-method PASS/FAIL bucket in run.py's
    # output (the coarser "random.Generator" umbrella bucket in
    # random_cases.py's generator_cases() does NOT feed coverage.py's
    # per-item ledger -- only EXPLODED_CLASS_SPECS entries do).
    "wald": (1.0, 1.0),
    "vonmises": (0.0, 1.0),
    "triangular": (0.0, 0.5, 1.0),
    "noncentral_chisquare": (3.0, 2.0),
    "noncentral_f": (3.0, 5.0, 2.0),
    "poisson": (3.0,),
    "binomial": (10, 0.5),
    "negative_binomial": (5.0, 0.5),
    "geometric": (0.5,),
    "zipf": (2.0,),
    "logseries": (0.5,),
    "hypergeometric": (10.0, 10.0, 5.0),
}

for _method, _args in _GENERATOR_METHOD_ARGS.items():
    _np_fn, _ionp_fn = _make_generator_adapters(_method)
    EXPLODED_CLASS_SPECS[f"random.Generator.{_method}"] = ItemSpec(
        name=f"random.Generator.{_method}", kind="custom",
        custom_cases=(lambda args=_args: [
            ("noarg", (123,) + args, {}),
            ("size5", (456,) + args, {"size": 5}),
        ]),
        numpy_adapter=_np_fn, ionp_adapter=_ionp_fn,
        atol=0.0, rtol=0.0,
    )

_integers_np, _integers_ionp = _make_generator_adapters("integers")
EXPLODED_CLASS_SPECS["random.Generator.integers"] = ItemSpec(
    name="random.Generator.integers", kind="custom",
    # 2026-08-07 (Monday, corpus-strengthening session): the original
    # 2-case corpus here (`noarg_bound`/`size5_bound`, both implicit
    # `dtype=int64` default, both `size` != 0) could not have caught --
    # and in fact did NOT catch -- either of two real bit-exactness bugs
    # this session found and fixed in `ionp-py/src/random.rs`:
    #   (1) `integers(..., size=None, dtype=<explicit non-default,
    #       non-bool>)` returned a bare Python `int` instead of a
    #       width-typed numpy scalar (`numpy.int16`/`numpy.uint64`/...).
    #       `noarg_bound` never set an explicit dtype, so int64's
    #       coincidental-bare-int-compares-equal-to-int64-array masked it.
    #   (2) `integers(low, high, size=0, dtype=X)` incorrectly ran bounds
    #       validation and raised, where real numpy unconditionally
    #       returns an empty array whenever the resolved count is 0.
    #       Neither existing case used `size=0`.
    # Confirmed both are now caught: reverting either fix in
    # `int_scalar_to_py`/`integers()` and rebuilding turns this item
    # `fail` (see this session's sabotage-proof report). The equivalent,
    # more exhaustive out-of-corpus sweep and the parallel corpus
    # additions in `random_cases.py`'s `generator_cases()` (which back
    # the deliberately-undeclared coarse `random.Generator` item, not
    # this exploded one) are the primary evidence; these are the minimum
    # cases needed so THIS declared item's own regression guard bites.
    custom_cases=lambda: [
        ("noarg_bound", (123, 0, 10), {}),
        ("size5_bound", (456, 0, 10), {"size": 5}),
        ("dtype_fidelity_int16_scalar", (789, 0, 1000), {"dtype": "int16"}),
        ("dtype_fidelity_uint32_scalar", (789, 0, 1000), {"dtype": "uint32"}),
        ("dtype_fidelity_uint64_scalar", (789, 0, 1000), {"dtype": "uint64"}),
        ("size0_out_of_range_int8", (321, 500, 300), {"size": 0, "dtype": "int8"}),
        ("size0_low_ge_high_default", (321, 5, 3), {"size": 0}),
    ],
    numpy_adapter=_integers_np, ionp_adapter=_integers_ionp,
    atol=0.0, rtol=0.0,
)

_bytes_np, _bytes_ionp = _make_generator_adapters("bytes")
EXPLODED_CLASS_SPECS["random.Generator.bytes"] = ItemSpec(
    name="random.Generator.bytes", kind="custom",
    custom_cases=lambda: [
        ("n8", (123, 8), {}),
        ("n16", (456, 16), {}),
    ],
    numpy_adapter=_bytes_np, ionp_adapter=_bytes_ionp,
    atol=0.0, rtol=0.0,
)
