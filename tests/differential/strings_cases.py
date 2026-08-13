"""Differential cases + registry override for the `numpy.char`/`numpy.strings`
block (28 primary items: add, the ten unary predicates, and the six
comparison ops, each exposed under both `char.*` and `strings.*`).

WHY THIS FILE OVERRIDES EXISTING REGISTRY ENTRIES (not adds new ones):
`np.strings.add`/`np.strings.isalpha`/etc are genuinely `numpy.ufunc`
instances (verified: `type(np.strings.add) is numpy.ufunc`), so
`ufunc_introspect.UFUNC_NAMES` already includes all 28 `char.*`/`strings.*`
names, and `ufunc_registry.py` already auto-derives `kind="ufunc"`
`ItemSpec` entries for them and merges them into `registry.REGISTRY`. Those
auto-derived entries resolve correctly on the numpy side, but their
`resolve_ionp()` path (`ItemSpec.resolve_ionp`'s `kind == "ufunc"` branch)
unconditionally converts every numpy array argument via
`make_ionp_array_converter` -> `anionpy.array()`, which raises `TypeError` for
string dtypes (`anionpy.ndarray`/`ionp_core::DType` has no string variant --
see `ionp-py/src/strings.rs`'s module docstring for why this block
deliberately never constructs one). So all 28 entries exist today and FAIL.

This file replaces those 28 entries with `kind="custom"` ones,
`convert_ionp_args=False` (the documented escape hatch in
`registry.py`'s `ItemSpec.convert_ionp_args` field -- built for exactly
this: "fixtures ... deliberately built from real numpy semantics rather
than standing in for a real anionpy.ndarray call"). `numpy_path`/`ionp_path`
default to `name`, so `spec.resolve_numpy()`/`resolve_ionp()` fall through
to plain dotted-attribute lookup (`numpy.char.add`, `anionpy.char.add`, ...)
-- both sides get the exact same real numpy `S`/`U` array object, matching
how a real caller would use `anionpy.char`/`anionpy.strings` directly.

ARCHITECTURAL NOTE on WHY THIS OVERRIDE HAPPENS FROM `run.py`, NOT HERE:
`registry.py`'s own tail (where `LINALG_SPECS`/`INPLACE_SPECS`/etc merge
in) executes to completion BEFORE `ufunc_registry.py`'s
`from registry import REGISTRY` line returns -- so entries appended there
would already be present in `registry.REGISTRY` by the time
`ufunc_registry.py`'s collision guard (`_collisions = set(UFUNC_SPECS) &
set(registry.REGISTRY)`) runs, and that guard raises `AssertionError` on
ANY collision, override-intent or not. The only point in the import graph
that runs strictly after `ufunc_registry.py`'s merge completes is
`run.py`'s module body itself (imported before `from registry import
REGISTRY`... no -- see run.py: `import ufunc_registry` is the second
import, this module is imported as a THIRD, subsequent line, so by the
time this module's top-level code below runs, `registry.REGISTRY` already
contains the 28 kind="ufunc" entries this file intentionally replaces).
`test_differential.py` picks this up automatically too, since it does
`from run import build_cases`, which imports (and therefore fully
executes) `run.py` as a module.

Corpus design, per the declaration bar (task brief): every dtype/content/
shape/keyword/foreign-input axis that's cheap to sweep in one file --
S1/S4/S16 and U1/U4/U16/U24 widths, ASCII/Latin-1/CJK/emoji(astral)/
combining-mark/embedded-NUL/trailing-whitespace/empty content, plus
Arabic-Indic and circled-digit content specifically because they are the
two documented Unicode-classification edge cases this block's
`ionp-core/src/strings.rs` handles explicitly (Nd category via
`unicode_general_category`, and a hardcoded Numeric_Type=Digit table for
`isdigit` -- see that module's docstrings for the exact, still-incomplete
boundary of that table). Shapes: 0-d, empty, 1-D, 2-D, 3-D, and
broadcasting pairs for `add`/comparisons. Foreign input: plain Python
`str`/`bytes` and `numpy.str_`/`numpy.bytes_` scalars mixed with real
arrays, plus the S/U dtype-mismatch cases that must raise (exercising the
real `numpy._core._exceptions._UFuncBinaryResolutionError` /
`_UFuncNoLoopError` exception classes).

CLASS C (foreign/numeric dtype input) and CLASS D (Python list / bare
non-str/bytes scalar input) cases (`_unary_foreign_cases`,
`_add_foreign_cases`, `_comparison_foreign_cases` below): added after an
independent, wider probe found `ionp-py/src/strings.rs`'s first attempt at
these two classes was itself a fresh defect, not a fix -- it delegated the
`Foreign` branch of every string ufunc straight to `numpy.strings.<name>`/
`numpy.<name>`, which for `add`/the six comparisons on numeric input
returns REAL NUMPY'S OWN COMPUTED ANSWER standing in as anionpy's, and for
the ten unary predicates raised REAL NUMPY'S OWN EXCEPTION OBJECT --
either way, the differential comparison was numpy-vs-numpy through a
one-line indirection, passing regardless of whether anionpy's own code path
was ever exercised. `np.strings.add is np.add` does NOT mean "anionpy may
call numpy.add"; it means "anionpy.strings.add must literally BE anionpy.add".
Fixed: `add`/comparisons on foreign-dtype operands now route to anionpy's own
`ionp_core::ufunc::binary_op` (`ionp_binary_dispatch` in
ionp-py/src/strings.rs) -- genuinely computing via anionpy's own engine for
numeric operands, and raising anionpy's own genuine "unsupported numpy dtype"
`TypeError` (from `extract_array`, since `ionp_core::NdArray` has no
string dtype at all) for the S-vs-U-mismatch and numeric-vs-string cases.
The ten unary predicates now construct numpy's own deterministic
not-a-loop message text themselves (`numpy_no_loop_type_error`, built from
the ufunc name and the input's real dtype CLASS NAME -- input-side
metadata read off the array numpy handed us, not an answer numpy
computed) and raise a plain `TypeError`, never importing real numpy to
produce a VALUE or an EXCEPTION on any path exercised by these cases.
Real numpy's actual exception classes here (`_UFuncNoLoopError`,
`_UFuncBinaryResolutionError` -- both private `TypeError` subclasses,
verified via `.__mro__`) are declared as accepted via
`exception_equivalences` below rather than imported into
`ionp-py/src/strings.rs` to construct instances of them.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
import registry
from registry import ItemSpec

# ---------------------------------------------------------------------------
# Shared content corpus
# ---------------------------------------------------------------------------

# Content strings, deliberately spanning every category the declaration bar
# calls out. Kept as plain `str`; each dtype-width case below encodes this
# same list into the target dtype (truncating is never allowed to silently
# happen -- every entry here is short enough to fit U24/S24 unpadded).
_CONTENT_U = [
    "",
    "a",
    "ABC",
    "AbC",
    "  spaced  ",
    "CAFÉ",
    "café",
    "日本語",
    "\U0001F600\U0001F525",  # astral emoji (surrogate-pair-width codepoints)
    "z1",
    "Title Case",
    "MiXeD_Case9",
    "éclair",  # combining acute accent (NFD 'é')
    "trailing \t\n",
    "1234",
    "١٢٣",  # Arabic-Indic digits (Unicode Nd, non-ASCII)
    "①②③",  # circled digit one/two/three (Numeric_Type=Digit, not Nd)
    "\x00emb\x00ed",
    "²³¹",  # superscript 2 3 1 (Numeric_Type=Digit, not Nd)
]

# `S` (bytes) content: ASCII + high-byte Latin-1, per the C-locale rule
# (`ionp-core/src/strings.rs`'s `is_bytes` branch treats codepoints > 127
# as never alphabetic/decimal/space regardless of what Latin-1 character
# they'd name). encode("latin-1") keeps every codepoint <= 0xFF representable.
_CONTENT_S = [c.encode("latin-1") for c in [
    "",
    "a",
    "ABC",
    "  sp  ",
    "CAFE",
    "cafe",
    "\xe9clair",  # Latin-1 'é' as a raw high byte -- NOT ascii-alphabetic
    "z1",
    "1234",
    "\x00e\x00",
    "Z9x",
]]

_U_WIDTHS = [1, 4, 16, 24]
_S_WIDTHS = [1, 6, 16, 24]


def _u_arrays():
    for width in _U_WIDTHS:
        yield f"U{width}", np.array(_CONTENT_U, dtype=f"<U{width}")


def _s_arrays():
    for width in _S_WIDTHS:
        yield f"S{width}", np.array(_CONTENT_S, dtype=f"S{width}")


def _shape_variants_u(width=10):
    # 0-d, empty, 1-D, 2-D, 3-D -- every shape rank the declaration bar
    # asks for, all built off the same short word list so results are
    # human-checkable in a failure message.
    words6 = ["ab", "CD", "e", "日本", "", "Z9"]
    return [
        ("0d", np.array("hello world", dtype=f"<U{width}")),
        ("empty", np.array([], dtype=f"<U{width}")),
        ("1d", np.array(words6, dtype=f"<U{width}")),
        ("2d", np.array(words6, dtype=f"<U{width}").reshape(2, 3)),
        ("3d", np.array(words6 + words6, dtype=f"<U{width}").reshape(2, 2, 3)),
    ]


# ---------------------------------------------------------------------------
# custom_cases builders
# ---------------------------------------------------------------------------

def _unary_cases():
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/full", (arr,), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/full", (arr,), {}))
    for tag, arr in _shape_variants_u():
        cases.append((f"shape_{tag}", (arr,), {}))
    cases += _unary_foreign_cases()
    return cases


def _unary_u_only_cases():
    # isdecimal/isnumeric: real numpy has no `S`-dtype loop at all (raises
    # `_UFuncNoLoopError`) -- see `strings_isdecimal`/`strings_isnumeric` in
    # ionp-py/src/strings.rs. `S` cases are included here deliberately, to
    # exercise that this anionpy raises the identical real exception class.
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/full", (arr,), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/S_raises", (arr,), {}))
    for tag, arr in _shape_variants_u():
        cases.append((f"shape_{tag}", (arr,), {}))
    cases += _unary_foreign_cases()
    return cases


def _add_cases():
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/self", (arr, arr), {}))
        cases.append((f"{tag}/reversed", (arr, arr[::-1]), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/self", (arr, arr), {}))

    # broadcasting: column x row
    a_col = np.array(["x", "yy", "zzz"], dtype="<U5").reshape(3, 1)
    b_row = np.array(["1", "22"], dtype="<U5").reshape(1, 2)
    cases.append(("broadcast_col_row", (a_col, b_row), {}))

    # 0-d operands (exercises the "0-d array behaves like a scalar for
    # output-width purposes" rule -- see `Loaded`'s itemsize_chars override
    # in ionp-py/src/strings.rs)
    z = np.array("hello world", dtype="<U20")
    cases.append(("0d_self", (z, z), {}))
    zs = np.array(b"hello", dtype="S20")
    cases.append(("0d_self_S", (zs, zs), {}))

    # empty array
    empty_u = np.array([], dtype="<U5")
    cases.append(("empty_self", (empty_u, empty_u), {}))

    # foreign scalar operands: plain str/bytes and numpy scalar types,
    # mixed with a real array (matches ufunc_corpus.py's
    # `string_scalar_operands()` call-form philosophy for this block)
    arr = np.array(["ab", "cd"], dtype="<U5")
    cases.append(("arr_plus_str", (arr, "Z"), {}))
    cases.append(("str_plus_arr", ("Z", arr), {}))
    cases.append(("arr_plus_npstr", (arr, np.str_("q9")), {}))
    arr_s = np.array([b"ab", b"cd"], dtype="S5")
    cases.append(("arrS_plus_bytes", (arr_s, b"Z"), {}))
    cases.append(("bytes_plus_arrS", (b"Z", arr_s), {}))

    # dtype-mismatch: must raise (real numpy's
    # `_UFuncBinaryResolutionError`)
    cases.append(("mismatch_S_plus_str", (arr_s, "suffix"), {}))
    cases.append(("mismatch_S_plus_npstr", (arr_s, np.str_("nsfx")), {}))
    cases.append(("mismatch_U_plus_bytes", (arr, b"suffix"), {}))

    cases += _add_foreign_cases()
    return cases


def _unary_foreign_cases():
    """Class C (foreign/numeric dtype -- must raise, via anionpy's OWN
    reproduction of numpy's not-a-loop message, never numpy's own
    exception object) and class D (Python list / bare str/bytes scalar --
    must be accepted, via implicit `numpy.asarray()` conversion) cases,
    shared by every unary predicate (`_unary_cases`) and by the two
    `U`-only predicates (`_unary_u_only_cases`, where foreign-numeric-dtype
    behaves identically to the `U`-only case -- both raise -- so this list
    is reused there unchanged; the `S`-only-raises case those two already
    special-case via `_unary_u_only_cases`'s own `_s_arrays()` loop)."""
    cases = []
    for dtname in ("int64", "float32", "int32", "uint8", "bool_", "complex128"):
        arr = np.array([1, 2, 3], dtype=getattr(np, dtname))
        cases.append((f"foreign_{dtname}_raises", (arr,), {}))
    cases.append(("list_of_str", (["ab", "CD", ""],), {}))
    cases.append(("list_of_bytes", ([b"ab", b"CD", b""],), {}))
    cases.append(("bare_str_scalar", ("hello",), {}))
    cases.append(("bare_bytes_scalar", (b"hello",), {}))
    return cases


def _add_foreign_cases():
    """Class C: foreign numeric operands must genuinely COMPUTE (anionpy's own
    `ionp_core::ufunc::binary_op`, not numpy's `add` standing in for it --
    the case this whole re-declaration round exists to actually exercise,
    not just claim). Class D: list operands must be accepted."""
    cases = []
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([10, 20, 30], dtype=np.int64)
    cases.append(("foreign_int64_computes", (a, b), {}))
    af = np.array([1.5, 2.5], dtype=np.float32)
    bf = np.array([0.5, 0.25], dtype=np.float32)
    cases.append(("foreign_float32_computes", (af, bf), {}))
    cases.append(("list_plus_list", (["ab", "cd"], ["X", "Y"]), {}))
    # numeric-vs-string mixed operand: real numpy raises `_UFuncNoLoopError`
    # here (verified: distinct from the both-string `_UFuncBinaryResolution
    # Error` case `_add_cases`'s `mismatch_*` entries already cover).
    cases.append(("mismatch_int_plus_str", (a, np.array(["x", "y", "z"])), {}))
    return cases


def _case_transform_cases():
    """Case-conversion group (`upper`/`lower`/`swapcase`/`title`/
    `capitalize`) -- `_vec_string`-based, NOT real numpy ufuncs (confirmed:
    `type(np.strings.upper) is not numpy.ufunc`), so foreign-dtype input
    raises a plain `TypeError` ("string operation on non-string array") and
    a 0-d input is NOT scalarized (stays a 0-d `ndarray`, itemsize NOT
    shrunk to content -- distinct from `strip`'s group below; see
    `ionp-py/src/strings.rs`'s `encode_string_array_noscalar` docstring).
    Reuses `_unary_cases`' shape/dtype/foreign coverage directly since the
    input-side rules (dtype kinds, shapes, foreign/list/scalar handling) are
    the same corpus, just exercised through a transform instead of a
    predicate."""
    return _unary_cases()


def _strip_cases():
    """`strip`/`lstrip`/`rstrip` -- real ufuncs in numpy, unlike the
    case-conversion group: 0-d input DOES shrink itemsize to the actual
    post-strip content length and DOES scalarize (mirrors `add`'s existing
    0-d handling), confirmed directly against real numpy 2.5.1. Adds a
    handful of `chars`-argument cases (numpy accepts a scalar `str`/`bytes`
    of chars to strip, not just whitespace) on top of the shared
    `_unary_cases()` no-`chars` coverage."""
    cases = list(_unary_cases())
    cases.append(("chars_arg_u", (np.array(["xxhixx", "yyayy", "zz"], dtype="<U10"), "xy"), {}))
    cases.append(("chars_arg_s", (np.array([b"xxhixx", b"yyayy", b"zz"], dtype="S10"), b"xy"), {}))
    cases.append(("chars_arg_none_content", (np.array(["  hi  ", "no_strip_chars_here"], dtype="<U10"), None), {}))
    return cases


def _comparison_foreign_cases():
    """Same rationale as `_add_foreign_cases`, for the six comparison ops."""
    cases = []
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([1, 5, 2], dtype=np.int64)
    cases.append(("foreign_int64_computes", (a, b), {}))
    cases.append(("list_vs_list", (["ab", "cd"], ["ab", "ce"]), {}))
    cases.append(("mismatch_int_vs_str", (a, np.array(["x", "y", "z"])), {}))
    return cases


def _comparison_cases():
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/self_vs_reversed", (arr, arr[::-1]), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/self_vs_reversed", (arr, arr[::-1]), {}))

    a_col = np.array(["x", "yy", "zzz"], dtype="<U5").reshape(3, 1)
    b_row = np.array(["1", "22"], dtype="<U5").reshape(1, 2)
    cases.append(("broadcast_col_row", (a_col, b_row), {}))

    z = np.array("hello world", dtype="<U20")
    cases.append(("0d_self", (z, z), {}))

    arr = np.array(["ab", "cd"], dtype="<U5")
    cases.append(("arr_vs_str", (arr, "cd"), {}))
    cases.append(("str_vs_arr", ("cd", arr), {}))
    cases.append(("arr_vs_npstr", (arr, np.str_("ab")), {}))

    arr_s = np.array([b"ab", b"cd"], dtype="S5")
    cases.append(("mismatch_S_vs_str", (arr_s, "suffix"), {}))
    cases.append(("mismatch_S_vs_npstr", (arr_s, np.str_("nsfx")), {}))

    cases += _comparison_foreign_cases()
    return cases


# ---------------------------------------------------------------------------
# Registry override
# ---------------------------------------------------------------------------

_UNARY_NAMES = ["isalpha", "isalnum", "isdigit", "isspace", "islower", "isupper", "istitle"]
_UNARY_U_ONLY_NAMES = ["isdecimal", "isnumeric"]
_COMPARISON_NAMES = ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]
_CASE_TRANSFORM_NAMES = ["upper", "lower", "swapcase", "title", "capitalize"]
_STRIP_NAMES = ["strip", "lstrip", "rstrip"]


def _numpy_exc_type(fn, *args):
    """Captures real numpy's own exception CLASS for a known-raising call,
    without hardcoding its (private, `numpy._core._exceptions`-internal,
    not-guaranteed-stable-across-versions) import path. Used only to build
    `exception_equivalences` dict keys below -- declaring "anionpy's plain
    `TypeError` is an accepted stand-in for numpy's own private `TypeError`
    subclass here", never to construct an instance of numpy's exception (see
    this file's module docstring, and `numpy_no_loop_type_error`/
    `ionp_binary_dispatch` in ionp-py/src/strings.rs, for why anionpy never
    imports numpy to produce that instance itself)."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001 - probing numpy's own exception type, not app code
        return type(exc)
    raise AssertionError(f"{fn} did not raise on {args!r} while probing numpy's exception type")


# Both verified directly against real numpy 2.5.1 (see this file's module
# docstring): `_UFuncNoLoopError` for every unary predicate on foreign/S-vs-
# U-mismatched input, PLUS `add`/comparisons on foreign-numeric-vs-string
# mixed operands; `_UFuncBinaryResolutionError` specifically for `add` on
# two operands that are BOTH string dtype but mismatched S/U kinds. Both are
# private subclasses of the public `TypeError` (`.__mro__` confirmed).
_UFUNC_NO_LOOP_ERROR = _numpy_exc_type(np.strings.isalpha, np.array([1, 2, 3]))
_UFUNC_BINARY_RESOLUTION_ERROR = _numpy_exc_type(
    np.strings.add, np.array([b"a"]), np.array(["b"])
)

_UNARY_EXC_EQUIV = {_UFUNC_NO_LOOP_ERROR: {TypeError}}
_ADD_EXC_EQUIV = {_UFUNC_NO_LOOP_ERROR: {TypeError}, _UFUNC_BINARY_RESOLUTION_ERROR: {TypeError}}
_COMPARISON_EXC_EQUIV = {_UFUNC_NO_LOOP_ERROR: {TypeError}}

_OVERRIDES: dict[str, registry.ItemSpec] = {}
for _prefix in ("char", "strings"):
    _OVERRIDES[f"{_prefix}.add"] = ItemSpec(
        name=f"{_prefix}.add", kind="custom", atol=0.0, rtol=0.0,
        convert_ionp_args=False, custom_cases=_add_cases,
        exception_equivalences=_ADD_EXC_EQUIV,
    )
    _OVERRIDES[f"{_prefix}.str_len"] = ItemSpec(
        name=f"{_prefix}.str_len", kind="custom", atol=0.0, rtol=0.0,
        convert_ionp_args=False, custom_cases=_unary_cases,
        exception_equivalences=_UNARY_EXC_EQUIV,
    )
    for _n in _UNARY_NAMES:
        _OVERRIDES[f"{_prefix}.{_n}"] = ItemSpec(
            name=f"{_prefix}.{_n}", kind="custom", atol=0.0, rtol=0.0,
            convert_ionp_args=False, custom_cases=_unary_cases,
            exception_equivalences=_UNARY_EXC_EQUIV,
        )
    for _n in _UNARY_U_ONLY_NAMES:
        _OVERRIDES[f"{_prefix}.{_n}"] = ItemSpec(
            name=f"{_prefix}.{_n}", kind="custom", atol=0.0, rtol=0.0,
            convert_ionp_args=False, custom_cases=_unary_u_only_cases,
            exception_equivalences=_UNARY_EXC_EQUIV,
        )

for _n in _COMPARISON_NAMES:
    _OVERRIDES[f"strings.{_n}"] = ItemSpec(
        name=f"strings.{_n}", kind="custom", atol=0.0, rtol=0.0,
        convert_ionp_args=False, custom_cases=_comparison_cases,
        exception_equivalences=_COMPARISON_EXC_EQUIV,
    )

# `char` has no comparison functions in real numpy (only `strings` does --
# `np.char.equal`/etc do not exist); confirmed empirically and by the 28-name
# corpus list itself (no `char.equal` among them). Nothing to override there.

_missing = set(_OVERRIDES) - set(registry.REGISTRY)
if _missing:
    raise AssertionError(
        f"strings_cases.py: {sorted(_missing)} expected to already be present "
        f"in registry.REGISTRY (as kind='ufunc' auto-derived entries from "
        f"ufunc_registry.py) before this override -- if this fires, either "
        f"ufunc_introspect.py's UFUNC_NAMES stopped including these, or this "
        f"module is being imported before ufunc_registry.py (see this file's "
        f"module docstring for the required import order)."
    )
assert len(_OVERRIDES) == 28, f"expected exactly 28 char/strings overrides, got {len(_OVERRIDES)}"

# ---------------------------------------------------------------------------
# NEW entries (not overrides): case-conversion + strip family
# ---------------------------------------------------------------------------
#
# Unlike the 28 names above, `numpy.char.upper`/`numpy.strings.upper`/...
# and `numpy.char.strip`/`numpy.strings.strip`/... are NOT `numpy.ufunc`
# instances -- confirmed both via `type(np.strings.upper) is not
# numpy.ufunc` and via `tools/numpy_surface.json` itself, which classifies
# every one of these 16 names as `kind == "func"`, not `"ufunc"`. So
# `ufunc_registry.py` never auto-derives a (broken) `kind="ufunc"` entry for
# them the way it does for `add`/the predicates/the comparisons -- there is
# nothing in `registry.REGISTRY` to override. These are genuinely NEW
# registry additions, asserted below to confirm that understanding (if one
# of these names unexpectedly already has an entry, something upstream
# changed and silently colliding with it would be wrong).
_NEW_ITEMS: dict[str, registry.ItemSpec] = {}
for _prefix in ("char", "strings"):
    for _n in _CASE_TRANSFORM_NAMES:
        _NEW_ITEMS[f"{_prefix}.{_n}"] = ItemSpec(
            name=f"{_prefix}.{_n}", kind="custom", atol=0.0, rtol=0.0,
            convert_ionp_args=False, custom_cases=_case_transform_cases,
            exception_equivalences=_UNARY_EXC_EQUIV,
        )
    for _n in _STRIP_NAMES:
        _NEW_ITEMS[f"{_prefix}.{_n}"] = ItemSpec(
            name=f"{_prefix}.{_n}", kind="custom", atol=0.0, rtol=0.0,
            convert_ionp_args=False, custom_cases=_strip_cases,
            exception_equivalences=_UNARY_EXC_EQUIV,
        )

_unexpected_collision = set(_NEW_ITEMS) & set(registry.REGISTRY)
if _unexpected_collision:
    raise AssertionError(
        f"strings_cases.py: {sorted(_unexpected_collision)} already present "
        f"in registry.REGISTRY -- expected these 16 case-transform/strip "
        f"names to be genuinely new (kind='func' in numpy_surface.json, not "
        f"auto-derived as kind='ufunc'). If this fires, something upstream "
        f"started auto-deriving them and this block should switch to "
        f"overriding like the 28 names above instead of adding fresh."
    )
assert len(_NEW_ITEMS) == 16, f"expected exactly 16 new case-transform/strip items, got {len(_NEW_ITEMS)}"

registry.REGISTRY.update(_OVERRIDES)
registry.REGISTRY.update(_NEW_ITEMS)

# ---------------------------------------------------------------------------
# NEW entries, round 3: pad family (center/ljust/rjust/zfill) + search
# family (count/find/rfind/index/rindex/startswith/endswith) -- 22 items
# (11 names x 2 prefixes). Neither family is a real `numpy.ufunc` (confirmed
# via `type(np.strings.center) is not numpy.ufunc`, same as the case-
# transform/strip round above), so these are genuinely NEW registry
# entries, not overrides.
#
# PAD FAMILY -- a THIRD, previously undocumented output-shape rule
# (distinct from both case-transform and strip above), confirmed directly
# against real numpy 2.5.1: 0-d input does NOT scalarize (like
# case-transform), but output itemsize is `max(width, actual post-op
# content length)` recomputed fresh from the RESULT (unlike either prior
# group, which key off the INPUT's itemsize one way or another). A genuine
# numpy-2.5.1-internal bug is reproduced byte-exact rather than avoided: a
# zero-size array raises `ValueError('zero-size array to reduction
# operation maximum which has no identity')` (an internal `width.max()`
# reduction called before checking array size) -- confirmed identical for
# all four names. `center`'s left/right padding bias was cross-checked
# against real numpy AND real CPython `str.center` during this pass (a
# pre-existing `ionp-core::strings::StrElem::center` had the bias
# backwards -- extra pad char to the right instead of the left on an odd
# total; fixed here, see that file's updated doc comment).
#
# Two input classes are DELIBERATELY EXCLUDED from this family's corpus,
# each a disclosed, non-silent gap (see `ionp-py/src/strings.rs`'s
# `decode_fillchar`/`pad_fn_with_fillchar!` doc comments for why): a
# foreign-dtype `a`, and a foreign-dtype `fillchar` -- real numpy's actual
# behavior on either is an internal quirk
# (`ValueError("invalid literal for int() with base 10: ...")` for some
# inputs) that is not a clean, deterministic shape worth chasing this pass;
# anionpy raises its own plain `TypeError` for both instead.
#
# SEARCH FAMILY -- a FOURTH output-shape personality: int64 output
# (count/find/rfind/index/rindex) or bool output (startswith/endswith), 0-d
# input DOES scalarize (like strip, unlike case-transform/pad). `sub`/
# `prefix`/`suffix` and `start`/`end` are restricted to scalar operands (the
# same restriction already accepted for `strip`'s `chars` argument).
# `index`/`rindex` raise numpy's own real `ValueError('substring not
# found')` verbatim on any not-found element (confirmed, and this is
# already a genuine public numpy class, not a private one -- no
# `exception_equivalences` entry needed for that specific error).
#
# Foreign-dtype-input error message: confirmed a message shape distinct
# from every one of the three already established for `add`/comparisons --
# a 4-element dtype-class tuple, always ending
# `_PyLongDType, _PyLongDType` (the underlying gufunc's `start`/`end`
# positions). Covered here for: foreign `a` + string `sub` (`Int64DType`
# class observed), and string `a` + mismatched-kind string `sub` (S-vs-U).
# DELIBERATELY EXCLUDED (disclosed gap, not chased this pass): a bare
# non-string Python scalar (e.g. plain `5`) passed as `sub` -- numpy's
# NEP-50 "weak scalar" dispatch reports THAT specific case as
# `_PyLongDType` rather than the `Int64DType` `numpy.asarray(5).dtype`
# alone would give (confirmed via direct probe:
# `type(np.asarray(5).dtype).__name__ == 'Int64DType'`, but
# `np.strings.find(a, 5)`'s actual error text names `_PyLongDType`) --
# reproducing that exactly would require modeling numpy's weak-scalar type
# promotion in the loader, disproportionate effort for this pass. Both
# operands foreign is likewise excluded (undocumented combination).
# ---------------------------------------------------------------------------

_PAD_FILLCHAR_NAMES = ["center", "ljust", "rjust"]
_SEARCH_I64_NAMES = ["find", "rfind", "count"]
_SEARCH_BOOL_NAMES = ["startswith", "endswith"]
_SEARCH_INDEX_NAMES = ["index", "rindex"]


def _pad_fillchar_cases():
    """`center`/`ljust`/`rjust` -- share an identical (a, width, fillchar)
    call signature, so one builder serves all three (each function's own
    per-element semantics differ, but the differential harness calls
    `numpy_path`/`ionp_path` with the SAME args on both sides, so sharing
    the args here is exactly right -- it does not imply the outputs must
    match across names, only that the plumbing does)."""
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/w1_default_fill", (arr, 1), {}))
        cases.append((f"{tag}/w30_default_fill", (arr, 30), {}))
        cases.append((f"{tag}/w12_custom_fill", (arr, 12, "*"), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/w1_default_fill", (arr, 1), {}))
        cases.append((f"{tag}/w30_default_fill", (arr, 30), {}))
        cases.append((f"{tag}/w12_custom_fill", (arr, 12, b"*"), {}))
    for tag, arr in _shape_variants_u():
        if tag == "empty":
            continue  # zero-size-array bug case, covered separately below
        cases.append((f"shape_{tag}/w20", (arr, 20), {}))

    # 0-d specifically (does NOT scalarize -- distinct from strip/search)
    z = np.array("hi", dtype="<U20")
    cases.append(("0d_wide_fill", (z, 8, "-"), {}))
    cases.append(("0d_narrow", (z, 1), {}))

    # zero-size array: real numpy-2.5.1-internal bug, reproduced byte-exact
    empty_u = np.array([], dtype="<U5")
    cases.append(("zero_size_raises", (empty_u, 3), {}))
    empty_s = np.array([], dtype="S5")
    cases.append(("zero_size_raises_S", (empty_s, 3), {}))

    # negative width (clamped to 0 -- a no-op pad)
    cases.append(("negative_width", (np.array(["hi", "hello"]), -5), {}))

    # width narrower than content (no-op, output width == max content len)
    cases.append(("width_narrower_than_content", (np.array(["hi", "hello"]), 2), {}))

    # wrong fillchar length -- must raise TypeError (both empty and multi-char)
    a = np.array(["hi", "hello"])
    cases.append(("fillchar_empty_raises", (a, 5, ""), {}))
    cases.append(("fillchar_multichar_raises", (a, 5, "ab"), {}))

    return cases


def _zfill_cases():
    """`zfill` -- no `fillchar` argument, but shares every other pad-family
    rule (0-d non-scalarizing, content-derived output width, the same
    zero-size-array bug)."""
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/w1", (arr, 1), {}))
        cases.append((f"{tag}/w30", (arr, 30), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/w1", (arr, 1), {}))
        cases.append((f"{tag}/w30", (arr, 30), {}))
    for tag, arr in _shape_variants_u():
        if tag == "empty":
            continue
        cases.append((f"shape_{tag}/w20", (arr, 20), {}))

    z = np.array("42", dtype="<U20")
    cases.append(("0d", (z, 8), {}))

    # signed content -- exercises the "pad after the sign, not before" rule
    signed = np.array(["-3", "+12", "-", "no_sign_here"], dtype="<U20")
    cases.append(("signed_content", (signed, 8), {}))
    signed_s = np.array([b"-3", b"+12"], dtype="S20")
    cases.append(("signed_content_S", (signed_s, 8), {}))

    empty_u = np.array([], dtype="<U5")
    cases.append(("zero_size_raises", (empty_u, 3), {}))

    cases.append(("negative_width", (np.array(["hi", "hello"]), -5), {}))
    return cases


def _search_common_cases(bool_output=False):
    """Shared corpus for the (a, sub, start, end) call signature -- used by
    `find`/`rfind`/`count` (int64 output) and `startswith`/`endswith` (bool
    output) alike; the args are identical, only the expected VALUE dtype
    differs, which the differential harness derives from the real numpy
    call, not from anything declared here."""
    cases = []
    needle_u = "l" if not bool_output else "h"
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/default_start_end", (arr, "a"), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/default_start_end", (arr, b"a"), {}))
    for tag, arr in _shape_variants_u():
        cases.append((f"shape_{tag}", (arr, "a"), {}))

    a = np.array(["hello", "world", "", "aabbaabb"], dtype="<U20")
    cases.append(("start_positive", (a, "a", 2), {}))
    cases.append(("start_end_positive", (a, "a", 1, 5), {}))
    cases.append(("start_negative", (a, "a", -4), {}))
    cases.append(("end_negative", (a, "a", 0, -1), {}))
    cases.append(("start_end_negative", (a, "a", -6, -1), {}))
    cases.append(("end_none_explicit", (a, "a", 0, None), {}))
    cases.append(("start_out_of_range", (a, "a", 100), {}))
    cases.append(("end_out_of_range", (a, "a", 0, 100), {}))
    cases.append(("start_gt_end", (a, "a", 5, 1), {}))
    cases.append(("empty_sub", (a, ""), {}))
    cases.append(("sub_longer_than_content", (a, "aabbaabbaabb"), {}))

    # 0-d specifically (int/bool output DOES scalarize here, unlike pad)
    z = np.array("hello world", dtype="<U20")
    cases.append(("0d", (z, "o"), {}))

    # bytes/S content, needle present and absent
    sarr = np.array([b"hello", b"world", b""], dtype="S20")
    cases.append(("S_present", (sarr, b"l"), {}))
    cases.append(("S_absent", (sarr, b"z"), {}))

    # foreign `a` (Int64DType) + string `sub` -- must raise the 4-tuple
    # not-a-loop message
    fint = np.array([1, 2, 3], dtype=np.int64)
    cases.append(("foreign_a_raises", (fint, "a"), {}))

    # S-vs-U kind-mismatched `sub` -- must raise the same message shape,
    # with both operand orders
    cases.append(("kind_mismatch_U_array_S_sub", (a, np.array([b"a"])[0]), {}))
    cases.append(("kind_mismatch_S_array_U_sub", (sarr, np.array(["l"])[0]), {}))

    # REGRESSION CORPUS (Monday, 2026-08-01): `start > len(element)` boundary
    # rows. Withdrawn-then-fixed bug: when `start` exceeds an element's own
    # length, even an EMPTY substring must NOT be found -- anionpy used to clamp
    # `start` down to `len` and report a hit there. The original round-3
    # corpus never combined a short/empty element with a `start` past its own
    # end (every prior boundary case used a fixed-length array with `start`
    # values chosen relative to the WHOLE array's typical length, not per-
    # element), which is exactly why 30/30 passed while this diverged. See
    # `anionpy/_state/char_strings.py` round-3/round-4 notes for the full
    # CPython `ADJUST_INDICES` derivation this corpus is checked against.
    boundary_u = np.array(["", "a", "ab"], dtype="<U5")
    for _st in range(4):
        cases.append((f"boundary_start{_st}_emptysub_default_end", (boundary_u, "", _st), {}))
        cases.append((f"boundary_start{_st}_emptysub_end4", (boundary_u, "", _st, 4), {}))
        cases.append((f"boundary_start{_st}_asub_default_end", (boundary_u, "a", _st), {}))
        cases.append((f"boundary_start{_st}_asub_end4", (boundary_u, "a", _st, 4), {}))
    boundary_s = np.array([b"", b"a", b"ab"], dtype="S5")
    for _st in range(4):
        cases.append((f"boundary_S_start{_st}_emptysub_default_end", (boundary_s, b"", _st), {}))
        cases.append((f"boundary_S_start{_st}_emptysub_end4", (boundary_s, b"", _st, 4), {}))
        cases.append((f"boundary_S_start{_st}_asub_default_end", (boundary_s, b"a", _st), {}))
        cases.append((f"boundary_S_start{_st}_asub_end4", (boundary_s, b"a", _st, 4), {}))

    return cases


def _search_i64_cases():
    return _search_common_cases(bool_output=False)


def _search_bool_cases():
    return _search_common_cases(bool_output=True)


def _index_cases():
    """`index`/`rindex` -- identical corpus to `find`/`rfind` for the
    "found" cases (same underlying method, just a different not-found
    policy), plus dedicated not-found cases exercising the real, PUBLIC
    `ValueError('substring not found')` (not a private-subclass case --
    no `exception_equivalences` entry needed for this specific error)."""
    cases = [c for c in _search_common_cases(bool_output=False) if "foreign" not in c[0] and "mismatch" not in c[0]]
    a = np.array(["hello", "world", "abc"], dtype="<U20")
    cases.append(("notfound_raises", (a, "zzz"), {}))
    cases.append(("notfound_one_of_several_raises", (a, "z"), {}))
    sarr = np.array([b"hello", b"world"], dtype="S20")
    cases.append(("notfound_raises_S", (sarr, b"zzz"), {}))
    return cases


_PAD_SEARCH_EXC_EQUIV = {_UFUNC_NO_LOOP_ERROR: {TypeError}}

_NEW_ITEMS_2: dict[str, registry.ItemSpec] = {}
for _prefix in ("char", "strings"):
    for _n in _PAD_FILLCHAR_NAMES:
        _NEW_ITEMS_2[f"{_prefix}.{_n}"] = ItemSpec(
            name=f"{_prefix}.{_n}", kind="custom", atol=0.0, rtol=0.0,
            convert_ionp_args=False, custom_cases=_pad_fillchar_cases,
        )
    _NEW_ITEMS_2[f"{_prefix}.zfill"] = ItemSpec(
        name=f"{_prefix}.zfill", kind="custom", atol=0.0, rtol=0.0,
        convert_ionp_args=False, custom_cases=_zfill_cases,
    )
    for _n in _SEARCH_I64_NAMES:
        _NEW_ITEMS_2[f"{_prefix}.{_n}"] = ItemSpec(
            name=f"{_prefix}.{_n}", kind="custom", atol=0.0, rtol=0.0,
            convert_ionp_args=False, custom_cases=_search_i64_cases,
            exception_equivalences=_PAD_SEARCH_EXC_EQUIV,
        )
    for _n in _SEARCH_BOOL_NAMES:
        _NEW_ITEMS_2[f"{_prefix}.{_n}"] = ItemSpec(
            name=f"{_prefix}.{_n}", kind="custom", atol=0.0, rtol=0.0,
            convert_ionp_args=False, custom_cases=_search_bool_cases,
            exception_equivalences=_PAD_SEARCH_EXC_EQUIV,
        )
    for _n in _SEARCH_INDEX_NAMES:
        _NEW_ITEMS_2[f"{_prefix}.{_n}"] = ItemSpec(
            name=f"{_prefix}.{_n}", kind="custom", atol=0.0, rtol=0.0,
            convert_ionp_args=False, custom_cases=_index_cases,
            exception_equivalences=_PAD_SEARCH_EXC_EQUIV,
        )

_unexpected_collision_2 = set(_NEW_ITEMS_2) & set(registry.REGISTRY)
if _unexpected_collision_2:
    raise AssertionError(
        f"strings_cases.py: {sorted(_unexpected_collision_2)} already present "
        f"in registry.REGISTRY -- expected these 22 pad/search names to be "
        f"genuinely new (kind='func' in numpy_surface.json, not auto-derived "
        f"as kind='ufunc'). If this fires, something upstream started "
        f"auto-deriving them and this block should switch to overriding "
        f"like the 28-name round above instead of adding fresh."
    )
assert len(_NEW_ITEMS_2) == 22, f"expected exactly 22 new pad/search items, got {len(_NEW_ITEMS_2)}"

registry.REGISTRY.update(_NEW_ITEMS_2)

# ---------------------------------------------------------------------------
# NEW entries, round 4 (Monday, 2026-08-01): replace, multiply (both
# `char`/`strings` -- genuinely different foreign-error paths), partition/
# rpartition (`strings` only -- `char.partition`/`.rpartition` are a
# different, non-identical function returning a single trailing-size-3-dim
# array instead of a 3-tuple; NOT implemented this pass, disclosed gap, no
# entry added), encode/decode (both `char`/`strings` -- confirmed literally
# the same bound Rust function object between the two modules, same as
# `replace`), and the six char-only legacy comparisons plus
# `compare_chararrays` -- 17 items total (2+2+1+1+2+2+6+1). None of these
# are real `numpy.ufunc` instances (confirmed via `type(np.strings.replace)
# is not numpy.ufunc`, same check as every `kind="func"` round above), so
# these are genuinely NEW registry entries, not overrides.
#
# EXCLUDED FROM THIS CORPUS (disclosed, undeclared gaps -- see
# `ionp-py/src/strings.rs`'s doc comments on the relevant functions for
# each):
#   - `old`/`new`/`sep` as a foreign-dtype or wrong-string-kind argument:
#     anionpy raises its own placeholder `TypeError`, not numpy's real
#     internal quirk message.
#   - `encode`ing an S-dtype array / `decode`ing a U-dtype array, and
#     foreign-dtype `a` to either: anionpy raises its own placeholder
#     `TypeError`; real numpy raises `AttributeError`/a different
#     `TypeError` text (confirmed via direct probe this session: `"type
#     object 'bytes' has no attribute 'encode'"` / `"'str' has no
#     attribute 'decode'"` for the wrong-kind case).
#   - any encoding name outside {utf-8, ascii, latin-1} except to confirm
#     the `LookupError` text for a genuinely-unrecognized name (`'bogus'`);
#     a registered-but-non-text codec name (e.g. `'rot13'`) gets a
#     DIFFERENT real numpy message anionpy does not reproduce.
#   - `errors=` parameter to encode/decode beyond the implicit default.
#   - utf-8 decode's "invalid continuation byte" sub-case (distinct from
#     "invalid start byte" and "unexpected end of data", both covered).
#   - the S-vs-U kind-mismatch `NotImplemented`-return case for the six
#     char comparisons / `compare_chararrays`: confirmed correct via an
#     independent ad-hoc smoke test this session (real numpy returns the
#     literal `NotImplemented` singleton, symmetric both operand orders),
#     but NOT added to this file's corpus -- `harness.compare_values` has
#     no path for a bare `NotImplemented` return (it expects `.dtype`/
#     `.shape`-bearing array-likes), so exercising it here would need a
#     dedicated adapter this round doesn't add. Documented, not silently
#     dropped.
# ---------------------------------------------------------------------------


def _replace_cases():
    """`replace(a, old, new, count=-1)` -- "edit"-family output-shape rule
    (distinct from case-transform/strip/pad/search, confirmed this
    session): 0-d input does NOT scalarize, output width is content-derived
    but FLOORS AT 1 (never 0) -- see `strings_replace`'s doc comment in
    ionp-py/src/strings.rs. Hits the same zero-size-array-on-empty-INPUT
    bug as the pad family (`pad_zero_size_error` reused verbatim there)."""
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/basic", (arr, "a", "Z"), {}))
        cases.append((f"{tag}/count1", (arr, "a", "Z", 1), {}))
        cases.append((f"{tag}/count0", (arr, "a", "Z", 0), {}))
        cases.append((f"{tag}/count_negative", (arr, "a", "Z", -1), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/basic", (arr, b"a", b"Z"), {}))
    for tag, arr in _shape_variants_u():
        if tag == "empty":
            continue  # zero-size-array bug, covered separately below
        cases.append((f"shape_{tag}", (arr, "a", "Z"), {}))

    z = np.array("hello world", dtype="<U20")
    cases.append(("0d", (z, "o", "0"), {}))

    a = np.array(["abc", "xyz", ""], dtype="<U10")
    cases.append(("notfound", (a, "q", "Z"), {}))
    cases.append(("new_longer_than_old", (a, "a", "ZZZZZ"), {}))
    cases.append(("new_shorter_than_old", (np.array(["aaaa", "bbbb"], dtype="<U10"), "aa", "Z"), {}))
    cases.append(("new_empty", (a, "a", ""), {}))
    cases.append(("all_emptied_result", (np.array(["aaa", "aa", "a"], dtype="<U10"), "a", ""), {}))

    empty_u = np.array([], dtype="<U5")
    cases.append(("zero_size_raises", (empty_u, "a", "b"), {}))
    empty_s = np.array([], dtype="S5")
    cases.append(("zero_size_raises_S", (empty_s, b"a", b"b"), {}))

    return cases


def _multiply_cases():
    """Shared (a, n) corpus for BOTH `strings.multiply` and `char.multiply`
    -- same call signature and matching-kind VALUES (confirmed identical
    for real string input this session), differing only in their
    foreign-dtype error path -- see `multiply_impl`'s doc comment in
    ionp-py/src/strings.rs. Registered below with a DIFFERENT
    `exception_equivalences` per prefix accordingly."""
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/scalar_n3", (arr, 3), {}))
        cases.append((f"{tag}/scalar_n0", (arr, 0), {}))
        cases.append((f"{tag}/scalar_n_negative", (arr, -2), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/scalar_n2", (arr, 2), {}))

    a = np.array(["ab", "cd", "e", "日本", "", "Z9"], dtype="<U10")
    cases.append(("array_n_matching_shape", (a, [0, 1, 2, 3, 0, 1]), {}))
    cases.append(("array_n_broadcast_scalar_a", (np.array("hi", dtype="<U10"), [0, 1, 2]), {}))
    cases.append(("array_n_broadcast_col_row", (a.reshape(2, 3), np.array([0, 1, 2]).reshape(1, 3)), {}))

    z = np.array("hello", dtype="<U20")
    cases.append(("0d_n3", (z, 3), {}))
    cases.append(("0d_n0", (z, 0), {}))

    empty_u = np.array([], dtype="<U5")
    cases.append(("zero_size_raises", (empty_u, 2), {}))

    fint = np.array([1, 2, 3], dtype=np.int64)
    cases.append(("foreign_raises", (fint, 2), {}))

    return cases


def _partition_common_cases():
    """Shared (a, sep) corpus for both `strings.partition` and
    `strings.rpartition` -- same call signature; VALUES differ (leftmost vs
    rightmost separator match), graded independently per item since each
    resolves its own `numpy_path`/`ionp_path` and calls both sides with the
    same args. `multi_output=True` below (real numpy returns a genuine
    3-tuple of arrays, confirmed via `type(...) is tuple`, `len(...) ==
    3`) -- same mechanism `setops_cases.py` already uses for
    `intersect1d`/etc."""
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/sep_a", (arr, "a"), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/sep_a", (arr, b"a"), {}))
    for tag, arr in _shape_variants_u():
        if tag == "empty":
            continue  # zero-size-array bug, covered separately below
        cases.append((f"shape_{tag}/sep_dash", (arr, "-"), {}))

    p = np.array(["a=b=c", "noequals", "", "=x", "x=", "==="], dtype="<U20")
    cases.append(("eq_sep", (p, "="), {}))
    cases.append(("multichar_sep", (np.array(["a::b::c", "no_sep", "::"], dtype="<U20"), "::"), {}))
    cases.append(("sep_at_start", (np.array(["=abc"], dtype="<U20"), "="), {}))
    cases.append(("sep_at_end", (np.array(["abc="], dtype="<U20"), "="), {}))
    cases.append(("sep_not_found_all_empty_component", (np.array(["", "abc"], dtype="<U20"), "="), {}))

    z = np.array("a=b", dtype="<U20")
    cases.append(("0d", (z, "="), {}))

    pb = np.array([b"a=b=c", b"noequals", b""], dtype="S20")
    cases.append(("bytes", (pb, b"="), {}))

    empty_u = np.array([], dtype="<U5")
    cases.append(("zero_size_raises", (empty_u, "="), {}))

    cases.append(("empty_sep_raises", (p, ""), {}))

    return cases


def _encode_cases():
    """`encode(a, encoding='utf-8')` -- U-array input only (S-array/foreign
    `a` are documented, disclosed gaps, see this section's header). Does
    NOT hit the zero-size-array bug that `replace`/`multiply`/`partition`/
    `rpartition` share (confirmed: empty input array succeeds normally)."""
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/utf8", (arr, "utf-8"), {}))
        cases.append((f"{tag}/default_encoding", (arr,), {}))
    for tag, arr in _shape_variants_u():
        cases.append((f"shape_{tag}/utf8", (arr, "utf-8"), {}))

    ascii_ok = np.array(["abc", "", "Z9 !"], dtype="<U10")
    cases.append(("ascii_ok", (ascii_ok, "ascii"), {}))
    cases.append(("ascii_fail_single", (np.array(["a日b"], dtype="<U10"), "ascii"), {}))
    cases.append(("ascii_fail_multi_run", (np.array(["a日本b語c"], dtype="<U20"), "ascii"), {}))
    latin1_ok = np.array(["abcé", "café", ""], dtype="<U10")
    cases.append(("latin1_ok", (latin1_ok, "latin-1"), {}))
    cases.append(("latin1_fail", (np.array(["aĀb"], dtype="<U10"), "latin-1"), {}))

    # case-insensitive / alternate spellings of the three supported names
    cases.append(("utf8_nodash", (ascii_ok, "UTF8"), {}))
    cases.append(("utf8_uppercase_dash", (ascii_ok, "UTF-8"), {}))
    cases.append(("ascii_uppercase", (ascii_ok, "ASCII"), {}))
    cases.append(("latin1_alt_spelling", (latin1_ok, "iso-8859-1"), {}))
    cases.append(("latin1_uppercase", (latin1_ok, "LATIN-1"), {}))

    cases.append(("unknown_encoding_raises", (ascii_ok, "bogus"), {}))

    empty_u = np.array([], dtype="<U5")
    cases.append(("empty_array_no_raise", (empty_u, "utf-8"), {}))

    return cases


def _decode_cases():
    """`decode(a, encoding='utf-8')` -- S-array input only, same U/S-kind
    scope note as `_encode_cases`. utf-8 decode error grouping: only
    "unexpected end of data" (truncated valid lead byte) groups a
    multi-byte run; "invalid start byte" is always single-byte-reported.
    ascii decode NEVER groups. utf-8's "invalid continuation byte"
    sub-case is a documented, disclosed gap (see this section's header) --
    deliberately excluded here."""
    # NOTE: deliberately NOT sweeping `_s_arrays()` here (unlike every other
    # width sweep in this file) -- `_CONTENT_S` includes high-byte Latin-1
    # content (e.g. `"\xe9clair".encode("latin-1")`) that is byte-valid
    # Latin-1 but, read as UTF-8, is exactly the excluded "invalid
    # continuation byte" case (0xE9 is a valid 3-byte UTF-8 lead byte,
    # followed by plain ASCII continuation bytes that are not valid UTF-8
    # continuations) -- confirmed this session (this is what first
    # surfaced the gap in this corpus draft). Width sweep below uses
    # ASCII-only content instead, which is valid UTF-8 by construction;
    # genuine multi-byte UTF-8 content is exercised separately below via
    # `utf8_multibyte_ok`.
    cases = []
    for width in _S_WIDTHS:
        ascii_s = np.array([b"", b"a", b"ABC", b"z1", b"MiXeD9"], dtype=f"S{width}")
        cases.append((f"S{width}_ascii/utf8", (ascii_s, "utf-8"), {}))
        cases.append((f"S{width}_ascii/default_encoding", (ascii_s,), {}))

    words_s = np.array([b"ab", b"CD", b"e", b"", b"Z9"], dtype="S10")
    cases.append(("shape_1d", (words_s,), {}))
    cases.append(("shape_2d", (words_s[:4].reshape(2, 2),), {}))
    cases.append(("shape_0d", (np.array(b"hello", dtype="S10"),), {}))
    cases.append(("shape_empty", (np.array([], dtype="S10"),), {}))

    utf8_content = np.array(["woréld".encode("utf-8"), "日本語".encode("utf-8"), b""], dtype="S20")
    cases.append(("utf8_multibyte_ok", (utf8_content, "utf-8"), {}))

    cases.append(("ascii_fail", (np.array([b"\x80"], dtype="S5"), "ascii"), {}))
    cases.append(("ascii_fail_multi_bytes_still_single_report", (np.array([b"ab\x80\x81cd"], dtype="S10"), "ascii"), {}))
    cases.append(("utf8_invalid_start_byte", (np.array([b"\xff"], dtype="S5"), "utf-8"), {}))
    cases.append(("utf8_invalid_start_byte_mid", (np.array([b"ab\xffcd"], dtype="S10"), "utf-8"), {}))
    cases.append(("utf8_truncated_2byte", (np.array([b"\xc2"], dtype="S5"), "utf-8"), {}))
    cases.append(("utf8_truncated_3byte", (np.array([b"\xe2\x82"], dtype="S5"), "utf-8"), {}))
    cases.append(("utf8_truncated_4byte", (np.array([b"\xf0\x9f\x98"], dtype="S5"), "utf-8"), {}))
    cases.append(("latin1_ok", (np.array([b"\xe9", b"\xff", b"abc"], dtype="S10"), "latin-1"), {}))

    cases.append(("utf8_nodash", (utf8_content, "UTF8"), {}))
    cases.append(("unknown_encoding_raises", (words_s, "bogus"), {}))

    empty_s = np.array([], dtype="S5")
    cases.append(("empty_array_no_raise", (empty_s, "utf-8"), {}))

    return cases


def _char_comparison_cases():
    """Shared (a, b) corpus for all six `char.*` legacy comparisons --
    unlike `strings.*`'s comparisons (real ufuncs, `_comparison_cases`
    above), these right-strip trailing whitespace from BOTH operands before
    comparing (confirmed directly this session), so this corpus
    deliberately includes trailing-whitespace-only-differing pairs the
    `strings` corpus doesn't need. The S-vs-U kind-mismatch
    `NotImplemented` case is deliberately excluded -- see this section's
    header."""
    cases = []
    for tag, arr in _u_arrays():
        cases.append((f"{tag}/self_vs_reversed", (arr, arr[::-1]), {}))
    for tag, arr in _s_arrays():
        cases.append((f"{tag}/self_vs_reversed", (arr, arr[::-1]), {}))

    a = np.array(["ab ", "cd", "z  ", "  ", ""], dtype="<U10")
    b = np.array(["ab", "cd ", "zz", "", " "], dtype="<U10")
    cases.append(("rstrip_matters", (a, b), {}))

    z = np.array("hello ", dtype="<U20")
    cases.append(("0d", (z, np.array("hello", dtype="<U20")), {}))

    arr = np.array(["ab", "cd"], dtype="<U5")
    cases.append(("arr_vs_str", (arr, "cd"), {}))
    cases.append(("str_vs_arr", ("cd", arr), {}))

    fint = np.array([1, 2, 3], dtype=np.int64)
    cases.append(("foreign_raises", (arr, fint), {}))

    return cases


def _compare_chararrays_cases():
    """`compare_chararrays(a1, a2, cmp, rstrip)` -- the general form behind
    the six comparisons above; same rstrip semantics, plus its own
    `cmp`-string validation. Kind-mismatch `NotImplemented` case excluded,
    same reason as `_char_comparison_cases`."""
    cases = []
    a = np.array(["ab ", "cd", "z  ", "  ", ""], dtype="<U10")
    b = np.array(["ab", "cd ", "zz", "", " "], dtype="<U10")
    for cmp in ["==", "!=", "<", "<=", ">", ">="]:
        cases.append((f"cmp_{cmp}_rstrip", (a, b, cmp, True), {}))
        cases.append((f"cmp_{cmp}_norstrip", (a, b, cmp, False), {}))
    cases.append(("bad_cmp_raises", (a, b, "~=", True), {}))

    fint = np.array([1, 2, 3], dtype=np.int64)
    arr_u = np.array(["ab", "cd"], dtype="<U5")
    cases.append(("foreign_raises", (arr_u, fint, "==", True), {}))
    return cases


_MULTIPLY_EXC_EQUIV = {_UFUNC_NO_LOOP_ERROR: {TypeError}}

_NEW_ITEMS_3: dict[str, registry.ItemSpec] = {}
for _prefix in ("char", "strings"):
    _NEW_ITEMS_3[f"{_prefix}.replace"] = ItemSpec(
        name=f"{_prefix}.replace", kind="custom", atol=0.0, rtol=0.0,
        convert_ionp_args=False, custom_cases=_replace_cases,
    )
    _NEW_ITEMS_3[f"{_prefix}.encode"] = ItemSpec(
        name=f"{_prefix}.encode", kind="custom", atol=0.0, rtol=0.0,
        convert_ionp_args=False, custom_cases=_encode_cases,
    )
    _NEW_ITEMS_3[f"{_prefix}.decode"] = ItemSpec(
        name=f"{_prefix}.decode", kind="custom", atol=0.0, rtol=0.0,
        convert_ionp_args=False, custom_cases=_decode_cases,
    )

_NEW_ITEMS_3["strings.multiply"] = ItemSpec(
    name="strings.multiply", kind="custom", atol=0.0, rtol=0.0,
    convert_ionp_args=False, custom_cases=_multiply_cases,
    exception_equivalences=_MULTIPLY_EXC_EQUIV,
)
_NEW_ITEMS_3["char.multiply"] = ItemSpec(
    name="char.multiply", kind="custom", atol=0.0, rtol=0.0,
    convert_ionp_args=False, custom_cases=_multiply_cases,
)
_NEW_ITEMS_3["strings.partition"] = ItemSpec(
    name="strings.partition", kind="custom", atol=0.0, rtol=0.0,
    convert_ionp_args=False, custom_cases=_partition_common_cases,
    multi_output=True,
)
_NEW_ITEMS_3["strings.rpartition"] = ItemSpec(
    name="strings.rpartition", kind="custom", atol=0.0, rtol=0.0,
    convert_ionp_args=False, custom_cases=_partition_common_cases,
    multi_output=True,
)
for _n in _COMPARISON_NAMES:
    _NEW_ITEMS_3[f"char.{_n}"] = ItemSpec(
        name=f"char.{_n}", kind="custom", atol=0.0, rtol=0.0,
        convert_ionp_args=False, custom_cases=_char_comparison_cases,
    )
_NEW_ITEMS_3["char.compare_chararrays"] = ItemSpec(
    name="char.compare_chararrays", kind="custom", atol=0.0, rtol=0.0,
    convert_ionp_args=False, custom_cases=_compare_chararrays_cases,
)

_NEW_ITEMS_3["char.strings_multiply"] = ItemSpec(
    name="char.strings_multiply", kind="custom", atol=0.0, rtol=0.0,
    convert_ionp_args=False, custom_cases=_multiply_cases,
    exception_equivalences=_MULTIPLY_EXC_EQUIV,
)
# char.strings_partition / char.strings_rpartition: NOT registered here.
# See anionpy/_state/char_strings.py's round-6 docstring -- an out-of-corpus
# probe found the underlying shared Rust `strings_partition`/
# `strings_rpartition` (already declared exact under `strings.*`) silently
# mismatches real numpy when `sep` is wider than the array's declared
# per-element dtype width (numpy truncates `sep` to match; anionpy does not),
# so this alias is declined, not declared.

_unexpected_collision_3 = set(_NEW_ITEMS_3) & set(registry.REGISTRY)
if _unexpected_collision_3:
    raise AssertionError(
        f"strings_cases.py: {sorted(_unexpected_collision_3)} already "
        f"present in registry.REGISTRY -- expected these 17 round-4 names "
        f"to be genuinely new. If this fires, something upstream started "
        f"auto-deriving them and this block should switch to overriding "
        f"instead of adding fresh."
    )
assert len(_NEW_ITEMS_3) == 18, f"expected exactly 17 round-4 + 1 round-6 item, got {len(_NEW_ITEMS_3)}"

registry.REGISTRY.update(_NEW_ITEMS_3)
