"""anionpy's top-level array-creation + shape-manipulation differential
registry entries (zeros/ones/empty/full [+ _like], arange/linspace/eye/
identity, asarray/copy/ascontiguousarray, and the top-level view wrappers
reshape/ravel/transpose/swapaxes/moveaxis/squeeze/expand_dims/
broadcast_to).

NEW FILE (same pattern as linalg_cases.py/inplace_cases.py/
testing_cases.py: builds a dict of ItemSpecs, merged into registry.REGISTRY
from the bottom of registry.py with a collision check). This is the
differential-spec half of the "top-level array creation + shape
manipulation" task block; ionp-core/src/creation.rs and
ionp-py/src/creation.rs are this same task's other half.

Everything here is kind="custom": every item in this block takes shape
tuples, axis ints, dtype objects, or fill scalars alongside (or instead of)
its array argument, which is exactly what kind="custom" exists for (see
registry.py's ItemSpec docstring) -- none of these are a bare f(array) or
f(array, array) call kind="unary"/"binary" assumes, and only the view
wrappers even have a receiver to hang kind="method" call_forms off of (and
even those are exposed as anionpy.reshape(a, ...) top-level functions, not
ndarray methods, so "ndarray."-prefixed resolution does not apply either).

Two real bugs were found and fixed in ionp-core/src/creation.rs while
building this file (both documented in the task's final report, not
repeated in full here):

  - `broadcast_to` performed NO shape-compatibility validation at all.
    An incompatible target shape produced a garbage view (wrong strides)
    that didn't fail until something later walked it, which crashed the
    whole process with a Rust panic (`repr.rs`, "index out of bounds")
    instead of raising numpy's ValueError. Fixed by validating with
    `shape::broadcast_shapes` before constructing the view. Covered here
    by `broadcast_to_cases()`'s `*_raises` entries.

  - `squeeze`/`expand_dims`/`swapaxes`/`moveaxis` used to raise plain
    `IndexError` for an out-of-bounds axis where numpy raises
    `numpy.exceptions.AxisError` (a genuine subclass of BOTH ValueError and
    IndexError, not a plain IndexError). FIXED at the Rust layer (both
    `ionp-py/src/creation.rs`'s and `ndarray_attrs.rs`'s copies of
    `normalize_axis` now raise `IonpError::AxisError`, routed through the
    same real-`numpy.exceptions.AxisError`-raising machinery
    `to_py_err`/`axis_error()` in `lib.rs` already used for the six
    reduction methods) -- so these four ItemSpecs no longer need
    `exception_equivalences` to paper over the mismatch.

`empty`/`empty_like` get dedicated `numpy_adapter`/`ionp_adapter`s that
canonicalize the (deliberately garbage/uninitialized) contents to zero
before comparison -- per the task's explicit rule, `empty`'s *contents*
are never a valid basis for comparison, only its shape and dtype are. The
adapter still calls the real `empty`/`empty_like` and still compares real
shape/dtype output, so a wrong shape or wrong dtype is still caught; only
the value comparison is neutralized.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec, _reshape_copy_false_reachable
from corpus import unary_corpus

_CORPUS = unary_corpus()


def _corpus_cases(label_suffix: str = "") -> list[tuple[str, tuple, dict]]:
    return [(c.label + label_suffix, (c.value,), {}) for c in _CORPUS]


# ---------------------------------------------------------------------------
# dtype= STRING SPELLING GRAMMAR (docs/DTYPE-SPELLING-GAP-2026-08-06.md).
#
# `dtype_name_to_dtype` (ionp-py/src/lib.rs) used to be a flat 15-arm match
# on CANONICAL names only ("bool", "int8", ... "complex128"); every other
# spelling numpy's own `np.dtype(str)` grammar accepts -- single char codes
# ('f'), char+itemsize codes ('f4'), native byte-order-prefixed codes
# ('<f4'/'=f4'/'|f4'), and C-name aliases ('float32'/'single'/'f4' is NOT
# the same list, 'single' IS) -- fell through to a raised TypeError, even
# though every one of them names a dtype anionpy already has one of its 14
# `DType`s for. Nine items declared `exact` in `_state/toplevel.py`
# (zeros/ones/empty/full/eye/identity/linspace/asarray/zeros_like) were
# FALSE against this gap because their corpora only ever spelled dtypes
# canonically. `_DTYPE_SPELLING_CASES` below is that spec, written BEFORE
# the Rust-side fix (per this task's own verification-first instruction),
# and is reused (not copy-pasted) across every affected item's own
# `*_cases()` below via `_dtype_spelling_kwarg_cases()`.
#
# Deliberately NOT included here (see the gap doc's "Scope note" and "Not
# checked" sections):
#   - `S`/`U`/`V`/`O`/`M`/`m` and `str`/`bytes`/`object`/`void`/
#     `datetime64`/`timedelta64` -- these name dtypes anionpy has NONE of (no
#     string/object/void/datetime dtype exists in the 14-member `DType`
#     enum at all); they are separate, already-tracked absent-dtype items
#     and MUST keep raising. `_DTYPE_SPELLING_RAISES_CASES` below locks
#     that in as a regression guard so this fix cannot quietly widen into
#     covering them.
#   - Big-endian `'>'`-prefixed multi-byte codes (`'>f4'`, `'>i4'`, ...).
#     Measured live (see the fix's own commit message / KNOWN-DIFFERENCES.md
#     for the numpy 2.5.1 measurement): `np.dtype('>f4') != np.dtype('f4')`
#     (different `.byteorder`, different `tobytes()` layout) -- a REAL,
#     distinct dtype anionpy's little-endian-only architecture cannot
#     represent, so real numpy does NOT raise for it while anionpy correctly
#     DOES. That is an intentional, permanent architectural divergence
#     (documented in KNOWN-DIFFERENCES.md), not something to encode as an
#     "exact match" differential case -- doing so would either falsely fail
#     forever (numpy succeeds, anionpy raises) or, if "fixed" by silently
#     aliasing to little-endian, would be a correctness bug strictly worse
#     than the one this task closes. Verified out-of-corpus instead (see
#     the task's own verification script), same as `'>'`-prefixed
#     itemsize-1 codes (`'>b1'`/`'>i1'`/`'>u1'`/`'>?'`/`'>b'`/`'>B'`), which
#     ARE safe (byteorder is genuinely inapplicable at 1 byte, confirmed
#     `np.dtype('>i1') == np.dtype('i1')` live) and so ARE covered below,
#     folded into the `<`/`=`/`|` prefix group since all four behave
#     identically for itemsize-1 dtypes.
_DTYPE_SPELLING_CASES: list[tuple[str, str]] = [
    # single char codes -> (name)
    ("char_bool", "?"), ("char_int8", "b"), ("char_uint8", "B"),
    ("char_int16", "h"), ("char_uint16", "H"), ("char_int32", "i"),
    ("char_uint32", "I"), ("char_int64_l", "l"), ("char_uint64_L", "L"),
    ("char_int64_q", "q"), ("char_uint64_Q", "Q"), ("char_float16", "e"),
    ("char_float32", "f"), ("char_float64", "d"), ("char_complex64", "F"),
    ("char_complex128", "D"),
    # 'g'/'G' (longdouble/clongdouble): on THIS platform (verified live,
    # numpy 2.5.1 / macOS arm64) np.dtype('g').itemsize == 8 == float64's,
    # i.e. there is no true extended-precision type here, so aliasing them
    # to F64/C128 is a genuinely correct answer on this platform -- NOT
    # true on a platform with real 80/128-bit extended precision (x86
    # long double), where this alias would be wrong. Recorded deliberately
    # (see ionp-py/src/lib.rs's dtype_name_to_dtype comment) as a
    # platform-dependent choice, not an oversight.
    ("char_longdouble_g", "g"), ("char_clongdouble_G", "G"),
    # char + itemsize codes
    ("sized_bool", "b1"), ("sized_int8", "i1"), ("sized_int16", "i2"),
    ("sized_int32", "i4"), ("sized_int64", "i8"), ("sized_uint8", "u1"),
    ("sized_uint16", "u2"), ("sized_uint32", "u4"), ("sized_uint64", "u8"),
    ("sized_float16", "f2"), ("sized_float32", "f4"), ("sized_float64", "f8"),
    ("sized_complex64", "c8"), ("sized_complex128", "c16"),
    # native byte-order prefixes ('<' little/native, '=' native, '|' N/A):
    # a representative subset, not the full cross-product -- one shared
    # mechanism (prefix-strip-then-recurse), same reasoning
    # `_reshape_forms` etc already use elsewhere in this file.
    ("prefix_lt_f4", "<f4"), ("prefix_eq_f8", "=f8"), ("prefix_pipe_b1", "|b1"),
    ("prefix_lt_i4", "<i4"), ("prefix_eq_c16", "=c16"), ("prefix_pipe_u1", "|u1"),
    ("prefix_lt_char_d", "<d"), ("prefix_eq_char_i", "=i"),
    # '>' prefix on an itemsize-1 code: genuinely safe (byteorder doesn't
    # apply at 1 byte -- verified `np.dtype('>i1') == np.dtype('i1')`).
    ("prefix_gt_i1", ">i1"), ("prefix_gt_u1", ">u1"), ("prefix_gt_b1", ">b1"),
    ("prefix_gt_char_b", ">b"), ("prefix_gt_char_B", ">B"), ("prefix_gt_char_q", ">?"),
    # C-name aliases
    ("alias_float", "float"), ("alias_double", "double"), ("alias_half", "half"),
    ("alias_single", "single"), ("alias_int", "int"), ("alias_uint", "uint"),
    ("alias_intp", "intp"), ("alias_uintp", "uintp"),
    ("alias_longlong", "longlong"), ("alias_ulonglong", "ulonglong"),
    ("alias_ubyte", "ubyte"), ("alias_byte", "byte"),
    ("alias_short", "short"), ("alias_ushort", "ushort"),
    ("alias_intc", "intc"), ("alias_uintc", "uintc"),
    ("alias_csingle", "csingle"), ("alias_cdouble", "cdouble"),
    ("alias_longdouble", "longdouble"), ("alias_clongdouble", "clongdouble"),
]

# Spellings that must keep raising against anionpy AND against real numpy
# alike -- grammar-adjacent strings numpy itself rejects too (wrong case,
# stray whitespace, an unassigned itemsize), so these ARE legitimate
# "both sides raise TypeError" differential cases (verified live: every
# one of these raises `TypeError: data type '...' not understood` under
# real numpy 2.5.1 as well, not just under anionpy).
#
# `S`/`U`/`V`/`O`/`M`/`m` and `str`/`bytes`/`object`/`void`/`datetime64`/
# `timedelta64` are DELIBERATELY NOT here: real numpy `np.zeros(3,
# dtype=X)` SUCCEEDS for every one of them (verified live, e.g.
# `dtype='object'` -> a real object-dtype array of `0`s), so an ionp-side
# raise for those is a PERMANENT, ARCHITECTURAL divergence (anionpy has no
# string/object/void/datetime `DType` at all), not a "both sides agree"
# case -- encoding it as a plain differential comparison here would be a
# guaranteed-forever failure, exactly the trap `'>f4'` above is. Locked in
# as an ionp-only raise-regression check instead (see this task's
# verification script / KNOWN-DIFFERENCES.md), never as a differential
# corpus entry.
_DTYPE_SPELLING_RAISES_CASES: list[str] = [
    "B1", "H2", "I4", "L8", "Q8", "e2", "g8", "G16", "F8", "D16",
    "i0", "u1 ", "f4 ", " f4", "Float32", "FLOAT32", "INT8", "",
]


def _dtype_spelling_kwarg_cases(shape_args: tuple, base_kwargs: dict | None = None
                                 ) -> list[tuple[str, tuple, dict]]:
    """`(label, shape_args, {**base_kwargs, "dtype": spelling})` for every
    in-scope spelling in `_DTYPE_SPELLING_CASES`, plus one `*_raises` entry
    per out-of-scope spelling in `_DTYPE_SPELLING_RAISES_CASES` -- shared
    by every `*_cases()` below that takes a `dtype=` kwarg, so the grammar
    is specified ONCE and reused, not copy-pasted per item."""
    base_kwargs = base_kwargs or {}
    out = []
    for label, spelling in _DTYPE_SPELLING_CASES:
        out.append((f"dtype_spelling/{label}", shape_args, {**base_kwargs, "dtype": spelling}))
    for spelling in _DTYPE_SPELLING_RAISES_CASES:
        out.append((f"dtype_spelling/raises_{spelling!r}", shape_args,
                     {**base_kwargs, "dtype": spelling}))
    return out


# ---------------------------------------------------------------------------
# zeros / ones / empty / full
# ---------------------------------------------------------------------------

_SHAPES = [(), (0,), (1,), (5,), (2, 3), (3, 1, 4), (0, 3), (2, 0, 3)]
_DTYPES = [None, np.bool_, np.int8, np.int32, np.int64, np.uint8,
           np.float32, np.float64, np.complex64, np.complex128]


def zeros_cases():
    out = []
    for shape in _SHAPES:
        for dt in _DTYPES:
            out.append((f"shape={shape}/dtype={dt}", (shape,), {"dtype": dt}))
    out.append(("order_F", ((2, 3),), {"order": "F"}))
    out.append(("bare_int_shape", (5,), {}))
    out.append(("negative_dim_raises", ((-1, 2),), {}))
    out.append(("bad_shape_type_raises", ("not a shape",), {}))
    # like=: numpy no-ops this for a plain ndarray target (accepts None,
    # and any object carrying __array_function__); non-array-like non-None
    # values raise TypeError (verified directly against numpy 2.5.1).
    out.append(("like_none_explicit", ((3,),), {"like": None}))
    out.append(("like_non_array_function_raises", ((3,),), {"like": 5}))
    out.extend(_dtype_spelling_kwarg_cases(((5,),)))
    return out


def ones_cases():
    out = []
    for shape in _SHAPES:
        for dt in _DTYPES:
            out.append((f"shape={shape}/dtype={dt}", (shape,), {"dtype": dt}))
    out.append(("order_F", ((2, 3),), {"order": "F"}))
    out.append(("bare_int_shape", (5,), {}))
    out.append(("negative_dim_raises", ((-1, 2),), {}))
    out.append(("like_none_explicit", ((3,),), {"like": None}))
    out.append(("like_non_array_function_raises", ((3,),), {"like": "x"}))
    out.extend(_dtype_spelling_kwarg_cases(((5,),)))
    return out


def full_cases():
    out = []
    fills = [0, 1, -7, 3.5, float("nan"), float("inf"), True, 2 + 3j]
    for shape in [(), (1,), (5,), (2, 3), (0, 3)]:
        for fill in fills:
            out.append((f"shape={shape}/fill={fill!r}", (shape, fill), {}))
    out.append(("explicit_dtype_override", ((3,), 1, np.float32), {}))
    out.append(("explicit_dtype_int_from_float_fill", ((3,), 2.9, np.int64), {}))
    out.append(("explicit_dtype_negative_float_truncate", ((3,), -2.9, np.int32), {}))
    out.append(("bare_int_shape", (4, 9), {}))
    out.append(("like_none_explicit", ((3,), 1), {"like": None}))
    out.append(("like_non_array_function_raises", ((3,), 1), {"like": object()}))
    # RECLAIM 2026-08-01: two bugs the coordinator's independent post-merge
    # probe found (undeclared in 790fced) are now fixed in
    # ionp-py/src/lib.rs and locked in here so they can't silently regress.
    #   Bug 1 -- complex fill value with a non-complex target dtype used to
    #   PANIC (Rust unreachable!() -> PanicException across FFI, not
    #   catchable via `except Exception`) instead of following numpy's real
    #   rule: bool target -> (re != 0 or im != 0); any other target ->
    #   discard the imaginary part and cast `re` alone by that target's own
    #   normal float-cast rule.
    for dt in (np.bool_, np.int8, np.uint8, np.int64, np.float32, np.float64):
        out.append((f"complex_fill_non_complex_dtype/dt={dt.__name__}",
                     ((2,), 3 + 4j), {"dtype": dt}))
    out.append(("complex_fill_imag_only_makes_bool_true", ((2,), 0 + 5j), {"dtype": np.bool_}))
    out.append(("complex_fill_zero_makes_bool_false", ((2,), 0 + 0j), {"dtype": np.bool_}))
    #   Bug 2 -- out-of-range/non-finite float fill values previously used
    #   Rust's own saturating `v as i8`/`v as u16` etc (saturates directly
    #   at the narrow target width), which does not match numpy's real
    #   cast: NaN -> 0; 32/64-bit int targets saturate natively at that
    #   width (no change needed there, already matched); 8/16-bit targets
    #   go through a SIGNED 32-bit saturating conversion FIRST (regardless
    #   of the target's own signedness), then truncate the low 8/16 bits of
    #   that i32 as the target type (plain wrapping truncation). Verified
    #   exhaustively against real numpy 2.5.1 -- see lib.rs's
    #   `int_buffer_from_f64` doc comment for the full value x dtype
    #   matrix this was reverse-engineered from.
    for fill in (float("inf"), float("-inf"), float("nan"), 200.0, -200.0, 1e20, -1e20,
                 127.9, -128.9, 128.0, -129.0, 255.5, 256.5, -1.5):
        for dt in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
                   np.int64, np.uint64):
            out.append((f"narrow_int_cast/fill={fill!r}/dt={dt.__name__}",
                         ((2,), fill), {"dtype": dt}))
    # NOT covered: a string fill_value. numpy accepts it (produces a
    # '<U4'-dtype string array); anionpy has no string dtype at all (13
    # numeric/bool dtypes only, see ndarray_from_numpy's supported list in
    # lib.rs), so `full`'s existing, correct TypeError there is a real
    # reflection of an already-known, out-of-scope absent-coverage gap
    # (string dtypes), not a `full`-specific defect -- testing it here
    # would just be re-discovering "anionpy has no strings" under this
    # item's name.
    # CLASS B (array-like fill_value must broadcast, not just repeat
    # element 0): withdrawn in fdcdfb6 because `np.full((2,2), [[2.,1.],
    # [0.,3.]])` broadcast in numpy but silently returned `[[2,2],[2,2]]`
    # in anionpy, and `np.full((2,2), [7.,8.])` broadcast in numpy but raised
    # `TypeError: full()'s fill_value must be a bool/int/float/complex
    # scalar` in anionpy. Fixed in creation.rs via `fill_material` +
    # `broadcast_or_err`; these cases FAIL before that fix and PASS after.
    out.append(("array_fill_exact_shape", ((2, 2), [[2.0, 1.0], [0.0, 3.0]]), {}))
    out.append(("array_fill_row_broadcast", ((2, 2), [7.0, 8.0]), {}))
    out.append(("array_fill_col_broadcast", ((2, 2), [[1.0], [2.0]]), {}))
    out.append(("array_fill_scalar_list_broadcast", ((2, 2), [5]), {}))
    out.append(("array_fill_int_list_broadcast", ((3, 2), [1, 2]), {}))
    out.append(("array_fill_tuple_broadcast", ((2, 2), (7.0, 8.0)), {}))
    out.append(("array_fill_1d_1elem", ((2, 2), [9.0]), {}))
    out.append(("array_fill_bad_shape_1d", ((2, 2), [1, 2, 3]), {}))
    out.append(("array_fill_bad_shape_2d", ((2, 2), [[1, 2, 3], [4, 5, 6]]), {}))
    out.append(("array_fill_order_F", ((2, 3), [1.0, 2.0, 3.0]), {"order": "F"}))
    out.append(("array_fill_dtype_none_infers_int", ((2, 2), [7, 8]), {}))
    out.append(("array_fill_dtype_override", ((2, 2), [7.9, 8.9]), {"dtype": np.int32}))
    out.extend(_dtype_spelling_kwarg_cases(((5,), 1)))
    return out


def _empty_adapter(call):
    def adapter(*args, **kwargs):
        r = call(*args, **kwargs)
        arr = np.asarray(r)
        return np.zeros(arr.shape, dtype=arr.dtype)
    return adapter


def empty_cases():
    out = []
    for shape in _SHAPES:
        for dt in (None, np.int32, np.float64, np.complex128, np.bool_):
            out.append((f"shape={shape}/dtype={dt}", (shape,), {"dtype": dt}))
    out.append(("negative_dim_raises", ((-1,),), {}))
    out.append(("like_none_explicit", ((3,),), {"like": None}))
    out.append(("like_non_array_function_raises", ((3,),), {"like": 5}))
    out.extend(_dtype_spelling_kwarg_cases(((5,),)))
    return out


# ---------------------------------------------------------------------------
# _like family
# ---------------------------------------------------------------------------

def zeros_like_cases():
    out = _corpus_cases()
    for c in _CORPUS[:8]:
        out.append((c.label + "/dtype_override", (c.value,), {"dtype": np.int16}))
        out.append((c.label + "/shape_override", (c.value,), {"shape": (2, 2)}))
    for c in _CORPUS[:5]:
        out.append((c.label + "/subok_false", (c.value,), {"subok": False}))
        out.append((c.label + "/subok_true", (c.value,), {"subok": True}))
    out.extend(_dtype_spelling_kwarg_cases((_CORPUS[0].value,)))
    return out


def ones_like_cases():
    out = _corpus_cases()
    for c in _CORPUS[:8]:
        out.append((c.label + "/dtype_override", (c.value,), {"dtype": np.float32}))
    for c in _CORPUS[:5]:
        out.append((c.label + "/subok_false", (c.value,), {"subok": False}))
    return out


def empty_like_cases():
    out = [(c.label, (c.value,), {}) for c in _CORPUS]
    for c in _CORPUS[:8]:
        out.append((c.label + "/dtype_override", (c.value,), {"dtype": np.int64}))
        out.append((c.label + "/shape_override", (c.value,), {"shape": (3,)}))
    for c in _CORPUS[:5]:
        out.append((c.label + "/subok_false", (c.value,), {"subok": False}))
    return out


def full_like_cases():
    out = []
    fills = [0, 1, -3, 2.5, True]
    for c in _CORPUS[:20]:
        for fill in fills[:2]:
            out.append((f"{c.label}/fill={fill!r}", (c.value, fill), {}))
    for c in _CORPUS[:5]:
        out.append((c.label + "/dtype_override", (c.value, 9, np.float32), {}))
        out.append((c.label + "/subok_false", (c.value, 1), {"subok": False}))
    # RECLAIM 2026-08-01: same complex-fill-panic and narrow-int-cast bugs
    # as `full` above (both share `weak_scalar_buffer` in lib.rs) -- see the
    # comment block in `full_cases` for the full bug description.
    base = np.zeros((2,), dtype=np.float64)
    for dt in (np.bool_, np.int8, np.uint8, np.int64, np.float32, np.float64):
        out.append((f"complex_fill_non_complex_dtype/dt={dt.__name__}",
                     (base, 3 + 4j), {"dtype": dt}))
    for fill in (float("inf"), float("-inf"), 200.0, -200.0, 128.0, -129.0):
        for dt in (np.int8, np.uint8, np.int16, np.uint16):
            out.append((f"narrow_int_cast/fill={fill!r}/dt={dt.__name__}",
                         (base, fill), {"dtype": dt}))
    # CLASS B (see full_cases' comment block): same array-like
    # fill_value-must-broadcast bug, shared `fill_material`/
    # `broadcast_or_err` fix in creation.rs, plus `full_like`-specific
    # coverage of the prototype-order-aware relayout path (`like_perm`)
    # combined with a genuinely multi-element fill_value -- an order='K'/
    # F-contiguous-prototype combination the plain `full` cases above
    # can't exercise since `full`'s order is restricted to 'C'/'F' only.
    proto2x2 = np.zeros((2, 2))
    proto_f = np.asfortranarray(np.zeros((2, 3)))
    out.append(("array_fill_exact_shape", (proto2x2, [[2.0, 1.0], [0.0, 3.0]]), {}))
    out.append(("array_fill_broadcast", (proto2x2, [7.0, 8.0]), {}))
    out.append(("array_fill_bad_shape", (proto2x2, [1, 2, 3]), {}))
    out.append(("array_fill_F_proto_K_order", (proto_f, [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]), {}))
    out.append(("array_fill_shape_override", (proto_f, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), {"shape": (6,)}))
    out.append(("array_fill_scalar_still_works_F_proto", (proto_f, 9.0), {}))
    return out


# ---------------------------------------------------------------------------
# arange / linspace / eye / identity
# ---------------------------------------------------------------------------

def arange_cases():
    return [
        ("stop_only_int", (5,), {}),
        ("stop_only_float", (5.5,), {}),
        ("start_stop", (2, 9), {}),
        ("start_stop_step", (0, 10, 2), {}),
        ("negative_step", (10, 0, -1), {}),
        ("float_step", (0, 1, 0.1), {}),
        ("float_step_odd", (0, 2, 0.3), {}),
        ("negative_float_step", (5, -5, -1.5), {}),
        ("empty_range", (5, 0), {}),
        ("zero_stop", (0,), {}),
        ("dtype_override_int_to_float", (5,), {"dtype": np.float32}),
        ("dtype_override_float_to_int", (0, 5, 1.0), {"dtype": np.int32}),
        ("step_zero_raises", (0, 5, 0), {}),
        ("all_negative", (-5, -1), {}),
        ("large_range", (0, 10000), {}),
        ("like_none_explicit", (5,), {"like": None}),
        ("like_non_array_function_raises", (5,), {"like": 5}),
    ]


def linspace_cases():
    out = [
        ("basic", (0, 1), {}),
        ("num_10", (0, 1, 10), {}),
        ("num_0", (0, 1, 0), {}),
        ("num_1", (0, 1, 1), {}),
        ("no_endpoint", (0, 1, 10, False), {}),
        # axis=: array-valued start/stop broadcast together, then `num`
        # samples are inserted at `axis` in the OUTPUT (not the input)
        # shape -- verified against numpy 2.5.1 directly (moveaxis-style
        # placement, positive and negative axis, plus the default axis=0
        # form already covered by the scalar cases above).
        ("array_start_stop_axis0_default",
         (np.array([0.0, 10.0]), np.array([1.0, 20.0]), 5), {}),
        ("array_start_stop_axis_neg1",
         (np.array([0.0, 10.0]), np.array([1.0, 20.0]), 5), {"axis": -1}),
        ("array_start_stop_axis1_2d",
         (np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([[10.0, 11.0], [12.0, 13.0]]), 4),
         {"axis": 1}),
        ("array_start_scalar_stop_broadcast",
         (np.array([0.0, 1.0, 2.0]), 10.0, 6), {"axis": 0}),
        ("array_start_stop_axis_out_of_range_raises",
         (np.array([0.0, 1.0]), np.array([1.0, 2.0]), 3), {"axis": 5}),
        # retstep=True is NOT covered here: it makes both numpy and anionpy
        # return a (array, float) tuple, and harness.compare_values only
        # knows how to grade a single .dtype-carrying result (or, via
        # multi_output, a tuple of same-shaped arrays for nout>1 ufuncs) --
        # neither fits a (ndarray, python-float) pair. Verified correct by
        # hand instead (both the returned array AND the returned step
        # matched numpy bit-for-bit for linspace(0,10,5,retstep=True) and
        # linspace(0,10,5,endpoint=False,retstep=True)); not declared here
        # because "verified by hand" is not "has a passing differential
        # spec" -- this is a genuine harness gap (no tuple-of-mixed-types
        # comparison path), not an anionpy gap, left open rather than faked.
        ("negative_range", (5, -5, 11), {}),
        ("float_start_stop", (0.5, 2.75, 7), {}),
        ("dtype_override", (0, 1, 5), {"dtype": np.float32}),
        ("start_equals_stop", (3, 3, 4), {}),
        ("negative_num_raises", (0, 1, -1), {}),
        # RECLAIM 2026-08-01: for genuinely-integer dtypes numpy floors the
        # float sequence before narrowing (already correct), but for
        # dtype=bool specifically it does NOT floor first -- it casts each
        # raw float element by nonzero-ness directly (x != 0.0). anionpy used
        # to floor unconditionally (dt.is_integer() || dt == Bool in
        # creation.rs), which wrongly floored 0.25/0.5/0.75 down to 0
        # before the bool cast, e.g. linspace(0,1,5,dtype=bool) gave
        # [F,F,F,F,T] instead of numpy's [F,T,T,T,T]. Fixed by scoping the
        # floor to dt.is_integer() only.
        ("dtype_bool", (0, 1, 5), {"dtype": np.bool_}),
        ("dtype_bool_endpoint_false", (0, 1, 5), {"dtype": np.bool_, "endpoint": False}),
        ("dtype_bool_all_nonzero", (1, 2, 4), {"dtype": np.bool_}),
        ("dtype_bool_negative_range", (-2, 2, 5), {"dtype": np.bool_}),
    ]
    out.extend(_dtype_spelling_kwarg_cases((0, 1, 5)))
    return out


def eye_cases():
    out = [
        ("square_3", (3,), {}),
        ("rect_3x5", (3, 5), {}),
        ("rect_5x3", (5, 3), {}),
        ("offset_k1", (4, None, 1), {}),
        ("offset_k_neg1", (4, None, -1), {}),
        ("offset_k_large", (3, 3, 5), {}),
        ("zero_size", (0,), {}),
        ("dtype_override", (3, None, 0, np.int32), {}),
        ("negative_n_raises", (-1,), {}),
        # order=: prior to this fix `eye` had NO order parameter at all --
        # a real bug (hardcoded Order::C), not just missing coverage, see
        # task report. `compare_values` only checks dtype/shape/values (no
        # memory-layout check), so this exercises that `order='F'` is
        # accepted and produces the same logical values, not a strides
        # assertion -- verified separately by hand that the returned array
        # is genuinely F-contiguous.
        ("order_F", (3, 5), {"order": "F"}),
        ("order_C_explicit", (4,), {"order": "C"}),
        # N=/M= keyword names, added 2026-08-01: real numpy's `eye` first
        # two params are POSITIONAL_OR_KEYWORD named `N`/`M` (no `/` in its
        # signature) -- `np.eye(N=3, M=5)` genuinely works. Found via this
        # task's audit re-run that anionpy's binding used lowercase `n=`/`m=`
        # and rejected the real numpy call form outright; fixed in
        # ionp-py/src/creation.rs (see that function's own doc comment for
        # the full bug report) and verified bit-exact here.
        ("keyword_N_M", (), {"N": 3, "M": 5, "k": 1}),
        ("like_none_explicit", (3,), {"like": None}),
        ("like_non_array_function_raises", (3,), {"like": 5}),
    ]
    out.extend(_dtype_spelling_kwarg_cases((3,)))
    return out


def identity_cases():
    out = [
        ("size_1", (1,), {}),
        ("size_3", (3,), {}),
        ("size_0", (0,), {}),
        ("dtype_override", (3, np.complex128), {}),
        ("negative_raises", (-2,), {}),
        ("like_none_explicit", (3,), {"like": None}),
        ("like_non_array_function_raises", (3,), {"like": 5}),
    ]
    out.extend(_dtype_spelling_kwarg_cases((3,)))
    return out


# ---------------------------------------------------------------------------
# asarray / copy / ascontiguousarray
# ---------------------------------------------------------------------------

def asarray_cases():
    out = _corpus_cases()
    for c in _CORPUS[:5]:
        out.append((c.label + "/dtype_cast", (c.value,), {"dtype": np.float64}))
    # order=: forces a re-layout when the source isn't already that order.
    # copy=: tri-state (None/True/False); False raises when a copy can't
    # be avoided. Run across the FULL corpus (mixed C/F/strided layouts)
    # so both the "already matches, no-copy" and "must-copy" branches of
    # `copy=False` get genuinely exercised per-array, not hand-picked.
    for c in _CORPUS:
        out.append((c.label + "/order_C", (c.value,), {"order": "C"}))
        out.append((c.label + "/order_F", (c.value,), {"order": "F"}))
        out.append((c.label + "/copy_true", (c.value,), {"copy": True}))
        out.append((c.label + "/copy_false_order_K", (c.value,), {"copy": False}))
        out.append((c.label + "/copy_false_order_C", (c.value,), {"copy": False, "order": "C"}))
    out.extend(asarray_int_range_cases())
    out.extend(_dtype_spelling_kwarg_cases((_CORPUS[0].value,)))
    return out


def asarray_int_range_cases():
    """`asarray()` shares `array()`'s `array_impl`/`ndarray_from_pylist`
    plumbing for any non-ndarray source (a plain Python list/tuple/scalar
    is NOT ingested via the numpy-buffer path -- only an already-`np.ndarray`
    or `anionpy.ndarray` source is, verified live: `anionpy.asarray(np.array([...],
    dtype='uint64'))` round-trips the full uint64 range correctly today,
    while `anionpy.asarray([2**64-1], dtype='uint64')` raises the identical
    `OverflowError: Python int too large to convert to C long` `array()`
    does). See registry.py's `_array_custom_cases` for the full root-cause
    writeup (ionp-py/src/lib.rs's `flatten_nested` PyInt arm) -- not
    repeated here. These cases are the `asarray`-specific regression guard
    for the SAME bug via the SAME `convert_ionp_args=False` list-literal
    path `asarray`'s ItemSpec already uses; every one is expected to FAIL
    against anionpy today (root cause is in `lib.rs`, out of this task's
    file-ownership scope -- reported, not fixed)."""
    return [
        ("uint64_high_half_list", ([2**64 - 1],), {"dtype": "uint64"}),
        ("uint64_boundary_2pow63_list", ([2**63],), {"dtype": "uint64"}),
        ("int64_min_list", ([-(2**63)],), {"dtype": "int64"}),
        ("int64_max_list", ([2**63 - 1],), {"dtype": "int64"}),
        ("int64_precision_loss_regression", ([2**62 + 5],), {"dtype": "int64"}),
        ("uint64_overflow_2pow64_list", ([2**64],), {"dtype": "uint64"}),
        ("uint8_overflow_256_list", ([256],), {"dtype": "uint8"}),
    ]


def copy_cases():
    out = _corpus_cases()
    for c in _CORPUS[:5]:
        out.append((c.label + "/subok_false", (c.value,), {"subok": False}))
        out.append((c.label + "/subok_true", (c.value,), {"subok": True}))
    return out


def ascontiguousarray_cases():
    out = _corpus_cases()
    for c in _CORPUS[:5]:
        out.append((c.label + "/dtype_cast", (c.value,), {"dtype": np.float32}))
        out.append((c.label + "/like_none_explicit", (c.value,), {"like": None}))
    out.append(("like_non_array_function_raises", (_CORPUS[0].value,), {"like": 5}))
    return out


# ---------------------------------------------------------------------------
# View wrappers: reshape / ravel / transpose / swapaxes / moveaxis /
# squeeze / expand_dims / broadcast_to.
#
# Blanket-applying an axis-taking call across the FULL shape/dtype/view
# corpus (which includes 0-d and 1-d arrays) deliberately also exercises
# the out-of-bounds-axis path on both sides for every array whose ndim is
# too small for the chosen axis -- both numpy and anionpy are expected to
# raise there, which is itself a real (and, per the AxisError-vs-IndexError
# bug above, NOT free) assertion, not dead weight.
# ---------------------------------------------------------------------------

class _ViewCase:
    """A reshape(copy=False) probe built from a genuine INGESTION-NATIVE
    view on each side, not a numpy-sourced view fed through
    `anionpy.array()`.

    Why this class exists (2026-08-01, replacing the old
    `_reshape_copy_false_reachable` skip guard): that guard's premise --
    "any non-C/F-contiguous numpy view gets silently collapsed to a fresh
    contiguous copy at ingestion, so anionpy's reshape(copy=False) can never
    see the original strides" -- is real, but it was being used to excuse
    NOT testing the actual view-discovery algorithm in ionp-core's
    `reshape` at all (the corpus's `view/*` fixtures are without exception
    numpy-sourced views passed through `anionpy.array()`). The bug this task
    fixed (`shape::attempt_nocopy_reshape`, ported from numpy's own
    `_attempt_nocopy_reshape` in `_core/src/multiarray/shape.c`) lives
    entirely on the anionpy side, in `NdArray::reshape`/`reshape_with_order`,
    and only ever runs on a view that anionpy ITSELF produced via its own
    slicing/transpose/swapaxes -- i.e. it cannot be exercised by handing
    anionpy a numpy view, ingestion-boundary or not.

    So each `_ViewCase` builds its own view independently on each side:
    `base` (a plain, contiguous numpy array -- ingesting IT loses nothing)
    is ingested once into a real `anionpy.ndarray`, then `ops` (the same
    `[("getitem", key) | ("transpose", None) | ("swapaxes", (a, b)), ...]`
    sequence `probe_reshape.py`'s out-of-corpus fuzz probe uses) are
    replayed identically on the numpy array and the anionpy array, producing
    two independently-constructed but structurally-equivalent strided
    views. `reshape(..., copy=False)` is then called on both and compared
    for outcome (ValueError type match, or shape/dtype/value match) --
    see `_reshape_view_numpy_adapter`/`_reshape_view_ionp_adapter` below.

    Every case's expected direction (succeeds vs raises) was verified
    against real numpy 2.5.1 directly before being encoded here (see this
    task's scratch verification, not repeated in-line) -- none were
    guessed from the algorithm's expected behavior.
    """

    __slots__ = ("base", "ops", "target_shape", "order")

    def __init__(self, base, ops, target_shape, order="C"):
        self.base = base
        self.ops = ops
        self.target_shape = target_shape
        self.order = order


def _apply_view_ops(arr, ops):
    """Replay a `_ViewCase.ops` sequence against either a numpy.ndarray or
    an anionpy.ndarray -- both expose the same `__getitem__`/`.T`/
    `.swapaxes()` surface, so one function drives both sides."""
    for kind, arg in ops:
        if kind == "getitem":
            arr = arr[arg]
        elif kind == "transpose":
            arr = arr.T
        elif kind == "swapaxes":
            arr = arr.swapaxes(*arg)
        else:  # pragma: no cover - programmer error in a case definition
            raise ValueError(f"unknown view op kind {kind!r}")
    return arr


def _reshape_view_numpy_adapter(arg, shape=None, **kwargs):
    if isinstance(arg, _ViewCase):
        view = _apply_view_ops(arg.base, arg.ops)
        return np.reshape(view, arg.target_shape, order=arg.order, copy=False)
    return np.reshape(arg, shape, **kwargs)


def _reshape_view_ionp_adapter(arg, shape=None, **kwargs):
    import anionpy

    if isinstance(arg, _ViewCase):
        # `arg.base` is always plain C- or F-contiguous (see the cases
        # below), so this one ingestion copy loses nothing -- the view is
        # then built with anionpy's OWN slicing/transpose/swapaxes, exactly
        # like `probe_reshape.py`'s fuzz probe, never by ingesting an
        # already-strided numpy array.
        ionp_arr = arg.base if isinstance(arg.base, anionpy.ndarray) else anionpy.array(arg.base)
        view = _apply_view_ops(ionp_arr, arg.ops)
        return anionpy.reshape(view, arg.target_shape, order=arg.order, copy=False)
    ionp_arr = arg if isinstance(arg, anionpy.ndarray) else anionpy.array(arg)
    return anionpy.reshape(ionp_arr, shape, **kwargs)


def _reshape_native_view_cases():
    """New, real `reshape(copy=False)` cases built from genuine
    ionp-native views (see `_ViewCase`'s docstring), covering BOTH
    directions the fixed `attempt_nocopy_reshape` algorithm can take, for
    both `order="C"` and `order="F"`. Every outcome below was verified
    against real numpy 2.5.1 before being encoded."""
    out = []

    # --- view IS possible: copy=False must succeed and match numpy -----
    out.append((
        "native_view/2d_col_stride2_to_1d",
        (_ViewCase(np.arange(24.0).reshape(4, 6),
                    [("getitem", (slice(None), slice(None, None, 2)))],
                    (12,)),),
        {},
    ))
    out.append((
        "native_view/1d_stride2_to_2d",
        (_ViewCase(np.arange(20.0),
                    [("getitem", slice(None, None, 2))],
                    (5, 2)),),
        {},
    ))
    out.append((
        "native_view/3d_inner_stride2_flatten",
        (_ViewCase(np.arange(48.0).reshape(2, 3, 8),
                    [("getitem", (slice(None), slice(None), slice(None, None, 2)))],
                    (24,)),),
        {},
    ))
    out.append((
        "native_view/3d_inner_stride2_regroup",
        (_ViewCase(np.arange(48.0).reshape(2, 3, 8),
                    [("getitem", (slice(None), slice(None), slice(None, None, 2)))],
                    (2, 1, 12)),),
        {},
    ))
    out.append((
        "native_view/f_contig_flatten",
        (_ViewCase(np.asfortranarray(np.arange(12.0).reshape(4, 3)),
                    [], (12,), order="F"),),
        {},
    ))
    out.append((
        "native_view/f_last_axis_slice_regroup",
        (_ViewCase(np.asfortranarray(np.arange(24.0).reshape(4, 3, 2)),
                    [("getitem", (slice(None), slice(None), slice(0, 1)))],
                    (4, 3), order="F"),),
        {},
    ))
    out.append((
        "native_view/f_middle_axis_partial_slice",
        (_ViewCase(np.asfortranarray(np.arange(24.0).reshape(4, 3, 2)),
                    [("getitem", (slice(None), slice(1, None), slice(None)))],
                    (4, 2, 2), order="F"),),
        {},
    ))

    # --- view is genuinely IMPOSSIBLE: copy=False must raise ValueError -
    out.append((
        "native_view/2d_transposed_flatten_raises",
        (_ViewCase(np.arange(12.0).reshape(3, 4),
                    [("transpose", None)], (12,)),),
        {},
    ))
    out.append((
        "native_view/2d_reversed_rows_flatten_raises",
        (_ViewCase(np.arange(12.0).reshape(3, 4),
                    [("getitem", slice(None, None, -1))], (12,)),),
        {},
    ))
    out.append((
        "native_view/3d_swapaxes_flatten_raises",
        (_ViewCase(np.arange(24.0).reshape(2, 3, 4),
                    [("swapaxes", (0, 1))], (24,)),),
        {},
    ))
    out.append((
        "native_view/3d_middle_slice_flatten_raises",
        (_ViewCase(np.arange(24.0).reshape(2, 3, 4),
                    [("getitem", (slice(None), slice(1, None), slice(None)))],
                    (16,)),),
        {},
    ))
    out.append((
        "native_view/f_stride2_slice_raises",
        (_ViewCase(np.asfortranarray(np.arange(12.0).reshape(4, 3)),
                    [("getitem", (slice(None), slice(None, None, 2)))],
                    (8,), order="F"),),
        {},
    ))
    return out


def reshape_cases():
    out = []
    for c in _CORPUS:
        out.append((c.label + "/flatten", (c.value, (-1,)), {}))
        out.append((c.label + "/same_shape", (c.value, c.value.shape), {}))
    out.append(("reshape_fail/incompatible_size", (np.arange(6), (4,)), {}))
    out.append(("reshape_neg1_infer", (np.arange(12), (3, -1)), {}))
    out.append(("reshape_order_F", (np.arange(6).reshape(2, 3), (3, 2)), {"order": "F"}))
    # copy=: tri-state, run across the FULL layout corpus (C/F/strided/
    # transposed) with a `-1` (flatten) target -- whether numpy needs to
    # copy to flatten depends on the specific layout, so this genuinely
    # exercises both the "view is possible, copy=False succeeds" and the
    # "view is NOT possible, copy=False raises ValueError" branches per
    # array, not a hand-picked single case.
    #
    # `copy=False` is skipped for four specific `view/*` corpus entries --
    # `2d_col_slice_noncontig`, `2d_row_reverse`, `3d_transpose_axes`,
    # `3d_middle_slice` -- a genuine, acknowledged architectural fact, NOT
    # an excuse for skipping the reshape ALGORITHM fix (that fix is now
    # directly, honestly exercised in both directions by
    # `_reshape_native_view_cases()` above, using ionp-native views built
    # by anionpy's own slicing/transpose/swapaxes -- never by ingesting an
    # already-strided numpy array, which is exactly the case these four
    # corpus fixtures are and exactly why they can't reach it): this
    # crate's whole architecture (lib.rs's own module doc, and `array()`'s
    # existing `copy=False` divergence note in ionp-py/src/lib.rs) is that
    # ingesting any real external `numpy.ndarray` ALWAYS physically copies
    # it into a fresh ionp-owned buffer, and that ingestion only preserves
    # genuine F-contiguity -- everything else (these four, all
    # neither-C-nor-F strided views) is silently normalized to a fresh
    # C-contiguous copy during that ONE-TIME ingestion, before
    # `reshape()`/`ndarray.reshape` is ever called. By the time
    # `reshape(copy=False)` runs, the array it sees genuinely IS already
    # contiguous, so it correctly, honestly returns a view with zero
    # further copying -- it has no way to know (and no representation
    # capable of knowing) that the ORIGINAL numpy source needed a copy to
    # reach that state, because that copy already happened, invisibly, at
    # the ingestion boundary. Real numpy raises `ValueError: Unable to
    # avoid creating a copy while reshaping.` for these four because ITS
    # reshape sees the true original strides; anionpy's `reshape()` cannot
    # reproduce that answer without anionpy storing genuine external strided
    # views, which it deliberately does not do. Measured directly (scratch
    # verification): the other seven `view/*` entries
    # (`1d_reverse_negstride`, `1d_stride2`, `1d_stride2_negative`,
    # `2d_transpose`, `2d_fully_reversed`, `2d_fortran_order`,
    # `2d_diag_view`) all genuinely agree between numpy and anionpy on
    # `copy=False` (some because F-contiguity IS preserved through
    # ingestion, some because both sides independently agree no copy is
    # needed) and ARE exercised below; `copy=True`/`copy=None` are
    # unaffected by this gap for all eleven `view/*` entries (verified) and
    # keep exercising the full corpus.
    for c in _CORPUS:
        out.append((c.label + "/copy_true", (c.value, (-1,)), {"copy": True}))
        if _reshape_copy_false_reachable(c.value):
            out.append((c.label + "/copy_false", (c.value, (-1,)), {"copy": False}))
        out.append((c.label + "/copy_none", (c.value, (-1,)), {"copy": None}))
    out.extend(_reshape_native_view_cases())
    return out


def ravel_cases():
    out = [(c.label, (c.value,), {}) for c in _CORPUS]
    out.append(("order_F", (np.arange(6).reshape(2, 3),), {"order": "F"}))
    return out


def transpose_cases():
    out = [(c.label + "/no_axes", (c.value,), {}) for c in _CORPUS]
    out.append(("explicit_axes_3d", (np.arange(24).reshape(2, 3, 4), (2, 0, 1)), {}))
    out.append(("explicit_axes_2d_identity", (np.arange(6).reshape(2, 3), (0, 1)), {}))
    out.append(("negative_axes", (np.arange(24).reshape(2, 3, 4), (-1, -2, -3)), {}))
    return out


def swapaxes_cases():
    out = [(c.label, (c.value, 0, 1), {}) for c in _CORPUS]
    out.append(("negative_axes_3d", (np.arange(24).reshape(2, 3, 4), -1, -2), {}))
    # Explicit out-of-range-axis cases (not just incidental via the 0-d/1-d
    # corpus members above): both axis1 and axis2 positions, and negative
    # out-of-range too. Real numpy 2.5.1 raises `numpy.exceptions.AxisError`
    # for all of these -- now that `normalize_axis` (ionp-py/src/creation.rs)
    # raises the real `AxisError` via `IonpError::AxisError`, anionpy matches.
    out.append(("axis1_out_of_range", (np.arange(6).reshape(2, 3), 5, 0), {}))
    out.append(("axis2_out_of_range", (np.arange(6).reshape(2, 3), 0, 5), {}))
    out.append(("axis_neg_out_of_range", (np.arange(6).reshape(2, 3), -5, 0), {}))
    return out


def moveaxis_cases():
    out = [(c.label, (c.value, 0, -1), {}) for c in _CORPUS]
    out.append(("multi_axis_3d", (np.arange(24).reshape(2, 3, 4), (0, 1), (2, 1)), {}))
    out.append(("source_out_of_range", (np.arange(6).reshape(2, 3), 5, 0), {}))
    out.append(("destination_out_of_range", (np.arange(6).reshape(2, 3), 0, 5), {}))
    # 0-d out-of-range: the two cases above are 2-D, and anionpy raised the
    # correct AxisError for them while still raising a plain IndexError on
    # 0-d input -- so this item read GREEN with a real exception-type
    # divergence underneath it. The harness compares exception type strictly
    # (harness.py: `type(ionp_exc) not in allowed`); the blindness was the
    # corpus never exercising ndim==0. Measured out-of-corpus at 2,160
    # divergences, all exception-type, zero value/shape.
    out.append(("0d_source_out_of_range", (np.array(5), 0, 0), {}))
    out.append(("0d_source_negative_out_of_range", (np.array(5), -1, 0), {}))
    out.append(("0d_destination_out_of_range", (np.array(5.5), 0, 1), {}))
    out.append(("1d_source_out_of_range", (np.arange(3), 3, 0), {}))
    out.append(("1d_negative_out_of_range", (np.arange(3), -2, 0), {}))
    return out


def squeeze_cases():
    out = [(c.label + "/no_axis", (c.value,), {}) for c in _CORPUS]
    # axis=0 excluded for ndim==0 arrays specifically: numpy's squeeze
    # special-cases a BARE (non-tuple) axis 0 (and -1) as always-valid on a
    # 0-d array (`np.squeeze(np.array(5), axis=0)` succeeds and returns the
    # array unchanged -- confirmed directly; NOT the general
    # axis-normalization rule the rest of numpy's axis-taking functions
    # use, since `np.core.numeric.normalize_axis_tuple(0, 0)` on its own
    # raises AxisError for the exact same (axis=0, ndim=0) pair, and even
    # squeeze's own TUPLE-axis form `axis=(0,)` does NOT get this special
    # case -- only a bare int does). FIXED in both `creation.rs`'s and
    # `ndarray_attrs.rs`'s `squeeze` wrappers with a dedicated ndim==0 +
    # bare-int-axis-in-{0,-1} bypass; covered below by the dedicated
    # `squeeze_0d_axis0_noop`/`squeeze_0d_axis_neg1_noop` cases, and the
    # exclusion here can now be lifted for the general corpus loop too.
    out += [(c.label + "/axis0", (c.value,), {"axis": 0}) for c in _CORPUS if c.value.ndim != 0]
    out.append(("axis_tuple", (np.zeros((1, 3, 1)), (0, 2)), {}))
    out.append(("squeeze_0d_axis0_noop", (np.array(5),), {"axis": 0}))
    out.append(("squeeze_0d_axis_neg1_noop", (np.array(5.5),), {"axis": -1}))
    # Tuple-form axis on a 0-d array does NOT get the no-op special case --
    # numpy still raises AxisError here (verified directly).
    out.append(("squeeze_0d_axis_tuple_still_raises", (np.array(5),), {"axis": (0,)}))
    out.append(("out_of_range", (np.zeros((1, 3, 1)),), {"axis": 5}))
    return out


def expand_dims_cases():
    out = [(c.label + "/axis0", (c.value, 0), {}) for c in _CORPUS]
    out += [(c.label + "/axis_neg1", (c.value, -1), {}) for c in _CORPUS]
    out.append(("out_of_range", (np.arange(3), 5), {}))
    # 0-d out-of-range -- same corpus blindness as moveaxis above: the 1-D
    # case passed while 0-d input still raised plain IndexError instead of
    # numpy.exceptions.AxisError.
    out.append(("0d_out_of_range", (np.array(5), 2), {}))
    out.append(("0d_negative_out_of_range", (np.array(5), -3), {}))
    out.append(("0d_axis0_valid", (np.array(5), 0), {}))
    out.append(("0d_axis_neg1_valid", (np.array(5.5), -1), {}))
    out.append(("1d_negative_out_of_range", (np.arange(3), -4), {}))
    return out


def broadcast_to_cases():
    return [
        ("scalar_to_3x3", (np.array(5), (3, 3)), {}),
        ("1d_to_2d", (np.array([1, 2, 3]), (4, 3)), {}),
        ("row_broadcast", (np.zeros((1, 3)), (5, 3)), {}),
        ("identity_broadcast", (np.arange(6).reshape(2, 3), (2, 3)), {}),
        ("add_leading_dims", (np.arange(6).reshape(2, 3), (4, 2, 3)), {}),
        ("zero_dim_broadcast", (np.zeros((0, 3)), (5, 0, 3)), {}),
        ("noncontig_input", (np.arange(20, dtype=np.float64).reshape(4, 5)[:, ::-1], (4, 4, 5)), {}),
        ("incompatible_raises", (np.arange(24).reshape(2, 3, 4), (9, 9)), {}),
        ("fewer_target_dims_raises", (np.arange(6).reshape(2, 3), (3,)), {}),
        ("mismatched_trailing_raises", (np.arange(6).reshape(2, 3), (2, 4)), {}),
        ("subok_false", (np.arange(6).reshape(2, 3), (2, 3)), {"subok": False}),
        ("subok_true", (np.arange(6).reshape(2, 3), (4, 2, 3)), {"subok": True}),
    ]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _build_creation_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["zeros"] = ItemSpec(name="zeros", kind="custom", custom_cases=zeros_cases)
    specs["ones"] = ItemSpec(name="ones", kind="custom", custom_cases=ones_cases)
    specs["full"] = ItemSpec(name="full", kind="custom", custom_cases=full_cases)
    specs["empty"] = ItemSpec(
        name="empty", kind="custom", custom_cases=empty_cases,
        numpy_adapter=_empty_adapter(np.empty),
        ionp_adapter=_empty_adapter(__import__("anionpy").empty),
    )

    # convert_ionp_args=False added 2026-08-01 to zeros_like/ones_like/
    # full_like/empty_like/copy/broadcast_to (below), alongside a real
    # `ionp-py/src/creation.rs` fix (`extract_or_ingest_ndarray`, replacing
    # `extract_pyarray` for exactly these six). Before that Rust fix, every
    # one of these functions raised PyO3's own "'ndarray' object is not an
    # instance of 'ndarray'" TypeError for a genuine external
    # `numpy.ndarray` argument (confirmed directly against
    # `anionpy.zeros_like(np.zeros(3), subok=False)` etc, independent of
    # `subok`'s value -- `check_subok` is an intentional no-op, not the
    # actual gate); the default True here was silently hiding that,
    # because `make_ionp_array_converter` was ingesting `c.value` into a
    # real `anionpy.ndarray` BEFORE these functions ever saw it, so the corpus
    # never actually exercised the foreign-numpy-input path these
    # functions are documented (and now fixed) to accept. Flipping to
    # False, same as `asarray` above, makes every EXISTING case in
    # `zeros_like_cases()` etc a genuine foreign-numpy-input test with no
    # duplication; the only kwargs any of these cases pass alongside the
    # array (`dtype=`, `shape=`, `subok=`, a scalar `fill`) are never
    # numpy.ndarray instances, so this cannot silently change what a
    # non-array argument receives.
    specs["zeros_like"] = ItemSpec(
        name="zeros_like", kind="custom", custom_cases=zeros_like_cases, convert_ionp_args=False,
    )
    specs["ones_like"] = ItemSpec(
        name="ones_like", kind="custom", custom_cases=ones_like_cases, convert_ionp_args=False,
    )
    specs["full_like"] = ItemSpec(
        name="full_like", kind="custom", custom_cases=full_like_cases, convert_ionp_args=False,
    )
    specs["empty_like"] = ItemSpec(
        name="empty_like", kind="custom", custom_cases=empty_like_cases, convert_ionp_args=False,
        numpy_adapter=_empty_adapter(np.empty_like),
        ionp_adapter=_empty_adapter(__import__("anionpy").empty_like),
    )

    specs["arange"] = ItemSpec(name="arange", kind="custom", custom_cases=arange_cases)
    specs["linspace"] = ItemSpec(name="linspace", kind="custom", custom_cases=linspace_cases)
    specs["eye"] = ItemSpec(name="eye", kind="custom", custom_cases=eye_cases)
    specs["identity"] = ItemSpec(name="identity", kind="custom", custom_cases=identity_cases)

    # asarray specifically needs convert_ionp_args=False: it IS the numpy
    # -> anionpy boundary function (`anionpy.asarray` delegates straight to
    # `lib.rs`'s `array()`, which only accepts a real numpy.ndarray --
    # checked via `__array_interface__`/type-name -- or a Python
    # list/tuple). The default True behavior (auto-convert every
    # numpy.ndarray argument to a real anionpy.ndarray first, via
    # make_ionp_array_converter) is exactly backwards for this one item:
    # confirmed directly (`anionpy.asarray(a_real_numpy_array)` works for
    # every claimed dtype; every case failed with "unsupported numpy
    # dtype" under the default True, which was this test's own
    # methodology bug, not an anionpy defect).
    #
    # `copy` USED TO be the opposite case (kept the default True): its Rust
    # side (`extract_pyarray`) required an already-ionp.ndarray input and
    # rejected a raw numpy.ndarray outright (confirmed directly:
    # `anionpy.copy(raw_numpy_array)` -> TypeError), unlike `asarray`, it was
    # not the numpy-ingestion boundary itself. FIXED 2026-08-01
    # (`extract_or_ingest_ndarray`, see the `zeros_like` comment above for
    # the full story); `copy` now takes `convert_ionp_args=False` too, for
    # the same reason -- every existing case becomes a genuine
    # foreign-numpy test with no duplication needed.
    #
    # `ascontiguousarray` had the IDENTICAL underlying defect (still called
    # `extract_pyarray`; confirmed directly:
    # `anionpy.ascontiguousarray(np.zeros(3))` raised the same TypeError) --
    # authorized and FIXED 2026-08-01 in the same follow-up pass as
    # `ravel`/`transpose`/`swapaxes`/`moveaxis`/`squeeze`/`expand_dims`
    # below (all seven use the same `extract_or_ingest_ndarray` fix as
    # `zeros_like` et al above; `reshape` stayed explicitly off-limits and
    # was NOT touched). `convert_ionp_args=False` here for the same reason
    # as everywhere else in this file: every existing case becomes a
    # genuine foreign-numpy test with no duplication.
    specs["asarray"] = ItemSpec(
        name="asarray", kind="custom", custom_cases=asarray_cases, convert_ionp_args=False,
    )
    specs["copy"] = ItemSpec(
        name="copy", kind="custom", custom_cases=copy_cases, convert_ionp_args=False,
    )
    specs["ascontiguousarray"] = ItemSpec(
        name="ascontiguousarray", kind="custom", custom_cases=ascontiguousarray_cases,
        convert_ionp_args=False,
    )

    # numpy_adapter/ionp_adapter added 2026-08-01 alongside
    # `_reshape_native_view_cases()`/`_ViewCase` above: every EXISTING case
    # here still passes a plain (array, shape) pair through unchanged (see
    # `_reshape_view_numpy_adapter`/`_reshape_view_ionp_adapter`'s
    # fallthrough branch), but a `_ViewCase` case builds a genuine
    # ionp-native view independently on each side instead of letting the
    # default numpy.ndarray auto-conversion (`convert_ionp_args=True`,
    # kept here) ingest an already-strided numpy view and silently lose
    # its stride information -- see `_ViewCase`'s docstring for why that
    # auto-conversion is exactly backwards for these specific cases.
    specs["reshape"] = ItemSpec(
        name="reshape", kind="custom", custom_cases=reshape_cases,
        numpy_adapter=_reshape_view_numpy_adapter,
        ionp_adapter=_reshape_view_ionp_adapter,
    )
    # ravel/transpose/swapaxes/moveaxis/squeeze/expand_dims: same
    # extract_or_ingest_ndarray fix + convert_ionp_args=False story as
    # ascontiguousarray immediately above (2026-08-01 follow-up pass).
    # None of these cases' extra args (axis ints/tuples, an `order`
    # string) are ever numpy.ndarray instances, so flipping this cannot
    # silently change what a non-array argument receives.
    specs["ravel"] = ItemSpec(
        name="ravel", kind="custom", custom_cases=ravel_cases, convert_ionp_args=False,
    )
    specs["transpose"] = ItemSpec(
        name="transpose", kind="custom", custom_cases=transpose_cases, convert_ionp_args=False,
    )
    specs["swapaxes"] = ItemSpec(
        name="swapaxes", kind="custom", custom_cases=swapaxes_cases, convert_ionp_args=False,
    )
    specs["moveaxis"] = ItemSpec(
        name="moveaxis", kind="custom", custom_cases=moveaxis_cases, convert_ionp_args=False,
    )
    specs["squeeze"] = ItemSpec(
        name="squeeze", kind="custom", custom_cases=squeeze_cases, convert_ionp_args=False,
    )
    specs["expand_dims"] = ItemSpec(
        name="expand_dims", kind="custom", custom_cases=expand_dims_cases, convert_ionp_args=False,
    )
    # broadcast_to: same fix/story as zeros_like et al above -- FIXED
    # 2026-08-01 (`extract_or_ingest_ndarray`); `convert_ionp_args=False`
    # makes every existing `broadcast_to_cases()` entry a genuine
    # foreign-numpy test (the second positional arg is always a plain
    # shape tuple, never a numpy.ndarray, so this cannot change what it
    # receives).
    specs["broadcast_to"] = ItemSpec(
        name="broadcast_to", kind="custom", custom_cases=broadcast_to_cases,
        convert_ionp_args=False,
    )

    return specs


CREATION_SPECS = _build_creation_specs()
