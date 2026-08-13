"""Differential specs for the SET OPERATIONS / DIFF-UNIQUE / SORTING-
REMAINDER block: `unique, unique_counts, unique_inverse, unique_all, diff,
ediff1d, trim_zeros, intersect1d, isin, setdiff1d, setxor1d, union1d` --
backed by `ionp-core/src/setops.rs` and the thin wrappers in
`ionp-py/src/setops.rs`.

Deliberately NOT declared here, with reasons (see also
`anionpy/_state/toplevel.py`'s block comment for this family, which repeats
this for ledger readers who don't read this file):

  - `unique_values` (the Array API function, distinct from `unique`):
    real numpy documents it (verified via `inspect.getsource(np.unique_values)`)
    as `np.unique(x, equal_nan=False, sorted=False)`, and numpy >=2.3 uses,
    per its own source comment, "a faster algorithm that does not rely on
    sorting, and hence the results are no longer implicitly sorted" -- an
    intentionally UNSPECIFIED hash-based order. This only produces an
    observable difference from a sorted order for small-bounded-range
    integer dtypes (int8/int16 hit numpy's fast hash path in practice);
    verified directly that the value SETS match anionpy bit-for-bit and only
    the ORDER differs, and that order is not pinned by any spec anionpy could
    target. Not a bug to chase -- permanently out of scope for a bit-exact
    harness. `unique(ar, sorted=False)` on int/uint/complex dtypes with no
    return_index/return_inverse/return_counts flag hits the exact same
    unspecified-hash-order landmine (bare numpy `unique()`'s own
    `sorted=False` argument routes through the same fast hash path
    `unique_values` uses) -- `sorted=` is therefore only exercised here
    against `sorted=True` (verified byte-exact; `sorted=False` is
    deliberately NOT probed or claimed).

`unique(a, axis=...)` IS implemented and declared exact (2026-08-01):
real numpy's own dispatcher (`_arraysetops_impl.unique`, read via
`inspect.getsource`) hardcodes `axis=None` internally whenever
`ar.ndim <= 1` -- an explicit `axis=0`/`axis=-1` on a 1-D array is
VALIDATED (raises `AxisError` if out of range) but otherwise byte-
identical to `axis=None`, `equal_nan` included. For `ar.ndim >= 2`, numpy
takes a genuinely different path: `moveaxis(ar, axis, 0)` -> reshape to
2-D `(n, m)` -> view each row as a structured/void dtype -> run the
flat-unique algorithm on THAT. Because a void dtype never satisfies
numpy's own `aux.dtype.kind in "cfmM"` equal_nan-special-case gate, this
path NEVER collapses NaN regardless of `equal_nan`'s value (verified
empirically both ways against live numpy 2.5.1), and never takes the
hash-fast-path either (structured dtype isn't hash-eligible), so
`sorted=` has zero observable effect once `axis=` is given. Signed zero:
no forced tie-break (unlike the flat path's `-0.0`-before-`+0.0` rule) --
rows differing only in sign of zero merge via a plain stable sort with
the FIRST-occurring row's sign surviving as representative (verified with
two input orderings). `ionp-core/src/setops.rs`'s `unique_axis` implements
this exactly; see its own doc comment for the moveaxis/transpose
arithmetic.

`intersect1d(..., return_indices=True)` IS implemented and declared exact
(2026-08-01), backed by `ionp-core/src/setops.rs`'s
`intersect1d_return_indices` (see its doc comment for the full derivation
from `inspect.getsource(np.intersect1d)`). `intersect1d`'s ItemSpec below
uses `numpy_adapter`/`ionp_adapter` to always compare a tuple (1-tuple
when `return_indices=False`, the real 3-tuple when `True`) under
`multi_output=True` uniformly -- real numpy's own return SHAPE depends on
`return_indices`, which a single `ItemSpec.multi_output` flag can't
express per-case (same reason `unique`'s tuple-returning Array API
siblings are graded as separate registry items rather than mixed into
`unique` itself).

One real bug was found and fixed while building this file's evidence:
`ionp-core/src/setops.rs`'s own hand-derived complex sort comparator
(bucket-by-any-NaN-component, then lexicographic real/imag within bucket)
independently repeated a mistake `ionp-core/src/sort.rs`'s `complex_cmp`
had already found and fixed in an earlier task: numpy's real complex `LT`
(`npysort_common.h.src`'s `@TYPE@_LT`) is NOT that bucketed rule. Fixed by
deleting setops.rs's own comparator and delegating to the already-verified
`sort::complex_cmp` (6000-case randomized stress sweep, 0 mismatches, see
that function's doc comment) instead of maintaining two independently
"verified" (but disagreeing on the general case) complex orderings in one
crate. `fixup_complex_nan_group_representative` (setops.rs, the targeted
post-pass that makes complex `equal_nan=True` canonicalizing-path NaN-group
representative selection match numpy's min-original-index rule) was
re-verified against the corrected ordering and needed no further change.

Every item below was probed against real numpy 2.5.1 comparing raw bytes
(dtype + shape + `.tobytes()`, never `allclose`) via a standalone sweep
script (2374 total comparisons across this file's full function set before
this registry file existed) -- see this task's report for the exact
compared/mismatch counts per function. This file's `custom_cases()`
builders below are the same corpus construction, now wired into the
regular differential harness instead of living only in a throwaway script.

Same file-ownership-fence merge pattern as sort_cases.py / linalg_cases.py
/ etc: a standalone `SETOPS_SPECS` dict, merged into registry.REGISTRY at
the tail of registry.py with a collision check.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401

import corpus
from registry import ItemSpec

# ---------------------------------------------------------------------------
# Shared small-array corpus for set operations (intersect1d/union1d/
# setdiff1d/setxor1d/isin) -- these need genuine value OVERLAP between two
# arrays to actually exercise the set logic, which corpus.py's generic
# binary_corpus() (independent random floats, broadcast-shape-focused) does
# not reliably provide. Mirrors the ad hoc probe corpus that already
# confirmed 0 mismatches across 32-per-item comparisons for each of these
# five functions, now promoted into the permanent registry.
# ---------------------------------------------------------------------------

def _set_pairs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [
        ("overlap_int", np.array([1, 2, 2, 3, 5]), np.array([2, 3, 4])),
        ("empty_a", np.array([], dtype=np.float64), np.array([1.0, 2.0])),
        ("empty_b", np.array([1.0, 2.0]), np.array([], dtype=np.float64)),
        ("nan_both_sides", np.array([1.0, np.nan, 2.0, np.nan]), np.array([np.nan, 2.0, 9.0])),
        ("signed_zero", np.array([-0.0, 0.0, 1.0]), np.array([0.0, -1.0])),
        ("dup_heavy", np.array([1, 1, 1, 2, 2]), np.array([1, 3, 3])),
        ("mixed_dtype_int_float", np.array([1, 2, 3], dtype=np.int32), np.array([2, 3, 4], dtype=np.float64)),
        ("complex_with_nan", np.array([1 + 2j, 3 + 4j, complex(np.nan, 0)]),
         np.array([3 + 4j, 5 + 6j, complex(0, np.nan)])),
        ("no_overlap", np.array([1, 2, 3]), np.array([10, 20, 30])),
        ("identical", np.array([1, 2, 3]), np.array([1, 2, 3])),
        ("bool", np.array([True, False, True]), np.array([False, False])),
    ] + _set_pairs_dtype_sweep()


# ---------------------------------------------------------------------------
# Dtype-sweep pairs, added 2026-08-01: the corpus above was reported blind
# on 9 of 14 supported dtypes -- it only ever exercised the numpy DEFAULT
# int width (platform int64), float64, int32, (default) complex128, and
# bool. A setops item's Rust backing (`ionp-core/src/setops.rs`) dispatches
# on dtype internally (see e.g. its per-width integer sort paths and its
# hand-derived complex comparator, both called out in this file's module
# docstring), so a corpus that never actually constructs a float16/float32/
# complex64/int8/int16/uint8/uint16/uint32/uint64 array cannot exercise --
# and therefore cannot falsify -- whatever those paths do. This block adds
# every one of those dtypes orthogonally, on top of (not instead of) the
# shape/edge-case corpus above: overlap, a dtype-specific empty-operand
# case, and (for the float/complex kinds, where NaN-collation and signed-
# zero handling are precision-sensitive per numpy's own `equal_nan` special
# case gate documented at the top of this file) a NaN-carrying case too --
# a float16 NaN and a float64 NaN are not the same bit pattern question.
# ---------------------------------------------------------------------------

_SETOPS_SWEEP_DTYPES: dict[type, tuple[list, list]] = {
    np.float16: ([1.0, 2.0, 2.0, 3.0, 5.0], [2.0, 3.0, 4.0]),
    np.float32: ([1.0, 2.0, 2.0, 3.0, 5.0], [2.0, 3.0, 4.0]),
    np.complex64: ([1 + 2j, 3 + 4j, 3 + 4j, 5 + 0j], [3 + 4j, 5 + 6j, 7 + 8j]),
    np.complex128: ([1 + 2j, 3 + 4j, 3 + 4j, 5 + 0j], [3 + 4j, 5 + 6j, 7 + 8j]),
    np.int8: ([-3, -1, -1, 0, 5, 127], [-1, 0, 4, 127]),
    np.int16: ([-30000, -1, -1, 0, 5, 30000], [-1, 0, 4, 30000]),
    np.int64: ([1, 2, 2, 3, 5], [2, 3, 4]),
    np.uint8: ([0, 1, 1, 3, 5, 255], [1, 0, 4, 255]),
    np.uint16: ([0, 1, 1, 3, 5, 60000], [1, 0, 4, 60000]),
    np.uint32: ([0, 1, 1, 3, 5, 4_000_000_000], [1, 0, 4, 4_000_000_000]),
    np.uint64: ([0, 1, 1, 3, 5, 18_000_000_000_000_000_000], [1, 0, 4, 18_000_000_000_000_000_000]),
}


def _set_pairs_dtype_sweep() -> list[tuple[str, np.ndarray, np.ndarray]]:
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for dt, (a_vals, b_vals) in _SETOPS_SWEEP_DTYPES.items():
        name = np.dtype(dt).name
        a = np.array(a_vals, dtype=dt)
        b = np.array(b_vals, dtype=dt)
        out.append((f"dtype_sweep/{name}/overlap", a, b))
        out.append((f"dtype_sweep/{name}/empty_a", np.array([], dtype=dt), b))
        out.append((f"dtype_sweep/{name}/empty_b", a, np.array([], dtype=dt)))
        if np.dtype(dt).kind in "fc":
            nan = dt(np.nan) if np.dtype(dt).kind == "f" else dt(complex(np.nan, 0))
            nan2 = dt(np.nan) if np.dtype(dt).kind == "f" else dt(complex(0, np.nan))
            a_nan = np.array(list(a_vals) + [nan, nan2], dtype=dt)
            b_nan = np.array(list(b_vals) + [nan2], dtype=dt)
            out.append((f"dtype_sweep/{name}/nan_both_sides", a_nan, b_nan))
    return out


# ---------------------------------------------------------------------------
# FORM axis (2026-08-02, CLASS A closure task) for the binary set ops:
# `intersect1d`/`setdiff1d`/`setxor1d`/`union1d`/`isin` are all kind="custom"
# and (`isin`/`intersect1d` included) route through `_wrap_custom_conversion`,
# which -- per `make_ionp_array_converter`'s own doc comment, already
# established for the unary builders below -- only rewrites actual
# `numpy.ndarray` instances into `anionpy.ndarray`, never a bare Python list/
# tuple/range CONTAINER. A sampled subset of `_set_pairs()` gets both
# operands converted to list/tuple form together here, so e.g.
# `anionpy.intersect1d([1, 2, 2, 3, 5], [2, 3, 4])` is genuinely exercised, not
# just `anionpy.intersect1d(np.array(...), np.array(...))`.
# ---------------------------------------------------------------------------

def _form_axis_binary_pairs(stride: int = 3) -> list[tuple[str, object, object]]:
    out: list[tuple[str, object, object]] = []
    pairs = _set_pairs()
    for i, (label, a, b) in enumerate(pairs):
        if i % stride != 0:
            continue
        out.append((f"{label}/form_list", a.tolist(), b.tolist()))
        out.append((f"{label}/form_tuple", tuple(a.tolist()), tuple(b.tolist())))
        out.append((f"{label}/form_mixed_list_ndarray", a.tolist(), b))
    return out


def _intersect1d_cases():
    out = []
    for label, a, b in _set_pairs():
        for au in (False, True):
            for ri in (False, True):
                out.append((
                    f"{label}/assume_unique={au}/return_indices={ri}",
                    (a, b),
                    {"assume_unique": au, "return_indices": ri},
                ))
    for label, a, b in _form_axis_binary_pairs():
        out.append((f"{label}/assume_unique=False/return_indices=False", (a, b), {}))
    return out


def _intersect1d_numpy_adapter(a, b, assume_unique=False, return_indices=False):
    """Always returns a tuple -- `intersect1d`'s ItemSpec below is
    `multi_output=True` uniformly (real numpy switches between a plain
    array and a 3-tuple depending on `return_indices`, which a single
    `ItemSpec.multi_output` can't express per-case; see this file's
    `unique`/`unique_all` split docstring for the established precedent of
    handling a flag-dependent return SHAPE via separate machinery rather
    than mixing shapes under one comparator). Wrapping the plain-array
    form in a 1-tuple changes nothing about what's compared -- `int1d` is
    still graded index-for-index, byte-exact, against numpy's real output.
    """
    out = np.intersect1d(a, b, assume_unique=assume_unique, return_indices=return_indices)
    return out if return_indices else (out,)


def _intersect1d_ionp_adapter(a, b, assume_unique=False, return_indices=False):
    # `a`/`b` arrive already converted to `anionpy.ndarray` by
    # `ItemSpec._wrap_custom_conversion` (applies to `ionp_adapter`-based
    # items too, per its own docstring) -- this is NOT a raw numpy call.
    import anionpy
    result = anionpy.intersect1d(a, b, assume_unique=assume_unique, return_indices=return_indices)
    return result if return_indices else (result,)


def _setdiff1d_cases():
    out = []
    for label, a, b in _set_pairs():
        for au in (False, True):
            out.append((f"{label}/assume_unique={au}", (a, b), {"assume_unique": au}))
    for label, a, b in _form_axis_binary_pairs():
        out.append((f"{label}/assume_unique=False", (a, b), {}))
    return out


def _setxor1d_cases():
    out = []
    for label, a, b in _set_pairs():
        for au in (False, True):
            out.append((f"{label}/assume_unique={au}", (a, b), {"assume_unique": au}))
    for label, a, b in _form_axis_binary_pairs():
        out.append((f"{label}/assume_unique=False", (a, b), {}))
    return out


def _union1d_cases():
    out = [(label, (a, b), {}) for label, a, b in _set_pairs()]
    out += [(label, (a, b), {}) for label, a, b in _form_axis_binary_pairs()]
    out += _union1d_weak_scalar_cases()
    return out


def _union1d_weak_scalar_cases():
    """Guard for the PHANTOM fixed 2026-08-04: a WEAK scalar operand.

    Every pair above is (array, array), which is exactly why `union1d` sat
    declared "exact" while giving a bare Python scalar a fixed strong dtype.
    numpy's `union1d` is literally `unique(np.concatenate((ar1, ar2),
    axis=None))`, so it inherits `concatenate`'s NEP 50 weak promotion:
    `np.union1d(np.array([1,0,2], np.int8), 1)` is int8, anionpy gave int64.
    56 of a 168-cell sweep diverged, on DTYPE only -- values agreed.

    Its three siblings are deliberately NOT given this grid, because they do
    not share the behaviour and adding it would encode a false rule:
    `intersect1d`, `setdiff1d` and `setxor1d` each `np.asanyarray` their
    inputs first, which makes a bare Python int a strong int64 before any
    promotion. All three measured clean on the same sweep. `union1d` is the
    only one that hands the raw operand to `concatenate`.

    The out-of-range integers are the load-bearing rows. Concatenation's
    overflow behaviour is a WRAPPING cast -- a third mode, distinct from the
    `OverflowError` of plain arithmetic and from the value-widening of the
    comparison ops -- and only the exact wrapped values pin it:

        np.union1d(np.array([1,0,2], np.int8),  300)   -> [ 0, 1, 2, 44] int8
        np.union1d(np.array([1,0,2], np.uint8), -7)    -> [ 0, 1, 2,249] uint8
        np.union1d(np.array([1,0,2], np.int8),  2**63) -> [ 0, 1, 2]     int8

    44 is `300 as i8`; 249 is `-7 as u8`; `2**63 as i8` is 0, which `unique`
    then merges into the existing 0, shortening the result. A fix that
    raised, or one that widened the dtype, fails all three. `10**100` is
    included because it IS still an error, just numpy's generic
    `OverflowError: Python int too large to convert to C long` rather than a
    bounds complaint.

    Both operand positions are covered: `np.union1d(300, int8_arr)` is the
    same int8 result, and a one-sided fix would still pass a one-sided grid.
    """
    dtypes = ["bool", "int8", "uint8", "int16", "uint16", "int32", "int64",
              "uint64", "float16", "float32", "float64", "complex64",
              "complex128"]
    scalars = [
        ("py_bool", True), ("py_int", 1), ("py_int_big", 300),
        ("py_int_neg", -7), ("py_int_huge", 2**63), ("py_int_vast", 10**100),
        ("py_float", 1.5), ("py_float_big", 1e20), ("py_nan", float("nan")),
        ("py_complex", 1 + 2j),
        ("np_bool", np.bool_(1)), ("np_int8", np.int8(1)),
        ("np_int64", np.int64(1)), ("np_uint8", np.uint8(1)),
        ("np_float16", np.float16(1.5)), ("np_float32", np.float32(1.5)),
        ("np_float64", np.float64(1.5)), ("np_complex64", np.complex64(1 + 2j)),
        ("arr_int8", np.array([1, 9], np.int8)),
        ("arr_float16", np.array([1.5], np.float16)),
        ("arr_float64", np.array([1.5])),
        ("arr_2d", np.array([[1, 9]], np.int8)),
        ("arr_0d", np.array(2, np.int8)),
        ("arr_empty", np.array([], np.int8)),
        ("list", [1, 2]), ("tuple", (1, 2)),
    ]
    out = []
    for name in dtypes:
        base = np.array([1, 0, 2], dtype=name)
        for slabel, s in scalars:
            out.append((f"weak/{name}/{slabel}/right", (base, s), {}))
            out.append((f"weak/{name}/{slabel}/left", (s, base), {}))
    return out


def _isin_cases():
    out = []
    for label, a, b in _set_pairs():
        for inv in (False, True):
            out.append((f"{label}/invert={inv}", (a, b), {"invert": inv}))
    for label, a, b in _form_axis_binary_pairs():
        out.append((f"{label}/invert=False", (a, b), {}))
    # CLASS C (kind= never validated): withdrawn in fdcdfb6 because
    # `np.isin(a, b, kind='bogus')` raises `ValueError: Invalid kind:
    # 'bogus'. Please use None, 'sort' or 'table'.` in real numpy but anionpy
    # ran happily and returned a result -- 'sort'/'table' themselves
    # already produced correct results (core_setops::isin's single
    # algorithm matches numpy's output regardless of which internal
    # algorithm numpy would have picked), only the upfront validation was
    # missing. Fixed in ionp-py/src/setops.rs's `isin` wrapper.
    label0, a0, b0 = _set_pairs()[0]
    for kind in (None, "sort", "table"):
        out.append((f"{label0}/kind={kind!r}", (a0, b0), {"kind": kind}))
    out.append((f"{label0}/kind_invalid", (a0, b0), {"kind": "bogus"}))
    out.append((f"{label0}/kind_invalid_empty", (a0, b0), {"kind": ""}))
    return out


# ---------------------------------------------------------------------------
# unique family. `unique_counts`/`unique_inverse`/`unique_all` (Array API
# functions) are always tuple-returning -> multi_output=True. `unique`
# itself (the classic numpy function) returns a PLAIN array when no
# return_index/return_inverse/return_counts flag is set, and a TUPLE when
# any is -- a fixed ItemSpec can't switch `multi_output` per-case, so this
# file declares `unique` against its plain-array-returning call forms only
# (bare call, plus equal_nan=True/False) rather than mixing return shapes
# under one comparator. The return_index=True/return_inverse=True/
# return_counts=True combination is NOT an untested code path: it is
# exercised together (values+index+inverse+counts, real numpy's default
# legacy-signature equal_nan=True) by `unique_all`'s own coverage below,
# which shares `unique_general`'s Rust implementation
# (`ionp-core/src/setops.rs`) with `unique`'s flag-combination path --
# verified directly (probe script's `unique_all4` case, 84 comparisons,
# 0 mismatches) that combination.
# ---------------------------------------------------------------------------

def _unique_bare_cases():
    out = []
    for c in corpus.unary_corpus():
        out.append((f"{c.label}/bare", (c.value,), {}))
        out.append((f"{c.label}/equal_nan_true", (c.value,), {"equal_nan": True}))
        out.append((f"{c.label}/equal_nan_false", (c.value,), {"equal_nan": False}))
        # `sorted=` (numpy 2.x keyword): accepted-and-ignored on anionpy's side
        # (see ionp-py/src/setops.rs's doc comment) -- only exercised here
        # against `sorted=True` (the only value verified byte-exact; see
        # this file's module docstring for why `sorted=False`'s genuinely
        # different, order-unspecified-for-some-dtypes behavior is the same
        # landmine `unique_values` stays undeclared for).
        out.append((f"{c.label}/sorted_true", (c.value,), {"sorted": True}))
    # FORM axis (2026-08-02, CLASS A closure task): `unique` is kind="custom"
    # -- a raw Python list/tuple/range/scalar reaches `anionpy.unique()`
    # unconverted (see this file's module-level FORM-axis note above
    # `_form_axis_binary_pairs`), so `anionpy.unique([3, 1, 2, 1])` is a
    # genuinely different, previously-untested code path from
    # `anionpy.unique(np.array([3, 1, 2, 1]))`.
    for c in corpus.form_axis_cases(sample_stride=8):
        out.append((f"{c.label}/form", (c.value,), {}))
    return out


def _unique_axis_cases():
    """`unique(ar, axis=...)` -- both the `ndim<=1` (validated-but-ignored,
    byte-identical to `axis=None` per real numpy's own dispatcher hardcoding
    `axis=None` whenever `ar.ndim <= 1`) and the genuinely different
    `ndim>=2` moveaxis + structured/void-row-dedup path. Covers axis=0/1/-1,
    negative axes, empty arrays, 0-d/2-d inputs, and duplicate rows (all of
    which are int arrays with deliberately repeated rows/columns to
    exercise the row-dedup grouping, not just distinct-row inputs).
    """
    out = []
    for c in corpus.unary_corpus():
        ndim = c.value.ndim
        if ndim == 0:
            # `axis=0` on a 0-d array must raise (AxisError) -- numpy's own
            # `normalize_axis_index`/`normalize_axis_tuple` validation runs
            # even though `ndim<=1` later ignores the axis VALUE.
            out.append((f"{c.label}/axis=0", (c.value,), {"axis": 0}))
            continue
        axes = sorted({0, ndim - 1, -1, -ndim})
        for ax in axes:
            out.append((f"{c.label}/axis={ax}", (c.value,), {"axis": ax}))
        # out-of-range axis must raise the real (unprefixed) AxisError
        out.append((f"{c.label}/axis_oob", (c.value,), {"axis": ndim}))
        out.append((f"{c.label}/axis_oob_neg", (c.value,), {"axis": -ndim - 1}))

    # Duplicate-row / duplicate-column 2-D int and float cases, explicitly
    # constructed (unary_corpus() is not guaranteed to contain exact-
    # duplicate rows for every dtype) -- these are the cases that actually
    # exercise unique_axis's row-grouping logic, not just its plumbing.
    dup_arrays = [
        ("dup_rows_int", np.array([[1, 2], [1, 2], [3, 4], [1, 2]], dtype=np.int32)),
        ("dup_cols_int", np.array([[1, 1, 3], [2, 2, 4]], dtype=np.int32)),
        ("dup_rows_float_nan", np.array([[1.0, np.nan], [1.0, np.nan], [2.0, 3.0]])),
        ("dup_rows_signed_zero", np.array([[-0.0, 1.0], [0.0, 1.0], [2.0, 2.0]])),
        ("empty_2d", np.zeros((0, 3), dtype=np.int32)),
        ("empty_axis1", np.zeros((3, 0), dtype=np.int32)),
    ]
    for label, a in dup_arrays:
        for ax in (0, 1, -1, -2):
            for eq in (True, False):
                out.append((f"{label}/axis={ax}/equal_nan={eq}", (a,), {"axis": ax, "equal_nan": eq}))
            out.append((f"{label}/axis={ax}/sorted_true", (a,), {"axis": ax, "sorted": True}))
    return out


def _unique_tuple_cases():
    out = [(c.label, (c.value,), {}) for c in corpus.unary_corpus()]
    # FORM axis (2026-08-02, CLASS A closure task): backs unique_counts/
    # unique_inverse/unique_all -- same rationale as _unique_bare_cases above.
    out += [(f"{c.label}/form", (c.value,), {}) for c in corpus.form_axis_cases(sample_stride=8)]
    return out


# ---------------------------------------------------------------------------
# diff / ediff1d / trim_zeros
# ---------------------------------------------------------------------------

def _diff_cases():
    out = []
    for c in corpus.unary_corpus():
        if c.value.ndim == 0:
            continue
        for n in (0, 1, 2, 3):
            axes = [0]
            if c.value.ndim > 1:
                axes.append(1)
            axes.append(-1)
            for axis in axes:
                out.append((f"{c.label}/n={n}/axis={axis}", (c.value,), {"n": n, "axis": axis}))
    # prepend/append, and the bool-must-raise case
    for dt in ("int32", "float64", "complex128"):
        a = np.array([1, 2, 4, 7, 0], dtype=dt)
        pre = np.array([9], dtype=dt)
        app = np.array([9, 10], dtype=dt)
        out.append((f"prepend/{dt}", (a,), {"prepend": pre}))
        out.append((f"append/{dt}", (a,), {"append": app}))
        out.append((f"prepend_append/{dt}", (a,), {"prepend": pre, "append": app}))
    out.append(("bool_raises", (np.array([True, False, True]),), {}))

    # -----------------------------------------------------------------
    # prepend/append regression coverage added 2026-08-02: before this,
    # `anionpy.diff(a, prepend=0)` panicked (Rust `PanicException`, indexing
    # `shape[axis]` on a 0-d array inside `concat_along_axis`) and
    # `anionpy.diff(a, append=25)` raised the wrong exception type
    # ("all input arrays must have the same number of dimensions" instead
    # of numpy's real byte-exact message) -- both because a scalar
    # prepend/append was never broadcast to length-1-along-axis before
    # hitting the shared concatenate path the way real numpy's own
    # `diff()` does (`prepend = np.broadcast_to(prepend, shape)` when
    # `prepend.ndim == 0`, read from
    # `numpy/lib/_function_base_impl.py`). Fixed in
    # `ionp-core/src/setops.rs` (`diff_broadcast_operand` +
    # `concat_along_axis` hardening). Confirmed FAILING against the
    # pre-fix build (`PanicException`/wrong-ValueError) and PASSING
    # against the post-fix build for every case below, plus manually
    # verified out-of-corpus against live numpy 2.5.1.
    # -----------------------------------------------------------------

    # Bare scalar prepend/append (the originally reported repro).
    out.append(("prepend_scalar/int", (np.array([1, 4, 9, 16]),), {"prepend": 0}))
    out.append(("append_scalar/int", (np.array([1, 4, 9, 16]),), {"append": 25}))
    out.append((
        "prepend_append_scalar/n=2/int",
        (np.array([1, 4, 9, 16]),),
        {"n": 2, "prepend": 0, "append": 25},
    ))
    # Scalar prepend/append that forces ordinary dtype promotion through
    # the shared concatenate path (int32+float scalar -> float64;
    # uint8+out-of-range-int scalar -> platform-default-int-then-promoted).
    out.append((
        "prepend_scalar/int32_float_promote",
        (np.array([1, 4, 9, 16], dtype=np.int32),),
        {"prepend": 1.5},
    ))
    out.append((
        "append_scalar/uint8_oob_promote",
        (np.array([1, 4, 9, 16], dtype=np.uint8),),
        {"append": 300},
    ))
    # Scalar prepend/append on bool and complex dtypes -- exercises the
    # 0-d-broadcast path against dtypes not covered by the plain
    # prepend/append block above.
    out.append((
        "prepend_scalar/bool",
        (np.array([True, False, True]),),
        {"prepend": True},
    ))
    out.append((
        "prepend_scalar/complex",
        (np.array([1 + 2j, 3 + 4j, 5 + 6j]),),
        {"prepend": 0},
    ))
    # n=0 short-circuits before prepend/append (and even before shape
    # validation) is ever consulted -- must return `a` completely
    # unprocessed, even with a wildly incompatible prepend shape.
    out.append((
        "n=0_incompatible_prepend",
        (np.array([1, 2, 3]),),
        {"n": 0, "prepend": np.zeros((2, 2))},
    ))
    # Empty array with a scalar prepend.
    out.append((
        "empty_prepend_scalar",
        (np.array([], dtype=np.float64),),
        {"prepend": 1},
    ))
    # 2-D, non-default and negative axis, scalar prepend.
    _a2 = np.array([[1, 2, 4, 7], [10, 20, 40, 70]])
    out.append(("2d_prepend_scalar/axis=1", (_a2,), {"axis": 1, "prepend": 0}))
    out.append(("2d_prepend_scalar/axis=-1", (_a2,), {"axis": -1, "prepend": 0}))
    # Array-valued prepend/append of matching rank (broadcast-eligible
    # only for true 0-d operands; a matching-rank array must pass through
    # to concatenate unchanged and just concatenate normally).
    out.append((
        "2d_append_array_matching_rank/axis=0",
        (_a2,), {"axis": 0, "append": np.array([[1, 1, 1, 1]])},
    ))
    # Mismatched-rank append (1-D operand against a 2-D array): must raise
    # numpy's exact "all the input arrays must have same number of
    # dimensions" ValueError, not the old wrong message and not a panic.
    out.append((
        "2d_append_rank_mismatch/axis=1",
        (_a2,), {"axis": 1, "append": np.array([1, 1])},
    ))
    # Matching rank but mismatched non-axis-dimension size: must raise
    # numpy's exact "all the input array dimensions except for the
    # concatenation axis must match exactly" ValueError.
    out.append((
        "2d_append_shape_mismatch/axis=1",
        (_a2,), {"axis": 1, "append": np.zeros((3, 1))},
    ))
    # Out-of-range axis with a prepend given -- must raise the real
    # (unprefixed) AxisError referencing `a`'s own ndim, computed before
    # prepend/append are ever touched.
    out.append((
        "axis_oob_with_prepend",
        (_a2,), {"axis": 5, "prepend": 0},
    ))

    # -----------------------------------------------------------------
    # Array-like (non-ndarray) prepend/append coverage added 2026-08-02:
    # before this, `anionpy.diff(a, prepend=[1])` raised `TypeError: ufunc
    # operand must be an anionpy.ndarray` -- real numpy coerces any
    # array-like via `np.asarray` before concatenating (Python scalars
    # already worked here, since `extract_array` special-cases bare
    # bool/int/float/complex, but a bare Python list/tuple hit neither
    # that scalar branch nor the `hasattr("dtype")` numpy-object branch).
    # Fixed in `ionp-py/src/setops.rs` (`extract_array_like`, falling
    # back to the same nested-list/tuple ingestion `anionpy.array()` itself
    # uses when `extract_array` rejects a `list`/`tuple`). These cases
    # pass RAW Python lists/tuples (not `np.array(...)`) as the
    # `prepend=`/`append=` kwarg value -- deliberately unconverted by the
    # harness (`make_ionp_array_converter` only rewrites `np.ndarray`
    # instances, on purpose; see its own doc comment), so both the numpy
    # side and the anionpy side receive the identical Python list/tuple
    # object and must each do their own array-like coercion.
    # -----------------------------------------------------------------
    out.append(("prepend_list/int", (np.array([1, 2, 3, 4]),), {"prepend": [1]}))
    out.append(("append_list/int", (np.array([1, 2, 3, 4]),), {"append": [9]}))
    out.append(("prepend_list_len2/int", (np.array([1, 2, 3, 4]),), {"prepend": [1, 2]}))
    out.append(("prepend_tuple/int", (np.array([1, 2, 3, 4]),), {"prepend": (1,)}))
    out.append(("prepend_tuple_len2/int", (np.array([1, 2, 3, 4]),), {"prepend": (1, 2)}))
    out.append(("prepend_empty_list/int", (np.array([1, 2, 3, 4]),), {"prepend": []}))
    out.append(("prepend_bool_list/uint8", (np.array([1, 2, 3, 4], dtype=np.uint8),), {"prepend": [True]}))
    out.append(("prepend_float_list/int32", (np.array([1, 2, 3, 4], dtype=np.int32),), {"prepend": [1.5]}))
    # NOTE: a Python list of COMPLEX literals (e.g. `[0j]`) is out of
    # scope here -- `anionpy.array()`'s own nested-list ingestion
    # (`flatten_nested`/`LeafKind` in `ionp-py/src/lib.rs`, off-limits
    # for this task) only classifies bool/int/float leaves, so
    # `anionpy.array([0j])` already raises today independent of diff/
    # ediff1d. A plain real-valued list against a complex `a` (numpy
    # promotes it through the ordinary concatenate dtype rule) IS in
    # scope and covered here instead.
    out.append((
        "prepend_append_list/complex128",
        (np.array([1 + 2j, 3 + 4j, 5 + 6j]),),
        {"prepend": [0], "append": [1, 2]},
    ))
    # Nested list -> higher-rank prepend against a 1-D `a`: must raise
    # numpy's exact same-ndim ValueError (matches the already-covered
    # `2d_append_rank_mismatch` case's message, now reached via a raw
    # list instead of a pre-built `np.ndarray`).
    out.append((
        "prepend_nested_list_rank_mismatch",
        (np.array([1, 2, 3, 4]),), {"prepend": [[1, 2]]},
    ))
    # Nested list matching a 2-D `a`'s rank, concatenated along axis=0.
    out.append((
        "2d_append_nested_list/axis=0",
        (_a2,), {"axis": 0, "append": [[1, 1, 1, 1]]},
    ))
    # n=0 short-circuit must still ignore a raw-list prepend entirely,
    # same as the already-covered `n=0_incompatible_prepend` ndarray case.
    out.append((
        "n=0_incompatible_prepend_list",
        (np.array([1, 2, 3]),), {"n": 0, "prepend": [[9, 9], [9, 9]]},
    ))

    # -----------------------------------------------------------------
    # Omitted-vs-explicit-`None` coverage added 2026-08-02: real numpy's
    # true default for `prepend`/`append` is a private `np._NoValue`
    # sentinel (NOT `None`), so `np.diff(a, prepend=None)` is a genuinely
    # different call from omitting `prepend` -- it builds a mixed
    # int/`NoneType` object array via `np.asanyarray(None)` that then
    # crashes in the elementwise Python `-` inside `np.concatenate`/the
    # diff loop, raising `TypeError: unsupported operand type(s) for -:
    # '<dtype-kind>' and 'NoneType'` (append-side: operand order flips).
    # `n`'s real default IS the concrete int `1`, but `n=None` explicitly
    # still fails differently and later than an omitted `n` would, since
    # numpy compares the actual passed object (`n < 0`) rather than
    # re-deriving a default: `TypeError: '<' not supported between
    # instances of 'NoneType' and 'int'`.
    #
    # Before this fix, `anionpy.diff`'s `prepend`/`append`/`n` were plain
    # typed `Option<&Bound<PyAny>>`/`usize` pyo3 parameters -- pyo3's
    # blanket `Option<T>: FromPyObject` impl collapses "keyword omitted"
    # and "keyword explicitly passed `None`" to the identical Rust `None`,
    # so `anionpy.diff(a, prepend=None)` silently behaved exactly like
    # omitting `prepend` (`array([1, 1, 1])`) instead of raising -- a
    # genuine, confirmed divergence from live numpy 2.5.1. Fixed in
    # `ionp-py/src/setops.rs` by rewriting `diff` to take raw
    # `*args`/`**kwargs` (the same technique already used by
    # `Ufunc::__call__`'s `out=` keyword in `lib.rs` for the identical
    # absent-vs-present-holding-None problem) and `ionp-core/src/setops.rs`'s
    # new `DiffOperand` enum (`Omitted`/`ExplicitNone`/`Value`) to carry
    # the distinction through to the actual error. `ediff1d`'s
    # `to_begin=`/`to_end=` are DELIBERATELY NOT touched here -- their
    # real, documented numpy default genuinely is `None`, so
    # `to_begin=None`/omitted are legitimately identical there (verified
    # live, see `_ediff1d_cases`'s own coverage above).
    # -----------------------------------------------------------------
    for dt in ("int32", "uint8", "float64", "complex128", "bool"):
        if dt == "bool":
            a = np.array([True, False, True, True])
        else:
            a = np.array([1, 2, 4, 7], dtype=dt)
        out.append((f"prepend_none/{dt}", (a,), {"prepend": None}))
        out.append((f"append_none/{dt}", (a,), {"append": None}))
        out.append((f"prepend_append_none/{dt}", (a,), {"prepend": None, "append": None}))
    out.append(("n_none", (np.array([1, 2, 4, 7]),), {"n": None}))
    out.append(("n_negative", (np.array([1, 2, 4, 7]),), {"n": -1}))
    out.append(("n_negative_large", (np.array([1, 2, 4, 7]),), {"n": -5}))
    # n=0 must still short-circuit around explicit-None prepend/append,
    # exactly like it already does for an incompatible-shape/list prepend
    # above (`n=0_incompatible_prepend`/`n=0_incompatible_prepend_list`).
    out.append((
        "n=0_prepend_none",
        (np.array([1, 2, 3]),), {"n": 0, "prepend": None},
    ))
    out.append((
        "n=0_append_none",
        (np.array([1, 2, 3]),), {"n": 0, "append": None},
    ))

    # -----------------------------------------------------------------
    # n=0/axis CROSSING coverage added 2026-08-02: the coordinator's own
    # 20,930-case grid caught what a per-axis (n varied separately from
    # axis, axis varied separately from n) grid could not -- `n=0`'s
    # short-circuit must run BEFORE axis is even inspected, so
    # `np.diff(a, n=0, axis=None)` returns `a` unchanged (real numpy never
    # normalizes `axis=None` into a concrete axis at all when n==0) where
    # `anionpy.diff` used to raise `TypeError: 'NoneType' object cannot be
    # interpreted as an integer` -- the wrapper in `ionp-py/src/setops.rs`
    # was extracting `axis` into a concrete `isize` unconditionally,
    # before ever calling into `core_setops::diff` (which already had the
    # right internal ordering, but that's no help if the caller blows up
    # first). Fixed by moving an identical `n==0` short-circuit into the
    # wrapper itself, ahead of resolving `axis`/`prepend`/`append` at all.
    # Covers axis=None, an out-of-range axis, AND explicit-None
    # prepend/append together with a bad axis, all under n=0 -- every one
    # of these must return `a` untouched, never raising or even looking at
    # the invalid keyword.
    # -----------------------------------------------------------------
    out.append(("n=0_axis_none", (np.array([1, 2, 3]),), {"n": 0, "axis": None}))
    out.append(("n=0_axis_oob", (np.array([1, 2, 3]),), {"n": 0, "axis": 5}))
    out.append(("n=0_axis_oob_neg", (np.array([1, 2, 3]),), {"n": 0, "axis": -5}))
    out.append((
        "n=0_axis_none_prepend_none",
        (np.array([1, 2, 3]),), {"n": 0, "axis": None, "prepend": None},
    ))
    out.append((
        "n=0_axis_none_append_none",
        (np.array([1, 2, 3]),), {"n": 0, "axis": None, "append": None},
    ))
    out.append((
        "n=0_axis_oob_prepend_append_none",
        (np.array([1, 2, 3]),),
        {"n": 0, "axis": 5, "prepend": None, "append": None},
    ))
    # Same crossing on a 2-D array and an explicit-in-range axis, to make
    # sure the fix didn't accidentally break the ordinary (non-crossing)
    # n=0 short-circuit for a valid axis -- must still return `a`
    # unchanged.
    out.append((
        "n=0_axis_valid_2d",
        (_a2,), {"n": 0, "axis": 1, "prepend": None, "append": [9, 9]},
    ))
    # Empty array, n=0, axis=None crossing -- exercises the short-circuit
    # on the zero-element path too (distinct from the explicit-None-on-
    # empty object-dtype gap, which only applies when n != 0 and the
    # subtraction path is actually reached; see KNOWN-DIFFERENCES.md).
    out.append((
        "n=0_axis_none_empty",
        (np.array([], dtype=np.float64),), {"n": 0, "axis": None, "prepend": None},
    ))

    # -----------------------------------------------------------------
    # Empty-diff-axis + BOTH prepend/append present coverage, added
    # 2026-08-02 per the coordinator's 18,480-case grid: 198 mismatches,
    # all "prepend=None AND append=None together on an empty array" --
    # narrower than the KNOWN-DIFFERENCES.md entry previously claimed.
    # When the diff axis has length 0 but EVERY OTHER axis is non-empty
    # and both prepend and append were actually passed (None or not),
    # `combined` gets exactly 2 real elements along the diff axis and a
    # genuine subtraction executes -- fully representable, no object
    # dtype needed. These cases must now PASS (previously they were
    # wrongly excluded as part of the whole empty+None blocker).
    # -----------------------------------------------------------------
    empty1d = np.array([], dtype=np.int64)
    out.append(("empty1d_both_none", (empty1d,), {"prepend": None, "append": None}))
    out.append(("empty1d_prepend_none_append_real", (empty1d,), {"prepend": None, "append": np.array([9])}))
    out.append(("empty1d_prepend_real_append_none", (empty1d,), {"prepend": np.array([9]), "append": None}))
    out.append(("empty1d_both_real_no_none", (empty1d,), {"prepend": np.array([9]), "append": np.array([8])}))

    # Same crossing on a 2-D array where the DIFF axis is length-0 but the
    # OTHER axis is non-empty (so per-row subtraction genuinely runs, one
    # real pair per row) -- as opposed to the orthogonal case where the
    # OTHER axis is length-0 (still vacuous/blocked, deliberately not
    # added here; see KNOWN-DIFFERENCES.md).
    a_4_0 = np.zeros((4, 0), dtype=np.int64)
    out.append(("shape4x0_axis1_both_none", (a_4_0,), {"axis": 1, "prepend": None, "append": None}))
    out.append((
        "shape4x0_axis1_prepend_none_append_real",
        (a_4_0,), {"axis": 1, "prepend": None, "append": np.zeros((4, 1), dtype=np.int64)},
    ))
    a_0_4 = np.zeros((0, 4), dtype=np.int64)
    out.append(("shape0x4_axis0_both_none", (a_0_4,), {"axis": 0, "prepend": None, "append": None}))
    out.append((
        "shape0x4_axis0_prepend_real_append_none",
        (a_0_4,), {"axis": 0, "prepend": np.zeros((1, 4), dtype=np.int64), "append": None},
    ))

    # 3-D crossing: diff axis (axis=1) is length-0, the OTHER two axes
    # (0 and 2) are both non-empty -- same "both present -> real
    # subtract" regime, one more dimension up.
    a_3_0_2 = np.zeros((3, 0, 2), dtype=np.int64)
    out.append(("shape3x0x2_axis1_both_none", (a_3_0_2,), {"axis": 1, "prepend": None, "append": None}))

    # FORM axis (2026-08-02, CLASS A closure task) on `a` ITSELF (as
    # opposed to the `prepend=`/`append=` list/tuple coverage added above,
    # which is a separate, already-covered axis): `anionpy.diff([1, 4, 9, 16])`
    # with no ndarray anywhere in the call.
    for c in corpus.form_axis_cases(sample_stride=8):
        out.append((f"{c.label}/form", (c.value,), {}))
    return out


def _ediff1d_cases():
    out = []
    for c in corpus.unary_corpus():
        out.append((c.label, (c.value,), {}))
    for dt in ("int32", "float64"):
        a = np.array([1, 5, 2, 8], dtype=dt)
        tb = np.array([-1, -2], dtype=dt)
        te = np.array([100], dtype=dt)
        out.append((f"begin_end/{dt}", (a,), {"to_begin": tb, "to_end": te}))

    # -----------------------------------------------------------------
    # Regression coverage added 2026-08-02: found during the required
    # sweep for other reachable callers of the rank-0/dtype-homogeneity
    # bug in `concat_along_axis` -- `ediff1d` never pre-cast `to_begin`/
    # `to_end` to a common dtype before concatenating (unlike
    # intersect1d/union1d/setdiff1d/setxor1d, which all call
    # `common_dtype_cast` first), so a dtype MISMATCH between `to_begin`
    # and `ary` hit `concat_along_axis`'s `do_concat!` `unreachable!()`
    # arm -- a second, distinct Rust panic beyond the one named in this
    # task. Fixed by giving `ediff1d` its own (numpy-real, genuinely
    # different from `diff`'s) same-kind-cast validation rule, verified
    # live: `np.can_cast(to_begin.dtype, ary.dtype, casting="same_kind")`
    # must hold or numpy raises
    # "dtype of `to_begin`/`to_end` must be compatible with input `ary`
    # under the `same_kind` rule." -- checked BEFORE the bool-ary-
    # rejection check (verified ordering: `to_begin=1.5` against a bool
    # `ary` raises the to_begin TypeError, not the boolean-subtract one;
    # `to_begin=True`, same-kind-valid, does then hit the boolean-subtract
    # TypeError).
    # -----------------------------------------------------------------
    out.append((
        "to_begin_same_kind_narrower_dtype",
        (np.array([1, 4, 9, 16], dtype=np.int32),),
        {"to_begin": np.array([2], dtype=np.int8)},
    ))
    out.append((
        "to_end_same_kind_narrower_dtype",
        (np.array([1, 4, 9, 16], dtype=np.float32),),
        {"to_end": np.array([2], dtype=np.int8)},
    ))
    out.append((
        "to_begin_not_same_kind",
        (np.array([1, 4, 9, 16], dtype=np.int32),),
        {"to_begin": 1.5},
    ))
    out.append((
        "bool_ary_to_begin_not_same_kind",
        (np.array([True, False, True]),),
        {"to_begin": 1.5},
    ))
    out.append((
        "bool_ary_to_begin_same_kind_then_bool_subtract_raises",
        (np.array([True, False, True]),),
        {"to_begin": True},
    ))

    # -----------------------------------------------------------------
    # Array-like (non-ndarray) to_begin/to_end coverage added 2026-08-02,
    # same defect and same fix as `diff`'s prepend/append list block
    # above (`extract_array_like` in `ionp-py/src/setops.rs`) -- raw
    # Python lists/tuples, deliberately left unconverted by the harness
    # so both numpy and anionpy do their own array-like coercion.
    # -----------------------------------------------------------------
    out.append(("to_begin_list/int", (np.array([1, 5, 2, 8]),), {"to_begin": [-1]}))
    out.append(("to_end_list/int", (np.array([1, 5, 2, 8]),), {"to_end": [100]}))
    out.append(("to_begin_tuple/int", (np.array([1, 5, 2, 8]),), {"to_begin": (-1, -2)}))
    out.append((
        "begin_end_list/float64",
        (np.array([1.0, 5.0, 2.0, 8.0]),),
        {"to_begin": [-1.5], "to_end": (100.0, 200.0)},
    ))
    out.append(("to_begin_empty_list/int", (np.array([1, 5, 2, 8]),), {"to_begin": []}))
    out.append((
        "to_begin_list_not_same_kind",
        (np.array([1, 4, 9, 16], dtype=np.int32),),
        {"to_begin": [1.5]},
    ))

    # FORM axis (2026-08-02, CLASS A closure task) on `ary` itself (as
    # opposed to the `to_begin=`/`to_end=` list/tuple coverage above, a
    # separate axis): `anionpy.ediff1d([1, 5, 2, 8])` with no ndarray anywhere.
    for c in corpus.form_axis_cases(sample_stride=8):
        out.append((f"{c.label}/form", (c.value,), {}))
    return out


def _trim_zeros_cases():
    arrays = [
        np.array([0, 0, 1, 2, 0, 3, 0, 0]),
        np.array([0, 0, 0]),
        np.array([1, 2, 3]),
        np.array([0.0, -0.0, 1.0, 0.0]),
        np.array([0 + 0j, 1 + 1j, 0 + 0j]),
        np.array([True, False, True, False]),
    ]
    out = []
    for a in arrays:
        for trim in ("fb", "f", "b", "FB", "", "bf"):
            out.append((f"{a.dtype}/trim={trim!r}", (a,), {"trim": trim}))
            out.append((f"{a.dtype}/trim={trim!r}/axis=None", (a,), {"trim": trim, "axis": None}))
            out.append((f"{a.dtype}/trim={trim!r}/axis=0", (a,), {"trim": trim, "axis": 0}))
            out.append((f"{a.dtype}/trim={trim!r}/axis=-1", (a,), {"trim": trim, "axis": -1}))

    # N-D axis coverage: bounding-box-over-whole-array, all-zero-forces-
    # empty-regardless-of-trim-flags, non-trimmed-axes-keep-full-extent,
    # negative axes, sequence-of-axes, duplicate-axis error, out-of-range
    # (prefixed) AxisError, and the empty-`axis=()`-is-identity case.
    nd_arrays = [
        ("nd_basic", np.array([[0, 0, 0], [0, 1, 0], [0, 2, 3], [0, 0, 0]])),
        ("nd_all_zero", np.zeros((3, 4), dtype=np.int32)),
        ("nd_float_signed_zero", np.array([[0.0, -0.0], [1.0, 0.0], [0.0, 0.0]])),
        ("nd_3d", np.arange(24).reshape(2, 3, 4) % 5 - 2),
        ("nd_0d", np.array(0)),
        ("nd_0d_nonzero", np.array(5)),
    ]
    for label, a in nd_arrays:
        ndim = a.ndim
        for trim in ("fb", "f", "b"):
            out.append((f"{label}/trim={trim!r}/axis=None", (a,), {"trim": trim, "axis": None}))
            out.append((f"{label}/trim={trim!r}/axis=empty", (a,), {"trim": trim, "axis": ()}))
            for ax in range(ndim):
                out.append((f"{label}/trim={trim!r}/axis={ax}", (a,), {"trim": trim, "axis": ax}))
                out.append((f"{label}/trim={trim!r}/axis={ax - ndim}", (a,), {"trim": trim, "axis": ax - ndim}))
            if ndim >= 2:
                out.append((f"{label}/trim={trim!r}/axis=seq", (a,), {"trim": trim, "axis": [0, 1]}))
                out.append((f"{label}/trim={trim!r}/axis=dup", (a,), {"trim": trim, "axis": [0, 0]}))
            out.append((f"{label}/trim={trim!r}/axis_oob", (a,), {"trim": trim, "axis": ndim}))
            out.append((f"{label}/trim={trim!r}/axis_oob_neg", (a,), {"trim": trim, "axis": -ndim - 1}))

    # FORM axis (2026-08-02, CLASS A closure task): `trim_zeros` is
    # kind="custom" -- `anionpy.trim_zeros([0, 0, 1, 2, 0])` with no ndarray.
    for c in corpus.form_axis_cases(sample_stride=8):
        out.append((f"{c.label}/form", (c.value,), {}))
    return out


def _unique_all_bare_and_axis_cases():
    return _unique_bare_cases() + _unique_axis_cases()


SETOPS_SPECS: dict[str, ItemSpec] = {
    "unique": ItemSpec(
        name="unique", kind="custom",
        custom_cases=_unique_all_bare_and_axis_cases,
        atol=0.0, rtol=0.0,
    ),
    "unique_counts": ItemSpec(
        name="unique_counts", kind="custom",
        custom_cases=_unique_tuple_cases,
        multi_output=True,
        atol=0.0, rtol=0.0,
    ),
    "unique_inverse": ItemSpec(
        name="unique_inverse", kind="custom",
        custom_cases=_unique_tuple_cases,
        multi_output=True,
        atol=0.0, rtol=0.0,
    ),
    "unique_all": ItemSpec(
        name="unique_all", kind="custom",
        custom_cases=_unique_tuple_cases,
        multi_output=True,
        atol=0.0, rtol=0.0,
    ),
    "diff": ItemSpec(
        name="diff", kind="custom",
        custom_cases=_diff_cases,
        atol=0.0, rtol=0.0,
    ),
    "ediff1d": ItemSpec(
        name="ediff1d", kind="custom",
        custom_cases=_ediff1d_cases,
        atol=0.0, rtol=0.0,
    ),
    "trim_zeros": ItemSpec(
        name="trim_zeros", kind="custom",
        custom_cases=_trim_zeros_cases,
        atol=0.0, rtol=0.0,
    ),
    "intersect1d": ItemSpec(
        name="intersect1d", kind="custom",
        custom_cases=_intersect1d_cases,
        numpy_adapter=_intersect1d_numpy_adapter,
        ionp_adapter=_intersect1d_ionp_adapter,
        multi_output=True,
        atol=0.0, rtol=0.0,
    ),
    "union1d": ItemSpec(
        name="union1d", kind="custom",
        custom_cases=_union1d_cases,
        atol=0.0, rtol=0.0,
    ),
    "setdiff1d": ItemSpec(
        name="setdiff1d", kind="custom",
        custom_cases=_setdiff1d_cases,
        atol=0.0, rtol=0.0,
    ),
    "setxor1d": ItemSpec(
        name="setxor1d", kind="custom",
        custom_cases=_setxor1d_cases,
        atol=0.0, rtol=0.0,
    ),
    "isin": ItemSpec(
        name="isin", kind="custom",
        custom_cases=_isin_cases,
        atol=0.0, rtol=0.0,
    ),
}


# ---------------------------------------------------------------------------
# unique_counts / unique_inverse / unique_all RESULT-CONTAINER identity
# (2026-08-03).
#
# Why these three probe items exist: the ordinary `unique_counts` item PASSED
# while anionpy returned a bare `tuple` and numpy returned a `UniqueCountsResult`
# namedtuple. Verified by mutation -- reverting the container to a plain tuple
# left all 212 cases green. The comparator unpacks a returned tuple and
# compares the arrays inside it, so the container is exactly the thing it
# cannot see, and the item was therefore undeclarable no matter how many
# array-valued cases it passed. A passing test is necessary, not sufficient.
#
# These items make the container the COMPARED VALUE. Each adapter encodes its
# OWN library's result -- class name plus field names -- as int64 character
# codes. There is no literal expectation anywhere in this file: numpy's side
# is derived live from numpy's object, anionpy's from anionpy's, and they agree only
# if anionpy genuinely reproduces the Array-API result type. An assertion against
# a hardcoded "UniqueCountsResult" string would have been a tautology dressed
# as a test.
#
# Field ORDER is encoded, not just membership: the fields are positional
# (`values, indices, inverse_indices, counts` for unique_all), so a container
# with the right names in the wrong order is a real defect and must not pass.
# ---------------------------------------------------------------------------

def _container_identity(result):
    """Encode a result container's observable identity as int64 char codes.

    `type(...).__name__` and `_fields` are both part of the documented,
    user-visible API surface -- `r.values` is how the Array-API spec says
    these are read. Missing `_fields` entirely (a bare tuple) encodes as the
    class name alone, which cannot collide with any namedtuple's encoding
    because the separator is always present for a namedtuple.
    """
    name = type(result).__name__
    fields = ",".join(getattr(result, "_fields", ()))
    return np.array([ord(c) for c in f"{name}|{fields}"], dtype=np.int64)


def _container_cases():
    return [
        ("ints_with_repeats", (np.array([1, 2, 2, 3]),), {}),
        ("all_unique", (np.array([5, 1, 9]),), {}),
        ("single", (np.array([7]),), {}),
        ("empty", (np.array([], dtype=np.int64),), {}),
        ("floats_with_nan", (np.array([1.0, np.nan, 1.0]),), {}),
        ("bools", (np.array([True, False, True]),), {}),
        ("two_d_is_flattened", (np.array([[1, 2], [2, 3]]),), {}),
    ]


for _fn in ("unique_counts", "unique_inverse", "unique_all"):
    SETOPS_SPECS[f"{_fn}_container_identity"] = ItemSpec(
        name=f"{_fn}_container_identity", kind="custom",
        custom_cases=_container_cases,
        numpy_adapter=(lambda f: (lambda x: _container_identity(getattr(np, f)(x))))(_fn),
        # `anionpy` imported inside the call, not at module scope: these case
        # modules are imported while registry.py is still assembling, and a
        # top-level anionpy import here would add an import-order dependency
        # that no other file in this directory has.
        ionp_adapter=(lambda f: (lambda x: _container_identity(
            getattr(__import__("anionpy"), f)(x))))(_fn),
    )
del _fn
