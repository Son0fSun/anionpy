"""anionpy's array-manipulation block differential registry entries:
concatenate/stack + the hstack/vstack/row_stack/dstack/column_stack
wrappers, flip/fliplr/flipud, roll, tile, repeat, broadcast_shapes/
broadcast_arrays, atleast_1d/2d/3d, and the diagonal family
(diag/diagonal/diagflat/tril/triu/trace).

NEW FILE (same collision-checked-merge pattern as creation_cases.py/
order_cases.py/reduction_cases.py/sort_cases.py -- builds a dict of
ItemSpecs, merged into registry.REGISTRY from the bottom of registry.py).
ionp-core/src/manip.rs and ionp-py/src/manip.rs are this same task's other
two files.

Everything here is kind="custom": every item takes a list/tuple of arrays,
an axis int, a shape tuple, or some mix thereof, never a bare f(array) or
f(array, array) call kind="unary"/"binary" assumes.

Scope note (deliberate, per this task's "twelve genuinely verified items
beat forty declared on hope" directive): the array-manipulation block has
roughly 40 numpy items in it; this file (and its ionp-core/ionp-py
siblings) implements and tests a 24-item subset chosen for being
mechanically related and independently verifiable in the time available,
PLUS (2026-08-01, first pass) split/array_split/hsplit/vsplit/dsplit,
insert/delete/append/resize, rot90, rollaxis -- 11 more items, 35 total --
PLUS (2026-08-01, second pass) the grid/index-construction family:
diag_indices, diag_indices_from, tril_indices, triu_indices, ix_, indices,
meshgrid, ravel_multi_index, unravel_index, fill_diagonal -- 10 more items,
45 total.
PLUS (2026-08-01, third pass) the Tier-1 gather/scatter family: take, put,
take_along_axis, put_along_axis, compress -- 5 more items, 50 total. `take`/
`compress` also cover `out=` (exact-shape match, unsafe dtype cast allowed,
same-identity return -- a bespoke contract distinct from the ufunc family's
`out=`, see `write_exact_shape_out` in ionp-core/src/manip.rs) via their
single adapter's `_out_shape`/`_out_dtype`/`_bad_out` sentinel kwargs.
NOT attempted (not "divergence found", simply not built): pad, block (both
require multi-mode support that would leave anionpy raising wrong errors if
only partially built), plus the Tier-2 gather/scatter family (choose, place,
select). See the task's final report for the full accounting.

split/array_split/hsplit/vsplit/dsplit return a plain Python `list` of
arrays (both from real numpy and from anionpy) -- `list` is not `tuple`, so
`harness.compare_multi_output`'s `isinstance(np_out, tuple)` gate would
reject every case outright if the raw resolved functions were used
directly with `multi_output=True`. Each of these five items therefore
declares a `numpy_adapter`/`ionp_adapter` pair that does nothing but
`tuple(...)` the real call's result (see `_split_family_adapters` below)
-- the underlying call and its exception behavior are completely
unchanged, only the container type wrapping the parts list becomes
tuple-comparable to satisfy the multi-output-aware harness path.

`atleast_1d`/`atleast_2d`/`atleast_3d`: real numpy's multi-array call form
returns a `tuple` of arrays (confirmed via `type(np.atleast_1d(a, b))`),
which anionpy's binding now also does (see manip.rs's `atleast_fn!` macro
comment) -- but the differential harness's `compare_values`/
`compare_multi_output` machinery grades a SINGLE numpy_path/ionp_path
resolution against ONE spec-wide `multi_output` flag, and this function's
return shape (bare array vs tuple) depends on the CALL, not the item, so
one spec can't honestly declare both. This file therefore only exercises
the single-array call form through the harness; the multi-array tuple
form was manually spot-checked in-session (`anionpy.atleast_1d(a, b)` against
`np.atleast_1d(a, b)`, byte-identical per-element, correct tuple return)
but is not part of the automated differential ledger for this item.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec
import corpus as _corpus_mod
from corpus import unary_corpus, ALL_DTYPES, SWEEP_DTYPES, _fill

_CORPUS = unary_corpus()


def _corpus_cases(label_suffix: str = "") -> list[tuple[str, tuple, dict]]:
    return [(c.label + label_suffix, (c.value,), {}) for c in _CORPUS]


def _form_cases(label_suffix: str = "", sample_stride: int = 8) -> list[tuple[str, tuple, dict]]:
    """FORM axis (2026-08-02, CLASS A closure task): every item in this file
    is kind="custom" and takes its array as a plain positional argument to a
    top-level `anionpy.<name>()` function, so a raw Python list/tuple/range/
    scalar genuinely reaches anionpy's own coercion code unconverted (see
    `corpus.form_axis_cases()`'s docstring, and the identical pattern
    already wired into reduction_cases.py/stats_cases.py/setops_cases.py).
    """
    return [
        (c.label + label_suffix, (c.value,), {})
        for c in _corpus_mod.form_axis_cases(sample_stride=sample_stride)
    ]


# ---------------------------------------------------------------------------
# concatenate / stack / hstack / vstack / row_stack / dstack / column_stack
# ---------------------------------------------------------------------------

def _stack_family_arrays():
    """A variety of same-shape array groups (2 or 3 arrays) across dtypes/
    ndims, for concatenate/stack/h/v/d/column-stack sweeps."""
    rng = np.random.default_rng(20260801)
    out = []
    for dtype in SWEEP_DTYPES:
        a = rng.standard_normal((3, 4)).astype(dtype) if np.dtype(dtype).kind in "fc" \
            else rng.integers(0, 50, size=(3, 4)).astype(dtype)
        b = rng.standard_normal((3, 4)).astype(dtype) if np.dtype(dtype).kind in "fc" \
            else rng.integers(0, 50, size=(3, 4)).astype(dtype)
        out.append((np.dtype(dtype).name, a, b))
    return out



# ---------------------------------------------------------------------------
# concatenate/stack/hstack/vstack `dtype=` x `casting=` crossed grid
#
# The corpus above (and the differential suite generally) varies dtype=,
# casting=, input dtype, and memory layout INDEPENDENTLY -- e.g. a single
# `dtype_override` case per function, a `casting=` axis never exercised at
# all previously. That is exactly why 2176/3456 mismatches across these
# four functions went undetected: an axis crossed only against itself, or
# never crossed at all, cannot see a bug that only appears when dtype= and
# casting= interact (DEFECT 2 in this task's report) or when casting= is
# 'no'/'equiv' specifically (DEFECT 1). This block is the crossed grid that
# DID find them: 6 input dtypes x 6 dtype= targets (including "omitted") x
# 6 casting= values (including "omitted") x 4 memory layouts = 864 cases
# per function, 3456 across all four -- reproducing the exact grid size
# from this task's own investigation. See manip.rs's `check_concat_casting`
# / `concat_target_dtype` for the fix these cases guard.
# ---------------------------------------------------------------------------

_GRID_INPUT_DTYPES = [np.bool_, np.int8, np.uint8, np.int32, np.float32, np.complex64]
_GRID_DTYPE_TARGETS = [None, "int8", "uint8", "int32", "float32", "complex128"]
_GRID_CASTING_VALUES = [None, "no", "equiv", "safe", "same_kind", "unsafe"]


def _grid_layout_pair(dtype, seed):
    """Two same-shape (3, 4) arrays of `dtype` for each of four memory
    layouts: C-contiguous, Fortran-order, sliced/non-contiguous (stride-2
    column slice of a wider array), and transposed (stride-swapped view of
    a (4, 3) base). Distinct `seed` per input dtype so different dtypes
    don't share byte patterns; deterministic across runs like the rest of
    this corpus."""
    rng = np.random.default_rng(seed)

    def fresh(shape):
        return _fill(rng, shape, dtype)

    out = {}
    out["c_contig"] = (fresh((3, 4)), fresh((3, 4)))
    out["fortran"] = (np.asfortranarray(fresh((3, 4))), np.asfortranarray(fresh((3, 4))))
    wide_a, wide_b = fresh((3, 8)), fresh((3, 8))
    out["sliced_noncontig"] = (wide_a[:, ::2], wide_b[:, ::2])
    tall_a, tall_b = fresh((4, 3)), fresh((4, 3))
    out["transposed"] = (tall_a.T, tall_b.T)
    return out


def _casting_dtype_grid_kwargs():
    """Every (dtype=, casting=) combination -- 6 targets (5 real dtype
    strings + "omit the kwarg entirely") x 6 casting values (the 5 legal
    strings + "omit the kwarg entirely") = 36 kwarg dicts."""
    out = []
    for target in _GRID_DTYPE_TARGETS:
        for casting in _GRID_CASTING_VALUES:
            kwargs = {}
            if target is not None:
                kwargs["dtype"] = target
            if casting is not None:
                kwargs["casting"] = casting
            label = f"target_{target or 'none'}/casting_{casting or 'default'}"
            out.append((label, kwargs))
    return out


_GRID_KWARGS = _casting_dtype_grid_kwargs()


def _casting_dtype_grid_cases(wrap):
    """`wrap(a, b) -> args`, adapting a same-shape array pair into whatever
    positional-argument shape the target function expects (a single-list
    positional arg for all four of concatenate/stack/hstack/vstack)."""
    out = []
    for i, dtype in enumerate(_GRID_INPUT_DTYPES):
        layouts = _grid_layout_pair(dtype, seed=20260802 + i)
        for layout_name, (a, b) in layouts.items():
            for kw_label, kwargs in _GRID_KWARGS:
                label = f"grid/{np.dtype(dtype).name}/{layout_name}/{kw_label}"
                out.append((label, wrap(a, b), dict(kwargs)))
    return out


def concatenate_cases():
    out = []
    groups = _stack_family_arrays()
    for name, a, b in groups:
        out.append((f"axis0/{name}", ([a, b],), {}))
        out.append((f"axis1/{name}", ([a, b],), {"axis": 1}))
        out.append((f"axis_neg1/{name}", ([a, b],), {"axis": -1}))
        out.append((f"three_arrays/{name}", ([a, b, a],), {"axis": 0}))
    # 1-D
    out.append(("1d_two", ([np.array([1, 2, 3]), np.array([4, 5])],), {}))
    out.append(("1d_empty_plus_nonempty", ([np.array([], dtype=np.int64), np.array([1, 2])],), {}))
    # axis=None flattens
    out.append(("axis_none_2d", ([np.arange(6).reshape(2, 3), np.arange(4).reshape(2, 2)],), {"axis": None}))
    # 3-D, negative axis
    out.append(("3d_axis2", ([np.arange(24).reshape(2, 3, 4), np.arange(24).reshape(2, 3, 4)],), {"axis": 2}))
    out.append(("3d_axis_neg2", ([np.arange(24).reshape(2, 3, 4), np.arange(24).reshape(2, 3, 4)],), {"axis": -2}))
    # mixed dtype promotion
    out.append(("promote_i32_f64", ([np.arange(4, dtype=np.int32), np.arange(4, dtype=np.float64)],), {}))
    out.append(("promote_bool_i8", ([np.array([True, False]), np.array([1, 2], dtype=np.int8)],), {}))
    out.append(("promote_f32_c64", ([np.arange(3, dtype=np.float32), np.arange(3, dtype=np.complex64)],), {}))
    # dtype= override
    out.append(("dtype_override", ([np.arange(4, dtype=np.int32), np.arange(4, dtype=np.int32)],), {"dtype": np.float64}))
    # foreign list-of-lists input
    out.append(("list_of_lists", ([[1, 2], [3, 4]],), {}))
    # error cases
    out.append(("mismatched_shapes_raises", ([np.zeros((2, 3)), np.zeros((2, 4))],), {"axis": 0}))
    out.append(("ndim_mismatch_raises", ([np.zeros((2, 3)), np.zeros((2, 3, 1))],), {}))
    out.append(("axis_out_of_range_raises", ([np.zeros((2, 3)), np.zeros((2, 3))],), {"axis": 5}))
    out.append(("empty_sequence_raises", ([],), {}))
    out.append(("zero_d_raises", ([np.array(5), np.array(6)],), {}))
    # DEFECT 3: a generator is a real Python iterable but NOT a `PySequence`
    # (no `__len__`/`__getitem__`-by-index protocol) -- numpy rejects it
    # with a function-specific TypeError (see manip.rs's
    # `CONCATENATE_NOT_A_SEQUENCE_MSG`/`STACK_NOT_A_SEQUENCE_MSG`); a plain
    # `try_iter()`-based ingestion would happily drain it instead. Both the
    # numpy call and the anionpy call below share this literal generator
    # object (see harness.py's `_freshen`, which passes non-ndarray/tuple/
    # list/dict args through unchanged) -- safe here because the rejection
    # must happen on the object's TYPE before either side ever iterates it.
    out.append(("generator_rejected_raises", ((np.array([1, 2]) for _ in range(2)),), {}))
    out.extend(_casting_dtype_grid_cases(lambda a, b: ([a, b],)))
    # FORM axis (2026-08-02, CLASS A closure task): the *sequence itself* is
    # already a plain Python list ("list_of_lists" above covers list-of-
    # lists-of-ints), but here the OUTER sequence is a tuple/generator-free
    # iterable holding non-ndarray, non-nested-list array-like ELEMENTS
    # (range, bare tuple), each of which must independently array-coerce.
    out.append(("form_tuple_of_ranges", ((range(3), range(3, 6)),), {}))
    out.append(("form_mixed_ndarray_and_list", ([np.array([1, 2, 3]), [4, 5, 6]],), {}))
    out += _concat_weak_scalar_cases()
    return out


def _concat_weak_scalar_cases():
    """NEP 50 weak-scalar operands -- `concatenate` is the ONLY member of
    this whole block that ever sees a bare Python scalar *as* a scalar, so
    this grid lives here and deliberately does not get replicated onto
    `hstack`/`vstack`/`stack`/`append`: verified live against numpy 2.5.1,
    `hstack`'s `atleast_1d` turns the bare scalar into a strong 1-d array
    before concatenation (`np.hstack((int8_arr, 300))` really is int64),
    and `vstack`/`stack` raise on shape long before dtype matters. Adding
    the grid there would assert the WRONG rule.

    The rule under test: promote over the STRONG operands only, then
    wrapping-cast each weak scalar into that dtype. The load-bearing rows
    are the ones where "wrapping" is visible rather than incidental --
    `(int8_arr, 300)` -> int8 [1,0,2,44], `(bool_arr, 300)` -> int64
    (a weak int against a bool array retargets to the default, it does not
    stay bool), and `(float16_arr, 10**100)` -> float16 inf while
    `(int8_arr, 10**100)` -> OverflowError, because the overflow is
    INT-TARGET-SPECIFIC.

    The strong/weak spellings are interleaved on purpose: `[300]` (a plain
    list, STRONG -- it goes through asarray) and `np.int64(300)` (STRONG)
    must NOT wrap, while bare `300` must. A fix that treats "not an
    ndarray" as "weak" passes the wrapping rows and fails these.

    Both operand orders are generated because the 0-d rejection in
    `concatenate` is position-sensitive (see `ionp-core/src/manip.rs`):
    at `axis=0` a 0-d operand at index 0 gets "zero-dimensional arrays
    cannot be concatenated" and anywhere else gets the ndim-mismatch
    message.
    """
    scalars = [
        300, -300, 44, 0, 1, True, False, 2.5, -0.5, 1e100, 10 ** 100,
        -(10 ** 100), 2 ** 63, -(2 ** 63), 1j, (2 + 3j), 255, 256, -1,
    ]
    strong_spellings = [
        ("list", [300]), ("nparr", np.array([300])), ("npscal", np.int64(300)),
        ("f64scal", np.float64(2.5)), ("tuple", (300,)),
        ("bool_", np.bool_(True)), ("f32scal", np.float32(2.0)),
    ]
    bases = [
        ("i8", np.array([1, 0, 2], np.int8)),
        ("i16", np.array([1], np.int16)),
        ("f16", np.array([1, 0, 2], np.float16)),
        ("bool", np.array([True], np.bool_)),
        ("c64", np.array([1, 0, 2], np.complex64)),
        ("u8", np.array([1, 2], np.uint8)),
        ("u64", np.array([1], np.uint64)),
        ("i64", np.array([1], np.int64)),
        ("f32", np.array([1.5], np.float32)),
    ]
    out = []
    for bname, base in bases:
        for i, sc in enumerate(scalars):
            out.append((f"weak/{bname}/{i:02d}/after", ((base, sc),), {"axis": None}))
            out.append((f"weak/{bname}/{i:02d}/before", ((sc, base),), {"axis": None}))
        # three-operand form: the weak scalar must fold into the promotion
        # of the two STRONG operands, not become a third strong one.
        out.append((f"weak3/{bname}", ((base, 300, np.array([1], np.int16)),), {"axis": None}))
    for sname, sc in strong_spellings:
        out.append((f"strong/{sname}", ((np.array([1, 0, 2], np.int8), sc),), {"axis": None}))
    # all-weak falls back to the default dtype, not to "smallest that fits"
    out.append(("allweak/int", ((300, 400),), {"axis": None}))
    out.append(("allweak/float", ((1.5, 2),), {"axis": None}))
    out.append(("allweak/bool_int", ((True, 300),), {"axis": None}))
    out.append(("allweak/huge", ((10 ** 100, 1),), {"axis": None}))
    # position-sensitive 0-d rejection at axis=0
    a8 = np.array([1, 0, 2], np.int8)
    z = np.array(5)
    out.append(("zerod/scalar_first_axis0", ((300, a8),), {}))
    out.append(("zerod/scalar_last_axis0", ((a8, 300),), {}))
    out.append(("zerod/arr_first_axis0", ((z, a8),), {}))
    out.append(("zerod/arr_last_axis0", ((a8, z),), {}))
    out.append(("zerod/arr_mid_axis0", ((a8, z, a8),), {}))
    return out


def stack_cases():
    out = []
    groups = _stack_family_arrays()
    for name, a, b in groups:
        out.append((f"axis0/{name}", ([a, b],), {}))
        out.append((f"axis1/{name}", ([a, b],), {"axis": 1}))
        out.append((f"axis2_new_last/{name}", ([a, b],), {"axis": 2}))
        out.append((f"axis_neg1/{name}", ([a, b],), {"axis": -1}))
    out.append(("1d_three", ([np.array([1, 2]), np.array([3, 4]), np.array([5, 6])],), {"axis": 0}))
    out.append(("0d_arrays", ([np.array(1), np.array(2), np.array(3)],), {}))
    out.append(("dtype_override", ([np.arange(4, dtype=np.int32), np.arange(4, dtype=np.int32)],), {"dtype": np.float32}))
    out.append(("mismatched_shapes_raises", ([np.zeros((2, 3)), np.zeros((3, 2))],), {}))
    out.append(("axis_out_of_range_raises", ([np.zeros((2, 3)), np.zeros((2, 3))],), {"axis": 10}))
    # DEFECT 3, `stack`'s own wording -- see the identical note on
    # `concatenate_cases`'s `generator_rejected_raises` above.
    out.append(("generator_rejected_raises", ((np.array([1, 2]) for _ in range(2)),), {}))
    out.extend(_casting_dtype_grid_cases(lambda a, b: ([a, b],)))
    out.append(("form_mixed_ndarray_and_list", ([np.array([1, 2]), [3, 4]],), {}))
    return out


def hstack_cases():
    out = []
    groups = _stack_family_arrays()
    for name, a, b in groups:
        out.append((f"2d/{name}", ([a, b],), {}))
    out.append(("1d", ([np.array([1, 2, 3]), np.array([4, 5])],), {}))
    out.append(("0d", ([np.array(1), np.array(2)],), {}))
    out.append(("3d", ([np.arange(24).reshape(2, 3, 4), np.arange(24).reshape(2, 3, 4)],), {}))
    out.append(("mismatched_raises", ([np.zeros((2, 3)), np.zeros((3, 3))],), {}))
    out.append(("generator_rejected_raises", ((np.array([1, 2]) for _ in range(2)),), {}))
    out.extend(_casting_dtype_grid_cases(lambda a, b: ([a, b],)))
    out.append(("form_mixed_ndarray_and_list", ([np.array([1, 2, 3]), [4, 5]],), {}))
    return out


def vstack_cases():
    out = []
    groups = _stack_family_arrays()
    for name, a, b in groups:
        out.append((f"2d/{name}", ([a, b],), {}))
    out.append(("1d", ([np.array([1, 2, 3]), np.array([4, 5, 6])],), {}))
    out.append(("0d", ([np.array(1), np.array(2)],), {}))
    out.append(("mismatched_raises", ([np.zeros((2, 3)), np.zeros((2, 4))],), {}))
    out.append(("generator_rejected_raises", ((np.array([1, 2]) for _ in range(2)),), {}))
    out.extend(_casting_dtype_grid_cases(lambda a, b: ([a, b],)))
    out.append(("form_mixed_ndarray_and_list", ([np.array([1, 2, 3]), [4, 5, 6]],), {}))
    return out


def row_stack_cases():
    # row_stack is a pure alias of vstack -- same coverage.
    return vstack_cases()


def dstack_cases():
    out = []
    groups = _stack_family_arrays()
    for name, a, b in groups:
        out.append((f"2d/{name}", ([a, b],), {}))
    out.append(("1d", ([np.array([1, 2, 3]), np.array([4, 5, 6])],), {}))
    out.append(("0d", ([np.array(1), np.array(2)],), {}))
    out.append(("form_mixed_ndarray_and_list", ([np.array([1, 2, 3]), [4, 5, 6]],), {}))
    return out


def column_stack_cases():
    out = []
    out.append(("1d_two", ([np.array([1, 2, 3]), np.array([4, 5, 6])],), {}))
    out.append(("1d_three", ([np.array([1, 2]), np.array([3, 4]), np.array([5, 6])],), {}))
    out.append(("2d_two", ([np.arange(6).reshape(3, 2), np.arange(3).reshape(3, 1)],), {}))
    out.append(("mixed_1d_2d", ([np.array([1, 2, 3]), np.arange(6).reshape(3, 2)],), {}))
    out.append(("float_dtype", ([np.array([1.5, 2.5]), np.array([3.5, 4.5])],), {}))
    out.append(("form_mixed_ndarray_and_list", ([np.array([1, 2, 3]), [4, 5, 6]],), {}))
    return out


# ---------------------------------------------------------------------------
# flip / fliplr / flipud
# ---------------------------------------------------------------------------

def flip_cases():
    out = _corpus_cases()
    extra = [
        ("axis0_2d", (np.arange(12).reshape(3, 4), 0), {}),
        ("axis1_2d", (np.arange(12).reshape(3, 4), 1), {}),
        ("axis_neg1_2d", (np.arange(12).reshape(3, 4), -1), {}),
        ("axis_tuple_2d", (np.arange(12).reshape(3, 4), (0, 1)), {}),
        ("axis0_3d", (np.arange(24).reshape(2, 3, 4), 0), {}),
        ("axis_tuple_3d", (np.arange(24).reshape(2, 3, 4), (1, 2)), {}),
        ("axis_none_default", (np.arange(12).reshape(3, 4),), {}),
        ("1d", (np.arange(5),), {}),
        ("axis_out_of_range_raises", (np.arange(6).reshape(2, 3), 5), {}),
        ("repeated_axis_raises", (np.arange(6).reshape(2, 3), (0, 0)), {}),
    ]
    return [(l, a, k) for l, a, k in extra] + [(f"corpus/{l}", (v,), {}) for l, (v,), _ in out] + _form_cases("/form")


def fliplr_cases():
    out = []
    for c in _CORPUS:
        if c.value.ndim >= 2:
            out.append((f"corpus/{c.label}", (c.value,), {}))
    out.append(("1d_raises", (np.arange(5),), {}))
    out.append(("0d_raises", (np.array(5),), {}))
    out += _form_cases("/form")
    return out


def flipud_cases():
    out = []
    for c in _CORPUS:
        if c.value.ndim >= 1:
            out.append((f"corpus/{c.label}", (c.value,), {}))
    out.append(("0d_raises", (np.array(5),), {}))
    out += _form_cases("/form")
    return out


# ---------------------------------------------------------------------------
# roll
# ---------------------------------------------------------------------------

def roll_cases():
    out = []
    a2 = np.arange(12).reshape(3, 4)
    a1 = np.arange(7)
    a3 = np.arange(24).reshape(2, 3, 4)
    out.append(("1d_pos_shift", (a1, 2), {}))
    out.append(("1d_neg_shift", (a1, -2), {}))
    out.append(("1d_shift_exceeds_len", (a1, 100), {}))
    out.append(("1d_shift_zero", (a1, 0), {}))
    out.append(("flat_no_axis_2d", (a2, 3), {}))
    out.append(("axis0", (a2, 1, 0), {}))
    out.append(("axis1", (a2, 2, 1), {}))
    out.append(("axis_neg1", (a2, 1, -1), {}))
    out.append(("axis_tuple_shift_tuple", (a2, (1, 2), (0, 1)), {}))
    out.append(("axis_repeated", (a3, (1, 1), (0, 0)), {}))
    out.append(("3d_axis2", (a3, 1, 2), {}))
    for dtype in SWEEP_DTYPES:
        arr = (np.arange(10) % 5).astype(dtype)
        out.append((f"dtype/{np.dtype(dtype).name}", (arr, 3), {}))
    out.append(("empty_array", (np.array([], dtype=np.int64), 3), {}))
    out.append(("shift_axis_len_mismatch_raises", (a2, (1, 2, 3), (0, 1)), {}))
    # Regression coverage for the 2026-08-01 false declaration: a scalar
    # `shift` broadcasts across every named axis in a tuple `axis=` (numpy
    # does NOT require `len(shift) == len(axis)` here), and a tuple `shift`
    # with `axis=None` is valid (the shifts sum onto the single flattened
    # axis) -- both previously raised in anionpy where real numpy succeeds.
    out.append(("scalar_shift_tuple_axis", (a2, 1, (0, 1)), {}))
    out.append(("scalar_shift_tuple_axis_3d", (a3, 1, (0, 2)), {}))
    out.append(("tuple_shift_axis_none", (a2, (1, 2), None), {}))
    out.append(("tuple_shift_axis_none_3elem", (a1, (1, 2, 3), None), {}))
    # Axis length 1 broadcasting against a longer shift tuple (the same
    # mechanism `axis=None` uses internally, per numpy's own `roll` source:
    # `axis=None` recurses as `roll(a.ravel(), shift, 0)`).
    out.append(("shift_tuple_axis_len1", (a2, (1, 2), (0,)), {}))
    # FORM axis (2026-08-02, CLASS A closure task): `a` as a raw Python
    # list/tuple/range, no ndarray. `roll` needs a second positional
    # (`shift`), which corpus.form_axis_cases()'s Case values don't carry,
    # so this is a small hand-built set rather than a `_form_cases()` call.
    out.append(("form_list_1d", ([1, 2, 3, 4, 5], 2), {}))
    out.append(("form_tuple_1d", ((1, 2, 3, 4, 5), -2), {}))
    out.append(("form_range_1d", (range(6), 2), {}))
    out.append(("form_list2d", ([[1, 2, 3], [4, 5, 6]], 1, 0), {}))
    return out


# ---------------------------------------------------------------------------
# tile / repeat
# ---------------------------------------------------------------------------

def tile_cases():
    out = []
    out.append(("1d_scalar_reps", (np.array([1, 2, 3]), 2), {}))
    out.append(("1d_tuple_reps", (np.array([1, 2, 3]), (3,)), {}))
    out.append(("2d_reps_2d", (np.arange(6).reshape(2, 3), (2, 2)), {}))
    out.append(("2d_reps_1d_pads", (np.arange(6).reshape(2, 3), (3,)), {}))
    out.append(("1d_reps_extra_dims", (np.array([1, 2]), (2, 3)), {}))
    out.append(("0d", (np.array(5), (3,)), {}))
    out.append(("reps_zero", (np.arange(4).reshape(2, 2), (0, 2)), {}))
    out.append(("3d", (np.arange(8).reshape(2, 2, 2), (1, 2, 1)), {}))
    for dtype in SWEEP_DTYPES:
        arr = np.arange(4).reshape(2, 2).astype(dtype)
        out.append((f"dtype/{np.dtype(dtype).name}", (arr, (2, 1)), {}))
    # FORM axis (2026-08-02, CLASS A closure task): `A` as a raw Python
    # list/tuple/range, no ndarray -- `tile` needs a second positional
    # (`reps`), same reasoning as `roll_cases`'s hand-built block above.
    out.append(("form_list_1d", ([1, 2, 3], 2), {}))
    out.append(("form_tuple_1d", ((1, 2, 3), (3,)), {}))
    out.append(("form_range_1d", (range(4), 2), {}))
    out.append(("form_list2d", ([[1, 2, 3], [4, 5, 6]], (2, 2)), {}))
    out.extend(_tile_reps_spelling_cases())
    return out


def _tile_reps_spelling_cases():
    """The `reps` SPELLING axis, which the block above has no coverage of.

    Every case above passes `reps` as a plain int or a tuple of plain
    ints, and that is exactly the shape of the blind spot: `tile` was
    declared "exact" while 11 of 36 `reps` spellings diverged, because
    anionpy coerced `reps` with a single "list of non-negative ints" helper
    and numpy does something structurally different -- its observable
    behaviour falls out of WHICH primitive fails FIRST inside a short
    Python function (tuple-ize, an all-ones fast path that never coerces
    at all, a Python `*` for `shape_out`, `repeat`'s own count coercion,
    then `__index__` on the products). See the long comment on
    `tile` in `ionp-py/src/manip.rs` for the measured contract.

    Three properties this grid is specifically built to hold down, each
    of which was a live bug and none of which a plain int/tuple case can
    see:

      * `np.tile(arr, 1.0)` SUCCEEDS -- `1.0 == 1`, so the fast path
        returns before any integer conversion. So do `True`, `np.True_`,
        `np.float64(1.0)`, `(1.0, 1.0)`, `[]`, `np.array([])` and
        `{1: 2}` (a dict tuple-izes to its keys). anionpy raised TypeError
        for all of them.
      * the same `reps=1.0` with a LIST operand RAISES, because the fast
        path is gated on `isinstance(A, ndarray)`. Both spellings are in
        here; a fix that drops the operand-kind gate passes one and
        fails the other.
      * the `'float' object cannot be interpreted as an integer` message
        comes from the FINAL reshape of `shape_out`, not from `repeat`
        (which accepts `2.0`, `2.5`, `"2"`). That is why the element is
        named with the dtype of the PRODUCT: `np.array([2., 1.])` reports
        `'numpy.float64'`, and `(1.0, 2.0)` reports the SECOND element
        because `nrep != 1` skips the first.

    The negative/huge entries pin the error-ORDER, which is the part most
    likely to regress silently: `-1.0` is a repeat-loop ValueError and
    not the reshape TypeError, `1j` and `np.array(2.0)` have their own
    distinct TypeErrors, `2**63` is an OverflowError, and on an EMPTY
    operand the `n > 0` guard skips the loop so the reshape message wins
    instead. `-(2**63)` is in here twice on purpose, against a 1-d and a
    2-d base: numpy's "negative dimensions" test is on the WRAPPING intp
    product `rows * count`, so 2 * -2**63 wraps to a legal 0 while
    3 * -2**63 does not.
    """
    reps = [
        1.0, 2.0, 0.0, -1.0, 2.5, -0.5, np.float64(2.0), np.float64(1.0),
        np.float32(2.0), True, False, np.True_, np.False_, np.int8(2),
        np.uint8(2), np.uint64(2), 2, 0, 1, 3,
        (2.0,), (1.0, 2.0), (1.0, 1.0), [2.0, 1.0], [], (), (1,), (2, 3),
        (1, 1), (0, 2), (2, 0), (2, -1),
        np.array(2.0), np.array([2.0, 1.0]), np.array([2, 1]), np.array(2),
        np.array([]), np.array([[2]]), [[2]], 1j, "2", "x", None,
        np.float16(2.0), 1e100, 2 ** 63, -(2 ** 63), {1: 2}, range(2),
        float("nan"), float("inf"), b"2", np.complex64(2), -3,
    ]
    bases = [
        ("i8_1d", np.array([1, 0, 2], np.int8)),
        ("f64_2d", np.arange(6).reshape(2, 3).astype(np.float64)),
        ("c64_0d", np.array(1, np.complex64)),
        ("u16_empty", np.array([], np.uint16)),
        ("bool_empty2d", np.zeros((0, 3), np.bool_)),
        ("list_1d", [1, 2, 3]),
        ("tuple_2d", ((1, 2), (3, 4))),
    ]
    out = []
    for bname, base in bases:
        for i, r in enumerate(reps):
            out.append((f"reps_spelling/{bname}/{i:02d}", (base, r), {}))
    return out


def repeat_cases():
    # NOTE on the 2026-08-01 false declaration ("`repeat` rejects an
    # anionpy.ndarray-valued `repeats`, accepts a numpy array or list"): that
    # specific bug class is NOT expressible as a case here. `harness.py`'s
    # `run_case` calls the real-numpy reference AND the anionpy function with
    # the exact same frozen args (see `_freshen`/`run_case`) -- it never
    # converts an argument to an `anionpy.ndarray` before calling numpy. If a
    # case's `repeats` were an actual `anionpy.ndarray` instance, the numpy
    # reference call itself would fail before ever reaching anionpy: real
    # numpy's `np.repeat` converts a non-ndarray `repeats` via `__array__`,
    # and `anionpy.ndarray.__array__` (`ionp-py/src/lib.rs`, declared "exact"
    # in `anionpy/_state/ndarray.py`) does not yet accept the modern
    # `dtype=`/`copy=` keywords numpy's array-protocol passes, so numpy
    # itself raises `TypeError: ndarray.__array__() takes no arguments (1
    # given)` -- a genuine, separate, out-of-scope bug in `__array__`, not
    # in `repeat`. Fixing this class of regression detection would require
    # either fixing `__array__`'s NEP signature (a different file/task) or
    # a harness change to call anionpy with ionp-native args (an explicit,
    # coordinator-held, ledger-wide decision -- see the `PATH-TO-100.md`
    # blind spot #3 discussion). The actual fix (`usize_list_from_pyobj` in
    # `ionp-py/src/manip.rs` now accepts `PyRef<PyArray>`) was verified by
    # a manual out-of-corpus probe instead: ionp-array, numpy-array, and
    # list `repeats` all produce byte-identical output.
    out = []
    out.append(("1d_scalar_repeats", (np.array([1, 2, 3]), 2), {}))
    out.append(("1d_per_element_repeats", (np.array([1, 2, 3]), [1, 2, 3]), {}))
    out.append(("axis0", (np.arange(6).reshape(2, 3), 2, 0), {}))
    out.append(("axis1", (np.arange(6).reshape(2, 3), 3, 1), {}))
    out.append(("axis_neg1", (np.arange(6).reshape(2, 3), 2, -1), {}))
    out.append(("axis0_per_row_repeats", (np.arange(6).reshape(2, 3), [1, 3], 0), {}))
    out.append(("no_axis_flattens", (np.arange(6).reshape(2, 3), 2), {}))
    out.append(("repeats_zero", (np.array([1, 2, 3]), 0), {}))
    out.append(("3d_axis1", (np.arange(24).reshape(2, 3, 4), 2, 1), {}))
    for dtype in SWEEP_DTYPES:
        arr = np.arange(4).astype(dtype)
        out.append((f"dtype/{np.dtype(dtype).name}", (arr, 3), {}))
    # FORM axis (2026-08-02, CLASS A closure task): `a` as a raw Python
    # list/tuple/range, no ndarray -- `repeat` needs a second positional
    # (`repeats`), same reasoning as `roll_cases`'s hand-built block above.
    out.append(("form_list_1d", ([1, 2, 3], 2), {}))
    out.append(("form_tuple_1d", ((1, 2, 3), [1, 2, 3]), {}))
    out.append(("form_range_1d", (range(4), 2), {}))
    return out


# ---------------------------------------------------------------------------
# broadcast_shapes / broadcast_arrays
# ---------------------------------------------------------------------------

def broadcast_shapes_cases():
    out = [
        ("simple", ((3, 1), (1, 4)), {}),
        ("identical", ((2, 3), (2, 3)), {}),
        ("scalar_and_shape", ((), (3, 4)), {}),
        ("three_shapes", ((5, 1, 3), (1, 4, 1), (5, 4, 3)), {}),
        ("leading_dims", ((3,), (2, 1, 3)), {}),
        ("all_ones", ((1, 1), (1, 1)), {}),
        ("empty_dims", ((0, 3), (1, 3)), {}),
        ("single_shape", ((3, 4),), {}),
        ("incompatible_raises", ((3, 4), (5, 4)), {}),
    ]
    return out


def broadcast_arrays_cases():
    out = [
        ("simple", (np.array([1, 2, 3]), np.array([[1], [2]])), {}),
        ("three_arrays", (np.zeros((3, 1)), np.zeros((1, 4)), np.zeros((3, 4))), {}),
        ("identical_shapes", (np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)), {}),
        ("scalar_broadcast", (np.array(5), np.zeros((3, 4))), {}),
        ("single_array", (np.arange(6).reshape(2, 3),), {}),
        ("incompatible_raises", (np.zeros((3, 4)), np.zeros((5, 4))), {}),
    ]
    return out


# ---------------------------------------------------------------------------
# atleast_1d / atleast_2d / atleast_3d (single-array call form only --
# see module docstring)
# ---------------------------------------------------------------------------

def atleast_1d_cases():
    return _corpus_cases() + _form_cases("/form")


def atleast_2d_cases():
    return _corpus_cases() + _form_cases("/form")


def atleast_3d_cases():
    return _corpus_cases() + _form_cases("/form")


# ---------------------------------------------------------------------------
# diag / diagflat / diagonal / tril / triu / trace
# ---------------------------------------------------------------------------

def diag_cases():
    out = []
    out.append(("1d_k0", (np.array([1, 2, 3]),), {}))
    out.append(("1d_k_pos", (np.array([1, 2, 3]), 2), {}))
    out.append(("1d_k_neg", (np.array([1, 2, 3]), -1), {}))
    out.append(("1d_empty", (np.array([], dtype=np.int64),), {}))
    out.append(("2d_extract_k0", (np.arange(9).reshape(3, 3),), {}))
    out.append(("2d_extract_k_pos", (np.arange(12).reshape(3, 4), 1), {}))
    out.append(("2d_extract_k_neg", (np.arange(12).reshape(3, 4), -1), {}))
    out.append(("2d_non_square", (np.arange(6).reshape(2, 3),), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (np.arange(4).astype(dtype),), {}))
    out.append(("3d_raises", (np.arange(8).reshape(2, 2, 2),), {}))
    out += _form_cases("/form")
    return out


def diagflat_cases():
    out = []
    out.append(("1d", (np.array([1, 2, 3]),), {}))
    out.append(("1d_k1", (np.array([1, 2, 3]), 1), {}))
    out.append(("1d_k_neg2", (np.array([1, 2, 3]), -2), {}))
    out.append(("2d_ravels", (np.arange(6).reshape(2, 3),), {}))
    out.append(("0d", (np.array(5),), {}))
    out.append(("scalar_python_int", (7,), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (np.arange(3).astype(dtype),), {}))
    out += _form_cases("/form")
    return out


def diagonal_cases():
    out = []
    out.append(("2d_default", (np.arange(12).reshape(3, 4),), {}))
    out.append(("2d_offset_pos", (np.arange(12).reshape(3, 4), 1), {}))
    out.append(("2d_offset_neg", (np.arange(12).reshape(3, 4), -1), {}))
    out.append(("2d_offset_out_of_range", (np.arange(12).reshape(3, 4), 10), {}))
    out.append(("3d_default_batched", (np.arange(24).reshape(2, 3, 4),), {}))
    out.append(("3d_axis1_axis2", (np.arange(24).reshape(2, 3, 4), 0, 1, 2), {}))
    out.append(("3d_axis0_axis2", (np.arange(24).reshape(2, 3, 4), 0, 0, 2), {}))
    out.append(("3d_negative_axes", (np.arange(24).reshape(2, 3, 4), 0, -1, -2), {}))
    out.append(("4d_batched", (np.arange(120).reshape(2, 3, 4, 5), 0, 2, 3), {}))
    out.append(("4d_batched_other_axes", (np.arange(120).reshape(2, 3, 4, 5), 1, 0, 1), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (np.arange(9).reshape(3, 3).astype(dtype),), {}))
    out.append(("axis1_eq_axis2_raises", (np.arange(12).reshape(3, 4), 0, 0, 0), {}))
    out.append(("1d_raises", (np.arange(5),), {}))
    out += _form_cases("/form")
    return out


def tril_cases():
    out = []
    out.append(("square_k0", (np.arange(9).reshape(3, 3),), {}))
    out.append(("square_k1", (np.arange(9).reshape(3, 3), 1), {}))
    out.append(("square_k_neg1", (np.arange(9).reshape(3, 3), -1), {}))
    out.append(("rect_tall", (np.arange(12).reshape(4, 3),), {}))
    out.append(("rect_wide", (np.arange(12).reshape(3, 4),), {}))
    out.append(("k_beyond_range", (np.arange(9).reshape(3, 3), 5), {}))
    out.append(("k_beyond_neg_range", (np.arange(9).reshape(3, 3), -5), {}))
    out.append(("batched_3d", (np.arange(18).reshape(2, 3, 3),), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (np.arange(9).reshape(3, 3).astype(dtype),), {}))
    # Regression coverage for the 2026-08-01 false declaration: numpy
    # broadcasts a 1-D input against a square `tri` mask (row-wise) instead
    # of requiring >= 2-D, and a 0-D input hits numpy's own `tri(*())`
    # missing-argument TypeError.
    out.append(("1d_k0", (np.arange(6),), {}))
    out.append(("1d_k_pos", (np.arange(6), 2), {}))
    out.append(("1d_k_neg", (np.arange(6), -2), {}))
    out.append(("1d_empty", (np.array([], dtype=np.int64),), {}))
    out.append(("0d_raises", (np.array(5),), {}))
    out += _form_cases("/form")
    return out


def triu_cases():
    out = []
    out.append(("square_k0", (np.arange(9).reshape(3, 3),), {}))
    out.append(("square_k1", (np.arange(9).reshape(3, 3), 1), {}))
    out.append(("square_k_neg1", (np.arange(9).reshape(3, 3), -1), {}))
    out.append(("rect_tall", (np.arange(12).reshape(4, 3),), {}))
    out.append(("rect_wide", (np.arange(12).reshape(3, 4),), {}))
    out.append(("k_beyond_range", (np.arange(9).reshape(3, 3), 5), {}))
    out.append(("k_beyond_neg_range", (np.arange(9).reshape(3, 3), -5), {}))
    out.append(("batched_3d", (np.arange(18).reshape(2, 3, 3),), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (np.arange(9).reshape(3, 3).astype(dtype),), {}))
    # See tril_cases()'s matching note -- same fix, mirrored bug.
    out.append(("1d_k0", (np.arange(6),), {}))
    out.append(("1d_k_pos", (np.arange(6), 2), {}))
    out.append(("1d_k_neg", (np.arange(6), -2), {}))
    out.append(("1d_empty", (np.array([], dtype=np.int64),), {}))
    out.append(("0d_raises", (np.array(5),), {}))
    out += _form_cases("/form")
    return out


def trace_cases():
    out = []
    out.append(("square_default", (np.arange(9).reshape(3, 3),), {}))
    out.append(("offset_pos", (np.arange(12).reshape(3, 4), 1), {}))
    out.append(("offset_neg", (np.arange(12).reshape(3, 4), -1), {}))
    out.append(("rect", (np.arange(12).reshape(4, 3),), {}))
    out.append(("3d_batched", (np.arange(24).reshape(2, 3, 4),), {}))
    out.append(("3d_explicit_axes", (np.arange(24).reshape(2, 3, 4), 0, 1, 2), {}))
    out.append(("dtype_override", (np.arange(9).reshape(3, 3), 0, 0, 1, np.float64), {}))
    for dtype in ALL_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (np.arange(9).reshape(3, 3).astype(dtype),), {}))
    # accumulator-widening check: int8 input should trace without overflow
    # (real numpy widens integer accumulators -- see manip.rs's
    # `trace_accum_dtype`, verified against real numpy 2.5.1 directly).
    out.append(("int8_accum_widen", (np.full((3, 3), 100, dtype=np.int8),), {}))
    out.append(("uint8_accum_widen", (np.full((3, 3), 200, dtype=np.uint8),), {}))
    out += _form_cases("/form")
    return out


# ---------------------------------------------------------------------------
# split / array_split / hsplit / vsplit / dsplit
# ---------------------------------------------------------------------------

def _split_family_adapters(numpy_name: str):
    """See this file's module docstring for why these tuple-wrapping
    adapters exist. `numpy_name` selects the real numpy function
    (`np.split`/`np.array_split`/...); the anionpy side always resolves the
    identically-named attribute on the real `anionpy` package (lazy import,
    matching this repo's established `import anionpy` pattern for adapters --
    see e.g. linalg_cases.py/lib_cases.py)."""

    def numpy_adapter(*args, **kwargs):
        return tuple(getattr(np, numpy_name)(*args, **kwargs))

    def ionp_adapter(*args, **kwargs):
        import anionpy
        return tuple(getattr(anionpy, numpy_name)(*args, **kwargs))

    return numpy_adapter, ionp_adapter


def split_cases():
    out = []
    x = np.arange(9)
    m = np.arange(16).reshape(4, 4)
    out.append(("equal_sections", (x, 3), {}))
    out.append(("indices_list", (x, [2, 5]), {}))
    out.append(("indices_beyond_len", (x, [2, 5, 20]), {}))
    out.append(("indices_empty_list", (x, []), {}))
    out.append(("axis1_2d", (m, 2), {"axis": 1}))
    out.append(("axis_neg1", (m, 2), {"axis": -1}))
    out.append(("unequal_raises", (x, 4), {}))
    out.append(("zero_sections_raises", (x, 0), {}))
    out.append(("neg_sections_raises", (x, -3), {}))
    out.append(("axis_oob_raises", (m, 2, 5), {}))
    for dtype in SWEEP_DTYPES:
        arr = np.arange(9).astype(dtype)
        out.append((f"dtype/{np.dtype(dtype).name}", (arr, 3), {}))
    # FORM axis, error-matching (2026-08-02): real numpy's `split()` touches
    # `ary.shape[axis]` directly with NO asarray fallback whenever
    # `indices_or_sections` is a scalar int (unlike `array_split`, which
    # wraps the same access in `try/except AttributeError: len(ary)`) -- so
    # a raw list/tuple/range genuinely raises AttributeError under real
    # numpy too, uniformly regardless of ndim. These cases are NOT hardcoded
    # expectations: the differential harness runs both numpy and anionpy live
    # and requires the same exception class + message text from each: see
    # ionp-py/src/manip.rs's `require_shape_attr`, which reproduces the
    # mechanism (a bare Python attribute lookup) rather than the message.
    out.append(("form_list1d", ([1, 2, 3, 4, 5, 6], 2), {}))
    out.append(("form_list3d", ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2), {}))
    out.append(("form_range", (range(4), 2), {}))
    return out


def array_split_cases():
    out = []
    x = np.arange(9)
    m = np.arange(16).reshape(4, 4)
    out.append(("uneven_sections", (x, 4), {}))
    out.append(("more_sections_than_elems", (np.arange(3), 5), {}))
    out.append(("indices_list", (x, [2, 5]), {}))
    out.append(("axis1_2d", (m, 3), {"axis": 1}))
    out.append(("zero_sections_raises", (x, 0), {}))
    out.append(("neg_sections_raises", (x, -1), {}))
    for dtype in SWEEP_DTYPES:
        arr = np.arange(7).astype(dtype)
        out.append((f"dtype/{np.dtype(dtype).name}", (arr, 3), {}))
    out.append(("form_list_1d", ([1, 2, 3, 4, 5, 6], 3), {}))
    out.append(("form_range_1d", (range(9), 3), {}))
    return out


def hsplit_cases():
    out = []
    x = np.arange(9)
    m = np.arange(16).reshape(4, 4)
    d = np.arange(24).reshape(2, 3, 4)
    out.append(("1d", (x, 3), {}))
    out.append(("2d", (m, 2), {}))
    out.append(("3d", (d, 2), {}))
    out.append(("0d_raises", (np.array(5), 2), {}))
    out.append(("unequal_raises", (x, 4), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (m.astype(dtype), 2), {}))
    # FORM axis, error-matching (2026-08-02): real numpy's `hsplit` does a
    # lenient `_nx.ndim(ary) == 0` check first (fallback to asarray, so a
    # raw list/range never trips THIS branch), then touches `ary.ndim`
    # strictly right after regardless of which branch it takes -- so every
    # form here (1-D, 3-D, range) uniformly raises AttributeError under
    # real numpy. See ionp-py/src/manip.rs's `hsplit` for the matching
    # lenient-then-strict dispatch (mechanism reproduced, not the message).
    out.append(("form_list1d", ([1, 2, 3, 4, 5, 6], 3), {}))
    out.append(("form_list3d", ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2), {}))
    out.append(("form_range", (range(4), 2), {}))
    return out


def vsplit_cases():
    out = []
    m = np.arange(16).reshape(4, 4)
    d = np.arange(24).reshape(2, 3, 4)
    out.append(("2d", (m, 2), {}))
    out.append(("3d", (d, 2), {}))
    out.append(("1d_raises", (np.arange(4), 2), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (m.astype(dtype), 2), {}))
    # FORM axis, error-matching (2026-08-02): real numpy's `vsplit` does a
    # LENIENT `_nx.ndim(ary) < 2` check first (asarray-fallback, so it
    # actually SUCCEEDS on a raw list/range and computes a real ndim), then
    # only delegates to `split(ary, ..., 0)` -- whose own strict
    # `ary.shape[axis]` is what rejects a raw list. The upshot: the
    # exception CLASS genuinely varies by form here, unlike
    # hsplit/rollaxis/dsplit's uniform-across-forms AttributeError --
    # form_list1d's ndim (1) fails the lenient check itself (ValueError),
    # while form_list3d's ndim (3) passes it and only then hits the strict
    # shape gate (AttributeError). Confirmed live against numpy 2.5.1 and
    # reproduced via the same lenient/strict dispatch in
    # ionp-py/src/manip.rs's `vsplit` (mechanism, not hardcoded message).
    out.append(("form_list1d", ([1, 2, 3, 4], 2), {}))
    out.append(("form_list3d", ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2), {}))
    out.append(("form_range", (range(4), 2), {}))
    return out


def dsplit_cases():
    out = []
    d = np.arange(24).reshape(2, 3, 4)
    out.append(("3d", (d, 2), {}))
    out.append(("2d_raises", (np.arange(16).reshape(4, 4), 2), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (d.astype(dtype), 2), {}))
    # FORM axis, error-matching (2026-08-02): same lenient-then-strict shape
    # as vsplit above, but dsplit's own low-ndim ValueError threshold is
    # <3, so a raw list3d (ndim 3, passes the lenient check) still falls
    # through to the strict `.shape` touch and genuinely raises
    # AttributeError under real numpy; a raw list1d/range (ndim 1, < 3)
    # raises the lenient-path ValueError instead, same as numpy.
    out.append(("form_list1d", ([1, 2, 3, 4], 2), {}))
    out.append(("form_list3d", ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2), {}))
    out.append(("form_range", (range(4), 2), {}))
    return out


# ---------------------------------------------------------------------------
# insert / delete / append / resize
# ---------------------------------------------------------------------------

def insert_cases():
    out = []
    x = np.arange(9)
    m = np.arange(12).reshape(3, 4)
    out.append(("scalar_idx_scalar_val", (x, 2, 99), {}))
    out.append(("scalar_idx_neg", (x, -1, 99), {}))
    out.append(("list_idx_list_val", (x, [2, 3], [99, 100]), {}))
    out.append(("slice_obj", (x, slice(1, 3), [99, 100]), {}))
    out.append(("bool_obj", (x, x % 2 == 0, 99), {}))
    out.append(("axis_row", (m, 1, [7, 8, 9, 10]), {"axis": 0}))
    out.append(("axis_col_scalar_idx", (m, 1, [7, 8, 9]), {"axis": 1}))
    out.append(("no_axis_ravels", (m, 1, 99), {}))
    out.append(("oob_raises", (x, 100, 1), {}))
    out.append(("scalar_broadcast_val", (x, [1, 2, 3], 0), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (x.astype(dtype), 2, 99), {}))
    out.append(("form_list_1d", ([1, 2, 3, 4, 5], 2, 99), {}))
    out.append(("form_range_1d", (range(6), 2, 99), {}))

    # -------------------------------------------------------------------
    # Saturating-cast defect #4 (2026-08-02): `f64 as isize` in the old
    # `parse_insert_obj` SATURATED (NaN->0, out-of-range->isize::MIN/MAX)
    # instead of matching numpy's own behaviour for a floating `obj`.
    # `np.insert` treats a scalar `obj` as a SLICE bound (`TypeError` when
    # in-bounds, `IndexError` naming the ORIGINAL `obj` when out of
    # bounds), NOT a fancy index like `np.delete` does -- every case below
    # was measured live against numpy 2.5.1, `.venv/bin/python`,
    # fresh-process scripts, `except BaseException`.
    # -------------------------------------------------------------------
    out.append(("float_ordinary_frac_raises", (x, 2.7, 99), {}))
    out.append(("float_negative_frac_raises", (x, -1.5, 99), {}))
    out.append(("float_zero_raises", (x, 0.0, 99), {}))
    out.append(("float_exact_integral_raises", (x, 2.0, 99), {}))
    out.append(("float_exact_integral_neg_raises", (x, -2.0, 99), {}))
    out.append(("float_nan_raises", (x, float("nan"), 99), {}))
    out.append(("float_inf_raises", (x, float("inf"), 99), {}))
    out.append(("float_neg_inf_raises", (x, float("-inf"), 99), {}))
    out.append(("float_huge_pos_raises", (x, 1e30, 99), {}))
    out.append(("float_huge_neg_raises", (x, -1e30, 99), {}))
    out.append(("float_exact_2pow63_raises", (x, float(2**63), 99), {}))
    out.append(("float_exact_neg_2pow63_raises", (x, float(-(2**63)), 99), {}))
    out.append(("float_just_below_2pow63_raises", (x, float(2**63 - 1024), 99), {}))
    # NOTE: np.float64(...)/np.float32(...) bare-scalar `obj` forms are
    # deliberately NOT covered here -- `anionpy.array(np.float64(2.0))` itself
    # already raises `TypeError: 'float64' object is not an instance of
    # 'ndarray'` (verified live), a pre-existing gap in `ndarray_from_numpy`
    # (ionp-py/src/lib.rs, out of this task's edit scope) unrelated to the
    # saturating-cast defect this task fixes -- see final report.
    out.append(("float_at_pos_bound_raises", (x, float(len(x)), 99), {}))
    out.append(("float_at_neg_bound_raises", (x, -float(len(x)), 99), {}))
    out.append(("float_just_past_pos_bound_raises", (x, float(len(x)) + 1e-7, 99), {}))
    out.append(("float_just_past_neg_bound_raises", (x, -float(len(x)) - 1e-7, 99), {}))
    out.append(("float_with_axis_raises", (m, 2.7, 99), {"axis": 0}))
    out.append(("float_inf_with_axis_raises", (m, float("inf"), 99), {"axis": 0}))
    out.append(("float_oob_with_axis_raises", (m, 5.0, 99), {"axis": 0}))
    out.append(("float_with_axis_none_raises", (m, 2.7, 99), {}))
    out.append(("float_oob_with_axis_none_raises", (m, 50.0, 99), {}))
    out.append(("float_list_mixed_frac_raises", (x, [1.5, 2.5], 99), {}))
    out.append(("float_list_all_integral_raises", (x, [1.0, 2.0], 99), {}))
    out.append(("float_array_single_frac_raises", (x, np.array([1.5]), 99), {}))
    out.append(("float_list_single_nan_raises", (x, [float("nan")], 99), {}))
    out.append(("float_list_single_inf_raises", (x, [float("inf")], 99), {}))
    out.append(("int_list_ok", (x, [1, 2], 99), {}))
    out.append(("int_array_ok", (x, np.array([1, 2]), 99), {}))
    out.append(("bool_array_ok", (x, np.array([True, False, True, False, True, False, True, False, True]), 99), {}))
    out.append(("empty_list_ok", (x, [], 99), {}))
    out.append(("empty_float_array_raises", (x, np.array([]), 99), {}))
    out.append(("empty_float_array_typed_raises", (x, np.array([], dtype=float), 99), {}))
    out.append(("empty_int_array_ok", (x, np.array([], dtype=np.intp), 99), {}))
    out.append(("int_scalar_ok_control", (x, 1, 99), {}))
    out.append(("int_scalar_neg_ok_control", (x, -1, 99), {}))
    return out


def delete_cases():
    out = []
    x = np.arange(9)
    m = np.arange(12).reshape(3, 4)
    out.append(("scalar_idx", (x, 2), {}))
    out.append(("scalar_idx_neg", (x, -1), {}))
    out.append(("list_idx", (x, [1, 3, 5]), {}))
    out.append(("slice_obj", (x, slice(1, 4)), {}))
    out.append(("bool_mask", (x, x % 2 == 0), {}))
    out.append(("axis0", (m, 1), {"axis": 0}))
    out.append(("axis1", (m, [0, 2]), {"axis": 1}))
    out.append(("no_axis_ravels", (m, 2), {}))
    out.append(("oob_raises", (x, 100), {}))
    out.append(("bool_badlen_raises", (x, np.array([True, False])), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (x.astype(dtype), 2), {}))
    out.append(("form_list_1d", ([1, 2, 3, 4, 5], 2), {}))
    out.append(("form_range_1d", (range(6), 2), {}))

    # -------------------------------------------------------------------
    # Saturating-cast defect #4 (2026-08-02): same root cause as
    # `insert`'s cases above (`f64 as isize` saturating instead of
    # matching numpy), but `np.delete` treats ANY non-int/non-bool `obj`
    # -- scalar or array, any ndim, any size -- as a fancy INDEX ARRAY,
    # not a slice bound: real numpy's fancy-index-assignment machinery
    # rejects a floating dtype UNCONDITIONALLY (never a bounds check, no
    # TypeError branch, no per-magnitude tier -- axis/size never even
    # appear in the message). Every case measured live against numpy
    # 2.5.1, `.venv/bin/python`, fresh-process scripts.
    # -------------------------------------------------------------------
    out.append(("float_ordinary_frac_raises", (x, 2.7), {}))
    out.append(("float_negative_frac_raises", (x, -1.5), {}))
    out.append(("float_zero_raises", (x, 0.0), {}))
    out.append(("float_exact_integral_raises", (x, 2.0), {}))
    out.append(("float_exact_integral_neg_raises", (x, -2.0), {}))
    out.append(("float_nan_raises", (x, float("nan")), {}))
    out.append(("float_inf_raises", (x, float("inf")), {}))
    out.append(("float_neg_inf_raises", (x, float("-inf")), {}))
    out.append(("float_huge_pos_raises", (x, 1e30), {}))
    out.append(("float_huge_neg_raises", (x, -1e30), {}))
    out.append(("float_exact_2pow63_raises", (x, float(2**63)), {}))
    out.append(("float_exact_neg_2pow63_raises", (x, float(-(2**63))), {}))
    out.append(("float_just_below_2pow63_raises", (x, float(2**63 - 1024)), {}))
    # NOTE: np.float64(...)/np.float32(...) bare-scalar `obj` forms are
    # deliberately NOT covered here -- same pre-existing `ndarray_from_numpy`
    # gap noted in `insert_cases` above, out of this task's edit scope.
    out.append(("float_with_axis_raises", (m, 2.7), {"axis": 0}))
    out.append(("float_inf_with_axis_raises", (m, float("inf")), {"axis": 0}))
    out.append(("float_oob_val_with_axis_raises", (m, 50.0), {"axis": 0}))
    out.append(("float_with_axis_none_raises", (m, 2.7), {}))
    out.append(("float_list_mixed_frac_raises", (x, [1.5, 2.5]), {}))
    out.append(("float_list_all_integral_raises", (x, [1.0, 2.0]), {}))
    out.append(("float_array_single_frac_raises", (x, np.array([1.5])), {}))
    out.append(("float_list_single_nan_raises", (x, [float("nan")]), {}))
    out.append(("float_list_single_inf_raises", (x, [float("inf")]), {}))
    out.append(("float_2d_array_raises", (x, np.array([[1.5, 2.5]])), {}))
    out.append(("int_list_ok", (x, [1, 2]), {}))
    out.append(("int_array_ok", (x, np.array([1, 2])), {}))
    out.append(("empty_list_ok", (x, []), {}))
    out.append(("empty_float_array_raises", (x, np.array([])), {}))
    out.append(("empty_float_array_typed_raises", (x, np.array([], dtype=float)), {}))
    out.append(("empty_int_array_ok", (x, np.array([], dtype=np.intp)), {}))
    out.append(("int_scalar_ok_control", (x, 1), {}))
    out.append(("int_scalar_neg_ok_control", (x, -1), {}))
    return out


def append_cases():
    out = []
    x = np.arange(9)
    m = np.arange(12).reshape(3, 4)
    out.append(("flatten_no_axis", (x, [9, 10]), {}))
    out.append(("axis0", (m, [[1, 2, 3, 4]]), {"axis": 0}))
    out.append(("axis1", (m, [[1], [2], [3]]), {"axis": 1}))
    out.append(("axis_mismatch_raises", (m, [[1, 2, 3]]), {"axis": 0}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (x.astype(dtype), np.array([9, 10]).astype(dtype)), {}))
    # FORM axis (2026-08-02, CLASS A closure task): `arr` itself as a raw
    # Python list/range (append's `values` was already list-form-covered
    # by every case above -- it never touches `np.array()` at all here).
    out.append(("form_list_1d", ([1, 2, 3], [9, 10]), {}))
    out.append(("form_range_1d", (range(5), [9, 10]), {}))
    return out


def resize_cases():
    out = []
    x = np.arange(9)
    out.append(("smaller", (x, (2, 2)), {}))
    out.append(("larger_cyclic", (x, (3, 4)), {}))
    out.append(("same_size_reshape", (x, (3, 3)), {}))
    out.append(("scalar_shape", (x, 5), {}))
    out.append(("empty_input", (np.array([]), (3, 3)), {}))
    out.append(("neg_shape_raises", (x, (-1, 2)), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (x.astype(dtype), (3, 4)), {}))
    out.append(("form_list_1d", ([1, 2, 3, 4, 5], (2, 2)), {}))
    out.append(("form_range_1d", (range(9), (3, 4)), {}))
    return out


# ---------------------------------------------------------------------------
# rot90 / rollaxis
# ---------------------------------------------------------------------------

def rot90_cases():
    out = []
    m = np.arange(16).reshape(4, 4)
    d = np.arange(24).reshape(2, 3, 4)
    out.append(("default", (m,), {}))
    out.append(("k2", (m, 2), {}))
    out.append(("k3", (m, 3), {}))
    out.append(("k_neg1", (m, -1), {}))
    out.append(("k4_noop", (m, 4), {}))
    out.append(("explicit_axes", (m, 1, (1, 0)), {}))
    out.append(("3d_axes02", (d, 1, (0, 2)), {}))
    out.append(("axes_equal_raises", (m, 1, (0, 0)), {}))
    out.append(("axes_oob_raises", (m, 1, (0, 5)), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (m.astype(dtype),), {}))
    out.append(("form_list2d", ([[1, 2], [3, 4]],), {}))
    return out


def rollaxis_cases():
    out = []
    d = np.arange(24).reshape(2, 3, 4)
    out.append(("default_start0", (d, 2), {}))
    out.append(("explicit_start", (d, 2, 1), {}))
    out.append(("neg_axis", (d, -1), {}))
    out.append(("axis_eq_start_noop", (d, 0, 0), {}))
    out.append(("axis_oob_raises", (d, 5), {}))
    out.append(("start_oob_raises", (d, 0, 10), {}))
    out.append(("start_neg_oob_raises", (d, 0, -10), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (d.astype(dtype), 2), {}))
    # FORM axis, error-matching (2026-08-02): real numpy's `rollaxis` starts
    # with an unconditional STRICT `n = a.ndim` -- no lenient asarray
    # fallback at all, unlike hsplit/dsplit/vsplit's own preliminary
    # lenient check -- so a raw list/tuple/range genuinely raises
    # AttributeError uniformly, for every ndim. See ionp-py/src/manip.rs's
    # `rollaxis`, which reproduces the same unconditional strict gate.
    out.append(("form_list1d", ([1, 2, 3, 4], 0), {}))
    out.append(("form_list3d", ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 0), {}))
    out.append(("form_range", (range(4), 0), {}))
    return out


# ---------------------------------------------------------------------------
# grid/index construction: diag_indices[_from], tril_indices/triu_indices,
# ix_, indices, meshgrid, ravel_multi_index/unravel_index, fill_diagonal.
# (2026-08-01 pass; ionp-core/src/manip.rs and ionp-py/src/manip.rs are this
# same task's other two files.) All of diag_indices/diag_indices_from/
# tril_indices/triu_indices/ix_/meshgrid/unravel_index/indices(sparse=True)
# return a real numpy `tuple` already (confirmed via `type(...)` against
# real numpy 2.5.1) -- no tuple-wrapping adapter needed, just
# multi_output=True directly on the resolved function.
# ---------------------------------------------------------------------------

def diag_indices_cases():
    out = []
    out.append(("default_ndim2", (4,), {}))
    out.append(("ndim3", (3, 3), {}))
    out.append(("ndim1", (5, 1), {}))
    out.append(("n_zero", (0,), {}))
    out.append(("n_negative", (-2,), {}))
    # 2026-08-02: saturating-cast family regression coverage. `n` as a
    # float ultimately bottoms out in `np.arange(n)`'s own length
    # computation (`np.diag_indices`'s source is a bare `np.arange(n)`), so
    # these probe the SAME axes as `np.arange` itself: non-finite, beyond
    # `isize`/C-long range (both directions -- real numpy errors on a
    # hugely-negative float too, it does not short-circuit to an empty
    # range the way a naive `bound <= 0.0 -> 0` clamp would), the exact
    # float64 value `2**63` (numpy's raw-double range check and its
    # separate byte-size check land on this one double from opposite
    # sides, giving a THIRD, distinct message), and ordinary fractional/
    # negative/zero floats that stay in range.
    out.append(("n_float_nan", (float("nan"),), {}))
    out.append(("n_float_inf", (float("inf"),), {}))
    out.append(("n_float_neg_inf", (float("-inf"),), {}))
    out.append(("n_float_huge", (1e30,), {}))
    out.append(("n_float_huge_negative", (-1e30,), {}))
    out.append(("n_float_two_pow_63", (9223372036854775808.0,), {}))
    out.append(("n_float_frac", (2.7,), {}))
    out.append(("n_float_negative", (-1.0,), {}))
    out.append(("n_float_zero", (0.0,), {}))
    return out


def diag_indices_from_cases():
    out = []
    out.append(("square_2d", (np.arange(9).reshape(3, 3),), {}))
    out.append(("cube_3d", (np.arange(27).reshape(3, 3, 3),), {}))
    out.append(("non_square_raises", (np.arange(12).reshape(3, 4),), {}))
    out.append(("1d_raises", (np.arange(5),), {}))
    out.append(("unequal_3d_raises", (np.arange(24).reshape(2, 3, 4),), {}))
    return out


def tril_indices_cases():
    out = []
    out.append(("square_k0", (4,), {}))
    out.append(("k_pos", (4, 1), {}))
    out.append(("k_neg", (4, -1), {}))
    out.append(("rect_m", (3, 0, 5), {}))
    out.append(("rect_m_tall", (5, 0, 3), {}))
    out.append(("n_zero", (0,), {}))
    out.append(("n_negative", (-1,), {}))
    out.append(("m_negative", (3, 0, -1), {}))
    out.append(("k_beyond_range", (3, 10), {}))
    out.append(("k_beyond_neg_range", (3, -10), {}))
    # 2026-08-02: saturating-cast family regression coverage -- see
    # `diag_indices_cases`'s comment above for the full rationale. Both
    # `n` and `m` route through the same `size_arg_to_ceil_isize` ->
    # `arange_len_from_zero` path when float-valued.
    out.append(("n_float_nan", (float("nan"),), {}))
    out.append(("n_float_inf", (float("inf"),), {}))
    out.append(("n_float_neg_inf", (float("-inf"),), {}))
    out.append(("n_float_huge", (1e30,), {}))
    out.append(("n_float_huge_negative", (-1e30,), {}))
    out.append(("n_float_two_pow_63", (9223372036854775808.0,), {}))
    out.append(("n_float_frac", (2.7,), {}))
    out.append(("m_float_nan", (3, 0, float("nan")), {}))
    out.append(("m_float_inf", (3, 0, float("inf")), {}))
    out.append(("m_float_huge", (3, 0, 1e30), {}))
    return out


def triu_indices_cases():
    out = []
    out.append(("square_k0", (4,), {}))
    out.append(("k_pos", (4, 1), {}))
    out.append(("k_neg", (4, -1), {}))
    out.append(("rect_m", (3, 0, 5), {}))
    out.append(("rect_m_tall", (5, 0, 3), {}))
    out.append(("n_zero", (0,), {}))
    out.append(("n_negative", (-1,), {}))
    out.append(("m_negative", (3, 0, -1), {}))
    out.append(("k_beyond_range", (3, 10), {}))
    out.append(("k_beyond_neg_range", (3, -10), {}))
    # 2026-08-02: saturating-cast family regression coverage -- see
    # `diag_indices_cases`'s comment above for the full rationale.
    out.append(("n_float_nan", (float("nan"),), {}))
    out.append(("n_float_inf", (float("inf"),), {}))
    out.append(("n_float_neg_inf", (float("-inf"),), {}))
    out.append(("n_float_huge", (1e30,), {}))
    out.append(("n_float_huge_negative", (-1e30,), {}))
    out.append(("n_float_two_pow_63", (9223372036854775808.0,), {}))
    out.append(("n_float_frac", (2.7,), {}))
    out.append(("m_float_nan", (3, 0, float("nan")), {}))
    out.append(("m_float_inf", (3, 0, float("inf")), {}))
    out.append(("m_float_huge", (3, 0, 1e30), {}))
    return out


def ix__cases():
    out = []
    out.append(("two_1d", (np.array([1, 3]), np.array([0, 2, 4])), {}))
    out.append(("three_1d", (np.array([0, 1]), np.array([2, 3]), np.array([1])), {}))
    out.append(("bool_mask", (np.array([True, False, True]), np.array([0, 1])), {}))
    out.append(("mixed_bool_int", (np.array([True, True, False]), np.array([2, 0])), {}))
    out.append(("single_array", (np.array([1, 2, 3]),), {}))
    out.append(("empty_array", (np.array([], dtype=np.int64),), {}))
    out.append(("2d_raises", (np.arange(4).reshape(2, 2),), {}))
    return out


def indices_cases():
    out = []
    out.append(("2d_dense", ((3, 4),), {}))
    out.append(("2d_sparse", ((3, 4),), {"sparse": True}))
    out.append(("1d", ((5,),), {}))
    out.append(("3d_dense", ((2, 3, 2),), {}))
    out.append(("3d_sparse", ((2, 3, 2),), {"sparse": True}))
    out.append(("dtype_float", ((3, 3),), {"dtype": np.float64}))
    out.append(("dtype_bare_int", ((3, 3),), {"dtype": int}))
    out.append(("empty_dims", ((),), {}))
    out.append(("zero_dim", ((0,),), {}))
    out.append(("list_dimensions", ([2, 3],), {}))
    return out


def meshgrid_cases():
    out = []
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    z = np.array([6, 7])
    out.append(("two_xy_default", (x, y), {}))
    out.append(("two_ij", (x, y), {"indexing": "ij"}))
    out.append(("two_sparse_xy", (x, y), {"sparse": True}))
    out.append(("two_sparse_ij", (x, y), {"sparse": True, "indexing": "ij"}))
    out.append(("three_xy", (x, y, z), {}))
    out.append(("three_ij", (x, y, z), {"indexing": "ij"}))
    out.append(("three_sparse_ij", (x, y, z), {"sparse": True, "indexing": "ij"}))
    out.append(("single_array", (x,), {}))
    out.append(("copy_false", (x, y), {"copy": False}))
    out.append(("invalid_indexing_raises", (x, y), {"indexing": "bogus"}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (x.astype(dtype), y.astype(dtype)), {}))
    return out


def ravel_multi_index_cases():
    out = []
    out.append(("basic_2d", ((np.array([1, 2]), np.array([0, 3])), (3, 4)), {}))
    out.append(("order_f", ((np.array([1, 2]), np.array([0, 3])), (3, 4)), {"order": "F"}))
    out.append(("mode_wrap", ((np.array([-1, 5]), np.array([0, 3])), (3, 4)), {"mode": "wrap"}))
    out.append(("mode_clip", ((np.array([-1, 5]), np.array([0, 3])), (3, 4)), {"mode": "clip"}))
    out.append(("mode_per_dim", ((np.array([-1, 5]), np.array([0, 10])), (3, 4)), {"mode": ("wrap", "clip")}))
    out.append(("mode_raise_oob", ((np.array([5]), np.array([0])), (3, 4)), {}))
    out.append(("dims_list", ((np.array([1]), np.array([2])), [3, 4]), {}))
    out.append(("bool_coords", ((np.array([True, False]), np.array([0, 1])), (2, 2)), {}))
    out.append(("float_coords_raises", ((np.array([1.0]), np.array([0.0])), (3, 4)), {}))
    out.append(("wrong_length_raises", ((np.array([1]),), (3, 4)), {}))
    out.append(("3d", ((np.array([1]), np.array([2]), np.array([0])), (2, 3, 4)), {}))
    return out


def unravel_index_cases():
    out = []
    out.append(("basic", (np.array([1, 5, 11]), (3, 4)), {}))
    out.append(("order_f", (np.array([1, 5, 11]), (3, 4)), {"order": "F"}))
    out.append(("scalar_int", (7, (3, 4)), {}))
    out.append(("shape_list", (np.array([1, 5]), [3, 4]), {}))
    out.append(("3d", (np.array([1, 23]), (2, 3, 4)), {}))
    out.append(("out_of_bounds_raises", (np.array([12]), (3, 4)), {}))
    out.append(("negative_raises", (np.array([-1]), (3, 4)), {}))
    return out


def fill_diagonal_cases():
    out = []
    out.append(("square_scalar", (np.zeros((4, 4), dtype=np.int64), 7), {}))
    out.append(("square_array_exact_len", (np.zeros((3, 3), dtype=np.int64), [1, 2, 3]), {}))
    out.append(("square_array_cyclic_tile", (np.zeros((4, 4), dtype=np.int64), [1, 2]), {}))
    out.append(("rect_tall_no_wrap", (np.zeros((5, 3), dtype=np.int64), 9), {}))
    out.append(("rect_tall_wrap_true", (np.zeros((5, 3), dtype=np.int64), 9), {"wrap": True}))
    out.append(("rect_wide", (np.zeros((3, 5), dtype=np.int64), 9), {}))
    out.append(("batched_3d_equal_dims", (np.zeros((3, 3, 3), dtype=np.int64), 5), {}))
    out.append(("1d_raises", (np.zeros(4, dtype=np.int64), 1), {}))
    out.append(("3d_unequal_raises", (np.zeros((2, 3, 4), dtype=np.int64), 1), {}))
    for dtype in SWEEP_DTYPES:
        out.append((f"dtype/{np.dtype(dtype).name}", (np.zeros((3, 3), dtype=dtype), 5), {}))
    return out


# ---------------------------------------------------------------------------
# take / put / take_along_axis / put_along_axis / compress (2026-08-01,
# third pass -- the Tier-1 gather/scatter block). take/take_along_axis/
# compress are pure (adapters just call the real function through, same as
# every other item above); put/put_along_axis mutate their first argument
# in place and return None, so they get the same copy-first/assert-None
# adapter pattern `_fill_diagonal_adapters` established above. take/
# compress's `out=` support additionally gets its own dedicated case
# functions below (`take_out_cases`/`compress_out_cases`) with a bespoke
# adapter that allocates a caller-owned buffer via `anionpy.empty`/`np.empty`
# (never `anionpy.asarray(np_array)`, which would silently copy instead of
# sharing storage) and asserts same-object identity on return, per this
# task's out= verification requirement.
# ---------------------------------------------------------------------------

def _take_adapters():
    """Single adapter pair for the WHOLE `take` item (plain calls AND
    `out=` calls) -- `ItemSpec`/`resolve_ionp` supports exactly one
    adapter per item, so the plain-call cases and the `out=`-identity
    cases below both flow through this one pair rather than two separate
    items. Plain cases (no `_out_shape` kwarg) just forward to
    `np.take`/`anionpy.take` unchanged -- `_wrap_custom_conversion` (see
    registry.py) already auto-converts any numpy-origin positional/keyword
    argument reaching the anionpy side, so `a`/`indices` arrive pre-converted
    to real `anionpy.ndarray`s here.

    `out=` cases (an `_out_shape` kwarg present) instead allocate the
    caller-owned target buffer THEMSELVES, via `np.empty`/`anionpy.empty`
    (never `anionpy.asarray(some_numpy_array)`, which would copy instead of
    sharing storage -- this task's explicit out= verification
    requirement), then assert `result is out` before returning the
    mutated buffer for the harness's normal value comparison -- verified
    live that `np.take(..., out=out) is out`, unsafe dtype casting is
    allowed (float->int truncates), and a shape mismatch raises the fixed
    `ValueError: output array does not match result of ndarray.take`
    text (`_bad_out="shape"`/`_bad_out="type"` cases exercise that path,
    routed through unconditionally so the raised exception itself is what
    gets compared, not a value).
    """

    def numpy_adapter(a, indices, axis=None, mode="raise", _out_shape=None, _out_dtype=None, _bad_out=None):
        if _bad_out == "type":
            return np.take(a, indices, axis=axis, mode=mode, out=[1, 2, 3])
        if _out_shape is not None:
            shape = (3,) if _bad_out == "shape" else _out_shape
            out = np.empty(shape, dtype=_out_dtype)
            result = np.take(a, indices, axis=axis, mode=mode, out=out)
            if _bad_out is None:
                assert result is out, "numpy take out= must return the same object"
            return out
        return np.take(a, indices, axis=axis, mode=mode)

    def ionp_adapter(a, indices, axis=None, mode="raise", _out_shape=None, _out_dtype=None, _bad_out=None):
        import anionpy
        if _bad_out == "type":
            return anionpy.take(a, indices, axis=axis, mode=mode, out=[1, 2, 3])
        if _out_shape is not None:
            shape = (3,) if _bad_out == "shape" else _out_shape
            out = anionpy.empty(shape, dtype=_out_dtype)
            result = anionpy.take(a, indices, axis=axis, mode=mode, out=out)
            if _bad_out is None:
                assert result is out, "anionpy take out= must return the same object"
            return out
        return anionpy.take(a, indices, axis=axis, mode=mode)

    return numpy_adapter, ionp_adapter


def take_cases():
    out = []
    a = np.arange(24).reshape(2, 3, 4)
    out.append(("axis0", (a, [0, 1]), {"axis": 0}))
    out.append(("axis1", (a, [0, 2, 1]), {"axis": 1}))
    out.append(("axis2", (a, [-1, 0]), {"axis": 2}))
    out.append(("axis_none_flattens", (a, [0, 5, 23]), {}))
    out.append(("negative_axis", (a, [0, 1]), {"axis": -1}))
    out.append(("repeated_indices", (a, [0, 0, 1, 1]), {"axis": 0}))
    out.append(("bool_indices_cast_like_int", (a, np.array([True, False])), {"axis": 0}))
    out.append(("mode_wrap_negative", (a, [-1, -10]), {"axis": 0, "mode": "wrap"}))
    out.append(("mode_wrap_oob_positive", (a, [10]), {"axis": 0, "mode": "wrap"}))
    out.append(("mode_clip_negative", (a, [-5]), {"axis": 0, "mode": "clip"}))
    out.append(("mode_clip_oob_positive", (a, [10]), {"axis": 0, "mode": "clip"}))
    out.append(("mode_raise_negative_one_wrap_ok", (a, [-1]), {"axis": 0, "mode": "raise"}))
    out.append(("mode_raise_oob_positive_raises", (a, [5]), {"axis": 0}))
    out.append(("mode_raise_oob_negative_raises", (a, [-10]), {"axis": 0}))
    out.append(("mode_bogus_raises", (a, [0]), {"axis": 0, "mode": "bogus"}))
    out.append(("axis_oob_positive_raises", (a, [0]), {"axis": 5}))
    out.append(("axis_oob_negative_raises", (a, [0]), {"axis": -5}))
    out.append(("empty_indices_nonempty_axis", (a, np.array([], dtype=np.int64)), {"axis": 0}))
    out.append(("empty_axis_empty_indices_ok", (np.empty((0, 3)), np.array([], dtype=np.int64)), {"axis": 0}))
    out.append(("empty_axis_nonempty_indices_raises", (np.empty((0, 3)), np.array([0])), {"axis": 0}))
    out.append(("float_indices_type_error", (a, np.array([0.5])), {"axis": 0}))
    out.append(("0d_array_axis0", (np.array(5), [0]), {"axis": 0}))
    out.append(("0d_array_axis_neg1", (np.array(5), [0]), {"axis": -1}))
    out.append(("0d_array_axis1_raises", (np.array(5), [0]), {"axis": 1}))
    out.append(("0d_array_axis_none", (np.array(5), [0, 0]), {}))
    out.append(("single_element", (np.array([42]), [0, 0, 0]), {"axis": 0}))
    out.append(("2d_indices_shape", (a, np.array([[0, 1], [1, 0]])), {"axis": 0}))
    for dtype in SWEEP_DTYPES:
        d = np.arange(12).reshape(3, 4).astype(dtype) if np.dtype(dtype).kind != "b" \
            else (np.arange(12).reshape(3, 4) % 2).astype(dtype)
        out.append((f"dtype/{np.dtype(dtype).name}", (d, [0, 2]), {"axis": 0}))
    # out= cases, routed through `_take_adapters`'s `_out_shape`/`_out_dtype`/
    # `_bad_out` sentinel kwargs (stripped by the adapter, never real numpy
    # kwargs) -- verified live: exact shape required, unsafe dtype casting
    # allowed, same-identity return, non-array `out=` raises TypeError.
    a2 = np.arange(12).reshape(3, 4).astype(np.int32)
    out.append(("out_exact_dtype_match", (a2, [0, 2]), {"axis": 0, "_out_shape": (2, 4), "_out_dtype": np.int32}))
    out.append(("out_unsafe_upcast_to_float64", (a2, [0, 2]), {"axis": 0, "_out_shape": (2, 4), "_out_dtype": np.float64}))
    out.append((
        "out_unsafe_downcast_truncates",
        (a2.astype(np.float64) + 0.9, [0, 2]),
        {"axis": 0, "_out_shape": (2, 4), "_out_dtype": np.int32},
    ))
    out.append(("out_axis_none", (a2, [0, 5, 11]), {"_out_shape": (3,), "_out_dtype": np.int64}))
    out.append(("out_wrong_shape_raises", (a2, [0, 2]), {"axis": 0, "_out_shape": (3,), "_out_dtype": np.int32, "_bad_out": "shape"}))
    out.append(("out_non_array_raises", (a2, [0, 2]), {"axis": 0, "_bad_out": "type"}))
    out += _take_index_ingestion_cases()
    out += _take_empty_axis_matrix_cases()
    return out


def _take_index_ingestion_cases():
    """numpy does not INFER a dtype for a `take` index argument and then cast
    it -- it builds the array with `intp` REQUESTED, so an untyped Python
    sequence is coerced elementwise through Python's own `int()`, while an
    argument that is already an array keeps its dtype and faces the ordinary
    `same_kind` cast check. The two routes disagree on the same input:
    `a.take([])` is a valid empty intp index, `a.take(np.array([]))` is a
    float64 array and a TypeError.

    A numpy SCALAR takes the coercion route too (it is not an array), which is
    why `np.float64(0.5)` succeeds where `np.array(0.5)` does not -- and why
    the 0-d array's message says "Cannot cast scalar from" rather than
    "Cannot cast array data from". That noun is chosen by ndim, verified live
    across take/put/choose/repeat.

    These operands are deliberately raw Python lists/tuples and numpy scalars:
    `make_ionp_array_converter` passes both through unconverted (it converts
    only `numpy.ndarray`), so these cells genuinely exercise anionpy's own
    untyped-sequence ingestion rather than a pre-converted anionpy array.
    """
    out = []
    a = np.arange(4, dtype=np.int8)
    seqs = [
        ("empty_list", []), ("empty_tuple", ()), ("nested_empty", [[]]),
        ("nested_empty_2", [[], []]), ("int_list", [0, 2]), ("tuple_ints", (0, 2)),
        ("float_list_truncates", [0.5]), ("float_list_zero", [0.0]),
        ("float_list_negative", [-0.5]), ("bool_list", [True, False]),
        ("nested_int_list", [[0, 1], [1, 0]]), ("bare_int", 0),
        ("bare_float_truncates", 1.9), ("bare_bool", True), ("bare_negative", -1),
    ]
    for name, idx in seqs:
        out.append((f"ingest/{name}", (a, idx), {}))
        out.append((f"ingest/{name}/axis0", (a, idx), {"axis": 0}))
    # Scalars vs 0-d arrays: the whole point of the distinction.
    for name, idx in [
        ("np_float64", np.float64(0.5)), ("np_int8", np.int8(1)),
        ("np_bool", np.bool_(True)), ("np_int64", np.int64(2)),
    ]:
        out.append((f"ingest/scalar/{name}", (a, idx), {}))
    for name, idx in [
        ("0d_float_cast_error", np.array(0.5)),
        ("0d_complex_cast_error", np.array(0.5 + 0j)),
        ("1d_float_cast_error", np.array([0.5])),
        ("2d_float_cast_error", np.array([[0.5]])),
        ("0d_int_ok", np.array(1)),
        ("1d_empty_int_ok", np.array([], dtype=np.int64)),
        ("1d_empty_float_cast_error", np.array([], dtype=np.float64)),
    ]:
        out.append((f"ingest/array/{name}", (a, idx), {}))
    return out


def _take_empty_axis_matrix_cases():
    """Index validation is gated on the OUTER block count,
    `prod(shape[:axis])`: numpy checks the indices inside a loop over those
    blocks, so when that product is 0 no index is ever examined, however wild.
    When it is non-zero every index is checked even though the result is empty
    because some INNER axis has length 0 -- which is where anionpy used to return
    an empty array instead of raising.

    Separately, the static "cannot do a non-empty take from an empty axes."
    message is numpy's only when the RESULT would have been non-empty; if the
    result is empty anyway, numpy falls through to the ordinary per-index path
    (out-of-bounds text against a size of 0 under `raise`, plain success under
    `clip`).

    `mode='wrap'` against an EMPTY take axis is deliberately absent: numpy
    2.5.1 infinite-loops on it (reproduced on
    `np.zeros((0, 0)).take([0], axis=0, mode='wrap')`), so it has no reference
    answer, and including it would hang this suite rather than fail it.
    """
    out = []
    shapes = [(3, 0, 2), (0, 3, 2), (3, 2, 0), (0, 0), (3, 0), (0, 3), (1, 0, 5), (0,)]
    idxs = [("in_range", [0]), ("oob_pos", [8]), ("oob_neg", [-8]),
            ("mixed", [0, 3]), ("empty", np.array([], dtype=np.int64))]
    for shape in shapes:
        base = np.zeros(shape, dtype=np.int8)
        for axis in list(range(len(shape))) + [None]:
            axlen = base.size if axis is None else shape[axis]
            for iname, idx in idxs:
                tag = f"empty_axis/{'x'.join(map(str, shape))}/ax{axis}/{iname}"
                out.append((tag, (base, idx), {"axis": axis}))
                out.append((f"{tag}/clip", (base, idx), {"axis": axis, "mode": "clip"}))
                if axlen != 0:
                    out.append((f"{tag}/wrap", (base, idx), {"axis": axis, "mode": "wrap"}))
    return out


def _put_adapters():
    """`put` mutates its first argument in place and returns `None`
    (confirmed live) -- same copy-first/assert-None pattern as
    `_fill_diagonal_adapters` above, since `run_case` hands the SAME frozen
    `args` tuple to both sides and a raw in-place call would let one side
    observe the other's mutation.
    """

    def numpy_adapter(a, ind, v, mode="raise"):
        target = a.copy()
        result = np.put(target, ind, v, mode=mode)
        assert result is None, "numpy put must return None"
        return target

    def ionp_adapter(a, ind, v, mode="raise"):
        import anionpy
        target = anionpy.array(a) if isinstance(a, np.ndarray) else a.copy()
        result = anionpy.put(target, ind, v, mode=mode)
        assert result is None, "anionpy put must return None"
        return target

    return numpy_adapter, ionp_adapter


def put_cases():
    out = []
    a = np.arange(10)
    out.append(("basic", (a, [0, 2], [99, 88]), {}))
    out.append(("scalar_value_broadcasts", (a, [0, 1, 2], 7), {}))
    out.append(("repeated_indices_last_wins", (a, [0, 0], [1, 2]), {}))
    out.append(("mode_wrap_oob", (a, [15], [1]), {"mode": "wrap"}))
    out.append(("mode_wrap_negative", (a, [-15], [1]), {"mode": "wrap"}))
    out.append(("mode_clip_oob", (a, [15], [1]), {"mode": "clip"}))
    out.append(("mode_raise_oob_raises", (a, [15], [1]), {}))
    out.append(("mode_bogus_raises", (a, [0], [1]), {"mode": "bogus"}))
    out.append(("empty_indices_noop", (a, np.array([], dtype=np.int64), np.array([], dtype=np.int64)), {}))
    out.append(("empty_target_raises", (np.empty((0,)), [0], [1]), {}))
    out.append(("2d_target_flat_indices", (np.arange(12).reshape(3, 4), [0, 5, 11], [100, 200, 300]), {}))
    out.append(("2d_target_negative_flat_index", (np.arange(12).reshape(3, 4), [-1], [999]), {}))
    for dtype in SWEEP_DTYPES:
        d = np.arange(10).astype(dtype) if np.dtype(dtype).kind != "b" else (np.arange(10) % 2).astype(dtype)
        out.append((f"dtype/{np.dtype(dtype).name}", (d, [0, 2], [1, 1]), {}))
    return out


def take_along_axis_cases():
    out = []
    a = np.arange(12).reshape(3, 4)
    out.append(("axis1_basic", (a, np.array([[0, 1], [2, 0], [1, 3]])), {"axis": 1}))
    out.append(("axis0_basic", (a, np.array([[0, 1, 2, 0]])), {"axis": 0}))
    out.append(("default_axis_neg1", (a, np.array([[0], [1], [2]])), {}))
    out.append(("negative_axis", (a, np.array([[0], [1], [2]])), {"axis": -1}))
    out.append(("broadcast_single_row_idx", (a, np.array([[0, 1, 2, 3]])), {"axis": 0}))
    out.append(("axis_none_flatten", (a, np.array([0, 5, 11])), {}))
    out.append(("axis_none_2d_indices_raises", (a, np.array([[0, 1]])), {"axis": None}))
    out.append(("bool_indices_raise", (a, np.array([[True, False, True, False]])), {"axis": 0}))
    out.append(("float_indices_raise", (a, np.array([[0.0, 1.0, 2.0, 3.0]])), {"axis": 0}))
    out.append(("ndim_mismatch_raises", (a, np.zeros((3, 4, 1), dtype=np.int64)), {"axis": 1}))
    out.append(("broadcast_shape_conflict_raises", (a, np.zeros((3, 3), dtype=np.int64)), {"axis": 0}))
    out.append(("axis_oob_raises", (a, np.array([[0]])), {"axis": 5}))
    out.append(("0d_array_raises", (np.array(5), np.array([0])), {"axis": 0}))
    out.append(("negative_index_value", (a, np.array([[-1, -2, -3, -4]])), {"axis": 0}))
    for dtype in SWEEP_DTYPES:
        d = np.arange(12).reshape(3, 4).astype(dtype) if np.dtype(dtype).kind != "b" \
            else (np.arange(12).reshape(3, 4) % 2).astype(dtype)
        out.append((f"dtype/{np.dtype(dtype).name}", (d, np.array([[0, 1, 2, 0]])), {"axis": 0}))
    return out


def _put_along_axis_adapters():
    """`put_along_axis` mutates its first argument in place and returns
    `None` -- same copy-first/assert-None pattern as `_put_adapters` above.
    `axis` is a required positional in real numpy (no default), reproduced
    here unchanged.
    """

    def numpy_adapter(arr, indices, values, axis):
        target = arr.copy()
        result = np.put_along_axis(target, indices, values, axis)
        assert result is None, "numpy put_along_axis must return None"
        return target

    def ionp_adapter(arr, indices, values, axis):
        import anionpy
        target = anionpy.array(arr) if isinstance(arr, np.ndarray) else arr.copy()
        result = anionpy.put_along_axis(target, indices, values, axis)
        assert result is None, "anionpy put_along_axis must return None"
        return target

    return numpy_adapter, ionp_adapter


def put_along_axis_cases():
    out = []
    a = np.arange(12).reshape(3, 4)
    out.append(("axis1_basic", (a, np.array([[0], [1], [2]]), np.array([[100], [200], [300]]), 1)))
    out.append(("axis0_basic", (a, np.array([[0, 1, 2, 0]]), np.array([[9, 8, 7, 6]]), 0)))
    out.append(("negative_axis", (a, np.array([[0], [1], [2]]), np.array([[100], [200], [300]]), -1)))
    out.append(("scalar_values_broadcast", (a, np.array([[0], [1], [2]]), 77, 1)))
    out.append(("axis_none_flatten", (a, np.array([0, 5, 11]), np.array([100, 200, 300]), None)))
    out.append(("axis_none_2d_indices_raises", (a, np.array([[0, 1]]), np.array([[1, 2]]), None)))
    out.append(("ndim_mismatch_raises", (a, np.zeros((3, 4, 1), dtype=np.int64), np.zeros((3, 4, 1)), 1)))
    out.append(("bool_indices_raise", (a, np.array([[True, False, True, False]]), np.array([[1, 2, 3, 4]]), 0)))
    out.append(("index_oob_raises", (a, np.array([[10]]), np.array([[1]]), 0)))
    out.append(("axis_oob_raises", (a, np.array([[0]]), np.array([[1]]), 5)))
    # value-shape-mismatch boundary (own bespoke ValueError text, distinct
    # from `broadcast_to`'s generic message and from `take_along_axis`'s
    # IndexError -- live-measured against numpy 2.5.1, see
    # `shape_broadcastable_to` in ionp-core/src/manip.rs). indices shape
    # (3,1) against a (3,4) target -> indexing result shape (3,1).
    out.append(("value_shape_too_wide_raises", (a, np.array([[0], [1], [2]]), np.zeros((3, 4), dtype=a.dtype), 1)))
    out.append(("value_shape_wrong_nonunit_raises", (a, np.array([[0], [1], [2]]), np.zeros((3, 2), dtype=a.dtype), 1)))
    out.append(("value_shape_extra_dim_raises", (a, np.array([[0], [1], [2]]), np.zeros((3, 1, 1), dtype=a.dtype), 1)))
    out.append(("value_shape_1d_wrong_ndim_raises", (a, np.array([[0], [1], [2]]), np.zeros((3,), dtype=a.dtype), 1)))
    out.append(("value_shape_broadcastable_unit_ok", (a, np.array([[0], [1], [2]]), np.array([[77]]), 1)))
    normalized = [(label, args, {}) for (label, args) in out]
    for dtype in SWEEP_DTYPES:
        d = np.arange(12).reshape(3, 4).astype(dtype) if np.dtype(dtype).kind != "b" \
            else (np.arange(12).reshape(3, 4) % 2).astype(dtype)
        normalized.append((
            f"dtype/{np.dtype(dtype).name}",
            (d, np.array([[0, 1, 2, 0]]), np.array([[1, 1, 1, 1]]).astype(dtype), 0),
            {},
        ))
    return normalized


def _compress_adapters():
    """Single adapter pair for the WHOLE `compress` item -- mirrors
    `_take_adapters` above (same `_out_shape`/`_out_dtype`/`_bad_out`
    sentinel-kwarg design, same reasoning: exactly one adapter per
    `ItemSpec`, so plain calls and out= calls share it). Verified live:
    `np.compress(..., out=out) is out`, exact shape required, unsafe dtype
    casting allowed.
    """

    def numpy_adapter(condition, a, axis=None, _out_shape=None, _out_dtype=None, _bad_out=None):
        if _bad_out == "type":
            return np.compress(condition, a, axis=axis, out=[1, 2, 3])
        if _out_shape is not None:
            shape = (3,) if _bad_out == "shape" else _out_shape
            out = np.empty(shape, dtype=_out_dtype)
            result = np.compress(condition, a, axis=axis, out=out)
            if _bad_out is None:
                assert result is out, "numpy compress out= must return the same object"
            return out
        return np.compress(condition, a, axis=axis)

    def ionp_adapter(condition, a, axis=None, _out_shape=None, _out_dtype=None, _bad_out=None):
        import anionpy
        if _bad_out == "type":
            return anionpy.compress(condition, a, axis=axis, out=[1, 2, 3])
        if _out_shape is not None:
            shape = (3,) if _bad_out == "shape" else _out_shape
            out = anionpy.empty(shape, dtype=_out_dtype)
            result = anionpy.compress(condition, a, axis=axis, out=out)
            if _bad_out is None:
                assert result is out, "anionpy compress out= must return the same object"
            return out
        return anionpy.compress(condition, a, axis=axis)

    return numpy_adapter, ionp_adapter


def compress_cases():
    out = []
    a = np.arange(12).reshape(3, 4)
    out.append(("axis0_basic", (np.array([True, False, True]), a), {"axis": 0}))
    out.append(("axis1_basic", (np.array([True, False, True, False]), a), {"axis": 1}))
    out.append(("negative_axis", (np.array([True, False, True]), a), {"axis": -1}))
    out.append(("axis_none_flattens", (np.array([True, False] * 6), a), {}))
    out.append(("all_false_empty_result", (np.array([False, False, False]), a), {"axis": 0}))
    out.append(("condition_longer_than_axis_raises", (np.array([True] * 5), a), {"axis": 0}))
    out.append(("condition_shorter_than_axis_ok", (np.array([True]), a), {"axis": 0}))
    out.append(("condition_not_1d_raises", (np.array([[True]]), a), {"axis": 0}))
    out.append(("axis_oob_raises", (np.array([True]), a), {"axis": 5}))
    out.append(("0d_array_axis0", (np.array([True]), np.array(5)), {"axis": 0}))
    out.append(("int_condition_truthy", (np.array([1, 0, 2]), a), {"axis": 0}))
    for dtype in SWEEP_DTYPES:
        d = np.arange(12).reshape(3, 4).astype(dtype) if np.dtype(dtype).kind != "b" \
            else (np.arange(12).reshape(3, 4) % 2).astype(dtype)
        out.append((f"dtype/{np.dtype(dtype).name}", (np.array([True, False, True]), d), {"axis": 0}))
    a2 = np.arange(12).reshape(3, 4).astype(np.int32)
    cond2 = np.array([True, False, True])
    out.append(("out_exact_dtype_match", (cond2, a2), {"axis": 0, "_out_shape": (2, 4), "_out_dtype": np.int32}))
    out.append(("out_unsafe_upcast_to_float64", (cond2, a2), {"axis": 0, "_out_shape": (2, 4), "_out_dtype": np.float64}))
    out.append(("out_wrong_shape_raises", (cond2, a2), {"axis": 0, "_out_shape": (3,), "_out_dtype": np.int32, "_bad_out": "shape"}))
    out.append(("out_non_array_raises", (cond2, a2), {"axis": 0, "_bad_out": "type"}))
    out += _compress_empty_axis_matrix_cases()
    return out


def _compress_empty_axis_matrix_cases():
    """`compress` IS a `take` over the selected positions, so it inherits
    `take`'s outer-block gating exactly: a condition longer than the axis is an
    out-of-bounds index, and whether numpy notices depends on whether
    `prod(shape[:axis])` is non-zero. anionpy silently returned an empty array
    (with the wrong length, at that) for the cases where numpy raises.

    Kept as its own generator rather than folded into the take matrix because
    the failure is worth seeing under this item's own name -- these two items
    have one root cause and can regress together, and a reader who only greps
    `compress` should still find the guard.
    """
    out = []
    shapes = [(3, 0, 2), (0, 3, 2), (3, 2, 0), (0, 0), (3, 0), (0, 3), (1, 0, 5)]
    for shape in shapes:
        base = np.zeros(shape, dtype=np.int8)
        for axis in list(range(len(shape))) + [None]:
            for cname, cond in [
                ("empty", []), ("one_true", [True]), ("all_false_3", [False] * 3),
                ("long_4", [True] * 4), ("long_9", [True] * 9),
                ("alternating_6", [True, False] * 3),
            ]:
                out.append((
                    f"empty_axis/{'x'.join(map(str, shape))}/ax{axis}/{cname}",
                    (np.array(cond, dtype=bool), base), {"axis": axis},
                ))
    return out


def _indices_adapters():
    """`indices` returns a single stacked array when `sparse=False` (the
    default) but a tuple of `ndim` arrays when `sparse=True` (confirmed
    against real numpy) -- one spec can't honestly declare a bare
    `multi_output` flag for both return shapes (same tension the module
    docstring documents for atleast_1d/2d/3d's single-array-vs-tuple call
    forms). These adapters normalize BOTH sides identically: wrap a plain
    array result in a 1-tuple, pass an already-returned tuple through
    unchanged -- so `indices_cases()` can exercise dense AND sparse in the
    same automated item via `multi_output=True` without favoring either
    shape.
    """

    def _norm(x):
        return x if isinstance(x, tuple) else (x,)

    def numpy_adapter(*args, **kwargs):
        return _norm(np.indices(*args, **kwargs))

    def ionp_adapter(*args, **kwargs):
        import anionpy
        return _norm(anionpy.indices(*args, **kwargs))

    return numpy_adapter, ionp_adapter


def _fill_diagonal_adapters():
    """`fill_diagonal` mutates its first argument in place and returns
    `None` (confirmed via real numpy). The harness's `run_case`/`_call`
    hands the SAME frozen `args` tuple to both the numpy call and the anionpy
    call (see registry.py's `build_ufunc_dispatcher` docstring for why that
    matters), so calling the real function directly here would mutate the
    corpus array in place for one side and leave the other reading already-
    mutated data. These adapters copy the target array first (mirroring the
    `.at()` mutation-safety pattern in registry.py), assert the real
    contract (return value is None) on EACH side independently -- so if
    either implementation ever stopped returning None that side would raise
    and be reported as a mismatch rather than silently pass -- and hand back
    the mutated copy for the harness's normal value comparison.
    """

    def numpy_adapter(a, val, wrap=False):
        target = a.copy()
        result = np.fill_diagonal(target, val, wrap=wrap)
        assert result is None, "numpy fill_diagonal must return None"
        return target

    def ionp_adapter(a, val, wrap=False):
        import anionpy
        target = anionpy.array(a) if isinstance(a, np.ndarray) else a.copy()
        result = anionpy.fill_diagonal(target, val, wrap=wrap)
        assert result is None, "anionpy fill_diagonal must return None"
        return target

    return numpy_adapter, ionp_adapter


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _build_manip_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["concatenate"] = ItemSpec(name="concatenate", kind="custom", custom_cases=concatenate_cases)
    specs["stack"] = ItemSpec(name="stack", kind="custom", custom_cases=stack_cases)
    specs["hstack"] = ItemSpec(name="hstack", kind="custom", custom_cases=hstack_cases)
    specs["vstack"] = ItemSpec(name="vstack", kind="custom", custom_cases=vstack_cases)
    # row_stack: intentionally NOT registered as a differential item.
    # numpy_surface.json declares it, but real numpy 2.5.1 (this repo's
    # pinned reference/venv version) has already REMOVED `np.row_stack`
    # entirely (`AttributeError: module 'numpy' has no attribute
    # 'row_stack'`, confirmed directly in this venv) -- there is no numpy
    # reference left to differentially compare against. anionpy's own
    # `row_stack` (a plain alias of `vstack`, see manip.rs) is still built
    # and importable, but is NOT declared exact in toplevel.py since it
    # cannot be verified the way every other declared item is.
    specs["dstack"] = ItemSpec(name="dstack", kind="custom", custom_cases=dstack_cases)
    specs["column_stack"] = ItemSpec(name="column_stack", kind="custom", custom_cases=column_stack_cases)

    specs["flip"] = ItemSpec(name="flip", kind="custom", custom_cases=flip_cases)
    specs["fliplr"] = ItemSpec(name="fliplr", kind="custom", custom_cases=fliplr_cases)
    specs["flipud"] = ItemSpec(name="flipud", kind="custom", custom_cases=flipud_cases)

    specs["roll"] = ItemSpec(name="roll", kind="custom", custom_cases=roll_cases)
    specs["tile"] = ItemSpec(name="tile", kind="custom", custom_cases=tile_cases)
    specs["repeat"] = ItemSpec(name="repeat", kind="custom", custom_cases=repeat_cases)

    specs["broadcast_shapes"] = ItemSpec(
        name="broadcast_shapes", kind="custom", custom_cases=broadcast_shapes_cases,
        scalar_like=True,
    )
    specs["broadcast_arrays"] = ItemSpec(
        name="broadcast_arrays", kind="custom", custom_cases=broadcast_arrays_cases,
        multi_output=True,
    )

    specs["atleast_1d"] = ItemSpec(name="atleast_1d", kind="custom", custom_cases=atleast_1d_cases)
    specs["atleast_2d"] = ItemSpec(name="atleast_2d", kind="custom", custom_cases=atleast_2d_cases)
    specs["atleast_3d"] = ItemSpec(name="atleast_3d", kind="custom", custom_cases=atleast_3d_cases)

    specs["diag"] = ItemSpec(name="diag", kind="custom", custom_cases=diag_cases)
    specs["diagflat"] = ItemSpec(name="diagflat", kind="custom", custom_cases=diagflat_cases)
    specs["diagonal"] = ItemSpec(name="diagonal", kind="custom", custom_cases=diagonal_cases)
    specs["tril"] = ItemSpec(name="tril", kind="custom", custom_cases=tril_cases)
    specs["triu"] = ItemSpec(name="triu", kind="custom", custom_cases=triu_cases)
    specs["trace"] = ItemSpec(name="trace", kind="custom", custom_cases=trace_cases)

    _split_np, _split_ionp = _split_family_adapters("split")
    specs["split"] = ItemSpec(
        name="split", kind="custom", custom_cases=split_cases,
        numpy_adapter=_split_np, ionp_adapter=_split_ionp, multi_output=True,
    )
    _asplit_np, _asplit_ionp = _split_family_adapters("array_split")
    specs["array_split"] = ItemSpec(
        name="array_split", kind="custom", custom_cases=array_split_cases,
        numpy_adapter=_asplit_np, ionp_adapter=_asplit_ionp, multi_output=True,
    )
    _hsplit_np, _hsplit_ionp = _split_family_adapters("hsplit")
    specs["hsplit"] = ItemSpec(
        name="hsplit", kind="custom", custom_cases=hsplit_cases,
        numpy_adapter=_hsplit_np, ionp_adapter=_hsplit_ionp, multi_output=True,
    )
    _vsplit_np, _vsplit_ionp = _split_family_adapters("vsplit")
    specs["vsplit"] = ItemSpec(
        name="vsplit", kind="custom", custom_cases=vsplit_cases,
        numpy_adapter=_vsplit_np, ionp_adapter=_vsplit_ionp, multi_output=True,
    )
    _dsplit_np, _dsplit_ionp = _split_family_adapters("dsplit")
    specs["dsplit"] = ItemSpec(
        name="dsplit", kind="custom", custom_cases=dsplit_cases,
        numpy_adapter=_dsplit_np, ionp_adapter=_dsplit_ionp, multi_output=True,
    )

    specs["insert"] = ItemSpec(name="insert", kind="custom", custom_cases=insert_cases)
    specs["delete"] = ItemSpec(name="delete", kind="custom", custom_cases=delete_cases)
    specs["append"] = ItemSpec(name="append", kind="custom", custom_cases=append_cases)
    specs["resize"] = ItemSpec(name="resize", kind="custom", custom_cases=resize_cases)
    specs["rot90"] = ItemSpec(name="rot90", kind="custom", custom_cases=rot90_cases)
    specs["rollaxis"] = ItemSpec(name="rollaxis", kind="custom", custom_cases=rollaxis_cases)

    specs["diag_indices"] = ItemSpec(
        name="diag_indices", kind="custom", custom_cases=diag_indices_cases,
        multi_output=True,
    )
    specs["diag_indices_from"] = ItemSpec(
        name="diag_indices_from", kind="custom", custom_cases=diag_indices_from_cases,
        multi_output=True,
    )
    specs["tril_indices"] = ItemSpec(
        name="tril_indices", kind="custom", custom_cases=tril_indices_cases,
        multi_output=True,
    )
    specs["triu_indices"] = ItemSpec(
        name="triu_indices", kind="custom", custom_cases=triu_indices_cases,
        multi_output=True,
    )
    specs["ix_"] = ItemSpec(
        name="ix_", kind="custom", custom_cases=ix__cases,
        multi_output=True,
    )
    _indices_np, _indices_ionp = _indices_adapters()
    specs["indices"] = ItemSpec(
        name="indices", kind="custom", custom_cases=indices_cases,
        numpy_adapter=_indices_np, ionp_adapter=_indices_ionp, multi_output=True,
    )
    specs["meshgrid"] = ItemSpec(
        name="meshgrid", kind="custom", custom_cases=meshgrid_cases,
        multi_output=True,
    )
    specs["ravel_multi_index"] = ItemSpec(
        name="ravel_multi_index", kind="custom", custom_cases=ravel_multi_index_cases,
    )
    specs["unravel_index"] = ItemSpec(
        name="unravel_index", kind="custom", custom_cases=unravel_index_cases,
        multi_output=True,
    )
    _fd_np, _fd_ionp = _fill_diagonal_adapters()
    specs["fill_diagonal"] = ItemSpec(
        name="fill_diagonal", kind="custom", custom_cases=fill_diagonal_cases,
        numpy_adapter=_fd_np, ionp_adapter=_fd_ionp,
    )

    _take_np, _take_ionp = _take_adapters()
    specs["take"] = ItemSpec(
        name="take", kind="custom", custom_cases=take_cases,
        numpy_adapter=_take_np, ionp_adapter=_take_ionp,
    )
    _put_np, _put_ionp = _put_adapters()
    specs["put"] = ItemSpec(
        name="put", kind="custom", custom_cases=put_cases,
        numpy_adapter=_put_np, ionp_adapter=_put_ionp,
    )
    specs["take_along_axis"] = ItemSpec(
        name="take_along_axis", kind="custom", custom_cases=take_along_axis_cases,
    )
    _pal_np, _pal_ionp = _put_along_axis_adapters()
    specs["put_along_axis"] = ItemSpec(
        name="put_along_axis", kind="custom", custom_cases=put_along_axis_cases,
        numpy_adapter=_pal_np, ionp_adapter=_pal_ionp,
    )
    _compress_np, _compress_ionp = _compress_adapters()
    specs["compress"] = ItemSpec(
        name="compress", kind="custom", custom_cases=compress_cases,
        numpy_adapter=_compress_np, ionp_adapter=_compress_ionp,
    )

    return specs


MANIP_SPECS = _build_manip_specs()


# ---------------------------------------------------------------------------
# 0-d x explicit-axis boundary for `repeat` / `ndarray.repeat` (2026-08-03).
#
# The INVERSE of the reduction-family defect closed the same day: there anionpy
# ACCEPTED axes numpy rejects, here it REJECTED an axis numpy accepts.
# numpy promotes a 0-d operand to shape (1,) BEFORE validating `axis`, so
# `repeat(np.array(3.0), 2, axis=0)` returns shape (2,) and an out-of-range
# axis reports "array of dimension 1" -- never "dimension 0".
#
# Sweeps `repeats` as well as `axis`, because `repeats=0` and `repeats=3`
# exercise different branches of the target-length resolution than the
# `repeats=2` a single spot-check would have used, and an out-of-range axis
# must raise identically for every one of them.
# ---------------------------------------------------------------------------

def _repeat_zero_d_axis_cases():
    cases = []
    for dlabel, val in [("float64", 3.0), ("float64_nan", np.nan),
                        ("int", 7), ("bool", True)]:
        for alabel, ax in [("axis_omitted", "OMIT"), ("axis_none", None),
                           ("axis_0", 0), ("axis_neg1", -1), ("axis_1", 1),
                           ("axis_neg2", -2), ("axis_2", 2)]:
            for reps in (0, 1, 2, 3):
                kwargs = {} if ax == "OMIT" else {"axis": ax}
                cases.append((f"zero_d/{dlabel}/{alabel}/reps{reps}",
                              (np.array(val), reps), kwargs))
    return cases


def _repeat_append_zero_d_axis_cases():
    spec = specs.get("repeat") if "specs" in globals() else None
    if spec is None:
        spec = MANIP_SPECS.get("repeat")
    assert spec is not None, "0-d boundary: no spec for 'repeat'"
    prev = spec.extra_cases

    def wrapped(prev=prev):
        base = list(prev()) if prev is not None else []
        return base + _repeat_zero_d_axis_cases()

    spec.extra_cases = wrapped


_repeat_append_zero_d_axis_cases()
