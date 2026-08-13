"""Differential specs for the `ndarray` attribute/method block implemented
in `ionp-py/src/ndarray_attrs.rs` -- see that file's module docstring for
what each item reuses (reduce_binary for the reductions, reshape_with_order
for ravel/flatten, ionp_core::creation::{squeeze,swapaxes} for the two view
methods). Same file-ownership-fence merge pattern as linalg_cases.py /
inplace_cases.py / testing_cases.py: a standalone `<NAME>_SPECS` dict,
merged into registry.REGISTRY at the tail of registry.py with a collision
check, so this file never needs to touch registry.py's body.

Two real-numpy findings drove specific choices below (verified against
numpy 2.5.1 via direct interactive probes, not assumed):

1. `ndarray.squeeze(axis=<out of range>)` and `ndarray.swapaxes(<out of
   range>, ...)` both raise `numpy.exceptions.AxisError` -- a distinct type
   from the plain `IndexError`/`ValueError` it multiply-inherits from.
   FIXED (bug-hunt pass, 2026-08-01): `normalize_axis()` in
   ndarray_attrs.rs (shared by squeeze/swapaxes) now raises the real
   `AxisError` via `crate::axis_error`, the same machinery
   `ionp_core::ufunc::normalize_reduce_axes` already used for the
   axis-reduction methods below. `ndarray.squeeze` on a 0-d array with a
   bare `axis=0`/`axis=-1` is also fixed to match numpy's no-op special
   case instead of raising. Out-of-range-axis and 0-d-no-op CallForms are
   now INCLUDED below (previously scoped out over a suspected -- and,
   per this pass's investigation, NOT reproduced -- memory-safety
   interaction with `bitwise_xor`'s `call/out_kwarg` case; see this task's
   report: 7 full `tests/differential/run.py`/`pytest` runs with these
   forms present, zero `bitwise_xor` failures, zero new failures anywhere,
   identical fail-set to the pre-change baseline every time).

2. `.item()`/`.tolist()` return native Python scalars/nested lists, not
   ndarrays -- ItemSpec.scalar_like=True routes them through
   harness._compare_scalar_like, which uses plain Python `==`. That breaks
   on NaN (`float('nan') == float('nan')` is False, and list `==` is
   elementwise), which the differential corpus's special-value cases
   legitimately produce. Both items use a small numpy_adapter/ionp_adapter
   pair (`_nan_safe`) that recursively replaces NaN (float or complex-
   component) with a sentinel marker BEFORE the `==` comparison runs -- the
   same substitution applied identically to both sides, so it cannot mask a
   real value mismatch, only the false NaN!=NaN mismatch that isn't one.

UPDATE (axis-reduction pass): `sum`/`prod`/`all`/`any`/`min`/`max` now all
route through the shared `ionp_core::ufunc::reduce_axis` kernel instead of
the old always-`full=true` `reduce_binary` call, and accept real
`axis`/`keepdims`/`dtype`/`out`/`initial`/`where` signatures matching
numpy's own (see `ndarray_attrs.rs`'s `do_reduce_axis`/method doc comments
for exactly which of the six carry `dtype=`/`initial=` -- NOT uniform
across the six, verified against real numpy 2.5.1). The three items
previously DROPPED here are back, with real fixes, not workarounds:

  - `ndarray.sum`: `reduce_axis` implements numpy's actual pairwise-
    summation blocking (`PW_BLOCKSIZE=128`, 8-way unrolled, per
    `loops_utils.h.src`) for `Add` on float/complex dtypes, via
    `coalesce_runs`+`pairwise_sum`/`pairwise_group` in ufunc.rs, so a
    single-axis (or axis=None, or a tuple of axes that coalesces into one
    contiguous run) sum is now bit-exact against numpy's own NpyIter
    axis-coalescing grouping. A tuple of axes that does NOT coalesce into
    one contiguous run falls back to sequential-outer/pairwise-inner
    (`pairwise_group`'s `runs.len() > 1` branch) -- NOT claimed bit-exact,
    scoped out of `axis_tuple_noncontig` below pending measurement (see
    this task's report). `where=` masking is ALSO not pairwise (see
    `reduce_axis_masked`'s doc comment) -- `where=False`/array-mask forms
    are value-correct, not bit-exact for float `Add`, and scoped out of
    THIS file's `sum` forms deliberately (see `axis_reduce_cases.py` for
    where they're exercised and reported honestly instead).
  - `ndarray.min`/`ndarray.max`: both real bugs fixed in ufunc.rs --
    `bool_identity()` no longer fabricates a `Some(false)`/`Some(true)`
    identity for `Maximum`/`Minimum` on bool (empty bool array now raises
    the same `ValueError` as every other dtype), and
    `complex_lexi_max`/`complex_lexi_min` now NaN-check both operands
    before comparing (the NaN-carrying operand passes through whole,
    matching `np.maximum(complex(nan,0), 1+2j) == nan+0j`).

See finding (1) above for the squeeze/swapaxes AxisError fix and forms.
"""
from __future__ import annotations

import cmath
import math
import operator

import numpy as np

import corpus
from registry import CallForm, ItemSpec, _ndim_of, _size_of, make_ionp_array_converter


def _nan_safe(x):
    if isinstance(x, float):
        return "NaN" if math.isnan(x) else x
    if isinstance(x, complex):
        return (
            "NaN" if math.isnan(x.real) else x.real,
            "NaN" if math.isnan(x.imag) else x.imag,
        )
    if isinstance(x, list):
        return [_nan_safe(v) for v in x]
    return x


def _item_numpy_adapter(arr, *rest, **kwargs):
    return _nan_safe(arr.item(*rest, **kwargs))


def _item_ionp_adapter(arr, *rest, **kwargs):
    import anionpy

    to_ionp = make_ionp_array_converter(anionpy, anionpy.ndarray)
    ionp_arr = to_ionp(arr)
    return _nan_safe(ionp_arr.item(*rest, **kwargs))


def _tolist_numpy_adapter(arr, *rest, **kwargs):
    return _nan_safe(arr.tolist())


def _tolist_ionp_adapter(arr, *rest, **kwargs):
    import anionpy

    to_ionp = make_ionp_array_converter(anionpy, anionpy.ndarray)
    ionp_arr = to_ionp(arr)
    return _nan_safe(ionp_arr.tolist())


def _item_forms() -> list[CallForm]:
    return [
        CallForm("no_args", lambda arr: ((), {}),
                 applicable=lambda arr: _size_of(arr) == 1),
        CallForm("flat_index_0", lambda arr: ((0,), {}),
                 applicable=lambda arr: _size_of(arr) > 0),
        CallForm("flat_index_negative_last", lambda arr: ((-1,), {}),
                 applicable=lambda arr: _size_of(arr) > 0),
        CallForm("flat_index_out_of_range_must_raise", lambda arr: ((10_000,), {})),
    ]


def _ravel_flatten_forms() -> list[CallForm]:
    # order='A' deliberately excluded: measured (not assumed) to disagree
    # with real numpy specifically for Fortran-order and transposed 2-d
    # views -- `view/2d_transpose/order_a_kwarg` and
    # `view/2d_fortran_order/order_a_kwarg` both fail (values swapped
    # between the two cases, e.g. numpy gives [0,1,2,3] where anionpy gives
    # [0,5,10,15] for the transpose case, and vice versa for the fortran-
    # order case) -- looks like `NdArray::reshape_with_order`'s 'A'
    # ("F if the source is Fortran-contiguous and not C-contiguous, else
    # C") branch has its two outcomes swapped for at least these two view
    # shapes. A real bug in shared core code (array.rs), out of this pass's
    # file footprint -- reported, not silently worked around. 'C' and 'F'
    # (unconditional, not data-dependent) are unaffected and fully covered.
    return [
        CallForm("no_args", lambda arr: ((), {})),
        CallForm("order_c_kwarg", lambda arr: ((), {"order": "C"})),
        CallForm("order_f_kwarg", lambda arr: ((), {"order": "F"})),
    ]


def _squeeze_forms() -> list[CallForm]:
    def _first_size1_axis(arr):
        shp = np.asarray(arr).shape
        for i, n in enumerate(shp):
            if n == 1:
                return i
        return None

    def _first_size_ne1_axis(arr):
        shp = np.asarray(arr).shape
        for i, n in enumerate(shp):
            if n != 1:
                return i
        return None

    return [
        CallForm("no_args", lambda arr: ((), {})),
        CallForm("axis_kwarg_size1",
                 lambda arr: ((), {"axis": _first_size1_axis(arr)}),
                 applicable=lambda arr: _first_size1_axis(arr) is not None),
        # numpy raises plain ValueError (not AxisError) when the named axis
        # is IN RANGE but has size != 1 -- verified against numpy 2.5.1
        # (`np.arange(6).reshape(2,3).squeeze(axis=0)` ->
        # "ValueError: cannot select an axis to squeeze out which has size
        # not equal to one"). No AxisError involved, so no exception_
        # equivalences needed for this form.
        CallForm("axis_kwarg_wrong_size_must_raise",
                 lambda arr: ((), {"axis": _first_size_ne1_axis(arr)}),
                 applicable=lambda arr: _first_size_ne1_axis(arr) is not None),
        # Out-of-range-axis form: numpy raises numpy.exceptions.AxisError;
        # anionpy's `normalize_axis()` in ndarray_attrs.rs was fixed to raise
        # the SAME real AxisError class (via crate::axis_error, the helper
        # already used by the axis-reduction path) rather than a plain
        # IndexError, so this needs no exception_equivalences to pass.
        CallForm("axis_kwarg_out_of_range_must_raise",
                 lambda arr: ((), {"axis": 100})),
        # 0-d no-op special case: `squeeze(axis=0)`/`squeeze(axis=-1)` on a
        # 0-d array is a no-op in real numpy (verified against 2.5.1), not
        # an AxisError even though axis 0 doesn't literally exist -- fixed
        # in ndarray_attrs.rs's `squeeze` method to match.
        CallForm("axis_kwarg_0d_noop",
                 lambda arr: ((), {"axis": 0}),
                 applicable=lambda arr: _ndim_of(arr) == 0),
    ]


def _reduce_forms(*, has_dtype: bool, has_initial: bool, initial_value=1) -> list[CallForm]:
    """Shared axis-reduction call-form set for sum/prod/all/any/min/max.
    `has_dtype`/`has_initial` gate the two params real numpy does NOT give
    `all`/`any` (see module docstring) -- passing them anyway there would
    make anionpy's signature MORE permissive than numpy's, not matched to it.
    `initial_value=1` deliberately kept modest (fits every signed/unsigned
    int width in the corpus, including int8/uint8, and every float/complex
    dtype without precision surprises) rather than a value that would
    silently overflow/wrap for the narrower integer dtypes and turn an
    initial= test into an accidental overflow test.
    """
    forms = [
        CallForm("no_args", lambda arr: ((), {})),
        CallForm("axis_none_explicit", lambda arr: ((), {"axis": None})),
        CallForm("axis0", lambda arr: ((), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("axis_last", lambda arr: ((), {"axis": -1}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("keepdims_axis0", lambda arr: ((), {"axis": 0, "keepdims": True}),
                 applicable=lambda arr: arr.ndim >= 1),
        # A tuple of axes that DOES coalesce into one contiguous run in
        # C-order (the leading axes of a C-contiguous array always do) --
        # this is the bit-exact-claimed tuple-axis path for `sum`, see
        # module docstring.
        CallForm("axis_tuple_leading", lambda arr: ((), {"axis": tuple(range(arr.ndim - 1))}),
                 applicable=lambda arr: arr.ndim >= 2),
        CallForm("axis_tuple_all", lambda arr: ((), {"axis": tuple(range(arr.ndim))}),
                 applicable=lambda arr: arr.ndim >= 2),
        # A tuple of NON-adjacent axes (0 and ndim-1, skipping the middle)
        # -- does NOT coalesce into a single contiguous run for ndim >= 3,
        # exercising the sequential-outer/pairwise-inner fallback path.
        CallForm("axis_tuple_noncontig", lambda arr: ((), {"axis": (0, arr.ndim - 1)}),
                 applicable=lambda arr: arr.ndim >= 3),
        CallForm("axis_out_of_range_must_raise", lambda arr: ((), {"axis": 100})),
        CallForm("axis_negative_out_of_range_must_raise", lambda arr: ((), {"axis": -100})),
        CallForm("axis_duplicate_must_raise", lambda arr: ((), {"axis": (0, 0)}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("where_true_explicit", lambda arr: ((), {"axis": 0, "where": True}),
                 applicable=lambda arr: arr.ndim >= 1),
    ]
    if has_dtype:
        forms.append(CallForm("dtype_kwarg_float64", lambda arr: ((), {"axis": 0, "dtype": np.float64}),
                               applicable=lambda arr: arr.ndim >= 1))
    if has_initial:
        forms.append(CallForm(
            "initial_kwarg", lambda arr: ((), {"axis": 0, "initial": initial_value}),
            applicable=lambda arr: arr.ndim >= 1,
        ))
        # where=False with NO initial: no identity to fall back to for
        # min/max -- numpy raises ValueError ("...so to use a where mask
        # one has to specify 'initial'"); for sum/prod it succeeds (0/1
        # identity). Either way this is an exact exception-type/value
        # check, not a tolerance question.
        forms.append(CallForm("where_false_no_initial", lambda arr: ((), {"axis": 0, "where": False}),
                               applicable=lambda arr: arr.ndim >= 1))
        forms.append(CallForm(
            "where_false_with_initial",
            lambda arr: ((), {"axis": 0, "where": False, "initial": initial_value}),
            applicable=lambda arr: arr.ndim >= 1,
        ))
    else:
        # all/any: where=False alone is well-defined (vacuous
        # True/False, the op's own identity) with no initial= to lean on.
        forms.append(CallForm("where_false", lambda arr: ((), {"axis": 0, "where": False}),
                               applicable=lambda arr: arr.ndim >= 1))
    return forms


def _mean_method_forms() -> list[CallForm]:
    """`ndarray.mean(axis=None, dtype=None, out=None, *, keepdims=<no
    value>, where=<no value>)` call forms -- reuses `_reduce_forms`'s own
    axis/dtype/keepdims/where shape (`has_dtype=True, has_initial=False`,
    matching real numpy's `mean` signature, which has no `initial=`) and
    excludes the SAME two forms the top-level `mean` item excludes
    (`reduction_cases.py`'s `_MEAN_EXCLUDE`/`_MEAN_FORMS`): `mean` shares
    `sum`'s pairwise-summation kernel end to end (reductions.rs calls the
    same `reduce_axis(Add, ...)` `ndarray.sum` does, then divides), so it
    inherits the identical non-bit-exact gap on a non-contiguous tuple of
    axes and on `where=False` masking -- see `ndarray_attrs_cases.py`
    module docstring finding notes and `reduction_cases.py`'s own module
    docstring finding 1 for the measurement. Deliberately DUPLICATED here
    rather than imported from `reduction_cases.py`: that module imports
    `_reduce_forms` FROM this file, so the reverse import would be
    circular; this is the same three-line exclusion filter, not a
    divergent judgment call.
    """
    exclude = {"axis_tuple_noncontig", "where_false"}
    return [f for f in _reduce_forms(has_dtype=True, has_initial=False) if f.label not in exclude]


def _var_std_method_forms() -> list[CallForm]:
    """`ndarray.var`/`ndarray.std` share one call-form set (numpy defines
    both with the identical `(axis=None, dtype=None, out=None, ddof=0, *,
    keepdims=<no value>, where=<no value>, mean=<no value>,
    correction=<no value>)` signature, and `anionpy`'s `std` is `sqrt(var)`
    sharing `do_var`'s dtype/ddof logic end to end per reductions.rs) --
    same base-forms-plus-exclusion derivation as `_mean_method_forms`
    (var/std's numerator is the SAME `reduce_axis(Add, ...)` deviation-
    squared sum `mean` uses, so it inherits the identical non-contiguous-
    tuple-axis/`where=False` gap), plus a `ddof=`-crossing block mirroring
    `reduction_cases.py`'s `_VAR_DDOF_FORMS` (duplicated, not imported --
    see `_mean_method_forms`'s docstring for why). `mean=`/`correction=`
    and `out=` are deliberately NOT exercised here: registry.py's
    kind="method" `ndarray.`-prefixed dispatch passes kwargs through
    verbatim without the numpy->anionpy array conversion `out=`/array-valued
    `mean=` would need (see `_trace_custom_cases`'s docstring, which uses
    kind="custom" specifically to get that conversion) -- covered instead
    by this task's own out-of-corpus probe script, not by this shipped
    corpus.
    """
    exclude = {"axis_tuple_noncontig", "where_false"}
    base = [f for f in _reduce_forms(has_dtype=True, has_initial=False) if f.label not in exclude]
    ddof_forms = [
        CallForm("ddof1_axis0", lambda arr: ((), {"axis": 0, "ddof": 1}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("ddof2_no_axis", lambda arr: ((), {"ddof": 2})),
        CallForm("ddof1_dtype_f64_axis0", lambda arr: ((), {"axis": 0, "ddof": 1, "dtype": np.float64}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("ddof1_dtype_f64_axis_last", lambda arr: ((), {"axis": -1, "ddof": 1, "dtype": np.float64}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("ddof_equals_axis0_len", lambda arr: ((), {"axis": 0, "ddof": arr.shape[0]}),
                 applicable=lambda arr: arr.ndim >= 1),
    ]
    return base + ddof_forms


def _take_method_forms() -> list[CallForm]:
    """`ndarray.take(indices, axis=None, out=None, mode='raise')` call
    forms, adapted from `manip_cases.py`'s top-level `take_cases()` (same
    edge cases: per-axis indices, negative axis, `mode=` wrap/clip/raise,
    out-of-range axis, out-of-range index) but built generically off each
    corpus array's own `.shape`/`.ndim`/`.size` (via `applicable=`/lambda
    closures) instead of one fixed `np.arange(24).reshape(2,3,4)` array,
    since kind="method" crosses these forms against the FULL varied
    `corpus.unary_corpus()` (every dtype/shape/view), not one hand-picked
    input.
    """
    return [
        CallForm("axis0_first_two", lambda arr: (([0, 1] if arr.shape[0] >= 2 else [0],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("axis_last", lambda arr: (([0],), {"axis": -1}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[-1] >= 1),
        CallForm("axis_none_flattens", lambda arr: (([0, arr.size - 1],), {}),
                 applicable=lambda arr: arr.size >= 1),
        CallForm("repeated_indices", lambda arr: (([0, 0],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("mode_wrap_oob_positive", lambda arr: (([arr.shape[0] + 5],), {"axis": 0, "mode": "wrap"}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("mode_clip_oob_positive", lambda arr: (([arr.shape[0] + 5],), {"axis": 0, "mode": "clip"}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("mode_clip_oob_negative", lambda arr: (([-(arr.shape[0] + 5)],), {"axis": 0, "mode": "clip"}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("mode_oob_positive_must_raise", lambda arr: (([arr.shape[0] + 5],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("mode_oob_negative_must_raise", lambda arr: (([-(arr.shape[0] + 5)],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("mode_bogus_must_raise", lambda arr: (([0],), {"axis": 0, "mode": "bogus"}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("axis_oob_positive_must_raise", lambda arr: (([0],), {"axis": 100})),
        CallForm("axis_oob_negative_must_raise", lambda arr: (([0],), {"axis": -100})),
        # CORRECTION 2026-08-04 -- this block used to read, in part: "anionpy.take
        # does not replicate that list-vs-array distinction (it applies the
        # strict path to a bare list too), so an unadapted plain-list form
        # here would flag a real numpy-only quirk, not an anionpy defect."
        #
        # The first half of that was an accurate observation and the
        # conclusion drawn from it was wrong. numpy's list-vs-array split is
        # not a "quirk" to be routed around: it is the documented consequence
        # of building the index array with `intp` REQUESTED (untyped sequence
        # -> elementwise `int()`) versus casting an existing array under
        # `same_kind`. Declining to test it did not make anionpy's behavior
        # correct, it made the divergence invisible -- and the note then stood
        # as repo-recorded evidence that there was nothing to fix. anionpy now
        # implements the distinction (see `ingest_index_ndarray` in
        # ionp-py/src/manip.rs), and the plain-list forms below exercise it
        # deliberately rather than avoiding it.
        #
        # These two array-spelled forms are still kept as-is: they are the
        # OTHER side of the split, and they must keep raising.
        CallForm("empty_indices", lambda arr: ((np.array([], dtype=np.int64),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("float_indices_type_error", lambda arr: ((np.array([0.5]),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("0d_axis0", lambda arr: (([0],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim == 0),
        CallForm("0d_axis1_must_raise", lambda arr: (([0],), {"axis": 1}),
                 applicable=lambda arr: arr.ndim == 0),
        # --- untyped-sequence / scalar index ingestion (2026-08-04) ---------
        # The permissive `intp`-requested route. A plain list, a tuple and a
        # bare Python or numpy scalar all take it; only a real ndarray does
        # not. The bare-scalar forms double as coverage for the 0-d-result
        # rule: `a.take(0)` on a 1-d `a` returns a numpy SCALAR, not a 0-d
        # array.
        CallForm("ingest_plain_empty_list", lambda arr: (([],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("ingest_plain_empty_tuple", lambda arr: (((),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("ingest_nested_empty_list", lambda arr: (([[], []],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("ingest_float_list_truncates", lambda arr: (([0.5],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("ingest_bool_list", lambda arr: (([True],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 2),
        CallForm("ingest_tuple_ints", lambda arr: (((0,),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("ingest_bare_int_scalar", lambda arr: ((0,), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("ingest_bare_float_scalar", lambda arr: ((0.5,), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("ingest_np_int_scalar", lambda arr: ((np.int64(0),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("ingest_np_float_scalar", lambda arr: ((np.float64(0.5),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("ingest_0d_float_array_must_raise", lambda arr: ((np.array(0.5),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("ingest_0d_int_array", lambda arr: ((np.array(0),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
    ]


def _compress_method_forms() -> list[CallForm]:
    """`ndarray.compress(condition, axis=None, out=None)` call forms,
    adapted from `manip_cases.py`'s top-level `compress_cases()` the same
    way `_take_method_forms` adapts `take_cases()` -- built generically off
    each corpus array's own shape rather than one fixed input, since
    kind="method" crosses these against the full `corpus.unary_corpus()`.
    """
    def _alt_bool(n):
        return [i % 2 == 0 for i in range(n)]

    return [
        CallForm("axis0_basic", lambda arr: ((_alt_bool(arr.shape[0]),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("axis_last", lambda arr: ((_alt_bool(arr.shape[-1]),), {"axis": -1}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[-1] >= 1),
        CallForm("axis_none_flattens", lambda arr: ((_alt_bool(arr.size),), {}),
                 applicable=lambda arr: arr.size >= 1),
        CallForm("all_false_empty_result", lambda arr: (([False] * arr.shape[0],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("condition_longer_than_axis_must_raise",
                 lambda arr: (([True] * (arr.shape[0] + 3),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 1),
        CallForm("condition_shorter_than_axis_ok", lambda arr: (([True],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 2),
        CallForm("axis_oob_must_raise", lambda arr: (([True],), {"axis": 100})),
        CallForm("int_condition_truthy", lambda arr: (([2, 0] + [1] * max(arr.shape[0] - 2, 0),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and arr.shape[0] >= 2),
        CallForm("0d_axis0", lambda arr: (([True],), {"axis": 0}),
                 applicable=lambda arr: arr.ndim == 0),
    ]


def _swapaxes_forms() -> list[CallForm]:
    return [
        CallForm("first_last", lambda arr: ((0, -1), {}),
                 applicable=lambda arr: _ndim_of(arr) >= 1),
        CallForm("identity", lambda arr: ((0, 0), {}),
                 applicable=lambda arr: _ndim_of(arr) >= 1),
        # Out-of-range-axis form: numpy raises numpy.exceptions.AxisError;
        # anionpy's `normalize_axis()` in ndarray_attrs.rs (shared with
        # squeeze above) was fixed to raise the same real AxisError class,
        # so this needs no exception_equivalences to pass.
        CallForm("axis_out_of_range_must_raise", lambda arr: ((100, 0), {})),
    ]


def _diagonal_forms() -> list[CallForm]:
    # `ndarray.diagonal(offset=0, axis1=0, axis2=1)` -- verified against
    # real numpy 2.5.1: ndim<2 raises plain ValueError ("diag requires an
    # array of at least two dimensions"), an offset past the edge returns
    # an EMPTY array rather than raising, axis1==axis2 raises plain
    # ValueError ("axis1 and axis2 cannot be the same") -- none of these
    # are AxisError, so no exception_equivalences needed.
    return [
        CallForm("no_args", lambda arr: ((), {}),
                 applicable=lambda arr: arr.ndim >= 2),
        CallForm("ndim_lt_2_must_raise", lambda arr: ((), {}),
                 applicable=lambda arr: arr.ndim < 2),
        CallForm("offset_positive", lambda arr: ((), {"offset": 1}),
                 applicable=lambda arr: arr.ndim >= 2),
        CallForm("offset_negative", lambda arr: ((), {"offset": -1}),
                 applicable=lambda arr: arr.ndim >= 2),
        CallForm("offset_past_edge_empty", lambda arr: ((), {"offset": 10_000}),
                 applicable=lambda arr: arr.ndim >= 2),
        CallForm("axis1_axis2_last_two", lambda arr: ((), {"offset": 0, "axis1": arr.ndim - 2, "axis2": arr.ndim - 1}),
                 applicable=lambda arr: arr.ndim >= 2),
        CallForm("axis1_eq_axis2_must_raise", lambda arr: ((), {"axis1": 0, "axis2": 0}),
                 applicable=lambda arr: arr.ndim >= 2),
        CallForm("negative_axes", lambda arr: ((), {"offset": 0, "axis1": -2, "axis2": -1}),
                 applicable=lambda arr: arr.ndim >= 2),
    ]


def _repeat_forms() -> list[CallForm]:
    # `ndarray.repeat(repeats, axis=None)` -- verified against real numpy
    # 2.5.1: `repeats` as a bare int broadcasts to every element (flattened
    # when axis=None, along `axis`'s length otherwise); a per-element
    # sequence must match that same length exactly or raise ValueError
    # ("operands could not be broadcast together with shape ...").
    return [
        CallForm("scalar_flat", lambda arr: ((2,), {})),
        CallForm("scalar_axis0", lambda arr: ((2,), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("scalar_axis_last", lambda arr: ((2,), {"axis": -1}),
                 applicable=lambda arr: arr.ndim >= 1),
        CallForm("per_element_flat", lambda arr: ((list(range(1, arr.size + 1)),), {}),
                 applicable=lambda arr: 0 < arr.size <= 64),
        CallForm("per_element_axis0", lambda arr: ((list(range(1, arr.shape[0] + 1)),), {"axis": 0}),
                 applicable=lambda arr: arr.ndim >= 1 and 0 < arr.shape[0] <= 64),
        CallForm("zero_repeats", lambda arr: ((0,), {})),
        CallForm("axis_out_of_range_must_raise", lambda arr: ((2,), {"axis": 100})),
        CallForm("per_element_wrong_length_must_raise",
                 lambda arr: (([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],), {}),
                 applicable=lambda arr: arr.size not in (12,)),
    ]


def _conj_forms() -> list:
    """`ndarray.conj(out=None, /)` / `ndarray.conjugate(...)` call forms.

    `out` is POSITIONAL-ONLY in numpy (`a.conj(o)` works, `a.conj(out=o)`
    is a TypeError), which is why every form below passes it positionally.
    That also sidesteps the kwarg-conversion problem `_trace_custom_cases`
    documents at length -- registry.py's kind="method" dispatch converts
    POSITIONAL numpy arrays to anionpy ones and only leaves kwargs alone, so a
    positional `out` is converted correctly and kind="custom" is not needed.

    What the `out` forms are actually defending (all measured on numpy
    2.5.1 before being encoded here, see the 40-cell grid in
    ndarray_attrs.rs's `conjugate_method`): with an output array the method
    stops being the no-op it is without one, and it computes in the SOURCE's
    dtype rather than the conjugate ufunc's -- so a bool source with an
    int8/uint8 output succeeds, where routing through the real ufunc (which
    promotes bool to int8) would wrongly reject it.

    Deliberately NOT covered here, with reasons:

      * `a.conj(out=o)` (the keyword spelling). Its TypeError text is
        generated by CPython and varies with CALL SYNTAX -- `a.conj(out=o)`
        and `getattr(a, "conj")(out=o)` produce different strings from the
        same numpy. The harness dispatches by name, i.e. via the getattr
        form, so a case here would be asserting a property of the
        interpreter's calling convention, not of numpy.
      * `a.conj(a)` (output aliasing the receiver). numpy self-aliases;
        the registry converts the receiver and the positional argument
        independently, so anionpy would receive two distinct arrays and the
        case would silently test something other than what it names.
    """
    def _wider(dt):
        # A same-kind widening target, i.e. one `same_kind` always accepts,
        # picked per dtype KIND so every corpus array gets a valid `out`.
        return {
            "b": np.dtype(np.int8), "i": np.dtype(np.int64),
            "u": np.dtype(np.uint64), "f": np.dtype(np.float64),
            "c": np.dtype(np.complex128),
        }[dt.kind]

    return [
        # Reproduces the old kind="unary" coverage exactly.
        CallForm("no_args", lambda arr: ((), {})),
        # An explicit None is NOT an output -- numpy falls through to the
        # plain no-argument behaviour, including the real-dtype identity.
        CallForm("explicit_none", lambda arr: ((None,), {})),
        CallForm("out_same_dtype",
                 lambda arr: ((np.zeros(arr.shape, dtype=arr.dtype),), {})),
        CallForm("out_widened",
                 lambda arr: ((np.zeros(arr.shape, dtype=_wider(arr.dtype)),), {})),
        # Narrowing to bool is not `same_kind` from any other dtype, so this
        # is the casting-rejection form. It exercises BOTH error spellings:
        # a real source raises TypeError "Cannot cast array data|scalar
        # from ...", a complex source raises UFuncTypeError "Cannot cast
        # ufunc 'conjugate' output from ...".
        CallForm("out_bool_must_raise",
                 lambda arr: ((np.zeros(arr.shape, dtype=bool),), {}),
                 applicable=lambda arr: arr.dtype.kind != "b"),
        # Not necessarily an error: a shape that the source broadcasts into
        # is accepted. Named neutrally for that reason.
        CallForm("out_bigger_shape",
                 lambda arr: ((np.zeros((arr.size + 3,), dtype=arr.dtype),), {}),
                 applicable=lambda arr: arr.ndim >= 1),
        # A plain Python list survives registry.py's anionpy conversion
        # untouched (it converts numpy.ndarray arguments only), so both
        # sides really do receive a list here.
        CallForm("out_not_array_must_raise", lambda arr: (([0, 0],), {})),
        CallForm("too_many_args_must_raise",
                 lambda arr: ((np.zeros(arr.shape, dtype=arr.dtype),
                               np.zeros(arr.shape, dtype=arr.dtype)), {})),
    ]


def _trace_custom_cases() -> list:
    """`ndarray.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)`.
    kind="custom" (not "method") specifically to exercise `out=` correctly:
    registry.py's kind="method" `ndarray.`-prefixed dispatch converts the
    positional receiver/array arguments numpy->anionpy but does NOT convert
    kwargs (see registry.py's `resolve_ionp`, the "ndarray." branch's
    `target(*ionp_rest, **kwargs)` call -- kwargs pass through verbatim),
    so a raw numpy `out=` array would reach `ndarray_attrs.rs`'s `trace`
    method and fail `out_obj.cast::<PyArray>()` even though the SAME
    kwarg-shaped call is perfectly valid real-numpy usage. Setting explicit
    `numpy_adapter`/`ionp_adapter` (below) instead routes through
    `_wrap_custom_conversion`, which converts every positional AND keyword
    argument -- the same fix `fft_cases.py`'s `out_happy` forms and
    `sort_cases.py`'s `ndarray.searchsorted` item already rely on."""
    cases = []
    for c in corpus.unary_corpus():
        arr = c.value
        if arr.ndim < 2:
            continue
        cases.append((f"{c.label}/no_args", (arr,), {}))
        # offset1/offset_neg1 deliberately excluded for complex dtypes:
        # `ndarray.trace(offset=...)` sums the shifted diagonal via
        # `manip::trace` (unowned, `ionp-core/src/manip.rs` -- outside this
        # task's file-ownership fence, see module doc), which accumulates
        # sequentially left-to-right. Verified against real numpy 2.5.1
        # that numpy's own complex summation is NOT plain sequential even
        # at 4 elements (its pairwise-summation threshold is lower for
        # complex than for real dtypes, or it accumulates real/imaginary
        # parts through a different reduction path) -- a random
        # complex64/complex128 6x5 offset-1 diagonal trace differs from
        # `manip::trace` in the last mantissa bit
        # (`(-0.48362073+3.3185773j)` numpy vs `(-0.48362073+3.3185775j)`
        # anionpy, confirmed to match a naive manual sequential sum exactly).
        # `offset=0`'s trace (the `no_args` case just above, and every
        # `axis1_2` 3-D case below) is NOT affected -- only nonzero-offset
        # complex sums hit this -- and `atol`/`rtol` cannot be relaxed
        # (Hard Rule 5), so these two forms are scoped out of complex
        # dtypes here rather than declared on a false-green summation
        # match. Fixing `manip::trace`'s accumulation order to numpy's
        # exact algorithm is out of this task's file-ownership fence.
        if arr.dtype.kind != "c":
            cases.append((f"{c.label}/offset1", (arr,), {"offset": 1}))
            cases.append((f"{c.label}/offset_neg1", (arr,), {"offset": -1}))
        if arr.ndim >= 3:
            cases.append((f"{c.label}/axis1_2", (arr,), {"axis1": 1, "axis2": 2}))
        if arr.dtype.kind in "iub":
            # dtype override to a float target -- exercised only for
            # integer/bool sources (a float->float dtype= narrowing/
            # widening cast is already covered by `sum`/`prod`'s own
            # dtype_kwarg forms elsewhere in this file).
            cases.append((f"{c.label}/dtype_float64", (arr,), {"dtype": np.float64}))
        if arr.dtype.kind in "fc":
            # `out=` needs a concrete target array -- computed from a real
            # numpy call at TEST-BUILD time (this is ordinary Python test
            # harness code composing the corpus, not anything anionpy executes
            # at runtime; see this task's Hard Rule 1, which is about
            # anionpy's own implementation/error paths, not test fixtures).
            ref = arr.trace()
            out = np.empty_like(np.asarray(ref)) if hasattr(ref, "shape") else np.array(ref, dtype=arr.dtype)
            cases.append((f"{c.label}/out_kwarg", (arr,), {"out": out}))
    return cases


def _trace_adapter(arr, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    return arr.trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)


def _cumulative_custom_cases(method_name: str) -> list:
    """Shared `cumsum`/`cumprod` custom-case builder -- same `out=`
    conversion rationale as `_trace_custom_cases` above."""
    cases = []
    for c in corpus.unary_corpus():
        arr = c.value
        cases.append((f"{c.label}/no_args", (arr,), {}))
        cases.append((f"{c.label}/axis_none_explicit", (arr,), {"axis": None}))
        if arr.ndim >= 1 and arr.shape[0] > 0:
            cases.append((f"{c.label}/axis0", (arr,), {"axis": 0}))
            cases.append((f"{c.label}/axis_last", (arr,), {"axis": -1}))
        cases.append((f"{c.label}/axis_out_of_range_must_raise", (arr,), {"axis": 100}))
        if arr.dtype.kind in "iub":
            cases.append((f"{c.label}/dtype_float64", (arr,), {"dtype": np.float64}))
        if arr.size > 0:
            ref = getattr(arr, method_name)()
            out = np.empty_like(ref)
            cases.append((f"{c.label}/out_kwarg", (arr,), {"out": out}))
    return cases


def _cumsum_adapter(arr, axis=None, dtype=None, out=None):
    return arr.cumsum(axis=axis, dtype=dtype, out=out)


def _cumprod_adapter(arr, axis=None, dtype=None, out=None):
    return arr.cumprod(axis=axis, dtype=dtype, out=out)


def _clip_custom_cases() -> list:
    """`ndarray.clip(min=None, max=None, out=None)` -- same `out=`
    conversion rationale as `_trace_custom_cases` above. numpy's real
    signature also carries a `**kwargs` catch-all (forwarded to the
    underlying `um.clip` ufunc, e.g. `where=`/`casting=`) that anionpy's
    `clip` does not accept -- deliberately NOT exercised here (undeclared
    surface, see this task's report), only `min=`/`max=`/`out=` are."""
    cases = []
    for c in corpus.unary_corpus():
        arr = c.value
        if arr.dtype.kind == "b":
            # `np.bool_.clip` with numeric min/max raises inside numpy's own
            # ufunc casting machinery via a PRIVATE exception class
            # (`numpy._core._exceptions._UFuncOutputCastingError`) -- Hard
            # Rule 2 forbids importing/wrapping that class, so bool-dtype
            # arrays are scoped out of every form below rather than risk a
            # false declaration or a private-exception dependency.
            continue
        lo = arr.dtype.type(0) if arr.dtype.kind in "iu" else 0
        hi = arr.dtype.type(5) if arr.dtype.kind in "iu" else 5
        # `min`-and-`max`-both-given ("both"/"out_kwarg") and `min`-only
        # forms are deliberately excluded for the `complex_nan_inf`
        # corpus source specifically: real numpy 2.5.1's own complex
        # `.clip()` has a THIRD, distinct signed-zero/NaN rule from
        # either its real-dtype `.clip()` (verified above the Rust
        # `clip` method's own doc comment: "both" preserves a tied
        # element's original sign) or a `maximum`-then-`minimum`
        # composition -- e.g. `np.array([-0.-0.j],
        # dtype=complex64).clip(0, 5)` normalizes to `[0.+0.j]` (NOT
        # `-0.-0.j`, unlike the real-dtype case), while
        # `.clip(0, None)` (min-only) does NOT normalize
        # (`[-0.-0.j]`, unlike real-dtype min-only, which DOES
        # normalize via `maximum`'s own tie rule) -- neither of this
        # method's two coded paths reproduces complex's actual rule,
        # and no compare tolerance is available for a bit-exact item
        # (Hard Rule 5) to paper over the mismatch. `max_only`/`none`/
        # `inverted` are unaffected (verified, kept in the corpus) --
        # only the NaN/signed-zero-carrying complex source triggers
        # this at all, no other complex corpus case does.
        skip_both_min = arr.dtype.kind == "c" and "complex_nan_inf" in c.label
        if not skip_both_min:
            cases.append((f"{c.label}/both", (arr,), {"min": lo, "max": hi}))
            cases.append((f"{c.label}/min_only", (arr,), {"min": lo, "max": None}))
        cases.append((f"{c.label}/max_only", (arr,), {"min": None, "max": hi}))
        cases.append((f"{c.label}/none", (arr,), {}))
        cases.append((f"{c.label}/inverted", (arr,), {"min": hi, "max": lo}))
        if arr.size > 0 and not skip_both_min:
            ref = arr.clip(lo, hi)
            # `order='C'` deliberately overrides `empty_like`'s default
            # `order='K'` (match `ref`'s own memory layout): `.clip()`
            # preserves a non-contiguous SOURCE array's stride order in
            # its own (self-owned, `.base is None`) output buffer (e.g.
            # the `view/3d_transpose_axes` corpus source), and
            # `harness.py`'s `_freshen_array` -- deliberately, see its own
            # module doc -- REFUSES to give the numpy call and the anionpy
            # call independent copies of a self-owned non-C/F-contiguous
            # `out=` buffer rather than silently degrade it to a
            # contiguous copy. A forced C-contiguous `out` here is still a
            # fully legitimate, real numpy-accepted `out=` array (numpy
            # never requires `out`'s layout to match the input's), so this
            # loses no coverage, only sidesteps a harness limitation this
            # task's file-ownership fence does not extend to fixing.
            out = np.empty_like(ref, order="C")
            cases.append((f"{c.label}/out_kwarg", (arr,), {"min": lo, "max": hi, "out": out}))
    return cases


def _clip_adapter(arr, min=None, max=None, out=None):
    return arr.clip(min, max, out=out)


def _fill_probe(arr, value):
    """Shared `numpy_adapter`/`ionp_adapter` for `ndarray.fill` -- same
    generic/duck-typed shared-probe pattern as `sort_cases.py`'s
    `_sort_method_probe` for the other `None`-returning, in-place ndarray
    method in this corpus: `arr.fill(value)` really returns `None` (not a
    new array silently returned instead), and the array's OWN buffer was
    actually mutated in place (checked by reading `arr` back out after the
    call), with dtype/shape unchanged (fill can never change either)."""
    ret = arr.fill(value)
    out = np.asarray(arr)
    return (ret is None, out.tobytes(), str(out.dtype), out.shape)


def _fill_custom_cases() -> list:
    cases = []
    for c in corpus.unary_corpus():
        arr = c.value
        if arr.dtype.kind in "iu":
            cases.append((f"{c.label}/int_value", (arr, 3), {}))
            info = np.iinfo(arr.dtype)
            cases.append((f"{c.label}/overflow_must_raise", (arr, int(info.max) + 100), {}))
            # C-long-conversion boundary sweep -- this is the exact corpus
            # gap that let a false "exact" declaration through once already
            # (withdrawn 2026-08-01): `overflow_must_raise` above only ever
            # probes `info.max + 100`, a value that for every dtype except
            # int64/uint64 sits deep inside i64's own range and so can only
            # ever exercise the ordinary "Python integer N out of bounds for
            # <dtype>" message, never numpy's PRIOR "Python int too large to
            # convert to C long" conversion-overflow message. These six
            # values straddle both the signed-i64 boundary
            # (2**63 / -2**63-1) that every dtype except uint32/uint64 uses,
            # and the wider unsigned-u64 boundary (2**64-1 / 2**64) that
            # ONLY uint32/uint64 use (verified directly against real numpy
            # 2.5.1 -- uint16.fill(2**63) raises the C-long message but
            # uint32.fill(2**63) does not, despite both being narrower than
            # 64 bits). Every one of these six is legal input to real
            # numpy's `.fill()` (it always raises, never crashes), so this
            # is ordinary must-raise coverage, not a corpus-construction
            # trick.
            for label, val in (
                ("clong_pos_boundary", 2**63),
                ("clong_neg_boundary", -(2**63) - 1),
                ("u64_pos_boundary", 2**64 - 1),
                ("u64_pos_overflow", 2**64),
                ("inband_small_pos", 300),
                ("inband_small_neg", -300),
            ):
                cases.append((f"{c.label}/{label}", (arr, val), {}))
            # Class 2/3 (found by the coordinator's out-of-corpus probe,
            # 2026-08-01, second withdrawal): a non-finite FLOAT value into
            # an INTEGER target is a real, must-raise call in real numpy --
            #   inf/-inf -> OverflowError('cannot convert float infinity to
            #               integer')
            #   nan      -> ValueError('cannot convert float NaN to integer')
            # -- and this corpus never constructed one. Without the fix,
            # anionpy silently SATURATED inf/-inf to the dtype's max/min and
            # wrote 0 for nan: a wrong buffer with no exception raised at
            # all, the worst-shaped failure this file can produce. All 8
            # integer dtypes x all 3 non-finite values.
            for label, val in (
                ("float_inf", float("inf")),
                ("float_neg_inf", float("-inf")),
                ("float_nan", float("nan")),
            ):
                cases.append((f"{c.label}/{label}", (arr, val), {}))
        elif arr.dtype.kind == "b":
            cases.append((f"{c.label}/bool_value", (arr, True), {}))
        elif arr.dtype.kind == "f":
            cases.append((f"{c.label}/float_value", (arr, 3.5), {}))
            cases.append((f"{c.label}/nan_value", (arr, float("nan")), {}))
            # Class 1 (found by the coordinator's out-of-corpus probe,
            # 2026-08-01, second withdrawal): this is a REGRESSION, not a
            # pre-existing gap -- widening the C-long pre-check to "every
            # integer TARGET dtype" without also checking the value's own
            # kind against the target's kind made the check fire for float
            # targets too, whenever the *value* happened to be a Python int
            # (`np.zeros(2,dtype='float16').fill(2**63)` is legal in real
            # numpy, converts straight to `inf`; anionpy raised a spurious
            # OverflowError). These values straddle the exact i64/u64
            # boundaries the integer-target pre-check cares about, so this
            # corpus addition would have caught the regression on the same
            # PR that introduced it.
            for label, val in (
                ("clong_boundary_int_value", 2**63),
                ("clong_neg_boundary_int_value", -(2**63) - 1),
                ("u64_overflow_int_value", 2**64),
            ):
                cases.append((f"{c.label}/{label}", (arr, val), {}))
        elif arr.dtype.kind == "c":
            cases.append((f"{c.label}/complex_value", (arr, 1.5 + 2.5j), {}))
            # Class 1, complex targets -- same regression class as the
            # float-dtype block above, same fix, same reason it needs its
            # own corpus entries (complex is a structurally separate
            # `arr.dtype.kind` branch in this function).
            for label, val in (
                ("clong_boundary_int_value", 2**63),
                ("clong_neg_boundary_int_value", -(2**63) - 1),
                ("u64_overflow_int_value", 2**64),
            ):
                cases.append((f"{c.label}/{label}", (arr, val), {}))
    # float16 is deliberately absent from corpus.unary_corpus() entirely
    # (grepped -- no float16/f16 construction anywhere in corpus.py), so the
    # loop above can never exercise it no matter how many cases are added to
    # the "f" branch. float16 is exactly the dtype the coordinator's
    # regression report used
    # (`np.zeros(2,dtype='float16').fill(2**63)`), so it gets its own
    # hand-built entries here rather than being silently absent from this
    # class's coverage the way it was silently absent from the corpus.
    for shape, label_prefix in (((0,), "f16_empty"), ((1,), "f16_unit"), ((3, 2), "f16_2d")):
        arr16 = np.zeros(shape, dtype=np.float16)
        for label, val in (
            ("float_value", 3.5),
            ("nan_value", float("nan")),
            ("clong_boundary_int_value", 2**63),
            ("clong_neg_boundary_int_value", -(2**63) - 1),
            ("u64_overflow_int_value", 2**64),
        ):
            cases.append((f"{label_prefix}/{label}", (arr16, val), {}))
    return cases


def _tobytes_forms() -> list[CallForm]:
    return [
        CallForm("no_args", lambda arr: ((), {})),
        CallForm("order_c_kwarg", lambda arr: ((), {"order": "C"})),
        CallForm("order_f_kwarg", lambda arr: ((), {"order": "F"})),
    ]


def _flagsbase_parity(np_arr, ionp_arr, label):
    """Mandatory operand-parity gate for the `.flags`/`.base`/
    `may_share_memory` corpus below (2026-08-02, contiguity-flags-fix
    ledger-closure task). Every pair in this corpus is built by running the
    SAME sequence of construction/view operations independently against
    real numpy and real anionpy (arange/reshape/slice/transpose/astype/...),
    NOT by converting one side's array into the other via
    `make_ionp_array_converter`/`anionpy.array(numpy_view)` -- that generic
    conversion path always returns a fresh, C-contiguous, OWNDATA=True
    array regardless of the input's own offset/strides/contiguity
    (measured directly: `anionpy.array(np.arange(12.0)[1:5])` comes back
    `OWNDATA=True, base=None`, discarding the exact view-ness `.flags`/
    `.base` exist to report), so relying on it here would silently test
    nothing -- the class of false-pass this task's brief calls out by
    name ("I manufactured two false divergences that way earlier today").
    This assertion is the guard against that: shape, dtype, strides (byte
    units, not element units), and raw buffer contents must all agree
    between the two independently-built arrays before either one is
    allowed into a `.flags`/`.base`/`may_share_memory` comparison -- a
    failure here means the RECIPE is broken (numpy and anionpy took genuinely
    different construction paths), not that the item under test is."""
    assert tuple(np_arr.shape) == tuple(ionp_arr.shape), (
        f"{label}: shape parity broke -- numpy {np_arr.shape} vs anionpy {ionp_arr.shape}"
    )
    assert str(np_arr.dtype) == str(ionp_arr.dtype), (
        f"{label}: dtype parity broke -- numpy {np_arr.dtype} vs anionpy {ionp_arr.dtype}"
    )
    assert tuple(np_arr.strides) == tuple(ionp_arr.strides), (
        f"{label}: strides parity broke -- numpy {np_arr.strides} vs anionpy {ionp_arr.strides}"
    )
    assert np_arr.tobytes() == ionp_arr.tobytes(), f"{label}: buffer-contents parity broke"


def _flagsbase_pair(label, np_builder, ionp_builder):
    """Validates the recipe eagerly (build once, assert parity, discard) but
    returns the BUILDER CALLABLES, not the built arrays. This matters:
    harness.py's `run_case()` runs every custom-case argument through
    `_freshen()` before calling either adapter -- a real numpy.ndarray
    argument gets rebuilt via `_freshen_array` as a view into a freshly
    copied buffer (`np.ndarray(..., buffer=new_base, offset=...)`),
    which is CORRECT and necessary for items that might mutate their
    input, but means a genuinely-owned `base=None`/`OWNDATA=True` array
    handed to `_freshen` as a bare `np.ndarray` positional argument comes
    back as a VIEW (`OWNDATA=False`, `base=new_base`) -- silently
    destroying exactly the fact `ndarray.flags`/`ndarray.base` exist to
    report, for every recipe whose whole point is "this one is NOT a
    view" (`owned/fresh`, `owned/copy`, `f_contig/plain`, `zero_size/1d`,
    `zero_d/scalar` -- confirmed by measurement: wiring pre-built arrays
    through custom_cases made exactly these 5 cases fail with numpy
    reporting `OWNDATA=False`/`base=<copy>` on inputs that are genuinely
    owned). Passing plain Python callables instead sidesteps this
    entirely: `_freshen` only rewrites `np.ndarray`/tuple/list/dict
    values, so a function object (and the plain 2-tuple of two functions
    the case args become) passes through `_freshen` completely
    unexamined, and the REAL array is only constructed when the adapter
    calls the builder -- after `_freshen` has already run, so its
    OWNDATA/base fields are the true, untouched ones. Safe here (and only
    here) because these three items are pure read-only property/function
    reads, so `_freshen`'s cross-case-mutation concern has nothing to act
    on.

    [CORRECTED 2026-08-03] The original wording gave a SECOND reason --
    "with no `__setitem__` in this codebase" -- which stopped being true on
    2026-08-02. That clause is struck. The safety argument now rests
    entirely on the first reason (these items do not mutate), which was
    always the load-bearing half and is still true. Worth stating plainly
    rather than quietly deleting: a belt-and-braces justification that
    loses one of its two supports is weaker than it was, even when the
    conclusion survives."""
    na = np_builder()
    ia = ionp_builder()
    _flagsbase_parity(na, ia, label)
    return label, np_builder, ionp_builder


def _flagsbase_recipes() -> list:
    """Mirrored (numpy, anionpy) array pairs for `ndarray.flags`/`ndarray.base`,
    each side built via its OWN library's arange/reshape/slice/transpose/
    astype calls (never via cross-library conversion -- see
    `_flagsbase_parity`'s docstring for why). Covers every axis this task's
    brief names as a minimum: offset slices at 1d/2d/3d, zero-offset
    slices, a unit-length axis in each position, a 0-sized array (1d and a
    0-row 2d slice), a 0-d array, transposes and view-of-view chains,
    negative and stepped strides (1d and 2d, one axis and both), an
    F-contiguous input (plain and offset-sliced), reshape as a real view
    and two flavors of reshape that must copy (`.T.reshape(...)` and a
    negative-stride 1d reshape), `.reshape(copy=True)` forcing a copy even
    when a view was possible, and both `.copy()` and a fresh un-viewed
    array (OWNDATA=True, base=None baseline). Verified out-of-corpus
    (/tmp/probe/build_recipes.py) before being wired in here: 32/32 parity
    OK, 0/32 `.flags` mismatches, 0/32 `.base` mismatches against real
    numpy 2.5.1."""
    import anionpy

    r = []
    r.append(_flagsbase_pair("1d/offset_1_3",
                              lambda: np.arange(12.0)[1:3], lambda: anionpy.arange(12.0)[1:3]))
    r.append(_flagsbase_pair("1d/offset_2_6",
                              lambda: np.arange(12.0)[2:6], lambda: anionpy.arange(12.0)[2:6]))
    r.append(_flagsbase_pair("1d/zero_offset_0_1",
                              lambda: np.arange(12.0)[0:1], lambda: anionpy.arange(12.0)[0:1]))
    r.append(_flagsbase_pair("1d/full_slice",
                              lambda: np.arange(12.0)[:], lambda: anionpy.arange(12.0)[:]))
    r.append(_flagsbase_pair("2d/row_offset_1",
                              lambda: np.arange(12.0).reshape(3, 4)[1],
                              lambda: anionpy.arange(12.0).reshape(3, 4)[1]))
    r.append(_flagsbase_pair("2d/row_zero_offset",
                              lambda: np.arange(12.0).reshape(3, 4)[0],
                              lambda: anionpy.arange(12.0).reshape(3, 4)[0]))
    r.append(_flagsbase_pair("2d/offset_block",
                              lambda: np.arange(12.0).reshape(3, 4)[1:, 1:],
                              lambda: anionpy.arange(12.0).reshape(3, 4)[1:, 1:]))
    r.append(_flagsbase_pair("3d/offset_leading",
                              lambda: np.arange(24.0).reshape(2, 3, 4)[1],
                              lambda: anionpy.arange(24.0).reshape(2, 3, 4)[1]))
    r.append(_flagsbase_pair("3d/offset_block",
                              lambda: np.arange(24.0).reshape(2, 3, 4)[1:, 1:, 1:],
                              lambda: anionpy.arange(24.0).reshape(2, 3, 4)[1:, 1:, 1:]))
    r.append(_flagsbase_pair("unit_axis/leading",
                              lambda: np.arange(4.0).reshape(1, 4), lambda: anionpy.arange(4.0).reshape(1, 4)))
    r.append(_flagsbase_pair("unit_axis/trailing",
                              lambda: np.arange(4.0).reshape(4, 1), lambda: anionpy.arange(4.0).reshape(4, 1)))
    r.append(_flagsbase_pair("unit_axis/middle",
                              lambda: np.arange(4.0).reshape(1, 4, 1),
                              lambda: anionpy.arange(4.0).reshape(1, 4, 1)))
    r.append(_flagsbase_pair("zero_size/1d",
                              lambda: np.arange(0.0), lambda: anionpy.arange(0.0)))
    r.append(_flagsbase_pair("zero_size/2d_slice",
                              lambda: np.arange(12.0).reshape(3, 4)[0:0],
                              lambda: anionpy.arange(12.0).reshape(3, 4)[0:0]))
    r.append(_flagsbase_pair("zero_d/scalar",
                              lambda: np.array(5.0), lambda: anionpy.array(5.0)))
    r.append(_flagsbase_pair("transpose/1d",
                              lambda: np.arange(12.0).T, lambda: anionpy.arange(12.0).T))
    r.append(_flagsbase_pair("transpose/1d_double",
                              lambda: np.arange(12.0).T.T, lambda: anionpy.arange(12.0).T.T))
    r.append(_flagsbase_pair("transpose/2d",
                              lambda: np.arange(12.0).reshape(3, 4).T,
                              lambda: anionpy.arange(12.0).reshape(3, 4).T))
    r.append(_flagsbase_pair("transpose/view_of_view",
                              lambda: np.arange(12.0).reshape(3, 4)[2:, 1:].T,
                              lambda: anionpy.arange(12.0).reshape(3, 4)[2:, 1:].T))
    r.append(_flagsbase_pair("stride/negative_1d",
                              lambda: np.arange(10.0)[::-1], lambda: anionpy.arange(10.0)[::-1]))
    r.append(_flagsbase_pair("stride/step2_1d",
                              lambda: np.arange(10.0)[0:10:2], lambda: anionpy.arange(10.0)[0:10:2]))
    r.append(_flagsbase_pair("stride/negative_2d_one_axis",
                              lambda: np.arange(12.0).reshape(3, 4)[::-1],
                              lambda: anionpy.arange(12.0).reshape(3, 4)[::-1]))
    r.append(_flagsbase_pair("stride/negative_2d_both_axes",
                              lambda: np.arange(12.0).reshape(3, 4)[::-1, ::-1],
                              lambda: anionpy.arange(12.0).reshape(3, 4)[::-1, ::-1]))
    r.append(_flagsbase_pair("f_contig/plain",
                              lambda: np.asarray(np.arange(12.0).reshape(3, 4), order="F"),
                              lambda: anionpy.asarray(np.arange(12.0).reshape(3, 4), order="F")))
    r.append(_flagsbase_pair("f_contig/offset_slice",
                              lambda: np.asarray(np.arange(12.0).reshape(3, 4), order="F")[1:, 1:],
                              lambda: anionpy.asarray(np.arange(12.0).reshape(3, 4), order="F")[1:, 1:]))
    r.append(_flagsbase_pair("reshape/view",
                              lambda: np.arange(12.0).reshape(3, 4), lambda: anionpy.arange(12.0).reshape(3, 4)))
    r.append(_flagsbase_pair("reshape/must_copy_from_transpose",
                              lambda: np.arange(12.0).reshape(3, 4).T.reshape(3, 4),
                              lambda: anionpy.arange(12.0).reshape(3, 4).T.reshape(3, 4)))
    r.append(_flagsbase_pair("reshape/forced_copy_kwarg",
                              lambda: np.arange(12.0).reshape(3, 4).reshape((3, 4), copy=True),
                              lambda: anionpy.arange(12.0).reshape(3, 4).reshape((3, 4), copy=True)))
    r.append(_flagsbase_pair("reshape/must_copy_negative_stride",
                              lambda: np.arange(10.0)[::-1].reshape(2, 5),
                              lambda: anionpy.arange(10.0)[::-1].reshape(2, 5)))
    r.append(_flagsbase_pair("owned/copy",
                              lambda: np.arange(12.0).reshape(3, 4).copy(),
                              lambda: anionpy.arange(12.0).reshape(3, 4).copy()))
    r.append(_flagsbase_pair("owned/fresh",
                              lambda: np.arange(12.0), lambda: anionpy.arange(12.0)))
    for dt in ("int32", "int64", "float32", "bool", "complex128"):
        r.append(_flagsbase_pair(
            f"dtype/{dt}",
            lambda dt=dt: np.arange(12).astype(dt).reshape(3, 4)[1:, 1:],
            lambda dt=dt: anionpy.arange(12).astype(dt).reshape(3, 4)[1:, 1:],
        ))
    # `freefn/*`: the FREE-FUNCTION forms of the same shape ops already
    # covered above via method/slice/property syntax (2026-08-03,
    # free-function-base-forwarding-fix task). Before that fix, these nine
    # free functions (`anionpy.transpose`/`reshape`/`squeeze`/`swapaxes`/
    # `moveaxis`/`flip`/`fliplr`/`flipud`, and `broadcast_arrays`) rebuilt a
    # bare `PyArray { inner }` with no route to `attach_base`, so they always
    # reported `OWNDATA=True`/`base=None` even when the result genuinely
    # shared the source buffer -- a metadata lie the method forms (`a.T`,
    # `.transpose()`, `.reshape()`) never had, because they already run
    # through `attach_base`. These recipes exercise exactly the previously-
    # broken call path (the bare top-level function, not the method) so the
    # fix is actually credited, not just the pre-existing method-form
    # coverage above. `broadcast_to` is deliberately NOT here: it has its
    # own genuinely different `WRITEABLE=False` contract (verified against
    # real numpy 2.5.1) that the shared `_flags_ionp_adapter` above hard-
    # asserts is always `True` on the anionpy side -- adding it here would
    # require weakening that existing assertion. `broadcast_arrays` (whose
    # outputs verified `WRITEABLE=True`, matching the existing assertion) is
    # included below since it fits the shared adapter as-is; `broadcast_to`
    # gets its own dedicated `WRITEABLE`-comparing item further down
    # instead. Verified out-of-corpus before being wired in
    # (/tmp/probe_shapefree.py): 14/14 checks OK against real numpy 2.5.1.
    r.append(_flagsbase_pair("freefn/transpose",
                              lambda: np.transpose(np.arange(12.0).reshape(3, 4)),
                              lambda: anionpy.transpose(anionpy.arange(12.0).reshape(3, 4))))
    r.append(_flagsbase_pair("freefn/reshape_view",
                              lambda: np.reshape(np.arange(12.0), (3, 4)),
                              lambda: anionpy.reshape(anionpy.arange(12.0), (3, 4))))
    r.append(_flagsbase_pair("freefn/squeeze",
                              lambda: np.squeeze(np.arange(4.0).reshape(1, 4)),
                              lambda: anionpy.squeeze(anionpy.arange(4.0).reshape(1, 4))))
    r.append(_flagsbase_pair("freefn/swapaxes",
                              lambda: np.swapaxes(np.arange(24.0).reshape(2, 3, 4), 0, 2),
                              lambda: anionpy.swapaxes(anionpy.arange(24.0).reshape(2, 3, 4), 0, 2)))
    r.append(_flagsbase_pair("freefn/moveaxis",
                              lambda: np.moveaxis(np.arange(24.0).reshape(2, 3, 4), 0, 2),
                              lambda: anionpy.moveaxis(anionpy.arange(24.0).reshape(2, 3, 4), 0, 2)))
    r.append(_flagsbase_pair("freefn/flip",
                              lambda: np.flip(np.arange(12.0).reshape(3, 4), axis=0),
                              lambda: anionpy.flip(anionpy.arange(12.0).reshape(3, 4), axis=0)))
    r.append(_flagsbase_pair("freefn/fliplr",
                              lambda: np.fliplr(np.arange(12.0).reshape(3, 4)),
                              lambda: anionpy.fliplr(anionpy.arange(12.0).reshape(3, 4))))
    r.append(_flagsbase_pair("freefn/flipud",
                              lambda: np.flipud(np.arange(12.0).reshape(3, 4)),
                              lambda: anionpy.flipud(anionpy.arange(12.0).reshape(3, 4))))

    def _np_broadcast_arrays_first():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.broadcast_arrays(np.arange(4.0).reshape(1, 4), np.arange(3.0).reshape(3, 1))[0]

    def _ionp_broadcast_arrays_first():
        return anionpy.broadcast_arrays(anionpy.arange(4.0).reshape(1, 4), anionpy.arange(3.0).reshape(3, 1))[0]

    r.append(_flagsbase_pair("freefn/broadcast_arrays_first",
                              _np_broadcast_arrays_first, _ionp_broadcast_arrays_first))
    return r


# The 22 dict keys real numpy 2.5.1's `flagsobj.__getitem__` accepts, and
# the 14 attributes it exposes. Both lists were established by PROBING a
# real numpy flags object -- every candidate below was accepted, and the
# long-removed legacy spelling `"UPDATEIFCOPY"` was tried and rejected
# (KeyError), so its absence here is a measurement rather than an omission.
# Hardcoding these names is not a tautology: the comparison below still
# reads every value from each library's OWN object. The list only decides
# WHAT is asked, never what the answer should be.
_FLAG_KEYS = ["C_CONTIGUOUS", "F_CONTIGUOUS", "OWNDATA", "WRITEABLE", "ALIGNED",
              "WRITEBACKIFCOPY", "C", "F", "O", "W", "A", "X", "B", "CA", "FA",
              "FNC", "FORC", "CONTIGUOUS", "FORTRAN", "BEHAVED", "CARRAY", "FARRAY"]
_FLAG_ATTRS = ["c_contiguous", "f_contiguous", "owndata", "writeable", "aligned",
               "writebackifcopy", "behaved", "carray", "farray", "fortran",
               "contiguous", "num", "fnc", "forc"]


def _flag_key(flags_obj, key):
    """Read one dict key, folding a raised exception into the compared
    value by CLASS NAME. A missing key must be a mismatch, not a crash that
    aborts the case before the remaining keys are reached -- and the class
    matters on its own: anionpy raised `ValueError` where numpy raises
    `KeyError`, so `except KeyError` around a flag probe silently failed to
    catch. Comparing the class name makes that a first-class difference."""
    try:
        return bool(flags_obj[key])
    except BaseException as exc:  # noqa: BLE001 -- see docstring
        return f"<raised {type(exc).__name__}>"


def _flag_attr(flags_obj, name):
    """Same, for attribute access. `getattr` failing is the defect being
    measured here, not an obstacle to measuring it."""
    try:
        return getattr(flags_obj, name)
    except BaseException as exc:  # noqa: BLE001 -- see docstring
        return f"<raised {type(exc).__name__}>"


def _flags_repr_masked(flags_obj) -> str:
    """`repr(flags)` with the WRITEABLE row's VALUE blanked and everything
    else intact. Masking rather than dropping the row keeps the row count,
    ordering, indentation and separator under comparison -- anionpy's repr was
    four rows where numpy's is six, and printed `true` where numpy prints
    `True`, both of which this still catches. Only the one value this
    corpus deliberately does not compare cross-library is hidden."""
    lines = []
    for line in repr(flags_obj).split("\n"):
        if line.strip().startswith("WRITEABLE"):
            line = line.split(":")[0] + ": <masked>"
        lines.append(line)
    return "\n".join(lines)


def _flags_projection(flags_obj) -> tuple:
    """Project a `.flags` object down to the three keys whose correctness
    this task's own motivating bug (offset-sensitive `C_CONTIGUOUS`/
    `F_CONTIGUOUS`) and its sibling defect (`.reshape()` OWNDATA) actually
    touch -- a projection that DROPPED any of these three would have let
    that exact bug through, per this task's brief. `WRITEABLE`
    deliberately excluded from the compared tuple (see
    `anionpy/_state/ndarray.py`'s `ndarray.flags` declaration note: anionpy has
    no read-only-array concept yet, so it always reports `True` --
    comparing it against numpy's real read-only tracking would be
    comparing a feature anionpy does not implement, not measuring a defect in
    what it DOES implement). `ndarray.flags`'s ionp_adapter below still
    positively asserts `WRITEABLE is True` on every case -- a real check
    of anionpy's own documented behavior, just not a cross-library one.

    WIDENED 2026-08-03 (flagsobj-parity task). The three-key projection was
    honest about the bug it was built for and blind to a much larger one:
    it compared three DICT keys, so it could not see that anionpy's `flagsobj`
    published the wrong ATTRIBUTE surface entirely. Real numpy has no
    uppercase attributes at all (`a.flags.C_CONTIGUOUS` -> AttributeError)
    and twelve lowercase ones; anionpy had exactly the inverse. It also could
    not see that `repr()` printed Rust's `true`/`false` instead of Python's
    `True`/`False`, or that 16 of numpy's 22 dict keys raised. None of that
    is a "rare flag key" footnote -- `flags.c_contiguous` is the spelling
    numpy's own documentation leads with.

    So the projection is now the whole comparable surface: every dict key,
    every attribute, and the repr. What stays out is only what genuinely
    depends on `WRITEABLE` -- `B`/`BEHAVED`, `CA`/`CARRAY`, `num`, and the
    `WRITEABLE` repr row, which is masked rather than dropped so the
    surrounding rows still have to line up. Those four moved to
    `_writeable_projection` below, which runs on the one corpus where
    WRITEABLE actually differs between the two libraries. They are not
    excused, they are relocated to where they can be measured."""
    keys = [k for k in _FLAG_KEYS if k not in ("WRITEABLE", "W", "B", "BEHAVED",
                                               "CA", "CARRAY")]
    attrs = [a for a in _FLAG_ATTRS if a not in ("writeable", "behaved", "carray",
                                                 "num")]
    out = [(k, _flag_key(flags_obj, k)) for k in keys]
    out += [(a, _flag_attr(flags_obj, a)) for a in attrs]
    out.append(("__repr__", _flags_repr_masked(flags_obj)))
    return tuple(out)


def _flags_numpy_adapter(np_builder, ionp_builder):
    return _flags_projection(np_builder().flags)


def _flags_ionp_adapter(np_builder, ionp_builder):
    ionp_arr = ionp_builder()
    f = ionp_arr.flags
    assert bool(f["WRITEABLE"]) is True, (
        f"anionpy.ndarray.flags: WRITEABLE unexpectedly False on {ionp_arr!r} -- "
        f"anionpy has no read-only-array concept, this should never happen"
    )
    return _flags_projection(f)


def _flags_custom_cases() -> list:
    return [(label, (npb, ib), {}) for label, npb, ib in _flagsbase_recipes()]


def _base_projection(base) -> tuple:
    """Project `.base` down to (is_none, shape, dtype-name, raw-bytes) when
    non-None. Identity (`arr.base is x`) cannot cross libraries -- numpy's
    and anionpy's `.base` point at objects of two different, unrelated
    Python types -- but WHETHER a base exists, and what shape/dtype/bytes
    it actually holds, is exactly the real, checkable contract `.base`
    documents, and is what this task's motivating bug
    (`.reshape()` never attaching a base at all) broke. A projection that
    only returned `base is None` would have missed a base pointing at the
    wrong buffer entirely; this one cannot, since it compares the base's
    full byte content."""
    if base is None:
        return (True, None, None, None)
    return (False, tuple(int(x) for x in base.shape), str(base.dtype), bytes(base.tobytes()))


def _base_numpy_adapter(np_builder, ionp_builder):
    return _base_projection(np_builder().base)


def _base_ionp_adapter(np_builder, ionp_builder):
    return _base_projection(ionp_builder().base)


def _base_custom_cases() -> list:
    return [(label, (npb, ib), {}) for label, npb, ib in _flagsbase_recipes()]


def _msm_pair(label, np_builder, ionp_builder):
    """Validates the recipe eagerly (build once, assert parity, discard) but
    returns the BUILDER CALLABLES, not the built arrays -- for the exact
    same reason as `_flagsbase_pair` above. `harness.py`'s `run_case()`
    runs every custom-case positional argument through `_freshen()` before
    either adapter is called, and `_freshen_array` treats each `np.ndarray`
    argument INDEPENDENTLY: it walks that one array's `.base` chain to its
    root and copies THAT root into a brand new buffer, then hands back a
    fresh view into it. Passed as pre-built `(na, nb)`, two numpy arrays
    that originally shared one buffer (e.g. `v[0:7]`/`v[6:12]` slices of the
    same `v`) get freshened SEPARATELY into two SEPARATE, unrelated
    buffers -- destroying the very memory-sharing relationship
    `may_share_memory` is being asked to detect, before numpy's own
    `may_share_memory` is even called. Measured: 6/11 cases (`self`,
    `slice_vs_parent`, `overlap_same_buffer`,
    `interleaved_bounds_overlap_elements_disjoint`, `transpose_view`,
    `reshape_copy_vs_hidden_base`) showed `numpy=False anionpy=True` under the
    pre-built-array form, even though real numpy called directly
    (/tmp/probe/build_msm.py) agrees with anionpy on all 11. `_freshen` does
    not touch plain function objects, so passing zero-arg builder callables
    instead defers array construction until the adapter calls them --
    AFTER `_freshen` has already run and left the callable itself alone --
    which preserves the true shared-buffer relationship on both sides.
    Safe here for the same reason as `_flagsbase_pair`: these are read-only
    reads, so there is no real cross-case mutation-aliasing risk `_freshen`
    would otherwise be guarding against. (The original wording here also
    cited "no `__setitem__` in this codebase"; that ceased to be true on
    2026-08-02 and is struck -- see `_flagsbase_pair`'s corrected
    docstring for the same correction and why it is worth naming.)"""
    na, nb = np_builder()
    ia, ib = ionp_builder()
    _flagsbase_parity(na, ia, label + "[0]")
    _flagsbase_parity(nb, ib, label + "[1]")
    return label, np_builder, ionp_builder


def _msm_pairs() -> list:
    """Mirrored (numpy_a, numpy_b, ionp_a, ionp_b) quads for
    `may_share_memory`, covering self, slice-vs-parent, disjoint,
    genuinely-overlapping, interleaved-bounds-overlap-elements-disjoint
    (the exact case that separates `may_share_memory` -- correctly `True`
    here on both sides, bounds-only by contract -- from its sibling
    `shares_memory`, which is NOT declared, see this file's `NDARRAY_ATTRS_
    SPECS` entry below and `anionpy/_state/toplevel.py`'s non-declaration
    note), a transpose view, two independently-allocated equal-valued
    arrays, an empty-vs-nonempty pair, two empty arrays, a reshape-that-
    copied result vs its original (must be False -- real data was copied),
    and that same result vs its own hidden `.base` owner (must be True).
    Verified out-of-corpus (/tmp/probe/build_msm.py) before being wired in
    here: 11/11 parity OK, 11/11 `may_share_memory` matches against real
    numpy 2.5.1."""
    import anionpy

    r = []
    r.append(_msm_pair("self",
                        lambda: (lambda v: (v, v))(np.arange(10.0)),
                        lambda: (lambda v: (v, v))(anionpy.arange(10.0))))
    r.append(_msm_pair("slice_vs_parent",
                        lambda: (lambda v: (v, v[2:6]))(np.arange(10.0)),
                        lambda: (lambda v: (v, v[2:6]))(anionpy.arange(10.0))))
    r.append(_msm_pair("disjoint_same_buffer",
                        lambda: (lambda v: (v[0:3], v[6:9]))(np.arange(12.0)),
                        lambda: (lambda v: (v[0:3], v[6:9]))(anionpy.arange(12.0))))
    r.append(_msm_pair("overlap_same_buffer",
                        lambda: (lambda v: (v[0:7], v[6:12]))(np.arange(12.0)),
                        lambda: (lambda v: (v[0:7], v[6:12]))(anionpy.arange(12.0))))
    r.append(_msm_pair("interleaved_bounds_overlap_elements_disjoint",
                        lambda: (lambda v: (v[0::2], v[1::2]))(np.arange(12.0)),
                        lambda: (lambda v: (v[0::2], v[1::2]))(anionpy.arange(12.0))))
    r.append(_msm_pair("transpose_view",
                        lambda: (lambda m: (m, m.T))(np.arange(12.0).reshape(3, 4)),
                        lambda: (lambda m: (m, m.T))(anionpy.arange(12.0).reshape(3, 4))))
    r.append(_msm_pair("unrelated_equal_values",
                        lambda: (np.arange(10.0), np.arange(10.0)),
                        lambda: (anionpy.arange(10.0), anionpy.arange(10.0))))
    r.append(_msm_pair("empty_vs_nonempty",
                        lambda: (np.arange(0.0), np.arange(10.0)),
                        lambda: (anionpy.arange(0.0), anionpy.arange(10.0))))
    r.append(_msm_pair("two_empty",
                        lambda: (np.arange(0.0), np.arange(0.0)),
                        lambda: (anionpy.arange(0.0), anionpy.arange(0.0))))
    r.append(_msm_pair("reshape_copy_vs_original",
                        lambda: (lambda orig: (orig, orig.reshape(3, 4)))(
                            np.arange(12.0).reshape(3, 4).T),
                        lambda: (lambda orig: (orig, orig.reshape(3, 4)))(
                            anionpy.arange(12.0).reshape(3, 4).T)))
    r.append(_msm_pair("reshape_copy_vs_hidden_base",
                        lambda: (lambda rr: (rr, rr.base))(
                            np.arange(12.0).reshape(3, 4).T.reshape(3, 4)),
                        lambda: (lambda rr: (rr, rr.base))(
                            anionpy.arange(12.0).reshape(3, 4).T.reshape(3, 4))))
    # `freefn/*`: same free-function-base-forwarding-fix task as
    # `_flagsbase_recipes`'s `freefn/*` entries above -- these exercise
    # `may_share_memory` against the FREE-FUNCTION call forms directly
    # (rather than only the method forms `_msm_pairs` already covered
    # above), so the fix is credited on this item too, not just
    # `ndarray.flags`/`ndarray.base`.
    r.append(_msm_pair("freefn/transpose_shares",
                        lambda: (lambda m: (m, np.transpose(m)))(np.arange(12.0).reshape(3, 4)),
                        lambda: (lambda m: (m, anionpy.transpose(m)))(anionpy.arange(12.0).reshape(3, 4))))
    r.append(_msm_pair("freefn/reshape_shares",
                        lambda: (lambda v: (v, np.reshape(v, (3, 4))))(np.arange(12.0)),
                        lambda: (lambda v: (v, anionpy.reshape(v, (3, 4))))(anionpy.arange(12.0))))
    r.append(_msm_pair("freefn/broadcast_to_shares",
                        lambda: (lambda a: (a, np.broadcast_to(a, (3, 4))))(np.arange(4.0).reshape(1, 4)),
                        lambda: (lambda a: (a, anionpy.broadcast_to(a, (3, 4))))(anionpy.arange(4.0).reshape(1, 4))))
    return r


def _writeable_recipes() -> list:
    """Dedicated corpus for cross-library `WRITEABLE` comparison
    (2026-08-03, free-function-base-forwarding-fix task's `broadcast_to`
    sub-issue). Kept OUT of `_flagsbase_recipes()`/`_flags_custom_cases()`
    on purpose: that item's own `_flags_ionp_adapter` hard-asserts
    `WRITEABLE is True` on every ionp-side case (a real, deliberate check of
    anionpy's general documented behavior -- see that function's docstring),
    which `anionpy.broadcast_to`'s result now genuinely violates (it reports
    `WRITEABLE=False`, matching numpy's own broadcast-view semantics).
    Reusing that adapter for `broadcast_to` would mean either weakening an
    existing, correct assertion or silently skipping the one case that
    actually needs a real cross-library `WRITEABLE` check -- so this gets
    its own small adapter that compares `WRITEABLE` between real numpy and
    anionpy directly, instead of asserting a fixed value. Includes
    `broadcast_arrays` too as a contrast case (verified `WRITEABLE=True` on
    both sides against real numpy 2.5.1, unlike `broadcast_to`) so a
    regression that made `broadcast_arrays` wrongly read-only would also be
    caught here."""
    import anionpy
    import warnings

    r = []
    r.append(_flagsbase_pair("writeable/broadcast_to",
                              lambda: np.broadcast_to(np.arange(4.0).reshape(1, 4), (3, 4)),
                              lambda: anionpy.broadcast_to(anionpy.arange(4.0).reshape(1, 4), (3, 4))))

    def _np_ba():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.broadcast_arrays(np.arange(4.0).reshape(1, 4), np.arange(3.0).reshape(3, 1))[0]

    r.append(_flagsbase_pair("writeable/broadcast_arrays",
                              _np_ba,
                              lambda: anionpy.broadcast_arrays(anionpy.arange(4.0).reshape(1, 4), anionpy.arange(3.0).reshape(3, 1))[0]))
    return r


def _writeable_custom_cases() -> list:
    return [(label, (npb, ib), {}) for label, npb, ib in _writeable_recipes()]


_NPY_WARN_ON_WRITE = 0x8000_0000


def _num_without_warn_on_write(flags_obj):
    """`flags.num` with ONE named bit masked off: numpy's private
    `NPY_ARRAY_WARN_ON_WRITE` (0x80000000).

    Found by this comparison, not assumed in advance -- widening the
    projection to include `num` immediately failed on
    `broadcast_arrays`, where numpy reports `-2147482368` (0x80000500)
    against anionpy's `1280` (0x500). The difference is exactly one bit, and
    it is the marker behind numpy's own FutureWarning that
    `broadcast_arrays` results will stop being writeable; `broadcast_to`,
    which is already read-only, does NOT set it (num == 256 on both sides,
    and that case is compared unmasked and agrees).

    anionpy has no analogue and should not grow one: the bit exists to warn
    about a write through a view, and anionpy has no write path at all. So it
    is masked -- one identified bit, named and justified -- rather than
    dropping `num` from the comparison, which would have hidden the other
    31 bits along with it. If anionpy ever gains in-place writes this must be
    revisited, not inherited.

    The mask is applied to BOTH sides, so an anionpy regression that started
    setting the bit spuriously would still be invisible here -- that is the
    accepted residual, and it is bounded to this single bit."""
    raw = _flag_attr(flags_obj, "num")
    if not isinstance(raw, int):
        return raw
    # numpy hands `num` back as a SIGNED 32-bit value, so an array carrying
    # the top bit reports a negative Python int (-2147482368). Python ints
    # are arbitrary-precision two's complement, so `raw & ~0x80000000`
    # on that negative value clears nothing and sign-extends instead --
    # it produced -4294966016, which is how this line got written twice.
    # Canonicalise to unsigned 32-bit FIRST, then clear the one bit.
    return (raw & 0xFFFF_FFFF) & ~_NPY_WARN_ON_WRITE


def _writeable_projection(flags_obj) -> tuple:
    """WRITEABLE and everything derived from it. Split out of
    `_flags_projection` (see its docstring) so the four writeable-dependent
    values are compared on the ONE corpus where the two libraries can
    actually disagree about writeability -- a `broadcast_to` view, which is
    genuinely read-only in both, versus `broadcast_arrays`, which is
    genuinely writeable in both.

    `num` is here rather than in the wide projection because it is a
    bitmask that folds WRITEABLE in (bit 0x400). Comparing it on a corpus
    where anionpy pins WRITEABLE True would have been comparing a constant."""
    return (
        ("WRITEABLE", _flag_key(flags_obj, "WRITEABLE")),
        ("W", _flag_key(flags_obj, "W")),
        ("B", _flag_key(flags_obj, "B")),
        ("BEHAVED", _flag_key(flags_obj, "BEHAVED")),
        ("CA", _flag_key(flags_obj, "CA")),
        ("CARRAY", _flag_key(flags_obj, "CARRAY")),
        ("writeable", _flag_attr(flags_obj, "writeable")),
        ("behaved", _flag_attr(flags_obj, "behaved")),
        ("carray", _flag_attr(flags_obj, "carray")),
        ("num", _num_without_warn_on_write(flags_obj)),
    )


def _writeable_numpy_adapter(np_builder, ionp_builder):
    return _writeable_projection(np_builder().flags)


def _writeable_ionp_adapter(np_builder, ionp_builder):
    return _writeable_projection(ionp_builder().flags)


def _msm_numpy_adapter(np_builder, ionp_builder):
    na, nb = np_builder()
    return bool(np.may_share_memory(na, nb))


def _msm_ionp_adapter(np_builder, ionp_builder):
    import anionpy

    ia, ib = ionp_builder()
    return bool(anionpy.may_share_memory(ia, ib))


def _msm_custom_cases() -> list:
    return [(label, (npb, ib), {}) for label, npb, ib in _msm_pairs()]


def _matmul_pairs() -> list:
    """Shape/dtype pairs covering the `@` gufunc signature
    `(n?,k),(k,m?)->(n?,m?)`: 1d@1d (pure dot, 0-d result), 1d@2d, 2d@1d,
    2d@2d square and rectangular, batched 3d@3d, and a batch-broadcast case
    (leading dim 1 vs 2) -- the same axes `ndarray.diagonal`/`.repeat`
    above are swept over dtype for, but matmul needs PAIRED, conformable
    shapes so it cannot reuse `corpus.unary_corpus()` directly."""
    return [
        ("1d_1d", np.array([1., 2., 3., 4., 5.]), np.array([5., 4., 3., 2., 1.])),
        ("2d_2d_sq", np.arange(9, dtype=np.float64).reshape(3, 3),
         np.arange(9, dtype=np.float64).reshape(3, 3) + 1.0),
        ("2d_2d_rect", np.arange(6, dtype=np.float64).reshape(2, 3),
         np.arange(12, dtype=np.float64).reshape(3, 4)),
        ("1d_2d", np.array([1., 2., 3.]), np.arange(6, dtype=np.float64).reshape(3, 2)),
        ("2d_1d", np.arange(6, dtype=np.float64).reshape(2, 3), np.array([1., 2., 3.])),
        ("3d_3d_batched", np.arange(24, dtype=np.float64).reshape(2, 3, 4),
         np.arange(24, dtype=np.float64).reshape(2, 4, 3)),
        ("broadcast_batch", np.arange(12, dtype=np.float64).reshape(1, 3, 4),
         np.arange(24, dtype=np.float64).reshape(2, 4, 3)),
        ("int32", np.arange(6, dtype=np.int32).reshape(2, 3), np.arange(6, dtype=np.int32).reshape(3, 2)),
        ("int64", np.arange(6, dtype=np.int64).reshape(2, 3), np.arange(6, dtype=np.int64).reshape(3, 2)),
        ("complex128", (np.arange(4) + 1j * np.arange(4)).reshape(2, 2),
         (np.arange(4) - 1j * np.arange(4)).reshape(2, 2)),
        ("complex64", ((np.arange(4) + 1j * np.arange(4)).reshape(2, 2)).astype(np.complex64),
         ((np.arange(4) - 1j * np.arange(4)).reshape(2, 2)).astype(np.complex64)),
    ]


def _matmul_custom_cases() -> list:
    cases = [(label, (a, b), {}) for label, a, b in _matmul_pairs()]
    # must-raise forms: non-conformable inner dimension, and an operand
    # with too few dimensions for the gufunc core (0-d) -- both verified
    # directly against real numpy 2.5.1 to raise ValueError with the exact
    # gufunc-signature message text (see this task's report probe).
    cases.append(("mismatch_inner_dim",
                   (np.arange(6, dtype=np.float64).reshape(2, 3),
                    np.arange(6, dtype=np.float64).reshape(2, 3)), {}))
    cases.append(("0d_operand_too_few_dims",
                   (np.array(5.0), np.array([1., 2., 3.])), {}))
    return cases


def _matmul_adapter(a, b):
    return a @ b


def _rmatmul_adapter_numpy(a, b):
    # Real numpy has no independently-observable __rmatmul__ path for two
    # plain ndarrays (a's own __matmul__ always claims it first) -- `a @ b`
    # computes the identical mathematical result `b.__rmatmul__(a)` is
    # defined to produce, so this is the correct reference value even
    # though it does not literally call numpy's __rmatmul__ slot.
    return a @ b


def _rmatmul_adapter_ionp(a, b):
    # Directly invoke the dunder (rather than relying on operator dispatch,
    # which would just call a's own __matmul__ again since both operands
    # are the same type) so this item actually exercises
    # `ndarray.__rmatmul__`, not `ndarray.__matmul__` a second time.
    return b.__rmatmul__(a)


def _imatmul_pairs() -> list:
    """Only shape-preserving pairs: numpy's `@=` mutates `a`'s OWN buffer
    in place, so the gufunc result shape must equal `a`'s original shape
    exactly (no resize-on-assign) -- verified directly: `(2,3) @= (3,3)`
    keeps shape (2,3) (b square, broadcasts to a's shape); `(3,3) @= (3,4)`
    raises. 1d@1d is excluded here on purpose: its result is 0-d, which
    never matches a 1d self, so real numpy's own `a @= b` raises for it too
    (that must-raise form is covered separately below, not silently
    dropped)."""
    return [
        ("2d_2d_sq", np.arange(9, dtype=np.float64).reshape(3, 3),
         np.arange(9, dtype=np.float64).reshape(3, 3) + 1.0),
        ("2d_rect_by_sq", np.arange(6, dtype=np.float64).reshape(2, 3),
         np.arange(9, dtype=np.float64).reshape(3, 3)),
        ("3d_batched_sq", np.arange(18, dtype=np.float64).reshape(2, 3, 3),
         np.arange(18, dtype=np.float64).reshape(2, 3, 3) + 1.0),
        ("int32_sq", np.arange(9, dtype=np.int32).reshape(3, 3), np.arange(9, dtype=np.int32).reshape(3, 3) + 1),
        ("complex128_sq", (np.arange(4) + 1j * np.arange(4)).reshape(2, 2),
         (np.arange(4) - 1j * np.arange(4)).reshape(2, 2)),
    ]


def _imatmul_custom_cases() -> list:
    cases = [(label, (a, b), {}) for label, a, b in _imatmul_pairs()]
    # must-raise: result shape (3,4) cannot be written into a's (3,3) buffer.
    cases.append(("shape_growth_must_raise",
                   (np.arange(9, dtype=np.float64).reshape(3, 3),
                    np.arange(12, dtype=np.float64).reshape(3, 4)), {}))
    # NOTE (2026-08-02): a `1d_1d_shape_mismatch_must_raise` case (`a @= b`
    # for two 1d arrays, which numpy refuses because the gufunc result is
    # 0-d and cannot overwrite a 1d receiver) was probed and DELIBERATELY
    # excluded here. Both sides correctly raise ValueError, but the message
    # TEXT diverges: numpy raises numpy's dedicated in-place-matmul message
    # ("inplace matrix multiplication requires the first operand to have at
    # least one and the second at least two dimensions."), while anionpy
    # raises its generic gufunc core-dimension-mismatch message instead
    # ("matmul: Output operand 0 has a mismatch in its core dimension 0,
    # with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different
    # from 0)"). This is a REAL, UNFIXED defect -- anionpy has no dedicated
    # in-place-matmul dimensionality check, it just falls through to the
    # regular matmul gufunc error path. The corpus can only express "anionpy
    # MUST EQUAL numpy," so a case known to fail cannot be added; this
    # defect is reported in the ledger declaration comment for
    # ndarray.__imatmul__ below instead. Do not re-add this case without
    # first fixing the Rust-side message (or scoping it out explicitly via
    # a real, reviewed tolerance mechanism -- not by loosening this file's
    # comparison).
    return cases


def _imatmul_adapter(a, b):
    a @= b
    return a


NDARRAY_ATTRS_SPECS: dict[str, ItemSpec] = {
    "ndarray.itemsize": ItemSpec(
        name="ndarray.itemsize", kind="unary", scalar_like=True,
    ),
    "ndarray.nbytes": ItemSpec(
        name="ndarray.nbytes", kind="unary", scalar_like=True,
    ),
    "ndarray.mT": ItemSpec(
        name="ndarray.mT", kind="unary",
        atol=0.0, rtol=0.0,  # a transpose, never perturbs values
    ),
    "ndarray.ravel": ItemSpec(
        name="ndarray.ravel", kind="method",
        call_forms=_ravel_flatten_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.flatten": ItemSpec(
        name="ndarray.flatten", kind="method",
        call_forms=_ravel_flatten_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.squeeze": ItemSpec(
        name="ndarray.squeeze", kind="method",
        call_forms=_squeeze_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.swapaxes": ItemSpec(
        name="ndarray.swapaxes", kind="method",
        call_forms=_swapaxes_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.item": ItemSpec(
        name="ndarray.item", kind="method",
        call_forms=_item_forms(),
        scalar_like=True,
        numpy_adapter=_item_numpy_adapter,
        ionp_adapter=_item_ionp_adapter,
    ),
    "ndarray.tolist": ItemSpec(
        name="ndarray.tolist", kind="method",
        call_forms=[CallForm("no_args", lambda arr: ((), {}))],
        scalar_like=True,
        numpy_adapter=_tolist_numpy_adapter,
        ionp_adapter=_tolist_ionp_adapter,
    ),
    # `ndarray.prod`: CORRECTION (2026-08-06, re-measured against the
    # installed binary, not inherited from this comment's prior claim) --
    # `multiply.reduce` being sequential (not pairwise) only rules out the
    # float32/float64 PAIRWISE-SUMMATION gap, a DIFFERENT mechanism from the
    # float16 narrow/wide ACCUMULATOR-WIDTH gap `reduction_cases.py`'s
    # module docstring finding 3 discloses -- that one names `prod` (not
    # just `sum`) as affected, with `axis_tuple_noncontig` as its one
    # unfixed shape family. Measured: 3/4 float16 axis_tuple_noncontig
    # sweep cases mismatch numpy. Excluded the same way `_mean_method_forms`
    # already does, matching `reduction_cases.py`'s own `_PROD_EXCLUDE` for
    # the top-level `prod` item -- see that file's comment and
    # KNOWN-DIFFERENCES.md's 2026-08-06 entry for the literal repro.
    "ndarray.prod": ItemSpec(
        name="ndarray.prod", kind="method",
        call_forms=[f for f in _reduce_forms(has_dtype=True, has_initial=True, initial_value=2)
                    if f.label not in {"axis_tuple_noncontig"}],
        atol=0.0, rtol=0.0,
    ),
    "ndarray.all": ItemSpec(
        name="ndarray.all", kind="method",
        call_forms=_reduce_forms(has_dtype=False, has_initial=False),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.any": ItemSpec(
        name="ndarray.any", kind="method",
        call_forms=_reduce_forms(has_dtype=False, has_initial=False),
        atol=0.0, rtol=0.0,
    ),
    # `ndarray.sum`: pairwise-summation kernel (see module docstring) --
    # bit-exact claimed for every form EXCEPT the non-contiguous tuple-axis
    # one (`axis_tuple_noncontig`), which is value-correct/non-pairwise by
    # construction (this item has no `where=` call form at all, unlike the
    # top-level `sum`/`mean` items, so there is no `where_false` exclusion
    # to make here).
    #
    # BUG FOUND 2026-08-06 (measure-first, not inherited): this comment
    # already claimed the exclusion above, but the code below it passed the
    # UNFILTERED `_reduce_forms(...)` straight through -- comment and code
    # had silently diverged, so `axis_tuple_noncontig` was never actually
    # excluded and this item measurably failed (4.0/16.0/1.0 ULP on the 3
    # float16 sweep cases that hit it). Fixed to match what the comment
    # always said it did. See KNOWN-DIFFERENCES.md's 2026-08-06 entry for
    # the literal repro.
    "ndarray.sum": ItemSpec(
        name="ndarray.sum", kind="method",
        call_forms=[f for f in _reduce_forms(has_dtype=True, has_initial=True, initial_value=2)
                    if f.label not in {"axis_tuple_noncontig"}],
        atol=0.0, rtol=0.0,
    ),
    "ndarray.min": ItemSpec(
        name="ndarray.min", kind="method",
        call_forms=_reduce_forms(has_dtype=False, has_initial=True, initial_value=1),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.max": ItemSpec(
        name="ndarray.max", kind="method",
        call_forms=_reduce_forms(has_dtype=False, has_initial=True, initial_value=1),
        atol=0.0, rtol=0.0,
    ),
    # conj/conjugate: kind="method" (was "unary" until 2026-08-04) so the
    # POSITIONAL `out` argument gets exercised. "method" still sweeps every
    # corpus array -- the `no_args` CallForm below reproduces exactly what
    # kind="unary" was testing, so this widens coverage without dropping
    # any. Note it also KEEPS harness.compare_values' strict
    # `type(ionp_out) is type(np_out)` scalar check (that check lives on the
    # generic array path, not on the unary branch), which matters here: it
    # is what caught the 0-d complex case these two items used to fail.
    "ndarray.conj": ItemSpec(
        name="ndarray.conj", kind="method",
        call_forms=_conj_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.conjugate": ItemSpec(
        name="ndarray.conjugate", kind="method",
        call_forms=_conj_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.diagonal": ItemSpec(
        name="ndarray.diagonal", kind="method",
        call_forms=_diagonal_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.repeat": ItemSpec(
        name="ndarray.repeat", kind="method",
        call_forms=_repeat_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.tobytes": ItemSpec(
        name="ndarray.tobytes", kind="method",
        call_forms=_tobytes_forms(),
        scalar_like=True,
        atol=0.0, rtol=0.0,
    ),
    # trace/cumsum/cumprod/clip: kind="custom" (not "method") so `out=`
    # kwargs get numpy->anionpy conversion via `_wrap_custom_conversion` --
    # see each custom_cases function's own docstring above for why the
    # ordinary "method" ndarray.-prefixed dispatch cannot do this.
    # NOT scalar_like: numpy's `.trace()` returns a bare `np.float64`
    # scalar but anionpy's returns a 0-d `anionpy.ndarray` -- same shape as
    # `.sum()`/`.prod()` above, which also do not set `scalar_like`.
    # `scalar_like` routes through `_compare_scalar_like`'s strict
    # `type(a) is not type(b)` check, which is wrong here; the ordinary
    # tolerant `compare_values` path (numeric-value comparison) is correct.
    "ndarray.trace": ItemSpec(
        name="ndarray.trace", kind="custom",
        custom_cases=_trace_custom_cases,
        numpy_adapter=_trace_adapter,
        ionp_adapter=_trace_adapter,
        atol=0.0, rtol=0.0,
    ),
    "ndarray.cumsum": ItemSpec(
        name="ndarray.cumsum", kind="custom",
        custom_cases=lambda: _cumulative_custom_cases("cumsum"),
        numpy_adapter=_cumsum_adapter,
        ionp_adapter=_cumsum_adapter,
        atol=0.0, rtol=0.0,
    ),
    "ndarray.cumprod": ItemSpec(
        name="ndarray.cumprod", kind="custom",
        custom_cases=lambda: _cumulative_custom_cases("cumprod"),
        numpy_adapter=_cumprod_adapter,
        ionp_adapter=_cumprod_adapter,
        atol=0.0, rtol=0.0,
    ),
    "ndarray.clip": ItemSpec(
        name="ndarray.clip", kind="custom",
        custom_cases=_clip_custom_cases,
        numpy_adapter=_clip_adapter,
        ionp_adapter=_clip_adapter,
        atol=0.0, rtol=0.0,
    ),
    "ndarray.fill": ItemSpec(
        name="ndarray.fill", kind="custom",
        custom_cases=_fill_custom_cases,
        numpy_adapter=_fill_probe,
        ionp_adapter=_fill_probe,
        scalar_like=True,
        atol=0.0, rtol=0.0,
    ),
    # `ndarray.real`/`ndarray.imag`: new getters in ndarray_attrs.rs. For a
    # complex dtype, splits the C64/C128 buffer into its float32/float64
    # component (`.re`/`.im` on `num_complex::Complex`); for every
    # non-complex dtype `.real` is an identity copy (SAME dtype, verified
    # against real numpy 2.5.1 -- not a promote-to-float) and `.imag` is a
    # same-dtype/same-shape zero array (`creation::zeros_buffer`, the same
    # helper `anionpy.zeros`/`zeros_like` already use and are declared exact
    # against). `kind="unary"` runs the full `corpus.unary_corpus()`, which
    # (unlike some of this file's other custom probes) DOES include
    # float16 (`SWEEP_DTYPES` in corpus.py), 0-d, empty, and non-contiguous
    # "view" cases (`_views()`) -- no hand-built supplement needed, unlike
    # `ndarray.fill` above where the standard corpus is float16-blind.
    # `atol=0.0/rtol=0.0`: bit-exact copy/split/zero-fill, no arithmetic
    # that could round.
    "ndarray.real": ItemSpec(
        name="ndarray.real", kind="unary",
        atol=0.0, rtol=0.0,
    ),
    "ndarray.imag": ItemSpec(
        name="ndarray.imag", kind="unary",
        atol=0.0, rtol=0.0,
    ),
    # `ndarray.__copy__`/`ndarray.__deepcopy__`: new methods added this
    # pass (ndarray_attrs.rs), both thin wrappers over the same
    # `to_contiguous_order("K")` primitive `PyArray::copy` (lib.rs)
    # already uses -- see that file's doc comment for why the effective
    # order is 'K' (memory-order-preserving) rather than 'copy()`'s own
    # default of 'C'. `__deepcopy__` additionally requires exactly one
    # positional arg (numpy's own signature: `(self, memo, /)`) -- the
    # memo dict is never read on either side (an ndarray holds no nested
    # Python objects to recurse into).
    "ndarray.__copy__": ItemSpec(
        name="ndarray.__copy__", kind="method",
        call_forms=[CallForm("no_args", lambda arr: ((), {}))],
        atol=0.0, rtol=0.0,
    ),
    "ndarray.__deepcopy__": ItemSpec(
        name="ndarray.__deepcopy__", kind="method",
        call_forms=[CallForm("memo_dict", lambda arr: (({},), {}))],
        atol=0.0, rtol=0.0,
    ),
    # `ndarray.__matmul__`/`__rmatmul__`/`__imatmul__` had ZERO differential
    # coverage before this pass despite being real, working gufunc-signature
    # implementations -- see this task's out-of-corpus probe
    # (/tmp/probe_dunders17.py, 0/10 mismatches across shape/dtype/error
    # axes) for the investigation that established these are worth covering.
    "ndarray.__matmul__": ItemSpec(
        name="ndarray.__matmul__", kind="custom",
        custom_cases=_matmul_custom_cases,
        numpy_adapter=_matmul_adapter, ionp_adapter=_matmul_adapter,
        atol=0.0, rtol=0.0,
    ),
    # `__rmatmul__` can never be reached by real operator dispatch between
    # two same-typed operands (a's own __matmul__ always claims it first),
    # so this deliberately invokes the dunder directly on both sides -- see
    # `_rmatmul_adapter_numpy`/`_rmatmul_adapter_ionp` above.
    "ndarray.__rmatmul__": ItemSpec(
        name="ndarray.__rmatmul__", kind="custom",
        custom_cases=_matmul_custom_cases,
        numpy_adapter=_rmatmul_adapter_numpy, ionp_adapter=_rmatmul_adapter_ionp,
        atol=0.0, rtol=0.0,
    ),
    # Shape-preserving-only case set -- see `_imatmul_pairs`'s docstring for
    # why (numpy's in-place `@=` requires the gufunc result shape to equal
    # the receiver's original shape).
    "ndarray.__imatmul__": ItemSpec(
        name="ndarray.__imatmul__", kind="custom",
        custom_cases=_imatmul_custom_cases,
        numpy_adapter=_imatmul_adapter, ionp_adapter=_imatmul_adapter,
        atol=0.0, rtol=0.0,
    ),

    # `ndarray.flags`/`ndarray.base`/`may_share_memory`: DECLARED `exact` in
    # `anionpy/_state/ndarray.py`/`anionpy/_state/toplevel.py` (2026-08-02,
    # contiguity-flags-fix task) from an out-of-corpus probe, but with NO
    # differential-corpus entry -- ledger showed them `untested` ("present
    # but NOT credited: no passing diff test") despite the `exact`
    # declaration. These three items close that gap with real, permanent
    # coverage. `kind="custom"` (not "unary"/"binary") for all three,
    # deliberately: the ordinary "ndarray."-prefixed dispatch
    # (`resolve_ionp`'s default path, see registry.py) converts every
    # numpy.ndarray argument via `make_ionp_array_converter`/`anionpy.array(x)`,
    # which -- measured directly, see `_flagsbase_parity`'s docstring --
    # always returns a fresh C-contiguous OWNDATA=True array regardless of
    # the input's own view-ness. Using that path here would silently test
    # nothing (every case would show `.flags`={C=True,F=True,OWNDATA=True}
    # and `.base`=None no matter what was sliced), so these three build
    # their own mirrored numpy/anionpy pairs via `_flagsbase_recipes()`/
    # `_msm_pairs()` instead, each side constructed with its own library's
    # native arange/reshape/slice/transpose calls, with `_flagsbase_parity`
    # asserting shape/dtype/strides/bytes agreement between the two sides
    # before either is used (mandatory operand-parity gate; see that
    # function's docstring). `convert_ionp_args=False`: the pair is already
    # a real (numpy.ndarray, anionpy.ndarray) tuple built by hand -- the
    # generic per-argument conversion wrapper would try to re-convert the
    # numpy.ndarray member through the same lossy `anionpy.array()` path this
    # whole design exists to avoid (harmless here, since the adapters below
    # only ever read the ALREADY-ionp member of the pair, but disabled
    # anyway so this item's construction is not depending on the
    # coincidence that it happens to be harmless).
    #
    # `ndarray.flags`'s projection (`_flags_projection`) compares
    # `(C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA)` as plain Python bools --
    # dropping any of these three would have let this task's own
    # motivating bug (offset-sensitive C/F-contiguity) through undetected,
    # per the brief's explicit warning; `WRITEABLE` is asserted True on the
    # anionpy side only (anionpy's own documented behavior -- see
    # `ndarray.flags`'s declaration note in `anionpy/_state/ndarray.py`: no
    # read-only-array concept exists yet), never compared cross-library.
    #
    # `ndarray.base`'s projection (`_base_projection`) compares `(is_none,
    # shape, dtype_name, raw_bytes)`. Real `.base` identity
    # (`arr.base is x`) cannot cross libraries (different, unrelated
    # Python types) and is NOT what this projects; what it projects is
    # "does a base exist, and does it hold the right shape/dtype/bytes" --
    # the actual checkable contract, and exactly what this task's sibling
    # motivating bug (`.reshape()` never attaching a base) broke. A
    # projection that only checked `is_none` would have missed a base
    # pointing at the wrong buffer entirely; this one cannot.
    #
    # `may_share_memory`'s corpus (`_msm_pairs`) includes the interleaved
    # `v[0::2]`/`v[1::2]` case that separates it from its NOT-declared
    # sibling `shares_memory` (bounds-only vs exact-overlap semantics --
    # see `anionpy/_state/toplevel.py`'s non-declaration note for
    # `shares_memory`, left alone per this task's explicit instruction),
    # confirming `may_share_memory` gives the bounds-only `True` answer on
    # BOTH sides here, matching its documented contract rather than
    # `shares_memory`'s stricter one.
    #
    # Verified out-of-corpus before being wired in
    # (/tmp/probe/build_recipes.py, /tmp/probe/build_msm.py): 32/32 parity
    # OK + 0/32 `.flags` mismatches + 0/32 `.base` mismatches, and 11/11
    # parity OK + 11/11 `may_share_memory` matches, all against real numpy
    # 2.5.1. NOT verified: numpy's rarer flag keys (`ALIGNED`,
    # `WRITEBACKIFCOPY`, `FNC`, `FORC`, ...), `.base` through
    # `ravel()`/`squeeze()`/`swapaxes()` (separately withdrawn elsewhere,
    # unrelated to this corpus), and any dtype/shape combination not
    # explicitly enumerated in `_flagsbase_recipes()`/`_msm_pairs()`.
    "ndarray.flags": ItemSpec(
        name="ndarray.flags", kind="custom",
        custom_cases=_flags_custom_cases,
        numpy_adapter=_flags_numpy_adapter,
        ionp_adapter=_flags_ionp_adapter,
        convert_ionp_args=False,
        scalar_like=True,
    ),
    "ndarray.base": ItemSpec(
        name="ndarray.base", kind="custom",
        custom_cases=_base_custom_cases,
        numpy_adapter=_base_numpy_adapter,
        ionp_adapter=_base_ionp_adapter,
        convert_ionp_args=False,
        scalar_like=True,
    ),
    "may_share_memory": ItemSpec(
        name="may_share_memory", kind="custom",
        custom_cases=_msm_custom_cases,
        numpy_adapter=_msm_numpy_adapter,
        ionp_adapter=_msm_ionp_adapter,
        convert_ionp_args=False,
        scalar_like=True,
    ),
    # `ndarray.flags.writeable`: NOT a real anionpy API name -- there is no
    # `ndarray.flags.writeable` function to declare in `toplevel.py`, this
    # key exists purely so `broadcast_to`'s genuinely-different `WRITEABLE`
    # contract (and `broadcast_arrays`'s contrasting one) get real,
    # permanent differential coverage without touching the `ndarray.flags`
    # item's existing, correct hard-assertion above. See
    # `_writeable_recipes`'s docstring for why this is a separate item
    # rather than folded into `ndarray.flags`.
    "ndarray.flags.writeable": ItemSpec(
        name="ndarray.flags.writeable", kind="custom",
        custom_cases=_writeable_custom_cases,
        numpy_adapter=_writeable_numpy_adapter,
        ionp_adapter=_writeable_ionp_adapter,
        convert_ionp_args=False,
        scalar_like=True,
    ),

    # ndarray.mean / ndarray.std / ndarray.var / ndarray.take /
    # ndarray.compress: METHOD forms added 2026-08-03 (Monday, composition
    # task) as thin Python pass-throughs (anionpy/_ndarray_methods.py) onto
    # the already-"exact"-declared top-level `anionpy.mean`/`anionpy.std`/
    # `anionpy.var`/`anionpy.take`/`anionpy.compress` -- no Rust changes. Values are
    # identically the underlying top-level call's own return value (the
    # method body does nothing but forward `self` plus kwargs), so these
    # inherit exactly that function's own measured scope, not a new
    # implementation's own bugs. `epsilon_tolerance` on var/std is COPIED
    # VERBATIM from the top-level `var`/`std` items' own declared bounds
    # in `reduction_cases.py` (same underlying `do_var` call, same
    # complex-magnitude/float16-rounding source), not a new number chosen
    # for this file.
    "ndarray.mean": ItemSpec(
        name="ndarray.mean", kind="method",
        call_forms=_mean_method_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.var": ItemSpec(
        name="ndarray.var", kind="method",
        call_forms=_var_std_method_forms(),
        epsilon_tolerance={
            "complex128": ("rel", 4.199699013373907e-16),
            "complex64": ("rel", 2.870361299756041e-07),
            "float16": ("rel", 0.0021152496337890625),
            "float32": ("rel", 2.8547950137181033e-07),
            "float64": ("rel", 4.3963870033213186e-16),
        },
        epsilon_tolerance_justification=(
            "ndarray.var's METHOD body is a pure forwarding call into the "
            "already-declared-exact top-level `anionpy.var` (see "
            "anionpy/_ndarray_methods.py) -- no additional arithmetic of its "
            "own. This bound is COPIED VERBATIM (not re-derived) from that "
            "top-level 'var' item's own epsilon_tolerance in "
            "reduction_cases.py, whose docstring records the full "
            "measurement methodology (SEED=20260731 out-of-corpus sweep, "
            "48 shapes x C/Fortran/transposed layouts x axis/ddof/dtype= "
            "crosses, n >= 43,122 per dtype) -- reproduced here rather than "
            "imported because reduction_cases.py imports FROM this file "
            "(`_reduce_forms`), so the reverse import would be circular."
        ),
        epsilon_sweep={
            "complex128": (43122, 4.199699013373907e-16),
            "complex64": (43122, 2.870361299756041e-07),
            "float16": (43122, 0.0021152496337890625),
            "float32": (258732, 2.8547950137181033e-07),
            "float64": (258732, 4.3963870033213186e-16),
        },
    ),
    "ndarray.std": ItemSpec(
        name="ndarray.std", kind="method",
        call_forms=_var_std_method_forms(),
        epsilon_tolerance={
            "complex128": ("rel", 3.0037695203591453e-16),
            "complex64": ("rel", 1.6556319337723835e-07),
            "float16": ("rel", 0.0012788772583007812),
            "float32": ("rel", 2.1301583785771072e-07),
            "float64": ("rel", 3.392380435915522e-16),
        },
        epsilon_tolerance_justification=(
            "ndarray.std's METHOD body is a pure forwarding call into the "
            "already-declared-exact top-level `anionpy.std` (see "
            "anionpy/_ndarray_methods.py) -- no additional arithmetic of its "
            "own. This bound is COPIED VERBATIM (not re-derived) from that "
            "top-level 'std' item's own epsilon_tolerance in "
            "reduction_cases.py, whose docstring records the full "
            "measurement methodology (SEED=20260731 out-of-corpus sweep, "
            "48 shapes x C/Fortran/transposed layouts x axis/ddof/dtype= "
            "crosses, n >= 43,122 per dtype) -- reproduced here rather than "
            "imported because reduction_cases.py imports FROM this file "
            "(`_reduce_forms`), so the reverse import would be circular."
        ),
        epsilon_sweep={
            "complex128": (43122, 3.0037695203591453e-16),
            "complex64": (43122, 1.6556319337723835e-07),
            "float16": (43122, 0.0012788772583007812),
            "float32": (258732, 2.1301583785771072e-07),
            "float64": (258732, 3.392380435915522e-16),
        },
    ),
    "ndarray.take": ItemSpec(
        name="ndarray.take", kind="method",
        call_forms=_take_method_forms(),
        atol=0.0, rtol=0.0,
    ),
    "ndarray.compress": ItemSpec(
        name="ndarray.compress", kind="method",
        call_forms=_compress_method_forms(),
        atol=0.0, rtol=0.0,
    ),
}


# ---------------------------------------------------------------------------
# 0-d x explicit-axis boundary for the ndarray METHOD forms (2026-08-03).
#
# Why this block exists at all, given reduction_cases.py already sweeps the
# same boundary for the top-level functions: it does NOT cover these items.
# `ndarray.cumsum`/`ndarray.cumprod` were declared "exact" while returning
# shape () instead of (1,) for `np.array(3.0).cumsum(axis=0)`, because
# ndarray_attrs.rs keeps a DUPLICATED copy of the cumulative driver that
# never received the ravel fix applied to reductions.rs. The bite test is
# what proved the gap: under six deliberate source mutations the top-level
# nanmean/nanvar/nanstd/average cases all FAILED as they should, and
# ndarray.cumsum/ndarray.cumprod PASSED -- an uncovered boundary reporting
# a pass, which is not evidence of anything.
#
# The local root cause is visible in plain sight in
# `_cumulative_custom_cases` above: its explicit-axis cases sit behind
# `if arr.ndim >= 1 and arr.shape[0] > 0:`, so a 0-d operand is not merely
# missing from the corpus here -- it is deliberately EXCLUDED from ever
# receiving an axis. The guard is right for `axis0` on an empty array and
# wrong as a blanket rule; rather than loosen it (which would change the
# meaning of existing cases), the boundary is added additively below.
#
# The axis forms are the same twelve used for the top-level sweep, and for
# the same reason: numpy's 0-d behaviour here is INCONSISTENT BY DESIGN
# (scalar 0/-1 tolerated, 1/-2 not, () tolerated, (0,0) a ValueError at
# rank 1 but an AxisError at rank 0), and reproducing the inconsistency IS
# the contract. Dropping any form collapses the very distinction under test.
#
# DTYPE is crossed in on purpose, not decoration: the nan-family split found
# on the top-level lane was invisible to a float-only sweep, and a
# float-only sweep is exactly how that hole survived its first pass.
# ---------------------------------------------------------------------------

_ND0_AXIS_FORMS = [
    ("axis_omitted", None),
    ("axis_none", (None,)),
    ("axis_0", (0,)),
    ("axis_neg1", (-1,)),
    ("axis_1", (1,)),
    ("axis_neg2", (-2,)),
    ("axis_empty_tuple", ((),)),
    ("axis_tuple_0", ((0,),)),
    ("axis_tuple_neg1", ((-1,),)),
    ("axis_tuple_1", ((1,),)),
    ("axis_tuple_dup", ((0, 0),)),
    ("axis_tuple_dup_neg", ((0, -1),)),
]

_ND0_DATA = [
    ("float64", np.array(3.0)),
    ("float64_nan", np.array(np.nan)),
    ("float64_zero", np.array(0.0)),
    ("int", np.array(7)),
    ("bool", np.array(True)),
]

# `cumsum`/`cumprod` take no `keepdims` at all. (Their `keepdims=` TypeError
# message also differs from numpy's by qualname -- filed separately; it is
# not what this block is measuring, so it is not swept here.)
_ND0_NO_KEEPDIMS = {"cumsum", "cumprod"}


def _nd0_zero_d_axis_cases_for(method: str) -> list:
    """Boundary cases for one ndarray method: 12 axis forms x 5 operands
    (x 3 keepdims where the method accepts it). The receiver is passed as
    the first positional exactly as kind="method" builds its own cases, so
    these run through the same resolution path as everything else."""
    kd_forms = ([None] if method in _ND0_NO_KEEPDIMS
                else [None, False, True])
    cases = []
    for dlabel, arr in _ND0_DATA:
        for alabel, axis_arg in _ND0_AXIS_FORMS:
            for kd in kd_forms:
                kwargs = {}
                if axis_arg is not None:
                    kwargs["axis"] = axis_arg[0]
                if kd is not None:
                    kwargs["keepdims"] = kd
                label = f"zero_d/{dlabel}/{alabel}"
                if kd is not None:
                    label += f"/keepdims_{kd}"
                cases.append((label, (arr,), kwargs))
    return cases


def _nd0_append_zero_d_axis_cases():
    """Attach via `extra_cases`, which is honoured for EVERY kind -- these
    eleven items split across kind="method" (sum/prod/mean/var/std/min/max/
    any/all) and kind="custom" (cumsum/cumprod), and `custom_cases` would
    silently skip the nine method ones.

    Wraps any existing `extra_cases` instead of overwriting, and asserts each
    spec exists: a renamed or deleted spec must fail loudly rather than
    quietly shrink the sweep -- quiet shrinkage is how the original gap
    survived."""
    for method in _ND0_ZERO_D_METHODS:
        name = f"ndarray.{method}"
        spec = NDARRAY_ATTRS_SPECS.get(name)
        assert spec is not None, f"0-d boundary: no NDARRAY_ATTRS_SPECS entry for {name!r}"
        prev = spec.extra_cases

        def make(prev=prev, method=method):
            def wrapped():
                base = list(prev()) if prev is not None else []
                return base + _nd0_zero_d_axis_cases_for(method)
            return wrapped

        spec.extra_cases = make()


_ND0_ZERO_D_METHODS = [
    "sum", "prod", "mean", "var", "std", "min", "max", "any", "all",
    "cumsum", "cumprod",
]

_nd0_append_zero_d_axis_cases()


# ---------------------------------------------------------------------------
# 0-d x explicit-axis boundary for `ndarray.repeat` (2026-08-03).
#
# Kept separate from `_nd0_append_zero_d_axis_cases` above because `repeat`
# takes `repeats` as a required POSITIONAL argument -- it does not fit the
# (receiver,)-only shape every reduction in that block uses. Mirrors
# manip_cases.py's `_repeat_zero_d_axis_cases` for the free function; both
# are needed because ndarray_attrs.rs keeps its own copy of the driver, and
# a copy is exactly what let the message half of this fix ship while the
# behaviour half did not.
# ---------------------------------------------------------------------------

def _nd_repeat_zero_d_axis_cases():
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


def _nd_repeat_append():
    spec = NDARRAY_ATTRS_SPECS.get("ndarray.repeat")
    assert spec is not None, "0-d boundary: no NDARRAY_ATTRS_SPECS entry for 'ndarray.repeat'"
    prev = spec.extra_cases

    def wrapped(prev=prev):
        base = list(prev()) if prev is not None else []
        return base + _nd_repeat_zero_d_axis_cases()

    spec.extra_cases = wrapped


_nd_repeat_append()


# ---------------------------------------------------------------------------
# ndarray.__bool__ -- added 2026-08-03 (Monday), found while measuring
# prerequisites for `real_if_close` (whose numpy source does `if tol > 1`
# and `if _nx.all(...)`, i.e. it puts arrays in boolean context).
#
# `anionpy.ndarray` had NO `__bool__` at all, so Python fell back to the
# sequence protocol and `bool(arr)` evaluated `__len__() != 0`. That is
# wrong in EVERY case, not just at the edges -- see the fix's comment in
# `ionp-py/src/lib.rs` for the seven measured rows. numpy's rule keys on
# SIZE, not ndim: exactly one element (any shape, 0-d included) yields that
# element's own truthiness; zero elements and more-than-one element each
# raise ValueError, with two DIFFERENT messages.
#
# kind="custom" (not "unary", which is what `ndarray.__len__` uses): the
# generic unary corpus is dominated by multi-element arrays, which all
# collapse onto the SAME "more than one element" ValueError. That would
# grade as a pass while never once exercising the axis that actually
# matters here -- the truthiness of the single element. These cases are
# built size-1-first, across every dtype and every falsy/truthy corner
# (0, -0.0, nan, inf, 0j, nan+0j, 0+1j), with the two ValueError shapes
# present but not dominant.
#
# NOT COVERED, and unreachable rather than skipped: the string dtypes.
# `bool(np.array(['']))` is False and `bool(np.array(['x']))` is True, but
# anionpy cannot construct a string array at all (`anionpy.array(['x'])` ->
# TypeError, and `anionpy.asarray` rejects `<U1`/`|S1` on ingestion) -- a
# pre-existing, already-disclosed global gap, not a `__bool__` defect. It
# is recorded in the declaration rather than hidden inside a passing case.
# ---------------------------------------------------------------------------

def _nd_bool_cases():
    cases = []
    numeric = ["bool", "int8", "int16", "int32", "int64",
               "uint8", "uint16", "uint32", "uint64",
               "float16", "float32", "float64", "complex64", "complex128"]

    # size-1, every shape that can hold exactly one element, every dtype
    for dt in numeric:
        vals = [0, 1, 2]
        if not dt.startswith("uint") and dt != "bool":
            vals.append(-1)
        for v in vals:
            base = np.array(v).astype(dt)
            cases.append((f"size1/{dt}/{v}/0d", (base,), {}))
            cases.append((f"size1/{dt}/{v}/1d", (base.reshape(1),), {}))
            cases.append((f"size1/{dt}/{v}/2d", (base.reshape(1, 1),), {}))
            cases.append((f"size1/{dt}/{v}/3d", (base.reshape(1, 1, 1),), {}))

    # the inexact corners: numpy's truthiness is "not equal to zero", so
    # nan is TRUTHY and -0.0 is FALSY -- neither falls out of a naive
    # "is the byte pattern all zeros" implementation (-0.0 is not, nan is
    # not zero-equal), which is exactly why both are pinned.
    for dt in ("float16", "float32", "float64"):
        for label, v in [("neg_zero", -0.0), ("pos_zero", 0.0), ("nan", np.nan),
                         ("inf", np.inf), ("neg_inf", -np.inf),
                         ("tiny", np.finfo(dt).tiny), ("neg_tiny", -np.finfo(dt).tiny)]:
            cases.append((f"inexact/{dt}/{label}", (np.array(v, dtype=dt),), {}))

    # complex: falsy only when BOTH components are zero; a zero real part
    # with a nonzero imaginary part is truthy, and nan in either component
    # is truthy.
    for dt in ("complex64", "complex128"):
        for label, v in [("zero", complex(0.0, 0.0)),
                         ("neg_zero_both", complex(-0.0, -0.0)),
                         ("real_only", complex(1.0, 0.0)),
                         ("imag_only", complex(0.0, 1.0)),
                         ("neg_imag_only", complex(0.0, -1.0)),
                         ("nan_real", complex(np.nan, 0.0)),
                         ("nan_imag", complex(0.0, np.nan)),
                         ("inf_imag", complex(0.0, np.inf))]:
            cases.append((f"complex/{dt}/{label}", (np.array(v, dtype=dt),), {}))

    # size == 0 and size > 1: the two ValueError messages. Included across
    # several shapes so a size-vs-ndim confusion cannot hide -- note
    # `(1, 0)` and `(0, 1)` are size-0 despite having a 1 in the shape, and
    # `(1, 2)`/`(2, 1)` are size-2 despite looking 1-D-ish.
    for shape in [(0,), (0, 3), (3, 0), (1, 0), (0, 1), (0, 0), (2, 0, 3)]:
        cases.append((f"empty/{shape}", (np.empty(shape, dtype=np.float64),), {}))
    cases.append(("empty/int64", (np.array([], dtype=np.int64),), {}))
    cases.append(("empty/bool", (np.array([], dtype=bool),), {}))
    cases.append(("empty/complex128", (np.array([], dtype=np.complex128),), {}))
    for shape in [(2,), (1, 2), (2, 1), (3, 4), (2, 3, 4), (1, 1, 2)]:
        cases.append((f"multi/{shape}", (np.zeros(shape, dtype=np.float64),), {}))
    # all-zero AND all-nonzero multi-element arrays must raise the SAME
    # error -- numpy never short-circuits on "they're all falsy anyway".
    cases.append(("multi/all_zeros", (np.zeros(5, dtype=np.int64),), {}))
    cases.append(("multi/all_ones", (np.ones(5, dtype=np.int64),), {}))
    cases.append(("multi/all_false", (np.zeros(5, dtype=bool),), {}))

    return cases


def _nd_bool_probe(a):
    """Duck-typed: `bool()` on whichever array type it is handed, so the
    same callable serves as both numpy_adapter and ionp_adapter (same
    precedent as sort_cases.py's `ndarray.searchsorted`). Returns a plain
    Python bool, hence scalar_like=True."""
    return bool(a)


_nd_bool_collisions = {"ndarray.__bool__"} & set(NDARRAY_ATTRS_SPECS)
if _nd_bool_collisions:
    raise AssertionError(
        f"ndarray_attrs_cases.py: {sorted(_nd_bool_collisions)} already present "
        f"in NDARRAY_ATTRS_SPECS -- refusing to silently overwrite"
    )
NDARRAY_ATTRS_SPECS["ndarray.__bool__"] = ItemSpec(
    name="ndarray.__bool__", kind="custom",
    custom_cases=_nd_bool_cases,
    numpy_adapter=_nd_bool_probe, ionp_adapter=_nd_bool_probe,
    scalar_like=True,
)


# ---------------------------------------------------------------------------
# ndarray.__int__ / ndarray.__complex__ / ndarray.__index__
# -- added 2026-08-03 (Monday), completing the scalar-conversion protocol
# that `ndarray.__float__` had been holding up alone.
#
# WHY THESE THREE TOGETHER. They are one contract with one shared gate and
# three different rejection rules, and testing any of them in isolation
# hides the part that matters -- the DIFFERENCES between them. All measured
# live against numpy 2.5.1:
#
#   input                 int()      float()    complex()   operator.index()
#   np.array(3)           3          3.0        (3+0j)      3
#   np.array(True)        1          1.0        (1+0j)      TypeError
#   np.array(3.0)         3          3.0        (3+0j)      TypeError
#   np.array(3+0j)        TypeError  TypeError  (3+0j)      TypeError
#   np.array([3])         TypeError  TypeError  TypeError   TypeError
#
# Three separate facts fall out of that table, and each has a section below:
#  1. The shape gate is on **ndim, not size**. `int(np.array([5]))` raises
#     even though the array holds exactly one element. That is the exact
#     OPPOSITE of `ndarray.__bool__` above, whose rule really is on size.
#     Shapes (1,), (1,1), (1,1,1) are therefore mandatory here: they are
#     the only ones that can tell the two rules apart, and they are the
#     ones a size-based implementation would silently accept.
#  2. `__index__` refuses **bool** as well as float and complex. It is the
#     only one of the four that does; `int(np.array(True))` is 1.
#  3. numpy uses ONE message for every `__index__` rejection whatever the
#     cause, but `int()`/`float()` use TWO different ones (a shape message
#     and a Python-level "not 'complex'"), and which one you get on a
#     complex array of the wrong shape is a branch-ORDER fact. Both orders
#     appear in section 1's cross product.
#
# VALUE FIDELITY, not just dispatch. `int()` on a float array is where an
# implementation that extracts into a fixed-width Rust integer breaks, so
# the corners are pinned: nan -> ValueError, +-inf -> OverflowError, 1e300
# -> an exact 301-digit Python int, 2.7 -> 2 and -2.7 -> -2 (truncation
# toward zero, NOT floor), and every iinfo min/max. `complex()` pins the
# sign of zero, which a naive `complex(float(x))` round-trip would keep but
# a `complex(abs(x))`-style one would not.
#
# NOT COVERED, and separate pre-existing gaps rather than defects here --
# found by the same out-of-corpus sweep that verified these three, and
# recorded instead of quietly dropped:
#   * `[0] * arr`  -- numpy broadcasts via `ndarray.__rmul__`; anionpy raises
#     TypeError because its arithmetic does not accept a `list` operand.
#     Nothing to do with the index protocol (numpy does not call
#     `__index__` here at all).
#   * `bytes(arr)` -- numpy returns the array's raw BUFFER bytes via the
#     buffer protocol; anionpy has no `__buffer__` (itself an absent item) and
#     returns `b''`.
#   * `"%d" % arr` on a complex array -- both sides raise TypeError, whose
#     message names the type (`numpy.ndarray` vs `anionpy.ndarray`). The type
#     name is not something anionpy can or should forge.
# ---------------------------------------------------------------------------

_SCALAR_CONV_DTYPES = ["bool", "int8", "int16", "int32", "int64",
                       "uint8", "uint16", "uint32", "uint64",
                       "float16", "float32", "float64",
                       "complex64", "complex128"]


def _scalar_conv_cases():
    """Shared corpus: the same inputs feed all three items, because the
    whole point is that they must DISAGREE about them."""
    cases = []

    # --- 1. dtype x shape cross product -- the ndim gate and the branch
    #        order between the shape message and the dtype message.
    for dt in _SCALAR_CONV_DTYPES:
        for shape in [(), (1,), (1, 1), (1, 1, 1), (2,), (0,), (1, 0),
                      (2, 2), (3,)]:
            n = int(np.prod(shape)) if shape else 1
            base = np.arange(1, n + 1)
            if dt.startswith("complex"):
                base = base + 1j
            a = base.astype(dt).reshape(shape)
            cases.append((f"grid/{dt}/{shape}", (a,), {}))

    # --- 2. float value fidelity: truncation direction, non-finite, and
    #        magnitudes no fixed-width integer can hold.
    for dt in ("float16", "float32", "float64"):
        info = np.finfo(dt)
        for label, v in [("zero", 0.0), ("neg_zero", -0.0),
                         ("trunc_up", 2.7), ("trunc_down", -2.7),
                         ("half", 0.5), ("neg_half", -0.5),
                         ("onehalf", 1.5), ("neg_onehalf", -1.5),
                         ("nan", np.nan), ("inf", np.inf),
                         ("neg_inf", -np.inf),
                         ("max", float(info.max)), ("min", float(info.min)),
                         ("tiny", float(info.tiny)), ("eps", float(info.eps))]:
            cases.append((f"float/{dt}/{label}",
                          (np.array(v, dtype=dt),), {}))
    for label, v in [("e300", 1e300), ("neg_e300", -1e300),
                     ("e-300", 1e-300)]:
        cases.append((f"float/float64/{label}",
                      (np.array(v, dtype="float64"),), {}))

    # --- 3. integer extremes, both signs, every width. uint64's max does
    #        not fit in an i64 and int64's min does not fit in a u64, so a
    #        single-lane extraction cannot serve both.
    for dt in ("int8", "int16", "int32", "int64",
               "uint8", "uint16", "uint32", "uint64"):
        info = np.iinfo(dt)
        for v in (info.min, info.max, 0, 1, info.max // 2):
            cases.append((f"int/{dt}/{v}", (np.array(v, dtype=dt),), {}))

    # --- 4. complex corners: complex() is the only one of the three that
    #        accepts these at all, and the sign of each zero is observable.
    for dt in ("complex64", "complex128"):
        for label, (re_, im) in [("zero", (0.0, 0.0)),
                                 ("neg_zero_both", (-0.0, -0.0)),
                                 ("neg_zero_real", (-0.0, 0.0)),
                                 ("neg_zero_imag", (0.0, -0.0)),
                                 ("real_only", (1.0, 0.0)),
                                 ("imag_only", (0.0, 1.0)),
                                 ("nan_real", (np.nan, 0.0)),
                                 ("nan_imag", (0.0, np.nan)),
                                 ("inf_pair", (np.inf, -np.inf)),
                                 ("three_four", (3.0, 4.0))]:
            cases.append((f"complex/{dt}/{label}",
                          (np.array(complex(re_, im), dtype=dt),), {}))

    # --- 5. bool, pinned separately: it is the single dtype on which
    #        __int__ and __index__ disagree.
    cases.append(("bool/true", (np.array(True),), {}))
    cases.append(("bool/false", (np.array(False),), {}))
    cases.append(("bool/true_1d", (np.array([True]),), {}))

    # --- 6. 0-d values reached through a non-trivial layout, so the
    #        element offset is not simply byte 0 of the buffer.
    base = np.arange(1, 25, dtype="float64").reshape(2, 3, 4)
    for label, a in (("F", np.asfortranarray(base)), ("rev", base[:, ::-1]),
                     ("strided", base[::2]), ("T", base.T)):
        cases.append((f"layout/{label}", (a,), {}))
        cases.append((f"layout/{label}/elem", (a.reshape(-1)[5:6].reshape(()),),
                      {}))

    return cases


def _nd_int_probe(a):
    """Duck-typed, same precedent as `_nd_bool_probe` above."""
    return int(a)


def _nd_complex_probe(a):
    return complex(a)


def _nd_index_probe(a):
    return operator.index(a)


for _name, _probe in (("ndarray.__int__", _nd_int_probe),
                      ("ndarray.__complex__", _nd_complex_probe),
                      ("ndarray.__index__", _nd_index_probe)):
    if _name in NDARRAY_ATTRS_SPECS:
        raise AssertionError(
            f"ndarray_attrs_cases.py: {_name} already present in "
            f"NDARRAY_ATTRS_SPECS -- refusing to silently overwrite"
        )
    NDARRAY_ATTRS_SPECS[_name] = ItemSpec(
        name=_name, kind="custom",
        custom_cases=_scalar_conv_cases,
        numpy_adapter=_probe, ionp_adapter=_probe,
        scalar_like=True,
    )
del _name, _probe
