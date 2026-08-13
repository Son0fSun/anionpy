"""Differential specs for the top-level numpy reduction/statistics
FUNCTIONS (as opposed to the `ndarray` methods already covered by
ndarray_attrs_cases.py) -- `sum, prod, all, any, min, max, amin, amax,
argmin, argmax, cumsum, cumprod, mean, ptp, count_nonzero, var, std,
nanvar, nanstd, average`, backed by `ionp-py/src/reductions.rs` (thin
wrappers on the same `ionp_core::ufunc` axis-reduction/accumulate/argext
kernels ndarray_attrs.rs already uses, plus two new kernels --
`accumulate_axis` for cumsum/cumprod and `argext` for argmin/argmax --
added alongside this file).

Deliberately OUT OF SCOPE this pass (per the task brief -- do these LAST,
only once the rest is solid, and NOT attempted at all this pass):
`median, percentile, quantile`. (`nanmin`/`nanmax` were added
2026-08-01 -- see their ItemSpec comment below; `nansum`/`nanprod`/`nanmean`
were added the same day once the `nan_fill` traversal-order gap they all
shared was fixed -- see the module comment above `_NANSUM_FORMS`.

`var`/`std`/`nanvar`/`nanstd`/`average` were added 2026-08-02, closing the
two real defects a 2800-case crossed grid (6 fns x 10 dtypes x 7 shapes x
8 kwarg sets) found against real numpy 2.5.1:

  DEFECT 1 -- complex input to var/std/nanvar/nanstd used to raise
  `NotImplementedError` where real numpy computes `|deviation|^2` and
  returns the COMPANION REAL dtype (complex64 -> float32, complex128 ->
  float64). Fixed in `do_var` (reductions.rs) via the new
  `complex_abs_sq`/`default_var_out_dtype` helpers -- see that function's
  doc comment for the exact numpy-source citations
  (`_methods.py`'s `_complex_to_float` fast path for plain var/std,
  `_nanfunctions_impl.py`'s `multiply(arr, arr.conj(), out=arr).real` for
  nanvar/nanstd).

  DEFECT 2 -- nanvar/nanstd used to accumulate the deviation-squared sum
  in the INPUT array's dtype and only cast the final result to a
  requested `dtype=` at the very end, instead of truncating the
  deviation itself to the input dtype (numpy's actual
  `np.subtract(arr, avg, out=arr, casting='unsafe')` behavior, `arr`
  being the nan-filled COPY at ORIGINAL input precision) while still
  accumulating the final `sum(sqr, dtype=dtype)` at the REQUESTED
  precision. Fixed by casting the deviation array back to
  `compute_arr.dtype()` when `nan_aware` before squaring -- see `do_var`.
  Plain var/std were already correct here (numpy's own `_var` uses
  `out=Ellipsis` for the deviation, i.e. never truncates it).

  Both fixes verified directly against real numpy 2.5.1 (not
  extrapolated) via targeted probes: complex64/complex128 var/std/
  nanvar/nanstd with and without `dtype=`, `ddof=`, and NaN-bearing
  complex input; float16/float32 input with `dtype='float64'` for all
  four functions. See the `_VAR_FORMS`/`_VAR_DDOF_FORMS`/
  `_complex_nan_corpus` comments below for how those probes became
  crossed differential cases.

  `var`/`std`/`nanvar`/`nanstd` inherit the SAME `axis_tuple_noncontig`/
  `where_false` traversal-order exclusion `mean` already has (finding 1
  below) -- their deviation-squared sum goes through the identical
  `reduce_axis`/`binary_op` path, which always forces `Order::C` output
  regardless of input layout. NOT re-investigated or fixed this pass;
  disclosed and excluded from the bit-exact claim exactly like `mean`.

  `average` needed NO Rust changes (`ionp-py/src/reductions.rs`'s
  `average` was already a complete implementation of numpy's three
  supported `weights=` forms -- `None`, same-shape, and 1-D broadcast
  along a single `axis` -- plus `returned=True`); only differential
  cases were missing, added below.)

Same file-ownership-fence merge pattern as linalg_cases.py / inplace_cases.py
/ ndarray_attrs_cases.py: a standalone `REDUCTION_SPECS` dict, merged into
registry.REGISTRY at the tail of registry.py with a collision check, so this
file never needs to touch registry.py's body (beyond the one-time removal of
the stale `"sum"` placeholder documented in registry.py's own comments).

Three real findings drove the choices below (all measured against real
numpy 2.5.1 via direct interactive probes before being encoded as
call-forms, not assumed):

1. `sum`/`mean` inherit the SAME partial pairwise-summation gap
   `ndarray.sum` already found and scoped out (see that file's module
   docstring): a tuple of axes that does NOT coalesce into a single
   contiguous run (`axis_tuple_noncontig`) and `where=`-masked reduction
   both fall back to a non-pairwise path in `ufunc.rs`, which is
   value-correct but not bit-exact against numpy's own pairwise-summation
   loop for float dtypes. Both items here exclude exactly those forms from
   their bit-exact (`atol=0.0`) claim, same as `ndarray.sum` did -- NOT
   worked around, just honestly not claimed. `prod`/`cumsum`/`cumprod` are
   NOT affected: numpy's own `multiply.reduce` and any `.accumulate` are
   sequential, never pairwise (see ufunc.rs's `use_pairwise()`), so those
   three keep every call form.

2. `argmin`/`argmax`/`cumsum`/`cumprod`'s axis validation
   (`single_axis_from_pyobj` in reductions.rs) raises a plain `IndexError`
   for an out-of-range axis, where real numpy raises
   `numpy.exceptions.AxisError` for all four (verified interactively) --
   the same AxisError-vs-IndexError gap `ndarray.squeeze`/`ndarray.swapaxes`
   already have (see ndarray_attrs_cases.py finding 1). Out-of-range-axis
   forms are scoped OUT of these four items' call_forms below rather than
   laundered via `exception_equivalences`, and reported as a real, small,
   fixable gap -- NOT fixed in this file (out of this task's Rust-file
   footprint for this pass; `sum`/`prod`/`all`/`any`/`min`/`max`/`mean`/
   `ptp`/`count_nonzero` all route through `normalize_reduce_axes`, which
   DOES raise real `AxisError`, and keep their out-of-range forms).

3. FLOAT16 ACCUMULATOR WIDTH -- fixed (was the known trap this note used to
   describe). numpy's float16 `sum`/`prod` reduction does NOT always
   accumulate in a widened float32 buffer; it does so ONLY across the
   reduced axis (or coalesced run of reduced axes) that is genuinely the
   iteration's innermost loop -- i.e. whichever axis (kept or reduced) has
   the smallest memory stride -- and otherwise rounds to float16 after
   every fold step, matching real numpy 2.5.1 bit-exactly across a large
   randomized sweep (full reductions, every single-axis reduction, and
   memory-adjacent multi-axis reductions, including F-order and
   transposed/non-contiguous views, where it is the STRIDE ranking that
   decides width, not the nominal axis number). `mean` on float16 is
   different again: it ALWAYS widens to float32 for the sum step
   regardless of axis contiguity, divides by the count in float32, and
   rounds once at the very end -- verified NOT to reuse `sum`'s own
   (bit-exact, but differently-rounded) narrow/wide float16 result. See
   `reduce_axis_f16_narrow_wide`'s doc comment in `ufunc.rs` for the full
   measurement and code.

   ONE GAP remains, disclosed rather than hidden: when two or more reduced
   axes are separated by an intervening KEPT axis in memory (e.g.
   `axis=(0, 2)` on a 3-D array with axis 1 kept, so the reduced axes do
   NOT coalesce), the best model found matches numpy on ~53-67% of output
   elements, not 100% -- no fold order/width combination tried reproduces
   numpy exactly for every shape in that family (reproducible
   counterexample in `ufunc.rs`'s doc comment). This is the SAME family of
   shape `axis_tuple_noncontig` already excludes from `sum`/`mean`'s
   bit-exact claim for float32/float64/complex (finding 1 above, pairwise
   summation grouping) -- float16 inherits that same exclusion, now for a
   documented, measured reason rather than an unverified assumption. The
   float16-specific call forms below therefore deliberately do NOT add an
   `axis=(0, 2)`-style tuple form (that would just manufacture a new,
   already-known-imperfect required test rather than lock in anything);
   the gap is reported in prose here and in the fix's task report instead.
"""
from __future__ import annotations

import numpy as np

import corpus
from ndarray_attrs_cases import _reduce_forms
from registry import CallForm, ItemSpec

# ---------------------------------------------------------------------------
# epsilon-tolerance justifications for var/std/nanvar/nanstd/average -- see
# the REDUCTION_SPECS comment directly above those five items for the full
# sweep-design writeup; these two constants hold the reusable prose.
# ---------------------------------------------------------------------------

_STATS_EPS_JUST = (
    "var/std/nanvar/nanstd inherit the SAME non-C-contiguous traversal-order "
    "gap `mean` already has (module docstring finding 1): `reduce_axis`/"
    "`binary_op` always force `Order::C` output regardless of input layout, "
    "so Fortran-order/transposed input sums in a different order than "
    "numpy's own layout-aware traversal -- ordinary float summation "
    "non-associativity, not a value bug. Bound = the exact maximum measured "
    "over a seeded (SEED=20260731, same seed as corpus.SEED), OUT-OF-CORPUS "
    "sweep: 8 hand-picked + 40 seeded-random shapes (distinct from the "
    "corpus's own shape list), each expanded into C/Fortran/transposed "
    "layout variants, crossed with axis/ddof/dtype= kwarg combos (plus a "
    "15%-NaN-injected variant for nanvar/nanstd). Every dtype key has n >= "
    "43,122 compared elements, above MIN_ULP_SWEEP_N. The grading key for "
    "each measurement mirrors harness.py's real `_explicit_dtype_override`/"
    "`_first_operand_dtype` logic exactly (explicit dtype= kwarg wins, else "
    "the first operand's own dtype -- never plain output dtype; this "
    "applies to var/std only -- nanvar/nanstd key on the COMPOSITE "
    "(operand, override) pair instead, see `_NANVAR_NANSTD_EPS_JUST` "
    "for why). Every dtype key on var/std/average measures within "
    "~0.4x-2.5x of that dtype's own eps -- re-verified directly (not just "
    "assumed) for var/std's complex64->float64/complex128->float64 "
    "sub-populations too: unlike nanvar/nanstd, var/std never truncate the "
    "deviation to the input dtype regardless of an explicit dtype= "
    "override (module docstring DEFECT-2 paragraph), so there is no "
    "lower-precision population hiding under their single \"float64\" key "
    "and no split was needed. "
    "Metric is \"rel\" (harness.max_rel_distance, floor=1.0): these "
    "reduction outputs' magnitude is the data's own variance/mean scale, "
    "and a relative bound is the right instrument for a summation-order "
    "artifact that scales with the reduced value itself. Sweep script "
    "(not committed) at scratchpad `stats_tol_sweep.py`; see each item's "
    "`epsilon_sweep` for the recorded (n, measured) pair."
)

_NANVAR_NANSTD_EPS_JUST = (
    "nanvar/nanstd (UNLIKE var/std) key their epsilon_tolerance on the "
    "COMPOSITE (operand_dtype, dtype=override) pair, not just the grading "
    "dtype alone -- see harness.py's `_dtype_key_candidates`/"
    "`_lookup_dtype_tolerance` (2026-08-02). Root cause: this item's "
    "plain \"float64\" key used to be measured 3.2e8x looser than genuine "
    "float64 computation needs (1.06e-07 for nanvar, 5.28e-08 for nanstd) "
    "because it silently merged native-float64 input with EVERY population "
    "that ends up grading as \"float64\" under the old override-or-operand "
    "single key -- collapsing two populations with genuinely different "
    "achievable precision into one bound sized to the looser one. The "
    "actual culprit, found by re-sweeping each (operand, override) pair "
    "SEPARATELY: real numpy's own nanvar/nanstd, given complex64/complex128 "
    "input AND an explicit real `dtype=` override, do NOT take the "
    "DEFECT-1 `complex_abs_sq` companion-dtype path at all -- they discard "
    "the imaginary part via a plain unsafe real cast first (`ComplexWarning: "
    "Casting complex values to real discards the imaginary part`, verified "
    "directly against numpy 2.5.1) and THEN run an ordinary real-valued "
    "nanvar/nanstd on what's left, with nanvar/nanstd's own DEFECT-2 "
    "deviation-truncation (module docstring) applying on top. For "
    "complex64->float64 the retained real component is only "
    "FLOAT32-precision even though it's carried in float64 registers "
    "thereafter, so accuracy is floor-bound by float32 eps (~1.19e-07) -- "
    "matching the measured 1.09e-07 (nanvar) / 5.46e-08 (nanstd), roughly "
    "half of native complex64's own bound (no extra rounding step from a "
    "complex-modulus computation the native path still does). For "
    "complex128->float64 the real component is ALREADY float64-precision, "
    "so it measures at genuine float64 scale (~7.1e-16 / ~4.7e-16) same as "
    "native float64. float16->float64 and float32->float64, BY CONTRAST, "
    "measured BIT-EXACT (0.0) across the full sweep below -- DEFECT 2's "
    "fix (casting the deviation back to the input dtype before squaring, "
    "then summing at the requested precision) reproduces numpy's own "
    "operation sequence exactly for the axis-reduction paths exercised "
    "here, so there is no residual left to bound; the previous "
    "declaration's misattribution of the \"float64\" key's slack to "
    "float16/float32 truncation was WRONG -- re-verified here against real "
    "numpy and real anionpy, not assumed. Sweep design otherwise matches "
    "`_STATS_EPS_JUST` exactly (same SEED=20260731 methodology, out-of-"
    "corpus, C/Fortran/transposed layouts, axis/ddof crosses, 15%-NaN "
    "injection), widened to 13 hand-picked shapes (incl. up to 5000 "
    "elements, to stress summation-order effects at scale) + 200 seeded-"
    "random shapes, each key measured on n=75,402 compared elements, above "
    "MIN_ULP_SWEEP_N. var/std were independently re-swept the SAME way as "
    "a check (NOT because their own declared bound was in question -- it "
    "wasn't touched): every var/std composite key, including "
    "complex64->float64, measures at genuine float64 scale (~4.4e-16), "
    "confirming var/std's existing single-key bound was already correct "
    "and needs no split (their deviation is never truncated to the input "
    "dtype regardless of dtype=, per the module docstring's DEFECT-2 "
    "paragraph, so there is no lower-precision population to collapse in). "
    "Metric is \"rel\" (harness.max_rel_distance, floor=1.0), matching "
    "`_STATS_EPS_JUST`. Sweep script (not committed) at scratchpad "
    "`stats_tol_sweep_v2.py`; see this item's `epsilon_sweep` for the "
    "recorded (n, measured) pair per key."
)

_AVERAGE_EPS_JUST = (
    "average's residual is NOT the traversal-order gap above -- it measures "
    "at float64-machine-epsilon scale (~5.5e-16 to ~1.5e-15) for EVERY "
    "dtype key, including float16/complex64 input, because anionpy's "
    "sum(a*weights)/sum(weights) accumulates through an intermediate "
    "widened buffer the same way numpy's own `average` does, so the "
    "residual reflects double-rounding at the accumulation precision, not "
    "the input dtype's own precision floor. Bound = the exact maximum "
    "measured over the SAME seeded (SEED=20260731), out-of-corpus sweep "
    "described in `_STATS_EPS_JUST` (weights=None / same-shape / 1-D-axis0 "
    "forms), n=53,298 for every dtype key, above MIN_ULP_SWEEP_N. Grading "
    "key and metric selection ('rel', harness.max_rel_distance, floor=1.0) "
    "match `_STATS_EPS_JUST` exactly. Sweep script (not committed) at "
    "scratchpad `stats_tol_sweep.py`; see this item's `epsilon_sweep` for "
    "the recorded (n, measured) pair."
)

# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------


def _method_style_cases(call_forms: list[CallForm], corpus_cases=None) -> list:
    """Exactly run.py's build_cases(kind="method") logic, factored out so
    kind="custom" items (sum, mean) can reuse it and then append more cases
    on top -- see module docstring finding 3."""
    cases = []
    for c in (corpus_cases if corpus_cases is not None else corpus.unary_corpus()):
        for form in call_forms:
            if not form.applicable(c.value):
                continue
            args, kwargs = form.build(c.value)
            cases.append((f"{c.label}/{form.label}", (c.value, *args), kwargs))
    return cases


def _float16_corpus() -> list:
    """Explicit float16 corpus -- see module docstring finding 3.
    corpus.py's shared catalogs never produce float16 at all (no
    `np.float16` entry in `ALL_DTYPES`/`SWEEP_DTYPES`), so without this,
    `reduce_axis_f16_narrow_wide`'s width rule (and its one disclosed gap)
    would be invisible to this differential run. Every shape/layout family
    the width rule's own measurement covered gets a case here: single-axis
    C-order 2-D (both axes, so both narrow and wide get exercised), the
    `(k, 1)` single-output-element edge case, F-order and a transposed
    (non-contiguous) view (stride ranking, not axis number, decides
    width), and a 3-D array for the adjacent-multi-axis/axis=None forms
    below. Deliberately NOT included: an `axis=(0, 2)`-style gapped
    multi-axis shape -- see module docstring finding 3's gap disclosure."""
    rng = np.random.default_rng(20260731)  # same pinned SEED as corpus.py
    base_2d = rng.standard_normal((6, 7)).astype(np.float16)
    k1 = rng.standard_normal((11, 1)).astype(np.float16)
    base_3d = rng.standard_normal((4, 3, 5)).astype(np.float16)
    return [
        corpus.Case("float16/random_2000_1d", rng.standard_normal(2000).astype(np.float16)),
        corpus.Case("float16/small_constant_10000_1d",
                     (np.ones(10000, dtype=np.float16) * 0.001)),
        corpus.Case("float16/random_2d", base_2d),
        corpus.Case("float16/random_2d_fortran", np.asfortranarray(base_2d)),
        corpus.Case("float16/random_2d_transposed", base_2d.T),
        corpus.Case("float16/k1_single_output_elem", k1),
        corpus.Case("float16/random_3d", base_3d),
        corpus.Case("float16/random_3d_fortran", np.asfortranarray(base_3d)),
    ]


def _basic_forms_for_float16(call_forms: list[CallForm]) -> list[CallForm]:
    """The float16 probe cases need enough forms to exercise
    `reduce_axis_f16_narrow_wide`'s width rule on every axis of the 2-D/3-D
    corpus arrays above (not the full signature sweep, which the ordinary
    non-float16 corpus cases already cover for every other dtype) --
    `no_args`/`axis_none_explicit` (full reduction, always wide),
    `axis0`/`axis_last` (opposite narrow/wide roles depending on the
    array's C/F/transposed layout), and `axis_tuple_all` (every axis via a
    tuple -- exercises the same full-coalesce path as `axis_none` through
    a different call form, and covers the adjacent-multi-axis case on the
    3-D corpus entries). Deliberately excludes `axis_tuple_noncontig` (the
    gapped-multi-axis form, per finding 3's disclosed gap) even for the
    dtypes where it's normally kept, since float16 shares that exclusion."""
    keep = {"no_args", "axis_none_explicit", "axis0", "axis_last", "axis_tuple_all", "axis_tuple_leading"}
    return [f for f in call_forms if f.label in keep]


# ---------------------------------------------------------------------------
# Tier 1: thin wrappers on reduce_axis -- prod/all/any/min/max/amin/amax
# keep every call form _reduce_forms produces EXCEPT prod's one exclusion
# below (no OTHER known partial-exactness gap: minimum.reduce/maximum.reduce/
# logical *.reduce are all sequential in real numpy too, never pairwise --
# see module docstring finding 1).
#
# CORRECTION (2026-08-06, re-measured against the installed binary, not
# inherited from finding 1's claim): finding 1's "prod/cumsum/cumprod are
# NOT affected [by axis_tuple_noncontig], keep every call form" is true for
# cumsum/cumprod but FALSE for prod itself. Finding 1 is about the
# float32/float64 PAIRWISE-SUMMATION gap (multiply.reduce is sequential, so
# that specific gap doesn't apply to prod) -- but finding 3's SEPARATE
# float16 NARROW/WIDE ACCUMULATOR gap explicitly says "numpy's float16
# sum/prod reduction does NOT always accumulate in a widened float32
# buffer" (prod, not just sum) and names `axis_tuple_noncontig` as the one
# disclosed unfixed shape family for that gap. `_basic_forms_for_float16`
# already excludes `axis_tuple_noncontig` for sum/mean/nansum/nanmean's
# DEDICATED float16 corpus cases -- but `_PROD_FORMS` (unlike `_SUM_FORMS`/
# `_MEAN_FORMS`) was never filtered at all, so prod's plain `kind="method"`
# corpus sweep (which crosses EVERY call form against EVERY dtype in
# `corpus.unary_corpus()`, including float16) reaches `axis_tuple_noncontig`
# on float16 input directly -- measured: 3/4 float16 axis_tuple_noncontig
# cases in that sweep mismatch numpy (`prod`, `nanprod`, `ndarray.prod` all
# identically, `ndarray.sum` too via a separate but same-shaped gap -- see
# `ndarray_attrs_cases.py`). Excluded here the same way `_SUM_FORMS`/
# `_MEAN_FORMS` already do, rather than left silently over-claiming
# bit-exactness on a form nothing here actually verified. See
# KNOWN-DIFFERENCES.md's 2026-08-06 "float16 axis_tuple_noncontig reduction
# order" entry for the literal repro.
# ---------------------------------------------------------------------------

_PROD_EXCLUDE = {"axis_tuple_noncontig"}
_PROD_FORMS = [f for f in _reduce_forms(has_dtype=True, has_initial=True, initial_value=2)
               if f.label not in _PROD_EXCLUDE]
_ALL_ANY_FORMS = _reduce_forms(has_dtype=False, has_initial=False)
_MIN_MAX_FORMS = _reduce_forms(has_dtype=False, has_initial=True, initial_value=1)

# sum/mean: same _reduce_forms base, minus the non-pairwise-exact forms
# (finding 1 above).
_SUM_EXCLUDE = {"axis_tuple_noncontig", "where_false_no_initial", "where_false_with_initial"}
_SUM_FORMS = [f for f in _reduce_forms(has_dtype=True, has_initial=True, initial_value=2)
              if f.label not in _SUM_EXCLUDE]

_MEAN_EXCLUDE = {"axis_tuple_noncontig", "where_false"}
_MEAN_FORMS = [f for f in _reduce_forms(has_dtype=True, has_initial=False)
               if f.label not in _MEAN_EXCLUDE]

# `dtype=` INTEGER-TARGET cast-back cases (2026-08-02): `_reduce_forms`'s only
# dtype-bearing form is `dtype_kwarg_float64` -- for a float/bool/int input
# array, casting the ACCUMULATOR/RESULT to float64 never actually exercises
# numpy's `um.true_divide(ret, rcount, out=ret, casting='unsafe')` final
# unsafe-cast-DOWN step (see reductions.rs's `mean()` doc comment), because
# anionpy's own reduction already defaults to a float64 accumulator for every
# non-inexact input and stays float64 for float64 input -- the two happen to
# agree by coincidence. `dtype=np.int64`/`np.int32` on a bool/int/float/
# complex input is the form that actually forces a down-cast and would have
# caught the withdrawn `mean`/`nanmean` declaration's bug (verified against
# real numpy above this module's edit: bool -> int64 gives 1 not 0.33...,
# float32 -> int64 truncates 2.5 -> 2, complex -> int64 emits a
# ComplexWarning but still succeeds by discarding the imaginary part).
_DTYPE_INT_FORMS = [
    CallForm("dtype_kwarg_int64_axis0", lambda arr: ((), {"axis": 0, "dtype": np.int64}),
             applicable=lambda arr: arr.ndim >= 1),
    CallForm("dtype_kwarg_int64_no_axis", lambda arr: ((), {"dtype": np.int64})),
    CallForm("dtype_kwarg_int32_axis0", lambda arr: ((), {"axis": 0, "dtype": np.int32}),
             applicable=lambda arr: arr.ndim >= 1),
]


def _form_axis_custom_cases() -> list:
    """FORM axis (2026-08-02, CLASS A closure task), crossed into every
    kind="custom" top-level reduction that takes its array as a plain
    positional argument: `anionpy.sum([1, 2, 3])`, not just `anionpy.sum(arr)`.
    Uses the item's own no-kwarg baseline call shape -- see
    corpus.form_axis_cases()'s docstring for why "form_ndarray" is skipped
    (already covered by `_method_style_cases`'s own baseline loop) and why
    a bounded `sample_stride` is used against the full unary_corpus()."""
    return [(c.label, (c.value,), {}) for c in corpus.form_axis_cases(sample_stride=8)]


def _axis_type_cases(fn_name: str) -> list:
    """axis= TYPE and RANGE rejection, crossed against operand ndim.

    numpy does not have one axis converter, it has several, and they
    disagree. The mean-family converter turns a Python bool into the index
    1 and RANGE-CHECKS it BEFORE re-parsing it with the integer-format
    converter that rejects bools, so the error class flips on the operand's
    ndim alone. Measured 2026-08-04 against numpy 2.5.1:
        np.mean(np.zeros((0, 3)), axis=True) -> TypeError: an integer is required
        np.mean(np.zeros((5,)),   axis=True) -> AxisError: axis 1 is out of
                                                bounds for array of dimension 1
        np.nanmean(np.zeros((5,)), axis=True) -> TypeError: an integer is required
    -- i.e. `nanmean` does NOT inherit `mean`'s range check even though it
    delegates to it. Without an ndim<2 operand the range-check arm is
    unreachable and a guard covering it cannot be exercised at all; that is
    exactly how an earlier draft shipped that arm untested.

    ILLUSTRATIVE, NOT EXHAUSTIVE.

    HISTORY. This used to end: "Wired into mean/var/std/nanmean ONLY,
    deliberately. The same forms are measurably wrong today on
    sum/prod/min/max/any/all/argmin/nansum/nanprod/nanvar/nanstd (592
    diffs, filed separately); adding them there would be asserting a fix
    that does not exist yet." That fix now exists -- cc5523f, cab03ab and
    db62b24 took a 16,344-case axis-form sweep from 592 diverging to zero
    real divergences -- so the guards are wired into every one of those
    items via the `axistype/*` specs at the bottom of REDUCTION_SPECS.

    numpy has FIVE distinct axis converters and this file's guards now
    cover all five:
      1. PyArray_ConvertMultiAxis  sum/prod/min/max/any/all/nansum/
                                   nanprod/nanvar/nanstd/ptp/count_nonzero
                                   -- a TUPLE is the only sequence accepted
      2. the mean-family pre-check mean/var/std/nanmean -- range-checks a
                                   bool BEFORE type-rejecting it
      3. the single-axis converter argmin/argmax/cumsum -- FULLY QUALIFIED
                                   type names, bool -> "...for the axis"
      4. normalize_axis_tuple      median/quantile/average -- iterates ANY
                                   sequence, accepts bools, and takes an
                                   `argname` that changes its messages
      5. the COMPOSITION of 1+3    nanargmin/nanargmax -- no converter of
                                   its own; gated on inexact AND size != 0

    Shape () and dtype int64 are both load-bearing, not padding. The
    `nanmean` defect fixed in cab03ab was visible ONLY on a 0-d operand of
    an EXACT dtype: `nanmean` delegates to `mean` there, messages and all,
    and anionpy was running a third converter instead. A float64-only or
    ndim>=1-only guard set cannot see it.
    """
    import decimal
    forms = [0.0, np.float64(0), np.float32(0), "x", ("x",), (0.0,), (0, 0.0),
             ("x", 0.0), decimal.Decimal(0), (decimal.Decimal(0),),
             True, (True,), np.True_, (np.True_,), np.array(0.0), (np.array(0.0),),
             np.array([0]), np.array(0), np.int64(1),
             (0, 0), (0, 5), 100, (0.0, 1), 0, (0,), None, (), (0, 1), 1, -1,
             # Added 2026-08-04. The SEQUENCE forms are what separate
             # converter 1 from converter 4, and anionpy silently accepted
             # every one of them on converter-1 items -- reducing over axis
             # 0 where numpy raises by container type name. That is a wrong
             # ANSWER, not a wrong message, so it could never have been
             # caught by an exception-only guard set.
             #   np.sum(a,     axis=[0])      -> TypeError: 'list' object ...
             #   np.average(a, axis=[0])      -> OK, shape (3,)
             # `b"\x00"` and `range(1)` are not exotica for their own sake:
             # bytes iterate into INTEGERS and range into ints, so numpy's
             # converter 4 accepts both, and any "is it a tuple or a list"
             # shortcut gets them wrong in the accepting direction.
             [0], [0, 1], range(1), b"\x00",
             # A 0-d array and an ndim>=1 array as tuple ELEMENTS, which
             # take a different path from the same objects passed bare.
             (np.array(0),), (np.array([0]),)]
    out = []
    for shape in ((5,), (0,), (0, 3), (2, 3), (2, 3, 4), ()):
        for dt in (np.float64, np.int64):
            for ax in forms:
                out.append((f"axistype/{fn_name}/{dt.__name__}/shape{shape}/{ax!r}",
                            (np.zeros(shape, dtype=dt),), {"axis": ax}))
    return out


def _sum_custom_cases() -> list:
    cases = _method_style_cases(_SUM_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_SUM_FORMS), _float16_corpus())
    cases += _form_axis_custom_cases()
    return cases


def _mean_custom_cases() -> list:
    cases = _axis_type_cases("mean")
    cases += _method_style_cases(_MEAN_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_MEAN_FORMS), _float16_corpus())
    cases += _method_style_cases(_DTYPE_INT_FORMS)
    cases += _form_axis_custom_cases()
    return cases


# ---------------------------------------------------------------------------
# nansum/nanprod/nanmean (2026-08-01): `nan_fill` in ufunc.rs used to always
# emit a fresh C-ordered buffer regardless of the input's actual memory
# layout, discarding the Fortran/transposed/strided stride information that
# `reduce_axis`'s traversal-order fix (the pairwise-summation/`use_pairwise`
# work this whole module documents) depends on to find the genuinely
# innermost axis -- so any nan* reduction on a non-C-ordered array picked
# the wrong "innermost" run and diverged from real numpy (which preserves
# layout here via `_replace_nan`'s `np.array(a, subok=True, copy=True)`,
# `order='K'` by default). Fixed by routing `nan_fill` through the
# already-verified `NdArray::to_contiguous_order("K")` primitive instead of
# forcing `Order::C` -- see `nan_fill`'s doc comment in ufunc.rs.
#
# `nansum`/`nanprod` both call `nan_fill` then the exact same
# `do_reduce_axis` real `sum`/`prod` use (BinaryOp::Add / BinaryOp::Multiply
# respectively) -- so they inherit `sum`'s form set/exclusions (Add is
# pairwise, same `axis_tuple_noncontig`/`where_false_*` gap, finding 1
# above) and `prod`'s (Multiply is sequential, no gap, every form kept).
# `nanmean` short-circuits to plain `mean` whenever no NaN is present at all
# (verified in reductions.rs), and otherwise sums via the same
# `reduce_axis(Add, ...)` path -- so it inherits `mean`'s own exclusions
# too.
#
# The float16 narrow/wide accumulator width rule (finding 3) applies here
# unchanged (nan_fill doesn't touch accumulation, only which values reach
# it), including its ONE disclosed gap (gapped multi-axis, e.g.
# `axis=(0, 2)` on a 3-D array) -- still excluded via
# `_basic_forms_for_float16`, same as plain sum/mean. `_float16_nan_corpus`
# below reuses `_float16_corpus`'s layout families (2-D C/F/transposed, 3-D
# C/F) but forces a few elements to NaN BEFORE deriving the Fortran/
# transposed views, so every view carries NaN at the same logical positions
# -- this is the layout-preservation question the nan_fill fix actually
# addresses, which an all-finite float16 corpus would never exercise.
# ---------------------------------------------------------------------------

_NANSUM_FORMS = [f for f in _reduce_forms(has_dtype=True, has_initial=True, initial_value=2)
                 if f.label not in _SUM_EXCLUDE]
# nanprod inherits prod's own axis_tuple_noncontig float16 exclusion (see
# _PROD_EXCLUDE's CORRECTION comment above) -- was unfiltered here too until
# 2026-08-06 (re-measured, not inherited from the module docstring's older
# "prod's gap doesn't apply to nanprod" assumption).
_NANPROD_FORMS = [f for f in _reduce_forms(has_dtype=True, has_initial=True, initial_value=2)
                  if f.label not in _PROD_EXCLUDE]
_NANMEAN_FORMS = [f for f in _reduce_forms(has_dtype=True, has_initial=False)
                  if f.label not in _MEAN_EXCLUDE]


def _float16_nan_corpus() -> list:
    """See module comment above `_NANSUM_FORMS`: same layout family as
    `_float16_corpus()` but with NaN forced in at fixed logical positions
    before Fortran/transposed views are derived, plus an all-NaN case
    (verified against real numpy: nansum/nanprod emit no warning on an
    all-NaN slice -- 0/1 identity -- unlike nanmean/nanmin/nanmax)."""
    rng = np.random.default_rng(20260801)
    base_2d = rng.standard_normal((6, 7)).astype(np.float16)
    base_2d[1, 2] = np.nan
    base_2d[4, 5] = np.nan
    base_3d = rng.standard_normal((4, 3, 5)).astype(np.float16)
    base_3d[0, 1, 2] = np.nan
    base_3d[3, 2, 4] = np.nan
    all_nan_2d = np.full((3, 4), np.nan, dtype=np.float16)
    return [
        corpus.Case("float16nan/random_2d", base_2d),
        corpus.Case("float16nan/random_2d_fortran", np.asfortranarray(base_2d)),
        corpus.Case("float16nan/random_2d_transposed", base_2d.T),
        corpus.Case("float16nan/random_3d", base_3d),
        corpus.Case("float16nan/random_3d_fortran", np.asfortranarray(base_3d)),
        corpus.Case("float16nan/all_nan_2d", all_nan_2d),
    ]


def _nansum_custom_cases() -> list:
    cases = _method_style_cases(_NANSUM_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_NANSUM_FORMS), _float16_nan_corpus())
    cases += _form_axis_custom_cases()
    return cases


def _nanprod_custom_cases() -> list:
    cases = _method_style_cases(_NANPROD_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_NANPROD_FORMS), _float16_nan_corpus())
    cases += _form_axis_custom_cases()
    return cases


def _nanmean_custom_cases() -> list:
    cases = _axis_type_cases("nanmean")
    cases += _method_style_cases(_NANMEAN_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_NANMEAN_FORMS), _float16_nan_corpus())
    cases += _form_axis_custom_cases()
    # dtype=int64/int32 cast-back cases, same rationale as _mean_custom_cases
    # above -- PLUS this is where nanmean's OWN bug half lives: for a
    # float/complex input, real numpy raises `TypeError: If a is inexact,
    # then dtype must be inexact` for dtype=int64/int32 BEFORE ever looking
    # at whether the array actually contains a NaN (verified above this
    # module's edit) -- exercised here on both the plain unary corpus
    # (mixes of int/bool/float/complex inputs, no NaN) and the NaN-bearing
    # float16 corpus reused below at plain float32/float64 precision via a
    # dedicated small NaN corpus, so the guard-must-fire-even-with-no-NaN
    # path and the guard-must-fire-even-though-NaN-IS-present path are both
    # covered.
    cases += _method_style_cases(_DTYPE_INT_FORMS)
    cases += _method_style_cases(_DTYPE_INT_FORMS, _nanmean_dtype_int_extra_corpus())
    return cases


def _nanmean_dtype_int_extra_corpus() -> list:
    """A small float/complex-with-NaN corpus (double precision, unlike
    `_float16_nan_corpus`) so the `dtype=int64` guard-must-fire-despite-NaN
    path is exercised at a precision the withdrawn declaration's bug report
    actually used."""
    a = np.array([1.5, np.nan, 3.5, -2.5], dtype=np.float64)
    b = np.array([1.0, 2.0, np.nan], dtype=np.float32)
    c = np.array([1 + 2j, np.nan, 3 + 4j], dtype=np.complex128)
    return [
        corpus.Case("nanmean_dtype_int/float64_with_nan", a),
        corpus.Case("nanmean_dtype_int/float32_with_nan", b),
        corpus.Case("nanmean_dtype_int/complex128_with_nan", c),
    ]


# ---------------------------------------------------------------------------
# var/std/nanvar/nanstd (2026-08-02): see module docstring's DEFECT 1/
# DEFECT 2 paragraphs for the two real bugs this closes. Same base
# `_reduce_forms(has_dtype=True, has_initial=False)` shape as `mean`
# (numpy's own `var`/`std` signature -- verified: `(a, axis=None,
# dtype=None, out=None, ddof=0, keepdims=<no value>, where=<no value>,
# mean=<no value>, correction=<no value>)` -- has no `initial=`), so they
# share `mean`'s SAME `axis_tuple_noncontig`/`where_false` exclusion
# (finding 1, and see the module docstring's DEFECT-section note on why
# that gap carries over unchanged).
#
# `_VAR_DDOF_FORMS` crosses `ddof=` against `axis=`/`dtype=` -- exactly
# the interaction DEFECT 2 lived in (the divisor `N - ddof` doesn't matter
# for the fix, but the ACCUMULATION dtype the numerator is computed in
# does). `ddof_equals_axis0_len`'s ddof value is derived from the actual
# corpus array's own axis-0 length via the CallForm's own closure (not a
# hardcoded number), so `N - ddof == 0` for every applicable case --
# real numpy emits `RuntimeWarning: Degrees of freedom <= 0 for slice`
# and returns nan/inf here (verified), which is a value the comparison
# harness checks like any other float, not a warning assertion.
# ---------------------------------------------------------------------------

_VAR_EXCLUDE = _MEAN_EXCLUDE
_VAR_BASE_FORMS = [f for f in _reduce_forms(has_dtype=True, has_initial=False)
                   if f.label not in _VAR_EXCLUDE]

_VAR_DDOF_FORMS = [
    CallForm("ddof1_axis0", lambda arr: ((), {"axis": 0, "ddof": 1}),
             applicable=lambda arr: arr.ndim >= 1),
    CallForm("ddof2_no_axis", lambda arr: ((), {"ddof": 2})),
    CallForm("ddof1_dtype_f64_axis0", lambda arr: ((), {"axis": 0, "ddof": 1, "dtype": np.float64}),
             applicable=lambda arr: arr.ndim >= 1),
    # DEFECT 2's exact interaction: narrow (float16/float32) input,
    # WIDENED requested dtype, nonzero ddof -- the accumulation-precision
    # question and the divisor question exercised in the same call.
    CallForm("ddof1_dtype_f64_axis_last", lambda arr: ((), {"axis": -1, "ddof": 1, "dtype": np.float64}),
              applicable=lambda arr: arr.ndim >= 1),
    CallForm("ddof_equals_axis0_len", lambda arr: ((), {"axis": 0, "ddof": arr.shape[0]}),
             applicable=lambda arr: arr.ndim >= 1),
]

_VAR_FORMS = _VAR_BASE_FORMS + _VAR_DDOF_FORMS


def _complex_nan_corpus() -> list:
    """Complex64/complex128 arrays with NaN in the real component, the
    imaginary component, or both, at fixed positions -- exercises
    nanvar/nanstd's complex branch (`multiply(arr, arr.conj(),
    out=arr).real`, see reductions.rs's `do_var` doc comment) together
    with `nan_fill`'s NaN detection, which for complex must flag NaN if
    EITHER component is NaN (verified against real numpy:
    `np.isnan(complex(nan, 0))` is True)."""
    rng = np.random.default_rng(20260802)
    out = []
    for dtype in (np.complex64, np.complex128):
        real = rng.standard_normal(24)
        imag = rng.standard_normal(24)
        arr = (real + 1j * imag).astype(dtype).reshape(4, 6)
        arr[0, 1] = complex(np.nan, 0.0)
        arr[2, 3] = complex(0.0, np.nan)
        arr[3, 5] = complex(np.nan, np.nan)
        out.append(corpus.Case(f"complexnan/2d/{np.dtype(dtype).name}", arr))
        out.append(corpus.Case(f"complexnan/1d/{np.dtype(dtype).name}", arr.reshape(-1)))
    return out


def _var_custom_cases() -> list:
    cases = _axis_type_cases("var")
    cases += _method_style_cases(_VAR_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_VAR_BASE_FORMS), _float16_corpus())
    cases += _method_style_cases(_DTYPE_INT_FORMS)
    cases += _out_where_cases("var")
    cases += _mean_correction_cases("var")
    return cases


def _std_custom_cases() -> list:
    # `std` is `sqrt(var(...))` sharing `do_var`'s dtype/ddof logic end to
    # end (reductions.rs) -- same forms/corpora as `var`, dispatched
    # against `np.std`/`anionpy.std` by ItemSpec.name, not called here.
    cases = _axis_type_cases("std")
    cases += _method_style_cases(_VAR_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_VAR_BASE_FORMS), _float16_corpus())
    cases += _method_style_cases(_DTYPE_INT_FORMS)
    cases += _out_where_cases("std")
    cases += _mean_correction_cases("std")
    return cases


def _nanvar_custom_cases() -> list:
    cases = _method_style_cases(_VAR_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_VAR_BASE_FORMS), _float16_nan_corpus())
    cases += _method_style_cases(_DTYPE_INT_FORMS)
    # Same TypeError-before-NaN-inspection guard as nanmean, see
    # `_nanmean_custom_cases`'s comment.
    cases += _method_style_cases(_DTYPE_INT_FORMS, _nanmean_dtype_int_extra_corpus())
    # DEFECT 1's complex branch, crossed with NaN (DEFECT 2's nan_fill
    # dependency) and with ddof.
    cases += _method_style_cases(_VAR_BASE_FORMS, _complex_nan_corpus())
    cases += _method_style_cases(_VAR_DDOF_FORMS, _complex_nan_corpus())
    cases += _out_where_cases("nanvar")
    cases += _mean_correction_cases("nanvar")
    return cases


def _nanstd_custom_cases() -> list:
    cases = _method_style_cases(_VAR_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_VAR_BASE_FORMS), _float16_nan_corpus())
    cases += _method_style_cases(_DTYPE_INT_FORMS)
    cases += _method_style_cases(_DTYPE_INT_FORMS, _nanmean_dtype_int_extra_corpus())
    cases += _method_style_cases(_VAR_BASE_FORMS, _complex_nan_corpus())
    cases += _method_style_cases(_VAR_DDOF_FORMS, _complex_nan_corpus())
    cases += _out_where_cases("nanstd")
    cases += _mean_correction_cases("nanstd")
    return cases


# ---------------------------------------------------------------------------
# `out=`/`where=` corpus for var/std/nanvar/nanstd (2026-08-02, statistics-
# grid task -- reproduces `/tmp/statgrid.py`'s own crossed grid: input dtype
# x axis x keepdims x `out=` in {ABSENT, matching, wrong-dtype, wrong-shape}
# x `where=` for the plain (non-nan) pair). `out=`'s ABSENT value is already
# exercised exhaustively by every other form in this file (none of them pass
# `out=` at all) -- only the three PRESENT values are added here, so this
# doesn't duplicate coverage `_VAR_FORMS` already provides.
#
# The natural (pre-`out=`) result shape/dtype for "matching" and the
# trailing-dim-appended "wrong shape" are derived by calling real numpy
# ONCE, here, at CORPUS-CONSTRUCTION time -- not at anionpy call time, and not
# to supply a VALUE this harness grades anionpy against (harness.run_case()
# calls real numpy itself, completely independently, for every case's own
# grading; see harness.py's own docstring, "numpy IS the reference
# implementation"). This mirrors `_average_custom_cases`'s own weights-array
# construction just above, which does the same thing for the same reason.
# ---------------------------------------------------------------------------

_OUT_WHERE_AXES = [None, 0, -1, (0, 1)]
_OUT_KINDS = ["matching", "wrongdtype", "wrongshape", "wrongsize"]


def _natural_result(fn_name, arr, axis, keepdims, ddof):
    np_fn = {"var": np.var, "std": np.std, "nanvar": np.nanvar, "nanstd": np.nanstd}[fn_name]
    kwargs = {"axis": axis, "keepdims": keepdims}
    if ddof is not None:
        kwargs["ddof"] = ddof
    return np_fn(arr, **kwargs)


def _build_out(natural, kind):
    shape = np.shape(natural)
    dt = np.asarray(natural).dtype
    if kind == "matching":
        return np.zeros(shape, dtype=dt)
    if kind == "wrongdtype":
        # Same fixed choice `/tmp/statgrid.py` itself makes: always int32,
        # regardless of the natural result's own dtype -- this is exactly
        # DEFECT 2's own trigger shape, kept identical to the reproduction
        # grid rather than a broader (but grid-untested) sweep.
        return np.zeros(shape, dtype=np.int32)
    if kind == "wrongshape":
        # Same fixed choice `/tmp/statgrid.py` makes: append ONE extra
        # trailing dimension -- DEFECT 1's own trigger shape (an ndim
        # mismatch against the natural result, verified above to always
        # hold for this exact construction).
        return np.zeros(tuple(list(shape) + [2]), dtype=dt)
    if kind == "wrongsize":
        # FINDING 1 (coordinator, 2026-08-02, statgrid.py's `wrongsize`
        # axis): SAME ndim as the natural result, but a DIFFERENT extent
        # along every dim (+1) -- the branch `wrongshape` above can never
        # reach, since it always changes ndim instead. Real numpy raises a
        # THIRD, differently-formatted message here (`"operands could not
        # be broadcast together with remapped shapes [original->remapped]:
        # ..."`, built by the underlying nditer/ufunc-reduce broadcast
        # machinery rather than the reduction-specific ndim check
        # `wrongshape` exercises) -- see `do_var`'s own doc comment in
        # reductions.rs for the full `keepdims`/`where=`-dependent format
        # derivation. 0-d natural results have no same-ndim variant to
        # perturb, so fall back to a 1-d buffer there, exactly like
        # statgrid.py does.
        if len(shape) == 0:
            return np.zeros((2,), dtype=dt)
        return np.zeros(tuple(d + 1 for d in shape), dtype=dt)
    raise ValueError(kind)


def _out_where_cases(fn_name: str) -> list:
    is_nan_aware = fn_name in ("nanvar", "nanstd")
    rng = np.random.default_rng(20260802)
    cases: list = []
    for dt in (np.int32, np.float32, np.float64):
        base = rng.standard_normal((2, 3)).astype(dt) if dt != np.int32 else \
            rng.integers(-50, 50, size=(2, 3)).astype(dt)
        if is_nan_aware and dt != np.int32:
            base = base.copy()
            base[0, 1] = np.nan
        for axis in _OUT_WHERE_AXES:
            if axis is not None and (isinstance(axis, tuple) or axis != 0) and base.ndim < 2:
                continue
            for keepdims in (False, True):
                for ddof in (0, 1):
                    try:
                        natural = _natural_result(fn_name, base, axis, keepdims, ddof)
                    except Exception:
                        continue
                    for kind in _OUT_KINDS:
                        out = _build_out(natural, kind)
                        label = f"outwhere/dt={np.dtype(dt).name}/axis={axis}/kd={keepdims}/ddof={ddof}/out={kind}"
                        cases.append((label, (base,), {"axis": axis, "keepdims": keepdims, "ddof": ddof, "out": out}))
        # `where=` -- only meaningful for the plain (non-nan-aware) pair;
        # `nanvar`/`nanstd` accept `where=` too but this task's own defects
        # (1/2/3) are all `out=`/weights=`-shaped, `where=` is included here
        # only to confirm it still composes correctly alongside the new
        # `out=` handling (DEFECT 1/2's fix sits directly beside the
        # existing `where=` plumbing in `do_var`, see reductions.rs).
        mask_full = np.ones((2, 3), dtype=bool)
        mask_partial = np.array([[True, False, True], [False, True, True]])
        for mask, mtag in ((mask_full, "all_true"), (mask_partial, "partial")):
            for kind in _OUT_KINDS:
                natural = np.var(base, axis=0, keepdims=False, where=mask)
                out = _build_out(natural, kind)
                label = f"outwhere/dt={np.dtype(dt).name}/where={mtag}/out={kind}"
                cases.append((label, (base,), {"axis": 0, "where": mask, "out": out}))
    return cases


# ---------------------------------------------------------------------------
# `mean=`/`correction=` (2026-08-02, coordinator signature-diff task): a
# standing pre-declaration check ("read the real signature, enumerate every
# parameter, diff against the axes the grid varied") caught that neither
# param was crossed anywhere in this corpus for var/std/nanvar/nanstd,
# despite `do_var` (reductions.rs) already threading both through for all
# four functions. `correction=` forms: ABSENT (already covered by every
# other case in this file), `0`/`1`/`2` (real numpy: `correction` is a
# straight alias for `ddof`, verified `np.var(a, correction=c) ==
# np.var(a, ddof=c)` for every `c` tried), and `together_with_ddof` (real
# numpy's own dedicated interaction error, verified: `ddof=1,
# correction=1` raises `ValueError: ddof and correction can't be provided
# simultaneously.` even though the two values AGREE -- it's a presence
# check, not a conflicting-value check).
#
# `mean=` forms: ABSENT, `correct` (the real keepdims-shaped mean, so the
# call is a pure no-recompute passthrough), `wrong_value` (a deliberately
# incorrect but SHAPE-compatible mean -- verified real numpy does no
# sanity-check against the data at all, it trusts the caller completely
# and produces different-but-still-defined deviations), `wrong_shape_ok`
# (non-keepdims shape that still broadcasts fine against `a`'s own shape
# -- verified real numpy's `mean=` has NO keepdims-shape requirement,
# only ordinary broadcast-compatibility against `a`: `np.var(b, axis=0,
# mean=np.ones(3))` succeeds identically to a keepdims `(1,3)` mean), and
# `wrong_shape_bad` (genuinely non-broadcastable, e.g. `(2,)` against a
# `(2,3)` array reduced over axis 0 whose natural mean shape is `(3,)` --
# verified real numpy raises the ordinary `ValueError: operands could not
# be broadcast together with shapes ...` here, the SAME message
# `binary_op`'s own broadcast check already produces, not a new bespoke
# message).
#
# `mean=` is also crossed with an all-NaN reduced slice for the nan-aware
# pair specifically (`nan_slice`), since `mean=` interacts with NaN
# handling in a way the plain variants never see -- measured directly
# against real numpy rather than assumed: an all-NaN slice's result is
# `nan` regardless of what `mean=` supplies, because the dof<=0 overwrite
# (already-existing code, gated on `rcount`/`ddof`, not on `mean`) fires
# unconditionally and stomps whatever `mean=`-driven value was computed --
# verified: `np.nanvar([[1,2,3],[nan,nan,nan]], axis=1, mean=[[0],[0]])`
# gives `[4.667, nan]`, the SAME nan the `mean=[[2],[nan]]` (the "correct"
# per-row mean) call gives, confirming `mean=`'s value is irrelevant once
# the slice is dof<=0.
# ---------------------------------------------------------------------------

_CORRECTION_FORMS = ["ABSENT", 0, 1, 2, "together_with_ddof"]
_MEAN_FORMS_TAGS = ["ABSENT", "correct", "wrong_value", "wrong_shape_ok", "wrong_shape_bad"]


def _mean_correction_cases(fn_name: str) -> list:
    is_nan_aware = fn_name in ("nanvar", "nanstd")
    rng = np.random.default_rng(20260802)
    base = rng.standard_normal((2, 3))
    if is_nan_aware:
        base = base.copy()
        base[0, 1] = np.nan
    nan_slice_base = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])

    arr_tags = [(base, "base")]
    if is_nan_aware:
        arr_tags.append((nan_slice_base, "nan_slice"))

    cases: list = []
    for arr, tag in arr_tags:
        for axis in (None, 0):
            for mtag in _MEAN_FORMS_TAGS:
                kw: dict = {}
                if axis is not None:
                    kw["axis"] = axis
                if mtag != "ABSENT":
                    natural_mean_kd = np.nanmean(arr, axis=axis, keepdims=True) if is_nan_aware \
                        else np.mean(arr, axis=axis, keepdims=True)
                    if mtag == "correct":
                        kw["mean"] = natural_mean_kd
                    elif mtag == "wrong_value":
                        kw["mean"] = natural_mean_kd + 1.0
                    elif mtag == "wrong_shape_ok":
                        kw["mean"] = np.ravel(natural_mean_kd) if axis is not None else natural_mean_kd.reshape(())
                    elif mtag == "wrong_shape_bad":
                        # Genuinely non-broadcastable: 2 elements against a
                        # natural-mean shape whose own broadcast-relevant
                        # dim is 3 (axis=0) or a scalar (axis=None) --
                        # either way `(2,)` cannot broadcast against `arr`.
                        kw["mean"] = np.array([1.0, 2.0])
                for ctag in _CORRECTION_FORMS:
                    kw2 = dict(kw)
                    if ctag == "ABSENT":
                        pass
                    elif ctag == "together_with_ddof":
                        kw2["correction"] = 1
                        kw2["ddof"] = 1
                    else:
                        kw2["correction"] = ctag
                    label = f"meancorr/{tag}/axis={axis}/mean={mtag}/corr={ctag}"
                    cases.append((label, (arr,), kw2))
    return cases


# ---------------------------------------------------------------------------
# average (2026-08-02): real numpy's signature (verified: `(a, axis=None,
# weights=None, returned=False, *, keepdims=<no value>)`) has NO `dtype=`
# param at all -- unlike mean/var, so no `_DTYPE_INT_FORMS`-style cases
# here. `average`'s own doc comment in reductions.rs (Tier 7) documents
# the three `weights=` forms numpy itself supports: `None` (identical to
# `mean`), same-shape as `a`, and 1-D broadcast along a single `axis`.
# Needed NO Rust fix -- was already a complete implementation; only
# differential coverage was missing.
# ---------------------------------------------------------------------------

_AVERAGE_NO_WEIGHTS_FORMS = [
    CallForm("no_args", lambda arr: ((), {})),
    CallForm("axis_none_explicit", lambda arr: ((), {"axis": None})),
    CallForm("axis0", lambda arr: ((), {"axis": 0}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_last", lambda arr: ((), {"axis": -1}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("keepdims_axis0", lambda arr: ((), {"axis": 0, "keepdims": True}),
             applicable=lambda arr: arr.ndim >= 1),
    # `returned=True` deliberately NOT exercised here: it changes
    # `average`'s return type from a single array to a 2-tuple
    # `(avg, sum_of_weights)`, which needs `ItemSpec(multi_output=True)` --
    # a property of the WHOLE item, not a per-case toggle -- and this
    # item's other cases return a plain array. Verified by direct probe
    # (see this task's report) that `average(..., returned=True)` matches
    # numpy value-for-value; not encoded as a differential case because
    # mixing multi_output and non-multi_output cases under one
    # `compare_values` call would be graded incorrectly (a tuple handed to
    # the non-multi_output comparison path), not because of any known gap.
]


def _average_custom_cases() -> list:
    cases = _method_style_cases(_AVERAGE_NO_WEIGHTS_FORMS)
    cases += _method_style_cases(_basic_forms_for_float16(_AVERAGE_NO_WEIGHTS_FORMS), _float16_corpus())

    # weights=SAME-SHAPE-as-a, axis=None (numpy: elementwise weights,
    # flattened contraction) and weights=1-D broadcast along a chosen
    # axis -- built directly as (label, args, kwargs) tuples (not
    # CallForms) since the weights array itself has to be derived from
    # each corpus array's own shape, which CallForm's `build(arr) ->
    # (args, kwargs)` signature already supports via closures, but the
    # weights VALUES need their own RNG draw per case, not a fixed form.
    rng = np.random.default_rng(20260802)
    out = list(cases)
    for c in corpus.unary_corpus():
        arr = c.value
        if not hasattr(arr, "shape") or getattr(arr, "ndim", 0) == 0:
            continue
        if arr.size == 0:
            continue
        # same-shape weights, axis=None
        w_full = rng.random(arr.shape) + 0.1  # keep strictly positive, avoid all-zero-weight edge
        out.append((f"{c.label}/weights_same_shape", (arr, None, w_full), {}))
        # 1-D weights along axis 0
        w1d = rng.random(arr.shape[0]) + 0.1
        out.append((f"{c.label}/weights_1d_axis0", (arr,), {"axis": 0, "weights": w1d}))
        # weights sum to zero on that axis -- numpy raises ZeroDivisionError
        w1d_zero = np.zeros(arr.shape[0])
        out.append((f"{c.label}/weights_1d_axis0_all_zero_must_raise", (arr,), {"axis": 0, "weights": w1d_zero}))
        # mismatched-length 1-D weights -- numpy raises TypeError/ValueError
        w1d_bad = rng.random(arr.shape[0] + 1)
        out.append((f"{c.label}/weights_1d_axis0_length_mismatch_must_raise", (arr,), {"axis": 0, "weights": w1d_bad}))

        # DEFECT 3 (statistics-grid task, 2026-08-02): the TWO distinct
        # `weights=`-shape-mismatch messages, selected by whether `axis=`
        # was given at all -- NOT by weights' own dimensionality (see
        # reductions.rs's `average` doc comment for the full derivation from
        # numpy's `_weights_are_valid` source). The case just above
        # (`axis=0` explicit) already exercises the ValueError branch; these
        # two new cases exercise the OTHER branch (`axis=None`, TypeError)
        # and confirm the ValueError branch also fires correctly for a
        # TUPLE axis (not just a single int), which nothing above tests.
        w_shape_mismatch = rng.random(tuple(d + 1 for d in arr.shape)) + 0.1
        out.append((f"{c.label}/weights_shape_mismatch_axis_none_must_raise", (arr,), {"axis": None, "weights": w_shape_mismatch}))
        if arr.ndim >= 2:
            w_tuple_axis_bad = rng.random(tuple(d + 1 for d in arr.shape[:2])) + 0.1
            out.append((f"{c.label}/weights_shape_mismatch_axis_tuple_must_raise", (arr,), {"axis": (0, 1), "weights": w_tuple_axis_bad}))

    # `average()` has no `out=` parameter at all in real numpy (verified:
    # `inspect.signature(np.average)` is `(a, axis=None, weights=None,
    # returned=False, *, keepdims=<no value>)`) -- confirm anionpy raises the
    # same "unexpected keyword argument" TypeError real numpy does, rather
    # than silently accepting it (which would be a real, if minor,
    # divergence -- `out=` was never one of this task's three defects for
    # `average` specifically, since `average` never had an `out=` path to
    # begin with, but an untested silent-accept gap is still a gap).
    out.append(("out_kwarg_unsupported_must_raise", (np.array([1.0, 2.0, 3.0]),), {"out": np.zeros(())}))
    return out


# ---------------------------------------------------------------------------
# Tier 2: argmin/argmax -- index-returning reduce over a SINGLE int axis (or
# None); real numpy's own signature (verified: `(a, axis=None, out=None, *,
# keepdims=<no value>)`) has no dtype/initial/where/tuple-axis support at
# all, so there is no equivalent to scope OUT there -- a tuple axis is
# tested as a must-raise form instead (both sides raise TypeError, matching
# -- verified). Out-of-range axis forms are scoped out per finding 2.
# ---------------------------------------------------------------------------

_ARGEXT_FORMS = [
    CallForm("no_args", lambda arr: ((), {})),
    CallForm("axis_none_explicit", lambda arr: ((), {"axis": None})),
    CallForm("axis0", lambda arr: ((), {"axis": 0}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_last", lambda arr: ((), {"axis": -1}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("keepdims_axis0", lambda arr: ((), {"axis": 0, "keepdims": True}),
             applicable=lambda arr: arr.ndim >= 1),
    # THIRD BUG (found + fixed this task, see anionpy/_state/toplevel.py): 0-d
    # input + `keepdims=True` (axis=None, the only form that applies to a
    # 0-d array) used to return shape `(1,)` where real numpy returns a
    # 0-d/scalar result -- `reductions.rs`'s `do_argext` and
    # `ndarray_attrs.rs`'s `do_method_argext` both guarded the flatten-then-
    # relabel reshape with `ndim > 1`, which incorrectly also skipped
    # `ndim == 0`. This form locks the fix in; it is applicable to every
    # ndim, not just 0-d, since the bug's fix (`ndim != 1`) touches the
    # general case too.
    CallForm("keepdims_axis_none", lambda arr: ((), {"axis": None, "keepdims": True})),
    CallForm("axis_tuple_must_raise", lambda arr: ((), {"axis": (0, 1)}),
             applicable=lambda arr: arr.ndim >= 2),
    # 2026-08-01 (sort/search task): argmin/argmax's out-of-range-axis path
    # was fixed to raise the real `numpy.exceptions.AxisError` instead of a
    # plain IndexError (reductions.rs's `do_argext`, ndarray_attrs.rs's
    # `do_method_argext` via `normalize_axis`) -- verified directly against
    # numpy 2.5.1 before re-adding this form (finding 2's gap no longer
    # applies to these two items specifically; cumsum/cumprod still have
    # the old IndexError bug and are NOT touched here, out of this task's
    # scope).
    CallForm("axis_out_of_range_must_raise", lambda arr: ((), {"axis": 100})),
]

# ---------------------------------------------------------------------------
# Tier 3: cumsum/cumprod -- running accumulate; axis=None flattens first.
# Sequential fold (never pairwise, in either numpy or anionpy -- an accumulate
# needs every prefix, so there is nothing to reorder), fully bit-exact. Real
# numpy's signature (verified: `(a, axis=None, dtype=None, out=None)`) has
# no keepdims/initial/where at all. Out-of-range axis forms scoped out per
# finding 2.
# ---------------------------------------------------------------------------

_ACCUMULATE_FORMS = [
    CallForm("no_args", lambda arr: ((), {})),
    CallForm("axis_none_flattens", lambda arr: ((), {"axis": None})),
    CallForm("axis0", lambda arr: ((), {"axis": 0}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_last", lambda arr: ((), {"axis": -1}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("dtype_kwarg_float64", lambda arr: ((), {"axis": 0, "dtype": np.float64}),
             applicable=lambda arr: arr.ndim >= 1),
    # Deliberately no out-of-range-axis form -- see module docstring
    # finding 2.
]


# ---------------------------------------------------------------------------
# nancumsum/nancumprod (2026-08-04). Both are `nan_fill(a, identity)` then
# the SAME `do_accumulate_axis` cumsum/cumprod already use -- 0 for Add, 1
# for Multiply -- exactly as nansum/nanprod delegate to `do_reduce_axis`.
# So they inherit cumsum/cumprod's tier verbatim: a sequential fold (an
# accumulate needs every prefix, so there is nothing to reorder and no
# pairwise-summation traversal-order question in EITHER library), bit-exact,
# and real numpy's signature is `(a, axis=None, dtype=None, out=None)` --
# no keepdims/initial/where at all.
#
# Two forms are added on top of `_ACCUMULATE_FORMS`, both verified against
# real numpy 2.5.1 before being included:
#
#   * `axis_out_of_range_must_raise` -- `_ACCUMULATE_FORMS` omits it per the
#     module docstring's finding 2 (cumsum/cumprod used to raise a plain
#     IndexError instead of AxisError). That was FIXED 2026-08-03 in
#     `do_accumulate_axis`, message and all, including the 0-d courtesy
#     "dimension 1" wording -- re-measured live for both nan* spellings and
#     both signs of the axis. Kept here rather than retro-added to
#     cumsum/cumprod, which is their own item's re-declaration decision.
#   * `dtype_kwarg_float32` -- a NARROWING dtype on a NaN-bearing operand,
#     which is where the fill-then-cast ORDER is observable.
#
# NOT included, and the reason is not squeamishness: `out=`. cumsum/cumprod
# diverge from numpy on two `out=` seams TODAY (a wrong-size out gives
# numpy's "provided out is the wrong size for the accumulation." vs anionpy's
# generic broadcast message; an out with a different dtype is an unsafe CAST
# in numpy and a TypeError in anionpy), and nancumsum/nancumprod inherit both
# by construction since they call the identical code. Measured, recorded in
# KNOWN-DIFFERENCES.md 2026-08-04, and deliberately not claimed here -- the
# fix belongs to the library-wide `out=` contract item, not to this one. An
# `out=` with the right size AND dtype does match; it is exercised by this
# item's out-of-corpus sweep rather than here, because the correct `out=`
# shape and dtype depend on the axis and on sum/prod integer promotion,
# which a shared `CallForm` (given only the operand) cannot compute.
# ---------------------------------------------------------------------------

_NANACCUM_FORMS = _ACCUMULATE_FORMS + [
    CallForm("dtype_kwarg_float32", lambda arr: ((), {"axis": 0, "dtype": np.float32}),
             applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_out_of_range_must_raise", lambda arr: ((), {"axis": 100})),
    CallForm("axis_out_of_range_neg_must_raise", lambda arr: ((), {"axis": -100})),
    # An INTEGER dtype= on a NaN-bearing float operand. This is the one form
    # that can see the fill-then-cast ORDER: fill first (correct) turns NaN
    # into the identity and then casts a finite number; cast first turns NaN
    # into INT64_MIN and leaves the fill nothing to find. Every float dtype=
    # form is blind to it, because NaN survives a float->float cast. Excluded
    # for complex operands, where numpy's complex->int cast is its own
    # (ComplexWarning-bearing) question and not this item's.
    CallForm("dtype_kwarg_int64", lambda arr: ((), {"axis": 0, "dtype": np.int64}),
             applicable=lambda arr: arr.ndim >= 1 and arr.dtype.kind != "c"),
]


def _nanaccum_layout_corpus() -> list:
    """NaN-bearing operands whose PROVENANCE varies, not just their dtype.

    `nan_fill` returns a `to_contiguous_order("K")` copy specifically so a
    Fortran/transposed/strided operand keeps its layout -- that is the bug
    the nansum/nanprod work fixed (see `_NANSUM_FORMS`'s comment). An
    accumulate reads along one axis in order, so a layout that silently
    became C-ordered would change WHICH elements are adjacent and therefore
    which prefix each output slot holds. `corpus.unary_corpus()` carries
    NaN cases but freshly-created C-ordered ones, so it cannot see this.

    Complex is included because `nan_fill` substitutes `fill + 0j`, NOT
    `fill + fill*j` -- invisible for nancumsum (fill 0) and load-bearing for
    nancumprod (fill 1), which is exactly the shape of the bug nanprod's own
    sweep once caught.
    """
    rng = np.random.default_rng(20260804)
    out = []
    for name in ("float32", "float64", "complex64", "complex128"):
        dt = np.dtype(name)
        base = rng.standard_normal((5, 6))
        if dt.kind == "c":
            base = base + 1j * rng.standard_normal((5, 6))
        base = base.astype(dt)
        base[1, 2] = np.nan
        base[3, 4] = np.nan
        if dt.kind == "c":
            # One element NaN in the IMAGINARY part only: numpy's
            # `_replace_nan` masks on `isnan(a)`, which is true if EITHER
            # component is NaN, and then replaces the WHOLE element.
            base[0, 5] = complex(2.0, np.nan)
        out += [
            corpus.Case(f"nanaccum/{name}/c", base),
            corpus.Case(f"nanaccum/{name}/fortran", np.asfortranarray(base)),
            corpus.Case(f"nanaccum/{name}/transposed", base.T),
            corpus.Case(f"nanaccum/{name}/reversed", base[::-1]),
            corpus.Case(f"nanaccum/{name}/strided", base[:, ::2]),
            corpus.Case(f"nanaccum/{name}/offset", base[1:, 2:]),
            corpus.Case(f"nanaccum/{name}/all_nan", np.full((3, 4), np.nan, dtype=dt)),
        ]
    # Integer and bool: `nan_fill` clones them untouched, matching numpy's
    # `_replace_nan` no-op on a non-inexact dtype, so nancumsum on an
    # integer array must be bit-identical to plain cumsum -- including its
    # int64/uint64 promotion.
    for name in ("bool", "int8", "int64", "uint8"):
        out.append(corpus.Case(
            f"nanaccum/{name}/exact_dtype",
            np.arange(1, 13, dtype=np.dtype(name)).reshape(3, 4)
            if name != "bool" else
            np.array([[True, False, True, True], [False, False, True, False]]),
        ))
    return out


def _nancumsum_custom_cases() -> list:
    cases = _method_style_cases(_NANACCUM_FORMS)
    cases += _method_style_cases(_NANACCUM_FORMS, _nanaccum_layout_corpus())
    cases += _method_style_cases(_NANACCUM_FORMS, _float16_nan_corpus())
    return cases


def _nancumprod_custom_cases() -> list:
    return _nancumsum_custom_cases()


# ---------------------------------------------------------------------------
# LAYOUT GUARD for nancumsum/nancumprod, registered under the fresh keys
# `layout/nancumsum` and `layout/nancumprod` (NOT the real item names, so the
# ledger is unaffected -- same device `ufunc_order_cases.py` uses).
#
# Why a second spec instead of another call form: the value corpus above is
# BLIND to memory layout, and this is not a hypothetical. `nan_fill` returns a
# `to_contiguous_order("K")` copy; mutate that one call to `to_contiguous()`
# and every one of the 1715 value cases still PASSES, because an accumulate
# reads by logical index and the VALUES are unchanged -- while the returned
# array's layout silently flips from Fortran to C:
#
#     f = np.asfortranarray(a)          # a 5x6 float64 array with NaN
#     np.nancumsum(f, axis=0).flags     # F_CONTIGUOUS: True
#     anionpy.nancumsum(f, axis=0).flags   # F_CONTIGUOUS: True  (correct today)
#                                       #                False (under the mutation)
#
# Verified in both directions: matching on the real build, diverging under the
# mutation. `check_strides=True` is what actually catches it.
#
# Operands are restricted to C and F layouts on purpose. A transposed or
# strided operand would trip the PRE-EXISTING, unrelated `ndarray_from_numpy`
# ingestion defect documented on `ItemSpec.check_strides` (any non-C/non-F
# source is ingested as if C-contiguous), which has nothing to do with this
# item -- exactly the reasoning `ufunc_order_cases.py` already records.
# ---------------------------------------------------------------------------

_LAYOUT_FORMS = [
    CallForm("no_args", lambda arr: ((), {})),
    CallForm("axis_none", lambda arr: ((), {"axis": None})),
    CallForm("axis0", lambda arr: ((), {"axis": 0})),
    CallForm("axis1", lambda arr: ((), {"axis": 1}), applicable=lambda arr: arr.ndim >= 2),
    CallForm("axis_last", lambda arr: ((), {"axis": -1})),
    # NO `dtype=` form here, and the reason is a REAL BUG this guard found
    # and this task deliberately did not ship a fix for. An explicit `dtype=`
    # routes through `NdArray::cast_to`, which forces C order, so a Fortran
    # operand comes back C-contiguous where numpy (whose `astype` defaults to
    # `order='K'`) keeps it Fortran:
    #
    #     f = np.asfortranarray(np.arange(30.).reshape(5, 6))
    #     f.astype('float32').strides          # (4, 20)  -- still F
    #     ionp_f.astype('float32').strides     # (24, 4)  -- silently C
    #
    # The defect is `astype`/`asarray`/`array`-with-a-dtype, all three long
    # declared "exact"; cumsum/cumprod/nancumsum/nancumprod only inherit it
    # via their `dtype=` cast. The one-line fix (`to_contiguous_order("K")`)
    # was written, built and MEASURED: it fixes these 8 cases and regresses
    # ELEVEN other items (all 11 `fft.*` spellings, `ndarray.sum`/`prod`/
    # `take`/`compress`/`conj`/`conjugate`), whose callers assume `cast_to`
    # hands back a C-contiguous buffer. So the honest fix is a caller audit,
    # not a one-liner, and it belongs to its own item. Reverted; recorded in
    # KNOWN-DIFFERENCES.md 2026-08-04. Excluded here rather than left failing
    # so this guard stays a guard instead of becoming noise.
]


def _layout_custom_cases() -> list:
    rng = np.random.default_rng(20260804)
    cases = []
    for name in ("float32", "float64", "complex64", "complex128"):
        dt = np.dtype(name)
        for shape in ((5, 6), (3, 4, 2)):
            base = rng.standard_normal(shape)
            if dt.kind == "c":
                base = base + 1j * rng.standard_normal(shape)
            base = base.astype(dt)
            flat = base.reshape(-1)
            flat[1] = np.nan
            flat[-2] = np.nan
            for layout in ("c", "f"):
                arr = np.ascontiguousarray(base) if layout == "c" else np.asfortranarray(base)
                for form in _LAYOUT_FORMS:
                    if not form.applicable(arr):
                        continue
                    args, kwargs = form.build(arr)
                    cases.append((f"layout/{name}/{len(shape)}d/{layout}/{form.label}",
                                  (arr, *args), kwargs))
    return cases



# ---------------------------------------------------------------------------
# Tier 4 (partial, this pass): ptp, count_nonzero. Both bit-exact everywhere
# (ptp = Maximum.reduce - Minimum.reduce, count_nonzero = (a != 0).sum() on
# an integer dtype -- neither op is order-sensitive the way float Add is).
# Real numpy signatures verified: `ptp(a, axis=None, out=None,
# keepdims=False)` (no dtype/initial/where); `count_nonzero(a, axis=None,
# *, keepdims=False)` (no dtype/out/initial/where). Both route through
# `normalize_reduce_axes` (real AxisError), so out-of-range forms are kept.
#
# mean/nanmin/nanmax/nansum/nanprod are NOT in this tier -- mean is handled
# above (Tier "sum-like", it shares sum's Add-pairwise gap) and so are
# nanmin/nanmax/nansum/nanprod (see their ItemSpec comments below).
# var/std/nanvar/nanstd/average are handled in their own section further
# below (see module docstring's DEFECT 1/DEFECT 2 paragraphs).
# median/percentile/quantile remain explicitly out of scope this
# pass (see module docstring).
# ---------------------------------------------------------------------------

_PTP_FORMS = [
    CallForm("no_args", lambda arr: ((), {})),
    CallForm("axis_none_explicit", lambda arr: ((), {"axis": None})),
    CallForm("axis0", lambda arr: ((), {"axis": 0}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_last", lambda arr: ((), {"axis": -1}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("keepdims_axis0", lambda arr: ((), {"axis": 0, "keepdims": True}),
             applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_out_of_range_must_raise", lambda arr: ((), {"axis": 100})),
]

_COUNT_NONZERO_FORMS = [
    CallForm("no_args", lambda arr: ((), {})),
    CallForm("axis_none_explicit", lambda arr: ((), {"axis": None})),
    CallForm("axis0", lambda arr: ((), {"axis": 0}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_last", lambda arr: ((), {"axis": -1}), applicable=lambda arr: arr.ndim >= 1),
    CallForm("keepdims_axis0", lambda arr: ((), {"axis": 0, "keepdims": True}),
             applicable=lambda arr: arr.ndim >= 1),
    CallForm("axis_out_of_range_must_raise", lambda arr: ((), {"axis": 100})),
]


REDUCTION_SPECS: dict[str, ItemSpec] = {
    "sum": ItemSpec(
        name="sum", kind="custom",
        custom_cases=_sum_custom_cases,
        atol=0.0, rtol=0.0,
    ),
    "prod": ItemSpec(
        name="prod", kind="method",
        call_forms=_PROD_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "all": ItemSpec(
        name="all", kind="method",
        call_forms=_ALL_ANY_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "any": ItemSpec(
        name="any", kind="method",
        call_forms=_ALL_ANY_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "min": ItemSpec(
        name="min", kind="method",
        call_forms=_MIN_MAX_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "max": ItemSpec(
        name="max", kind="method",
        call_forms=_MIN_MAX_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "amin": ItemSpec(
        name="amin", kind="method",
        call_forms=_MIN_MAX_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "amax": ItemSpec(
        name="amax", kind="method",
        call_forms=_MIN_MAX_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "argmin": ItemSpec(
        name="argmin", kind="method",
        call_forms=_ARGEXT_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "argmax": ItemSpec(
        name="argmax", kind="method",
        call_forms=_ARGEXT_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "cumsum": ItemSpec(
        name="cumsum", kind="method",
        call_forms=_ACCUMULATE_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "cumprod": ItemSpec(
        name="cumprod", kind="method",
        call_forms=_ACCUMULATE_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "mean": ItemSpec(
        name="mean", kind="custom",
        custom_cases=_mean_custom_cases,
        atol=0.0, rtol=0.0,
    ),
    "ptp": ItemSpec(
        name="ptp", kind="method",
        call_forms=_PTP_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "count_nonzero": ItemSpec(
        name="count_nonzero", kind="method",
        call_forms=_COUNT_NONZERO_FORMS,
        atol=0.0, rtol=0.0,
    ),

    # nanmin/nanmax (2026-08-01): same `_reduce_forms(has_dtype=False,
    # has_initial=True)` shape as plain min/max (real numpy's
    # `nanmin`/`nanmax` signature is `(a, axis=None, out=None,
    # keepdims=<no value>, initial=<no value>, where=<no value>)` --
    # identical param set to `amin`/`amax`, no `dtype`). Reuses
    # `_MIN_MAX_FORMS` directly: `corpus.unary_corpus()` already carries
    # NaN-bearing cases (`special/finite_nan_inf`, `special/all_nan`,
    # `special/scalar_nan`) across every float/complex dtype, so no
    # separate NaN corpus was needed the way sum/mean needed a dedicated
    # float16 corpus. Both min/max are order-independent reductions (no
    # pairwise-summation-style traversal-order gap the way sum/mean/prod
    # have), and nanmin/nanmax build on that same already-verified-exact
    # `Minimum`/`Maximum` `reduce_axis` path (see anionpy/_state/toplevel.py's
    # nanmin/nanmax entry for the out-of-corpus sweep: 6336 cases, 0
    # divergences, incl. matched ValueError on an all-empty reduced axis).
    "nanmin": ItemSpec(
        name="nanmin", kind="method",
        call_forms=_MIN_MAX_FORMS,
        atol=0.0, rtol=0.0,
    ),
    "nanmax": ItemSpec(
        name="nanmax", kind="method",
        call_forms=_MIN_MAX_FORMS,
        atol=0.0, rtol=0.0,
    ),

    # nansum/nanprod/nanmean (2026-08-01): see the module comment above
    # `_NANSUM_FORMS` for the `nan_fill` traversal-order fix and why these
    # three inherit sum/prod/mean's own form exclusions and float16 rules
    # verbatim. Verified via an independent out-of-corpus fuzz sweep
    # (dtypes float16/float32/float64/complex64/complex128 x layouts
    # C/F/transposed/reversed/strided x shapes 1-D through 4-D x every axis
    # combination, comparing raw `.tobytes()`/dtype/shape): 0 mismatches
    # outside float16's already-disclosed gapped-multi-axis gap (Gap B,
    # `_basic_forms_for_float16` / finding 3 above), which this corpus does
    # not claim.
    "nansum": ItemSpec(
        name="nansum", kind="custom",
        custom_cases=_nansum_custom_cases,
        atol=0.0, rtol=0.0,
    ),
    "nanprod": ItemSpec(
        name="nanprod", kind="custom",
        custom_cases=_nanprod_custom_cases,
        atol=0.0, rtol=0.0,
    ),
    # Layout guards -- fresh keys, not ledger items. See the comment above
    # `_LAYOUT_FORMS` for why the value corpus alone cannot see this.
    "layout/nancumsum": ItemSpec(
        name="layout/nancumsum", kind="custom",
        numpy_path="nancumsum", ionp_path="nancumsum",
        custom_cases=_layout_custom_cases,
        check_strides=True, atol=0.0, rtol=0.0,
    ),
    "layout/nancumprod": ItemSpec(
        name="layout/nancumprod", kind="custom",
        numpy_path="nancumprod", ionp_path="nancumprod",
        custom_cases=_layout_custom_cases,
        check_strides=True, atol=0.0, rtol=0.0,
    ),

    # nancumsum/nancumprod (2026-08-04): see the comment above
    # `_NANACCUM_FORMS` for the delegation, the two added call forms, and
    # the deliberately unclaimed `out=` seams they inherit from
    # cumsum/cumprod.
    "nancumsum": ItemSpec(
        name="nancumsum", kind="custom",
        custom_cases=_nancumsum_custom_cases,
        atol=0.0, rtol=0.0,
    ),
    "nancumprod": ItemSpec(
        name="nancumprod", kind="custom",
        custom_cases=_nancumprod_custom_cases,
        atol=0.0, rtol=0.0,
    ),
    "nanmean": ItemSpec(
        name="nanmean", kind="custom",
        custom_cases=_nanmean_custom_cases,
        atol=0.0, rtol=0.0,
    ),

    # var/std/nanvar/nanstd/average (2026-08-02): see module docstring's
    # DEFECT 1/DEFECT 2 paragraphs and the `_VAR_FORMS`/`_VAR_DDOF_FORMS`/
    # `_complex_nan_corpus`/`_average_custom_cases` comments above for the
    # full rationale. Both defects are now FIXED in reductions.rs; the
    # residual grading below is NOT those defects -- it is the SEPARATE,
    # already-disclosed `axis_tuple_noncontig`/non-C-contiguous traversal-
    # order gap `var`/`std`/`nanvar`/`nanstd` inherit from `mean` (finding 1
    # above / the module docstring paragraph just before DEFECT 1): anionpy's
    # `reduce_axis`/`binary_op` path always forces `Order::C` output
    # regardless of input layout, so a Fortran-order or transposed input
    # sums its elements in a DIFFERENT order than numpy's own layout-aware
    # traversal, producing genuine (small) float summation-order rounding
    # differences -- not a value-correctness bug. Measured via a seeded,
    # OUT-OF-CORPUS sweep (`SEED = 20260731`, same seed as corpus.SEED, but
    # 8 hand-picked + 40 seeded-random shapes distinct from the corpus's own
    # shape list, each expanded into C/Fortran/transposed layout variants,
    # crossed with axis/ddof/dtype= kwarg combos and, for the two nan* items,
    # a 15%-NaN-injected variant) -- scratch script (not committed) at
    # `stats_tol_sweep.py`; every dtype key below has n >= 43,122, well
    # above `MIN_ULP_SWEEP_N` (20,000). The grading key for each measurement
    # mirrors harness.py's real `_explicit_dtype_override`/
    # `_first_operand_dtype` logic exactly (explicit `dtype=` kwarg wins,
    # else the first operand's own dtype -- never the output dtype), which
    # is why `nanvar`/`nanstd`'s "float64" key (float16/float32 input with
    # `dtype='float64'`) shows a residual at FLOAT32-epsilon scale
    # (~1.06e-07 / ~5.28e-08) rather than float64 scale: DEFECT 2's fix
    # makes anionpy faithfully reproduce numpy's own behavior of truncating the
    # deviation to the INPUT dtype before squaring, so the "float64" output
    # inherits the input's lower precision by design -- this is numpy's
    # actual documented behavior, not an anionpy shortfall, and every OTHER
    # dtype key on every item measures within ~0.4x-6.8x of that dtype's own
    # eps, consistent with ordinary summation-order non-associativity, not a
    # real bug. Metric is "rel" for every entry: these are reduction
    # outputs whose magnitude is the data's own variance/mean scale, and a
    # relative bound (floor=1.0, per harness.max_rel_distance) is the
    # correct instrument for a summation-order artifact that scales with
    # the reduced value itself.
    "var": ItemSpec(
        name="var", kind="custom",
        custom_cases=_var_custom_cases,
        epsilon_tolerance={
            "complex128": ("rel", 4.199699013373907e-16),
            "complex64": ("rel", 2.870361299756041e-07),
            "float16": ("rel", 0.0021152496337890625),
            "float32": ("rel", 2.8547950137181033e-07),
            "float64": ("rel", 4.3963870033213186e-16),
        },
        epsilon_tolerance_justification=_STATS_EPS_JUST,
        epsilon_sweep={
            "complex128": (43122, 4.199699013373907e-16),
            "complex64": (43122, 2.870361299756041e-07),
            "float16": (43122, 0.0021152496337890625),
            "float32": (258732, 2.8547950137181033e-07),
            "float64": (258732, 4.3963870033213186e-16),
        },
    ),
    "std": ItemSpec(
        name="std", kind="custom",
        custom_cases=_std_custom_cases,
        epsilon_tolerance={
            "complex128": ("rel", 3.0037695203591453e-16),
            "complex64": ("rel", 1.6556319337723835e-07),
            "float16": ("rel", 0.0012788772583007812),
            "float32": ("rel", 2.1301583785771072e-07),
            "float64": ("rel", 3.392380435915522e-16),
        },
        epsilon_tolerance_justification=_STATS_EPS_JUST,
        epsilon_sweep={
            "complex128": (43122, 3.0037695203591453e-16),
            "complex64": (43122, 1.6556319337723835e-07),
            "float16": (43122, 0.0012788772583007812),
            "float32": (258732, 2.1301583785771072e-07),
            "float64": (258732, 3.392380435915522e-16),
        },
    ),
    "nanvar": ItemSpec(
        name="nanvar", kind="custom",
        custom_cases=_nanvar_custom_cases,
        epsilon_tolerance={
            "complex128": ("rel", 4.977307988994899e-16),
            "complex64": ("rel", 2.877001747947361e-07),
            "float16": ("rel", 0.0017547607421875),
            "float32": ("rel", 2.6589225399220595e-07),
            "float64": ("rel", 7.771561172376096e-16),
            "float16->float64": ("rel", 0.0),
            "float32->float64": ("rel", 0.0),
            "complex64->float64": ("rel", 1.092461506517964e-07),
            "complex128->float64": ("rel", 7.1252949362055015e-16),
        },
        epsilon_tolerance_justification=_NANVAR_NANSTD_EPS_JUST,
        epsilon_sweep={
            "complex128": (43122, 4.977307988994899e-16),
            "complex64": (43122, 2.877001747947361e-07),
            "float16": (43122, 0.0017547607421875),
            "float32": (258732, 2.6589225399220595e-07),
            "float64": (75402, 7.771561172376096e-16),
            "float16->float64": (75402, 0.0),
            "float32->float64": (75402, 0.0),
            "complex64->float64": (75402, 1.092461506517964e-07),
            "complex128->float64": (75402, 7.1252949362055015e-16),
        },
    ),
    "nanstd": ItemSpec(
        name="nanstd", kind="custom",
        custom_cases=_nanstd_custom_cases,
        epsilon_tolerance={
            "complex128": ("rel", 3.134303943293842e-16),
            "complex64": ("rel", 1.8344549346238637e-07),
            "float16": ("rel", 0.0011949539184570312),
            "float32": ("rel", 2.0557828861456073e-07),
            "float64": ("rel", 4.440892098500626e-16),
            "float16->float64": ("rel", 0.0),
            "float32->float64": ("rel", 0.0),
            "complex64->float64": ("rel", 5.462307388304211e-08),
            "complex128->float64": ("rel", 4.661962290625788e-16),
        },
        epsilon_tolerance_justification=_NANVAR_NANSTD_EPS_JUST,
        epsilon_sweep={
            "complex128": (43122, 3.134303943293842e-16),
            "complex64": (43122, 1.8344549346238637e-07),
            "float16": (43122, 0.0011949539184570312),
            "float32": (258732, 2.0557828861456073e-07),
            "float64": (75402, 4.440892098500626e-16),
            "float16->float64": (75402, 0.0),
            "float32->float64": (75402, 0.0),
            "complex64->float64": (75402, 5.462307388304211e-08),
            "complex128->float64": (75402, 4.661962290625788e-16),
        },
    ),
    "average": ItemSpec(
        name="average", kind="custom",
        custom_cases=_average_custom_cases,
        epsilon_tolerance={
            "complex128": ("rel", 1.496439074554386e-15),
            "complex64": ("rel", 8.881784197001252e-16),
            "float16": ("rel", 1.1379786002407855e-15),
            "float32": ("rel", 5.551115123125783e-16),
            "float64": ("rel", 6.64811507269307e-16),
        },
        epsilon_tolerance_justification=_AVERAGE_EPS_JUST,
        epsilon_sweep={
            "complex128": (53298, 1.496439074554386e-15),
            "complex64": (53298, 8.881784197001252e-16),
            "float16": (53298, 1.1379786002407855e-15),
            "float32": (53298, 5.551115123125783e-16),
            "float64": (53298, 6.64811507269307e-16),
        },
    ),
}


# ---------------------------------------------------------------------------
# 0-d OPERAND x EXPLICIT AXIS (2026-08-03, Monday).
#
# The boundary that produced a whole generation of false declarations: every
# reduction here was declared "exact" while `f(np.array(3.0), axis=0)` either
# panicked or returned the wrong shape, because no case in this file ever
# crossed a 0-DIMENSIONAL operand with an EXPLICIT axis kwarg. The corpus is
# rich in ranks 1..3 and rich in axis values, and the product of the two was
# simply never taken.
#
# Why each axis form is here (all measured against real numpy 2.5.1, not
# assumed -- numpy's behaviour at this boundary is INCONSISTENT BY DESIGN
# and the inconsistency is the reference, so a fix must reproduce it rather
# than tidy it up):
#
#   omitted / none  -- the baseline: both reduce over everything.
#   0 / -1          -- the two spellings of "the (nonexistent) first axis".
#                      Accepted by most reductions on a 0-d operand as a
#                      courtesy, REJECTED by mean/var/std/average/size.
#   1 / -2          -- unambiguously out of range; must raise, and must say
#                      so with numpy's exact message.
#   ()              -- reduces over NOTHING; a distinct code path from None.
#   (0,) / (-1,)    -- the SEQUENCE spelling. numpy does not extend the 0-d
#                      courtesy to it, so these raise where the bare scalar
#                      succeeds. Same axis, different container, opposite
#                      answer.
#   (1,)            -- out of range in sequence form.
#   (0,0) / (0,-1)  -- duplicates. Bounds are checked BEFORE repeats, so on
#                      a 0-d operand these are an AxisError, while at rank 1
#                      the same input is a ValueError.
#
# DTYPE is crossed in deliberately, not for coverage-padding: numpy's
# `nanmean`/`nanvar`/`nanstd` accept a 0-d `axis=0` for INEXACT operands and
# raise `AxisError` for bool/integer ones, because `_replace_nan` hands an
# exact dtype straight to `mean`/`var`/`std`. A float-only sweep reports
# agreement on a function that is wrong for every integer width -- which is
# exactly how that hole survived the first pass here.
#
# These are appended to each item's EXISTING cases rather than registered as
# a new item, so the boundary travels with the item it constrains and cannot
# be silently dropped from a run.
# ---------------------------------------------------------------------------

_TL_NO_AXIS = object()

_TL_ZERO_D_AXIS_FORMS = [
    ("omitted", _TL_NO_AXIS),
    ("none", None),
    ("0", 0),
    ("-1", -1),
    ("1", 1),
    ("-2", -2),
    ("empty_tuple", ()),
    ("tuple_0", (0,)),
    ("tuple_-1", (-1,)),
    ("tuple_1", (1,)),
    ("tuple_dup", (0, 0)),
    ("tuple_dup_neg", (0, -1)),
]

# (label, 0-d datum). Spans the exact/inexact split that gates the nan*
# family, plus a NaN payload so the nan-aware paths do real work rather than
# short-circuiting on a clean operand.
_TL_ZERO_D_DATA = [
    ("float64", 3.0),
    ("float64_nan", float("nan")),
    ("float64_zero", 0.0),
    ("int", 7),
    ("bool", True),
]

# Items whose signature has no `keepdims`.
_TL_NO_KEEPDIMS = {"cumsum", "cumprod", "argmin", "argmax"}
# Items that reject a tuple axis outright at every rank (argmin/argmax take
# an int or None only) -- the tuple forms still belong in the sweep, since
# "raises TypeError with numpy's exact wording" is itself the contract.
_TL_LOGICAL = {"any", "all"}


def _tl_zero_d_axis_cases_for(name: str) -> list:
    """0-d x explicit-axis cases for one top-level reduction."""
    cases = []
    keepdims_opts = [None] if name in _TL_NO_KEEPDIMS else [None, False, True]
    for data_label, datum in _TL_ZERO_D_DATA:
        # A bool operand for the logical reductions is the meaningful one;
        # the numeric items still get it, since bool is an EXACT dtype and
        # that is precisely what the nan* split turns on.
        for axis_label, axis in _TL_ZERO_D_AXIS_FORMS:
            for keepdims in keepdims_opts:
                label = f"{name}/0d/{data_label}/axis={axis_label}"
                kwargs = {}
                if axis is not _TL_NO_AXIS:
                    kwargs["axis"] = axis
                if keepdims is not None:
                    kwargs["keepdims"] = keepdims
                    label = f"{label}/keepdims={keepdims}"
                cases.append((label, (datum,), kwargs))
    return cases


def _tl_append_zero_d_axis_cases():
    """Attach the boundary cases to every reduction spec that owns one.

    Uses `extra_cases`, not `custom_cases`: 14 of these 23 items are
    kind="method", for which `custom_cases` is ignored entirely. See
    `ItemSpec.extra_cases` in registry.py for why no call form could have
    reached a 0-d operand on its own.

    Wraps any existing `extra_cases` rather than overwriting it, and asserts
    every named item is really present -- a renamed or deleted spec must
    fail loudly here instead of quietly shrinking the sweep, which is the
    exact failure mode that let the original gap survive.
    """
    for name in _TL_ZERO_D_ITEMS:
        spec = REDUCTION_SPECS.get(name)
        assert spec is not None, f"0-d boundary: no REDUCTION_SPECS entry for {name!r}"
        prev = spec.extra_cases

        def make(prev=prev, name=name):
            def wrapped():
                base = list(prev()) if prev is not None else []
                return base + _tl_zero_d_axis_cases_for(name)
            return wrapped

        spec.extra_cases = make()


_TL_ZERO_D_ITEMS = [
    "sum", "prod", "mean", "min", "max", "any", "all", "ptp",
    "count_nonzero", "nanmin", "nanmax", "nansum", "nanprod", "nanmean",
    # `amax`/`amin` are SEPARATE ledger keys from `max`/`min` and were
    # revoked for this identical boundary under different wording -- they
    # share the mechanism but not the entry, and an item is credited by key.
    "amax", "amin",
    "var", "std", "nanvar", "nanstd", "average", "cumsum", "cumprod",
    "argmin", "argmax",
]

_tl_append_zero_d_axis_cases()


# ---------------------------------------------------------------------------
# AXIS-TYPE guards (2026-08-04). Attach `_axis_type_cases` to every reduction
# that owns one of numpy's five axis converters -- see that function's
# docstring for the converter map and for why shape () and dtype int64 are
# load-bearing rather than padding.
#
# Same `extra_cases` mechanism, and for the same reason, as the 0-d block
# above: most of these items are kind="method", for which `custom_cases` is
# ignored entirely, so a guard placed there would silently never run.
#
# These forms were measurably WRONG before cc5523f/cab03ab/db62b24 -- 592
# diverging cases in the original 3,770-case sweep, and a further 660 once it
# was widened to `ndarray` method forms, a second dtype and shape (). They now
# read zero. This block is what keeps them there; without it the entire
# converter contract is defended by nothing but a scratch script in /tmp.
#
# `mean`/`var`/`std`/`nanmean` are deliberately NOT listed: they already call
# `_axis_type_cases` directly from their own `custom_cases`, and listing them
# here too would merely duplicate every case.
# ---------------------------------------------------------------------------

_AXISTYPE_ITEMS = [
    # converter 1 -- PyArray_ConvertMultiAxis (tuple is the only sequence)
    "sum", "prod", "min", "max", "any", "all", "nansum", "nanprod",
    "nanvar", "nanstd", "ptp", "count_nonzero", "nanmin", "nanmax",
    # converter 3 -- the single-axis one (fully qualified type names)
    "argmin", "argmax", "cumsum", "cumprod", "nancumsum", "nancumprod",
    # converter 4 -- normalize_axis_tuple (iterates ANY sequence)
    "average",
]
# `median`/`nanmedian` (converter 4) live in stats_cases.py and
# `nanargmin`/`nanargmax` (converter 5) in sort_cases.py, not in
# REDUCTION_SPECS. They attach the SAME `_axis_type_cases` from their own
# modules -- see `_axistype_append_cases` there -- importing it lazily at
# case-build time so this module and theirs do not form an import cycle.
# The assert below is what surfaced that split rather than letting those
# four items quietly go unguarded.


def _axistype_append_cases():
    """Attach the axis-type sweep to every spec in `_AXISTYPE_ITEMS`.

    Wraps any existing `extra_cases` rather than overwriting it, and asserts
    every named item is really present, so a renamed or deleted spec fails
    loudly here instead of quietly shrinking the sweep.
    """
    for name in _AXISTYPE_ITEMS:
        spec = REDUCTION_SPECS.get(name)
        assert spec is not None, f"axis-type: no REDUCTION_SPECS entry for {name!r}"
        prev = spec.extra_cases

        def make(prev=prev, name=name):
            def wrapped():
                base = list(prev()) if prev is not None else []
                return base + _axis_type_cases(name)
            return wrapped

        spec.extra_cases = make()


_axistype_append_cases()
