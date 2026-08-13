"""Differential `ItemSpec`s for `median`, `nanmedian`, `percentile`,
`quantile`, `nanpercentile`, `nanquantile` -- backed by
`ionp-core/src/stats.rs` (sort-based reduction kernels) and
`ionp-py/src/stats.rs` (thin marshaling wrappers).

All six are `kind="custom"`: `quantile`/`percentile`/`nanquantile`/
`nanpercentile` need a `q` argument alongside the array (not a plain
unary/binary corpus shape), and `median`/`nanmedian` are included here
rather than in `reduction_cases.py` so all six share one array/dtype/shape
sweep and one exclusion rationale.

Every case below is `atol=0.0, rtol=0.0` (bit-exact) -- confirmed reachable
across the WHOLE non-excluded call-form space by an out-of-corpus probe
(`tests/differential/_probe_stats.py`, 129,320 calls, 12 dtypes x 8 shapes x
axis x keepdims x 13 methods x 4 `q` forms, real numpy 2.5.1): 0 value
mismatches, 0 error-type/message mismatches. The only mismatches that probe
still finds (7,729 of 129,320, all `shape/dtype`, never `value`) are the two
related dtype-preservation gaps documented and EXCLUDED below -- neither is
worked around or hidden, both are real, both are narrow.

---------------------------------------------------------------------------
DELIBERATELY EXCLUDED call-forms (measured, not assumed -- same rigor as
the `reduction_cases.py` `_SUM_EXCLUDE`/`_MEAN_EXCLUDE` precedent)
---------------------------------------------------------------------------

1. WEAK-SCALAR-Q DTYPE PRESERVATION (float16/float32 input, an INTERPOLATED
   method, `q` passed as a bare Python scalar). Real numpy's own
   `_quantile`/`_lerp` (read via `inspect.getsource`) set a `weak_q` flag
   True only when the caller's `q` was a bare Python `int`/`float` (NOT an
   array, not even a 1-element one) -- when true, `gamma` is converted to a
   plain Python `float` ("weak" under NEP 50), which does NOT force
   `float16`/`float32` array promotion in `_lerp`'s arithmetic, so the
   output keeps the input's own narrow dtype. When `q` is an array (even
   `[0.5]`), `gamma` stays a float64-dtyped array and ordinary NEP 50
   array-array promotion forces float64 output regardless of input dtype.
   anionpy's `ionp-py/src/stats.rs::q_from_pyobj` collapses every `q` form
   (bare scalar, 0-d array, 1-element list) to the same `Vec<f64>` at the
   Python/Rust boundary, discarding whether the ORIGINAL call used a bare
   scalar -- replicating this would need a new `weak_q: bool` field
   threaded from the pyfunction binding all the way through `QuantileArgs`
   into `build_regular`/`build_nan_variant`'s F64-casting decision. Live-
   verified via direct comparison: `np.quantile(f16_arr, 0.5)` -> float16;
   `np.quantile(f16_arr, np.array(0.5))` -> float64;
   `np.quantile(f16_arr, [0.0, 0.25, 0.5, 0.75, 1.0])` -> float64.
   EXCLUDED forms: any (float16 or float32 dtype) x (interpolated method:
   everything except inverted_cdf/closest_observation/lower/higher/nearest)
   x (bare-Python-scalar `q`) combination.

2. ALL-NaN-SLICE DTYPE PRESERVATION for `nanquantile`/`nanpercentile` on
   float16/float32 input with an interpolated method -- a DIFFERENT
   manifestation of the same underlying narrow-dtype-preservation gap (not
   the empty-axis/zero-rows short-circuit already handled correctly in
   `quantile_axis`, which DOES preserve dtype via `nan_scalar` -- this is
   the case where the axis/slice is nonempty but every value in it happens
   to be NaN). Live-verified: `np.nanquantile(np.full(4, np.nan,
   dtype=np.float16), [0.1, 0.9], method="linear")` stays float16 (shape
   `(2,)`, matching `len(q)`); anionpy's `build_nan_variant` unconditionally
   pushes `f64::NAN` into a `Buffer::F64` for the `valid[r] == 0` case,
   losing the narrow dtype the same way case 1 does. Root cause: numpy's
   `nanquantile` processes each 1-D slice via `_nanquantile_1d`, which for
   an all-NaN slice returns a same-dtype NaN-filled array directly, bypassing
   the weak_q/gamma promotion machinery entirely -- so this happens
   regardless of whether `q` was a bare scalar or an array (broader than
   gap 1). EXCLUDED forms: any (float16 or float32 dtype) x (interpolated
   method) x (all-NaN array) combination, for `nanquantile`/`nanpercentile`
   only (plain `quantile`/`percentile` never see this path).

3. MULTI-DIMENSIONAL `q` (`q.ndim > 1`). Real numpy accepts an N-D `q`
   array as long as N is anything (docstring claims "scalar or 1d" but the
   actual implementation does not enforce that -- live-verified:
   `np.quantile(np.array([1., 2., 3.]), np.array([[0.1, 0.2], [0.3, 0.4]]))`
   returns a `(2, 2)`-shaped result, no error). anionpy's `q_from_pyobj`
   explicitly rejects anything but 0-d/1-d `q` with `ValueError("q must be
   a scalar or 1d")`, which is itself numpy's OWN historical error message
   for an unrelated, narrower restriction that numpy no longer enforces --
   not implemented; would need `QuantileArgs`/`build_regular` reworked to
   carry an arbitrary output q-shape instead of the current flat `Vec<f64>`
   length. EXCLUDED for all six items.

4. `weights=` (numpy 2.0+ `quantile`/`percentile`/`nanquantile`/
   `nanpercentile` keyword) -- not in `ionp-py/src/stats.rs`'s pyfunction
   signatures at all (calling with `weights=` raises Python's own
   "unexpected keyword argument" TypeError, not a numpy-shaped error).
   EXCLUDED for the four quantile-family items (median/nanmedian never had
   a `weights` parameter in numpy either).

5. Integer-dtype magnitude `> 2**53` for INTERPOLATED methods only --
   inherited from the same float64-intermediate-precision limitation as
   every other anionpy reduction that widens integers through an f64 pipeline
   (see `reduction_cases.py`'s own dtype notes); direct-index methods are
   exact for any integer magnitude since they never arithmetic-combine two
   samples. Not separately re-verified this pass (no int64/uint64 corpus
   value here exceeds 2**53 magnitude to begin with), listed for parity
   with the pre-existing precedent, not because a fresh mismatch was found.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

import corpus
from registry import ItemSpec

STATS_SPECS: dict[str, ItemSpec] = {}

# ---------------------------------------------------------------------------
# Shared corpus
# ---------------------------------------------------------------------------

_ALL_METHODS = [
    "inverted_cdf", "averaged_inverted_cdf", "closest_observation",
    "interpolated_inverted_cdf", "hazen", "weibull", "linear",
    "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint",
    "nearest",
]
_DIRECT_METHODS = {"inverted_cdf", "closest_observation", "lower", "higher", "nearest"}
_INTERP_METHODS = [m for m in _ALL_METHODS if m not in _DIRECT_METHODS]

_DTYPES = [np.bool_, np.int8, np.int64, np.uint16, np.float16, np.float32, np.float64]
_NARROW_FLOATS = {np.float16, np.float32}

_SHAPES = [(1,), (2,), (5,), (2, 3), (3, 4, 2), (0,), (2, 0, 3)]

_Q_ARRAY = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def _make_array(dt, shape, rng):
    if dt == np.bool_:
        return rng.integers(0, 2, size=shape).astype(bool)
    if np.issubdtype(dt, np.integer):
        return rng.integers(-100, 100, size=shape).astype(dt)
    return (rng.standard_normal(size=shape) * 12).astype(dt)


def _axes_for(ndim):
    if ndim == 0:
        return [None]
    return [None] + list(range(-ndim, ndim))


def _skip_weak_scalar_q(dt, method, q_is_bare_scalar):
    """Gap 1 above: bare-scalar `q` + narrow float + interpolated method."""
    return dt in _NARROW_FLOATS and method in _INTERP_METHODS and q_is_bare_scalar


def _skip_allnan_narrow_interp(dt, method, is_allnan, is_nan_variant):
    """Gap 2 above: all-NaN slice + narrow float + interpolated method, nan*
    variants only."""
    return is_nan_variant and is_allnan and dt in _NARROW_FLOATS and method in _INTERP_METHODS


# ---------------------------------------------------------------------------
# median / nanmedian -- no `q`/`method`, just axis/keepdims/dtype/shape
# ---------------------------------------------------------------------------

def _median_cases(is_nan_variant: bool):
    rng = np.random.default_rng(20260801)
    cases = []
    for dt in _DTYPES:
        for shape in _SHAPES:
            arr = _make_array(dt, shape, rng)
            for axis in _axes_for(arr.ndim):
                for keepdims in (False, True):
                    label = f"dt={np.dtype(dt).name}/shape={shape}/axis={axis}/kd={keepdims}"
                    cases.append((label, (arr,), {"axis": axis, "keepdims": keepdims}))
    # all-NaN slices (float dtypes only; irrelevant to non-nan median's
    # value but exercises the shared has_nan-detection path either way)
    for dt in (np.float16, np.float32, np.float64):
        for shape in ((4,), (2, 3)):
            arr = np.full(shape, np.nan, dtype=dt)
            cases.append((f"allnan/dt={np.dtype(dt).name}/shape={shape}", (arr,), {}))
    # mixed-NaN rows
    mixed = np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]])
    cases.append(("mixed_nan/axis1", (mixed,), {"axis": 1}))
    # 0-d
    for dt in _DTYPES:
        v = np.array(True) if dt == np.bool_ else np.array(5, dtype=dt)
        cases.append((f"0d/dt={np.dtype(dt).name}", (v,), {}))
    # FORM axis (2026-08-02, CLASS A closure task): median/nanmedian take
    # their array as a plain positional argument to a top-level anionpy
    # function -- crossed here with the same corpus.unary_corpus() dtype/
    # shape/view axis already used elsewhere in this module's siblings
    # (reduction_cases.py's `_form_axis_custom_cases`), not a separate
    # one-off sweep.
    for c in corpus.form_axis_cases(sample_stride=8):
        cases.append((c.label, (c.value,), {}))
    cases.extend(_stats_edge_cases(is_nan_variant=is_nan_variant))
    return cases


# ---------------------------------------------------------------------------
# 2026-08-04 REGRESSION BLOCK -- seven divergence classes found by an
# 82,504-case out-of-corpus sweep (/tmp/qsweep2.py: 8 dtypes x 12 operand
# PROVENANCES x 20 kwarg sets x per-function `q` forms, plus method,
# error, non-array-input, `out=`, aliasing and STRIDE blocks). The corpus
# above passed 100% while that sweep found 3,941 diffs, so every case below
# exists because the corpus could not see it -- provenance (F-order,
# transposed, strided, offset views), operand dtype (complex), `q` TYPE
# (bool/integer/0-d/numpy scalar), axis TYPE (float, str, numpy scalar,
# 0-d array), size-0 operands, and result LAYOUT.
#
# ILLUSTRATIVE, NOT EXHAUSTIVE: these are the seven classes the sweep
# surfaced, not a claim that the space is now closed.

def _layout_operands(dt=np.float64):
    """Same VALUES, four different memory provenances. The corpus builds
    every operand the same way (fresh C-contiguous `_make_array`), which is
    exactly why the F-order `axis=()` layout bug (class 7) survived it."""
    base = (np.arange(30, dtype=np.float64).reshape(5, 6) - 14.5).astype(dt)
    return [
        ("c", base),
        ("f", np.asarray(base, order="F")),
        ("T", base.T),
        ("strided", base[:, ::2]),
        ("offset", base[1:, 2:]),
    ]


def _stats_edge_cases(*, is_nan_variant, q_scale=None):
    """`q_scale=None` -> median/nanmedian (no `q`); 1.0 -> quantile family;
    100.0 -> percentile family."""
    has_q = q_scale is not None
    def call(arr, kw, q=None):
        return ((arr,) if q is None else (arr, q)), kw
    cases = []

    # --- class 7: result LAYOUT. `median` inherits the operand's memory
    # order for `axis=()`; `quantile`/`percentile` do not (both measured
    # 2026-08-04). Every other axis form already matched. `check_strides`
    # on the spec is what makes these bite -- without it they compare only
    # values and pass either way.
    for name, arr in _layout_operands():
        for kw in ({"axis": ()}, {"axis": 0}, {"axis": 0, "keepdims": True},
                   {"axis": (0, 1)}, {"axis": None, "keepdims": True}):
            args, k = call(arr, kw, 0.5 * q_scale if has_q else None)
            cases.append((f"layout/{name}/{kw}", args, k))
    # keepdims inserts a BROADCAST (stride-0) axis, including in front of a
    # `q` axis for the quantile family.
    for name, arr in _layout_operands():
        if has_q:
            cases.append((f"kdstride/{name}/qlist", (arr, [0.1 * q_scale, 0.9 * q_scale]),
                          {"axis": 0, "keepdims": True}))
            cases.append((f"kdstride/{name}/qscalar", (arr, 0.5 * q_scale),
                          {"axis": 1, "keepdims": True}))
        else:
            cases.append((f"kdstride/{name}", (arr,), {"axis": 1, "keepdims": True}))

    # --- class 5: complex `median` finishes through `mean()`, which
    # divides by the count with Smith's algorithm; `x * 0.0` is NaN for an
    # infinite x, so `median([inf+0j]) == inf+nanj`. Odd (divide by 1) and
    # even (divide by 2) lengths both go through it.
    cplx = np.array([complex(np.inf, 0.0), complex(1.0, np.inf),
                     complex(-np.inf, 3.0), complex(np.inf, np.inf),
                     complex(1.0, 2.0)], dtype=np.complex128)
    for n in (1, 2, 3, 4, 5):
        args, k = call(cplx[:n], {}, 0.5 * q_scale if has_q else None)
        cases.append((f"complex/inf/n={n}", args, k))
    args, k = call(cplx.astype(np.complex64), {}, 0.5 * q_scale if has_q else None)
    cases.append(("complex/c64", args, k))
    args, k = call(np.array([[complex(np.inf, 0.0), 1 + 1j], [2 + 0j, complex(0.0, np.inf)]]),
                   {"axis": 1}, 0.5 * q_scale if has_q else None)
    cases.append(("complex/axis1", args, k))

    # --- class 3 + 4: a size-0 operand. The `nan*` variants delegate to
    # `nanmean` BEFORE anything else -- which DROPS `q` entirely (the
    # result is `nanmean`-shaped, not `len(q)`-shaped) and never validates
    # `method`. The plain variants instead reach `_ureduce`'s reshape,
    # which cannot infer `-1` from a size-0 shape and raises.
    for shape in ((0,), (0, 3), (2, 0, 3)):
        empty = np.zeros(shape, dtype=np.float64)
        for kw in ({}, {"axis": 0}, {"axis": ()}, {"axis": (0, 1)} if len(shape) > 1 else {"axis": (0,)},
                   {"axis": None, "keepdims": True}, {"axis": (), "keepdims": True}):
            for q in ([0.1 * q_scale, 0.9 * q_scale], 0.5 * q_scale) if has_q else (None,):
                args, k = call(empty, kw, q)
                cases.append((f"empty/{shape}/{kw}/q={q}", args, k))
        for dt in (np.int64, np.bool_, np.float16):
            args, k = call(np.zeros(shape, dtype=dt), {"axis": 0},
                           0.5 * q_scale if has_q else None)
            cases.append((f"empty/{shape}/dt={np.dtype(dt).name}", args, k))
    if has_q:
        cases.append(("empty/method_unvalidated", (np.zeros((0, 3)), 0.5 * q_scale),
                      {"axis": 0, "method": "nope"}))

    # --- axis TYPE errors. Three distinct numpy messages, and which one
    # you get depends on the operand dtype (a size-0 integer operand routes
    # through `nanmean` -> `mean`, whose C converter has its own float
    # branch). Covered on both an ordinary and a size-0 operand for exactly
    # that reason.
    import decimal
    bad_axes = [0.0, np.float64(0), np.float32(0), "x", ("x",), (0.0,), (0, 0.0),
                ("x", 0.0), np.array(0.0), (np.array(0.0),), decimal.Decimal(0),
                (decimal.Decimal(0),), np.True_, (np.True_,), None if False else (0, 0),
                (0, 5), 100, (0.0, 1)]
    # Array-valued axes belong on EVERY operand, not just the full one: a
    # size-0 operand takes the `nanmean` delegation, whose converter
    # disagrees with `median`'s about all of them. Measured 2026-08-04:
    #     np.median(np.zeros((2, 3)), axis=np.array([0]))       -> OK (3,)
    #     np.nanmedian(np.zeros((0, 3)), axis=np.array([0]))    -> TypeError:
    #         only integer scalar arrays can be converted to a scalar index
    #     np.nanmedian(np.zeros((0, 3)), axis=np.array(0))      -> OK (3,)
    # The 0-d integer array is the one that caught a SILENT wrong answer
    # rather than a wrong message: anionpy extracted an EMPTY axis vector from
    # it through the sequence protocol and returned the operand unreduced.
    # Named `axistype/` not `axisok/` because whether a form is accepted is
    # operand-dependent; the harness compares, it does not presume.
    array_axes = [np.array(0), (np.array(0),), np.array([0]), (np.array([0]),),
                  np.array(1), np.int64(1), True, (True,)]
    # 1-D operands are NOT redundant with the 2-D ones. numpy's mean-family
    # axis converter turns a Python/numpy bool into the index 1 and
    # RANGE-CHECKS it before re-parsing it with the converter that rejects
    # bools, so the reported error flips on ndim alone. Measured 2026-08-04:
    #     np.mean(np.zeros((0, 3)), axis=True) -> TypeError: an integer is required
    #     np.mean(np.zeros((5,)),   axis=True) -> AxisError: axis 1 is out of
    #                                             bounds for array of dimension 1
    # Without an ndim<2 operand the range-check arm is unreachable and a
    # guard covering it cannot be exercised at all.
    for operand, tag in ((np.zeros((2, 3)), "full"), (np.zeros((0, 3)), "empty"),
                         (np.zeros((0, 3), dtype=np.int64), "empty_int"),
                         (np.zeros((5,)), "vec"), (np.zeros((0,)), "vec_empty")):
        for ax in bad_axes + array_axes:
            args, k = call(operand, {"axis": ax}, 0.5 * q_scale if has_q else None)
            cases.append((f"axistype/{tag}/{ax!r}", args, k))

    if has_q:
        # --- class 1: integral `q` takes the ELEMENT, preserving the
        # operand's dtype (`np.quantile(int8_arr, 1)` is int8, not
        # float64). numpy reaches this through `method_props["fix_gamma"]
        # is None` OR a `linear` virtual-index array that came out with an
        # INTEGER dtype -- which only happens when `q` itself was integral.
        # `percentile` is structurally exempt: `q / 100` is always float.
        for dt in (np.int8, np.int64, np.uint16, np.bool_, np.float16, np.float32):
            arr = np.array([3, 1, 4, 1, 5], dtype=dt)
            for q in (0, 1, True, False, np.int64(1), np.array(1), [0, 1], (0, 1),
                      np.array([0, 1]), 0.0, 1.0, np.float64(1)):
                if q_scale != 1.0:
                    q = q if isinstance(q, (list, tuple)) else q
                qq = ([v * q_scale for v in q] if isinstance(q, (list, tuple))
                      else q * q_scale)
                cases.append((f"qintegral/dt={np.dtype(dt).name}/q={q!r}",
                              (arr, qq), {}))
                for method in ("linear", "lower", "nearest", "midpoint"):
                    cases.append((f"qintegral/dt={np.dtype(dt).name}/q={q!r}/m={method}",
                                  (arr, qq), {"method": method}))

        # --- class 2: the 0-d NaN-substitution dtype flip. `_quantile`'s
        # tail replaces a 0-d result with `arr[-1]` when `out is None`, so a
        # NaN-carrying float16 slice comes back float16 even though the
        # NaN-free call returns float64.
        for dt in (np.float16, np.float32, np.float64):
            arr = np.array([1.0, np.nan], dtype=dt)
            for q in (0.5 * q_scale, np.float64(0.5 * q_scale), [0.5 * q_scale]):
                for kd in (False, True):
                    cases.append((f"nan0d/dt={np.dtype(dt).name}/q={q!r}/kd={kd}",
                                  (arr, q), {"keepdims": kd}))
    return cases


STATS_SPECS["median"] = ItemSpec(
    name="median", kind="custom", custom_cases=lambda: _median_cases(False),
    atol=0.0, rtol=0.0, check_strides=True,
)
STATS_SPECS["nanmedian"] = ItemSpec(
    name="nanmedian", kind="custom", custom_cases=lambda: _median_cases(True),
    atol=0.0, rtol=0.0, check_strides=True,
)


# ---------------------------------------------------------------------------
# quantile / percentile / nanquantile / nanpercentile
# ---------------------------------------------------------------------------

def _quantile_family_cases(as_percentile: bool, is_nan_variant: bool):
    rng = np.random.default_rng(20260801)
    cases = []
    scale = 100.0 if as_percentile else 1.0

    for dt in _DTYPES:
        for shape in _SHAPES:
            arr = _make_array(dt, shape, rng)
            for axis in _axes_for(arr.ndim):
                for keepdims in (False, True):
                    for method in _ALL_METHODS:
                        # array-form q -- always safe (never weak-scalar).
                        q_arr = [v * scale for v in _Q_ARRAY]
                        label = f"dt={np.dtype(dt).name}/shape={shape}/axis={axis}/kd={keepdims}/m={method}/q=array"
                        cases.append((label, (arr, q_arr), {"axis": axis, "method": method, "keepdims": keepdims}))

                        # bare-scalar q -- excluded for narrow-float +
                        # interpolated method (gap 1).
                        if not _skip_weak_scalar_q(dt, method, True):
                            q_scalar = 0.5 * scale
                            label2 = f"dt={np.dtype(dt).name}/shape={shape}/axis={axis}/kd={keepdims}/m={method}/q=scalar"
                            cases.append((label2, (arr, q_scalar), {"axis": axis, "method": method, "keepdims": keepdims}))

    # all-NaN slices -- excluded for narrow-float + interpolated method on
    # the nan* variants only (gap 2); plain quantile/percentile still
    # propagate NaN normally through the ordinary lerp path and ARE
    # covered here.
    for dt in (np.float16, np.float32, np.float64):
        for shape in ((4,), (2, 3)):
            arr = np.full(shape, np.nan, dtype=dt)
            for method in _ALL_METHODS:
                if _skip_allnan_narrow_interp(dt, method, True, is_nan_variant):
                    continue
                q_arr = [v * scale for v in _Q_ARRAY]
                label = f"allnan/dt={np.dtype(dt).name}/shape={shape}/m={method}"
                cases.append((label, (arr, q_arr), {"method": method}))

    # mixed-NaN rows
    mixed = np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]])
    for method in _ALL_METHODS:
        q_arr = [v * scale for v in (0.25, 0.75)]
        cases.append((f"mixed_nan/m={method}", (mixed, q_arr), {"axis": 1, "method": method}))

    # 0-d input
    for dt in _DTYPES:
        v = np.array(True) if dt == np.bool_ else np.array(5, dtype=dt)
        for method in _ALL_METHODS:
            q_arr = [v_ * scale for v_ in _Q_ARRAY]
            cases.append((f"0d/dt={np.dtype(dt).name}/m={method}", (v, q_arr), {"method": method}))

    # error triggers -- range checks, bad method name, complex rejection,
    # empty array, axis out of bounds, boolean+interpolated TypeError.
    err_arr = np.array([1.0, 2.0, 3.0])
    lo, hi = (0.0, 100.0) if as_percentile else (0.0, 1.0)
    cases.append(("err/out_of_range_high", (err_arr, hi + 50 * scale), {}))
    cases.append(("err/out_of_range_low", (err_arr, lo - 0.1 * scale), {}))
    cases.append(("err/bad_method", (err_arr, 0.5 * scale), {"method": "bogus"}))
    cases.append(("err/axis_oob", (err_arr, 0.5 * scale), {"axis": 5}))
    cases.append(("err/empty_array", (np.array([], dtype=np.float64), 50.0 * scale), {}))
    cases.append(("err/complex_input", (np.array([1 + 1j, 2 + 2j]), 50.0 * scale), {}))
    cases.append(("err/bool_linear", (np.array([True, False]), 0.5 * scale), {"method": "linear"}))

    cases.extend(_stats_edge_cases(is_nan_variant=is_nan_variant, q_scale=scale))
    return cases


for _as_pct, _name in ((False, "quantile"), (True, "percentile")):
    STATS_SPECS[_name] = ItemSpec(
        name=_name, kind="custom",
        custom_cases=(lambda ap=_as_pct: _quantile_family_cases(ap, False)),
        atol=0.0, rtol=0.0, check_strides=True,
    )
for _as_pct, _name in ((False, "nanquantile"), (True, "nanpercentile")):
    STATS_SPECS[_name] = ItemSpec(
        name=_name, kind="custom",
        custom_cases=(lambda ap=_as_pct: _quantile_family_cases(ap, True)),
        atol=0.0, rtol=0.0, check_strides=True,
    )


# ---------------------------------------------------------------------------
# AXIS-TYPE guards (2026-08-04). These items own one of numpy's five axis
# converters and were measurably wrong before cc5523f/cab03ab/db62b24; the
# sweep that found it now reads zero. This is what keeps it there.
#
# The case builder lives in reduction_cases.py (see `_axis_type_cases` for
# the converter map and for why shape () and dtype int64 are load-bearing).
# It is imported INSIDE the closure, i.e. at case-build time rather than at
# module import time, because reduction_cases.py and this module would
# otherwise form an import cycle through registry.py.
# ---------------------------------------------------------------------------

def _axistype_append_cases(specs, names):
    for name in names:
        spec = specs.get(name)
        assert spec is not None, f"axis-type: no spec entry for {name!r}"
        prev = spec.extra_cases

        def make(prev=prev, name=name):
            def wrapped():
                from reduction_cases import _axis_type_cases
                base = list(prev()) if prev is not None else []
                return base + _axis_type_cases(name)
            return wrapped

        spec.extra_cases = make()


_axistype_append_cases(STATS_SPECS, ["median", "nanmedian"])
