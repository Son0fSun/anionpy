"""Differential coverage for the Array-API/2-D-construction cluster added
2026-08-03: `permute_dims`, `clip`, `cumulative_sum`, `cumulative_prod`,
`tri`.

NEW FILE, same collision-checked-merge pattern as manip_compose_cases.py --
build a dict of ItemSpecs, merged into registry.REGISTRY from the bottom of
registry.py.

All five are Python-only compositions in `anionpy/__init__.py` over
already-verified Rust primitives (`transpose`, `ndarray.clip`, `cumsum`,
`cumprod`, `arange`, `greater_equal`, `astype`, `concatenate`,
`full_like`). No Rust changed.

WHAT IS DELIBERATELY NOT EXERCISED HERE, and why (these are exclusions from
the TEST, distinct from the three siblings excluded from the
IMPLEMENTATION -- see `__init__.py`'s block comment for `geomspace`,
`logspace` and `concat`):

  - `clip(..., out=...)` and `cumulative_*(..., out=...)`: `out=` writes
    into a caller-supplied array, and the whole `out=` contract was
    entangled with the no-`__setitem__`/no-true-views architectural
    absence. Testing it here would be testing that absence, not these
    five. Excluded, not passed-and-ignored.

    [CORRECTED 2026-08-03] Half of that entanglement is gone:
    `ndarray.__setitem__` landed 2026-08-02 (declared 2026-08-03, 2439-case
    corpus) and real views landed with it, so `out=` is no longer blocked
    on an architectural absence. The exclusion above STANDS for this file
    regardless -- `out=` is still not what these five items are about, and
    scope is the reason, not impossibility. But it is now queued work
    rather than a wall, and the word "absence" above should not be read as
    still describing the codebase.
  - `cumulative_*(out=..., include_initial=True)` together: anionpy raises
    NotImplementedError with a message naming the reason, where numpy
    succeeds. This is a REAL divergence and it is why the exclusion above
    is not cosmetic.
  - `tri(like=...)`: the `__array_function__` dispatch protocol. anionpy
    cannot construct a foreign array type, so it raises rather than
    silently handing back an anionpy array to a caller who asked for
    something else. numpy would dispatch. Divergent by construction.

The `_NoValue` sentinel dance in `clip` IS exercised, in both directions
(missing-positional -> TypeError, mixed positional-and-keyword ->
ValueError), because those two error types are the only observable
difference between numpy's signature and the obvious `a_min=None` one, and
an implementation that got them wrong would otherwise pass every value
case.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec


# ---------------------------------------------------------------------------
# Shared input pool. Deliberately spans dtype families (bool/int/uint/float/
# complex/float16), ranks 0-3, empties, non-contiguous views, and the
# special float values -- `clip` and the cumulative pair all have distinct
# nan/inf behaviour that a float64-only pool would never reach.
# ---------------------------------------------------------------------------

def _pool():
    return [
        ("f64_1d", np.array([0.0, 1.5, -2.5, 3.25, -0.0])),
        ("f64_special", np.array([np.nan, np.inf, -np.inf, 0.0, 1.0])),
        ("f32_1d", np.array([1.5, -2.5, 3.0], dtype=np.float32)),
        ("f16_1d", np.array([1.5, -2.5, 3.0], dtype=np.float16)),
        ("i64_1d", np.array([-3, -1, 0, 2, 7])),
        ("i8_1d", np.array([-128, 0, 127], dtype=np.int8)),
        ("u8_1d", np.array([0, 1, 255], dtype=np.uint8)),
        ("bool_1d", np.array([True, False, True])),
        ("c128_1d", np.array([1 + 2j, -3 + 0j, 0 - 1j])),
        ("c64_1d", np.array([1 + 2j, -3 + 0j], dtype=np.complex64)),
        ("f64_2d", np.arange(6.0).reshape(2, 3)),
        ("i64_3d", np.arange(24).reshape(2, 3, 4)),
        ("f64_2d_T", np.arange(6.0).reshape(2, 3).T),
        ("f64_strided", np.arange(12.0)[::3]),
        ("f64_0d", np.array(2.5)),
        ("i64_0d", np.array(3)),
        ("f64_empty", np.array([], dtype=np.float64)),
        ("i64_empty_2d", np.zeros((0, 3), dtype=np.int64)),
        ("f64_empty_axis", np.zeros((2, 0, 4))),
        ("list_int", [1, 2, 3]),
        ("list_nested", [[1.0, 2.0], [3.0, 4.0]]),
        ("py_float", 2.5),
        ("py_int", 4),
    ]


def _rank(v):
    return np.asarray(v).ndim


# ---------------------------------------------------------------------------
# permute_dims -- in real numpy this is literally `transpose` (verified:
# `np.permute_dims is np.transpose`), so the cases are the ones that would
# catch a WRAPPER drifting from its target: explicit axes, negative axes,
# wrong-length axes, duplicate axes.
# ---------------------------------------------------------------------------

def _permute_dims_cases():
    out = [(lbl, (v,), {}) for lbl, v in _pool()]
    a3 = np.arange(24).reshape(2, 3, 4)
    a2 = np.arange(6.0).reshape(2, 3)
    for lbl, arr, axes in [
        ("3d/identity", a3, (0, 1, 2)),
        ("3d/reverse", a3, (2, 1, 0)),
        ("3d/rotate", a3, (1, 2, 0)),
        ("3d/negative", a3, (-1, -2, -3)),
        ("3d/mixed_sign", a3, (0, -1, 1)),
        ("2d/swap", a2, (1, 0)),
        ("2d/list_axes", a2, [1, 0]),
        # error paths -- both sides must reject identically
        ("3d/too_few", a3, (0, 1)),
        ("3d/too_many", a3, (0, 1, 2, 0)),
        ("3d/duplicate", a3, (0, 0, 1)),
        ("3d/out_of_range", a3, (0, 1, 3)),
        ("2d/oob_negative", a2, (0, -3)),
    ]:
        out.append((f"axes/{lbl}", (arr, axes), {}))
    return out


# ---------------------------------------------------------------------------
# clip -- the interesting surface is the ARGUMENT protocol, not the values.
# ---------------------------------------------------------------------------

def _clip_cases():
    out = []
    for lbl, v in _pool():
        out.append((f"pos/{lbl}", (v, 0, 2), {}))
        out.append((f"kw/{lbl}", (v,), {"min": 0, "max": 2}))
    a = np.arange(10)
    f = np.array([np.nan, np.inf, -np.inf, 0.0, 5.0])
    out += [
        # one-sided (None means "do not clip this edge")
        ("none_lower", (a, None, 5), {}),
        ("none_upper", (a, 3, None), {}),
        ("none_both", (a, None, None), {}),
        ("kw_none_lower", (a,), {"min": None, "max": 5}),
        ("kw_none_upper", (a,), {"min": 3, "max": None}),
        ("kw_neither", (a,), {}),
        # inverted interval -- numpy documents that a_min > a_max yields
        # a_max everywhere, and performs no check
        ("inverted", (a, 8, 1), {}),
        ("inverted_float", (f, 4.0, -4.0), {}),
        # array bounds, broadcast against the operand
        ("array_min", (a, np.arange(10) - 2, 8), {}),
        ("array_both", (a, np.zeros(10), np.full(10, 4)), {}),
        ("broadcast_2d", (np.arange(6).reshape(2, 3), np.array([1, 2, 3]), 5), {}),
        # nan/inf bounds
        ("nan_bound", (f, np.nan, 1.0), {}),
        ("inf_bound", (f, -np.inf, np.inf), {}),
        # dtype-crossing bounds (the result dtype is the interesting part)
        ("int_operand_float_bound", (a, 1.5, 7.5), {}),
        ("float_operand_int_bound", (f, 0, 1), {}),
        ("bool_operand", (np.array([True, False]), False, True), {}),
        ("complex_bound", (np.array([1 + 1j]), 0, 1), {}),
        # ERROR PATHS -- the reason clip needs a sentinel at all
        ("missing_a_max", (a, 1), {}),
        ("mixed_pos_and_kw_min", (a, 1, 8), {"min": 0}),
        ("mixed_pos_and_kw_max", (a, 1, 8), {"max": 9}),
        ("mixed_pos_and_kw_both", (a, 1, 8), {"min": 0, "max": 9}),
    ]

    # ---- bound-dtype promotion grid -------------------------------------
    # ADDED 2026-08-04 (Monday). The dtype-crossing rows above are four
    # hand-picked pairs, and all four happened to be pairs where the
    # promotion is trivial. `clip` was declared "exact" while carrying TWO
    # independent promotion defects, neither of which those four could see:
    #
    #  (1) The min-only/max-only branches routed float16 operands through
    #      `extreme_f16_tie_first`, which computes AND RETURNS at f16, on a
    #      test of the OPERAND's dtype alone. A wider bound was therefore
    #      narrowed to the operand rather than promoting the output. For a
    #      complex bound that DISCARDS THE IMAGINARY PART -- a wrong value,
    #      not just a wrong dtype:
    #          np.clip(f16([1,0,2]), 1+2j, None)
    #              np  : complex64 [1+2j, 1+2j, 2+0j]
    #              anionpy: float16   [1.,   1.,   2.  ]
    #  (2) `clip_bound`/`scalar_or_array_against` called `scalar_against`
    #      with no strong-operand gate, so a numpy scalar that SUBCLASSES a
    #      Python scalar type (`np.float64` of `float`, `np.int64` of `int`,
    #      `np.bool_` of `bool`) was treated as NEP 50 weak and narrowed:
    #          np.clip(f16([1,0,2]), np.float64(1), None) -> float64 (was f16)
    #          np.clip(c64([1,0,2]), np.float64(1), None) -> complex128
    #      `np.float32`/`np.complex64` are NOT Python-scalar subclasses and
    #      so were never affected -- which is exactly why defect (2) hid
    #      behind the first fix until the grid grew an `np.float64` column.
    #
    # Hence a full cross-product rather than more hand-picked pairs: every
    # operand dtype against every BOUND SPELLING (bare Python scalar / numpy
    # scalar / 0-d-equivalent / 1-d array), which is the axis both defects
    # lived on. Bound spelling is a distinct axis from bound dtype here --
    # `1.0`, `np.float64(1)` and `np.array([1.0])` are three different
    # promotion paths in numpy and must stay three cases.
    _CLIP_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32",
                    "int64", "uint64", "float16", "float32", "float64",
                    "complex64", "complex128"]
    _CLIP_BOUNDS = [
        ("py_int", 1), ("py_float", 1.0), ("py_complex", 1 + 2j),
        ("py_bool", True),
        ("np_bool", np.bool_(1)), ("np_int8", np.int8(1)),
        ("np_int64", np.int64(1)), ("np_uint8", np.uint8(1)),
        ("np_float16", np.float16(1)), ("np_float32", np.float32(1)),
        ("np_float64", np.float64(1)), ("np_complex64", np.complex64(1 + 2j)),
        ("np_complex128", np.complex128(1 + 2j)),
        ("arr_int8", np.array([1, 1, 1], dtype=np.int8)),
        ("arr_float16", np.array([1, 1, 1], dtype=np.float16)),
        ("arr_float32", np.array([1.0, 1, 1], dtype=np.float32)),
        ("arr_float64", np.array([1.0, 1, 1])),
        ("arr_complex64", np.array([1 + 2j] * 3, dtype=np.complex64)),
    ]
    for _dt in _CLIP_DTYPES:
        _operand = np.array([1, 0, 2], dtype=_dt)
        for _blbl, _b in _CLIP_BOUNDS:
            out.append((f"promote/{_dt}/{_blbl}/min", (_operand, _b, None), {}))
            out.append((f"promote/{_dt}/{_blbl}/max", (_operand, None, _b), {}))
            out.append((f"promote/{_dt}/{_blbl}/both", (_operand, _b, _b), {}))

    # ---- signed-zero ties and NaN bounds -------------------------------
    # ADDED 2026-08-03 (Monday) after this very case set was caught being
    # BLIND. `clip` passed everything above while an out-of-corpus probe
    # found 255 divergences against real numpy 2.5.1: NaN bounds were not
    # propagated at all, and the signed-zero tie rule was wrong for
    # complex and for array-valued bounds. A case set that goes green over
    # a defect this size is not a guard, it is decoration -- so the exact
    # axes it was missing are pinned here.
    #
    # The three axes that actually matter, none of which the block above
    # varies: the SIGN of a zero bound, the SPELLING of the bound (Python
    # scalar / 0-d array / 1-d array -- numpy's weak-promotion split makes
    # these three different operations), and the operand dtype (f16,
    # f32/f64 and complex each follow a different tie rule).
    for dt in ("float16", "float32", "float64", "complex64", "complex128"):
        for ai, av in enumerate((0.0, -0.0)):
            operand = np.array([av] * 4, dtype=dt)
            for mi, mn in enumerate((0.0, -0.0)):
                for xi, mx in enumerate((0.0, -0.0, 1.0, -1.0)):
                    for kmn, spell_mn in (
                        ("py", lambda v, d=dt: v),
                        ("0d", lambda v, d=dt: np.array(v, dtype=d)),
                        ("1d", lambda v, d=dt: np.array([v] * 4, dtype=d)),
                    ):
                        for kmx, spell_mx in (
                            ("py", lambda v, d=dt: v),
                            ("0d", lambda v, d=dt: np.array(v, dtype=d)),
                            ("1d", lambda v, d=dt: np.array([v] * 4, dtype=d)),
                        ):
                            out.append((
                                f"tie/{dt}/a{ai}/mn{mi}{kmn}/mx{xi}{kmx}",
                                (operand, spell_mn(mn), spell_mx(mx)),
                                {},
                            ))

    # NaN bounds, both scalar and array-valued, with NaN in the operand
    # too -- a NaN BOUND must propagate elementwise while a NaN OPERAND
    # must survive, and only cases carrying both distinguish the two.
    nan, inf = np.nan, np.inf
    for dt in ("float32", "float64", "complex64", "complex128"):
        op = np.array([2.0, -3.0, nan, 0.0, -0.0, inf, -inf, 1.0], dtype=dt)
        for lbl, mn, mx in (
            ("nan_min", nan, 1.0),
            ("nan_max", -1.0, nan),
            ("nan_both", nan, nan),
            ("nan_min_only", nan, None),
            ("nan_max_only", None, nan),
            ("inf_min", inf, None),
            ("neginf_max", None, -inf),
        ):
            out.append((f"nanb/{dt}/{lbl}", (op, mn, mx), {}))
        arr_mn = np.array([nan, 0.0, nan, -9.0, 1.0, nan, 0.0, -0.0], dtype=dt)
        arr_mx = np.array([1.0, nan, nan, 9.0, 2.0, 0.0, nan, 0.0], dtype=dt)
        out.append((f"nanb/{dt}/arr_min", (op, arr_mn, None), {}))
        out.append((f"nanb/{dt}/arr_max", (op, None, arr_mx), {}))
        out.append((f"nanb/{dt}/arr_both", (op, arr_mn, arr_mx), {}))
        out.append((f"nanb/{dt}/arr_min_scalar_max", (op, arr_mn, 1.0), {}))
        out.append((f"nanb/{dt}/scalar_min_arr_max", (op, -1.0, arr_mx), {}))

    # Integer and bool operands with a NaN bound: numpy promotes the whole
    # operation to float64 rather than erroring, so the RESULT DTYPE is
    # the assertion here, not just the values.
    for dt in ("int8", "int32", "int64", "uint8", "bool"):
        iop = np.array([1, 0, 1, 0], dtype=dt)
        out.append((f"nanb/{dt}/nan_min", (iop, nan, None), {}))
        out.append((f"nanb/{dt}/nan_both", (iop, nan, nan), {}))
        out.append((f"nanb/{dt}/int_bounds", (iop, 0, 1), {}))

    return out


# ---------------------------------------------------------------------------
# cumulative_sum / cumulative_prod -- what separates these from cumsum/
# cumprod is the axis=None rule (an ERROR for ndim>=2, where cumsum
# silently ravels) and include_initial.
# ---------------------------------------------------------------------------

def _cumulative_cases():
    out = []
    for lbl, v in _pool():
        r = _rank(v)
        # axis=None: legal only for 0-d/1-d (atleast_1d makes 0-d rank 1);
        # for rank>=2 BOTH sides must raise, which is the point of keeping
        # these cases rather than filtering them out.
        out.append((f"default/{lbl}", (v,), {}))
        out.append((f"init/{lbl}", (v,), {"include_initial": True}))
        for ax in range(-r, r) if r else [0, -1]:
            out.append((f"axis{ax}/{lbl}", (v,), {"axis": ax}))
            out.append((f"axis{ax}_init/{lbl}", (v,), {"include_initial": True,
                                                       "axis": ax}))
    a1 = np.array([1, 2, 3, 4])
    a2 = np.arange(6).reshape(2, 3)
    fa = np.array([1.0, np.nan, 2.0, np.inf])
    out += [
        ("dtype_f64", (a1,), {"dtype": np.float64}),
        ("dtype_f32", (a1,), {"dtype": np.float32}),
        ("dtype_i8_overflow", (np.array([100, 100, 100], dtype=np.int8),),
         {"dtype": np.int8}),
        ("dtype_c128", (a1,), {"dtype": np.complex128}),
        ("dtype_bool", (np.array([True, True, False]),), {"dtype": np.bool_}),
        ("dtype_init", (a1,), {"dtype": np.float64, "include_initial": True}),
        ("nan_inf", (fa,), {}),
        ("nan_inf_init", (fa,), {"include_initial": True}),
        ("2d_axis0_init", (a2,), {"axis": 0, "include_initial": True}),
        ("2d_axis1_init", (a2,), {"axis": 1, "include_initial": True}),
        ("2d_axis_neg_init", (a2,), {"axis": -2, "include_initial": True}),
        # error paths
        ("2d_axis_none", (a2,), {}),
        ("3d_axis_none", (np.arange(8).reshape(2, 2, 2),), {}),
        ("axis_out_of_range", (a1,), {"axis": 2}),
        ("axis_out_of_range_neg", (a1,), {"axis": -3}),
        ("2d_axis_none_init", (a2,), {"include_initial": True}),
    ]
    return out


# ---------------------------------------------------------------------------
# tri
# ---------------------------------------------------------------------------

def _tri_cases():
    out = []
    for n in (0, 1, 2, 3, 5):
        out.append((f"N{n}", (n,), {}))
        for m in (0, 1, 3, 6):
            out.append((f"N{n}_M{m}", (n, m), {}))
            for k in (-7, -2, -1, 0, 1, 2, 7):
                out.append((f"N{n}_M{m}_k{k}", (n, m, k), {}))
    for dt in (bool, int, float, np.int8, np.uint8, np.int64, np.float32,
               np.float16, np.complex128, np.bool_):
        name = getattr(dt, "__name__", str(dt))
        out.append((f"dtype_{name}", (3, 4, 1), {"dtype": dt}))
        out.append((f"dtype_{name}_str", (3, 4), {"dtype": np.dtype(dt).name}))
    out += [
        ("kw_all", (), {"N": 3, "M": 4, "k": -1, "dtype": np.int32}),
        ("kw_M_only", (4,), {"M": 2}),
        ("kw_k_only", (4,), {"k": 2}),
        # `operator.index` paths: a bool IS an index, a float is not, and
        # numpy 2.5 deprecated (not yet removed) the float case.
        ("bool_N", (True, 3), {}),
        # numpy 2.5 DEPRECATED but still ACCEPTS non-integer N/M/k: it warns
        # and coerces rather than raising. These now PASS.
        #
        # CORRECTION 2026-08-03 (Monday), same day the original note was
        # written. The note that stood here blamed
        # `anionpy.arange(3.0, dtype=int8)` for raising where numpy coerces,
        # and concluded "an arange divergence surfacing through tri, not a
        # tri bug". BOTH HALVES WERE WRONG, and the correction is left
        # visible instead of swept up:
        #   - `anionpy.arange(3.0, dtype=int8)` does not raise. It never did.
        #     Probed directly across 17 float/int/dtype spellings: identical
        #     to numpy on every one.
        #   - It WAS a `tri` bug, and mine: this file's `tri` built its row
        #     vector with `reshape(N, 1)`, which needs an integer N. numpy's
        #     own tri uses `greater_equal.outer` and never needs N as a
        #     shape at all. `reshape(-1, 1)` fixes it.
        # The blocker was filed from the traceback's neighbourhood rather
        # than from a probe, and it cost a day. A blocker you have not
        # reproduced in isolation is a guess wearing a bug's clothes -- the
        # same failure mode already recorded twice in predicate_cases.py and
        # in the `order=` revocation notes.
        ("float_N", (3.0,), {}),
        ("float_N_M", (3.0, 4.0), {}),
        ("float_k", (3, 4, 1.0), {}),
        ("float_N_frac", (3.5,), {}),
        ("npfloat_N", (np.float64(3.0),), {}),
        # NOW PASSING (2026-08-03) -- kept as the regression guard for the
        # two REAL blockers described below, both since fixed in arange
        # (200e003 and the integer-fill commit that follows it). Note that
        # blocker 2's diagnosis below is CORRECT ABOUT THE SYMPTOM AND
        # WRONG ABOUT THE CAUSE, and is left standing for that reason: it
        # reads the divergence as start-sign-specific, "exactly the range
        # where trunc(start) == 0 but start < 0". It is not sign-specific
        # at all. numpy seeds an integer buffer with trunc(start) and
        # trunc(start+step) and fills by their DIFFERENCE, so the
        # effective step is a difference of truncations -- start=-0.5
        # merely happens to be a place where that difference is 0 instead
        # of 1. The evidence cited for the sign hypothesis (start=+0.5 and
        # start=-1.5 "both agree") was real and was consistent with the
        # true rule too; agreement on two neighbours is not a boundary.
        # The general rule fixes cases the sign hypothesis would have
        # left broken, e.g. arange(-1.5, 3.5, 2, dtype=int8), where the
        # effective step is 1 from a nominal step of 2.
        #
        # ORIGINAL NOTE, preserved:
        # STILL FAILING ON PURPOSE -- the two REAL blockers, found by an
        # out-of-corpus sweep (15,120 cases over N x M x k x dtype, 1,782
        # divergences) once the reshape bug was gone. `tri` stays
        # undeclared until both are fixed. Both are `arange` defects for
        # real this time, each reproduced in isolation:
        #
        # 1. A numpy-scalar `k` is rejected outright: anionpy's arange raises
        #    TypeError("expected a bool/int/float scalar") for
        #    `np.float32(1.5)`, which numpy accepts. ~1,600 of the 1,782.
        # 2. Integer-dtype arange with a fractional start in (-1, 0) --
        #    exactly the range where trunc(start) == 0 but start < 0 --
        #    diverges:
        #        np.arange(-0.5, 3.5, dtype=int8) -> [0, 0, 0, 0]
        #      anionpy.arange(-0.5, 3.5, dtype=int8) -> [0, 1, 2, 3]
        #    Verified this is start-sign-specific, not general truncation:
        #    start=+0.5 and start=-1.5 both agree with numpy. numpy's tri
        #    hits it whenever k is a positive fraction, since it builds
        #    columns as `arange(-k, M - k, dtype=_min_int(...))`.
        ("npfloat32_k", (3, 4, np.float32(1.5)), {}),
        ("npfloat32_k_only", (2, None, np.float32(0.5)), {}),
        ("frac_pos_k", (1, 4, 0.5), {}),
        ("frac_pos_k_square", (2, 2, 0.5), {}),
        ("frac_pos_k_tall", (5, 4, 0.5), {}),
        ("numpy_int_N", (np.int64(3), np.int32(4)), {}),
        # error paths
        ("negative_N", (-1,), {}),
        ("negative_M", (3, -2), {}),
        ("str_N", ("3",), {}),
        ("none_k", (3, 3, None), {}),
    ]
    return out


def _build_arrayapi_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}
    specs["permute_dims"] = ItemSpec(
        name="permute_dims", kind="custom", custom_cases=_permute_dims_cases,
        numpy_path="permute_dims", ionp_path="permute_dims",
    )
    specs["clip"] = ItemSpec(
        name="clip", kind="custom", custom_cases=_clip_cases,
        numpy_path="clip", ionp_path="clip",
    )
    specs["cumulative_sum"] = ItemSpec(
        name="cumulative_sum", kind="custom", custom_cases=_cumulative_cases,
        numpy_path="cumulative_sum", ionp_path="cumulative_sum",
    )
    specs["cumulative_prod"] = ItemSpec(
        name="cumulative_prod", kind="custom", custom_cases=_cumulative_cases,
        numpy_path="cumulative_prod", ionp_path="cumulative_prod",
    )
    specs["tri"] = ItemSpec(
        name="tri", kind="custom", custom_cases=_tri_cases,
        numpy_path="tri", ionp_path="tri",
    )
    return specs


ARRAYAPI_SPECS = _build_arrayapi_specs()
