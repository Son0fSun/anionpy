"""Seeded adversarial input corpus for anionpy differential testing.

Deterministic on purpose: one pinned seed, no wall-clock or OS entropy. Every
array here is either a fixed literal (special values, dtype boundaries) or
derived from `np.random.default_rng(SEED)`, so the corpus is byte-identical
across machines and across runs. A "fail" produced against this corpus is
reproducible, not a coin flip -- that is the entire point of pinning it.

This module builds inputs only. It contains no comparisons and no notion of
right/wrong -- that lives in harness.py. It also does no arithmetic on
behalf of anionpy; every value here is either a literal or produced by numpy,
which is the reference implementation, not the thing under test.
"""
from __future__ import annotations

import dataclasses

import numpy as np

# Changing this is a deliberate, reviewed act -- like re-pinning numpy_surface.json.
# It changes what "reproducible failure" means for every item in the registry.
SEED = 20260731


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# dtype catalog
# ---------------------------------------------------------------------------

BOOL_DTYPE = [np.bool_]
INT_DTYPES = [np.int8, np.int16, np.int32, np.int64]
UINT_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
FLOAT_DTYPES = [np.float16, np.float32, np.float64]
COMPLEX_DTYPES = [np.complex64, np.complex128]
ALL_DTYPES = BOOL_DTYPE + INT_DTYPES + UINT_DTYPES + FLOAT_DTYPES + COMPLEX_DTYPES

# A smaller set used for the shape sweep (full ALL_DTYPES x every shape would
# be thousands of arrays; this set already exercises every numeric kind).
SWEEP_DTYPES = [np.bool_, np.int8, np.int32, np.int64, np.uint8, np.uint32,
                np.float16, np.float32, np.float64, np.complex64, np.complex128]

SHAPES = {
    "0d": (),
    "1elem": (1,),
    "empty1d": (0,),
    "1d": (7,),
    "2d": (3, 4),
    "3d": (2, 3, 4),
    "rank4": (2, 2, 3, 2),
    "rank5": (2, 2, 2, 2, 2),
    "empty2d": (0, 5),
    "empty_any_axis": (3, 0, 2),
}


def _fill(rng: np.random.Generator, shape: tuple[int, ...], dtype) -> np.ndarray:
    """Fill an array of `shape`/`dtype` with values appropriate to its kind."""
    dt = np.dtype(dtype)
    n = int(np.prod(shape)) if shape else 1
    if dt.kind == "b":
        data = rng.integers(0, 2, size=n).astype(bool)
    elif dt.kind in "iu":
        info = np.iinfo(dt)
        # keep it away from the exact overflow edge here; edge cases are
        # covered explicitly below so a random hit isn't the only coverage.
        lo = max(info.min, -1_000_000)
        hi = min(info.max, 1_000_000)
        data = rng.integers(lo, hi, size=n, dtype=np.int64 if dt.itemsize < 8 else np.int64)
        data = data.astype(dt)
    elif dt.kind == "f":
        data = rng.standard_normal(size=n).astype(dt) * 10
    elif dt.kind == "c":
        real = rng.standard_normal(size=n)
        imag = rng.standard_normal(size=n)
        data = (real + 1j * imag).astype(dt)
    else:  # pragma: no cover - not reachable with our dtype catalog
        raise ValueError(f"unhandled dtype kind {dt.kind!r}")
    return data.reshape(shape)


@dataclasses.dataclass(frozen=True)
class Case:
    label: str
    value: object


def _shape_sweep() -> list[Case]:
    rng = _rng()
    out = []
    for shape_name, shape in SHAPES.items():
        for dtype in SWEEP_DTYPES:
            arr = _fill(rng, shape, dtype)
            out.append(Case(f"sweep/{shape_name}/{np.dtype(dtype).name}", arr))
    return out


def _special_float_values() -> list[Case]:
    out = []
    specials = [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0]
    for dtype in (np.float16, np.float32, np.float64):
        arr = np.array(specials, dtype=dtype)
        out.append(Case(f"special/finite_nan_inf/{np.dtype(dtype).name}", arr))
        out.append(Case(f"special/all_nan/{np.dtype(dtype).name}",
                         np.full(5, np.nan, dtype=dtype)))
        out.append(Case(f"special/scalar_nan/{np.dtype(dtype).name}",
                         np.array(np.nan, dtype=dtype)))
        out.append(Case(f"special/scalar_neg_zero/{np.dtype(dtype).name}",
                         np.array(-0.0, dtype=dtype)))
    # denormals (subnormals): smallest positive subnormal for each float kind
    for dtype in (np.float16, np.float32, np.float64):
        tiny_sub = np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype))
        arr = np.array([tiny_sub, -tiny_sub, tiny_sub * 2, 0.0], dtype=dtype)
        out.append(Case(f"special/denormal/{np.dtype(dtype).name}", arr))
    # complex specials
    for dtype in (np.complex64, np.complex128):
        arr = np.array([complex(np.nan, 0), complex(0, np.nan),
                         complex(np.inf, -np.inf), complex(-0.0, -0.0)], dtype=dtype)
        out.append(Case(f"special/complex_nan_inf/{np.dtype(dtype).name}", arr))
        # All four sign combinations of a zero complex, spelled via
        # `complex(re, im)` explicitly -- NOT `-0.0+0j` (that expression
        # parses as `(-0.0) + (0+0j)` and silently renormalizes the real
        # part's sign back to +0.0; see this module's other 2026-08-13
        # additions). `complex(-0.0, -0.0)` already existed above (folded
        # into the nan/inf array); the other three combinations did not
        # exist anywhere in this corpus before #86.
        out.append(Case(f"special/complex_signed_zero/{np.dtype(dtype).name}",
                         np.array([complex(0.0, 0.0), complex(-0.0, 0.0),
                                    complex(0.0, -0.0), complex(-0.0, -0.0)],
                                   dtype=dtype)))
        # Complex denormal: both components at the smallest positive
        # subnormal magnitude for this dtype's component width, plus a
        # denormal/zero mix -- unary_corpus's real-float denormal case
        # (above) has no complex counterpart before this ticket.
        component_dt = np.float32 if dtype is np.complex64 else np.float64
        tiny_sub_c = np.nextafter(np.array(0, dtype=component_dt),
                                    np.array(1, dtype=component_dt))
        out.append(Case(f"special/complex_denormal/{np.dtype(dtype).name}",
                         np.array([complex(tiny_sub_c, tiny_sub_c),
                                    complex(-tiny_sub_c, tiny_sub_c),
                                    complex(tiny_sub_c, 0.0)], dtype=dtype)))
    return out


def _integer_boundaries() -> list[Case]:
    out = []
    for dtype in INT_DTYPES:
        info = np.iinfo(dtype)
        arr = np.array([info.min, info.max, info.min + 1, info.max - 1, 0], dtype=dtype)
        out.append(Case(f"boundary/int_extremes/{np.dtype(dtype).name}", arr))
    for dtype in UINT_DTYPES:
        info = np.iinfo(dtype)
        arr = np.array([0, 1, info.max, info.max - 1], dtype=dtype)
        out.append(Case(f"boundary/uint_extremes/{np.dtype(dtype).name}", arr))
    return out


def _views() -> list[Case]:
    """Non-contiguous, negative-stride, transposed, sliced, Fortran-order."""
    out = []
    base2d = np.arange(20, dtype=np.float64).reshape(4, 5)
    base3d = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
    base1d = np.arange(10, dtype=np.float64)

    out.append(Case("view/1d_reverse_negstride", base1d[::-1]))
    out.append(Case("view/1d_stride2", base1d[::2]))
    out.append(Case("view/1d_stride2_negative", base1d[::-2]))
    out.append(Case("view/2d_transpose", base2d.T))
    out.append(Case("view/2d_col_slice_noncontig", base2d[:, 1::2]))
    out.append(Case("view/2d_row_reverse", base2d[::-1, :]))
    out.append(Case("view/2d_fully_reversed", base2d[::-1, ::-1]))
    out.append(Case("view/2d_fortran_order", np.asfortranarray(base2d)))
    out.append(Case("view/3d_transpose_axes", np.transpose(base3d, (2, 0, 1))))
    out.append(Case("view/3d_middle_slice", base3d[:, 1:3, :]))
    out.append(Case("view/2d_diag_view", np.diagonal(base2d[:4, :4])))

    # CONTIGUOUS views that carry a nonzero OFFSET. Added 2026-08-03.
    #
    # Every case above is non-contiguous: reversed, strided, transposed,
    # Fortran-ordered, or sliced on a non-leading axis. Not one of them was
    # a plain leading-axis slice -- the simplest view there is, and the
    # only kind that stays C-contiguous while carrying an offset.
    #
    # That single omission let a real bug live in `astype`, `asarray` and
    # `array` -- all three long declared "exact" -- until `copyto`'s
    # out-of-corpus sweep tripped over it. `NdArray::cast_to` hardcoded
    # `offset: 0`, so `np.arange(9.)[2:5].astype('int8')` returned
    # `[0,1,2]` instead of `[2,3,4]`: right dtype, right shape, wrong
    # elements. The exotic views all PASSED, because the code path they
    # take (`to_contiguous()`) gathers through the offset honestly; only
    # the simple one was broken. A corpus made entirely of hard cases
    # missed a bug that only easy cases could catch.
    #
    # These belong in the SHARED corpus rather than in one item's file
    # because the defect was in shared core machinery -- anything that
    # casts, copies or reads a view routes through it, so every unary item
    # in the suite should be asking this question.
    out.append(Case("view/1d_offset_contig", base1d[3:]))
    out.append(Case("view/1d_offset_contig_bounded", base1d[3:7]))
    out.append(Case("view/1d_offset_contig_tail", base1d[9:]))
    out.append(Case("view/2d_offset_contig_rows", base2d[2:]))
    out.append(Case("view/2d_offset_contig_one_row", base2d[3:4]))
    out.append(Case("view/3d_offset_contig_planes", base3d[1:]))
    return out


def _random_extra() -> list[Case]:
    """A handful of larger seeded-random arrays per dtype, for volume."""
    rng = _rng()
    out = []
    for dtype in ALL_DTYPES:
        arr = _fill(rng, (6, 5), dtype)
        out.append(Case(f"random/6x5/{np.dtype(dtype).name}", arr))
    return out


def unary_corpus() -> list[Case]:
    """Every input a unary (single-array) API item must be run against."""
    cases: list[Case] = []
    cases += _shape_sweep()
    cases += _special_float_values()
    cases += _integer_boundaries()
    cases += _views()
    cases += _random_extra()
    return cases


# ---------------------------------------------------------------------------
# binary / broadcasting corpus
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Pair:
    label: str
    a: object
    b: object


def binary_corpus() -> list[Pair]:
    rng = _rng()
    out: list[Pair] = []

    # same shape, same dtype
    for dtype in SWEEP_DTYPES:
        a = _fill(rng, (3, 4), dtype)
        b = _fill(rng, (3, 4), dtype)
        out.append(Pair(f"same_shape/{np.dtype(dtype).name}", a, b))

    # broadcastable pairs
    out.append(Pair("broadcast/col_row", _fill(rng, (3, 1), np.float64),
                     _fill(rng, (1, 4), np.float64)))
    out.append(Pair("broadcast/scalar_array", np.float64(2.5),
                     _fill(rng, (3, 4), np.float64)))
    out.append(Pair("broadcast/0d_array", np.array(3, dtype=np.int32),
                     _fill(rng, (2, 3), np.int32)))
    out.append(Pair("broadcast/trailing_dims", _fill(rng, (5,), np.float64),
                     _fill(rng, (3, 5), np.float64)))
    out.append(Pair("broadcast/empty_with_scalar", np.zeros((0, 3)),
                     np.float64(4.0)))

    # incompatible shapes -- numpy MUST raise ValueError; anionpy must match
    out.append(Pair("broadcast_fail/mismatched_2d", _fill(rng, (3, 4), np.float64),
                     _fill(rng, (2, 5), np.float64)))
    out.append(Pair("broadcast_fail/1d_mismatch", _fill(rng, (3,), np.float64),
                     _fill(rng, (4,), np.float64)))
    out.append(Pair("broadcast_fail/3d_mismatch", _fill(rng, (2, 3, 4), np.float64),
                     _fill(rng, (2, 3, 5), np.float64)))

    # mixed dtypes (promotion rules)
    out.append(Pair("mixed_dtype/int32_float64", _fill(rng, (3, 3), np.int32),
                     _fill(rng, (3, 3), np.float64)))
    out.append(Pair("mixed_dtype/bool_int8", _fill(rng, (4,), np.bool_),
                     _fill(rng, (4,), np.int8)))
    out.append(Pair("mixed_dtype/float32_complex128", _fill(rng, (3,), np.float32),
                     _fill(rng, (3,), np.complex128)))
    out.append(Pair("mixed_dtype/uint8_int8", _fill(rng, (4,), np.uint8),
                     _fill(rng, (4,), np.int8)))

    # integer overflow / wraparound on fixed-width types
    out.append(Pair("overflow/int8_add_wraps",
                     np.array([np.iinfo(np.int8).max, np.iinfo(np.int8).max - 1], dtype=np.int8),
                     np.array([1, 1], dtype=np.int8)))
    out.append(Pair("overflow/uint8_sub_wraps",
                     np.array([0, 1], dtype=np.uint8),
                     np.array([1, 1], dtype=np.uint8)))
    out.append(Pair("overflow/int64_near_max",
                     np.array([np.iinfo(np.int64).max, 0], dtype=np.int64),
                     np.array([1, 0], dtype=np.int64)))

    # non-contiguous vs contiguous
    base = np.arange(20, dtype=np.float64).reshape(4, 5)
    out.append(Pair("stride_mismatch/contig_vs_view", base, base[:, ::-1]))
    out.append(Pair("stride_mismatch/negstride_vs_negstride", base[::-1], base[::-1]))

    # NaN/inf interplay
    out.append(Pair("special/nan_vs_finite",
                     np.array([np.nan, 1.0, np.inf]),
                     np.array([1.0, np.nan, -np.inf])))

    # Denormal (subnormal) magnitudes and explicit signed zeros -- #86.
    #
    # Every array built above this point comes from either `_fill` (which
    # draws from `rng.standard_normal() * 10`, a distribution that never
    # lands anywhere near 5e-324 or produces an exact `-0.0`) or a small
    # fixed literal with no zero/subnormal member at all. Measured directly
    # (probe, 2026-08-13): `min(|x| for x in binary_corpus() if x != 0)`
    # was `0.02` before this addition -- i.e. every binary/broadcasting
    # item's pairwise corpus was ENTIRELY blind to denormal magnitudes and
    # to signed zero, even though `unary_corpus()`'s `_special_float_values`
    # already covers both for single-array items. A binary op (add/sub/mul/
    # compare/...) run element-by-element against a denormal or a signed
    # zero on ONE side and a normal value on the other is a genuinely
    # different code path from either side being tested alone -- so the gap
    # was not merely redundant with the unary corpus, it was untested
    # entirely for this whole item family.
    for dtype in (np.float16, np.float32, np.float64):
        tiny_sub = np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype))
        denorm_a = np.array([tiny_sub, -tiny_sub, tiny_sub * 2, 0.0, -0.0], dtype=dtype)
        denorm_b = np.array([-0.0, tiny_sub, 0.0, tiny_sub, -tiny_sub], dtype=dtype)
        out.append(Pair(f"special/denormal_pair/{np.dtype(dtype).name}", denorm_a, denorm_b))
        # denormal/signed-zero on one side only, ordinary values on the other
        normal_side = _fill(rng, (5,), dtype)
        out.append(Pair(f"special/denormal_vs_normal/{np.dtype(dtype).name}",
                         denorm_a, normal_side))
    for dtype in (np.complex64, np.complex128):
        # complex(-0.0, 0.0) spelled explicitly -- NOT `-0.0+0j`, which
        # Python parses as `(-0.0) + (0+0j)` and silently normalizes the
        # real part's sign back to +0.0 (see docs/ write-up for this
        # ticket, and the module docstring above for the general warning).
        comp_a = np.array([complex(-0.0, 0.0), complex(0.0, -0.0),
                            complex(-0.0, -0.0)], dtype=dtype)
        comp_b = np.array([complex(0.0, -0.0), complex(-0.0, 0.0),
                            complex(0.0, 0.0)], dtype=dtype)
        out.append(Pair(f"special/complex_signed_zero_pair/{np.dtype(dtype).name}",
                         comp_a, comp_b))

    return out


def binary_corpus_as_tuples() -> list[tuple[str, object, object]]:
    return [(p.label, p.a, p.b) for p in binary_corpus()]


# ---------------------------------------------------------------------------
# scalar right-hand operands -- for binary-op *call-form* coverage.
#
# binary_corpus() above only ever pairs an array with another array. That is
# exactly the gap that let `ndarray.__add__` pass while `a + 1` raised
# TypeError: real numpy accepts a python int/float/bool/complex, a numpy
# scalar, and a 0-d array on the right-hand side of every arithmetic dunder,
# and each of those is a genuinely different code path (Python-object
# unboxing vs numpy-scalar vs 0-d-array-broadcast). This corpus exists so
# run.py's kind="binary_op" case builder can pair every array in
# unary_corpus() against every one of these, in addition to the
# array-vs-array pairs already covered by binary_corpus().
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# input FORM axis -- 2026-08-02, CLASS A (rejects-input-form-numpy-accepts)
# closure task.
#
# Every case above builds a real numpy.ndarray and hands it straight to the
# item under test. That is deliberate for VALUE coverage (dtype/shape/
# stride/special-value edges), but it means the corpus never actually
# exercises the FORM an argument arrives in: real numpy accepts a Python
# list, a nested list, a tuple, a `range`, or a bare Python scalar anywhere
# an array-like is accepted, not just a real ndarray. `extract_array_like`/
# `array_impl` (ionp-py/src/lib.rs) were widened to accept exactly those
# forms; this is the corpus-side half of that fix -- without it, every case
# below still starts life as an ndarray and the coercion path stays
# untested no matter how correct the Rust side is.
#
# `array_like_forms(value)` is the crossed part: given an ndarray VALUE
# already produced by one of the functions above, it hands back that same
# logical value re-expressed as a nested Python list and a nested Python
# tuple (via `.tolist()`, which also changes the numpy-inferred dtype
# exactly the way real numpy's own list/tuple ingestion does -- e.g. a
# float16 array's tolist() is a list of Python floats, which numpy infers
# as float64, not float16 -- so this genuinely exercises dtype inference,
# not just "does it raise"). "ndarray" (the value unchanged) is included
# as one member of the returned list, per this task's explicit requirement
# that ABSENT/ndarray remain one value of the FORM axis, not the only one.
#
# `extra_array_like_forms()` covers the forms that do NOT generalize to an
# arbitrary corpus value -- `range` (only meaningful for a 1-D arithmetic
# integer progression) and a bare Python scalar/nested-bool-list (only
# meaningful standalone, not derived from an arbitrary shape/dtype) -- as a
# small, fixed, dedicated set, mirroring how `scalar_operands()` above
# already covers the equivalent call-form gap for `binary_op`'s RIGHT-hand
# operand. Consumers cross this into their EXISTING case loops (see run.py,
# ufunc_cases.py, reduction_cases.py, setops_cases.py, stats_cases.py,
# manip_cases.py) rather than being a standalone sweep of its own.
# ---------------------------------------------------------------------------


def _to_nested_tuple(x):
    if isinstance(x, list):
        return tuple(_to_nested_tuple(v) for v in x)
    return x


def array_like_forms(value: np.ndarray) -> list[tuple[str, object]]:
    """(form_label, value) pairs for the FORM axis, derived from a single
    ndarray `value`. Always includes ("form_ndarray", value) unchanged, so
    ndarray remains one value of the axis, not the only one. Also includes
    ("form_list", ...) and ("form_tuple", ...) via `.tolist()`, when that
    succeeds (it always does for numpy's own numeric/bool dtypes, which is
    every dtype this corpus ever produces)."""
    out: list[tuple[str, object]] = [("form_ndarray", value)]
    try:
        lst = value.tolist()
    except Exception:
        return out
    out.append(("form_list", lst))
    out.append(("form_tuple", _to_nested_tuple(lst)))
    return out


def extra_array_like_forms() -> list[Case]:
    """A small, fixed set of FORM-axis values that don't generalize to an
    arbitrary corpus ndarray: `range`, bare Python int/float/bool, and a
    couple of literal nested list/bool forms -- deliberately mirroring
    `classify.py`'s own FORMS dict (list1d/list2d/tuple/pyint/pyfloat/bool/
    float/range), so the corpus now covers, crossed into every consumer
    that calls this, exactly the forms that instrument was already
    checking against `anionpy.__ion_state__`-declared items."""
    return [
        Case("form_range", range(6)),
        Case("form_pyint", 7),
        Case("form_pyint_neg", -3),
        Case("form_pyfloat", 3.5),
        Case("form_pybool", True),
        Case("form_list1d_int", [1, 2, 3, 4]),
        Case("form_list1d_float", [1.5, 2.5, -3.0]),
        Case("form_list1d_bool", [True, False, True]),
        Case("form_list2d", [[1, 2, 3], [4, 5, 6]]),
        Case("form_tuple_nested", ((1, 2), (3, 4))),
        Case("form_empty_list", []),
        Case("form_large_int_mixed",
             [4611686018427387904, 4611686018427387905, 3]),  # 2**62, 2**62+1, 3
    ]


def form_axis_cases(pool: list[Case] | None = None, sample_stride: int = 1,
                     include_extra: bool = True) -> list[Case]:
    """The FORM axis crossed with `pool` (default `unary_corpus()`):
    every `sample_stride`-th base case gets its non-ndarray forms
    (form_list/form_tuple) appended as additional Cases, plus
    `extra_array_like_forms()` once. `sample_stride` bounds the combinatorial
    cost on large pools (e.g. `unary_corpus()`'s ~190 cases) the same way
    `ufunc_corpus.sampled()` already does elsewhere in this suite -- a
    stride of 1 means every base case is crossed, matching this task's
    "crossed axis, not a one-off sweep" requirement; a larger stride still
    crosses the SAME axis against a deterministic, representative subset of
    the existing dtype/shape/view axes rather than dropping the crossing
    entirely.
    """
    base = pool if pool is not None else unary_corpus()
    out: list[Case] = []
    for i, c in enumerate(base):
        if i % sample_stride != 0:
            continue
        for label, val in array_like_forms(c.value):
            if label == "form_ndarray":
                continue  # already covered by the caller's own baseline loop
            out.append(Case(f"{c.label}/{label}", val))
    if include_extra:
        out += extra_array_like_forms()
    return out


def scalar_operands() -> list[tuple[str, object]]:
    return [
        ("python_int_pos", 3),
        ("python_int_neg", -7),
        ("python_int_zero", 0),
        ("python_float_pos", 2.5),
        ("python_float_neg", -1.25),
        ("python_bool_true", True),
        ("python_bool_false", False),
        ("python_complex", 1.5 + 2.5j),
        ("numpy_int64_scalar", np.int64(5)),
        ("numpy_int8_scalar", np.int8(3)),
        ("numpy_float64_scalar", np.float64(3.5)),
        ("numpy_float32_scalar", np.float32(1.5)),
        ("numpy_bool_scalar", np.bool_(True)),
        ("numpy_complex128_scalar", np.complex128(1 + 1j)),
        ("zero_d_array_float", np.array(4.0)),
        ("zero_d_array_int", np.array(4, dtype=np.int32)),
        # Signed zero and denormal RHS operands -- #86. None of the entries
        # above ever pass a `-0.0`, a bare denormal, or a genuinely signed-
        # zero complex as the scalar operand of a binary-op call form (`a +
        # <this>`), so that whole call-form axis was as blind to both
        # properties as `binary_corpus()` was before this ticket's other
        # addition, for exactly the same reason: nothing in the existing
        # set of values ever takes on either property.
        ("python_float_neg_zero", -0.0),
        ("python_float_pos_zero", 0.0),
        ("python_complex_neg_zero", complex(-0.0, 0.0)),
        ("numpy_float64_neg_zero", np.float64(-0.0)),
        ("numpy_float32_neg_zero", np.float32(-0.0)),
        ("numpy_complex128_signed_zero", np.complex128(complex(-0.0, -0.0))),
        ("numpy_float64_denormal", np.nextafter(np.float64(0.0), np.float64(1.0))),
        ("numpy_float32_denormal", np.nextafter(np.float32(0.0), np.float32(1.0))),
        ("zero_d_array_neg_zero", np.array(-0.0)),
    ]
