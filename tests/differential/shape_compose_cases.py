"""Differential coverage for `anionpy/_shape_compose.py`: `kron`.

Same collision-checked-merge pattern as typecheck_cases.py -- builds a dict
of ItemSpecs, merged into registry.REGISTRY from the bottom of registry.py.
`kron` is a Python-only transcription of numpy's OWN Python implementation
(`numpy/lib/_shape_base_impl.py`) over already-existing anionpy primitives
(`asanyarray`, `array`, `reshape`, `expand_dims`, `multiply`) -- no Rust
changes, no accumulation of any kind.

kind="custom" (not "binary"): the generic binary corpus broadcasts its two
operands against each other, which is exactly what `kron` does NOT do. Every
axis that matters here -- the rank ASYMMETRY between the two arguments, the
0-d early return, the alternating expand_dims interleave, the final reshape
-- is invisible to a corpus of same-shape pairs.

WHAT THE CORPUS DELIBERATELY TARGETS
------------------------------------
1. **Rank asymmetry, both orders.** `a` is promoted to at least `b`'s rank
   (`ndmin=b.ndim`); `b` is never promoted to `a`'s. Every rank pair 0..3
   appears in BOTH orders, so a symmetric-promotion mutant cannot hide in
   the half of the space where the two happen to agree.
2. **The interleave.** `a` gets the ODD inserted axes and `b` the EVEN ones.
   Swapping them still produces an array of the right shape and dtype
   containing exactly the same multiset of products -- only in the wrong
   ORDER. Non-square, non-symmetric operands with distinguishable values
   are therefore mandatory; a 2x2 of ones would grade the swap as a pass.
   `_ramp()` builds operands whose every element is distinct.
3. **The 0-d early return.** `nda == 0 or ndb == 0` short-circuits to a
   plain `multiply` BEFORE any shape work. Both `kron(0d, nd)` and
   `kron(nd, 0d)` appear, plus `kron(0d, 0d)`, plus the Python-scalar forms
   that only become 0-d after ingestion.
4. **Empty axes.** A zero-length axis in either operand, at every position,
   including `(0,)` against `(0,)`. The output is empty but its SHAPE is
   still the elementwise product and still observable.
5. **Layout.** Fortran-order, reversed-stride and sliced inputs, in both
   positions and in every combination. This is the only axis that touches
   the `flags.contiguous` reshape.
6. **Value fidelity.** Non-finite grids (7x7 outer of nan/inf/-inf/+-0.0)
   per inexact dtype, and integer iinfo extremes per integer dtype, where
   the products overflow. Values are compared byte-for-byte; anionpy's missing
   overflow RuntimeWarning is a pre-existing, already-disclosed global gap
   and is not re-litigated here.
7. **dtype promotion**, all 14x14 ionp-constructible pairs, at four rank
   combinations including the 0-d early-return branch (whose promotion goes
   through a different call than the main branch's).

DELIBERATE EXCLUSIONS (recorded, not hidden)
--------------------------------------------
* `np.matrix` input. numpy's `is_any_mat` branch returns a `matrix`; anionpy
  has no ndarray subclass machinery at all, so the branch is dead code, not
  a divergence to test.
* object- and string-dtype input: the pre-existing, disclosed ingestion gap
  (`anionpy.array(['x'])` raises TypeError).
* The overflow RuntimeWarning itself (see 6).
"""
from __future__ import annotations

import itertools

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec

_DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
           "uint32", "uint64", "float16", "float32", "float64",
           "complex64", "complex128"]


def _ramp(shape, dtype, start=1):
    """An array whose every element is distinct, so an interleave swap or a
    transposed block layout cannot come out byte-identical."""
    n = int(np.prod(shape)) if shape else 1
    if dtype == "bool":
        # bool cannot hold distinct values; use an asymmetric pattern
        # instead so at least the ORDER is still observable.
        vals = np.array([(i % 3) != 0 for i in range(n)], dtype=bool)
    elif dtype.startswith("complex"):
        vals = (np.arange(start, start + n) + 1j * np.arange(n, 0, -1))
    else:
        vals = np.arange(start, start + n)
    return np.asarray(vals).astype(dtype).reshape(shape)


def kron_cases():
    out = []
    rng = np.random.default_rng(20260803)

    # --- 1. rank asymmetry, every pair 0..3, BOTH orders --------------------
    rank_shapes = {0: (), 1: (3,), 2: (2, 3), 3: (2, 3, 2)}
    for ra, rb in itertools.product(range(4), range(4)):
        a = _ramp(rank_shapes[ra], "float64")
        b = _ramp(rank_shapes[rb], "float64", start=101)
        out.append((f"rank/{ra}x{rb}", (a, b), {}))

    # --- 2. the interleave: non-square, all-distinct operands --------------
    for sa, sb in (((2, 3), (3, 2)), ((3, 2), (2, 3)), ((1, 4), (4, 1)),
                   ((4, 1), (1, 4)), ((2, 3), (2, 2)), ((2, 2, 3), (3, 1, 2)),
                   ((1, 2, 3), (3, 2, 1)), ((5,), (2,)), ((2,), (5,))):
        a = _ramp(sa, "float64")
        b = _ramp(sb, "float64", start=101)
        out.append((f"interleave/{sa}x{sb}", (a, b), {}))

    # --- 3. the 0-d early return -------------------------------------------
    z = np.array(7.0)
    for sb in ((), (3,), (2, 3), (2, 2, 2), (0,)):
        b = _ramp(sb, "float64", start=101)
        out.append((f"zerod_left/{sb}", (z, b), {}))
        out.append((f"zerod_right/{sb}", (b, z), {}))
    out.append(("zerod_both", (np.array(3.0), np.array(5.0)), {}))
    out.append(("pyscalar_both", (3.0, 5.0), {}))
    out.append(("pyscalar_left", (3.0, _ramp((2, 3), "float64")), {}))
    out.append(("pyscalar_right", (_ramp((2, 3), "float64"), 3.0), {}))
    out.append(("pybool_left", (True, _ramp((2, 2), "int64")), {}))
    out.append(("pycomplex_right", (_ramp((2, 2), "float64"), 2 + 1j), {}))

    # --- 4. empty axes ------------------------------------------------------
    empties = [(0,), (0, 3), (3, 0), (0, 0), (1, 0), (0, 1), (2, 0, 3)]
    for sa in empties:
        a = _ramp(sa, "float64")
        for sb in ((2,), (2, 2), (), (0,)):
            b = _ramp(sb, "float64", start=101)
            out.append((f"empty_left/{sa}x{sb}", (a, b), {}))
            out.append((f"empty_right/{sb}x{sa}", (b, a), {}))

    # --- 5. layout: the flags.contiguous branch ----------------------------
    base = _ramp((4, 5), "float64")
    variants = {
        "c": base,
        "f": np.asfortranarray(base),
        "rev_cols": base[:, ::-1],
        "rev_rows": base[::-1],
        "strided": base[::2, ::2],
        "transposed": base.T,
    }
    for na, va in variants.items():
        for nb, vb in variants.items():
            out.append((f"layout/{na}x{nb}", (va, vb), {}))

    # --- 6a. non-finite grids ----------------------------------------------
    sp = [0.0, -0.0, np.nan, np.inf, -np.inf, 1.0, -1.0]
    for dt in ("float16", "float32", "float64", "complex64", "complex128"):
        a = np.array(sp, dtype=dt).reshape(7, 1)
        b = np.array(sp, dtype=dt).reshape(1, 7)
        out.append((f"nonfinite/{dt}", (a, b), {}))
        out.append((f"nonfinite_rev/{dt}", (b, a), {}))
        out.append((f"nonfinite_0d/{dt}", (np.array(np.nan, dtype=dt), a), {}))

    # --- 6b. integer extremes (products overflow; values still compared) ----
    for dt in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
               "uint64"):
        info = np.iinfo(dt)
        vals = np.array([info.min, -1 if info.min < 0 else 0, 0, 1, 2,
                         info.max], dtype=dt)
        out.append((f"extremes/{dt}", (vals.reshape(6, 1),
                                       vals.reshape(1, 6)), {}))
        out.append((f"extremes_1d/{dt}", (vals, vals), {}))

    # --- 7. dtype promotion, 14x14, four rank combinations -----------------
    for da, db in itertools.product(_DTYPES, _DTYPES):
        for sa, sb in (((2, 3), (2, 2)), ((3,), (2,)), ((), (2, 2)),
                       ((2, 2), ())):
            a = _ramp(sa, da)
            b = _ramp(sb, db, start=2)
            out.append((f"dtype/{da}x{db}/{sa}x{sb}", (a, b), {}))

    # --- 8. list / nested-list ingestion ------------------------------------
    out.append(("list_1d", ([1, 2, 3], [4, 5]), {}))
    out.append(("list_2d", ([[1, 2], [3, 4]], [[5, 6], [7, 8]]), {}))
    out.append(("list_mixed_rank", ([1, 2], [[3, 4], [5, 6]]), {}))
    out.append(("list_mixed_rank_rev", ([[3, 4], [5, 6]], [1, 2]), {}))
    out.append(("list_float", ([1.5, 2.5], [0.5]), {}))
    out.append(("list_bool", ([True, False], [True, True]), {}))
    out.append(("list_empty", ([], [1, 2]), {}))
    out.append(("list_empty_rev", ([1, 2], []), {}))
    out.append(("tuple_1d", ((1, 2, 3), (4, 5)), {}))

    # --- 9. random shapes/dtypes -------------------------------------------
    for trial in range(24):
        ra = int(rng.integers(1, 4))
        rb = int(rng.integers(1, 4))
        sa = tuple(int(x) for x in rng.integers(1, 4, size=ra))
        sb = tuple(int(x) for x in rng.integers(1, 4, size=rb))
        dt = _DTYPES[int(rng.integers(0, len(_DTYPES)))]
        out.append((f"random/{trial}/{dt}/{sa}x{sb}",
                    (_ramp(sa, dt), _ramp(sb, dt, start=3)), {}))

    return out


def _build_shape_compose_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}
    specs["kron"] = ItemSpec(name="kron", kind="custom",
                             custom_cases=kron_cases)
    return specs


SHAPE_COMPOSE_SPECS = _build_shape_compose_specs()


# ---------------------------------------------------------------------------
# binary_repr / base_repr -- added 2026-08-03 (Monday). Implemented in
# anionpy/_intrepr_compose.py as transcriptions of numpy's OWN Python bodies
# (numpy/_core/numeric.py). Filed here rather than in a fifth cases module
# because the shape_compose_cases <-> SHAPE_COMPOSE_SPECS <-> registry merge
# is already wired; the docstring above describes `kron` only.
#
# kind="custom" for the obvious reason: neither takes an array operand at
# all in its documented signature. Both take a Python integer.
#
# WHAT THE CORPUS TARGETS
# 1. `binary_repr`'s zero branch. `'0' * (width or 1)` -- and it never calls
#    `err_if_insufficient`. So `binary_repr(0, width=0)` is `'0'` while
#    `binary_repr(1, width=0)` RAISES. Both are pinned; an implementation
#    that hoists the width check above the zero branch fails only on the
#    first, and only if width 0 is in the corpus.
# 2. The gh-8679 boundary. For negative input with a width, the
#    two's-complement width is decremented when `-num` is an exact power of
#    two. Every `+-2**k` and `+-(2**k + 1)` for k up to 8, plus 16/31/32/63/
#    64/200, is present specifically so the off-by-one has somewhere to
#    show.
# 3. The width seam. For each number, the widths swept span "far too small",
#    "exactly binwidth - 1", "exactly binwidth", "binwidth + 1" and "far too
#    large" -- the insufficient-width ValueError lives at that seam, and its
#    message embeds BOTH width and binwidth (f-string `{width=}` form), so a
#    transcription that computes the right string but reports the wrong
#    binwidth still fails.
# 4. Bignums. 2**200 and 2**64 are included in both functions. They are the
#    reason neither may be moved to a fixed-width Rust core, and they are
#    where an implementation that extracts to i64/u64 breaks.
# 5. `base_repr`'s abs/sign split. `abs(int(number))` for the magnitude but
#    `number < 0` for the sign, so a fractional negative gives a sign with
#    no digits: `base_repr(-0.5) == '-'`, and `res or '0'` does NOT rescue
#    it because `['-']` is truthy. Pinned with and without padding.
# 6. Both functions' inherited error surfaces: invalid bases (0, 1, -1, 37,
#    100), non-integer bases (which pass validation and fail later inside
#    `digits[num % base]`), negative padding, and every input type
#    `operator.index`/`int()` accepts or refuses.
#
# ONE MEASURED DIVERGENCE, EXCLUDED HERE AND DISCLOSED IN THE DECLARATION:
# `binary_repr(-5, width=<1-D array>)`. numpy reaches `'1' * outwidth` with
# `outwidth` an array and raises TypeError from its own string-multiply
# ufunc; anionpy raises TypeError from its arithmetic, which does not accept a
# `str` operand. Same exception TYPE, different message, and the difference
# is the `str`/`list` sibling of the already-recorded
# `ndarray.__rmul__`-with-a-sequence-operand gap -- not a `binary_repr`
# defect. `width` is documented `int`; a 1-element ARRAY width is
# out-of-contract input. It is excluded from the corpus rather than wrapped
# in a tolerance, and recorded in the declaration and KNOWN-DIFFERENCES.md
# rather than dropped. 0-d array widths are NOT excluded wholesale: they
# agree with numpy everywhere except the same zero branch, which is the only
# place the width array is itself an operand of a `str` multiply. The exact
# rule, and the row-by-row measurement behind it, is on the width-type loop
# below.
# ---------------------------------------------------------------------------

_BR_NUMS = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9,
            15, -15, 16, -16, 17, -17, 31, -31, 32, -32, 33, -33,
            127, -127, 128, -128, 129, -129, 255, -255, 256, -256,
            2**16, -(2**16), 2**31, -(2**31), 2**32, -(2**32),
            2**63, -(2**63), 2**64 - 1, -(2**64),
            2**200, -(2**200), 2**200 + 1, -(2**200 + 1)]
_BR_WIDTHS = [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33,
              63, 64, 65, 199, 200, 201, 202, 300]


def binary_repr_cases():
    out = []
    for n in _BR_NUMS:
        out.append((f"default/{n}", (n,), {}))
        for w in _BR_WIDTHS:
            out.append((f"pos/{n}/{w}", (n, w), {}))
            out.append((f"kw/{n}/{w}", (n,), {"width": w}))

    # input types accepted or refused by operator.index
    for n in (0, 1, -1, 5, -5, 127, -128, 255):
        for dt in ("int8", "int16", "int32", "int64",
                   "uint8", "uint16", "uint32", "uint64"):
            info = np.iinfo(dt)
            if info.min <= n <= info.max:
                out.append((f"type/scalar/{dt}/{n}",
                            (np.dtype(dt).type(n),), {}))
                out.append((f"type/arr0d/{dt}/{n}",
                            (np.array(n, dtype=dt),), {}))
                out.append((f"type/arr1d/{dt}/{n}",
                            (np.array([n], dtype=dt),), {}))
        out.append((f"type/bool/{n}", (bool(n % 2),), {}))
        out.append((f"type/float/{n}", (float(n),), {}))
        out.append((f"type/npf64/{n}", (np.float64(n),), {}))
        out.append((f"type/arrf64/{n}", (np.array(float(n)),), {}))
        out.append((f"type/complex/{n}", (complex(n),), {}))
        out.append((f"type/str/{n}", (str(n),), {}))
        out.append((f"type/none/{n}", (None,), {}))
        out.append((f"type/list/{n}", ([n],), {}))

    # width TYPES. The array-width class was measured row by row before it
    # was written down, and it is NOT the blanket divergence it first looked
    # like: for `width=np.array(8)` every one of num in
    # {1, -1, 5, -5, -4, 255} agrees exactly, including the two's-complement
    # branch, and `width=np.array(2)` reproduces the insufficient-width
    # ValueError on both sides. Only ONE branch diverges -- the zero branch,
    # `'0' * (width or 1)`, which is the single place where the width array
    # is itself an operand of a `str` multiply:
    #   numpy: str.__mul__ defers, ndarray.__rmul__ runs the string-multiply
    #          ufunc and raises "The 'out' kwarg is necessary ..." (TypeError
    #          for an int array, UFuncTypeError for a float one);
    #   anionpy:  ndarray.__rmul__ rejects a `str` operand outright.
    # Same failure of the same missing feature, and it is the `str` sibling
    # of the `[0] * arr` gap recorded in 4e4b8a5 -- an absence in
    # `ndarray.__rmul__`, not a defect in `binary_repr`. Matching numpy's
    # text would mean forging numpy's string-multiply ufunc internals, which
    # is forbidden. So `n == 0` is skipped for array widths ONLY -- the
    # narrowest exclusion that covers the actual divergence, stated as a
    # rule about which branch is reached rather than as a list of rows that
    # happened to fail. Every other num keeps its array-width case.
    # np.array([8]), the 1-D form, is absent for the same reason at every
    # num: `width` is documented `int`, so a 1-element array is
    # out-of-contract input that reaches the same `__rmul__` hole.
    for label, w in (("int", 8), ("zero", 0), ("bool_true", True),
                     ("bool_false", False), ("npint64", np.int64(8)),
                     ("npfloat64", np.float64(8.0)), ("pyfloat", 8.0),
                     ("str", "8"), ("none", None), ("list", [8]),
                     ("arr0d_int", np.array(8)),
                     ("arr0d_float", np.array(8.0)),
                     ("arr0d_int2", np.array(2)),
                     ("arr0d_int64", np.array(8, dtype="int64")),
                     ("arr0d_uint8", np.array(8, dtype="uint8"))):
        _is_arr = isinstance(w, np.ndarray)
        for n in (-5, 5, 0, 1, -1, -4, 255, -255, 2**200):
            if _is_arr and n == 0:
                continue
            out.append((f"wtype/{label}/{n}", (n, w), {}))
    return out


_BASE_NUMS = [0, 1, -1, 2, -2, 7, -7, 10, -10, 35, -35, 36, -36,
              255, -255, 256, 2**32, -(2**32), 2**64, -(2**64),
              2**200, -(2**200)]
_BASES = [2, 3, 4, 5, 7, 8, 10, 16, 26, 35, 36, 37, 1, 0, -1, -2, 100]
_PADS = [0, 1, 2, 3, 7, -1, -5]


def base_repr_cases():
    out = []
    for n in _BASE_NUMS:
        out.append((f"default/{n}", (n,), {}))
        out.append((f"kw/{n}", (n,), {"base": 16, "padding": 3}))
        for b in _BASES:
            out.append((f"base/{n}/{b}", (n, b), {}))
        for p in _PADS:
            out.append((f"pad/{n}/{p}", (n, 2, p), {}))
    # the abs/sign split: magnitude truncates to 0 but the sign test still
    # fires, so the result is a bare '-' with no digits.
    for v in (0.0, -0.0, 0.5, -0.5, 0.9, -0.9, 1.5, -1.5, 2.7, -2.7,
              1e-9, -1e-9, np.float64(-0.5), np.float32(-0.5),
              np.float16(-0.5)):
        for p in (0, 3):
            out.append((f"frac/{v!r}/{p}", (v, 2, p), {}))
    for label, v in (("arr0d_negfrac", np.array(-0.5)),
                     ("arr0d_posfrac", np.array(0.5)),
                     ("arr0d_negint", np.array(-3)),
                     ("arr0d_posint", np.array(3)),
                     ("arr0d_bool", np.array(True)),
                     ("arr1d", np.array([3])),
                     ("arr0d_complex", np.array(3 + 0j))):
        for p in (0, 3):
            out.append((f"arr/{label}/{p}", (v, 2, p), {}))
    # base and padding TYPES: a non-integer base survives validation and
    # fails later inside digits[num % base]; that surface is inherited.
    for label, b in (("float_exact", 2.0), ("float_frac", 2.5),
                     ("bool", True), ("npint64", np.int64(3)),
                     ("npfloat64", np.float64(3.0)), ("str", "3"),
                     ("none", None), ("arr0d_int", np.array(3)),
                     ("arr0d_float", np.array(3.0))):
        out.append((f"btype/{label}", (10, b), {}))
    for label, p in (("float_exact", 1.0), ("float_frac", 2.5),
                     ("bool", True), ("npint64", np.int64(2)),
                     ("str", "2"), ("none", None)):
        out.append((f"ptype/{label}", (10, 2, p), {}))
    return out


for _nm, _cases in (("binary_repr", binary_repr_cases),
                    ("base_repr", base_repr_cases)):
    if _nm in SHAPE_COMPOSE_SPECS:
        raise AssertionError(
            f"shape_compose_cases.py: {_nm} already present in "
            f"SHAPE_COMPOSE_SPECS -- refusing to silently overwrite"
        )
    SHAPE_COMPOSE_SPECS[_nm] = ItemSpec(name=_nm, kind="custom",
                                        custom_cases=_cases,
                                        scalar_like=True)
del _nm, _cases
