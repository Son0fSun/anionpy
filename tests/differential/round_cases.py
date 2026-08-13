"""Differential coverage for `anionpy/_round_compose.py`: `round`, `around`,
`ndarray.round`.

Same collision-checked-merge pattern as compare_compose_cases.py --  builds a
dict of ItemSpecs, merged into registry.REGISTRY from the bottom of
registry.py.

WHAT IS UNDER TEST. numpy's `np.round` is `_wrapfunc(a, 'round', ...)`, i.e.
it is `ndarray.round`, which is the C driver `PyArray_Round`
(numpy/_core/src/multiarray/calculation.c). That driver is NOT a ufunc: it is
a branch tree that CALLS the `multiply` / `true_divide` / `rint` ufuncs. So
the faithful anionpy shape is a Python driver over anionpy's own ufuncs, and
`anionpy/_round_compose.py` is a literal transcription of that branch tree. Only
the three things no ufunc call can express live in Rust
(`ionp-core/src/round.rs`): `PyArray_CopyInto`, the `arr.real=`/`arr.imag=`
setters, and numpy's `power_of_ten` repeated-multiplication table.

Because the arithmetic is anionpy's own already-verified ufuncs, all three
`Cannot cast ufunc 'multiply'/'divide'/'rint' output from dtype('X') to
dtype('Y') with casting rule 'same_kind'` messages match byte-for-byte
without any wording being transcribed by hand -- they come out of the same
ufunc machinery on both sides. The cases below deliberately exercise each of
those three, so that if the driver ever stops routing through the ufuncs the
message identity stops being free and the divergence bites here.

kind="custom" (not "unary"): every interesting axis of `round` is a keyword
(`decimals=`, `out=`) that the generic unary corpus does not vary, and the
0-d return-shape rule below is invisible to it.

BRANCH ORDER IS OBSERVABLE -- the cases that pin it:

  1. The INTEGER branch is tested BEFORE `decimals == 0`. Consequence:
     `round(int8([15, 25, -15]), -1)` does NOT go down the integer
     short-circuit (that only fires for `decimals >= 0`), but
     `round(int8(...), 0)` does, returning a plain copy rather than
     `rint`'s float. Both are pinned (`int_negative_decimals/*` and
     `dtype_sweep/*/dec_0`).

  2. `bool` is neither `PyArray_ISINTEGER` nor complex, so `decimals != 0`
     drops it into the multiply/divide branch, where the freshly-allocated
     bool working array makes the ufunc refuse. `decimals == 0` instead
     reaches `rint`, which promotes bool -> float16. This asymmetry is not
     a special case anywhere in the code; it falls out of transcribing the
     branch order, and `dtype_sweep/bool/*` is what proves it stayed that
     way.

  3. The `out.size != a.size` check is tested BEFORE the complex branch,
     and it is a SIZE check, not a shape check. So a right-size/wrong-shape
     `out=` produces a *different* ValueError than a wrong-size one:
     wrong size -> "invalid output shape" (raised by the driver itself),
     right size / wrong shape -> "operands could not be broadcast together
     ..." (raised later, by whichever ufunc runs first). Both wordings are
     pinned (`out_wrong_size_raises` vs `out_same_size_wrong_shape_raises`).

RETURN SHAPE: 0-d BECOMES A SCALAR, BUT ONLY WITHOUT `out=`. `round(0-d)`
returns a numpy scalar (the driver's trailing `ret[()]`), while
`round(0-d, out=o)` returns `o`, still 0-d. Pinned by `zero_d/*` and
`out_0d`.

`decimals` IS A C int, NOT a Python int, and the three OverflowError
wordings on the way out are three DIFFERENT messages depending on how far
past the boundary the value is (C long conversion first, then the C int
range check). All three are pinned in `decimals_overflow/*`.

DELIBERATELY NOT COVERED HERE, because it is deliberately NOT IMPLEMENTED:
`np.fix`. It emits a DeprecationWarning anionpy has no channel to emit, and it
diverges from `round` on two dtypes anyway (`fix` on bool returns bool where
`round` returns float16; `fix` on complex128 raises where `round`
succeeds). It stays an absent item -- see `_round_compose.py`'s docstring.
Likewise `np.copyto`: `_copy_into` is only its private half, and `copyto`'s
`casting=`/`where=` surface has not been measured, so `copyto` also stays
absent.

RuntimeWarnings ("invalid value encountered in multiply") that real numpy
raises for the overflowing-scale cases are NOT part of what is compared
here -- only values and exceptions are, exactly as in
compare_compose_cases.py's non-finite atol/rtol cases.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec

# Every real dtype anionpy and numpy share. `longdouble`/`clongdouble` are
# absent from anionpy entirely (`TypeError: data type 'longdouble' not
# understood`), so they are out of scope for this item, not silently
# skipped inside a passing case.
_DTYPES = [
    "bool",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "complex64", "complex128",
]

# `decimals` values chosen to hit each branch: negative (divide-first),
# zero (bare rint), positive (multiply-first), and one value large enough
# that `power_of_ten`'s loop regime runs instead of its 9-entry table.
_DECIMALS = [-17, -3, -1, 0, 1, 2, 5, 17]


def _values_for(dt):
    """Data per dtype. Deliberately includes ties (`x.5`), signed zeros,
    non-finites, and values that straddle the rounding boundary at the
    decimals used above -- NOT just a bland ramp."""
    if dt == "bool":
        return [True, False, True, True, False]
    if dt.startswith("uint"):
        info = np.iinfo(dt)
        return [0, 1, 5, 15, 25, 99, 100, 101, min(255, info.max), info.max]
    if dt.startswith("int"):
        info = np.iinfo(dt)
        return [info.min, -101, -100, -25, -15, -1, 0, 1, 15, 25, 100, 101, info.max]
    if dt.startswith("complex"):
        parts = [-2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 3.5,
                 np.nan, np.inf, -np.inf, 1.005, -1.005, 12345.6789]
        return [complex(re, im) for re, im in zip(parts, list(reversed(parts)))]
    # float16/32/64
    return [-2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 3.5,
            np.nan, np.inf, -np.inf, 1.005, -1.005, 0.125, -0.125, 123.456]


def _round_common_cases():
    """The case list shared by `round`, `around` and `ndarray.round` --
    they are the same driver reached three ways, so testing them against
    three separately-written corpora would test the corpora, not the
    driver. Divergence between the three entry points is instead pinned by
    the `alias_*` cases at the bottom, which are entry-point specific by
    construction (each spec resolves to a different callable)."""
    out = []
    rng = np.random.default_rng(20260803)

    # --- half-to-even ties, the single most load-bearing behaviour -------
    # numpy rounds ties to EVEN, and preserves the sign of a zero result.
    for dt in ("float16", "float32", "float64"):
        ties = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=dt)
        out.append((f"ties_half_to_even/{dt}", (ties,), {}))
        out.append((f"ties_half_to_even/{dt}/dec_1", (ties / 10, ), {"decimals": 1}))
    # -0.0 must survive as -0.0, not become +0.0 (the harness compares
    # signed zeros bitwise unless an item is explicitly exempted -- this
    # item is NOT exempted).
    out.append(("signed_zero_float64",
                (np.array([-0.0, 0.0, -0.4, 0.4, -0.6], dtype=np.float64),), {}))

    # --- full dtype x decimals cross product -----------------------------
    # bool x (decimals != 0) is a RAISING cell (see docstring point 2);
    # it is included, not skipped, precisely because it raises.
    for dt in _DTYPES:
        base = np.array(_values_for(dt), dtype=dt)
        for dec in _DECIMALS:
            out.append((f"dtype_sweep/{dt}/dec_{dec}", (base,), {"decimals": dec}))

    # --- integer dtypes with NEGATIVE decimals: the branch that does real
    #     arithmetic in float64 and casts back to the integer dtype, with
    #     all the wraparound that implies at the type's extremes ----------
    for dt in ("int8", "int16", "int32", "int64",
               "uint8", "uint16", "uint32", "uint64"):
        info = np.iinfo(dt)
        vals = [0, 1, 15, 25, 99, 149, 150, 151, info.max]
        if info.min < 0:
            vals = [info.min, -151, -150, -149, -25, -15, -1] + vals
        arr = np.array([v for v in vals if info.min <= v <= info.max], dtype=dt)
        for dec in (-1, -2, -3, -30):
            out.append((f"int_negative_decimals/{dt}/dec_{dec}", (arr,), {"decimals": dec}))

    # --- complex: real and imaginary parts rounded INDEPENDENTLY, with
    #     the untouched component preserved bit-for-bit ------------------
    for dt in ("complex64", "complex128"):
        parts = [-0.0, 0.0, 0.5, 1.5, 2.5, np.nan, np.inf, -np.inf, 1.005]
        grid = np.array([complex(re, im) for re in parts for im in parts], dtype=dt)
        for dec in (-1, 0, 1, 2):
            out.append((f"complex_grid/{dt}/dec_{dec}", (grid,), {"decimals": dec}))

    # --- random data, every inexact dtype, wide exponent range ----------
    for dt in ("float32", "float64"):
        mags = np.array([float(10.0 ** int(e)) for e in range(-6, 7)], dtype=dt)
        data = (rng.standard_normal(13).astype(dt) * mags).astype(dt)
        for dec in (-4, -1, 0, 1, 3, 8):
            out.append((f"random_magnitudes/{dt}/dec_{dec}", (data,), {"decimals": dec}))

    # --- decimals passed POSITIONALLY vs by keyword ----------------------
    pos = np.array([1.2345, -1.2345, 2.5, -2.5])
    out.append(("decimals_positional", (pos, 2), {}))
    out.append(("decimals_keyword", (pos,), {"decimals": 2}))
    out.append(("decimals_default_omitted", (pos,), {}))
    # a numpy integer scalar / 0-d integer array is index-able, so
    # `operator.index` accepts it where a float would not
    out.append(("decimals_numpy_scalar", (pos,), {"decimals": np.int64(2)}))
    out.append(("decimals_bool", (pos,), {"decimals": True}))

    # --- decimals that are not integers at all: operator.index rejects ---
    out.append(("decimals_float_raises", (pos,), {"decimals": 1.5}))
    out.append(("decimals_float_integral_raises", (pos,), {"decimals": 2.0}))
    out.append(("decimals_none_raises", (pos,), {"decimals": None}))
    out.append(("decimals_str_raises", (pos,), {"decimals": "2"}))

    # --- the three DISTINCT OverflowError wordings (docstring point 4) ---
    out.append(("decimals_overflow/long_max_plus_1", (pos,), {"decimals": 2**63}))
    out.append(("decimals_overflow/long_min_minus_1", (pos,), {"decimals": -(2**63) - 1}))
    out.append(("decimals_overflow/int_max_plus_1", (pos,), {"decimals": 2**31}))
    out.append(("decimals_overflow/int_min_minus_1", (pos,), {"decimals": -(2**31) - 1}))
    # ...and the two values that are IN range and therefore must NOT raise.
    # `-(2**31)` is the exact case that caught the `power_of_ten(i32)`
    # defect during bring-up: `abs(-2**31)` does not fit an i32, so the
    # first implementation raised `OverflowError: out of range integral
    # type conversion attempted` where numpy returns all-nan. If that
    # parameter ever narrows back to i32, this case bites.
    out.append(("decimals_int_min_boundary", (pos,), {"decimals": -(2**31)}))
    out.append(("decimals_int_min_boundary_bool",
                (np.array([True, False]),), {"decimals": -(2**31)}))
    # the positive boundary is deliberately NOT tested at `2**31 - 1`:
    # numpy's own C loop really does spin two billion times there (anionpy's
    # is value-identically short-circuited at inf), so the case would cost
    # seconds of suite time to prove a value already proven by `dec_400`.
    out.append(("decimals_large_positive", (pos,), {"decimals": 400}))
    out.append(("decimals_large_negative", (pos,), {"decimals": -400}))
    # `power_of_ten` switches from its 9-entry table to a multiplication
    # loop at n == 9; pin both sides of that seam.
    for dec in (8, 9, 10, 22, 23, 24, -8, -9, -10, -22, -23, -24):
        out.append((f"power_of_ten_seam/dec_{dec}", (pos,), {"decimals": dec}))

    # --- out=: happy paths ------------------------------------------------
    src = np.array([1.2345, -1.2345, 2.5, -2.5])
    out.append(("out_same_dtype", (src,), {"out": np.empty(4, dtype=np.float64)}))
    out.append(("out_same_dtype_dec2", (src, 2), {"out": np.empty(4, dtype=np.float64)}))
    out.append(("out_wider_dtype", (src.astype(np.float32),),
                {"out": np.empty(4, dtype=np.float64)}))
    out.append(("out_int_input_int_out",
                (np.array([15, 25, -15], dtype=np.int64), -1),
                {"out": np.empty(3, dtype=np.int64)}))
    out.append(("out_int_input_dec0",
                (np.array([15, 25, -15], dtype=np.int64), 0),
                {"out": np.empty(3, dtype=np.int64)}))
    out.append(("out_complex", (np.array([1.25 + 2.75j, -0.5 - 1.5j]), 1),
                {"out": np.empty(2, dtype=np.complex128)}))
    out.append(("out_0d", (np.array(2.5),), {"out": np.empty((), dtype=np.float64)}))
    out.append(("out_2d", (np.arange(6, dtype=np.float64).reshape(2, 3) + 0.5,),
                {"out": np.empty((2, 3), dtype=np.float64)}))

    # --- out=: the four distinct error surfaces --------------------------
    out.append(("out_wrong_size_raises", (src,), {"out": np.empty(3, dtype=np.float64)}))
    out.append(("out_same_size_wrong_shape_raises",
                (np.arange(6, dtype=np.float64).reshape(2, 3),),
                {"out": np.empty(6, dtype=np.float64)}))
    out.append(("out_narrowing_dtype_raises", (src,),
                {"out": np.empty(4, dtype=np.float32)}))
    out.append(("out_int_from_float_raises", (src,),
                {"out": np.empty(4, dtype=np.int64)}))
    out.append(("out_int_narrowing_copy_raises",
                (np.array([300, 400, 500], dtype=np.int64), 0),
                {"out": np.empty(3, dtype=np.uint8)}))
    out.append(("out_complex_from_real_raises", (src,),
                {"out": np.empty(4, dtype=np.complex128)}))
    out.append(("out_real_from_complex_raises",
                (np.array([1.5 + 2.5j, 3.5 + 4.5j]),),
                {"out": np.empty(2, dtype=np.float64)}))
    out.append(("out_not_an_array_raises", (src,), {"out": [0.0, 0.0, 0.0, 0.0]}))
    # COMPLEX + bad `out=`: the case that actually distinguishes docstring
    # point 3. A wrong-SIZE `out=` must be rejected by the driver's own
    # size check ("invalid output shape") BEFORE the complex branch gets to
    # assign into it -- if that check ever moves below the complex branch,
    # `_set_real` reports a broadcast failure instead and this bites.
    # Verified to bite 2026-08-03 by a mutant with the two swapped; the
    # real-vs-complex `out=` cases above do NOT catch that reordering.
    cplx = np.array([1.25 + 2.75j, -0.5 - 1.5j])
    out.append(("out_complex_wrong_size_raises", (cplx,),
                {"out": np.empty(3, dtype=np.complex128)}))
    out.append(("out_complex_same_size_wrong_shape_raises",
                (np.array([[1.25 + 2.75j, -0.5 - 1.5j], [0.5 + 0.5j, 2.5 + 2.5j]]),),
                {"out": np.empty(4, dtype=np.complex128)}))
    out.append(("out_none_explicit", (src,), {"out": None}))

    # --- 0-d input: scalar out, NOT a 0-d array (docstring point 5) ------
    for dt in ("float64", "float32", "int64", "complex128", "bool"):
        val = np.array(2.5).astype(dt) if dt != "complex128" else np.array(2.5 + 3.5j)
        out.append((f"zero_d/{dt}", (val,), {}))
        if dt != "bool":
            out.append((f"zero_d/{dt}/dec_1", (val,), {"decimals": 1}))

    # --- empty arrays: every branch must survive a size-0 input ----------
    for dt in ("float64", "int64", "complex128", "bool"):
        e = np.array([], dtype=dt)
        out.append((f"empty/{dt}", (e,), {}))
        out.append((f"empty/{dt}/dec_2", (e,), {"decimals": 2}))
    out.append(("empty_2d", (np.empty((0, 3), dtype=np.float64),), {"decimals": 1}))

    # --- memory layout: F-contiguous and non-contiguous inputs -----------
    m = np.arange(12, dtype=np.float64).reshape(3, 4) + 0.5
    out.append(("layout_c_contiguous", (m,), {"decimals": 0}))
    out.append(("layout_f_contiguous", (np.asfortranarray(m),), {"decimals": 0}))
    out.append(("layout_f_contiguous_dec1", (np.asfortranarray(m / 3),), {"decimals": 1}))
    out.append(("layout_transposed", (np.ascontiguousarray(m.T),), {"decimals": 1}))
    out.append(("layout_3d", (np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 7,),
                {"decimals": 3}))

    # --- non-array input: the `asarray` front door -----------------------
    out.append(("input_python_list", ([1.5, 2.5, -0.5],), {}))
    out.append(("input_nested_list", ([[1.5, 2.5], [-0.5, 3.5]],), {"decimals": 0}))
    out.append(("input_python_float", (2.5,), {}))
    out.append(("input_python_int", (25,), {"decimals": -1}))
    out.append(("input_python_bool", (True,), {}))
    out.append(("input_python_complex", (1.25 + 2.75j,), {"decimals": 1}))
    out.append(("input_tuple", ((1.5, 2.5, 3.5),), {"decimals": 0}))

    return out


def round_cases():
    return _round_common_cases()


def around_cases():
    return _round_common_cases()


def ndarray_round_cases():
    """Same corpus minus the cases whose first argument is not an array --
    `ndarray.round` is a METHOD, so `[1.5, 2.5].round()` is not a thing
    that exists on either side, and a case that raises AttributeError
    identically on both sides would be a test of Python, not of anionpy."""
    return [
        (label, args, kwargs)
        for (label, args, kwargs) in _round_common_cases()
        if isinstance(args[0], np.ndarray)
    ]


def _identity_checked(fn):
    """`out=` identity contract: `round(a, out=o) is o`. numpy holds this
    unconditionally, so a failure here surfaces as "numpy returned
    normally, anionpy raised AssertionError instead" -- an ordinary,
    already-handled divergence shape, not a new grading path. Same
    mechanism and rationale as fft_cases.py's `_identity_checked`.

    Note the `out` this closure sees on the anionpy side is ALREADY the
    converted `anionpy.ndarray` that `fn` itself receives (registry.py's
    `_wrap_custom_conversion` runs first), so the check compares against
    the same object rather than a numpy array anionpy never saw."""
    def adapter(*args, **kwargs):
        out = kwargs.get("out")
        ret = fn(*args, **kwargs)
        if out is not None and ret is not out:
            raise AssertionError(
                "out= identity contract broken: the returned object is not "
                "the same object as `out` (numpy's own contract is "
                "`round(a, out=o) is o == True`)"
            )
        return ret

    return adapter


def _numpy_adapter(name):
    return _identity_checked(getattr(np, name))


def _ionp_adapter(name):
    """`import anionpy` inside the closure, not at module scope -- this module
    is imported by registry.py during collection, before the extension is
    necessarily importable."""
    def adapter(*args, **kwargs):
        import anionpy
        return _identity_checked(getattr(anionpy, name))(*args, **kwargs)

    return adapter


def _method_probe(a, decimals=0, out=None):
    """Duck-typed: resolves `.round` on whichever array type it is handed,
    so the SAME callable serves as both `numpy_adapter` and `ionp_adapter`
    (established precedent: sort_cases.py's `ndarray.searchsorted`)."""
    ret = a.round(decimals, out=out) if out is not None else a.round(decimals)
    if out is not None and ret is not out:
        raise AssertionError(
            "out= identity contract broken: the returned object is not "
            "the same object as `out`"
        )
    return ret


def _build_round_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}
    specs["round"] = ItemSpec(
        name="round", kind="custom", custom_cases=round_cases,
        numpy_adapter=_numpy_adapter("round"), ionp_adapter=_ionp_adapter("round"),
        atol=0.0, rtol=0.0,
    )
    specs["around"] = ItemSpec(
        name="around", kind="custom", custom_cases=around_cases,
        numpy_adapter=_numpy_adapter("around"), ionp_adapter=_ionp_adapter("around"),
        atol=0.0, rtol=0.0,
    )
    specs["ndarray.round"] = ItemSpec(
        name="ndarray.round", kind="custom", custom_cases=ndarray_round_cases,
        numpy_adapter=_method_probe, ionp_adapter=_method_probe,
        atol=0.0, rtol=0.0,
    )
    return specs


ROUND_SPECS = _build_round_specs()
