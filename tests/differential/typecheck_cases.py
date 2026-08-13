"""Differential coverage for `anionpy/_typecheck_compose.py`: `real_if_close`.

Same collision-checked-merge pattern as compare_compose_cases.py -- builds a
dict of ItemSpecs, merged into registry.REGISTRY from the bottom of
registry.py. `real_if_close` is a Python-only transcription of numpy's OWN
Python implementation (`numpy/lib/_type_check_impl.py`) over already-existing
anionpy primitives (`asanyarray`, `multiply`, `absolute`, `less`, `all`,
`ndarray.real`, `ndarray.imag`) -- no Rust changes. See
`_typecheck_compose.py`'s module docstring for the transcribed source and the
four load-bearing details behind it.

kind="custom" (not "unary"): the whole item turns on the SECOND argument.
The generic unary corpus varies input arrays only, so it would never reach
the `tol > 1` gate, never reach the eps-dtype promotion, and never reach any
of the bad-`tol` error paths -- it would grade as a pass while exercising
exactly one of the function's four branches.

WHAT THE CORPUS DELIBERATELY TARGETS
------------------------------------
1. **The threshold seam.** Cases are built so the imaginary part sits AT
   `eps*tol`, one ULP under it, and one ULP over it, for both complex
   dtypes. A corpus of "obviously tiny" and "obviously large" imaginary
   parts would pass against a threshold that is off by a factor of two.
2. **The eps DTYPE axis.** `tol=100` (weak) and `tol=np.int64(100)` /
   `np.float64(100)` (strong) compute the complex64 threshold in float32 and
   float64 respectively -- genuinely different numbers
   (1.1920929e-05 vs 1.1920928955078125e-05). Both appear -- but see the
   long comment on section 11b: for an INTEGER `tol` below 2**24 the two
   paths are bit-identical (the product is a pure exponent shift), so the
   obvious seam cases do NOT separate them. A weak-Python-float-eps mutant
   -- exactly what `anionpy.finfo(...).eps` would give you today -- survived
   this whole corpus at 0/1204 until 11b was added. That section exists
   solely to make this axis bite, and it does: 10/1244.
3. **The strict `>` gate.** `tol=1`, `tol=True`, `tol=False`, `tol=0` and
   negatives take the ABSOLUTE-tolerance branch, where `tol` is used raw.
   `tol=1` vs `tol=2` is the single case that separates `>` from `>=`.
4. **Bad `tol` types.** numpy never touches these -- Python's own `>` raises
   -- so the messages are inherited, not transcribed. Included anyway,
   because "inherited" is a claim that has to be measured.
5. **Array-valued `tol`.** This is the path that goes through
   `ndarray.__bool__` (declared 2026-08-03), producing numpy's two
   different ambiguous-truth-value ValueErrors for the multi-element and
   empty cases. `real_if_close` was blocked on that item.
6. **Non-complex passthrough**, every dtype anionpy can build, including the
   ones where `tol` is nonsense: the early return happens BEFORE the `tol`
   comparison, so `real_if_close(int_array, "x")` must NOT raise. That
   branch-ORDER fact is observable and is tested directly.

DELIBERATE EXCLUSIONS (recorded, not hidden)
--------------------------------------------
* object- and string-dtype input (`real_if_close(None)` -> object array in
  numpy; `anionpy.asanyarray(None)` raises TypeError). Pre-existing, already
  disclosed ingestion gap, not a `real_if_close` defect.
* `clongdouble` input: numpy's `issubclass(type_, complexfloating)` covers
  it; anionpy has no longdouble family at all. UNREACHABLE, not skipped.
* The view/base relationship of the returned real part. numpy's `a.real` is
  a VIEW, anionpy's is a copy; VALUES and dtype match everywhere. Same
  declared-on-value divergence already recorded for asarray/asanyarray.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec

# The two IEEE machine epsilons, spelled the same way _typecheck_compose.py
# spells them, so the straddling values below are computed from the same
# definition the implementation uses rather than from numpy's finfo.
_EPS = {"complex64": 2.0**-23, "complex128": 2.0**-52}


def real_if_close_cases():
    out = []
    rng = np.random.default_rng(20260803)

    # --- 1. threshold seam: |imag| at, just under, and just over eps*tol ----
    #     `_step` walks one representable step in the array's own dtype so
    #     "just over" is genuinely the next float, not a fudge factor.
    for dt in ("complex64", "complex128"):
        rdt = "float32" if dt == "complex64" else "float64"
        for tol in (2, 100, 1000, np.int64(100), np.float64(100),
                    np.float32(100)):
            thr = np.array(_EPS[dt] * float(tol), dtype=rdt)
            below = np.nextafter(thr, np.array(0.0, dtype=rdt))
            above = np.nextafter(thr, np.array(np.inf, dtype=rdt))
            for lbl, v in (("at", thr), ("below", below), ("above", above),
                           ("zero", np.array(0.0, dtype=rdt)),
                           ("neg_at", -thr), ("neg_above", -above)):
                a = np.array([complex(1.0, float(v))], dtype=dt)
                out.append((f"seam/{dt}/{tol!r}/{lbl}", (a, tol), {}))
                # and with a second, definitely-small element alongside, so
                # the `all()` reduction is actually exercised rather than
                # short-circuited by a one-element array.
                a2 = np.array([complex(1.0, float(v)), complex(2.0, 0.0)],
                              dtype=dt)
                out.append((f"seam_pair/{dt}/{tol!r}/{lbl}", (a2, tol), {}))

    # --- 2. the strict `>` gate: absolute-tolerance branch -------------------
    for dt in ("complex64", "complex128"):
        a = np.array([complex(2.1, 0.5), complex(5.2, 0.25)], dtype=dt)
        for tol in (1, 2, True, False, 0, -1, -100, 0.5, 1.0, 1.0000000001,
                    np.bool_(True), np.bool_(False)):
            out.append((f"gate/{dt}/{tol!r}", (a, tol), {}))
        exact_real = np.array([complex(2.1, 0.0), complex(5.2, -0.0)],
                              dtype=dt)
        for tol in (0, False, np.bool_(False)):
            out.append((f"gate_zero_imag/{dt}/{tol!r}", (exact_real, tol), {}))

    # --- 3. default tol (=100), keyword vs positional ------------------------
    for dt in ("complex64", "complex128"):
        a = np.array([2.1 + 4e-14j, 5.2 + 3e-15j], dtype=dt)
        out.append((f"default/{dt}", (a,), {}))
        out.append((f"kw_tol/{dt}", (a,), {"tol": 1000}))
        out.append((f"pos_tol/{dt}", (a, 1000), {}))
        out.append((f"kw_tol_small/{dt}", (a,), {"tol": 1}))

    # --- 4. non-finite real and imaginary parts ------------------------------
    special = [0.0, -0.0, np.nan, np.inf, -np.inf, 5e-324, -5e-324, 1e-300,
               1.0]
    for dt in ("complex64", "complex128"):
        for r in special:
            for i in special:
                a = np.array([complex(r, i)], dtype=dt)
                for tol in (100, 1, 0, np.inf, np.nan):
                    out.append(
                        (f"special/{dt}/{r!r}/{i!r}/{tol!r}", (a, tol), {})
                    )

    # --- 5. non-finite / extreme tol ----------------------------------------
    for dt in ("complex64", "complex128"):
        a = np.array([1 + 1e-20j, 2 + 1e-3j], dtype=dt)
        for tol in (np.inf, -np.inf, np.nan, 1e30, 1e-30, 2**53, 2**63 - 1,
                    np.float16(100)):
            out.append((f"tol_extreme/{dt}/{tol!r}", (a, tol), {}))

    # --- 6. bad tol types: Python's own `>` supplies the message -------------
    a = np.array([1 + 1e-20j], dtype="complex128")
    bad = [("str", "x"), ("none", None), ("complex", 1j), ("list", [10]),
           ("empty_list", []), ("tuple", (1,)), ("dict", {}),
           ("np_str", np.str_("a")), ("np_complex128", np.complex128(2)),
           ("np_complex64", np.complex64(2)), ("bytes", b"x"),
           ("nested_list", [[1], [2]])]
    for lbl, tol in bad:
        out.append((f"badtol/{lbl}", (a, tol), {}))

    # --- 7. array-valued tol -- the ndarray.__bool__ path --------------------
    #     (registry.py converts these numpy arrays to anionpy arrays for the
    #      anionpy side, so both sides really do take their own __bool__.)
    for dt in ("complex64", "complex128"):
        a = np.array([1 + 1e-20j, 2 + 0j], dtype=dt)
        arr_tols = [
            ("0d", np.array(1000)),
            ("0d_float", np.array(0.5)),
            ("0d_bool", np.array(True)),
            ("size1_1d", np.array([1000])),
            ("size1_2d", np.array([[1000]])),
            ("size1_false", np.array([False])),
            ("size1_tiny", np.array([1e-30])),
            ("size1_int32", np.array([1000], dtype="int32")),
            ("multi", np.array([10, 20])),
            ("multi_2d", np.array([[10, 20], [30, 40]])),
            ("empty", np.array([])),
            ("empty_2d", np.array([]).reshape(0, 3)),
        ]
        for lbl, tv in arr_tols:
            out.append((f"arrtol/{dt}/{lbl}", (a, tv), {}))

    # --- 8. non-complex passthrough, every ionp-constructible dtype ----------
    #     including with a nonsense `tol`: the early return happens BEFORE
    #     the `tol > 1` comparison, so these must NOT raise.
    for dt in ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
               "uint32", "uint64", "float16", "float32", "float64"):
        base = np.array([0, 1, 2, 3], dtype=dt)
        out.append((f"real_pass/{dt}", (base,), {}))
        out.append((f"real_pass_badtol/{dt}", (base, "x"), {}))
        out.append((f"real_pass_nonetol/{dt}", (base, None), {}))
        out.append((f"real_pass_arrtol/{dt}", (base, np.array([10, 20])), {}))
        out.append((f"real_pass_tol0/{dt}", (base, 0), {}))

    # --- 9. shapes: 0-d, empty, multi-dimensional, non-contiguous -----------
    for dt in ("complex64", "complex128"):
        out.append((f"0d_small/{dt}", (np.array(1 + 1e-20j, dtype=dt), 100), {}))
        out.append((f"0d_large/{dt}", (np.array(1 + 1e-3j, dtype=dt), 100), {}))
        for shape in ((0,), (0, 4), (4, 0), (1, 1), (2, 3), (2, 2, 2),
                      (1, 5), (5, 1)):
            n = int(np.prod(shape))
            arr = (rng.normal(size=n)
                   + 1j * rng.normal(size=n) * 1e-20).astype(dt).reshape(shape)
            out.append((f"shape/{dt}/{shape}", (arr, 100), {}))
            big = (rng.normal(size=n)
                   + 1j * rng.normal(size=n)).astype(dt).reshape(shape)
            out.append((f"shape_big_imag/{dt}/{shape}", (big, 100), {}))
        f_order = np.asfortranarray(
            np.array([[1 + 1e-20j, 2 + 0j], [3 + 0j, 4 + 0j]], dtype=dt)
        )
        out.append((f"fortran_order/{dt}", (f_order, 100), {}))
        out.append((f"non_contig/{dt}", (f_order[:, ::-1], 100), {}))

    # --- 10. non-array input (asanyarray does the ingestion) ----------------
    out.append(("list_complex", ([1 + 0j, 2 + 0j], 100), {}))
    out.append(("list_complex_big_imag", ([1 + 1j, 2 + 0j], 100), {}))
    out.append(("nested_list_complex", ([[1 + 0j], [2 + 0j]], 100), {}))
    out.append(("list_float", ([1.0, 2.0], 100), {}))
    out.append(("list_int", ([1, 2], 100), {}))
    out.append(("list_bool", ([True, False], 100), {}))
    out.append(("python_scalar_complex", (1 + 0j, 100), {}))
    out.append(("python_scalar_complex_big", (1 + 1j, 100), {}))
    out.append(("python_scalar_float", (1.0, 100), {}))
    out.append(("python_scalar_int", (1, 100), {}))
    out.append(("python_scalar_bool", (True, 100), {}))
    out.append(("empty_list", ([], 100), {}))

    # --- 11b. the eps-DTYPE seam proper -------------------------------------
    #     Section 1's seam values do NOT separate a float32 eps from a
    #     float64 one, and that is a fact about binary floating point, not an
    #     oversight: for an INTEGER `tol` below 2**24, `2**-23 * tol` is a
    #     pure exponent shift, exactly representable in float32, so both
    #     paths compute bit-identical thresholds. A mutant that used a weak
    #     Python-float eps (which is exactly what `anionpy.finfo(...).eps`
    #     returns today) survived the entire rest of this corpus -- 0/1204 --
    #     until these cases were added.
    #
    #     The separation only exists where the float32 product rounds DOWN
    #     relative to the exact float64 product (rounding UP is provably
    #     undetectable: no float32 value lies in the resulting gap). The
    #     `tol` values below were found by scanning for exactly that
    #     condition; the imaginary part is then set to the float32 threshold
    #     ITSELF, which the correct float32 path rejects (`v < v` is False,
    #     keep complex) and a float64 path accepts (`v < thr64` is True,
    #     convert to real). complex128 is untouched by this axis -- there is
    #     no wider eps to promote to -- which is why only complex64 appears.
    _E32 = np.float32(2.0**-23)
    for tol in (3.3, 7.7, 12345.678, 16777217.0, 1e38):
        thr32 = np.float32(_E32 * np.float32(tol))
        for lbl, tv in ((f"weak/{tol!r}", tol),
                        (f"strong_f32/{tol!r}", np.float32(tol))):
            a = np.array([complex(1.0, float(thr32))], dtype="complex64")
            out.append((f"eps_dtype_seam/{lbl}/at", (a, tv), {}))
            for step, sl in ((-1, "below"), (1, "above")):
                v = np.nextafter(thr32,
                                 np.array(step * np.inf, dtype="float32"))
                a2 = np.array([complex(1.0, float(v))], dtype="complex64")
                out.append((f"eps_dtype_seam/{lbl}/{sl}", (a2, tv), {}))
            a3 = np.array([complex(1.0, float(thr32)), complex(2.0, 0.0)],
                          dtype="complex64")
            out.append((f"eps_dtype_seam_pair/{lbl}/at", (a3, tv), {}))

    # --- 11. random magnitudes, out of the seam's reach ---------------------
    for dt in ("complex64", "complex128"):
        for trial in range(8):
            n = 6
            re = rng.normal(size=n) * 10.0 ** rng.integers(-3, 4)
            im = rng.normal(size=n) * 10.0 ** rng.integers(-20, 2, size=n)
            arr = (re + 1j * im).astype(dt)
            for tol in (100, 1, 1000):
                out.append((f"random/{dt}/{trial}/{tol}", (arr, tol), {}))

    return out


def _build_typecheck_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}
    specs["real_if_close"] = ItemSpec(
        name="real_if_close", kind="custom", custom_cases=real_if_close_cases
    )
    return specs


TYPECHECK_SPECS = _build_typecheck_specs()
