"""Coverage declarations for the small standalone namespace blocks
(`emath` first; `lib`/`ctypeslib`/`rec`/`polynomial` items land here too if
and when they clear the same bar) that don't yet warrant their own file.

See `anionpy/_state/testing.py`'s module docstring for why this is split out
per-block rather than one shared dict: the split lets independent agents
declare coverage without serializing through the same file. The COMMENTS
ARE THE EVIDENCE here too -- an entry without a recorded reason is not
better than no entry.
"""

MISC_NAMESPACES_STATE = {

    # numpy.emath (a.k.a. numpy.lib.scimath) block, declared 2026-08-01.
    # `ionp_core::emath` (ionp-core/src/emath.rs) implements numpy's own
    # `_fix_real_lt_zero` / `_fix_int_lt_zero` / `_fix_real_abs_gt_1` /
    # `_tocomplex` promotion rules (transcribed directly from numpy 2.5.1's
    # installed `numpy/lib/_scimath_impl.py`, not guessed) as a thin
    # dispatch layer in front of `ufunc.rs`'s existing complex ufunc
    # loops -- this module writes zero new transcendental math, only the
    # branch-cut trigger condition and the dtype-promotion table.
    #
    # UPDATED 2026-08-01 (separate pass, dedicated to the 4 pre-existing
    # `ufunc.rs` bugs this block's original comment identified but did not
    # own/fix): `emath.log2` and `emath.log10` are now declared below --
    # both root causes were fixed in `ufunc.rs` (owned by that pass) and
    # re-verified against the REAL differential harness, not assumed:
    #   - `emath.log2`/`emath.log10` on complex input were 1 ULP off
    #     because the complex `Log2`/`Log10` arms in `math_unary_complex`
    #     computed `c_log(z) / T::LN_2()` (division by a rounded constant)
    #     instead of `c_log(z) * T::LOG2_E()` (multiply by the reciprocal
    #     constant) -- mathematically equivalent, but IEEE-754 rounds the
    #     two differently, and numpy's own C99 `clog2`/`clog10` use the
    #     multiply form. Fixed by switching to multiply-by-reciprocal.
    #     Verified: `emath.log2` and `emath.log10` now each pass their FULL
    #     differential corpus, 144/144, bit-exact -- both declared below.
    #   - `emath.logn`'s noise traced to `complex_div` (Smith's algorithm),
    #     not `emath.logn`'s own `log(x)/log(n)` composition: the cross
    #     terms (`b*rat+a` etc.) were computed with plain multiply/add
    #     instead of FMA, while the final reciprocal-then-multiply scaling
    #     step was (correctly) left alone. Fixed by adding FMA to the cross
    #     terms only (`mul_add_ext`). Verified: `emath.logn` went from
    #     19/28 to 26/28 -- the 2 remaining failures
    #     (`broadcast/scalar_array`, `broadcast/empty_with_scalar`) are a
    #     DIFFERENT, NOT-fixed-by-this-pass bug: `TypeError('float64'
    #     object is not an instance of ndarray')` raised by `emath.rs`'s own
    #     scalar-argument dispatch (a file this task does not own), not a
    #     numeric mismatch -- confirmed the numeric/precision half of
    #     `logn` is now fully clean.
    #   - `emath.power`: NOT a 1-ULP rounding gap -- `ufunc.rs`'s complex
    #     Power loop was unconditionally computing `exp(p * log(x))`
    #     instead of special-casing small integer exponents the way
    #     numpy's C99 `npy_cpow`/`npy_cpowi` does. Fixed by adding
    #     exponentiation-by-squaring (`complex_powi`) for real,
    #     integer-valued exponents with `|n| < 100` (numpy's own cutoff,
    #     confirmed empirically: `np.power(-5+0j, k)` is exact through
    #     `k=99`, visibly noisy from `k=100` on). Verified: the exact
    #     small-integer cases that used to be wrong now pass, but the FULL
    #     item is still 7/28 failing, for reasons NOT covered by this fix: 2 are the
    #     same `emath.rs` scalar-dispatch TypeError as `logn` above (not
    #     owned here); the other 5 (int32/int64/float32/complex64/
    #     complex128) are a genuinely SEPARATE, still-open defect in the
    #     GENERAL (non-integer-exponent, or `|n|>=100`) complex power path,
    #     which still calls `num_complex::Complex::powc` unchanged --
    #     1-19 ULP rounding noise plus a real `inf+nanj` vs numpy's
    #     `inf+0j`/`inf+infj` overflow-handling divergence on large-result
    #     int-promoted-to-complex cases. An Annex-G `c_exp(y*c_log(x))`
    #     replacement for that general path was tried and measured WORSE
    #     (more failures, plus a new signed-zero regression on zero-base
    #     cases) -- see `complex_powc_f32`/`complex_powc_f64`'s doc comment
    #     in ufunc.rs, and toplevel.py's `power` entry for the same finding
    #     from the top-level side. Reverted; `emath.power` stays undeclared
    #     -- it does not clear its full corpus and this task's own rule is
    #     against declaring an item that has not been verified byte-exact
    #     out of corpus. `emath.logn` also stays undeclared for the same
    #     reason (2/28 still failing, even though the reason is now
    #     precisely isolated to code this task does not own).
    #
    # "emath.logn": RE-VERIFIED AND DECLARED 2026-08-02 (independent
    # audit, no Rust changed since the comment above). The 2 corpus
    # failures cited above (`broadcast/scalar_array`,
    # `broadcast/empty_with_scalar`) NOW PASS -- re-ran those exact two
    # cases directly (`emath.logn(np.float64(2.5), array)` and
    # `emath.logn(zeros((0,3)), np.float64(4.0))`) and both match numpy
    # exactly; the scalar-argument dispatch TypeError this comment
    # attributed to `emath.rs` is gone. Re-verified further OUT OF CORPUS
    # with a dedicated 28-case sweep the existing corpus doesn't cover:
    # scalar x/scalar n branch-cut coverage (x in {8,1,0,-1,nan,inf},
    # n in {2,1,0,-2,nan,inf}, negative-base x negative-n, 0**0-equivalent,
    # complex x and/or complex n), array-x/scalar-n, scalar-x/array-n,
    # array/array, the two originally-failing broadcast/empty shapes above
    # plus empty-vs-empty, and dtype coverage (float32/64, complex64/128,
    # int32-promoted). 0/28 mismatches (value, shape, dtype, and exception
    # class all matched; numpy's own `RuntimeWarning`s on 0/negative-base
    # divide-by-zero were incidental, not divergences). Measured tolerance:
    # 0.0 relative error (bit-exact) on a separate 2000-case float32/
    # float64 seeded sweep -- declared "exact" on genuinely bit-exact
    # grounds, no epsilon needed.
    "emath.logn": "exact",
    # "emath.arccos": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): ufunc/binop order='K' propagation: numpy
    #   ufuncs/binops default to order='K', preserving a non-C-contiguous (e.g. transposed)
    #   input's memory order in the output; anionpy always emits C-contiguous. Hand-verified case:
    #   b.T + b.T, b=arange(12).reshape(3,4).astype(f64): numpy strides=(8,32) anionpy=(24,8).
    #   Values equal, bytes differ (different traversal order under tobytes('A'), same
    #   underlying reason as the reduction bucket). Re-declare when: the ufunc/binop dispatch
    #   path honors order='K' (propagates non-contiguous input layout to the output) instead of
    #   unconditionally emitting C-contiguous output.
    # RE-DECLARED 2026-08-02 (Monday, order='K' output-layout fix task):
    #   ionp-core/src/emath.rs now votes order='K' output layout using the
    #   original pre-promotion operand(s) via a new `apply_k_order` helper
    #   (sidesteps the fact that `cast_to` forces C-contiguous during the
    #   real-to-complex promotion cast, which would otherwise silently
    #   erase the caller's original layout). Verified with an out-of-corpus
    #   probe across C/F/transposed/strided/negative-stride inputs (with
    #   value ranges chosen to also trigger complex promotion): strides AND
    #   tobytes() match numpy exactly in every case. See
    #   docs/order-k-propagation-fix.md for the full derivation.
    "emath.arccos": "exact",
    # "emath.arcsin": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): ufunc/binop order='K' propagation: numpy
    #   ufuncs/binops default to order='K', preserving a non-C-contiguous (e.g. transposed)
    #   input's memory order in the output; anionpy always emits C-contiguous. Hand-verified case:
    #   b.T + b.T, b=arange(12).reshape(3,4).astype(f64): numpy strides=(8,32) anionpy=(24,8).
    #   Values equal, bytes differ (different traversal order under tobytes('A'), same
    #   underlying reason as the reduction bucket). Re-declare when: the ufunc/binop dispatch
    #   path honors order='K' (propagates non-contiguous input layout to the output) instead of
    #   unconditionally emitting C-contiguous output.
    # RE-DECLARED 2026-08-02 (Monday, order='K' output-layout fix task):
    #   ionp-core/src/emath.rs now votes order='K' output layout using the
    #   original pre-promotion operand(s) via a new `apply_k_order` helper
    #   (sidesteps the fact that `cast_to` forces C-contiguous during the
    #   real-to-complex promotion cast, which would otherwise silently
    #   erase the caller's original layout). Verified with an out-of-corpus
    #   probe across C/F/transposed/strided/negative-stride inputs (with
    #   value ranges chosen to also trigger complex promotion): strides AND
    #   tobytes() match numpy exactly in every case. See
    #   docs/order-k-propagation-fix.md for the full derivation.
    "emath.arcsin": "exact",
    # "emath.arctanh": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): ufunc/binop order='K' propagation: numpy
    #   ufuncs/binops default to order='K', preserving a non-C-contiguous (e.g. transposed)
    #   input's memory order in the output; anionpy always emits C-contiguous. Hand-verified case:
    #   b.T + b.T, b=arange(12).reshape(3,4).astype(f64): numpy strides=(8,32) anionpy=(24,8).
    #   Values equal, bytes differ (different traversal order under tobytes('A'), same
    #   underlying reason as the reduction bucket). Re-declare when: the ufunc/binop dispatch
    #   path honors order='K' (propagates non-contiguous input layout to the output) instead of
    #   unconditionally emitting C-contiguous output.
    # RE-DECLARED 2026-08-02 (Monday, order='K' output-layout fix task):
    #   ionp-core/src/emath.rs now votes order='K' output layout using the
    #   original pre-promotion operand(s) via a new `apply_k_order` helper
    #   (sidesteps the fact that `cast_to` forces C-contiguous during the
    #   real-to-complex promotion cast, which would otherwise silently
    #   erase the caller's original layout). Verified with an out-of-corpus
    #   probe across C/F/transposed/strided/negative-stride inputs (with
    #   value ranges chosen to also trigger complex promotion): strides AND
    #   tobytes() match numpy exactly in every case. See
    #   docs/order-k-propagation-fix.md for the full derivation.
    "emath.arctanh": "exact",
    # "emath.log": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): ufunc/binop order='K' propagation: numpy
    #   ufuncs/binops default to order='K', preserving a non-C-contiguous (e.g. transposed)
    #   input's memory order in the output; anionpy always emits C-contiguous. Hand-verified case:
    #   b.T + b.T, b=arange(12).reshape(3,4).astype(f64): numpy strides=(8,32) anionpy=(24,8).
    #   Values equal, bytes differ (different traversal order under tobytes('A'), same
    #   underlying reason as the reduction bucket). Re-declare when: the ufunc/binop dispatch
    #   path honors order='K' (propagates non-contiguous input layout to the output) instead of
    #   unconditionally emitting C-contiguous output.
    # RE-DECLARED 2026-08-02 (Monday, order='K' output-layout fix task):
    #   ionp-core/src/emath.rs now votes order='K' output layout using the
    #   original pre-promotion operand(s) via a new `apply_k_order` helper
    #   (sidesteps the fact that `cast_to` forces C-contiguous during the
    #   real-to-complex promotion cast, which would otherwise silently
    #   erase the caller's original layout). Verified with an out-of-corpus
    #   probe across C/F/transposed/strided/negative-stride inputs (with
    #   value ranges chosen to also trigger complex promotion): strides AND
    #   tobytes() match numpy exactly in every case. See
    #   docs/order-k-propagation-fix.md for the full derivation.
    "emath.log": "exact",
    # "emath.log2": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): ufunc/binop order='K' propagation: numpy
    #   ufuncs/binops default to order='K', preserving a non-C-contiguous (e.g. transposed)
    #   input's memory order in the output; anionpy always emits C-contiguous. Hand-verified case:
    #   b.T + b.T, b=arange(12).reshape(3,4).astype(f64): numpy strides=(8,32) anionpy=(24,8).
    #   Values equal, bytes differ (different traversal order under tobytes('A'), same
    #   underlying reason as the reduction bucket). Re-declare when: the ufunc/binop dispatch
    #   path honors order='K' (propagates non-contiguous input layout to the output) instead of
    #   unconditionally emitting C-contiguous output.
    # RE-DECLARED 2026-08-02 (Monday, order='K' output-layout fix task):
    #   ionp-core/src/emath.rs now votes order='K' output layout using the
    #   original pre-promotion operand(s) via a new `apply_k_order` helper
    #   (sidesteps the fact that `cast_to` forces C-contiguous during the
    #   real-to-complex promotion cast, which would otherwise silently
    #   erase the caller's original layout). Verified with an out-of-corpus
    #   probe across C/F/transposed/strided/negative-stride inputs (with
    #   value ranges chosen to also trigger complex promotion): strides AND
    #   tobytes() match numpy exactly in every case. See
    #   docs/order-k-propagation-fix.md for the full derivation.
    "emath.log2": "exact",
    # "emath.log10": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): ufunc/binop order='K' propagation: numpy
    #   ufuncs/binops default to order='K', preserving a non-C-contiguous (e.g. transposed)
    #   input's memory order in the output; anionpy always emits C-contiguous. Hand-verified case:
    #   b.T + b.T, b=arange(12).reshape(3,4).astype(f64): numpy strides=(8,32) anionpy=(24,8).
    #   Values equal, bytes differ (different traversal order under tobytes('A'), same
    #   underlying reason as the reduction bucket). Re-declare when: the ufunc/binop dispatch
    #   path honors order='K' (propagates non-contiguous input layout to the output) instead of
    #   unconditionally emitting C-contiguous output.
    # RE-DECLARED 2026-08-02 (Monday, order='K' output-layout fix task):
    #   ionp-core/src/emath.rs now votes order='K' output layout using the
    #   original pre-promotion operand(s) via a new `apply_k_order` helper
    #   (sidesteps the fact that `cast_to` forces C-contiguous during the
    #   real-to-complex promotion cast, which would otherwise silently
    #   erase the caller's original layout). Verified with an out-of-corpus
    #   probe across C/F/transposed/strided/negative-stride inputs (with
    #   value ranges chosen to also trigger complex promotion): strides AND
    #   tobytes() match numpy exactly in every case. See
    #   docs/order-k-propagation-fix.md for the full derivation.
    "emath.log10": "exact",
    # "emath.sqrt": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): ufunc/binop order='K' propagation: numpy
    #   ufuncs/binops default to order='K', preserving a non-C-contiguous (e.g. transposed)
    #   input's memory order in the output; anionpy always emits C-contiguous. Hand-verified case:
    #   b.T + b.T, b=arange(12).reshape(3,4).astype(f64): numpy strides=(8,32) anionpy=(24,8).
    #   Values equal, bytes differ (different traversal order under tobytes('A'), same
    #   underlying reason as the reduction bucket). Re-declare when: the ufunc/binop dispatch
    #   path honors order='K' (propagates non-contiguous input layout to the output) instead of
    #   unconditionally emitting C-contiguous output.
    # RE-DECLARED 2026-08-02 (Monday, order='K' output-layout fix task):
    #   ionp-core/src/emath.rs now votes order='K' output layout using the
    #   original pre-promotion operand(s) via a new `apply_k_order` helper
    #   (sidesteps the fact that `cast_to` forces C-contiguous during the
    #   real-to-complex promotion cast, which would otherwise silently
    #   erase the caller's original layout). Verified with an out-of-corpus
    #   probe across C/F/transposed/strided/negative-stride inputs (with
    #   value ranges chosen to also trigger complex promotion): strides AND
    #   tobytes() match numpy exactly in every case. See
    #   docs/order-k-propagation-fix.md for the full derivation.
    "emath.sqrt": "exact",

    # numpy.lib / numpy.ctypeslib block, declared 2026-08-01. Three of the
    # twelve `lib.*`/`ctypeslib.*` surface items, implemented in
    # `anionpy/lib.py`/`anionpy/ctypeslib.py`, verified via
    # tests/differential/lib_cases.py (15+7+3 = 25 cases, all bit-exact
    # pass, no tolerance of any kind declared or needed):
    #   - `lib.NumpyVersion`: ported VERBATIM from numpy 2.5.1's own
    #     `numpy/lib/_version.py` -- pure string/int comparison, no array
    #     involved anywhere in it.
    #   - `lib.Arrayterator`: ported VERBATIM from numpy 2.5.1's own
    #     `numpy/lib/_arrayterator_impl.py` -- a lazy buffered slicing
    #     iterator whose entire body is Python-int index bookkeeping
    #     (start/stop/step) plus `self.var[slice_]`, delegating the actual
    #     sub-array extraction to whatever `__getitem__` the wrapped object
    #     provides. Verified against a REAL `anionpy.ndarray` receiver (see
    #     lib_cases.py's `convert_ionp_args` comment on this item -- the
    #     differential case deliberately exercises `anionpy.ndarray.shape`/
    #     `.ndim`/tuple-of-slice `__getitem__`, not numpy's). One
    #     documented, undeclared gap: `Arrayterator.flat` needs
    #     `ndarray.flat`, which `anionpy.ndarray` does not implement (out of
    #     this task's file ownership -- see anionpy/lib.py's module
    #     docstring); `lib.Arrayterator` the class is what
    #     numpy_surface.json actually lists, and it is fully portable.
    #   - `ctypeslib.load_library`: ported VERBATIM from numpy 2.5.1's own
    #     `numpy/ctypeslib/_ctypeslib.py` -- pure `os.path`/`ctypes.cdll`
    #     lookup, zero ndarray dependency of any kind.
    # `ctypeslib.c_intp` DECLARED 2026-08-03 (Wave 1, "connect what already
    # exists" audit). Previously implemented (anionpy/ctypeslib.py) and
    # verified by hand to match `np.ctypeslib.c_intp` exactly on this
    # platform (both `ctypes.c_long`), but left undeclared on the premise
    # that a bare module constant has "no call semantics, so it has no
    # differential ItemSpec" -- that premise was about the ABSENCE of a
    # registry entry, not a genuine correctness gap, and it went stale the
    # moment a kind="custom" ItemSpec that compares the constant ONCE,
    # directly (no varying input needed), was added to lib_cases.py: see
    # `LIB_SPECS["ctypeslib.c_intp"]` there (its own comment records the
    # live re-verification: `numpy.ctypeslib.c_intp is anionpy.ctypeslib.c_intp
    # is ctypes.c_long` on this arm64 Darwin/LP64 box). Re-run confirms
    # `[PASS] ctypeslib.c_intp (1/1 cases)`; it now has real automated
    # regression coverage behind it, so the "stays absent" reasoning no
    # longer applies.
    "lib.NumpyVersion": "exact",
    "lib.Arrayterator": "exact",
    "ctypeslib.load_library": "exact",
    "ctypeslib.c_intp": "exact",
}
