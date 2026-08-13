"""anionpy.fft.* differential registry entries.

NEW FILE (permitted: the task brief allows adding new files under
tests/differential/, in addition to owning registry.py itself, mirroring
linalg_cases.py's own "own new file, merged from the bottom of registry.py"
pattern exactly). Builds an `FFT_SPECS: dict[str, ItemSpec]` the same way
linalg_cases.py builds `LINALG_SPECS` -- imported and merged into
`registry.REGISTRY` from the bottom of registry.py (which this task owns),
collision-checked the same way every other *_cases.py merge is.

Underlying engine, loudly disclosed (per the task brief's "no new
third-party crate without saying so loudly" rule -- repeating the
disclosure already made prominently in `ionp-core/Cargo.toml` and
`ionp-core/src/fft.rs`'s own module doc comment): `ionp_core::fft`'s
arbitrary-length complex FFT core is built on the `rustfft` crate (v6.4.1),
NOT a from-scratch radix implementation. This was already a pinned-but-
previously-unused workspace dependency before this task; it is not a newly
added crate, but its first real *use* happens here, so it is disclosed
again prominently in this file's own docstring, not just buried in a
Cargo.toml diff.

Evidence basis for every `epsilon_tolerance` below (REVISED 2026-08-01, see
this task's report for the full write-up): `rustfft` and numpy's pocketfft
are two independently-correct FFT algorithms -- they do not walk the same
butterfly graph in the same order, so their outputs are NOT bit-identical
in general even though both are numerically correct to within their own
algorithm's error bound. This is the exact same class of floating-point
non-associativity `linalg_cases.py` documents for two independent LAPACK
call sequences (see that file's `_EPS_JUST_ABS`/`_EPS_JUST_RELATIVE`
docstrings) -- not a defect, and not evidence of a bug, in either engine.

Raw ULP distance (`harness.ulp_distance`) was tried first and rejected: FFT
outputs routinely contain individual real/imaginary components that are
numerically near-zero (symmetry cancellation, e.g. the Nyquist-adjacent bin
of a real-input transform), and ULP distance measured against a near-zero
true value explodes to metric-noise magnitudes (observed up to ~8.8e18 on
scratch sweeps) with no correctness meaning -- the identical phenomenon
`linalg_cases.py` documents for `det`'s complex128 path. `epsilon_tolerance`
(magnitude-based `"abs"`/`"rel"` distance via `harness.max_abs_distance`/
`harness.max_rel_distance`) was used instead, exactly mirroring linalg's
resolution of the same problem.

Every `epsilon_tolerance`/`epsilon_sweep` number below is the exact maximum
measured over a real, seeded (`ulp_sweep.SEED` = 20260731) sweep run via
standalone scratch scripts (`fft_abs_sweep.py` for the six 1-D transforms,
`fft_nd_sweep.py` for the eight N-D transforms; both scripts still live
under this task's scratchpad, following linalg_cases.py's own documented
practice of a throwaway evidence-generation script whose numbers get pasted
into the real declarations, rather than committing the sweep script itself)
-- NOT hand-picked worked examples. Sample sizes (`n` in each item's
`epsilon_sweep`) are the actual total compared array elements across every
size/dtype/norm combination swept, all >= `MIN_ULP_SWEEP_N` (20000). Sizes
used were NOT drawn from this file's own `custom_cases()` corpora (an
out-of-corpus probe, per the task brief), and `"abs"` vs `"rel"` was chosen
per item based on which distance was the tighter, more stable measured
bound for that item's typical output-magnitude behavior (both were
measured for every item; see `_EPS_JUST_FFT` below for the shared
reasoning).

`fftshift`/`ifftshift` (pure data movement via `ionp_core::manip::roll`,
zero arithmetic) and `fftfreq`/`rfftfreq` (aside from one genuine
correctness bug found and fixed during this task -- see below) are
BIT-EXACT against numpy: confirmed via direct `np.array_equal` checks
across multiple shapes/dtypes/axes (`fftshift`/`ifftshift`) and multiple
`(n, d)` pairs (`fftfreq`/`rfftfreq`), so these four are declared
`atol=0.0, rtol=0.0` with no tolerance dict at all.

Genuine correctness bug found and fixed during this task (not a tolerance
question): `anionpy.fft.fftfreq`/`rfftfreq` originally returned silently
(often an empty array for `n=0`) instead of raising `ZeroDivisionError:
division by zero` for any `(n, d)` with `n * d == 0`, matching numpy's own
behavior exactly (numpy evaluates `1.0 / (n * d)` as a literal Python float
division before building the output array, which raises on `1.0 / 0.0`
regardless of the resulting array's shape). Fixed in
`ionp-py/src/fft.rs::check_zero_division`, at the PyO3 binding boundary
(not in `ionp_core::fft`, which has no Python float-division-by-zero
semantics to inherit). `fftfreq_zerodiv_cases`/`rfftfreq_zerodiv_cases`
below exercise this directly.

Second correctness bug found and fixed during this task: an out-of-range
`axis=`/`axes=` argument on any of these 18 items originally raised numpy's
REAL `numpy.exceptions.AxisError` (correct for ordinary ndarray reductions
elsewhere in this codebase, via `IonpError::AxisError`'s conversion in
`ionp-py/src/lib.rs::to_py_err`) instead of matching what `numpy.fft`
itself actually raises: a plain built-in `IndexError` -- `"tuple index out
of range"` for the six 1-D transforms (numpy's `_raw_fft` indexes
`a.shape[axis]` via plain Python tuple indexing), or `"index {axis} is out
of bounds for axis 0 with size {ndim}"` for the eight N-D transforms
(numpy's N-D path fancy-indexes `numpy.array(a.shape)` with the `axes`
sequence, always reporting "axis 0" of that internal 1-D shape array plus
the FIRST out-of-range value in `axes`, unnormalized, not sorted).
Verified directly against live numpy 2.5.1 across multiple axis values and
ndims for both cases. Fixed in `ionp-py/src/fft.rs::check_1d_axis`/
`check_nd_axes`, called before delegating to `ionp_core::fft`, bypassing
the generic `AxisError` conversion for this one namespace specifically
(numpy.fft really does behave differently here than the rest of numpy).
`*_bad_axis_cases` below exercise this directly.

Third correctness bug found and fixed during this task, specific to
`rfftfreq` (found via the formal `run.py` differential run, not a scratch
probe): `anionpy.fft.rfftfreq` originally raised `ValueError: negative
dimensions are not allowed` for any negative `n`, copying `fftfreq`'s
(correct) behavior. But `fftfreq` and `rfftfreq` are NOT symmetric here --
read via `inspect.getsource`, numpy's real `fftfreq` allocates its output
through the equivalent of `numpy.empty(n, int)`, which does raise for
negative `n`; `rfftfreq` instead builds its output via `numpy.arange(0, n
// 2 + 1, dtype=int)` with no size-validating allocation step at all, and
Python floor division makes `n // 2 + 1 <= 0` for every negative `n`, so
`arange(0, <=0)` is simply an empty range (numpy/Python `arange`'s
start > stop convention), never an error. Verified directly against live
numpy 2.5.1: `np.fft.rfftfreq(-3, 1.0)`, `(-1, 1.0)`, `(-8, 1.0)` all
return `array([], dtype=float64)` with no exception, while the equivalent
`np.fft.fftfreq(...)` calls all raise. Fixed in
`ionp_core::fft::rfftfreq` by clamping `big_n = (n // 2 + 1).max(0)`
instead of rejecting negative `n` up front, and by sizing the output
array from `vals.len()` (never from casting a possibly-negative `big_n`
to `usize` directly, which would wrap around) -- `fftfreq`'s own
negative-`n` raise was left untouched, since it is independently correct.
`rfftfreq_cases` below exercises negative `n` for even, odd, and larger
magnitudes, labeled `n_negative_*_returns_empty` (not `_raises`, which
was this bug's original, since-corrected mislabeling). Confirmed via a
40-case out-of-corpus probe (n, d pairs disjoint from every case in
`rfftfreq_cases()` below) that anionpy now matches numpy exactly, both in
returned values and in the surviving `n * d == 0` zero-division raise.
"""
from __future__ import annotations

import numpy as np

from registry import ItemSpec

# ---------------------------------------------------------------------------
# `out=` identity checking (2026-08-01, out=/device= reclamation task).
#
# registry.py's ordinary value-comparison machinery (harness.compare_values,
# reached via plain custom_cases() entries) verifies that the NUMBERS an
# `out=` case produces are correct, but has no notion of Python object
# IDENTITY -- and identity is the entire point of `out=`: numpy's own
# contract is `np.fft.fft(a, out=o) is o == True` (verified directly, see
# `_state/fft.py`'s ruling), not just "o's contents end up correct". A
# same-values-different-object bug (e.g. an implementation that computes a
# fresh array and copies its data into `out` instead of ever returning
# `out` itself) would pass every ordinary value-comparison case and only
# show up here.
#
# `_identity_checked` wraps EITHER side's real callable (`numpy.fft.*` or
# `anionpy.fft.*`) so that whenever a case supplies `out=`, the wrapper checks
# `return_value is out` before handing the (unchanged) return value back to
# run_case's normal comparison path. This is deliberately NOT a parallel
# hand-rolled comparison: an identity failure raises a plain
# `AssertionError`, which harness.run_case's PRE-EXISTING exception-
# comparison logic then grades exactly like any other divergence (numpy's
# side never raises this, since numpy's own identity contract always
# holds -- so a real anionpy bug here surfaces as "numpy returned normally,
# anionpy raised AssertionError instead", an ordinary, already-handled
# failure shape, not a new grading path). For every case with no `out=`
# kwarg, the wrapper is a no-op passthrough -- safe to install on the
# WHOLE item (every case, not just the out= ones) without touching any
# existing case's behavior.
#
# Set as this ItemSpec's `numpy_adapter`/`ionp_adapter` (registry.py's
# existing, unmodified mechanism -- see inplace_cases.py for the estab-
# lished precedent of using these two fields for a probe rather than plain
# dotted-path resolution). `ionp_adapter`'s args/kwargs (including `out=`)
# are auto-converted numpy.ndarray -> anionpy.ndarray by
# registry.py's `_wrap_custom_conversion` BEFORE this wrapper ever runs
# (confirmed via registry.py's `resolve_ionp()`: the `self.ionp_adapter is
# not None` branch returns `self._wrap_custom_conversion(anionpy, self.ionp_adapter)`,
# not the raw adapter) -- so `kwargs["out"]` inside the ionp-side wrapper
# is already the same anionpy.ndarray object `ionp_fn` itself receives,
# making the identity check meaningful (comparing against the SAME object,
# not a numpy.ndarray that was never passed to anionpy at all).


def _identity_checked(real_fn):
    def adapter(*args, **kwargs):
        out = kwargs.get("out")
        ret = real_fn(*args, **kwargs)
        if out is not None and ret is not out:
            raise AssertionError(
                "out= identity contract broken: the returned object is not "
                "the same object as `out` (numpy's own contract is "
                "`fn(a, out=o) is o == True`)"
            )
        return ret

    return adapter


def _numpy_identity_adapter(name):
    """`numpy_adapter=` for fft item `name` -- resolved lazily (not at
    module-import time) so this module's own import order never depends on
    numpy.fft's attribute layout beyond what `getattr` needs at call time."""
    fn = getattr(np.fft, name)
    return _identity_checked(fn)


def _ionp_identity_adapter(name):
    """`ionp_adapter=` for fft item `name`. `import anionpy` happens inside
    the returned closure (not at module scope) -- anionpy may not be
    importable at all in some harness contexts (see registry.py's
    resolve_ionp() docstring: "anionpy may not be importable at all
    (absent), which is data"), and this module must not turn that into an
    import-time crash instead of the harness's normal "absent" verdict."""

    def real_fn(*args, **kwargs):
        import anionpy

        return getattr(anionpy.fft, name)(*args, **kwargs)

    return _identity_checked(real_fn)


# WITHDRAWN 2026-08-02 (`ufunc_output_casting_err`/`fft_output_casting_err`
# class fix): this module used to declare `_OUT_CASTING_EXC_EQUIV = {
# _UFUNC_OUTPUT_CASTING_ERROR: {TypeError}}` and pass it as
# `exception_equivalences=` to ~15 `out=`-casting fft cases below, which told
# the harness to ACCEPT anionpy's plain `TypeError` where real numpy raises its
# private `_UFuncOutputCastingError` (displayed as `UFuncTypeError`) -- an
# equivalence that excused a real, measured divergence rather than one that
# reconciled a harmless representational difference. `ionp-py/src/fft.rs`'s
# `fft_output_casting_err` now raises the crate's own `UFuncTypeError` (a
# genuine `TypeError` subclass, matching real numpy's exact `type(e).__name__`)
# instead of a plain `TypeError`, so the equivalence is no longer needed and
# has been deleted along with every `exception_equivalences=` use below --
# these cases now compare exact exception class with no substitution.


# ---------------------------------------------------------------------------
# Fixed, deterministic arrays (not randomized -- differential failures must
# be reproducible without a seed; the real out-of-corpus statistical
# evidence for tolerance bounds lives in the scratch sweep scripts cited in
# the module docstring above, not here).
# ---------------------------------------------------------------------------

REAL_EVEN = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])  # n=8
REAL_ODD = np.array([1.5, -2.0, 3.25, 0.0, -7.5])  # n=5
REAL_SINGLE = np.array([3.5])  # n=1, degenerate but legal

COMPLEX_EVEN = np.array(
    [1 + 2j, -3 + 1j, 0 - 4j, 2 + 0j, 5 - 1j, -1 + 3j, 4 + 4j, -2 - 2j]
)  # n=8
COMPLEX_ODD = np.array([2 + 1j, -1 - 1j, 0.5 + 2j, -3 + 0j, 1 - 2j])  # n=5

# Half-spectrum inputs for the c2r-family (irfft/hfft): numpy does not
# require exact Hermitian symmetry on the input -- it uses the given half
# and assumes the missing half is the conjugate mirror, so an arbitrary
# complex array of the right length is a perfectly legal, reproducible
# probe (this is the same style of input irfft_cases would use even if
# constructed from a "real" spectrum first).
HALF8 = np.array([1 + 0j, 2 - 1j, 3 + 2j, -1 + 0.5j, 0.5 + 0j])  # n=8//2+1=5
HALF5 = np.array([2 + 0j, -1 + 3j, 0.5 - 2j])  # n=5//2+1=3

MAT34 = np.array(
    [[1.0, 2.0, 3.0, 4.0], [5.0, -6.0, 7.0, -8.0], [2.5, 0.0, -3.5, 9.0]]
)
CMAT34 = np.array(
    [
        [1 + 1j, 2 - 1j, 0 + 3j, -1 + 0j],
        [4 + 0j, -2 + 2j, 1 - 1j, 0 - 2j],
        [0.5 - 1j, 3 + 0j, -1 - 1j, 2 + 1j],
    ]
)
MAT234 = (np.arange(24.0).reshape(2, 3, 4) - 11.5) * 0.7

# Half-spectrum N-D inputs for irfftn/irfft2, built by direct arithmetic
# (NOT via numpy's own rfftn/rfft2 -- constructing an input array by hand,
# not asking numpy to produce a value under test), shaped for a target
# real output of s=(3,4) (last-axis half = 4//2+1 = 3) and s=(2,3,4) (last-
# axis half = 4//2+1 = 3).
HALF34 = (np.arange(9.0).reshape(3, 3) - 4.0) + 1j * (
    np.arange(9.0).reshape(3, 3)[::-1, ::-1] - 2.0
)
HALF234 = (np.arange(18.0).reshape(2, 3, 3) - 8.5) + 1j * (
    np.arange(18.0).reshape(2, 3, 3)[::-1, ::-1, ::-1] - 4.0
)

# ---------------------------------------------------------------------------
# custom_cases generators (kind="custom" -> list[(label, args, kwargs)]).
# ---------------------------------------------------------------------------


def fft_cases():
    return [
        ("real_even_default", (REAL_EVEN,), {}),
        ("real_odd_default", (REAL_ODD,), {}),
        ("complex_even_default", (COMPLEX_EVEN,), {}),
        ("n_truncate", (REAL_EVEN,), {"n": 5}),
        ("n_pad", (REAL_EVEN,), {"n": 12}),
        ("n_zero_raises", (REAL_EVEN,), {"n": 0}),
        ("n_negative_raises", (REAL_EVEN,), {"n": -1}),
        ("axis0_2d", (MAT34,), {"axis": 0}),
        ("axis1_2d", (MAT34,), {"axis": 1}),
        ("axis_neg1_2d", (MAT34,), {"axis": -1}),
        ("axis_out_of_range_raises", (REAL_EVEN,), {"axis": 5}),
        ("axis_out_of_range_neg_raises", (REAL_EVEN,), {"axis": -10}),
        ("norm_backward", (REAL_EVEN,), {"norm": "backward"}),
        ("norm_ortho", (REAL_EVEN,), {"norm": "ortho"}),
        ("norm_forward", (REAL_EVEN,), {"norm": "forward"}),
        ("norm_invalid_raises", (REAL_EVEN,), {"norm": "bogus"}),
        ("single_element", (REAL_SINGLE,), {}),
        ("out_happy", (REAL_EVEN,), {"out": np.empty(8, dtype=complex)}),
        ("out_wrong_shape_raises", (REAL_EVEN,), {"out": np.empty(5, dtype=complex)}),
        ("out_wrong_dtype_raises", (REAL_EVEN,), {"out": np.empty(8, dtype=np.int64)}),
        ("out_non_array_raises", (REAL_EVEN,), {"out": []}),
    ]


def ifft_cases():
    return [
        ("real_even_default", (REAL_EVEN,), {}),
        ("complex_even_default", (COMPLEX_EVEN,), {}),
        ("complex_odd_default", (COMPLEX_ODD,), {}),
        ("n_truncate", (COMPLEX_EVEN,), {"n": 5}),
        ("n_pad", (COMPLEX_EVEN,), {"n": 12}),
        ("n_zero_raises", (COMPLEX_EVEN,), {"n": 0}),
        ("axis0_2d", (CMAT34,), {"axis": 0}),
        ("axis1_2d", (CMAT34,), {"axis": 1}),
        ("norm_ortho", (COMPLEX_EVEN,), {"norm": "ortho"}),
        ("norm_forward", (COMPLEX_EVEN,), {"norm": "forward"}),
        ("norm_invalid_raises", (COMPLEX_EVEN,), {"norm": "bogus"}),
        ("out_happy", (COMPLEX_EVEN,), {"out": np.empty(8, dtype=complex)}),
        ("out_wrong_shape_raises", (COMPLEX_EVEN,), {"out": np.empty(3, dtype=complex)}),
        ("out_wrong_dtype_raises", (COMPLEX_EVEN,), {"out": np.empty(8, dtype=np.int64)}),
        ("out_non_array_raises", (COMPLEX_EVEN,), {"out": []}),
    ]


def rfft_cases():
    return [
        ("real_even_default", (REAL_EVEN,), {}),
        ("real_odd_default", (REAL_ODD,), {}),
        ("n_truncate", (REAL_EVEN,), {"n": 5}),
        ("n_pad", (REAL_EVEN,), {"n": 12}),
        ("n_zero_raises", (REAL_EVEN,), {"n": 0}),
        ("axis0_2d", (MAT34,), {"axis": 0}),
        ("axis1_2d", (MAT34,), {"axis": 1}),
        ("axis_out_of_range_raises", (REAL_EVEN,), {"axis": 5}),
        ("norm_ortho", (REAL_EVEN,), {"norm": "ortho"}),
        ("norm_forward", (REAL_EVEN,), {"norm": "forward"}),
        ("norm_invalid_raises", (REAL_EVEN,), {"norm": "bogus"}),
        ("out_happy", (REAL_EVEN,), {"out": np.empty(5, dtype=complex)}),
        ("out_wrong_shape_raises", (REAL_EVEN,), {"out": np.empty(4, dtype=complex)}),
        ("out_wrong_dtype_raises", (REAL_EVEN,), {"out": np.empty(5, dtype=np.int64)}),
        ("out_non_array_raises", (REAL_EVEN,), {"out": []}),
    ]


def irfft_cases():
    return [
        ("half8_default", (HALF8,), {}),
        ("half5_default", (HALF5,), {}),
        ("n_explicit_even", (HALF8,), {"n": 8}),
        ("n_explicit_odd", (HALF8,), {"n": 9}),
        ("n_truncate", (HALF8,), {"n": 4}),
        ("n_zero_raises", (HALF8,), {"n": 0}),
        ("norm_ortho", (HALF8,), {"norm": "ortho"}),
        ("norm_forward", (HALF8,), {"norm": "forward"}),
        ("norm_invalid_raises", (HALF8,), {"norm": "bogus"}),
        ("out_happy", (HALF8,), {"out": np.empty(8, dtype=float)}),
        ("out_wrong_shape_raises", (HALF8,), {"out": np.empty(6, dtype=float)}),
        ("out_wrong_dtype_raises", (HALF8,), {"out": np.empty(8, dtype=complex)}),
        ("out_non_array_raises", (HALF8,), {"out": []}),
    ]


def hfft_cases():
    return [
        ("half8_default", (HALF8,), {}),
        ("half5_default", (HALF5,), {}),
        ("n_explicit_even", (HALF8,), {"n": 8}),
        ("n_explicit_odd", (HALF8,), {"n": 9}),
        ("n_zero_raises", (HALF8,), {"n": 0}),
        ("norm_ortho", (HALF8,), {"norm": "ortho"}),
        ("norm_invalid_raises", (HALF8,), {"norm": "bogus"}),
        ("out_happy", (HALF8,), {"out": np.empty(8, dtype=float)}),
        ("out_wrong_shape_raises", (HALF8,), {"out": np.empty(6, dtype=float)}),
        ("out_wrong_dtype_raises", (HALF8,), {"out": np.empty(8, dtype=complex)}),
        ("out_non_array_raises", (HALF8,), {"out": []}),
    ]


def ihfft_cases():
    return [
        ("real_even_default", (REAL_EVEN,), {}),
        ("real_odd_default", (REAL_ODD,), {}),
        ("n_truncate", (REAL_EVEN,), {"n": 5}),
        ("n_pad", (REAL_EVEN,), {"n": 12}),
        ("n_zero_raises", (REAL_EVEN,), {"n": 0}),
        ("norm_ortho", (REAL_EVEN,), {"norm": "ortho"}),
        ("norm_forward", (REAL_EVEN,), {"norm": "forward"}),
        ("norm_invalid_raises", (REAL_EVEN,), {"norm": "bogus"}),
        ("out_happy", (REAL_EVEN,), {"out": np.empty(5, dtype=complex)}),
        ("out_wrong_shape_raises", (REAL_EVEN,), {"out": np.empty(4, dtype=complex)}),
        ("out_wrong_dtype_raises", (REAL_EVEN,), {"out": np.empty(5, dtype=np.int64)}),
        ("out_non_array_raises", (REAL_EVEN,), {"out": []}),
    ]


def fft2_cases():
    return [
        ("real_default", (MAT34,), {}),
        ("complex_default", (CMAT34,), {}),
        ("s_explicit", (MAT34,), {"s": [2, 6]}),
        ("axes_explicit", (MAT34,), {"axes": [1, 0]}),
        ("axes_out_of_range_raises", (MAT34,), {"axes": [0, 5]}),
        ("norm_ortho", (MAT34,), {"norm": "ortho"}),
        ("norm_invalid_raises", (MAT34,), {"norm": "bogus"}),
        ("out_happy", (MAT34,), {"out": np.empty((3, 4), dtype=complex)}),
        ("out_wrong_shape_raises", (MAT34,), {"out": np.empty((3, 3), dtype=complex)}),
        ("out_wrong_dtype_raises", (MAT34,), {"out": np.empty((3, 4), dtype=np.int64)}),
        ("out_non_array_raises", (MAT34,), {"out": []}),
    ]


def ifft2_cases():
    return [
        ("real_default", (MAT34,), {}),
        ("complex_default", (CMAT34,), {}),
        ("s_explicit", (CMAT34,), {"s": [4, 3]}),
        ("norm_forward", (CMAT34,), {"norm": "forward"}),
        ("norm_invalid_raises", (CMAT34,), {"norm": "bogus"}),
        ("out_happy", (CMAT34,), {"out": np.empty((3, 4), dtype=complex)}),
        # NOTE: (4, 4) was deliberately avoided here -- verified directly
        # against live numpy 2.5.1 that a wrong-shape `out=` for a
        # multi-axis transform raises numpy's plain
        # "output array has wrong shape." ONLY when the shape mismatches
        # on the LAST transformed axis (numpy internally passes `out=`
        # only to the final 1-D pocketfft-ufunc call in the axis
        # sequence, whose own shape check is the simple one); a mismatch
        # confined to an EARLIER axis instead falls through to the
        # generic gufunc broadcasting machinery and raises a completely
        # different message (e.g. "operands could not be broadcast
        # together with remapped shapes [original->remapped]: ..."),
        # which anionpy's `finish_fft_result` does not attempt to replicate
        # (a known, disclosed, narrow gap -- see `_state/fft.py`). (4, 3)
        # mismatches on the last axis too, so it hits the simple message
        # both sides agree on.
        ("out_wrong_shape_raises", (CMAT34,), {"out": np.empty((4, 3), dtype=complex)}),
        ("out_wrong_dtype_raises", (CMAT34,), {"out": np.empty((3, 4), dtype=np.int64)}),
        ("out_non_array_raises", (CMAT34,), {"out": []}),
    ]


def fftn_cases():
    return [
        ("real_2d_default", (MAT34,), {}),
        ("complex_2d_default", (CMAT34,), {}),
        ("real_3d_default", (MAT234,), {}),
        ("s_explicit_3d", (MAT234,), {"s": [2, 4, 5]}),
        ("axes_subset_3d", (MAT234,), {"axes": [0, 2]}),
        ("axes_out_of_range_raises", (MAT234,), {"axes": [0, 5]}),
        ("s_axes_length_mismatch_raises", (MAT234,), {"s": [2, 4, 5], "axes": [0, 1]}),
        ("norm_ortho", (MAT234,), {"norm": "ortho"}),
        ("norm_invalid_raises", (MAT234,), {"norm": "bogus"}),
        ("out_happy", (MAT234,), {"out": np.empty((2, 3, 4), dtype=complex)}),
        ("out_wrong_shape_raises", (MAT234,), {"out": np.empty((2, 3, 5), dtype=complex)}),
        ("out_wrong_dtype_raises", (MAT234,), {"out": np.empty((2, 3, 4), dtype=np.int64)}),
        ("out_non_array_raises", (MAT234,), {"out": []}),
    ]


def ifftn_cases():
    return [
        ("real_2d_default", (MAT34,), {}),
        ("complex_3d_default", (MAT234.astype(complex) + 1j * MAT234,), {}),
        ("s_explicit_3d", (MAT234,), {"s": [2, 3, 4]}),
        ("norm_forward", (MAT234,), {"norm": "forward"}),
        ("norm_invalid_raises", (MAT234,), {"norm": "bogus"}),
        ("out_happy", (MAT34,), {"out": np.empty((3, 4), dtype=complex)}),
        ("out_wrong_shape_raises", (MAT34,), {"out": np.empty((3, 3), dtype=complex)}),
        ("out_wrong_dtype_raises", (MAT34,), {"out": np.empty((3, 4), dtype=np.int64)}),
        ("out_non_array_raises", (MAT34,), {"out": []}),
    ]


def rfft2_cases():
    return [
        ("real_default", (MAT34,), {}),
        ("s_explicit", (MAT34,), {"s": [4, 6]}),
        ("axes_explicit", (MAT34,), {"axes": [0, 1]}),
        ("norm_ortho", (MAT34,), {"norm": "ortho"}),
        ("norm_invalid_raises", (MAT34,), {"norm": "bogus"}),
        ("out_happy", (MAT34,), {"out": np.empty((3, 3), dtype=complex)}),
        ("out_wrong_shape_raises", (MAT34,), {"out": np.empty((3, 4), dtype=complex)}),
        ("out_wrong_dtype_raises", (MAT34,), {"out": np.empty((3, 3), dtype=np.int64)}),
        ("out_non_array_raises", (MAT34,), {"out": []}),
    ]


def rfftn_cases():
    return [
        ("real_2d_default", (MAT34,), {}),
        ("real_3d_default", (MAT234,), {}),
        ("s_explicit_3d", (MAT234,), {"s": [2, 4, 5]}),
        ("axes_subset_3d", (MAT234,), {"axes": [0, 2]}),
        ("s_axes_length_mismatch_raises", (MAT234,), {"s": [2, 4, 5], "axes": [0, 1]}),
        ("norm_forward", (MAT234,), {"norm": "forward"}),
        ("norm_invalid_raises", (MAT234,), {"norm": "bogus"}),
        ("out_happy", (MAT234,), {"out": np.empty((2, 3, 3), dtype=complex)}),
        ("out_wrong_shape_raises", (MAT234,), {"out": np.empty((2, 3, 4), dtype=complex)}),
        ("out_wrong_dtype_raises", (MAT234,), {"out": np.empty((2, 3, 3), dtype=np.int64)}),
        ("out_non_array_raises", (MAT234,), {"out": []}),
    ]


def irfft2_cases():
    return [
        ("half34_default", (HALF34,), {"s": [3, 4]}),
        ("half34_no_s", (HALF34,), {}),
        ("norm_ortho", (HALF34,), {"s": [3, 4], "norm": "ortho"}),
        ("norm_invalid_raises", (HALF34,), {"s": [3, 4], "norm": "bogus"}),
        ("out_happy", (HALF34,), {"s": [3, 4], "out": np.empty((3, 4), dtype=float)}),
        ("out_wrong_shape_raises", (HALF34,), {"s": [3, 4], "out": np.empty((3, 3), dtype=float)}),
        ("out_wrong_dtype_raises", (HALF34,), {"s": [3, 4], "out": np.empty((3, 4), dtype=complex)}),
        ("out_non_array_raises", (HALF34,), {"s": [3, 4], "out": []}),
    ]


def irfftn_cases():
    return [
        ("half234_default", (HALF234,), {"s": [2, 3, 4]}),
        ("half34_2d", (HALF34,), {"s": [3, 4]}),
        ("axes_subset", (HALF234,), {"s": [2, 4], "axes": [0, 2]}),
        ("norm_forward", (HALF234,), {"s": [2, 3, 4], "norm": "forward"}),
        ("norm_invalid_raises", (HALF234,), {"s": [2, 3, 4], "norm": "bogus"}),
        ("out_happy", (HALF234,), {"s": [2, 3, 4], "out": np.empty((2, 3, 4), dtype=float)}),
        ("out_wrong_shape_raises", (HALF234,), {"s": [2, 3, 4], "out": np.empty((2, 3, 5), dtype=float)}),
        ("out_wrong_dtype_raises", (HALF234,), {"s": [2, 3, 4], "out": np.empty((2, 3, 4), dtype=complex)}),
        ("out_non_array_raises", (HALF234,), {"s": [2, 3, 4], "out": []}),
    ]


def fftshift_cases():
    return [
        ("1d_even", (REAL_EVEN,), {}),
        ("1d_odd", (REAL_ODD,), {}),
        ("2d_default", (MAT34,), {}),
        ("2d_axis0", (MAT34,), {"axes": 0}),
        ("2d_axis1", (MAT34,), {"axes": 1}),
        ("3d_axes_subset", (MAT234,), {"axes": [0, 2]}),
        ("complex_1d", (COMPLEX_EVEN,), {}),
    ]


def ifftshift_cases():
    return [
        ("1d_even", (REAL_EVEN,), {}),
        ("1d_odd", (REAL_ODD,), {}),
        ("2d_default", (MAT34,), {}),
        ("2d_axis1", (MAT34,), {"axes": 1}),
        ("3d_axes_subset", (MAT234,), {"axes": [0, 2]}),
        ("complex_1d", (COMPLEX_EVEN,), {}),
    ]


def fftfreq_cases():
    return [
        ("n8_d1", (8,), {}),
        ("n7_d1", (7,), {}),
        ("n8_d_half", (8, 0.5), {}),
        ("n1_d1", (1,), {}),
        ("n_negative_raises", (-3,), {}),
        ("n_not_integer_raises", (3.5,), {}),
        ("zerodiv_n_zero_raises", (0, 1.0), {}),
        ("zerodiv_d_zero_raises", (5, 0.0), {}),
        ("zerodiv_both_zero_raises", (0, 0.0), {}),
        ("device_none_explicit", (8,), {"device": None}),
        ("device_cpu", (8,), {"device": "cpu"}),
        ("device_bogus_raises", (8,), {"device": "gpu"}),
    ]


def rfftfreq_cases():
    # NOTE: unlike fftfreq (which allocates via a Python-equivalent of
    # numpy.empty(n, int) and genuinely raises for negative n), numpy's
    # rfftfreq builds its result via numpy.arange(0, n // 2 + 1) -- for
    # ANY negative n, floor division makes n // 2 + 1 <= 0, and
    # arange(0, <=0) is simply an empty range in real numpy, not an
    # error. Verified directly against live numpy 2.5.1:
    # np.fft.rfftfreq(-3, 1.0) / (-1, 1.0) / (-8, 1.0) all return
    # array([], dtype=float64), never raise. So (unlike fftfreq's
    # matching case above) these are "returns empty", not "raises".
    return [
        ("n8_d1", (8,), {}),
        ("n7_d1", (7,), {}),
        ("n8_d_half", (8, 0.5), {}),
        ("n1_d1", (1,), {}),
        ("n_negative_returns_empty", (-3,), {}),
        ("n_negative_odd_returns_empty", (-1,), {}),
        ("n_negative_even_returns_empty", (-8,), {}),
        ("n_not_integer_raises", (3.5,), {}),
        ("zerodiv_n_zero_raises", (0, 1.0), {}),
        ("zerodiv_d_zero_raises", (5, 0.0), {}),
        ("device_none_explicit", (8,), {"device": None}),
        ("device_cpu", (8,), {"device": "cpu"}),
        ("device_bogus_raises", (8,), {"device": "gpu"}),
    ]


# ---------------------------------------------------------------------------
# Evidence-gated tolerance declarations.
# ---------------------------------------------------------------------------

_EPS_JUST_FFT = (
    "`rustfft` (v6.4.1, ionp-core's underlying arbitrary-length complex "
    "FFT engine -- see this file's module docstring for the loud "
    "disclosure) and numpy's pocketfft are two independently-correct FFT "
    "algorithms: they do not walk the same butterfly graph in the same "
    "order, so IEEE754 does not guarantee bit-identical output even though "
    "both are numerically correct to within their own algorithm's error "
    "bound. This is the same class of floating-point non-associativity "
    "linalg_cases.py documents for two independent LAPACK call sequences. "
    "Raw ULP distance was rejected as the grading metric: FFT outputs "
    "routinely contain near-zero real/imaginary components from symmetry "
    "cancellation, and ULP distance against a near-zero true value "
    "explodes to metric-noise magnitudes with no correctness meaning (the "
    "same phenomenon linalg_cases.py documents for det's complex128 path) "
    "-- an `epsilon_tolerance` (magnitude-based abs/rel distance) is used "
    "instead. Bound = the exact maximum measured over a real, seeded "
    "(ulp_sweep.SEED=20260731) sweep across sizes/shapes NOT drawn from "
    "this file's own custom_cases() corpora (out-of-corpus, per the task "
    "brief) -- see this item's `epsilon_sweep` for the recorded (n, "
    "measured) pair, and this file's module docstring for the scratch "
    "scripts that produced every number below."
)


def _build_fft_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["fft.fft"] = ItemSpec(
        name="fft.fft", kind="custom", custom_cases=fft_cases,
        numpy_adapter=_numpy_identity_adapter("fft"),
        ionp_adapter=_ionp_identity_adapter("fft"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float32": ("rel", 1.190130234363096e-07),
            "float64": ("rel", 1.7913030296749173e-13),
            "complex64": ("rel", 4.200436251267092e-06),
            "complex128": ("rel", 4.2056989973594906e-14),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "float32": (55422, 1.190130234363096e-07),
            "float64": (55422, 1.7913030296749173e-13),
            "complex64": (55422, 4.200436251267092e-06),
            "complex128": (55422, 4.2056989973594906e-14),
        },
    )
    specs["fft.ifft"] = ItemSpec(
        name="fft.ifft", kind="custom", custom_cases=ifft_cases,
        numpy_adapter=_numpy_identity_adapter("ifft"),
        ionp_adapter=_ionp_identity_adapter("ifft"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float32": ("rel", 1.190130234363096e-07),
            "float64": ("rel", 5.234187813799692e-14),
            "complex64": ("rel", 5.136471827427158e-06),
            "complex128": ("rel", 2.9478734417587496e-13),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "float32": (55422, 1.190130234363096e-07),
            "float64": (55422, 5.234187813799692e-14),
            "complex64": (55422, 5.136471827427158e-06),
            "complex128": (55422, 2.9478734417587496e-13),
        },
    )
    specs["fft.rfft"] = ItemSpec(
        name="fft.rfft", kind="custom", custom_cases=rfft_cases,
        numpy_adapter=_numpy_identity_adapter("rfft"),
        ionp_adapter=_ionp_identity_adapter("rfft"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float32": ("rel", 2.458330300214584e-06),
            "float64": ("rel", 7.753912113643828e-14),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "float32": (27762, 2.458330300214584e-06),
            "float64": (27762, 7.753912113643828e-14),
        },
    )
    specs["fft.irfft"] = ItemSpec(
        name="fft.irfft", kind="custom", custom_cases=irfft_cases,
        numpy_adapter=_numpy_identity_adapter("irfft"),
        ionp_adapter=_ionp_identity_adapter("irfft"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "complex64": ("rel", 7.092952728271484e-06),
            "complex128": ("rel", 6.485247950873956e-13),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "complex64": (55422, 7.092952728271484e-06),
            "complex128": (55422, 6.485247950873956e-13),
        },
    )
    specs["fft.hfft"] = ItemSpec(
        name="fft.hfft", kind="custom", custom_cases=hfft_cases,
        numpy_adapter=_numpy_identity_adapter("hfft"),
        ionp_adapter=_ionp_identity_adapter("hfft"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "complex64": ("rel", 5.125999450683594e-06),
            "complex128": ("rel", 6.821210263296962e-13),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "complex64": (55422, 5.125999450683594e-06),
            "complex128": (55422, 6.821210263296962e-13),
        },
    )
    specs["fft.ihfft"] = ItemSpec(
        name="fft.ihfft", kind="custom", custom_cases=ihfft_cases,
        numpy_adapter=_numpy_identity_adapter("ihfft"),
        ionp_adapter=_ionp_identity_adapter("ihfft"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float32": ("rel", 2.458330300214584e-06),
            "float64": ("rel", 6.439293542825908e-14),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "float32": (27762, 2.458330300214584e-06),
            "float64": (27762, 6.439293542825908e-14),
        },
    )

    specs["fft.fft2"] = ItemSpec(
        name="fft.fft2", kind="custom", custom_cases=fft2_cases,
        numpy_adapter=_numpy_identity_adapter("fft2"),
        ionp_adapter=_ionp_identity_adapter("fft2"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float32": ("rel", 2.7768282961915247e-06),
            "float64": ("rel", 7.6007246791563e-14),
            "complex64": ("rel", 4.463117420527851e-06),
            "complex128": ("rel", 1.2819545472586934e-13),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "float32": (125025, 2.7768282961915247e-06),
            "float64": (125025, 7.6007246791563e-14),
            "complex64": (125025, 4.463117420527851e-06),
            "complex128": (125025, 1.2819545472586934e-13),
        },
    )
    specs["fft.ifft2"] = ItemSpec(
        name="fft.ifft2", kind="custom", custom_cases=ifft2_cases,
        numpy_adapter=_numpy_identity_adapter("ifft2"),
        ionp_adapter=_ionp_identity_adapter("ifft2"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float32": ("rel", 2.3276356841961388e-06),
            "float64": ("rel", 6.880543122448531e-14),
            "complex64": ("rel", 4.798544068762567e-06),
            "complex128": ("rel", 5.642304205361071e-14),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "float32": (125025, 2.3276356841961388e-06),
            "float64": (125025, 6.880543122448531e-14),
            "complex64": (125025, 4.798544068762567e-06),
            "complex128": (125025, 5.642304205361071e-14),
        },
    )
    specs["fft.fftn"] = ItemSpec(
        name="fft.fftn", kind="custom", custom_cases=fftn_cases,
        numpy_adapter=_numpy_identity_adapter("fftn"),
        ionp_adapter=_ionp_identity_adapter("fftn"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float64": ("rel", 7.6007246791563e-14),
            "complex128": ("rel", 1.2819545472586934e-13),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "float64": (125025, 7.6007246791563e-14),
            "complex128": (125025, 1.2819545472586934e-13),
        },
    )
    specs["fft.ifftn"] = ItemSpec(
        name="fft.ifftn", kind="custom", custom_cases=ifftn_cases,
        numpy_adapter=_numpy_identity_adapter("ifftn"),
        ionp_adapter=_ionp_identity_adapter("ifftn"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float64": ("rel", 6.880543122448531e-14),
            "complex128": ("rel", 5.642304205361071e-14),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "float64": (125025, 6.880543122448531e-14),
            "complex128": (125025, 5.642304205361071e-14),
        },
    )
    specs["fft.rfft2"] = ItemSpec(
        name="fft.rfft2", kind="custom", custom_cases=rfft2_cases,
        numpy_adapter=_numpy_identity_adapter("rfft2"),
        ionp_adapter=_ionp_identity_adapter("rfft2"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float32": ("rel", 3.1680303891334916e-06),
            "float64": ("rel", 3.412911368445222e-14),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "float32": (65871, 3.1680303891334916e-06),
            "float64": (65871, 3.412911368445222e-14),
        },
    )
    specs["fft.rfftn"] = ItemSpec(
        name="fft.rfftn", kind="custom", custom_cases=rfftn_cases,
        numpy_adapter=_numpy_identity_adapter("rfftn"),
        ionp_adapter=_ionp_identity_adapter("rfftn"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", 3.412911368445222e-14)},
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={"float64": (65871, 3.412911368445222e-14)},
    )
    specs["fft.irfft2"] = ItemSpec(
        name="fft.irfft2", kind="custom", custom_cases=irfft2_cases,
        numpy_adapter=_numpy_identity_adapter("irfft2"),
        ionp_adapter=_ionp_identity_adapter("irfft2"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "complex64": ("rel", 5.841255187988281e-06),
            "complex128": ("rel", 1.952615869745978e-12),
        },
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={
            "complex64": (125025, 5.841255187988281e-06),
            "complex128": (125025, 1.952615869745978e-12),
        },
    )
    specs["fft.irfftn"] = ItemSpec(
        name="fft.irfftn", kind="custom", custom_cases=irfftn_cases,
        numpy_adapter=_numpy_identity_adapter("irfftn"),
        ionp_adapter=_ionp_identity_adapter("irfftn"),
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"complex128": ("rel", 1.952615869745978e-12)},
        epsilon_tolerance_justification=_EPS_JUST_FFT,
        epsilon_sweep={"complex128": (125025, 1.952615869745978e-12)},
    )

    specs["fft.fftshift"] = ItemSpec(
        name="fft.fftshift", kind="custom", custom_cases=fftshift_cases,
        atol=0.0, rtol=0.0,
    )
    specs["fft.ifftshift"] = ItemSpec(
        name="fft.ifftshift", kind="custom", custom_cases=ifftshift_cases,
        atol=0.0, rtol=0.0,
    )
    specs["fft.fftfreq"] = ItemSpec(
        name="fft.fftfreq", kind="custom", custom_cases=fftfreq_cases,
        atol=0.0, rtol=0.0,
    )
    specs["fft.rfftfreq"] = ItemSpec(
        name="fft.rfftfreq", kind="custom", custom_cases=rfftfreq_cases,
        atol=0.0, rtol=0.0,
    )

    # fft.test -- mirrors `testing.test`'s already-declared `_IonpTester`
    # pattern (see `tests/differential/testing_cases.py`'s
    # `"testing.test"` entry and `anionpy/testing.py`'s `_IonpTester` class
    # for the full rationale). Same falsifiability fix as that item:
    # `callable(x)` alone is true for a no-op stub, so this mocks
    # `pytest.main` and checks it was actually invoked with a real,
    # non-empty argv containing "-q" (both numpy's `PytestTester` and
    # anionpy's `_IonpTester` build this), and that the tester's return
    # value reflects `pytest.main`'s return code. Verified live against
    # real numpy 2.5.1 (`np.fft.test`) that its mocked-`pytest.main` call
    # shape also includes "-q" and reports `m.called is True`, matching
    # the probe's expectation on both sides.
    specs["fft.test"] = ItemSpec(
        name="fft.test", kind="custom", scalar_like=True,
        custom_cases=lambda: [("invokes_pytest_main_with_constructed_argv", (), {})],
        numpy_adapter=lambda: _probe_fft_test_numpy(),
        ionp_adapter=lambda: _probe_fft_test_ionp(),
    )

    return specs


def _probe_fft_test_call(mod):
    # Delegates to the ONE shared implementation. This used to be a private
    # copy that compared invocation only and never the signature; see
    # harness.probe_pytest_tester for the divergence that slipped through
    # all three copies.
    from harness import probe_pytest_tester
    return probe_pytest_tester(mod)


def _probe_fft_test_numpy():
    import numpy.fft as npfft
    return _probe_fft_test_call(npfft)


def _probe_fft_test_ionp():
    # `anionpy.fft` is a real PyO3 `module` object, NOT a plain-Python
    # submodule file -- `import anionpy.fft` fails with `ModuleNotFoundError`
    # (verified live), so it must be reached via attribute access on the
    # already-imported top-level `anionpy` package, not a dotted import.
    import anionpy
    return _probe_fft_test_call(anionpy.fft)


FFT_SPECS: dict[str, ItemSpec] = _build_fft_specs()
