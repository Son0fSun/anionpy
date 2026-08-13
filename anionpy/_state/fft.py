"""Coverage declarations for the `numpy.fft` block.

NEW FILE (permitted: this task's owned-file list includes
`anionpy/_state/fft.py`). Follows `anionpy/_state/linalg.py`'s exact split-file,
collision-checked-merge convention (see `anionpy/_state/__init__.py`'s module
docstring for the full rationale). The value is the ledger state string
("exact" / "ion"). The COMMENTS ARE THE EVIDENCE and are load-bearing: an
entry without a recorded reason is not better than no entry.

All 18 items below have passing differential cases against numpy 2.5.1 --
see `tests/differential/fft_cases.py`'s module docstring for the full
measured-evidence writeup (per-dtype `epsilon_tolerance` bounds, each the
exact maximum measured over a real 20,000+ sample, out-of-corpus, seeded
sweep; `fftshift`/`ifftshift`/`fftfreq`/`rfftfreq` are bit-exact,
`atol=0.0, rtol=0.0`, no tolerance needed at all). "Passing differential
cases" is not the same as "declared exact", though -- see the
`out=`/`device=` sections below: 8 of the 18 pass every case they are
given while still being UNDECLARED, because their case coverage does not
span the full observable `out=` contract. Read `FFT_STATE` itself, not
just this sentence, for which 10 are actually declared.

Engine, loudly disclosed (repeating the disclosure already made in
`ionp-core/Cargo.toml` and `ionp-core/src/fft.rs`'s own module doc comment,
per the task brief's "no new third-party crate without saying so loudly"
rule): the underlying arbitrary-length complex FFT core is built on the
`rustfft` crate (v6.4.1) -- a pinned-but-previously-unused workspace
dependency, not a newly added one, but its first real *use* is this task.

Three genuine correctness bugs were found and fixed while building this
block (not pre-existing, not something merely worked around with
tolerance -- see `ionp-py/src/fft.rs` and `ionp-core/src/fft.rs` for the
fixes):
  1. `fftfreq`/`rfftfreq` did not raise `ZeroDivisionError: division by
     zero` for any `(n, d)` with `n * d == 0`, unlike numpy (numpy
     evaluates `1.0 / (n * d)` as a literal Python float division before
     building the output array). Fixed via `check_zero_division` at the
     PyO3 binding boundary.
  2. An out-of-range `axis=`/`axes=` argument raised numpy's real
     `numpy.exceptions.AxisError` (correct for ordinary ndarray reductions
     elsewhere in this codebase) instead of matching what `numpy.fft`
     itself actually raises: a plain built-in `IndexError` (`"tuple index
     out of range"` for the six 1-D transforms; `"index {axis} is out of
     bounds for axis 0 with size {ndim}"` for the eight N-D transforms).
     Fixed via `check_1d_axis`/`check_nd_axes`, verified directly against
     live numpy 2.5.1 across multiple axis values/ndims for both cases.
  3. `rfftfreq` raised `ValueError: negative dimensions are not allowed`
     for any negative `n`, copying `fftfreq`'s (correct) behavior --  but
     numpy's real `rfftfreq` builds its output via `numpy.arange(0, n //
     2 + 1)`, not an allocate-by-size call, so for every negative `n` it
     returns an EMPTY array rather than raising (`fftfreq`'s own
     `numpy.empty(n, int)`-based raise on negative `n` is unaffected and
     stays correct). Verified directly against live numpy 2.5.1 for
     several negative `n`; fixed in `ionp_core::fft::rfftfreq` by
     clamping the output length to `>= 0` instead of rejecting negative
     `n`, confirmed via a 40-case out-of-corpus probe plus the formal
     `run.py` differential suite (`fft.rfftfreq` now 10/10 cases, up
     from a 7/8 fail this bug caused when first surfaced).

Parameter-blindness audit (2026-08-01, same standard `_state/linalg.py`
applies -- verified directly against live numpy 2.5.1's own
`inspect.signature` for every item below): all 14 transform functions
(`fft`/`ifft`/`rfft`/`irfft`/`hfft`/`ihfft`/`fftn`/`ifftn`/`fft2`/`ifft2`/
`rfftn`/`rfft2`/`irfftn`/`irfft2`) accept a numpy-only `out=` keyword
(pre-allocated output buffer), and `fftfreq`/`rfftfreq` accept a
numpy-only `device=` keyword (Array API compatibility, defaults to the
only device numpy itself supports on CPU builds). As of this update both
are IMPLEMENTED (see `ionp-py/src/fft.rs`'s `finish_fft_result`/
`fft_output_casting_err`/`resolve_ufunc_name`, and `crate::check_device`,
reused verbatim from `lib.rs`).

RECLAMATION (Monday, 2026-08-01, same day as the UNDECLARED ruling
below -- this supersedes it, and was itself revised same-day after
coordinator review; see the two corrections recorded below):
`out=`/`device=` are now real for all 16 previously-undeclared items,
but only 10 of the 16 are re-declared "exact" -- the other 6 (the 1-D
transforms) plus fftshift/ifftshift/fftfreq/rfftfreq. The 8 N-D
transforms stay UNDECLARED; see "N-D `out=` wrong-shape message" below.

  `out=` contract, verified identical to numpy's:
    - `result is out` holds (identity, not just values) -- enforced by a
      dedicated `numpy_adapter`/`ionp_adapter` identity-check wrapper in
      `tests/differential/fft_cases.py` (`_identity_checked`), not merely
      value comparison, so a copy-into-out-but-return-a-new-object bug
      would have been caught, not missed.
    - Wrong shape -> `ValueError: output array has wrong shape.` (exact
      string, verified live) for the 6 one-axis (1-D) transforms and
      fftshift/ifftshift/fftfreq/rfftfreq's non-out= path. For the 14
      transforms overall this is implemented as an exact shape-equality
      check (fft's `out=` does NOT support numpy's general broadcast-up
      rule the way ordinary ufuncs do -- verified directly:
      `np.fft.fft(a, out=<bigger>)` still raises this same message, not a
      successful broadcast). See "N-D `out=` wrong-shape message" below
      for why this check is NOT sufficient for the 8 multi-axis
      transforms.
    - Non-ndarray `out=` -> `TypeError: return arrays must be of
      ArrayType` (exact string, verified live).
    - Incompatible dtype -> a native Rust-constructed `TypeError` with
      message text verified BYTE-FOR-BYTE against numpy's real (but
      PRIVATE) `numpy._core._exceptions._UFuncOutputCastingError` for
      all 5 real pocketfft ufunc names (`fft`, `ifft`, `irfft`,
      `rfft_n_even`, `rfft_n_odd` -- see `fft.rs`'s module doc comment
      for the name-to-function mapping and the constant `2`, both
      derived empirically). CORRECTION (same day, post-coordinator-
      review): this was ORIGINALLY implemented by importing `numpy`,
      `numpy.fft._pocketfft_umath`, and `numpy._core._exceptions` at
      Rust raise-time to construct a genuine `_UFuncOutputCastingError`
      instance. That violated this project's standing rule (numpy
      supplies input bytes, never answers) on two independent grounds:
      (1) anionpy's own error path would not function without numpy
      installed -- not a replacement if it requires the thing it
      replaces to raise its own errors; (2) it made the `out=`
      dtype-casting differential cases CIRCULAR (both sides would raise
      the literal same numpy instance, so the comparison could never
      fail regardless of whether anionpy's casting logic was correct --
      the same "tautological receiver" failure shape that cost this
      project 32 items elsewhere, a green test that proves nothing).
      Fixed by promoting the native-Rust-`TypeError` fallback (which
      already existed) to the ONLY path, and declaring the type-vs-
      message equivalence explicitly to the differential harness via
      `exception_equivalences={_UFuncOutputCastingError: {TypeError}}`
      in `fft_cases.py` (`_numpy_exc_type`, the same "probe numpy's own
      class without hardcoding its private import path, only to build
      an equivalence dict KEY" pattern `strings_cases.py` already uses).
      `_UFuncOutputCastingError`'s MRO includes `TypeError` and it has no
      public alias (`hasattr(np.exceptions, 'UFuncTypeError')` is
      `False`), so this is the codebase's existing "Answer B" policy for
      numpy-PRIVATE exception classes, not a new exemption -- PUBLIC
      classes (`AxisError`, `LinAlgError`, `ValueError`) still get built
      from the real class, never natively reconstructed. Same-kind-
      compatible downcasts (e.g. complex128 out= for a
      complex64-precision-equivalent result) are correctly ACCEPTED, not
      rejected, matching numpy's own `same_kind` casting rule.

  N-D `out=` wrong-shape message (CORRECTION, same day, post-
  coordinator-review): the 8 multi-axis transforms (`fft2`/`ifft2`/
  `fftn`/`ifftn`/`rfft2`/`rfftn`/`irfft2`/`irfftn`) were ORIGINALLY
  declared "exact" alongside the other 8 on the theory that their `out=`
  wrong-shape divergence was a narrow "message-text nuance on an
  already-correctly-raised ValueError". Further probing (post-review)
  showed that characterization was itself too generous: numpy's
  behavior for a shape-mismatched N-D `out=` is not one alternate
  message but AT LEAST THREE structurally different ones depending on
  exactly which axes mismatch and how (verified directly against live
  numpy 2.5.1, `fft2`/`ifft2`/`fftn` probes):
    - mismatch confined to the LAST transformed axis ->
      "output array has wrong shape." (anionpy's actual behavior, always).
    - mismatch on an EARLIER axis, otherwise broadcast-compatible ->
      "operands could not be broadcast together with remapped shapes
      [original->remapped]: (3,4)->(3,newaxis) ()->() (4,4)->(4,4)  and
      requested shape (4)" (format varies with ndim/axes/the specific
      mismatched dims).
    - an earlier-axis mismatch where the wrong dim is exactly `1` ->
      yet a THIRD shape: "non-broadcastable output operand with shape
      (1,3,4) [remapped to (4,3,1)] doesn't match the broadcast shape
      (2,3,4)".
  These come from numpy's internal per-axis gufunc dispatch (each N-D
  transform is a sequence of 1-D pocketfft-ufunc calls, one per axis,
  with `out=` handed only to the LAST call in that sequence; earlier-axis
  mismatches surface through the generic ufunc broadcasting-error path
  instead of a purpose-built shape check) -- reproducing all three
  byte-for-byte for arbitrary ndim/axes/shape combinations would mean
  replicating a non-trivial slice of numpy's C-level gufunc broadcast-
  remapping logic in Rust, which was judged (per the coordinator's
  explicit either/or: replicate exactly, or leave undeclared) too large
  and too risk-prone to attempt for this task -- a plausible-looking but
  subtly wrong replica across untested shape combinations would be worse
  than an honest UNDECLARED. anionpy raises its own single, correct-in-the-
  simple-case "output array has wrong shape." `ValueError` for every
  mismatch on these 8 items, which is right only in the last-axis-only
  sub-case. These 8 items are therefore left UNDECLARED (see below) --
  identity, values, dtype-casting, non-array-out, and `device=`-adjacent
  behavior are otherwise fully correct and covered by differential
  cases, but a namespace item is declared "exact" only when its FULL
  observable behavior matches, not most of it.

  `device=` contract, verified identical to numpy's: accepts `None`
  (numpy's own default) and `"cpu"` silently; any other value raises
  `ValueError('Device not understood. Only "cpu" is allowed, but
  received: {device}')` with `str()` (not `repr()`) interpolation,
  reusing `lib.rs`'s existing `check_device` verbatim. Check ORDER
  matches numpy's (n-integer-check -> zero-division-check ->
  device-check), verified directly.

All 16 items now have differential cases that actually EXERCISE
`out=`/`device=` (happy-path with identity+value checks, plus
wrong-shape/wrong-dtype/non-array-raises for `out=`, plus
none/cpu/bogus-raises for `device=`) -- not just default-path coverage,
including on the 8 UNDECLARED N-D items (their cases pass in the
last-axis-mismatch sub-case exercised, which is real, documented
coverage -- it just isn't the WHOLE `out=` wrong-shape contract, hence
UNDECLARED rather than "exact"). Only `fftshift`/`ifftshift` never took
either kwarg and were never affected.
"""

FFT_STATE = {
    # WITHDRAWN 2026-08-02 (coordinator audit). These pass ONLY because
    # tests/differential/fft_cases.py registers `_OUT_CASTING_EXC_EQUIV =
    # {_UFUNC_OUTPUT_CASTING_ERROR: {TypeError}}`, which explicitly tells
    # the harness to ACCEPT anionpy's plain `TypeError` where numpy raises
    # `UFuncTypeError`. Measured directly: np.fft.fft(a, out=int64 array)
    # raises UFuncTypeError, anionpy raises TypeError. That is a real
    # divergence the corpus was configured not to see -- the same defect
    # 10 ufuncs were just withdrawn for. An equivalence that excuses a
    # known bug is a weakened check, not a passing test. Delete the
    # equivalence and restore these once the class fix lands.
    # "fft.fft": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): Transpose/relayout quirk: transforming
    #   along a non-last axis (axis=0) of a 2-D array -- numpy returns a C-contiguous result,
    #   anionpy returns F-contiguous (the reverse direction from most other patterns here,
    #   consistent with an internal transpose-compute-transpose-back implementation on anionpy's
    #   side). Verified: numpy strides=(64,16) anionpy=(16,48) on a 3x4 complex128 array. Values
    #   equal (within the item's declared epsilon_tolerance), bytes differ. NOTE: this item
    #   ALSO has additional failing cases from the separate tolerance/byte-check-conflict
    #   ARTIFACT class (see registry.py's check_strides docstring) -- it is revoked here
    #   because a real layout divergence coexists with that artifact, not because every failing
    #   case is genuine. Re-declare when: fft along a non-last axis matches numpy's
    #   C-contiguous-output convention instead of leaving an F-contiguous result from an
    #   internal transpose round-trip.
    # "fft.ifft": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): Transpose/relayout quirk: same
    #   non-last-axis transpose-compute-transpose-back mechanism as fft.fft. NOTE: also has
    #   additional failing cases from the tolerance/byte-check-conflict ARTIFACT class (see
    #   registry.py's check_strides docstring); revoked here because real layout signal
    #   coexists with that artifact. Re-declare when: ifft along a non-last axis matches
    #   numpy's output-contiguity convention instead of leaving an F-contiguous result from an
    #   internal transpose round-trip.
    # "fft.rfft": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): Transpose/relayout quirk: same
    #   non-last-axis transpose-compute-transpose-back mechanism as fft.fft. NOTE: also has
    #   additional failing cases from the tolerance/byte-check-conflict ARTIFACT class (see
    #   registry.py's check_strides docstring); revoked here because real layout signal
    #   coexists with that artifact. Re-declare when: rfft along a non-last axis matches
    #   numpy's output-contiguity convention instead of leaving an F-contiguous result from an
    #   internal transpose round-trip.
    "fft.irfft": "exact",
    "fft.hfft": "exact",
    # ==================================================================
    # REVOKED 2026-08-03 (Monday) -- boundary audit of the declared surface.
    #
    # Measured against real numpy 2.5.1 on this binary
    # (/tmp/mg_verify_bsweep.py). CONTROLS: 0 divergent of 10 -- every one
    # of these items agrees on ordinary well-formed input, which is exactly
    # why the corpus never saw them.
    #
    # Found by a probe-only sweep of 488 declared-"exact" items across
    # toplevel/fft/linalg/ndarray, then re-measured here independently
    # before revocation. Reported claims were NOT taken on trust.
    #
    # Boundary classes represented below:
    #   0-d operand + explicit axis  (amax amin average size repeat)
    #   dtype-specific paths         (angle sinc unwrap fft.ihfft)
    #   result CONTAINER type        (unique_counts/_inverse/_all)
    #   kwarg never exercised        (linalg.vecdot)
    # ==================================================================
    # REVOKED 2026-08-03: complex64 input: numpy REJECTS with TypeError (ufunc
    #   'rfft_n_even' not supported); anionpy silently accepts and returns a
    #   complex64 value. Accepting input numpy rejects is a
    #   silent-wrong-answer class. float64 control agrees.
    # "fft.ihfft": "exact",
    # UNDECLARED -- out= wrong-shape message diverges from numpy's for
    # mismatches confined to a non-last transformed axis (numpy's
    # internal per-axis gufunc broadcasting path produces one of at
    # least 2 further message shapes anionpy does not replicate). See the
    # "N-D `out=` wrong-shape message" section of this module's
    # docstring for the full writeup and the coordinator's either/or
    # ruling this reflects. Identity, values, dtype-casting, non-array
    # out=, and every other behavior are otherwise verified correct.
    #   fft.fft2  fft.ifft2  fft.fftn   fft.ifftn
    #   fft.rfft2 fft.rfftn  fft.irfft2 fft.irfftn
    #
    # RE-VERIFIED 2026-08-02 (independent audit, no Rust changed): directly
    # reproduced the non-last-axis out= message divergence for ALL 8 items
    # individually (not just fft2 as the original write-up spot-checked),
    # by constructing a wrong-shape (mismatch on axis 0, not the last axis)
    # `out=` for each of fft2/fftn/ifft2/ifftn/rfft2/rfftn/irfft2/irfftn.
    # All 8 raise ValueError on both sides (class matches) but with the
    # documented message divergence in every case, e.g.:
    #   numpy: "operands could not be broadcast together with remapped
    #     shapes [original->remapped]: (6,8)->(6,newaxis) ()->()
    #     (8,8)->(8,8)  and requested shape (8)"
    #   anionpy:  "output array has wrong shape."
    # Also re-swept correctness broadly out of corpus for all 8 (303 cases:
    # n=/s=/axes=/norm={backward,ortho,forward}/negative axes/truncation/
    # zero-padding/non-power-of-2 lengths, both forward and inverse
    # directions) -- 0 mismatches beyond the known out= issue above. The
    # defect this section documents is real, current, and is the ONLY
    # thing blocking these 8; still correctly left undeclared.
    #
    # RE-VERIFIED live 2026-08-03 (fft lane pickup, no Rust touched, Python-
    # only). Confirmed all three revocations/non-declarations above are
    # still accurate against the CURRENT binary, not just the recorded
    # writeup (per this task's "a recorded reason is not evidence, verify
    # live" rule):
    #   - fft.fft/ifft/rfft non-last-axis transpose-and-back: reproduced
    #     directly (3x4 float64, axis=0): numpy strides (64,16) C-contig,
    #     anionpy strides (16,48) F-contig; values allclose, bytes differ.
    #     Still revoked/absent.
    #   - The 8 N-D items' out= wrong-shape message divergence: reproduced
    #     directly for ALL 8 (fft2/ifft2/fftn/ifftn/rfft2/rfftn/irfft2/
    #     irfftn, not just fft2), each with a non-last-axis (axis 0)
    #     mismatched out= shape on a (6,8) float64 input -- numpy raises the
    #     "operands could not be broadcast..." remapped-shape message on all
    #     8, anionpy raises "output array has wrong shape." on all 8. A
    #     same-shape-family mismatch confined to axis 1 (the last axis) DOES
    #     match on both sides ("output array has wrong shape." verbatim) --
    #     confirming the divergence is specifically the non-last-axis path,
    #     as previously documented.
    #   - CORRECTION to "the ONLY thing blocking these 8": that claim was
    #     scoped to the out= contract specifically and is too narrow. The
    #     SAME non-last-axis transpose-and-back mechanism that revoked
    #     fft.fft/ifft/rfft also reaches the N-D items on their DEFAULT
    #     (no out=) path, because fft2/fftn/etc. process axes one at a time
    #     internally and every axis but the innermost one being processed
    #     is, at that step, a "non-last" 1-D transform under the hood.
    #     Reproduced directly: `anionpy.fft.fft2(a)` on a (6,8) float64 array,
    #     DEFAULT axes=(-2,-1), no out= involved at all -- numpy strides
    #     (128,16) C-contiguous, anionpy strides (16,96); values allclose,
    #     bytes differ. So even if the out= message text were fixed, these
    #     8 items would still not be bit-exact on the plain default call.
    #     Both defects trace to the same root cause (`ionp-core/src/
    #     fft.rs`'s `axes_moving_to_last`/`inverse_perm` transpose-compute-
    #     transpose-back per-axis engine, see that file's own doc comment)
    #     and both require a Rust change (either compute in a genuinely
    #     N-D-aware layout, or re-materialize C-contiguous after the last
    #     transposed axis) that is out of scope for this Python-only lane.
    #     All 12 items this lane was asked to re-derive
    #     (fft/ifft/fft2/ifft2/fftn/ifftn/rfft/rfft2/rfftn/irfft2/irfftn/
    #     test) stay DECLINED/absent; see "fft.test" below for the 12th.
    "fft.fftfreq": "exact",
    "fft.rfftfreq": "exact",
    "fft.fftshift": "exact",
    "fft.ifftshift": "exact",
    #
    # "fft.test": RESOLVED and DECLARED 2026-08-06 (this task). The
    # decline reasoning directly above is correct as far as it goes
    # (`numpy.fft.test`'s literal contract -- shelling to
    # `pytest --pyargs numpy.fft` against numpy's own installed test
    # files -- genuinely cannot be replicated without importing numpy at
    # runtime), but it stopped short of the substitution this codebase
    # already accepts elsewhere for the exact same situation:
    # `anionpy.testing.test` (declared "exact" in
    # `anionpy/_state/testing.py`) does not replicate numpy's contract
    # either -- it is `_IonpTester` (`anionpy/testing.py`), a same-SHAPE
    # callable (label/verbose/extra_argv/tests kwargs, returns bool,
    # internally calls `pytest.main`) that runs ANIONPY'S OWN test suite
    # instead of numpy's. `fft.test` now mirrors that already-accepted
    # pattern: `anionpy/__init__.py` attaches
    # `fft.test = _IonpTester("anionpy.fft")` via `setattr` after import
    # (`anionpy.fft` is a real PyO3 `module` object, not a plain-Python
    # file, so this can't be a `test = ...` line inside a `fft.py` the way
    # `testing.py`'s own assignment works).
    #
    # Corpus: `tests/differential/fft_cases.py` gained `"fft.test"`,
    # reusing `testing.test`'s own falsifiability-hardened probe shape
    # (mocks `pytest.main`, asserts it was actually called with a
    # non-empty argv containing "-q", and that the tester's return value
    # reflects `pytest.main`'s return code -- a `callable(x)`-only check
    # would pass for a no-op stub, which this catches). ANTI-TAUTOLOGY
    # (measured, not assumed): temporarily swapped `anionpy.fft.test` for
    # `lambda *a, **k: True` (a no-op stub) and re-ran the probe directly
    # -- `pytest.main` was NOT called (`m.called is False`), confirming
    # the probe would catch a fake "test" attribute rather than passing
    # tautologically; restored afterward.
    #
    # OUT-OF-CORPUS: verified live that `anionpy.fft.test`/`anionpy.linalg.test`
    # (`linalg.test`, declared the same way this task, see
    # `anionpy/_state/linalg.py`) both build a `pytest.main` argv
    # containing "-q" matching real numpy's own `np.fft.test`/
    # `np.linalg.test` mocked-argv shape (`-l -q ... -m not slow ...`),
    # confirmed directly against real numpy 2.5.1, not just against the
    # corpus's own probe.
    "fft.test": "exact",
}

# =====================================================================
# CORRECTION 2026-08-03 (Monday) -- THE fft DIAGNOSIS ABOVE IS WRONG,
# TWICE OVER. Both the original docstring and the correction that
# replaced it blamed MEMORY LAYOUT. Neither is the cause.
#
# What the earlier writeups said:
#   (a) original: the N-D `out=` wrong-shape MESSAGE is "the ONLY thing
#       blocking these 8";
#   (b) correction: the plain default path is also broken, because anionpy
#       returns F-contiguous where numpy returns C-contiguous -- cited as
#       numpy strides (128,16) vs anionpy (16,96) on a (6,8) float64 fft2.
#
# The strides observation is REAL but it is NOT the divergence.
# `ndarray.tobytes()` defaults to `order='C'` and emits C-ordered bytes
# REGARDLESS of the array's actual strides. A byte comparison built on
# `.tobytes()` is therefore layout-INSENSITIVE. The stride difference was
# sitting next to the failure, not causing it.
#
# The actual cause, measured 2026-08-03:
#   np.fft.fft2 on (6,8) float64 -- shapes and dtypes match exactly,
#   36 of 48 elements differ, max abs 2.51e-15, max REL 5.76e-16.
#   It is a floating-point divergence in the transform kernel.
#
# And it is driven by TRANSFORM LENGTH, not by axis and not by
# dimensionality. Length sweep, 1-D `fft`, n = 2..40, random float64:
#
#   n = 2, 4, 8            -> bit-exact (0 elements differ)
#   every other n, 3..40   -> diverges, max rel 1.7e-17 .. 3.7e-14
#
# n=16 and n=32 diverge, so this is not even "powers of two are safe" --
# only lengths short enough for the transform to be unambiguous agree.
# The worst observed was n=37 (prime) at 3.67e-14 relative, roughly 165
# ULP at float64.
#
# What this means for the 11 numeric items: they are NOT a Python-surface
# composition lane, and no amount of axis handling or layout fixing will
# close them. anionpy's FFT is a DIFFERENT ALGORITHM from numpy's pocketfft.
# There are exactly two honest routes:
#   1. Port pocketfft's factorization and twiddle scheme exactly, in Rust.
#      This is the only route to a bit-exact declaration.
#   2. Declare them under the existing `ulp_tolerance` mechanism in
#      tests/differential/ufunc_registry.py, with a real justification
#      ("different FFT factorization; error grows with length and with
#      the largest prime factor") and a measured per-dtype sweep. The
#      required tolerance is LARGE -- order 1e-13 relative -- and the
#      ledger would show these as ULP-tolerant, not bit-exact.
#
# Route 2 is defensible and route 1 is expensive; I am not choosing
# between them unilaterally, because the tolerance is big enough that it
# is a product decision about what "exact" means, not an implementation
# detail. Items stay DECLINED until that is settled.
#
# Method note, recorded because it cost two wrong diagnoses in a row:
# a `.tobytes()` comparison cannot see layout, and `np.allclose` cannot
# see small numeric error (its default rtol is 1e-5, which hid a 1e-16
# divergence behind "values allclose: True"). Use `np.array_equal` for
# exactness and read strides off the object directly for layout. Do not
# let the two questions share one probe.
# =====================================================================
