"""Coverage declarations for ticket #72's legacy polynomial API block
(`numpy.polyval`, `polyadd`, `polysub`, `polyder`, `polyint`, `polydiv`,
`poly`, `roots`, `polyfit`, `poly1d`) -- the pre-1.4, highest-degree-first
coefficient convention. NOT `anionpy.polynomial` (the modern
chebyshev/legendre/hermite/power-series package, declared separately in
`anionpy/_state/polynomial.py`).

Implementation: `anionpy/_polynomial_legacy.py`. Differential corpus:
`tests/differential/poly1d_legacy_cases.py` (`LEGACY_POLY_SPECS`, merged
into `tests/differential/registry.py`'s `REGISTRY`). See
`docs/TICKET-72-POLY1D-2026-08-08.md` for the full implementation writeup.

See `anionpy/_state/testing.py`'s module docstring for why coverage
declarations are split one-module-per-block rather than a single shared
dict: it lets independent agents declare without serializing through the
same file, with a loud `AssertionError` on any duplicate key across
blocks (see `anionpy/_state/__init__.py`).

WIRING NOTE: this is a landing pass over an implementation a prior agent
built but could not wire in (registry.py and every _state/*.py file were
locked by concurrent edits at the time). This module does not change
`anionpy/_polynomial_legacy.py`, `ionp-core/src/poly_legacy.rs`, or
`ionp-py/src/poly_legacy.rs` -- only wires the corpus into `registry.py`
and declares coverage here, after independently re-verifying every
declared item OUT of corpus (fresh seeded sweeps from `/private/tmp`,
comparing exact array equality including dtype -- not `allclose`, not a
re-run of the committed corpus).

======================================================================
CORRECTION to a prior agent's landing instructions (important -- read
before adding to this file)
======================================================================

The building agent's own report (see "What remains for the user to
land" in the ticket doc) asked for NINE items to be declared `exact`:
`polyval`, `polyadd`, `polysub`, `polyder`, `polyint`, `poly1d` (bare,
no caveat) PLUS `poly`, `roots`, `polyfit` ("state='exact' with the
epsilon caveat already declared in the registry entries"). That
instruction conflates "passes under a declared epsilon_tolerance" with
"exact" -- this project's `exact` means bit-identical, not "within a
measured slack." `roots` and `polyfit` are declared in the registry with
a real, evidence-gated `epsilon_tolerance` (not blind atol/rtol) because
they go through `anionpy.linalg.eigvals`/`lstsq`, independent
LAPACK-adjacent paths from real numpy's own -- ~5e-15 relative error
after sorting is the honest number (ordering itself matches real numpy
exactly -- 0 ordering-only differences observed across 400 out-of-corpus
trials in this landing pass). Note the metric matters here: an unfloored
relative-error denominator on a near-zero root can inflate this ~5e-15
figure to ~1.97 for that one component -- that is a bug in how the
metric floors near-zero denominators, not evidence the code is
imprecise. Either way, "close to zero error" is not "zero error", and
this file's bar is bit-exactness. Both remain UNDECLARED here,
deliberately, matching `roots`'/`polyfit`'s treatment in
`anionpy/_state/polynomial.py` for their modern-package counterparts
(`polynomial.polynomial.polyroots`/`polyfit`, also epsilon-toleranced,
also not declared "exact").

======================================================================
A SECOND, independently-measured correction (this landing pass, not the
building agent's claim): 6 of the 7 remaining "plain function" items
are ALSO not safely bit-exact, despite passing their own 38-case
committed corpus 38/38 and despite an earlier informal 400-trial sweep
reportedly finding 0 nonexact cases for them.
======================================================================

Out-of-corpus verification for this landing pass deliberately varied
the *kind* of input, not just size (per this ticket's own methodology
requirement): int8/16/32/64, uint8/16/32, float16/32/64, complex128,
bool, empty arrays, length-1, leading zeros, negative values, 0-d. Two
distinct, independently reproduced classes of divergence turned up:

1. NARROW-DTYPE non-preservation (the same, already-accepted category
   `anionpy/_state/polynomial.py`'s module docstring documents for the
   modern package's `polyval2d`/etc: values match, but the RETURNED
   DTYPE does not, for int8/16/32, uint8/16/32, and float32 inputs --
   e.g. `polyval(np.array([1,2],dtype=np.int32), 0)` returns real
   numpy's `int64` but anionpy's `float64`; `polyder(uint8_array)`
   likewise. All of `polyval`/`polyadd`/`polysub`/`polyder`/`poly`
   showed this.

2. *** RETRACTED 2026-08-08 -- THIS FINDING WAS FALSE. ***

   This slot previously documented a CALL-HISTORY-DEPENDENT divergence:
   the same (p, x) pair allegedly returning `float64` instead of
   `complex128` for a zero-length output, but only after a sequence of
   unrelated prior calls, and reproducing deterministically given a
   fixed call history (`polyval` 30/400, `polyadd` 8/400, `polysub`
   8/400, `polyder` 144/400, `polyint` 7/400, `poly` 72/400, stable
   across 3 fresh-process reruns). It was attributed to stale/reused
   buffer contents in the Rust dispatch layer.

   None of that was real. It was an artifact in the DETECTING
   INSTRUMENT. Both the sweep (`core_dtype_sweep.py`'s `arrs_equal`)
   and the replay script normalised each side with:

       iov = np.asarray(iov.tolist() if hasattr(iov, "tolist") else iov)

   For an EMPTY array, `.tolist()` returns `[]` regardless of source
   dtype, and `np.asarray([]).dtype` is unconditionally `float64`.
   That is stock numpy behaviour with an empty Python list and has
   nothing to do with anionpy:

       complex128 -> tolist(): [] -> np.asarray(...).dtype: float64
       int64      -> tolist(): [] -> np.asarray(...).dtype: float64
       float64    -> tolist(): [] -> np.asarray(...).dtype: float64

   Read directly, `iov.dtype` is `complex128` and matches numpy
   exactly. The probe was laundering every empty-output case through a
   dtype-collapsing round-trip and reporting the collapse as a
   divergence. Re-running the identical seeded sweep with `arrs_equal`
   comparing `np.asarray(iov)` directly gives `polyval` 0/400,
   `polyadd` 0/400, `polysub` 0/400, `polyder` 0/400, `polyint` 0/400.

   WHY THE "REPRODUCIBILITY" WAS SO CONVINCING, since that is the part
   worth learning from: an instrument bug is perfectly deterministic.
   Stable counts across fresh-process reruns felt like strong evidence
   of a real state-dependent defect, but a broken measurement repeats
   just as faithfully as a real effect -- reproducibility distinguishes
   signal from NOISE, and says nothing at all about whether the
   instrument is measuring the intended quantity. The "isolation does
   not reproduce it, full sequence does" asymmetry had the same
   innocent cause: the isolated replay happened to print a non-empty
   result, the sequenced one an empty one.

   The corrected sweep did surface one REAL divergence that had been
   buried in this noise -- see class 3 below.

3. `poly` ONLY: genuine ULP-level floating-point divergence on
   complex128 roots, reproducible FROM THE INPUT ALONE with no call
   history, max abs diff 2.220446049250313e-16 (1 ULP), from a
   different summation/convolution order when building the
   characteristic polynomial. Same category as `roots`/`polyfit`
   below: would need an evidenced epsilon or ULP tolerance to declare,
   and is NOT bit-exact. Not declared.

`polydiv` alone was independently stress-tested against exactly this
second failure class -- 500 unrelated `polyval`/`polyder` calls
(deliberately including many complex/empty-result combinations) run
immediately before a 200-trial `polydiv` sweep on complex128/float64
operands -- and held clean, 0/200, on top of 3x500 = 1500 trials of its
own ordinary (non-adversarial-ordering) sweep, 0 failures throughout,
plus 0/400 through the real differential harness via
`poly1d_legacy_cases.py`'s corpus. `polydiv` alone is declared below.

======================================================================
TICKET #76, first landing (2026-08-08): `polyadd`/`polysub` promoted to
`exact`; `polyval`/`polyder`/`polyint` held back on NEWLY MEASURED
defects that the earlier sweeps were structurally blind to.
======================================================================

The 5 items above were left undeclared pending verification to
`polydiv`'s own standard. That verification ran (18,000 trials: 3,600
per item on int64/float64/complex128, preceded by the same 500-call
adversarial-ordering stress `polydiv` got, deliberately including many
empty-result and complex combinations). All five measured 0/3600.

That sweep was NOT sufficient, and the way it failed is the point:

  * It only ever fed **1-D** coefficient arrays, so it could not see
    that `np.polyval` accepts an N-D `p` and Horners over the first
    axis (`np.polyval(np.ones((2,2)), 1)` -> `array([2., 2.])`) while
    anionpy raises `ValueError: p must be a 1d array`.
  * It only ever compared RETURN VALUES, so it could not see any
    error-path divergence at all. Zero exceptions were raised across
    all 18,000 trials -- which reads as "clean" but actually means the
    error paths were never exercised once.

A separate provenance-varied probe (sliced views, negative strides,
F-order columns, 0-d, Python lists, bare scalars, empty, length-1,
nan/inf, +-1e300) plus an explicit raise-parity probe found what the
big sweep could not. Three DISTINCT defects, not one -- they share a
family, not a cause:

  A. `polyval` with `p.ndim > 1`: numpy Horners over the first axis
     and returns a value; anionpy raises. `_polynomial_legacy.py:104`.
  B. `polyder`/`polyint` on 0-d or scalar `p`: numpy raises
     `TypeError: len() of unsized object` (incidental -- it is
     `len(p)` on an unsized object leaking out of numpy's own
     implementation); anionpy raises `ValueError: p must be a 1d
     array`. The harness compares exception TYPE **and** MESSAGE
     (`harness.py:1748`), so this is a live divergence, not cosmetics.
  C. `polyder`/`polyint` with `p.ndim >= 2`: numpy reaches its
     broadcast and raises `ValueError: operands could not be broadcast
     together with shapes (0,3) (0,)` (note: shapes differ between
     `polyder` and `polyint` for the same input); anionpy raises its
     own `ValueError` with a different message.

Ticket #76 continues to own A/B/C. The methodological lesson, recorded
because it has now cost this file two wrong conclusions in two days:
**a corpus is blind to any property its inputs never vary, and 18,000
trials of a blind corpus are exactly as blind as 1.** Trial count is
not evidence of coverage; input DIVERSITY is. The first landing pass
was misled by a dtype-collapsing probe, this one by a
dimension-invariant one, and in both cases the sample size was large
enough to feel authoritative.

======================================================================
TICKET #76, second landing (2026-08-08): A/B/C FIXED.
`polyval`/`polyder`/`polyint` promoted to `exact`.
======================================================================

Root causes, from reading real numpy's actual source
(`inspect.getsource`), not its docs:

  A. `polyval`'s reference implementation is `for pv in p: y = y*x + pv`
     -- a plain Horner loop over `p`'s first axis with NO ndim guard at
     all. Fixed by adding an N-D Horner path
     (`polyval_legacy_horner_nd`, `ionp-core/src/poly_legacy.rs`),
     composed entirely from `ufunc::binary_op` (already-verified
     broadcasting arithmetic, correct dtype promotion, numpy-exact
     error text) rather than a hand-written broadcaster -- dispatched
     from `_polyval_legacy_nd` (`ionp-py/src/poly_legacy.rs`) whenever
     `p.ndim != 1`. A genuinely 0-d `p` is NOT special-cased: numpy's
     own `for pv in p:` on a 0-d array raises
     `TypeError: iteration over a 0-d array`, and `_polyval_legacy`
     (`anionpy/_polynomial_legacy.py`) reproduces that for free with a
     bare `for _ in p: break` -- no new Rust error path needed.

  B. `polyder`'s reference implementation calls `len(p)` and computes
     its broadcast product UNCONDITIONALLY every call, even at the
     `m == 0` base case whose result it then discards. `polyint`'s
     `m == 0` case SHORT-CIRCUITS before calling `len()` at all. This
     asymmetry is real and now preserved bit-for-bit:
     `polyder(np.array(5), 0)` raises `TypeError: len() of unsized
     object` (measured, matches numpy) while `polyint(np.array(5), 0)`
     succeeds and returns a genuine 0-d `ndarray` (measured
     `type(r) is numpy.ndarray`, NOT a numpy scalar -- polyval's
     scalar-result convention does not apply here). Fixed with a bare
     `len(pc)` call in each function's N-D dispatch branch in
     `_polynomial_legacy.py`: Python's own `len()` on anionpy's 0-d
     array already raises the identical `TypeError` with the identical
     message, so no new Rust code was needed for this defect either --
     it is dispatch-layer validation, per this file's own
     Python-is-dispatch-only architecture rule.

  C. `polyder`'s `p[:-1] * np.arange(n, 0, -1)` and `polyint`'s
     `p / np.arange(n, 0, -1)` (then `np.concatenate` with the next
     integration constant) run UNCONDITIONALLY on N-D `p` too, and
     broadcast/concatenate along the first axis against a 1-d `arange`.
     Whether that succeeds, and which of two DIFFERENT numpy error
     texts it raises if not, is shape-dependent (confirmed: a trailing
     size-1 axis broadcasts and succeeds; a mismatched trailing axis
     raises `ValueError: operands could not be broadcast together with
     shapes ...`; polyint's post-division `concatenate` step raises a
     DIFFERENT `ValueError: all the input arrays must have same number
     of dimensions...` for shapes where the division itself would have
     succeeded). Fixed by composing `polyder_legacy_nd`/
     `polyint_legacy_nd` (`ionp-core/src/poly_legacy.rs`) from
     `ufunc::binary_op` and `manip::concatenate` -- both already proven
     to reproduce numpy's exact broadcast/concatenate error text
     (trailing space, no-space tuple formatting) elsewhere in this
     crate -- rather than hand-writing new broadcast logic that would
     have to re-earn that exactness from scratch.

Verification for this landing pass: a provenance-varied probe
(`/private/tmp/probe_cases.py`) covering 2-d and 3-d `p`, scalar and
array `x`, real/float/complex dtypes, 0-d/bare-scalar `p` at multiple
`m`, and N-D `p` shapes chosen to hit BOTH the broadcast-success and
the two distinct broadcast/concatenate-failure paths -- 20 targeted
cases, run directly against real numpy 2.5.1 and the freshly rebuilt
package, 20/20 byte-for-byte equal (exception type AND message where
either side raised, dtype+shape+bytes where neither did). All 20 cases
(plus the original 1-D corpus) were then wired into
`poly1d_legacy_cases.py` and run through the actual differential
harness (`tests/differential/run.py`): `polyval`/`polyder`/`polyint`
all verdict `pass`, `mechanism: none`, `tolerant: False` (bit-exact,
not epsilon-massaged) -- `/tmp/r76b1.json`. Full-suite failure count
unchanged at 33, and by NAME: none of the 33 failures are
`polyval`/`polyder`/`polyint`/`polydiv`/`polyadd`/`polysub` or any
`polynomial.polynomial.poly{val,der,int}` sibling -- the 33 are the
same pre-existing unrelated items (`ndarray.__ipow__`/`__pow__`,
`stack`, `hypot`, `finfo`/`iinfo` construction, the modern package's
`*fromroots`/`*mul`/`*pow` family sharing ticket #45's unreproduced
convolution order, etc.) as before this pass.

A FOURTH, unrelated defect was found and deliberately NOT fixed here,
being out of ticket #76's A/B/C scope: `polyint(p, m, k)` where `k` is
itself a nested list/array (e.g. `k=[[1,2],[3,4]]`) diverges even for
plain 1-D `p` -- confirmed independent of the N-D-`p` work in this
pass (`polyint([1,1,1], 2, [[1,2],[3,4]])` diverges on both 1-D and N-D
`p`, so this is not a regression introduced by the A/B/C fix). Real
numpy raises `ValueError: all the input arrays must have same number
of dimensions...` from its own `concatenate`; anionpy's k-validation
block (unchanged by this pass, pre-existing) raises `TypeError: only
0-dimensional arrays can be converted to Python scalars` instead. Not
in this ledger, not in the differential corpus, left for a future
ticket -- `k` as a nested (non-1-d-of-scalars) sequence is a corner
usage most callers would not hit, and manufacturing a "declined" case
in the corpus for a defect nobody asked this pass to fix would be
scope creep, not diligence.

DECLARED `exact` (6 items):
  - `polydiv`: bit-exact for int64/float64/complex128 operands (the
    "big three" dtypes, matching the scope precedent set for the modern
    package's `polynomial.polynomial.*` block above in this same
    `_state` package) -- ~3600 out-of-corpus trials across varied
    shapes (1-8), leading zeros, negative values, mixed-sign
    coefficients, exact division and division-with-remainder, plus the
    500-call adversarial-ordering stress test described above, 0
    failures. float16/float32/complex64/int8/16/32/uint8/16/32 are
    EXCLUDED from this declaration (same pre-existing narrow-dtype
    upcast-without-preserving-dtype gap documented above for the other
    6 items) -- e.g. `polydiv` on float32 operands returns real numpy's
    `float32` but anionpy's `float64`; values match, dtype does not.
  - `polyadd`, `polysub`: same int64/float64/complex128 scope and the
    same narrow-dtype exclusion as `polydiv`. Evidence: 3,600
    out-of-corpus trials each (shapes 1-9, leading zeros, empty
    operands, length-1, mixed dtypes) run AFTER the 500-call
    adversarial-ordering stress, 0 divergences on dtype, shape, or
    bytes; plus the provenance-varied probe (13 input kinds incl.
    sliced views, negative strides, F-order columns, 0-d, lists, bare
    scalars, nan/inf, +-1e300) at 0 divergences INCLUDING raise
    parity -- these two are the only members of the family that hold
    on N-D input and on every error path probed. The comparison used
    was itself controlled: it was shown to catch an injected dtype
    swap, a shape swap, a value swap, and a 1-ULP float difference
    before any result was trusted (the previous pass's failure was an
    UNCONTROLLED comparison, so this is now mandatory here).
  - `polyval`, `polyder`, `polyint`: same int64/float64/complex128
    scope and the same narrow-dtype exclusion as `polydiv`/`polyadd`/
    `polysub` (class-1 above still applies unchanged to these three --
    it was never in ticket #76's scope). Defects A/B/C (N-D `p`
    rejection, wrong exception type on 0-d/scalar `p`, wrong exception
    message on `p.ndim >= 2`) are FIXED as of this pass -- see the
    "TICKET #76, second landing" section above for root causes,
    fix locations, and evidence (20/20 targeted provenance-varied
    cases plus the full differential-harness run at `/tmp/r76b1.json`,
    verdict `pass`/`mechanism: none`/`tolerant: False` for all three).
    The pre-existing narrow-dtype non-preservation gap (class-1) is
    excluded from this declaration exactly as it is for the other
    three items in this family -- e.g. `polyval` on int32 input
    returns anionpy's `float64` where real numpy returns `int64`,
    values match, dtype does not.

NOT declared (3 items, all implemented, all passing their own
committed corpus, none bit-exact out of corpus by this pass's
measurement):
  - `poly`: class-1 AND class-3 (real 1-ULP complex128 divergence).
    Not declarable without an evidenced tolerance.
  - `roots`, `polyfit` (+ the registry's `polyfit_full`/`polyfit_cov`
    variants): ~5e-15 relative error after sorting (ordering itself
    matches real numpy exactly), not zero -- see the epsilon-tolerance
    correction above.

`poly1d` (the class itself) has NO items declared here: the committed
corpus (`poly1d_legacy_cases.py`) covers only the 9 free functions
above, not any `poly1d.<method>`/dunder -- there is nothing measured to
declare a class-level item against. Declining is not a finding, just an
absence of corpus for this pass to verify against.

DECLINED, matching the implementation's own documented scope boundary
(ticket #45's un-reproduced `np.convolve` summation order) -- not
implemented at all, so not applicable to this ledger:
  `polymul` (free function), `convolve`, `correlate`, `array_str`,
  `poly1d.__mul__`/`__rmul__` (non-scalar operand), `poly1d.__pow__`
  (n > 1), `polyval(p, x)` with `x` itself a `poly1d` (composition).
"""

POLY_LEGACY_STATE = {
    "polydiv": "exact",
    "polyadd": "exact",
    "polysub": "exact",
    "polyval": "exact",
    "polyder": "exact",
    "polyint": "exact",
}
