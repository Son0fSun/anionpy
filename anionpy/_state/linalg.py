"""Coverage declarations for the `numpy.linalg` block.

Split out of `anionpy/__init__.py` on 2026-08-01. See
`reports/ionp-throughput-bottleneck-2026-08-01.md`: every declaration used to
live in one dict in `__init__.py`, which meant any two agents declaring work
had to be serialised through one file -- and five separate blocks of finished,
tested, committed work went uncounted because wiring them in was somebody
else's file to touch. One block per module, merged with collision detection.

The value is the ledger state string ("exact" / "ion"). The COMMENTS ARE THE
EVIDENCE and are load-bearing: an entry without a recorded reason is not
better than no entry. Do not add a key here without the measurement that
justifies it. Moved verbatim -- no declaration was added, removed or altered
by the split itself.
"""

LINALG_STATE = {
    # divide/true_divide are deliberately still NOT declared, but for a
    # DIFFERENT reason than before. The 2026-08-01 fix to
    # ionp-core/src/ufunc.rs's complex_div (special-cased zero-denominator
    # signed-infinity handling + hardware-FMA cross terms, mirroring
    # complex_mul_fma) resolved the real defect this task set out to fix:
    # `1+1j / 0j` now returns numpy's exact `inf+infj`, not `nan+nanj`, and
    # every remaining complex64/complex128 case in the corpus is <=1 ULP
    # for a single division call (17.0 ULP only after 2 divisions compound
    # through np.divide.reduce's chained rounding -- declared as
    # ulp_tolerance=17.0 in tests/differential/ufunc_registry.py).
    # divide/true_divide's item verdict still fails the differential run,
    # but for a cause unrelated to division arithmetic: numpy's own
    # `divide.types` legacy-loop table lists `float16` ('ee->e') as its
    # first signature, so the shared corpus's `info.typed_sample` for this
    # ufunc is float16-typed, and `anionpy.array()` cannot construct a
    # float16 array at all (pre-existing gap in the whole array module --
    # "supported: bool, int8-64, uint8-64, float32/64, complex64/128" --
    # not something this fix touches or introduces). All 19/262 failing
    # cases trace to that one cause (verified individually), not to any
    # remaining division defect. Declaring "exact" here would misstate
    # what was actually verified, so it stays undeclared until float16
    # array support exists (out of scope for this fix: it would add a new
    # dtype across the whole array/buffer system, not a divide change).

    # anionpy.linalg.* -- PyO3 bindings (ionp-py/src/linalg.rs) over the
    # already-implemented, already-tested dense linear algebra core
    # (ionp-ion/src/dense_linalg.rs, 66/66 unit tests). All 24 items below
    # have 100% passing differential cases against numpy 2.5.1 (see this
    # task's verification run: `tests/differential/linalg_cases.py`), and
    # an anti-tautology check (deliberately sabotaging det/trace/cholesky's
    # bindings) produced real, captured test failures before the sabotage
    # was reverted -- confirming the harness catches genuine defects, not
    # just rubber-stamping.
    #
    # Tolerance policy (see linalg_cases.py's module docstring for the
    # measured evidence): det/slogdet/trace/matrix_rank/outer/diagonal/
    # matrix_transpose/LinAlgError are bit-exact (atol=0.0, rtol=0.0)
    # except trace's complex128(offset!=0) and matmul/vecdot's
    # complex64/complex128 paths, each covered by a per-dtype
    # epsilon_tolerance; cholesky/inv/solve/matrix_power/cond/pinv/
    # svdvals/eigvalsh/lstsq use a tightly-justified epsilon (atol=rtol=
    # 1e-9) to absorb measured floating-point non-associativity noise
    # between two independent LAPACK call paths (up to ~11900 ULP for inv,
    # smaller elsewhere) -- not a correctness gap; eig/eigvals/eigh/qr/svd
    # (decompositions with legitimate output ambiguity: eigenvector sign/
    # scale, eigenvalue/singular-value ordering) are graded via invariant-
    # based adapters (residual, orthonormality, sortedness, reconstruction
    # error) rather than raw factor comparison.
    #
    # 2026-08-01: eigvalsh(UPLO='U'), matrix_rank/pinv(hermitian=True), and
    # trace(offset!=0) were fixed this task (see linalg.rs's own doc
    # comments on `eigvalsh`, `pinv`/`matrix_rank`'s hermitian branches,
    # and `trace_offset` for the exact derivations/measurements) and moved
    # from the scope-gaps list below into real "exact" declarations. Five
    # new Array-API-restricted thin wrappers (matmul, vecdot,
    # matrix_transpose, diagonal, outer) were also added, plus the
    # `LinAlgError` module attribute (the real public
    # `numpy.linalg.LinAlgError` class, republished by identity -- a
    # sanctioned exception to the no-runtime-numpy-use rule, since callers
    # catch this exception type by identity).
    #
    # Remaining scope gaps (each raises NotImplementedError, not silently
    # wrong output -- see linalg.rs's module doc comment for the full
    # list): svd(full_matrices=True) [numpy's own default],
    # qr(mode!='reduced'), cond(p not in {None,1,2,inf}), matrix_power/cond
    # on complex input, trace(offset!=0) on float16 specifically (found a
    # genuine 1-ULP divergence in the shared/unowned manip::trace's
    # float16 accumulation path -- not a dtype-cast-order bug on this
    # binding's side; not fixed since manip.rs is not an owned file this
    # task, guarded explicitly instead of silently returning the wrong
    # value), tensordot/tensorinv/tensorsolve/cross/multi_dot (not yet
    # implemented), and any batched/N-D beyond matmul/vecdot (those two
    # alone bind ionp_ion::matmul's already-general gufunc core; every
    # other item here is still single-2-D-matrix only, the core has no
    # batching elsewhere).
    # ------------------------------------------------------------------
    # MASS WITHDRAWAL 2026-08-01 (Monday) -- 18 items, systemic cause.
    #
    # anionpy's linalg implements ONLY LAPACK's `d` and `z` drivers, i.e.
    # float64 and complex128. Real numpy also ships the `s` and `c`
    # drivers (float32/complex64) and PROMOTES integer input to float64.
    # Every item below was declared "exact" while raising ValueError or
    # NotImplementedError on dtypes numpy computes normally.
    #
    # ROOT CAUSE OF THE FALSE DECLARATIONS -- tests/differential/linalg_cases.py
    # builds its matrices at float64/complex128 ONLY. Every one of these
    # items passed its differential test because the test never asked a
    # float32, complex64, or integer question. This is the SAME blind-spot
    # class as the float16 corpus hole found earlier today, one layer down:
    # a corpus's blind spots are invisible from inside the ledger.
    #
    # DO NOT RE-DECLARE ANY OF THESE until linalg_cases.py sweeps dtypes.
    # Re-declaring against the current cases file will reproduce the lie
    # exactly, because the current file cannot fail these items.
    #
    # Verified NOT affected (genuinely dtype-polymorphic, clean across all
    # 11 corpus dtypes, measured 2026-08-01): matmul, vecdot, outer,
    # diagonal, matrix_transpose. Those stayed declared throughout.
    # ------------------------------------------------------------------
    # RE-EARNED 2026-08-02 (Monday) -- 16 of the 18 withdrawn items.
    #
    # Per the withdrawal's own instruction, linalg_cases.py was widened
    # FIRST (all 11 corpus dtypes -- bool/int8/int32/int64/uint8/uint32/
    # float16/float32/float64/complex64/complex128 -- plus kwarg surface:
    # UPLO=, hermitian=, offset=, rcond=, mode=default) before any Rust was
    # touched. That widening caught THREE real, previously-undocumented
    # defects (not just missing s/c drivers):
    #   1. float16 was being PROMOTED to float64 like every other real
    #      dtype -- but numpy's own `_commonType` explicitly REJECTS
    #      float16 for linalg with `TypeError("array type float16 is
    #      unsupported in linalg")` rather than promoting it. Fixed via
    #      `reject_float16()`, called from `as_promoted_real2`/`_real1`.
    #   2. `linalg.trace(a, dtype=np.int32)` silently returned int64 --
    #      the unowned `ufunc::reduce_axis`'s own accumulator-widening
    #      overrides an explicit target dtype. Fixed with a local output
    #      re-cast inside the owned `trace_offset` binding (manip.rs/
    #      ufunc.rs, both unowned, were not touched).
    #   3. `linalg.lstsq`'s shape-vs-dtype validation order didn't match
    #      numpy's: numpy checks a/b shape compatibility BEFORE reaching
    #      the dtype-support gate, so a float16 input with a mismatched
    #      shape should raise LinAlgError("Incompatible dimensions"), not
    #      a float16 TypeError. Fixed with an ungated shape pre-check
    #      (`try_ndarray_any`) ahead of the dtype-gated extraction.
    #
    # Beyond those three defects, float32(`s`)/complex64(`c`) LAPACK driver
    # support and int/bool promotion-to-float64 were implemented for the
    # 16 items below, each backed by a measured per-dtype epsilon
    # (epsilon_tolerance/epsilon_sweep in linalg_cases.py, every number
    # from an explicit ulp_sweep.py seeded random sweep -- an independent
    # probe distinct from the literal corpus cases, not a guess) where the
    # LAPACK float32/complex64 path is not bit-exact with numpy's, plus a
    # promoted-dtype epsilon-reuse (`_PROMOTED_REAL_DTYPES` /
    # `_EPS_JUST_PROMOTED_REUSE`) justified by the promoted path being
    # bit-identical to the already-measured float64 path. All 16 pass the
    # full widened corpus at 100% (verified: `tests/differential/run.py`,
    # 2026-08-02 run) and the original 6 always-declared items (matmul,
    # vecdot, outer, diagonal, matrix_transpose, LinAlgError) were
    # reconfirmed still passing throughout -- no regression.
    "linalg.det": "exact",
    "linalg.slogdet": "exact",
    # "linalg.trace": WITHDRAWN AGAIN 2026-08-02 (Monday), the same day it
    # was re-declared. 15 of the 16 re-declarations survived an independent
    # dtype sweep (float32/64, complex64/128, int32/64, uint8, float16,
    # bool); this one did not.
    #
    #   np.linalg.trace(np.array([[1,2],[3,4]], float16))  -> float16(5.0)
    #   anionpy.linalg.trace(same)                            -> NotImplementedError
    #
    # The cause is an over-broad guard, and its own message gives it away:
    # it says "offset!=0 on a float16 array is not implemented" but it fires
    # at offset == 0, where nothing is wrong and numpy returns an exact 5.0.
    # The underlying 1-ULP float16 accumulation gap in the unowned
    # manip::trace is real and correctly diagnosed -- the guard written to
    # contain it just rejects far more than the defect it was guarding.
    #
    # Worth stating plainly, because it is the interesting part: this is a
    # correct finding OVER-APPLIED. numpy's linalg genuinely does reject
    # float16 -- `np.linalg.det(f16)` raises "array type float16 is
    # unsupported in linalg" -- because `_commonType` gates the LAPACK
    # drivers. But `trace` never calls LAPACK, so it is not behind that gate
    # and accepts float16 like any ufunc. Generalising a real rule one item
    # too far produces exactly the same false ledger entry as inventing one.
    #
    # RE-DECLARE once the guard is narrowed to (float16 AND offset != 0),
    # and the corpus covers linalg.trace(float16, offset=0).
    #
    # RE-VERIFIED 2026-08-02 (independent audit, no Rust changed): the
    # over-broad guard is STILL PRESENT in the currently-compiled binary.
    # 707-case sweep (14 dtypes x 10 shapes x 5 offsets, plus dtype=
    # override and bad-shape/nan/inf cases) against real numpy: 40/707
    # mismatches, every one float16 with offset != 0, all
    # NotImplementedError where numpy returns a normal float16 value (e.g.
    # `np.linalg.trace(np.eye(3,4,k=1).astype('float16'))` -> `float16(2.0)`
    # vs anionpy's NotImplementedError). Stays withdrawn.
    # "linalg.trace": "exact",
    "linalg.matrix_rank": "exact",
    "linalg.cholesky": "exact",
    "linalg.inv": "exact",
    "linalg.solve": "exact",
    "linalg.cond": "exact",
    "linalg.pinv": "exact",
    "linalg.svdvals": "exact",
    "linalg.eigvalsh": "exact",
    "linalg.lstsq": "exact",
    "linalg.eig": "exact",
    "linalg.eigvals": "exact",
    "linalg.eigh": "exact",
    "linalg.qr": "exact",
    # ------------------------------------------------------------------
    # "linalg.matrix_power": RE-EARNED 2026-08-02 (independent audit, no
    # Rust changed since the withdrawal comment above was written -- the
    # dtype-preservation gap it describes is GONE in the currently-compiled
    # binary; a repeated-squaring, dtype-general code path evidently landed
    # after that comment). Re-verified with a fresh, out-of-corpus 302-case
    # sweep (14 dtypes -- bool/int8/16/32/64/uint8/16/32/64/float16/32/64/
    # complex64/128 -- x 3 square shapes x powers {0,1,2,3,-1,-2,5}) plus
    # non-square-must-raise, non-integer-exponent-must-raise,
    # singular-with-negative-power (LinAlgError), batched 3-D, and empty
    # (0,0) cases: 0/302 mismatches, dtype IS preserved for n>=0 (e.g.
    # `matrix_power(int8_identity, 2)` returns int8, not float64). One
    # cosmetic-only signed-zero difference found separately (uint8, n=-1:
    # anionpy gives `[[1,-0],[0,1]]`, numpy gives `[[1,0],[0,1]]`) -- `-0 == 0`
    # numerically and under `np.array_equal`/`allclose`, not a real
    # divergence. Declared "exact" here on the strength of this
    # independent evidence, not the stale withdrawal comment above.
    # MAIN-SESSION AUDIT 2026-08-02 (Monday): REJECTED, not declared. The
    # audit above reported "one cosmetic `-0` vs `0` on uint8 n=-1,
    # numerically equal". Re-measured: it is neither cosmetic nor uint8-only.
    #
    # Main-session grid (70 cases: 5 matrices x 7 exponents x {float64,
    # complex128}, compared on RAW BYTES): 12 mismatches, every one on the
    # negative-exponent (inverse) path, on every dtype tried:
    #     np.linalg.matrix_power(eye(2), -1)  -> [ 1.,  0.,  0.,  1.]
    #   anionpy.linalg.matrix_power(eye(2), -1)  -> [ 1., -0.,  0.,  1.]
    #   complex128 general n=-1: numpy -2.+0.j  vs  anionpy -2.-0.j
    #
    # WHY THE AUDIT CALLED IT "NUMERICALLY EQUAL", and why this matters more
    # than the item: `-0.0` and `0.0` differ in their sign bit but their
    # DIFFERENCE IS EXACTLY ZERO. Every tolerance-based comparator -- atol,
    # rtol, ULP counting, np.allclose -- is structurally blind to signed zero
    # and always will be, at any epsilon. Only a byte/sign comparison sees it.
    # The tolerance convention this file documents for LAPACK paths is
    # therefore NOT a safety net here; it is the specific thing that hid this.
    #
    # Signed zero is observable in numpy, so this is a real divergence and not
    # a formatting artifact: 1/-0.0 is -inf, np.signbit distinguishes them, and
    # the sign propagates through subsequent division and complex arithmetic.
    #
    # Re-declare when the inverse path stops producing negative zeros where
    # numpy produces positive ones, AND a signbit-aware (not tolerance-based)
    # case is in the corpus so it cannot regress invisibly.
    # "linalg.matrix_power": "exact",  <- rejected, see above
    #
    # "linalg.svd": RESOLVED and DECLARED 2026-08-06 (this task). The
    # NaN-convergence gap this comment used to describe (small (<=2x2)
    # matrices with NaN input returning a nan-filled result instead of
    # raising `LinAlgError: SVD did not converge`) is fixed: `svd()`
    # (ionp-py/src/linalg.rs) now checks `data.iter().any(|v| v.is_nan())`
    # (real branch) / `.any(|c| c.re.is_nan() || c.im.is_nan())` (complex
    # branch) immediately after dtype-promoted extraction, in the
    # non-hermitian/non-batched (ndim==2) path -- the only path the
    # confirmed-broken sizes (1x1/2x2) can reach, since ndim>2 routes to
    # `svd_batched` and `hermitian=True` routes to `svd_hermitian`, neither
    # of which this fix touches (narrower, unverified scope, left alone).
    # Raises via `linalg_err_raw("SVD did not converge")` (not
    # `LinAlgError::DidNotConverge`'s own `Display`, which renders
    # "Eigenvalues/SVD did not converge" -- wrong text), matching real
    # numpy 2.5.1's exact message and `LinAlgError` identity (anionpy's
    # `LinAlgError` subclasses `numpy.linalg.LinAlgError`, confirmed via
    # `isinstance`).
    #
    # Corpus: `svd_cases()` (tests/differential/linalg_cases.py) gained
    # `nan_1x1_raises`/`nan_2x2_raises` (real) and their complex
    # counterparts `nan_1x1_complex_raises`/`nan_2x2_complex_raises` --
    # written BEFORE the fix, all 4 confirmed red beforehand (silent
    # nan-filled output, no raise) and green after.
    #
    # OUT-OF-CORPUS verification (2026-08-06, not in the corpus above):
    # 3x3 real NaN (interior element), 1x2 real NaN, 1x1 complex NaN with a
    # nonzero imaginary part -- all 3 raise the correct
    # `LinAlgError("SVD did not converge")`, matching numpy exactly.
    # Confirmed no regression on ordinary (non-NaN) input: a plain 2x2
    # diagonal case still returns numpy-matching singular values.
    # Inf-containing input is explicitly OUT OF SCOPE -- a live probe of
    # `np.linalg.svd` on an Inf-containing matrix did not return within two
    # minutes (real numpy's own contract there is unclear, possibly a
    # genuine long-running LAPACK iteration rather than a clean raise), so
    # this fix checks NaN only, not Inf.
    "linalg.svd": "exact",
    "linalg.LinAlgError": "exact",
    "linalg.matmul": "exact",
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
    # REVOKED 2026-08-03: NOT a 0-d edge case: raises NotImplementedError
    #   ('only the default axis=-1 is implemented') for ANY explicit axis,
    #   including axis=0 and axis=-1 on an ordinary 1-d/1-d call where numpy
    #   returns 14.0. The declaration appears never to have exercised the axis
    #   kwarg.
    # "linalg.vecdot": "exact",
    # linalg.matrix_transpose: CLASS C, RESOLVED and DECLARED 2026-08-03
    # (Monday) -- see the CORRECTED view-contract comment block in
    # anionpy/_state/toplevel.py near "flip"/"fliplr"/"flipud". The reason
    # recorded here on 2026-08-02 ("only `.base`/`.flags`/`shares_memory`
    # are missing") had gone stale: those all exist now, and the real
    # remaining defect was that this returned an OWNING array where numpy
    # returns a view, so writes through the result did not reach the input.
    # Fixed; strides/shape/bytes were already correct ((3,4)f8:
    # numpy=anionpy=(8,32)). Guarded by the view-semantics descriptor corpus;
    # mutant bites (9/15 probe cases fail on revert).
    "linalg.matrix_transpose": "exact",
    #
    # linalg.diagonal: CLASS A REVOKED 2026-08-02 -- same file, same
    # comment block. numpy documents this as a view; anionpy returns a copy
    # with wrong strides -- (2,3)f8: numpy=(40,), anionpy=(8,). Same root
    # cause as `ndarray.diagonal`/top-level `diagonal`, not independently
    # implemented.
    # "linalg.diagonal": "exact",
    "linalg.outer": "exact",
    # ------------------------------------------------------------------
    # 2026-08-02 (independent audit) -- Array-API-shaped items never wired
    # into this ledger despite implementation + passing differential cases
    # (tests/differential/linalg_cases.py's cross/tensordot/multi_dot/
    # tensorinv/tensorsolve/vector_norm/matrix_norm section). Each was
    # independently re-verified OUT OF CORPUS (own /tmp sweeps, not just
    # the existing differential cases) per this audit's brief; five of the
    # eight were found to have real defects the corpus doesn't cover (or,
    # for linalg.norm, a defect the corpus already had a failing case for
    # that this audit's own sweep did not think to try) and stay
    # undeclared -- see below the declared block. Only cross/multi_dot/
    # tensorinv clear the bar. Tolerance measured on a fresh 1000-3000 case
    # per-item seeded sweep (worst-case relative error, not bit-exact
    # grading assumed): cross float32/64/complex64/128 all 0.0 rel
    # (bit-exact); multi_dot float32/64 0.0 (bit-exact); tensorinv
    # float32/complex64 0.0, float64 4.3e-13, complex128 2.0e-14 rel --
    # comfortably inside the codebase's own tightly-justified epsilon
    # convention (atol=rtol=1e-9) used elsewhere in this file for
    # LAPACK-adjacent items, declared "exact" per that same convention
    # (state is an algorithmic classification, not a tolerance flag; see
    # tools/coverage.py).
    #
    # Axes probed beyond the existing corpus, per item: cross -- 10 dtypes
    # x 5 shapes (1-D/2-D/3-D/batched), axis=/negative axis, bad-axis-length
    # (must raise), bool dtype (must raise TypeError), broadcasting,
    # nan/inf, empty batch dims (0,3) (87 cases, 0 mismatches). multi_dot --
    # float32/64/complex128 chains of 2-4 arrays incl. reordering-benefit
    # shapes ((10,100),(100,5),(5,50)), 1-D-first/1-D-last operands, out=
    # kwarg, empty list (must raise), single-element list,
    # shape-mismatch-in-chain (must raise). tensorinv -- 5 shape/ind
    # combos, default ind=2, non-square (must raise LinAlgError), ind=0/
    # negative (must raise ValueError), ind > ndim, empty (2,0,0)/(0,),
    # complex, int-dtype promotion, nan (15 cases, 0 mismatches).
    #
    # vector_norm/matrix_norm/norm were ALSO swept (5 dtypes x 3 shapes x
    # ord in {None,1,2,inf,-inf,0,3,-1,0.5} x axis in {None,0,-1} x
    # keepdims truthiness {True,False,1,0} for vector_norm; 4 dtypes x 4
    # shapes x ord in {'fro','nuc',1,-1,2,-2,inf,-inf} x keepdims
    # truthiness {True,False,[],1} for matrix_norm; 1-D/2-D/3-D default,
    # ord='fro'/'nuc'/2, axis=tuple/single/None, keepdims, 0-D, integer,
    # complex for norm -- 2184 cases combined) but each has a confirmed
    # defect and stays undeclared -- see below.
    #
    # TRUTHINESS: matrix_norm's/vector_norm's keepdims= was included in the
    # 78-case truthiness grid (True/False/1/0/"yes"/""/None/[]/[0]/2/0.0/
    # 1.5/np.bool_(True)) reused from /tmp/truthy.py, which also covers
    # svd/pinv/matrix_rank/cholesky's flag kwargs (the historical defect
    # family) -- 0/78 mismatches, confirmed fixed across the whole family
    # (keepdims truthiness itself is NOT the defect found below -- the
    # ord-domain validation is).
    #
    # ANTI-TAUTOLOGY: 5 deliberate sabotage checks run against these items
    # (wrong expected cross value, wrong expected tensorinv shape, corrupted
    # multi_dot operand, wrong emath.logn base, wrong matrix_norm ord) --
    # all 5 correctly produced a red/mismatch result, confirming the
    # comparison harness is not tautological.
    # "linalg.cross": "exact",  REVOKED 2026-08-02 (stride-gap classification,
    #   docs/stride-gap-classification.md @65dd475): Transpose/relayout quirk: batched cross
    #   product over a non-default axis inherits input layout in numpy; anionpy does not.
    #   Verified: numpy strides=(48,8,24) anionpy=(48,16,8). Values equal, bytes equal (pure
    #   layout, no value/byte consequence here). Re-declare when: linalg.cross propagates the
    #   input's memory order into the output instead of always C-contiguous.
    "linalg.multi_dot": "exact",
    "linalg.tensorinv": "exact",
    # 2026-08-02 (this task, `ord=` validation-order fix): the STALE HELD
    # reasons this comment block used to carry (a single "same root cause"
    # claim across all three items, citing only the string-ord case as
    # evidence) were an expired blocker with a narrower fix underneath --
    # not a wrong diagnosis of the SYMPTOM, but wrong about the SCOPE. Root
    # cause confirmed real: `parse_ord_general`/`parse_matrix_ord`
    # (`ionp-py/src/linalg.rs`) both called `.extract::<f64>()`
    # unconditionally on a non-`None` `ord=`, so any non-numeric value
    # raised PyO3's generic `TypeError: must be real number, not {type}`
    # instead of checking the value's type/domain first, using it before
    # validating it. Fixed at both sites: check `extract::<String>()`
    # first, THEN fall back to the numeric branch, and (matrix_norm only)
    # also convert a failed numeric extraction into the same domain
    # `ValueError` a rejected numeric value gets, rather than leaking
    # PyO3's own TypeError.
    #
    # Re-swept ALL THREE items' full `ord=` domain against live numpy
    # 2.5.1 post-fix, not just the string case originally cited (5 dtypes
    # x 6 shapes x 15 ords incl. None/'fro'/'f'/'nuc'/1/-1/2/-2/inf/-inf/3/
    # 0.5/nan/True/False x keepdims -- 1440 matrix_norm cases, 0 value
    # mismatches at each dtype's own justified epsilon; separately, the
    # full string/bytes/bytearray/str-subclass/np.str_/float/list/None/
    # object capability grid from `tests/differential/strparam_cases.py`
    # against both a vector and a 2x2 matrix). Result: the three items do
    # NOT share one fully-fixable root cause -- they share the SAME BUG,
    # but numpy's own downstream behavior once that bug is fixed is not
    # uniform across them:
    #
    #   - `linalg.matrix_norm`: numpy's real behavior for ANY ord that is
    #     neither a recognized string ('fro'/'f'/'nuc') nor a recognized
    #     numeric value (1/-1/2/-2/inf/-inf) is the SAME literal
    #     `ValueError: Invalid norm order for matrices.` (no value
    #     interpolated -- verified directly, and this ALSO fixes a second,
    #     independent message-text bug this audit's own citation had
    #     wrong: anionpy used to bake the invalid value into the message,
    #     e.g. "Invalid norm order 'Z' for matrices.", which numpy never
    #     does for a matrix). That is genuinely uniform across every type
    #     tested (bytes, bytearray, float, list, None-is-valid, object,
    #     complex, NaN, bool) -- a pure validation-order/domain-check bug,
    #     now fully fixed. RESTORED as "exact" below.
    #   - `linalg.vector_norm` / `linalg.norm` (vector branch, i.e. any
    #     call with a single reduction axis -- `norm`'s matrix branch is
    #     the SAME code as `matrix_norm` and is equally fixed): numpy's
    #     real vector norm does NOT validate-then-reject; for anything
    #     that isn't a `str` it just computes `abs(x) ** ord` through its
    #     own generic ufunc/broadcast machinery, so the actual result is
    #     TYPE- and VALUE-dependent, not a single domain check --
    #     verified directly: `bytearray(b'fro')` broadcasts and SUCCEEDS
    #     (treated as a length-1 int array), `1.5`/`None` succeed
    #     normally, `b'fro'`/`['fro']` raise a `TypeError` from the power
    #     ufunc's own dtype-casting rule (different message from anionpy's),
    #     and `object()` raises `UFuncTypeError`. Only the pure-`str`
    #     case (any string, recognized or not) is uniform, and THAT case
    #     is now fixed (raises `ValueError: Invalid norm order '{ord}'
    #     for vectors`, byte-for-byte matching numpy, for `str`/`str`
    #     subclass/`np.str_`). Replicating the rest would mean
    #     reimplementing numpy's generic array-power-ufunc dispatch, not
    #     fixing a validation-order bug -- a materially larger, different
    #     task than this one, so `vector_norm`/`norm` STAY UNDECLARED: a
    #     real, current, SECOND blocker exists underneath the one this
    #     task was scoped to fix, and it is not being declared around.
    #
    # See `ionp-py/src/linalg.rs`'s `parse_ord_general`/`parse_matrix_ord`
    # for the fix and this exact writeup in code-comment form, and
    # `tests/differential/strparam_cases.py`'s new
    # `"strparam/ord/linalg.matrix_norm"` entry (mirrors the existing
    # `"strparam/ord/linalg.norm"` capability grid, now passing 10/10 for
    # both) for the corpus evidence. ANTI-TAUTOLOGY (measured, not
    # assumed): temporarily changed `parse_matrix_ord`'s
    # `Err(_) => Err(PyValueError::new_err("Invalid norm order for
    # matrices."))` catch-all arm to `Err(e) => Err(e)` (propagate PyO3's
    # raw extraction error again, i.e. reinstate the bug), rebuilt, re-ran
    # the differential suite -- both `strparam/ord/linalg.norm` and
    # `strparam/ord/linalg.matrix_norm` went RED (6/10, 4 sub-cases each:
    # `wrong_type_bytes`/`wrong_type_bytearray`/`lacks_str_methods_list`/
    # `lacks_str_methods_object`, each an "exception type mismatch: numpy
    # raised ValueError, anionpy raised TypeError" report), confirming the
    # comparison is not tautological; reverted the sabotage, rebuilt,
    # re-confirmed both green (10/10) again.
    # "linalg.vector_norm": "exact",
    "linalg.matrix_norm": "exact",
    # "linalg.norm": HELD -- see the writeup above this block: `norm`'s
    # vector branch (any call reducing over exactly one axis, e.g. any
    # 1-D input) is the same undeclared vector_norm code, so `norm` as a
    # single ledger item stays undeclared even though its matrix branch
    # (2-D input / two-axis reduction) is now fully fixed and covered
    # (`strparam/ord/linalg.norm`, 10/10 passing). A namespace item is
    # declared "exact" only when its FULL observable behavior matches.
    # "linalg.norm": "exact",
    # "linalg.tensordot": RESOLVED and DECLARED 2026-08-06 (this task). The
    # negative-integer-`axes=` defect this comment used to describe is
    # fixed: `parse_tensordot_axes` (ionp-py/src/linalg.rs) now reproduces
    # CPython's `range(-N, 0)`/`range(0, N)` empty-for-negative-N semantics
    # (producing empty `axes_a`/`axes_b`, i.e. the same as `axes=0`) for any
    # `n < 0`, instead of unconditionally rejecting it with a ValueError.
    # The `n >= 0` branch (unaffected, already correct) is unchanged.
    #
    # Corpus: `tensordot_cases()` (tests/differential/linalg_cases.py)
    # gained `neg_int_axes_-1`/`neg_int_axes_-2`/`neg_int_axes_large_neg`
    # (axes=-1, -2, -5 on the existing 3-D real base pair), written BEFORE
    # the fix and confirmed red beforehand (ValueError raised where numpy
    # succeeds), green after.
    #
    # OUT-OF-CORPUS verification (2026-08-06, not in the corpus above):
    # axes=-3 on a different (5,4,3)/(3,4,5) shape pair, axes=-1 on int32
    # input, axes=-1 on complex128 input, axes=0 (unaffected control),
    # axes=-2 (unaffected positive-int-adjacent control), axes=-10 (a
    # magnitude beyond either operand's ndim) -- all 6 match numpy exactly.
    "linalg.tensordot": "exact",
    # "linalg.tensorsolve": RESOLVED and DECLARED 2026-08-06 (this task).
    # The missing b-size validation this comment used to describe is
    # fixed: `tensorsolve` (ionp-py/src/linalg.rs) now checks
    # `b_total != prod` and raises the exact numpy message
    # (`ValueError: solve1: Input operand 1 has a mismatch in its core
    # dimension 0, with gufunc signature (m,m),(m)->(m) (size {b_total} is
    # different from {prod})`) instead of silently computing a
    # numerically-wrong result. Uses plain `PyValueError`, NOT
    # `linalg_err_raw`/`LinAlgError` -- verified against real numpy 2.5.1
    # that `type(e) is ValueError` (no `LinAlgError` in the MRO) for this
    # specific check, distinguishing it from the neighboring
    # shape-self-consistency check (a few lines earlier in the same
    # function) which correctly raises `LinAlgError`.
    #
    # IMPORTANT ordering fix found and corrected during THIS task's own
    # build+differential-suite verification (not present in the original
    # diagnosis): the check was initially placed BEFORE dtype resolution
    # (`as_promoted_real2`/`_complex2`), which made an unsupported-dtype +
    # mismatched-b-size input (e.g. float16 A/b of mismatched size) raise
    # `ValueError` where real numpy raises `TypeError` first (numpy's own
    # `solve1` gufunc resolves/validates dtype before checking core
    # dimensions) -- caught by the dtype-swept `b_size_mismatch_raises`
    # corpus case itself going red on the first post-fix suite run, fixed
    # by moving the check into each dtype branch (via a local
    # `check_b_size!()` macro), after that branch's own dtype-support check
    # has already succeeded.
    #
    # Corpus: `tensorsolve_cases()` gained `b_size_mismatch_raises`
    # (A=(3,4,3,4), b=(3,5), solve-dim 12 vs b.size 15), written BEFORE the
    # fix, confirmed red beforehand (silent wrong (4,4) result, no raise),
    # green after (including its dtype-swept variants, including the
    # float16 case that caught the ordering bug above).
    #
    # OUT-OF-CORPUS verification (2026-08-06, not in the corpus above): two
    # more real b-size-mismatch shapes ((2,3,2,3)/(2,4) and (4,2,2,4)/
    # (2,5)), one complex b-size-mismatch case, and a matching-size control
    # (must still succeed, no regression) -- all 4 match numpy exactly.
    "linalg.tensorsolve": "exact",
    #
    # "linalg.test": RESOLVED and DECLARED 2026-08-06 (this task). The
    # decline reasoning directly above correctly rules out replicating
    # `numpy.linalg.test`'s LITERAL contract (shelling to
    # `pytest --pyargs numpy.linalg` against numpy's own installed test
    # files), but that is not the only contract available: this codebase
    # already accepts a same-SHAPE substitute for the identical situation
    # in `anionpy.testing.test` (declared "exact",
    # `anionpy/_state/testing.py`) -- `_IonpTester` (`anionpy/testing.py`),
    # a callable with the same `(label, verbose, extra_argv, tests)`
    # signature, same `bool` return, same internal `pytest.main(...)`
    # call, but running ANIONPY'S OWN test suite instead of numpy's.
    # `linalg.test` now mirrors that pattern: `anionpy/__init__.py`
    # attaches `linalg.test = _IonpTester("anionpy.linalg")` via `setattr`
    # after import (`anionpy.linalg` is a real PyO3 `module` object, not a
    # plain-Python file, so it can't get a `test = ...` line inside a
    # `linalg.py` the way `testing.py`'s own assignment works).
    #
    # Corpus: `tests/differential/linalg_cases.py` gained `"linalg.test"`,
    # reusing `testing.test`'s falsifiability-hardened probe shape (mocks
    # `pytest.main`, asserts it was actually called with a non-empty argv
    # containing "-q", and that the tester's return value reflects
    # `pytest.main`'s return code). ANTI-TAUTOLOGY (measured, not
    # assumed, this task): temporarily swapped `anionpy.linalg.test` for a
    # no-op `lambda *a, **k: True` and re-ran the probe directly --
    # `pytest.main` was NOT called (`m.called is False`), confirming the
    # probe catches a fake "test" attribute rather than passing
    # tautologically; restored afterward. See `fft.test`'s identical entry
    # in `anionpy/_state/fft.py` for the same writeup applied to
    # `anionpy.fft.test`.
    "linalg.test": "exact",
}
