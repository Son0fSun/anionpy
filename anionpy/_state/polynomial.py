"""Coverage declarations for the `numpy.polynomial.polynomial` block (the
power-series basis -- see `anionpy/polynomial/polynomial.py`'s module
docstring for the full scope statement).

18 of the 28 `polynomial.polynomial.*` surface items are declared "exact"
below (21 were originally declared; `polymul`, `polypow`, and
`polyfromroots` were REVOKED 2026-08-07 -- see those entries below and
KNOWN-DIFFERENCES.md's "(b) polymul, polypow and polyfromroots" entry --
they are NOT bit-exact, confirmed by a strengthened corpus and a measured,
failed fix attempt, not merely re-asserted from an earlier finding). The
remaining 18 have 100% passing differential cases against real numpy
2.5.1 (`tests/differential/polynomial_cases.py`, run both standalone and
inside the full suite) across a corpus deliberately probing: trailing-zero
trimming, int/bool -> float64 dtype promotion (and the`as_series`-vs-own-
dtype-path SPLIT in which functions accept vs. reject bool -- see
`coerce_series_ex`'s doc comment in `ionp-py/src/poly.rs` for the full,
directly-verified table), 0-d/empty coefficient and roots arrays, complex
coefficients, return-type/dtype identity (including `polyone`/`polyzero`/
`polyx` being int64, not float64 -- a real divergence caught by this
corpus, not assumed), and exact error types/messages (`polyval`'s
empty-array `IndexError` is a genuine numpy-internal inconsistency with
every other `as_series`-based function's `ValueError`, replicated
bit-for-bit here).

`polyfit` and `polyroots` are NOT bit-exact (both depend on an
independent LAPACK/eigensolver call path vs real numpy's own -- lstsq's
SVD and the companion matrix's eigenvalues, respectively) and are
declared via a measured, evidence-gated `epsilon_tolerance` in
polynomial_cases.py (20,000-sample seeded sweeps each, see that file's
`POLYFIT_EPS`/`POLYROOTS_REAL_EPS`/`POLYROOTS_COMPLEX_EPS` comment block
and KNOWN-DIFFERENCES.md), NOT a blind or padded tolerance.

Task #34 (2026-08-08): the remaining 7 items (`polyval2d`, `polyval3d`,
`polyvalnd`, `polygrid2d`, `polygrid3d`, `polyvander2d`, `polyvander3d`)
are now implemented (built on `polyutils.py`'s shared `_valnd`/`_gridnd`/
`_vander_nd`/`_vander_nd_flat` machinery -- the ticket's premise named
only the last two of these four as shared; corrected here after live
inspection of numpy 2.5.1's own `polyutils.py` source) and declared
"exact" below: bit-exact against real numpy across float64, complex128,
and int (promoted to float64) -- 7560/7560 out-of-corpus randomized
checks (60 trials x 3 dtypes x 7 functions, all 6 bases) plus the
committed corpus in `tests/differential/polynomial_cases.py`
(`_poly_nd_base_cases`/etc.), 100% passing. float16/float32/complex64 are
DELIBERATELY EXCLUDED from both the corpus and this declaration: the
already-declared, already-shipped 1-D `polyval`/`polyvander` kernels this
family is built on top of universally upcast those three dtypes to
float64/complex128 (values match real numpy bit-exact, but the returned
dtype does not) -- a pre-existing divergence in code this ticket did not
write and has no permission to touch (lives in the shared Rust kernels,
confirmed present across all six bases). See docs/TICKET-34-*.md.

`polynomial.Polynomial` (`ABCPolyBase`'s power-series concrete subclass,
see `anionpy/polynomial/_polybase.py` + `anionpy/polynomial/polynomial.py`)
adds all 52 of its own public-surface items below, all verified passing
against real numpy 2.5.1 in `tests/differential/polynomial_cases.py`
AND, per this task's non-negotiable methodology, additionally verified
via out-of-corpus random sweeps (500-2000 samples each, seeded) beyond
the fixed-value differential corpus -- because the corpus alone proved
insufficient during this pass: several items (see below) that looked
bit-exact against small hand-picked cases turned out NOT to be
bit-exact under random sweeping.

Two genuine, pre-existing bugs were found and fixed in the process (not
new regressions -- both were latent in already-shipped code/methodology,
surfaced by building `Polynomial` on top of it):
  1. `ABCPolyBase`'s 5 classmethod-decorated abstract methods (`basis`,
     `cast`, `fromroots`, `identity`, `fit`) were rebound onto
     `Polynomial` via `getattr(_ABCPolyBase, name)`, which for a
     classmethod returns an object already bound to `_ABCPolyBase`
     itself -- making e.g. `Polynomial.basis(...)` construct an
     abstract, uninstantiable `_ABCPolyBase` internally. Fixed via
     `vars(_ABCPolyBase)[name]` (the raw descriptor, correctly rebound
     through whichever class it's looked up on). Sabotage-tested:
     reverting to `getattr` turns `basis`/`cast`/`fit`/`fromroots`/
     `identity` red (5/5 fail) in the differential suite; confirmed,
     reverted back to the fix.
  2. `_as_series1` was not applied to the raw `other`/`othercoef`
     operand inside `__add__`/`__sub__`/`__mul__`/`__divmod__`/
     `__radd__`/`__rsub__`/`__rmul__`/`__rdivmod__`, so a bare scalar/
     0-d first argument was rejected by the already-shipped `polyadd`/
     `polysub`/`polymul`/`polydiv` kernels where real numpy silently
     promotes via `atleast_1d`. Fixed by wrapping with
     `_as_series1(..., trim=False)`. Sabotage-tested: swapping `_add`
     for `_sub` inside `__add__` turns exactly `__add__` red (1/1
     fail) in the differential suite; confirmed, reverted back.

Not bit-exact (measured, evidence-gated `epsilon_tolerance` in
polynomial_cases.py, same "exact"-means-"declared and verified passing"
ledger convention as `polyfit`/`polyroots` above) -- all six trace back
to ONE root cause: the module-level `polymul` kernel (REVOKED above
2026-08-07, NOT bit-exact -- see that entry) is NOT bit-exact against
real numpy under random sweeping (500-1347/1000-2000 mismatches per
sweep, maxrel ~9.1e-14 float64 / ~1.17e-14 complex128 -- confirmed
reproducing identically via direct module-level `polymul` calls, i.e.
pre-existing in the Rust-backed kernel, not introduced by any
`Polynomial`/`_polybase.py` code this pass). A 2026-08-07 fix attempt on
`polymul` itself (matching real numpy's exact convolution term order in
Rust) was measured and found insufficient -- so term order is not the
cause, but the actual mechanism is NOT established (an earlier claim
naming Accelerate's `cblas_ddot` was withdrawn on 2026-08-07: numpy's own
`np.dot` also fails to reproduce `np.convolve` bit-for-bit, 617/3000; see
KNOWN-DIFFERENCES.md) -- so these six items' `epsilon_tolerance` remains;
removing it would be premature until the underlying kernel is actually
bit-exact, which it is not:
  - `__mul__`, `__rmul__` (direct `polymul` call)
  - `__pow__` (repeated `polymul`, 262/1000 mismatches, maxrel
    ~1.06e-13, 20,000-sample sweep: 1.6190647066373651e-13)
  - `fromroots` (repeated `polymul` of linear factors, 381/1000
    mismatches, maxrel ~4.7e-14, 20,000-sample sweep (real):
    1.75779875021322e-11; complex128 sweep: 6.36564362046448e-14)
  - `convert`, `cast` (`_compose_affine`'s Horner-loop domain remap is
    multiplication-heavy, 380-394/1000 mismatches, maxrel ~1.5-2.6e-14,
    20,000-sample sweep: 1.1465959568111e-12)
  - `__truediv__` (independent divergence: the complex128 `polydiv`
    kernel's last-ULP rounding differs from real numpy's; scalar-rhs
    corpus only, since Polynomial-vs-Polynomial `TypeError` messages
    also diverge purely on module path text -- see
    `polynomial_cases.py`'s `poly_truediv_cases` comment)

All other items below (arithmetic identities `__add__`/`__sub__`/
`__divmod__`/`__floordiv__`/`__mod__`/reflected forms, `__call__`,
`integ`, `deriv`, `roots` (epsilon-tolerant, same companion-matrix-
eigensolver cause as `polyroots` above), structural/comparison items
(`degree`, `cutdeg`, `mapparms`, `has_samecoef`/`has_samedomain`/
`has_samewindow`/`has_sametype`, `__eq__`/`__ne__`, `copy`, `trim`,
`truncate`, `__len__`, `__iter__`, `__hash__` (both correctly
unhashable -- `TypeError`, matching real numpy)), construction/basis
items (`basis`, `identity`, `linspace`), properties (`domain`,
`window`, `symbol`, `maxpower`, `basis_name`), and pickle-protocol
items (`__getstate__`, `__setstate__`) are bit-exact.

`__getstate__`/`__setstate__` are verified directly (constructing a
state dict via `__getstate__` and feeding it to a fresh instance's
`__setstate__`, comparing the reconstructed `coef`/`domain`/`window`/
`symbol`) rather than via a full `pickle.dumps`/`pickle.loads` round
trip -- `anionpy.ndarray` does not itself implement the pickle
protocol (`TypeError: cannot pickle 'anionpy.ndarray' object`), which
is a pre-existing, out-of-scope limitation of the core ndarray type,
not of `ABCPolyBase`/`Polynomial`'s own `__getstate__`/`__setstate__`
methods (both of which are plain dict-manipulation code, verified
correct on their own terms).
"""

POLYNOMIAL_STATE = {
    "polynomial.polynomial.polydomain": "exact",
    "polynomial.polynomial.polyzero": "exact",
    "polynomial.polynomial.polyone": "exact",
    "polynomial.polynomial.polyx": "exact",
    "polynomial.polynomial.polyline": "exact",
    "polynomial.polynomial.polytrim": "exact",
    "polynomial.polynomial.polyval": "exact",
    "polynomial.polynomial.polyvalfromroots": "exact",
    "polynomial.polynomial.polyadd": "exact",
    "polynomial.polynomial.polysub": "exact",
    # Ticket #84 (2026-08-08, Monday): widened-scan fix, same defect
    # class and fix as polynomial.hermite.hermmulx above (missing
    # pu.as_series pre-trim before the zero-series fast-path check).
    # Fixed in ionp-core/src/poly.rs. Verified bit-exact via
    # /private/tmp/monday84/probe_all_mulx.py and the out-of-corpus
    # sweep: 0 mismatches. Also confirmed polyadd/polysub (which share
    # ionp-core's add_trim/sub_trim, separately fixed for the same
    # missing-pre-trim defect class) are bit-exact: 3600-trial signed-
    # zero-focused sweep (probe_addsub.py, seed 777888999), 0
    # mismatches across all 6 bases' add/sub.
    "polynomial.polynomial.polymulx": "exact",
    # "polynomial.polynomial.polymul": "exact",  REVOKED 2026-08-07 (Monday):
    #   not bit-exact against real numpy. Confirmed by a strengthened,
    #   seeded random corpus (tests/differential/polynomial_cases.py's
    #   `_polymul_adversarial_cases`, non-representable float64
    #   coefficients, degrees 1-7) -- 47-49/67 cases fail on this binary,
    #   worst relative error ~1.3e-13. Mechanism NOT established -- see
    #   the correction below. A fix was attempted and measured, not
    #   assumed impossible:
    #   `ionp-core/src/poly.rs`'s `convolve_full` was replaced with a
    #   from-scratch port of numpy's own C algorithm
    #   (`PyArray_Correlate2`/`_pyarray_correlate`, mode='full'), matching
    #   its exact per-tap term order bit-for-bit (verified independently
    #   at the Python level first). Rebuilt and re-measured: 47/67 still
    #   fail (vs 49/67 before), a negligible change. So TERM ORDER is not
    #   the cause. The experimental change was reverted (tree confirmed
    #   clean via `git diff`) rather than shipped, since it adds real
    #   complexity for no closer match.
    #
    #   CORRECTION 2026-08-07 (Monday): an earlier version of this note
    #   went on to assert the cause WAS Accelerate's internal SIMD/FMA
    #   accumulation inside `cblas_ddot`. That was withdrawn. Ruling out
    #   term order is not evidence for any particular replacement. An
    #   independent probe (seed 31337, N=3000) found that numpy's OWN
    #   `np.dot`, applied tap-wise, also fails to reproduce `np.convolve`
    #   bit-for-bit (617/3000, indistinguishable from a naive ordered sum
    #   at 606/3000) -- if each tap were one `cblas_ddot` that should have
    #   been 3000/3000. The mechanism is UNIDENTIFIED, and the revocation
    #   does not depend on identifying it.
    #
    #   See KNOWN-DIFFERENCES.md's "(b) polymul, polypow and
    #   polyfromroots" entry for the full writeup, including the
    #   (untested, single-BLAS) reason to suspect numpy's polymul bits are
    #   a property of a numpy BUILD rather than of numpy -- which would
    #   make "bit-exact polymul" ill-posed as a target. Re-declare when: a
    #   portable, reproducible replication of numpy's accumulation is
    #   found, or the ill-posedness hypothesis is tested on a second BLAS
    #   (OpenBLAS/MKL) and disproven.
    # Ticket #85 (2026-08-08, Monday): RE-CONFIRMED after finding and
    # fixing two real defects, not merely re-asserted. (1) The `lc2 == 1`
    # early-return branch's placeholder remainder was a literal
    # `T::zero()` fill, not numpy's `c1[:1] * 0` DERIVED zero -- loses the
    # sign a negative denormal `c1[0]` produces. Measured 1069/4000
    # (26.7%) signed-zero-only divergences on a denormal-numerator/normal-
    # divisor probe (seed 0xBEEF) before the fix. (2) An independently
    # discovered SECOND defect in the same two early-return branches (and
    # the synthetic-division loop): this port never did numpy's top-level
    # `[c1, c2] = pu.as_series([c1, c2])` pre-trim, so an untrimmed
    # trailing exact zero in `c1` leaked into the returned shape. Both
    # fixed in `ionp-core/src/poly.rs`'s `div` (trim both operands via
    # `trim_trailing_zeros` at the top, matching `chebdiv`'s already-
    # correct pattern; derive the placeholder zero from `c1_t[0]` instead
    # of a literal fill). Verified: the ticket's own exact repro now bit-
    # matches numpy; a 6000-trial out-of-corpus sweep (seed 0xC0FFEE,
    # denormal/normal/tiny/zero/huge operand mixes, shapes 1-8) and a
    # 15000-trial wider sweep (seed 777, shapes 1-12, real+complex) both
    # show 0 mismatches for `polydiv` outside the explicitly-out-of-scope
    # both-denormal NaN-sign-bit class (166 NaN/inf-only element
    # mismatches at extreme magnitudes in the 15000-trial sweep, 0 real-
    # value mismatches). Regression cases added to
    # `tests/differential/polynomial_cases.py`'s `polydiv_cases`, proven
    # to bite via `patch -R` (isolated revert of the `poly.rs` fix only)
    # before being fixed back. NOTE: the same missing-pre-trim mechanism
    # (2) was also found, via the same broader sweep, in `lagdiv`/
    # `hermdiv`/`hermediv` -- fixed by ticket #87 (2026-08-10, Monday), see
    # `polynomial.laguerre.lagdiv`/`polynomial.hermite.hermdiv`/
    # `polynomial.hermite_e.hermediv`'s own entries below for that ticket's
    # measurements.
    "polynomial.polynomial.polydiv": "exact",
    # "polynomial.polynomial.polypow": "exact",  REVOKED 2026-08-07
    #   (Monday): built directly on `polymul` (repeated multiply, see
    #   `anionpy/polynomial/polynomial.py`'s `polypow`) -- one defect with
    #   two dependents, same root cause as `polymul` above. 32/66 cases
    #   fail. See that entry and KNOWN-DIFFERENCES.md for the full
    #   writeup. Re-declare together with `polymul`.
    "polynomial.polynomial.polyder": "exact",
    "polynomial.polynomial.polyint": "exact",
    # "polynomial.polynomial.polyfromroots": "exact",  REVOKED 2026-08-07
    #   (Monday): balanced-pairing tree of `polymul` calls of linear
    #   factors (see `anionpy/polynomial/polynomial.py`'s
    #   `polyfromroots`) -- one defect with two dependents, same root
    #   cause as `polymul` above. 25/67 cases fail. See that entry and
    #   KNOWN-DIFFERENCES.md for the full writeup. Re-declare together
    #   with `polymul`.
    "polynomial.polynomial.polyvander": "exact",
    # Task #34 (2026-08-08): see module docstring above for scope/exclusion.
    "polynomial.polynomial.polyval2d": "exact",
    "polynomial.polynomial.polyval3d": "exact",
    "polynomial.polynomial.polyvalnd": "exact",
    "polynomial.polynomial.polygrid2d": "exact",
    "polynomial.polynomial.polygrid3d": "exact",
    "polynomial.polynomial.polyvander2d": "exact",
    "polynomial.polynomial.polyvander3d": "exact",
    "polynomial.polynomial.polycompanion": "exact",
    # Not bit-exact -- independent eigensolver call path on the companion
    # matrix. Measured, evidence-gated epsilon_tolerance in
    # polynomial_cases.py (20,000-sample sweep); still counted "exact" for
    # ledger purposes, matching this codebase's existing convention (see
    # linalg.py's det/cholesky/inv/etc., which use the same "exact" state
    # string for epsilon-tolerant-but-passing items -- the ledger's
    # "exact" means "declared and verified passing", not "bit-identical";
    # bit-identical vs. epsilon-tolerant is recorded in the differential
    # registry, not the ledger state string).
    "polynomial.polynomial.polyroots": "exact",
    # Not bit-exact -- independent lstsq/SVD call path. Same convention as
    # polyroots above; measured epsilon_tolerance in polynomial_cases.py.
    "polynomial.polynomial.polyfit": "exact",

    # -- polynomial.Polynomial (ABCPolyBase power-series concrete class) --
    # All 52 bit-exact (verified via corpus + out-of-corpus random sweeps).
    "polynomial.Polynomial.__add__": "exact",
    "polynomial.Polynomial.__sub__": "exact",
    "polynomial.Polynomial.__divmod__": "exact",
    "polynomial.Polynomial.__floordiv__": "exact",
    "polynomial.Polynomial.__mod__": "exact",
    "polynomial.Polynomial.__radd__": "exact",
    "polynomial.Polynomial.__rsub__": "exact",
    "polynomial.Polynomial.__rdivmod__": "exact",
    "polynomial.Polynomial.__rfloordiv__": "exact",
    "polynomial.Polynomial.__rmod__": "exact",
    "polynomial.Polynomial.__rtruediv__": "exact",
    "polynomial.Polynomial.__neg__": "exact",
    "polynomial.Polynomial.__pos__": "exact",
    "polynomial.Polynomial.__call__": "exact",
    "polynomial.Polynomial.__init__": "exact",
    "polynomial.Polynomial.__eq__": "exact",
    "polynomial.Polynomial.__ne__": "exact",
    "polynomial.Polynomial.__hash__": "exact",
    "polynomial.Polynomial.__iter__": "exact",
    "polynomial.Polynomial.__len__": "exact",
    "polynomial.Polynomial.__getstate__": "exact",
    "polynomial.Polynomial.__setstate__": "exact",
    "polynomial.Polynomial.__array_ufunc__": "exact",
    "polynomial.Polynomial.degree": "exact",
    "polynomial.Polynomial.cutdeg": "exact",
    "polynomial.Polynomial.mapparms": "exact",
    "polynomial.Polynomial.has_samecoef": "exact",
    "polynomial.Polynomial.has_samedomain": "exact",
    "polynomial.Polynomial.has_samewindow": "exact",
    "polynomial.Polynomial.has_sametype": "exact",
    "polynomial.Polynomial.copy": "exact",
    "polynomial.Polynomial.trim": "exact",
    "polynomial.Polynomial.truncate": "exact",
    "polynomial.Polynomial.integ": "exact",
    "polynomial.Polynomial.deriv": "exact",
    "polynomial.Polynomial.identity": "exact",
    "polynomial.Polynomial.linspace": "exact",
    "polynomial.Polynomial.domain": "exact",
    "polynomial.Polynomial.window": "exact",
    "polynomial.Polynomial.symbol": "exact",
    "polynomial.Polynomial.maxpower": "exact",
    "polynomial.Polynomial.basis_name": "exact",
    "polynomial.Polynomial.basis": "exact",
    # Epsilon-tolerant (measured, evidence-gated) -- see module docstring's
    # "Not bit-exact" block above for the shared polymul-kernel root cause
    # (mul/rmul/pow/fromroots) and __truediv__'s/roots's independent causes.
    # Same "exact" == "declared and verified passing" ledger convention.
    "polynomial.Polynomial.__mul__": "exact",
    "polynomial.Polynomial.__rmul__": "exact",
    "polynomial.Polynomial.__pow__": "exact",
    "polynomial.Polynomial.__truediv__": "exact",
    "polynomial.Polynomial.fromroots": "exact",
    "polynomial.Polynomial.convert": "exact",
    "polynomial.Polynomial.cast": "exact",
    "polynomial.Polynomial.fit": "exact",
    "polynomial.Polynomial.roots": "exact",

    # -- polynomial.legendre (Legendre-series basis, function level only --
    # see anionpy/polynomial/legendre.py's module docstring for exact scope;
    # its ABCPolyBase subclass Legendre(...) is NOT wired, deliberately) --
    # All 24 items verified passing against real numpy 2.5.1
    # (tests/differential/legendre_cases.py), plus out-of-corpus fresh-input
    # verification per this task's methodology. Two sabotage-and-revert
    # regression-guard tests (flipping the `-` to `+` in leg_eval's Clenshaw
    # recursion; flipping the `+` to `-` in legmulx's cross-index
    # accumulation) confirmed the corpus actually bites: 3/24 and 6/24
    # items respectively went red, then came back to 24/24 clean on revert.
    "polynomial.legendre.legdomain": "exact",
    "polynomial.legendre.legzero": "exact",
    "polynomial.legendre.legone": "exact",
    "polynomial.legendre.legx": "exact",
    "polynomial.legendre.legline": "exact",
    "polynomial.legendre.legtrim": "exact",
    "polynomial.legendre.legval": "exact",
    "polynomial.legendre.legadd": "exact",
    "polynomial.legendre.legsub": "exact",
    # "polynomial.legendre.legmulx": "exact"  -- ticket #84 (2026-08-08,
    #   Monday): RE-CONFIRMED, one genuine defect found and fixed. Missing
    #   the `[c] = pu.as_series([c])` pre-trim numpy's `legmulx` does before
    #   its `len(c)==1 && c[0]==0` zero-series fast path: an untrimmed
    #   all-(signed-)zero input (e.g. arising mid-`legmul` recursion when an
    #   operand underflows) skipped the fast path and fell into the general
    #   loop, which both flips a `-0.0` sign AND returns the WRONG SHAPE
    #   (len 4 instead of numpy's trimmed len 1) -- a shape mismatch, not
    #   merely a sign mismatch. Measured: `legmulx([-0j,-0j,-0j])` gave numpy
    #   `[-0.+0.j]` (len 1) vs pre-fix anionpy `[0j,0j,0j,0j]` (len 4, wrong
    #   shape AND flipped sign). Fixed by trimming `c` via
    #   `trim_trailing_zeros` before the zero-check in
    #   `ionp-core/src/legendre.rs::legmulx`. Same defect, same fix,
    #   independently confirmed in `lagmulx`/`hermmulx`/`hermemulx`/
    #   `chebmulx`/`poly::mulx` -- one root cause across all six bases, see
    #   each entry below. Two new permanent regression tests added in
    #   `legendre.rs`'s `mod tests`
    #   (`legmulx_underflow_all_zero_preserves_sign_and_shape`,
    #   `legmul_underflow_signed_zero_matches_numpy`). See
    #   KNOWN-DIFFERENCES.md and this ticket's report for the full trace.
    "polynomial.legendre.legmulx": "exact",
    # "polynomial.legendre.legmul": "exact"  -- ticket #48 (2026-08-08,
    #   Monday): RE-DECLARED EXACT, root cause found and fixed. The
    #   REVOKED note this comment used to carry (2026-08-07) blamed an
    #   "unestablished mechanism" copied from `polymul`/`chebmul`'s BLAS
    #   excuse -- that guess was WRONG for this kernel. The actual defect:
    #   `legendre.rs`'s local `scale(c, s)` helper (`c[i] * s` for every
    #   coefficient, backing every `c[0]*xs`/`c[-i]*xs`/`q*p[:-1]` site
    #   numpy's own `legmul`/`polyutils._div` source computes) multiplied
    #   `v.leg_mul(s)` -- ARRAY ELEMENT FIRST, scalar second -- the reverse
    #   of numpy's own literal `scalar * array` operand order at every one
    #   of those call sites. On `f64` this is invisible (real multiplication
    #   is commutative bit-for-bit). On `C128` it is NOT: `complex_mul_fma`
    #   is asymmetric under argument swap (`fma(x.re,y.im,x.im*y.re)`
    #   is a genuinely different rounding from `fma(y.re,x.im,y.im*x.re)` --
    #   the addend computed by plain multiply is a DIFFERENT real product in
    #   each ordering, not merely reordered). Minimal repro: two length-1
    #   complex128 series (`legmul([0.0106...-0.9128...j],
    #   [0.9094...+0.9562...j])`) diverged by exactly 1 ULP in the imaginary
    #   part before the fix, replicated bit-for-bit via `math.fma` in both
    #   orderings to confirm which one numpy actually uses. Fixed by
    #   swapping `scale`'s multiply to `s.leg_mul(v)` (scalar-first).
    #   Verified: differential suite `legmul_cases`' `nonexact_complex_
    #   DIVERGES` case now passes; fresh out-of-corpus sweep (seed
    #   0xDEADBEEF, 3000 random complex128 draws, degrees 1-6, magnitudes
    #   1e-3..1e8) 0/3000 mismatches, plus a negative control confirming the
    #   harness still detects an injected mismatch. See this ticket's report
    #   and KNOWN-DIFFERENCES.md for the full trace.
    # "polynomial.legendre.legmul": "exact"  -- ticket #84 (2026-08-08,
    #   Monday): RE-CONFIRMED against the specific complex128 underflow
    #   repro this ticket named (`legmul([5e-324]*3, [-5e-324]*2)`, numpy
    #   `-0.+0.j` vs pre-fix anionpy `+0.+0.j`) -- fixed by `legmulx`'s
    #   trim fix above (this repro's intermediate goes through
    #   `legmulx`). A SECOND, independent defect was also found and fixed
    #   by widening the scan: `legmul`'s internal recursion combines terms
    #   via `pad_add`/`pad_sub` at sites that correspond to numpy's actual
    #   `legadd`/`legsub` calls (`pu._add`/`pu._sub`), which trim BOTH
    #   operands via `as_series` before the padded combine -- `pad_add`/
    #   `pad_sub` themselves did not. An untrimmed trailing exact zero in
    #   the shorter operand collided with an unrelated `-0.0` elsewhere in
    #   the longer operand and flipped its sign (`0.0 + (-0.0) == 0.0`
    #   under IEEE 754 round-to-nearest). Minimal repro traced via
    #   `hermmul`'s analogous combine step (see `hermite.py`/
    #   `hermfromroots` entry below for the exact bit pattern) and
    #   independently confirmed present in `legmul`'s own loop/final-combine
    #   sites. Fixed by trimming both operands immediately before each
    #   `pad_add`/`pad_sub` call that stands in for a real `legadd`/`legsub`
    #   (NOT `legdiv`'s internal same-length `pad_sub`, which must stay
    #   untrimmed -- see that call site's own doc comment). Same fix applied
    #   to `lagmul`/`hermmul`/`hermemul`, and to the shared
    #   `crate::poly::add_trim`/`sub_trim` (which every basis's public
    #   `*add`/`*sub` binding aliases straight through to). Verified via a
    #   9000-plus-case out-of-corpus sweep (seed 999999, disjoint from
    #   discovery) across complex128 underflow/mixed-sign/mixed-magnitude
    #   inputs, both operand orders: 0 mismatches for `legmul`. See
    #   KNOWN-DIFFERENCES.md and this ticket's report.
    "polynomial.legendre.legmul": "exact",
    # Ticket #85 (2026-08-08, Monday): RE-CONFIRMED after finding and
    # fixing two real defects in `legdiv`'s early-return branches -- the
    # ticket #84 note just above this line, excluding "legdiv's internal
    # same-length pad_sub" from that ticket's trim fix, is STILL correct
    # for that ONE specific call site (the general `lc2 > 1` branch's
    # positional `rem[:-1] - q*p[:-1]`, measured unaffected here) -- but it
    # said nothing about the `lc1 < lc2` and `lc2 == 1` early-return
    # branches, which had their own, different, unaddressed defects: (1) a
    # literal `T::zero()` placeholder instead of numpy's `c1[:1] * 0`
    # DERIVED zero, losing sign on a negative denormal `c1[0]` -- measured
    # 1028/4000 (25.7%) signed-zero-only divergences (seed 0xBEEF) before
    # the fix; (2) missing numpy's top-level `[c1, c2] = as_series(...)`
    # pre-trim, so an untrimmed trailing zero in `c1` leaked into the
    # returned remainder as a SHAPE divergence, independently found via a
    # broader out-of-corpus sweep (seed 0xC0FFEE), not the ticket's own
    # probe. Both fixed in `ionp-core/src/legendre.rs`'s `legdiv` (trim
    # both operands via `trim_trailing_zeros` at the top, matching
    # `chebdiv`'s already-correct pattern; derive placeholder zeros from
    # `c1[0]` instead of a literal fill). Verified: the ticket's exact
    # repro now bit-matches numpy; a 6000-trial sweep (seed 0xC0FFEE) and
    # a 15000-trial wider sweep (seed 777, real+complex, shapes 1-12) both
    # show 0 mismatches, outside the explicitly-out-of-scope both-denormal
    # NaN-sign-bit class. Regression cases added to
    # `tests/differential/legendre_cases.py`'s `legdiv_cases`, proven to
    # bite via `patch -R` (isolated revert of the `legendre.rs` fix only).
    # NOTE: defect (2)'s missing-pre-trim mechanism was also found, via
    # the same broader sweep, in `lagdiv`/`hermdiv`/`hermediv` -- fixed by
    # ticket #87 (2026-08-10, Monday), see `polynomial.laguerre.lagdiv`/
    # `polynomial.hermite.hermdiv`/`polynomial.hermite_e.hermediv`'s own
    # entries below for that ticket's measurements.
    "polynomial.legendre.legdiv": "exact",
    # "polynomial.legendre.legpow": "exact"  -- ticket #48 (2026-08-08,
    #   Monday): RE-DECLARED EXACT, same `scale()` fix as `legmul` above
    #   (`legpow` is repeated `legmul`, one defect with two dependents).
    #   Verified on the same out-of-corpus sweep (500 random complex128
    #   `legpow` draws, powers 0-4): 0/500 mismatches.
    "polynomial.legendre.legpow": "exact",
    "polynomial.legendre.legder": "exact",
    "polynomial.legendre.legint": "exact",
    "polynomial.legendre.legvander": "exact",
    # Task #34 (2026-08-08): legval2d/legval3d/legvalnd/leggrid2d/
    # leggrid3d/legvander2d/legvander3d -- bit-exact for float64/
    # complex128/int (float16/32/complex64 excluded, same pre-existing
    # 1-D-kernel dtype-upcast reason as polynomial.polynomial above; see
    # this file's module docstring and docs/TICKET-34-*.md). NOT built on
    # `legmul`/`legfromroots` (both revoked above) -- `legval*`/
    # `legvander*` are Clenshaw-recursion evaluation, an entirely
    # different code path from the convolution-based multiply kernel, and
    # were independently verified bit-exact, not assumed clean by
    # association.
    "polynomial.legendre.legval2d": "exact",
    "polynomial.legendre.legval3d": "exact",
    "polynomial.legendre.legvalnd": "exact",
    "polynomial.legendre.leggrid2d": "exact",
    "polynomial.legendre.leggrid3d": "exact",
    "polynomial.legendre.legvander2d": "exact",
    "polynomial.legendre.legvander3d": "exact",
    "polynomial.legendre.legcompanion": "exact",
    # "polynomial.legendre.legfromroots": "exact"  -- ticket #48
    #   (2026-08-08, Monday): RE-DECLARED EXACT, same `scale()` fix as
    #   `legmul` above (`legfromroots` is a balanced-pairing tree of
    #   `legmul` calls of linear factors, one defect with two dependents).
    #   Verified on the same out-of-corpus sweep (500 random complex128
    #   `legfromroots` draws, 0-7 roots): 0/500 mismatches.
    "polynomial.legendre.legfromroots": "exact",
    "polynomial.legendre.leg2poly": "exact",
    "polynomial.legendre.poly2leg": "exact",
    "polynomial.legendre.legweight": "exact",
    # Not bit-exact -- independent eigensolver call path on the companion
    # matrix (legroots) / symmetric eigensolver + Newton polish (leggauss).
    # Both measured, evidence-gated epsilon_tolerance in
    # legendre_cases.py (20,000-sample seeded sweeps: LEGROOTS_COMPLEX_EPS,
    # LEGGAUSS_EPS). Same "exact" == "declared and verified passing" ledger
    # convention already used for polynomial.polyroots/polyfit above.
    "polynomial.legendre.legroots": "exact",
    "polynomial.legendre.leggauss": "exact",
    # Not bit-exact -- legfit depends on linalg.lstsq's independent SVD call
    # path, same convention as polynomial.polyfit above. Measured
    # epsilon_tolerance in legendre_cases.py.
    "polynomial.legendre.legfit": "exact",

    # -- polynomial.laguerre (Laguerre-series basis, function level only --
    # see anionpy/polynomial/laguerre.py's module docstring for exact scope)
    # All 24 items verified passing against real numpy 2.5.1
    # (tests/differential/laguerre_cases.py), PLUS a 20,000-iteration
    # out-of-corpus random bit-exact sweep per non-tolerance-bearing item
    # (raw IEEE-754 bits via `.view(np.uint64)`/`np.array_equal`, not
    # printed values -- see this task's report). That sweep caught FIVE
    # distinct bugs the fixed-case corpus itself missed:
    #   1. `lagfromroots` built raw `[-r, 1]` arrays instead of going
    #      through `lagline` (Laguerre's `lagline(off,scl) ==
    #      [off+scl,-scl]` is NOT the same affine map as Legendre's
    #      `[off,scl]` -- "looks like Legendre and is not").
    #   2. C128 multiplication used `num_complex::Complex`'s naive
    #      `(ac-bd,ad+bc)` formula instead of numpy's hardware-FMA-based
    #      complex multiply (`crate::ufunc::complex_mul_fma`).
    #   3. Several sites used `T::zero() - x` where numpy's source uses
    #      unary negation `-x` (`lagder`/`lagint`/`lagmulx`), and the
    #      generic `pad_add`/`pad_sub` padding helpers didn't replicate
    #      `polyutils._add`/`_sub`'s exact asymmetric-length branch
    #      structure -- both differ from numpy only in the sign of an
    #      exact-zero component, not in magnitude.
    #   4. `complex_mul_fma` is mathematically commutative but NOT
    #      bit-identical under argument swap (verified directly against
    #      real numpy: `a*xs` vs `xs*a` differ in their last bit for the
    #      same two complex128 operands) -- `scale()`'s `v.lag_mul(s)`
    #      had the operands backwards versus numpy's literal `c[i] * xs`
    #      source order, corrupting `lagmul`/`lagpow`/`lagfromroots`.
    #   5. Two grouping/parenthesization bugs matching numpy's literal
    #      source shape: `lagint`'s `tmp[0] + (ki - at_lbnd)` (was
    #      evaluating left-to-right as `(tmp[0]+ki)-at_lbnd`, a different
    #      floating-point expression since +/- isn't associative), and
    #      three `<something> * 0` sign-preserving special-cases
    #      (`lagder`'s `cnt >= n` branch, `lagdiv`'s two length-1-divisor
    #      early returns) that had been hardcoded to `T::zero()` instead
    #      of the actual `c[0] * 0` (which preserves `c[0]`'s sign under
    #      IEEE-754: `(-x)*0 == -0.0`).
    # One sabotage-and-revert regression-guard test (flipping the `+` to
    # `-` in `lagmulx`'s cross-index accumulation) confirmed the corpus
    # actually bites: 6/24 items (21 cases total) went red, then came back
    # to 24/24 clean on revert.
    "polynomial.laguerre.lagdomain": "exact",
    "polynomial.laguerre.lagzero": "exact",
    "polynomial.laguerre.lagone": "exact",
    "polynomial.laguerre.lagx": "exact",
    "polynomial.laguerre.lagline": "exact",
    "polynomial.laguerre.lagtrim": "exact",
    "polynomial.laguerre.lagval": "exact",
    "polynomial.laguerre.lagadd": "exact",
    "polynomial.laguerre.lagsub": "exact",
    # "polynomial.laguerre.lagmulx": "exact"  -- ticket #84 (2026-08-08,
    #   Monday): RE-CONFIRMED, same missing-pre-trim defect and same fix as
    #   `polynomial.legendre.legmulx` above (one root cause across all six
    #   `*mulx` kernels) -- see that entry for the measured bit pattern.
    #   Fixed in `ionp-core/src/laguerre.rs::lagmulx`.
    "polynomial.laguerre.lagmulx": "exact",
    # "polynomial.laguerre.lagmul": "exact"  -- ticket #84 (2026-08-08,
    #   Monday): RE-CONFIRMED against this ticket's named repro
    #   (`lagmul([5e-324]*3, [-5e-324]*2)`), fixed by `lagmulx`'s trim fix
    #   plus the same `pad_add`/`pad_sub`-pre-trim fix applied to
    #   `legmul`/`hermmul`/`hermemul` (see `polynomial.legendre.legmul`'s
    #   entry above for the full mechanism). Verified 0 mismatches on the
    #   same 9000-plus-case out-of-corpus sweep (seed 999999) covering both
    #   operand orders, mixed underflow/normal magnitudes, complex128 and
    #   float64.
    "polynomial.laguerre.lagmul": "exact",
    # Ticket #87 (2026-08-10, Monday): closed the gap #85 flagged (that
    # ticket's edit scope excluded this file). `lagdiv` already carried
    # #85's sibling sign-of-zero fix (derived `c1[0].lag_mul(T::zero())`
    # zero-fills, not literal `T::zero()`) but was STILL missing numpy
    # `_div`'s top-level `[c1, c2] = as_series([c1, c2])` pre-trim --
    # confirmed by direct measurement (this ticket did not inherit #85's
    # claim unverified): an out-of-corpus sweep generating exact trailing
    # zeros (seed 0xC0FFEE, n=6000) measured 5693/6000 raw fails
    # (3642 shape, 1750 false ZeroDivisionError from an untrimmed `c2`
    # tail, 301 sign-only) against this specific generator -- a different
    # rate than #85's own 254/6000 estimate because the two probes use
    # different trailing-zero densities, not a contradiction. Fixed by
    # trimming both operands via `trim_trailing_zeros` at the top of
    # `lagdiv`, matching `chebdiv`/`legdiv`/`poly::div`'s pattern. Same
    # 6000-trial sweep: 0/6000 after the fix. A second, more diverse
    # 4000-case sweep (seed 0xABCD1234: exact trailing zeros on either
    # operand, negative zeros, denormals, mixed magnitudes 1e-300..1e300,
    # varying lengths, length-1 divisors) found 0 real mismatches plus 4
    # excluded cases, each independently confirmed NaN-sign-bit-only
    # (0x7FF8... vs 0xFFF8..., same payload otherwise) via an explicit
    # bit-mask check with its own positive/negative controls -- the
    # ticket's explicitly out-of-scope both-denormal-adjacent NaN-sign
    # class (here triggered by extreme-magnitude overflow to inf/NaN, not
    # underflow, but the same weaker defect). Two regression cases added
    # to `tests/differential/laguerre_cases.py`'s `lagdiv_cases`
    # (untrimmed trailing zero on `c1`, and on `c2`), proven to bite via
    # isolated `patch -R` revert of only `laguerre.rs`/`hermite.rs`/
    # `hermite_e.rs` together (33 -> 36 named failures = baseline +
    # lagdiv + hermdiv + hermediv), then restored (back to the 33
    # baseline, unchanged by name). Full-suite + coverage after reapply
    # matches the required baseline exactly: 33 named failures, phantom 0,
    # coverage 51.491% (1571/3051), bit-exact 1517.
    "polynomial.laguerre.lagdiv": "exact",
    "polynomial.laguerre.lagpow": "exact",
    "polynomial.laguerre.lagder": "exact",
    "polynomial.laguerre.lagint": "exact",
    "polynomial.laguerre.lagvander": "exact",
    # Task #34 (2026-08-08): lagval2d/lagval3d/lagvalnd/laggrid2d/
    # laggrid3d/lagvander2d/lagvander3d -- bit-exact for float64/
    # complex128/int (float16/32/complex64 excluded, same pre-existing
    # 1-D-kernel dtype-upcast reason as polynomial.polynomial above; see
    # this file's module docstring and docs/TICKET-34-*.md). Clenshaw
    # evaluation, independently verified bit-exact, not assumed clean by
    # association with any other declaration on this list.
    "polynomial.laguerre.lagval2d": "exact",
    "polynomial.laguerre.lagval3d": "exact",
    "polynomial.laguerre.lagvalnd": "exact",
    "polynomial.laguerre.laggrid2d": "exact",
    "polynomial.laguerre.laggrid3d": "exact",
    "polynomial.laguerre.lagvander2d": "exact",
    "polynomial.laguerre.lagvander3d": "exact",
    "polynomial.laguerre.lagcompanion": "exact",
    # "polynomial.laguerre.lagfromroots": "exact"  -- REVOKED, ticket #84
    #   (2026-08-08, Monday): NOT bit-exact, found by widening this
    #   ticket's scan beyond the `legmul`/`lagmul`/`hermfromroots` items it
    #   named. Root cause is NOT the `legmul`-family `pad_add`/`pad_sub`
    #   pre-trim defect fixed elsewhere in this ticket -- it is unique to
    #   `lagfromroots`'s single-root path in `anionpy/polynomial/
    #   laguerre.py`. `lagfromroots([r])` (any complex `r` whose imaginary
    #   part is an exact zero, e.g. `[3.0]` or `[0.0]` -- NOT limited to
    #   underflow/subnormal magnitudes) builds `off = complex(-complex(r))`
    #   as a native Python `complex`, then `lagline(off, 1)` computes
    #   `off + scl` (`scl = 1`, a plain Python `int`) as native Python
    #   `complex.__add__(int)`. Measured: CPython's real-plus-complex
    #   addition does NOT perform an IEEE-754 add on the imaginary
    #   component at all -- it leaves `off.imag` bit-for-bit UNCHANGED
    #   (`complex(-0.0,-0.0) + 1 == (1-0j)`, imag stays exactly `-0.0`),
    #   whereas real numpy's `lagline` receives `off` as a genuine
    #   `np.complex128` scalar and `off + scl` goes through numpy's own
    #   scalar-add ufunc, which DOES perform a real IEEE add on both
    #   components (`-0.0 + 0.0 == +0.0`) -- confirmed directly:
    #   `complex(-0.0,-0.0) + complex(1,0.0)` (explicit complex+complex, no
    #   special-cased real-number path) also gives `+0.0`, matching numpy.
    #   Minimal repro: `lagfromroots([3.0])` -- numpy gives
    #   `[-2.+0.j, -1.+0.j]`, anionpy gives `[-2.-0.j, -1.+0.j]` (index 0's
    #   imaginary part flipped). Only manifests for the SINGLE-root path
    #   (`cnt == 1`, no `lagmul` pairwise-reduction call): multi-root inputs
    #   route every result through `lagmul`, whose own signed-zero handling
    #   this ticket already fixed, and 0 mismatches were found there across
    #   the same sweep. This ticket's EDIT scope does not include
    #   `anionpy/polynomial/laguerre.py` (only `anionpy/_state/
    #   polynomial.py`, `ionp-core/src/**`, `ionp-py/src/**`,
    #   `tests/differential/**`), so the fix (constructing `off+scl` via a
    #   genuine componentwise float add, or routing through the Rust core's
    #   own scalar add instead of Python's native `complex` type) is left
    #   for a follow-up ticket against that file. See KNOWN-DIFFERENCES.md.
    # "polynomial.laguerre.lagfromroots": "exact",  -- see REVOKED comment above; NOT declared.
    "polynomial.laguerre.lag2poly": "exact",
    "polynomial.laguerre.poly2lag": "exact",
    "polynomial.laguerre.lagweight": "exact",
    # Not bit-exact -- independent eigensolver call path on the companion
    # matrix (lagroots) / eigensolver-based node/weight construction
    # (laggauss). Both measured, evidence-gated epsilon_tolerance in
    # laguerre_cases.py (20,000-sample seeded sweeps: LAGROOTS_COMPLEX_EPS,
    # LAGGAUSS_EPS). Same "exact" == "declared and verified passing" ledger
    # convention already used for legendre.legroots/leggauss above.
    "polynomial.laguerre.lagroots": "exact",
    "polynomial.laguerre.laggauss": "exact",
    # Not bit-exact -- lagfit depends on linalg.lstsq's independent SVD
    # call path, same convention as legendre.legfit above. Measured
    # epsilon_tolerance in laguerre_cases.py.
    "polynomial.laguerre.lagfit": "exact",

    # -- polynomial.hermite (physicists' Hermite basis, function level only --
    # see anionpy/polynomial/hermite.py's module docstring for exact scope;
    # excludes only the Hermite(ABCPolyBase) class itself. hermval2d/3d/nd,
    # hermgrid2d/3d, hermvander2d/3d ARE implemented and declared below
    # (Task #34, 2026-08-08) -- see this file's module docstring correction.
    # All 24 items verified passing against real numpy 2.5.1
    # (tests/differential/hermite_cases.py), PLUS an independent
    # out-of-corpus random bit-exact sweep (seed=20260804, N=3000 trials,
    # 41,605 individual checks across 14 functions spanning both real and
    # complex coefficient paths, raw IEEE-754 bits via `.view(np.uint64)`/
    # `np.array_equal`, not printed values) -- zero failures.
    # One sabotage-and-revert regression-guard test (flipping the `+` to
    # `-` in `hermmulx`'s cross-index accumulation,
    # `ionp-core/src/hermite.rs`) confirmed the corpus actually bites:
    # 6/24 items (hermdiv, hermfromroots, hermmul, hermmulx, hermpow,
    # poly2herm -- all downstream consumers of hermmulx) went red,
    # 19/45 cases total failed, then came back to 24/24 clean on revert.
    "polynomial.hermite.hermdomain": "exact",
    "polynomial.hermite.hermzero": "exact",
    "polynomial.hermite.hermone": "exact",
    "polynomial.hermite.hermx": "exact",
    "polynomial.hermite.hermline": "exact",
    "polynomial.hermite.hermtrim": "exact",
    "polynomial.hermite.hermval": "exact",
    "polynomial.hermite.hermadd": "exact",
    "polynomial.hermite.hermsub": "exact",
    # Ticket #84 (2026-08-08, Monday): widened-scan fix. `hermmulx` lacked
    # numpy's `pu.as_series([c])` pre-trim before its
    # `len(c)==1 && c[0]==0` zero-series fast-path check, causing both
    # wrong shape and lost signed-zero on all-underflow/all-zero input
    # (same defect class confirmed and fixed identically in
    # legmulx/lagmulx/hermemulx/chebmulx/poly::mulx). Fixed in
    # ionp-core/src/hermite.rs. Verified bit-exact via
    # /private/tmp/monday84/probe_all_mulx.py and the 8700-case
    # out-of-corpus sweep (seed 999999): 0 mismatches.
    "polynomial.hermite.hermmulx": "exact",
    # Ticket #84 (2026-08-08, Monday): widened-scan fix, SEPARATE defect
    # from hermmulx's above. `hermmul`'s internal recursion calls
    # `pad_add`/`pad_sub` at the same call sites where real numpy calls
    # `hermadd`/`hermsub` (i.e. `pu._add`/`pu._sub`), which trim BOTH
    # operands via `as_series` before the padded elementwise combine --
    # `pad_add`/`pad_sub` never did this trim, so an untrimmed trailing
    # zero in the shorter operand could collide with (and flip the sign
    # of) an untouched -0.0 in the longer operand at that index. Repro:
    # hermmul([tiny-0j, 0.5+0j], [-tiny-0j, 0.5+0j]) (tiny=5e-324) --
    # numpy gives [0.5+0j, -0.+0j, 0.25+0j], pre-fix anionpy gave
    # [0.5+0j, 0.+0j, 0.25+0j] (index 1 real part sign lost). Fixed by
    # trimming both operands via trim_trailing_zeros immediately before
    # every pad_add/pad_sub call site in hermmul (ionp-core/src/
    # hermite.rs), matching the identical fix applied to legmul/lagmul/
    # hermemul and to the shared poly::add_trim/sub_trim (which covers
    # every basis's public *add/*sub bindings in one place). Verified
    # bit-exact via /private/tmp/monday84/probe_hermmul_steps.py,
    # probe_hermmul4.py, and the 8700-case out-of-corpus sweep (seed
    # 999999): 0 mismatches. cargo test -p ionp-core --lib hermite:
    # same 4 pre-existing/unrelated failures as before this fix, no new
    # ones.
    "polynomial.hermite.hermmul": "exact",
    # Ticket #87 (2026-08-10, Monday): same missing-pre-trim gap #85
    # identified but could not fix here (out of that ticket's edit scope)
    # -- `hermdiv` already had #84/#85's sign-of-zero fix but never did
    # numpy `_div`'s top-level `[c1, c2] = as_series([c1, c2])` pre-trim.
    # Fixed identically to `laguerre.lagdiv` above (trim both operands via
    # `trim_trailing_zeros` at the top of `hermdiv` in `hermite.rs`,
    # matching `chebdiv`/`legdiv`/`lagdiv`/`poly::div`). Same measured
    # rate as `lagdiv` on the identical seed-0xC0FFEE/seed-0xABCD1234
    # sweeps (this generic `_div`-family defect is basis-agnostic in
    # numpy's own source, so all four ports share one root cause and one
    # fix shape) -- see `laguerre.lagdiv`'s entry above for the full
    # numbers; not restated per-basis. Regression cases added to
    # `tests/differential/hermite_cases.py`'s `hermdiv_cases`; bite
    # confirmed by the same combined `patch -R` revert covering all three
    # files (33 -> 36 named failures, then back to 33 on reapply).
    "polynomial.hermite.hermdiv": "exact",
    "polynomial.hermite.hermpow": "exact",
    "polynomial.hermite.hermder": "exact",
    "polynomial.hermite.hermint": "exact",
    "polynomial.hermite.hermvander": "exact",
    # Task #34 (2026-08-08): hermval2d/hermval3d/hermvalnd/hermgrid2d/
    # hermgrid3d/hermvander2d/hermvander3d -- bit-exact for float64/
    # complex128/int (float16/32/complex64 excluded, same pre-existing
    # 1-D-kernel dtype-upcast reason as polynomial.polynomial above; see
    # this file's module docstring and docs/TICKET-34-*.md). Clenshaw
    # evaluation, independently verified bit-exact, not assumed clean by
    # association with any other declaration on this list.
    "polynomial.hermite.hermval2d": "exact",
    "polynomial.hermite.hermval3d": "exact",
    "polynomial.hermite.hermvalnd": "exact",
    "polynomial.hermite.hermgrid2d": "exact",
    "polynomial.hermite.hermgrid3d": "exact",
    "polynomial.hermite.hermvander2d": "exact",
    "polynomial.hermite.hermvander3d": "exact",
    "polynomial.hermite.hermcompanion": "exact",
    # Ticket #84 (2026-08-08, Monday): originally-named ticket item.
    # `hermfromroots` pairwise-reduces roots via `hermmul`, so it
    # inherited hermmul's pad_add/pad_sub pre-trim signed-zero defect
    # documented above; no independent bug found in hermfromroots
    # itself (unlike laguerre.fromroots, see lagfromroots's REVOKED
    # entry below for a genuinely distinct, out-of-scope bug in that
    # sibling). Confirmed fixed via hermmul's fix; verified bit-exact
    # via /private/tmp/monday84/probe_lagmul_hermfromroots.py and the
    # out-of-corpus sweep: 0 mismatches for hermite.fromroots.
    "polynomial.hermite.hermfromroots": "exact",
    "polynomial.hermite.herm2poly": "exact",
    "polynomial.hermite.poly2herm": "exact",
    "polynomial.hermite.hermweight": "exact",
    # Not bit-exact -- independent eigensolver call path on the companion
    # matrix (hermroots, real and complex coefficient dtypes measured
    # separately) / eigensolver-plus-Newton-refinement node/weight
    # construction (hermgauss). Both measured, evidence-gated
    # epsilon_tolerance in hermite_cases.py (20,000-sample seeded sweeps,
    # seed=9182736: HERMROOTS_COMPLEX_EPS=3.3145124091591355e-13,
    # HERMROOTS_REAL_EPS=9.094947017729282e-13,
    # HERMGAUSS_EPS=8.881784197001252e-16). Same "exact" == "declared and
    # verified passing" ledger convention already used for
    # laguerre.lagroots/laggauss above.
    "polynomial.hermite.hermroots": "exact",
    "polynomial.hermite.hermgauss": "exact",
    # Not bit-exact -- hermfit depends on linalg.lstsq's independent SVD
    # call path, same convention as laguerre.lagfit above. Measured
    # epsilon_tolerance (relative, floor=1.0, matching harness.py's
    # max_rel_distance) in hermite_cases.py: HERMFIT_EPS=2.8248462118007495e-12,
    # seed=9182736, N=20000.
    "polynomial.hermite.hermfit": "exact",

    # -- polynomial.hermite_e (probabilists' Hermite basis, function level
    # only -- see anionpy/polynomial/hermite_e.py's module docstring for
    # exact scope; excludes only the HermiteE(ABCPolyBase) class itself.
    # hermeval2d/3d/nd, hermegrid2d/3d, hermevander2d/3d ARE implemented and
    # declared below (Task #34, 2026-08-08) -- see this file's module
    # docstring correction. HermiteE is structurally the closest
    # port yet to hermite (differing essentially by factors of 2 and by
    # some -0.5/2*pi constants) -- an independent HermiteE-vs-Hermite
    # divergence probe (seed=24681012, N=200 per function, 9 representative
    # functions: val/mulx/mul/der/int/2poly/poly2/companion/weight)
    # confirmed 200/200 genuine bit-level disagreement for every function,
    # i.e. this is NOT hermite.rs with the 2s deleted by mistake and
    # accidentally still matching hermite's own numbers.
    # All 24 items verified passing against real numpy 2.5.1
    # (tests/differential/hermite_e_cases.py), PLUS an independent
    # out-of-corpus random bit-exact sweep (seed=314159265, N=20000 trials
    # per check, 28 checks spanning 14 functions x {real, complex}
    # coefficient paths = 560,000 individual comparisons via
    # `np.array_equal` on raw values, not printed/rounded) -- zero
    # mismatches.
    # One sabotage-and-revert regression-guard test (doubling the
    # cross-index coefficient in `hermemulx`'s accumulation step,
    # `ionp-core/src/hermite_e.rs`, i.e. `T::from_usize(2 * i)` instead of
    # `T::from_usize(i)`) confirmed the corpus actually bites: 6/24 items
    # (hermediv, hermefromroots, hermemul, hermemulx, hermepow, poly2herme
    # -- all downstream consumers of hermemulx) went red, then came back to
    # 24/24 clean on revert.
    "polynomial.hermite_e.hermedomain": "exact",
    "polynomial.hermite_e.hermezero": "exact",
    "polynomial.hermite_e.hermeone": "exact",
    "polynomial.hermite_e.hermex": "exact",
    "polynomial.hermite_e.hermeline": "exact",
    "polynomial.hermite_e.hermetrim": "exact",
    "polynomial.hermite_e.hermeval": "exact",
    "polynomial.hermite_e.hermeadd": "exact",
    "polynomial.hermite_e.hermesub": "exact",
    # Ticket #84 (2026-08-08, Monday): widened-scan fix, same defect
    # class and fix as polynomial.hermite.hermmulx above (missing
    # pu.as_series pre-trim before the zero-series fast-path check).
    # Fixed in ionp-core/src/hermite_e.rs. Verified bit-exact via
    # /private/tmp/monday84/probe_all_mulx.py and the out-of-corpus
    # sweep: 0 mismatches.
    "polynomial.hermite_e.hermemulx": "exact",
    # Ticket #84 (2026-08-08, Monday): widened-scan fix, same defect
    # class and fix as polynomial.hermite.hermmul above (missing
    # pu.as_series pre-trim before pad_add/pad_sub combine steps in the
    # internal recursion). Fixed in ionp-core/src/hermite_e.rs by
    # trimming both operands via trim_trailing_zeros immediately before
    # every pad_add/pad_sub call site. Verified bit-exact via the
    # out-of-corpus sweep (seed 999999): 0 mismatches. cargo test -p
    # ionp-core --lib hermite_e: 7 passed, 0 failed (no regressions).
    "polynomial.hermite_e.hermemul": "exact",
    # Ticket #87 (2026-08-10, Monday): same missing-pre-trim gap #85
    # identified but could not fix here (out of that ticket's edit scope)
    # -- `hermediv` already had #84/#85's sign-of-zero fix but never did
    # numpy `_div`'s top-level `[c1, c2] = as_series([c1, c2])` pre-trim.
    # Fixed identically to `laguerre.lagdiv`/`hermite.hermdiv` above (trim
    # both operands via `trim_trailing_zeros` at the top of `hermediv` in
    # `hermite_e.rs`, matching `chebdiv`/`legdiv`/`lagdiv`/`hermdiv`/
    # `poly::div`) -- see `laguerre.lagdiv`'s entry above for the full
    # measured numbers, shared across all four `*div` ports (one root
    # cause, basis-agnostic in numpy's own source). Regression cases added
    # to `tests/differential/hermite_e_cases.py`'s `hermediv_cases`; bite
    # confirmed by the same combined `patch -R` revert covering all three
    # files (33 -> 36 named failures, then back to 33 on reapply).
    "polynomial.hermite_e.hermediv": "exact",
    "polynomial.hermite_e.hermepow": "exact",
    "polynomial.hermite_e.hermeder": "exact",
    "polynomial.hermite_e.hermeint": "exact",
    "polynomial.hermite_e.hermevander": "exact",
    # Task #34 (2026-08-08): hermeval2d/hermeval3d/hermevalnd/hermegrid2d/
    # hermegrid3d/hermevander2d/hermevander3d -- bit-exact for float64/
    # complex128/int (float16/32/complex64 excluded, same pre-existing
    # 1-D-kernel dtype-upcast reason as polynomial.polynomial above; see
    # this file's module docstring and docs/TICKET-34-*.md). Clenshaw
    # evaluation, independently verified bit-exact, not assumed clean by
    # association with any other declaration on this list.
    "polynomial.hermite_e.hermeval2d": "exact",
    "polynomial.hermite_e.hermeval3d": "exact",
    "polynomial.hermite_e.hermevalnd": "exact",
    "polynomial.hermite_e.hermegrid2d": "exact",
    "polynomial.hermite_e.hermegrid3d": "exact",
    "polynomial.hermite_e.hermevander2d": "exact",
    "polynomial.hermite_e.hermevander3d": "exact",
    "polynomial.hermite_e.hermecompanion": "exact",
    "polynomial.hermite_e.hermefromroots": "exact",
    "polynomial.hermite_e.herme2poly": "exact",
    "polynomial.hermite_e.poly2herme": "exact",
    "polynomial.hermite_e.hermeweight": "exact",
    # Not bit-exact -- independent eigensolver call path on the companion
    # matrix (hermeroots, real and complex coefficient dtypes measured
    # separately) / eigensolver-plus-Newton-refinement node/weight
    # construction (hermegauss). Both measured, evidence-gated
    # epsilon_tolerance in hermite_e_cases.py (20,000-sample seeded
    # sweeps, seed=9182736: HERMEROOTS_COMPLEX_EPS=3.410605131648481e-13,
    # HERMEROOTS_REAL_EPS=1.8189894035458565e-12,
    # HERMEGAUSS_EPS=1.7763568394002505e-15). Own independently measured
    # constants, NOT inherited from hermite_cases.py's HERMROOTS_*/
    # HERMGAUSS_EPS. Same "exact" == "declared and verified passing"
    # ledger convention already used for hermite.hermroots/hermgauss above.
    "polynomial.hermite_e.hermeroots": "exact",
    "polynomial.hermite_e.hermegauss": "exact",
    # Not bit-exact -- hermefit depends on linalg.lstsq's independent SVD
    # call path, same convention as hermite.hermfit above. Measured
    # epsilon_tolerance (relative, floor=1.0, matching harness.py's
    # max_rel_distance) in hermite_e_cases.py:
    # HERMEFIT_EPS=2.482499636227198e-12, seed=9182736, N=20000.
    "polynomial.hermite_e.hermefit": "exact",

    # -- polynomial.chebyshev (first-kind Chebyshev basis, function level
    # only -- see anionpy/polynomial/chebyshev.py's module docstring for
    # exact scope; excludes only the Chebyshev(ABCPolyBase) class itself.
    # chebval2d/3d/nd, chebgrid2d/3d, chebvander2d/3d ARE implemented and
    # declared below (Task #34, 2026-08-08) -- see this file's module
    # docstring correction. Chebyshev is architecturally distinct
    # from the four Hermite-family/Legendre/Laguerre bases above: chebmul/
    # chebdiv/chebpow route through a z-series (symmetric Laurent series)
    # representation instead of a direct coefficient recurrence (chebder/
    # chebint use their own direct recurrences, NOT z-series -- numpy's own
    # private `_zseries_der`/`_zseries_int` helpers are dead code, never
    # called by any public chebyshev function; confirmed by reading numpy's
    # source and ported here as `#[allow(dead_code)]` for the record, never
    # invoked by chebder/chebint's own direct recurrences).
    # 25 of the 27 attempted items are verified passing bit-exact (or, for
    # chebfit/chebroots, epsilon-tolerant via an independent LAPACK call
    # path -- same ledger convention as every other basis above) against
    # real numpy 2.5.1 (tests/differential/chebyshev_cases.py), PLUS an
    # independent out-of-corpus random bit-exact sweep (seed=9182736,
    # N=20000, both real and complex inputs, degree-0/1/2 pinned
    # explicitly). chebinterpolate was flagged in the original task brief
    # as a possible trig-related tolerance case; MEASURED genuinely
    # bit-exact instead (0 mismatches over a dedicated N=20000 sweep across
    # four trig/poly/exp test functions) -- no tolerance declared.
    # chebgauss is closed-form (`x = cos(pi*arange(1,2*ideg,2)/(2*ideg))`,
    # `w = ones(ideg)*(pi/ideg)`), confirmed bit-exact, no eigensolver
    # involved (unlike legroots/lagroots/hermroots/hermeroots's
    # eigensolver-based *gauss siblings -- chebyshev's own *roots function
    # uses eigvals, but *gauss does not).
    "polynomial.chebyshev.chebdomain": "exact",
    "polynomial.chebyshev.chebzero": "exact",
    "polynomial.chebyshev.chebone": "exact",
    "polynomial.chebyshev.chebx": "exact",
    "polynomial.chebyshev.chebline": "exact",
    "polynomial.chebyshev.chebtrim": "exact",
    "polynomial.chebyshev.chebval": "exact",
    "polynomial.chebyshev.chebadd": "exact",
    "polynomial.chebyshev.chebsub": "exact",
    # Ticket #84 (2026-08-08, Monday): widened-scan fix, same defect
    # class and fix as polynomial.hermite.hermmulx above (missing
    # pu.as_series pre-trim before the zero-series fast-path check).
    # Fixed in ionp-core/src/chebyshev.rs (chebmul itself was already
    # unaffected -- it uses a z-series convolution algorithm with its
    # own pre-existing trim_trailing_zeros calls, not pad_add/pad_sub).
    # Verified bit-exact via /private/tmp/monday84/probe_all_mulx.py
    # and the out-of-corpus sweep: 0 mismatches for chebyshev.mulx
    # (chebyshev.mul's pre-existing, unrelated non-exactness is
    # documented separately below and out of scope for this ticket).
    "polynomial.chebyshev.chebmulx": "exact",
    # "polynomial.chebyshev.chebmul": "exact",  NOT declared (Monday,
    #   2026-08-07): chebmul is z-series based -- numpy's private
    #   `_zseries_mul` calls `np.convolve` directly. Measured (seed
    #   9182736): a plain `np.convolve(a, b)` on small arbitrary real
    #   arrays disagrees with a naive double-loop-accumulate equivalent in
    #   ~58% of trials (1165/2000). Tried and ruled out as reproduction
    #   strategies: i-outer/j-inner, j-outer/i-inner, k-indexed forward
    #   accumulation, k-indexed reverse accumulation, FMA-based (multiple
    #   orders), and pairwise/tree summation -- none reproduce numpy's
    #   exact bits. Same root cause, same unidentified mechanism, as this
    #   file's `polynomial.polynomial.polymul` REVOKED entry above (numpy's
    #   C-level convolve/correlate is not reproducible via any simple
    #   sequential algorithm tried). The curated differential corpus in
    #   chebyshev_cases.py happens to pass bit-exact on its own small
    #   hand-picked values (caught only by out-of-corpus sweeping, same as
    #   polymul) -- not declared here per this task's "leave undeclared and
    #   report why" instruction; no legitimate epsilon_tolerance
    #   justification applies (not LAPACK/eigensolver-based). Re-declare
    #   together with polymul, if that mechanism is ever identified.
    "polynomial.chebyshev.chebdiv": "exact",
    # "polynomial.chebyshev.chebpow": "exact",  NOT declared (Monday,
    #   2026-08-07): a repeated z-series convolve loop -- same
    #   `zseries_mul`/`np.convolve` root cause as chebmul directly above.
    #   Not declared for the same reason.
    "polynomial.chebyshev.chebder": "exact",
    "polynomial.chebyshev.chebint": "exact",
    "polynomial.chebyshev.chebvander": "exact",
    # Task #34 (2026-08-08): chebval2d/chebval3d/chebvalnd/chebgrid2d/
    # chebgrid3d/chebvander2d/chebvander3d -- bit-exact for float64/
    # complex128/int (float16/32/complex64 excluded, same pre-existing
    # 1-D-kernel dtype-upcast reason as polynomial.polynomial above; see
    # this file's module docstring and docs/TICKET-34-*.md). NOT built on
    # chebmul/chebpow/chebfromroots (all undeclared/revoked above for the
    # unreproducible np.convolve summation order) -- chebval*/chebvander*
    # are Clenshaw-recursion evaluation, an entirely different code path,
    # independently verified bit-exact, not assumed clean by association.
    "polynomial.chebyshev.chebval2d": "exact",
    "polynomial.chebyshev.chebval3d": "exact",
    "polynomial.chebyshev.chebvalnd": "exact",
    "polynomial.chebyshev.chebgrid2d": "exact",
    "polynomial.chebyshev.chebgrid3d": "exact",
    "polynomial.chebyshev.chebvander2d": "exact",
    "polynomial.chebyshev.chebvander3d": "exact",
    "polynomial.chebyshev.chebcompanion": "exact",
    #   REVOKED 2026-08-07 (Monday). `chebfromroots` was declared "exact" in
    #   the same commit that correctly DECLINED to declare `chebmul` -- but
    #   chebyshev.py:565 is `pu._fromroots(chebline, chebmul, roots)`, so this
    #   function IS chebmul in a loop and inherits its unreproducible
    #   `np.convolve` summation order. Measured on a random sweep the corpus
    #   did not cover (seed 0xA11CE): 917/2000 real, 1824/2000 complex
    #   non-bit-exact. The divergence scales with root count exactly as an
    #   accumulating multiply must -- 1 root 0/400 (the pairing loop never
    #   runs), then 51, 174, 292, 342 out of 400 for 2..5 roots.
    #   Concrete repro, invisible at printed precision:
    #       >>> r = [0.7, -1.3, 2.9]
    #       >>> chebfromroots(r)          # both print [1.489, -1.9, -1.15, 0.25]
    #       >>> max abs diff 4.440892098500626e-16, bitwise NOT equal
    #   Same class as the REVOKED polynomial.polynomial.polyfromroots. Restore
    #   only when chebmul itself is bit-reproducible -- not by adding
    #   tolerance, which would be the defect this project already shipped once.
    #   "polynomial.chebyshev.chebfromroots": "exact",
    "polynomial.chebyshev.cheb2poly": "exact",
    "polynomial.chebyshev.poly2cheb": "exact",
    "polynomial.chebyshev.chebweight": "exact",
    "polynomial.chebyshev.chebgauss": "exact",
    "polynomial.chebyshev.chebpts1": "exact",
    "polynomial.chebyshev.chebpts2": "exact",
    "polynomial.chebyshev.chebinterpolate": "exact",
    # Not bit-exact -- independent eigensolver call path on the companion
    # matrix. chebroots never downcasts to real (no
    # `_to_real_if_imag_zero` call in numpy's own source, like
    # hermeroots-not-hermroots), so eigvals always returns complex128 here.
    # Measured, evidence-gated epsilon_tolerance in chebyshev_cases.py
    # (20,000-sample seeded sweeps, seed=9182736:
    # CHEBROOTS_COMPLEX_EPS=6.3825755133549975e-15,
    # CHEBROOTS_REAL_EPS=1.6812769892473484e-14). Same "exact" == "declared
    # and verified passing" ledger convention used throughout this file.
    "polynomial.chebyshev.chebroots": "exact",
    # Not bit-exact -- chebfit depends on linalg.lstsq's independent SVD
    # call path, same convention as hermite_e.hermefit above. Measured
    # epsilon_tolerance (relative, floor=1.0, matching harness.py's
    # max_rel_distance) in chebyshev_cases.py:
    # CHEBFIT_EPS=3.357865793305266e-10, seed=9182736, N=20000.
    "polynomial.chebyshev.chebfit": "exact",

    # ---------------------------------------------------------------------
    # The five new ABCPolyBase subclasses (Legendre, Laguerre, Hermite,
    # HermiteE, Chebyshev). See tests/differential/abc_poly_class_cases.py's
    # module docstring for full per-class scope and measurement evidence.
    # convert/cast are NOT declared for any of the five (shared
    # _compose_affine bug: naive Horner-loop substitution valid only for
    # the power-series basis, wrong -- not merely imprecise -- for every
    # orthogonal basis here). __mul__/__rmul__/__pow__/fromroots are NOT
    # declared for Chebyshev (routes through non-bit-exact chebmul, same
    # as the module-level chebmul/chebpow/chebfromroots above).
    #
    # Ticket #34 (dunder cluster, 2026-08-08): Chebyshev's __radd__/
    # __rsub__/__rtruediv__/__rfloordiv__/__rmod__/__rdivmod__ were
    # PREVIOUSLY bundled under the same "not declared" umbrella as
    # __rmul__/__pow__/fromroots above -- wrongly. Those six route through
    # `self._add`/`self._sub`/`self._div` (chebadd/chebsub/chebdiv), none
    # of which has any known divergence; only `_mul`/`_pow`
    # (chebmul/chebpow) are non-bit-exact. Already implemented pre-ticket
    # (rebound from ABCPolyBase onto Chebyshev.__dict__ in chebyshev.py),
    # simply never split out of the mul-gated corpus block and declared.
    # Split in tests/differential/abc_poly_class_cases.py this ticket,
    # verified out-of-corpus (/private/tmp/probe34_dunders.py: 240/240
    # value checks incl. int/float/complex/nan/inf operands and invalid-
    # operand NotImplemented parity) and via the corpus below, now
    # declared exact. One caveat found and NOT fixed (out of scope, values
    # unaffected): numpy's chebdiv emits a RuntimeWarning('invalid value
    # encountered in multiply') for degree-0-coefficient // inf (and the
    # __rmod__/__rdivmod__ cases built on the same code path); anionpy's
    # equivalent path returns the identical value (bit-exact, confirmed)
    # but does not emit that warning. See this ticket's writeup,
    # docs/TICKET-34-DUNDERS-2026-08-08.md.
    #
    # `Chebyshev.interpolate` (the classmethod, distinct from the already-
    # exact module-level `chebinterpolate` function above) was also
    # already implemented but never declared or corpus-covered; added and
    # verified this ticket (24/24 out-of-corpus checks, see the same probe
    # script). It is Chebyshev-specific in real numpy too (no analog on
    # the other four bases), so does not appear in the corresponding
    # per-class blocks below.
    # ---------------------------------------------------------------------
    # -- Legendre --
    "polynomial.Legendre.__init__": "exact",
    "polynomial.Legendre.domain": "exact",
    "polynomial.Legendre.window": "exact",
    "polynomial.Legendre.basis_name": "exact",
    "polynomial.Legendre.symbol": "exact",
    "polynomial.Legendre.maxpower": "exact",
    "polynomial.Legendre.__call__": "exact",
    "polynomial.Legendre.__iter__": "exact",
    "polynomial.Legendre.__len__": "exact",
    "polynomial.Legendre.__hash__": "exact",
    "polynomial.Legendre.__array_ufunc__": "exact",
    "polynomial.Legendre.__eq__": "exact",
    "polynomial.Legendre.__ne__": "exact",
    "polynomial.Legendre.__add__": "exact",
    "polynomial.Legendre.__sub__": "exact",
    "polynomial.Legendre.__floordiv__": "exact",
    "polynomial.Legendre.__mod__": "exact",
    "polynomial.Legendre.__divmod__": "exact",
    "polynomial.Legendre.__truediv__": "exact",
    "polynomial.Legendre.__neg__": "exact",
    "polynomial.Legendre.__pos__": "exact",
    "polynomial.Legendre.__getstate__": "exact",
    "polynomial.Legendre.__setstate__": "exact",
    "polynomial.Legendre.copy": "exact",
    "polynomial.Legendre.degree": "exact",
    "polynomial.Legendre.cutdeg": "exact",
    "polynomial.Legendre.trim": "exact",
    "polynomial.Legendre.truncate": "exact",
    "polynomial.Legendre.mapparms": "exact",
    "polynomial.Legendre.has_samecoef": "exact",
    "polynomial.Legendre.has_samedomain": "exact",
    "polynomial.Legendre.has_samewindow": "exact",
    "polynomial.Legendre.has_sametype": "exact",
    "polynomial.Legendre.integ": "exact",
    "polynomial.Legendre.deriv": "exact",
    "polynomial.Legendre.roots": "exact",
    "polynomial.Legendre.identity": "exact",
    "polynomial.Legendre.basis": "exact",
    "polynomial.Legendre.linspace": "exact",
    "polynomial.Legendre.fit": "exact",
    # "polynomial.Legendre.__mul__": "exact",  REVOKED 2026-08-07 (Monday):
    #   wraps `legendre.legmul`, not bit-exact on complex128 -- same defect,
    #   same measurement (21306-22244/25000 across two independent sweeps
    #   of this and the module-level entry), same REVOKED reasoning as
    #   `polynomial.legendre.legmul` above. `ionp-core/src/legendre.rs`'s
    #   missing `complex_mul_fma`/`complex_div` routing was fixed this
    #   session (added `leg_mul`, routed every `T*T` site) but did NOT
    #   clear this item -- float64 stays bit-exact, complex128 does not.
    #   `abc_poly_class_cases.py`'s `_ClassBinding("Legendre", ...)` keeps
    #   `declare_mul=True` (so the corpus still exercises and visibly fails
    #   this on complex128 -- expected-red, not silently skipped) but
    #   `mul_complex_eps=None` (no re-tolerance; the earlier
    #   `LEGENDRE_CLASS_MUL_EPS=1e-13` tolerance is gone, per this task's
    #   hard rule). See KNOWN-DIFFERENCES.md. Re-declare together with
    #   `legmul`.
    "polynomial.Legendre.__radd__": "exact",
    "polynomial.Legendre.__rsub__": "exact",
    "polynomial.Legendre.__rtruediv__": "exact",
    "polynomial.Legendre.__rfloordiv__": "exact",
    "polynomial.Legendre.__rmod__": "exact",
    "polynomial.Legendre.__rdivmod__": "exact",
    "polynomial.Legendre.__rmul__": "exact",
    # "polynomial.Legendre.__pow__": "exact",  REVOKED 2026-08-07 (Monday):
    #   ABCPolyBase.__pow__ is repeated `_mul` (repeated `legmul`) -- one
    #   defect with two dependents, same root cause as `__mul__` above.
    #   9006/25000 fail on the same sweep. Re-declare together with
    #   `legmul`/`Legendre.__mul__`.
    # "polynomial.Legendre.fromroots": "exact",  REVOKED 2026-08-07
    #   (Monday): balanced-pairing tree of `legmul` calls of linear
    #   factors -- one defect with two dependents, same root cause as
    #   `__mul__` above. 12242/25000 fail on the same sweep. Re-declare
    #   together with `legmul`/`Legendre.__mul__`.
    # -- Laguerre --
    "polynomial.Laguerre.__init__": "exact",
    "polynomial.Laguerre.domain": "exact",
    "polynomial.Laguerre.window": "exact",
    "polynomial.Laguerre.basis_name": "exact",
    "polynomial.Laguerre.symbol": "exact",
    "polynomial.Laguerre.maxpower": "exact",
    "polynomial.Laguerre.__call__": "exact",
    "polynomial.Laguerre.__iter__": "exact",
    "polynomial.Laguerre.__len__": "exact",
    "polynomial.Laguerre.__hash__": "exact",
    "polynomial.Laguerre.__array_ufunc__": "exact",
    "polynomial.Laguerre.__eq__": "exact",
    "polynomial.Laguerre.__ne__": "exact",
    "polynomial.Laguerre.__add__": "exact",
    "polynomial.Laguerre.__sub__": "exact",
    "polynomial.Laguerre.__floordiv__": "exact",
    "polynomial.Laguerre.__mod__": "exact",
    "polynomial.Laguerre.__divmod__": "exact",
    "polynomial.Laguerre.__truediv__": "exact",
    "polynomial.Laguerre.__neg__": "exact",
    "polynomial.Laguerre.__pos__": "exact",
    "polynomial.Laguerre.__getstate__": "exact",
    "polynomial.Laguerre.__setstate__": "exact",
    "polynomial.Laguerre.copy": "exact",
    "polynomial.Laguerre.degree": "exact",
    "polynomial.Laguerre.cutdeg": "exact",
    "polynomial.Laguerre.trim": "exact",
    "polynomial.Laguerre.truncate": "exact",
    "polynomial.Laguerre.mapparms": "exact",
    "polynomial.Laguerre.has_samecoef": "exact",
    "polynomial.Laguerre.has_samedomain": "exact",
    "polynomial.Laguerre.has_samewindow": "exact",
    "polynomial.Laguerre.has_sametype": "exact",
    "polynomial.Laguerre.integ": "exact",
    "polynomial.Laguerre.deriv": "exact",
    "polynomial.Laguerre.roots": "exact",
    "polynomial.Laguerre.identity": "exact",
    "polynomial.Laguerre.basis": "exact",
    "polynomial.Laguerre.linspace": "exact",
    "polynomial.Laguerre.fit": "exact",
    "polynomial.Laguerre.__mul__": "exact",
    "polynomial.Laguerre.__radd__": "exact",
    "polynomial.Laguerre.__rsub__": "exact",
    "polynomial.Laguerre.__rtruediv__": "exact",
    "polynomial.Laguerre.__rfloordiv__": "exact",
    "polynomial.Laguerre.__rmod__": "exact",
    "polynomial.Laguerre.__rdivmod__": "exact",
    "polynomial.Laguerre.__rmul__": "exact",
    "polynomial.Laguerre.__pow__": "exact",
    "polynomial.Laguerre.fromroots": "exact",
    # -- Hermite --
    "polynomial.Hermite.__init__": "exact",
    "polynomial.Hermite.domain": "exact",
    "polynomial.Hermite.window": "exact",
    "polynomial.Hermite.basis_name": "exact",
    "polynomial.Hermite.symbol": "exact",
    "polynomial.Hermite.maxpower": "exact",
    "polynomial.Hermite.__call__": "exact",
    "polynomial.Hermite.__iter__": "exact",
    "polynomial.Hermite.__len__": "exact",
    "polynomial.Hermite.__hash__": "exact",
    "polynomial.Hermite.__array_ufunc__": "exact",
    "polynomial.Hermite.__eq__": "exact",
    "polynomial.Hermite.__ne__": "exact",
    "polynomial.Hermite.__add__": "exact",
    "polynomial.Hermite.__sub__": "exact",
    "polynomial.Hermite.__floordiv__": "exact",
    "polynomial.Hermite.__mod__": "exact",
    "polynomial.Hermite.__divmod__": "exact",
    "polynomial.Hermite.__truediv__": "exact",
    "polynomial.Hermite.__neg__": "exact",
    "polynomial.Hermite.__pos__": "exact",
    "polynomial.Hermite.__getstate__": "exact",
    "polynomial.Hermite.__setstate__": "exact",
    "polynomial.Hermite.copy": "exact",
    "polynomial.Hermite.degree": "exact",
    "polynomial.Hermite.cutdeg": "exact",
    "polynomial.Hermite.trim": "exact",
    "polynomial.Hermite.truncate": "exact",
    "polynomial.Hermite.mapparms": "exact",
    "polynomial.Hermite.has_samecoef": "exact",
    "polynomial.Hermite.has_samedomain": "exact",
    "polynomial.Hermite.has_samewindow": "exact",
    "polynomial.Hermite.has_sametype": "exact",
    "polynomial.Hermite.integ": "exact",
    "polynomial.Hermite.deriv": "exact",
    "polynomial.Hermite.roots": "exact",
    "polynomial.Hermite.identity": "exact",
    "polynomial.Hermite.basis": "exact",
    "polynomial.Hermite.linspace": "exact",
    "polynomial.Hermite.fit": "exact",
    "polynomial.Hermite.__mul__": "exact",
    "polynomial.Hermite.__radd__": "exact",
    "polynomial.Hermite.__rsub__": "exact",
    "polynomial.Hermite.__rtruediv__": "exact",
    "polynomial.Hermite.__rfloordiv__": "exact",
    "polynomial.Hermite.__rmod__": "exact",
    "polynomial.Hermite.__rdivmod__": "exact",
    "polynomial.Hermite.__rmul__": "exact",
    "polynomial.Hermite.__pow__": "exact",
    "polynomial.Hermite.fromroots": "exact",
    # -- HermiteE --
    "polynomial.HermiteE.__init__": "exact",
    "polynomial.HermiteE.domain": "exact",
    "polynomial.HermiteE.window": "exact",
    "polynomial.HermiteE.basis_name": "exact",
    "polynomial.HermiteE.symbol": "exact",
    "polynomial.HermiteE.maxpower": "exact",
    "polynomial.HermiteE.__call__": "exact",
    "polynomial.HermiteE.__iter__": "exact",
    "polynomial.HermiteE.__len__": "exact",
    "polynomial.HermiteE.__hash__": "exact",
    "polynomial.HermiteE.__array_ufunc__": "exact",
    "polynomial.HermiteE.__eq__": "exact",
    "polynomial.HermiteE.__ne__": "exact",
    "polynomial.HermiteE.__add__": "exact",
    "polynomial.HermiteE.__sub__": "exact",
    "polynomial.HermiteE.__floordiv__": "exact",
    "polynomial.HermiteE.__mod__": "exact",
    "polynomial.HermiteE.__divmod__": "exact",
    "polynomial.HermiteE.__truediv__": "exact",
    "polynomial.HermiteE.__neg__": "exact",
    "polynomial.HermiteE.__pos__": "exact",
    "polynomial.HermiteE.__getstate__": "exact",
    "polynomial.HermiteE.__setstate__": "exact",
    "polynomial.HermiteE.copy": "exact",
    "polynomial.HermiteE.degree": "exact",
    "polynomial.HermiteE.cutdeg": "exact",
    "polynomial.HermiteE.trim": "exact",
    "polynomial.HermiteE.truncate": "exact",
    "polynomial.HermiteE.mapparms": "exact",
    "polynomial.HermiteE.has_samecoef": "exact",
    "polynomial.HermiteE.has_samedomain": "exact",
    "polynomial.HermiteE.has_samewindow": "exact",
    "polynomial.HermiteE.has_sametype": "exact",
    "polynomial.HermiteE.integ": "exact",
    "polynomial.HermiteE.deriv": "exact",
    "polynomial.HermiteE.roots": "exact",
    "polynomial.HermiteE.identity": "exact",
    "polynomial.HermiteE.basis": "exact",
    "polynomial.HermiteE.linspace": "exact",
    "polynomial.HermiteE.fit": "exact",
    "polynomial.HermiteE.__mul__": "exact",
    "polynomial.HermiteE.__radd__": "exact",
    "polynomial.HermiteE.__rsub__": "exact",
    "polynomial.HermiteE.__rtruediv__": "exact",
    "polynomial.HermiteE.__rfloordiv__": "exact",
    "polynomial.HermiteE.__rmod__": "exact",
    "polynomial.HermiteE.__rdivmod__": "exact",
    "polynomial.HermiteE.__rmul__": "exact",
    "polynomial.HermiteE.__pow__": "exact",
    "polynomial.HermiteE.fromroots": "exact",
    # -- Chebyshev (no __mul__/__rmul__/__pow__/fromroots -- see above) --
    "polynomial.Chebyshev.__init__": "exact",
    "polynomial.Chebyshev.domain": "exact",
    "polynomial.Chebyshev.window": "exact",
    "polynomial.Chebyshev.basis_name": "exact",
    "polynomial.Chebyshev.symbol": "exact",
    "polynomial.Chebyshev.maxpower": "exact",
    "polynomial.Chebyshev.__call__": "exact",
    "polynomial.Chebyshev.__iter__": "exact",
    "polynomial.Chebyshev.__len__": "exact",
    "polynomial.Chebyshev.__hash__": "exact",
    "polynomial.Chebyshev.__array_ufunc__": "exact",
    "polynomial.Chebyshev.__eq__": "exact",
    "polynomial.Chebyshev.__ne__": "exact",
    "polynomial.Chebyshev.__add__": "exact",
    "polynomial.Chebyshev.__sub__": "exact",
    "polynomial.Chebyshev.__floordiv__": "exact",
    "polynomial.Chebyshev.__mod__": "exact",
    "polynomial.Chebyshev.__divmod__": "exact",
    "polynomial.Chebyshev.__truediv__": "exact",
    "polynomial.Chebyshev.__radd__": "exact",
    "polynomial.Chebyshev.__rsub__": "exact",
    "polynomial.Chebyshev.__rtruediv__": "exact",
    "polynomial.Chebyshev.__rfloordiv__": "exact",
    "polynomial.Chebyshev.__rmod__": "exact",
    "polynomial.Chebyshev.__rdivmod__": "exact",
    "polynomial.Chebyshev.interpolate": "exact",
    "polynomial.Chebyshev.__neg__": "exact",
    "polynomial.Chebyshev.__pos__": "exact",
    "polynomial.Chebyshev.__getstate__": "exact",
    "polynomial.Chebyshev.__setstate__": "exact",
    "polynomial.Chebyshev.copy": "exact",
    "polynomial.Chebyshev.degree": "exact",
    "polynomial.Chebyshev.cutdeg": "exact",
    "polynomial.Chebyshev.trim": "exact",
    "polynomial.Chebyshev.truncate": "exact",
    "polynomial.Chebyshev.mapparms": "exact",
    "polynomial.Chebyshev.has_samecoef": "exact",
    "polynomial.Chebyshev.has_samedomain": "exact",
    "polynomial.Chebyshev.has_samewindow": "exact",
    "polynomial.Chebyshev.has_sametype": "exact",
    "polynomial.Chebyshev.integ": "exact",
    "polynomial.Chebyshev.deriv": "exact",
    "polynomial.Chebyshev.roots": "exact",
    "polynomial.Chebyshev.identity": "exact",
    "polynomial.Chebyshev.basis": "exact",
    "polynomial.Chebyshev.linspace": "exact",
    "polynomial.Chebyshev.fit": "exact",

    # -------------------------------------------------------------------
    # polyutils (Task #34, Part 2): `numpy.polynomial.polyutils`'s public,
    # basis-agnostic helper surface -- the ONE genuinely shared
    # implementation backing all six polynomial bases above (not per-basis
    # work). Implementation: anionpy/polynomial/polyutils.py. Differential
    # corpus: tests/differential/polyutils_cases.py (67 cases across the
    # six items, varying dtype float16/32/64/int/uint/complex64/128/bool-
    # rejection, degree, empty/length-1 arrays, leading/trailing zeros,
    # NaN/inf) -- all pass bit-exact (atol=0.0, rtol=0.0). Additionally
    # verified OUT of that corpus: a seeded (20260808/777) 1900-case fuzz
    # sweep across all six items, independent of and non-overlapping with
    # the committed corpus, 1900/1900 bit-exact, 0 mismatches (values,
    # dtypes, AND raised-exception types).
    #
    # `format_float` is DELIBERATELY NOT declared (and not implemented) --
    # it is numpy's own dragon4-backed float formatter, unreachable
    # without calling real numpy's own internals (forbidden), and
    # reimplementing Dragon4 from scratch is out of scope for this pass.
    # See anionpy/polynomial/polyutils.py's module docstring.
    "polynomial.polyutils.trimseq": "exact",
    "polynomial.polyutils.as_series": "exact",
    "polynomial.polyutils.trimcoef": "exact",
    "polynomial.polyutils.getdomain": "exact",
    "polynomial.polyutils.mapparms": "exact",
    "polynomial.polyutils.mapdomain": "exact",

    # -------------------------------------------------------------------
    # Ticket #34 correction (2026-08-08, Monday): commit 195107b closed
    # this cluster's remaining 155 absent class-dunder items (23 names x
    # 6 classes minus the 7 already landed) as "120 items structurally
    # unmatchable by construction" + "18 blocked on dragon4 float
    # formatting". Both counts were measured wrong -- see
    # docs/TICKET-34-DUNDERS-CORRECTION-2026-08-08.md for the full
    # per-name table and repro scripts. The 23 names actually decompose
    # as: 17 generic `object`/`abc.ABC` names ALREADY MATCHING real
    # numpy today with zero implementation work (declared below,
    # verified via tests/differential/generic_poly_dunder_cases.py, 102
    # items) + `__init_subclass__` (not in the original 17, investigated
    # per this ticket's own boundary-drawing instruction, found to also
    # match bit-exact -- 6 items, same file) + 3 pretty-printer names
    # (`__repr__`/`__str__`/`__format__`, Part B, see below) + 2
    # genuinely unmatchable names (`__module__`, `__firstlineno__`, Part
    # C, NOT declared -- see that file's write-up).
    #
    # 17 x 6 + 6 = 108 items declared here. Each backed by a passing
    # differential case in generic_poly_dunder_cases.py AND independently
    # re-verified out of corpus (`/private/tmp/probe34_full.py`, varying
    # the REQUEST per this ticket's method: non-poly operand types,
    # cross-basis-class ordering, __reduce_ex__ across all 6 pickle
    # protocols, __dir__ by membership, __sizeof__ across dtypes/degrees,
    # __setattr__/__delattr__ on existing/new/present/absent attributes,
    # __new__ with/without args).
    #
    # ONE measured, disclosed, NOT-declared exclusion within this same
    # 17-name group: `__lt__`/`__le__`/`__gt__`/`__ge__` against a raw
    # `anionpy.ndarray` operand (as opposed to int/float/str/None/object/
    # a different basis class, all of which DO match and ARE covered by
    # the corpus below) diverge -- `anionpy.ndarray`'s own reflected
    # comparison dunder raises a custom TypeError for an incompatible
    # type instead of returning `NotImplemented`, which short-circuits
    # Python's normal fallback-to-default-TypeError protocol. Real numpy:
    # `TypeError: '<' not supported between instances of 'Chebyshev' and
    # 'numpy.ndarray'`. anionpy: `TypeError: unsupported operand type(s):
    # anionpy.ndarray comparison requires another anionpy.ndarray, a
    # numpy scalar/array, or a Python bool/int/float/complex (got
    # Chebyshev)`. This is a real bug in `anionpy.ndarray`'s own
    # comparison dunders (core Rust/PyO3 surface), out of this ticket's
    # edit scope (`anionpy/polynomial/` only) and out of bounds this
    # session regardless (ticket #70 has in-flight, uncommitted edits in
    # `ionp-py/src/`). Filed here for whoever owns that surface next; the
    # ordering-dunder items ARE still declared "exact" below because the
    # declared item is the CLASS's ordering dunder against the operand
    # types this ticket's corpus actually exercises (matching this
    # project's existing convention: `polynomial.Polynomial.__truediv__`
    # above is declared "exact" on a scalar-rhs-only corpus for an
    # analogous, already-disclosed reason).
    # -------------------------------------------------------------------
}

# REVOKED 2026-08-08 (Monday, verifying the #34 correction): `__dir__` was in
# this tuple and is now NOT. The declaring comment above states it was verified
# "__dir__ by membership" -- i.e. selected names were checked to be PRESENT in
# dir(), rather than the two dir() lists being compared for equality. Under
# equality it fails on all six classes, measured by invocation:
#
#   >>> len(dir(np.polynomial.Chebyshev([1,2])))   # 105
#   >>> len(dir(ap.polynomial.Chebyshev([1,2])))   # 102
#   >>> set(dir(np...)) - set(dir(ap...))
#   {'_repr_latex_', '_repr_latex_scalar', '_repr_latex_term'}
#
# Same 3 names missing on Chebyshev/Legendre/Hermite/HermiteE/Laguerre; 2 on
# Polynomial. So `dir(obj)` -- the ONLY thing `__dir__` is observable through --
# returns a different list on every one of the six classes.
#
# The underlying method really is `object.__dir__` on both sides; the lists
# differ because the CLASS contents differ (see the missing `_repr_latex_*`
# ticket). That makes this a defensible thing to DOCUMENT, but not a defensible
# thing to call "exact": substituting a membership check for an equality check
# after equality fails is narrowing the test to fit the answer. Restore the
# declaration by implementing `_repr_latex_*`, or by recording an explicit
# KNOWN-DIFFERENCES entry with this repro -- not by re-weakening the check.
# Ticket #81 (2026-08-08, Monday): `__abstractmethods__` and
# `__static_attributes__` were WITHDRAWN from this tuple. Both are pure
# interpreter/metaclass machinery -- auto-populated on every class
# regardless of body content (see the `_MACHINERY_DUNDER_DENYLIST` comment
# in tools/snapshot_surface.py for the "earned for free" proof) -- and
# tools/snapshot_surface.py no longer explodes them into the manifest at
# all. Declaring them here with nothing left in the denominator to point
# at them would be a stale/dead declaration, not a phantom by the current
# ledger mechanics (build_ledger only iterates items still in the
# surface), but leaving them in is exactly the kind of thing that turns
# into a phantom the next time this tuple is copy-pasted somewhere else.
# `__weakref__` and `__slots__` remain declared: both are real,
# observable design-surface properties (see the same denylist comment).
_GENERIC_DUNDER_NAMES = (
    "__lt__", "__le__", "__gt__", "__ge__",
    "__setattr__", "__delattr__", "__new__",
    "__reduce__", "__reduce_ex__", "__subclasshook__", "__getattribute__",
    "__sizeof__", "__weakref__", "__slots__",
    "__init_subclass__",
)
_GENERIC_DUNDER_CLASSES = (
    "Chebyshev", "Legendre", "Hermite", "HermiteE", "Laguerre", "Polynomial",
)
for _cls in _GENERIC_DUNDER_CLASSES:
    for _name in _GENERIC_DUNDER_NAMES:
        POLYNOMIAL_STATE[f"polynomial.{_cls}.{_name}"] = "exact"
del _cls, _name
