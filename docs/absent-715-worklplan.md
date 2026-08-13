# The 715 absent items — work plan

Survey by a read-only agent 2026-08-03, **with my corrections**. Binary
1785740822 (00:07:02), stable across every measurement below. Where I re-measured
a claim myself it is marked MEASURED BY ME and my result governs.

## Namespace breakdown (MEASURED BY ME, from `coverage.py --list absent`)

| namespace | absent |
|---|---:|
| `<toplevel>` | 249 |
| **`ma.*`** | **225** |
| `ndarray.*` | 97 |
| `random.*` | 63 |
| `char.*` | 24 |
| `fft.*` | 12 |
| `linalg.*` | 11 |
| `rec.*` | 9 |
| `polynomial.*` | 8 |
| `ctypeslib.*` | 5 |
| `strings.*`, `lib.*` | 4 each |
| `testing.*` | 3 |
| `emath.*` | 1 |

**`ma.*` = 225 items, 19% of the whole 1180 target, zero Rust, and it was
invisible to every plan I had written before today.** `hasattr(ionp,"ma")` is
False. Masked arrays need a new array-like type (data + mask) plus masked
semantics for ~200 functions, and a decision about how that composes with the
current immutable `ndarray`. This is the single largest lever in the ledger and
it needs its own design doc before anyone estimates it. **Escalate to Mother.**

## *** CORRECTION: `random` is NOT a multi-week all-or-nothing project ***

The survey flagged `random.*` as "suspect, likely construction-only, do not
partial-declare." **MEASURED BY ME — that is wrong, and the truth is much
better.**

Of 31 distributions probed against `numpy.random.default_rng(99)`:

**BIT-EXACT: 21. DIFFERS: 0. MISSING: 10.**

Bit-exact (dtype + shape + `tobytes()` identical to numpy's PCG64 stream):
`random, standard_normal, standard_exponential, standard_cauchy, exponential,
normal, uniform, gamma, beta, chisquare, standard_t, f, laplace, logistic,
lognormal, pareto, power, rayleigh, weibull, gumbel, integers`

Also verified: same seed reproduces; different seeds diverge.

Of numpy's 45 `Generator` methods, **23 are present and 22 are missing**:
`binomial, bit_generator, choice, dirichlet, geometric, hypergeometric,
logseries, multinomial, multivariate_hypergeometric, multivariate_normal,
noncentral_chisquare, noncentral_f, permutation, permuted, poisson,
negative_binomial, shuffle, spawn, triangular, vonmises, wald, zipf`

So the hard part — a bit-exact PCG64 stream and the ziggurat/transform
machinery on top of it — **is already done and provably correct.** What remains
is discrete/multivariate distributions and the shuffle family, each an
independent kernel over a stream that already matches. `random` is not a
research project; it is a finishing job, and a large block of near-free
declarations sits in the 21 already-exact ones.

The all-or-nothing warning was right in ONE respect and I am keeping it: a
distribution can match on mean/variance and be byte-wrong. Every declaration
here must be `tobytes()`-compared against numpy at a fixed seed, never
statistically compared. That bar is met by the 21 above.

## *** CORRECTION: `__imatmul__` is real in-place, not a rebind ***

The survey reported `ndarray.__imatmul__` absent from `vars(ndarray)` and
concluded `a @= b` was silently falling back to `a = a.__matmul__(b)`.
**MEASURED BY ME: `"__imatmul__" in vars(ionp.ndarray)` is True; after `a @= b`
the object id is UNCHANGED and the values ARE updated — identical behaviour to
numpy on the same recipe.** It is genuinely in-place.

This weakens the "immutability blocks 10 items" bucket: `x[0] = 9` does raise
`TypeError`, so `__setitem__`/`__delitem__` are genuinely blocked, but the
in-place ARITHMETIC dunders must each be measured rather than assumed dead by
association. Do not treat that bucket as a block of 10 without per-item proof.

## The genuinely valuable finding: "passes today's corpus but declaring it
would be a LIE"

This bucket (~35 items) is the survey's best work and I am keeping it intact.
Every member currently shows `verdict: pass`, so a naive declare-everything-
green pass would ship all of them as false positives — the exact sin that cost
16 revocations this morning. Families:

- **view/metadata lie:** free-fn `transpose/reshape/squeeze/swapaxes/moveaxis/
  broadcast_to/broadcast_arrays` — NOW FIXED by f3848b1, re-check before declaring.
- **real copy where numpy views:** `ravel`, and `diag/diagonal/rot90/rollaxis`
  (needs true aliasing).
- **size-1-axis stride convention:** `atleast_1d/2d/3d`, `expand_dims`.
- **F-order/output layout:** `argwhere, hstack, vstack, sort, sort_complex,
  unique, delete, linspace, concatenate, stack, trace, diff`.
- **matmul-family kwarg gap:** `matmul/matvec/vecdot/vecmat` reject
  `order=`/`casting=`/`subok=` that numpy 2.5.1 accepts — passes only because
  the corpus is thin.
- **wrong container:** `linalg.svd` (`tuple` vs `SVDResult`).
- **`arange`** negative-step unsigned clamping + fractional-step rule (two prior
  attempts failed); **`array`** missing `ndmax=`; **`asarray`** view preservation.

## Structurally blocked — decisions for Mother, not effort

- **dtype enum (~22 items).** `ionp-core/src/dtype.rs` has exactly 14 numeric
  variants; no str/bytes/object/void/datetime. Blocks `object_, str_, bytes_,
  void, datetime64, timedelta64, record, recarray, busday_*, datetime_as_string,
  datetime_data` and all 9 `rec.*`. One dtype expansion unlocks all of them.
  NOTE: most of `char.*`/`strings.*` is NOT blocked — those are Python-string
  ops over `strings.rs`, which already works.
- **immutability.** `__setitem__`/`__delitem__` genuinely blocked. The in-place
  arithmetic dunders are NOT — see the correction above.

## Recommended sequence

1. Bucket A declare-only (~20, zero Rust): `finfo, nanstd, nanvar`, 11 `fft.*`,
   `linalg.{cross,tensordot,matrix_power,tensorsolve}`, several `ndarray.*` attrs.
2. **The 21 bit-exact `random` distributions** — declare with `tobytes()` tests.
   Largest near-free block found today; the survey would have deferred it.
3. Top-level wiring of `outer/tensordot/cross` (+ `matrix_power`, `tensorsolve`
   — MEASURED BY ME as also linalg-only and undeclared). One line each.
4. Re-verify the 7 view items now that f3848b1 landed; declare what survives.
5. Size-1-axis stride fix (4 items), then F-order root cause (~12 items).
6. `dot/vdot/inner`, `cumulative_sum/cumulative_prod`.
7. The remaining 22 `random` distributions — independent kernels over a stream
   already proven correct.
8. `ma` design doc → Mother. `einsum`, polynomial, file I/O, printing last.

## Explicitly NOT measured

- Per-item cost for the ~40 moderate-kernel items — sized by reading module
  presence, not by building.
- Whether `fliplr`/`flipud` share `flip()`'s defect (inferred from sibling).
- Whether `linalg.diagonal`/`linalg.matrix_transpose` inherit top-level defects.
- The 22 missing `Generator` methods were probed for PRESENCE only; the 10 I
  called MISSING in the distribution sweep raised AttributeError, not wrong values.
- dtype variation (uint/int32/float32/float16) anywhere in this report.
- `ma`'s overlap with the dtype-blocked and immutability-blocked buckets.
