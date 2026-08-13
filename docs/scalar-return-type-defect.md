# Full reductions return a 0-d array where numpy returns a scalar

**Status:** measured 2026-08-02 against binary mtime 22:25:53. **16 items
currently declared `exact` are wrong.** Revocation pending — the declaration
files are held by a concurrent build agent.

## The defect

Every ionp operation that fully reduces to a single value returns a 0-d
`ionp.ndarray`. Real numpy returns a numpy **scalar** — `np.float64`,
`np.int64`, `np.bool_`. These are different types with different behaviour.

Confirmed across **22 functions**, every dtype tried, one root cause in the
reduction/contraction result-wrapping path:

`sum, mean, max, min, amax, amin, ptp, argmax, argmin, nansum, nanmean,
nanmax, nanmin, count_nonzero, any, all, prod, std, var, trace,
matmul(1-d,1-d), vecdot`

`cumsum`/`cumprod` are correct — they don't fully reduce.

Axis-wise reductions (`sum(a, axis=0)`) are also correct: both sides return
arrays. **Only the full-reduction path is affected.**

## It is not a type quibble — 11 of 15 behavioural probes diverge

| probe | numpy | ionp |
|---|---|---|
| `float(r)` | `6.0` | **TypeError** |
| `hash(r)` | `6` | **TypeError: unhashable** |
| `{r: 'v'}` (dict key) | works | **TypeError** |
| `json.dumps(float(r))` | `'6.0'` | **TypeError** |
| `r.is_integer()` | `True` | **AttributeError** |
| `isinstance(r, float)` | `True` | `False` |
| `str(r)` | `'6.0'` | `'6.'` |
| `repr(r)` | `'np.float64(6.0)'` | `'array(6.)'` |
| `r + 1` | `float64` | `ndarray` |
| `r[()]` | `float64` | `ndarray` |
| `len(r)` | TypeError (unsized) | TypeError (different message) |

`float(sum(x))` raising `TypeError` is not a compatibility nuance; it breaks
ordinary user code on the single most common call in the library.

Note this is a *stacked* defect: numpy's own 0-d arrays support `float()`
fine. ionp's 0-d array is worse than a numpy 0-d array — it lacks `__float__`
entirely. Fixing the scalar-return path does not automatically fix that; both
need doing.

## Why it survived — instrument blindness, and the largest instance yet

`compare_values`' **array path** compares dtype, shape and values through
`np.asarray()` on both sides. For this defect:

```
np.asarray(np.float64(6.0))      -> array(6.), dtype float64, shape ()
np.asarray(ionp 0-d ndarray)     -> array(6.), dtype float64, shape ()
```

Every axis the instrument inspects agrees. **The check cannot fail on this
defect.** It compares what an object *contains* and never what it *is*.

`_compare_scalar_like` DOES check `type(np_out) is type(ionp_out)` — but only
items carrying `scalar_like=True` reach it, and those are for genuinely
non-array returns (`.shape` tuples, `.ndim` ints, `.dtype` objects). Every
reduction is an array item and takes the blind path.

This is the same family as the signed-zero blindness, the byte-comparison
blindness, the tuple-walking blindness and the fixture that destroyed the
property under test. It is the biggest member so far: **16 declared items.**

## Consequences for the ledger

16 `exact` declarations are false and must be revoked: `sum, mean, max, min,
amax, amin, ptp, argmax, argmin, nansum, nanmean, nanmax, nanmin,
count_nonzero, any, all`.

**DONE 2026-08-02.** All 16 revoked in `ionp/_state/toplevel.py` (key commented
out in place, revocation reason and the superseded declaration both preserved).
Verified by two-step ledger: **457 → 441 (37.373%)**, phantom 0, untested 0,
failing still 5. (This doc originally read "445 → 429"; the builder's `bcb461c`
landed +12 between the measurement and the revocation, so both endpoints move by
12. The delta of 16 is the number that was ever load-bearing.)

A further 6 are undeclared and carry the same defect, so nothing is lost
there: `prod, std, var, trace, matmul, vecdot`. (`std`/`var` were already
known-broken for reduction layout — this is a second, independent defect in
the same items.)

## The fix, and the wrong fix

Right: make the full-reduction path wrap its result in the numpy-scalar
equivalent for the result dtype, not a 0-d array. One place, 22 items.

**Wrong, and I want this written down:** do not "fix" the tests by comparing
`.item()` on both sides, and do not set `scalar_like=True` on reductions to
force a type check. The first builds the identical blindness into a second
instrument; the second abuses a flag whose semantics are for non-array
returns and routes around the defect rather than finding it. The instrument
needs a return-type check on the ARRAY path — that is the real repair, and it
will likely expose more than these 22.

## Open question this raises for the harness

Should the array path assert `type(np_out) is type(ionp_out)` outright?
Probably not verbatim — numpy sometimes returns subclasses, and ionp arrays
are legitimately not `np.ndarray`. The narrow, defensible rule is: **if numpy
returned a numpy scalar, ionp must return the matching numpy scalar type; if
numpy returned an ndarray, ionp must return an ionp array.** That
distinction is cheap to test and catches exactly this class.
