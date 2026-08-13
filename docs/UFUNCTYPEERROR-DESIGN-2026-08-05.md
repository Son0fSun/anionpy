# `UFuncTypeError` — design note, and the private-path problem

**Date:** 2026-08-05
**Measured against:** numpy 2.5.1, live, on the installed `.so`. No rebuild;
no source changed by this note. Read-only investigation.
**Status:** task #15 is now specified. Not started. One prerequisite decision
(§4) is not mine to take alone.

## 1. Why this is worth more than its one suite failure

`UFuncTypeError` appears in revocation comments across the ledger, not just in
the one failing item:

| file | mentions |
|---|---|
| `ionp/_state/toplevel.py` | 11 |
| `ionp/_state/fft.py` | 3 |
| `ionp/_state/char_strings.py` | 2 |
| `ionp/_state/linalg.py` | 1 |
| `ionp/_state/ndarray.py` | 1 |

Those are declarations held back because ionp raises a plain `TypeError` where
numpy raises `UFuncTypeError`. The count above is a count of *comments*, not of
blocked items — some comments cite it in passing and some items are blocked for
additional independent reasons. **It is an upper bound on the payoff, not a
promise.** Anyone sequencing off this table must re-derive the per-item list
rather than reading "18 items" out of it. **ILLUSTRATIVE, NOT EXHAUSTIVE.**

## 2. What numpy actually raises — measured, not assumed

`numpy._core._exceptions` defines one public-facing name and four subclasses:

```
UFuncTypeError               -> TypeError
  _UFuncCastingError         -> UFuncTypeError
    _UFuncInputCastingError  -> _UFuncCastingError
    _UFuncOutputCastingError -> _UFuncCastingError
  _UFuncNoLoopError          -> UFuncTypeError
  _UFuncBinaryResolutionError-> _UFuncNoLoopError
```

Probed behaviour (**ILLUSTRATIVE, NOT EXHAUSTIVE** — 7 probes, one numpy
version, no `out=` dtype sweep, no structured/void dtypes):

```
>>> np.add(np.arange(3,dtype=np.float64), 1, casting='safe', out=np.zeros(3,dtype=np.int32))
UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32')
>>> np.sign(np.array([True]))
UFuncTypeError: ufunc 'sign' did not contain a loop with signature matching types ...
>>> np.less(np.array(['a']), np.array([1]))
UFuncTypeError: ufunc 'less' did not contain a loop with signature matching types ...
>>> np.bitwise_count(np.array([1.0]))
TypeError:      ufunc 'bitwise_count' not supported for the input types, ...
```

Note the last one: **not every ufunc type failure is a `UFuncTypeError`.** The
"not supported for the input types" path is a plain `TypeError`. Implementing
this cannot be done by blanket-promoting every ufunc `TypeError` — that would
convert a currently-correct case into a wrong one. A composition can be wrong
when every primitive in it is right.

### 2a. A stale claim in the repo, found in passing

`ionp-core/src/ufunc.rs:221` says:

```
///     np.equal.reduce(np.array([1,2,3], dtype=np.int32))   # now: rich UFuncTypeError
```

Measured on numpy 2.5.1:

```
>>> np.equal.reduce(np.array([1,2,3], dtype=np.int32))
TypeError: No loop matching the specified signature and casting was found for ufunc equal
```

`type(e).__name__` is literally `TypeError` — not a `UFuncTypeError` subclass.
Whoever implements #15 must not take that comment as a specification. A
recorded reason in the repo is not evidence about the current numpy. Correcting
the comment belongs in the same commit as the implementation, not before it —
it is only worth touching that file once.

## 3. The class is cheap. The messages are the work.

`tests/differential/harness.py:1677` compares raised class **names**, and
`:1688` then compares **exact message text** through `_normalize_exc_message`,
which scrubs only provably nondeterministic spans and explicitly does **not**
normalize whitespace (numpy's own messages carry load-bearing doubled and
trailing spaces).

So shipping a class named `UFuncTypeError` flips the type check and leaves the
message check exactly as failing as it is today. The deliverable is per-branch
message reproduction:

- `Cannot cast ufunc '{name}' {input|output} {n} from {dtype!r} to {dtype!r} with casting rule '{rule}'`
- `ufunc '{name}' did not contain a loop with signature matching types {types}`

with numpy's own repr formatting of dtypes and the `<class '...'>` vs
`dtype('...')` distinction visible in the probes above — the two no-loop probes
render their type tuples *differently* (`<cla...` for `sign`/`less` with a str
operand, `(dtyp...` for `add`). That difference is not yet explained and must
be before any message is written.

## 4. The prerequisite decision: numpy's class is PRIVATE

This is where the plan's Phase 1 rule does not carry over cleanly.

Phase 1 decided: *"Own classes; subclass numpy's when present."* That works for
the two classes already built that way, because both are **public**:
`numpy.linalg.LinAlgError`, `numpy.exceptions.AxisError`. The harness encodes
exactly those two at `harness.py:116`:

```python
_IONP_COMPAT_EXC_NAMES = frozenset({"LinAlgError", "AxisError"})
```

and `_ionp_compat_exc()` admits a pair only when numpy's class is a **direct
base** of ionp's, ionp's `__module__` is `"ionp"`, and numpy's module starts
with `"numpy"` (`harness.py:142-147`).

`UFuncTypeError` is **not public**. It is absent from `numpy.exceptions`
(measured: `AttributeError`) and reachable only as
`numpy._core._exceptions.UFuncTypeError`. Subclassing it means the shipped
package takes a hard runtime dependency on a numpy **private** module path —
in a project whose entire Phase 1 was spent making numpy *optional*.

Three options, none obviously right:

| option | cost |
|---|---|
| (a) subclass `numpy._core._exceptions.UFuncTypeError` when importable, else `TypeError` | matches existing idiom; depends on a private path numpy may move without notice; needs a third entry in `_IONP_COMPAT_EXC_NAMES` |
| (b) always subclass `TypeError`, never numpy's | zero numpy dependency; `except numpy...UFuncTypeError:` in user code stops catching ionp errors — a real drop-in-compat regression |
| (c) declare the divergence and keep raising plain `TypeError` | free; leaves every item in §1 blocked, permanently |

The `except TypeError:` direction — by far the common one in real user code —
**already works today**, because `UFuncTypeError` subclasses `TypeError` and
ionp already raises `TypeError`. Only `except UFuncTypeError:` and
`type(e).__name__` checks distinguish. That materially lowers the stakes of
(b) and (c) and should be weighed before defaulting to (a) on consistency
grounds.

**I am not taking this decision unilaterally.** It changes what "numpy
optional" means, which was Mother's call in Phase 1 and stays hers here.

## 5. Not checked

- Whether the five subclasses need to be distinguishable at all, or whether
  every currently-blocked item only ever observes the base name.
- The `<class '...'>` vs `dtype('...')` rendering difference in no-loop
  messages (§3). Leading suspect: operand-vs-dtype provenance in numpy's
  formatter, untested.
- Any numpy version other than 2.5.1.
- Whether the four `fft`/`char_strings`/`linalg`/`ndarray` mentions are true
  blockers or passing references.
- Whether `out=`-related paths in `tests/differential/ufunc_out_axis_cases.py:18`
  share a code path with the call-site paths probed here.

## 6. Verdict

Specified, not started, and **correctly blocked** at §4 rather than guessed at.
The work after that decision is message-text archaeology, not class plumbing —
budget accordingly.

---

# CORRECTION, 2026-08-06 — §4 is decided, and §6's "not started" was WRONG

**Measured live against the installed `.so`, not read off this file.**

## The decision

Mother chose **option (b): always subclass plain `TypeError`, never numpy's.**
Reasoning and accepted costs recorded in `PHASE-0-DECISIONS-2026-08-06.md`.

## The thing this document got wrong

§6 declared the work "specified, not started." **It ships today, and it ships
exactly as option (b) requires.** `ionp-py/src/lib.rs:49`:

```rust
pyo3::create_exception!(_ionp, UFuncTypeError, PyTypeError);
```

Measured identity — `ionp.add(..., casting='safe', out=<int32>)`:

```
class      : <class '_ionp.UFuncTypeError'>
mro        : ['_ionp.UFuncTypeError', 'builtins.TypeError', ...]
IS numpy's UFuncTypeError        : False
caught by `except np...UFuncTypeError:` : False
caught by `except TypeError:`           : True
```

Messages already match numpy verbatim on the probes I ran:

```
add safe-cast out : numpy UFuncTypeError "Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32')"
                    ionp  UFuncTypeError  <identical>
sign(bool)        : both UFuncTypeError, identical no-loop text
bitwise_count(f8) : both plain TypeError, identical text   <-- §2's trap, handled correctly
```

Note the third: the "not supported for the input types" path correctly stays a
**plain** `TypeError` in ionp too. §2 warned that blanket-promotion would break
this case. It was not broken.

**ILLUSTRATIVE, NOT EXHAUSTIVE** — 3 probes, numpy 2.5.1, no `out=` dtype
sweep, no structured/void dtypes, no per-subclass distinction, and none of the
18 `_state` revocation comments re-tested against the current binary.

## A gap I looked for and did NOT find

I expected `ionp.UFuncTypeError` to be missing from the public surface, since
`hasattr(ionp, "UFuncTypeError")` is `False`. Measured before filing it:

```
>>> hasattr(numpy, "UFuncTypeError")
False
```

numpy does not expose it publicly either — it lives only at
`numpy._core._exceptions`. ionp exposing it only as `_ionp.UFuncTypeError` is
therefore **correct parity, not a defect.** Recording the non-finding because
a hypothesis a measurement killed is worth as much as one it confirmed, and I
was one commit away from "fixing" matching behaviour into a divergence.

Related and genuinely absent, but a different item: `ionp.exceptions` does not
exist at all (`np.exceptions` is 7 items). That is Phase 4 coverage work, not
this task.

## Status

**Task #15: DONE, verified out of corpus.** No code change required by the
decision — it ratified what already shipped. The `_state` revocation comments
in §1 that cite `UFuncTypeError` as the blocker are now **suspect**: they may
be stale, or those items may be blocked for the independent message-text
reasons in §3. Each must be re-measured individually before any of the up-to-18
items is declared. That re-measurement has NOT been done.

The stale-comment defect flagged in §2a (`ufunc.rs:221`) is still open and
still needs correcting in whatever commit next touches that file.
