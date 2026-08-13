# `.accumulate` / `.reduceat` are 1-D only — and `.reduceat` fails SILENTLY

**Date:** 2026-08-05
**Measured against:** numpy 2.5.1, live, on the `.so` built at `effe853`.
**Status:** open defect, not scheduled. Found while fixing an unrelated bug.
**Severity:** one of the two failure modes is a silent wrong answer.

## How this was found

While fixing power's negative-exponent check for `.accumulate`/`.reduceat`
(commits `8e5fa57`, `effe853`), the implementation turned out not to need
multi-axis handling at all: `accumulate_math_binary` and
`reduceat_math_binary` have **no `axis` parameter in their Rust signatures**.
They are hard-coded 1-D.

That is a much larger, entirely separate, pre-existing gap. It was correctly
declined as out of scope for that fix and is recorded here instead of being
absorbed into an unrelated commit.

## Measured — and this is NOT power-specific

`np.add`, not `np.power`. Every ufunc is affected.

```
>>> a = np.array([[1,2,3],[4,5,6]], dtype=np.int64)

>>> np.add.accumulate(a, axis=0)
[[1, 2, 3], [5, 7, 9]]
>>> ionp.add.accumulate(a, axis=0)
ValueError: buffer of length 2 cann...          # axis=1 and axis=-1 likewise

>>> np.add.reduceat(a, [0,1], axis=1)
[[1, 5], [4, 11]]
>>> ionp.add.reduceat(a, [0,1], axis=1)
[1, 4]                                          # <-- NO ERROR. WRONG SHAPE. WRONG VALUES.
```

**ILLUSTRATIVE, NOT EXHAUSTIVE** — one dtype, one ufunc, 2-D only, four axis
values. No 3-D, no `out=`, no `dtype=`, no non-contiguous input.

## Why the two modes rank differently

`.accumulate` **raises**. A caller finds out. The message is misleading
(`"buffer of length 2 cannot..."` names an internal buffer, not the
unsupported `axis=`), but the failure is loud.

`.reduceat` **returns a plausible-looking array of the wrong shape with the
wrong values and no warning.** `[1, 4]` is a perfectly ordinary int64 array.
Nothing downstream can tell it is garbage. That is strictly worse than the
crash, and it is the reason this note exists rather than a line in a backlog.

A silent wrong answer in a numpy replacement is the most expensive defect
class we can ship: the whole value proposition is that callers do not have to
check.

## What is NOT yet known

- Whether `axis=0` on `.reduceat` is also silently wrong or happens to be
  right by accident on the default axis. Not probed.
- Whether the 1-D restriction reaches the Python binding as an explicit
  rejection anywhere, or whether the `[1, 4]` result is the binding silently
  dropping the `axis=` kwarg. **Leading suspect: the kwarg is accepted and
  ignored.** Untested — and if confirmed, the same silent-ignore may affect
  other kwargs on these two methods.
- Whether `.reduce` (which IS axis-aware) shares any of this code path.
- Whether any currently-declared item in `ionp.__ion_state__` depends on
  these methods and would therefore be a FALSE declaration today. **This must
  be checked before the next declaration sweep.**

## Corpus status

The differential corpus does **not** cover N-D `.accumulate`/`.reduceat`.
That is why the suite reports 35 failures and none of them is this. Same
shape of problem as the `order='K'` multi-operand blind spot recorded in
`STRIDE-PERM-GUARD-2026-08-05.md`: a real divergence in code the suite
reports as covered, because nothing looks for it.

Per the standing loop, the corpus case must be written **before** any fix,
and landing it will move the suite baseline upward. That delta is the
instrument starting to measure something it was blind to, not a regression,
and whoever lands it must say so in the same commit.

## Commit-hygiene defect recorded in the same breath

Commits `8e5fa57` and `effe853` are correct in content — independently
re-verified, 18/18 behavioural checks matching numpy, including the
`reduceat(a,[0,3,3]) -> [729,-4,0]` degenerate-segment passthrough.

But both **omit the `[Monday]` prefix and the `Co-Authored-By` trailer** that
this repo requires. The immediately preceding commit `7026b15`, from the same
author, has both. History is deliberately NOT rewritten to fix them: five
sessions share this tree and the hashes are already cited in the task log.
The omission is recorded here instead, which is what the attribution policy
is actually for — an auditable trail, not a tidy one.
