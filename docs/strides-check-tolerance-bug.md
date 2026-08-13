# `_strides_match` conflates layout with value — diagnosis and fix design

**Status:** diagnosed 2026-08-02, NOT yet implemented. Must be fixed before
`ItemSpec.check_strides` is ever enabled broadly.

**Location:** `tests/differential/harness.py:955-1027`, called from `run_case`
at line 1505. Gate is `ItemSpec.check_strides: bool = False`
(`tests/differential/registry.py:869`).

## What it does today

`_strides_match` is a POST-check: `run_case` calls it only after
`compare_values` has already said the VALUES are correct. It then checks three
things through `np.asarray()` on both sides:

1. buffer-protocol `.strides` equality
2. `C_CONTIGUOUS` / `F_CONTIGUOUS` flag equality
3. `tobytes('A')` — literal physical byte sequence in memory order

## The defect

Step 3 has **no concept of tolerance**. `compare_values` has four passing
branches — bit-exact, exact-integer, ULP-tolerant, and epsilon-tolerant. For
the two tolerant branches, the values are legitimately *not* bit-identical;
that is the entire point of the declared `ulp_tolerance` /
`epsilon_tolerance`, each of which carries a written justification.

For such an item, `tobytes('A')` differs **even when the layout is perfect**,
and the failure is reported as:

> physical byte layout mismatch (tobytes('A'), the ground truth strides/flags
> only describe)

which names layout as the defect when the real difference is a last-ulp value
difference the item is explicitly declared to permit. An item's own declared
tolerance is silently overridden by a check running downstream of it.

Measured blast radius: **16 items** would be mis-flagged this way. Every one of
them is a false positive — a correct item reported as a layout defect.

This is the same family as lessons #42 and #50: an instrument whose failure
mode is *structurally* unable to distinguish the thing it names from a
different thing entirely.

## Why the byte check exists at all — and why it is the wrong instrument

Step 3 was added for a real defect: ionp's `order='F'` implementation was found
to **relabel the `.strides` metadata field without reordering the underlying
buffer**. The original version of this function read `ionp_out.strides` — the
component's own self-report — and compared it to numpy's real strides. That is
a test that asks a component to confirm its own claim, so it could not fail.

The correction routed both sides through `np.asarray()`. But `np.asarray` reads
the strides the buffer protocol *exports*. If the relabel bug is present in the
buffer-protocol export itself, `asarray` inherits the lie and step 1 agrees with
it. Step 3's byte comparison is genuinely independent of that lie — which is
why it was added.

So the byte check is catching something real. It is simply catching it with an
instrument that also responds to value differences, and therefore cannot tell
the two apart.

## The fix

Replace the value-sensitive ground truth with a value-**independent** one.

**Step 0 (new): self-report vs buffer-protocol agreement, on the ionp side
alone.**

```
reported = tuple(getattr(ionp_out, "strides", ()) or ())
observed = tuple(np.asarray(ionp_out).strides)
if reported and reported != observed:  -> relabel defect
```

This catches the exact bug step 3 was written for — a `.strides` label that
disagrees with the physical buffer — and it involves **no values whatsoever**,
so no tolerance can interact with it. It is also a strictly sharper test: it
localises the defect to ionp rather than inferring it from a byte diff against
numpy.

**Step 3 (amended): make it tolerance-aware.**

`compare_values` already returns `tolerant` as its third element. Thread that
into `_strides_match` and run the `tobytes('A')` comparison **only when the
comparison was bit-exact**. When the item passed tolerantly, steps 0/1/2 stand
alone — and with step 0 present they cover the original defect completely.

Do NOT instead relax step 3 with a tolerance of its own. Byte sequences in
memory order are not a numeric space; "close bytes" is not a meaningful
predicate, and building one would manufacture exactly the kind of blunt
instrument that measures a gap rather than finding one (lesson #52).

## What this does not fix

- It does not validate layout for tolerant items beyond strides + flags. That
  is a real, disclosed reduction in coverage for those 16 items, and it is the
  correct trade: strides + flags + step 0 is a sound layout check, whereas the
  current byte check is an unsound one.
- It says nothing about whether output memory layout should be part of the
  compatibility contract at all. That remains an open question for Mother.
