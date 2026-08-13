# Contributing to anionpy

Read this before you write a line of code. The bar here is different from most
projects: **the differential test corpus is the contribution, not a formality
attached to one.**

## The workflow, in order

1. **Write the differential corpus first.** Before implementing anything, add
   or extend a case file under `tests/differential/` (see `registry.py` and
   any existing `*_cases.py` for the shape) that feeds real NumPy and the not-
   yet-implemented `anionpy` item the same inputs. It will fail — that's
   expected and correct. It is the specification.
2. **Implement.** Make the corpus pass. Nothing else counts as "implemented."
   A function that exists, runs, and looks plausible but has no passing
   differential case is not credited — see `tools/coverage.py`'s `undeclared`
   and `absent` states.
3. **Verify OUT of corpus.** Run the differential suite and confirm the new
   item passes on inputs beyond the ones you hand-picked while writing it —
   edge cases, dtypes, shapes you did not specifically design for. A corpus
   that only tests what the implementation was built against tests nothing.
4. **Declare.** Only once the item passes does it get declared in
   `__ion_state__`. Declaring before the corpus is green is exactly the
   failure mode this project exists to avoid — see `KNOWN-DIFFERENCES.md` for
   real audited instances of that going wrong.

## How comparison works

Results are compared **byte-based**, via `.tobytes()` — not `==`, not
`np.allclose()`. Bit-for-bit or it does not count as exact.

A **tolerance cannot be added to make a mismatch go away.** `ItemSpec` in
`tests/differential/registry.py` enforces this structurally, not by
convention: `__post_init__` unconditionally refuses to construct any
`ItemSpec` with a non-zero legacy `atol`/`rtol`, regardless of what
justification text accompanies it. There used to be a path where a
justification string alone was enough to accept an arbitrary tolerance; it let
unearned slack into the ledger and was removed. The only tolerance mechanism
that remains constructible (`epsilon_tolerance` / `ulp_tolerance`) requires:

- a per-dtype bound,
- backed by a seeded sweep of at least the registry's declared minimum sample
  count,
- where the declared bound must **equal** the measured maximum for that dtype
  — not merely be greater than it. Padding a tolerance "for headroom" is
  rejected the same as an unjustified one.

If your change needs a tolerance, the sweep that produced the number is part
of the contribution, in-tree, not a claim in a commit message.

## A green suite is necessary but not sufficient

Passing the differential suite means your change didn't break anything the
suite already checks. It does not mean the new item is correct on inputs the
suite doesn't cover, and it does not excuse skipping step 3 above. If you
found a real divergence from NumPy and chose to leave it undeclared or
document it rather than paper over it, that is the correct call — see
`KNOWN-DIFFERENCES.md` for the standing record of exactly that kind of
decision, including at least one case (`matrix.__setitem__`) that was
implemented, found divergent, and deliberately declined rather than force-fit
into a passing state.

## Running things

- Interpreter: use the project's `.venv`, not a system Python.
- Differential suite: `python tests/differential/run.py --out <fresh-path>.json`
  — always a fresh output path; a stale JSON silently produces misleading
  coverage numbers.
- Coverage: `python tools/coverage.py --tests <path-from-above>.json`. Run
  with no `--tests` and it silently reports 0.0% — always pass the file.
- The compiled extension is shared mutable state across concurrent sessions
  working on the same tree. See `DEVELOPING.md` for what that has already
  cost this project once.

## What a PR needs

- The corpus addition/change, and the implementation, in the same PR — not
  split across a "tests later" follow-up.
- A fresh coverage measurement if the change affects declared items, quoting
  the commit it was measured at.
- No edits to case files purely to make a failing case pass without also
  fixing the underlying divergence (or explicitly declining and documenting
  it, per the process above).

## License

By contributing, you agree your contribution is licensed under this
project's MIT license (see `LICENSE`).
