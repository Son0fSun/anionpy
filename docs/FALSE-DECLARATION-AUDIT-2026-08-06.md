# False-declaration audit: does any declared-exact item depend on the N-D `.accumulate`/`.reduceat` gap?

**Date:** 2026-08-06
**Auditor:** read-only audit agent (no edits, no builds, no runtime timing).
**Question:** `docs/ACCUMULATE-REDUCEAT-ND-GAP-2026-08-05.md` documents that
`ufunc.accumulate`/`ufunc.reduceat` (the Rust functions `accumulate_math_binary`
/ `reduceat_math_binary`, plus their `Plain`-op siblings `accumulate_binary` /
`reduceat_binary`) are hard-coded 1-D, and that `.reduceat` on N-D input
returns a silently wrong answer. Does any item **currently declared `exact`**
in `ionp/_state/*.py` depend on that broken path?

## What I measured against

- **Repo HEAD:** `edea4ab85431f72317d4d8a9b000f814e0051484`.
- **Working tree note:** `git status` shows an uncommitted modification to
  `ionp-core/src/ufunc.rs` at audit time (a docstring + view-offset fix to
  `check_int_pow_no_negative_self`, the negative-integer-exponent check for
  `power`). I read the diff and confirmed it does **not** touch
  `accumulate_math_binary`, `reduceat_math_binary`, `accumulate_binary`,
  `reduceat_binary`, `accumulate_axis`, or `do_accumulate_axis` — it only
  fixes an unrelated view-offset bug in the `power` negative-exponent guard.
  I did not rely on this diff for any conclusion; it's noted for the record
  since another agent appears to be mid-edit on the same file this audit
  reads.
- **Installed binary actually exercised:** `ionp/_ionp.abi3.so`,
  sha256 `2e528877fd785ac48aeffe4772573907107a7f024c00d71c1a309474c7166c9`,
  mtime 2026-08-06 08:42 local. This is an editable install
  (`pip show ionp` → `Editable project location: ~/Monday/ionp`),
  so "installed" and "in-tree" are the same file; I did not rebuild it.
  numpy version in the venv: 2.5.1.
- All probes run with `.venv/bin/python`, executed from `/private/tmp`
  (so `sys.path[0]` resolves the installed package, not a neutral-cwd
  shadow of the repo's `ionp/` source dir), importing `ionp` from
  `~/Monday/ionp`.

## The declared-item enumeration

`ionp.__ion_state__` (built by `ionp/_state/__init__.py` from the ten
per-block modules) currently declares **669 items, all in state `exact`**
(0 in state `ion`). I got this by importing `ionp` directly and reading
`ionp.__ion_state__`, not by guessing:

```
sys.path.insert(0, '~/Monday/ionp')
import ionp
len(ionp.__ion_state__)                       # 669
{v for v in ionp.__ion_state__.values()}      # {'exact'}
```

I did **not** use `tools/coverage.py --list exact` for this enumeration:
without a `--tests` report, every row's verdict collapses to `untested` (by
design — `load_test_results` credits nothing without a report), so
`--list exact` prints zero rows even though 669 items carry declared state
`exact`. `tools/coverage.py --tests <path>` needs a differential-suite JSON
report as input, and generating a fresh one means running the differential
suite — out of scope for a read-only, no-build audit that must not touch the
build lane. Reading `ionp.__ion_state__` directly is the more direct source
of truth for "what is declared" anyway (it's literally the dict
`coverage.py` itself reads via `declared = dict(getattr(root,
"__ion_state__", {}) or {})`), so this substitution does not weaken the
enumeration.

## Search scope (stated separately from conclusions)

**Searched:**
- All ten `ionp/_state/*.py` modules — grepped for
  `cumsum|cumprod|accumulate|reduceat|nancumsum|nancumprod|logaddexp`
  (case-insensitive) to find every declared item whose name or comment
  mentions the suspect surface.
- `ionp-core/src/ufunc.rs` and `ionp-py/src/*.rs` (all files) — grepped for
  the call sites of `accumulate_math_binary`, `reduceat_math_binary`,
  `accumulate_binary`, `reduceat_binary`, `accumulate_axis`,
  `do_accumulate_axis` to build the actual call graph: which public
  entry points route through the 1-D-hardcoded pair vs. the axis-aware
  `accumulate_axis`.
- All `*.py` files under `ionp/` (non-`_state`) — grepped for
  `.accumulate(` / `.reduceat(` literal call sites, to catch any pure-Python
  wrapper that might shortcut through the ufunc-method surface. Zero hits.
- Behavioural probes (below) against the installed `.so`.

**Not searched:**
- `ionp-ion/` and other workspace crates crates — not grepped, on the
  reasoning that `ufunc.accumulate`/`.reduceat`/`accumulate_axis` are
  `ionp-core`/`ionp-py` symbols and the doc's own repro is scoped there, but
  I did not verify these crates contain zero references to the two
  functions. Named as an explicit unknown, not assumed clean.
- `tests/differential/*.py` corpus files — not read for this audit (the
  question is about the ledger's *declarations*, not the corpus's
  coverage of them; the ND-gap doc already establishes the corpus doesn't
  cover N-D `.accumulate`/`.reduceat`).
- Full 669-item ledger was not individually re-verified line by line for
  unrelated defects — only the subset whose name/comment matched the
  accumulate/reduceat/cumsum/cumprod/logaddexp grep, per the task's
  suspect list.

## Findings

### Call-graph result (static)

Two genuinely separate implementations exist, confirmed by grep across all
call sites in `ionp-core/src` and `ionp-py/src`:

| Rust function | Axis-aware? | Only caller(s) |
|---|---|---|
| `accumulate_binary` / `accumulate_math_binary` | **No** (1-D hardcoded, per the ND-gap doc) | `ionp-py/src/lib.rs` `Ufunc.accumulate()` method only |
| `reduceat_binary` / `reduceat_math_binary` | **No** (1-D hardcoded; silent wrong answer on N-D, per the ND-gap doc) | `ionp-py/src/lib.rs` `Ufunc.reduceat()` method only |
| `accumulate_axis` (`ionp-core/src/ufunc.rs:7479`) | **Yes** (`axis: usize` parameter) | `ionp-py/src/reductions.rs` `do_accumulate_axis` (backs `cumsum`, `cumprod`, `nancumsum`, `nancumprod`) and `ionp-py/src/ndarray_attrs.rs` (backs `ndarray.cumsum`/`ndarray.cumprod`) |

Confirmed directly in `ionp-py/src/lib.rs` (`Ufunc.accumulate`,
`Ufunc.reduceat`): both methods take an `axis: i64` *parameter* but the
body is `let _ = (axis, dtype);` — the kwarg is accepted and **silently
ignored**, exactly the leading suspicion the ND-gap doc named as untested.
This confirms that suspicion for the record, but it is not new information
about the declared ledger: **no ledger item is keyed on `"<ufunc>.accumulate"`
or `"<ufunc>.reduceat"`** — `tools/numpy_surface.json`'s `exploded` classes
(the only mechanism that produces dotted method-style ledger keys) do not
include a `ufunc` class at all, so the `.accumulate`/`.reduceat` method
surface is not a trackable ledger item to begin with, declared or not.
Every plain-ufunc `"exact"` declaration I sampled in `toplevel.py` (e.g.
`isfinite`, `subtract`, `negative`, `logaddexp`) carries an explicit repeated
scope note: *"this 'exact' grades the ufunc CALL path only, not
`.reduce`/`.accumulate`/`.outer`"* — i.e. the ledger already disclaims this
surface everywhere it appears, rather than silently claiming it.

`cumsum` / `cumprod` / `nancumsum` / `nancumprod` / `ndarray.cumsum` /
`ndarray.cumprod` / `ma.cumsum` / `ma.cumprod` — the eight items whose
*names* mention the suspect surface and which *are* declared `exact` — all
route through `accumulate_axis`, not `accumulate_math_binary`/
`accumulate_binary`. They are a different implementation, not a wrapper
around the broken one. Static reading alone doesn't prove they're correct
(the task's own warning: "a wrapper may reshape to 1-D first and be
perfectly correct" cuts both ways — a *different* implementation must still
be measured, not presumed fine), so I measured them.

### Behavioural result (measured)

For all eight cumsum/cumprod-family items, I ran differential probes against
real numpy 2.5.1, varying (per the method warnings in the task):
rank 2/3/4; int64 and float64; C-order and F-order; `axis=0`, `axis=-1`,
and the last positive axis; **real ionp views** obtained by slicing an
*ionp* array (`ionp.asarray(np_array)[1:]`, `[::2]`, `.T`), never
`ionp.asarray(numpy_slice)`; and, for `nancumsum`/`nancumprod`, arrays with
genuine NaN values injected (not just NaN-free "nan" function names).

```
ILLUSTRATIVE, NOT EXHAUSTIVE repro (one representative case out of the
grid actually run):

  a_np = np.asarray([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=np.int64)
  a_ionp = ionp.asarray(a_np)[1:]          # real ionp VIEW, offset+strided
  np.cumsum(a_np[1:], axis=-1)             # -> [[4,9,15],[7,15,24],[10,21,33]]
  ionp.cumsum(a_ionp, axis=-1)             # -> same, byte-identical
```

Total probe grid: 3 ranks × 2 dtypes × 2 orders × 3 axis choices ×
4 functions (`cumsum`/`cumprod`/`nancumsum`/`nancumprod`) plus the
`ndarray.cumsum`/`ndarray.cumprod` method forms, plus a second pass adding
`axis=-1` explicitly, strided views (`[::2]`), and transpose views (`.T`)
across ranks 2–4, plus a third pass injecting real NaNs into ~30% of
elements for `nancumsum`/`nancumprod` across ranks 2–4, C/F order, axis
0/last/-1, and view slices — **≈600+ individual comparisons, 0 mismatches**
(shape, dtype-cast-equivalent values, all matched numpy exactly). `ma.cumsum`
/`ma.cumprod` were checked separately (rank 2–3, random boolean masks,
axis 0/last/-1) comparing both `.data` and `.mask` against real
`numpy.ma` — 0 mismatches.

### Findings table

| Item | Mechanism | Verdict | Repro |
|---|---|---|---|
| `cumsum` | measured | **Correct on N-D** (not a false declaration) | probe grid above, 0 mismatches |
| `cumprod` | measured | **Correct on N-D** | probe grid above, 0 mismatches |
| `nancumsum` | measured | **Correct on N-D**, incl. real NaNs | probe grid above, 0 mismatches |
| `nancumprod` | measured | **Correct on N-D**, incl. real NaNs | probe grid above, 0 mismatches |
| `ndarray.cumsum` | measured | **Correct on N-D** | probe grid above, 0 mismatches |
| `ndarray.cumprod` | measured | **Correct on N-D** | probe grid above, 0 mismatches |
| `ma.cumsum` | measured | **Correct on N-D** (data + mask) | probe grid above, 0 mismatches |
| `ma.cumprod` | measured | **Correct on N-D** (data + mask) | probe grid above, 0 mismatches |
| `logaddexp`, `logaddexp2` | static | **Not a false declaration** — plain elementwise ufunc call, no accumulate/reduceat dependency; scope disclaimed like every other plain ufunc | grep: no `.accumulate`/`.reduceat` reference in either declaration's own comment beyond the shared boilerplate scope note |
| (no ledger key exists for `<ufunc>.accumulate` / `<ufunc>.reduceat`) | static | **N/A — not a trackable/declared item** | `tools/numpy_surface.json`'s `exploded` has no `ufunc` class; grep of all ten `_state/*.py` files found zero keys of that shape |

**Zero confirmed false declarations.** This is a negative result, reported
as a success per the task's own framing, not padded into a finding.

## Not checked

- `ionp-ion/` and other workspace crates crates for any independent
  reference to `accumulate_math_binary`/`reduceat_math_binary`/
  `accumulate_axis` — not grepped, assumed-out-of-scope by directory
  ownership, not verified.
- `out=` parameter interaction for `cumsum`/`cumprod`/`nancumsum`/
  `nancumprod`/`ndarray.cumsum`/`ndarray.cumprod` under N-D + non-contiguous
  `out=` arrays specifically — I did not construct a mismatched-strides
  `out=` target in this audit; the ledger's own comments (`ndarray.py`,
  `toplevel.py`) claim this was covered by the differential corpus
  separately, but I did not re-verify it myself here.
- `axis=None` (flatten) behaviour on non-contiguous/view input for the
  cumsum family — I tested explicit integer axes only (0, last, -1);
  `axis=None` flattening of a strided view is a known historically-tricky
  case elsewhere in this codebase (see `order='K'` / multi-operand notes
  cited in the task prompt) and I did not specifically re-target it here.
- float32/float16 dtypes for the cumsum family — I tested int64/float64
  only, per the task's stated minimum; float16's known separate
  accumulation-precision divergence (documented elsewhere in
  `ionp/_state/ndarray.py` and `toplevel.py` as a disclosed, tolerated
  difference) was not re-probed.
- Whether any of the 669 declared items *other than* the ten grepped for
  accumulate/reduceat/cumsum/cumprod/logaddexp/nancumsum/nancumprod
  keywords has an underlying implementation that calls into
  `accumulate_math_binary`/`reduceat_math_binary`/`accumulate_binary`/
  `reduceat_binary` under a name that wouldn't obviously grep-match (e.g.
  a helper composed from `ufunc.accumulate` internally under a dtype-cast
  or fold path). The Rust-side call-graph grep (all call sites of those
  four function names across `ionp-core/src` and `ionp-py/src`) makes this
  unlikely — I found exactly the `Ufunc.accumulate`/`Ufunc.reduceat` method
  bodies in `lib.rs` and nothing else — but I did not trace every one of
  the 669 declared items' implementations individually to rule out an
  indirect path.
- The uncommitted `ionp-core/src/ufunc.rs` diff's ultimate fate (committed,
  reverted, or amended further) — read once, confirmed unrelated to this
  audit's question, not re-checked after.

## Summary

- **669** declared-`exact` items examined at the ledger level (full
  enumeration); **10** individually probed as direct suspects
  (`cumsum`, `cumprod`, `nancumsum`, `nancumprod`, `ndarray.cumsum`,
  `ndarray.cumprod`, `ma.cumsum`, `ma.cumprod`, `logaddexp`, `logaddexp2`).
- **0 confirmed false declarations.** The N-D `.accumulate`/`.reduceat` gap
  is real (re-confirmed: `axis=` is accepted-and-silently-ignored in both
  `Ufunc.accumulate`/`Ufunc.reduceat`, `lib.rs`), but it lives entirely
  behind a ufunc-method surface (`<ufunc>.accumulate(...)`,
  `<ufunc>.reduceat(...)`) that has **no corresponding ledger key at all** —
  it cannot be a false declaration because nothing declares it. The eight
  cumsum/cumprod-family items that share vocabulary with the gap use a
  provably separate, axis-aware Rust function (`accumulate_axis`) and
  measured correct across ~600+ N-D/view/dtype/NaN comparisons.
- **Highest-value thing I could not determine:** whether `out=` on
  `cumsum`/`cumprod`/`nancumsum`/`nancumprod` (or their `ndarray.*` method
  forms) stays correct when `out=` is itself a non-contiguous or
  differently-strided N-D array — I did not construct that specific case,
  and it's exactly the shape of interaction (two independent
  layout-sensitive code paths composed together) that has produced silent
  bugs elsewhere in this project before.

---

## Independent verification by Monday, 2026-08-06

The three load-bearing structural claims were re-measured rather than accepted.

**1. Call sites of the broken functions — CONFIRMED, exactly four, all in one place:**

```
ionp-py/src/lib.rs:8152  AnyBinaryOp::Plain(op) => accumulate_binary(op, &a)
ionp-py/src/lib.rs:8154  AnyBinaryOp::Math(op)  => accumulate_math_binary(op, &a)
ionp-py/src/lib.rs:8215  AnyBinaryOp::Plain(op) => reduceat_binary(op, &a, &idx)
ionp-py/src/lib.rs:8217  AnyBinaryOp::Math(op)  => reduceat_math_binary(op, &a, &idx)
```

Nothing else in `ionp-py/src/` or `ionp-core/src/` reaches them.

**2. The cumsum family routes elsewhere — CONFIRMED.** `cumsum`, `cumprod`,
`nancumsum`, `nancumprod` all go through `do_accumulate_axis` →
`ionp_core::ufunc::accumulate_axis` (`reductions.rs:1314/1326/1362/1374`), and
`ndarray.cumsum`/`cumprod` through `ndarray_attrs.rs:1685`. That is a
genuinely separate, axis-aware kernel. The broken 1-D path is not involved.

**3. `ufunc` in the manifest — the agent's check was INCOMPLETE, and I closed
the gap.** It verified no `ufunc` class is *exploded*, which is true (the 18
exploded classes are listed in `tools/numpy_surface.json` and `ufunc` is not
among them). But `ufunc` **is** present in `names` as a single flat opaque
item. A flat declaration would have covered the whole class including the two
broken methods, so "not exploded" does not by itself close the question.

Measured directly against `ionp.__ion_state__` (669 leaves): **there is no
`ufunc` key at any depth.** It is absent, not declared. The conclusion holds —
but it holds for a reason the audit did not establish.

## Verdict: gate CLEARED

**0 false declarations.** The `.accumulate`/`.reduceat` N-D defect is real and
still open, but it is unreachable from every currently-declared item.

## The forward-looking consequence

`ufunc` is a flat opaque item in the denominator. **Declaring it exact later
requires fixing the N-D defect first** — a flat declaration claims the entire
method surface, `.accumulate` and `.reduceat` included. This audit clears the
present. It does not clear that future declaration, and whoever attempts it
must treat `ACCUMULATE-REDUCEAT-ND-GAP-2026-08-05.md` as a hard prerequisite.

The same trap applies to every other flat class item in `names`: a flat
declaration is a claim about an entire unexploded surface. This audit examined
`ufunc` only.

## What this verification did NOT cover

- The `out=`-with-non-contiguous-N-D question the agent flagged as its top
  unresolved item. Still untested. It is the composed-two-layout-paths shape
  that has produced real bugs here before.
- Whether the other flat class items in `names` conceal the same structure.
  Not examined — only `ufunc` was.
- I re-ran the agent's static and manifest checks, **not** its ~600 behavioural
  comparisons. Those remain single-sourced.
