# ionp — work breakdown to zero `absent`

Derived from `tools/numpy_surface.json` (numpy **2.5.1**, **1,180 items**, canonical venv).
Not hand-counted: regenerate with `tools/snapshot_surface.py` and these numbers move with it.

## The denominator, by block

| block | items | notes |
|---|---:|---|
| top-level `np.*` | 477 | 106 ufunc, 283 func, 70 class, 18 const |
| `ma` | 225 | masked arrays — largest single module |
| `ndarray` methods/attrs + dunders | 164 | 70 + 94 |
| `char` + `strings` | 111 | 28 of them ufuncs |
| `random` | 63 | 9 classes (Generator, BitGenerators) + 54 funcs |
| `testing` | 50 | assertion helpers — pure logic, no arithmetic |
| `linalg` | 33 | **where Ion's structural win lands** |
| `fft` | 19 | **Ion's other landing zone** |
| `emath` 9, `rec` 9, `polynomial` 8, `ctypeslib` 6, `lib` 6 | 38 | long tail |

**134 ufuncs total** (106 top-level + 17 `strings` + 11 `char`). These are *one engine*, not 134
implementations — loop generation over the dtype matrix, in Rust. That is the single highest
leverage block on the board: ~11% of the surface from one correctly-built mechanism.

## Sequencing rationale

The blocks are not equal-difficulty per item, and ordering by count is wrong. Ordering by
*dependency*:

1. **`ndarray` core (164)** — everything else returns one. Nothing can be credited until dtypes,
   strides, broadcasting and the operator protocol are right. **In progress; currently the only
   block with any passing items.**
2. **ufunc engine (134)** — depends on 1. Largest leverage-per-unit-work on the board.
3. **`linalg` (33) + `fft` (19)** — depends on 1+2. Small item count, but this is where Ion's
   441× actually lands, so it carries the performance thesis rather than the coverage number.
4. **`testing` (50)** — depends on 1. Pure comparison logic, no arithmetic, no Ion content.
   Cheap, mechanical, and unblocks *our own* test ergonomics.
5. **`char`/`strings` (111)** — depends on 1+2 (28 are ufuncs). No spectral structure; all `exact`.
6. **`random` (63)** — self-contained. Bit-exact reproduction of numpy's stream is the hard part,
   not speed: PCG64/Philox/SFC64 must match seed-for-seed or the differential tests are unpassable.
   **Flagged as the block most likely to force honest `KNOWN-DIFFERENCES` entries.**
7. **`ma` (225)** — depends on nearly everything. Largest block, deliberately last. Mostly dispatch
   and mask bookkeeping over core ops; the arithmetic delegates down to Rust, so the Python-skin
   rule holds.
8. **long tail (38)** — `emath`, `rec`, `polynomial`, `ctypeslib`, `lib`.

## Honest risk register

- **`random` bit-exactness** may be unachievable without reimplementing numpy's exact generator
  internals. If so it gets a `KNOWN-DIFFERENCES` entry, not a redefined bar.
- **`ma` at 225 items** is 19% of the surface in the single least-glamorous corner. It is where a
  "100%" claim most plausibly quietly becomes 81%.
- **`ctypeslib`/`lib`** expose implementation-coupled surface that may have no meaningful
  standalone semantics. Candidates for an explicit, argued exclusion list — which must be
  *argued in writing and subtracted from the denominator visibly*, never silently dropped.
- No block is credited by writing it here. The ledger credits declared + resolvable + tested.
