# The 93 failing declared items — root-cause clusters

Classified 2026-08-02. Cluster survey by a read-only agent (message-text +
grep); **every claim below marked MEASURED was re-measured by me directly**
against binary mtime 1785740090 (23:54:50), stable across the run.

## The structural finding: two disjoint populations

The four root causes tracked all day (F-order output layout, empty-array stride
convention, free-function OWNDATA/.base metadata, Phase-2 view write-through)
**do not intersect the 93 failing items at all.** Those causes live in
`diff`/`reshape`/`alias-view` groups, which fold to *undeclared* items.

- The 4 known causes = **absent** surface area (items we have not claimed).
- The 93 = **declared-but-wrong** surface area (items we claimed and got wrong).

These are different queues with different urgency. A declared-but-wrong item is
a lie in the ledger; an absent item is only unwritten. Work the 93 first.

## Cluster A — scalar demotion MISSING (~84 items). RETURN TYPE axis.

Full reduction / contraction returning 0-d gives an ionp 0-d `ndarray` where
numpy gives a genuine numpy scalar. Same defect class as the ufunc fix in
`358142c`; the shared helper `numpy_scalar_from_0d` (ionp-py/src/lib.rs:4504)
already exists and is proven at 4+ call sites.

MEASURED BY ME, 8/8 mismatching:

| call | numpy | ionp |
|---|---|---|
| `linalg.det` | `float64` | `ndarray` |
| `linalg.matrix_rank` | `int64` | `ndarray` |
| `linalg.cond` | `float64` | `ndarray` |
| `linalg.norm` | `float64` | `ndarray` |
| `emath.sqrt/log/power` on 0-d | `float64` | `ndarray` |

`emath.rs` has **zero** references to the helper — the whole emath 0-d path is
unwired. Five linalg fns (det 874, slogdet 910, matrix_rank 2919, cond 3110,
lstsq 3255) likewise.

## Cluster A' — WRONG CONTAINER TYPE, not scalar demotion. NEW; 1+ items.

**`linalg.slogdet`: numpy returns `SlogdetResult` (a named tuple); ionp returns
a plain `tuple`.** MEASURED BY ME.

This was folded into Cluster A by the survey because the failure *message* says
"return-type mismatch". It is a different defect on a different object: the
container, not the elements. Wiring `numpy_scalar_from_0d` into slogdet will
NOT fix it.

**Method lesson (#60):** a classifier that reads failure MESSAGES groups defects
by how they are DESCRIBED, not by what they ARE. Two unrelated defects that
phrase alike land in one bucket. Cousin of #57 — an instrument that inspects a
representation cannot distinguish objects the representation flattens together.
Audit implication: check every other multi-return in linalg for the same thing
(`eig`, `eigh`, `qr`, `svd`, `lstsq` all return numpy named tuples).

## Cluster B — scalar demotion EXCESS (6 items). RETURN TYPE axis, INVERSE of A.

`char.equal, char.not_equal, char.less, char.greater, char.less_equal,
char.greater_equal`.

MEASURED BY ME, 6/6: `np.char.equal("a","b")` -> `ndarray shape=()`;
`ionp.char.equal("a","b")` -> Python `bool`.

**COLLISION WARNING, load-bearing.** Clusters A and B pull in OPPOSITE
directions on the same `ndim()==0` predicate. A blanket "0-d result becomes a
scalar" rule fixes 84 and breaks 6. numpy's `char.*` comparisons always return
an array. Any Cluster-A fix MUST carve char.* out explicitly, and must be
regression-checked against these 6 before landing.

## Clusters C-F — the long tail. ERROR MESSAGE / ERROR TYPE axes.

- **C (message wording):** `copy`, `ndarray.flatten`, `searchsorted`, `zeros` —
  correct exception CLASS, generic message (`"'X' object is not an instance of
  'str'"`) vs numpy's parameter-specific (`"order must be str, not X"`).
  Sharedness INFERRED from identical phrasing, NOT grep-confirmed.
- **D (missing validation):** `add` silently accepts invalid `casting=`/`order=`
  values numpy rejects.
- **E (excess validation):** `ndarray.flatten`, `searchsorted`, `zeros` reject
  plain `bytes` where numpy accepts.
- **F (wrong exception class):** `zeros` invalid dtype-string — numpy
  `TypeError`, ionp `ValueError`.

## Order by items-fixed-per-unit-work

1. **Cluster A** (~84) — helper exists, mechanical wiring. Must carve out char.*.
2. **Cluster B** (6) — do WITH A, not after; A's fix is what threatens it.
3. **Cluster A'** — audit all linalg multi-returns for named-tuple types.
4. **C** (4) — grep for the shared validator first; sharedness unverified.
5. **D/E/F** (1-3 each) — one-off kwarg validation. Lowest yield.

## Explicitly NOT measured

- Where the ufunc `.reduce()` full-reduction wrapping lives (`ionp-core/src/ufunc.rs`, unopened).
- Whether `ndarray.max/min/argmax/argmin/all/any/clip` method forms share code with the top-level functions or need independent wiring.
- Why `linalg::matrix_norm` still fails despite two confirmed helper call sites — a third uncovered branch is INFERRED, not located.
- Rust source confirmation for clusters B, C, D, E, F — message-text only.
- (RESOLVED: named-tuple defect now MEASURED in full -- see cluster A' section. 7 items affected, 5 correctly plain.)
- Whether `unique_inverse`/`unique_values` and any non-linalg named-tuple returns beyond unique_counts/unique_all are affected.

## Cluster A' — MEASURED IN FULL: 7 items, wrong CONTAINER type

Re-measured by me directly (binary 1785740090, stable). numpy returns a
**named tuple**; ionp returns a plain `tuple` with no `_fields`:

| item | numpy container | ionp |
|---|---|---|
| `linalg.slogdet` | `SlogdetResult` | `tuple` |
| `linalg.eig` | `EigResult` | `tuple` |
| `linalg.eigh` | `EighResult` | `tuple` |
| `linalg.qr` | `QRResult` | `tuple` |
| `linalg.svd` | `SVDResult` | `tuple` |
| `unique_counts` | `UniqueCountsResult` | `tuple` |
| `unique_all` | `UniqueAllResult` | `tuple` |

**The boundary matters more than the list.** These numpy calls return a PLAIN
tuple and are already CORRECT in ionp — do not "fix" them:

`linalg.lstsq`, `modf`, `divmod`, `frexp`, `nonzero`.

So the rule is NOT "linalg multi-returns are named" and NOT "multi-output
returns are named". `lstsq` is linalg, multi-output, and plain. The survey
listed `lstsq` under cluster A; on this axis it is correct. A blanket rule
over-fires on five call sites. **numpy is the per-function arbiter; there is no
derivable pattern.**

Values and element types are NOT implicated — this is purely the container.
Attribute access (`.eigenvalues`, `.S`, `.counts`) is what breaks, and no
value-comparing test can see it, which is why it survived this long.
