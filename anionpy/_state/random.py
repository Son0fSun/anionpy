"""Coverage declarations for the `numpy.random` block.

2026-08-07 CORRECTION (Monday, this session): the docstring this file used
to carry claimed `tools/numpy_surface.json` only has WHOLE-CLASS
granularity for `random.Generator` (no `random.Generator.integers`
sub-entry), and declared RANDOM_STATE empty on that basis. **That premise
is false on the current manifest and always was checkable in one line:**

    >>> import json; d = json.load(open("tools/numpy_surface.json"))
    >>> "random.Generator" in d["exploded"]
    True

`random.Generator` (and `random.RandomState`) ARE in `surface["exploded"]`,
exactly like `ndarray`/`ma.MaskedArray`/`dtype`/`finfo`/`iinfo` --
`tools/coverage.py`'s `all_items()` expands each into one ledger key per
public/dunder method (`random.Generator.integers`, `random.Generator.beta`,
...), the same mechanism `ndarray.reshape` already uses. A second session
had already acted correctly on this (see `tests/differential/
exploded_class_cases.py`'s `random.Generator.<method>` block, dated before
this one) and built 24 working per-method `ItemSpec`s; this file simply
never caught up to declare against them. The lesson is exactly the task
brief's own warning: a recorded reason in this repo is not evidence about
the current binary (or, here, the current manifest) -- it must be
re-checked, not re-cited. Kept below as a still-true record of what
`SeedSequence`/`PCG64`/`PCG64DXSM` genuinely lack (those three are NOT in
`exploded` -- `random.SeedSequence` etc. resolve as flat, whole-class
`names` entries -- so the "class missing most of its surface stays
undeclared" reasoning DOES still correctly apply to them).

`anionpy.random` implements a bit-exact-verified subset of 5 of the 63
`random.*` top-level surface items (`tools/numpy_surface.json`):
`SeedSequence`, `PCG64`, `PCG64DXSM`, `Generator`, `default_rng`. See
`ionp-core/src/random/{mod,seed_sequence,pcg64,bounded,distributions,
discrete}.rs` and `ionp-py/src/random.rs` for the implementation, and
`tests/differential/random_cases.py` +
`tests/differential/exploded_class_cases.py` for the differential cases.

2026-08-13 UPDATE (this session): 12 more `Generator.<method>` items wired,
tested (Rust-core unit tests with numpy-2.5.1-captured reference vectors,
THEN a Python-layer differential-suite pass on a freshly rebuilt `.so`),
and declared below -- see the DECLARED list and the dated block in
RANDOM_STATE itself for the exact evidence run. `Generator` now implements
36 of its ~30+ public methods (some numpy_surface entries double-count
aliases), missing only `.choice()`, `.permutation()`, `.shuffle()`,
`.permuted()`, `.dirichlet()`, `.multinomial()`, `.multivariate_normal()`,
`.multivariate_hypergeometric()`, and `.bit_generator` -- the bare
`random.Generator` class item stays undeclared below since those are
still missing.

WHAT REMAINS UNDECLARED, and why (whole-class, non-exploded items --
`Generator` itself is exploded, see DECLARED below, but these three are
not):
  - `PCG64`/`PCG64DXSM` implement the constructor and `.random_raw()`
    only. Missing: `.state` (get/set property -- numpy's documented
    save/restore mechanism), `.advance()`, `.jumped()`.
  - `SeedSequence` implements the constructor and `.generate_state()`
    only. Missing: `.entropy`/`.spawn_key`/`.pool_size`/`.pool`/`.state`
    attributes, `.spawn()`.
  - `random.Generator` (the BARE, unqualified class item, as opposed to
    its exploded `random.Generator.<method>` children below) stays
    undeclared: `Generator` implements 36 of its ~30+ public methods (see
    DECLARED below) but is still missing `.choice()`, `.permutation()`,
    `.shuffle()`, `.permuted()`, `.dirichlet()`, `.multinomial()`,
    `.multivariate_normal()`, `.multivariate_hypergeometric()`, and the
    `.bit_generator` attribute -- declaring the bare class item would
    claim this whole surface, which is false.
  - `random.RandomState` (whole class, not exploded... it IS in
    `exploded`, but anionpy implements 0 of its ~45 methods --
    `hasattr(anionpy.random, "RandomState")` is `False`) stays absent,
    correctly.

DECLARED (2026-08-07 session). 24 items, all EXPLODED
`random.Generator.<method>` children plus `random.default_rng` (a
top-level `names` func entry, un-exploded, matching numpy's own
single-optional-arg signature exactly):

  `random.default_rng`,
  `random.Generator.random`, `.integers`, `.bytes`,
  `.standard_normal`, `.normal`, `.standard_exponential`, `.exponential`,
  `.standard_gamma`, `.gamma`, `.beta`, `.chisquare`, `.f`, `.uniform`,
  `.standard_cauchy`, `.standard_t`, `.pareto`, `.weibull`, `.power`,
  `.laplace`, `.gumbel`, `.logistic`, `.lognormal`, `.rayleigh`.

DECLARED (2026-08-13, this session). 12 more EXPLODED
`random.Generator.<method>` children, all newly implemented in
`ionp-core/src/random/{distributions,discrete}.rs` +
`ionp-py/src/random.rs`:

  `random.Generator.wald`, `.vonmises`, `.triangular`,
  `.noncentral_chisquare`, `.noncentral_f`, `.poisson`, `.binomial`,
  `.negative_binomial`, `.geometric`, `.zipf`, `.logseries`,
  `.hypergeometric`.

EVIDENCE for the 12-item 2026-08-13 batch, same two-layer bar as the
2026-08-07 batch below:

(1) Rust-core unit tests (`ionp-core/src/random/distributions.rs` and the
new `#[cfg(test)]` module in `discrete.rs`): each new distribution has a
`_seed42` test with a numpy-2.5.1-captured reference vector as a doc
comment, hardcoded `assert_eq!`. Two FMA-contraction bugs were found and
fixed via this layer BEFORE any Python-layer testing was attempted:
`noncentral_chisquare`'s `Chi2 + n * n` and `vonmises`'s `1 + s * Z` /
`Y * (2 - Y) - V` are single C expressions that get FMA-contracted by
numpy's compiled build; Rust needs `.mul_add()` in the same spots to match
bit-for-bit (same class of bug the 2026-08-07 session found in
`uniform`/`normal`/`laplace`). All `random::` unit tests (46 total) pass
after the fix.

(2) Python-layer differential suite (`tests/differential/random_cases.py`
`generator_cases()`, this session's ~58-case addition covering happy-path
cases at multiple seeds/regimes crossing internal algorithm branch
boundaries -- e.g. `binomial`'s inversion vs BTPE branch, `geometric`'s
search vs inversion branch, `vonmises`'s tiny/normal/huge-kappa Taylor
branches -- plus CONS_*-transcribed error-path cases with exact numpy
message text, `size=0` cases, and a >2**63 seed case): a fresh
`flock /tmp/ionp-build.lock ./.venv/bin/maturin develop --release`
rebuild (needed because a concurrent agent's commit had transiently
broken/staled the installed `.so` mid-session -- `ImportError: cannot
import name 'dot'`, resolved by the rebuild, unrelated to this file)
followed by `tests/differential/run.py` on the current tree gave
`[PASS] random.Generator (113/113 cases) -- ok` with all 24 pre-existing
+ 12 new per-method buckets showing `[PASS]` and zero `[FAIL]` lines
anywhere matching random/Generator/poisson/binomial/negative_binomial/
geometric/zipf/logseries/hypergeometric/wald/vonmises/triangular/
noncentral in a fresh `FAILURE-BASELINE.txt` diff (35 total current fail
names, all in `array`/`polynomial`/`hermite`/`ufunc`/other agents'
domains, none touching `random`).

EVIDENCE, in two layers:

(1) Pre-existing corpus (`random_cases.py`'s `generator_cases()` +
`default_rng_cases()`, and `exploded_class_cases.py`'s
`_GENERATOR_METHOD_ARGS`/`integers`/`bytes` blocks): every case bit-exact
(dtype+shape+value/byte identity, `atol=rtol=0.0`, no tolerance
mechanism), including error paths matched by exception TYPE and exact
MESSAGE TEXT. `tests/differential/run.py` confirms all 24 items `pass`
with `mechanism: none` (bit-exact, not ULP/epsilon-tolerant) in the
current run (`/tmp/rRND1.json`).

(2) OUT-OF-CORPUS verification (this session, NOT cases already in the
corpus above -- a standalone probe script, not committed, run from
`/private/tmp` against the installed package): swept
  - 6 seeds: 1, 12345, 999999937, 2**63-1, 2**63, 2**63+12345,
    2**64-1, 2**64-7, and a >2**64 (~10**21) seed (SeedSequence's
    arbitrary-precision entropy pool accepts it),
  - sizes: `None` (bare scalar), `0`, `1`, `5`, `(2,3)`, `(0,4)`,
    `(3,0,2)`, `(2,2)`,
  - every explicit `integers()` dtype (`int8`/`16`/`32`/`64`,
    `uint8`/`16`/`32`/`64`, `bool`) crossed with 8 `(low, high)` pairs
    including reversed (`low>high`) and out-of-dtype-range pairs,
  - `endpoint=True`/`False`,
  - `random()`'s `dtype=float32`/`float64` at scalar AND array size,
  - negns-size error path (`size=-3`) on 6 representative methods,
  against real numpy 2.5.1, bit-comparing via
  `np.ascontiguousarray(x).view(uint64/uint32)` (never plain
  `np.asarray(x).view(...)`, which raises on non-contiguous output and
  would misscore a clean function as broken -- the exact bug this
  project's brief warns about). Total: ~3000 out-of-corpus cases, 0
  divergences AFTER two fixes made mid-session (see below) -- before the
  fixes, the same probe found 2 real, reproducible divergences:

  BUG 1 (dtype-fidelity, scalar `integers()`): `Generator.integers(...,
  size=None, dtype=<non-default, non-bool>)` returned a bare Python `int`
  instead of a width-typed numpy scalar (`numpy.int16`/`numpy.uint32`/
  ...). Only the DEFAULT dtype (`int64`) coincidentally survived
  `np.asarray(x).dtype` comparison, since a bare Python int's `asarray`
  dtype is always `(u)int64` -- every EXPLICIT narrower dtype at
  `size=None` diverged (confirmed: `type(np.random.default_rng(5)
  .integers(0, 100, dtype=np.int8))` is `numpy.int8`; anionpy's was
  `int`). `bool` was and remains correct as a bare Python `bool` --
  that's numpy's OWN actual behavior for `dtype=bool`, confirmed
  directly, not a second bug. FIXED in `ionp-py/src/random.rs`'s
  `int_scalar_to_py`: routes non-bool dtypes through the same
  `numpy_scalar_from_0d` mechanism `lib.rs` already uses elsewhere in the
  crate for exact-type-matching scalar returns (builds a real 1-element
  0-d `NdArray` of the target dtype, mints a real numpy scalar
  constructor call when numpy is importable -- this is a container-type
  fix, not a numpy-computed-VALUE dependency; the drawn VALUE is
  unchanged).

  BUG 2 (`size=0` bounds-validation): `Generator.integers(low, high,
  size=0, dtype=...)` raised on out-of-range or `low>=high` bounds where
  real numpy returns an empty array UNCONDITIONALLY, skipping validation
  whenever the resolved element count is 0 -- confirmed:
  `np.random.default_rng(1).integers(500, 300, size=0, dtype=np.int8)` ->
  `array([], dtype=int8)`, no error, despite `low>high` AND both being out
  of `int8`'s range. numpy's own `_bounded_integers.pyx` fill functions
  early-return on `cnt==0` before ever touching the range/bounds
  computation. FIXED in `ionp-py/src/random.rs`'s `integers()`: resolves
  `size=` to a concrete shape BEFORE running `check_integer_bounds`, and
  skips validation whenever the resolved count is 0. `size=None` (the
  scalar path) always draws exactly one value and is therefore never
  exempt -- unaffected.

  Both bugs are now represented in `random_cases.py`'s corpus (search for
  "Declaration-pass additions (2026-08-07 session)"), confirmed RED before
  the fix and GREEN after, so a future revert of either fix is caught
  there, not only by this out-of-corpus report.

SCOPE BOUNDARY, explicit and loud-failing, kept undeclared-inside (not a
silent gap): none of the 24 items above support numpy's documented
array-like/broadcast parameter forms (`integers(low=[0,0], high=10)`,
`normal(loc=np.array([1.0, 2.0]))`, etc.) -- only scalar Python-float/int
parameters plus a `size=` output shape are wired up. This mirrors
`linalg.svd`'s already-accepted precedent (`anionpy/_state/linalg.py`,
2026-08-06: declared for its non-batched `ndim==2` path, batched
explicitly scoped out and left alone) of declaring a covered call-form
subset while explicitly scoping out a documented-but-unsupported one --
the load-bearing difference from the `percentile`/`ndarray.real` WITHDRAWN
precedents (`anionpy/_state/toplevel.py`, `anionpy/_state/ndarray.py`) is
that those were divergences found INSIDE the supposedly-covered path
(silently wrong dtype/aliasing on the ordinary call), not an explicitly
scoped-out parameter TYPE that fails loudly. Confirmed loud-failure, by
hand, on representative methods: `integers(low=[0,0], high=[10,20])` ->
`TypeError: 'list' object cannot be interpreted as an integer`;
`normal(loc=np.array([1.0,2.0]), scale=1.0, size=2)` -> `TypeError: only
0-dimensional arrays can be converted to Python scalars`;
`gamma(shape=[1.0,2.0], scale=1.0)` -> `TypeError: must be real number,
not list`; `random(size=3, out=np.zeros(3))` -> `TypeError:
Generator.random() got an unexpected keyword argument 'out'` (same for
`integers(..., out=...)`) -- never a silent wrong answer, always a clean
Python `TypeError` before any draw happens. `default_rng`'s own scope
boundary is the same shape: numpy's documented `seed=` also accepts
array-like-of-ints, `SeedSequence`, `BitGenerator`, and `Generator`
instances; anionpy accepts `None`/int only, and anything else raises
`TypeError` cleanly (confirmed: `anionpy.random.default_rng([1,2,3])` ->
`TypeError: SeedSequence expects int or sequence of ints for entropy not
[1, 2, 3]` -- ironically a MORE informative message than the silent
success on numpy's side, but the point here is only that it fails loudly,
not that it matches numpy's error).

CRITICAL, empirically-verified correction baked into the implementation
(not a declaration-time claim, see `ionp-core/src/random/bounded.rs`'s and
`ionp-py/src/random.rs`'s own module docs for the full derivation):
`np.random.default_rng(seed)` in the installed numpy 2.5.1 instantiates
plain `PCG64` (XSL-RR), NOT `PCG64DXSM` -- confirmed directly via
`type(np.random.default_rng(42).bit_generator)`.

OUT OF SCOPE FOR THIS PASS (2026-08-13), not attempted, not silently
assumed working: `Generator.dirichlet`, `.multinomial`,
`.multivariate_normal`, `.multivariate_hypergeometric` (array-valued
alpha/pvals parameter plumbing, not yet wired), `Generator.choice`/
`.permutation`/`.shuffle`/`.permuted` (flagged high-risk for subtle
bit-stream consumption-order bugs -- must match numpy's exact draw order,
not just produce a statistically-uniform permutation), `MT19937`/
`Philox`/`SFC64` BitGenerators, and the entire legacy `RandomState`
surface (`random.seed`/`random.rand`/`random.randn`/`random.randint`/etc,
and the `RandomState` class itself, ~45 methods) -- next items in the
task's specified order, none started yet.
"""

RANDOM_STATE: dict[str, str] = {
    "random.default_rng": "exact",
    "random.Generator.random": "exact",
    "random.Generator.integers": "exact",
    "random.Generator.bytes": "exact",
    "random.Generator.standard_normal": "exact",
    "random.Generator.normal": "exact",
    "random.Generator.standard_exponential": "exact",
    "random.Generator.exponential": "exact",
    "random.Generator.standard_gamma": "exact",
    "random.Generator.gamma": "exact",
    "random.Generator.beta": "exact",
    "random.Generator.chisquare": "exact",
    "random.Generator.f": "exact",
    "random.Generator.uniform": "exact",
    "random.Generator.standard_cauchy": "exact",
    "random.Generator.standard_t": "exact",
    "random.Generator.pareto": "exact",
    "random.Generator.weibull": "exact",
    "random.Generator.power": "exact",
    "random.Generator.laplace": "exact",
    "random.Generator.gumbel": "exact",
    "random.Generator.logistic": "exact",
    "random.Generator.lognormal": "exact",
    "random.Generator.rayleigh": "exact",
    # 2026-08-13 (this session): 12 more Generator methods wired in
    # ionp-py/src/random.rs, verified bit-exact through the Python layer
    # by tests/differential/run.py against a freshly rebuilt .so on a
    # then-current tree (random.Generator (113/113 cases) -- ok, zero
    # FAIL lines matching random/Generator/poisson/binomial/geometric/
    # zipf/logseries/hypergeometric/wald/vonmises/triangular/noncentral
    # in the fresh FAILURE-BASELINE diff). Includes CONS_* constraint
    # error-path cases (verbatim numpy message transcription) and two
    # FMA-contraction fixes (noncentral_chisquare, vonmises) found and
    # fixed this session -- see ionp-core/src/random/distributions.rs.
    "random.Generator.wald": "exact",
    "random.Generator.vonmises": "exact",
    "random.Generator.triangular": "exact",
    "random.Generator.noncentral_chisquare": "exact",
    "random.Generator.noncentral_f": "exact",
    "random.Generator.poisson": "exact",
    "random.Generator.binomial": "exact",
    "random.Generator.negative_binomial": "exact",
    "random.Generator.geometric": "exact",
    "random.Generator.zipf": "exact",
    "random.Generator.logseries": "exact",
    "random.Generator.hypergeometric": "exact",
}
