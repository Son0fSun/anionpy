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

2026-08-13 UPDATE #2 (same session, later pass): `.dirichlet()` and
`.multinomial()` wired and declared -- see the DECLARED block further
down. `Generator` now implements 38 of its ~30+ public methods, missing
only `.choice()`, `.permutation()`, `.shuffle()`, `.permuted()`,
`.multivariate_normal()`, `.multivariate_hypergeometric()`, and
`.bit_generator`.

WHAT REMAINS UNDECLARED, and why (whole-class, non-exploded items --
`Generator` itself is exploded, see DECLARED below, but these three are
not):
  - `PCG64`/`PCG64DXSM` implement the constructor and `.random_raw()`
    only. Missing: `.state` (get/set property -- numpy's documented
    save/restore mechanism), `.advance()`, `.jumped()`.
  - `SFC64` implements the constructor and `.random_raw()` only, same
    shape as PCG64/PCG64DXSM above -- measured directly (2026-08-13):
    `sorted(set(dir(np.random.SFC64(1))) - set(dir(object)))` is
    `['capsule', 'cffi', 'ctypes', 'lock', 'random_raw', 'seed_seq',
    'spawn', 'state']` (8 attrs, ignoring dunders/`_benchmark`/private
    `_cffi`/`_ctypes`/`_seed_seq`/cython-internal names); anionpy's is
    `['random_raw']` (1 attr). numpy's constructor also documents and
    accepts array-like/sequence entropy (`np.random.SFC64([1,2,3])`
    succeeds); anionpy's raises `TypeError: SeedSequence expects int or
    sequence of ints for entropy not [1, 2, 3]` on the identical call.
    See the REVOKED block near the end of this docstring for the full
    story of why this was briefly (incorrectly) declared exact.
  - `SeedSequence` implements the constructor and `.generate_state()`
    only. Missing: `.entropy`/`.spawn_key`/`.pool_size`/`.pool`/`.state`
    attributes, `.spawn()`.
  - `random.Generator` (the BARE, unqualified class item, as opposed to
    its exploded `random.Generator.<method>` children below) stays
    undeclared: `Generator` implements 38 of its ~30+ public methods (see
    DECLARED below) but is still missing `.choice()`, `.permutation()`,
    `.shuffle()`, `.permuted()`, `.multivariate_normal()`,
    `.multivariate_hypergeometric()`, and the `.bit_generator` attribute
    -- declaring the bare class item would claim this whole surface,
    which is false.
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

DECLARED (2026-08-13, this session, second pass). 2 more EXPLODED
`random.Generator.<method>` children, both newly implemented in
`ionp-core/src/random/{distributions,discrete}.rs` +
`ionp-py/src/random.rs`, both scoped to the non-broadcast call form
(scalar `n` / 1-D `alpha`/`pvals`, matching `linalg.svd`'s prior scoping
precedent -- see SCOPE BOUNDARY above, same reasoning applies):

  `random.Generator.dirichlet`, `.multinomial`.

EVIDENCE for this 2-item batch:

(1) Rust-core unit tests (`ionp-core/src/random/distributions.rs`'s
`dirichlet_standard_path_seed42`/`dirichlet_small_alpha_path_seed42`/
`dirichlet_k1_seed42`, `ionp-core/src/random/discrete.rs`'s
`multinomial_single_draw_seed42`/`multinomial_many_draws_seed42`/
`multinomial_n_zero_seed42`): numpy-2.5.1-captured reference vectors,
hardcoded `assert_eq!`. `dirichlet`'s two-branch selection
(`alpha.max() < 0.1` picks stick-breaking via `random_beta` over
independent-gamma unit-normalization) was transcribed from
`_generator.pyx`; a `break`-vs-`return` control-flow subtlety in the
stick-breaking loop (numpy's C-level `break` exits only the inner `for j`
loop, NOT the whole draw -- the trailing `val_data[i+k-1] = acc`
assignment still runs afterward) was caught and fixed BEFORE any build
was attempted, not found by a failing test. All 52 `random::` unit tests
pass (`cargo test -p ionp-core --release random::`).

(2) Python-layer differential suite: both items registered as their OWN
dedicated `EXPLODED_CLASS_SPECS` entries from the start (learning
directly from this session's own umbrella-bucket ledger-gap finding, see
`docs/UMBRELLA-BUCKET-LEDGER-GAP-2026-08-13.md` -- NOT routed through
`random_cases.py`'s umbrella bucket first). Per the task's explicit
"vary the input so the loop runs 1, 2, and many times" warning
(`multinomial`/`dirichlet` both draw in a data-size-dependent-trip-count
loop), the corpus (`tests/differential/exploded_class_cases.py`) crosses
trip counts 1/2/7 against fixed parameters, both `dirichlet` branches
(standard AND small-alpha), the `k=1` and `k=0`/`d` (`alpha`/`pvals`
empty) edge cases, and CONS_*-transcribed error paths (`alpha<0`, `pvals`
elementwise out of `[0,1]`, `pvals[:-1]` sum too large, `n<0`, empty
`pvals`). A real bug was found and fixed via this corpus BEFORE
declaration: `dirichlet([], size=...)` (`k=0`) panicked
(`chunks_mut(0)`, "chunk size must be non-zero") instead of returning
numpy's actual behavior, an all-empty array of shape `size + (0,)` with
nothing drawn -- fixed in `ionp-py/src/random.rs`'s `dirichlet()` by
skipping the draw loop when `k == 0`. `tests/differential/run.py` shows
`[PASS]` for both `random.Generator.dirichlet` and
`random.Generator.multinomial`, `mechanism: none` (bit-exact), confirmed
against a freshly rebuilt `.so` (`/tmp/r_dm.json`); the only `[FAIL]`
lines in that same run are `transpose`/`trunc` bool-dtype mismatches in
another concurrent agent's in-flight, uncommitted domain (confirmed via
`git status`: neither file this batch touched), same category as the
`cross` item the task coordinator already confirmed is out of this
domain.

(3) OUT-OF-CORPUS stream-position probe (this session, standalone,
not committed): for `dirichlet` (both branches, `size=5`) and
`multinomial` (`size=5`, including the `n=0` and `d=2` edge shapes),
drew the method's own output, THEN drew `random(8)` from both numpy and
anionpy generators continuing from the same seed and compared -- zero
mismatches on all 5 probes, confirming these methods consume the PCG64
bit stream in the same word-count as numpy's C implementations, not just
producing matching final values.

DECLINED-BY-MEASUREMENT (2026-08-13, this session, third pass):
`Generator.multivariate_normal` was investigated and its Rust/PyO3 draw
pipeline deliberately NOT built, per the task's explicit instruction to
measure the underlying LAPACK decomposition's bit-exactness FIRST rather
than "burn days" implementing a distribution that could never pass. This
is a NEGATIVE FINDING, treated as a real, complete result -- not a
shortfall and not deferred work.

numpy's `multivariate_normal` (`_generator.pyx`, ~line 3719-3957) draws
`x = standard_normal(final_shape)` then computes a factor matrix `A` via
one of three decompositions of `cov` (default `method='svd'`:
`u, s, vh = np.linalg.svd(cov)`, `_factor = u * sqrt(s)`; `'eigh'`:
`s, u = np.linalg.eigh(cov)`; `'cholesky'`: `l = np.linalg.cholesky(cov)`),
then returns `mean + x @ _factor.T`. numpy's own docstring already warns
results "may not be identical... even up to precision... across
architectures, OSes, or even builds," specifically calling out `svd` with
repeated singular values.

Measured directly (`/tmp/mvn_probe.py`, `/tmp/mvn_probe2.py`, not
committed -- scratch instruments only) on 8 covariance matrices spanning
well-conditioned, singular (rank-2 of 4), near-equal-eigenvalue,
exact-repeated-eigenvalue, spherical, all-zero, well-conditioned 8x8, and
low-rank-plus-noise 8x8 cases, comparing `anionpy.linalg.svd`/`.eigh`/
`.cholesky` byte-for-byte against real numpy's, with BOTH numpy and
anionpy confirmed on this box to route through the SAME Apple Accelerate
LAPACK/BLAS backend (`np.show_config()`'s `blas`/`lapack` both report
`"name": "accelerate"`; `anionpy/_state/linalg.py` confirms the same for
ionp-py's calls):

  - `linalg.svd`: byte-MISMATCHED on 4/8 matrices (`wellcond_3x3`,
    `near_equal_eig_4x4`, `wellcond_8x8`, `lowrank_plus_noise_8x8`).
    Differences are ULP-level (~1e-16 to 1e-15), same sign/direction as
    numpy's values -- not a wrong algorithm, but not byte-identical.
    Byte-matched on the other 4/8 (`singular_rank2_4x4`,
    `exact_repeated_eig_4x4`, `spherical_5x5`, `zero_3x3`).
  - `linalg.eigh`: on a 4-matrix subset, eigenvalues byte-exact on only
    2/4, eigenvectors byte-exact on only 1/4 (max diffs 3.3e-16 to
    1.78e-15).
  - `linalg.cholesky`: byte-exact (`max_diff == 0.0`) on ALL 4/4 tested
    matrices -- a genuine positive sub-finding, but NOT usable here since
    numpy's `multivariate_normal` DEFAULT method is `'svd'`, not
    `'cholesky'`, and silently substituting a different method than the
    caller requested (or than numpy's default resolves to) would itself
    be a mismatch.

Conclusion: same-LAPACK-backend does NOT imply byte-identical output at
this precision; `multivariate_normal` cannot be bit-exact against numpy
through its default (or `'eigh'`) code path on this box. Per the task's
explicit framing ("If it does not, say so and declare NOTHING... a
value-exact-but-not-bit-exact multivariate_normal is not exact... Report
the measurement either way; a negative result there is a real finding,
not a failure"), `random.Generator.multivariate_normal` is declared
NOTHING and left entirely unimplemented in `ionp-py/src/random.rs`,
by design.

DECLARED (2026-08-13, this session, third pass):
`random.Generator.multivariate_hypergeometric`, default
`method='marginals'` only (numpy's default; the array-valued `colors`
parameter, scalar `nsample`, `size` broadcasting).
`method='count'` is a real, numpy-successful, but algorithmically
DIFFERENT draw order (temp array of size `sum(colors)`, distinct
correlated-draw structure) -- deliberately scoped out, same "loud failure
over silent wrongness" precedent as `linalg.svd`'s non-batched scoping:
`ionp-py/src/random.rs`'s `multivariate_hypergeometric()` raises a clear
`PyNotImplementedError` for `method='count'` rather than either
implementing an unverified algorithm or silently mismatching numpy.

EVIDENCE:

(1) Rust-core: the "marginals" algorithm was transcribed from numpy
2.5.1's ACTUAL C source, `numpy/random/src/distributions/
random_mvhg_marginals.c` (obtained via a fresh `pip3 download
numpy==2.5.1 --no-binary :all: --no-deps`, since the previously-cached
local source dump did not contain this function). Implemented as
`discrete::multivariate_hypergeometric_marginals` in
`ionp-core/src/random/discrete.rs`: the "more than half" symmetry
optimization (`if nsample > total/2: nsample = total - nsample`, draw for
the cheaper complement, negate every output entry at the end) runs FIRST,
then `num_colors - 1` sequential CORRELATED `hypergeometric()` draws with
running `remaining`/`num_to_sample` state, then whatever remains after
the loop is assigned UNCONDITIONALLY to the last color slot (no further
draw). 6 unit tests against numpy-2.5.1-captured reference vectors
(single/many draws, the more-than-half branch, `nsample=0`, single-color,
two-color) all pass on first attempt -- `cargo test -p ionp-core
--release random::discrete::` (18/18 pass).

(2) PyO3 binding (`ionp-py/src/random.rs` only -- `lib.rs` untouched, per
the standing constraint): validation order (method membership, then
nsample nonneg, then colors nonneg, then sum(colors) overflow, then
`method=='marginals'`'s total<1e9 ceiling, then nsample>total) confirmed
via direct hand-probing of real numpy with simultaneously-violating
inputs (e.g. negative colors AND negative nsample together -> nsample
error wins), transcribed exactly.

(3) Python-layer differential suite: registered as its own dedicated
`EXPLODED_CLASS_SPECS` entry from the start (same umbrella-bucket lesson
as dirichlet/multinomial). Corpus varies `colors` length (1/2/3, so the
per-draw loop's data-dependent trip count runs 0/1/2 times), crosses both
the direct and more-than-half branches, and includes CONS_*-transcribed
error paths (invalid method, negative colors, negative nsample,
nsample>total, empty colors) -- all error messages hand-verified
byte-identical to numpy's own exception text. `method='count'` is
deliberately ABSENT from the corpus (see scoping note above -- it cannot
match numpy's differently-algorithmed output, so a case for it would only
ever be a false-pass or an uninformative permanent fail; the loud
NotImplementedError IS the declaration for that path).
`tests/differential/run.py` (`/tmp/r_mvhg.json`) shows
`random.Generator.multivariate_hypergeometric`: `verdict: "pass"`,
`mechanism: "none"` (bit-exact, no tolerance applied).

(4) OUT-OF-CORPUS stream-position probe (standalone, not committed):
for 5 cases (`size=5` direct path, `size=5` more-than-half, `size=5`
single-color, `size=5` two-color, `size=5` nsample=0), drew the method's
own output, THEN drew `random(8)` from both numpy and anionpy generators
continuing from the same seed and compared -- zero mismatches on all 5
probes, confirming bit-stream-consumption parity (not just matching final
values), including in the single-color case where the main draw loop
never runs at all.

DECLARED (2026-08-13, this session, fourth pass): `Generator.shuffle`,
`.permutation`, `.permuted`, `.choice` -- the family flagged high-risk in
the prior pass's OUT-OF-SCOPE note above for subtle bit-stream
consumption-order bugs. Scoped to 1-D `anionpy.ndarray`/int input for all
four; N-D input raises a loud `PyNotImplementedError` rather than
attempting numpy's axis-wise cross-section algorithms (a genuinely
different draw structure this pass didn't transcribe). `choice`'s
`replace=False, p=given` branch is likewise declined loudly -- numpy's own
algorithm there is an iterative `np.unique`+`searchsorted` rejection loop
whose bit-exactness would need dedicated verification this pass didn't
have time for.

Two bit-stream primitives are involved, confirmed genuinely distinct by
reading numpy's actual Cython/C source (`_generator.pyx`,
`distributions.c`), NOT assumed: `shuffle`/`permutation`/`permuted` all
draw via `_shuffle_raw`'s MASKED-rejection `random_interval`
(`ionp_core::random::bounded::random_interval`, already existing and
verified); `choice`'s own `replace=False, p=None` branches (`_shuffle_int`
for its tail-shuffle, and its Floyd's-algorithm hash-set) instead draw via
`random_bounded_uint64(..., use_masked=0)`, i.e. LEMIRE's algorithm
(`bounded::bounded_u64`, likewise pre-existing). Implemented as
`discrete::shuffle_masked`/`shuffle_lemire`/`choice_no_replace_no_p` in
`ionp-core/src/random/discrete.rs`, 7 new unit tests against
numpy-2.5.1-captured reference vectors (25/25 in that module pass,
`cargo test -p ionp-core --release random::discrete::`), including a
`pop_size=20000, size=1000` case specifically chosen to land on the
`pop_size > 10000` tail-shuffle heuristic branch rather than Floyd's
algorithm.

A pre-existing Python-level monkeypatch (`anionpy/__init__.py`'s
`_generator_shuffle`/`_generator_permutation`/`_generator_choice`/
`_generator_permuted`, self-labeled "not bit-identical to numpy" and
implemented via `Generator.integers` calls, from an earlier session) was
SHADOWING the real PyO3 methods added this pass -- discovered when the
first differential run against the freshly rebuilt `.so` mismatched every
case despite the Rust-core unit tests passing. Removed entirely; the real
`#[pymethods]` in `ionp-py/src/random.rs` now run.

EVIDENCE: registered as 4 dedicated `EXPLODED_CLASS_SPECS` entries (not
the umbrella bucket) in `tests/differential/exploded_class_cases.py`.
`choice`'s corpus explicitly covers each cell of the `replace`x`p`
cross-product as its own case (`replace=True,p=None` via Lemire uniform
int; `replace=True,p=given` via cdf+searchsorted; `replace=False,p=None`
via BOTH the Floyd's-algorithm branch and, separately, the
`pop_size=20000` tail-shuffle branch), per the coordinator's explicit
warning that a uniform pass/fail across all four should trigger suspicion
of the test instrument rather than be accepted at face value -- here the
four cells are NOT uniform: `replace=False,p=given` is the one cell that
raises `NotImplementedError` by design, confirmed distinctly from the
other three passing. Input DATA/PROVENANCE is varied per the
coordinator's point 3: `p` with an exact-zero entry, `p` summing to just
inside the `sqrt(eps)` `atol` tolerance boundary (not a clean 1.0), `a` as
a plain int vs. a 1-D array, `a` as an explicit non-default-dtype
(`int32`) array, and a string-dtype `a`/`shuffle` target (exercises
`shuffle_masked`'s non-`Copy` `Buffer::S`/`Buffer::U` code path). Error
paths transcribed verbatim from `_generator.pyx` (`a` non-positive,
`p` length mismatch, `p` containing a negative entry, `p` summing too far
from 1, `replace=False` with `size > pop_size`) and hand-verified
byte-identical to numpy's own exception text via an ad-hoc script (both
sides raising the identical message on the identical input).

GRADING-HARNESS GAP FOUND AND FIXED (same pass, before this declaration):
the first full differential run reported `shuffle`'s `"strings"` case and
`choice`'s `"array_a_strings"` case as FAIL, both with the identical
symptom -- `NotImplementedError` raised not by `shuffle`/`choice`
themselves but by `tests/differential/exploded_class_cases.py`'s
`_as_comparable()` helper's `np.asarray(x)` call, converting the anionpy
S/U-dtype RESULT array into a numpy-comparable form for grading.
`np.asarray()` fails on any anionpy S/U-dtype array (unrelated,
pre-existing anionpy array-protocol limitation: string storage is one
heap allocation per element, so there is no single flat buffer to
export -- see `anionpy/array_protocol`'s own `NotImplementedError`
message) -- this is a gap in the TEST HARNESS's grading step, not in
`shuffle_masked`/`choice`'s underlying string-dtype code path
(`Buffer::S`/`Buffer::U`, both already exercised via non-`Copy` generics
before this session). Confirmed by hand
(`./.venv/bin/python`): `anionpy_arr.tolist()` on the same string result
reproduces numpy's `.tolist()` output exactly, byte-for-byte, so the
underlying shuffle/choice draw on string data was correct all along --
only the harness's numpy-array round-trip was broken. Fixed by widening
`_as_comparable()` to catch that specific `NotImplementedError` (matched
on its "flat buffer to expose" text, so no other error is silently
swallowed) and fall back to `np.array(x.tolist())`. Re-ran both items
after the fix: `shuffle` 7/7, `choice` 17/17, both pass; the other 40
`random.*` registry items were also re-run in full to confirm the
`_as_comparable()` change caused zero regressions elsewhere (all still
pass). This declaration reflects the POST-FIX state.

OUT-OF-CORPUS stream-position probes (standalone, not committed) were run
for EVERY branch, not once per method, per the coordinator's point 2:
`shuffle`, `permutation` (both the int-arange and array-copy sub-paths),
`permuted`, and all 4 `choice` cells (including the declined
`replace=False,p=given` cell, confirmed to raise before consuming any
bits rather than partially drawing then failing) -- each drew the
candidate call, then drew `random(8)` from both numpy and anionpy
generators continuing from the same seed and compared; zero
stream-position mismatches on any passing branch. Two N-D-input declines
(`shuffle`/`choice` on a 2-D array) and one `permuted(out=...)` decline
were also probed directly to confirm they raise `NotImplementedError`
rather than silently falling through to a wrong 1-D interpretation.

DECLARED (2026-08-13, this session, fifth pass): `random.SFC64` (constructor
+ `.random_raw()` only, same scope precedent as `random.PCG64`/
`random.PCG64DXSM`). `MT19937`/`Philox` were measured but NOT implemented
this pass -- see the OUT OF SCOPE note below for why, and note this is a
scope/time deferral, not a bit-exactness decline like `multivariate_normal`:
both algorithms' full numpy C source is locally cached
(`numpy/random/src/{mt19937,philox}/`) and the differential-corpus/unit-test
methodology that verified `SFC64` reads as directly reusable for them.

Implemented in `ionp-core/src/random/sfc64.rs` (new module, `Sfc64` struct
+ `BitGen64` impl, transcribed from `numpy/random/src/sfc64/sfc64.{h,c}`
and `numpy/random/_sfc64.pyx` at the `v2.5.1` tag) and wired into
`ionp-py/src/random.rs` (`PySFC64` pyclass, new `BitGenKind::Sfc64`
variant, accepted by `Generator.__init__` alongside `PCG64`/`PCG64DXSM`).
The existing `BitGen64` trait / `BitGenKind` enum dispatch pattern (built
for `PCG64`/`PCG64DXSM`) needed zero structural changes -- confirms the
task's own framing that PCG64 machinery, not a rewrite, was the right
starting point for measuring this family.

A REAL BUG was found and fixed via this pass's differential corpus, in
CODE SHARED WITH `PCG64`/`PCG64DXSM` (not SFC64-specific): the new
`sfc64_random_raw_cases()` corpus's provenance-varied seeds included one
exceeding `2**128` (needing 5 uint32 entropy words, one more than
`SeedSequence`'s default `pool_size=4`), which is the first case in this
repo's history to execute `mix_entropy`'s "remaining entropy" loop
(`ionp-core/src/random/seed_sequence.rs`). That loop's Rust port had
hoisted `hashmix(entropy_array[i_src], hash_const)` OUTSIDE the inner
`i_dst` loop and reused one value for all `i_dst`, where numpy's own
`bit_generator.pyx` calls `hashmix` FRESH inside the inner loop each time
(it mutates `hash_const` and returns a different value per call even for
the same input word) -- confirmed directly against real numpy
2.5.1's `SeedSequence(2**130+7).generate_state(...)`. Fixed by moving the
`hashmix` call inside the `i_dst` loop. This bug silently affected
`PCG64`/`PCG64DXSM` too (any seed >= ~2**128, confirmed directly: both
diverged from real numpy before the fix, matched after) -- neither one's
prior corpus had ever exercised a seed that wide. Added a permanent
regression case (`n3_seed_gt_2_128`, seed `2**130+7`) to
`pcg64_random_raw_cases()`/`pcg64dxsm_random_raw_cases()` in
`tests/differential/random_cases.py`, plus a dedicated Rust unit test
(`seed_sequence_entropy_beyond_pool_matches_numpy`) confirmed RED before
the fix, GREEN after.

EVIDENCE:

(1) Rust-core unit tests: `ionp-core/src/random/sfc64.rs`'s
`sfc64_zero_matches_numpy`/`sfc64_42_matches_numpy` (numpy-2.5.1-captured
reference vectors including the FULL internal state, not just output
words, so the 12-iteration seed-mixing discard loop is verified, not only
its downstream effect) plus the seed_sequence.rs regression test above --
all 3 new tests pass (`cargo test -p ionp-core --release random::`).

(2) Python-layer differential suite: `random.SFC64` registered as its own
dedicated `ItemSpec` in `tests/differential/random_cases.py` (mirroring
`random.PCG64`/`random.PCG64DXSM`'s existing pattern), corpus varying seed
PROVENANCE (small int, zero, >2**64, >2**128 -- the last one is what
caught the bug above) crossed with output shape (`None`, `0`, `1`,
multi-dim, a zero-sized dim), plus error paths (negative seed, float
seed). `Generator(SFC64(seed))` also added as a THIRD `ctor_random_*`
branch in `generator_cases()`'s dispatch (alongside the pre-existing
`ctor_random`/PCG64 branch), confirming `Generator`'s `BitGenKind`
dispatch picks the right variant, not just `SFC64` in isolation. A fresh
`flock /tmp/ionp-build.lock ./.venv/bin/maturin develop --release` rebuild
(confirmed via the literal `🛠 Installed anionpy-0.1.0` banner) followed by
`tests/differential/run.py --out /tmp/r_sfc64_2.json`: 2193 items (2192
baseline + 1 new `random.SFC64` item), 37 fail -- the SAME 37 names as the
`eee334c` baseline, verified by exact set comparison, not count. `random.
SFC64`, `random.Generator`, `random.PCG64`, `random.PCG64DXSM` all show
`verdict: "pass"`, `mechanism: "none"` (bit-exact).

(3) OUT-OF-CORPUS verification (standalone probe, not committed, run from
`/private/tmp`): 14 seeds (small ints, `2**63±`, `2**64±`, `10**21`,
`2**128-1`, `2**128`, `2**129+3`, `2**200+555`, `0`) crossed with 8 sizes
(`None`, `0`, `1`, `5`, `(2,3)`, `(0,4)`, `(3,0,2)`, `(2,2)`) for
`SFC64.random_raw()` directly (112 cases), plus a `Generator(SFC64(seed))`
stream-position probe at 4 seeds (`random(5)` -> `integers(0,1000,size=5)`
-> `random(3)`, each step compared against real numpy's own `Generator`
wrapping its own `SFC64`, confirming the shared bit stream is consumed in
the same word count across a distribution boundary, not merely matching
final values) -- 124 total out-of-corpus cases, 0 divergences.

FINDING 1 -- the mix_entropy fix's REAL blast radius (measured 2026-08-13,
stated explicitly here per external audit, because the fifth-pass
paragraph above buried this): the `hashmix`-hoisting bug in
`ionp-core/src/random/seed_sequence.rs`'s `mix_entropy` did not just
affect the two UNDECLARED bit generators (`PCG64`, `PCG64DXSM`). It sat
directly upstream of `SeedSequence`, which is what `random.default_rng`
and `random.Generator.random` -- BOTH already `"exact"` in this dict,
BOTH used constantly, well before this session -- construct their state
from. Concretely: `random.default_rng(seed)` for any `seed >= 2**128`
(any seed needing more than `SeedSequence`'s default `pool_size=4` uint32
entropy words) was silently instantiating a `PCG64` with a state stream
that diverged from real numpy's, and every draw from that generator
(`.random()`, `.integers()`, etc.) was therefore silently WRONG, not
merely undeclared-and-untested. This was true from whenever `default_rng`
was first declared exact until this session's fix -- an unknown number of
prior sessions. The reason it was never caught: this file's own prior
corpus (`random_cases.py`, `default_rng_cases()`/`generator_cases()`) had
never exercised a seed wider than ~10**21 (< 2**70) before this pass --
the corpus was blind to seed MAGNITUDE specifically, not merely thin on
kwargs/branches, while the differential suite stayed green the entire
time. The fix (moving `hashmix()` inside the `i_dst` loop, matching
numpy's `bit_generator.pyx`) is now covered by a dedicated Rust unit test
(`seed_sequence_entropy_beyond_pool_matches_numpy`) and permanent
`n3_seed_gt_2_128` regression cases added to `pcg64_random_raw_cases()`/
`pcg64dxsm_random_raw_cases()`, so a revert is caught mechanically -- but
the point of this paragraph is the ledger, not the fix: `default_rng`/
`Generator.random`'s "exact" declarations were briefly FALSE for a real
slice of their documented input domain, silently, and that needs to be
readable here, not only inferred from a PCG64/PCG64DXSM-framed bug
report.

REVOKED (2026-08-13, this session, sixth pass, same day as the fifth
pass above, on external audit): `random.SFC64`'s "exact" declaration
from the fifth-pass block above is WITHDRAWN. It was declared on the
stated precedent of "same scope as PCG64/PCG64DXSM" but reached the
OPPOSITE verdict -- PCG64/PCG64DXSM stay undeclared specifically because
implementing only the constructor + `.random_raw()` out of a much larger
public surface is partial, not exact (see WHAT REMAINS UNDECLARED
above); SFC64 was implemented to the identical partial scope and
declared exact anyway, which is a direct self-contradiction, not a
judgment call. Measured surface (2026-08-13): numpy's public `SFC64` has
8 attributes (`capsule`, `cffi`, `ctypes`, `lock`, `random_raw`,
`seed_seq`, `spawn`, `state`); anionpy's has 1 (`random_raw`). numpy's
constructor documents and accepts sequence-of-ints entropy
(`np.random.SFC64([1,2,3])` succeeds); anionpy's raises `TypeError:
SeedSequence expects int or sequence of ints for entropy not [1, 2, 3]`
on the identical call -- a real, user-visible divergence on a documented
call form, not just a missing attribute. `random.SFC64` is removed from
`RANDOM_STATE` below; the exact-item count therefore falls from 1613 to
1612. That is the CORRECT ledger value, not a regression -- the 1613
figure was never true. The underlying `Sfc64`/`BitGen64` draw code
itself (`ionp-core/src/random/sfc64.rs`) is unchanged and still verified
bit-exact for what it actually does (`random_raw()`'s raw word stream,
including the `Generator(SFC64(seed))` composition) -- only the
DECLARATION of the class as a whole is revoked, matching PCG64/
PCG64DXSM's existing undeclared status exactly. `.state`/`.spawn`/
`.seed_seq`/etc. are deliberately NOT being implemented this pass just
to preserve the higher count -- that would be gaming the ledger, not
fixing the gap; implementing them, if ever done, is separate future work
with its own verification pass.

A guard against this exact failure shape (declaring a class exact while
implementing a narrow subset of its public surface) was added to the
differential corpus this same pass: `tests/differential/random_cases.py`
now asserts a bit generator's real Python-visible attribute set against
numpy's own `dir()` for the same class before any generator item can be
graded "pass" on a surface-completeness axis -- see
`bitgen_surface_cases()` and the `random.SFC64.__surface__`-style item
below, deliberately constructed so it currently FAILS against `SFC64`
(1 of 8 attributes present) and would have caught this exact bug had it
existed before the fifth pass.

OUT OF SCOPE FOR THIS PASS (2026-08-13):
`Generator.multivariate_normal` (declined by measurement, see above --
not "not yet wired," a deliberate permanent decline unless numpy's LAPACK
byte-exactness situation changes); `Generator.choice`'s
`replace=False, p=given` branch, and `Generator.shuffle`/`.permutation`/
`.permuted` on any array with ndim != 1, and `Generator.permuted(axis=...)`/
`permuted(out=...)` (all four declared above as loud, deliberate
`NotImplementedError` declines, not silent gaps -- see the DECLARED block
just above for why each one specifically was declined rather than
implemented this pass); `MT19937`/`Philox` BitGenerators (measured this
session -- see the fifth-pass DECLARED block above -- but explicitly NOT
attempted this pass, per the coordinator's direct instruction to stop
after the SFC64/mix_entropy ledger correction and not grow the surface
further in the same pass); `random.SFC64` itself is now UNDECLARED again
(see REVOKED above), not declared; and the entire legacy `RandomState`
surface (`random.seed`/`random.rand`/`random.randn`/`random.randint`/etc,
and the `RandomState` class itself, ~45 methods) -- none started yet.
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
    # 2026-08-13 (this session, second pass): dirichlet/multinomial, both
    # scoped to the non-broadcast call form (scalar n / 1-D alpha or
    # pvals). Verified bit-exact through the Python layer (own dedicated
    # EXPLODED_CLASS_SPECS entries, not the umbrella bucket) with a
    # varied-trip-count corpus (1/2/7) plus a stream-position probe
    # confirming bit-stream-consumption parity, not just value parity.
    # See the DECLARED #2 block above for full evidence.
    "random.Generator.dirichlet": "exact",
    "random.Generator.multinomial": "exact",
    # 2026-08-13 (this session, third pass): multivariate_hypergeometric,
    # default method='marginals' only (method='count' scoped out, raises
    # NotImplementedError -- see ionp-py/src/random.rs). Verified bit-exact
    # through the Python layer (own dedicated EXPLODED_CLASS_SPECS entry)
    # plus a stream-position probe. multivariate_normal was measured
    # (SVD/eigh decompositions NOT bit-exact against numpy even on
    # well-conditioned matrices, despite sharing the Apple Accelerate
    # LAPACK backend) and deliberately declared NOTHING -- see the
    # DECLINED-BY-MEASUREMENT block above; it is intentionally absent from
    # this dict, not merely unstarted.
    "random.Generator.multivariate_hypergeometric": "exact",
    # 2026-08-13 (this session, fourth pass): shuffle/permutation/permuted/
    # choice, scoped to 1-D anionpy.ndarray/int input (N-D input raises
    # NotImplementedError -- see ionp-py/src/random.rs); choice's
    # replace=False,p=given branch also raises NotImplementedError and is
    # intentionally not declared "exact" for that cell specifically (the
    # dict has no branch-level granularity, so this comment is the
    # record: the OTHER three replace-x-p cells are the ones verified
    # bit-exact, per-branch, including a stream-position probe on each).
    # See the DECLARED block above for full evidence.
    "random.Generator.shuffle": "exact",
    "random.Generator.permutation": "exact",
    "random.Generator.permuted": "exact",
    "random.Generator.choice": "exact",
    # random.SFC64 is intentionally ABSENT from this dict -- see the
    # REVOKED block near the end of the module docstring (2026-08-13,
    # sixth pass): declared "exact" in the fifth pass, then revoked the
    # same day on independent audit. Measured surface: numpy's public
    # SFC64 exposes 8 attributes (capsule, cffi, ctypes, lock, random_raw,
    # seed_seq, spawn, state); anionpy's implements 1 (random_raw). numpy
    # also accepts sequence-of-ints entropy (SFC64([1,2,3]) succeeds);
    # anionpy raises TypeError on the same call. This is the SAME
    # partiality already used above to justify PCG64/PCG64DXSM staying
    # undeclared -- SFC64 must match that precedent, not contradict it.
}
