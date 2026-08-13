# anionpy differential test harness

This is the instrument that decides whether an API item may be credited as
covered in `tools/coverage.py`'s ledger. It runs the same inputs through
real `numpy` and through `anionpy` and checks that they agree on values,
dtype, shape, and exceptions. It emits `{item: "pass"|"fail"}` JSON, which
is exactly what `python3 tools/coverage.py --tests <report>` consumes.

```
python3 tests/differential/run.py --out report.json
python3 tools/coverage.py --tests report.json
```

## Files

| file | job |
|---|---|
| `corpus.py` | the seeded, adversarial input corpus. No comparisons, no notion of right/wrong -- just inputs. |
| `harness.py` | the equivalence engine: given a numpy callable, an anionpy callable, and cases, decides pass/fail per case and per item. No test registration lives here. |
| `registry.py` | the table of API items under test (`ItemSpec` + `REGISTRY`), plus the ufunc-block machinery shared by both the plain and ufunc paths: `make_ionp_array_converter` (numpy.ndarray -> anionpy.ndarray, tautology-proof) and `build_ufunc_dispatcher` (routes a case's call-form tag to `__call__`/`.reduce`/`.accumulate`/`.outer`/`.reduceat`/`.at`, defensively copying mutable `out=`/`.at()` targets -- see its docstring). This is what you edit to add coverage for a new item. |
| `ufunc_introspect.py` | enumerates the 134 ufuncs from `tools/numpy_surface.json` (`UFUNC_NAMES`, never hand-typed) and derives each one's `UfuncInfo` (`.nin`/`.nout`/`.identity`/`.signature`/`.types`) plus, via real trial calls against numpy (`applicable_methods`), which of `reduce`/`accumulate`/`outer`/`reduceat`/`at`/`out=`/`where=`/`dtype=` that specific ufunc actually accepts. Looks only at numpy; no anionpy import, no comparisons. |
| `ufunc_corpus.py` | the string corpus for the 28 `char.*`/`strings.*` ufuncs (numeric ufuncs reuse `corpus.py` unchanged), plus the deterministic `sampled()` "every Nth" helper and its pinned `UNARY_SAMPLE_STRIDE` constant that keeps the binary-ufunc x scalar-operand cross product bounded without silent truncation. |
| `ufunc_cases.py` | turns a `UfuncInfo` + its `applicable_methods()` map into the full `(label, args, kwargs)` case list for one ufunc -- plain call over the full corpus, `dtype=`/`out=`/`where=` kwarg cases, and one method-form case group per applicable ufunc-protocol method. `build_cases_for_item(spec)` is the `kind="ufunc"` entry point `run.py` dispatches to. |
| `ufunc_registry.py` | builds the 134 `kind="ufunc"` `ItemSpec` entries (one per `ufunc_introspect.UFUNC_NAMES` name) and merges them into `registry.REGISTRY` as an import side effect; asserts the count is exactly 134 and refuses to silently overwrite a name collision. |
| `selftest.py` | deliberately broken (and one deliberately correct) shim functions, used to prove the harness actually catches bugs. Not part of the real registry -- none of its names exist in `numpy_surface.json`. |
| `run.py` | the JSON-emitting runner (`--out report.json`) and the case-building logic (`build_cases`) shared with pytest, including the `kind="ufunc"` branch that delegates to `ufunc_cases.build_cases_for_item`. Also runs the self-test demo (`--selftest`). |
| `test_differential.py` | pytest wrapper around `REGISTRY`, one parametrized test per item. |
| `test_selftest.py` | pytest wrapper around `selftest.py`'s proof-of-detection fixtures. |
| `conftest.py` | sys.path plumbing for pytest collection. |
| `_bootstrap.py` | sys.path plumbing shared by every module that needs `import anionpy`, so the suite works whether invoked via pytest, via `python3 tests/differential/run.py`, or from a different cwd. |

## Adding a new API item

Almost always this is one entry in `REGISTRY` in `registry.py`:

```python
"linalg.norm": ItemSpec(
    name="linalg.norm", kind="unary",   # or "binary" / "method" / "custom"
    atol=1e-9, rtol=1e-9,               # mandatory for float/complex results
),
```

`name` must be the exact dotted string as it appears in
`tools/numpy_surface.json` (bare name for top-level/submodule items,
`"ndarray.<attr>"` for methods/attrs and dunders) -- that string is the
coverage-ledger key, so this is not cosmetic.

Pick `kind` by call shape:

- **`unary`** -- `f(array)`. Runs against every array in `corpus.unary_corpus()`.
- **`binary`** -- `f(a, b)`. Runs against every pair in `corpus.binary_corpus()`,
  including shapes that must fail to broadcast.
- **`method`** -- `array.method(*args)` where the method needs extra
  arguments beyond the array itself (e.g. `reshape(shape)`). Supply
  `args_provider=lambda arr: [(label, (args_tuple, kwargs_dict)), ...]`.
- **`custom`** -- anything else (multi-output like `linalg.svd`, generator
  functions like `arange` that don't take an array at all, etc). Supply
  `custom_cases=lambda: [(label, args_tuple, kwargs_dict), ...]`.

If `anionpy`'s name for the thing differs from numpy's canonical name, set
`ionp_path` explicitly. If you need to bridge a non-numpy-shaped helper for
diagnostic purposes only, use `ionp_adapter` -- but see the honesty note
below before doing that in the real registry.

That's the whole recipe. No new test file, no boilerplate loop, no corpus
duplication -- this is what "design for ~1169 items" means in practice.

## Tolerance policy

- **Integer, unsigned, and bool results**: exact. `np.array_equal`, no
  tolerance parameter accepted, no exceptions.
- **dtype**: exact `np.dtype` equality, always. `float32` is never "close
  enough" to `float64`.
- **shape**: exact tuple equality, always.
- **exceptions**: if numpy raises, anionpy must raise the same exception type
  (or a type in that item's declared `exception_equivalences`, empty by
  default). If numpy does not raise, anionpy must not raise either.
- **verdict**: an item is `pass` only if every case in its applicable
  corpus passes. No partial credit -- `coverage.py`'s ledger has no slot
  for "mostly right", so neither does this harness.
- **Float and complex results: bit-exact by default, ULP tolerance opt-in
  only.** See `reports/ionp-ulp-tolerance-decision-2026-08-01.md` for the
  full argument; summary below.

  numpy's own float ufunc loops are not correctly rounded (measured:
  `np.abs` on complex64 disagrees with the platform's own scalar `hypotf`
  on 35.1% of 20,000 seeded inputs, by 1 ULP -- see
  `KNOWN-DIFFERENCES.md`), so demanding bit-exactness against numpy is not
  a well-posed bar for every float/complex item. The policy:

  1. **Default is bit-exact (ULP distance 0).** An undeclared float/complex
     item is graded exactly, full stop -- no implicit tolerance, no global
     default of any kind.
  2. **A per-item `ItemSpec.ulp_tolerance` (int, default `None`) is the
     ONLY way to relax that**, and it must be paired with a non-empty
     `ItemSpec.ulp_justification` string citing the recorded evidence that
     numpy's own loop is not correctly rounded for this item.
     `ItemSpec.__post_init__` (`registry.py`) **refuses to construct** an
     `ItemSpec` with `ulp_tolerance` set and no justification -- undeclared
     tolerance is structurally impossible, not merely discouraged.
     `ulp_tolerance` defaults to 1 ULP in practice; a larger value needs its
     own separate justification.
  3. **Comparison is by real ULP distance, never `np.isclose`/`np.allclose`.**
     `harness.ulp_distance(a, b, width)` implements Bruce Dawson's
     biased-integer total-order mapping: reinterpret the IEEE754 bit
     pattern as a monotonic integer, then take the absolute integer
     difference. `harness.max_ulp_distance(np_out, ionp_out, dtype)` is the
     array/complex-aware wrapper actually used by the harness -- complex
     values are compared **per-component** (real and imaginary parts each
     get their own ULP distance; the element's distance is the max of the
     two). Special-value rules (deliberately chosen, not incidental):
       - **NaN**: both-NaN is distance 0 (NaN-ness must match; sign/payload
         don't matter). Exactly one NaN is distance `+inf` -- never
         tolerated, at any declared tolerance.
       - **+-inf**: same-signed infinities are distance 0. Anything else
         touching an infinity (opposite-signed, or one side finite) is
         distance `+inf`.
       - **+0.0 vs -0.0**: distance `+inf` -- i.e. **NOT** within any finite
         ULP tolerance, even though the raw bit-order transform alone would
         put them 0 ULP apart. Deliberate: sign of zero is IEEE754-meaningful
         (branch cuts in complex sqrt/log/atan2 depend on it), so a harness
         whose job is catching wrong special-case branches must not treat "the
         sign bit is wrong" as "close enough". See `test_ulp_distance.py`.
  4. **`ItemSpec.atol`/`ItemSpec.rtol` (`np.allclose(..., equal_nan=True)`)
     remain as a legacy path**, used only when an item does NOT declare
     `ulp_tolerance` -- this is what keeps every item that predates the ULP
     mechanism graded identically to before (no item's verdict changes as a
     side effect of the mechanism existing). New items needing float/complex
     slack should declare `ulp_tolerance`, not `atol`/`rtol`.
  5. **The ledger distinguishes tolerant passes from exact ones.**
     `ItemResult.tolerant` is `True` iff the item's `pass` verdict required
     nonzero ULP slack on at least one case -- a bit-exact pass is
     `tolerant=False` even when a `ulp_tolerance` was declared and simply
     wasn't needed. `ItemResult.max_ulp_observed` records the worst-case ULP
     distance seen. `tools/coverage.py`'s human-readable ledger reports
     both `of which bit-exact vs numpy` and `of which ULP-tolerant` counts
     underneath the headline coverage percentage, so "100% coverage" can
     never silently come to mean "100% within some unstated slop" -- see
     that decision doc's point 3.

  As of 2026-08-01, **no item in the real `REGISTRY`/`ufunc_registry.py`
  declares `ulp_tolerance`** -- this mechanism is built and self-tested
  (see `test_ulp_distance.py` and the `selftest.ulp_sqrt_*` fixtures in
  `selftest.py`) but not yet applied to any real item; declaring specific
  items is separate, future work with its own evidence trail.

## The corpus (`corpus.py`)

Seeded with a single pinned constant (`SEED = 20260731`); changing it is a
deliberate, reviewed act, same as re-pinning `numpy_surface.json`. Covers:

- shapes: 0-d, 1-element, empty (including empty-on-a-middle-axis), 1-D
  through rank-5
- dtypes: bool, int8/16/32/64, uint8/16/32/64, float32/64, complex64/128
- special float/complex values: NaN, +inf, -inf, +0.0/-0.0, denormals
- integer dtype boundaries: min/max for every signed and unsigned width
- non-contiguous views, negative strides, transposes, sliced sub-views,
  Fortran-order arrays
- broadcasting pairs, including pairs that must **fail** to broadcast
- mixed-dtype pairs (int/float, bool/int, float/complex) to exercise
  promotion rules
- fixed-width integer overflow/wraparound pairs (`int8` near `INT8_MAX`,
  `uint8` near `0`, `int64` near `INT64_MAX`)

`unary_corpus()` and `binary_corpus()` are the two entry points; `method`
and `custom` kinds build on top of `unary_corpus()` or supply their own.

## Call-form coverage (added 2026-07-31, after a real false-positive)

On 2026-07-31 the harness credited `ndarray.reshape` and `ndarray.__add__`
as `pass` while, by hand: `reshape(-1)`, `reshape(4)`, `reshape(2, 2)`, and
`array + 1` all raised `TypeError` against real anionpy. The registry only ever
exercised ONE call form per item -- the tuple form for `reshape`, the
array-vs-array form for `__add__` -- so a partial implementation that only
supports that one overload was indistinguishable from a complete one.
Concurrently with that fix, re-grading also surfaced (and fixed) a second,
unrelated, and more fundamental bug: `ItemSpec.resolve_ionp()`'s
`"ndarray."`-prefixed path never converted the corpus's `numpy.ndarray`
inputs into real `anionpy.ndarray` instances before dispatching, so every
`"ndarray.<attr>"` item was silently calling **numpy's own** method on a
numpy array and comparing numpy against numpy -- a test that can never fail
regardless of what anionpy does. Both bugs independently produced the same
symptom (false `pass`); both are fixed now (`registry.py`'s `_to_ionp` /
`CallForm` machinery and `harness.py`'s `_dtype_of`, respectively).

Three mechanisms now guard against this class of hole recurring:

1. **`CallForm`** (`registry.py`) -- the declarative unit for one argument
   shape. `kind="method"` items declare `call_forms: list[CallForm]`
   (mandatory, non-empty -- `ItemSpec.__post_init__` refuses to construct
   the item otherwise), and every form is tried against every array in
   `corpus.unary_corpus()`. `_reshape_forms()`, `_transpose_forms()`,
   `_astype_forms()`, `_copy_forms()`, and `_getitem_forms()` are the
   current libraries; adding a new shape-taking method means writing one of
   these, not one `args_provider` lambda covering a single form.
2. **`kind="binary_op"`** -- dedicated case-building for the arithmetic
   dunders (`__add__`/`__radd__`/`__mul__`/`__rmul__`): every corpus array
   is paired against every *other* corpus array (`corpus.binary_corpus()`,
   unchanged) **and** against every scalar-like right-hand operand
   (`corpus.scalar_operands()`: python `int`/`float`/`bool`/`complex`,
   numpy scalars, a 0-d array) -- because `a + b` (both arrays), `a + 1`
   (python int), and `a + np.array(1.0)` (0-d array) are three genuinely
   different code paths in any real implementation, and crediting the first
   as proof of the other two is exactly the bug being fixed here.
3. **`_require_varargs_form_if_numpy_supports_it()`** -- an import-time
   assertion. For every `kind="method"` item it runs
   `inspect.signature()` against numpy's own bound method; if that
   signature declares a `VAR_POSITIONAL` parameter (as `reshape`'s and
   `transpose`'s real numpy signatures do: `a.reshape(*shape, ...)`) but no
   declared `CallForm` is flagged `exercises_varargs=True`, the registry
   fails to import. This is deliberately a hard failure, not a lint
   warning: a human forgetting to add the varargs form is precisely how the
   original bug slipped through. Where numpy's signature can't be
   introspected at all (true for some C-implemented callables), this check
   is a no-op and the *only* remaining safety net is #1's unconditional
   "call_forms must be explicit and non-empty" -- which is deliberate, per
   the task brief: fall back to requiring an explicit form list rather than
   silently testing one form.

See `registry.py`'s module docstring for the full design rationale, and
`selftest.py`'s `reshape_all_call_forms` / `reshape_tuple_form_only` pair
for a runnable proof: the latter is a shim that implements ONLY the
single-tuple reshape form (byte-for-byte the shape of the real bug) --
the OLD harness would have passed it, the NEW harness (via the identical
`_reshape_forms()` library the real `ndarray.reshape` item uses) fails it.

## The ufunc block (tests-first, added 2026-08-01)

`anionpy` implements no ufuncs today (`anionpy/__init__.py` exports `array`,
`ndarray`, `dtype`, `sum_f64` only, and declares no ufunc in
`__ion_state__`). This harness is nonetheless ready for the ufunc block
*before* it exists, per GOAL-ionp.md's ordering (ufunc engine is step 2:
"one engine, not 134 functions"): `ufunc_registry.py` adds one
`kind="ufunc"` `ItemSpec` per name in `ufunc_introspect.UFUNC_NAMES` --
**134 of 134**: 106 top-level (`add`, `sqrt`, `matmul`, `isnat`, `gcd`,
`divmod`, ...), 17 `np.strings.*`, 11 `np.char.*` -- to `registry.REGISTRY`,
enumerated straight from `tools/numpy_surface.json`, never hand-typed.

Because zero of the 134 are declared in `anionpy.__ion_state__`,
`registry.ItemSpec.resolve_ionp()`'s `kind="ufunc"` branch resolves to
`None` for every one of them (an `AttributeError` walking the dotted path
against the real `anionpy` module), and `harness.evaluate()`'s existing
absent-short-circuit fires before any case is built or run. Verified by a
real run: `python3 tests/differential/run.py --out report.json` grades all
134 as `fail`/absent with `(0/0 cases)`, in well under a second, no crash,
no hang -- and `python3 tools/coverage.py --tests report.json` shows them
landing in the `absent` bucket (not `phantom`, not `failing`), same as
every other undeclared item. This is the correct, honest pre-implementation
state, not a gap in the harness.

**Where each ufunc's test matrix comes from (never guessed):**

- Arity, output count, and dtype-loop set: `.nin`/`.nout`/`.types`/
  `.identity`/`.signature` read directly off the real numpy ufunc object
  (`ufunc_introspect.describe`). A per-ufunc "typed sample" (concrete input
  arrays in a dtype/shape that ufunc itself accepts) is parsed out of
  `.types`' legacy loop signatures (e.g. `"dd->d"` -> two `float64`
  arrays) -- this is what gets `isnat` a `datetime64` sample, `gcd`/
  `bitwise_count` an integer sample, and `ldexp` a mixed float+int sample
  without a single hand-maintained "this ufunc wants dtype X" rule. The 4
  gufuncs (`matmul`/`vecdot`/`vecmat`/`matvec`) get a literal shape-valid
  sample instead, since their `.types` loops describe dtype only, not the
  `(n,k),(k,m)->(n,m)`-style core-dimension contract that actually matters
  for them.
- Which of `reduce`/`accumulate`/`outer`/`reduceat`/`at` and the `out=`/
  `where=`/`dtype=` kwargs a *specific* ufunc accepts:
  `ufunc_introspect.applicable_methods()` **tries each one against real
  numpy** inside `warnings.catch_warnings()` and records whether it raised
  -- never assumed from `nin`/`nout` alone. This is what correctly
  discriminates gufuncs (`matmul` rejects `reduce`/`at`/`outer`),
  multi-output ufuncs (`divmod`/`modf`/`frexp` reject all five methods AND
  all three kwarg forms), string ufuncs (`char.str_len` rejects
  `reduce`/`accumulate`/`reduceat`), and dtype-restricted ufuncs, with zero
  hand-maintained per-ufunc exception rules.

**Corpus:** numeric ufuncs (106 of 134) reuse `corpus.py`'s existing seeded
corpus (`SEED = 20260731`) unchanged -- same empty/0-d/NaN/inf/-0.0/
negative-and-non-contiguous-stride/broadcasting/integer-overflow/NEP-50
coverage every other item gets. The 28 `char.*`/`strings.*` ufuncs get a
dedicated string corpus (`ufunc_corpus.py`, seeded off the same constant)
because the numeric corpus is the wrong tool twice over: several
`char.*`/`strings.*` names are the *literal same ufunc object* as their
numeric counterpart (`np.char.add is np.add` -> `True`; string dispatch
happens via NEP 42/43 DType loops invisible in `.types`), so testing them
under their `char.`/`strings.` name with numeric input would silently
re-test the numeric alias instead of real string semantics, while the
string-exclusive ufuncs (`str_len`, `isalpha`, ...) would just always raise
on numeric input.

**Cross-product bounding, visibly, not silently:** a binary ufunc's
"unary-pool x scalar-operand" cases would be 144 x 16 = 2304 per item
before bounding. `ufunc_corpus.sampled(pool, stride)` takes a fixed,
documented, deterministic "every Nth" slice (`UNARY_SAMPLE_STRIDE = 12`,
a module-level constant in `ufunc_corpus.py`, not a random subset and not a
silent truncation) for that cross product and for the `.reduce`/
`.accumulate`/`.outer`/`.reduceat`/`.at` method-form corpus slices. The
guaranteed-valid typed_sample case is always included in addition, so no
ufunc's method-form coverage depends entirely on the sampled slice landing
on a compatible shape.

**Actual numbers from a real run** (`.venv/bin/python3`, numpy 2.5.1):
134/134 ufuncs registered and graded; `ufunc_cases.build_cases_for_item`
over all 134 names builds **21,025 cases total** (max 262 for `add`,
`arctan2`, `atan2`, `bitwise_and`, `bitwise_left_shift`; min 3) in well
under a second, zero errors, zero `RuntimeWarning`s (the domain-restricted
kwarg-form cases -- e.g. `arccos`'s typed sample landing outside `[-1, 1]`
-- are built under `warnings.simplefilter("ignore")` /
`np.errstate(all="ignore")`, since the resulting NaN is handled correctly
by the existing `equal_nan=True` comparison policy and is not itself the
thing under test).

**Known, documented scope exclusions** (none of the 134 ufuncs are skipped
entirely; these are narrower, principled simplifications):

- `'O'` (object-dtype) legacy loops are never sampled -- there is no
  principled generic Python-object sample, and it is out of scope for this
  block (`ufunc_introspect._parse_loop` skips any loop signature containing
  `'O'`).
- `char.*`/`strings.*` items are tested exclusively via the dedicated
  string corpus under their string name; the fact that several are literal
  aliases of a numeric ufunc (`char.add is np.add`) is not separately
  exercised under the `char.`/`strings.` registry key (it *is* exercised
  under the numeric key, e.g. `"add"`, via the numeric corpus).
- Multi-output ufuncs (`divmod`/`modf`/`frexp`, `nout=2`) get only the
  plain-`call` form: real numpy itself rejects `out=`/`where=`/`dtype=`
  and all five protocol methods for these (verified by trial, not assumed
  -- see `applicable_methods`), so `ufunc_cases.py` correctly builds zero
  cases for those forms rather than a more elaborate tuple-`out=` case that
  numpy wouldn't accept either.
- `harness.py`'s `run_case`/`evaluate` gained a `multi_output` flag
  (`compare_multi_output`) so `divmod`/`modf`/`frexp` get correct
  arity-then-elementwise tuple comparison instead of the single-`.dtype`
  assumption the rest of the harness makes; `ufunc_registry.py` sets it
  from each ufunc's own `.nout > 1`, never hand-listed by name.

**Mutation safety:** `harness._call()` invokes the numpy-side and ionp-side
callables with the *same* `args`/`kwargs` objects, sequentially. `.at()`
mutates in place and `out=` writes into whatever array it's given, so
without protection the numpy call's side effect would corrupt what the
anionpy call then reads. `registry.build_ufunc_dispatcher()` defends against
this on every dispatch: it `.copy()`s the `.at()` target and any `out=`
kwarg array immediately before use, on both sides independently -- see its
docstring for the full reasoning, and `selftest.ufunc_add_silent_at` below
for a fixture that specifically exercises this path.

Proof this isn't a tautology: `selftest.py`'s four new `ufunc_*` fixtures
(below) run real, deliberately-broken ufunc-like shims through this exact
machinery -- `build_ufunc_dispatcher`, the real trial-detected
applicability, the real sampled corpus -- for two different real ufuncs
(`add`, `sqrt`), and the harness catches every one of them.

## Honesty notes (read before "fixing" a failing item)

- **Missing is `fail`, not a crash.** All 19 items `anionpy` currently
  declares in `anionpy.__ion_state__` (`array` plus 18 `ndarray.*`
  methods/attrs/dunders) are registered below, alongside `sum` and
  `linalg.svd` (not in `__ion_state__`, both fully absent). Re-graded
  2026-07-31 after the call-form and conversion-layer fixes above:
  **5 pass, 16 fail** (`ndarray.__array__`, `.ndim`, `.shape`, `.size`,
  `.__len__` pass; everything that returns an `ndarray` -- including
  `reshape` and `__add__` -- currently fails). Run `python3 run.py
  --verbose` for the full, current, reproducible breakdown; do not trust a
  stale number copied into this file over that live output.
- **The single biggest cause of the 16 fails is `ndarray.dtype` itself**:
  `anionpy`'s `.dtype` returns a custom `anionpy.dtype` object that
  `np.dtype(...)` refuses to coerce (`TypeError: Cannot interpret
  "dtype('float64')" as a data type`). Since almost every other item's
  result is an `ndarray` whose `.dtype` gets compared the same way, this
  one interop gap cascades into most of the other failures too -- it is
  reported as a plain "dtype mismatch" per case (see `harness.py`'s
  `_dtype_of`), not hidden or specially excused. Once anionpy's dtype objects
  are numpy-interoperable, expect several of today's 16 fails to collapse
  to their item's OWN remaining bug (missing call forms, missing scalar
  arithmetic, etc), not to `pass` automatically.
- **`sum` is registered against `anionpy.sum`, not `anionpy.sum_f64`.** The
  registry's `ionp_path` for the `"sum"` item is the *true* numpy-surface
  name, so the moment the Rust-side agent lands a numpy-signature-compatible
  `anionpy.sum` and declares it in `__ion_state__`, this same registry entry
  starts grading it -- no test-side change required. We deliberately did
  **not** silently adapt `sum_f64` into passing as `"sum"` in the real
  registry: `sum_f64` requires a contiguous `float64` `ndarray` and raises
  `TypeError` on anything else (other dtypes, non-contiguous views), and
  returns a bare Python `float` rather than a `np.float64` scalar. Grading
  it as `"sum"` against the full corpus would legitimately still be
  `fail` (dtype mismatch on every non-float64 case, exception-type
  mismatch on non-contiguous/other-dtype cases) -- which is the honest
  answer, not a harness defect. If you want to see the harness exercise
  `sum_f64` directly, that's what `selftest.correct_sum`'s sibling fixtures
  in `selftest.py` are for (see below) -- they are explicitly out-of-band
  from the coverage ledger.

## Proof the harness catches real bugs

`python3 tests/differential/run.py --selftest --verbose` runs a
seventeen-fixture proof-of-detection suite (`selftest.py`), fully
independent of the real `REGISTRY` (its item names never appear in
`numpy_surface.json`, so `coverage.py` never sees them). The first six
pre-date the ufunc block (reshape call-form / tautology-fix proofs); four
more (added 2026-08-01, morning) prove the ufunc-block machinery; the last
seven (added 2026-08-01, this task) prove the ULP-tolerance mechanism
specifically:

- `selftest.correct_sum` -- a correct `np.sum` passthrough. **Must pass.**
  If this doesn't pass, nothing else the harness reports can be trusted.
- `selftest.wrong_dtype_sum` -- always returns `float32` regardless of
  input dtype. **Must fail** on a dtype mismatch.
- `selftest.off_by_tolerance_sum` -- adds a fixed `1e-3` to the true sum,
  well outside the item's declared `atol=rtol=1e-9`. **Must fail** on a
  values-differ-beyond-tolerance mismatch (and, incidentally, also trips a
  dtype mismatch for every integer-input case, since `+ 1e-3` promotes to
  float64).
- `selftest.no_raise_on_bad_broadcast_add` -- swallows numpy's
  `ValueError` on incompatible shapes and silently truncates instead.
  **Must fail** with "numpy raised ValueError ... anionpy returned ... instead
  of raising" on the broadcast-incompatible corpus cases (and pass on the
  compatible ones -- but partial correctness is still `fail` for the item
  as a whole, which the run demonstrates).
- `selftest.reshape_all_call_forms` / `selftest.reshape_tuple_form_only` --
  see "Call-form coverage" above.
- `selftest.ufunc_add_correct` -- a fake ufunc object (`_FakeAddBase`) that
  forwards every protocol entry point (`__call__`, `.reduce`,
  `.accumulate`, `.outer`, `.reduceat`, `.at`) straight to the real
  `np.add`, wrapped through the real `registry.build_ufunc_dispatcher()`
  and run through the real, trial-detected `add` case set
  (`ufunc_cases.build_cases_for_item`, `numpy_path="add"`). **Must pass**
  -- if it doesn't, `build_ufunc_dispatcher` itself is broken, independent
  of any real anionpy bug.
- `selftest.ufunc_add_wrong_reduce` -- same base, but `.reduce` always
  returns `zeros_like` the real answer regardless of input. **Must fail**
  only on `reduce/*` cases (plain-call/accumulate/outer/reduceat/at cases
  still pass), proving the harness can localize a defect to one call form.
- `selftest.ufunc_add_silent_at` -- `.at()` silently no-ops instead of
  mutating its target in place (still returns `None`, exactly like a
  correct `.at()` would -- the bug is only visible in the mutated array).
  **Must fail** on `at/*` cases -- this is the specific mutation-safety
  path `build_ufunc_dispatcher`'s defensive `.copy()` exists for.
- `selftest.ufunc_sqrt_wrong_call` -- a second, differently-shaped ufunc
  (unary, `nin=1`): `__call__` returns `np.sqrt(x) + 1` instead of
  `np.sqrt(x)`, with `.at()` still forwarding correctly. **Must fail** on
  `call/*` cases while `at/*` cases pass -- proves the defect-localization
  works in the other direction (call broken, method correct) too, and on a
  second, non-`add` ufunc.

The following seven all run `np.sqrt` over the same fixed, hand-inspectable
corpus (`[4.0, 1.0, 0.25, 0.0, -1.0]`, `float32`) through a declared
`ulp_tolerance=1` (with a real justification string, since `ItemSpec`
refuses to construct otherwise) -- see "ULP tolerance policy" above:

- `selftest.ulp_sqrt_correct` -- a correct `np.sqrt` passthrough. **Must
  pass with `ItemResult.tolerant == False`** (bit-exact, ULP distance 0
  throughout) -- the positive control that the mechanism doesn't force
  every declared-tolerant item to read as "used the slack".
- `selftest.ulp_sqrt_one_ulp_off` -- every result nudged exactly 1 ULP
  (via repeated `np.nextafter`) off the true value. **Must pass with
  `ItemResult.tolerant == True`, `max_ulp_observed == 1.0`** -- the
  positive control that a genuinely 1-ULP-off result is exactly what this
  mechanism exists to accept.
- `selftest.ulp_sqrt_wrong_formula` -- multiplies the correct result by
  `1.001` (a real formula bug). **Must fail** -- misses by thousands of
  ULP, far outside `ulp_tolerance=1`.
- `selftest.ulp_sqrt_wrong_dtype` -- correct value, wrong dtype
  (`float64` instead of `float32`). **Must fail** on a plain dtype
  mismatch -- proves the structural dtype check is untouched by (and
  checked *before*) any declared ULP tolerance.
- `selftest.ulp_sqrt_wrong_zero_sign` -- correct everywhere except
  `sqrt(0.0)` comes back `-0.0` instead of `0.0`. **Must fail** (ULP
  distance `+inf`) -- proves the deliberate "+-0.0 is never within any
  finite tolerance" policy actually holds in the harness, not just in the
  docstring.
- `selftest.ulp_sqrt_wrong_nan` -- `sqrt` of the negative input returns
  `0.0` instead of `NaN`. **Must fail** (ULP distance `+inf`) -- proves
  "exactly one side NaN" is never tolerated.
- `selftest.ulp_sqrt_off_by_two` -- every result nudged exactly 2 ULP off
  the true value, one more than the declared `ulp_tolerance=1` permits.
  **Must fail** with `ULP distance 2.0 exceeds declared tolerance 1` -- the
  fixture that most directly proves the bound is exactly 1, not "roughly
  1 in practice".

Actual output from a real run (`2026-08-01`, `.venv/bin/python3`):

```
[PASS] selftest.correct_sum  (144/144 cases)  -- ok
[FAIL] selftest.no_raise_on_bad_broadcast_add  (25/28 cases)  -- 3/28 cases failed
    [broadcast_fail/mismatched_2d] numpy raised ValueError(operands could not be broadcast together with shapes (3,4) (2,5) ), anionpy returned array([...]) instead of raising
[FAIL] selftest.off_by_tolerance_sum  (3/144 cases)  -- 141/144 cases failed
    [sweep/0d/bool] dtype mismatch: numpy=int64 anionpy=float64
    [sweep/0d/float64] values differ beyond atol=1e-09 rtol=1e-09: numpy=array([-12.44186284]) anionpy=array([-12.44086284])
[PASS] selftest.reshape_all_call_forms  (1436/1436 cases)  -- ok
[FAIL] selftest.reshape_tuple_form_only  (432/1436 cases)  -- 1004/1436 cases failed
    [sweep/0d/bool/bare_int_full] numpy returned array([False]) normally, anionpy raised TypeError(reshape_tuple_form_only: only a single tuple-shape positional argument is supported ...) instead
[PASS] selftest.ufunc_add_correct  (262/262 cases)  -- ok
[FAIL] selftest.ufunc_add_silent_at  (254/262 cases)  -- 8/262 cases failed
    [at/sweep/1elem/int32] exact value mismatch (dtype int32)
    [at/sweep/1d/float32] values differ beyond atol=1e-09 rtol=1e-09: numpy=array([ 1.60512  , -3.2967448, ...]) anionpy=array([ 0.60512  , -3.2967448, ...])
[FAIL] selftest.ufunc_add_wrong_reduce  (253/262 cases)  -- 9/262 cases failed
    [reduce/typed_sample] exact value mismatch (dtype int64)
    [reduce/sweep/1d/float32] values differ beyond atol=1e-09 rtol=1e-09: numpy=array([24.846706]) anionpy=array([0.])
[FAIL] selftest.ufunc_sqrt_wrong_call  (50/158 cases)  -- 108/158 cases failed
    [call/sweep/0d/bool] values differ beyond atol=1e-09 rtol=1e-09: numpy=array([0.]) anionpy=array([1.])
[PASS] selftest.ulp_sqrt_correct  (1/1 cases)  -- ok
[FAIL] selftest.ulp_sqrt_off_by_two  (0/1 cases)  -- 1/1 cases failed
    [fixed] ULP distance 2.0 exceeds declared tolerance 1 (justification: 'selftest fixture, not a real registry item ...'): numpy=array([2. , 1. , 0.5, 0. ], dtype=float32) anionpy=array([2.0000005e+00, 1.0000002e+00, 5.0000012e-01, 2.8025969e-45], dtype=float32)
[PASS] selftest.ulp_sqrt_one_ulp_off  (1/1 cases)  -- ok
[FAIL] selftest.ulp_sqrt_wrong_dtype  (0/1 cases)  -- 1/1 cases failed
    [fixed] dtype mismatch: numpy=float32 anionpy=float64
[FAIL] selftest.ulp_sqrt_wrong_formula  (0/1 cases)  -- 1/1 cases failed
    [fixed] ULP distance 8389.0 exceeds declared tolerance 1 (...): numpy=array([2. , 1. , 0.5, 0. ], dtype=float32) anionpy=array([2.002 , 1.001 , 0.5005, 0.    ], dtype=float32)
[FAIL] selftest.ulp_sqrt_wrong_nan  (0/1 cases)  -- 1/1 cases failed
    [fixed] ULP distance inf exceeds declared tolerance 1 (...): numpy=array([2. , 1. , 0.5, 0. ], dtype=float32) anionpy=array([2. , 1. , 0.5, 0. ], dtype=float32)
[FAIL] selftest.ulp_sqrt_wrong_zero_sign  (0/1 cases)  -- 1/1 cases failed
    [fixed] ULP distance inf exceeds declared tolerance 1 (...): numpy=array([2. , 1. , 0.5, 0. ], dtype=float32) anionpy=array([ 2. ,  1. ,  0.5, -0. ], dtype=float32)
[FAIL] selftest.wrong_dtype_sum  (16/144 cases)  -- 128/144 cases failed
    [sweep/0d/bool] dtype mismatch: numpy=int64 anionpy=float32

SELF-TEST OK: harness correctly passed the 5 correct shim(s) and failed all 12 deliberately broken ones.
```

(trimmed to one representative failure line per item above; the real
run prints up to `MAX_FAILURES_KEPT=8` per item. Also runnable as `pytest
tests/differential/test_selftest.py -v`, which asserts all seventeen
verdicts and additionally asserts that every `fail` carries at least one
recorded diagnostic line, i.e. it never fails silently. The `ulp_sqrt_*`
fixtures are the direct evidence for "does a 1-ULP tolerance still catch
real defects": `wrong_formula`, `wrong_dtype`, `wrong_nan`,
`wrong_zero_sign`, and `off_by_two` all FAIL under `ulp_tolerance=1`;
only `correct` (bit-exact) and `one_ulp_off` (genuinely 1 ULP away) PASS.)

## Running

```
# real registry -> JSON report for coverage.py
python3 tests/differential/run.py --out report.json
python3 tools/coverage.py --tests report.json

# same registry, via pytest (one red/green test per item)
pytest tests/differential/test_differential.py -v

# proof-of-detection demo
python3 tests/differential/run.py --selftest --verbose
pytest tests/differential/test_selftest.py -v

# everything
pytest tests/differential/ -v
```
