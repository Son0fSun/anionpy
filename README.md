# anionpy

A NumPy-compatible array library with a native Rust core.

**Status: early, incomplete, and honest about it. Do not use this as a NumPy
replacement in production. Read the coverage table below before the feature
list — there isn't a feature list.**

## What this is

`anionpy` implements the NumPy API — the same names, the same signatures, the
same errors — on top of an array engine written in Rust. Python is the import
and dispatch surface only. No numerical loop runs in Python; every element-wise
operation, reduction, and linear-algebra call is executed in compiled code.

Correctness is defined against real NumPy, not against a specification of it.
Every implemented item is backed by a *differential* test: the same inputs are
fed to NumPy and to `anionpy`, and the results are compared bit-for-bit unless a
tolerance is explicitly recorded and justified. An item that is not backed by a
passing differential test does not count as implemented, even if the function
exists and appears to work.

NumPy is an **optional** dependency. `import anionpy` succeeds with no NumPy
installed. The `compat` extra pulls NumPy in so that `anionpy`'s exception types
subclass NumPy's, and so `numpy.asarray()` interop works.

**With one disclosed exception:** the `char` and `strings` modules require NumPy
at runtime, not just for interop. `anionpy` has no native string array type yet
— those functions take and return real `numpy.ndarray` objects with `S`/`U`
dtype, doing the work in Rust in between. Without NumPy the modules import but
nothing under them can be called. This covers 84 of the items counted below;
the details are in `KNOWN-DIFFERENCES.md`.

## Status

Measured by `tools/coverage.py --tests` against the full public NumPy surface:

```
COVERAGE: 51.59%   (1574/3051)     measured at eadc710a73e56cd0273daedf0469ec13a330d1f4
absent      1141   not implemented
undeclared   335   resolvable on anionpy but not declared in __ion_state__
phantom        0   declared but missing
failing        1   declared with a failing differential test
untested       0   present but not credited
```

**1141 of 3051 tracked API items do not exist here at all**, and a further 335
resolve on `anionpy` but are not yet declared in `__ion_state__` (see the
ledger's own `undeclared` note for what that means). The 51.59% is the share
that exists, is declared, *and* passes a differential test against real NumPy.
One differential item is currently failing rather than suppressed: `divmod`'s
`.outer` method leaked the wrong ufunc name (`'floor_divide'`/`'remainder'`
instead of `'divmod'`) into one unsupported-dtype `TypeError`. The source fix
is committed; this measurement predates the rebuild that picks it up, and is
reported as-measured rather than adjusted by hand.

Of the 1574, **1520 match NumPy bit-for-bit on every case in the corpus**; 3
required a recorded ULP tolerance, 49 a recorded epsilon tolerance, and 2 an
exemption for floating-point tie-breaking (all four figures from
`tools/coverage.py --tests`, same run as the headline number above).

A previous edition of this section also reported a *declared*-tolerance count
(items carrying a tolerance whether or not any corpus case actually exercised
it) alongside the *exercised* count above, to make the gap between "might need
slack" and "needed slack this run" visible. `tools/coverage.py`'s current
output does not surface that breakdown directly, and re-deriving it precisely
enough to publish was out of scope for this pass — rather than restate stale
numbers under a fresh coverage figure, this section states only what was
directly re-measured. Re-adding that comparison is tracked, not silently
dropped.

Every tolerance is justified in-tree and backed by a seeded sweep of at least
20,000 samples. None of them exists to make a mismatch go away — see
`KNOWN-DIFFERENCES.md`, which currently records a defect of exactly that kind
found by auditing our own "exact" claims at the bit level rather than the value
level.

A coverage figure is only meaningful attached to a commit, because the compiled
extension is shared mutable state across concurrent working sessions. `DEVELOPING.md`
documents the hour this cost us and the practice that prevents it.

The denominator was audited and corrected *upward* partway through this project,
because the previous one flattered the number — it counted most classes as a
single item while exploding `ndarray` into 164. The percentage dropped sharply
when that was fixed. A coverage figure that only ever rises is not being audited.

### What is missing, in the aggregate

A percentage does not tell you which 57% is absent, so:

- **Missing dtype kinds:** object, void/structured, bytes (`S`), unicode (`U`),
  `datetime64`, `timedelta64`. Anything that depends on these is absent, which
  includes structured/record arrays and all date arithmetic.
- **`random` is bit-compatible only where explicitly declared.** `default_rng`
  and 23 `Generator` sampling methods reproduce NumPy's stream byte-for-byte,
  verified across seeds spanning 0 to 2**64-1 and sizes including empty and
  zero-containing shapes. Everything else — `RandomState` (the legacy API, 0%
  implemented), `MT19937`/`Philox`/`SFC64`, the discrete distributions
  (`poisson`, `geometric`, `zipf`, `logseries`, `triangular`, `vonmises`,
  `wald`), and the shuffle/choice/permutation family — is absent. The bit
  generators `PCG64`/`PCG64DXSM` produce the correct stream but are left
  undeclared because their `.state`/`.advance`/`.jumped` surface is missing.
- **`ma` (masked arrays)** is largely unimplemented — 214 of its items are still
  absent. What exists is the `MaskedArray` operator surface (arithmetic,
  comparison and bitwise dunders), the four `_MaskedUnaryOperation` /
  `_MaskedBinaryOperation` / domained families that back the module-level
  functions, and a mask-aware reduction/inspection surface (`filled`,
  `compressed`, `count`, `sum`/`any`/`all`/`min`/`max`/`argmin`/`argmax`/`ptp`,
  the shape attributes, the conversion dunders and copy protocol), and the
  modulo dunders.
- **`polynomial` is partial.** All six basis modules — the power basis plus
  Chebyshev, Legendre, Laguerre, Hermite and HermiteE — and all six
  `ABCPolyBase` subclasses are implemented and tested. Still absent: the N-D
  composition helpers (`polyval2d` and friends) and `polyutils`. Twelve
  functions that multiply polynomial series are **deliberately undeclared**,
  for two *separate* unresolved reasons that should not be confused:
  - `polymul`, `polypow`, `polyfromroots`, `chebmul`, `chebpow` and
    `chebfromroots` route through `np.convolve`, whose summation order we have
    not reproduced bit-for-bit.
  - `legmul`, `legpow`, `legfromroots` and the corresponding `Legendre.__mul__`
    /`__pow__`/`fromroots` touch no `convolve` at all — Legendre multiplication
    is a three-term recurrence — and diverge on complex128 for a cause not yet
    identified. They were briefly declared with a 1e-13 tolerance; that
    tolerance was removed and the declarations revoked instead.

  All twelve compute correct values to within an ULP or two. They are simply
  not claimed as exact, because that is what "exact" means here.
- **Pickling works for `ndarray` values** (`pickle.dumps`/`loads` round-trip
  shape, dtype, and data). The pickle format is anionpy-native, not
  numpy's `_reconstruct` stream.
- **`memoryview` works** for non-empty real/bool arrays and empty 1-D
  (typed PEP 688 export of a C-contiguous copy). Complex and empty N-D
  fall back to `__array__` for `np.asarray`; `memoryview` raises there.
  This is a copy, not a live alias of the internal buffer.
- **Array printing** does not reproduce NumPy's summarization rules.
- Assorted per-item behavioural divergences are catalogued in the development
  tree rather than hidden. Where `anionpy` knowingly differs from NumPy, it is
  written down.

### What does work

The implemented core is the part most code actually touches: array creation and
reshaping, dtypes and casting rules, indexing and slicing, broadcasting,
element-wise ufuncs with `out=`/`where=`/`casting=`/`dtype=`, the
`reduce`/`accumulate`/`reduceat`/`outer` ufunc methods, reductions and their
`nan*` variants, sorting and searching, FFT, and a `linalg` subset backed by a
system BLAS/LAPACK.

## Install

There is **no published release**. The version in `pyproject.toml` is a
placeholder. Build from source:

```sh
python -m venv .venv
./.venv/bin/pip install "maturin>=1.9,<2.0"
./.venv/bin/maturin develop --release
```

Requires a Rust toolchain and a working BLAS/LAPACK (Apple Accelerate on macOS).

```python
import anionpy as np

a = np.arange(12, dtype=np.float64).reshape(3, 4)
print(a.sum(axis=1))
```

## Compatibility notes

- `requires-python` is currently pinned high. That pin is environmental — it
  reflects the only interpreter this has been built and tested against, not a
  language feature the code needs. It will be widened once a lower interpreter
  can actually be tested, not before.
- Exception types: when NumPy is installed, `anionpy`'s exception classes
  subclass NumPy's, so `except numpy.linalg.LinAlgError:` catches `anionpy`
  errors. When NumPy is absent they subclass the appropriate builtins.
- `anionpy` never calls NumPy to *produce* a value or an error message. NumPy
  may hand over input bytes; it never supplies answers. This is what makes the
  differential tests meaningful — otherwise they would be comparing NumPy to
  itself.

## Contributing

The bar for declaring an item implemented is deliberately high, and is not
"the tests pass":

1. Write the differential corpus **first**.
2. Implement.
3. Verify **out of corpus** — on inputs the corpus does not contain.
4. Only then declare it in the coverage ledger.

A green suite is necessary, not sufficient. This project has already had a case
where 1300/1300 cases passed both before *and* after a real memory-layout defect,
because the corpus compared values and never checked strides. It has since had a
second: three functions declared *exact* were non-bit-exact on more than half of
a random sweep, and the corpus stayed green throughout because its particular
inputs happened to agree. Both defects were invisible at printed precision and
found only by comparing raw bytes.

"Exact" here is a claim about bit-identity. If you declare it, compare bits.

There is a third variant, and it is the easiest one to walk into: declining to
declare a primitive and then declaring something built out of it. `chebmul` was
correctly left undeclared and `chebfromroots` was declared in the same commit —
but `chebfromroots` *is* `chebmul` in a loop, so it inherited the defect and
amplified it with every additional root. If you decline an item, grep for its
callers before you declare them.

A fourth: **comparing values can never detect a wrongly-signed zero**, because
the value difference is exactly zero. Four functions were declared exact while
returning `+0.0` where NumPy returns `-0.0` — NumPy derives its padding zeros by
multiplying (`prd[0] = c[0] * 0`), so the sign of the input propagates, while we
filled a literal zero. Every corpus passed throughout. A signed zero is
observable through `signbit` and through `1/x`, so this is a real divergence and
not a cosmetic one. If you claim bit-exactness, compare bits.

A fifth: **a corpus that compares answers cannot see diagnostics.** Four
`ma.MaskedArray` division dunders were declared exact against a corpus that
checked values, masks, dtypes and fill values — all of which agreed — while
`anionpy` emitted `RuntimeWarning: divide by zero` where NumPy emitted nothing,
because NumPy's `ma` deliberately suppresses those inside `errstate` and we did
not. Note the shape of the mistake that hid it: the code carried a comment
arguing no suppression was needed *because `anionpy` never raises here*. That
premise was true and the conclusion drawn from it was false — it reasoned about
raising and said nothing about warning. Record why, and make sure the why is
actually load-bearing for the claim it supports.

The same class of defect surfaced again immediately afterwards, and the second
instance is the more instructive one: NumPy's signalling rule for
`remainder`/`mod`/`fmod` by zero is **dtype-conditional** — it raises the divide
flag for integer operands and stays silent for float, because the float result is
a defined NaN. Seven items declared exact warned on both. `divmod` had the mirror
defect, staying silent where NumPy signalled, on both kinds. If you are matching
diagnostics, match them per-dtype: a rule that is right for `int64` can be wrong
for `float64` in the same function.

A sixth, which is the same mistake three times over and is worth stating as a
rule rather than as three anecdotes: **a corpus is blind to any property its
inputs never vary.** Nine masked-array dunders were declared exact while
promoting `int8` operands to `int64`, and the corpus compared `.dtype` correctly
— but never passed an explicit `dtype=`, so every input landed on the
platform-default type, which is precisely the promotion target, the one place
where correct and broken are identical. Nine more items were declared exact while
getting empty-array strides wrong in both directions, and no corpus contained an
array of zero extent. A third sweep, varying only the *memory order* of the
input, found F-contiguous arrays coming back C-contiguous from a further set of
declared-exact operations. Comparing the right field is not enough if every input
sits on the fixed point. Vary the input's dtype, its extent, its layout and its
provenance — not just the keyword arguments.

And a seventh, the one no test can catch at all: **a fix in the Rust core that the
Python binding never calls.** One binding carried an inline duplicate of the
kernel it was supposed to delegate to, so a correct core fix was dead code and
the defect survived it. After fixing a core function, verify the binding
actually reaches it — `grep` the core symbol in `ionp-py/src/`.

Adding tolerance to make a mismatch disappear, or narrowing a corpus to exclude a
failing shape without recording why, both count as breaking the build. Where a
divergence is inherent to the algorithm — eigensolvers, least squares — a
tolerance is legitimate, but it must be measured by a seeded sweep and justified
in the same commit. Where it comes from an accumulation order we chose, the fix
is to match the order, and the honest fallback is to *revoke* the claim rather
than widen it.

## License

MIT. See `LICENSE`.
