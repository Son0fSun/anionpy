# Changelog

All notable changes to this project are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).
This project has **not** made a stable release and does not yet follow semantic
versioning; until 1.0 any release may change behaviour.

## HONEST STATUS — read this before reading anything below

This library is **not** a drop-in NumPy replacement today, and this file must
never be edited to imply otherwise.

As of 2026-08-07, measured by `tools/coverage.py --tests`:

```
COVERAGE: 23.035%   (712/3091)
absent      2379   not implemented
phantom        0   declared but missing
failing        0   declared with a failing differential test
untested       0   present but not credited
```

**2379 of 3091 tracked API items are not implemented at all.** The 23.035% is
the number of items that exist AND are backed by a passing differential test
against real NumPy. It is deliberately measured against a denominator that was
audited and corrected *upward* (see `DENOMINATOR-AUDIT.md`) — an earlier
denominator flattered the figure by counting most classes as one item each
while exploding `ndarray` into 164.

Known systemic gaps that a percentage does not convey:

- **Missing dtype kinds:** object, void/structured, bytes (`S`), unicode (`U`),
  `datetime64`, `timedelta64`.
- **`random` is not bit-compatible** with NumPy's generators except where
  explicitly declared.
- Assorted per-item divergences are recorded in `KNOWN-DIFFERENCES.md` rather
  than hidden.

## [Unreleased]

Nothing has been published to any package index. Version `0.1.0` in
`pyproject.toml` is a placeholder, not a release.

### Blocking before any publish

- **LICENSE copyright holder.** MIT was chosen, but the holder line still reads
  `<<COPYRIGHT-HOLDER-PENDING>>`. That is not the implementer's to invent — the
  IP is owned by a separate legal entity, and a LICENSE file is a public,
  irrevocable grant. `tools/check_public_export.py` fails while the placeholder
  is present; this is deliberate, not a lint.

Resolved since the last entry: the package name (now `anionpy`), the license
choice (MIT), and the README (written).

### Added

- `alignment`, `kind` and `shape` attributes on the dtype object.
- `can_cast` support for NumPy's empty-list / empty-dict void-dtype spelling.
- NumPy's dtype **string grammar**: single-char codes (`'f'`, `'i'`, `'?'`),
  sized codes (`'f4'`, `'i8'`, `'u1'`), byte-order-prefixed forms and C-name
  aliases (`'single'`, `'double'`, `'intp'`). Previously only 15 canonical
  spellings were accepted, out of at least 65 NumPy accepts.
- `ComplexWarning` on complex-to-real casts, matching NumPy's silent-data-loss
  diagnostic.
- `UFuncTypeError`, subclassing plain `TypeError`.
- **Floating-point error subsystem.** `errstate`, `seterr`, `geterr`,
  `seterrcall`, `geterrcall`, and the `RuntimeWarning` / `FloatingPointError`
  behaviour they gate. `np.errstate(divide='raise')` now raises here instead of
  silently returning a result. Backed by a 100-case differential corpus that
  measured 95 failures before implementation and 100 passes after — including
  the discrimination that NumPy's default `under` mode is `'ignore'`, not
  `'warn'`, so a uniformly-failing corpus would have been an instrument bug.

  **None of those five names is credited in the coverage ledger, deliberately.**
  The corpus tests the subsystem as one aggregate item, and out-of-corpus
  probing then found four error-path divergences it never exercised (a raw
  `AttributeError` where NumPy raises `TypeError` for a non-callable
  `seterrcall` registrant, plus three message-text differences), and that
  `geterrcall` has no test case at all — it appears only in the corpus
  docstring. The subsystem works; the evidence does not yet support declaring
  the individual items. Divergences are recorded in `KNOWN-DIFFERENCES.md`.
- **20 `dtype` introspection attributes** declared exact: `byteorder`, `char`,
  `num`, `str`, `hasobject`, `isalignedstruct`, `isbuiltin`, `isnative`, `ndim`,
  `subdtype`, `names`, `metadata`, `fields`, plus seven that were already
  implemented and had simply never been wired to a corpus (`name`, `itemsize`,
  `kind`, `shape`, `alignment`, `__repr__`, `__str__`). Verified out of corpus
  on **type identity**, not value alone — 280 attribute pairs across 14 dtypes,
  0 mismatches. `finfo`/`iinfo` attributes were re-measured in the same pass and
  **declined**: two previously-recorded blockers turned out to be already fixed
  on the current binary, but the `dtype('O')` message text still diverges and a
  new big-endian gap was found (NumPy's `finfo`/`iinfo` values do not depend on
  byte order; ours rejects `'>f8'`/`'>i4'` outright). Recorded in
  `KNOWN-DIFFERENCES.md`.
- `py.typed` marker — the package ships its inline types.
- `tools/check_public_export.py`: a read-only guard that fails if a file which
  would be published contains an internal identifier.

### Fixed

- **`.reduceat` returned a silently wrong answer on N-D input.** It and
  `.accumulate` were 1-D only; neither raised. This produced no error and no
  warning — the worst failure class there is.
- `ndarray.__array__` signature, which broke `numpy.asarray(arr, dtype=...)`
  interop.
- `conj` / `conjugate` now preserve `order='K'` layout.
- Several `power` / `float_power` edge cases: zero base with non-integer
  negative exponent, negative-base validation on empty and length-1 reductions.
- **`ma` ufuncs did not collapse 0-d results.** NumPy's masked unary and binary
  operations end with `if not result.ndim: return masked if m else result` — a
  0-d operand yields a bare scalar, or the `masked` singleton when the mask is
  set, never a `MaskedArray`. Ours returned a `MaskedArray` in both cases, and
  in the fully-masked case it still exposed the data it was supposed to hide.
  17 items restored to exact (`abs`, `absolute`, `add`, `arcsinh`, `arctan`,
  `arctan2`, `ceil`, `conjugate`, `cosh`, `exp`, `fabs`, `floor`, `multiply`,
  `negative`, `sin`, `sinh`, `subtract`); six more consumers of the same two
  factories were corrected as a side effect.
- `finfo` dtype-resolution and error-path divergences.
- `bitwise_count` `casting=` divergence.

### Changed

- The published package description no longer names internal components. It
  describes the outcome only.
- `numpy` moved from a hard dependency to an optional `compat` extra;
  `import anionpy` succeeds with no NumPy installed.

### Known-failing

21 differential items fail as of this writing, listed with measured causes in
`PATH-TO-100.md`. Roughly two-thirds are architecturally blocked (complex-power
ULP, `isnat` pending `datetime64`, `promote_types`/`result_type` pending a void
dtype, `min_scalar_type` pending object dtype) rather than merely unfinished.

This count fell from 26 without 26 bugs being fixed. Exactly one was a code fix;
four items stopped failing because their corpora were scoped to exclude a real,
still-unfixed float16 reduction-order divergence. That scoping is disclosed in
`KNOWN-DIFFERENCES.md`, and all four items remain WITHDRAWN in the ledger rather
than declared — coverage held flat across the change. A falling failure count
with a flat ledger is scoping; a falling count with a rising ledger would mean a
divergence had been hidden and paid for.
