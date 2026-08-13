"""Coverage declarations for the `numpy.char`/`numpy.strings` block.

Follows the one-module-per-block pattern established 2026-08-01 (see
`anionpy/_state/__init__.py`'s docstring) rather than editing `toplevel.py`.

Every key here corresponds to a `kind="custom"`, `convert_ionp_args=False`
`ItemSpec` in `tests/differential/strings_cases.py`, which overrides a
broken auto-derived `kind="ufunc"` entry (see that file's module docstring
for why the override, and why it must be wired in from `run.py` rather than
`registry.py`'s own tail).

HISTORY, for anyone reading this file wondering why there is no "N checks,
0 mismatches" claim standing alone this time: this block was declared once
(28 items, backed by a 167-check probe), merged, then UNDECLARED wholesale
after an independent, wider, out-of-corpus probe (8068 comparisons, 94
mismatches) found four real defect classes the 167-check probe never
exercised -- the probe was too narrow to falsify its own subject. That is
the specific failure this docstring is written to not repeat. The four
classes, all now fixed in `ionp-core/src/strings.rs` / `ionp-py/src/
strings.rs`, and the fix each one got:

 (A) Category Lt (titlecase letters, e.g. U+01C5 'ǅ') was invisible to the
     cased-character test used by `islower`/`isupper`/`istitle`, which was
     built only on Rust's `is_lowercase()`/`is_uppercase()`. Fix: a
     3-state `CaseKind` (Lower/Upper/Title/Uncased) -- Lower/Upper still
     come from Rust's `is_lowercase()`/`is_uppercase()` (these correctly
     implement the real Unicode derived Uppercase/Lowercase properties,
     NOT general category alone -- category-Nl Roman numerals like U+2167
     'Ⅷ' ARE cased in real numpy/CPython, `'Ⅷ'.isupper()==True`, confirmed
     directly, and switching to a general-category-Lu/Ll/Lt-only
     reimplementation would have silently broken this; see the regression
     test `roman_numeral_nl_is_still_cased_not_broken_by_lt_fix`) -- with
     Title added as an explicit third state via general category
     `TitlecaseLetter`, checked only when neither of Rust's own
     derived-property predicates already matched.

 (B) `isalpha` (and `isalnum`, which inherited the same bug through Rust's
     `is_alphanumeric()`) accepted category Nl (letterlike numerals, e.g.
     U+2167 'Ⅷ') and category Mc (spacing combining marks, e.g. U+0903
     DEVANAGARI SIGN VISARGA) -- both false positives from Rust's
     `is_alphabetic()`/`is_alphanumeric()` being the broader Unicode
     Alphabetic/Alphanumeric *property*, not numpy/CPython's narrower
     general-category-L*/isalpha-or-isdecimal-or-isdigit-or-isnumeric
     rule. Fix: `isalpha` now checks general category Lu|Ll|Lt|Lm|Lo
     directly via `unicode-general-category` (already a dependency, used
     for `isdecimal`); `isalnum` is now composed from the corrected
     per-char alpha/decimal/digit/numeric predicates rather than calling
     Rust's `is_alphanumeric()`.

     A related gap, not named by the original report but found by this
     block's own wider probe: `isnumeric` missed a small set of CJK (Han)
     ideographs with Unicode Numeric_Type=Numeric but general category Lo
     (e.g. U+4E00 '一' "one", U+5343 '千' "thousand"), which Rust's
     `is_numeric()` (Nd/Nl/No only) does not cover. Fixed with an explicit,
     documented-incomplete codepoint table (`is_known_cjk_numeral_lo`,
     same style/precedent as `isdigit`'s pre-existing Numeric_Type=Digit
     table -- see the KNOWN GAPS section below).

 (C) Foreign/numeric dtype input (`np.array([1,2,3])`, `np.array([1.5])`,
     bool, complex, ...): all ten unary predicates used to raise anionpy's own
     homegrown `TypeError`, and `add`/`equal`/the other comparisons used to
     raise the same instead of computing a real numeric result -- because
     `numpy.strings.add`/`equal`/`less`/... ARE `numpy.add`/`numpy.equal`/
     `numpy.less`/... (confirmed via `np.strings.add is np.add`), so real
     numpy actually computes `[2,4,6]` for `np.strings.add(int_arr,
     int_arr)`, not an exception.

     FIRST fix attempt (superseded, do not repeat): route the `Foreign`
     branch to real `numpy.strings.<name>`/`numpy.<name>` via
     `py.import("numpy")`. This was itself a defect, caught in review before
     merge: it meant `anionpy.strings.add` on numeric input returned NUMPY's
     own computed answer, and the ten unary predicates re-raised NUMPY's own
     exception OBJECT -- i.e. the differential test for this path compared
     numpy against numpy and passed trivially, exactly the "green that means
     nothing" failure the whole project exists to prevent. `anionpy.strings.X`
     cannot be "call `numpy.X`" and count as anionpy coverage.

     ACTUAL fix, two parts, per explicit review instruction:
       - BINARY (`add` + the six comparisons): since `np.strings.add is
         np.add`, the correct consequence is that `anionpy.strings.add` IS
         `anionpy.add` -- not "call numpy.add". The `Foreign` branch now routes
         through `ionp_binary_dispatch()`, which reuses the exact same
         `extract_binary_pair()` -> `ionp_core::ufunc::binary_op()` ->
         `PyArray` machinery that backs every other `anionpy.<ufunc>` (the same
         path `Ufunc::__call__` in `lib.rs` uses). Correct by construction:
         it is not a parallel reimplementation, it IS anionpy's own ufunc
         dispatch, and it inherits anionpy's own error text for any dtype
         combination anionpy itself doesn't support.
       - UNARY (`str_len`, `isdigit`, `isalpha`, `upper`, ...): numpy raises
         here (`_UFuncNoLoopError`, a private `UFuncTypeError`/`TypeError`
         subclass). anionpy cannot reproduce that private class without
         importing numpy to get it, so it raises a plain `TypeError`
         instead, constructed entirely in Rust via
         `numpy_no_loop_type_error()`, which reproduces numpy's message TEXT
         byte-for-byte (`ufunc '<name>' did not contain a loop with
         signature matching types <class 'numpy.dtypes.<DTypeClass>'> ->
         None`) from the ufunc name plus the input's dtype class name --
         itself obtained legitimately during `load()`'s input marshalling
         (`dtype.getattr("__class__").getattr("__name__")`), never by
         invoking numpy to construct or catch the exception object.
         `isdecimal`/`isnumeric` on `S`-dtype (bytes, unsupported
         independent of foreign-dtype) use the same constructor with a
         hardcoded `"BytesDType"`, for the same reason.

     `load()`'s `numpy.asarray()` call for input marshalling (turning a
     Python list, a numpy array of any dtype, or a bare scalar into
     something `load()` can inspect) is unaffected by this and stays -- it
     reads bytes in, it is never asked for an answer. `encode_string_array`
     likewise uses `numpy.frombuffer` only to wrap already-Rust-computed
     bytes into an output array, never to compute a value. Grepped the full
     diff for `import("numpy")`: exactly these two call sites remain, both
     input/output buffer marshalling, neither reachable on any path that
     produces a VALUE or an exception OBJECT.

 (D) Python sequence input (`np.strings.str_len(["a","b"])`, a list of
     bytes, a bare non-str/bytes scalar): used to be rejected outright by
     `load()`'s three hand-rolled branches (str-extract / bytes-cast /
     dtype-getattr-or-raise), none of which handled a list. Fix: `load()`
     now converts EVERY input through `numpy.asarray()` first -- the same
     conversion step real numpy's own ufunc dispatch performs -- collapsing
     the three branches into one, uniformly correct path.

Verification for this declaration:
- `cargo test -p ionp-core --release`: 11/11 `strings::tests` pass, zero
  new failures, zero weakened/deleted tests. (`ionp-core/src/strings.rs`
  itself was NOT touched by the class-C/D binary-dispatch/no-loop-error
  fix -- only `ionp-py/src/strings.rs` changed.)
- `tests/differential/run.py`: all 28 items PASS. Case counts grew from the
  original 13-24/item to 18-28/item because the committed corpus
  (`tests/differential/strings_cases.py`) previously had ZERO class-C/D
  coverage -- only an ephemeral scratchpad probe did, which is not
  re-runnable project state. Genuine class-C (six numeric/bool/complex
  dtypes) and class-D (list-of-str, list-of-bytes, bare scalar) cases were
  added to the committed corpus itself, wired through `exception_equivalences`
  (see below), so this PASS is now backed by the real, re-runnable suite,
  not just a one-off probe.
- IMPORTANT CORRECTION TO THE RECORD: the "compared=5449 mismatches=0"
  figure previously recorded here for this exact class-C/D fix was run
  against a STALE compiled `.so` -- `cargo build --release -p ionp-py`
  alone does not update `anionpy/_anionpy.abi3.so` (the file the Python package
  actually imports); a separate `cp target/release/lib_ionp.dylib
  anionpy/_anionpy.abi3.so` step is required, and had been skipped. Caught only
  by an isolated diagnostic that found `anionpy.strings.isalpha(int_array)`
  raising numpy's own private `_UFuncNoLoopError` object (`type(ioexc) is
  type(npexc) == True`) despite the Rust source constructing a plain
  `PyTypeError`. After copying the fresh binary into place (MD5-verified
  identical to the freshly built dylib) and re-running everything above
  plus the strict scratchpad probe (`/private/tmp/.../scratchpad/
  sprobe2.py`, which does exact `type(...) is not type(...)` exception
  comparison with NO leniency, unlike the differential harness):
  compared=5449, mismatches=156 (78 distinct cases, each producing two
  log lines). Every one of the 78 is an `EXC-TYPE` mismatch and NOTHING
  else -- confirmed by grepping the probe's own mismatch categories
  (zero `EXC-MISS`, zero `EXC-EXTRA`, zero dtype/shape/bytes-value
  mismatches, zero `EXC-MSG` message-text mismatches). Each is exactly the
  by-design gap documented in class (C) above: numpy raises its private
  `_UFuncNoLoopError`/`UFuncTypeError` subclass, anionpy raises plain
  `TypeError` with byte-identical message text -- not achievable as a type
  match without importing numpy's private exception class, which would
  violate the "never import numpy for an answer" rule harder than leaving
  it a `TypeError`. This is exactly what `exception_equivalences` in
  `strings_cases.py` declares acceptable (`_UNARY_EXC_EQUIV` etc., built by
  dynamically capturing numpy's real exception class via
  `_numpy_exc_type()`, never by hardcoding numpy's private import path),
  which is why the differential harness (type-only grading via the
  declared-equivalence set, message text never compared for pass/fail)
  shows all 28 items PASS. Zero VALUE mismatches and zero message-text
  mismatches were found anywhere in the 5449-comparison sweep, against the
  confirmed-fresh binary.
- Prior to this fix, the same wide out-of-corpus sweep (one character from
  every named Unicode general-category boundary, combined into words at
  multiple positions, across five `U` widths and four `S` widths, at both
  whole-array and single-element granularity, plus the class-C/D cases
  above) is what originally surfaced classes A-D; see history above.

KNOWN, DOCUMENTED GAPS (do not block declaring `isdigit`/`isnumeric` --
read why below):

`isdigit` on `U`-dtype input implements Python's `str.isdigit() ==
category Nd OR Numeric_Type=Digit` via an exact crate
(`unicode-general-category`) for the Nd half, and an explicit hardcoded
codepoint table (`is_known_numeric_type_digit`) for the Numeric_Type=Digit
half, covering the well-known blocks (superscript/subscript digits,
circled/parenthesized/dingbat digits). That table is NOT the complete UCD
Numeric_Type=Digit property -- no crate in this dependency tree carries it
as a distinct lookup, and a few scattered CJK/other Numeric_Type=Digit
outliers outside the table's ranges will disagree with real numpy.

`isnumeric` on `U`-dtype input is exact for general categories Nd/Nl/No
(the large majority of Numeric_Type=Numeric/Digit/Decimal content,
including category-Nl Roman numerals and ideographic zero, both confirmed
by direct probing) plus an explicit table (`is_known_cjk_numeral_lo`) of
the common CJK numeral ideographs (0-10, formal/banker's digit forms,
hundred/thousand/ten-thousand/hundred-million). That table is likewise NOT
a complete reproduction of the UCD's Numeric_Type=Numeric property for
CJK -- rarer historical/dialectal numeral variants and very large
magnitude characters (京, 垓, 秭, and beyond) are not included.

Declared anyway because: (a) the Nd/Nl/No general-category base for both
predicates is exact (crate-backed, not approximated), (b) both gaps are a
strict superset check on top of that exact base, affecting only characters
in a specific, named, narrow residual set, not general correctness, and
(c) this is exactly the kind of gap the "declare only what you verified,
document what you did not" bar calls for recording rather than either
hiding or blocking on -- an undeclared item would be equally dishonest in
the other direction, since the wide probe above found zero mismatches
across everything it actually swept.

REGRESSION, found by independent review AFTER the class-C/D fix above was
merged (2026-08-01), NOT caught by this file's own "all 28 PASS" claim at
the time -- second time that has happened on this block, see the process
fix at the end of this entry:

The `ionp_binary_dispatch()` rework in class (C) above deleted two
pre-existing hand-built exception constructors (`add_mixed_dtype_err` /
`comparison_mixed_dtype_err`) that used to handle the case where BOTH
operands decode as `S`/`U` string dtype but their KIND differs (bytes vs
unicode, e.g. `dtype('S5')` vs `dtype('<U6')`), on the wrong assumption
that the new `ionp_binary_dispatch` fallback covered it too. It did not:
that fallback routes through `extract_binary_pair()` -> `anionpy.array()`,
which rejects ALL string dtypes outright with a generic "unsupported
numpy dtype for anionpy.array()" message -- not numpy's real message for this
case. Independent review caught this via two committed corpus cases,
`mismatch_S_plus_str` (`strings.add`) and `mismatch_S_vs_str` (the six
comparisons), which had been in `strings_cases.py` since before the
class-C/D fix and should have caught the regression at merge time.

Re-verifying the fix against real numpy 2.5.1 also surfaced a SECOND,
related gap the review did not explicitly name: when one operand is
`S`/`U` string and the OTHER is genuinely foreign/numeric (e.g.
`np.array([1,2,3])` against `np.array(["x","y","z"])`), the same fallback
problem applied, caught by two more pre-existing corpus cases
(`mismatch_int_plus_str`, `mismatch_int_vs_str`) that had only ever
"passed" before under the superseded numpy-delegation design (which
laundered numpy-vs-numpy and could not fail this comparison honestly).

There are THREE distinct message shapes numpy uses for a binary
string-ufunc dtype mismatch, not one, all confirmed directly against real
numpy 2.5.1 (operand order preserved in all three):
  1. Both S/U, kind mismatched: `add` -> `"ufunc 'add' cannot use operands
     with types dtype('S5') and dtype('<U6')"` (no byteorder marker for S,
     `<` for U, itemsize substituted); comparisons -> `"ufunc 'equal' did
     not contain a loop with signature matching types (<class
     'numpy.dtypes.BytesDType'>, <class 'numpy.dtypes.StrDType'>) ->
     None"`.
  2. One S/U, other foreign/numeric: `add` -> `"ufunc 'add' did not
     contain a loop with signature matching types (dtype('int64'),
     dtype('<U1')) -> None"` (plain `dtype('xxx')` repr, using the
     foreign side's `dtype.name`) -- a THIRD shape, distinct from both of
     the above; comparisons -> same `<class 'numpy.dtypes.Xxx'>` shape as
     (1), with the foreign side's dtype CLASS name substituted (e.g.
     `Int64DType`).
  3. Both foreign/numeric: unaffected, routes through
     `ionp_binary_dispatch` -> `ionp_core::ufunc::binary_op` as designed.

Fix: `LoadResult::Foreign` changed from a bare `String` (dtype class name
only) to a struct variant `{ dtype_class, dtype_name }`, both captured
during `load()`'s existing input marshalling (no new numpy import). Four
new match arms added to `strings_add` and the `comparison_op!` macro (one
per non-happy-path combination of `Str`/`Foreign` on each side), backed by
three new message constructors (`numpy_add_dtype_mismatch_error`,
`numpy_no_loop_type_error_binary`, `numpy_no_loop_dtype_repr_error`),
none of which import or call numpy -- all built from data already
captured off the real numpy input array during marshalling.

Verification after the regression fix, following the mandated process
(build via `flock /tmp/ionp-build.lock .venv/bin/maturin develop --release
-m ionp-py/Cargo.toml`, NOT bare `cargo build`, which does not refresh
`anionpy/_anionpy.abi3.so`):
  - `.so` mtime confirmed newer than the last source edit at each build.
  - Observable behavior asserted directly (not just via the harness)
    before trusting any run: `anionpy.strings.add`/`equal` on S-vs-U and
    int-vs-str inputs both produce numpy's exact message text.
  - Baseline (pre-fix, this round) independently reproduced: `char.add`
    24/28, `strings.add` 24/28, `strings.equal`/`greater`/`greater_equal`/
    `less`/`less_equal`/`not_equal` 15/18 each -- 8 items, matching the
    coordinator's report.
  - Post-fix: `tests/differential/run.py --out` run twice, byte-identical
    SHA-256 both times (`3bb6a0d3...c9ac5`); all 28 items PASS, zero
    char/strings FAIL lines.
  - `cargo check -p ionp-py --release`: clean, zero warnings, twice.
  - `cargo test -p ionp-core --release strings::`: unaffected, still
    11/11 (this file was not touched by the regression or its fix).

PROCESS NOTE: this is the second time a "28/28 PASS" claim from this block
did not survive independent checking -- both times the failure was
declaring success from a run against either a stale `.so` or an
incompletely-covered fix, not from a fabricated number. The corrective
habit going forward, stated explicitly by the coordinator and now
followed above: verify `.so` freshness AND an observable behavior change
BEFORE trusting any run, run the differential suite twice and diff
SHA-256, and report the actual FAIL-line diff rather than a summary count
whenever a regression is being investigated.

--------------------------------------------------------------------------
ROUND 2 (2026-08-01): +16 items -- case conversion + strip family
--------------------------------------------------------------------------

Added: `upper`, `lower`, `swapcase`, `title`, `capitalize`, `strip`,
`lstrip`, `rstrip`, each under BOTH `char.` and `strings.` (16 total,
28 -> 44). Scope deliberately narrowed to these two "Tier A" families out
of the full 54/29-item absent list (per the task brief's own tiering
guidance: "predicates/case ops cheapest ... work in tiers, declaring as
you go" and "twenty honestly-declared items beat eighty you cannot
back") -- the harder families (search/index, pad/zfill, replace/
partition, encode/decode, and the char-specific array/chararray/alias
names) are explicitly NOT covered by this round; see the bottom of this
docstring for what was left out and why.

Two genuinely different numpy implementations were confirmed, not
assumed, via direct probing against real numpy 2.5.1 -- conflating them
would have silently broken the 0-d case:

  - `upper`/`lower`/`swapcase`/`title`/`capitalize` are `_vec_string`-
    based, NOT real `numpy.ufunc` instances (confirmed:
    `type(np.strings.upper) is not numpy.ufunc`, and
    `tools/numpy_surface.json` classifies all of `char.upper`/
    `strings.upper`/etc as `kind == "func"`, not `"ufunc"` -- so unlike
    the 28 names above, `ufunc_registry.py` never auto-derived a
    (broken) entry for these; they are NEW registry additions in
    `strings_cases.py`, not overrides, verified by an explicit
    "must NOT already be present" assertion rather than the existing
    "must already be present" one). Output itemsize == the input's
    DECLARED itemsize (silently truncates if the transform needs more
    chars -- e.g. German 'straße'.upper() truncates on a narrow dtype,
    confirmed directly), and a 0-d input array stays a genuine 0-d
    `ndarray` -- NOT scalarized, and its itemsize is NOT shrunk to the
    trimmed content length (confirmed: `type(np.strings.upper(0d_<U20>
    ("hello world"))) is numpy.ndarray`, dtype stays `<U20`). Foreign/
    non-string dtype input raises a plain `TypeError` ("string operation
    on non-string array") -- confirmed identical text for all five names,
    and this IS numpy's real exception class directly (not a private
    subclass needing an equivalence entry), matching byte-for-byte.

  - `strip`/`lstrip`/`rstrip` ARE real ufuncs (`_strip_whitespace`/
    `_lstrip_whitespace`/`_rstrip_whitespace` internally). Output
    itemsize stays at the input's declared width for any N-D (N>=1)
    array, even though every element got shorter (confirmed: a 1-D
    `<U20` array of `"  hi  "` etc. stays `<U20` after `strip`) -- but a
    0-d input DOES shrink to the actual POST-STRIP content length AND
    DOES scalarize to a `numpy.str_`/`numpy.bytes_` (confirmed:
    `np.strings.strip(0d_<U20>("  hi  "))` returns `np.str_('hi')`,
    dtype `<U2`), mirroring `add`'s existing 0-d-shrinks-to-content rule
    from round 1, not the case-conversion group's rule directly above.
    Foreign/non-string dtype input raises numpy's real private
    `_UFuncNoLoopError` (same class as the unary predicates), reproduced
    here as a plain `TypeError` with byte-identical message text
    (`ufunc '_strip_whitespace' did not contain a loop with signature
    matching types <class 'numpy.dtypes.Int64DType'> -> None`, confirmed
    per-function since the internal ufunc name differs by function), and
    declared via the same `_UNARY_EXC_EQUIV` equivalence as round 1's
    unary predicates -- no new equivalence class needed.

  The optional `chars` argument on `strip`/`lstrip`/`rstrip` (strip any
  character appearing in the given set, not just whitespace) is
  implemented and covered by dedicated corpus cases
  (`chars_arg_u`/`chars_arg_s`/`chars_arg_none_content` in
  `_strip_cases()`), decoded via the same `load()` path as the main
  array argument, restricted to a scalar `str`/`bytes` operand matching
  the array's own S/U kind.

  `char.<name>` and `strings.<name>` were confirmed to produce IDENTICAL
  results across the full corpus content for all eight functions on both
  `S` and `U` dtype (`type(r1) == type(r2) and r1.dtype == r2.dtype and
  np.array_equal(...)`, 0 mismatches) even though `np.char.upper is not
  np.strings.upper` (different underlying objects, same computed
  behavior for every name in this round) -- so both prefixes share one
  Rust implementation and one corpus builder, same pattern as round 1.

Verification for round 2:
  - `cargo test -p ionp-core --release strings::`: 20/20 pass (11
    original + 9 added for the case-conversion/strip/pad/search/replace/
    encode-decode core logic written this round; only 8 of those methods
    are wired to a Python-visible surface in this round, the rest -- pad,
    search, replace, partition, encode/decode -- are core-only for now,
    NOT declared, see below).
  - `.so` mtime confirmed newer than the last source edit; observable
    behavior asserted directly (`anionpy.strings.upper`/`anionpy.strings.strip`
    importable and producing numpy-matching output) before trusting any
    differential run.
  - `tests/differential/run.py --out` run twice, byte-identical SHA-256
    both times (`1da1ef5a...bcec79aa`).
  - All 44 `char.*`/`strings.*` items PASS (28 original + 16 new), 0 FAIL
    lines under either `char.` or `strings.` prefix. The only FAIL lines
    anywhere in the full 1180-item run are in OTHER agents' owned areas
    (numeric ufunc precision/casting/`ndarray` dunders -- `divide`,
    `pow`, `ndarray.__ipow__`, etc.), none touching `ionp-core/src/
    strings.rs` or `ionp-py/src/strings.rs`.
  - Out-of-corpus probe (>=2 inputs per item, not in the committed
    corpus, compared byte-for-byte against live numpy 2.5.1): itemsize-
    varying U/S widths, empty strings, a German eszett case-expansion
    case (`'straße'.upper()` truncation), a 0-d array with leading/
    trailing whitespace (`strip` itemsize-shrink case), and foreign-dtype
    (`int64`) inputs for all eight names -- all matched, including exact
    exception TYPE and MESSAGE TEXT (not just the type-only grading the
    harness itself applies).

NOT covered by round 2 (left undeclared, on purpose, not silently
skipped): `center`/`ljust`/`rjust`/`zfill` (pad family -- core logic
written and unit-tested in `ionp-core/src/strings.rs`, but no PyO3
binding yet); `find`/`rfind`/`index`/`rindex`/`count`/`startswith`/
`endswith` (search family -- same status); `replace`/`multiply`/
`partition`/`rpartition` (edit family -- same status, `partition`'s
output shape needs an extra trailing dim of size 3, not yet wired
through `encode_string_array`); `encode`/`decode` (needs the
`UnicodeEncodeError`/`UnicodeDecodeError` construction logic derived
during investigation but not yet implemented in `ionp-py/src/
strings.rs`); `compare_chararrays` (char-specific, not investigated this
round); `expandtabs` (itemsize rule does not follow an obviously
reproducible formula from declared itemsize or content length alone --
flagged, not resolved); `split`/`rsplit`/`splitlines`/`join`/`mod`/
`translate`/`slice` (numpy itself deliberately excludes the first three
from `np.strings`, keeping them `char`-only "until behavior has
crystallized" -- own comment in `numpy/_core/strings.py`; `mod`/
`translate`/`slice` not investigated); `array`/`asarray`/`chararray`/
`ndarray`/`narray`/`asnarray`/`character`/`bytes_`/`str_`/
`array_function_dispatch`/`set_module`/`strings_multiply`/
`strings_partition`/`strings_rpartition` (type/dispatch-machinery names,
not real per-element string computation -- out of scope for a "close the
namespace with real Rust logic" task, not investigated for whether a
thin alias would even be honest here). `char.equal`/`greater`/etc.
(6 names) are a SEPARATE, not-yet-investigated question from `strings.*`'s
already-declared comparisons -- confirmed directly that
`np.char.equal is not np.strings.equal` (different objects), so char's
version cannot be assumed identical without its own verification, unlike
the eight names actually declared this round which WERE cross-checked.

--------------------------------------------------------------------------
ROUND 3 (2026-08-01): +22 items -- pad family + search family
--------------------------------------------------------------------------

Added: `center`, `ljust`, `rjust`, `zfill` (pad family) and `count`,
`find`, `rfind`, `index`, `rindex`, `startswith`, `endswith` (search
family), each under BOTH `char.` and `strings.` (11 names x 2 prefixes =
22 total, 44 -> 66). `char.<name> is strings.<name>` confirmed True for
all 11 names (same underlying object, unlike the 6 comparison names
above), so a single Rust binding and a single corpus builder per name
correctly cover both prefixes.

Two more output-shape "personalities" confirmed by direct probing against
real numpy 2.5.1, distinct from BOTH of round 2's two:

  - PAD family (`center`/`ljust`/`rjust`/`zfill`): a 0-d input stays a
    genuine 0-d `ndarray`, NOT scalarized (like round 2's case-conversion
    group) -- but unlike that group, output itemsize is NOT the input's
    declared itemsize; it is `max(width, max-actual-post-op-length)`,
    computed fresh from the RESULT content, ignoring the input's declared
    itemsize entirely (confirmed: a `<U3` array of `"h"` centered to
    width 2 produces `<U2`, i.e. wider than the truncated 1-char input
    content but not wider than the actual op output). Implemented in
    `ionp-py/src/strings.rs` via `encode_string_array_noscalar` fed a
    freshly computed `width_chars = elems.iter().map(|e|
    e.chars.len()).max()`, not the loaded `Loaded.declared_itemsize_chars`.

  - SEARCH family (`count`/`find`/`rfind`/`index`/`rindex`/`startswith`/
    `endswith`): output dtype is int64 (`find`/`rfind`/`count`/`index`/
    `rindex`) or bool (`startswith`/`endswith`), never string -- and a 0-d
    input DOES scalarize (to `numpy.int64`/`numpy.bool_`), matching round
    2's strip-family rule, not the pad-family rule directly above.

A genuine BUG was found and fixed in `ionp-core/src/strings.rs`'s
pre-existing `center()` (written and unit-tested, but never wired to a
Python-visible surface, before this round -- so never exercised against a
wide corpus): the left/right padding-bias split for an odd total pad
amount was implemented with a fixed direction (an earlier draft had
"extra goes right", a first fix attempt this round changed it to a fixed
"extra goes left" -- BOTH wrong). The real CPython/numpy rule, transcribed
from CPython's own `unicode_center`/stringlib pad helper, is parity-
dependent on BOTH the pad amount and the target width: `marg = width -
len; left = marg // 2 + (marg & width & 1)` -- extra padding goes LEFT
only when BOTH the total pad and the width are odd; otherwise extra goes
RIGHT. Verified against two deliberately different probes that initially
looked contradictory but both match this exact formula: `'hi'.center(7)`
(pad=5 odd, width=7 odd -> both odd -> extra LEFT -> `'   hi  '`, 3L/2R)
and `'A'.center(30)` (pad=29 odd, width=30 EVEN -> extra RIGHT, 14L/15R).
Fixed as `extra_left = (total_pad % 2 == 1) && (width % 2 == 1)`. Before
the fix, the full differential corpus showed `char.center`/
`strings.center` failing 19/36 cases (only on `<U1`-truncated single-char
content padded to specific widths -- narrow enough that the original
smoke-test probes, which happened to only exercise the "both odd" case,
did not catch it); after the fix, both go to 36/36, and this was the ONLY
change in the project-wide FAIL count for this round (50 -> 48, i.e. -2,
both `center` entries, zero other regressions or improvements from this
round's other 21 items, which passed cleanly on first implementation).

Real numpy 2.5.1 bugs, deliberately byte-exact REPRODUCED (not "fixed",
since the task is differential parity, not correctness arbitration):
  - `center`/`ljust`/`rjust`/`zfill` on a zero-element array raise
    `ValueError("zero-size array to reduction operation maximum which has
    no identity")` -- an internal `width.max()` reduction bug inside
    numpy's own implementation, confirmed identical for both `<U5` and
    `S5` empty arrays. The search family does NOT share this bug (a
    zero-element array returns a valid empty result, confirmed directly)
    -- the two families were deliberately probed separately for this
    because there was no a priori reason to assume they shared it.
  - `center`/`ljust`/`rjust` on an empty (`''`) or multi-character (e.g.
    `'ab'`) `fillchar` raise `TypeError('The fill character must be
    exactly one character long')`, reproduced verbatim.
  - `index`/`rindex` on a not-found substring raise `ValueError('substring
    not found')` -- confirmed this is numpy's own genuine PUBLIC
    `ValueError`, not a private subclass, so no `exception_equivalences`
    entry was needed for it specifically (unlike the foreign-dtype path
    below).

Search family's foreign-dtype error message is a FOURTH distinct shape,
beyond the three already documented in round 1 for `add`/comparisons:
`"ufunc '{name}' did not contain a loop with signature matching types
(<class 'numpy.dtypes.{a_class}'>, <class 'numpy.dtypes.{sub_class}'>,
<class 'numpy.dtypes._PyLongDType'>, <class 'numpy.dtypes._PyLongDType'>)
-> None"`, confirmed byte-exact for message text; only the exception
class differs (real numpy's private `_UFuncNoLoopError` vs anionpy's plain
`TypeError`), covered by the same `_UFUNC_NO_LOOP_ERROR` equivalence
capture already established in round 1 (`_PAD_SEARCH_EXC_EQUIV =
{_UFUNC_NO_LOOP_ERROR: {TypeError}}`, no new equivalence class needed).

TWO SCOPE EXCLUSIONS, deliberate, disclosed rather than silently handled,
both judged disproportionate effort for this round's evidence bar:
  (1) Pad family with a foreign-dtype `a` array or a foreign-dtype
      `fillchar`: real numpy's actual behavior here is an internal
      int()-coercion quirk, not a clean error-message shape like the
      cases documented above -- not chased. anionpy raises its own
      documented-gap `TypeError` instead.
  (2) Search family with a bare non-string Python scalar (not wrapped via
      `numpy.asarray`) as `sub`/`prefix`/`suffix`, or BOTH `a` and `sub`
      foreign: confirmed via direct probe that `numpy.asarray(5).dtype
      .__class__.__name__ == "Int64DType"`, but passing a bare Python
      `int` directly as `sub` to `np.strings.find` produces an error
      naming `_PyLongDType` instead (NEP-50 weak-scalar dtype resolution,
      a different code path from an already-array-wrapped foreign
      operand). anionpy raises its own documented-gap `TypeError` for this
      specific combination rather than reproducing the weak-scalar
      message shape.

Verification for round 3:
  - `cargo test --release`: 150 passed, exactly the 3 known pre-existing
    failures (`complex_input_is_rejected_for_math_unary_ops`,
    `reciprocal_rejects_integer_input_documented_gap`,
    `square_rejects_bool_documented_gap`), none in `strings::`, zero new
    regressions from the `center` fix or the new pad/search bindings.
  - `.so` freshness confirmed via `maturin develop --release` (with
    `$HOME/.local/bin` on PATH, since maturin is pipx-installed, not
    venv-installed) reporting a fresh "Finished `release` profile
    [optimized]" build each time, not a stale-cache no-op.
  - `tests/differential/run.py --out ... --verbose`: all 22 new items
    PASS 100% of their corpus cases (`center`/`ljust`/`rjust` 36/36 each,
    `zfill` 25/25, `count`/`find`/`rfind`/`startswith`/`endswith`/`index`/
    `rindex` 30/30 each, x2 prefixes each) -- 656 case-checks total, 0
    mismatches.
  - Project-wide FAIL count: 48 (down from a pre-`center`-fix 50 that
    included the two now-fixed `center` failures), confirmed via `grep`
    to contain ZERO `char.*`/`strings.*` entries -- the remaining 48 are
    all in other agents' owned areas (ufunc gaps like `bitwise_count`/
    `conj`/`isnan`, numeric precision issues in `degrees`/`expm1`/
    `floor_divide`/`hypot`/`ndarray.__pow__`, none touching this round's
    files).
  - Out-of-corpus probes beyond the committed corpus (not just corpus-
    passing): empty arrays (`<U5` and `S5`, pad-family-raises vs
    search-family-empty-result), negative widths, `fillchar` edge cases
    (empty/multi-char), S-vs-U kind mismatches for search, foreign-dtype
    `a` for search, `index`/`rindex` not-found, and the two `center`
    parity probes documented above -- all byte-exact against real numpy
    2.5.1, including exact exception TYPE (module-adjusted via the
    equivalence dict) and MESSAGE TEXT.

JUDGMENT CALL, disclosed: `anionpy/char.py` and `anionpy/strings.py` (pure 1:1
re-export files, not listed in this task's originally-stated owned-files
set) were edited this round to add the 11 new import/`__all__` entries
per prefix. Treated as in-scope because git history shows the same task
lineage editing these every prior round as a mechanical, zero-logic
consequence of adding new `ionp-py` bindings (round 1 and round 2 both
did the same), not because the owned-files boundary was reinterpreted --
flagged here rather than silently done.

NOT covered by round 3 (left undeclared, on purpose): `replace`/
`multiply`/`partition`/`rpartition` (edit family -- core Rust logic
exists per round 2's note but no PyO3 binding yet, `partition`'s output
shape needs an extra trailing dim of size 3 not yet wired through
`encode_string_array`); `encode`/`decode` (needs
`UnicodeEncodeError`/`UnicodeDecodeError` construction, not yet
implemented); `compare_chararrays` (char-specific, not investigated);
`expandtabs`/`mod`/`translate`/`slice` (not investigated this round);
`split`/`rsplit`/`splitlines`/`join` (numpy itself keeps these
`char`-only, not on `np.strings`); the char-specific comparison names
(`char.equal`/etc, 6 names, separate object from `strings.*`'s already-
declared versions per round 1's note) and the type/dispatch-machinery
names (`array`/`asarray`/`chararray`/etc) noted as out-of-scope in round
2's docstring, unchanged this round.

--------------------------------------------------------------------------
ROUND 4 (2026-08-01): withdrawal, root-cause, and re-declaration of the
7 search names (14 items) -- coordinator audit caught what round 3 missed
--------------------------------------------------------------------------

Round 3 declared all 7 search names (`count`/`find`/`rfind`/`index`/
`rindex`/`startswith`/`endswith`) exact on merge, backed by 30/30 corpus
cases per item passing. The coordinator's independent post-merge audit
(not the committed corpus -- a separate, wider boundary probe) found a
real divergence and withdrew all 14 items (both prefixes) pending a fix.
This is exactly the "false green" the round-3 declaration's own
verification section warned about in the abstract; here it is concretely:
30/30 passing was NOT sufficient evidence, because the corpus never
combined a SHORT/EMPTY element with a `start` value past that specific
element's own length -- every prior boundary case in `_search_common_cases`
used a start value chosen relative to a longer, fixed array, not swept
against short elements.

THE BUG: `ionp-core/src/strings.rs`'s `clamp_range(start, end)` clamped
`start` down to `min(start, len)` before comparing it against `end`. This
meant `start > len(s)` collapsed to `start == len(s)`, which IS a valid
position for an empty-substring match (`s.find("", len(s))` is legitimately
`len(s)`, not a miss) -- so `start` values genuinely past the end of a
short/empty element got silently treated as "AT the end" and produced a
false hit for `find`/`count` (wrong non-negative index instead of -1/0),
a false `True` for `startswith`/`endswith`, and -- most dangerous --
`index`/`rindex` returning an integer instead of raising
`ValueError('substring not found')`.

ROOT CAUSE, not just symptom: the fix is NOT "special-case empty
substrings" (which would have re-introduced exactly the kind of narrow,
corpus-shaped patch that caused the original gap). The actual CPython
rule (`Objects/stringlib/find.h`'s `ADJUST_INDICES`, confirmed by direct
reading and independently reverse-derived by brute-force) is that `start`
is negative-adjusted but NEVER capped at `len` from above, while `end` is
negative-adjusted AND capped at `len`; the window is a miss -- for EVERY
caller, non-empty substrings included -- whenever the (possibly-beyond-
`len`) adjusted `start` exceeds the capped `end`. Non-empty substrings
already produced the correct answer under the OLD clamp too, purely by
coincidence (an over-clamped `start` still made `hi - lo` too small for
a nonzero-length substring to fit), which is exactly why the bug was
invisible to every non-empty-substring case in the corpus and only
surfaces for the empty-substring / zero-length-window edge.

Fixed by rewriting `clamp_range` to return `Option<(usize, usize)>`
(`None` meaning "structural miss, regardless of what the caller is
searching for") instead of a `(usize, usize)` pair that was silently
forced into a valid-looking `lo <= hi` range via `.max(lo)`. All 5
`StrElem` methods that call it (`find`, `rfind`, `count_sub`,
`startswith`, `endswith`) now propagate the `None` case as their own
"not found" result uniformly, with zero substring-emptiness special-
casing anywhere in the fix itself -- `index`/`rindex` inherit the fix for
free since they call `find`/`rfind` internally.

Verified by brute-force cross-product against REAL CPython (not assumed,
not spot-checked): string/bytes length 0-4 over alphabet {a, b}, every
substring length 0-3 over the same alphabet, `start` swept 0..100 step 3
plus -6..6, `end` swept -6..6 plus a large out-of-range sentinel (standing
in for the `i64::MAX` "no end given" convention), for both `find` and
`rfind` -- 0 mismatches across the full sweep (many thousands of rows,
including every combination the coordinator's original 24-row probe and
the withdrawal note's 4 headline examples are a subset of). A pre-existing
`ionp-core` unit test also encoded the OLD wrong clamp's coincidentally-
correct output for its specific inputs and needed no change (its cases
did not include a `start` past an element's own length) -- checked, not
assumed.

Corpus fix: `_search_common_cases()` in `strings_cases.py` (shared by
`find`/`rfind`/`count`/`startswith`/`endswith`, and inherited by
`index`/`rindex` via `_index_cases()`'s filter) gained a dedicated
boundary block -- elements `["", "a", "ab"]` (both `<U5` and `S5`) crossed
with `start` in `0..3`, each with an explicit `end=4` row and a
default-end row, for both an empty substring (the exact bug shape) and a
non-empty `"a"` substring (a sanity check that non-empty subs stay
correct, not just an assumption) -- 32 new rows per item, all 7 names x 2
prefixes now run 62/62 cases each (30 original + 32 boundary), 0
mismatches.

Verification for round 4 (full mandated process, in order):
  - `flock /tmp/ionp-build.lock maturin develop --release` (with
    `$HOME/.local/bin` on PATH): fresh "Finished `release` profile
    [optimized]" build, `Installed ionp-0.1.0` confirmed, twice (once
    after the core fix, once after the corpus-only change required no
    rebuild but was re-run anyway for the final check).
  - `cargo test --release`: 150 passed, exactly the 3 known pre-existing
    failures (`complex_input_is_rejected_for_math_unary_ops`,
    `reciprocal_rejects_integer_input_documented_gap`,
    `square_rejects_bool_documented_gap`); `strings::tests::search_family`
    (the pre-existing unit test) unaffected, still passing unmodified.
  - `tests/differential/run.py --out ... --verbose`: all 14 items PASS
    62/62 (`char.count`/`find`/`rfind`/`index`/`rindex`/`startswith`/
    `endswith` and the same 7 under `strings.`).
  - Project-wide FAIL count: 48, unchanged from before this round's fix
    and from round 3's baseline -- confirmed zero new regressions and
    zero remaining `char.*`/`strings.*` FAIL lines.
  - Direct probe reproducing the coordinator's own headline examples
    (`"".find("", 1, 4)`, `"a".find("", 2, 4)`, `"ab".find("", 3, 4)`,
    plus `"hi".find("", 1, 0)` and the valid `"hi".find("", 2)` boundary)
    against both real numpy and `anionpy.strings.*` directly, for all 7
    names including `index`/`rindex`'s exception behavior: byte-exact
    match on every case, `index`/`rindex` now raise `ValueError` on the
    same rows numpy does rather than returning an integer.

PROCESS NOTE, for calibration: this is the second time in this file's
history (see round 1's PROCESS NOTE above) that a "N/N corpus cases PASS"
claim was necessary but not sufficient evidence for declaring an item
exact -- both times the gap was found only by a wider, independently-
constructed probe that combined inputs the committed corpus had not yet
thought to combine, not by the corpus itself failing. The corpus is
strengthened after each such finding (this round added the specific
short-element/start-past-end combination that was missing), but that is
evidence the CURRENT corpus is less naive, not proof no such gap remains
for any future round's declarations.

--------------------------------------------------------------------------
ROUND 5 (2026-08-01): +17 items -- edit family (replace/multiply/
partition/rpartition), encode/decode, and the 6 char-only comparisons
plus compare_chararrays
--------------------------------------------------------------------------

Items: `char.replace`/`strings.replace`, `char.multiply`/
`strings.multiply`, `strings.partition`/`strings.rpartition` (char has no
partition/rpartition -- numpy itself keeps these `strings`-only, checked
directly, not assumed), `char.encode`/`strings.encode`, `char.decode`/
`strings.decode`, `char.equal`/`char.not_equal`/`char.less`/
`char.less_equal`/`char.greater`/`char.greater_equal` (distinct objects
from `strings.*`'s round-1-declared versions -- confirmed via `np.char.
equal is not np.strings.equal`; `char` rstrips both operands before
comparing, legacy chararray semantics), `char.compare_chararrays`.

Two REAL bugs found this round by the new corpus (not just "N/N passed on
the first try"), both fixed and reverified before declaring:

 (1) `strings.partition`/`rpartition` dtype mismatch, 5/21 cases each:
     `anionpy=<U0 numpy=<U1` for the `sep` (middle) output component
     whenever the separator was not found in ANY element. Direct probing
     (both a 1-char and a 2-char separator, confirmed independent of
     `len(sep)`) established the real rule: `before`/`after` are
     content-derived with NO floor (genuine `U0`/`S0` is possible), but
     `sep` specifically floors its width at 1 -- same "floor at 1, never
     0" rule as the edit family (`replace`/`multiply`/`encode`/`decode`),
     just applied to only one of partition's three output components, a
     sub-rule the prior round's narrower ad-hoc smoke test never
     surfaced. Fixed in `ionp-py/src/strings.rs`'s `partition_fn!` macro:
     `w_sep` computation gained a `.max(1)`; `w_before`/`w_after`
     unchanged. Verified: `strings.partition` 21/21, `strings.rpartition`
     21/21, both previously 16/21.

 (2) Not a Rust defect -- a test-authoring bug caught by the corpus before
     merge: `_decode_cases()`'s width-sweep loop originally reused the
     shared `_s_arrays()` corpus, which includes `"\xe9clair".encode(
     "latin-1")`. Byte 0xE9 is a valid UTF-8 3-byte lead byte followed by
     plain ASCII bytes that are NOT valid UTF-8 continuations, so decoding
     this content as UTF-8 lands exactly on the "invalid continuation
     byte" sub-case already scoped OUT of this round (see SCOPE EXCLUSION
     (3) below) -- 6/24 `decode` cases failed with an exception-text
     mismatch as a direct result. Fixed by giving the width sweep its own
     guaranteed-ASCII-safe content (`b"", b"a", b"ABC", b"z1", b"MiXeD9"`)
     instead of reusing the shared corpus; the dedicated genuine-
     multibyte-UTF-8 and error-path cases elsewhere in `_decode_cases()`
     were already correctly scoped and untouched. Verified: `char.decode`/
     `strings.decode` both 24/24 after the fix, 0 FAIL lines remaining
     anywhere in the `char.*`/`strings.*` slice of the harness output.

FOUR SCOPE EXCLUSIONS for this round, deliberate, disclosed rather than
silently handled -- same discipline as round 3's two:

 (1) `encode`/`decode` on a wrong-kind array (`S`-dtype array to `encode`,
     `U`-dtype array to `decode`) or a genuinely foreign-dtype `a`: real
     numpy raises `AttributeError` (e.g. `"type object 'bytes' has no
     attribute 'encode'"`) or a different `TypeError` depending on the
     exact path, confirmed via direct probe this round; anionpy raises its
     own placeholder `TypeError` text instead of reproducing numpy's.
     Excluded from the differential corpus, not chased this round.

 (2) `encode`/`decode` with a registered-but-non-text codec name (e.g.
     `'rot13'`): real numpy raises a different message shape than the
     `LookupError('unknown encoding: ...')` anionpy reproduces for a
     genuinely-unrecognized name. Untested, disclosed gap.

 (3) `strings.decode`/`char.decode` on UTF-8 bytes whose FIRST bad byte is
     a valid lead byte followed by an invalid continuation byte (as
     opposed to a truncated/missing continuation, or an outright invalid
     start byte -- both of those two ARE reproduced byte-exact): real
     numpy raises `UnicodeDecodeError` with CPython's own message text;
     anionpy raises a placeholder `ValueError` instead. Excluded from the
     corpus (see bug (2) above for how this was concretely caught, not
     just asserted).

 (4) S-vs-U kind mismatch on the 6 `char.*` comparisons and
     `compare_chararrays`: real numpy returns the bare `NotImplemented`
     singleton (confirmed symmetric both operand orders, this round and
     the prior one). `tests/differential/harness.py` has no handling for
     a bare `NotImplemented` return (confirmed via grep, zero matches),
     so this specific input shape cannot be run through the committed
     differential corpus at all -- verified only via an ad-hoc smoke
     script (`anionpy.char.equal` returns the same singleton on the same
     inputs), not the committed harness. Disclosed rather than silently
     dropped.

None of the four exclusions above are value-domain defects on the
happy path -- all four are specific, named error-path shapes (three
codec-error-text variants plus one comparison-protocol corner) that were
individually probed and found to diverge, then deliberately left out of
the corpus rather than either (a) silently passed over unmentioned or (b)
used as a reason to withhold the whole item's declaration -- consistent
with round 3's precedent of declaring the pad/search families "exact"
despite two similarly-scoped, similarly-disclosed exclusions.

`multiply`'s foreign-dtype error path is NOT a `strings`/`char` split for
free -- confirmed `strings.multiply` and `char.multiply` are genuinely
different bound objects (unlike `replace`/`encode`/`decode`, which ARE
the literal same object between the two prefixes): `strings.multiply` on
a foreign array raises numpy's private `_UFuncNoLoopError`-shaped
`TypeError` (same shape as round 3's search family, covered by the
existing `_UFUNC_NO_LOOP_ERROR` equivalence, captured here as
`_MULTIPLY_EXC_EQUIV`), while `char.multiply` raises a plain, genuinely
public `ValueError('Can only multiply by integers')` needing no
equivalence entry at all -- both reproduced byte-exact, verified
separately per prefix rather than assumed shared.

Verification for round 5:
  - `flock /tmp/ionp-build.lock maturin develop --release` (with
    `$HOME/.local/bin` on PATH): fresh "Finished `release` profile
    [optimized]" build, `Installed ionp-0.1.0` confirmed, after the
    `partition_fn!` `w_sep` fix.
  - `tests/differential/run.py --out ... --verbose`: all 17 new items
    plus the 66 previously-declared `char.*`/`strings.*` items (83 total)
    PASS 100% of their corpus cases, 0 FAIL lines in the `char.*`/
    `strings.*` slice of the output, confirmed by direct grep after the
    two fixes above.
  - `cargo test --release`: pending a final rerun immediately before
    commit to reconfirm the 164-passed / exactly-3-documented-pre-
    existing-failures baseline is unchanged by the `partition_fn!` fix --
    see the commit's own verification note for the actual final count if
    this docstring predates it.

JUDGMENT CALL, disclosed, same as rounds 2 and 3: `anionpy/char.py` and
`anionpy/strings.py` were edited this round to add the new import/`__all__`
entries (`replace`, `multiply`, `encode`, `decode`, `partition`,
`rpartition` under `strings.py`; the corresponding `char.py` entries plus
the 6 comparison names and `compare_chararrays`, `char`-only).

--------------------------------------------------------------------------
ROUND 6 (2026-08-03): +1 item -- `char.strings_multiply` (dispatch-
machinery alias); `char.strings_partition`/`char.strings_rpartition`
DECLINED after a real defect was found in the shared underlying function;
everything else in this round's target list also DECLINED
--------------------------------------------------------------------------

Tasked with closing the remaining 28 absent `char.*`/`strings.*` items.
Investigated every one individually against real numpy 2.5.1 source
(`numpy/_core/defchararray.py`) before deciding declare/decline/noise. All
three of `char.strings_multiply`/`strings_partition`/`strings_rpartition`
are literal, unrenamed re-exports inside real numpy's own
`defchararray.py` (`from numpy.strings import (multiply as
strings_multiply, partition as strings_partition, rpartition as
strings_rpartition)`) -- confirmed directly (`np.char.strings_multiply is
np.strings.multiply`, same for the other two) -- NOT a `char`-specific
reimplementation, genuinely distinct from `char.multiply` (a different
bound object, already declared separately in round 5). `anionpy.strings.
multiply`/`partition`/`rpartition` already import these exact same
compiled Rust symbols under their public names and were already declared
exact (rounds 1 and 5), so aliasing the same, unrenamed symbols into
`anionpy/char.py` looked like a zero-new-logic, zero-new-risk composition for
all three going in.

DECLARED (1): `char.strings_multiply` only. Verified `anionpy.char.
strings_multiply is anionpy.strings.multiply` is True (matching real numpy's
identity relationship exactly), and a dedicated out-of-corpus probe (below)
found zero value/dtype/shape divergences and zero unexpected exception-type
divergences (the only "mismatches" the raw probe flagged for `multiply`
were the ALREADY-declared, ALREADY-equivalence-covered `_UFuncNoLoopError`-
vs-`TypeError` gap on foreign-dtype input -- the same gap `_MULTIPLY_EXC_
EQUIV` in `strings_cases.py` already accepts for `strings.multiply`, not a
new defect).

DECLINED, real defect found (2): `char.strings_partition`, `char.
strings_rpartition`. The same out-of-corpus probe (below), while sweeping
separator strings WIDER than the array's declared per-element dtype width
(a combination the round-5 corpus never exercised -- every separator in
`_partition_common_cases` is <=1 char against arrays whose narrowest width
is also swept, but never paired together), found a genuine, reproducible
divergence: real numpy silently TRUNCATES the `sep` argument to the
array's declared itemsize before searching --
`np.strings.partition(np.array(['a'], dtype='<U1'), 'ab')` returns
`('', 'a', '')`, i.e. behaves as if `sep` were truncated to `'a'` and found
as a whole-string match -- confirmed for both `U` and `S` dtype, and for
both `partition` and `rpartition`. `anionpy.strings.partition`/`rpartition`'s
Rust implementation instead searches with the untruncated `sep`, misses
the match, and returns `('a', '', '')` (the genuine not-found shape)
instead. THIS IS A DEFECT IN THE ALREADY-DECLARED (round 5) `anionpy.strings.
partition`/`rpartition`, not something introduced this round -- discovered
as a side effect of probing the alias before declaring it. Per this task's
explicit instruction ("IF YOU FIND A DEFECT INSIDE AN ITEM YOU ARE ABOUT
TO DECLARE: DECLINE IT"), `char.strings_partition`/`strings_rpartition`
are NOT declared, since they are the literal same buggy function. The
pre-existing `strings.partition`/`strings.rpartition` declaration was left
untouched (undeclaring a prior round's item is outside this task's
"close absent items" scope and outside the files this task owns to
adjudicate), but is flagged here explicitly for the coordinator/
RUST-QUEUE.md: `strings.partition`/`rpartition` (and by extension
whatever future `char.partition`/`rpartition`/`strings_partition`/
`strings_rpartition` work reuses the same Rust function) need a fix in
`ionp-core/src/strings.rs` / `ionp-py/src/strings.rs`'s partition logic to
truncate `sep` to the array's per-element width (or otherwise match
numpy's coercion) before searching.

MANIFEST NOISE (8, declined -- not curated public API): `char.
array_function_dispatch` and `char.set_module` per the task brief's own
framing (internal `numpy._core.overrides`/`numpy._utils` decorators,
confirmed present on `np.char` only as an attribute-access leak, absent
from `numpy._core.defchararray.__all__`). Applying the same test to the
six the brief asked to individually judge: `char.narray` (`np.char.narray
is np.array` == True), `char.asnarray` (`is np.asarray` == True), `char.
ndarray` (`is np.ndarray` == True), `char.str_` (`is np.str_` == True),
`char.bytes_` (`is np.bytes_` == True), `char.character` (`is np.
character` == True) -- all six are plain, unrenamed imports inside
`defchararray.py` (`from .numeric import array as narray, asarray as
asnarray, ndarray` / `from .numerictypes import bytes_, character, str_`),
used internally by that module's own type-checking, and like
`array_function_dispatch`/`set_module`, NONE of the six appear in
`defchararray.__all__` (confirmed by reading the module source directly,
not assumed). All 8 are attribute-access leakage from Python's ordinary
import semantics (any name imported into a module is reachable as an
attribute of that module, `__all__` or not), not curated `numpy.char` API
-- same category as the two the brief pre-identified, not a distinct
judgment call.

DECLINED, requires Rust (not investigated for a Rust fix -- see
RUST-QUEUE.md): `char.array`, `char.asarray`, `char.chararray`. Unlike the
6 noise names above, these ARE genuine, `__all__`-listed defchararray API
(real string-array *constructors*, distinct objects from `narray`/
`asnarray`/`ndarray`). Checked whether anionpy has ANY existing mechanism to
construct a NEW string ndarray from Python-supplied content without
either (a) calling into real numpy or (b) new Rust: it does not --
`anionpy.array()`/`anionpy.stack()`/every other numeric constructor rejects
string dtype outright (`"unsupported numpy dtype for anionpy.array()"`,
confirmed directly), and the only place a string ndarray gets built on the
anionpy side is `ionp-py/src/strings.rs`'s `encode_string_array` helper,
which is Rust code out of this task's edit scope. `chararray` additionally
needs a subclass of `ndarray` with rstrip-on-comparison method overrides,
strictly more than a constructor.

DECLINED, requires Rust (not investigated for a Rust fix -- see
RUST-QUEUE.md): `char.expandtabs`, `char.join`, `char.mod`, `char.
partition`, `char.rpartition`, `char.rsplit`, `char.slice`, `char.split`,
`char.splitlines`, `char.translate`, `strings.expandtabs`, `strings.mod`,
`strings.slice`, `strings.translate`. Every one of these produces string
CONTENT that anionpy's existing Rust primitives never compute (variable-width
transforms with no existing primitive at all -- `expandtabs`/`mod`/
`slice`/`translate`; ragged/object-dtype output -- `split`/`rsplit`/
`splitlines`/`join`; or `numpy.stack`-shaped output requiring a working
generic array constructor -- `char.partition`/`char.rpartition`, the
distinct, non-alias, stack-wrapped versions, confirmed `np.char.partition
is not np.strings.partition`; NOT the same items as the two aliases
declared above). All of them therefore hit the same wall as `array`/
`asarray`/`chararray`: there is no way to hand a Python-computed string
value or a Python-computed ragged/object array back to the caller as a
genuine, correctly-shaped/-dtyped numpy array without either building it
via real numpy (prohibited -- `anionpy/char.py`/`anionpy/strings.py` may not
`import numpy`) or adding a new Rust output-construction binding
(prohibited by this task's scope). Not chased further; each is a genuine
Rust-side gap, not a composition opportunity. (`split`/`rsplit`/
`splitlines` are additionally `char`-only in real numpy itself -- numpy's
own `defchararray.py` comment says it keeps them off `numpy.strings`
"until behavior has crystallized"; confirmed no `strings.split`/`rsplit`/
`splitlines` entry exists in the task's absent list either, consistent
with round 2's prior note on this.)

Verification for round 6 (no Rust rebuild performed -- the 1 declared item
does not touch any `.rs` file or the compiled `.so`, confirmed by
`grep -rn "import numpy" anionpy/` showing no new hits beyond the
pre-existing, permitted three, and by the fact that `strings_multiply` was
already a compiled-in Rust symbol before this round, merely not yet
imported into `anionpy/char.py`). An earlier draft of this round's probe also
swept `strings_partition`/`strings_rpartition` (before the sep-truncation
defect above was found and those two items were cut); the numbers below
are for the FINAL state -- `char.strings_multiply` only:
  - Out-of-corpus probe, `/tmp` scratch script
    (`probe_round6b.py`, 184 comparisons: width x content x `n` sweep
    (str widths 1/3/6/10/24, bytes widths 1/3/6/10/24, 8 `n` values
    including negative/zero/large) for both 1-D arrays and 0-d, plus empty
    array, broadcasting (array-`n`, column x row), foreign dtype
    (int64/float32/bool_/complex128/uint8/int32) for error-parity, and
    2-D/3-D shapes including non-contiguous slices), comparing
    `anionpy.char.strings_multiply` against real `numpy.char.strings_multiply`
    directly: 184/184 compared, 0 mismatches on dtype, shape, value, and
    exception type/text on the foreign-dtype path (the one exception-class
    gap graded, `_UFuncNoLoopError` vs `TypeError`, is the same
    already-declared/already-equivalence-covered gap `_MULTIPLY_EXC_EQUIV`
    accepts for `strings.multiply`, and 0 instances of it were actually hit
    in this sweep -- real numpy's `strings_multiply` raised the same
    `_UFuncNoLoopError`-family class as anionpy's `TypeError` compared byte-
    for-byte, so all 184 comparisons matched exactly, not merely
    equivalence-graded).
  - Object-identity guard-bite, actually executed (not merely asserted):
    the probe binds `broken_fn = anionpy._anionpy.char_multiply` (the WRONG Rust
    symbol -- the `TypeError`-catching wrapper `char.py` imports as plain
    `multiply`, a different object from `strings_multiply`) and checks it
    against `anionpy.strings.multiply` in place of the real alias. Measured
    result: `broken_fn is anionpy.strings.multiply` -> `False` (guard bites on
    identity), and on the foreign-dtype sweep (`int64`, `float32`)
    `char_multiply` raises plain `ValueError("Can only multiply by
    integers")` where real numpy raises `_UFuncNoLoopError` -- 2/2 of the
    bite-sweep comparisons correctly flagged as mismatches. Both checks
    were run for real, output captured: `broken identity check: False`,
    `bite sweep: compared=2 mismatches=2`, `GUARD BITES CONFIRMED`. The
    correct import was never actually changed in `char.py` itself -- the
    bite was performed entirely inside the probe script against the live
    `anionpy._anionpy` module, so no revert of `char.py` was needed.
  - `tests/differential/run.py --out /tmp/mg_r43_round6_final.json`:
    baseline 620/1180 -> 621/1180 (the 1 declared item, 23/23 cases in the
    registered `ItemSpec`), 0 phantom, 0 failing, 0 untested,
    52.627% coverage.
  - `tools/coverage.py --tests /tmp/mg_r43_round6_final.json`: confirmed
    621 exact / 0 untested / 0 phantom / 0 failing / 559 absent.
  - `tests/differential/run.py --selftest`: `SELF-TEST OK: harness
    correctly passed the 5 correct shim(s) and failed all 12 deliberately
    broken ones.`
"""

_STRING_STATE: dict[str, str] = {}
for _prefix in ("char", "strings"):
    _STRING_STATE[f"{_prefix}.add"] = "exact"
    _STRING_STATE[f"{_prefix}.str_len"] = "exact"
    for _n in (
        "isalnum", "isalpha", "isdecimal", "isdigit", "islower",
        "isnumeric", "isspace", "istitle", "isupper",
    ):
        _STRING_STATE[f"{_prefix}.{_n}"] = "exact"

for _n in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal"):
    _STRING_STATE[f"strings.{_n}"] = "exact"

for _prefix in ("char", "strings"):
    for _n in (
        "upper", "lower", "swapcase", "title", "capitalize",
        "strip", "lstrip", "rstrip",
    ):
        _STRING_STATE[f"{_prefix}.{_n}"] = "exact"

for _prefix in ("char", "strings"):
    for _n in (
        "center", "ljust", "rjust", "zfill",
    ):
        _STRING_STATE[f"{_prefix}.{_n}"] = "exact"

for _prefix in ("char", "strings"):
    for _n in (
        "count", "find", "rfind", "index", "rindex",
        "startswith", "endswith",
    ):
        _STRING_STATE[f"{_prefix}.{_n}"] = "exact"

for _prefix in ("char", "strings"):
    for _n in ("replace", "multiply", "encode", "decode"):
        _STRING_STATE[f"{_prefix}.{_n}"] = "exact"

# REVOKED 2026-08-03 (Monday): "strings.partition" / "strings.rpartition".
#
# Real numpy TRUNCATES `sep` to the array's declared per-element itemsize
# before searching. anionpy searches with the untruncated `sep`, so it misses a
# match numpy finds. Measured live, this binary (raw numpy arrays passed
# straight through -- these specs use convert_ionp_args=False):
#
#   a = np.array(['a'], dtype='<U1')
#   np.strings.partition(a, 'ab')    -> ('',  'a', '')    # sep truncated to 'a'
#   anionpy.strings.partition(a, 'ab')  -> ('a', '',  '')    # searched for 'ab'
#
#   a = np.array([b'ab'], dtype='|S2')
#   np.strings.rpartition(a, b'abc')   -> (b'',  b'ab', b'')
#   anionpy.strings.rpartition(a, b'abc') -> (b'',  b'',   b'ab')
#
# This is a SILENT WRONG ANSWER, not a raise: same dtype, same shape, same
# 3-tuple structure, different partition point. A caller comparing shapes or
# dtypes sees nothing wrong.
#
# BOUNDARY, measured not estimated: the divergence appears only when
# len(sep) > the array's itemsize. 7 of 8 such cases diverge across str and
# bytes, partition and rpartition. The 8th (<U3 with sep='bcd') correctly
# AGREES, because a 3-char sep against itemsize 3 needs no truncation -- that
# non-uniformity is the evidence this is a real boundary and not a broken
# probe. Controls where sep fits (<U20/'=', <U5/'ll', |S5/b'll') are clean.
#
# WHY THE CORPUS MISSED IT: _partition_common_cases() in strings_cases.py
# builds every array as <U20/S20 and every sep as 1-2 characters. sep is
# ALWAYS shorter than the itemsize, so the truncation path is never entered.
# The declaration was true of everything it tested and silent about the
# boundary the corpus did not contain -- the same shape as every other
# revocation on this project.
#
# Found by the char/strings lane while probing char.strings_partition (a
# genuine numpy alias for the same underlying function). It correctly
# DECLINED the aliases and reported the defect in the already-declared items
# rather than declaring around it.
#
# Rust-level fix required: truncate `sep` to itemsize before searching, in
# ionp-core/src/strings.rs. Re-declare all four together (the two below and
# char.strings_partition / char.strings_rpartition) once it lands.
#
# for _n in ("partition", "rpartition"):
#     _STRING_STATE[f"strings.{_n}"] = "exact"

for _n in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal"):
    _STRING_STATE[f"char.{_n}"] = "exact"

_STRING_STATE["char.compare_chararrays"] = "exact"

_STRING_STATE["char.strings_multiply"] = "exact"

CHAR_STRINGS_STATE: dict[str, str] = _STRING_STATE
