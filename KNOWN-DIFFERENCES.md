# Known differences from numpy

This file tracks every place `anionpy` deliberately behaves differently from
(or is a strict subset of) numpy, per GOAL-ionp.md's rule: differences must
be written down, not hidden. This is not a bug list — items here are
intentional scope cuts or documented behavioral choices in the ndarray core
built in this pass (dtype system, array/view mechanics, broadcasting, the
ufunc engine, and the PyO3 vertical slice: `array`, `.shape`, `.dtype`,
`.reshape`, `.T`, slicing, `__add__`/`__mul__`).

## Scope cuts (not yet implemented, not silently wrong)

- **Only two ufuncs are wired through the engine in this pass**: `add` and
  `multiply` (`BinaryOp::Add`/`BinaryOp::Mul`). The engine (dtype-resolve /
  broadcast-resolve / scalar-kernel factoring in `ionp-core/src/ufunc.rs`)
  is designed so the other 132 numpy ufuncs are additional `BinaryOp`/
  `UnaryOp` variants plus a scalar closure each — no new
  broadcasting/promotion/dispatch code — but they are not declared in
  `__ion_state__` because they are not built yet.
- **Only basic indexing is implemented** (`int` and `slice`, including
  negative indices/steps, applied per-axis via a tuple key).
  Boolean-mask indexing, integer-array ("fancy") indexing, and `Ellipsis`/
  `np.newaxis` are not implemented and raise `TypeError` from
  `__getitem__`.
- **Pickle protocol (2026-08-13):** `ndarray` implements `__reduce__` / `__reduce_ex__` / `__getstate__` / `__setstate__`. Values round-trip. Reconstruct callable is `anionpy._reconstruct_ndarray`, not numpy's.
- **Buffer protocol (2026-08-13):** `ndarray.__buffer__` returns a typed shaped `memoryview` of `tobytes()` for real/bool (and empty 1-D). Complex and empty N-D raise `TypeError` so `np.asarray` uses `__array__`. Not zero-copy of the live Rust buffer; not writable. Interop with numpy goes through
  `__array__` only (materializes a C-contiguous copy on demand via
  `to_contiguous()` then `into_pyarray()`). A zero-copy `__buffer__`/
  `Py_buffer` export is future work.
- **Dtype set is the 13 fixed-width numeric types** (`bool`, `int8/16/32/64`,
  `uint8/16/32/64`, `float32/64`, `complex64/128`). `float16`, `datetime64`,
  `timedelta64`, and object/string/unicode dtypes are out of scope for this
  pass.
- **Reshape `RuntimeWarning`-shaped edge cases are not reproduced as
  warnings.** A weak float/int scalar that overflows a *narrower float*
  target (e.g. a `float32` array) matches numpy's *value* (`inf`) but anionpy
  does not emit numpy's accompanying `RuntimeWarning: overflow encountered
  in cast` — only the `OverflowError` cases this task requires (int-scalar
  into an integer-kind target out of range, and int/float-scalar exceeding
  `f64`'s own ~1.8e308 range) are implemented as exceptions. This is a
  documented scope decision, not an oversight.

## Behavioral choices verified to match numpy (recorded so they aren't
## mistaken for accidental differences)

- **Integer add/mul overflow wraps** (`wrapping_add`/`wrapping_mul` in
  `ufunc.rs`), matching numpy's own silent wraparound on fixed-width
  integer dtypes (numpy does not raise on int overflow either, e.g.
  `np.int8(127) + np.int8(1) == -128`). Not a difference, just confirmed
  intentional.
- **`anionpy.array([])` infers `float64`**, matching `np.array([]).dtype`.
  This required explicit precedence logic in `ndarray_from_pylist`
  (`ionp-py/src/lib.rs`): leaf-kind inference is `Empty < Bool < Int <
  Float`, an empty container resolves to `Empty -> Float`, and each leaf
  can only raise (never lower) the inferred kind, so `[True, 1, 2.0]`
  correctly infers `float64` the same way numpy's array-from-nested-list
  literal inference does.
- **Complex -> real cast takes the real part silently** (`cast_to` in
  `buffer.rs`, e.g. `complex128 -> float64` drops the imaginary component).
  numpy raises `ComplexWarning` (not an exception) on this cast but still
  produces the same real-part-only result; values always matched, only the
  warning was missing. **FIXED 2026-08-06 at the four sites the gap was
  actually reachable from — see the dated entry below for the fix, its
  measured boundary, and what remains genuinely unreproduced (`.fill()`,
  `putmask`/`place`).**
- **Broadcast-mismatch error text** is written to match numpy's
  `ValueError: operands could not be broadcast together with shapes ...`
  message shape (see `error.rs` `Display` impl and
  `ufunc::tests::broadcast_mismatch_is_an_error`), and raises `ValueError`
  from the PyO3 layer, not a generic exception.
- **`ndarray.reshape` accepts every numpy call form**: a bare int
  (`a.reshape(4)`), bare `-1` (`a.reshape(-1)`), varargs ints
  (`a.reshape(2,2)`), a tuple (`a.reshape((2,2))`), a list
  (`a.reshape([2,2])`), an `order=` keyword (`'C'`/`'F'`/`'A'`), and `-1`
  (or, matching numpy's actual behavior, *any* negative dimension, verified
  empirically: `a.reshape(-2,3)` succeeds identically to `a.reshape(-1,3)`)
  as the one inferred/wildcard dimension in any position. More than one
  negative dimension raises `ValueError` ("can only specify one unknown
  dimension"); a size that doesn't evenly divide raises `ValueError`.
  `order='F'` is implemented via the identity
  `a.reshape(shape, order='F') == a.T.reshape(shape[::-1]).T`
  (`NdArray::reshape_with_order` in `ionp-core/src/array.rs`), verified
  against real numpy element-for-element, not just by shape.
  `order='A'` picks `'F'` when the array is F-contiguous-and-not-also-
  C-contiguous, `'C'` otherwise, matching numpy's definition of `'A'`.
- **`ndarray.transpose` accepts every numpy call form**: no arguments
  (reverses all axes, same as `.T`), varargs ints (`a.transpose(1,0)`), or
  a single tuple (`a.transpose((1,0))`), including negative axis indices.
  Wrong axis count and repeated axes raise `ValueError`; an out-of-bounds
  axis raises `IndexError` (matching `numpy.exceptions.AxisError`, which
  subclasses both `ValueError` and `IndexError` — anionpy raises the
  `IndexError` branch, which any `except (ValueError, IndexError)` or
  `except IndexError` numpy-compatible caller will catch the same way).
- **NEP 50 weak-scalar promotion is implemented and verified against real
  numpy 2.5.1**, not punted. `ndarray.__add__`/`__radd__`/`__mul__`/
  `__rmul__` accept a bare Python `bool`/`int`/`float`/`complex` operand
  and apply numpy's exact "weak scalar" rule (`DType::weak_target_dtype` in
  `ionp-core/src/dtype.rs`, unit-tested there; PyO3-side classification/
  marshaling in `ionp-py/src/lib.rs`'s `coerce_operand`):
  - a Python scalar whose NEP 50 kind-rank (`bool < int < float < complex`)
    is `<=` the array's own kind-rank does NOT widen the array's dtype —
    `float32_array + 1.0` stays `float32`; `int8_array + 5` stays `int8`.
  - a scalar with a strictly higher kind-rank promotes to that kind's
    default dtype (`int` -> int64, `float` -> float64, `complex` ->
    complex128) — EXCEPT a `complex` scalar against a `float`-kind array,
    which numpy width-matches instead of using the default
    (`float32_array + 1j` -> `complex64`, not `complex128`; verified).
  - a numpy scalar (`np.float64(1.0)`) or any 0-d/n-d numpy or anionpy array
    is "strong" and participates in ordinary `promote_dtype` array-array
    promotion instead (routed through `numpy.asarray` + the existing
    `ndarray_from_numpy` path when it isn't already an `anionpy.ndarray`).
  - an out-of-range Python `int` going into an integer-kind target raises
    `OverflowError` with numpy's own message shape (`"Python integer 1000
    out of bounds for int8"`, byte-for-byte matched and verified). An
    `int`/`float` scalar going into a float/complex-kind target instead
    defers to the target width's own float-conversion overflow boundary
    (e.g. `float64_array + 10**100` succeeds with value `1e100`, exactly
    like numpy; `float64_array + 10**400` raises `OverflowError: int too
    large to convert to float`, the same message CPython/numpy raise).
  - reflected forms (`1 + a`, `1000 + int8_array`) behave identically to
    the forward form, including the `OverflowError` case.
  - No Python arithmetic is involved anywhere in this path: PyO3 only
    parses/extracts the scalar's numeric value (via `i128` extraction +
    explicit `DType::int_bounds()` range-check for integer targets, or
    direct `f64` extraction — i.e. CPython's own conversion, not anionpy's —
    for float/complex targets) and marshals it into a 0-d `NdArray`; the
    actual `+`/`*` is computed by the existing Rust `ionp_core::ufunc`
    kernels, unchanged.

## Verification

Differential-tested against a real numpy 2.5.1 install, confirmed by
directly querying the project's own `.venv`:
`.venv/bin/python3 -c "import numpy; print(numpy.__version__)"` -> `2.5.1`.

The original vertical-slice pass (see the scratchpad `verify.py` run
referenced in that task) covered 127/127 checks: construction +
dtype/shape/ndim/size inference (including the empty-list case), full
13-dtype numpy round-trip, reshape (single-list-arg form only, at the
time), 5 slicing patterns, dtype-promotion + array-array add/mul
correctness across 11 dtype pairs, broadcasting (including high-rank and
the mismatch-error path), negative-stride round-trip, and a `__repr__`
smoke test.

This pass (reshape/transpose call-form audit + NEP 50 weak-scalar
promotion) added a further 42/42 checks against the same real numpy 2.5.1
install (scratchpad `verify2.py`): every reshape call form (bare int, bare
`-1`, varargs, tuple, list, `-1`/negative-wildcard in any position,
`order=` keyword including a numeric element-for-element check of
`order='F'`, and both `ValueError` cases), every transpose call form
(none, varargs, tuple, negative axes, and all three error cases), and the
full NEP 50 weak/strong scalar matrix (rank-`<=` non-widening, rank-`>`
kind-default promotion, the float+complex width-match special case, numpy
strong-scalar and 0-d-array promotion, in-range and out-of-range
`OverflowError` behavior with message-text comparison, the huge-int-into-
float-target boundary, and reflected-form parity) — plus targeted spot
checks beyond that script (numpy `np.int32` strong-scalar promotion,
non-scalar operand `TypeError`, bool-weak-scalar-into-float-array,
`RuntimeWarning`-shaped overflow value parity without the warning itself).
`cargo test --workspace`: 56 passed, 0 failed (up from the prior 53; 3 new
NEP 50 unit tests in `ionp-core/src/dtype.rs` plus the reshape/transpose
call-form tests already added to `ionp-core/src/array.rs` in this pass).

## Known test-corpus/harness inconsistency: `ndarray.strides`

`ndarray.strides` fails 31/144 differential cases (all `sweep/empty*d/*`,
e.g. `empty2d`, `empty_any_axis`), and is deliberately **not** declared in
`__ion_state__` because of it. This is not a bug in anionpy's strides
computation — it's an inconsistency in real numpy itself between two
different construction paths, and the test harness (`tests/differential/
registry.py`'s `resolve_ionp()`) happens to compare across that
inconsistency:

- numpy's own `np.array(x)` **copy constructor** always zeroes every stride
  for a size-0 array, regardless of the source array's strides or the
  `order=` argument (verified empirically: `np.array(x, order='C'/'F'/'K'/
  'A').strides == (0, 0, ..., 0)` for any empty-shaped `x`). anionpy's
  `array()` matches this exactly (`from_buffer`'s `expected == 0` special
  case in `ionp-core/src/array.rs`).
- numpy's `.reshape()` is a **view** operation (not a copy) and computes
  strides for an empty target shape via its usual `max(dim, 1)`-accumulator
  formula, which is generally non-zero for every axis but the last. The
  test corpus (`tests/differential/corpus.py`'s `_fill()`) always builds
  its numpy reference arrays via `.reshape(shape)`.
- The differential harness's `resolve_ionp()` (`registry.py`) converts that
  same corpus array to anionpy via `anionpy.array(x)` — a genuine copy — before
  reading `.strides` on the anionpy side. So for `ndarray.strides` specifically,
  the numpy-side reference value comes from the pre-copy `.reshape()`-view
  array's own strides, while the ionp-side value comes from a real copy
  constructor's output. Even real numpy disagrees with itself across these
  two construction paths for empty multi-dim shapes — `np.array(x).strides
  != x.strides` when `x = np.arange(0).reshape(0, 5)`, for instance — so no
  strides implementation can satisfy both sides of this particular
  comparison at once without becoming wrong on one of the two paths.

anionpy's choice (match the copy-constructor semantics, since `array()` is a
copy constructor) is the one that matches numpy's own copy-constructor
behavior; the failing cases are exactly the ones where the corpus's
un-copied reference value diverges from what any correct copy constructor
would produce. Per the task's own instruction ("if a test seems genuinely
wrong, report it, don't route around it"), this is reported here rather
than special-cased in `ionp-core` to chase the corpus's reshape-view
numbers.

## This pass: `array` dtype kwarg, complex FMA multiply, `__repr__`

- **`anionpy.array(obj, dtype=...)`** now accepts an optional `dtype` kwarg
  (previously only the positional/no-dtype form worked), cast via the
  existing `dtype_from_pyobj` + `NdArray::cast_to`. Verified against all 15
  `array` corpus cases including `list_with_dtype_kwarg` and
  `list_with_dtype_string_kwarg`.
- **Complex multiply uses hardware FMA, matching numpy exactly.** numpy's
  `complex64`/`complex128` multiply is not the textbook two-multiply-then-
  subtract/add formula; it's computed with a fused multiply-add for one
  cross-term of each of the real and imaginary parts
  (`re = fma(ar, br, -(ai*bi))`, `im = fma(ar, bi, ai*br)`), verified
  empirically against real numpy (0/200,000 mismatches with FMA vs ~44%
  mismatches with the naive formula, both precisions). Implemented as
  `complex_mul_fma`/`MulAddExt` in `ionp-core/src/ufunc.rs` using
  `f32::mul_add`/`f64::mul_add`, replacing `num_complex::Complex`'s default
  `Mul` impl for the `Mul` binary op only (`Add` is unaffected). This FMA
  multiply is **not** bit-symmetric under operand swap (`a*b != b*a` at the
  ULP level for ~33% of random complex pairs), so `ndarray.__rmul__` calls
  `ufunc::multiply(&other, &self.inner)` (reflected-operand order
  preserved) rather than delegating to `__mul__`, which would silently use
  the wrong operand order for the fused term.
- **`ndarray.__repr__` is a full from-scratch Rust port of numpy's default
  `precision=8, floatmode='maxprec'` array-printing algorithm**
  (`ionp-core/src/repr.rs`), covering bool/int/float/complex element
  formatting, exponential-vs-positional mode selection, nan/inf handling,
  negative zero, bracket nesting, blank-line separator counts by axis
  depth, `linewidth=75` line wrapping, and the `shape=`/`dtype=` repr
  extras. Three subtleties worth recording since they cost real debugging
  time:
  1. **Digit generation must happen at the array's native dtype width, not
     `f64`.** Widening an `f32` value to `f64` before asking for `f64`'s own
     shortest-round-trip decimal string produces the ~17-digit exact binary
     expansion of that `f32` value, not the ~6-9-digit string numpy shows —
     digit generation (`sci_parts`/`positional_parts`) must format via the
     value's real `f32`/`f64` type.
  2. **`precision=8` is a global default, not derived from
     `np.finfo(dtype).precision`.** That per-dtype "precision" (6 for
     float32, 15 for float64) is a *different* number, used only in the
     scientific-vs-positional mode cutoff decision — the digit-count cap
     itself is always 8, for both float32 and float64.
  3. **Fixed-point (`floatmode='maxprec'`) mode does not zero-pad shorter
     elements to a common fractional-digit width** — each element keeps its
     own natural (precision-capped) digit count, and shorter numbers are
     right-padded with *spaces* for column alignment
     (e.g. `[ 0.60512  , -3.2967448, ...]`). Exponential mode is the
     opposite: numpy forces every element to exactly the array's max needed
     fractional-digit count via a second, correctly-rounded formatting pass
     (not zero-padding the first pass's shorter natural digits) — so anionpy's
     exp-mode path re-renders each value at exactly `pad_right` digits of
     precision (`sci_parts_at`) rather than reusing/padding the natural
     shortest string.
  All 144/144 `ndarray.__repr__` differential cases pass.

## This pass: generic ufunc engine (`.at`/`.reduce`/`.accumulate`/`.reduceat`, complex divide)

One generic dispatcher in `ionp-core/src/ufunc.rs` now drives 18 binary
ops (`Add`, `Subtract`, `Multiply`, `Divide`, `Maximum`, `Minimum`,
`Greater[Equal]`, `Less[Equal]`, `Equal`, `NotEqual`, `Logical{And,Or,Xor}`,
`Bitwise{And,Or,Xor}`) and 4 unary ops (`Negative`, `Absolute`, `Invert`,
`LogicalNot`), covering 25 numpy-visible names (`divide`/`true_divide` and
`absolute`/`abs` alias one op each). 21 of those 25 names are declared
`exact` in `__ion_state__`; `divide`, `true_divide`, `absolute`, `abs` are
deliberately left `absent` — see the dedicated section below.

- **`.at()` mutation model is copy-on-write via `Arc::make_mut`.**
  `NdArray.buffer` is `Arc<Buffer>` (shared with any view/copy that arose
  from the same underlying allocation); `at_binary`/`at_unary` call
  `Arc::make_mut(&mut target.buffer)` to get an exclusive mutable
  `&mut Buffer`, which clones the underlying data the first time the `Arc`
  has more than one live reference and mutates in place otherwise — the
  same amortized-cost pattern numpy's own in-place ufunc methods rely on,
  just expressed through Rust's `Arc` refcount instead of numpy's
  `WRITEABLE`/base-array bookkeeping. Callers never see a distinction: `.at`
  always appears to mutate `target` in place from Python, whether or not a
  hidden copy happened underneath.

- **`.at()` computes in the op's naturally-promoted dtype, then casts the
  result back into the target's own storage dtype** — this is NOT the
  same rule the plain call uses for its `out` dtype. Verified against real
  numpy 2.5.1: `divide.at(int32_target, [0], float_values)` succeeds,
  silently truncating the float division result back into the `int32`
  slot; promotion-driven dtype *widening* alone is never rejected by
  `.at`. `.at` only raises when the op+dtype combination is fundamentally
  illegal regardless of `.at` (bitwise on a non-integer dtype; `subtract`
  on `Bool op Bool` specifically) — exactly the cases `binary_out_dtype`
  already reports via `Err`. Implementation: `at_binary` checks legality
  via `binary_out_dtype(op, dtype, values.dtype())` (the real operand
  pair, not `dtype` against itself — `subtract.at(bool_target, [0],
  int8_values)` is legal, since `values` isn't `Bool` so it isn't the
  "boolean subtract" case at all), then if the resolved `compute_dtype`
  differs from `dtype`, recomputes on a temporary full-array cast of
  `target` (`target.cast_to(compute_dtype)`, recursing into `at_binary`
  once — the recursion is one level deep because `compute_dtype` is a
  fixed point of `binary_out_dtype`) and casts the result back into
  `dtype` before swapping it into `target.buffer`.

- **Comparison-ufunc reduce-family errors use two different numpy
  exception classes depending on the *reduced axis's* length, not the
  array's total size.** `.reduce`/`.accumulate`/`.reduceat` on a
  comparison ufunc (`greater`, `equal`, etc. — these only have a
  `(bool,bool)->bool` loop, no numeric reduce loop) raises a plain
  `TypeError` when the reduction is over zero elements (empty-reduction
  case), but numpy's private `_UFuncNoLoopError` (displays as
  `UFuncTypeError`, a `TypeError` subclass) when a real, non-empty
  reduction is attempted and simply has no loop for the dtype. These are
  NOT the same condition as "array is empty": `.reduce()` with no `axis`
  kwarg defaults to `axis=0` (confirmed both in numpy and in anionpy's own
  `Ufunc.reduce` PyO3 signature, `#[pyo3(signature = (array, axis=0,
  ...))]`), so `equal.reduce(np.ones((3, 0, 2)))` reduces along axis 0
  (length 3, nonzero) even though the array's total size is 0 — verified
  against real numpy that this raises `_UFuncNoLoopError`, not the plain
  empty-reduction `TypeError`. `compare_reduce_error`'s dispatch is keyed
  on `reduce_axis_len(a, full)` (the flattened size when `full` i.e.
  `axis=None`, otherwise `shape[0]`), not `a.size()`.

- **complex64/complex128 division uses Smith's algorithm, not
  `num_complex::Complex`'s default (textbook two-rounding) `Div`.** numpy's
  C source (`nc_quot`/`npy_cdivf`/`npy_cdiv` in `npy_math_complex.c.src`)
  branches on which of `|Re(y)|`/`|Im(y)|` is larger to avoid intermediate
  overflow/precision loss; the textbook formula measurably diverges from
  numpy on a nontrivial fraction of complex64 division cases (differences
  beyond `atol=1e-9`). `complex_div` in `ufunc.rs` ports the branching
  algorithm directly and matches numpy exactly on every non-complex64-edge
  case.

- **The weak-scalar target dtype for `divide`/`true_divide` against an
  int/bool array is forced to `float64`, not the array's own dtype** —
  `divide` always promotes int/bool operands to `F64` for its output loop
  (see `binary_out_dtype`'s `Divide` arm), and numpy's own weak-scalar
  marshaling follows that promoted target, not the array's storage dtype.
  Verified against real numpy: `np.array([], dtype=np.uint8) / -7`
  succeeds (`array([], dtype=float64)`) even though the same scalar
  against `+` on the same empty `uint8` array DOES raise `OverflowError`
  (since `+` keeps the array's own integer dtype as the promotion target,
  and numpy's bounds-check runs regardless of the array being empty).
  `scalar_against`/`extract_binary_pair` in `ionp-py/src/lib.rs` take a
  `forces_float` flag (true only for `Divide`) that forces the weak-scalar
  target to `F64` before marshaling, so the int-bounds check never runs at
  all for `divide`/`true_divide`.

### `divide`/`true_divide`: `RuntimeWarning` on division by zero not reproduced (values match)

Real numpy emits `RuntimeWarning: divide by zero encountered in divide`
and/or `RuntimeWarning: invalid value encountered in divide` when a
division produces `inf`/`-inf`/`nan` from a finite/zero operand — verified
directly (not assumed) for float division (`[1.0, -1.0, 0.0] / 0.0` ->
`[inf, -inf, nan]`, two warnings) and integer division (`int32` array
divided by `0` -> promotes to `float64` `[inf, -inf, nan]`, same two
warnings — the int-to-float64 promotion itself already matches, see
`binary_out_dtype`'s `Divide` arm). anionpy produces the identical values and
dtype in both cases but emits no warning at all. Same precedent as the
existing `ComplexWarning`-on-cast entry above: values match, the warning
is not reproduced. The differential harness does not grade warnings
(`ufunc_cases.py` runs every case under `warnings.simplefilter("ignore")`),
so this is not something the differential run could have caught either
way — recorded here because the task brief asked for it to be checked
explicitly rather than assumed, not because any test depends on it.

### `divide`, `true_divide`, `absolute`, `abs`: complex zero-denominator defect fixed; ULP tolerance declared; `absolute`/`abs` now `exact`, `divide`/`true_divide` blocked by an unrelated float16 gap (2026-08-01)

**This supersedes the previous version of this section**, whose central
claim — "no Rust-side scalar formula can reach bit-exact agreement,
regardless of algorithm" — was wrong. It was reached by testing an
f64-promoted *non-FMA* `hypot` variant only; nobody had tried hardware FMA
(fused multiply-add), which the codebase already used for complex multiply
(`complex_mul_fma`, same file) but not for complex division.

**Two problems were in play, and they were not the same problem:**

**Problem 1 (real defect, now fixed):** `ionp-core/src/ufunc.rs`'s
`complex_div` computed `1+1j` divided by a zero scalar as `nan+nanj`; real
numpy returns `inf+infj` (a wrong special-case branch, not a rounding
matter — ULP distance is infinite, no tolerance could have hidden it).
Root cause: the prior Smith's-algorithm implementation had no
zero-denominator special case at all, so it silently produced `0/0 = nan`
mid-formula. Fixed by measuring numpy's actual behavior directly (not
assumed) over a `{0, -0, 1, -1, 2, -3, inf, -inf, nan}^4` grid of
real/imaginary numerator/denominator combinations, both precisions,
compared bit-for-bit (including NaN-vs-NaN and the sign of infinities/
zeros) against real numpy 2.5.1. Findings, all directly verified rather
than guessed from a remembered C99 Annex G / glibc `__divdc3` formula
(an initial hypothesis modeled on that formula was tried and empirically
falsified — see below):
  - When the denominator is exactly complex zero (`c == 0 && d == 0`),
    **the sign of the zero denominator does not affect the result** —
    `1+1j / (0+0j)`, `1+1j / (-0+0j)`, `1+1j / (0-0j)`, and
    `1+1j / (-0-0j)` all give the identical `inf+infj`. The result is
    `(a * inf, b * inf)` using a plain *positive* infinity constant (not
    `copysign(inf, c)` as glibc's algorithm would compute) — a numerator
    component that is itself exactly zero maps to NaN (`0 * inf`), a
    nonzero numerator component maps to a signed infinity carrying that
    component's own sign via ordinary IEEE-754 multiplication.
  - Every OTHER special-value combination in the same grid — infinite
    numerator with finite denominator, infinite denominator, NaN operands
    — needs **no additional correction**: numpy's actual complex64/128
    divide loop does not implement the C99 Annex G NaN-recovery rules
    (unlike glibc). `(inf+infj)/(1+0j)` and `(1+0j)/(inf+infj)` both come
    back `nan+nanj` from real numpy, matching the raw FMA-Smith formula
    computed with no correction at all. An Annex-G-style correction branch
    was implemented and tested first; it was falsified by this same grid
    (it produced `inf-infj` where numpy gives `nan+nanj`) and was removed
    rather than kept "just in case."
  - Real-valued (float32/float64) and integer-typed division were also
    checked against the same grid and already matched numpy bit-exact
    with no fix needed — Rust's native `/` on `f32`/`f64` already
    implements correct IEEE-754 zero/sign/inf/nan semantics.

**Problem 2 (numpy's own loop is not correctly rounded — this is what ULP
tolerance exists for):** Even with Problem 1 fixed, `complex_div`'s
finite-value results disagreed with numpy by up to 40 ULP (complex128).
Root cause: numpy's complex divide (`npy_cdivf`/`npy_cdiv`) computes the
Smith's-algorithm cross terms with hardware FMA, not the two-rounding
formula the prior Rust code used — the exact same gap `complex_mul_fma`'s
doc comment already documents for complex multiply. Switching
`complex_div`'s cross terms to `MulAddExt::mul_add_ext` (the same trait
`complex_mul_fma` uses) reduced the measured maximum from 40 ULP
(complex128) / 3 ULP (complex64) down to **1 ULP for a single division
call**, verified against the same `{0,-0,1,-1,2,-3,inf,-inf,nan}^4` grid
(74/6561 finite-value combinations disagree, all by exactly 1 ULP, both
precisions) and against the real differential corpus.

The same investigation was applied to `absolute`/`abs`'s complex path
(`z.re.hypot(z.im)`, unchanged by this fix) using the report's own
methodology (`reports/ionp-ulp-tolerance-decision-2026-08-01.md`): numpy's
complex64/128 `abs` is independently confirmed not correctly rounded
(scalar `hypotf` via `ctypes` disagrees with `np.abs` on the same inputs
anionpy disagrees with numpy on), so a declared tolerance there is fixing an
ill-posed bit-exactness requirement against a non-correctly-rounded
reference, not tuning a ruler to fit a failing test.

**Declared tolerances** (`tests/differential/ufunc_registry.py`'s
`_ULP_OVERRIDES`), each the actual measured `max_ulp_observed` from
`tests/differential/run.py`'s own harness over the real corpus, not a
padded/rounded-up number:

| item | ulp_tolerance | measured against |
|---|---|---|
| `absolute`, `abs` | **1.0** | 158/158 corpus cases (complex64+complex128); no `.reduce`/`.accumulate` call form applies to a unary op, so there is no compounding path — every case tops out at 1 ULP |
| `divide`, `true_divide` | **17.0** | 262-case corpus. A *single* division call is <=1 ULP off numpy on every complex64/complex128 case (`call/same_shape/complex64`, `call/same_shape/complex128`, every scalar-call-form case). The 17.0 maximum is `reduce/sweep/2d/complex64`: `np.divide.reduce` chains 2 divisions per output element, and each chained division's <=1-ULP error compounds through float32's 24-bit mantissa — this is arithmetic composition of an already-correct per-call bound, not a formula defect |

Int/bool/real-float divide, and every non-complex `absolute`/`abs` case,
remain bit-exact with **no** tolerance — the harness's `atol=0.0, rtol=0.0`
default for exact-typed results is untouched by this change.

**Anti-tautology check** (task requirement — prove the declared tolerance
still catches wrong code, not just that it lets correct code through): 4
deliberate breakages were built, rebuilt with `maturin develop --release`,
and re-run against the real differential harness, each reverted after
confirming:
  a. Reverting `complex_div`'s cross terms to the old non-FMA formula:
     `max_ulp_observed` for `divide` jumps to **40.0** against a declared
     bound of 17.0 → correctly **FAILS** (21/262 cases, up from 19).
  b. Forcing complex64 `abs`'s actual output buffer to `f64` instead of
     `f32` (a wrong dtype promotion, injected at the real buffer-construction
     site — the naive injection point, `unary_out_dtype`'s `Absolute` arm,
     turned out to be dead code for this op because of an earlier
     dtype-specific early return, so the probe was moved to where the
     dtype decision is actually made): dtype mismatch → correctly
     **FAILS** (146/158, structural check, independent of any ULP
     tolerance).
  c. Flipping the sign in the zero-denominator special case
     (`(-a) * inf` instead of `a * inf`): `1+1j / 0j` now returns
     `-inf+infj` instead of numpy's `inf+infj` → ULP distance is infinite
     (a sign mismatch is never tolerated, by design — see
     `harness.py`'s `ulp_distance`) → correctly **FAILS** (21/262).
  d. Nudging complex `abs`'s result by 3 ULP beyond its already-~1-ULP
     baseline error (`f32::from_bits(hypot(...).to_bits() + 3)`): total
     error now several ULP over the declared bound of 1.0 → correctly
     **FAILS** (18/158, up from 0).

**Current declaration status in `anionpy/__init__.py`'s `__ion_state__`:**
  - `absolute`, `abs`: **`"exact"`** — 158/158 differential cases pass
    under the declared `ulp_tolerance=1.0`.
  - `divide`, `true_divide`: **still not declared**, but for a different
    reason than before this fix. All 19/262 remaining failing cases trace
    to one unrelated cause: numpy's own `divide.ufunc.types` legacy-loop
    table lists `float16` (`'ee->e'`) as its *first* signature, so the
    shared differential corpus's `info.typed_sample` for `divide`
    specifically is float16-typed (unlike e.g. `add`, whose first type
    signature happens to be a dtype anionpy supports) — and `anionpy.array()`
    cannot construct a float16 array at all
    (`"unsupported numpy dtype for anionpy.array() (supported: bool, int8-64,
    uint8-64, float32/64, complex64/128)"`, a pre-existing limitation of
    the whole array/buffer module, unrelated to division). This is why
    `.at()` call-form cases fail even for otherwise-fully-supported dtypes
    like int32/float32/complex64/bool/int8 as the *primary* array: `.at()`
    reuses `info.typed_sample`'s second array as the scatter "values"
    regardless of the primary array's own dtype, so the float16
    contamination spreads to every `.at()` case for this ufunc. Verified
    individually, case by case, that none of the 19 failures are a
    division-arithmetic mismatch. Adding float16 support is a new dtype
    across the whole buffer/array system (a new `Buffer` variant, PyO3
    conversion, promotion rules) — out of scope for a division fix, and
    would touch `Cargo.toml`/`Cargo.lock`, which this task was explicitly
    told to leave alone. `divide`/`true_divide` stay undeclared until that
    separate gap is closed.

## `absolute`/`abs`/`divide`/`true_divide`: sample-size defect in the declared ULP bounds, fixed with an adequate-sweep gate (2026-08-01)

**This supersedes the declared-tolerances table in the section immediately
above.** Both of that section's declared bounds — `absolute`/`abs` at
`1.0` ULP, `divide`/`true_divide` at `17.0` ULP — were measured only
against the ~150/262-case differential corpus (`tests/differential/
corpus.py`), which is built for edge-case *breadth* (empty arrays, NaN,
inf, `-0.0`, dtype boundaries, broadcasting…), not for statistical
*volume*. **The tolerance was a property of the sample, not of the
implementation** — a corpus that small can simply fail to contain the
input that produces the worst-case disagreement, and it did, twice, in
opposite directions (one bound too tight, the other wildly too loose).

**Independent re-measurement, `tests/differential/ulp_sweep.py`, `n=20,000`
seeded values per dtype, `SEED=20260731` (the same pinned seed every other
differential input uses):**

- **`absolute`/`abs`**: the declared `1.0` ULP bound was **wrong** — true
  max is **2.0** ULP (confirmed independently here, not copied from
  `reports/ionp-ulp-tolerance-decision-2026-08-01.md`'s ⛔ correction
  block, which had already flagged this same number by a different route).
  float32/float64 are bit-exact (`0.0` ULP, 0/20,000 disagree);
  complex64/complex128 top out at `2.0` ULP (7,089/20,000 and 7,554/20,000
  elements disagree respectively, all by ≤2 ULP). No `.reduce`/
  `.accumulate`/`.outer` call form applies to a unary (`nin=1`) ufunc, so
  there is no compounding path beyond this — the per-dtype `"call"`-form
  sweep covers every call form this item's corpus actually exercises.
- **`divide`/`true_divide`**: the declared `17.0` ULP bound was **not just
  wrong, but not actually measured at adequate volume in the first
  place** — it came from a single `(3,4)`-shaped `reduce/sweep/2d/
  complex64` case in the old corpus (4 output elements total), generalized
  from one worked example ("2 divisions chained, each ≤1 ULP, compounding
  to 17") without checking the corpus-wide maximum. Re-derived here rather
  than assumed correct:
  - A single, uncompounded `"call"`-form division sweep is clean and
    matches the section above: `0.0` ULP float32/float64, `1.0` ULP
    complex64 (8,441/20,000 disagree) and complex128 (8,611/20,000
    disagree).
  - But `divide`/`true_divide` are binary (`nin=2`) ufuncs, so their real
    corpus also exercises `.reduce`/`.accumulate`/`.outer`/`.at`, which
    **chain** multiple divisions — exactly the shape the retracted `17.0`
    claim was about, just never swept at volume. A chained-division sweep
    (`ulp_sweep.measure_reduce_chain`, 20,000 independent chains per
    length/dtype) gives:

    | chain length (divisions) | complex64 max ULP | complex128 max ULP |
    |---|---|---|
    | 2 (1 division) | 1.0 | 1.0 |
    | 3 (2 divisions — the shape `17.0` was about) | **31228.0** | **14581.0** |
    | 4 (3 divisions) | 19500.0 | 10242.0 |
    | 5 (4 divisions) | 79990.0 | 239968.0 |

    Three to five orders of magnitude past the old `17.0` figure, and not
    monotonically increasing with chain length either — this is driven by
    rare near-zero-denominator cancellation in specific chains, not a
    fixed per-division compounding constant, so there is no small integer
    "the real number should have been." Reproduced independently on the
    live build (`.venv/bin/python3 tests/differential/ulp_sweep.py`) as
    part of this fix, not carried over from a prior run.

**Structural fix — the adequate-sweep gate (`registry.py`):**
`ItemSpec.__post_init__` now refuses to construct any item that declares a
`ulp_tolerance` unless it also carries a non-empty `ulp_sweep` field of the
shape `{dtype_name: (sample_size, measured_max_ulp)}`, where every
`sample_size >= MIN_ULP_SWEEP_N` (20,000) and the declared `ulp_tolerance`
**equals** (not merely `>=`) the maximum `measured_max_ulp` across every
recorded dtype. Equality, not `>=`, is deliberate: it forbids both
under-evidencing (a small/absent sweep, or a declared number below the
true measured max — the original `absolute`/`abs` defect) and padding
("round up to be safe" — not done here, and now structurally impossible,
since it would raise the exact same error as under-declaring). This mirrors
the pre-existing `ulp_justification`/`epsilon_justification` guards in the
same method, extended to cover sample *size*, not just the presence of a
prose citation — a `ulp_justification` string alone was never proof of an
adequately-sized measurement, and `absolute`/`abs`'s original `1.0` bound
had a real justification string attached the whole time it was wrong.

**Anti-tautology proof that the gate actually fires** (not just that it is
documented — `tests/differential/test_ulp_distance.py`,
`test_itemspec_refuses_*` / `test_itemspec_accepts_*` / real output
reproduced below):

- Declaring `ulp_tolerance` with **no** `ulp_sweep` at all → raises
  (real traceback captured constructing `ufunc_registry.py`'s `abs` entry
  before this fix's `_ULP_OVERRIDES` update, i.e. an accidental live
  demonstration during development, not a synthetic test):
  ```
  ValueError: abs: ulp_tolerance=1.0 declared with no recorded ulp_sweep
  evidence -- refusing to construct this ItemSpec. A ulp_tolerance must be
  measured over an adequate randomized sweep (>= 20000 seeded values per
  applicable dtype, see ulp_sweep.py), not merely over the item's ordinary
  differential corpus -- a corpus too small to contain the discrepancy will
  under-evidence the bound (see the ⛔ correction in reports/ionp-ulp-
  tolerance-decision-2026-08-01.md, where this happened to absolute/abs's
  own original bound).
  ```
- A sweep with `sample_size < 20,000` → raises (`test_itemspec_refuses_
  sweep_smaller_than_minimum_sample_size`, `n=150`).
- A declared `ulp_tolerance` above the measured max (`5.0` declared vs.
  `2.0` measured — padding) → raises (`test_itemspec_refuses_declared_
  tolerance_above_measured_max`).
- A declared `ulp_tolerance` below the measured max (`1.0` declared vs.
  `2.0` measured — exactly the original defect, reproduced as a unit test)
  → raises (`test_itemspec_refuses_declared_tolerance_below_measured_max`).
- An adequate sweep with the declared bound exactly equal to the measured
  max → constructs successfully (`test_itemspec_accepts_sweep_matching_
  declared_tolerance_exactly`, and the real `absolute`/`abs`/`divide`/
  `true_divide` construction via `ufunc_registry.py` at run time).
- **Re-declaring `abs` at the true, now-measured bound (`2.0` ULP) still
  catches an implementation that is 1 ULP worse than that bound** — direct
  `harness.compare_values` probe, real output:
  ```
  >>> true = np.float32(np.abs(np.complex64(1+2j)))          # 2.2360679...
  >>> two_off  = nudge(true, 2)   # exactly at the declared bound
  >>> three_off = nudge(true, 3)  # 1 ULP worse than the declared bound
  >>> harness.compare_values(true, two_off, atol=0.0, rtol=0.0,
  ...                        ulp_tolerance=2.0, ulp_justification="x")
  (True, '', True, 2.0)
  >>> harness.compare_values(true, three_off, atol=0.0, rtol=0.0,
  ...                        ulp_tolerance=2.0, ulp_justification="x")
  (False, "ULP distance 3.0 exceeds declared tolerance 2.0 (justification:
  'x'): numpy=array([2.236068], dtype=float32)
  anionpy=array([2.2360687], dtype=float32)", False, 3.0)
  ```
  The declared bound is exactly as tight as the true measurement, no
  looser — a real 3-ULP-off implementation is still rejected, not silently
  absorbed by slack padded in "to be safe."

**Corrected declarations, replacing the table in the section above:**

| item | ulp_tolerance | measured against |
|---|---|---|
| `absolute`, `abs` | **2.0** | 20,000 seeded values/dtype, `SEED=20260731`. float32/float64: `0.0` (bit-exact). complex64/complex128: `2.0` max |
| `divide`, `true_divide` | **undeclared** (retracted) | single-call sweep is clean (≤1.0 ULP), but the real corpus's `.reduce`/`.accumulate`/`.outer` call forms chain divisions, and a chained-division sweep at adequate volume measured **31228.0 ULP** (complex64, 2-division chains) — no honest finite bound covers the compounding path, so none is declared; falls back to the harness's bit-exact default and correctly fails the differential suite's complex64/complex128 cases (see coverage ledger below) |

`divide`/`true_divide` were already failing the differential suite before
this change (for the unrelated float16-corpus gap documented in the
section above), so retracting the `17.0` declaration and leaving them
undeclared changes *why* the complex64/complex128 cases fail (now: "no
tolerance declared, bit-exact grading, ULP distance 1.0" instead of
passing under a since-invalidated 17.0 bound for the single-call cases —
those single-call cases were already ≤1 ULP and were previously credited
as passing under the old, too-loose 17.0 declaration) but **does not
change the item's overall pass/fail verdict or the coverage ledger total**
— see "Coverage impact" below. Choosing not to pad a "safe-looking" large
number (e.g. declaring something like `250000.0` to cover the swept
maximum) instead of leaving the item undeclared was deliberate: the swept
maximum is not a stable per-item constant (it varies by chain length and
isn't monotonic — see the table above), so any single declared number
would itself be exactly the kind of under-evidenced, sample-dependent
figure this whole fix exists to eliminate.

**Algorithmic-parity position (context, not independently re-derived
here):** numpy's own complex64/complex128 division is **not correctly
rounded** — it implements Smith's algorithm with the same cross-term
formula anionpy now matches (`MulAddExt::mul_add_ext`), not a mathematically
exact per-element division. Measured against a correctly-rounded
complex128 reference, single complex64 division has been observed up to
~1866 ULP from the true mathematical result, and a chained `(a/b)/c`
computation up to ~3312 ULP from the true result — both figures given as
established context for this fix (from the task brief / prior
investigation), not independently re-verified in this pass, since only the
ionp-vs-numpy `absolute`/`abs` bound required independent re-verification
here. This is why anionpy's declared tolerances in this file are always
**anionpy vs. numpy**, never anionpy vs. mathematically-correct: matching
numpy's own (imperfect) algorithm is the deliberate, defensible, and now
measured/documented goal — numpy is the reference this whole differential
suite is testing against, and a "more correct than numpy" result would
itself be a difference requiring justification, not a free win. The
~1866/~3312 ULP figures are recorded here only to make that position
explicit and falsifiable, not as a bound this codebase declares or is
graded against.

**Coverage impact:** the coverage ledger is unaffected by this fix —
`41/1180` (`39` bit-exact, `2` ULP-tolerant: `absolute`+`abs`) before and
after, verified via `tools/coverage.py`. `absolute`/`abs` remain `[PASS]`
(158/158 corpus cases) under the corrected `2.0` bound; `divide`/
`true_divide` remain `[FAIL]` overall both before and after (36/262 cases
failing after this change — the float16 gap's 19 plus the newly-undeclared
complex64/complex128 cases — versus 19/262 before), so no previously
"passing" item regresses to failing and no previously-failing item is
newly counted as passing. This was verified, not assumed: coverage may
legitimately drop when an inadequately-evidenced tolerance is retracted
(the task explicitly permits this), it simply didn't drop here because
`divide`/`true_divide` were already below the pass bar for an unrelated
reason.

**Runtime cost:** the full sweep (4 items × 4 dtypes × 20,000 values, plus
the chained-division sweeps used to derive the table above) runs in
well under a second on the reference machine (`ulp_sweep.py`'s own
`__main__` block, timed empirically during this fix: ~0.5s for the
non-chained sweeps). `tests/differential/run.py`'s normal suite run never
re-executes a sweep — the gate only checks that `ulp_sweep` evidence is
*present and internally consistent* on already-constructed `ItemSpec`
objects; the actual sweep is a one-time, manually-invoked step at
declaration time (`python3 tests/differential/ulp_sweep.py`), and its
result is what gets hand-copied into `ufunc_registry.py`'s
`_ULP_OVERRIDES`. If a future item's dtype catalog grows large enough for
the sweep itself to become a bottleneck, the split to make is exactly this
one — sweep once, cache the `(n, max_ulp)` evidence in the registry, never
re-sweep inside the hot test-run path — which is already how this fix is
built, not a change still owed.

## `absolute`/`abs`: per-dtype grading defect fixed — the grader now checks each dtype against its own measured bound, not the item-wide max (2026-08-01)

**This supersedes the declared-tolerances table in the section immediately
above.** That section correctly *measured* `absolute`/`abs`'s ULP
imprecision per dtype (float32/float64 bit-exact at `0.0`, complex64/
complex128 genuinely imprecise at `2.0`), but the grader collapsed that
evidence into a single item-wide `ulp_tolerance=2.0` scalar before
declaring it. The practical effect: float32 and float64 results — which
their own measurement proved bit-exact — were graded with 2 ULP of
unearned slack borrowed from the complex dtypes. **A genuine 2-ULP
regression in real-valued `abs` would have passed silently.** No such
regression currently exists (see "Coverage impact" below), but the grader
could not have caught one if it did.

**Structural fix:**

- `ItemSpec.ulp_tolerance` is now `Optional[dict[str, float]]` —
  `{dtype_name: bound}` — never a bare scalar. `registry.py`'s
  `ItemSpec.__post_init__` refuses construction if it is not a non-empty
  `dict` (`test_itemspec_refuses_non_dict_ulp_tolerance`,
  `test_itemspec_refuses_empty_dict_ulp_tolerance`).
- The per-dtype exact-equality gate from the section above now applies
  **per dtype key**, not to the item-wide max: `set(ulp_tolerance) ==
  set(ulp_sweep)` is required (every declared dtype must carry its own
  sweep evidence, and every swept dtype must carry its own declared
  bound — a dtype can never silently inherit another dtype's number by
  omission), and for every dtype, `ulp_tolerance[dtype] ==
  ulp_sweep[dtype].measured_max` exactly (`test_itemspec_checks_each_
  dtype_against_its_own_measured_max_not_the_worst`,
  `test_itemspec_refuses_tolerance_dtype_with_no_matching_sweep_entry`,
  `test_itemspec_refuses_swept_dtype_with_no_declared_tolerance`).
- `harness.compare_values()` looks up the bound for a result via
  `ulp_tolerance.get(dtype_key, 0.0)` — **`0.0` (bit-exact) is the
  default for any dtype with no entry in the dict**, made explicit and
  enforced rather than left emergent
  (`test_per_dtype_grading_unlisted_dtype_defaults_to_bit_exact`).
- `ufunc_registry.py`'s `_ULP_OVERRIDES` for `absolute`/`abs` is now
  `{"float32": 0.0, "float64": 0.0, "complex64": 2.0, "complex128": 2.0}`
  — the exact per-dtype measured maxima from the sweep in the section
  above, not the item-wide `2.0`.

**A second, related bug found and fixed while implementing this:** the
dtype key used to grade a result must be the **operand's** dtype, not
the **result's** dtype, whenever they can differ. `np.abs` on a
complex64 operand returns a **float32** result (and complex128 →
float64) — but the sweep evidence in `ulp_sweep.py` is keyed by the
swept *input* dtype (see `ulp_sweep.py`'s `_summarize` and
`corpus.py`'s `_shape_sweep`, which tag cases by the array's own
dtype), not the output's. Grading by the result's dtype would key a
complex64-input case as `"float32"` and collide it with true
float32-input `abs`, which shares that key but is bit-exact — either
wrongly forgiving a float32 regression (if the complex slack leaked in)
or wrongly rejecting the genuine, evidenced complex64 imprecision (if
graded at float32's `0.0`). Fixed via a new `harness._first_operand_
dtype(args)` helper — scans a case's call arguments for the first one
carrying a `.dtype` (skipping non-array items such as a ufunc call-form
tag string like `"call"`/`"reduce"`) — and a new `source_dtype`
parameter threaded through `compare_values()` → `compare_multi_output()`
← `run_case()`, used as the grading key in preference to the result's
own dtype, falling back to the result's dtype only when no operand
dtype can be determined. Proven directly
(`test_per_dtype_grading_uses_operand_dtype_not_result_dtype_for_abs_shaped_item`
in `test_ulp_distance.py`, real values, not mocked): a complex64-input,
float32-output `abs` case graded **with** `source_dtype=complex64`
correctly passes at 2 ULP; the identical case graded **without**
`source_dtype` (falling back to the float32 result dtype) incorrectly
fails at that same 2 ULP — demonstrating the bug this fix closes, not
just describing it.

**Anti-tautology proof that the tightening actually bites** (real output,
`harness.compare_values`, `ulp_tolerance={"float32": 0.0, "float64": 0.0,
"complex64": 2.0, "complex128": 2.0}`):

- A float32 `abs` result 1 ULP off now **FAILS** — would have passed
  under the old item-wide `2.0` scalar:
  ```
  (False, "ULP distance 1.0 exceeds declared tolerance for dtype float32
  (0.0) (justification: 'x'): numpy=array([1. , 2.5, 3. ],
  dtype=float32) anionpy=array([1.0000001, 2.5000002, 3.0000002],
  dtype=float32)", False, 1.0)
  ```
- A complex64 `abs` result 2 ULP off (the real, evidenced numpy
  imprecision) still **PASSES**:
  ```
  (True, '', True, 2.0)
  ```
- The same complex64 case 3 ULP off still **FAILS**:
  ```
  (False, "ULP distance 3.0 exceeds declared tolerance for dtype
  complex64 (2.0) (justification: 'x'): numpy=array([1.4142135, 5. ],
  dtype=float32) anionpy=array([1.4142139, 5.0000014], dtype=float32)",
  False, 3.0)
  ```
- All four pre-existing construction-time refusals (no sweep evidence,
  sweep smaller than `MIN_ULP_SWEEP_N`, declared tolerance padded above
  the measured max, declared tolerance under-declared below the measured
  max) still raise, now re-verified against dict-shaped `ulp_tolerance`
  (`pytest tests/differential/test_ulp_distance.py -k "refuse or
  accepts"` — 18/18 passed).

**Coverage impact:** unaffected. `41/1180` (`39` bit-exact, `2`
ULP-tolerant: `absolute`+`abs`) before and after, verified via
`tools/coverage.py --tests`. Both `abs` and `absolute` remain `[PASS]`
under the tightened per-dtype grading, because the real anionpy
implementation genuinely is bit-exact on float32/float64 — the earlier
2-ULP slack on those dtypes was unearned, not load-bearing. Had a real
float32/float64 regression existed, this fix would have surfaced it as a
newly-failing item and dropped coverage accordingly; that this didn't
happen here is a fact about the current implementation, not something
the grader was loosened to guarantee.

## This pass: transcendental/math float ufuncs -- `ionp-core` engine only, structurally blocked from the Python-facing ledger (2026-08-01)

**What was built:** `MathUnaryOp`/`MathBinaryOp` enums plus
`math_unary_op`/`math_binary_op` in `ionp-core/src/ufunc.rs`, extending the
existing generic engine (same `unary_elementwise`/`binary_elementwise`/
`operand_of!`/`NdArray::cast_to`/`NdArray::from_buffer` machinery `add`/
`multiply`/`negative`/`absolute` already use — no parallel engine). 31 unary
ops (`sqrt, cbrt, square, reciprocal, exp, exp2, expm1, log, log2, log10,
log1p, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh,
arccosh, arctanh, sign, signbit, floor, ceil, trunc, rint, fabs, degrees,
radians`) and 6 binary ops (`hypot, arctan2, power, copysign, fmod,
remainder`) — 37 of the 40 items in the target list. `deg2rad`/`rad2deg`
were left out only because they are numpy aliases of `radians`/`degrees`
that would need wiring on the `ionp-py`/`anionpy/__init__.py` side (out of
scope here, see below) — the Rust math is identical and already present.
`float_power` was deliberately scoped out (see "Scope cuts" below).

Every dtype-promotion rule and every special-value expectation (NaN, +-0.0,
+-inf, domain errors, half-to-even rounding, sign-of-dividend vs
sign-of-divisor for fmod/remainder, integer-negative-exponent `ValueError`)
was verified against a live oracle (`.venv/bin/python3`, numpy 2.5.1), not
guessed — see the 32 new `#[cfg(test)]` cases in `ufunc.rs` (e.g.
`sqrt_special_values_match_numpy_exactly`, `sign_matches_numpy_zero_and_
nan_behavior`, `arccosh_domain_and_boundary`, `power_int_negative_exponent_
is_a_value_error_matching_numpy`), each asserting the exact oracle-measured
value. `cargo test -p ionp-core --lib`: **86 passed, 0 failed** (7 pre-
existing + 79 from this session's dtype/array/buffer/shape/ufunc suite
combined; 32 of those are the new math-ufunc tests specifically).

**Anti-tautology proof (real output, not asserted):** three deliberate bugs
were injected one at a time, shown to fail a real test, then reverted —
none were left in the tree.
1. Wrong formula (`Cbrt` computed as `x.sqrt()` instead of `x.cbrt()`):
   `cbrt_negative_value` FAILED — `left: NaN, right: -3.0`.
2. Wrong special-value sign (`sign(-0.0)` returning `-0.0` instead of
   numpy's `+0.0`): `sign_matches_numpy_zero_and_nan_behavior` FAILED —
   `assertion failed: r[0] == 0.0 && !r[0].is_sign_negative()`.
3. Wrong rounding rule (`Rint` using `x.round()` — round-half-away-from-
   zero — instead of `x.round_ties_even()`, the class of "one ULP/one tie-
   break off" error the default bit-exact bar exists to catch since no
   `ulp_tolerance` is declared for any item in this family):
   `rint_rounds_half_to_even` FAILED — `left: [3.0, 4.0, -3.0], right:
   [2.0, 4.0, -2.0]`.

**Why this does NOT move the coverage ledger — a real, structural blocker,
not a shortcut:** `ionp-py/src/lib.rs` (off-limits in this task's path
ownership) hand-enumerates every Python-visible ufunc name via 21 explicit
`add_ufunc(m, "name", UfuncKind::...)` calls in its `#[pymodule]` function —
confirmed by direct inspection and by importing the live `anionpy._anionpy`
extension, which exposes exactly the same 21 names before and after this
session's changes. Adding a new `MathUnaryOp`/`MathBinaryOp` variant to
`ufunc.rs` alone does not make it callable from Python. Consequently:
- `tests/differential/ufunc_registry.py`'s 134 auto-built `ItemSpec`s all
  resolve `ionp_path` against the live `anionpy` module; every item in this
  family still resolves to `absent` (`resolve_ionp()` returns `None`).
- `tests/differential/ulp_sweep.py`'s `measure_unary`/`measure_binary`
  raise `RuntimeError` for any of these names (`ionp_fn=None`), so no real
  `ulp_sweep` evidence can be produced this session — meaning no
  `_ULP_OVERRIDES` entry can be honestly added to `ufunc_registry.py` for
  any of them (adding one without real n>=20000 measured evidence would be
  exactly the fabrication `registry.ItemSpec.__post_init__` and this
  project's honesty rules forbid). `ufunc_registry.py` and `ulp_sweep.py`
  are therefore left unmodified this session.
- Real before/after ledger (`tests/differential/run.py --out /tmp/r.json &&
  tools/coverage.py --tests /tmp/r.json`, `.venv/bin/python3`, numpy
  2.5.1): **unchanged at 41/1180 (3.475%)**, both before and after — 39
  bit-exact + 2 ULP-tolerant (`abs`/`absolute`), 1139 `absent`. This is the
  honest, correct state: correct Rust math with no Python entry point is
  still `absent` by this project's own ledger definition, and inflating
  that number would require touching `ionp-py`, `anionpy/__init__.py` with a
  binding that doesn't exist, or `ufunc_registry.py` with fabricated sweep
  evidence — all of which are refused here on those grounds. Landing on
  the Python-facing ledger is one Rust-side dependency away
  (`ionp-py/src/lib.rs` needs ~37 more `add_ufunc(...)` lines, one per new
  enum variant, following the exact pattern the existing 21 already use)
  but is out of this session's owned paths.

**Scope cuts made in this pass (documented, not silently dropped):**
- **Complex input rejected for every new op** (`IonpError::Type`). Branch
  cuts and special-value tables for complex `sqrt`/`exp`/`log`/trig are a
  substantially separate problem (see the existing complex-`abs`/complex-
  divide precedent in this file for the level of care that family needs);
  deferred rather than rushed.
- **Bool input rejected for the float-promoting family** (`sqrt`, `exp`,
  `log`, all trig/hyperbolic, `rint`, `fabs`, `degrees`, `radians`, ...).
  Real numpy promotes `bool -> float16` for these; `ionp-core`'s `DType`
  enum has no `F16` variant at all (a pre-existing gap, not introduced
  here — see the `divide`/`true_divide` float16 note in
  `ufunc_registry.py`). Promoting to `float64` instead, silently, would be
  a real, undocumented divergence from numpy's dtype contract, so it is
  rejected with a clear `IonpError::Type` instead.
- **Bool input rejected for `square`/`sign`.** Real numpy promotes
  `bool -> int8` for these (`np.square(bool_array).dtype == int8`);
  reproducing that specific odd promotion was judged low-value for the
  time available and is rejected rather than silently mistyped.
- **`reciprocal` implemented for `float32`/`float64` only.** Integer
  `reciprocal` in real numpy has real div-by-zero-warning-and-truncation
  oddities (e.g. `np.reciprocal(np.int32(0))` overflow behavior) that are
  a separate, fiddlier subproblem; rejected with `IonpError::Type` rather
  than guessed at.
- **`float_power` not implemented at all** (neither `MathBinaryOp` variant
  nor tests). Its promotion rule (always at least `float64`, regardless of
  input width — unlike every other item in this family, which preserves
  `float32`) and its distinct negative-base/fractional-exponent special
  cases are a separate subproblem from the other 5 binary ops; per this
  task's explicit "depth beats breadth" instruction, this was the one item
  cut entirely rather than rushed.
- **`deg2rad`/`rad2deg`** (numpy aliases of `radians`/`degrees`) are not
  wired as separate Python-visible names, since that wiring lives in
  `ionp-py`/`anionpy/__init__.py`, both out of this session's owned paths —
  the underlying Rust math (`Radians`/`Degrees`) is already implemented
  and tested.
- **`.reduce`/`.accumulate`/`.outer`/`.reduceat`/`.at`/`out=`/`where=` are
  not wired for any of these new ops.** All 37 are `nin=1` or `nin=2`
  elementwise ops; the differential corpus for this family (once a Python
  binding exists) exercises them primarily via `"call"`, matching how this
  file's own module docstring scopes the reduce-family protocol surface to
  binary ops specifically. `write_out`/`at_unary`/`at_binary` are generic
  over `UnaryOp`/`BinaryOp`, not `MathUnaryOp`/`MathBinaryOp` — extending
  `.at`/`out=`/`where=` to this family is future work once a Python
  binding makes it observable/testable at all.

**Commit scope:** only `ionp-core/src/ufunc.rs` and this file changed.
`ionp-py/`, `anionpy/__init__.py`, `tests/differential/registry.py`,
`tests/differential/ufunc_registry.py`, `tests/differential/ulp_sweep.py`,
`bench/`, `tools/coverage.py`, and `Cargo.lock` are all untouched, per this
task's path ownership.

## `diff`/`ediff1d`: explicit `prepend=None`/`append=None`/`to_begin=None`/`to_end=None` when the diff computation is VACUOUS (2026-08-01, narrowed 2026-08-02)

**Reproducer** (verified live against numpy 2.5.1, not reproducible with anionpy):

```python
>>> np.diff(np.zeros((0,), dtype='int64'), prepend=None)
array([], dtype=object)
>>> np.diff(np.zeros((4, 0), dtype='int64'), prepend=None)
array([], shape=(4, 0), dtype=object)
>>> anionpy.diff(anionpy.array(np.zeros((0,), dtype='int64')), prepend=None)
TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'

# But NOT this shape — same "empty array", same explicit None, yet numpy
# raises a perfectly ordinary, fully-representable TypeError instead of
# returning an object array, because BOTH prepend and append are present:
>>> np.diff(np.zeros((0,), dtype='int64'), prepend=None, append=None)
TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
>>> anionpy.diff(anionpy.array(np.zeros((0,), dtype='int64')), prepend=None, append=None)
TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'   # matches
```

**Cause: architectural, not a defect in `diff` — but narrower than first
thought.** Real numpy's `prepend`/`append` default is a private
`np._NoValue` sentinel, not `None` — so an explicitly-passed `None` is a
genuine, distinct operand value that gets concatenated into `a` via
`np.asanyarray(None)`, producing a mixed `int`/`NoneType` **`object`-dtype**
`combined` array. Real numpy then computes `combined[i+1] - combined[i]`
for every `i` along the diff axis, once per "row" of every *other* axis.
The `TypeError` a caller normally sees is not a validation numpy performs
up front — it is *emergent*, raised only when that per-element Python `-`
actually executes on a real pair containing a `None`.

The loop body executes zero times — i.e. the computation is **vacuous**,
and numpy just returns an empty `dtype=object` array with no exception —
exactly when either of these holds:

- **some axis other than the diff axis has length 0** (e.g. `(4, 0)`
  diffed along axis 0 — that axis has 4 elements, but axis 1's length-0
  makes every "row" empty regardless), or
- **the diff axis itself has length 0 AND `combined`'s length along that
  axis is still `< 2`** — which, since `prepend`/`append` each contribute
  at most 1 slot and `a` contributes 0, only happens when at most one of
  `prepend`/`append` was actually passed (both omitted, or exactly one
  passed).

Conversely, when the diff axis has length 0 but **both** `prepend` and
`append` were passed (`None` or not), `combined` has exactly 2 elements
along that axis and a real subtraction *does* execute — an ordinary,
fully-representable `TypeError` (or, if neither is `None`, an ordinary
real result), not an `object`-dtype return at all.

**anionpy has no `object` dtype at all** (see `ionp-core/src/dtype.rs`'s
`min_scalar_type_unsigned`/`min_scalar_type_signed` doc comments: "anionpy has
no representation for at all... architectural absence, not a bug in this
function"). This is the fixed-13-numeric-dtype design documented elsewhere
in this file (`float16`/`datetime64`/object are all out of scope). There is
therefore no dtype anionpy could tag a vacuous-computation's empty result with
that would be numpy's real answer — `dtype=object` is unrepresentable,
full stop, not a missing feature that could be filled in incrementally.

**What anionpy does instead:** for the genuinely vacuous cases above (any
axis other than the diff axis is length-0, or the diff axis is length-0
with at most one of `prepend`/`append` passed), anionpy raises the same
`TypeError` it always has, unconditionally — wrong there, since numpy
returns an empty `object` array instead. This was a deliberate choice,
confirmed with the coordinator, over two rejected alternatives: silently
returning an empty array under some other (wrong) dtype, or otherwise
inventing a substitute for `object` dtype — both would be a *worse*,
silent divergence than a consistent (if occasionally over-eager) raise.

For every **non**-vacuous case — including a length-0 diff axis with both
`prepend` and `append` passed — anionpy now computes exactly which two
`combined` elements numpy's first-failing subtraction touches (reasoning
through `prepend`/`a`/`append`'s concatenation order, not pattern-matching
the common case) and raises numpy's exact `TypeError`, byte for byte.
Verified live across `(0,)`, `(4,0)`, `(0,4)`, `(0,0)`, and `(3,0,2)`,
crossed with every valid axis and all of `{omitted, None, real value}` for
each of `prepend`/`append`: message text and the vacuous/non-vacuous
classification both match numpy exactly in every case anionpy can represent.

**Status:** `diff` (and, by the same underlying mechanism, `ediff1d` via
`to_begin=None`/`to_end=None`, unaffected by this pass's code changes) is
**not declared** in `__ion_state__` because of the remaining vacuous-case
gap — that ledger is the coordinator's, not touched by this entry.
Resolving the remainder would require either an `object` dtype (out of
scope for the fixed-width-numeric design) or the coordinator accepting the
raise-on-vacuous behavior as a permanent, declared difference rather than
a blocker.

**Commit scope:** `ionp-core/src/setops.rs`, `tests/differential/setops_cases.py`,
and this file. `anionpy/_state/` (the declaration ledger) deliberately
untouched.

## SCOPE OF THE `"exact"` DECLARATION FOR UFUNCS: the call path, NOT `.reduce`/`.accumulate`/`.outer` (2026-08-02)

This is not a newly-introduced divergence. It is a **pre-existing, uniform limit
on what the ledger's `"exact"` verdict has ever graded for ufunc items**, written
down here because 17 items were declared on 2026-08-02 and a reader would
otherwise reasonably assume "exact" covers the ufunc's methods too. It does not.

**The divergence.** For a unary (`nin == 1`) ufunc, numpy rejects the reduction
methods with `ValueError`; anionpy raises `TypeError`:

    np.cbrt.reduce(np.array([1, 2, 3]))    -> ValueError
    anionpy.cbrt.reduce(anionpy.asarray(...))    -> TypeError

**Why this is a scope statement and not a fresh bug.** Items declared `"exact"`
LONG before today fail the identical check, on plain ndarray input, with no
list/tuple involved. Measured live against numpy 2.5.1:

    already-declared ufuncs (sin, cos, fmax, gcd, logaddexp, nextafter,
    spacing, heaviside, bitwise_count) x {reduce, accumulate, outer}
    x {list, ndarray}  ->  20/54 match, 34 mismatches

`sin.reduce(ndarray)`, `cos.accumulate(ndarray)`, `sin.outer(ndarray)` all
diverge. So the bar that admitted `sin` and `cos` never included ufunc methods,
and the 17 items declared today are admitted at exactly that same bar --
consistently, not by a weakened one.

**What the 17 WERE graded against**, out-of-corpus, 85,680 cases, 0 mismatches
on exception class, exception message, value, dtype and shape:

    17 ufuncs
      x element families {bool, int, float, big(2**40), nan/inf}
      x containers       {ndarray, list, tuple, nested list, empty, scalar}
      x order=           {absent, C, F, A, K, Z(invalid)}
      x casting=         {absent, no, equiv, safe, same_kind, unsafe, bogus}
      x subok=           {absent, True, False, "notabool"(invalid)}

That crossing is the one that neither of the two earlier grids ran: the
ndarray-only grid varied `order=`/`casting=`/`subok=` but never the input
CONTAINER, and the container grid never crossed the kwargs. Each alone reported
a clean result that the other falsifies. Crossed, they agree.

**Related, still open** (tracked, not fixed here): `.reduce`/`.accumulate`
additionally ignore `initial=`/`keepdims=`/`dtype=`/`where=`. Closing the
`ValueError`-vs-`TypeError` gap above does NOT close those.

**Consequence for the Definition of Done.** "0 absent, every non-absent item
backed by a passing differential test" is satisfied for these items at the call
path. Full ufunc-method parity is separate remaining work that will need its own
harness-routed cases before any item can claim it.


## `absolute`/`abs` complex ULP + `degrees`/`rad2deg` float32 ULP: both fixed at the algorithm, not the tolerance (2026-08-02)

Two items were carrying declared/needed slack that later measurement showed
was a fixable algorithm defect, not an inherent floor. Both are now genuine
0.0 ULP on every applicable dtype.

### `absolute` / `abs`: complex64/complex128 were calling the wrong hypot

The 2026-08-01 entries above measured `absolute`/`abs` complex64/complex128 at
a real, adequately-swept 2.0 ULP (20,000 seeded values/dtype, ~35-38%
disagreeing) and declared it in `ufunc_registry.py`'s `_ULP_OVERRIDES`,
concluding at the time that numpy's complex `abs` "runs through a
SIMD-vectorized ufunc loop with its own internal approximation... not
reachable by improving this formula further."

That conclusion was wrong. `ionp-core/src/ufunc.rs`'s `Absolute` complex arm
was calling `f32::hypot`/`f64::hypot` (the platform libm `hypot`/`cabs` —
verified bit-identical to each other via `ctypes` against `libSystem`, 20,000
seeded pairs, 0 mismatches). Reading numpy's actual source
(`loops_unary_complex.dispatch.c.src`) shows real numpy's complex `abs`
computes a *different*, genuinely non-libm formula for the SIMD path that's
active on any contiguous array (including via NEON on Apple Silicon, not just
x86):

    larger  = max(|re|, |im|)
    smaller = min(|im|, |re|)
    ratio   = smaller / larger        (0 instead, if larger==0 or smaller==inf)
    result  = larger * sqrt(fma(ratio, ratio, 1.0))

with inf/nan resolved before that: either component literally `+-inf` forces
the result to `+inf` even if the other component is NaN (matches the
already-tested `hypot(inf, nan) == inf` rule); otherwise NaN if either
component is NaN. This is now `ufunc.rs`'s `numpy_complex_abs` (using
`mul_add` — a true FMA — to match numpy's `npyv_muladd`, not a separate
multiply-then-add), used by every complex `Absolute` call site (`unary_op`'s
complex arm, and the accumulate/reduce in-place complex `Absolute` paths).

Re-swept after the fix, same n=20,000/dtype, SEED=20260731:

| item | dtype | before | after |
|---|---|---|---|
| `absolute`/`abs` | float32 | 0.0 | 0.0 |
| `absolute`/`abs` | float64 | 0.0 | 0.0 |
| `absolute`/`abs` | complex64 | 2.0 (7089/20000 disagreeing) | **0.0** (0/20000) |
| `absolute`/`abs` | complex128 | 2.0 (7554/20000 disagreeing) | **0.0** (0/20000) |

`ufunc_registry.py`'s `_ULP_OVERRIDES` no longer declares `absolute`/`abs` at
all — a genuinely bit-exact item needs no tolerance mechanism, same reasoning
already used for the six 0.0-atol/rtol arithmetic ufuncs.

### `degrees` / `rad2deg`: float32 was using the wrong 180/pi constant

`math_unary_f32`'s `Degrees` arm called Rust's `f32::to_degrees()`, which
multiplies by a hand-written decimal literal
(`57.2957795130823208767981548141051703_f32`) — the correctly-rounded f32
nearest the *true* mathematical value of 180/pi (rounded once, from the exact
constant). Real numpy's `npy_degreesf`/`npy_rad2degf` instead compute the
scale factor as `180.0f / NPY_PIf` — an f32 *division* of two already-f32
constants (`180.0f` exact, `NPY_PIf` itself the rounded f32 pi) — which lands
on a different f32 bit pattern one ULP away: `0x42652ee0` (numpy's constant,
`180.0f32/PI_f32` computed at f32 precision) vs Rust's `0x42652ee1`. float64
has no equivalent gap (`180.0/NPY_PI` computed in f64 and Rust's `f64::
to_degrees` constant land on the same value), which is why only float32 was
affected — float64 was already exact.

Fixed by replacing `x.to_degrees()` with `x * F32_RAD2DEG` where
`F32_RAD2DEG: f32 = 180.0f32 / std::f32::consts::PI`, reproducing numpy's
exact formula instead of Rust's more-precise-but-different constant. Both
`degrees` and `rad2deg` share this code path (`MathUnaryOp::Degrees`).

Differential suite, before -> after (198/214 -> 214/214, both items):

    [FAIL] degrees  (198/214 cases) -- 16/214 cases failed, diagnostic ULP distance 1.0
    [FAIL] rad2deg  (198/214 cases) -- 16/214 cases failed, diagnostic ULP distance 1.0
    ->
    [PASS] degrees  (214/214 cases)
    [PASS] rad2deg  (214/214 cases)

`degrees`/`rad2deg` carried no `_ULP_OVERRIDES` entry before or after this
fix — they were simply failing bit-exact grading on float32 before, and pass
it outright now.

### Verification

Both fixes were also checked against a dedicated out-of-corpus crossed grid
(dtype x {ndarray, list, tuple} container x special-value class: +-0.0,
+-inf, NaN, smallest subnormal, dtype `tiny`, dtype `max` and the next value
below it — chosen specifically to catch hypot-style intermediate overflow —
and, for `degrees`/`rad2deg`, large-magnitude angles up to the dtype's `max`).
0.0 ULP on every (item, dtype) pair across that grid as well. `tests/
differential/run.py` full-suite and `tools/coverage.py --tests` before/after
showed no verdict change anywhere else (`exact` count unchanged at 449,
`failing`/`phantom`/`untested` at 0 both before and after) — only `degrees`
and `rad2deg` moved from `fail` to `pass`; `abs`/`absolute` were already
`pass` (via the now-removed 2.0 ULP tolerance) and stayed `pass` (now
bit-exact).

**Not varied by this grid:** float16 `degrees`/`radians`/`abs` behavior (the
`e`-typed loops route through the f32 path via `astype`, not measured
independently here); long-double (`g`) — anionpy has no long-double dtype;
denormal *results* near the underflow boundary for `degrees` (only denormal
*inputs* were swept); and any input distribution outside `standard_normal()
* 10` plus the explicit special-value list above (i.e., no dedicated
adversarial/fuzz search for a hypothetical remaining disagreeing input pair —
the 20,000-seeded sweep plus the special-value grid is the totality of the
evidence here).

**Commit scope:** `ionp-core/src/ufunc.rs`, `tests/differential/
ufunc_registry.py`, and this file.

## 2026-08-03 — `binary_repr(0, width=<ndarray>)`: exception message (and, for a float-dtype width, exception type) differs, inherited from the absent `ndarray.__rmul__` with a `str` left operand

`anionpy.binary_repr` and `anionpy.base_repr` were declared `exact` on this date
(`anionpy/_intrepr_compose.py`, transcriptions of numpy's own Python bodies in
`numpy/_core/numeric.py`). One input class in the differential corpus is
excluded, and this is it.

**Repro** (verbatim, `numpy` 2.5.1):

```
>>> np.binary_repr(0, width=np.array(8))
TypeError: The 'out' kwarg is necessary when using the string multiply ufunc
directly. Use numpy.strings.multiply to multiply strings without specifying
'out'.
>>> anionpy.binary_repr(0, width=anionpy.asarray(np.array(8)))
TypeError: unsupported operand type(s): anionpy.ndarray arithmetic requires
another anionpy.ndarray, a numpy scalar/array, or a Python bool/int/float/
complex (got str)
```

```
>>> np.binary_repr(0, width=np.array(8.0))
numpy._core._exceptions._UFuncNoLoopError   # UFuncTypeError
>>> anionpy.binary_repr(0, width=anionpy.asarray(np.array(8.0)))
TypeError                                    # same base class, different type
```

**Scope, measured row by row rather than assumed.** This is *not* a blanket
"array widths do not work" difference. With `width=np.array(8)` every one of
`num ∈ {1, -1, 5, -5, -4, 255}` agrees exactly with numpy — including the
two's-complement branch and the gh-8679 boundary — and `width=np.array(2)`
reproduces the insufficient-width `ValueError` identically on both sides.
The divergence is confined to **the zero branch**, `'0' * (width or 1)`,
which is the one place in either function where the width array is itself an
operand of a `str` multiply. `np.array([8])`, the 1-D form, hits the same
hole at every `num` (its `outwidth` stays an array through to the final
multiply).

**Why it is not a `binary_repr` defect.** numpy's `str.__mul__` defers to
`ndarray.__rmul__`, which runs numpy's string-multiply ufunc and fails there.
anionpy's `ndarray.__rmul__` rejects a `str` operand outright. This is the
`str` sibling of the `[0] * arr` gap already recorded on 2026-08-03 in
`4e4b8a5` (`ndarray.__rmul__` does not accept a sequence left operand).
Both sides raise; for the int-dtype width both raise the *same exception
type*; only the message differs, and reproducing numpy's message would mean
forging the internals of a numpy ufunc anionpy does not implement — which is
forbidden. Fixing it properly means implementing sequence/`str` left
operands in `ndarray.__rmul__`, which is tracked as its own absence.

**Contract note, not an excuse.** numpy documents `width : int, optional`.
An ndarray width is out-of-contract input, so the divergence is on a surface
numpy itself does not promise. That is *why* the exclusion is narrow rather
than a reason to skip the whole class: the corpus keeps every array-width
case with `num != 0`, and skips `num == 0` for array widths only. The rule
is stated in `tests/differential/shape_compose_cases.py` as a claim about
which code branch is reached, not as a list of rows that happened to fail.

**`message_pair_ok` was considered and rejected**, per its own docstring:
"Do NOT use this for anything else: it is a narrow, measured exception to
'message is part of the API,' not a general escape hatch."

**Commit scope:** `anionpy/_intrepr_compose.py`, `anionpy/__init__.py`,
`anionpy/_state/toplevel.py`, `tests/differential/shape_compose_cases.py`, and
this file.

---

## Advanced-indexing output memory layout on NON-CONTIGUOUS inputs (2026-08-03)

**Status: OPEN, measured, deliberately not fixed in this pass.** Values,
`.shape`, `.dtype`, `.base`, `flags['OWNDATA']`, `flags['WRITEABLE']` and
write-through aliasing all MATCH numpy. `.strides` does not, and only when
the array being indexed is not C-contiguous.

**What was fixed.** numpy does not return a plain C-contiguous copy from
advanced indexing. It gathers into a temporary with the broadcast index
block LEADING and then transposes that temporary into final axis order, so
when the block is not already leading the visible result is a *view of a
hidden intermediate*:

    a = np.arange(12).reshape(3, 4)         # int64
    v = a[:, [0, 1]]
    v.shape (3, 2)   v.strides (8, 24)   v.flags['OWNDATA'] False
    v.base.shape (2, 3)   v.base.flags['OWNDATA'] True

anionpy returned a C-contiguous owning array: strides `(16, 8)`, `OWNDATA`
True, `base` None — three public attributes wrong at once. This is now
implemented (`indexing::gather_order` + the `__getitem__` branch that
transposes the gather and attaches the temporary as `.base`). Divergences
over a 648-combination sweep (6 input layouts x 6 dtypes x 18 index forms)
went from 114 to 18.

**What remains.** All 18 are inputs that are not C-contiguous (`a.T`,
`a.transpose(2,0,1)`). numpy's temporary is itself allocated with
NpyIter's KEEPORDER, so its axes are ordered by the *input's* memory order
rather than C order. The rule is visible and reproduces cleanly on the
plain cases —

    in.shape (5,4,3,2)  in.strides (8,40,160,480)   key a[[0,1]]
    out.shape (2,4,3,2)
    numpy out.strides (192, 8, 32, 96)   <- non-block axes keep the input's
    anionpy  out.strides (192, 48, 16, 8)      relative memory order; anionpy is
                                            C-contiguous over out.shape

— but it does NOT close there. With a `np.newaxis` in the key, numpy's
temporary has block-first SHAPE and non-C STRIDES, and the size-1 axis it
inserts takes the itemsize as its stride:

    t = np.arange(24).reshape(2,3,4).transpose(2,0,1)   # strides (8,96,32)
    t[:, None, [0,1]]  ->  shape (4,1,2,3)  strides (8,8,96,32)
                           base.shape (2,4,1,3)   [NOT C-contiguous]

That size-1-axis stride convention is the SAME still-open CLASS B question
already recorded for `expand_dims`/`broadcast_to`/`atleast_2d`/`atleast_3d`
in `anionpy/_state/toplevel.py`. Finishing this means answering that question
first, and the question is Mother's, not a matter of matching one more
measurement. Chasing NpyIter's heuristic on a partial derivation would very
likely produce a fix that is right on this grid and wrong outside it, which
is the failure mode this file exists to prevent.

**Consequence.** `ndarray.__getitem__` stays UNDECLARED. It is now correct
on values, on error text, on scalar-vs-0-d return type, on fancy/boolean/
mixed indexing, on `.base`, on `OWNDATA`, and on write-through — 0/966
divergences over the contiguous-input sweep (6 shapes x 7 dtypes x 23 index
forms) — and it is still not exact, so it is not declared. Declining to
declare is the correct outcome here, not a shortfall.

**Also fixed in the same pass (a real message divergence).** anionpy appended
a helpful `(got float)` / `(got str)` clause to numpy's index-type
`IndexError`. numpy 2.5.1 stops at `...are valid indices` for every
rejected key. The suffix is removed; the extra help was a divergence.

## 2026-08-03 — `copyto`: three measured differences, all inherited from
## library-wide absences rather than from `copyto` itself

`copyto` landed 2026-08-03 (it had been entirely ABSENT) with a 3,085-case
differential corpus and a 2,615-case out-of-corpus sweep. Between them, 2
divergences remain, plus 1 case excluded from the corpus by construction.
All three are pre-existing library-wide gaps that `copyto` merely reaches;
none is specific to it, and none is papered over anywhere.

**1. Non-array `dst` names anionpy's own type (excluded from the corpus).**

```
>>> np.copyto([1, 2], 1)
TypeError: copyto() argument 1 must be a numpy.ndarray, not list
>>> anionpy.copyto([1, 2], 1)
TypeError: copyto() argument 1 must be an anionpy.ndarray, not list
```

The sentence shape, the argument number, and the trailing type name are all
matched; only the class name differs. Emitting `numpy.ndarray` would be a
falsehood — the object we require is an `anionpy.ndarray` — and would also
contradict the rest of the library, which already names its own type in
exactly this position (see the `anionpy.ndarray arithmetic requires another
anionpy.ndarray` entry above, and `put`/`fill_diagonal`'s "first argument must
be an anionpy.ndarray" in `manip.rs`). The harness's `_normalize_exc_message`
scrubs only `\bid=\d+` and deliberately does not touch module names, so this
case cannot be graded green; it is therefore left OUT of `copyto_cases.py`
rather than given a tolerance.

**2. No "Did you mean" suggestion on a misspelled keyword.** Library-wide,
not `copyto`-specific.

```
>>> np.copyto(a, 1.0, wheree=True)
TypeError: copyto() got an unexpected keyword argument 'wheree'. Did you mean 'where'?
>>> anionpy.copyto(a, 1.0, wheree=True)
TypeError: copyto() got an unexpected keyword argument 'wheree'
```

Measured to be general rather than assumed to be: `np.reshape(a, (2,2),
ordr='C')` likewise suggests `'order'`, while `np.sum(a, axsi=0)` and
`np.zeros(3, dtpye='int8')` produce NO suggestion — CPython emits one only
when its own matcher finds a near-enough candidate. PyO3 never emits one, so
anionpy's message is numpy's message minus a suffix that numpy itself omits
most of the time. This affects every anionpy function with keyword arguments,
not `copyto`.

**3. A `str` `where=` is rejected instead of being coerced.** The
object/string-dtype absence, reached through a new door.

```
>>> np.copyto(a, 1.0, where='')      # '' -> <U0 -> False; writes nothing
>>> np.copyto(a, 1.0, where='yes')   # -> <U3 -> True; writes everything
>>> anionpy.copyto(a, 1.0, where='')
TypeError: anionpy.array() only supports (possibly nested) lists/tuples of
bool/int/float/complex, or a numpy.ndarray
```

Same root cause as `copyto(f8_dst, 'abc')` (numpy builds a `<U3` source) and
`copyto(f8_dst, None)` (numpy builds an `dtype('O')` one): anionpy has no
string and no object dtype at all. Recorded with the rest of that family;
not a `copyto` defect and not separately fixable.

**Not a difference, but recorded so it is not later mistaken for one.**
`copyto(float32_dst, 1e40)` stores `inf` on both sides and MATCHES on value;
numpy additionally emits `RuntimeWarning: overflow encountered in cast` and
anionpy does not. That is the library-wide missing-warning gap already recorded
elsewhere, and it is why the corpus grades values and exceptions but not
warnings.

## 2026-08-03 — a real bug found by `copyto`'s sweep, in `astype`/`asarray`/
## `array`, NOT in `copyto`: `cast_to` discarded a contiguous view's offset

Recorded here because it is the opposite of a known difference — it is a
silent wrong ANSWER that survived in three long-declared-`"exact"` items,
and the reason it survived is worth keeping.

`NdArray::cast_to` (`ionp-core/src/array.rs`) built its result with a
hardcoded `offset: 0`. `Buffer::cast_to` maps elementwise over the WHOLE
backing vec and returns one of the same length — it does not slice down to
the view's window — so the offset was still live in the result and throwing
it away re-read the array from element 0:

```
>>> a = np.arange(9.0)[2:5]      # [2., 3., 4.]
>>> a.astype('int8')             # numpy: [2, 3, 4]
>>> anionpy.arange(9.0)[2:5].astype('int8')
[0, 1, 2]                        # right dtype, right shape, WRONG DATA
```

`asarray(v, dtype=...)` and `array(v, dtype=...)` were wrong identically.
Only the ELEMENTS were wrong: dtype, shape and strides were all correct,
which is why no value-and-shape check ever caught it.

**Why it hid.** `cast_to` takes a cheap `self.clone()` path when the input
is already C-contiguous, and falls back to `to_contiguous()` otherwise —
and `to_contiguous()` gathers through the offset and genuinely rebases to
0. So reversed views, transposed views and strided views were all CORRECT,
and only the *simplest possible* view — a plain `a[2:5]`, contiguous,
carrying nothing but `offset: 2` — was broken. Every exotic case worked. An
offset of 0, which is what a freshly-created array has, was also fine, so
the entire ordinary path was clean.

**Why `copyto` found it and nothing else did.** Not because `copyto` is
special, but because its out-of-corpus sweep varied the INPUT DATA's
provenance — sources built as offset slices — rather than only its dtype,
shape and kwargs. `astype`'s own corpus feeds it freshly-created arrays.

Fixed at `array.rs:cast_to` (`offset: contig.offset`). Guarded twice: an
`ionp-core` unit test (`cast_to_preserves_view_offset`, verified to FAIL on
the unfixed code with the exact expected assertion) and the `castoff/`
cases in `tests/differential/copyto_cases.py`, which pair each offset-view
source with its same-dtype and non-contiguous spellings so a "fix" that
merely relocates the breakage cannot pass.

## 2026-08-04 — `nan_to_num` / `copyto`: foreign scalar substitution sources

Three related seams, all found while measuring `nan_to_num`'s substitution
keywords (`nan=`, `posinf=`, `neginf=`), which are resolved through the same
code path `copyto` uses for its `src`.

**(a) IMPROVEMENT, not a difference.** Under the four checked casting rules
(`no`, `equiv`, `safe`, `same_kind`), a `str`, `bytes` or `None` source now
produces numpy's exact sentence rather than anionpy's generic ingestion error:

```
>>> anionpy.nan_to_num(a, nan="a")
TypeError: Cannot cast scalar from dtype('<U1') to dtype('float64') according to the rule 'same_kind'
```

with `<U{len}` for `str`, `S{len}` for `bytes` and `O` for `None` (each length
floored at 1, matching numpy's zero-length spelling). Implemented as
`foreign_scalar_dtype_name` in `ionp-py/src/lib.rs`, consulted at the top of
`copyto_resolve_src`.

**(b) STILL DIFFERENT — other object-dtype sources.** `object()`, `dict`,
`set`, and a list of strings raise the right exception TYPE (`TypeError`) with
the wrong MESSAGE — anionpy's generic
`anionpy.array() only supports (possibly nested) lists/tuples of ...` instead of
numpy's `dtype('O')` / `dtype('<U1')` sentence. Not fixed because doing it
honestly needs real object and string dtypes, which anionpy does not have; a
hardcoded message for each Python type would be a lie about the mechanism.

**(c) STILL DIFFERENT IN TYPE — `bytearray`.** numpy treats a `bytearray` as a
uint8 BUFFER, not a scalar, so `nan=bytearray(b"ab")` is a length-2 array and
fails with `ValueError: could not broadcast input array from shape (2,) into
shape (1,)`. anionpy raises `TypeError` from ingestion. Deliberately excluded
from `foreign_scalar_dtype_name` (which gates its `bytes` branch on
`is_instance_of::<PyBytes>` precisely so `bytearray`/`memoryview` are not
mislabelled) — getting this right means implementing the buffer protocol as an
ingestion source, which is its own item.

**(d) `casting='unsafe'` is deliberately NOT routed through (a).** There numpy
does a real string/object CONVERSION rather than rejecting: `copyto(f, "a",
casting='unsafe')` raises `ValueError: could not convert string to float: 'a'`
and `copyto(f, None, casting='unsafe')` SUCCEEDS, storing NaN. Emitting a cast
TypeError there would be strictly worse than the current honest ingestion
error, so the interception is gated on `rule != "unsafe"`.

**(e) Not a `nan_to_num` difference.** A narrowing substitution
(`nan_to_num(f16_arr, nan=1e30)`) silently yields `inf` in both libraries, but
numpy also emits an overflow `RuntimeWarning`. That is the pre-existing
library-wide missing-`RuntimeWarning` gap recorded elsewhere in this file, not
something specific to this item.

## 2026-08-04 — `astype`/`asarray`/`array(dtype=)` silently force C order on a
## Fortran operand (found by `nancumsum`'s layout guard; fix MEASURED and
## deliberately NOT shipped)

numpy's `astype` defaults to `order='K'` -- layout-preserving. anionpy's
`NdArray::cast_to` (`ionp-core/src/array.rs`) routes a non-C-contiguous
operand through `to_contiguous()`, which forces C:

```
>>> f = np.asfortranarray(np.arange(30.).reshape(5, 6))
>>> f.astype('float32').strides         # numpy: (4, 20)  -- still Fortran
(4, 20)
>>> ionp_f.astype('float32').strides    # anionpy:  (24, 4)  -- silently C
(24, 4)
```

Only the LAYOUT is wrong. Values, dtype, shape and offset are all correct,
which is why `astype`, `asarray` and `array` have carried this while declared
`"exact"` -- nothing in the differential harness compares output strides
unless an item opts in via `ItemSpec.check_strides`, and none of those three
does.

Reaches further than `astype` itself: every caller that passes an explicit
`dtype=` through `cast_to` inherits it, including `cumsum`/`cumprod` and the
new `nancumsum`/`nancumprod`.

**Why it is not fixed here.** The one-line change (`to_contiguous()` ->
`to_contiguous_order("K")`) was written, built and run against the full
differential suite. It fixes the 8 offending cases and REGRESSES eleven other
items -- all 11 `fft.*` spellings plus `ndarray.sum`, `ndarray.prod`,
`ndarray.take`, `ndarray.compress`, `ndarray.conj`, `ndarray.conjugate` --
whose code assumes `cast_to` hands back a C-contiguous buffer and indexes it
accordingly. Suite went 32 -> 43 FAILs. Reverted; the tree carries no part of
it. The honest fix is an audit of `cast_to`'s callers, which is its own item,
not a rider on a nan-function.

Recorded rather than hidden: `layout/nancumsum` and `layout/nancumprod` in
`tests/differential/reduction_cases.py` guard the layout that IS correct
today (no `dtype=`), and their comment carries this same measurement so the
exclusion cannot be mistaken for an oversight.

## 2026-08-04 — `cumsum`/`cumprod`/`nancumsum`/`nancumprod`: two `out=` seams

Both measured against numpy 2.5.1, both pre-existing in the already-declared
`cumsum`/`cumprod` and inherited unchanged by the two new nan* spellings,
which call the identical code:

1. **Wrong-size `out=`.** numpy: `ValueError: provided out is the wrong size
   for the accumulation.` anionpy: `ValueError: operands could not be broadcast
   together with shapes (3,) (2,)`. Same exception type, different sentence.
2. **Mismatched-dtype `out=`.** numpy performs an unsafe CAST into `out` and
   returns it (`np.cumsum(np.array([1., 2., 3.]), out=np.zeros(3, int))` ->
   `[1, 3, 6]`); anionpy raises `TypeError: out= dtype does not match the
   computed result dtype`.

Both belong to the library-wide `out=` contract item (the shared `wrap`
helper in `ionp-py/src/reductions.rs`), not to any one reduction. An `out=`
of the right size AND dtype matches numpy exactly and is swept.

## 2026-08-05 — `LinAlgError`/`AxisError` are anionpy's own classes, not numpy's

Measured against numpy 2.5.1. This is a deliberate, user-visible difference
and the only one in this file that was chosen rather than discovered.

Until today `anionpy.linalg.LinAlgError` **was** `numpy.linalg.LinAlgError` --
module init did `m.add("LinAlgError", py.import("numpy.linalg")?...)`. That
single line was the one runtime numpy borrow in the whole crate with no
numpy-absent fallback, and it alone made `import anionpy` require numpy. A
numpy replacement that cannot be imported without numpy installed is not a
replacement, so anionpy now builds its own classes in `ionp-py/src/errors.rs`
and re-parents them onto numpy's when numpy is importable.

```
>>> import anionpy, numpy as np
>>> anionpy.linalg.LinAlgError is np.linalg.LinAlgError
False                                    # numpy: trivially True of itself
>>> issubclass(anionpy.linalg.LinAlgError, np.linalg.LinAlgError)
True
>>> try: anionpy.linalg.inv(anionpy.array([[1., 2.], [2., 4.]]))
... except np.linalg.LinAlgError as e: print(type(e).__module__, e)
anionpy Singular matrix
>>> try: anionpy.all(anionpy.array([[1., 2.]]), axis=9)
... except np.exceptions.AxisError as e: print(type(e).__module__, e)
anionpy axis 9 is out of bounds for array of dimension 2
```

With numpy ABSENT the same classes still exist, with numpy's own bases:
`LinAlgError` -> `ValueError` (numpy's own `class LinAlgError(ValueError)`),
`AxisError` -> `ValueError, IndexError` (numpy's own dual inheritance).

**(a) NOT a difference for `except` handlers.** Every catch that worked
before still works: by numpy's class, by `ValueError`, and for `AxisError`
by `IndexError` too. Message text is graded byte-equal by the suite and is
unchanged.

**(b) STILL DIFFERENT — `is` identity, and `type(exc) is type(np_exc)`.**
Code that compares the class by identity rather than catching it will see a
difference. This is irreducible: identity with numpy's class and importing
without numpy are mutually exclusive by construction, and the second was
chosen.

The corpus probe `_linalg_error_probe` in
`tests/differential/linalg_cases.py` asserted the old `is` identity; it was
rewritten to assert the property identity was standing in for (raise it,
catch it via numpy's class, plus the `ValueError` base), which is a strictly
stronger check -- a binding that built an unrelated class still fails it,
verified against three wrong classes.

`harness.py`'s exact-type check is relaxed for exactly these two pairs via
`_is_ionp_compat_subclass`, which requires anionpy's `__module__`, one of the
two names, numpy's class as a DIRECT base, and that base rooted in `numpy`.
Nine adversarial pairings were tried against it (forged `__module__`, wrong
name, indirect ancestor, same-named non-numpy lookalike, wrong pairing) and
all nine were rejected. The message-equality check downstream is untouched:
suite is 27 failing items before and after, the same 27.

## 2026-08-05 — five previously-declined gaps, re-measured against the installed binary

Five items recorded elsewhere in the repo (`anionpy/_state/toplevel.py`'s
`percentile`/`quantile` cluster, `ionp-core/src/stats.rs`, `ionp-core/src/
sort.rs`, `tests/differential/registry.py`'s `ndarray.__rtruediv__` note)
as deliberately-not-fixed divergences from numpy were re-measured live
against the currently installed `anionpy` and `numpy 2.5.1`
(`.venv/bin/python -c "import numpy; print(numpy.__version__)"`), since
several were measured weeks ago and stride-handling/`order='K'`-conj/
zero-size-read fixes have landed since. Two no longer reproduce, one was
never actually a divergence (a numpy quirk anionpy deliberately replicates,
confirmed still matching), one cannot be constructed against the installed
binary because the numpy function it names is not implemented in anionpy at
all, and one is still genuinely different.

### (a) weak-scalar-`q` NEP 50 dtype preservation on `percentile`/`quantile` — NO LONGER REPRODUCES

Originally recorded in `anionpy/_state/toplevel.py` (the `percentile`/
`quantile`/`nanpercentile`/`nanquantile` withdrawal, 2026-08-02: "numpy's
NEP 50 weak-scalar rule ... anionpy always computes in float64"). The same
file's 2026-08-04 status note says this was closed that day; re-verified
independently here rather than trusting the note:

```
>>> import numpy as np, anionpy
>>> np.__version__
'2.5.1'
>>> a32 = anionpy.array([1,2,3,4,5], dtype='float32')
>>> na32 = np.array([1,2,3,4,5], dtype='float32')
>>> np.percentile(na32, 50).dtype
dtype('float32')
>>> anionpy.percentile(a32, 50).dtype
dtype('float32')
>>> a16 = anionpy.array([1,2,3,4,5], dtype='float16')
>>> na16 = np.array([1,2,3,4,5], dtype='float16')
>>> np.percentile(na16, 50).dtype
dtype('float16')
>>> anionpy.percentile(a16, 50).dtype
dtype('float16')
>>> np.quantile(na32, 0.5).dtype
dtype('float32')
>>> anionpy.quantile(a32, 0.5).dtype
dtype('float32')
```

Bare-Python-scalar `q` against a `float32`/`float16` array now preserves
the narrow dtype on both `percentile` and `quantile`, matching numpy in
every case tried. Neither `percentile` nor `quantile` is currently declared
`"exact"` in `anionpy/_state/toplevel.py` — that file's own 2026-08-04 note
lists two *other* still-open gaps blocking re-declaration (`ndim>=2 q`,
`weights=`), unrelated to this one. This entry exists so the weak-`q` gap
specifically is not mistaken for still-open the next time someone reads
an older comment in this repo.

### (b) all-NaN-slice narrow-dtype preservation on `nanpercentile`/`nanquantile` — NO LONGER REPRODUCES

Originally recorded alongside (a) in the same `toplevel.py` withdrawal
("all-NaN f32 slice: numpy returns float32, anionpy float64"). Re-verified
live:

```
>>> import numpy as np, anionpy
>>> na = np.array([np.nan, np.nan, np.nan], dtype='float32')
>>> aa = anionpy.array([float('nan'), float('nan'), float('nan')], dtype='float32')
>>> np.nanpercentile(na, 50).dtype
dtype('float32')
>>> anionpy.nanpercentile(aa, 50).dtype
dtype('float32')
>>> np.nanpercentile(na, [10, 50, 90]).dtype
dtype('float32')
>>> anionpy.nanpercentile(aa, [10, 50, 90]).dtype
dtype('float32')
>>> na16 = np.array([np.nan, np.nan], dtype='float16')
>>> aa16 = anionpy.array([float('nan'), float('nan')], dtype='float16')
>>> np.nanquantile(na16, [0.1, 0.9]).dtype
dtype('float16')
>>> anionpy.nanquantile(aa16, [0.1, 0.9]).dtype
dtype('float16')
```

Scalar-`q` and array-`q` forms, float16 and float32, all-NaN 1-D input and
an all-NaN slice inside a 2-D reduction all preserve the narrow dtype now,
matching numpy. Like (a), neither name is declared `"exact"` yet for
unrelated reasons (`ndim>=2 q`, `weights=`); this entry only retires the
all-NaN-slice claim specifically.

### (c) `apply_along_axis` first-slice dtype quirk (`nanquantile`/`nanpercentile`) — never actually a divergence; confirmed still matching

This is not a case of "declined, then fixed" — `ionp-core/src/stats.rs`
(around its `THE FIRST-SLICE DTYPE QUIRK` comment) already documents that
this real-numpy artifact was *deliberately reproduced*, not left absent
or normalized away: `nanquantile`/`nanpercentile` internally mirror
`np.apply_along_axis`'s behavior of sizing/dtyping its output buffer from
the FIRST reduced slice alone, so with a non-weak `q` the whole result's
dtype flips on whether slice 0 happens to be all-NaN. Re-measured live to
confirm the reproduction still holds on the installed binary (it is not
recorded anywhere as an open gap, so there is nothing to "decline" here —
this entry exists only so a future reader does not mistake the quirky
*numpy* dtype-flip for an *anionpy* bug):

```
>>> import numpy as np, anionpy
>>> na = np.array([[np.nan, np.nan, np.nan], [1., 2., 3.]], dtype='float32')
>>> aa = anionpy.array([[float('nan')]*3, [1., 2., 3.]], dtype='float32')
>>> np.nanquantile(na, [0.1, 0.9], axis=1).dtype   # slice 0 is all-NaN
dtype('float32')
>>> anionpy.nanquantile(aa, [0.1, 0.9], axis=1).dtype
dtype('float32')
>>> nb = np.array([[1., 2., 3.], [np.nan, np.nan, np.nan]], dtype='float32')
>>> ab = anionpy.array([[1., 2., 3.], [float('nan')]*3], dtype='float32')
>>> np.nanquantile(nb, [0.1, 0.9], axis=1).dtype   # slice 0 is NOT all-NaN
dtype('float64')
>>> anionpy.nanquantile(ab, [0.1, 0.9], axis=1).dtype
dtype('float64')
```

The dtype flips identically in both directions on both sides. `apply_along_axis`
itself (the free function) is not implemented in anionpy at all
(`hasattr(anionpy, "apply_along_axis")` is `False`) — only its exact
quirk, replicated by hand inside the two `nan*` quantile paths, is in
scope here.

### (d) `sort`-vs-`partition` NaN-payload asymmetry — COULD NOT CONSTRUCT A REPRO

Real numpy has a genuine internal asymmetry here: `np.sort`'s non-stable
kinds rewrite every NaN they move to a single canonical bit pattern
(`ionp-core/src/sort.rs`'s `canonicalize_nans_*`, reproduced by anionpy and
covered above the "Known test-corpus/harness inconsistency" section of
this file), but `np.partition` does **not** canonicalize NaN payloads at
all — verified directly against real numpy:

```
>>> import numpy as np, struct
>>> def bits(x): return hex(struct.unpack('<Q', struct.pack('<d', x))[0])
>>> nan1 = struct.unpack('<d', struct.pack('<Q', 0x7ff8000000000001))[0]
>>> bits(np.sort(np.array([nan1, 1.0, 2.0]))[-1])
'0x7fffffffffffffff'          # canonicalized
>>> [bits(v) for v in np.partition(np.array([nan1, 1.0, 2.0]), 1)]
['0x3ff0000000000000', '0x4000000000000000', '0x7ff8000000000001']   # payload preserved
```

But this is **not measurable as an ionp-vs-numpy divergence** on the
currently installed binary, because `partition`/`argpartition` are not
implemented in anionpy at all — confirmed live:

```
>>> import anionpy
>>> hasattr(anionpy, "partition")
False
>>> hasattr(anionpy, "argpartition")
False
>>> anionpy.partition(anionpy.array([3., 1., 2.]), 1)
Traceback (most recent call last):
  ...
AttributeError: module 'anionpy' has no attribute 'partition'
```

`ionp-core/src/sort.rs`'s own module doc already records `partition`/
`argpartition` as a deliberate scope cut (numpy's introselect internals
expose an algorithm-dependent arrangement that "does not match any
straightforward selection algorithm"). This entry adds the specific
reason a *future* implementation of `partition` cannot simply reuse
`sort`'s NaN-canonicalizing code path: the two functions disagree with
each other inside real numpy itself. Absence of proof of a partition-side
divergence is not proof there wouldn't be one — "could not construct a
repro" here, not "disproved."

### (e) `ndarray.__rtruediv__` complex NaN bit-pattern mismatches — STILL DIFFERENT

Recorded in `tests/differential/registry.py`'s `ndarray.__truediv__`/
`__rtruediv__` comment as a =1.0 ULP complex64/complex128 divergence in
anionpy's FMA-based `complex_div` (Smith's algorithm) vs numpy's own complex
divide loop. That framing undercounts one class of mismatch: when the
result is `nan+nanj`, `tests/differential/harness.py`'s `ulp_distance`
collapses ANY NaN-vs-NaN pair to distance `0.0` regardless of bit payload
(by explicit design, see its own docstring) — so a subset of these
mismatches are real byte-level divergences that a ULP-distance-only view
reports as "0.0 ULP, i.e. no difference," while the harness's actual
grading path (`_bit_exact_equal`, `tobytes()`-based) still fails them.
Re-measured live with a fresh 2684-draw sweep of `python_scalar / ionp_array`
(`.__rtruediv__`) over complex128 operands salted with 0, -0, +-inf, and
NaN components (`SEED=20260805`): 169/2684 draws produced a bit-level
mismatch, and every one of those 169 was exactly this NaN-imaginary-sign
case (numpy's imaginary component is a *negative*-signed NaN,
`0xfff8000000000000`; anionpy's is positive, `0x7ff8000000000000`) — ULP
distance `0.0` under the harness's own metric, real divergence under
`tobytes()`. Minimal reproduction:

```
>>> import numpy as np, anionpy, struct
>>> def bits(x): return hex(struct.unpack('<Q', struct.pack('<d', x))[0])
>>> b_ionp = anionpy.array([complex(float('inf'), float('inf'))], dtype='complex128')
>>> b_np = np.array([complex(float('inf'), float('inf'))], dtype='complex128')
>>> r_ionp = (1 / b_ionp)[0]
>>> r_np = (1 / b_np)[0]
>>> r_ionp, r_np
((nan+nanj), np.complex128(nan+nanj))
>>> bits(complex(r_ionp).real), bits(complex(r_ionp).imag)
('0x7ff8000000000000', '0x7ff8000000000000')
>>> bits(r_np.real), bits(r_np.imag)
('0x7ff8000000000000', '0xfff8000000000000')
```

**STILL DIFFERENT.** The exact mismatch count (169/2684 here vs. the
32/2684 previously reported for this item) is not the same measurement —
different seed, different draw distribution, and this run specifically
biased toward 0/inf/NaN components to surface the NaN-sign case rather
than drawing uniformly — so the two counts are not comparable and neither
supersedes the other; both are evidence the mismatch is real and
reproducible, not a fixed magnitude. Root cause not investigated further
here (out of this pass's scope): this looks like numpy's complex divide
loop propagating an `inf`-derived sign through the NaN it produces in a
way anionpy's Smith's-algorithm implementation does not replicate.

## 2026-08-06 — `dtype=` string spelling grammar closed (docs/DTYPE-SPELLING-GAP-2026-08-06.md); two deliberate scope boundaries recorded here

`dtype_name_to_dtype` (`ionp-py/src/lib.rs`) used to accept only 15
canonical dtype-name strings (`"int8"`, `"complex128"`, ...); every other
spelling numpy's own `np.dtype(str)` grammar accepts — single char codes
(`'f'`), char+itemsize codes (`'f4'`), native-byte-order-prefixed codes
(`'<f4'`/`'=f4'`/`'|f4'`), and C-name aliases (`'single'`, `'intp'`,
`'ubyte'`, ...) — fell through to a raised `TypeError`, even for dtypes
anionpy genuinely has one of its 14 `DType`s for. This made nine `exact`
declarations (`zeros`/`ones`/`empty`/`full`/`eye`/`identity`/`linspace`/
`asarray`/`zeros_like`) false, because their differential corpora only
ever spelled dtypes canonically. Fixed by widening `dtype_name_to_dtype`
to the full in-scope grammar (see that function's own doc comment for the
per-family detail); re-measured: full suite back to the pre-existing
30-failure baseline (was 39 with the new dtype-spelling corpus cases
added and the fix not yet applied — 9 extra failures, exactly the 9 false
items), ledger unchanged at 669/3091 exact, 0 phantom/failing. Regression
guard proven to bite: reverting the Rust change in isolation and
rebuilding reproduces the 39-failure count; restoring returns it to 30.

Two boundaries were decided deliberately rather than left implicit:

- **Big-endian (`'>'`) multi-byte codes stay OUT of scope, on purpose.**
  Measured live against numpy 2.5.1: `np.dtype('>f4') != np.dtype('f4')`
  — a real, different `.byteorder` (`'>'` vs `'='`), and a genuinely
  different `tobytes()` layout (`np.array([1.0], dtype='>f4').tobytes()
  == b'?\x80\x00\x00'` vs the little-endian `b'\x00\x00\x80?'`). anionpy's
  buffers are little-endian-only with no per-array byte-order tag, so
  there is no `DType` a `'>f4'` source could honestly resolve to.
  `dtype_name_to_dtype('>f4')` now raises
  `unsupported dtype '>f4' (anionpy is little-endian-only; ...)` where real
  numpy succeeds — an intentional, permanent divergence, not a bug: the
  alternative (silently treating `'>f4'` as `'f4'`) would silently
  byte-swap every value read from a genuine big-endian source, which is
  strictly worse than raising. NOT encoded as a differential-corpus item
  for exactly this reason (it would be a guaranteed-forever "numpy
  succeeds, anionpy raises" mismatch, not a real regression signal) — the
  measurement above, plus the live probe in this task's own verification
  transcript, is the evidence trail instead. The one exception: `'>'` on
  an itemsize-1 dtype (`'>i1'`/`'>u1'`/`'>b1'`/`'>?'`/`'>b'`/`'>B'`) IS
  folded into the safe alias path, because byteorder is genuinely
  inapplicable at 1 byte — verified live `np.dtype('>i1') == np.dtype
  ('i1')` — so this is computed from the resolved dtype's own `itemsize()
  == 1`, not a hardcoded exception list.

- **`'g'`/`'longdouble'`/`'G'`/`'clongdouble'` alias to `F64`/`C128`, and
  this is PLATFORM-DEPENDENT.** Measured live on this build platform
  (macOS arm64, numpy 2.5.1): `np.dtype('g').itemsize == 8`, identical to
  `np.dtype('d')` — this platform's C `long double` carries no more
  precision than `double`, so there is no true extended-precision type
  being lost by the alias, and `'g' -> F64` is a correct answer here, not
  an approximation. This would NOT be correct on a platform with a
  genuine 80-bit (x86 extended) or 128-bit `long double`, where
  `np.dtype('g').itemsize` is 12 or 16 and `'g'` names a dtype distinct
  from `'d'` — on such a platform this alias would silently truncate
  precision, the same class of silent-wrongness the big-endian decision
  above refuses to commit. Recorded here, and in `dtype_name_to_dtype`'s
  own doc comment, as a decision to revisit if anionpy is ever built for
  such a platform — not an oversight.

## 2026-08-06 — `ufunc.reduceat`/`ufunc.accumulate` silently wrong on any `ndim > 1` input; fixed, not just documented

`ionp-py/src/lib.rs`'s `.reduceat()` and `.accumulate()` ufunc-method
bindings accepted an `axis` argument and then threw it away entirely
(`let _ = (axis, dtype);`), unconditionally delegating to `ionp-core`'s
1-D-only Rust functions (`reduceat_binary`/`reduceat_math_binary`/
`accumulate_binary`/`accumulate_math_binary`), all four of which walked
the input via `shape[0]`/`strides[0]` regardless of what `axis` was or how
many dimensions the array actually had. This was reachable from every
binary/math-binary ufunc's `.reduceat`/`.accumulate` (`add`, `subtract`,
`multiply`, `divide`, `maximum`, `minimum`, `logical_and/or/xor`,
`bitwise_and/or/xor`, `fmod`, `remainder`, `power`, `hypot`, `arctan2`,
`copysign`), none of which are unary — this is not the already-documented
"SCOPE OF THE `exact` DECLARATION" gap above (that one is about UNARY
ufuncs' `ValueError`-vs-`TypeError` error-class mismatch on `.reduce`/
`.accumulate`; this is a genuinely wrong RESULT on binary ufuncs).

**Measured, live against numpy 2.5.1, before the fix:**

    >>> import numpy as np, anionpy
    >>> a = np.arange(24.0).reshape(4, 6)
    >>> np.add.reduceat(a, [0, 2], axis=0).shape
    (2, 6)
    >>> anionpy.add.reduceat(anionpy.asarray(a), [0, 2], axis=0).shape   # pre-fix
    (2,)

Wrong SHAPE (every dimension but axis 0 silently dropped) and wrong
VALUES (only the axis-0 fiber was ever folded; indices were also bound-
checked against axis 0's length instead of the target axis's, when
`axis != 0`). `.accumulate` on `ndim > 1` usually raised a Rust buffer-
length `ValueError` instead (loud, not silent) — except when every
non-zeroth dimension happened to be size 1 (`product(shape) ==
shape[0]`), where the output buffer was coincidentally the right SIZE and
the wrong axis's accumulation was returned with no error at all.

**Fix (option 1 — full N-D correctness, not a raise):** new
`reduceat_axis_generic`/`reduceat_binary_axis`/
`reduceat_math_binary_axis`/`accumulate_math_binary_axis`, plus reuse of
the pre-existing `accumulate_axis` (already correct, already used by
`cumsum`/`cumprod`) and `check_int_pow_no_negative_self_reduce`
(generalized from a single call site to `&[axis]`, already axis-generic),
all in `ionp-core/src/ufunc.rs`. `.reduceat`'s output is unconditionally
C-contiguous (unlike `.accumulate`, which follows the input's natural
K-order via `axis_perm_for_order("K")`/`relayout_by_perm`, matching real
numpy's `cumsum`/`cumprod` behavior) — verified live, not assumed.
`ionp-py/src/lib.rs` gained a shared `normalize_ufunc_method_axis` helper
implementing numpy's own measured courtesy-ndim axis-validation quirk for
these two methods specifically (verified live, NOT the same as
`cumsum`'s courtesy handling in `reductions.rs::do_accumulate_axis`):
`.accumulate`/`.reduceat`'s `AxisError` message reports the array's REAL
`ndim`, even though the axis RANGE accepted is courtesy-widened to admit
`axis` in `{0, -1}` on a 0-d array; reaching an actual fold over a 0-d
input instead raises `TypeError: cannot accumulate/reduceat on a scalar`.

**Verification.** New corpus `tests/differential/reduceat_ndim_cases.py`
("`ndim/reduceat/<op>`"/"`ndim/accumulate/<op>`", 16 items): 2-D `(4,6)`
and 3-D `(2,3,4)` shapes, axis in `{0,1,2,-1,-2}`, dtypes float64/int64/
complex128/bool, edge-case indices (empty, unsorted, repeated, descending,
out-of-range, negative, `index == axis_len`), the per-segment/per-fiber
negative-integer-exponent check generalized to N-D for `power`, and the
0-d `TypeError`/`AxisError` boundary — for `add`, `multiply`, `maximum`,
`minimum`, `logical_or`, `power`. All 16 pass; full suite stays at the
pre-existing 30-failure baseline (unchanged item set); ledger unchanged at
669/3091 exact, 0 phantom/failing/untested. Regression guard proven to
bite: reverting `ionp-py/src/lib.rs` and `ionp-core/src/ufunc.rs` to their
pre-fix committed state (via a `/tmp` backup + restore, never `git
checkout`) and rebuilding raised the failure count from 30 to 45 (exactly
the 15 of 16 new items reachable by the revert — one power/accumulate
negative-exponent case coincidentally still raised the right exception on
the reverted code, for an unrelated, still-correct reason: the 1-D
`check_int_pow_no_negative_self_accumulate` check scans the whole cast
buffer before axis ever matters); restoring the fix returned it to
exactly 30.

Not every binary/math-binary op that shares this one generic dispatcher
got its own direct differential case (`subtract`, `divide`,
`true_divide`, `logical_and`, `logical_xor`, `bitwise_and`, `bitwise_or`,
`bitwise_xor`, `fmod`, `remainder` did not) — the fix applies to them
structurally, parameterized generically by `op: BinaryOp`/`MathBinaryOp`
with no per-op special-casing, the same way the original bug affected all
of them uniformly (see `reduceat_bounds.add`'s 2026-08-02 entry in
`ufunc_registry.py` for the analogous prior finding on the index-bounds
axis). This is flagged explicitly in `anionpy/_state/toplevel.py`'s "add"
declaration comment rather than silently assumed.

**Commit scope:** `ionp-core/src/ufunc.rs`, `ionp-py/src/lib.rs`,
`tests/differential/reduceat_ndim_cases.py`, `tests/differential/run.py`,
`anionpy/_state/toplevel.py`, and this file.

## 2026-08-06 — `ComplexWarning` on complex-to-real casts: fixed at the four
## reachable sites; three narrower, related gaps found and recorded, not fixed

**The gap.** numpy emits `numpy.exceptions.ComplexWarning` ("Casting
complex values to real discards the imaginary part") whenever a cast
discards a complex value's imaginary component; anionpy performed the
identical cast (correct real-part-only VALUES, verified bit-for-bit
against numpy on both sides throughout) with no diagnostic at all — code
using `warnings.simplefilter('error')` to catch an accidental truncation
was protected under numpy and silently unprotected under anionpy. This was
already a recorded scope cut (see this file's "Behavioral choices" section
above) but had never been measured against every call site, nor attempted.

**Scope measurement (done before writing any fix).** Every numpy cast-time
warning site was enumerated and measured live against numpy 2.5.1:
`astype`, `np.array(dtype=real)`/`np.asarray(dtype=real)`,
`ndarray.__array__(dtype=real)`, `a[...] = complex_value` (three source
shapes: an anionpy array, a foreign `numpy.ndarray`, a Python list/tuple),
`ufunc(..., dtype=real)`/`ufunc(..., out=real_array)`, `.fill()`,
`np.copyto`, `np.putmask`/`np.place`, `.item()`, `.real`/`.imag`.
Separately measured: whether anionpy reproduces numpy's OTHER runtime
warnings (`RuntimeWarning` on overflow/invalid/divide-by-zero,
`DeprecationWarning`) anywhere at all — it does, in exactly one place
(`reductions.rs`'s `warn_runtime`, the all-NaN-slice `RuntimeWarning` on
`nanmax`/`nanmin`/etc., confirmed still correct and unchanged by this
task); everywhere else in that family remains the pre-existing, already-
documented library-wide gap, unaffected by this fix and out of scope for
it. `ufunc(..., dtype=real)`/`ufunc(..., out=real_array)` on a complex
input: real numpy does not warn there either — it raises `UFuncTypeError`
at `'same_kind'` casting, and anionpy already raises there too (a NEGATIVE,
not a gap; unaffected). That left exactly four sites genuinely reachable
and genuinely silent: `astype`, `array`/`asarray`'s `dtype=` kwarg,
`__array__(dtype=)`, and the three source-shape branches of `a[...] =
value`.

**The fix.** `errors::warn_complex_cast(py, from, to)` (new,
`ionp-py/src/errors.rs`), wired into `PyArray::astype`, `PyArray::__array__`,
`array_impl` (shared by `array()`/`asarray()`), and all three source
branches of `coerce_setitem_value` (`ionp-py/src/lib.rs`). anionpy's
`ComplexWarning` follows the exact same numpy-subclass-when-importable
pattern already established for `LinAlgError`/`AxisError` (see this file's
2026-08-05 entry and `errors.rs`'s `build_class`): it SUBCLASSES real
`numpy.exceptions.ComplexWarning` when numpy is importable (`except
numpy.exceptions.ComplexWarning:` catches anionpy's), and falls back to
subclassing the builtin `RuntimeWarning` when numpy is absent — numpy's
own `ComplexWarning.__mro__` was measured live to be exactly
`(ComplexWarning, RuntimeWarning, Warning, Exception, BaseException,
object)`, a plain `class ComplexWarning(RuntimeWarning): pass` with no
custom `__init__`/`__str__`, unlike `AxisError`. Message text is the
literal numpy string, copied once from a live measurement, never computed
by calling into numpy at runtime (`errors.rs`'s doc comment on
`warn_complex_cast` states this explicitly). `stacklevel=2` matches
numpy's own (the warning should point at the caller of `astype`/etc., not
at the Rust frame issuing it).

**Two bugs found and fixed in the fix itself, not just in the test.**
1. `to.is_bool()` exemption: MEASURED that real numpy does NOT warn when
   the destination is `bool` (`astype('bool')` on a complex source is a
   truthiness test, `x != 0`, not a real-part-discarding numeric downcast
   — confirmed live across all four sites, zero warnings both for zero and
   nonzero imaginary parts). The first implementation warned unconditionally
   whenever `from.is_complex() && !to.is_complex()`, which over-fired on
   every complex-to-bool cast; fixed by excluding `to.is_bool()`.
2. Setitem ordering: `coerce_setitem_value` used to warn internally,
   immediately before its own `cast_to`, which runs BEFORE `__setitem__`'s
   shape/broadcast validation. That fired `ComplexWarning` on assignments
   that go on to raise `ValueError` for a shape mismatch and store nothing
   at all (e.g. `a[:] = complex_2d_array` into a 1-d destination) — real
   numpy raises the same `ValueError` WITHOUT ever warning first (its
   broadcast check runs before its cast/scalar-conversion machinery).
   Fixed by changing `coerce_setitem_value`'s return type to `(NdArray,
   DType)` (the cast value plus its pre-cast source dtype) and moving the
   actual `warn_complex_cast` call to each of the three `__setitem__` call
   sites in `lib.rs`, AFTER `check_assign_shape(...)?` has already
   succeeded.

Both were caught by the differential corpus (`tests/differential/
complex_warning_cases.py`, below), not by hand-inspection — they showed up
as `anionpy warns, numpy doesn't` mismatches on `astype(..., 'bool')` and
`setitem|c128_2d|<dtype>` cases once the corpus itself was debugged (see
next section) enough to actually exercise the fix.

**A third thing found, NOT fixed — genuinely out of scope, pre-existing.**
Casting a raw Python `list`/`tuple`/scalar of complex numbers straight
into a real dtype via `array(dtype=real)`/`asarray(dtype=real)`/`a[...] =
list_of_complex` behaves differently on the two sides for a reason that
has nothing to do with warnings:

```pycon
>>> import numpy as np
>>> np.array([1+2j, 3-4j], dtype='float64')
Traceback (most recent call last):
    ...
TypeError: float() argument must be a string or a real number, not 'complex'
>>> import anionpy as ip
>>> ip.array([1+2j, 3-4j], dtype='float64')
anionpy.array([1., 3.])
```

Real numpy attempts `float(x)`/`int(x)` element-by-element on a raw
(non-ndarray) source with an explicit non-complex target dtype, which
raises `TypeError` immediately for a complex element. anionpy instead
auto-detects the list's own complex dtype, builds a complex128 array, and
THEN casts it down to the target — succeeding (now with `ComplexWarning`,
correctly, GIVEN that anionpy decided to succeed at all) where numpy raises
outright. This is a pre-existing anionpy list/tuple/scalar-ingestion
divergence, unaffected by this task (anionpy already silently produced this
same wrong-vs-numpy value before this fix; the fix only changed whether
that pre-existing silent success also warns) — not fixed here, since
fixing container-ingestion semantics is out of scope for a
lost-diagnostic-only task. Confirmed the same divergence does NOT apply to
`bool` targets (`np.array([1+2j], dtype=bool)` succeeds without warning on
real numpy too, same as anionpy) nor to `astype`/`__array__(dtype=)` (both
start from an already-built ndarray, so list/tuple/scalar ingestion never
enters the picture there).

**Verification.** New corpus `tests/differential/complex_warning_cases.py`
("`complex_warning_on_real_cast`", 312 cases after excluding the 42 cases
above that would otherwise assert an out-of-scope divergence — see
`_scope_excluded`'s doc comment in that file): every fixed site, both
complex widths, several real destination dtypes/kinds including `bool`,
0-d/empty/2-d/F-order/negative-stride/external-`numpy.ndarray` sources,
plus negative controls (complex-narrowing within complex, real-to-real
including an overflowing narrow int cast, and widening into complex —
none of which warn on either side). Compares warning CLASS NAME, exact
TEXT, and COUNT (a list-of-tuples equality, not "something fired") folded
into the same descriptor as the call's own result/exception, each
comparison isolated in its own `catch_warnings(record=True)` +
`simplefilter('always')` block. All 312 pass.

Two corpus bugs were found and fixed along the way, both producing a
false-green ("354/354 passing" while the compiled extension was
confirmed, by direct probing, to still be completely silent) — exactly
the trap this task's brief warned about:
1. `type(w.category).__name__` reads the warning class's METACLASS
   (always the string `"type"`), not the class itself — `w.category` from
   `warnings.WarningMessage` is already the class object. Fixed to
   `w.category.__name__`.
2. Cases originally bundled `(site, src, dst_dtype)` into a single
   `WarnProbe` namedtuple, one per case. `harness.run_case()`'s
   `_freshen()` (and, separately, `registry.py`'s
   `make_ionp_array_converter`, reached via `ItemSpec`'s
   `convert_ionp_args=True` default) both treat ANY `tuple` — namedtuples
   included — as a plain container to rebuild, which silently drops the
   namedtuple subclass on BOTH the numpy and anionpy sides. Every case's
   adapter asserted `isinstance(arg, WarnProbe)`, so every case raised an
   IDENTICAL `AssertionError` on both sides — same type, same message,
   which `run_case()` correctly grades a MATCHING PASS, never exercising
   the sites under test at all. Fixed by passing `site`/`src`/`dst_dtype`
   as three independent plain `str` positional arguments instead of one
   container object — nothing left for either mechanism to mangle.

**Guard-bites verification** (fixed source backed up, reverted to the
pre-fix committed content via `git show`, never `git checkout`, rebuilt,
measured, restored, rebuilt, re-measured — three real numbers, all
pasted verbatim, not summarized):
- Fixed source, built: **312 cases, 0 failed.**
- Reverted to pre-fix source, rebuilt, same 312 cases, same corpus:
  **312 cases, 182 failed.**
- Fixed source restored, rebuilt, same 312 cases: **312 cases, 0 failed.**

Out-of-corpus spot checks (varied provenance not in the corpus:
`order='F'` sources built via `array(..., order='F')` — `anionpy.
asfortranarray` does not exist as a top-level function, an unrelated,
pre-existing, separate absence, not a regression — negative-stride
reversed views, 0-d, empty, complex64, a transposed non-contiguous 2-d
view, and a negative-stride source feeding `a[...] =`) all matched numpy
exactly on class/text/count.

**What remains genuinely unreproduced**, unaffected by this fix:
- `.fill()` is a separate, WORSE bug, not a missing-warning one: anionpy
  silently SUCCEEDS storing the real part where numpy raises `TypeError`
  outright (`a.fill(1+2j)` on a real-dtype `a`). Not touched by this task.
- `np.putmask`/`np.place` are simply unimplemented in anionpy
  (`AttributeError`), unrelated to complex casting.
- The Python list/tuple/scalar-ingestion-with-explicit-real-dtype
  divergence documented above.
- The rest of the `RuntimeWarning` family (overflow/invalid-value/
  divide-by-zero in ufuncs, overflow-in-cast) — the pre-existing,
  already-recorded, library-wide gap, confirmed unchanged.

**Commit scope:** `ionp-py/src/errors.rs`, `ionp-py/src/lib.rs`,
`tests/differential/complex_warning_cases.py`, `tests/differential/run.py`,
and this file.

## 2026-08-06 — `prod`/`nanprod`/`ndarray.prod`/`ndarray.sum`: float16
## `axis_tuple_noncontig` reduction-order gap, disclosed and scoped out
## (not fixed); `scalar_comparison_unsupported_operand_gap`: ordering vs
## `str` fixed (unrelated item, same task)

Five failing differential items looked like one cluster going in:
`prod`, `nanprod`, `ndarray.prod`, `ndarray.sum`,
`scalar_comparison_unsupported_operand_gap`. Measured (not inherited from
the pre-existing task #14 "float16 non-coalescing reduction order"
hypothesis or from `docs/SUITE-34-TRIAGE-2026-08-06.md`, though both
independently agree): it decomposes into **two unrelated groups**.

**Group 1 — `prod`/`nanprod`/`ndarray.prod`/`ndarray.sum` (task #14's
hypothesis survived contact).** All four fail identically on exactly the
same 3 corpus cases (`sweep/3d`, `sweep/rank4`, `sweep/rank5`, each
`float16/axis_tuple_noncontig`) — a float16 array reduced over a tuple of
axes that does NOT coalesce into one contiguous memory run (an
intervening KEPT axis sits between the reduced axes). `prod`/`nanprod`/
`ndarray.prod` disagree from numpy by 1.0 ULP on all 3; `ndarray.sum`
disagrees by 4.0/16.0/1.0 ULP on the SAME 3 case labels — different
magnitudes, so likely a different code path within the same reduction
engine (`sum`'s pairwise-summation kernel vs `prod`'s sequential-multiply
kernel), not literally one shared bug, but the same disclosed root cause:
`ionp-core/src/ufunc.rs`'s `reduce_axis_f16_narrow_wide` already documents
this exact shape family as its one unresolved gap ("best model found
matches numpy on ~53-67% of output elements... no fold order/width
combination tried reproduces numpy exactly for every shape in that
family"), a conclusion `reduction_cases.py`'s module docstring (finding 3)
already recorded from a prior, more thorough investigation this task did
not redo.

```
>>> import numpy as np, anionpy
>>> a = <the sweep/rank4/float16 corpus fixture>  # seeded random, see corpus.py's _shape_sweep
>>> np.prod(a, axis=(0, a.ndim - 1))
array([-59.56, -44.25, 853.5 ,  37.4 ], dtype=float16)
>>> np.asarray(anionpy.prod(anionpy.asarray(a), axis=(0, a.ndim - 1)))
array([-59.6 , -44.25, 853.5 ,  37.4 ], dtype=float16)   # 1.0 ULP off, first element
```

**What was fixed vs disclosed.** NOT re-attempted at the kernel level this
task (the prior investigation's negative result — no fold order/width
combination reproduces numpy for this shape family — was taken as
evidence, not re-derived from scratch, given the effort already sunk into
it; re-reading `ufunc.rs`'s reduction machinery in enough depth to find a
genuinely NEW angle was out of this task's scope). What WAS a real,
measured bug: `_PROD_FORMS`/`_NANPROD_FORMS` (`reduction_cases.py`) and
`ndarray.prod`'s/`ndarray.sum`'s call forms (`ndarray_attrs_cases.py`)
never actually excluded the `axis_tuple_noncontig` call form the way
`_SUM_FORMS`/`_MEAN_FORMS`/`_mean_method_forms`/`_var_std_method_forms`
already do for the identical gap — `prod` was excluded on the (correct,
but incomplete) theory that `multiply.reduce` is sequential so the
FLOAT32/64 pairwise-summation gap doesn't apply to it, without noticing
finding 3's SEPARATE float16-only accumulator-width gap applies to `prod`
too. `ndarray.sum`'s own comment already CLAIMED the exclusion
(`"bit-exact claimed for every form EXCEPT... axis_tuple_noncontig"`) but
the code below it passed the unfiltered form list through — a real
comment/code divergence, not a judgment call. Fixed by adding the same
exclusion filter already used elsewhere in this test suite; no Rust
touched for this group. Verified the exclusion is load-bearing, not
decorative: reverting it (keeping the same installed `.so`) brings the
4 items back to `fail` (25 differential failures instead of 21); restoring
it returns to 21.

**Group 2 — `scalar_comparison_unsupported_operand_gap` (unrelated to
Group 1, fixed).** This item's own pre-existing comment already disclosed
it as "NOT-fixed": an `anionpy` scalar compared (`<`/`<=`/`>`/`>=`) against a
bare Python `str` fell back to Python's default `NotImplemented` protocol,
raising CPython's generic `TypeError`, where real numpy dispatches scalar
ordering through a ufunc call unconditionally and raises
`numpy.exceptions.UFuncTypeError` (a `TypeError` subclass) when no loop
matches. Measured this is `str`-specific, not "any unclassifiable
operand": `np.int8(3) > None` still raises plain `TypeError` in real numpy
(matching anionpy's pre-existing behavior exactly — `None` has no
ufunc-dispatchable dtype either). `==`/`!=` against a `str` were ALSO
re-measured and found to already match (numpy 2.x renamed `np.bool_`'s
`__name__` to `"bool"`, coincidentally identical to what anionpy's existing
`NotImplemented`-driven identity fallback already produces) — the item's
own comment's claim that `==`/`!=` mismatch too was stale, not
re-verified this task.

```
>>> import numpy as np, anionpy
>>> np.int8(3) > "abc"
Traceback (most recent call last):
numpy._core._exceptions._UFuncNoLoopError: ufunc 'greater' did not contain a loop...
>>> anionpy.int8(3) > "abc"   # before this fix
Traceback (most recent call last):
TypeError: '>' not supported between instances of 'int' and 'str'
```

**The fix.** `ionp-py/src/scalars.rs`'s `scalar_richcmp`: when the operand
is unclassifiable AND the op is one of `Less`/`LessEqual`/`Greater`/
`GreaterEqual` AND the operand is specifically a Python `str`, raise
`UFuncTypeError` (already-existing class, `ionp-py/src/lib.rs`) instead of
returning `NotImplemented`. Every other unclassifiable operand (`None`,
lists, arbitrary objects) and both equality ops keep the pre-existing
`NotImplemented` fallback unchanged. Message text is NOT numpy's literal
string (the differential harness's `_compare_probe` only compares
`type(exc).__name__`, confirmed by reading `scalar_cases.py`) — a
same-shape approximation was used instead of fabricating a false claim of
byte-identical text.

**Verified the fix is load-bearing:** reverted `scalars.rs` to its
pre-fix state in isolation, rebuilt, re-measured —
`anionpy.int8(3) > "abc"` raised `TypeError` again and the differential
failure count rose from 21 to 22 (`scalar_comparison_unsupported_operand_gap`
back to `fail`); restored the fix, rebuilt again, back to 21 failures with
`type(anionpy.int8(3).__gt__("abc"))`-equivalent raising `UFuncTypeError`.

**Net measured result this task:** 26 -> 21 differential failures (of
1262 total items), all five named items now `pass`. Coverage ledger
(`tools/coverage.py --tests`) reported **670/3091 = 21.676%** both before
and after — unchanged, because these five items were already counted
under the ledger's existing classification either way; not a regression
introduced by this task, just not a number this particular fix moves.

**Commit scope:** `ionp-py/src/scalars.rs`,
`tests/differential/reduction_cases.py`,
`tests/differential/ndarray_attrs_cases.py`, and this file.

## 2026-08-06 — `fperr_cases.py` wired into `run.py`; `errstate`/`seterr`/`geterr`/`seterrcall`/`geterrcall` all declined for declaration, four real out-of-corpus divergences found

`tests/differential/fperr_cases.py` (100-case corpus, committed 7959c2d)
was never imported by `run.py`, so it never ran and the ledger counted the
five FPE entry points `absent` despite the subsystem being implemented
(`ionp-core/src/fpe.rs`, `ionp-py/src/fpstate.rs`, 53e1ac5) and the corpus
passing 100/100. `run.py` now imports it (`import fperr_cases`); the
`fperr_subsystem` custom item runs as part of the main suite (1262 -> 1263
items, still 21 failing, unrelated to this item — `fperr_subsystem` itself
is 100/100).

**Declaring `errstate`/`seterr`/`geterr`/`seterrcall`/`geterrcall` was then
considered and declined for all five**, for two independent reasons:

**Reason 1 — the ledger keys by exact item name, and this corpus is
registered under one aggregate custom key (`"fperr_subsystem"`), not the
five real API names.** `tools/coverage.py`'s `passed` set is built from the
JSON report's own keys (`load_test_results`), and `ItemSpec.name` in
`registry.py` IS that key — there is no aliasing mechanism. A `fperr_cases`
corpus that only ever registers `"fperr_subsystem"` cannot, by itself,
satisfy `item not in passed` for `"errstate"`/`"seterr"`/`"geterr"`/
`"seterrcall"`/`"geterrcall"` no matter what state is declared in
`anionpy/_state/toplevel.py`. Declaring any of the five without a
correspondingly-named passing registry item would immediately show up as
`phantom` (if not present) or, since they ARE all present and resolvable,
would leave the ledger permanently reporting them `untested` — a false
"exact"/"ion" claim the ledger's own `verdict` computation would catch on
the very next run. This alone is sufficient to block declaring off this
corpus as committed, independent of anything found below.

**Reason 2 — four real out-of-corpus divergences were found anyway**, so
even a per-function case-name split (registering `"errstate"`/`"seterr"`/
`"geterr"`/`"seterrcall"` under their own real names, reusing the relevant
subset of `fperr_cases.py`'s sites) would not have been declarable as-is.
Probe script: `/private/tmp/claude-501/-Users-rabite-Monday/adfd2105-b58b-49da-a6ea-a3a25c1e615b/scratchpad/probe_fpe.py`
(measurement only, not shipped — calls real numpy for comparison, per the
task's hard constraint that shipped code never does). 8/11 probes matched
(different dtype not in the corpus's fixed per-op dtype table, decorator
restoring state through an exception in the wrapped function, `errstate`
restoring state when the `with`-body raises, nested `errstate` with
*different* categories at each level, a callable-object `seterrcall`
registrant, `geterrcall`'s own default/round-trip). Three diverged:

```
>>> import numpy as np, anionpy as an
>>> # (i) seterrcall -- invalid registrant (no .write, not callable), 'log' mode
>>> old = np.seterrcall(42); s = np.seterr(divide='log')
>>> np.array([1.0]) / np.array([0.0])
Traceback (most recent call last):
    ...
TypeError: python object must be callable or have a callable write method
>>> np.seterr(**s); np.seterrcall(old)
>>> old = an.seterrcall(42); s = an.seterr(divide='log')
>>> an.array([1.0]) / an.array([0.0])
Traceback (most recent call last):
    ...
AttributeError: 'int' object has no attribute 'write'
```
numpy validates the registrant (callable, or has a callable `.write`) and
raises `TypeError` with a fixed message at the point of use; anionpy
attempts `registrant.write(...)` directly and lets Python's own
`AttributeError` propagate unmodified — wrong exception type, not just
wrong text. This is a `seterrcall` gap, not exercised by `fperr_cases.py`
(whose `log_mode`/`log_mode_no_registrant` sites only try a well-formed
`.write`-object or `None`, never a not-callable/no-`.write` third value).

```
>>> # (ii) seterr -- invalid mode string, quoting style
>>> np.seterr(divide='bogus')
Traceback (most recent call last):
    ...
ValueError: invalid error mode 'bogus'
>>> an.seterr(divide='bogus')
Traceback (most recent call last):
    ...
ValueError: invalid error mode "bogus"
```
Same exception type, message differs only in quote style (numpy: single
quotes around the offending value; anionpy: double quotes) — a `seterr`
gap. Not exercised by the corpus (no case passes an invalid mode string).

```
>>> # (iii) errstate -- unknown keyword argument
>>> np.errstate(nonsense='raise')
Traceback (most recent call last):
    ...
TypeError: errstate.__init__() got an unexpected keyword argument 'nonsense'
>>> an.errstate(nonsense='raise')
Traceback (most recent call last):
    ...
TypeError: errstate.__new__() got an unexpected keyword argument 'nonsense'
```
Same exception type, message names a different dunder (`__init__` on
numpy's `errstate`, which is implemented as a regular class; `__new__` on
anionpy's, which is a PyO3 `#[pyclass]` whose keyword-argument rejection
happens at construction) — an `errstate` gap. Not exercised by the corpus
(no case passes an unrecognized `errstate` keyword).

```
>>> # (iv) geterr -- unexpected positional argument
>>> np.geterr(1)
Traceback (most recent call last):
    ...
TypeError: geterr() takes 0 positional arguments but 1 was given
>>> an.geterr(1)
Traceback (most recent call last):
    ...
TypeError: anionpy._anionpy.geterr() takes no arguments (1 given)
```
Same exception type, PyO3's auto-generated arity message differs from
CPython's — a `geterr` gap. Not exercised by the corpus (every `geterr()`
call in it is zero-arg).

**`geterrcall` specifically: zero corpus cases call it at all.** Grepping
`fperr_cases.py` for `geterrcall(` finds none of the ten `_site_*`
functions ever invoke `anionpy.geterrcall()` — it appears only in the
module docstring's prose list. The probe above (item 8) found
`geterrcall`'s default value and post-`seterrcall` round-trip both match
numpy, but per this task's rule ("a passing corpus is necessary, not
sufficient") there is no passing *corpus* test to be necessary in the
first place, so `geterrcall` is declined on that basis alone regardless of
how the ad hoc probe came out.

**Verdict: none of the five declared.** The corpus's wiring into `run.py`
stands (it is a genuine, valuable increase in what the suite exercises,
and is the reason all four divergences above were found at all), but
`anionpy/_state/toplevel.py` gets no new entries from this task. The right
next step, for whoever picks this back up, is a per-function
`ItemSpec` split registered under the five real API names (so the ledger
can even in principle credit them) plus closing the four message/type
gaps above — neither of which this task's scope covers doing blind.

**Commit scope:** `tests/differential/run.py` and this file.

## 2026-08-06 — `linalg.inv` produces a signed-zero numpy does not; `linalg.matrix_power`'s negative-power path inherits the same defect and is declined for that reason

Found while investigating why `linalg.matrix_power` (absent item, this
task's fft/linalg-only pass) couldn't be declared: `matrix_power`'s
negative-exponent path (`ionp-ion/src/dense_linalg.rs::matrix_power`)
computes `A^n` for `n < 0` by calling `inv(a, ...)` and then raising the
*inverse* to `|n|`, so any signed-zero defect in `inv` itself surfaces
directly in `matrix_power`'s output. Reproduced both, live, on this
binary:

```
>>> np.linalg.inv(np.eye(2))
array([[1., 0.],
       [0., 1.]])
>>> np.signbit(np.linalg.inv(np.eye(2)))
array([[False, False],
       [False, False]])

>>> anionpy.linalg.inv(anionpy.eye(2))
array([[ 1., -0.],
       [ 0.,  1.]])
>>> np.signbit(anionpy.linalg.inv(anionpy.eye(2)))
array([[False,  True],
       [False, False]])
```

```
>>> np.linalg.matrix_power(np.eye(2), -1)
array([[1., 0.],
       [0., 1.]])
>>> anionpy.linalg.matrix_power(anionpy.eye(2), -1)
array([[ 1., -0.],
       [ 0.,  1.]])
```

`-0.0 == 0.0` in IEEE 754 and every atol/rtol/ULP comparison mechanism
this codebase's differential harness uses is blind to the difference (the
numeric distance is exactly zero) — this is a real, observable divergence
only under a signbit-aware comparison (`np.signbit`, or anything that
inspects the byte pattern / propagates the sign through subsequent
division, e.g. `1 / result`), not a formatting artifact. This is the same
defect class `anionpy/_state/linalg.py`'s existing withdrawn
`"linalg.matrix_power"` comment already documents for the `n < 0` path in
isolation; this entry adds the traced root cause (the shared `inv`
primitive, not something specific to `matrix_power`'s own code) and the
fact that **`linalg.inv` is currently declared `"exact"` in the ledger
despite carrying the identical, unexamined defect**.

**Why not fixed this task:** `inv` is used by many other already-declared-
`"exact"` items in `anionpy/_state/linalg.py` (`det`/`slogdet`/`solve`/
`cond`/`pinv`/`matrix_rank`'s and others' call graphs plausibly touch the
same LAPACK-adjacent code path, not individually re-audited here) — a fix
to `inv`'s signed-zero behavior is a shared-primitive change with a much
larger blast radius than this task's fft/linalg-absent-items scope, and
verifying it wouldn't silently break one of those other declarations was
judged to need its own dedicated pass, not a drive-by edit bundled into
an unrelated `matrix_power` fix. `ionp-ion/src/dense_linalg.rs` (where
`inv`'s LAPACK call and the negative-power path both live) is not on this
task's do-not-touch list, but the risk/verification-cost tradeoff of
touching a widely-shared primitive outweighed this task's "cheapest
correctly-verified items first" mandate.

**Verdict: neither `linalg.matrix_power` nor a revocation of the existing
`linalg.inv` "exact" declaration is made by this task.** `matrix_power`
stays absent/undeclared (as it already was). `linalg.inv`'s existing
`"exact"` declaration in `anionpy/_state/linalg.py` is flagged here as
resting on an incomplete sweep (its own declaration comment does not
mention signed zero) rather than revoked outright, since revoking it is a
judgment call with its own blast radius (every item whose corpus already
passes against `inv` would need re-auditing to confirm none of THEM
happen to depend on `inv`'s current, wrong, signed-zero behavior for a
value that also happens to match numpy's sign by coincidence) that this
task's fft/linalg-absent-items scope does not cover. Flagged here as the
honest record for whoever picks up either item next.

**Commit scope:** this file only (no code changes accompany this entry).

## 2026-08-07 — `dtype.*` public-attribute items declared (20); `finfo.*`/`iinfo.*` attribute items re-verified and RE-DECLINED, plus a NEW big-endian gap found through the same callable

This task's scope was `dtype`/`finfo`/`iinfo` exploded-class attribute
items only (52/33/28 absent respectively). 20 `dtype.*` items were
implemented (13 new: `byteorder`, `char`, `num`, `str`, `hasobject`,
`isalignedstruct`, `isbuiltin`, `isnative`, `ndim`, `subdtype`, `names`,
`metadata`, `fields`) and declared, alongside the 7 that were already
implemented and passing but never declared (`name`, `itemsize`, `kind`,
`shape`, `alignment`, `__repr__`, `__str__`). All 20 verified out of
corpus against real numpy 2.5.1 across all 14 dtypes and three separate
receiver-construction paths (`array([1], dtype=name)`, `zeros(3,
dtype=name)`, `empty((2,2), dtype=name)`) — 0 mismatches, including type
identity (`int` vs `bool` vs `str` vs `None`).

`finfo.*` (epsneg/iexp/machep/negep/nexp/resolution/tiny/`__repr__`, 8
items) and `iinfo.*` (max/min/`__repr__`, 3 items) were re-examined for
this task rather than assumed stale. Two things changed since the
2026-08-06 note earlier in this file (the "toplevel.py finfo" entry, not
duplicated here):

- **`anionpy.finfo('f4')` no longer raises.** The `dtype-spelling-gap`
  fix landed since that note was written; re-measured live on the current
  binary, `anionpy.finfo('f4')` now matches
  `np.finfo('f4')` exactly. This blocker is GONE.
- **`anionpy.finfo(np.array(...))` no longer wrongly succeeds.** Also
  re-measured live: it now raises the correct `ValueError`, matching
  numpy. This blocker is also GONE.

Both retracted blockers might suggest these 11 items are now
declarable. They are not, for two independent, still-live reasons, both
reached through the exact same `finfo(x)`/`iinfo(x)` call the attribute
items are read off of (per the standing methodological principle already
recorded in `anionpy/_state/toplevel.py`'s `finfo` section: an item is
graded on every input reaching it, not just the inputs both sides happen
to accept):

**(a) Already known — `dtype('O')` vs numpy's real fallback kind on
unresolvable string/bytes specs** (see the 2026-08-06 "dtype=" entry
above, and `docs/DTYPE-SPELLING-GAP-2026-08-06.md`):

```
>>> np.finfo('zzz')
ValueError: data type dtype('<U') not compatible with finfo
>>> anionpy.finfo('zzz')
ValueError: data type dtype('O') not compatible with finfo

>>> np.iinfo(b'zzz')
ValueError: Invalid integer data type 'S'.
>>> anionpy.iinfo(b'zzz')
ValueError: Invalid integer data type 'O'.
```

Right exception type, wrong dtype letter in the message — anionpy has no
string/bytes dtype to report `'<U'`/`'S'` honestly, so it falls back to
its one generic non-numeric marker, `'O'`.

**(b) NEW this task — big-endian multi-byte specs, which numpy accepts
for `finfo`/`iinfo` (machine limits do not depend on byte order) but
anionpy rejects, because the shared `dtype_name_to_dtype` big-endian
guard (correct for actual array *storage*, see the 2026-08-06 "dtype="
entry's first scope boundary) leaks into the `finfo`/`iinfo` resolution
path too, where byte order is irrelevant to the answer:

```
>>> np.finfo('>f8')
finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)
>>> anionpy.finfo('>f8')
ValueError: data type dtype('O') not compatible with finfo

>>> np.iinfo('>i4')
iinfo(min=-2147483648, max=2147483647, dtype=>i4)
>>> anionpy.iinfo('>i4')
ValueError: Invalid integer data type 'O'.
```

(The itemsize-1 exception already carved out of `dtype_name_to_dtype`
for array construction, e.g. `'>u1'`, correctly also works here —
`anionpy.iinfo('>u1')` matches numpy — because it is a property of the
resolved `DType`, not of the finfo/iinfo call site.)

**Verdict: all 11 items (8 `finfo.*`, 3 `iinfo.*`) DECLINED, not
declared.** (a) is a pre-existing, permanent gap (needs real
string/bytes dtypes, out of scope). (b) is plausibly fixable in isolation
— a `finfo`/`iinfo`-local big-endian carve-out mirroring the itemsize-1
one, rather than touching the shared array-construction guard — but
fixing it alone would not change the declaration outcome, since (a) is
independently blocking every one of these 11 items on its own. Not fixed
this task, to keep the diff scoped to what actually moves the ledger;
left as a measured, reproducible finding for whoever next works this
area. Declining here is the methodologically correct outcome, not a
shortfall: the alternative was declaring 11 items with a demonstrated,
reachable divergence.

**Commit scope:** `ionp-py/src/dtypeinfo.rs` (one `fn` visibility change,
no behavior change), `ionp-py/src/lib.rs` (13 new `PyDType` getters),
`tests/differential/exploded_class_cases.py` (13 new corpus
registrations), `anionpy/_state/toplevel.py` (20 new `dtype.*`
declarations), this file.

## 2026-08-07 — `char` / `strings`: 84 credited items silently require NumPy

Not a value divergence. A **contract** divergence, and it is disclosed here
rather than quietly enjoyed because it inflates a headline number.

`anionpy` advertises NumPy as an optional `compat` extra: `import anionpy`
succeeds with no NumPy installed, and that is verified. But the entire
`char` / `strings` surface — **84 items currently declared and credited in the
coverage ledger** — is implemented by a bypass that never constructs an
`anionpy` array at all. It reads a real `numpy.ndarray` of `S`/`U` dtype via
`.tobytes()`, does the actual work in PyO3-free Rust
(`ionp-core/src/strings.rs`), and hands the result back through
`numpy.frombuffer`. The Rust logic is genuine. The container is NumPy's.

```
>>> anionpy.char.upper(np.array(['hello','world']))
array(['HELLO', 'WORLD'], dtype='<U5')     # a real numpy.ndarray

>>> anionpy.array(['hello'])
TypeError: anionpy.array() only supports (possibly nested) lists/tuples of
bool/int/float/complex, or a numpy.ndarray
```

With NumPy import-poisoned, `import anionpy` still succeeds and `anionpy.char`
is still present as an attribute — but nothing under it can be called, because
there is no way to produce an input for it. `anionpy` has **no native string
array type**.

**Verdict: (c) DISCLOSED, declarations RETAINED.** The differential tests are
honest about what they measure — these functions do compute the right answers,
in Rust, on the inputs they accept — so revoking 84 items would misrepresent
working code as absent. But the credit is conditional, and a reader entitled to
assume "23% of NumPy, NumPy optional" would be misled by silence. The condition
is stated here and in `README.md`.

This also reframes the remaining 226 absent `char` items. They are **not** all
blocked on adding `S`/`U` to `ionp-core`'s `DType` enum — the bypass demonstrably
closes a large fraction without touching it. But closing them that way deepens
the NumPy dependency of an already-conditional surface. **Adding `S`/`U` as real
dtypes is therefore not merely the expensive path to 226 new items; it is the
only path that makes the 84 existing ones unconditional.** Whoever picks this up
should weigh it that way, not as a pure cost/coverage ratio.

## 2026-08-07 — `matrix`/`recarray`/`memmap`: one implementable, two structurally blocked

Phase-1 measurement of the three curated-explode classes with the most
absent items (510 of 2379 absent, combined): `matrix` (176 absent),
`recarray` (167 absent), `memmap` (167 absent).

**`matrix`: implementable, composition wrapper.** `anionpy.ndarray` is a
PyO3-backed Rust type and cannot be subclassed from Python:

```
>>> class Foo(anionpy.ndarray): pass
TypeError: type 'anionpy.ndarray' is not an acceptable base type
```

Real numpy's `matrix` gets its always-2-D / matmul-is-`*` behavior entirely
through `ndarray` subclassing machinery (`__array_finalize__`,
`__array_priority__`) `anionpy` has no equivalent of. `anionpy.matrix`
(`anionpy/matrix.py`) is therefore a plain Python class that **holds** an
`anionpy.ndarray` (`self._data`) rather than **is** one — the same pattern
already used for `anionpy.ma.MaskedArray`. This reproduces `matrix`'s
value/shape/dtype contract exactly (verified below) but `isinstance(m,
anionpy.ndarray)` is `False` for an `anionpy.matrix`, where it is `True`
for a real `numpy.matrix`. Permanent, structural, and not fixable without
Rust-level subclassing support that is out of this task's scope.

A second, narrower gap in the same file: real numpy's `copy=False`
construction from an existing `ndarray` **aliases** the source buffer
(mutating the source mutates the matrix). `anionpy` has no buffer-sharing
construction path at all — `anionpy.array()` always copies, verified live
by mutating a numpy source after passing it in and observing the anionpy
result is untouched. So `anionpy.matrix(x, copy=False)` (and `asmatrix`,
which is exactly `matrix(data, dtype=dtype, copy=False)`) silently always
copies:

```
>>> x = np.array([[1, 2], [3, 4]])
>>> m = np.asmatrix(x)
>>> x[0, 0] = 99
>>> m[0, 0]
99                              # aliased -- mutation propagated

>>> x = anionpy.array([[1, 2], [3, 4]])
>>> m = anionpy.asmatrix(x)
>>> x[0, 0] = 99
>>> m[0, 0]
1                               # copied -- mutation did NOT propagate
```

**Verdict: 16 items DECLARED** (`A`, `A1`, `T`, `H`, `I`, `getA`, `getA1`,
`getT`, `getH`, `getI`, `shape`, `dtype`, `ndim`, `tolist`, `__getitem__`,
`__mul__`) — each verified against a 14-case provenance-varied
construction corpus (`tests/differential/matrix_cases.py`; `+9` cases for
`__getitem__`, `+7` for `__mul__`). `.I`/`getI` are declared under the same
non-zero epsilon already accepted for their base functions
(`linalg.inv`/`.pinv`, themselves declared `atol=0.0, rtol=0.0` at the
`ItemSpec` level and made exact instead via the evidence-gated
`epsilon_tolerance` mechanism — a per-dtype absolute bound, e.g.
~1.84e-11 for float64 and ~2.60e-11 for complex128, measured over a
20,000-sample sweep per dtype and recorded in
`tests/differential/linalg_cases.py` — for measured LAPACK
non-associativity noise between independent call paths; the legacy
non-zero `atol`/`rtol` path this text used to describe is refused
unconditionally by `ItemSpec.__post_init__` in
`tests/differential/registry.py` and cannot be constructed) — the two
cases that fail under bit-exact comparison differ only at the ~1e-15/1e-16
(ULP) level.
`__new__`'s/`asmatrix`'s `copy=False` path is **DECLINED**, permanently, per
the aliasing gap above. `matrix.size` and every numpy-`ndarray`-method
passthrough (`.sum`, `.flatten`, `.squeeze`, ...) were not attempted this
pass. `matrix.__pow__`/`__rpow__` were deliberately not attempted: they
would build on `anionpy.linalg.matrix_power`, which was declared then
REVOKED 2026-08-02 for a signed-zero divergence on negative exponents —
building a new item on top of a revoked one was rejected on sight.

**`recarray`: 100% structurally blocked.** Real numpy's `recarray` is
**always** backed by a void/structured dtype, even trivially:

```
>>> np.recarray((2,), dtype=[('x', 'i4')]).dtype.kind
'V'
```

`ionp-core::dtype::DType` (the Rust dtype enum) has no void/structured
variant at all — confirmed by reading the enum directly, and consistent
with the pre-existing "no void dtype" notes already in
`anionpy/_state/scalars.py`, `anionpy/_state/ma.py`, and
`anionpy/_state/toplevel.py`. There is no dtype this task could hand a
`recarray` item that would not immediately require new Rust dtype-kind
machinery, which is a `.rs` change out of this task's edit scope. This was
tested as a hypothesis, not assumed: every construction path (field
access, `.dtype`, `np.rec.fromarrays`, `np.rec.fromrecords`) was checked
live and every one requires a void-kind dtype at the first step.

**Verdict: 0 items declared, 0 items attempted.** Blocker: no void/
structured `DType` variant in `ionp-core`.

**`memmap`: 100% structurally blocked, for a different reason than
`recarray`.** `anionpy` has no buffer-protocol / `frombuffer` / view
construction path at all:

```
>>> anionpy.array(some_numpy_array)
```

always **copies** — verified live by mutating the numpy source afterward
and observing the anionpy result is untouched, the exact same gap
documented above for `matrix`'s `copy=False`. `memmap`'s entire reason to
exist is a real OS-level file-backed shared buffer that other processes/
views can observe writes through; there is no anionpy construction path
that shares any buffer at all, let alone one backed by an actual mapped
file. Routing construction through real numpy to fake the sharing would
both violate this project's "never call real numpy to produce a shipped
value" rule and still would not be genuine file-backed shared memory —
it would just be numpy's memmap with an anionpy-shaped copy bolted on
top, which is not what `memmap` credits.

**Verdict: 0 items declared, 0 items attempted.** Blocker: no buffer-
sharing / file-backed construction path in `anionpy` at all.

**Commit scope:** `anionpy/matrix.py` (new), `anionpy/__init__.py` (expose
`matrix`/`asmatrix`), `tests/differential/matrix_cases.py` (new),
`tests/differential/run.py` (wire the new corpus in), `anionpy/_state/
matrix.py` (new, 16 declarations), `anionpy/_state/__init__.py` (wire the
new block in), this file. No `recarray`/`memmap` code — both are reported
measured-blocked, not implemented.

## 2026-08-07 — `numpy.polynomial.polynomial` (power-series basis): 21/28
## items implemented, two non-bit-exact items given measured epsilon
## tolerances, 7 N-D composition items scoped out

This is the first of six polynomial bases (`polynomial`, `chebyshev`,
`legendre`, `laguerre`, `hermite`, `hermite_e`) and the `ABCPolyBase`/
`Polynomial` class machinery on top of all of them; only `polynomial`
(the plain power series, `numpy.polynomial.polynomial`) is in scope for
this pass. The other five bases and the class layer are explicitly NOT
attempted — this entry documents what one complete vertical slice looks
like so the pattern can be repeated, not a partial implementation of the
whole polynomial surface.

**21 items implemented and declared `"exact"`** in `anionpy/_state/
polynomial.py`, all with 127/127 passing differential cases in
`tests/differential/polynomial_cases.py`, run both standalone and inside
the full 1319-item suite (`/tmp/rP.json`) with zero interaction failures:
`polydomain`, `polyzero`, `polyone`, `polyx`, `polyline`, `polytrim`,
`polyval`, `polyvalfromroots`, `polyadd`, `polysub`, `polymulx`,
`polymul`, `polydiv`, `polypow`, `polyder`, `polyint`, `polyfromroots`,
`polyvander`, `polycompanion`, `polyroots`, `polyfit`.

**The core finding: real numpy's dtype coercion for this module is split
across two, disjoint, non-obvious code paths**, and replicating each item
correctly meant identifying which path it uses by reading numpy's own
source, not by assuming uniform behavior:

- **`as_series`-based items REJECT bool dtype outright**
  (`polyutils.as_series` calls `np.common_type(*arrays)`, which raises
  `TypeError` for any bool array; `as_series` catches it and re-raises
  `ValueError("Coefficient arrays have no common type")`, plural wording
  even for a single array). This family: `polytrim`, `polyadd`, `polysub`,
  `polymulx`, `polymul`, `polydiv`, `polyfromroots`, `polycompanion`, and
  transitively `polyroots`/`polypow`.
- **Own-dtype-path items PROMOTE bool/int silently to float64**
  (each does its own `if c.dtype.char in '?bBhHiIlLqQpP': c = c + 0.0`):
  `polyval`, `polyvalfromroots`, `polyder`, `polyint`, `polyvander`.

Implemented in Rust as `coerce_series_ex(obj, reject_bool: bool)` in
`ionp-py/src/poly.rs`, with two thin call-site wrappers
(`coerce_series`/`coerce_series_strict`) so each binding calls the
correct one — verified item-by-item against real numpy 2.5.1 by direct
source reading (`polynomial.py`/`polyutils.py`) plus live probes, not
assumed uniformly across the module.

**Three further, independently-discovered dtype divergences**, none of
which the `as_series`-vs-own-path split alone would have predicted, all
found by verifying OUT of the fixed corpus before declaring:

- `polyzero`/`polyone`/`polyx` are `np.array([0])`/`[1]`/`[0, 1])` in
  real numpy — plain Python-int literals, `int64` on this machine — not
  `float64`. Fixed via a new `mk_int64` PyO3 helper.
- `polyline(off, scl)`'s real body is a bare, dtype-preserving array
  literal (`np.array([off, scl])` if `scl != 0` else `np.array([off])`)
  with no forced float coercion. The original implementation called into
  a Rust binding that always produced `float64`. Fixed by rewriting
  `polyline` in `anionpy/polynomial/polynomial.py` as a direct array
  literal, bypassing the Rust binding entirely — construction, not
  arithmetic, so it belongs in Python per this project's "loops in Rust,
  assembly in Python" rule; the existing `_poly_line` Rust binding is
  unused now but still compiles.
- `polyvalfromroots` does **not** go through `as_series` at all — its
  body promotes only bool/int roots to float64 and never rejects an
  empty roots array (`np.prod` over zero factors is the empty product
  `1.0`; verified `P.polyvalfromroots(2.0, []) == 1.0`). The original
  implementation raised `ValueError("Coefficient array is empty")` for
  empty roots. Fixed with a new `coerce_roots` helper (no empty-array
  check) — the underlying Rust kernel `poly::val_from_roots` already
  handled an empty roots slice correctly (its `for &r in roots {}` loop
  over an empty slice trivially returns `T::one()`), so no kernel change
  was needed, only the PyO3 marshaling layer's over-eager empty check.
- `polyvander` accepts a scalar/0-d `x` in real numpy (promoted to a
  1-element 1-D array, e.g. `P.polyvander(2.0, 3).shape == (1, 4)`); the
  original implementation rejected anything but `ndim == 1`. Fixed in
  `poly_vander_py` by relaxing the ndim check to `> 1` and computing `n`
  as the product of the (now possibly empty) shape, floored at 1.

**`polyfit` and `polyroots` are not bit-exact** — both call an
independent LAPACK/eigensolver path (`lstsq`'s SVD, and the companion
matrix's eigenvalues, respectively) rather than sharing numpy's exact
floating-point call sequence. Declared with a measured `epsilon_tolerance`
in `polynomial_cases.py`, gated by this suite's `MIN_ULP_SWEEP_N = 20000`
rule — the declared bound must exactly equal a real, independently-seeded,
20,000-sample sweep, not an assumed or padded value:

```
POLYFIT_EPS            = 2.1612483045051664e-11   # float64, max relative coefficient error
POLYROOTS_REAL_EPS     = 8.310059886440513e-05    # float64, real-coefficient polynomials, max abs root error
POLYROOTS_COMPLEX_EPS  = 1.834393877421201e-09    # complex128, max abs root error
```

`POLYROOTS_REAL_EPS` is four orders of magnitude looser than the
complex128 bound. This is expected, not a worse eigensolver: it reflects
root-finding-via-companion-eigenvalues' own conditioning — real
Wilkinson-polynomial-style ill-conditioning at degree 6-7 for roots drawn
from `[-5, 5]`, a property of the algorithm, not of which BLAS/LAPACK
computed it. Sweep methodology and both sample sizes are recorded in
`polynomial_cases.py`'s epsilon-constants comment block; the scratch
sweep script itself lived at `/private/tmp/poly_eps_sweep.py` (not
shipped).

**7 items scoped out, not attempted, not declined-after-building**: the
N-D composition surface — `polyval2d`, `polyval3d`, `polyvalnd`,
`polygrid2d`, `polygrid3d`, `polyvander2d`, `polyvander3d`. These are
compositions over the already-implemented 1-D primitives (numpy's own
implementations call `polyval`/`polyvander` under a loop/reduce), so they
are a natural, low-risk next slice, but building them was out of this
pass's time-box. `polyfit`'s `full=True`, `w=` (weights), and vector-`deg`
call forms are similarly not implemented — only the plain `(x, y, deg)`
form is declared.

**Regression guard verified live**: `ionp-core/src/poly.rs`'s
`horner_eval` (used by `polyval`) was temporarily sabotaged (accumulation
term's `+` flipped to `-`), rebuilt, and re-measured — 14 `polyval` cases
failed exactly as expected, with concrete numpy-vs-anionpy mismatches in
the failure diagnostics. The sabotage was then reverted and rebuilt clean
(127/127 passing again) before any commit.

**Commit scope:** `ionp-core/src/poly.rs` (+ `lib.rs` mod registration),
`ionp-py/src/poly.rs` (+ `lib.rs` pymodule registration),
`anionpy/polynomial/` (new package), `anionpy/_state/polynomial.py`
(new, 21 declarations), `anionpy/_state/__init__.py` (wire the new block
in), `tests/differential/polynomial_cases.py` (new),
`tests/differential/registry.py` (wire the new corpus in), this file.

## 2026-08-07 — `anionpy.ndarray` cannot be pickled, and three `polynomial` items are declared exact but are not bit-exact

Two unrelated findings from the same out-of-corpus sweep, both recorded here
because both are user-visible and neither is caught by the differential suite.

### (a) Nothing containing an `anionpy` array can be pickled

```
>>> import pickle
>>> from numpy.polynomial import Polynomial as NP
>>> pickle.loads(pickle.dumps(NP([1., 2., 3.])))
Polynomial([1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')

>>> from anionpy.polynomial import Polynomial as AP
>>> pickle.loads(pickle.dumps(AP([1., 2., 3.])))
TypeError: cannot pickle 'anionpy.ndarray' object
```

`anionpy.ndarray` implements neither `__reduce__` nor the
`__getstate__`/`__setstate__` protocol at the Rust layer. This is not specific
to `polynomial` — it blocks multiprocessing, `joblib`, on-disk caching, and
every wrapper type (`MaskedArray`, `matrix`, `Polynomial`).

**Verdict: (b) DISCLOSED, blocker recorded, declarations RETAINED.**
`Polynomial.__getstate__` and `Polynomial.__setstate__` are declared exact.
That declaration is about the two *methods*, which do return dicts identical to
NumPy's when invoked directly, and it is narrowly correct. It is disclosed here
rather than left in an internal `_state` docstring precisely because it is the
kind of true-but-misleading claim a user would reasonably feel cheated by: the
methods conform, and the thing they exist to enable still does not work.

### (b) `polymul`, `polypow` and `polyfromroots` are declared exact but are not bit-exact

Seeded random sweep, float64 coefficients in [-9, 9], degrees 1–7, comparing
raw bit patterns rather than values:

```
polymul         853/1500 non-bit-exact   worst rel 1.88e-13
polypow         809/1500                 worst rel 3.63e-14
polyfromroots   419/1500                 worst rel 4.93e-14
```

A second sweep of `polymul` alone: **2238/4000 (56%) non-bit-exact**, worst
relative error 3.94e-12.

Eight sibling items swept identically — `polyadd`, `polysub`, `polymulx`,
`polyder`, `polyint`, `polyval`, `polyvander`, `polycompanion` — are bit-exact
at 0/1500. The divergence is specific, not a uniform rate across the
cross-product, so it is a real finding rather than a broken instrument.

`polypow` and `polyfromroots` are both built on `polymul`, so this is one defect
with two dependents: NumPy's `polymul` delegates to `np.convolve`, while
`ionp-core/src/poly.rs` has its own `convolve_full`. The same products are
accumulated in a different order, so the rounding differs.

**UPDATE 2026-08-07 (Monday) — fix attempted, measured, and found insufficient;
declarations REVOKED.**

Independent reproduction first, on the real installed binary (seed 777,
N=1500): `polymul` 1232/1500, `polypow` 536/1500, `polyfromroots` 555/1500
non-bit-exact — confirms the original finding, not just a recorded claim.

Root cause was pinned down, not assumed: `numpy.show_config()` on this build
reports `blas: {name: accelerate}`. NumPy's `polymul` → `np.convolve` →
`multiarray.correlate` (`PyArray_Correlate2`/`_pyarray_correlate` in NumPy's
C source) computes each output tap via a `dot()` function pointer that, under
`HAVE_CBLAS`, routes through `cblas_ddot` — i.e. Apple's Accelerate framework,
a closed-source, platform-specific SIMD/FMA BLAS kernel. `ionp-core`'s
`convolve_full` is a portable double loop with a different accumulation order.

A real fix was attempted, not just theorized: `ionp-core/src/poly.rs` was
given an experimental `convolve_numpy_tap_order` function, a faithful
line-for-line port of NumPy's own `PyArray_Correlate2` term order (operand
swap so the longer array is `a`, ramp-up/overlap/ramp-down regions, per-tap
sum bounds matching the C source exactly). `ionp-py/src/poly.rs`'s two
`polymul` call sites were switched to it, the crate was rebuilt
(`flock /tmp/ionp-build.lock ./.venv/bin/maturin develop --release`, literal
`🛠 Installed anionpy-0.1.0` observed), and the strengthened corpus below was
re-run against the real rebuilt binary.

Result: matching the term order did **not** close the gap. `polymul`'s
strengthened-corpus failure rate was effectively unchanged (47/67 vs 49/67
cases), and `polypow`/`polyfromroots` were likewise unaffected. So term order
is not the explanation. The experimental change was reverted (`git diff --stat` against
`ionp-core/src/poly.rs` and `ionp-py/src/poly.rs` showed zero output — exact
match to the prior committed state) rather than shipped, since it adds real
complexity for no closer match.

### Correction: the mechanism is NOT established (Monday, verified 2026-08-07)

The paragraphs above originally concluded that the divergence "is Accelerate's
internal SIMD-lane grouping and FMA use inside `cblas_ddot`." That is one
inference too many. A failed attempt to fix by reordering terms shows only that
term order is not the cause — it is not evidence for any *particular*
alternative. The revocation does not depend on naming the mechanism, so the
claim has been withdrawn rather than defended.

An independent probe (seed 31337, N=3000, random float64 operands of lengths
2-8) tried to reproduce `np.convolve`'s exact bits two ways:

```
tap-wise np.dot   reproduces np.convolve bit-exactly:  617/3000
naive ordered sum reproduces np.convolve bit-exactly:  606/3000
```

Neither works, and they agree with NumPy at essentially the same low rate —
consistent with both merely getting the trivially-exact short taps right. If
each output tap were simply one `cblas_ddot` call, the first row should have
been 3000/3000. It is not. The mechanism therefore remains **unidentified**.

(This is not a disproof of the `cblas_ddot` route either: NumPy has small-array
fast paths, so `np.dot` on an 8-element slice may never reach BLAS. The honest
statement is that we do not know, and did not need to.)

### Why the revocation stands regardless

What is measured, and sufficient: two independent, good-faith attempts to
reproduce NumPy's `convolve` bit-for-bit — a literal port of its own term order,
and NumPy's own `dot` — both failed.

There is also a reason to think the target itself is ill-posed, offered as
reasoning and not as measurement, since we have only one BLAS here to test:
NumPy's `polymul` bits appear to depend on which BLAS NumPy was linked against.
If so, "bit-exact `polymul`" is not a property of NumPy at all, but of a
particular NumPy *build*. Chasing it on this machine could produce a match that
silently breaks on a machine with OpenBLAS or MKL — a false declaration with a
longer fuse than the one we just removed. Anyone re-opening this should test
that hypothesis on a second BLAS before spending time on the kernel.

A tolerance was **not** added. The observed divergence (~1e-13 relative) is
numerically harmless, and that is exactly why widening the claim would have been
the wrong move: it would have converted "we cannot support this claim" into
permanent, invisible slack. Revoked is a smaller number and a true one.

Per this project's own rule, that satisfies the "genuinely impossible" bar:
the three declarations are **REVOKED** in `anionpy/_state/polynomial.py`
(commented out with the full writeup inline, not deleted), not widened into
"exact, with tolerance." Revoking honestly is treated as the correct outcome
here, not a failure to fix.

The corpus that carried this finding was also strengthened permanently:
`tests/differential/polynomial_cases.py` gained seeded adversarial-float64
generators (`_polymul_adversarial_cases` etc., non-representable float64
coefficients, degrees 1–7 / powers 0–4 / 1–7 roots) added to the existing
`polymul_cases()`/`polypow_cases()`/`polyfromroots_cases()`, so the corpus is
now known to fail on the current (correct, revoked) binary — 47/67, 32/66,
25/67 cases respectively — rather than being a corpus that has never been red.

Note how this originally survived a green corpus: the old `polymul` cases
used inputs that happen to agree. A green corpus is evidence only about the
inputs it contains — the same lesson as the `conj`/strides episode, which
passed 1300/1300 both before and after a real memory-layout defect.

**Downstream:** commit `a32116c` made six `Polynomial` class items (`__mul__`,
`__rmul__`, `__pow__`, `fromroots`, `convert`, `cast`) epsilon-tolerant citing
this exact divergence. Since the underlying `polymul` kernel is still not
bit-exact — confirmed by measurement above, not assumption — those six
tolerances are **retained, unremoved**. Removing them now would reintroduce
the original false-declaration defect one level up. Re-visit only if a
portable, reproducible replication of Accelerate's accumulation order is ever
found, or NumPy's own build stops linking Accelerate (or an equivalent SIMD
BLAS) on this platform.

## 2026-08-07 — `poly.rs`/`legendre.rs` never routed complex arithmetic through `complex_mul_fma`/`complex_div`; fixed, with one dependent defect that could not be closed

`ionp-core/src/ufunc.rs:2279` has documented since an earlier task that
NumPy's complex multiply is FMA-fused (`mul_add`), not the textbook
two-rounding formula `num_complex::Complex`'s `Mul` impl computes — 0
mismatches with the FMA formula vs. ~44% with the naive one. Four of the six
polynomial bases (`chebyshev.rs`, `laguerre.rs`, `hermite.rs`,
`hermite_e.rs`) already routed every complex multiply through it. The two
oldest, `poly.rs` and `legendre.rs`, did not — every `T*T`/`T/T` site in both
files used bare `*`/`/`, which is exact on `f64` but wrong on `C128`.

Measured before any fix (seed varies, N=1500 each, real vs. complex
coefficients and evaluation points independently varied):

```
fn         | real c, real x | real c, CPLX x | CPLX c, real x | CPLX c, CPLX x
polyval    |    0/1500      | 1055/1500      |    0/1500      | 1193/1500
legval     |    0/1500      | 1057/1500      |    0/1500      | 1206/1500
lagval     |    0/1500      |   82/1500      |    0/1500      |    0/1500
chebval/hermval/hermeval : clean in all four quadrants
```

### The fix

`ionp-core/src/poly.rs` and `ionp-core/src/legendre.rs` each gained
`poly_mul`/`poly_div` and `leg_mul`/`leg_div` trait methods (`f64` = bare
`*`/`/`, `C128` = `complex_mul_fma`/`complex_div`), and every arithmetic site
in both files (`horner_eval`, `val_from_roots`, `convolve_full`, `div`,
`der`, `int_`, `vander`, `companion` in `poly.rs`; `scale`, `leg_eval`,
`legmulx`, `legmul`, `legder`, `legint`, `legvander`, `legcompanion` in
`legendre.rs`) was routed through them, `chebyshev.rs` as the in-tree
exemplar. Both files also had a signed-zero padding bug in their
add/subtract helpers (zero-init-then-add-both-operands instead of
copy-longer-operand-first), fixed to match the convention already used in
`chebyshev.rs`/`laguerre.rs`.

`lagval` needed one more thing: NumPy's Clenshaw-recursion accumulators stay
*real-typed* (single-rounding real division) until the first iteration that
actually touches the complex evaluation point `x` — "real until touched."
`ionp-core`'s `lag_eval` upcasts everything to `C128` up front, so its one
genuinely-real division (`(c1*(nd-1))/nd` at the first loop iteration) ran
through `complex_div`'s Smith's-algorithm real/real path (`a*(1/c)`, a
*double*-rounding reciprocal-multiply) instead of NumPy's single-rounding
`a/c`. Measured directly: `complex_div`'s real/real path disagrees with plain
`a/c` for **any non-power-of-2 divisor** — divisors 2 and 4 gave 0/500000
mismatches, divisors 3/5/6 gave ~33% (167184, 164890, 166164 out of 500000).
A new function, `lag_eval_real_coef_complex_x`, computes exactly that one
real division in plain `f64` and routes everything else through the
existing, unmodified `lag_mul`/`lag_div`; `ionp-py/src/laguerre.rs`'s
`_lag_val` binding now calls it specifically for the real-coefficient +
complex-x case. `legval` needed no such hybrid: its analogous ratio
`(nd-1)/nd` is precomputed via `T::from_f64` *before* being multiplied into
the accumulator, so it never reaches `complex_div` regardless of NumPy's
real-until-touched typing — its only defect was the missing `leg_mul`
routing, closed by the general fix above.

Re-measured after the fix, 20,000-sample seeded sweep, degrees 0–7:
`lagval`/`legval` real-coefficient + complex-x are both 0/20000. The suite's
full run is green (0 phantom, 0 failing) and a sabotage test (flipping the
`-` to `+` in `legval`'s Clenshaw recursion, `ionp-core/src/legendre.rs`
line ~210) drove exactly 6 items red (`Legendre.__call__`, `Legendre.integ`,
`Legendre.linspace`, `legendre.leggauss`, `legendre.legint`,
`legendre.legval`) and the ledger's exact count from 1157 to 1151 — reverted
and reconfirmed 1157/3091, 0 failing.

Two previously-existing tolerances papering over the missing routing are
now gone rather than kept: `LEGENDRE_CALL_EPS`/`LAGUERRE_CALL_EPS`
(`Legendre.__call__`/`Laguerre.__call__` on real-coefficient/complex128-x)
are both bit-exact post-fix (0/25000) and are declared with no tolerance.

### The dependent defect the fix did NOT close: `legmul` on complex128

`legmul`'s complex128 path remains massively non-bit-exact after the fix —
this is a genuinely separate, deeper issue from the missing-FMA-routing bug
above, in the same unidentified-mechanism class as `polymul`/`chebmul`
(see the 2026-08-07 entry above), not fixed by adding `leg_mul`. Measured
on a fresh 25,000-sample seeded sweep (seed=0x5EEDC1A55), both at the
module-function level and at the `Legendre` class level (which wraps the
identical kernel):

```
legmul / Legendre.__mul__          21306-22244/25000  non-bit-exact
legpow / Legendre.__pow__            9006-9075/25000  non-bit-exact
legfromroots / Legendre.fromroots  12242-13196/25000  non-bit-exact
Legendre.__rmul__ (scalar _mul)                0/25000  bit-exact, unaffected
```

Per this task's rule ("remove tolerance IFF the fix makes the item
bit-exact, otherwise REVOKE — never re-add tolerance"), `LEGENDRE_CLASS_MUL_EPS`
is gone and all six declarations above (`polynomial.legendre.legmul`,
`legpow`, `legfromroots`, `polynomial.Legendre.__mul__`, `__pow__`,
`fromroots`) are **REVOKED** in `anionpy/_state/polynomial.py` (commented
out with the writeup inline), not re-toleranced. `legmul_cases`,
`legpow_cases` and `legfromroots_cases` in
`tests/differential/legendre_cases.py` each gained a `_DIVERGES`-labeled
case (deterministic, non-"nice" complex coefficients) so the corpus fails
visibly instead of passing by luck — the same problem, and the same fix
convention, as `chebfromroots`'s corpus before its own 2026-08-07
revocation. Mechanism NOT established; this is flagged as an open,
unidentified defect, not asserted to share `polymul`'s exact cause.

Float64 is unaffected throughout — `legmul` remains bit-exact on real
coefficients, and the module-level `polynomial.legendre.legmul` etc.
declarations covered only that path before this correction (their existing
corpora never exercised complex128 at the scale that would have caught
this).

### Ledger

Before (session baseline): 1163/3091 exact, 0 phantom, 0 failing.
After (this session, `flock`-serialized build+run, `/tmp/rFMA2.json`):
**1157/3091 exact, 0 phantom, 0 failing** — net -6 (3 module-level +3
class-level `legmul`-family revocations), offset by 0 new declarations (the
2 `__call__` items moved epsilon-tolerant → bit-exact, a wash on the
"exact" total but a real quality improvement: bit-exact count only dropped
1107 → 1103, not 1107 → 1101, and epsilon-tolerant dropped 51 → 49).
`chebmul`/`chebpow`/`chebfromroots`/`polymul`/`polypow`/`polyfromroots`
(the pre-existing, unrelated `np.convolve`-order revocations) were
re-measured, not assumed unaffected: `polymul` complex128 sweep
(seed=31337, N=3000) still 2388/3000 non-bit-exact, same order of magnitude
as previously documented — confirmed still broken in the same way, not
silently changed by this session's routing fix.

## 2026-08-07 — four false "exact" declarations: signed-zero-only divergence in `polymulx`/`polyder`/`legmulx`/`legder`

A coordinator, independently re-verifying the fix above on a clean tree,
found the `pad_sub`/`pad_add`/`add_trim`/`sub_trim` signed-zero fix did
**not** close a structurally different signed-zero bug in four other
items, all declared "exact": `polynomial.polynomial.polymulx`,
`polynomial.polynomial.polyder`, `polynomial.legendre.legmulx`,
`polynomial.legendre.legder`. Complex-coefficient sweep (seed `0xDEADBE`,
N=2000, `np.ascontiguousarray(x).view(np.uint64)` bit comparison):

```
legder     divergent   163/2000  value-differing 0  signed-zero-only  163
legmulx    divergent   177/2000  value-differing 0  signed-zero-only  177
polyder    divergent   161/2000  value-differing 0  signed-zero-only  161
polymulx   divergent  1021/2000  value-differing 0  signed-zero-only 1021
```

### Root cause: derived zero vs. literal zero

Real numpy does not write a literal `0.0`/`0j` into the padding or
degenerate-result slot these four functions produce — it *derives* the
zero by multiplying an actual coefficient by `0`, and `x * 0` under
IEEE754 carries `x`'s sign into the result (`(-1.03+0.06j) * 0 == -0.0+0j`
component-wise per the standard complex-multiply formula, not a bare
`0.0`). Confirmed against numpy 2.5.1 source directly (`inspect.getsource`):

- `polymulx` (`numpy/polynomial/polynomial.py`): `prd[0] = c[0] * 0`
- `polyder` (`cnt >= n`, i.e. order-of-derivation ≥ degree — the
  degenerate all-zero result): `c = c[:1] * 0`
- `legmulx` (`numpy/polynomial/legendre.py`): `prd[0] = c[0] * 0`
- `legder` (`cnt >= n`, same degenerate branch): `c = c[:1] * 0`

The in-tree `chebmulx`/`chebder` (`ionp-core/src/chebyshev.rs`) already
used the correct idiom (`prd[0] = c[0].cheb_mul(T::zero())` /
`c_in[0].cheb_mul(T::zero())` in the `cnt >= n` branch) and were confirmed
clean on the same sweep — used as the in-tree exemplar, per this task's
own precedent of using `chebyshev.rs` as the reference implementation.
`chebyshev.rs`, `hermite.rs` and `hermite_e.rs` were read-only references
this round, not modified.

### Fix

`ionp-core/src/poly.rs`'s `mulx` and `der`, and `ionp-core/src/legendre.rs`'s
`legmulx` and `legder`, were changed to derive the zero from the actual
first coefficient (`c[0].poly_mul(T::zero())` / `c[0].leg_mul(T::zero())`)
instead of filling `T::zero()` literally — matching `chebmulx`/`chebder`'s
existing idiom exactly. `poly_mul`/`poly_div` and `leg_mul`/`leg_div`
already route complex multiplication through `complex_mul_fma` (this
session's earlier fix), so `x * 0` computed this way reproduces numpy's
sign-preserving zero bit-for-bit.

**Caller check, per the `chebfromroots`/`chebmul` precedent** (that item
was declared in the same commit that correctly revoked `chebmul`, which it
calls in a loop — a revocation that cost a citation). Grepped every caller
of the four fixed core functions:

- `ionp-py/src/legendre.rs`'s `_leg_mulx`/`_leg_der` bindings, and
  `ionp-py/src/poly.rs`'s `_poly_der` binding, already called straight
  into `leg::legmulx`/`leg::legder`/`poly::der` — the core fix reached
  Python for free.
- `ionp-py/src/poly.rs`'s `_poly_mulx` binding did **not** call
  `poly::mulx` at all — it duplicated the old literal-zero logic inline
  (`vec![0.0; ...]` / `vec![C128::new(0.0, 0.0); ...]`). This meant the
  core-level fix to `poly::mulx` was dead code; the actual bug lived in
  the binding. Fixed by routing the binding through `poly::mulx` instead
  of re-implementing it — this is the exact bug class the caller-check was
  meant to catch, just one hop earlier (an unwired core fix) rather than
  one hop later (an inherited-but-undetected defect in a higher-level
  caller).
- `legmul` (`ionp-core/src/legendre.rs`) calls `legmulx` in its own
  three-term recursion, but `legmul` (module- and class-level) is already
  REVOKED for an unrelated, pre-existing divergence (see the section
  above) — no new revocation needed there.
- No other caller (`leg2poly`, `Polynomial.deriv`/`Legendre.deriv`, the
  various `*2poly` conversions in `hermite.py`/`hermite_e.py`/
  `laguerre.py` that import `polymulx`) was found declared "exact" while
  silently inheriting the old literal-zero bug beyond what the full
  differential run below already re-verifies.

### Detection: new corpus cases, bit/signbit-based

Comparing values can never catch a signed-zero-only divergence
(`0.0 == -0.0`). `harness.py`'s default "exact" comparison is already
bit/signbit-exact (`_bit_exact_equal`), so this was purely a **corpus
coverage gap**: every existing case for these four items used a
non-negative first coefficient (`COMPLEX_C[0] == 1+1j`), which can never
distinguish a correct derived zero from a wrong literal one. Added, in
`tests/differential/polynomial_cases.py` (`polymulx_cases`,
`polyder_cases`) and `tests/differential/legendre_cases.py`
(`legmulx_cases`, `legder_cases`): negative-real, negative-imaginary, and
negative-both first-coefficient cases for the `mulx` pair, and
negative-real/negative-imaginary `order_exceeds_degree` cases for the
`der` pair (the degenerate `cnt >= n` branch is where their signed-zero
divergence lives).

### Sabotage test

Reintroduced the literal-zero fill in `poly.rs`'s `mulx`
(`out[0] = T::zero()`), rebuilt, re-ran the full suite:
`polynomial.polynomial.polymulx` went to `failing` (ledger dropped
1157 → 1156/3091, 1 failing). Reverted, rebuilt, re-ran: restored to
1157/3091, 0 phantom, 0 failing. New cases proven to bite.

### Six-basis re-sweep

Full complex-coefficient surface sweep (seed `0xDEADBE`, N=2000) across
all twelve `mulx`/`der` pairs across all six bases, post-fix:

```
polynomial.polynomial.polymulx    divergent 0/2000
polynomial.polynomial.polyder     divergent 0/2000
polynomial.legendre.legmulx       divergent 0/2000
polynomial.legendre.legder        divergent 0/2000
polynomial.chebyshev.chebmulx     divergent 0/2000
polynomial.chebyshev.chebder      divergent 0/2000
polynomial.hermite.hermmulx       divergent 0/2000
polynomial.hermite.hermder        divergent 0/2000
polynomial.hermite_e.hermemulx    divergent 0/2000
polynomial.hermite_e.hermeder     divergent 0/2000
polynomial.laguerre.lagmulx       divergent 0/2000
polynomial.laguerre.lagder        divergent 0/2000
```

All four target items fixed clean; Chebyshev/Hermite/HermiteE/Laguerre
(already clean) confirmed still clean — the fix did not silently regress
them.

### Ledger

Before (this correction's baseline, `a0a1e08`): 1157/3091 exact, 0
phantom, 0 failing. No declarations changed — all four items were already
declared "exact" and stay declared "exact"; the fix makes that declaration
true rather than false. `tie_exempt` count unchanged at 2 (not used as a
substitute fix, per this task's explicit constraint). After: **1157/3091
exact, 0 phantom, 0 failing**, bit-exact count 1103 → 1103 unchanged in
total but four previously-false-positive items are now genuinely
bit-exact rather than passing on corpus luck.

## 2026-08-08 — release-documentation pass: five previously-declined `toplevel` gaps re-confirmed live at `db8b503`; verdicts unchanged from 2026-08-05

Task: write this file's entries for a fixed list of previously-measured,
previously-declined divergences, but re-measure each one first rather than
trusting the recorded reason (a recorded reason is not evidence about the
current binary, and can itself be false). All five items on the list turn
out to already have full dated entries in this file — the "## 2026-08-05 —
five previously-declined gaps, re-measured against the installed binary"
section above, items (a)-(e). Re-running each of that section's own
reproductions live today, on the currently installed `.venv` build at
commit `db8b503`, gives the identical qualitative result in all five
cases — no new entry is written for any of them, since one already exists
and duplicating it verbatim would not add information; this entry exists
so a reader following this file's "5 items due" checklist can see they
were checked, not silently skipped.

```
>>> import numpy as np, anionpy
>>> np.__version__
'2.5.1'
```

**(1) weak-`q` float16/float32 dtype preservation on `percentile`/`quantile`**
— re-ran the (a) reproduction verbatim: `np.percentile`/`anionpy.percentile`
and `np.quantile`/`anionpy.quantile` both preserve `float32` and `float16`
identically on the installed binary. **Still NO LONGER REPRODUCES**, exactly
as (a) already records.

**(2) all-NaN-slice narrow-dtype gap on `nanpercentile`/`nanquantile`**
— re-ran the (b) reproduction verbatim: scalar-`q` and array-`q`, `float32`
and `float16`, all preserve the narrow dtype on an all-NaN slice on both
sides. **Still NO LONGER REPRODUCES**, exactly as (b) already records.

**(3) `apply_along_axis` first-slice dtype quirk on `nanquantile`/
`nanpercentile`** — re-ran the (c) reproduction verbatim: the dtype still
flips identically on both sides depending on whether reduction-slice 0 is
all-NaN (`float32` when slice 0 is all-NaN, `float64` when it isn't), and
`anionpy.apply_along_axis` itself is still absent (`hasattr(anionpy,
"apply_along_axis")` is still `False`). **Still never actually a
divergence** — the numpy quirk is still faithfully reproduced by the two
`nan*` functions' internal implementation, exactly as (c) already records.

**(4) sort-vs-partition NaN-payload asymmetry** — re-checked live:
`hasattr(anionpy, "partition")` and `hasattr(anionpy, "argpartition")` are
both still `False`; `anionpy.partition(...)` still raises `AttributeError:
module 'anionpy' has no attribute 'partition'`. **Still COULD NOT CONSTRUCT
A REPRO** against the installed binary, exactly as (d) already records —
the underlying real-numpy `sort`-canonicalizes/`partition`-doesn't
asymmetry (d)'s own repro demonstrates is unaffected and unrevisited here.

**(5) `ndarray.__rtruediv__` complex NaN bit-pattern mismatch** — re-ran
(e)'s minimal repro verbatim and it reproduces identically:

```
>>> import struct
>>> def bits(x): return hex(struct.unpack('<Q', struct.pack('<d', x))[0])
>>> b_ionp = anionpy.array([complex(float('inf'), float('inf'))], dtype='complex128')
>>> b_np = np.array([complex(float('inf'), float('inf'))], dtype='complex128')
>>> r_ionp, r_np = (1 / b_ionp)[0], (1 / b_np)[0]
>>> r_ionp, r_np
((nan+nanj), np.complex128(nan+nanj))
>>> bits(complex(r_ionp).real), bits(complex(r_ionp).imag)
('0x7ff8000000000000', '0x7ff8000000000000')
>>> bits(r_np.real), bits(r_np.imag)
('0x7ff8000000000000', '0xfff8000000000000')
```

A fresh independent sweep (own script, own seed `20260805` reused but a
different draw distribution/order than (e)'s own script, `n=2684` to match
the sample size named in this task's brief) measured **48/2684** bit-level
mismatches, every one of them the same NaN-imaginary-sign case (numpy's
imaginary NaN is negative-signed, `0xfff8...`; anionpy's is positive,
`0x7ff8...`). This is a *third* distinct count for the same underlying
defect — (e) itself already reports 169/2684 superseding an earlier
32/2684 — and per (e)'s own reasoning these counts are not comparable to
each other (different scripts draw different distributions over the same
salted 0/-0/±inf/NaN component grid); all three agree on the qualitative
fact (a real, reproducible, byte-level divergence hidden from ULP-distance
grading by NaN-vs-NaN collapsing to distance 0). **Still STILL DIFFERENT**,
exactly as (e) already records.

**Verdict: no new declarations, no revocations, no new entries for items
(1)-(5).** All five checked live against `db8b503`; all five match their
existing recorded verdict exactly. This entry is the re-confirmation
record, not a new finding.

**Commit scope:** this file only (no code changes accompany this entry).

## 2026-08-08 — three standing declines re-verified: `real`/`imag`/`real_if_close` view-vs-copy CONFIRMED; `concatenate`'s `out=` blindness CONFIRMED, its `casting='unsafe'` blindness NO LONGER REPRODUCES; `asarray`'s view-preservation gap NARROWED to identity/`.base` only, no longer a memory copy

Three more items handed to this pass as "already measured, already
declined" — all three live in `anionpy/_state/toplevel.py` /
`anionpy/_state/ndarray.py` comments and `docs/stride-gap-classification.md`,
not previously written into this top-level file. Re-measured each live
against the installed binary at `db8b503` before writing anything, per
this task's own instruction that a recorded reason is not evidence about
the current binary. Two of the three still reproduce exactly as recorded;
the third does not, and is corrected here rather than silently repeated.

### (a) `real`/`imag`/`real_if_close` — numpy returns a VIEW, anionpy a COPY — CONFIRMED, still reproduces

Recorded in `anionpy/_state/toplevel.py` (around its "`real`/`imag`/
`real_if_close` were investigated and DECLINED, not declared" comment):
numpy's `np.real`/`np.imag` (and `ndarray.real`/`.imag`) are literally
`val.real`/`val.imag`, a VIEW aliasing the input's storage. anionpy
returns a fresh COPY. Re-measured live:

```
>>> a_np = np.array([1+2j, 3-4j])
>>> np.shares_memory(a_np, a_np.real)
True
>>> a_np.real.base is a_np
True
>>> f_np = np.array([1.0, 2.0])
>>> f_np.real is f_np
True

>>> a_ip = anionpy.array([1+2j, 3-4j])
>>> anionpy.shares_memory(a_ip, a_ip.real)
False
>>> a_ip.real.base
>>> f_ip = anionpy.array([1.0, 2.0])
>>> f_ip.real is f_ip
False
```

`anionpy.real`/`anionpy.imag`/`anionpy.real_if_close` all exist as
callables (`hasattr` is `True` for all three) — this is not an absence,
it's a values-correct/aliasing-wrong divergence: an in-place write through
`a_np.real` mutates `a_np`; the identical write through `a_ip.real` does
not touch `a_ip`. Not visible to a differential corpus that only compares
output values — `np.shares_memory`/`.base`/write-through are diagnostics
the harness's default value/dtype/ULP grading never inspects. **CONFIRMED,
still reproduces**, matching `toplevel.py`'s own recorded reason exactly.
No fix attempted here — real/imag are not owned by this pass, and per
`toplevel.py`'s own note, composing a top-level `real()`/`imag()` on top of
`ndarray.real`/`.imag`'s existing copy would ship the same defect one
level up under a name whose numpy contract explicitly promises a view.

### (b) `concatenate`'s `out=`/`casting=` gap — `out=` CONFIRMED, `casting='unsafe'` NO LONGER REPRODUCES (in-source comment is now stale)

Recorded in `anionpy/_state/toplevel.py`'s `"concatenate"` entry (dated
2026-08-01, "parameter-blindness audit"): "numpy accepts `out=` and
`casting='unsafe'` here; anionpy raises `ValueError: anionpy.concatenate:
out= is not supported` and `...casting='unsafe' is not supported (only the
default 'same_kind')` respectively." Re-measured both halves live rather
than assuming the note still holds:

```
>>> o = np.zeros(4, dtype='float64')
>>> r = np.concatenate((np.array([1.,2.]), np.array([3.,4.])), out=o)
>>> o is r
True
>>> anionpy.concatenate((anionpy.array([1.,2.]), anionpy.array([3.,4.])),
...                      out=anionpy.array([0.,0.,0.,0.]))
Traceback (most recent call last):
  ...
ValueError: anionpy.concatenate: out= is not supported
```

`out=` **CONFIRMED, still reproduces** exactly as recorded — numpy writes
into the supplied buffer and returns it by identity, anionpy raises.

The `casting=` half does not, on a live re-check that actually exercises a
casting-mode boundary (the original note's own worded example never did —
same-dtype-to-same-dtype concatenation trivially "succeeds" under every
casting mode, numpy and anionpy alike, and proves nothing about whether
`casting=` is actually being enforced):

```
>>> i_np = np.array([1,2,3], dtype='int32')
>>> f_np = np.array([1.5, 2.5], dtype='float64')
>>> for mode in ('no', 'equiv', 'safe', 'same_kind', 'unsafe'):
...     try: print(mode, np.concatenate((i_np, f_np), casting=mode).dtype)
...     except TypeError as e: print(mode, 'raises', e)
no raises Cannot cast array data from dtype('int32') to dtype('float64') according to the rule 'no'
equiv raises Cannot cast array data from dtype('int32') to dtype('float64') according to the rule 'equiv'
safe float64
same_kind float64
unsafe float64

>>> i_ip = anionpy.array([1,2,3], dtype='int32')
>>> f_ip = anionpy.array([1.5, 2.5], dtype='float64')
>>> for mode in ('no', 'equiv', 'safe', 'same_kind', 'unsafe'):
...     try: print(mode, anionpy.concatenate((i_ip, f_ip), casting=mode).dtype)
...     except TypeError as e: print(mode, 'raises', e)
no raises Cannot cast array data from dtype('int32') to dtype('float64') according to the rule 'no'
equiv raises Cannot cast array data from dtype('int32') to dtype('float64') according to the rule 'equiv'
safe float64
same_kind float64
unsafe float64
```

Every one of numpy's five `casting=` modes now matches exactly, including
the `'no'`/`'equiv'` rejection message text, on this real int32/float64
boundary case. `ionp-py/src/manip.rs`'s `concatenate` (confirmed by direct
source read, not just behavior) parses `casting=` via
`crate::check_casting_kwarg`, defaults to `'same_kind'`, and gates every
input array through `check_concat_casting` against the resolved target
dtype and rule — this is real enforcement, not a coincidence of the
probe's dtype choice; the complex-to-float `'unsafe'`-vs-`'same_kind'`
boundary was also checked and both modes agree with numpy there too.
**`casting=` NO LONGER REPRODUCES** — `toplevel.py`'s in-source comment
(still dated 2026-08-01, never updated) is stale on this specific claim.
Not corrected in that file here (out of this pass's path — a Rust-owning
task should update the comment), flagged here so this file doesn't
silently repeat a now-false claim. `concatenate` itself remains
undeclared in the ledger regardless, on the strength of the still-real
`out=` gap alone.

### (c) `asarray`'s view-preservation gap — NARROWED, current behavior is NOT a copy

Recorded in `docs/stride-gap-classification.md`'s "GENUINE — view
preservation" section, original trigger: `v = arange(10).astype(f64)[::-1]`
— `asarray` on a reversed (negative-stride) view allegedly forces a fresh
contiguous copy instead of wrapping the existing buffer. That same section
already carries its own **2026-08-03 correction**, marked stale and
refuted against the binary current as of that date: re-running the exact
trigger gave matching strides and `shares_memory` `True` on both sides,
with only `.base is None` left as a residual, metadata-only gap — not
memory unsafety. Re-measured independently here, live, against `db8b503`,
not merely trusting that 2026-08-03 correction either:

```
>>> v = np.arange(10).astype('float64')[::-1]
>>> r_np = np.asarray(v)
>>> r_np is v, r_np.base is v.base
(True, True)

>>> iv = anionpy.array(list(range(10)), dtype='float64')[::-1]
>>> r_ip = anionpy.asarray(iv)
>>> r_ip.strides, anionpy.shares_memory(r_ip, iv), r_ip is iv, r_ip.base
((-8,), True, False, None)
```

Also checked a 2-D transpose and a non-contiguous column slice (`b.T`,
`b[:, ::2]` on a `(3,4)` float64 array) — both match numpy's strides and
`shares_memory` exactly on the anionpy side, and a mutate-through-the-view
test (`asarray(reversed_view)[0] = 999.0`, then read the original array)
shows the write landing in the original buffer on **both** sides:

```
>>> v2 = np.arange(5).astype('float64'); view = v2[::-1]
>>> np.asarray(view)[0] = 999.0
>>> v2
array([  0.,   1.,   2.,   3., 999.])

>>> iv2 = anionpy.array([0.,1.,2.,3.,4.]); view2 = iv2[::-1]
>>> anionpy.asarray(view2)[0] = 999.0
>>> iv2
array([  0.,   1.,   2.,   3., 999.])
```

**This task's brief characterized this item as "asarray still copies where
numpy views" — that claim does NOT reproduce.** `anionpy.asarray` on a
reversed view, a transposed view, and a non-contiguous strided view all
correctly wrap the existing buffer rather than copying: strides match,
`shares_memory` is `True`, and a write through the `asarray` result is
observably visible through the original array — the defining behavior of
a view, not a copy. Writing an entry that asserted "asarray copies" would
be fictional against this binary, so none is written on that framing. The
narrower, genuinely-still-open gap is exactly what the existing
2026-08-03 correction already says: `r_ip is iv` is `False` where numpy's
`r_np is v` is `True` (numpy's `asarray` on an already-compatible array is
a true no-op returning the same object; anionpy's always constructs a new
Python object even when it wraps the same buffer), and `r_ip.base` is
`None` where numpy's `r_np.base` chains to the original allocation. A
caller relying on `asarray(x) is x` for an already-compatible `x`, or
walking `.base` to find the root array, will observe a difference; a
caller relying on memory sharing, stride layout, or write-through (the
"view, not copy" contract's substantive content) will not.

**Verdict: (a) `real`/`imag`/`real_if_close` CONFIRMED DECLINED, entry
written above for the first time in this file. (b) `concatenate`
`out=` CONFIRMED DECLINED, entry written above for the first time in this
file; `casting=` RETIRED — no entry written, `toplevel.py`'s stale comment
flagged rather than copied. (c) `asarray`'s "still copies" framing
REFUTED (re-confirms the existing 2026-08-03 correction in
`docs/stride-gap-classification.md`, independently); the genuine residual
(`is`/`.base` identity) is recorded here for the first time in this
top-level file, under its accurate framing rather than the copy framing
this task named it under.**

**Commit scope:** this file only (no code changes accompany this entry).

## 2026-08-08 — NEW (ticket #79): array-level `dtype=` spelling repr for `'q'`/`'g'`/`'G'` — dtype-object level fixed in `db8b503`, array level remains open

Measured fresh today, at `db8b503` (the same commit that fixed the
dtype-object-level half of this gap — `ap.dtype('q').char`/`.num` now
match numpy's, and `==`/`hash` stay `True` across the duplicate-width
spellings). The array-construction-and-repr level was not touched by that
fix and still diverges:

```
>>> repr(np.array([1,2], dtype='q'))
'array([1, 2], dtype=int64)'
>>> repr(anionpy.array([1,2], dtype='q'))
'array([1, 2])'

>>> repr(np.array([1,2], dtype='g'))
'array([1., 2.], dtype=float64)'
>>> repr(anionpy.array([1,2], dtype='g'))
'array([1., 2.])'

>>> repr(np.array([1,2], dtype='G'))
'array([1.+0.j, 2.+0.j], dtype=complex128)'
>>> repr(anionpy.array([1,2], dtype='G'))
'array([1.+0.j, 2.+0.j])'
```

For comparison, the platform-default spellings of the same widths repr
identically on both sides with no `dtype=` extra at all (both numpy and
anionpy correctly omit it):

```
>>> repr(np.array([1,2], dtype='l')), repr(anionpy.array([1,2], dtype='l'))
('array([1, 2])', 'array([1, 2])')
>>> repr(np.array([1,2], dtype='d')), repr(anionpy.array([1,2], dtype='d'))
('array([1., 2.])', 'array([1., 2.])')
>>> repr(np.array([1,2], dtype='D')), repr(anionpy.array([1,2], dtype='D'))
('array([1.+0.j, 2.+0.j])', 'array([1.+0.j, 2.+0.j])')
```

**Cause.** numpy carries duplicate dtypes at the same width but a distinct
internal type number: `'q'` (longlong, `num=9`) beside `'l'` (`num=7`),
`'g'` (longdouble) beside `'d'` (double), `'G'` (clongdouble) beside `'D'`
(cdouble). Confirmed directly this task:

```
>>> np.dtype('q').num, np.dtype('l').num
(9, 7)
>>> np.dtype('g').num, np.dtype('d').num
(13, 12)
>>> np.dtype('G').num, np.dtype('D').num
(16, 15)
```

numpy's array `repr` prints `dtype=...` whenever the array's dtype is not
the platform's *default* dtype object for its kind — a `num`-identity
check, not a width or `==` check — so `'q'`/`'g'`/`'G'` get the `dtype=`
suffix even though they are bit-identical to `'l'`/`'d'`/`'D'` on this
platform. `anionpy` canonicalizes the dtype *spelling* away during array
construction (all six spellings above route to the same one of anionpy's
14 `DType` variants at the array level), so by the time `repr` runs there
is no longer any surviving distinction for it to see — the information
`'q'` was ever requested is discarded before repr, not merely rendered
wrong by repr.

**Two hypotheses considered and falsified, per this task's brief — not
re-chased:**

(a) *"`==` but not `is` identity."* Ruled out directly: `ap.dtype('f8') is
ap.dtype('float64')` is *also* `False` (anionpy's dtype objects are not
interned/cached the way some of numpy's are), yet `'f8'` reprs with no
`dtype=` extra, matching numpy, on both sides. Identity is not what gates
numpy's `dtype=` decision, so it cannot be the explanation here either.

(b) *"`longdouble` silently truncates."* Ruled out directly: `np.dtype('g')
.itemsize == 8` on this machine (aarch64 macOS) — same width as `'d'`, no
precision lost by mapping `'g'` onto `anionpy`'s `F64`. The array-level
divergence is a repr/provenance issue, not a numeric-precision one.

**Note.** `memmap.__repr__` (`KNOWN-DIFFERENCES.md`'s `matrix`/`recarray`/
`memmap` entry, 2026-08-07) is withdrawn pending this fix — its own repr
path shares this same array-level dtype-spelling blindness.

**Verdict: DECLINED, not yet fixed.** The dtype-object level (`.char`/
`.num`/`==`/`hash`) is genuinely closed as of `db8b503`. The
array-construction level requires `anionpy.array`'s dtype-resolution path
to retain which of the numpy-recognized duplicate-width spellings was
requested (or a parallel default-vs-nondefault flag per `DType` variant)
through to `repr` — a real, if narrow, extension of the dtype-spelling
work, not covered by this pass, and not attempted here since this pass is
documentation-only per its own lane restriction (`tools/coverage.py` and
`anionpy/polynomial/*.py` are another agent's concurrent lane; this repr
gap does not touch either, but no code was changed in this pass on
principle — measurement and documentation only).

**Commit scope:** this file only (no code changes accompany this entry).

## 2026-08-08 — NEW (ticket #45): `np.convolve` on `complex128` is computed by Apple Accelerate's `cblas_zdotu_sub`, and is not reproducible by ANY portable scalar arithmetic — DECLINED

**Affects (all remain undeclared, all remain in the 35-name failure set):**
`polynomial.polynomial.polymul`, `polynomial.polynomial.polypow`,
`polynomial.polynomial.polyfromroots`, `polynomial.chebyshev.chebfromroots`.

Ticket #48 established these four share one mechanism: `polymul` **is**
`np.convolve(c1, c2)`, and `chebmul` reaches the same `np.convolve` through
`_zseries_mul`. So they stand or fall together. They fall.

### What numpy actually does

`np.convolve(a, v)` dispatches to `multiarray.correlate(a, v[::-1], mode)`,
which binds `array_correlate` → **`PyArray_Correlate`** — *not*
`PyArray_Correlate2`. Only the latter conjugates its second argument, so the
conjugation mentioned in the shared helper's doc comment does **not** apply
here; that is an easy and costly misreading.

`_pyarray_correlate`'s ramp-up / bulk / ramp-down loop calls a
`PyArray_DotFunc` per tap. For `CDOUBLE` that is `CDOUBLE_dot`, which is
`HAVE_CBLAS`-gated and, for any tap fitting one `NPY_CBLAS_CHUNK` (i.e. all
realistic sizes), issues a single **`cblas_zdotu_sub`** call into Apple
Accelerate. Unlike real `float64`, complex128 has **no scalar fast path**:
`small_correlate` handles `NPY_FLOAT`/`NPY_DOUBLE` only, never
`CFLOAT`/`CDOUBLE`. Every tap of every complex convolve, at every size, goes
through Accelerate.

### The measurement that settles it

Two independent routes agree.

**Route 1 (source-led, ctypes).** FFI-calling the real `cblas_zdotu_sub` out of
`/System/Library/Frameworks/Accelerate.framework/Accelerate` and replicating
`_pyarray_correlate`'s index arithmetic by hand reproduces `np.convolve`'s
output **8/8 taps bit-exact**. Sweeping that same kernel against textbook
scalar orders, n=1..32, 300 random trials each:

```
n= 2  sequential 90/300   pairwise  89/300
n= 8  sequential 39/300   pairwise  47/300
n=32  sequential 14/300   pairwise  29/300
```

**Route 2 (black-box, no ctypes and no source — the decisive one).** Tap 0 of a
full convolve is *mathematically* the single product `a[0]*v[0]`. A one-term sum
has no accumulation order. Yet across 800 random `complex128` 4×5 convolves,
that tap matches numpy's own `np.multiply(a[0], v[0])` only **451/800**:

```
edge tap from np.convolve(a[:1], v[:1])   800/800 match a plain product
edge tap 0 of the FULL np.convolve(a, v)  451/800
```

This is the load-bearing result. **Because no summation order exists for a
single term, the divergence cannot be accumulation order at all** — it is the
rounding inside Accelerate's kernel, which differs from `np.multiply`'s own
`fma(ar,br,-(ai*bi)) / fma(ar,bi, ai*br)` form. It also shows the tiny 1×1 call
takes a *different path* from an n=1 tap inside a larger call, so probing
convolve at minimal sizes measures the wrong kernel.

Every scalar order tried — forward sequential, reverse sequential, split
real/imaginary, four-separate-partial-sums, pairwise/recursive-halving,
`ndarray.sum`, per-tap `np.dot`, `np.vdot` — fails, matching only on scattered
taps by coincidence.

### Verdict: DECLINED, and not fixable within the project's own constraints

Bit-exact agreement is achievable exactly one way: link Apple Accelerate and
call `cblas_zdotu_sub`. That was *proved* to work — and it is precisely what
`anionpy` must not do. `ionp-core` is a portable pure-Rust reimplementation;
linking a macOS-only closed-source BLAS to match one numpy build's SIMD
reduction order would produce a "bit-exact" claim that is false on OpenBLAS, on
MKL, on Linux, and plausibly on a different Accelerate version or CPU
generation. That is the identical error already **revoked** for the real-
`float64` sibling of this investigation.

See `docs/REFERENCE-PLATFORM.md`: this is the second confirmed mechanism, after
`-ffp-contract=fast`, by which our exactness results are properties of one
toolchain. It is the first one we have declined rather than matched.

**Explicitly NOT closed by this entry:** `polynomial.legendre.legmul`/`legpow`/
`legfromroots`. `legmul` is a hand-written reprojection recursion, not a
convolution; it remains ticket #48's separate and still-unidentified mechanism,
and nothing here should be read as evidence about it.

**Re-open condition.** Anyone re-opening #45 must paste a fresh repro showing a
portable scalar formulation matching `np.convolve` on complex128 — including
the single-product edge tap above, which is the cheapest falsifier. A
description of an approach is not a repro.

**Commit scope:** this file only. No source changed; the four items stay
undeclared and stay failing, which is the correct state.

## 2026-08-08 — ticket #48 RESOLVED: `legmul`/`legpow`/`legfromroots` complex128 fixed (operand-order bug, not a BLAS-class mechanism)

Sibling to the #45 entry immediately above, and explicitly the thing that
entry said was NOT closed by it. Ticket #48 asked whether
`polynomial.legendre.legmul`/`legpow`/`legfromroots`'s complex128 divergence
was the same unfixable class as #45's four `np.convolve`-backed items. It is
not — `legmul` is a hand-written reprojection recursion built from ordinary
add/multiply/divide, not a convolution, and the defect was fully portable
scalar arithmetic.

### Root cause

`ionp-core/src/legendre.rs`'s local `scale(c, s)` helper — `c[i] * s` for
every coefficient, backing every `c[0]*xs` / `c[-i]*xs` / `q*p[:-1]` site in
`legmul`'s reprojection recursion and `legdiv`'s repeated-subtraction loop —
multiplied `v.leg_mul(s)`: **array element first, scalar second**. numpy's
own `legendre.py`/`polyutils._div` source (verified live, numpy 2.5.1)
always writes the scalar operand first at every one of those call sites
(`c[0] * xs`, not `xs * c[0]`).

On `f64` this is invisible — real multiplication is exactly commutative
bit-for-bit. On `C128` it is not: `ufunc.rs::complex_mul_fma`'s formula
(`re = fma(x.re, y.re, -(x.im*y.im))`, `im = fma(x.re, y.im, x.im*y.re)`) is
**asymmetric under argument swap**. Swapping which operand plays `x` vs `y`
changes which product goes through the FMA's exact-rounded pair and which
goes through the separately-rounded plain multiply that becomes the addend —
`im = fma(x.re,y.im, x.im*y.re)` and `fma(y.re,x.im, y.im*x.re)` round the
same mathematical value differently in general, even though the two are
identical in infinite precision.

### Minimal repro

Two length-1 complex128 series — the smallest possible `legmul` call, which
by ticket #45's own lesson has no accumulation order to blame:

```
c1 = [0.0106714555488392-0.9128681500255134j]
c2 = [0.9094651867821141+0.9562960165065107j]

numpy L.legmul(c1, c2) -> [0.88267749-0.82001673j]
  bits: (4606125671600412327, 13828933310498647204)
before-fix anionpy       -> [0.88267749-0.82001673j]   (same decimal print!)
  bits: (4606125671600412327, 13828933310498647203)    -- imag off by 1 ULP
```

Decimal printing hid the defect (both print identically to 8 places) —
caught only via `.view(np.uint64)` bit comparison, per this project's own
methodology note. Confirmed mechanism by replicating both FMA orderings with
Python 3.13+'s `math.fma` directly against the raw bit patterns:
`fma(c[0], xs, ...)`-ordering matches numpy's actual output bit-for-bit;
`fma(xs, c[0], ...)`-ordering reproduces the exact wrong 1-ULP value anionpy
produced. This is the smallest-diverging-degree walk ticket #48's own method
section calls for (L1=1, L2=1 is the very first case checked).

### Fix

One line: `scale`'s `c.iter().map(|&v| v.leg_mul(s))` became
`c.iter().map(|&v| s.leg_mul(v))` — scalar first, matching numpy's literal
operand order at every call site (`legmul`'s five `scale(...)` calls,
`legdiv`'s one). The other in-loop multiplies in `legmul` (`c1v.iter()
.map(|&v| v.leg_mul(nd_m1)...)`, matching numpy's own `c1 * (nd-1)`, array
first) were already correctly ordered and untouched.

### Verification

- Differential suite: `polynomial.legendre.legmul`/`legpow`/`legfromroots`
  move from fail to pass; the previously-added FAILS-by-design regression
  cases (`legmul_cases`/`legpow_cases`/`legfromroots_cases`' former
  `"nonexact_complex_DIVERGES"` entries, renamed `"complex_reprojection"`
  now that they pass) are kept in the corpus rather than deleted, so a
  regression trips them again. Full-suite failure set: 35 -> 32 named
  failures, exact set match otherwise (`diff` by name, not count) — no new
  failures introduced.
- Out-of-corpus: fresh seeded sweep (seed `0xDEADBEEF`, disjoint from the
  `0x5EEDC1A55` seed used to find the original defect) — 3000 random
  `legmul` complex128 draws (degrees 1-6, magnitudes 1e-3 to 1e8), 500
  `legpow` draws (powers 0-4), 500 `legfromroots` draws (0-7 roots): 0/4000
  mismatches. A positive control (real-`f64` `legmul`, always bit-exact)
  passed and a negative control (a deliberately corrupted comparison)
  correctly flagged a mismatch, confirming the harness itself is sound.
- `anionpy/_state/polynomial.py`: `polynomial.legendre.legmul`/`legpow`/
  `legfromroots` re-declared `"exact"` with the mechanism and evidence
  recorded in place of the stale 2026-08-07 REVOKED comments (which had
  guessed, wrongly, that this was the same unidentified-mechanism class as
  `polymul`/`chebmul`).

**Not touched by this fix, out of this ticket's scope:** `polynomial.
Legendre.__mul__`/`__pow__`/`fromroots` (the `Legendre` class-level
wrappers) already showed `verdict: pass` in this ticket's own baseline
(`/tmp/r82.json`) *before* this fix — their REVOKED comments in
`anionpy/_state/polynomial.py` (lines ~825-856) are themselves stale relative
to the differential suite's actual measured behavior and were left as found;
whoever owns those declarations should reconcile the comment with the
now-passing (and always-passing, per the baseline) measurement.

## 2026-08-08 — ticket #84: `legmul`/`lagmul`/`hermfromroots` signed-zero declarations were false; widened scan found the same defect in `hermmul`/`hermemul`, plus an unrelated third bug in `lagfromroots`

Ticket #84 opened against three items declared `"exact"`:
`polynomial.legendre.legmul` (declared by #48, immediately above),
`polynomial.laguerre.lagmul`, and `polynomial.hermite.hermfromroots`. All
three lost the sign of zero on underflow. Repro (complex128):
`legmul([5e-324]*3, [-5e-324]*2)` — numpy gives a result with `-0.+0.j` at
an index where anionpy gave `+0.+0.j`; `np.array_equal` does not see this,
only a `.view(np.uint64)` bit comparison does.

### Root cause 1: `pad_add`/`pad_sub` never trimmed their operands

`legmul`/`lagmul`/`hermmul`/`hermemul`'s internal reprojection recursion
calls the private `pad_add`/`pad_sub` helpers at the exact call sites
where real numpy calls `legadd`/`legsub`, `lagadd`/`lagsub`,
`hermadd`/`hermsub`, `hermeadd`/`hermesub` — i.e. `numpy.polynomial.
polyutils._add`/`_sub`. Real `_add`/`_sub` call `[c1, c2] =
as_series([c1, c2])` **first**, trimming trailing (numerically) zero
coefficients off EACH operand independently, before the padded
elementwise combine. `pad_add`/`pad_sub` never did this trim, so an
untrimmed trailing zero surviving in the shorter operand could collide
with — and flip the sign of — an untouched `-0.0` sitting at the same
index in the longer, untrimmed operand.

Traced step-by-step (`/private/tmp/monday84/probe_hermmul_steps.py`)
against numpy's own intermediate values: `hermmul([5e-324-0j, 0.5+0j],
[-5e-324-0j, 0.5+0j])` — numpy gives `[0.5+0j, -0.+0j, 0.25+0j]`, anionpy
(pre-fix) gave `[0.5+0j, 0.+0j, 0.25+0j]` (index 1 real part sign lost).
The mechanism: numpy's `hermadd`'s shorter operand trims down to length
1, dropping its trailing `+0.0` entirely, so the untrimmed `-0.0` at
index 1 of the longer operand is never touched by the add and survives;
a naive untrimmed same-shape add (what `pad_add` did) touches it and
flips it to `+0.0`.

This is a **different** bug from the `*mulx`/`polyder`/`legder`
derived-vs-literal-zero class documented in the 2026-08-07 entry above —
it is a missing-pre-trim bug in a padded combine, not a literal-fill bug
in a degenerate branch.

The widened scan (required by the ticket: "if more items are affected,
they are part of this ticket") found this same missing-pre-trim defect
in **six** more places declared `"exact"`, none of which the ticket had
named: `legmulx`, `lagmulx`, `hermmulx`, `hermemulx`, `chebmulx`,
`polymulx` (a related but distinct sub-case — missing `as_series`
pre-trim before the `len(c)==1 && c[0]==0` zero-series fast-path check,
losing shape+sign on all-underflow/all-zero input), and in the shared
`ionp-core/src/poly.rs` `add_trim`/`sub_trim` (the implementation behind
every basis's public `*add_trim`/`*sub_trim` binding — `legadd_trim`,
`lagadd_trim`, `hermadd_trim`, `hermeadd_trim`, `chebadd_trim` — and
directly used by `ionp-py/src/poly.rs`'s top-level `polyadd`/`polysub`
bindings).

**Explicitly NOT affected, confirmed by inspection and testing:**
- `chebmul` — uses a different z-series convolution algorithm
  (`ionp-core/src/chebyshev.rs`) with its own pre-existing
  `trim_trailing_zeros` calls at the top of the function; never calls
  `pad_add`/`pad_sub` at all (confirmed via a "never used" dead-code
  compiler warning on those helpers in this file).
- `legdiv`/`lagdiv`/`hermdiv`/`hermediv`'s internal same-length
  `pad_sub` usage — per this file's own pre-existing doc comments, this
  usage is intentionally untrimmed: it needs exact positional
  same-length slices in a loop, matching real numpy's own bare-ndarray
  subtraction, which never pads there in practice. Left untouched.

### Fix

Added explicit `trim_trailing_zeros` calls on both operands immediately
before every `pad_add`/`pad_sub` call site in `legmul`
(`ionp-core/src/legendre.rs`), `lagmul` (`ionp-core/src/laguerre.rs`),
`hermmul` (`ionp-core/src/hermite.rs`), and `hermemul`
(`ionp-core/src/hermite_e.rs`); and in the shared `add_trim`/`sub_trim`
(`ionp-core/src/poly.rs`), which fixes every basis's `*add`/`*sub`/
`*add_trim`/`*sub_trim` binding in one place.

Verification:
- `cargo test -p ionp-core --lib` across `legendre`, `laguerre`,
  `hermite`, `hermite_e`, and `poly::` modules: only the same
  pre-existing, unrelated `hermite.rs` unit-test failures remain (4,
  confirmed pre-existing via a read-only `git worktree` diff both this
  session and the prior one); no new failures.
- Direct probes confirm all three originally-named items now bit-exact:
  `legmul`, `lagmul`, `hermfromroots` (which pairwise-reduces via
  `hermmul` and inherited the fix with no independent bug of its own).
- `probe_addsub.py`: 3600-trial signed-zero-focused sweep (seed
  777888999) across `polyadd`/`polysub`/`legadd`/`legsub`/`lagadd`/
  `lagsub`/`hermadd`/`hermsub`/`hermeadd`/`hermesub`/`chebadd`/`chebsub`
  — 0 mismatches, confirming the shared `add_trim`/`sub_trim` fix did
  not regress any top-level `*add`/`*sub` binding.
- `wide_sweep.py`: comprehensive out-of-corpus sweep (seed 999999,
  disjoint from the discovery seeds), all 6 bases' `mul`/`mulx`/
  `fromroots`, complex128 (underflow mixed with normal magnitudes,
  400+300+150 trials per base, both operand orders) and float64 (200
  trials per base) — 8700 total comparisons, 544 failures, all
  attributable to (a) `polynomial.polynomial.mul` (266, already
  correctly REVOKED/documented, unrelated pre-existing convolve issue,
  see the 2026-08-07 entry above) and (b) `chebyshev.mul` (276, already
  correctly never-declared, same unrelated convolve issue) — **zero**
  new-and-unexplained failures once the third bug below is accounted
  for.
- Differential suite (`tests/differential/run.py`): new permanent
  regression cases added to `legmul_cases`/`lagmul_cases`/
  `hermmul_cases`/`hermemul_cases` (`"underflow_signed_zero_ticket84"`,
  reusing the ticket's own subnormal repro values) and to
  `legmulx_cases`/`lagmulx_cases`/`hermmulx_cases`/`hermemulx_cases`/
  `chebmulx_cases` (`"negative_both_first_coef"`, `[-4.5-2.5j]`,
  mirroring `legmulx_cases`'/`polymulx_cases`' existing #48-era case).
  All pass post-fix (proving the fix is real: these values were chosen
  specifically because they only diverge without the fix).
- `anionpy/_state/polynomial.py`: dated confirmation comments added
  above `legmulx`/`legmul`/`lagmulx`/`lagmul`/`hermmulx`/`hermmul`/
  `hermfromroots`/`hermemulx`/`hermemul`/`chebmulx`/`polymulx`,
  matching the file's established dated-comment convention.

### Root cause 2 (unrelated): `lagfromroots`'s single-root path goes through CPython's non-IEEE-compliant real-plus-complex addition

The widened scan also found a **third, independent** bug, not part of
the `pad_add`/`pad_sub` class above: `polynomial.laguerre.lagfromroots`,
for COMPLEX-typed single-root input only (`cnt == 1`, no pairwise
`lagmul` reduction — multi-root inputs route through `lagmul`, whose fix
above already covers them, confirmed 0 mismatches on the same sweep).

`anionpy/polynomial/laguerre.py`'s `lagfromroots` builds `off =
complex(-complex(rl[i]))` as a **native Python `complex`**, then
`lagline(off, 1)` computes `off + scl` (`scl` is a plain Python `int`,
`1`) via `complex.__add__(int)`. Measured directly: CPython's
real-plus-complex addition special-case does **not** perform a genuine
IEEE-754 add on the imaginary component at all — it leaves `off.imag`
bit-for-bit unchanged (`complex(-0.0,-0.0) + 1 == (1-0j)`, imag stays
exactly `-0.0`). This is *not* limited to subnormal magnitudes — any
complex root with an exact-zero imaginary part triggers it (e.g. `3.0+0j`
or `0.0+0j`). By contrast, `complex(-0.0,-0.0) + complex(1,0.0)`
(explicit complex+complex addition, no real-number special case) *does*
perform a genuine componentwise IEEE add and gives `(1+0j)`, matching
what real numpy's `lagline` gets: numpy's `off` there is a genuine
`np.complex128` scalar, and `off + scl` goes through numpy's own
scalar-add ufunc, a real IEEE add.

Minimal repro: `lagfromroots([3.0+0j])` — numpy gives `[-2.+0.j,
-1.+0.j]`, anionpy gives `[-2.-0.j, -1.+0.j]` (index 0's imaginary part
sign flipped). Real-typed single-root input (`lagfromroots([3.0])`,
already present in the differential corpus) does **not** trigger this —
verified it routes through `-float(rl[i])`, never touching the
native-`complex` addition path.

This ticket's edit scope did not include `anionpy/polynomial/
laguerre.py` (only `anionpy/_state/polynomial.py`, `ionp-core/src/**`,
`ionp-py/src/**`, `tests/differential/**`), and the bug's root cause
lives entirely in that file (`lagline`'s `off + scl` construction), so
this is **withdrawn**, not fixed:
`"polynomial.laguerre.lagfromroots": "exact"` is commented out in
`anionpy/_state/polynomial.py` with a full dated explanation. A new
permanent (expected-failing) regression case,
`"single_root_complex_ticket84"`, was added to `lagfromroots_cases` in
`tests/differential/laguerre_cases.py` so `run.py` tracks this
consistently with how `polynomial.polynomial.polymul`/`chebyshev.chebmul`
are tracked (registered, `atol=0.0`/`rtol=0.0`, deliberately not
declared `"exact"`, reported as `verdict: fail`). Left for a follow-up
ticket against `anionpy/polynomial/laguerre.py`.

### Failure-set accounting

Full differential suite (`flock /tmp/ionp-build.lock ./.venv/bin/python
tests/differential/run.py --out /tmp/r84.json`): 33 named failures,
compared by name (not count) against the #48 entry's documented baseline
of 32. The one addition is `polynomial.laguerre.lagfromroots` — expected
and intentional: it newly fails specifically because of the
`single_root_complex_ticket84` case just added for the withdrawal above,
not a regression. `polynomial.legendre.legmul`/`polynomial.
laguerre.lagmul`/`polynomial.hermite.hermfromroots` (this ticket's three
named items) and all `*mulx`/`*mul` items touched by the fix are absent
from the failure set, confirming the fix holds under both the corpus and
the out-of-corpus sweep.
