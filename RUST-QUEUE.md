# RUST QUEUE — defects that cannot be closed from Python

Every entry below was **re-measured against the current binary** on 2026-08-03,
not copied from a prior note. A recorded reason is not evidence about the
binary you are holding.

Ledger at time of writing: **573/1180 (48.559%)** — phantom 0, failing 0,
untested 0, absent 607. Clean tree, selftest green.

The number went **down** 620 → 573 and that is the file working. Every step
was a revocation of something declared true and measured false:
−2 `strings.partition`/`rpartition`, −3 ndarray lane (`cfae20b`), −15 the
0-d × explicit-axis reduce family (item 2e), +1 `char.strings_multiply`,
+18 ma Phase 8 (`1af2001`), −32 the `ma` 0-d boundary (`2061256`),
−14 the declared-surface boundary audit (`21abf6f`).

**A coverage number that only ever rises is not being audited.**

**66 revocations in one session, and the dominant one is a single boundary:
the 0-d operand.** 52 of the 66 are it. Three separate audits went looking and
all three found more. Treat any declaration that has not been probed at 0-d as
unverified, not as passing.

The 14 in `21abf6f` are the important warning: **every one of them agrees on
ordinary input** (controls 0/10). They were found only because someone went
looking at a boundary on purpose. `linalg.vecdot` is the sharpest case — it
raises `NotImplementedError` for *any* explicit `axis`, including `axis=-1`,
its own documented default. That declaration never exercised the kwarg once.

**Standing rule earned the hard way (item 1): isolation is only visible if you
read the thing you did NOT touch.** A view test that reads the view, and a
copy test that reads the copy, both pass while the opposite defect ships.

---

## 1. `out=` into a VIEW silently detaches — **CLOSED `5fbac50` + `1d548d0`**

**Re-verified independently 2026-08-03 reading the PARENT, not the view:** all
three view cases write through, `shares_memory` stays True across the op.
Ledger 620/1180, 0 failing.

The first fix (`5fbac50`) closed this but opened a silent copy-on-write
corruption: an unconditional `unsafe` bypass of `Arc::make_mut` made
`ionp.array(a)` alias its source, so writing to the "copy" rewrote the
original. `Arc::make_mut` had been doing **two** jobs — splitting views
(wrong) and splitting genuine CoW (right) — and the bypass killed both.

`1d548d0` fixes it at **construction** time instead of write time
(`NdArray::detach_buffer()`, called eagerly by `array()` when the source is
an existing `ionp.ndarray`), which removes the ambiguity before any write
happens. The lane tried the write-time gate I asked for first and reported
that it is provably impossible: a `_base`-on-target check cannot distinguish
"root array with a live child view" (needs write-through) from "copy
pretending to be independent" (must not have it) — neither has a `_base`.
That is a correct refusal of my instruction and the right call.

Re-verified: all 3 previously-corrupting paths isolated, all 3
must-stay-aliased paths still alias, 0 corrupting paths remain.

**Historical record — why the suite never caught it:** no corpus case writes
into a copy and then re-reads the SOURCE. The test observed the object it
wrote to, never the object that was supposed to be unaffected.

---

## 1b. `array(a, copy=None)` now over-copies — NEW, opened by `1d548d0`

Found by me post-audit; the lane did not report it.

```
np.array(a, copy=None)    -> aliases  (copy-if-needed, and none is needed)
ionp.array(a, copy=None)  -> copies   (shares_memory False)
```

`copy=True` / `copy=False` / `copy=False, dtype=` (raises) / `asarray` /
`order='F'` / `ndmin=2` / `subok=True` all match numpy exactly. **`copy=None`
is the only divergence**, because the eager `detach_buffer()` runs
unconditionally rather than honouring copy-if-needed.

**Direction matters:** this over-copies. It is a performance and
aliasing-semantics divergence, NOT data corruption — strictly the safer
failure than what it replaced. `array` is `absent` in `TOPLEVEL_STATE`, so
the ledger is not lying. But `array` cannot be declared until this closes.

---

## 1c. ORIGINAL ENTRY (kept for the root-cause record)

**This is the largest open honesty risk in the project.** It affects roughly
111 items currently declared `exact`.

The prior record described this as "`out=` silently discarded." **That is
wrong and the wrong boundary matters.** `out=` works correctly for a
contiguous, independently-allocated `out`. It breaks *only* when `out` is a
view of another array.

Measured, this binary:

```
o = ionp.zeros(6); v = o[::2]
ionp.add(ionp.ones(3), ionp.ones(3), out=v)

  ionp.shares_memory(o, v)  BEFORE = True
  ionp.shares_memory(o, v)  AFTER  = False      <-- the tell
  parent o (numpy) = [2., 0., 2., 0., 2., 0.]
  parent o (ionp)  = [0., 0., 0., 0., 0., 0.]   <-- never written
  view   v (ionp)  = [2., 2., 2.]               <-- write landed HERE
```

Same for `o[1:4]` and `o.reshape(2,3)[0]`. Contiguous `out`, dtype-casting
`out`, F-order `out`, wrong-shape `out`, and `out` aliasing the input all
behave correctly — those are NOT the boundary.

**Why it is the worst class of defect:** the caller inspects `out` after the
call, sees the right numbers, and never discovers the parent was not updated.
It fails silently and it fails *plausibly*.

**Root cause:** `let buffer = Arc::make_mut(&mut out.buffer);` at
`ufunc.rs:8314` (in `write_out(...)`, ~8286). With refcount > 1 — which is
exactly what a live view guarantees — `make_mut` CLONES. The clone is then
written and the view is left pointing at a detached buffer.

**Fix:** a genuine strided write through the existing buffer. Do not clone
when `out` is a view; that is the entire bug. Related known defect, likely
the same root cause: the in-place dunders (`__iadd__` etc.) do
`slf.inner = out.cast_to(...)` — a full reassignment instead of a strided
write. That blocks 7 dunders and 12 view/aliasing always-copy defects.

**After fixing:** re-verify with a probe that reads the PARENT, not the view.
My first pass at this reported 0 divergences precisely because I read back the
view. The bite test is mandatory here.

---

## 2. Invalid `order=` silently accepted — **CLOSED `5fbac50`, re-declared**

Re-swept independently 2026-08-03 against the current binary: **0 victims**.
All five (`asarray`, `asanyarray`, `asarray_chkfinite`, `frexp`, `modf`) now
raise numpy's `ValueError` and are re-declared exact. The lane also found and
fixed a Python bug on its own initiative: `asanyarray` never forwarded
`order=` to its internal `asarray` calls despite its docstring claiming it did.

Original entry follows.

## 2b. ORIGINAL: Invalid `order=` silently accepted — 5 items REVOKED

Swept every top-level `exact` declaration. **Exactly 5 diverge**, measured:

| item                | numpy `order='Z'`    | ionp        |
|---------------------|----------------------|-------------|
| `asarray`           | `ValueError`         | no error    |
| `asanyarray`        | `ValueError`         | no error    |
| `asarray_chkfinite` | `ValueError`         | no error    |
| `frexp`             | `ValueError`         | no error    |
| `modf`              | `ValueError`         | no error    |

Also `order=''`. numpy's message: `order must be one of 'C', 'F', 'A', or 'K'
(got 'Z')`.

**The boundary:** the creation family (`array`, `zeros`, `ones`, `empty`,
`full`) validates `order=` correctly. The gap is in the **conversion /
ufunc-`out` argument parser only**. Fix there, then re-declare all 5.

Revoked in `5b4fa15`.

---

## 2c. `linalg.tensorsolve` — SILENT WRONG ANSWER + a hard Rust PANIC

Both measured 2026-08-03 against `1d548d0`. The first was found by the linalg
lane; the second by me while (wrongly) trying to refute the first.

**(a) Silent wrong answer — worst class.** With a well-conditioned `a` of
shape `(3,4,3,4)` and a `b` whose core dimension does not match:

```
b.shape = (3,5)
  numpy -> ValueError: solve1: Input operand 1 has a mismatch in its
                       core dimension 0
  ionp  -> returns a float64 array of shape (3,4).  No error.
```

The caller asked for the impossible and got a plausible-looking answer.

**NOTE FOR ANYONE RE-MEASURING THIS:** you must use a NON-SINGULAR `a`. My
first probe built `a` from `arange(...)`, which is singular, so ionp raised
`LinAlgError: Singular matrix` and the defect was completely masked — I
briefly and wrongly concluded the lane had over-claimed. The missing
core-dimension check is only reachable once the solve itself can proceed.

**(b) Hard panic.** Same well-conditioned `a`, `b.shape = (2,3)`:

```
thread '<unnamed>' panicked at ionp-ion/src/dense_linalg.rs:94:33:
index out of bounds: the len is 6 but the index is 6
  numpy -> ValueError (same core-dimension message)
  ionp  -> PanicException
```

A panic is not an exception — `PanicException` derives from `BaseException`,
so ordinary `except Exception` handling in caller code will not catch it. An
out-of-bounds index in shipped release code is also a correctness red flag in
its own right, independent of the API mismatch.

**Fix:** validate `b`'s core dimensions against `prod(a.shape[n:])` BEFORE
dispatching to the solver, and raise ionp's own `ValueError` with numpy's
message. That single check closes both (a) and (b).

---

## 2d. `linalg` — the other 8 declines, all re-measured and upheld

Independently re-verified (`linalg.trace` float16 and `matrix_power` signed
zero I confirmed byte-for-byte myself):

- `matrix_power(eye(2), -1)` — ionp yields `-0.0` where numpy yields `+0.0`.
  Raw bytes: numpy `...0000000000000000`, ionp `...0000000000000080`.
- `linalg.trace` — `offset != 0` on float16 raises `NotImplementedError`
  30/30 cases; numpy returns a normal float16. Over-broad guard.
- `linalg.svd` — NaN input on a 2x2: numpy raises `LinAlgError: SVD did not
  converge`, ionp silently returns NaN-filled `(U, S, Vt)`.
- `linalg.tensordot` — `axes=-1`: numpy computes an outer product (because
  `range(-1, 0)` is empty), ionp raises `ValueError`.
- `linalg.norm` / `linalg.vector_norm` — a non-`str` `ord` that broadcasts
  (e.g. `bytearray(b'fro')`, which is `[102,114,111]`) succeeds in numpy via
  generic power-ufunc dispatch; ionp raises `TypeError: must be real number`.
  Only reproducible when the operand axis has length 3 — it is a broadcast,
  so a length-2 axis raises in numpy too and hides the gap.
- `linalg.cross` — values equal, strides differ (`(96,8,24)` vs `(96,32,8)`).
- `linalg.diagonal`, `linalg.matrix_transpose` — numpy returns a true view,
  ionp returns a copy. Same class as item 5's view/aliasing defects.
- `linalg.test` — a `PytestTester` instance, not API. Manifest noise.

---

## 2e. Reduce family PANICS on 0-d operand + explicit axis — **15 REVOKED** (`e86f97c`)

Biggest single ledger correction of the session: **15 items declared `"exact"`
were live lies.** Measured 2026-08-03 (`/tmp/mg_0d_axis.py`), real numpy 2.5.1
against the current binary, not inferred.

With a **0-d operand** and an **explicit `axis=0`**:

| ionp behaviour | items |
|---|---|
| `PanicException` (13) | `sum` `nansum` `nanmean` `nanvar` `nanstd` `min` `max` `nanmin` `nanmax` `any` `all` `ptp` `count_nonzero` |
| wrong SHAPE, silent (2) | `cumsum` `cumprod` — return `()` where numpy returns `(1,)` |

```
index out of bounds: the len is 0 but the index is 0
    @ ionp-core/src/ufunc.rs:6564
```

**A PanicException is not an exception.** PyO3 derives it from
`BaseException`, so a caller's `except Exception` does not catch it. This is
strictly worse than returning a wrong value — it takes the caller's process
down through a handler they reasonably believed was total.

**CONTROL GROUP:** the identical 12-function sweep on a **1-D** operand with
`axis=0` gave **0 divergences**. The boundary is the 0-d operand, not the
`axis=` kwarg. Non-uniformity is the evidence this is real.

**THE FIX IS NOT "REJECT axis ON 0-d".** numpy's own behaviour here is
inconsistent, and it is still the reference:

- `sum` `prod` `min` `max` `any` `all` `nan*` — **accept** `axis=0` on a 0-d
  array and return the scalar.
- `mean` `var` `std` `median` — **raise** `AxisError`.
- `cumsum` `cumprod` — return shape `(1,)`.

Any fix must **reproduce that inconsistency**, not normalize it. Bounds-check
`axis` against `ndim` before indexing the shape vector, then branch per-family
to match the table above.

Also in scope, already undeclared so not a revocation: `median` **silently
returns a value** where numpy raises `AxisError`; `mean` `var` `std` `prod`
`nanprod` panic.

**Why the corpus missed all 15.** Prior probing varied kwargs
(`axis`/`keepdims`/`dtype`/`out`) and varied layouts
(C/F/transposed/strided/negstride) — but never crossed *"0-d operand"* with
*"explicit axis kwarg"* in the same case. Same shape as `strings.partition`:
a declaration true of everything tested and silent about a boundary the corpus
did not contain. That is now **ten-plus** instances of this one failure mode.

Re-declare only after a differential case that includes 0-d × explicit-axis is
passing. Credit: the ndarray lane (`cfae20b`) found it in `mean`/`var`/`std`
and flagged `nanvar`/`nanstd` as untested-but-suspect. Both suspicions were
correct; the blast radius was **5×** what it reported.

**The same root cause reaches the `ma` layer — 13 more items, `2061256`.**
`ma.all` `ma.alltrue` `ma.amax` `ma.amin` `ma.any` `ma.count`
`ma.count_masked` `ma.max` `ma.min` `ma.ptp` `ma.size` `ma.sometrue` `ma.sum`
all panic at `ufunc.rs:6564` because they compose over these same top-level
reductions. **One Rust fix closes both sets — 28 items in total.** Do not fix
them separately at the `ma` layer; that would paper over the root cause and
leave the top-level primitives still panicking.

---

## 2f. `ndarray.var`/`.std` METHODS accept `correction=` — CONFIRMED, undeclared

```
np : TypeError: _var() got an unexpected keyword argument 'correction'
ion: 9.583333333333334
```

Real numpy accepts `correction=` on the **module function** only, never on the
**method**. ionp over-forwards it on both `.var` and `.std`. Controls:
`ddof=1` agrees on both, and `np.var(a, correction=1)` / `np.std(...)` agree —
so the divergence is method-specific, not kwarg-specific.

**Not reproduced, do not chase:** the same lane's `mean=` claim. `.var(mean=2.0)`
and `.std(mean=2.0)` both agree with numpy. That half of the report was wrong.

---

## 2g. `take`/`compress` zero-length-axis OOB — **NOT REPRODUCED**

The ndarray lane reported that `take`/`compress` skip out-of-bounds validation
when the indexed axis has length 0. **31 cases, 0 divergences**, same binary
(no rebuild since `cfae20b`), controls raising correctly in both directions:
`take` on `(0,)`/`(3,0)`/`(0,3)` × `axis=0`/`1`/`None` × scalar and list
indices × `mode='raise'`/`'wrap'`/`'clip'`, and `compress` with a condition
longer than both an empty and a non-empty axis.

Left open rather than closed: a claim I could not reproduce is not the same as
a claim I disproved. Anyone re-opening this must **paste the exact repro**, not
the description.

---

## 2i. Boundary-audit findings, `21abf6f` — 14 items revoked

Re-measured independently before revocation; controls 0/10. Grouped by what a
fixer needs to touch:

**Rust-side, same `ufunc.rs:6564` root cause as item 2e** — `amax` `amin`
(numpy returns the scalar, ionp panics), `average` `size` (numpy raises
`AxisError`, ionp panics). `amax`/`amin` are **separate ledger keys** from
`min`/`max` and were re-declared 2026-08-02 without re-probing this boundary.
Add them to the 28 already counted in 2e — **32 items on one fix.**

**Over-strict, the inverse direction** — `repeat` and `ndarray.repeat` on a
0-d operand with `axis=0`: numpy **succeeds**, returning shape `(2,)`; ionp
raises `AxisError`. Worth stating plainly because every other finding in this
file runs the other way: ionp is not always too permissive.

**`linalg.vecdot` — `NotImplementedError` for any explicit `axis`.** Not a 0-d
issue at all. On an ordinary 1-d/1-d call, `axis=0` and `axis=-1` both raise,
where numpy returns `14.0`. `axis=-1` is vecdot's own documented default, so
passing it explicitly must behave identically to omitting it. **This is the
cheapest fix in the file and the one whose declaration was emptiest.**

**dtype-path defects** — `unwrap` is the serious one: on `uint64`/`int8`/
`bool_`/`float16` the output is **shifted by one element** (`[0,3,6,…]` vs
`[0,0,3,…]`), and `float16` also returns `float64`. The `float64` control is
exact, so this is entirely dtype-path-specific. Also `angle(bool_)` →
`float16` instead of `float64`; `sinc(complex64)` → upcast to `complex128`;
`fft.ihfft(complex64)` → numpy **rejects** with `TypeError`, ionp silently
accepts and returns a value.

**Result container type** — `unique_counts` / `unique_inverse` / `unique_all`
return plain tuples where numpy returns `NamedTuple` result classes. Values
agree in every case; attribute access (`r.counts`) fails on ionp. Likely
Python-fixable.

---

## 2h. NOT RUST-BLOCKED — two `ma` defects fixable from Python

Recorded here only so they are not lost. **This file is for defects that cannot
be closed from Python; these two can be, and belong to a future `ma` lane.**

**(a) 0-d results are wrapped instead of collapsed — 17 items revoked
(`2061256`).** `make_masked_unary` / `make_masked_binary` in `ionp/ma/core.py`
never test whether the computed result is 0-d, so they wrap it into a
`MaskedArray`. Real numpy collapses a 0-d result to a **bare scalar**, and to
the **`masked` singleton** when the result is masked.

```
ma.exp(0-d)            np -> float64        ion -> MaskedArray     (13/13)
ma.exp(0-d, mask=True) np -> masked         ion -> MaskedArray
```

The **values are all correct** — only the container is wrong, which is exactly
why a value-comparing corpus never saw it. Affected: `abs` `absolute` `add`
`arcsinh` `arctan` `arctan2` `ceil` `conjugate` `cosh` `exp` `fabs` `floor`
`multiply` `negative` `sin` `sinh` `subtract`.

The ma Phase 8 lane reported this itself and correctly declined to touch it as
out-of-scope; it is wider than the `add`/`subtract`/`multiply` it named. Its own
comparison family and `hypot` sidestep it via a scalar-safe factory and were
**not** revoked.

**(b) `ma` unary ufuncs reject positional `out=`.** `np.ma.sin(x, y)` treats
`y` as `out=`; ionp's wrapper takes one positional argument and raises
`TypeError`. Found as a by-product of a probe bug, so it is under-explored —
the full set of affected unary ufuncs has **not** been enumerated.

---

## 3. Real `numpy.exceptions.AxisError` raised from the compiled extension

```
type(e) is numpy.exceptions.AxisError   ->   True    (moveaxis, expand_dims)
```

This violates the project's hardest rule: **ionp must raise its own errors;
numpy may hand us input bytes, never answers.** A differential harness
structurally cannot catch this — it compares numpy against numpy and passes.

Both items must stay undeclared until ionp raises its own `AxisError`.

---

## 4. Present-but-failing (16 items)

All confirmed present on `ionp` and failing vs numpy:
`can_cast`, `emath.power`, `expm1`, `float_power`, `hypot`, `iinfo`,
`isdtype`, `isnat`, `ldexp`, `min_scalar_type`, `nanprod`, `pow`, `power`,
`prod`, `promote_types`, `result_type`.

Note `hypot` additionally fails 3/775 under `.at()`, and `pow`/`power` fail on
complex `.reduce`. There is a false comment in `ndarray_attrs_cases.py`
claiming `prod` is bit-exact — delete it when `prod` is fixed.

---

## 5. Architectural gates (do NOT start without Mother)

- **DType enum expansion** — 180 `DType::C128` + 246 `DType::F64` sites across
  19 files.
- **Buffer representation / interior mutability** — entangled with item 1.

**Correction to an earlier estimate of mine:** these two gates block only
about **53 of the absent items (~11%)**, not all of them. They are not the
reason coverage is at 51%.

---

## 6. Other Rust-blocked work

- `random` — 63 items, 44 unexamined. Needs a bit-exact PCG64 / MT19937 /
  Philox / SFC64 stack.
- FFT — ionp's FFT is a **different algorithm** from pocketfft; bit-exact only
  for n ∈ {2, 4, 8}. Needs ~1e-13 tolerance if ULP-declared. **Blocked on an
  open question for Mother: does "exact" mean bytes or values?**
- float16 repr; float16/float32 missing overflow `RuntimeWarning`.
- `unique_values` sorted-ordering; `finfo` typed scalars; `outer` positional-
  only args; `order='K'` for multi-output ufuncs; tuple-axis for `median`;
  complex128 `matmul`/`vecdot` 1-ULP.
- 3 pre-existing `cargo test -p ionp-core --release` `documented_gap` failures.

---

## BUILD FACT — do not get this wrong

A bare `cargo build` does **NOT** refresh the `.so`. Only:

```
flock /tmp/ionp-build.lock .venv/bin/maturin develop --release
```

and you must see `Installed ionp-0.1.0` in the output. Without that line you
are measuring the OLD binary and every conclusion you draw is void.

**Only ONE agent may do Rust work at a time**, and a rebuild invalidates any
concurrently-running lane's measurements.
