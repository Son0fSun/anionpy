# Floating-point error subsystem: measured spec (2026-08-06)

**Status:** measurement + corpus only. No `.rs` file touched by this task.
Everything below was re-run live against `numpy 2.5.1`
(`~/Monday/ionp/.venv/lib/python3.14/site-packages/numpy`,
confirmed via `numpy.__file__`) on 2026-08-06, from `/tmp` (never from the
repo root, to avoid `sys.path[0]` resolving the source tree's own `ionp/`
package instead of the installed extension). ionp's own state was
confirmed absent the same way: `hasattr(ionp, 'errstate'|'seterr'|'geterr'
|'seterrcall'|'geterrcall')` are all `False`, and every one of the eight
representative ufunc-error conditions below produces zero warnings on ionp
where numpy produces one or two. This document is the reference an
implementation task should build against; it makes no `.rs`-level design
decisions.

Every script referenced below is preserved at `/tmp/fpe_measure{1..5}.py`
for re-run (scratch, not part of this deliverable's edit scope).

## 1. The gap, reproduced

```
case             numpy warns                          ionp warns
div by zero      ['RuntimeWarning']                    []
0/0 invalid      ['RuntimeWarning']                    []
sqrt(-1)         ['RuntimeWarning']                    []
log(0)           ['RuntimeWarning']                    []
overflow f32     ['RuntimeWarning']                    []
mean of empty    ['RuntimeWarning','RuntimeWarning']   []
int overflow     []                                    []   <- correct parity, must stay this way
underflow f32    []  (default under='ignore')          []   <- correct parity (see 5.)
```

`hasattr(ionp, 'errstate' | 'seterr' | 'geterr' | 'seterrcall' |
'geterrcall')` -> `False` for all five, confirmed live.

## 2. Exact warning category / message / stacklevel, per ufunc shape

All `RuntimeWarning`, `warnings.simplefilter("always")`, one call per case,
`w.lineno`/`w.filename` recorded from the `warnings.catch_warnings(record=True)`
record.

| case | message | notes |
|---|---|---|
| `array/array` divide by zero (`1.0/0.0`) | `divide by zero encountered in divide` | |
| `array/scalar` divide by zero (`np.array([1.0])/0.0`) | `divide by zero encountered in divide` | same message form as array/array — the RHS being a bare Python scalar doesn't change the ufunc name in the text |
| `np.true_divide` | `divide by zero encountered in divide` | `true_divide` is an alias of `divide`; the message names `divide`, not `true_divide` |
| `//` floor_divide by zero (float) | `divide by zero encountered in floor_divide` | |
| `0.0/0.0` (invalid) | `invalid value encountered in divide` | different *category word* (`invalid value`, not `divide by zero`) for the same ufunc, driven by which IEEE flag actually raised |
| `sqrt(-1.0)` | `invalid value encountered in sqrt` | |
| `log(0.0)` | `divide by zero encountered in log` | **`log` at zero is filed under `divide`, not `invalid`** — the flag categories are about which of the 4 IEEE754 exception classes fired (`FE_DIVBYZERO`), not about which named ufunc-error-kind you'd guess by intuition |
| `log(-1.0)` | `invalid value encountered in log` | |
| `arcsin(2.0)` (domain error) | `invalid value encountered in arcsin` | |
| `power(-1.0, 0.5)` (non-integer exponent of negative base) | `invalid value encountered in power` | |
| `exp(1000.0)` (float64 overflow) | `overflow encountered in exp` | |
| `float32` multiply overflow | `overflow encountered in multiply` | |
| `float16` multiply overflow | `overflow encountered in multiply` | same message shape at every float width |
| `float32` multiply underflow (`1e-38 * 1e-38`) | *(none — `under` defaults to `'ignore'`)* | see §5 |
| `int8` `astype`/cast overflow (`np.array([1e300]).astype(int32)`) | `invalid value encountered in cast` | **cast overflow is a distinct message shape** (`in cast`, not `in <ufunc-name>`) — not itself an arithmetic op |
| `//` int64 floordiv by zero | `divide by zero encountered in floor_divide` | int dtypes DO participate in the divide/invalid machinery for `/`, `//`, `%` — only elementwise arithmetic *ufunc-flag* checks are int-exempt, not the whole subsystem (see §5) |
| `%` int64 mod by zero | `divide by zero encountered in remainder` | |
| int8 `127 + 1` (silent wraparound) | *(none, ever)* | never flagged by numpy at any `seterr` mode — this is not a togglable behavior, it is simply outside the 4-flag model entirely (see §5) |
| complex128 `(1+1j)/(0+0j)` | `divide by zero encountered in divide` | same message text as real division |
| complex128 overflow (`(1e200+1e200j)**2`-ish product) | `overflow encountered in multiply` | |
| complex `(0+0j)/(0+0j)` | `invalid value encountered in divide` | |
| complex `sqrt(-1+0j)` | *(none)* | complex sqrt has a defined branch value everywhere; no domain error exists for it the way real `sqrt(-1)` has one |

**Stacklevel** (`fpe_measure5.py`, a 3-deep call chain `level1 -> level2 ->
level3`, where only `level3` calls the dividing ufunc): the recorded
warning's `lineno`/`filename` point at **the literal source line inside
`level3` that calls the ufunc**, not at `level1`'s or `level2`'s call site
and not at any numpy-internal frame — i.e. numpy's C-level warning emission
is stacklevel-correct relative to the *direct Python caller of the ufunc*,
regardless of call depth, for a bare ufunc call.

This does **not** hold for the two-warning composite cases below —
`np.mean`/`np.std`/`np.var` on an empty/degenerate reduction warn from
*inside numpy's own Python-level `_methods.py`/`fromnumeric.py` — the
recorded `w.filename` is `.../numpy/_core/_methods.py` or
`.../numpy/_core/fromnumeric.py`, not the caller's file at all, because
those functions are themselves plain Python wrapping the ufunc call, one
frame further removed, and do not re-target the stacklevel back to their
own caller.

## 3. Composite (two-warning) cases

`np.mean(np.array([]))` fires **two** warnings, in this order:
1. `RuntimeWarning: Mean of empty slice` (from `numpy/_core/fromnumeric.py`)
2. `RuntimeWarning: invalid value encountered in scalar divide` (from
   `numpy/_core/_methods.py`, the `0/0` inside mean's own `sum/count`
   division)

`np.std(np.array([]))` fires **three**:
1. `Degrees of freedom <= 0 for slice`
2. `invalid value encountered in divide`
3. `invalid value encountered in scalar divide`

`np.var(np.array([1.0]), ddof=2)` (degenerate but non-empty, `N - ddof <
0`) fires **two**: `Degrees of freedom <= 0 for slice` +
`invalid value encountered in scalar divide`.

These are not single ufunc-flag events — they are numpy's own
higher-level Python code (`_methods.py`) independently calling
`warnings.warn` for a *semantic* condition (empty-slice, non-positive
dof) **in addition to** whatever the ufunc-flag machinery separately
fires for the resulting `0/0`. An implementation that only wires the
4-flag ufunc mechanism into `mean`/`std`/`var` and skips this second,
semantic warning will only ever emit one of the two (or three) messages
numpy emits for these cases.

## 4. The five API entry points

### `numpy.seterr(divide=None, over=None, under=None, invalid=None) -> dict`
Sets the **calling thread's** per-category mode (unspecified kwargs left
unchanged) and **returns the previous state** (a plain `dict`, a fresh
copy — mutating the returned dict does not affect internal state, confirmed
live). Accepts `all=<mode>` as a shorthand that sets all four categories at
once (still returns the *pre-call* per-category dict, not a collapsed
`{'all': ...}` shape). Invalid mode string -> `ValueError: invalid error
mode 'bogus'`. Unknown category kwarg -> plain `TypeError: seterr() got an
unexpected keyword argument 'bogus_kw'` (ordinary Python signature
rejection, not a numpy-specific error type).

### `numpy.geterr() -> dict`
Returns the calling thread's current state, always all four keys present:
`{'divide': ..., 'over': ..., 'under': ..., 'invalid': ...}`. Measured
default (fresh interpreter, fresh thread, no prior `seterr` call):
```
{'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
```

### `numpy.errstate(**kwargs)`
A class (`type(np.errstate(...))` is `numpy.errstate`) usable **both** as
a context manager (`__enter__`/`__exit__` present) **and** as a decorator
(`__call__` present — confirmed: `@np.errstate(divide='raise')` on a
function works and the state is restored to whatever it was before the
call once the decorated function returns, exception or not). Merely
*constructing* an `errstate(...)` object with no `with`/decorator use has
**zero effect** on `geterr()` — confirmed live (`es = np.errstate(...)`
followed by `np.geterr()` shows no change). Accepts `all=<mode>` the same
way `seterr` does. On exit (normal or exceptional), restores exactly the
state that was active before entry — proven by the nested case below.

### `numpy.seterrcall(func_or_log_object) -> previous`
Registers the target for the `'call'` and `'log'` modes (see §4.1/4.2
below). Returns the previous registrant (default: `None`, confirmed via
`np.geterrcall()` on an untouched thread).

### `numpy.geterrcall() -> current registrant or None`

### 4.1 The `'call'` mode
Registrant must be a plain callable `f(err_msg: str, flag: int)`. Measured
live: `np.seterr(divide='call')` + a zero-divide produces exactly one call
`handler('divide by zero', 1)` — note the message text handed to the
callback is the **short** category phrase (`'divide by zero'`), not the
full `'divide by zero encountered in divide'` sentence the warning path
uses; `flag` is a small integer (`1` observed for `divide`) — this is a
bitmask over the four categories, not a string.

### 4.2 The `'log'` mode
Registrant (via `seterrcall`) must expose a `.write(str)` method (duck
typed — anything file-like works, confirmed with a plain class defining
only `write`). With **no** registrant set, `'log'` mode raises immediately
at the point of the arithmetic op:
```
NameError: log specified for divide by zero (in divide) but no object with write method found.
```
With a registrant set, exactly one `.write()` call is made per triggering
op, with the message text **identical to the ordinary warning text**
prefixed `"Warning: "` — measured: `'Warning: divide by zero encountered
in divide\n'` (trailing newline included). This is a materially different
text shape from the `'call'` mode's callback argument (§4.1) — `'log'`'s
text is the long form with a `"Warning: "` prefix and trailing `\n`;
`'call'`'s is the short phrase with no prefix/suffix.

### 4.3 `'ignore'` / `'warn'` / `'raise'` / `'print'`, measured
- `'ignore'`: no warning, no side effect, normal (non-finite) result value
  unaffected. Confirmed: `np.errstate(divide='ignore')` around `1.0/0.0`
  produces zero recorded warnings and still returns `inf`.
- `'warn'`: the ordinary `RuntimeWarning` path described in §2. This is
  the default for `divide`/`over`/`invalid`.
- `'raise'`: raises **`builtins.FloatingPointError`** (there is no
  `numpy.FloatingPointError` — `hasattr(np, 'FloatingPointError')` is
  `False`; numpy raises the stdlib exception directly). MRO:
  `(FloatingPointError, ArithmeticError, Exception, BaseException,
  object)`. Message text is the **same long-form sentence** the `'warn'`
  path would have used (`"divide by zero encountered in divide"`), not a
  distinct raise-mode phrasing. Confirmed to fire for both float ufuncs
  *and* integer `//`/`%` by zero (`int64 // 0` under `errstate(divide=
  'raise')` raises `FloatingPointError`, not `ZeroDivisionError` — int
  divide-by-zero is filed under the exact same `'divide'` category as
  float divide-by-zero, it is only the *silent-wraparound* overflow case
  (§5) that int dtypes are ever exempt from).
- `'print'`: writes directly to the process's real file-descriptor-2
  stderr (a C-level `fprintf`/`PySys_WriteStderr`-style write), **not**
  through Python's `sys.stderr` object — `contextlib.redirect_stderr`
  (which only reassigns `sys.stderr`) does not capture it; it must be
  captured, if at all, at the OS file-descriptor level. Text observed on
  the real terminal: `"Warning: divide by zero encountered in divide"`.

### 4.4 Thread-local, not global
Confirmed live (`fpe_measure4.py`): a background thread calling
`np.seterr(divide='raise')` and sleeping does **not** change the main
thread's `np.geterr()` at any point during or after the background
thread's lifetime — the main thread's snapshot before, during, and after
the other thread's `seterr` call is unchanged
(`{'divide': 'warn', ...}` throughout), while the background thread's own
`geterr()` correctly shows `'raise'`. State is per-thread, not a process
global and not a plain module-level variable.

### 4.5 Nesting (`errstate` as context manager)
Confirmed live: `with errstate(divide='raise'): with errstate(divide=
'ignore'): <op>` — the inner `errstate` genuinely overrides the outer one
for its scope (`1.0/0.0` inside the inner block returns `inf` with no
raise), and on exiting the inner block the **outer** state
(`'raise'`) is restored and does fire (the same `1.0/0.0` re-run
immediately after the inner `with` exits raises `FloatingPointError`).
After both blocks exit, `geterr()` is back to the pre-existing default.
This is ordinary LIFO context-manager stacking, not a merge/union of
states — the inner block's mode is used exclusively while active.

## 5. What numpy does NOT flag, ever

- **Integer overflow** (`np.int8(127) + np.int8(1)` -> `-128`, silent
  wraparound): zero warnings/exceptions at **any** `seterr`/`errstate`
  setting, for any integer dtype, for `+`/`-`/`*`. This is not a togglable
  "ignore" state — there is no `seterr` category that covers it at all;
  the 4-category model (`divide`/`over`/`under`/`invalid`) only governs
  IEEE754 floating-point exception flags, and CPU integer overflow does
  not raise one. **This existing (correct) silence must not regress** —
  do not wire integer arithmetic into the new subsystem's `over`/`invalid`
  categories.
- **Integer divide/mod by zero is the one exception to the above** — `//`
  and `%` on integer operands with a zero divisor DO go through the
  `'divide'` category exactly like float division (§4.3), because numpy
  implements integer floor-divide/remainder as ufuncs that explicitly set
  the `FE_DIVBYZERO`-equivalent software flag themselves (there is no
  hardware IEEE754 integer-divide-by-zero flag to piggyback on — this is
  numpy choosing to synthesize the same flag for a case CPU integer div
  would otherwise trap or produce a platform-defined result for).
- **Complex `sqrt` of a negative real** (`sqrt(-1+0j)`): defined branch
  value everywhere on the complex plane, never a domain error — no
  warning, confirmed.
- **A zero-size reduction whose accumulator doesn't divide** (`np.sum` of
  an empty array, or `np.sum` over a zero-length axis): the *identity*
  value (`0.0` for `sum`) is returned with **zero** warnings — this is
  distinct from `np.mean`/`np.std`/`np.var` of an empty array, which DO
  warn (§3), because those additionally divide by the (zero) count;
  `sum`'s empty case never divides by anything.
- **float32/float64 underflow, by default**: `under` defaults to
  `'ignore'` (see §6) — this is a *default*, not a structural exemption
  like integer overflow; setting `errstate(under='warn')` does make an
  underflowing multiply warn (not independently re-verified with an
  explicit non-default `under` setting in this pass, but implied directly
  by `under` being an ordinary member of the same 4-key `geterr()` dict as
  `divide`/`over`/`invalid`, which are all confirmed togglable).

## 6. Default state

Measured, fresh interpreter, no prior `seterr` call:
```
{'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
```
This is `numpy`'s own documented, source-level default (set once in
`numpy`'s C initialization, not derived from any environment variable or
platform CPU flag query observed during this measurement pass) — three
categories `'warn'`, `under` alone `'ignore'`. This spec did not have
access to a second OS/CPU architecture to test in this pass, so "platform-
independent" is stated on the strength of it being a fixed compile-time
default in numpy's own source rather than a runtime CPU-flag probe, not on
having literally re-run this on a second machine — flag this as unverified
across platforms if that distinction matters to the implementation task.

## 7. Interaction with `out=`, empty-axis reductions, complex dtypes

- **`out=`**: `np.divide(a, b, out=preallocated)` still fires the
  ordinary `RuntimeWarning` exactly as the non-`out=` form does — confirmed
  live, `out=` does not suppress or alter the warning path.
- **Reduction over an empty axis**: `np.mean(np.zeros((0,3)), axis=0)`
  (reducing the zero-length axis, leaving shape `(3,)`) fires the *same*
  two-warning composite as whole-array empty mean (§3) — `Mean of empty
  slice` + `invalid value encountered in divide` — and returns
  `[nan, nan, nan]`. `np.sum` over an empty axis (`np.sum(np.zeros((3,0)),
  axis=1)`, shape `(3,)` of zeros) fires **zero** warnings (identity value,
  no division — consistent with §5's zero-size-sum rule).
- **Complex dtypes**: complex128/complex64 divide-by-zero and invalid
  (`0/0`) both route through the ordinary `'divide'`/`'invalid'`
  categories with message text identical in *shape* to the real-dtype
  case (`"divide by zero encountered in divide"`,
  `"invalid value encountered in divide"` — the ufunc name in the message
  does not change for complex operands). Complex overflow
  (`(1e200+1e200j) * (1e200+1e200j)`, exceeding float64's range in the
  underlying real/imag components) fires `overflow encountered in
  multiply`, same category and message shape as real overflow. Complex
  `sqrt` of a negative real number is a defined value, never a domain
  error (§5).

## 8. Summary table for implementation

| `seterr` mode | measured behavior |
|---|---|
| `'ignore'` | silent, no side effect |
| `'warn'` | `RuntimeWarning`, long-form message, `"<condition> encountered in <ufunc-name>"` (or `"in cast"` for cast overflow), stacklevel = direct Python caller of the ufunc |
| `'raise'` | `builtins.FloatingPointError` (not a numpy-namespaced class — there is none), same long-form message text as `'warn'` |
| `'print'` | writes `"Warning: <long-form message>"` directly to fd-2 stderr, bypassing `sys.stderr` |
| `'call'` | calls the `seterrcall`-registered `f(short_phrase: str, flag: int)`; short phrase differs textually from the long-form warning/raise message |
| `'log'` | calls `.write("Warning: <long-form message>\n")` on the `seterrcall`-registered object; `NameError` if none registered |

State is **per-thread**. `errstate` nests as an ordinary LIFO context
manager/decorator, restoring the prior state exactly on exit. Integer
overflow is never flagged at any setting (must stay that way); integer
divide/mod by zero IS flagged, under `'divide'`, identically to float.
