# Polynomial-Basis Source Audit: Laguerre / Hermite / HermiteE vs. Legendre

**This is an audit of numpy's source, not a specification of ours.** It records
what numpy's Python reference implementation literally does — delegation
patterns, arithmetic grouping, trimming, scaling constants — as a map of
where a from-Legendre-pattern-matching implementer would go wrong.

**ILLUSTRATIVE OF RISK, NOT EXHAUSTIVE.** A function that is not flagged here
is not thereby certified safe. This document covers the functions read during
this audit; it is not a substitute for testing each function against real
numpy `.tobytes()` output.

Source: `numpy 2.5.1`, read from
`~/Monday/ionp/.venv/lib/python3.14/site-packages/numpy/polynomial/`
(`polyutils.py`, `legendre.py`, `laguerre.py`, `hermite.py`, `hermite_e.py`).

---

## 0. Why this matters

Floating-point arithmetic is not associative. `(a*(b-1))/b` and
`a*((b-1)/b)` can differ in the last bit. numpy's four orthogonal-polynomial
modules (`legendre`, `laguerre`, `hermite`, `hermite_e`) share a common
*shape* — each implements the same ~31-function public API by delegating to
common helpers in `polyutils.py` and using a three-term recurrence for
`mul`/`val`/`der`/`int`/`vander` — but the recurrences are **not** the same
polynomial family's recurrence copy-pasted with different constants. Each
basis has a structurally different recurrence (different terms present,
different operations, different groupings), and assuming "same shape as
Legendre, just change the constant" is exactly the trap that produces
code that returns the right *value* but the wrong *bits*.

## 1. Shared machinery (`polyutils.py`)

All five modules delegate these to the identical shared implementation.
Get `polyutils.py` bit-exact once and it is correct for all bases —
diverging accidentally in a per-basis reimplementation of any of these is a
pure regression, not a feature.

- **`trimseq(seq)`** — walks backward from the end, returns `seq[:i+1]`
  where `i` is the index of the last nonzero element; if all-zero, returns
  `seq[:1]` (i.e. the empty-coefficient case collapses to a single zero
  coefficient, never a zero-length array).
- **`as_series(alist, trim=True)`** — converts a list of array-likes to a
  common minimal float/complex dtype, makes each entry 1-D, and (by default)
  **trims trailing zeros on every input** before returning. This is the
  single biggest source of "why did my intermediate array have a different
  length than numpy's" bugs — trimming happens on the *way in*, before the
  recurrence runs, not just on the way out.
- **`trimcoef(c, tol=0)`** — the public-facing trim, used by `*trim`
  functions; same trailing-zero walk as `trimseq` but with a tolerance
  and additional all-below-tolerance zeroing pass first.
- **`_fromroots(line_f, mul_f, roots)`** — divide-and-conquer: sorts roots,
  builds a list of degree-1 `line_f(-r, 1)` factors, then repeatedly
  pairwise-multiplies the list with `mul_f` in a bottom-up merge order
  (indices `1::2` merged with `0::2` each pass, remainder pushed to the
  next round) until one polynomial remains. The pairing order is a merge
  tree, not a naive left-to-right fold — the accumulated rounding is
  **not** the same as folding roots one at a time.
- **`_div(mul_f, c1, c2)`** — polynomial long division by repeated
  subtraction: `while len(rem) >= len(c2): p = mul_f([0]*(len(rem)-len(c2)) + [1], c2); q = rem[-1]/p[-1]; rem = rem[:-1] - q*p[:-1]`. Uses each basis's own `mul_f`
  internally, so division bit-patterns depend on that basis's `mul`
  grouping.
- **`_fit(vander_f, x, y, deg, rcond=None, full=False, w=None)`** —
  builds the Vandermonde matrix via the basis's own `vander_f`, optionally
  weights, computes per-column scale `scl = sqrt((lhs*lhs).sum(1))`,
  `scl[scl==0] = 1`, then solves via **`np.linalg.lstsq(lhs.T/scl, rhs.T, rcond)`**
  and divides the result by `scl` afterward. This is SVD-based — see §5,
  not bit-exact-certifiable in general.
- **`_as_int`, `_deprecate_as_int`** — integer-coercion helpers for degree
  arguments; not arithmetic-sensitive.

## 2. Cross-module comparison table (the traps, at a glance)

| Behavior | Legendre | Laguerre | Hermite (phys.) | HermiteE (prob.) |
|---|---|---|---|---|
| `*x` array (degree-1 identity coeffs) | `[0, 1]` | `[1, -1]` | `[0, 1/2]` | `[0, 1]` |
| `*line(off, scl)` | `[off, scl]` | `[off+scl, -scl]` | `[off, scl/2]` | `[off, scl]` |
| `*mulx` uses division? | yes (`/s` where `s=i+j`) | **no** (pure int recurrence) | **no** (halves only) | **no** |
| `*mul` recurrence shape | single-term-scaled: `c1*(2nd-1)/nd` added after `mulx` | subtraction-then-divide: `(  (2nd-1)*c1 - mulx(c1) ) / nd` | no division anywhere; `mulx(c1)*2` | no division anywhere; `mulx(c1)` (no `*2`) |
| `*val` grouping | **divide-first**: `c1 * ((nd-1)/nd)` | **multiply-first**: `(c1*(nd-1))/nd` | precomputes `x2=x*2`, reused | no precompute, plain `x` |
| `*der` accumulation offset | two-off (`c[j-2] += c[j]`) | one-off (`c[j-1] += c[j]`) | none — direct `der[j-1] = 2*j*c[j]` | none — `der[j-1] = j*c[j]` |
| `*int` special first term | `tmp[2] = c[1]/3` | `tmp[1] = -c[0]` (no division at all in lagint) | `tmp[1] = c[0]/2` | `tmp[1] = c[0]` (no `/2`) |
| `*vander` recurrence | division (`/i`) | not read in this pass — **verify before relying on it** | pure multiply, no division | pure multiply, no division |
| companion matrix scaled? | yes, symmetric, `scl=1/sqrt(2n+1)` | **no** — docstring says "already symmetric...no scaling applied" | yes, cumulative-product `scl` | yes, cumulative-product `scl`, no factor-of-2 |
| companion last-column op | `-=` | `+=` (sign differs!) | `-=` | `-=` |
| `*roots` calls `_to_real_if_imag_zero`? | **no** | **yes** | **yes** | **no** |
| `*gauss` weight normalization | `w *= 2./w.sum()` | `w /= w.sum()` (no `2.`) | `w *= sqrt(pi)/w.sum()` | `w *= sqrt(2*pi)/w.sum()` |
| `*gauss` symmetrizes x/w? | yes | **no** (domain not symmetric) | yes | yes |
| `n==2` `*2poly` special case | (n/a — legendre uses general loop) | (n/a) | **`c[1] *= 2; return c`** | **`return c`** (no doubling!) |
| weight function | n/a (domain [-1,1], no weight fn in same sense) | `exp(-x)`, domain `[0, inf)` | `exp(-x**2)` | `exp(-0.5*x**2)` |

The last three rows are the sharpest traps: `herm2poly` and `herme2poly`
have the *same* `n==2` special case in the source, and one of them doubles
the coefficient while the other doesn't — copying one implementation to
produce the other without re-deriving the constant will silently produce
wrong (not just non-bit-exact, actually numerically wrong) output. Likewise
`_to_real_if_imag_zero` is called by exactly two of the four `*roots`
functions (`lagroots`, `hermroots`) and not the other two
(`legroots`, `hermeroots`) — there is no pattern by "physicists' vs.
probabilists'" or "bounded vs. unbounded domain" that predicts this; it has
to be checked per function, not inferred.

## 3. Per-function detail: Laguerre

Domain `[0, inf)`, weight `exp(-x)`. Recurrence: `L_{n+1}(x) = ((2n+1-x)L_n(x) - n*L_{n-1}(x)) / (n+1)`.

- **`lagline(off, scl)`**: `return np.array([off + scl, -scl])` if `scl != 0`
  else `[off]`. **Divergence**: Legendre's `legline` returns `[off, scl]`
  directly — Laguerre's is `[off+scl, -scl]`, an actual different affine
  map, not just a different constant slot. An implementer assuming "same
  shape, different values" will get the wrong polynomial, not just wrong
  bits.
- **`lagfromroots`**: delegates to `polyutils._fromroots(lagline, lagmul, roots)`.
  Divergence risk inherited entirely from `lagline`/`lagmul` above/below.
- **`lagadd`/`lagsub`**: standard `as_series` + pad-shorter-with-zeros +
  elementwise add/sub, trimmed via `trimseq` on return. No divergence from
  Legendre pattern here — these are basis-agnostic in numpy's source (same
  code body pattern across all four modules).
- **`lagmulx(c)`**: pure three-term integer-coefficient recurrence, **no
  division**:
  ```python
  prd = np.empty(len(c) + 1, dtype=c.dtype)
  prd[0] = c[0]
  prd[1] = -c[0]
  for i in range(1, len(c)):
      prd[i + 1] = -c[i] * (i + 1)
      prd[i] += c[i] * (2 * i + 1)
      prd[i - 1] -= c[i] * i
  ```
  **Divergence**: `legmulx` divides by `s = i+j` in its recurrence
  (`prd[j] = (c[i]*j)/s`); `lagmulx` has zero divisions. Assuming a
  division belongs here (because Legendre has one) produces wrong output,
  not just wrong bit-pattern.
- **`lagmul(c1, c2)`**: Clenshaw-style, but note the grouping:
  ```python
  c0 = lagadd(c[-2] * xs, lagsub((2 * nd - 1) * c1, lagmulx(c1)) / nd)
  ...
  c1 = lagadd(tmp, lagsub((2 * nd - 1) * c1, lagmulx(c1)) / nd)
  ...
  return lagadd(c0, lagsub(c1, lagmulx(c1)))
  ```
  (paraphrased structure — the operative point is `lagsub(X, lagmulx(c1))
  / nd`, i.e. **subtract then divide**, in contrast to Legendre's
  `legadd(tmp, (legmulx(c1) * (2*nd-1)) / nd)`, i.e. **scale then divide,
  no subtraction inside the division**.) These are different arithmetic
  trees; even choosing correct final values, evaluating them in
  Legendre's tree shape will not reproduce Laguerre's rounding.
- **`lagval(x, c)`**: multiply-then-divide grouping:
  ```python
  c0 = c[-i] - (c1 * (nd - 1)) / nd
  c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
  ```
  final step: `c0 + c1 * (1 - x)`.
  **Divergence, the headline trap**: Legendre's `legval` groups as
  `c1 * ((nd - 1) / nd)` — **divide first, then multiply** — while
  Laguerre's is `(c1 * (nd - 1)) / nd` — **multiply first, then divide**.
  Both are mathematically `c1*(nd-1)/nd` but are different floating-point
  operations and will differ in the last bit for generic inputs. This is
  the single most implementer-hostile difference in the whole audit
  because it looks identical unless you read the parenthesization
  character-by-character.
- **`lagder(c, m, scl, axis)`**: 
  ```python
  der[j - 1] = -c[j]
  c[j - 1] += c[j]
  ```
  with base case `der[0] = -c[1]`.
  **Divergence**: Legendre's `legder` has a *two*-index-back accumulation
  (`der[j-1] = (2j-1)*c[j]; c[j-2] += c[j]`) with a distinct `j==1` special
  case (`der[1] = 3*c[2]`). Laguerre's is a *one*-index-back accumulation
  with no multiplicative coefficient at all beyond the sign flip. Different
  recursion depth/shape entirely, not a relabeled constant.
- **`lagint(c, m, k, lbnd, scl, axis)`**: **no division anywhere**:
  ```python
  tmp[1] = -c[0]
  for j in range(1, n):
      tmp[j] += c[j]
      tmp[j + 1] = -c[j]
  ```
  then the usual `k`/`lbnd` constant-of-integration fixup (evaluate at
  `lbnd`, subtract from `tmp[0]`) shared in shape with other modules.
  **Divergence**: Legendre's `legint` divides (`t = c[j]/(2*j+1)`) with a
  special `tmp[2] = c[1]/3` seed; Laguerre integrates with pure
  accumulation, no division at all. An implementer expecting a `/(2j+1)`-
  style term here will not find one.
- **`lagvander(x, deg)`**: recurrence not fully re-verified in this pass —
  flagging as **unaudited-in-detail**, should be read directly before
  relying on any grouping claim for it. (All other `*vander` functions in
  the sibling modules use the corresponding `*val`-style three-term
  recurrence with the same per-module division/no-division pattern as
  `*mulx`; Laguerre's `mulx` has no division, so `lagvander` likely follows
  suit, but this was not confirmed character-by-character and should not
  be assumed bit-exact without direct comparison.)
- **`lagcompanion(c)`**: explicitly **unscaled** — the docstring states
  "the Laguerre polynomials are already symmetric... no scaling is
  applied". Main diagonal `mid = 2.*np.arange(n)+1.` is **nonzero**
  (contrast: Legendre's/Hermite's companion diagonals are zero before the
  final-column adjustment). Off-diagonal `top = bot = -np.arange(1, n)`.
  Last column: `mat[:, -1] += (c[:-1] / c[-1]) * n` — note **`+=`**, while
  Legendre's equivalent step is `mat[:, -1] -= ...`. Sign of this
  adjustment is basis-specific, not a copy-paste constant.
- **`lagroots(c)`**: builds companion matrix, calls `np.linalg.eigvals`
  (general, non-symmetric eigensolver — companion is not symmetric here
  since it's unscaled), sorts by real part, and **does** call
  `numpy.linalg._linalg._to_real_if_imag_zero(r)` before returning.
  Legendre's `legroots` does **not** call this helper (its companion is
  symmetric so it uses `eigvalsh` and the imaginary-part issue doesn't
  arise the same way). This is a real structural divergence, not
  incidental.
- **`laggauss(deg)`**: builds `lagcompanion`, uses `eigvals` (not
  `eigvalsh` — companion isn't symmetric), one Newton correction step
  using `lagval`/`lagder`-equivalent (`_normed_lagrange`-style — verify
  exact private helper name if reimplementing), then:
  `w = 1/(m*dfl)/(m/dfl).sum()`-style normalization ending in
  **`w /= w.sum()`** — no leading `2.` and, critically, **no
  symmetrization pass** (`x = (x - x[::-1])/2`-style code that Legendre
  and both Hermite variants have) because Laguerre's domain `[0, inf)` is
  not symmetric about a midpoint the way `[-1,1]` and `(-inf,inf)` are.
  Applying Legendre's symmetrization step to Laguerre nodes would be
  mathematically wrong, not just non-bit-exact.
- **`lagweight(x)`**: `np.exp(-x)`.

## 4. Per-function detail: Hermite (physicists') and HermiteE (probabilists')

These two modules are the cleanest paired illustration of "scaling
constant" divergence — HermiteE is structurally identical to Hermite with
essentially every factor-of-2 (or its sqrt) removed, **except** for two
functions where the removal is easy to get backwards (see below).

Recurrence (physicists'): `H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)`.
Recurrence (probabilists'): `He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)`.

| Function | Hermite (`hermite.py`) | HermiteE (`hermite_e.py`) |
|---|---|---|
| `*x` | `[0, 1/2]` | `[0, 1]` |
| `*line(off,scl)` | `[off, scl/2]` | `[off, scl]` |
| `*mulx` | `prd[1]=c[0]/2; prd[i+1]=c[i]/2; prd[i-1]+=c[i]*i` | `prd[1]=c[0]; prd[i+1]=c[i]; prd[i-1]+=c[i]*i` |
| `*mul` | `c0=hermsub(c[-i]*xs, c1*(2*(nd-1)))`; final `hermadd(c0, hermmulx(c1)*2)` | `c0=hermesub(c[-i]*xs, c1*(nd-1))`; final `hermeadd(c0, hermemulx(c1))` (no `*2`) |
| `*val` | precomputes `x2 = x*2`, reused every iteration; `c1*(2*(nd-1))` | no precompute; plain `x`; `c1*(nd-1)` |
| `*der` | `der[j-1] = (2*j)*c[j]` | `der[j-1] = j*c[j]` |
| `*int` | `tmp[1]=c[0]/2; tmp[j+1]=c[j]/(2*(j+1))` | `tmp[1]=c[0]; tmp[j+1]=c[j]/(j+1)` |
| `*vander` | `v[1]=x2; v[i]=v[i-1]*x2 - v[i-2]*(2*(i-1))` | `v[1]=x; v[i]=v[i-1]*x - v[i-2]*(i-1)` |
| `*2poly`, `n==2` case | `c[1] *= 2; return c` | `return c` (**no** doubling) |
| companion `scl` | `hstack((1., 1/sqrt(2*arange(n-1,0,-1))))`, then `multiply.accumulate(...)[::-1]` | `hstack((1., 1/sqrt(arange(n-1,0,-1))))`, then same accumulate/reverse — **no factor of 2 inside the sqrt** |
| companion `top`/`bot` | `sqrt(.5 * arange(1,n))` | `sqrt(arange(1,n))` (no `.5`) |
| companion last col | `mat[:,-1] -= scl*c[:-1]/(2.0*c[-1])` | `mat[:,-1] -= scl*c[:-1]/c[-1]` (no `2.0`) |
| `*roots` calls `_to_real_if_imag_zero`? | **yes** | **no** |
| normed-n helper | `_normed_hermite_n`: `c1*sqrt(2./nd)` step | `_normed_hermite_e_n`: `c1*sqrt(1./nd)` step (not `2./nd`) |
| `*gauss` df scale | `df = _normed_hermite_n(x,ideg-1) * sqrt(2*ideg)` | `df = _normed_hermite_e_n(x,ideg-1) * sqrt(ideg)` (no `2*`) |
| `*gauss` weight norm | `w *= sqrt(pi)/w.sum()` | `w *= sqrt(2*pi)/w.sum()` |
| `*gauss` symmetrizes? | yes | yes |
| weight fn | `exp(-x**2)` | `exp(-0.5*x**2)` |

**The two sharpest traps in this pair:**

1. **`herm2poly` vs `herme2poly`, the `n==2` branch.** Both modules special-
   case degree-2 conversion to power basis. Hermite's doubles the linear
   coefficient (`c[1] *= 2`); HermiteE's returns the array unmodified. If
   an implementer writes one by editing a copy of the other and forgets to
   delete this line (or forgets to *add* it going the other direction),
   the result is a **wrong value**, not merely a rounding difference — this
   is not the kind of bug bit-exactness testing against a handful of
   generic inputs will always catch quickly if degree-2 isn't specifically
   exercised.
2. **`hermroots` vs `hermeroots` and `_to_real_if_imag_zero`.** Hermite's
   root finder calls the private numpy helper
   `numpy.linalg._linalg._to_real_if_imag_zero`; HermiteE's does not — it's
   just `r.sort(); return r`. This is a real API-shape divergence: any
   port of `hermeroots` that reflexively includes the "clean up
   near-zero-imaginary-part roots" step (because Hermite has it, or because
   Legendre... doesn't, or because Laguerre does) needs to consult this
   table, not analogy, because there is no consistent rule predicting which
   of the four modules call it (`legroots`: no, `lagroots`: yes,
   `hermroots`: yes, `hermeroots`: no).

## 5. Functions that legitimately need measured tolerance, not bit-exactness

These are flagged as **inherently high divergence risk** because they rely
on iterative numerical linear algebra (`np.linalg.eigvals`, `eigvalsh`,
`lstsq`) whose bit-pattern output depends on the LAPACK backend, BLAS
threading, and internal iteration counts — not just on the literal Python
source arithmetic. Matching numpy's *source structure* for these does not
guarantee matching numpy's *bits*, even in principle, unless the underlying
linear-algebra routine is byte-for-byte the same compiled implementation.

- **`lagcompanion` / `hermcompanion` / `hermecompanion` / `legcompanion`**
  (construction is deterministic float arithmetic and *should* be
  achievable bit-exact — it's what's fed *into* the eigensolver that isn't).
- **`lagroots`, `hermroots`, `hermeroots`, `legroots`** — all call
  `np.linalg.eigvals` or `eigvalsh` on the companion matrix. Eigenvalue
  algorithms are iterative; convergence order and internal pivoting are
  backend-dependent. **I cannot certify these as bit-exact-achievable** in
  a from-scratch reimplementation unless the reimplementation calls the
  exact same LAPACK routine numpy's build calls (and even then, BLAS
  threading nondeterminism has bitten this project before per the
  `.tobytes()` verification standard already in use). This should be
  measured-tolerance-verified, not asserted bit-exact.
- **`laggauss`, `hermgauss`, `hermegauss`, `leggauss`** — same eigensolver
  dependency as `*roots`, **plus** a Newton refinement step whose fixed
  point depends on the eigensolver's initial guess's exact bit-pattern.
  Same conclusion: **not bit-exact-certifiable** by source-reading alone;
  needs measured tolerance.
- **`lagfit`, `hermfit`, `hermefit`, `legfit`** (via shared `polyutils._fit`)
  — call `np.linalg.lstsq`, which is SVD-based. Same class of problem as
  above: **not bit-exact-certifiable** from source structure alone.

Being unable to certify these as bit-exact is a legitimate finding, not a
gap in this audit — it reflects a real property of iterative numerical
linear algebra, and no amount of literal-source-reading changes that.
Everything else in this document (the `*mulx`/`*mul`/`*val`/`*der`/`*int`/
`*vander`/`*2poly`/`*line`/`*add`/`*sub` family, and the deterministic part
of `*companion` construction) is closed-form float arithmetic and should be
achievable bit-exact if — and only if — the exact operation order and
grouping documented above is reproduced.

## 6. Functions not covered in detail in this pass

`lagvander` (recurrence not verified character-by-character — see §3),
`*vander2d`/`*vander3d` (delegate to shared `polyutils._vander_nd`/
`_vander_nd_flat`, not individually re-derived per basis here),
`*grid2d`/`*grid3d`, `*pow` (delegates to shared `polyutils._pow`),
`*trim`, `*copy`, class wrapper methods (`Laguerre`, `Hermite`, `HermiteE`
classes in each module, which mostly delegate to the module-level functions
above). None of these were found to contain surprising divergences during
this pass, but per the top-of-document caveat, absence of a note here is
not certification.

---

## Independent verification (Monday, 2026-08-07)

This document was produced by a subagent. Three of its claims were spot-checked
directly against numpy's installed source before it was allowed to steer any
implementation work. **Three of N is a spot-check, not a certification** — the
unverified remainder carries the author's confidence, not mine.

**Claim: `lagval` and `legval` group the Clenshaw ratio oppositely. CONFIRMED,
and it is worse than stated.**

```
legendre.py:907   legval:  c0 = c[-i] - c1 * ((nd - 1) / nd)    <- divide first
laguerre.py:885   lagval:  c0 = c[-i] - (c1 * (nd - 1)) / nd    <- multiply first
```

The inversion additionally runs *within* Laguerre: `lagval:885` multiplies first
while `lagmul:503` divides first. So "check the grouping per function" is not
merely per-module advice — two functions in the same file disagree, exactly as
`legval`/`legmul` did. That pairing is the defect we already paid for once.

**Claim: `legmulx` divides by `s = i + j`, `lagmulx` has no division. CONFIRMED.**

```
legendre.py:52    prd[j] = (c[i] * j) / s
laguerre.py:49    prd[i + 1] = -c[i] * (i + 1)      # no division anywhere
```

**Claim: `herm2poly` doubles `c[1]` at `n == 2`, `herme2poly` does not. CONFIRMED.**

```
hermite.py:185-187     if n == 2: c[1] *= 2 ; return c
hermite_e.py:187-188   if n == 2: return c
```

Note this one is a *wrong value*, not a rounding difference — it will not show up
as a small ULP divergence, and it only fires at degree exactly 2. Any corpus that
samples degrees randomly without pinning `n == 2` can miss it entirely. Pin it.

### Unverified and explicitly flagged by the author

`lagvander`'s recurrence was not read character-by-character. Do not rely on any
grouping claim for it without reading the source first. The author flagging its
own gap is the reason the rest of the document is worth reading.

### On the functions declared un-certifiable

The author's position — that `*roots`, `*gauss` and `*fit` cannot be asserted
bit-exact because they bottom out in LAPACK eigendecomposition and SVD — matches
what we independently measured for the power basis and Legendre, where all three
families needed recorded epsilon tolerances derived from seeded sweeps. That is
consistent, but consistency is not proof: each tolerance must still be measured
per function on at least 20,000 samples, not inherited from a sibling basis
because the argument sounds the same.

---

## Late addition: `_normed_hermite_*_n` — a gap this document had (Monday, 2026-08-07)

The HermiteE implementer reported one divergence in this private helper that the
audit above does not mention. Read against numpy's installed source, there are
**four**, and the fourth is the one that matters.

```
                       hermite.py:1629-1644          hermite_e.py:1543-1558
n == 0 base return     1/sqrt(sqrt(np.pi))           1/sqrt(sqrt(2*np.pi))
c1 initialiser         1./sqrt(sqrt(np.pi))          1./sqrt(sqrt(2*np.pi))
loop, c1 update        c1 * x * np.sqrt(2./nd)       c1 * x * np.sqrt(1./nd)
final return           c0 + c1 * x * np.sqrt(2)      c0 + c1 * x
```

The first three are constant substitutions — the kind of thing a careful
"copy the sibling and change the constants" port gets right. The fourth is not a
constant: HermiteE's return expression has **no trailing factor at all**. A port
that transcribed Hermite and then hunted for twos to adjust would carry
`* np.sqrt(2)` across intact and be wrong by exactly that factor at every degree
`n >= 1`, while `n == 0` — the branch a quick smoke test is most likely to hit —
stayed correct.

`c0` is identical in both. That is not reassurance; it is why three of the four
lines look substitutable.

This helper feeds `hermgauss`/`hermegauss` weight normalisation, so the error
surfaces as wrong quadrature weights rather than as an obviously wrong
polynomial value.

**Two corrections to how this document should be read.** First, its coverage of
the *private* helpers is thinner than its coverage of the public surface, and
the gap was found by an implementer working past it rather than by the audit.
Second, the reported gap was itself incomplete: one divergence was named, four
exist. A gap report is a starting point for reading the source, never a
substitute for it.
