# Reference platform for anionpy's bit-exactness claims

**Measured 2026-08-08.** This document exists because of ticket #83. It is a scope
statement, not a disclaimer: it records *what our exactness declarations are claims
about*, so that nobody — including a future me — reads "bit-exact vs numpy" as a
platform-independent guarantee it was never measured to be.

## The claim, stated precisely

The ledger currently reports **1517 items bit-exact vs numpy** (corrected 2026-08-13;
this file previously said 1515 — see the dated section below for the re-measurement that
caught it). Every one of those was
measured by running the differential suite against one particular numpy binary on one
particular machine. The honest form of the claim is:

> On the platform described below, anionpy's output is bit-identical to numpy's.

It is **not**:

> anionpy's output is bit-identical to whatever numpy produces anywhere.

Nothing in the suite has ever been executed on x86-64, on glibc, on a GCC-built numpy,
or against an OpenBLAS/MKL-backed numpy. Those results are unknown — not assumed good,
not assumed bad. **Unknown.**

## The reference numpy

| Property | Value |
|---|---|
| numpy | 2.5.1 |
| C compiler | clang 15.0.0 (Apple) |
| C++ compiler | clang 15.0.0 (Apple) |
| cython | 3.2.8 |
| BLAS | Accelerate (system) |
| LAPACK | Accelerate (system) |
| host / build CPU | aarch64, little-endian, darwin |

Reproduce with:

```
.venv/bin/python -c "import numpy as np; print(np.show_config('dicts'))"
```

## The reference host and our own toolchain

| Property | Value |
|---|---|
| OS | macOS 26.3.1 |
| Arch | arm64 (Apple silicon) |
| Python | CPython 3.14.6 |
| rustc | 1.94.1 |
| cargo | 1.94.1 |
| maturin | 1.14.1 |
| ABI | abi3-py314 |
| Rust profile | release (stock Cargo defaults -- no `[profile]` block in the workspace: `opt-level=3`, `lto=false`, `codegen-units=16`) |
| target | aarch64-apple-darwin |

## Why the compiler is load-bearing, not incidental

This is the finding that turned #83 from pedantry into a real ticket.

Ticket #47 closed a 55-ULP divergence in complex `expm1`. The cause was **not** a
different formula and **not** a different rounding mode. numpy's C is compiled by Apple
clang with `-ffp-contract=fast` in effect, which silently fuses a multiply followed by an
add into a single hardware FMA — one rounding step where the source text implies two. The
fix was to route the matching Rust expression through `mul_add_ext` at the exact site
clang chose to fuse.

That means the fix reproduces **a compiler's decision**, not a library's algorithm. A
numpy built by GCC, or by clang with a different contraction setting, may fuse a
different set of sites — and then our carefully matched FMA becomes the divergence rather
than the cure.

Current blast radius, measured 2026-08-08:

```
grep -rn "mul_add" --include="*.rs" ionp-core/src ionp-py/src
```

**40 sites across 3 files:** `ionp-core/src/ufunc.rs` (27),
`ionp-core/src/random/distributions.rs` (12), `ionp-py/src/lib.rs` (1).

Each of those is a place where we deliberately matched — or deliberately avoided — a
contraction we observed in an Apple-clang build. Each is therefore a potential
platform-conditional result.

A second, independent instance of the same class of problem: the reference numpy links
**Accelerate** for BLAS/LAPACK. Any item whose numpy implementation dispatches into BLAS
inherits Accelerate's accumulation order. An OpenBLAS or MKL build would give a different
one, and no amount of care in our Rust changes that.

## What would settle it

In increasing cost:

1. **Document the reference platform.** This file. Nearly free, and it converts a silent
   assumption into a stated scope.
2. **Run the differential suite once on x86-64 Linux against a GCC-built,
   OpenBLAS-backed numpy.** The *number* is the deliverable: how many of the 1515
   bit-exact declarations survive. If it is 1515, this ticket closes and the platform
   note stays as documentation. If it is materially fewer, we have a real design
   question and actual data to answer it with.
3. **Only if (2) comes back bad:** per-platform declaration states, so an item can be
   exact on one target and ULP-tolerant on another.

Step 2 has not been run. Until it has, any claim about non-arm64 behaviour in this repo
is speculation and must be labelled as such.

### Step 2 is host-blocked, not merely unscheduled (surveyed 2026-08-08)

Every reachable machine was measured, not assumed:

| Host | Arch | OS | Verdict as a step-2 host |
|---|---|---|---|
| this Mac | arm64 | macOS 26.3.1 | *is* the reference platform; cannot contrast with itself |
| <linux-host> | **aarch64** | Ubuntu 24.04.3 | wrong toolchain host — see below |
| <macos-host-b> | arm64 | Darwin | second macOS box; not a contrast. Read-only, not ours |
| <linux-host-c> | **aarch64** | Ubuntu 24.04.4 | another party's box; off limits without a gate |

**There is no x86-64 machine on this network.** Step 2 as originally written cannot be run
on available hardware.

The <linux-host> box is the interesting near-miss. It carries precisely the contrasting
toolchain this ticket cares about — numpy 2.4.4 built by **gcc 14.2.1** against
**scipy-openblas 0.3.31**, on **glibc 2.39** — but it is unusable as-is: Python 3.12 (abi3
needs ≥3.14), no rustc/cargo/maturin at all, numpy 2.4.4 rather than the reference 2.5.1
(a version confound stacked on top of the toolchain variable), and its disk is at 98% with
27 GB free while hosting a large working corpus. Provisioning it is a resource decision
belonging to its owner, not a measurement.

**A distinction worth keeping straight before anyone runs this.** Step 2 as phrased varies
two things at once — *architecture* (x86-64 vs aarch64) and *toolchain* (GCC/OpenBLAS/glibc
vs clang/Accelerate/libSystem). The two mechanisms this document actually identifies as
load-bearing are both **toolchain**: clang's `-ffp-contract=fast` FMA fusion, and the
Accelerate BLAS accumulation order. Architecture is the *less* likely culprit of the two.

So a run on **aarch64 Linux** — GCC, OpenBLAS, glibc, architecture held constant — isolates
the named mechanisms and is the cheaper and cleaner first experiment, even though it is not
what step 2 literally says. It would not settle x86-64. It would settle whether the FMA and
BLAS matching are portable, which is the part we have an actual causal story for.

Recommended revision, therefore: **step 2a** = aarch64 Linux / GCC / OpenBLAS (isolates
toolchain); **step 2b** = x86-64 (isolates architecture), only if 2a comes back clean and
the release claim still needs it. Neither has been run. Do not cite either as evidence.

### 2026-08-13: step 2a actually measured (aarch64 Linux / GCC / OpenBLAS)

The build blocker documented in `PORTABILITY-LINUX-GCC.md` (accelerate-only `blas-src`/
`lapack-src` in `ionp-ion`) was fixed on the canonical tree before this run, by commit
`72242f9` ("ionp-ion: make BLAS/LAPACK backend target-conditional (fix #88)") — that fix
is what made this measurement possible; it was not done as part of this ticket.

**Host:** `<linux-host>`, already provisioned per the prior survey: numpy **2.5.1** (version
held constant to remove that confound), gcc 14.2.1, glibc 2.39, scipy-openblas
0.3.33.112.0, `anionpy` `maturin develop`-installed and verified functional
(`ap.matmul` through OpenBLAS returns correct values) before anything else ran.

**One environment fix was required to run the suite at all**, unrelated to any numerical
result: `tests/differential/memmap_cases.py` hardcodes `tempfile.mkdtemp(dir="/private/tmp")`,
a macOS-only path that does not exist on Linux. `/private/tmp` was created on the host
(`sudo mkdir -p /private/tmp && sudo chmod 1777 /private/tmp`, mirroring `/tmp`'s standard
permissions) rather than editing the test file, since ticket #83 forbids editing tests.
No repo file was touched to make this run possible.

**Headline, from the differential suite's own JSON verdicts** (methodology: failure set =
`{k for k,v in d.items() if v['verdict']=='fail'}`, exactly as specified):

- Suite total: **2141** items (matches the macOS baseline's own recorded total of 2141 in
  `PORTABILITY-LINUX-GCC.md`, confirming identical registry composition across platforms —
  this comparison is not decorative; it demonstrably detects real divergence, below).
- macOS-pass candidates (2141 total − 33 pinned macOS failures): **2108**.
- **N_survived = 2000** — items passing on macOS that also pass on Linux.
- **Newly failing on Linux: 108** (full list below).
- **Newly passing on Linux: 1** — `polynomial.polynomial.polyfromroots` (was in the
  macOS-33; passes on this Linux/OpenBLAS run).
- 32 of the macOS-33 remain failing on Linux (unsurprising — those were never claimed
  bit-exact anywhere).

Separately, `tools/coverage.py --tests <results.json>` on the Linux host reports, at the
declaration level (not the raw-item level above): `exact` state = 1485 declared items, of
which **1440 bit-exact vs numpy** on this Linux run, against **1517** on macOS. These two
numbers (108 newly-failing raw items vs. a 77-item drop in the declaration ledger) are
different granularities of the same underlying data — the differential-suite JSON keys
(2141) are test-registry entries, not a 1:1 map to `__ion_state__` declarations (3051
surface items, 1485 declared `exact`) — and reconciling them item-for-item was not done;
it is out of this ticket's scope. Both numbers point the same direction: real, non-trivial
loss of bit-exactness on the aarch64/GCC/OpenBLAS platform, concentrated in named clusters
below.

**NEWLY FAILING ON LINUX, BY NAME, IN FULL (108), with max_ulp / mechanism from the JSON:**

FMA-contraction-consistent (complex arithmetic, cbrt, polynomial evaluation, random
distributions — all multiply-add-heavy code paths, i.e. exactly the class of code the
40 measured `mul_add` sites exist to control):

```
cbrt (ulp 2.0)                          dtype/ufunc/cbrt (ulp 1.0)
divide (ulp inf)                        true_divide (ulp inf)
dtype/ufunc/divide (ulp 4.36e18)        multiply (ulp 101.0)
reciprocal (ulp 2.0)                    ndarray.__truediv__ (ulp 64.0)
alias/self/__imul__ (ulp n/a)           alias/self/__ipow__ (ulp n/a)
alias/self/__itruediv__ (ulp n/a)       alias/view/__itruediv__ (ulp n/a)
ndarray.__itruediv__ (ulp n/a)          emath.logn (ulp 66.0)
random.Generator (ulp 3.0)              random.Generator.chisquare (ulp 4.0)
random.Generator.f (ulp 3.0)
polynomial.chebyshev.chebgrid2d (ulp 2.0)   polynomial.chebyshev.chebgrid3d (ulp 32.0)
polynomial.chebyshev.chebval3d (ulp 1.0)    polynomial.chebyshev.chebvalnd (ulp 1.0)
polynomial.hermite.hermgrid2d (ulp 1.0)     polynomial.hermite.hermgrid3d (ulp 4.0)
polynomial.hermite.hermval3d (ulp 2.0)      polynomial.hermite.hermvalnd (ulp 2.0)
polynomial.hermite_e.hermegrid2d (ulp 4.0)  polynomial.hermite_e.hermegrid3d (ulp 1.0)
polynomial.laguerre.laggrid2d (ulp 1.0)     polynomial.laguerre.laggrid3d (ulp 1.0)
polynomial.laguerre.lagval (ulp 2.0)        polynomial.laguerre.lagval3d (ulp 1.0)
polynomial.laguerre.lagvalnd (ulp 1.0)      polynomial.legendre.leggrid2d (ulp 1.0)
polynomial.legendre.leggrid3d (ulp 2.0)     polynomial.legendre.legval (ulp 4.0)
polynomial.legendre.legval3d (ulp 1.0)      polynomial.legendre.legvalnd (ulp 1.0)
polynomial.polynomial.polygrid2d (ulp 1.0)  polynomial.polynomial.polygrid3d (ulp 4.0)
polynomial.polynomial.polyval3d (ulp 1.0)   polynomial.polynomial.polyvalnd (ulp 1.0)
```
(41 items)

BLAS/reduction-accumulation-order-consistent — `linalg.lstsq` genuinely dispatches into
LAPACK/BLAS; `var`/`nanvar` are `mechanism=epsilon` reductions whose summation order is
exactly the sensitive quantity even though they don't call BLAS directly; `cumprod`,
`nancumprod`, `nanprod`, `prod` and their `ndarray.*` aliases are reduction-order-sensitive
but do **not** dispatch into BLAS (BLAS doesn't do elementwise cumulative products) —
grouped here as "accumulation-order-adjacent," not literally the BLAS mechanism:

```
linalg.lstsq (ulp 8.73e18, epsilon)     var (ulp 3.0, epsilon)
nanvar (ulp 5.37e8, epsilon)            cumprod (ulp 305.0)
nancumprod (ulp 305.0)                  layout/nancumprod (ulp 157.0)
nanprod (ulp 17.0)                      prod (ulp 17.0)
ndarray.cumprod (ulp 305.0)             ndarray.prod (ulp 17.0)
```
(10 items)

Fits **neither** named mechanism — four distinct, separately-diagnosed residual causes:

1. **`longdouble`/`clongdouble` type-width gap** (not a compiler/FMA/BLAS effect — a real
   C-type-width difference: this Linux/glibc/aarch64 numpy has a genuine 128-bit
   `float128`/`complex256`, while macOS numpy aliases `longdouble` to `float64`; anionpy
   hardcodes the macOS mapping):
   `asarray, dtype.__new__, dtype.__repr__, dtype.__str__, dtype.alignment,
   dtype.itemsize, dtype.name, dtype.str, empty, eye, identity, linspace, ones, zeros,
   zeros_like` (15 items, all `max_ulp=None`/`0.0`, verified in the raw log as
   `dtype_spelling/*longdouble*` or `*clongdouble*` cases only).

2. **Narrow float→int cast undefined behavior** (C UB for casting an out-of-range float
   to `uint8`/`uint16`/`int32`; result is compiler-codegen-dependent, not FMA and not
   BLAS): `concatenate, hstack, vstack, full, full_like, ndarray.astype,
   dtype/ufunc/abs, dtype/ufunc/add, dtype/ufunc/bitwise_and, dtype/ufunc/bitwise_count,
   dtype/ufunc/bitwise_invert, dtype/ufunc/bitwise_not, dtype/ufunc/bitwise_or,
   dtype/ufunc/ceil, dtype/ufunc/conj, dtype/ufunc/conjugate, dtype/ufunc/divide,
   dtype/ufunc/floor, dtype/ufunc/floor_divide, dtype/ufunc/fmax, dtype/ufunc/fmin,
   dtype/ufunc/gcd, dtype/ufunc/invert, dtype/ufunc/lcm, dtype/ufunc/maximum,
   dtype/ufunc/minimum, dtype/ufunc/multiply, dtype/ufunc/negative, dtype/ufunc/positive,
   dtype/ufunc/reciprocal, dtype/ufunc/sign, dtype/ufunc/square, dtype/ufunc/trunc,
   dtype/ufunc/true_divide, tan` (32 items — every `dtype/ufunc/*` failure not already
   listed under FMA above is this same `casting_unsafe`/`target_uint8` pattern, verified
   by grep against the raw log).

3. **`int32` negative-decimal rounding** (`around`/`round`/`ndarray.round` at
   `dec=-1`/`dec=-3` on `int32` — cause not diagnosed further; grouping only by symptom):
   `around, round, ndarray.round` (3 items).

4. **libc floating-point exception/warning-flag reporting differs between glibc and Apple
   libm** for `divmod`/`remainder`/masked-array `%` on special values (nan/inf/0) — the
   numeric *results* match in the log excerpts checked; what differs is whether a
   `RuntimeWarning` fires: `complex_warning_on_real_cast, fperr_subsystem, ma_warning_imod,
   ma_warning_mod_dunder` (4 items).

Unclassified residual — genuinely doesn't fit any bucket above and is flagged rather than
guessed at: `crossing/ufunc/complex_divide_branch_coverage` (892/14872 sub-cases failed,
`max_ulp=0.0` reported on cases where both sides print as `nan+nanj`) — most likely a NaN
payload/bit-pattern difference between glibc and Apple libm on the same branch-coverage
sweep that ULP-distance-0.0 can't see through a naive compare, but this was not confirmed
at the byte level and should not be asserted as fact (1 item).

**What this comparison cannot see:** it is a same-registry, same-commit, verdict-only diff
— it would not detect a case where the *test itself* is insensitive to the platform (e.g.
an item whose differential cases happen not to exercise any FMA-fusible expression at all
would show `pass` on both platforms with no information about portability). It also
depends on the macOS-33 baseline being current and on the two runs sharing an identical
registry, which was checked (2141 items each side, exact total match) rather than assumed.
It says **nothing about x86-64** — that is step 2b, not run.

No declaration, tolerance, or test was changed to produce this result. Linux results JSON
committed at `ionp/tests/differential/results/r83-linux-aarch64-gcc-openblas.json`.

## What this is explicitly NOT

It is **not** an argument to loosen tolerances. "It might differ on another platform" is
not a reason to stop measuring exactness on the platform we can actually measure. Every
declaration here was earned against a real binary and stays earned. The only thing this
document changes is the *scope* those declarations are asserted over.

Nor is it a reason to remove the FMA matching in the 40 sites above. Matching Apple
clang is correct on the reference platform, which is the only platform we have evidence
about. Removing it would trade a measured pass for an unmeasured hope.

## Re-measurement

Per the standing rule that **a decline is not durable** — and neither is a platform
snapshot — the tables above must be re-measured, not merely cited, before they are used
to justify a decision. If numpy, the OS, or the Rust toolchain moves, this file is stale
until someone re-runs the two commands at the top and updates the date in the heading.

### Correction to the step-2a write-up: what was measured vs. what was inferred (2026-08-13)

The headline numbers above reproduce exactly against the committed JSON — 2141 items,
108 newly failing, 1 newly passing (`polynomial.polynomial.polyfromroots`), 1440 vs 1517
bit-exact. Those stand. Two claims in the presentation do not, and the correction makes
the finding *stronger*, not weaker.

**1. "with max_ulp / mechanism from the JSON" overstates provenance.** The JSON's
`mechanism` field is literally `none` for **105 of the 108**, and `epsilon` for the other 3:

```
105  none
  3  epsilon
```

So the FMA / BLAS / neither grouping was *inferred from item names*, not read from the
results. It may well be right — but it is a hypothesis, and labelling it "from the JSON"
lends it evidence it does not have. `max_ulp` **is** genuinely from the JSON; `mechanism`
as used in that section is not.

**2. The `longdouble` cluster is real but was mis-located, and it is much bigger than 15.**
Grepping the 108 names for `longdouble|float128|clongdouble` returns **zero** — the affected
items are named `dtype/ufunc/add`, `asarray`, `eye`, `full` and so on. The cluster is real;
it just isn't visible by name.

Measured directly, on the two hosts, same numpy 2.5.1:

| | `longdouble` itemsize | `dtype.name` | `np.float128` exists |
|---|---|---|---|
| macOS arm64 (reference) | **8** | `float64` | **False** |
| Linux aarch64 / glibc | **16** | `float128` | **True** |

`clongdouble` likewise 16 vs 32 bytes. On glibc/aarch64 `long double` is a genuine
128-bit type; on macOS it is an alias for `float64`. **numpy on Linux therefore exposes a
dtype that numpy on macOS does not have at all.**

**Why this is the most important sentence in this document:** `anionpy`'s dtype surface was
built and declared against a numpy where `longdouble` is a lie. That is not a
floating-point *matching* problem and no amount of FMA care addresses it — it is a missing
type.

**The measured discriminator.** Of the 108 newly-failing items, **37 have `max_ulp` exactly
0**. A failure at zero ULP distance is by construction *not* an arithmetic divergence — it
is a dtype, shape, repr, or exception difference. Twenty of the 37 are `dtype/ufunc/*`
items; the rest are `around`, `asarray`, `concatenate`, `empty`, `eye`, `full` and kin —
the dtype-surface, uniformly.

Full measured distribution:

```
 37  max_ulp == 0        not arithmetic at all
 30  max_ulp 1-4         FMA-contraction-shaped
 11  max_ulp 5-1000
  7  max_ulp > 1000      too large to be a rounding difference
 23  max_ulp null
```

**Stated honestly:** the itemsize table and the 37/30/11/7/23 split are *measured*. That
the 37 are caused by `float128`'s presence is an *inference* — strongly supported by the
type-width measurement and by the items being the dtype surface, but not separately
confirmed case-by-case. Anyone acting on it should confirm one case directly first.

**Revised reading of the ticket.** "108 of 2108 items are toolchain-fragile" is the wrong
summary. The better one: roughly a third of the regression is a **missing 128-bit type**,
a portability gap with a concrete fix; the FMA-shaped arithmetic residue is around 30
items, materially smaller than the raw count suggests. Both are real. They need different
work, and conflating them would send that work in the wrong direction.

Still says nothing about x86-64. Step 2b has not been run.
