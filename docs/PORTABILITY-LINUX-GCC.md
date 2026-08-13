# Step 2a: aarch64 Linux / GCC / OpenBLAS — RESULT: BLOCKED BEFORE MEASUREMENT

**Measured 2026-08-10.** Ticket #83, step 2a. This is a measurement job; nothing in
`ionp-core`, `ionp-ion`, `ionp-py`, or any declaration was modified. The finding below
*is* the deliverable, not a placeholder for one.

## Headline

**No count of surviving bit-exact declarations exists for this run.** The differential
suite never executed on the Linux host because `anionpy` **does not compile** there. The
build fails at `cargo build` for `ionp-ion`, before Python, before pyo3, before any
numpy comparison. This is a harder finding than "N/1517 survive" — it means the
portability question this ticket set out to answer is currently unanswerable on any
non-Apple host, for a reason that has nothing to do with FMA contraction or BLAS
accumulation order.

## The blocker, exactly

`ionp-py/Cargo.toml` depends unconditionally on `ionp-ion`:

```toml
ionp-core = { path = "../ionp-core" }
ionp-ion  = { path = "../ionp-ion" }
```

`ionp-ion/Cargo.toml` pins its BLAS/LAPACK backend to Accelerate with no alternative and
no `target_os` gating:

```toml
cblas = "0.5.0"
blas-src = { version = "0.14.0", default-features = false, features = ["accelerate"] }
lapack = "0.20.0"
lapack-src = { version = "0.13.0", default-features = false, features = ["accelerate"] }
```

`accelerate-src` links via `#[link(kind = "framework", ...)]`, which is only meaningful
on Apple targets. On `aarch64-unknown-linux-gnu` this is a hard compile-time error, not
a runtime fallback:

```
error: library kind `framework` is only supported on Apple targets
error: could not compile `accelerate-src` (lib) due to 1 previous error
💥 maturin failed
  Caused by: Failed to build a native library through cargo
```

There is no `cfg(target_os = "linux")` branch anywhere in `ionp-ion/src` or
`ionp-core/src` that selects a different backend (checked with
`grep -rn "target_os" ionp-ion/src ionp-core/src` — no hits), and no feature flag
exposed on `ionp-ion` to swap `"accelerate"` for `"openblas"` or `"system"` at build
time. The dependency is not optional and not configurable from outside the manifest.
Making it so would require editing `Cargo.toml` and very likely gating call sites in
`ionp-ion/src` — i.e., leaving the scope of a measurement ticket and entering the scope
of a portability *fix*. Ticket #83 step 2a is explicitly measurement-only ("If you find
yourself editing Rust, you have left scope"), so no such edit was made, on the host or
in the canonical tree.

Note this is a **different class of problem** than the two mechanisms
`REFERENCE-PLATFORM.md` names (FMA contraction, Accelerate accumulation order). Those
are numerical-portability risks *within* code that compiles everywhere. This is a
build-portability blocker: the crate graph itself will not produce a `.so` on Linux, so
the numerical question is currently moot for anything that links `ionp-ion` — and
because `ionp-py` depends on `ionp-ion` unconditionally, that blocks the *entire*
extension module, including the purely-portable parts of `ionp-core` that have nothing
to do with BLAS.

## What was established (provisioning succeeded; build did not)

Everything up to `cargo build` worked cleanly and is recorded here for whoever picks
this back up.

### Toolchain achieved on the host — verified, pasted

```
numpy version: 2.5.1
{'Build Dependencies': {'blas': {'detection method': 'pkgconfig',
                                 'found': True,
                                 'include directory': '/opt/_internal/cpython-3.14.3/lib/python3.14/site-packages/scipy_openblas64/include',
                                 'lib directory': '/opt/_internal/cpython-3.14.3/lib/python3.14/site-packages/scipy_openblas64/lib',
                                 'name': 'scipy-openblas',
                                 'openblas configuration': 'OpenBLAS '
                                                           '0.3.33.112.0  '
                                                           'USE64BITINT '
                                                           'DYNAMIC_ARCH '
                                                           'NO_AFFINITY '
                                                           'neoversev2 '
                                                           'MAX_THREADS=64',
                                 'pc file directory': '/project/.openblas',
                                 'version': '0.3.33.112.0'},
                        'lapack': {'detection method': 'pkgconfig',
                                   'found': True,
                                   'include directory': '/opt/_internal/cpython-3.14.3/lib/python3.14/site-packages/scipy_openblas64/include',
                                   'lib directory': '/opt/_internal/cpython-3.14.3/lib/python3.14/site-packages/scipy_openblas64/lib',
                                   'name': 'scipy-openblas',
                                   'openblas configuration': 'OpenBLAS '
                                                             '0.3.33.112.0  '
                                                             'USE64BITINT '
                                                             'DYNAMIC_ARCH '
                                                             'NO_AFFINITY '
                                                             'neoversev2 '
                                                             'MAX_THREADS=64',
                                   'pc file directory': '/project/.openblas',
                                   'version': '0.3.33.112.0'}},
 'Compilers': {'c': {'commands': 'cc',
                     'linker': 'ld.bfd',
                     'name': 'gcc',
                     'version': '14.2.1'},
               'c++': {'commands': 'c++',
                       'linker': 'ld.bfd',
                       'name': 'gcc',
                       'version': '14.2.1'},
               'cython': {'commands': 'cython',
                          'linker': 'cython',
                          'name': 'cython',
                          'version': '3.2.8'}},
 'Machine Information': {'build': {'cpu': 'aarch64', 'endian': 'little',
                                    'family': 'aarch64', 'system': 'linux'},
                         'host': {'cpu': 'aarch64', 'endian': 'little',
                                  'family': 'aarch64', 'system': 'linux'}},
 'SIMD Extensions': {'baseline': ['NEON', 'NEON_FP16', 'NEON_VFPV4', 'ASIMD'],
                     'found': ['ASIMDHP', 'ASIMDDP', 'ASIMDFHM', 'SVE']}}
```

This is exactly the confound-free target the ticket asked for: numpy **2.5.1**, the same
version as the macOS reference (no version confound, unlike the 2.4.4 previously
surveyed on this same host in `REFERENCE-PLATFORM.md`), built by **gcc 14.2.1** against
**scipy-openblas 0.3.33.112.0** (an OpenBLAS variant, not Accelerate), on **aarch64
Linux**, matching architecture to isolate toolchain per the step-2a rationale.

### Side-by-side toolchain table

| Property | macOS reference (Accelerate) | Linux host (this run) |
|---|---|---|
| numpy | 2.5.1 | 2.5.1 |
| OS | macOS 26.3.1 | Ubuntu 24.04.3 |
| Arch | arm64 (Apple silicon) | aarch64 |
| C compiler | clang 15.0.0 (Apple) | gcc 14.2.1 |
| BLAS | Accelerate (system) | scipy-openblas 0.3.33.112.0 |
| LAPACK | Accelerate (system) | scipy-openblas 0.3.33.112.0 |
| Python | CPython 3.14.6 | CPython 3.14.7 |
| rustc | 1.94.1 | 1.97.1 (stable) |
| maturin | 1.14.1 | 1.14.1 |
| ABI | abi3-py314 | abi3-py314 |
| target | aarch64-apple-darwin | aarch64-unknown-linux-gnu |
| **`anionpy` build result** | succeeds (shipping .so) | **fails at `cargo build` — `ionp-ion` will not compile** |

### Provisioning steps executed (all user-local, all under `$HOME`)

```bash
# 1. uv
curl -LsSf https://astral.sh/uv/install.sh | sh          # -> ~/.local/bin

# 2. Python 3.14
export PATH=$HOME/.local/bin:$PATH
uv python install 3.14                                    # -> installed 3.14.7

# 3. venv + deps (prebuilt manylinux wheel, NOT built from source)
mkdir -p ~/anionpy-portability
cd ~/anionpy-portability
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python numpy==2.5.1 maturin --only-binary numpy

# 4. Rust (rustup was already present from a prior session on this box;
#    this run updated it to stable 1.97.1, all still under ~/.cargo, ~/.rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal

# 5. Source, excluding Mac build artifacts
rsync -avz --exclude 'target/' --exclude '.venv/' --exclude '*.so' \
  ~/Monday/ionp/ <linux-host>:~/anionpy-portability/ionp/

# 6. Build attempt (this is the step that failed)
cd ~/anionpy-portability/ionp
~/anionpy-portability/.venv/bin/maturin develop --release
# -> error: library kind `framework` is only supported on Apple targets
# -> error: could not compile `accelerate-src` (lib) due to 1 previous error
# -> 💥 maturin failed
```

No differential suite run, no `tests/differential/run.py`, no `tools/coverage.py` — there
is no `.so` for either to import. No results JSON was produced or copied back; there is
nothing at `~/anionpy-portability/r83_linux.json` on the host and nothing was copied to
`/tmp` on the Mac, because nothing was generated.

## Baseline (macOS/Apple-clang/Accelerate), for reference — unchanged by this run

From `/tmp/r85verify.json` on the Mac, re-verified by direct JSON inspection during this
job (2141 total cases, keys `max_ulp`/`mechanism`/`tolerant`/`verdict`):

- 33 failures, 0 failing (distinct field), 0 phantom, coverage 51.491% (1571/3051),
  **1517 bit-exact**.
- The 33-name failure set (verified by direct read of the file, not summarized):

```
alias/nonaliased/__ipow__       alias/self/__ipow__            alias/view/__ipow__
emath.power                     expm1                          finfo.__init__
finfo.__new__                   float_power                    hypot
iinfo.__init__                  iinfo.__new__                  isnat
ldexp                           matrix.__setitem__             memmap.__new__/unsupported_dtype
memmap.__repr__                 min_scalar_type                ndarray.__ipow__
ndarray.__pow__                 ndarray.__rpow__                ndarray.__rtruediv__
order/ufunc/isnat                out_axis/ufunc/isnat            polynomial.chebyshev.chebfromroots
polynomial.laguerre.lagfromroots polynomial.polynomial.polyfromroots
polynomial.polynomial.polymul    polynomial.polynomial.polypow   pow
power                            promote_types                   result_type
stack
```

This is the macOS baseline only, included so the two platform tables sit side by side.
It was **not** re-run in this job; it is quoted from the existing artifact for context.

## Set differences (per the ticket's deliverable)

Not computable. There is no Linux failure set to diff against the macOS failure set,
because there is no Linux run. Reporting "0 new failures" or "N/1517 survive" would be a
fabricated number dressed as a measurement — exactly the failure mode this ticket exists
to prevent. The honest answer is: **unknown, blocked at compile time, not measured.**

## Mechanism analysis

Not performed, and cannot be, without a working build. The FMA-contraction and
BLAS-accumulation-order hypotheses from `REFERENCE-PLATFORM.md` remain exactly as
speculative as they were before this run — this job neither confirms nor refutes them.
The one new, confirmed fact is upstream of both: a third, previously undocumented
mechanism exists — a **build-time platform dependency**, not a numerical one — and it
blocks measurement of the other two entirely on any non-Apple target as the crate is
currently structured.

## Recommendation (out of scope to act on here; noting it because it's the honest next step)

Before step 2a can produce a number, `ionp-ion`'s BLAS/LAPACK backend selection needs to
become buildable on a second platform — e.g. `target_os`-conditional `blas-src`/
`lapack-src` features (`accelerate` on macOS, `openblas`-family or `system` elsewhere),
or making `ionp-ion` itself an optional/feature-gated dependency of `ionp-py` if the
differential surface under test doesn't actually need it. Either is a real code change
with its own review, and is explicitly **not** something this measurement ticket did or
should do. Whoever picks this up should re-run steps 1–5 above unchanged (the
provisioning is proven and reversible) and only step 6 needs a different crate graph to
attempt.

## Disk usage on <linux-host>

| Point | `df --output=avail -BG /` |
|---|---|
| Before any install | 30G |
| After uv + Python 3.14 + venv + numpy/maturin install | 30G |
| After rustup update | 30G |
| After rsync of source (~107M) | 30G |
| After failed build attempt (11M partial `target/`) | 30G |

Never approached the 10G stop threshold; disk was not a constraint in this run.
Directory sizes at end of job: `~/anionpy-portability` 107M (incl. 11M partial
`ionp/target/` from the failed build attempt), `~/.cargo` 678M, `~/.rustup` 1.5G,
`~/.local/share/uv` 90M. The `~/.cargo`/`~/.rustup` sizes reflect a rustup installation
that already existed on this host from a prior, unrelated session (this job only updated
it to stable 1.97.1); nothing about that state is specific to this job's failure.

## Cleanup commands (not executed — nothing was left in a dangerous or disk-threatening
state, and a future attempt at the same job would need most of it again; recorded here
per instructions so a full teardown is available on request)

```bash
ssh <linux-host> 'rm -rf ~/anionpy-portability'                 # this job's own tree, incl. partial target/
ssh <linux-host> 'rm -rf ~/.rustup ~/.cargo'                     # only if the pre-existing rustup install should also go
ssh <linux-host> 'rm -rf ~/.local/bin/uv ~/.local/bin/uvx ~/.local/share/uv'
```

Nothing under `/data`, `/etc`, `/usr`, `/opt`, or `/var` was touched. No service on
this host was started, stopped, restarted, or probed. `sudo` and `apt` were never
invoked.

## Bottom line

Step 2a did not produce a surviving-count. It produced a more fundamental finding: as
currently structured, `anionpy` cannot be built at all on a non-Apple target, for a
reason (unconditional `accelerate`-only `blas-src`/`lapack-src` in `ionp-ion`) that is
independent of, and prior to, the FMA-contraction and BLAS-accumulation-order questions
`REFERENCE-PLATFORM.md` raises. Until that dependency is made platform-conditional —
which is a source change and out of scope for this measurement ticket — step 2a (and by
extension step 2b) remain unrunnable, not merely unrun.

## 2026-08-13 update: the blocker above is resolved; step 2a has now been measured

Commit `72242f9` ("ionp-ion: make BLAS/LAPACK backend target-conditional (fix #88)") made
the `blas-src`/`lapack-src` selection in `ionp-ion/Cargo.toml` `target_os`-conditional,
which is exactly the fix this file's "Recommendation" section called for. That commit was
not part of this measurement job — it had already landed on the canonical tree by the time
this re-run happened.

With that fix in place, the same host (`<linux-host>`) built and ran `anionpy` cleanly
against numpy 2.5.1 / gcc 14.2.1 / scipy-openblas 0.3.33.112.0. The full result — headline
numbers, the complete newly-failing/newly-passing name lists, and mechanism grouping — is
recorded in `REFERENCE-PLATFORM.md` under the dated section **"2026-08-13: step 2a
actually measured (aarch64 Linux / GCC / OpenBLAS)"**, not duplicated here, since that file
is the one making the scope claim this number bears on.

One unrelated environment issue was hit and fixed non-invasively: `tests/differential/
memmap_cases.py` hardcodes a macOS-only `/private/tmp` path for its scratch directory,
which doesn't exist on Linux. `/private/tmp` was created on the host
(`sudo mkdir -p /private/tmp && sudo chmod 1777 /private/tmp`) rather than editing the
test, per this ticket's no-test-edits rule. This is orthogonal to the build blocker above
and to the FMA/BLAS mechanisms — it is purely a hardcoded-path portability gap in the test
harness itself, worth fixing properly (e.g. `tempfile.gettempdir()`) in a follow-up that
isn't this measurement ticket.

Headline: of the 2141 differential-suite items, 2000 that pass on macOS also pass on this
Linux/OpenBLAS build; 108 that pass on macOS fail here; 1 that fails on macOS (
`polynomial.polynomial.polyfromroots`) passes here. At the declaration ledger level,
`tools/coverage.py` on this host reports 1440 bit-exact vs numpy, against 1517 on macOS.
Disk on the host stayed flat at 12G free throughout (well above the 8G stop threshold).
