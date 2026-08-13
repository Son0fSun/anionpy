# Developing anionpy

Toolchain foundation stage (GOAL-ionp.md, ordering step 0). This document
covers build/test only — no feature work has landed yet.

## Versions resolved (2026-07-31, this machine)

- rustc / cargo 1.94.1
- Python 3.14.6 (`/opt/homebrew/bin/python3`)
- maturin 1.14.1 (installed via pipx — see below)
- pyo3 0.29.0 — this is the first pyo3 release with standard `abi3` support
  for Python 3.14 (`abi3-py314` feature). Do not go below 0.29.0 on this
  interpreter.
- numpy (rust-numpy) 0.29.0 — matches pyo3 0.29.0, required.
- rustfft 6.4.1, faer 0.24.4, num-complex 0.4.6, rayon 1.12.0
- Python-side numpy: 2.5.1 in the local `.venv` (repo/task baseline is
  2.4.2; 2.5.1 is what `pip install numpy` resolved to today and is what was
  actually used for the verification run below — re-pin to 2.4.2 if exact
  version parity with the coverage-ledger baseline matters for a given task)

## Installing maturin

Python 3.14 here is Homebrew-managed (PEP 668 externally-managed
environment) — plain `pip3 install maturin` is blocked, and
`--break-system-packages` is banned by project policy. What worked:

```sh
brew install pipx
pipx install maturin
```

This installs the `maturin` binary to `~/.local/bin/maturin` (isolated pipx
venv, does not touch the Homebrew Python site-packages). Add `~/.local/bin`
to `PATH` if it isn't already:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

`pip3 install --user maturin` was not tried since the pipx path worked on
the first attempt; try that fallback only if pipx is unavailable.

## Python virtualenv

`maturin develop` needs an activated venv with `pip` write access to install
the built wheel into (the system/Homebrew Python is externally-managed and
will refuse). One-time setup:

```sh
cd ionp
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip numpy
```

## Build

Rust workspace only (every crate in the workspace):

```sh
cd ionp
cargo build --workspace
```

This requires `.cargo/config.toml` (already committed) which passes
`-undefined dynamic_lookup` to the linker on macOS. Without it, plain
`cargo build`/`cargo test` fails to link `ionp-py` — a pyo3
`extension-module` cdylib is deliberately *not* linked against libpython
(it resolves those symbols from the host Python process at `dlopen` time
instead), so macOS `ld` sees the Py_* symbols as undefined at build-link
time unless told to defer resolution. maturin's own build path sets this
flag internally; plain `cargo build` does not unless this config file tells
it to. Reference: https://pyo3.rs/main/building-and-distribution#macos

Python-importable extension module (`_anionpy`, wired into the `anionpy` package
via pyproject.toml's `module-name = "anionpy._anionpy"`):

```sh
cd ionp
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"   # if maturin isn't already on PATH
maturin develop --release
```

`maturin develop` builds the `ionp-py` crate (manifest-path in
`pyproject.toml` points at `ionp-py/Cargo.toml`) and installs the wheel
editable into the active venv.

## Test

Rust unit tests (all crates):

```sh
cd ionp
cargo test --workspace
```

Python-level differential tests live in `tests/differential/` (empty
scaffold at this stage — populated starting ordering step 1).

## Rebuilding after a Rust change

```sh
cd ionp
source .venv/bin/activate
maturin develop --release
```

Re-run for every change to any crate in the workspace — `maturin develop`
recompiles the whole dependency chain feeding `ionp-py` and reinstalls the
wheel. There is no
separate "just relink" shortcut; the compile is what's slow (~7-25s clean,
faster incremental), not the install.

## Measurement provenance: a suite result is not attributable to a commit

The installed `.so` is **shared mutable state**. Several agents and sessions may
work in this tree at once, and any of them can rebuild at any moment. A suite run
that overlaps someone else's `maturin develop` measures a binary corresponding to
no commit at all — and it does so silently, producing a plausible-looking JSON
full of verdicts that belong to nothing.

This is not hypothetical. On 2026-08-07 a run recorded as "measured at `fd254cd`,
clean tree" reported `ma.allequal` as a failing declared item, which blocks the
Definition of Done outright. It cost roughly an agent-hour. The item was never
broken: it passed in isolation, it passed under an explicit order-dependence
discriminator (full suite then isolated re-check inside one interpreter), and it
passed a fresh rebuild-and-rerun. The result was real; the attribution was not.

Note the failure class. It is the `conj`/strides lesson one level up. There, a
corpus was evidence only about the axis it happened to look at. Here, a
**measurement was evidence only about a binary it never identified**. Both feel
like data and neither is, until you name what was actually observed.

So, before trusting any suite number:

1. `git status --porcelain` — confirm the tree is clean, or that only your own
   files are modified.
2. Rebuild, and see the literal `🛠 Installed anionpy` line. A failed build
   silently leaves the previous `.so` in place.
3. Serialize through the lock, for the **suite run as well as the build** —
   a reader racing a writer is the whole problem:
   ```sh
   flock /tmp/ionp-build.lock ./.venv/bin/maturin develop --release
   flock /tmp/ionp-build.lock ./.venv/bin/python tests/differential/run.py --out /tmp/r.json
   ```
4. Record the commit hash *with* the numbers. "24.393%" is not a fact;
   "24.393% at `f32edc9`, built and observed" is.

Do **not** reason from an mtime. That was already insufficient — maturin skips
the copy when wheel content is identical — and with concurrent writers it is
worse than useless.

The durable fix, not yet implemented: have `run.py` stamp the git hash and a
fingerprint of the loaded extension into its own output, so a result carries its
provenance instead of depending on the operator's memory of what they built.

## Verifying the seam (what step 3 of the task proved)

```sh
cd ionp
source .venv/bin/activate
python3 -c "
import numpy as np
import anionpy
x = np.arange(10.0)
print('anionpy.sum_f64(x) =', anionpy.sum_f64(x))
print('np.sum(x)       =', x.sum())
"
```

Expected/actual output (captured 2026-07-31):

```
anionpy.sum_f64(x) = 45.0
np.sum(x)       = 45.0
```

The addition happens in `ionp-core::sum_f64` (plain Rust `.iter().sum()`);
`ionp-py::sum_f64` only marshals the `PyReadonlyArray1<f64>` across FFI;
`anionpy/__init__.py` only re-exports. No Python arithmetic anywhere in the
path — confirmed by reading the three files, not by convention.

## Coverage ledger

```sh
cd ionp
source .venv/bin/activate
python3 tools/coverage.py
```

At this stage (toolchain foundation only, zero feature work) this correctly
reports 0/1169 (0.0%), all in state `absent`. That is the honest number for
where the project stands right now — see GOAL-ionp.md for the ordering
that closes it out.
