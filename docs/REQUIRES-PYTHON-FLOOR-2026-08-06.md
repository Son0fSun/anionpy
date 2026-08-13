# requires-python floor: measured, not assumed (2026-08-06)

## Question

`pyproject.toml` declares `requires-python = ">=3.14"`. The release plan asked
whether that is a real constraint or an accident, before anyone considers
widening it. This doc answers that with a measured/inferred/unverified split.
**No file outside this doc was edited.** `pyproject.toml` is untouched, per
instruction — this is a recommendation, not an enactment.

## Bottom line

**3.14 is not a technical constraint from pyo3, rust-numpy, or the Python
surface. It is an environmental accident: this machine only has Python 3.14
installed, and the abi3 feature was set to match the one interpreter available
at the time, not derived from a requirement.** The recorded justification in
`DEVELOPING.md` ("pyo3 0.29.0 — this is the first pyo3 release with standard
abi3 support for Python 3.14") is MEASURED FALSE — see below. Nothing in the
Rust dependency graph or the Python package source requires 3.14 or even
3.11+. The floor can very likely be lowered; the honest caveat is that nobody
has built or run a lower-abi3 wheel to confirm it, because no interpreter
below 3.14 exists on this box (MEASURED, see §5).

---

## 1. What the abi3 floor means (MEASURED)

`ionp-py/Cargo.toml`:
```
pyo3 = { version = "0.29.0", features = ["extension-module", "abi3-py314", "multiple-pymethods"] }
numpy = { version = "0.29.0", features = ["half"] }
```
`Cargo.lock` confirms the resolved versions actually built against:
```
pyo3            0.29.0
numpy           0.29.0
pyo3-build-config 0.29.0
```
pyo3 0.29.0's own `Cargo.toml` (read from
`~/.cargo/registry/src/.../pyo3-0.29.0/Cargo.toml`) defines the abi3 feature
family as a strictly cascading ladder:
```
abi3-py38  = ["abi3-py39",  "pyo3-ffi/abi3-py38"]
abi3-py39  = ["abi3-py310", "pyo3-ffi/abi3-py39"]
abi3-py310 = ["abi3-py311", "pyo3-ffi/abi3-py310"]
abi3-py311 = ["abi3-py312", "pyo3-ffi/abi3-py311"]
abi3-py312 = ["abi3-py313", "pyo3-ffi/abi3-py312"]
abi3-py313 = ["abi3-py314", "pyo3-ffi/abi3-py313"]
abi3-py314 = ["abi3-py315", "pyo3-ffi/abi3-py314"]
abi3-py315 = ["abi3",       "pyo3-ffi/abi3-py315"]
```
This confirms the standard abi3 contract by construction: selecting
`abi3-pyX.Y` builds a wheel against the stable ABI **starting at** X.Y —
forward-compatible with X.Y and later, and it will not load on anything
earlier (the earlier interpreter's stable-ABI struct layout doesn't have the
symbols the built extension expects). **The declared feature name IS the
runtime floor**, not just a build-time flag. So `abi3-py314` in
`ionp-py/Cargo.toml` is the literal source of the 3.14 floor — it is not
inherited from anywhere else in the dependency graph (confirmed in §3).

## 2. Is anything in the Python surface 3.14-specific? (MEASURED)

Searched all of `ionp/**/*.py` (21,986 lines across 28 files) for:
- PEP 695 syntax (`class Foo[T]`, `def f[T]`, `type X = ...`) — **none found**
- `except*` (PEP 654, 3.11+) — **none found**
- `tomllib`, `ExceptionGroup`/`BaseExceptionGroup`, `zoneinfo`,
  `contextlib.chdir`, `itertools.batched`, `pathlib.Path.walk`, `StrEnum` —
  **none found**
- Newer `typing` members (`Self`, `override`, `TypeVarTuple`, `Unpack`,
  `Never`, `LiteralString`, `Required`/`NotRequired`, `dataclass_transform`) —
  **none found** (there is in fact no `import typing` / `from typing import`
  anywhere in the tree at all)
- `sys.version_info` / `__future__` gating — **found one real hit**:
  `ionp/testing.py:148`:
  ```python
  if sys.version_info < (3, 13):
      ...  # manual direct_url.json parsing
  else:
      IS_EDITABLE = _ionp_dist.origin.dir_info.editable
  ```
  This is a runtime branch that **actively supports Python versions below
  3.13** — the code was written expecting to run on interpreters the current
  `requires-python` floor now forbids. This is direct textual evidence inside
  the tree that the 3.14 floor is inconsistent with the code's own design,
  not a deliberate 3.14-only decision.
- All 12 files using `from __future__ import annotations` — this is
  standard (available since 3.7) and imposes no floor of its own.

**Conclusion: nothing in the Python source requires 3.14, or even anything
newer than 3.9.** `sys.version_info < (3, 13)` implies the author anticipated
running on <3.13.

## 3. What would lowering the abi3 feature cost? (MEASURED + one INFERRED item)

- pyo3 0.29.0 supports abi3 floors from `abi3-py38` through `abi3-py315`
  (§1) — MEASURED from its `Cargo.toml`.
- rust-numpy (`numpy` crate) 0.29.0's own `Cargo.toml`
  (`~/.cargo/registry/src/.../numpy-0.29.0/Cargo.toml`) depends on pyo3 with
  `default-features = false, features = ["macros"]` and defines **no abi3
  feature of its own** — it does not gate or influence the abi3 floor at all.
  The floor is controlled entirely by `ionp-py`'s own `pyo3` dependency line.
  MEASURED.
- Searched `ionp-py/src/**/*.rs` (34,597 lines) for any CPython-3.14-only FFI
  usage (`immutable_type`, `PyLong_Import`/`Export` (PEP 757, 3.14+),
  `PyIter_NextItem`, `PyImport_ImportModuleAttr`) that pyo3's own changelog
  flags as 3.14-gated — **none found**. MEASURED (grep, read-only, no `.rs`
  file touched or built).
- **DEVELOPING.md's stated justification is measured false.** It says: "pyo3
  0.29.0 — this is the first pyo3 release with standard abi3 support for
  Python 3.14 (abi3-py314 feature)." Checked pyo3's `Cargo.toml` at tags
  `v0.28.0` and `v0.27.0` directly from GitHub (`raw.githubusercontent.com`):
  both already define `abi3-py314 = ["abi3", "pyo3-build-config/abi3-py314",
  "pyo3-ffi/abi3-py314"]`, cascading the same way. The `abi3-py314` feature
  is not new in 0.29.0; it predates it by at least two minor versions. This
  is a MEASURED correction of a recorded-but-unverified claim, matching
  exactly the failure mode the task brief warned about ("a recorded reason in
  this repo is not evidence about the current tree").
- INFERRED (not independently built/tested): given pyo3 supports
  `abi3-py311`, `abi3-py312`, `abi3-py313` as ordinary cascade members
  identical in structure to `abi3-py314`, and rust-numpy imposes no
  additional floor, lowering `ionp-py/Cargo.toml`'s feature to e.g.
  `abi3-py312` or `abi3-py311` should build and produce a wheel loadable on
  that version and above. This was **not built** — see §5 for why, and do
  not read this paragraph as a test result.

## 4. Does numpy's own floor bind first? (MEASURED)

`numpy` is a true optional runtime dependency (`[project.optional-dependencies]
compat = ["numpy>=2"]`); core `ionp` imports with no numpy installed at all
(per `pyproject.toml` comment and `ionp-py/src/errors.rs` reference — not
re-verified by running, since that would require a build; taken from the
`pyproject.toml` comment as read, not executed).

Checked PyPI's `requires_python` metadata directly for numpy releases
(`curl https://pypi.org/pypi/numpy/<version>/json`, MEASURED, live network
call, not memory):

| numpy version | `requires_python` |
|---|---|
| 2.0.0 | `>=3.9` |
| 2.1.0 | `>=3.10` |
| 2.2.0 | `>=3.10` |
| 2.3.0 | `>=3.11` |
| 2.5.1 (the compat target named in this repo) | `>=3.12` |

So numpy's *own* current-release floor (3.12) is higher than pyo3's minimum
supported abi3 floor (3.8) but lower than ionp's current declared floor
(3.14). Two consequences:

- If `ionp`'s `requires-python` were lowered to, say, 3.11 or 3.12, a user on
  that interpreter installing the `compat` extra would get `numpy>=2`
  resolved by pip to whatever numpy release still supports their
  interpreter (e.g. numpy 2.3.x on 3.11, not 2.5.1) — this works today for
  any package with an unpinned floor; it is how the ecosystem functions, not
  something special to ionp. **This is INFERRED from pip's standard
  resolution behavior, not tested against a real numpy install on <3.14.**
- numpy's floor does **not** justify or explain ionp's 3.14 floor — numpy
  itself would tolerate 3.11 or 3.12 easily. If anything, numpy's floor is
  evidence *against* 3.14 being necessary: the most numpy-recent constraint
  in the whole dependency picture is 3.12, two full versions below what
  `ionp` currently demands.

## 5. What was NOT tested (stated plainly)

- **No wheel was built in this session at all** — building was explicitly
  out of scope (build lane held by another agent) and this task was
  completed by static inspection of `Cargo.toml`, `Cargo.lock`, vendored
  crate sources under `~/.cargo/registry`, PyPI JSON metadata, and grep over
  the Python and Rust source trees. No `.rs` file was edited, no `cargo
  build`/`maturin develop` was run, `target/` and `.venv` were not touched
  beyond read-only inspection (`.venv/bin/python -c "import numpy; print(...)"`
  to confirm the installed numpy version, and `.venv/bin/python --version`).
- **This machine has exactly one Python interpreter: 3.14.6**
  (`/opt/homebrew/bin/python3`, confirmed via `python3 --version` and
  `ls /opt/homebrew/bin`). There is no pyenv, no `/Library/Frameworks/
  Python.framework` alternate version, no cached 3.11/3.12/3.13 anywhere
  found via filesystem search. It is therefore **not possible on this box to
  build an `abi3-py311`/`py312`/`py313` wheel and actually load it on a
  matching lower interpreter** to confirm runtime compatibility. Any claim
  that lowering the floor "will work" is INFERRED from the abi3 contract and
  pyo3's declared feature support, not verified by running a lower-version
  wheel against a lower-version interpreter.
- Whether `import ionp` truly works with zero numpy installed (claimed by
  the `pyproject.toml` comment) was **not independently re-verified** in this
  session — no build occurred to test it.

## Recommendation

Lowering the abi3 floor below 3.14 is not blocked by any dependency version,
any pyo3/rust-numpy feature gate, or any syntax/API actually used in the
Python or Rust source — every one of those was checked directly against the
resolved `Cargo.lock` versions, vendored crate manifests, and PyPI metadata,
not assumed from memory. The one piece of internal evidence (`testing.py:148`)
actively anticipates running below 3.13. The documented rationale for having
picked `abi3-py314` in the first place (`DEVELOPING.md`) is measurably wrong
about pyo3 history and reads as "3.14 was the only interpreter on the build
machine," not "3.14 was required."

**Recommend widening** — a reasonable target is `abi3-py311` or `abi3-py312`
(numpy 2.5.1's own floor, so the `compat` extra's newest release stays
reachable on the widened floor too) — **contingent on actually building and
loading a wheel against a lower interpreter to confirm the abi3 forward-
compatibility claim in practice**, since that step could not be performed on
this machine (§5). This doc does not modify `pyproject.toml`; that edit and
the interpreter-matrix build/test to validate it are follow-up work for
whoever holds the build lane next.
