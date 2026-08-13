# Public scope

This repository is the **anionpy** project: a NumPy-compatible array library
with a native Rust core.

## Included

- `anionpy/` — Python package
- `ionp-core/` — Rust array core
- `ionp-py/` — PyO3 extension (`anionpy._anionpy`)
- `ionp-ion/` — structured linear-algebra operators used by the product path
- `tests/`, `tools/`, product docs, MIT license

## Not included

- Private monorepo history from the development workspace
- Internal research crates that are not dependencies of `ionp-py`
- Host-specific infrastructure notes and private IP assessments

## Publishing rule

Public releases are produced by copying this ship set into a **new** git
repository (history starts at commit one). Do not republish a filtered clone
of a private parent workspace.
