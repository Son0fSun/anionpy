# Phase 0 decisions (public record)

**Date:** 2026-08-06 (updated for public export 2026-08-13)

This file records product decisions that still apply to the published tree.
Private IP review notes are not published.

## Decided

| # | Topic | Decision |
|---|--------|----------|
| 1 | Public package name | `anionpy` |
| 2 | License | MIT |
| 3 | NumPy at runtime | Optional for core import; optional `compat` extra |
| 4 | `UFuncTypeError` | Subclass plain `TypeError` (not a private NumPy class) |
| 5 | Research / non-product crates | Not included in this repository |
| 6 | History | Fresh public repository; no private monorepo history |

## Rationale (short)

- **Package name:** outward-facing name is `anionpy`. Directory and Cargo crate prefixes may still say `ionp-*` for historical reasons; that is an open rename, not a second public package name.
- **MIT:** chosen deliberately; copyright holder is stated in `LICENSE`.
- **Optional NumPy:** `import anionpy` must work without NumPy installed. Exception: `char` / `strings` currently require NumPy (see `KNOWN-DIFFERENCES.md`).
- **UFuncTypeError:** public NumPy exception classes (`LinAlgError`, `AxisError`) are fair to align with when present; private NumPy modules are not a runtime dependency.
- **Scope of this repo:** only the NumPy-compatible array library and its test/tooling surface. Internal research operators that are not on the `ionp-py` dependency path are omitted.

## Not decided here

- Whether every Cargo crate renames off the `ionp-*` prefix.
- Whether `requires-python = ">=3.14"` widens after measurement.

## Standing rule

A decision recorded is not proof of tree state. Re-run tests and
`tools/check_public_export.py` before release claims.
