# Phase 0 decisions (public record)

**Date:** 2026-08-06 (updated for public export 2026-08-13)

Private IP review notes are not published.

## Decided

| # | Topic | Decision |
|---|--------|----------|
| 1 | Public package name | `anionpy` |
| 2 | License | MIT |
| 3 | NumPy at runtime | Optional for core import; optional `compat` extra |
| 4 | `UFuncTypeError` | Subclass plain `TypeError` |
| 5 | Research / non-product crates | Not included in this repository |
| 6 | History | Fresh public repository; no private monorepo history |

## Standing rule

A decision recorded is not proof of tree state. Re-run tests and
`tools/check_public_export.py` before release claims.
