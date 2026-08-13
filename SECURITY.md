# Security Policy

`anionpy` is a NumPy-compatible array library with a native Rust core. It
parses and processes numeric buffers, dtype descriptors, and array metadata
supplied by the caller — memory-safety and denial-of-service issues in that
path are taken seriously.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a suspected security
vulnerability. Instead, report it privately using GitHub's
[private vulnerability reporting](../../security/advisories/new) feature on
this repository ("Security" tab → "Report a vulnerability").

Include, where you can:

- the affected version/commit,
- a minimal reproduction (input that triggers the issue),
- what you observed (crash, UB, memory disclosure, hang, etc.) and, if known,
  why you believe it's security-relevant rather than an ordinary bug.

## Scope

In scope:

- Memory safety issues in the Rust core (`ionp-core`, `ionp-ion`, `ionp-py`)
  reachable from Python-level `anionpy` calls, including via `unsafe` blocks
  or FFI boundaries.
- Panics or crashes triggerable by untrusted array data, dtype strings, or
  shape/stride combinations that should instead produce an ordinary Python
  exception.
- Integer overflow or out-of-bounds access in buffer/stride arithmetic.

Out of scope / not a vulnerability by itself:

- A numerical result differing from NumPy's (that's a correctness bug — file
  it as a normal issue, and see `KNOWN-DIFFERENCES.md` for the standing list).
- Resource exhaustion from a caller deliberately requesting an enormous
  allocation (`anionpy.zeros(10**18)` running out of memory is expected
  behavior, the same as it is in NumPy).

## Response

This is a small project without a dedicated security team or a fixed SLA.
Reports will be acknowledged as promptly as possible, and a fix or mitigation
will be prioritized once a report is confirmed. Credit will be offered in the
fix's changelog entry unless you ask not to be named.
