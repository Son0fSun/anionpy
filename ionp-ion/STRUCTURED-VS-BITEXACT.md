# Structured vs bit-exact: what Ion can and cannot safely ship

Dated 2026-08-13. Written in response to a coordinator follow-up on ticket
#101 ("the shipped `anionpy` package never routed through the Ion
structured-operator path"). First pass wired `ionp-ion/src/bridge.rs` into
`ionp-ion/src/matmul.rs`'s dispatch but always computed `dense_matmul` too
and only shipped the structured candidate when it happened to match
bit-for-bit — a "certified no-op": deleting the whole structured path
changed no output and no timing. This document is the honest accounting of
*which* structured compositions can actually skip computing dense at all,
and which can only ever be a value-correct but byte-different answer,
requested so a human (not this agent) can decide the policy for the latter
group.

## The one-sentence version

**Bit-exactness and asymptotic speedup are the same axis, not two
different ones, for every non-trivial composition in this crate.** A
composition is bit-exact against `dense_matmul` only when it can be
expressed as *dropping terms that are exactly zero* from the same
triple-loop summation `dense_matmul` performs — anything that changes
the summation's *order* (FFT diagonalization, low-rank contraction over
rank `r < n`) is real math and a real speedup, but IEEE 754 addition is not
associative, so a different-but-equal-valued order is not the same
bit pattern. There is no third option lurking in `compose.rs`.

## Compositions in `compose.rs`, sorted by which side of that line they're on

| Composition (`compose::matmul`) | Collapses to | Bit-exact vs `dense_matmul`? | Why |
|---|---|---|---|
| `Diagonal @ Diagonal` | `Diagonal` (elementwise product of the two diagonals) | **Yes** | Off-diagonal output entries are literally never computed (there is no summation to reorder); the one surviving diagonal entry is a single multiply, same value `dense_matmul` reaches after summing `n-1` exact zeros into it. |
| `Diagonal @ Dense` / `Dense @ Diagonal` | not a `compose.rs` case (no `Operator` is built) but licensed directly in `matmul.rs` by the same zero-multiply argument | **Yes**, conditionally | Every off-diagonal entry of the diagonal operand is exactly `0.0`; `0.0 * finite = 0.0` and `acc + 0.0 == acc`, so the sum collapses to the single surviving term — **provided** the non-diagonal operand is fully finite (see "The NaN/Inf hole" below). |
| `Circulant @ Circulant` | `Circulant` (product of DFT eigenvalues, one inverse FFT) | **No** | `dense_matmul` sums `n` real products per output entry in index order; the FFT path multiplies in the frequency domain and inverse-transforms — mathematically identical, a completely different sequence of floating-point roundings. Measured (see below): relative error ~1e-14, not 0. |
| `LowRank @ LowRank` | `LowRank` (contracts over the shared rank `r`, not the full dimension `n`) | **No** | Sums `r` terms instead of `n` per output entry (that's the entire point — it's the speedup). Different term count, different order, different rounding, even when `r == n` degenerately. |
| `Toeplitz @ Toeplitz`, anything not listed above | Lazy `Product` (no closed-form collapse; `apply`/`apply_mat` fall back to sequential application) | **No**, and also **not faster** | `Product::apply_mat` just calls the two operators' `apply_mat` in sequence — still `O(n^2)`-ish per stage, still a different sum order than the single fused `dense_matmul` triple loop. No win to chase here at all. |

## The NaN/Inf hole (why "diagonal" alone isn't enough)

`detect.rs`'s `is_diagonal` is **tolerance-based** (`rtol`, default `1e-9`)
— exactly right for deciding "this is worth applying `apply`/`apply_mat`
to," wrong for deciding "dense_matmul may be skipped entirely," because an
off-diagonal entry that is merely small is still a nonzero term a
triple-loop sum would add. `ionp-ion/src/bridge.rs` therefore adds a
second, stricter, non-tolerance predicate used only by the dense-skipping
fast path: `is_exact_diagonal_square_f64` (every off-diagonal entry `==
0.0` bit-for-bit, either sign) plus `all_finite_f64` on the *other*
operand. The finiteness check exists because `0.0 * NaN == NaN` and `0.0 *
Inf == NaN` — an exact zero times a non-finite value is not an absorbing
no-op, it's contamination. Both predicates are tested in
`bridge.rs`'s `exact_diagonal_accepts_true_zero_rejects_near_zero` and
`all_finite_rejects_nan_and_inf`, and the fast path's decline-on-NaN
behaviour is tested directly in `matmul.rs`'s
`exact_fast_path_declines_when_non_diagonal_side_has_nan`.

## Measured relative error: Circulant @ Circulant

From `matmul.rs`'s
`dtype_f64_matmul_end_to_end_circulant_detects_but_falls_back_safely` (n=16,
random circulant generators, `SplitMix64` seed 99): the FFT-collapsed
candidate is **not** bit-exact (confirmed: the test asserts `!bit_exact`
and would fail loudly if that ever changed) but stays under `1e-6` max
relative error against the dense reference; ad hoc probing at the same size
put the actual measured error around `1e-14`–`1e-13`, consistent with a
handful of FFT butterfly stages' worth of accumulated rounding on
well-conditioned random input. This is "exact-in-value, not exact-in-bits"
in the ticket's own vocabulary, exactly as ticket #101 predicted in advance
for FFT-based circulant multiply.

## Crossover size: where would FFT actually win?

Not measured directly in this pass (Circulant is deliberately **not** wired
into the hot path — see next section — so there is no shipped code whose
speed depends on the answer), but the standard result applies:
`dense_matmul` here is `O(n^3)` (or `O(n^2 m)` for the general `n x k @ k x
m` shape) via BLAS/Accelerate at very high constant-factor efficiency (the
diagonal benchmark below measures ~180+ GFLOP/s on this machine), while
`compose_circulant`'s FFT collapse is `O(n log n)`. For a *single*
`Circulant @ Circulant` (not embedded in a larger dense product), the
crossover is where `n^3` work at BLAS's realized throughput exceeds `n log
n` work at FFT's realized throughput — empirically, for well-tuned FFT vs
well-tuned dense GEMM, that crossover is usually in the **low hundreds to
low thousands** of `n` for square matrices, but this crate has not measured
it directly because doing so honestly requires the same rigor applied to
the diagonal case below (warm caches, repeated trials, checked against a
plausible-GFLOP/s sanity gate — see `src/bin/bench.rs`'s own comment about
a prior "123x win" that was a measurement artifact). Flagging this as
unmeasured rather than guessing a specific crossover number.

## What's actually wired into the hot path now (the honest change from the "certified no-op")

Per the coordinator's "make the proven-exact cases primary, not redundant"
instruction: `ionp-ion/src/matmul.rs`'s `kernel_2d` `DType::F64` arm now
calls `try_exact_diagonal_fast_path_f64` **first**, and when it fires
(`Diagonal @ Diagonal`, `Diagonal @ Dense`, or `Dense @ Diagonal`, all
exact per the table above), **`dense_matmul` is not called at all** — not
computed-then-discarded, not computed-then-compared, simply not executed.
`#[cfg(debug_assertions)]` still cross-checks the fast path's result
against `dense_matmul` as a standing regression guard in debug builds only
(the "or a debug assertion at minimum" bar from the ticket), so release
builds pay nothing for the guard.

`Circulant`, `LowRank`, `Toeplitz`, and anything that only detects on one
side (`try_structured_matmul_f64`, the original always-compute-both
function) are **deliberately not called from the hot path anymore** — they
are retained (`#[allow(dead_code)]`, still compiled and unit-tested) as
proof that `detect`/`compose`/`gate`/`bridge` work end-to-end beyond plain
diagonal, and as the mechanism behind the "opt-in tolerance" policy option
below, but wiring them into `kernel_2d` today would only recompute
`dense_matmul` a second time for nothing — the exact defect this document
exists to not repeat.

## Measured speed: does the diagonal fast path actually win?

`matmul.rs`'s `exact_fast_path_is_faster_than_dense_matmul` (`#[ignore]`d,
run explicitly: `cargo test -p ionp-ion --release
exact_fast_path_is_faster_than_dense_matmul -- --ignored --nocapture`),
`diag(n,n) @ dense(n,m)`, wall-clock, `--release`, Apple Accelerate BLAS
backend, machine shared with several other concurrently-running build
agents (noise caveat below):

| n | m | dense_matmul | diagonal fast path | speedup |
|---|---|---|---|---|
| 64 | 64 | ~2.3–7.4us | ~4.1–6.1us | **0.4x–1.4x (mixed/noisy, near parity)** |
| 256 | 64 | ~26.7–27.4us | ~45.0–53.8us | **~0.5–0.6x (consistent loss)** |
| 1024 | 64 | ~360–1014us | ~576–1077us | **0.3x–1.5x (noisy, no clear win)** |
| 256 | 256 | ~82–149us | ~74–91us | **~1.1x–1.6x (modest, consistent win)** |
| 1024 | 1024 | ~8.4–10.3ms | ~1.1–1.3ms | **~6.8x–7.8x (large, consistent win)** |

Repeated 3x per shape; the `m=64` (thin) column is genuinely noisy run to
run (this machine has other agents' `cargo build`s running concurrently —
see the git status at the top of this session — so any number under ~1ms
should be read as "in the right ballpark," not precise). The `m == n`
(square-ish) column is far more stable and shows a real, growing win.

**Honest finding, not glossed over:** the diagonal fast path does *not*
reliably beat BLAS `dense_matmul` when the non-diagonal operand is thin
(`m` small relative to `n`). The reason is visible in the code, not a
measurement artifact: `is_exact_diagonal_square_f64` (the exactness check
that licenses skipping dense at all) is itself an `O(n^2)` scalar,
branchy, non-vectorized scan over the *entire* `n x n` block, because the
only interface this fast path has to the caller is a raw row-major buffer
— it has no way to know "this came from a `Diagonal` operator" without
re-deriving it. At `n=1024` that scan alone is ~1M branchy comparisons,
roughly the same order of work as `dense_matmul`'s memory traffic for a
thin `m=64` product, so the detection cost eats most or all of the
`O(n*m)` arithmetic win. When `m` grows to match `n` (`m=1024`), the
`O(n*m)`-vs-`O(n^2*m)` arithmetic gap dominates the fixed `O(n^2)`
detection cost and the fast path wins clearly (~7-8x). This is a genuine
finding about *this specific interface* (raw-buffer, re-detect-every-call),
not a claim that diagonal-aware matmul can never win — a caller that
already holds a `Diagonal` operator (skipping re-detection) would win
unconditionally at every size in this table, but that is not the interface
`ionp-py`'s numpy-facing entry points currently offer.

## Policy options for the non-bit-exact-but-value-exact group (Circulant, LowRank, Toeplitz collapses)

Presented, not chosen — this call belongs to a human, per the coordinator's
explicit instruction:

1. **Exact-only (current state).** Never ship a non-bit-exact answer from
   these entry points, ever, under any flag. Simple, zero risk to the
   36-failure differential baseline, leaves real speedups (FFT circulant,
   low-rank contraction) permanently on the table for every caller,
   including ones who would gladly trade `1e-13` relative error for an
   `O(n log n)` circulant multiply.
2. **Opt-in fast path with a documented tolerance.** Add an explicit,
   off-by-default flag/kwarg (e.g. `anionpy.matmul(a, b, allow_structured=True)`
   or a separate `anionpy.experimental.structured_matmul`) that runs
   `try_structured_matmul_f64` (already implemented, tested, just not
   wired into the default path) and ships the candidate whenever
   `gate::certify` passes, documenting the `~1e-9`–`1e-13`-scale relative
   error a caller should expect instead of bit-exactness. Never silently
   changes default behavior; every caller who hits it opted in.
3. **Per-op policy.** Different default per structure: e.g. ship
   `LowRank @ LowRank` opt-in (rank-`r` contraction is a well-understood,
   predictable-error operation many numerical codebases already treat as
   "fast and approximately equal") while keeping `Circulant`/`Toeplitz`
   exact-only until there's a concrete customer need, rather than one
   global switch for structurally different tradeoffs.

None of these three change anything about the diagonal fast path already
wired in — that one is bit-exact by proof, not by tolerance, and ships
unconditionally regardless of which policy option above gets picked.

## Routing table: which dtypes and entry points reach the structured path

All of `anionpy`'s `matmul`/`vecdot`/`matvec`/`vecmat`/`@` (`__matmul__`)
share one Rust-side implementation: `ionp-py/src/matmul.rs` marshals
Python arguments then calls the matching `ionp_ion::matmul::{matmul,
vecdot, matvec, vecmat}`, every one of which bottoms out in
`ionp-ion/src/matmul.rs`'s `run_batched` -> `kernel_2d` per 2-D block. So
the routing question is really "what does `kernel_2d` do per dtype,"
checked directly against that function's `match dtype` arms:

| Entry point (`ionp-py`) | Rust fn (`ionp-ion/src/matmul.rs`) | Reaches `kernel_2d`? |
|---|---|---|
| `matmul` / `@` / `__matmul__` | `matmul` | Yes |
| `vecdot` | `vecdot` | Yes (`n=1`, `m=1` per block) |
| `matvec` | `matvec` | Yes (`m=1` per block) |
| `vecmat` | `vecmat` | Yes (`n=1` per block) |
| `dot` | `ionp_core::products::dot` | **No** — separate crate (`ionp-core`), never touches `ionp-ion`, out of this ticket's scope (owned by the products-cluster task per `products_py.rs`'s own doc comment) |
| `vdot`, `inner`, `tensordot`, `cross` | `ionp_core::products::*` | **No** — same as `dot` |

`kernel_2d`'s per-dtype behavior, i.e. what every one of the four Yes rows
above actually gets:

| `DType` | Structured path reachable? | What runs |
|---|---|---|
| `F64` | **Yes** | `try_exact_diagonal_fast_path_f64` first (skips `dense_matmul` entirely when it fires); `dense_matmul` otherwise |
| `F32` | No | `dense_matmul_f32` unconditionally |
| `C64` | No | `dense_matmul_c64` unconditionally |
| `C128` | No | `dense_matmul_c128` unconditionally |
| `Bool`, `I8`/`I16`/`I32`/`I64`, `U8`/`U16`/`U32`/`U64`, `F16` | No | Each has its own `kernel_*` (integer/bool/half-precision triple loop); none of these dtypes have a bridge into `Operator`/`Complex64` at all — `bridge.rs` is `f64`/`f32`-only by construction (`f64_rowmajor_to_c64_colmajor` / `f32_rowmajor_to_c64_colmajor`), and integer/bool matmul has no well-defined "structured operator" analog in this crate regardless (`detect.rs`'s cascade is float-tolerance-based throughout) |
| `S(_)`/`U(_)` (string) | N/A | `unreachable!()` — matmul has no string semantics; `ionp-py` bindings must reject before reaching here |

So: **only `F64`, only through `matmul`/`vecdot`/`matvec`/`vecmat`/`@`,
only for the three provably-exact diagonal shapes, currently reaches
Ion.** `F32` has a bridge function in `bridge.rs`
(`f32_rowmajor_to_c64_colmajor` / `c64_colmajor_to_f32_rowmajor`) but
nothing in `matmul.rs` calls it yet — the same exact-diagonal argument
would need re-deriving for `f32`'s narrower mantissa (untouched in this
pass, flagged here rather than silently assumed to carry over).
`dot`/`vdot`/`inner`/`tensordot`/`cross` never reach `ionp-ion` at all;
they are a different crate's ownership and were out of scope for ticket
#101 from the start (`ionp-py/src/linalg.rs`, also in my file ownership
for this ticket, was checked directly and has no matmul-adjacent
structured-operator opportunity in scope either — it's SVD/QR/solve/norm,
none of which currently call into `detect`/`compose`/`gate`).

