# Products / linalg cluster — ground truth

Measured 2026-08-02 against binary mtime **22:33:46** (stable across every run
below; checked before and after). Operand parity asserted on every comparison:
same shape, same strides, same `tobytes()` on both sides before any result is
compared.

## Near-free items: correct Rust that is simply not wired up

Three functions are fully implemented, measured GREEN across float64/float32/
int64/complex128, and reachable **only** as `ionp.linalg.X`. numpy exposes them
at top level too. Verified directly:

| name | top level | under `linalg` | `__ion_state__` |
|---|---|---|---|
| `outer` | **no** | yes | `linalg.outer: exact` |
| `tensordot` | **no** | yes | **undeclared either way** |
| `cross` | **no** | yes | **undeclared either way** |
| `matmul` | yes | yes | `linalg.matmul: exact` |
| `vecdot` | yes | yes | `linalg.vecdot: exact` |

So there are **up to 5 items available for close to no numeric work**: three
top-level re-exports (one `m.add_function(...)` line each in
`ionp-py/src/lib.rs`, following the existing `matmul`/`vecdot` pattern), plus
`tensordot` and `cross` which are already correct under `linalg` and simply
never declared. Each still needs a differential case before it counts —
undeclared-but-working is not coverage, it is an unwritten test.

`tensordot` note: it returns an ndarray even for a full contraction
(`axes=2`), which matches real numpy — numpy's `tensordot` does **not** return
a scalar there, unlike `sum`/`vecdot`/`dot` on 1-d. So it does not inherit the
scalar-return defect. Do not "fix" it into a scalar.

## `diff` — the withdrawal comment in `_state/toplevel.py` is STALE

The commented-out declaration says `diff` was withdrawn because scalar
`prepend=`/`append=` panic across the Rust FFI boundary. **That is no longer
true on this binary.** Measured, 12 cases:

plain 1-d, `n=2`, scalar `prepend`, scalar `append`, both scalars together,
array-form `prepend`, `n=2`+`prepend`, `n=0`, float32 `prepend`, complex
`prepend`, bool input — **11 of 12 agree with numpy exactly on values AND
strides. No panic anywhere.**

This matters in its own right: a stale comment holding down a working item is a
**false negative** in the ledger — the exact mirror image of the sixteen false
positives revoked this morning. Both come from trusting a written claim over a
measurement.

`diff` nonetheless stays withdrawn, for two defects the comment does not
mention:

1. **Empty input strides.** `diff(array([], int64))` → numpy strides `(0,)`,
   ionp `(8,)`.
2. **F-order output layout.** `diff` on an F-contiguous `(4,3)` along `axis=0`:
   numpy returns strides `(8,24)`, `F_CONTIGUOUS=True`; ionp returns `(24,8)`,
   `F_CONTIGUOUS=False`. Values identical.

**Defect 2 is the important one, and it is not local to `diff`.** `diff` is a
shape-changing op, not a reduction. The C-contiguous-output-forcing behaviour
was previously understood as a *reduction* layout bug; it reaches here too. The
root cause is general output-layout selection, and the fix should be scoped to
that, not patched per-function.

**Defect 1 is also not local.** The shape-cluster re-probe independently found
`reshape` of a size-0 1-d array giving numpy `(0,)` vs ionp `(itemsize,)`.
Empty-array stride convention is a third cross-cutting theme in its own right.

## The rest of the cluster — sized by reading the Rust, not the numpy docs

**Dispatchers over primitives that already exist** (`ionp_ion::matmul::matmul`,
`ionp_ion::matmul::vecdot`, `linalg::tensordot`):
- `dot` — 1-d·1-d → `vecdot`, 2-d·2-d → `matmul`, N-d → `tensordot(axes=([-1],[-2]))`
- `vdot` — `vecdot(ravel(a), ravel(b))`
- `inner` — `tensordot(a, b, axes=([-1],[-1]))`

All three will surface the 0-d-vs-scalar full-reduction defect on their 1-d
paths. That is the concurrent root-cause fix's job, not a reason to block them.

**Thin wrappers over the existing `cumsum`/`cumprod` accumulate kernel:**
`cumulative_sum`, `cumulative_prod` — the Array-API names, plus an
`include_initial` kwarg and a mandatory axis for ndim>1. (Unverified: whether
`cumsum`/`cumprod`'s own output layout is correct — these would inherit it.)

**Genuinely new kernels, moderate:** `interp` (can build on the existing
`searchsorted`), `convolve`/`correlate` (one shared sliding-window kernel powers
both; FFT is NOT a shortcut here — numpy's are direct, and an FFT round-trip
would not reproduce them bit-for-bit), `kron`, `trapezoid`, `gradient`.

**`einsum` + `einsum_path` — one multi-week subsystem, not two items.** Nothing
exists: no subscript parser, no N-ary contraction executor, no path costing.
`einsum_path` is not independently buildable — it is path analysis over the very
machinery `einsum` needs, so building it first means either duplicating the
parser or shipping something that cannot be validated against a working
`einsum`. Lowest yield per unit work in the cluster by a wide margin.

## Recommended order

1. Wire `outer`/`tensordot`/`cross` to top level; declare `tensordot`/`cross`. Up to 5 items, no new math.
2. `dot`/`vdot`/`inner` as dispatchers. 3 items, no new math.
3. `cumulative_sum`/`cumulative_prod`. 2 items, thin.
4. The general output-layout fix (serves `diff`, the reductions, and probably more) + empty-array stride convention.
5. `interp`, then `convolve`+`correlate` as a pair, then `kron`, `trapezoid`, `gradient`.
6. `einsum`/`einsum_path` last, scoped and budgeted as a project.

## Explicitly NOT measured — do not read absence as a pass

- `bool` dtype and 0-d inputs on `outer`/`tensordot`/`cross`.
- `diff` on 3-d, 0-d, and **negative-stride** inputs.
- `cumsum`/`cumprod` output strides/layout.
- No item here was prototyped or compiled; all sizing is from reading Rust
  signatures and call sites under a read-only constraint.
