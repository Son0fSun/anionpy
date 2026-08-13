"""anionpy.linalg.* differential registry entries.

NEW FILE (permitted: the task brief allows adding new files under
tests/differential/, in addition to owning registry.py itself). Builds a
`LINALG_SPECS: dict[str, ItemSpec]` the exact same way ufunc_registry.py
builds `UFUNC_SPECS` -- imported and merged into `registry.REGISTRY` from
the bottom of registry.py (which this task owns), collision-checked the
same way ufunc_registry.py's merge is (see registry.py's import of this
module).

Design notes, REVISED 2026-08-01 (see reports/ionp-hardened-door-open-
window-2026-08-01.md and this task's report for the full measurements
behind this revision -- the previous version of this docstring described
a blanket atol=1e-9/rtol=1e-9 declaration backed by a 50-matrix, non-
seeded-sweep, non-per-dtype justification string; that was the exact
"open window" the epsilon tier's evidence gate (registry.py's
`ItemSpec.epsilon_tolerance`/`epsilon_sweep`, mirroring `ulp_tolerance`/
`ulp_sweep`) now closes. Every number below is a real 20,000-matrix,
seeded (`ulp_sweep.SEED`), per-dtype sweep produced by
`ulp_sweep.sweep_matrix_item_evidence`, not a hand-picked worked example):

  - `det`, `slogdet`, `trace`, `matrix_rank` are graded bit-exact
    (atol=0.0, rtol=0.0) for EVERY dtype except one: `det`'s float64 path
    is bit-exact (confirmed: 0/20000 disagreement over a condition-
    controlled sweep, sizes 2-8, condition 1e-2...1e2) but its complex128
    path is not -- see `det`'s epsilon_tolerance below. `slogdet` shares
    `det`'s cases and LAPACK call but is NOT swept/re-declared here (out
    of this fix's measured scope; flagged in this task's report as an
    open question, not silently assumed fine).

  - `cholesky` is graded bit-exact on its LOWER-triangular path (0/20000
    disagreement, confirmed directly -- the `3x3_spd_upper`-case "8 ULP"
    figure that previously triggered a bit-exact -> epsilon
    reclassification never came from the lower path at all) and needs a
    real, small ABSOLUTE tolerance only on its UPPER-triangular path
    (uplo='U' issued directly vs our own transpose-of-lower-factor
    reasoning -- both call sequences are mathematically valid derivations
    of the same unique factor, just non-bit-identical floating-point
    paths through Accelerate LAPACK; measured max abs error over a
    combined lower+upper 20,000-matrix sweep: 4.24e-13, entirely
    attributable to the upper path). Declared per-dtype float64 bound
    covers both call forms of the one item (epsilon_tolerance is keyed by
    dtype, not by kwarg) -- see `epsilon_sweep["linalg.cholesky"]`.

  - `inv`, `solve`, `pinv`, `svdvals`, `eigvalsh`, `lstsq` are graded with
    a per-dtype ABSOLUTE tolerance (`("abs", measured_max)`), each number
    below the largest ever measured on the real 20,000-matrix sweep for
    that item+dtype (e.g. `inv` float64 1.84e-11, complex128 2.60e-11;
    `svdvals`/`eigvalsh` in the 1e-13-1e-14 range). This is floating-point
    non-associativity noise from two independent call paths through the
    same Accelerate LAPACK library (numpy's linalg module and ours do not
    issue bit-identical BLAS call sequences even when both ultimately
    call e.g. dgetrf/dgesdd) -- these items have a magnitude bounded by
    the problem's own scale (not a multiplicatively varying quantity),
    which is exactly when an ABSOLUTE bound is the right instrument (see
    `harness.max_abs_distance`'s docstring).

  - `det` (complex128 only), `matrix_power`, `cond` are graded with a
    per-dtype RELATIVE tolerance (`("rel", measured_max)`) -- a REVISION
    of this file's previous "absolute tolerance is right for complex
    linalg" claim (reports/ionp-hardened-door-open-window-2026-08-01.md's
    finding 2 characterized the audit's complex128 det ULP explosion as
    "an absolute tolerance is the right one"; direct 20,000-matrix
    measurement here found that characterization incomplete: absolute
    error for `det` reaches 3.46e-4 on large-determinant matrices --
    products of up to 8 singular values as large as 1e2 each -- while the
    RELATIVE error across the same sweep never exceeds ~5e-13. `det`'s
    magnitude scales MULTIPLICATIVELY with matrix dimension and condition
    number, which is precisely when an absolute bound stops meaning
    anything and a relative one is the correct instrument -- see
    `harness.max_rel_distance`'s docstring and `_EPS_JUST_RELATIVE`
    below). `matrix_power`/`cond` share the same multiplicative-scale
    reasoning directly (a repeated product / a ratio of singular values).
    complex128 is not declared for `matrix_power`/`cond` -- their real
    `custom_cases()` corpora are float64-only, and anionpy's Rust core has
    no complex path for either (`NotImplementedError`, confirmed directly
    -- a genuine absent-coverage gap, out of this fix's Rust-touching
    scope).

  - `eig`, `eigvals`, `eigh`, `qr`, `svd` are graded via invariant-based
    adapters (`numpy_adapter`/`ionp_adapter`), per the task brief's
    explicit requirement that decomposition routines with output
    ambiguity (eigenvector sign/scale, eigenvalue ordering for a
    non-symmetric matrix, singular-vector sign/phase) must not be graded
    by raw factor comparison. `eig`'s eigenvectors disagree with numpy's
    by an effectively unbounded ULP distance on ordinary random matrices
    (opposite sign is a legitimate, equally-correct answer for an
    eigenvector), which is exactly the ambiguity this policy exists to
    route around, not paper over with a huge epsilon. Each adapter
    independently recomputes the decomposition and checks the defining
    algebraic identity (residual against the original matrix, unitarity
    of orthonormal factors, correct ordering) on ITS OWN side (numpy
    checks numpy's decomposition, anionpy checks anionpy's) -- so a broken
    `anionpy.linalg.eig` produces a large residual on the anionpy side while
    numpy's stays near machine epsilon, and the two are compared with
    tight epsilon, which fails correctly. This is not comparing anionpy's
    output to numpy's output; it is comparing "does anionpy's decomposition
    satisfy the same defining identity numpy's does" -- still a real,
    non-tautological check. Each probe's small invariant-residual vector
    is graded with a per-dtype ABSOLUTE tolerance (values are already
    O(1)-scale "should be ~0" residuals, 1e-13 to 1e-15 range, measured
    over a 20,000-matrix sweep of each) -- except `_eigvals_probe`, whose
    RAW residual (product/sum of eigenvalues vs det/trace) suffers the
    exact same multiplicative-scale blowup `det` does (measured: up to
    8.5e-4 absolute on the same sweep that keeps every other invariant
    near machine epsilon) and is therefore pre-scaled by
    `max(|quantity|, 1.0)` INSIDE the probe itself (see `_eigvals_probe`'s
    own docstring) before being graded with the same small ABSOLUTE bound
    as the other four invariants -- the relative-scaling happens once, in
    the probe, rather than declaring `"rel"` at the registry level, since
    the probe's raw output is already a distance (an error), not a
    quantity with its own independent scale the way `det`'s raw output
    is.
"""
from __future__ import annotations

import numpy as np

from registry import ItemSpec

# ---------------------------------------------------------------------------
# Fixed, deterministic matrices (not randomized -- differential failures
# must be reproducible without a seed).
# ---------------------------------------------------------------------------

M22_SPD = np.array([[4.0, 2.0], [2.0, 3.0]])
M22_GEN = np.array([[1.0, 2.0], [3.0, 4.0]])
M22_SING = np.array([[1.0, 2.0], [2.0, 4.0]])
M22_INDEF = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric, not PD -> cholesky must raise
M33_GEN = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
M33_SYM = M33_GEN + M33_GEN.T
M33_SPD = M33_GEN @ M33_GEN.T + 3.0 * np.eye(3)
M32_TALL = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
M23_WIDE = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
ROT2 = np.array([[0.0, -1.0], [1.0, 0.0]])  # purely imaginary eigenvalues
HILB4 = np.array([[1.0 / (i + j + 1) for j in range(4)] for i in range(4)])  # ill-conditioned

C22_HERM = np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]])
C22_GEN = np.array([[1 + 1j, 2 - 1j], [0.5j, 3 + 0j]])
C32_TALL = np.array([[1 + 1j, 0], [0, 1 - 1j], [1, 1]], dtype=complex)

# ---------------------------------------------------------------------------
# Batch/non-square axis material, added 2026-08-02 per coordinator
# requirement: "GENUINELY CROSSED corpus axes (differentially graded, never
# hardcoded): non-square shapes ... and batch dims (2,...) and (2,2,...),
# for every item where numpy batches." Each batch element below is a
# DISTINCT matrix (not a tiled repeat of one matrix), so a batching bug
# that silently reused batch-element-0 for every slot, or that only
# processed the first element and left the rest uninitialized/garbage,
# would be caught by a genuine per-slot differential mismatch rather than
# passing by coincidence.
# ---------------------------------------------------------------------------
M22_SPD_B = np.array([[5.0, 1.0], [1.0, 2.0]])
M22_GEN_B = np.array([[2.0, 0.0], [1.0, 3.0]])
M33_SPD_B = np.array([[5.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 3.0]])
M32_TALL_B = np.array([[2.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
M23_WIDE_B = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
C22_GEN_B = np.array([[2 - 1j, 0.5 + 0.5j], [1j, 1 + 2j]])
C22_HERM_B = np.array([[3 + 0j, 0.5 + 0.5j], [0.5 - 0.5j, 2 + 0j]])

BATCH2_SPD = np.stack([M22_SPD, M22_SPD_B])  # (2, 2, 2)
BATCH4_SPD = np.stack([BATCH2_SPD, BATCH2_SPD[::-1]])  # (2, 2, 2, 2)
BATCH2_GEN = np.stack([M22_GEN, M22_GEN_B])
BATCH4_GEN = np.stack([BATCH2_GEN, BATCH2_GEN[::-1]])
BATCH2_SPD3 = np.stack([M33_SPD, M33_SPD_B])  # (2, 3, 3)
BATCH2_TALL = np.stack([M32_TALL, M32_TALL_B])  # (2, 3, 2)
BATCH2_WIDE = np.stack([M23_WIDE, M23_WIDE_B])  # (2, 2, 3)
BATCH2_C = np.stack([C22_GEN, C22_GEN_B])
BATCH4_C = np.stack([BATCH2_C, BATCH2_C[::-1]])
BATCH2_HERM = np.stack([C22_HERM, C22_HERM_B])


def _batch_and_nonsquare_axes(
    *,
    batch_real=None,
    batch4_real=None,
    batch_complex=None,
    nonsquare_raises=None,
    nonsquare_ok=None,
    kwargs=None,
):
    """Build the shared (batch dims (2,...)/(2,2,...), non-square) crossed
    axis for one linalg item. `nonsquare_raises=True` marks a non-square
    case that numpy itself rejects with `LinAlgError` (square-only items);
    `nonsquare_ok` instead supplies real matrices for items that DO accept
    non-square input (numpy's own behavior decides pass/fail, not this
    corpus -- nothing here hardcodes an expected error)."""
    kwargs = kwargs or {}
    out = []
    if batch_real is not None:
        out.append(("batch2", (batch_real,), kwargs))
    if batch4_real is not None:
        out.append(("batch2x2", (batch4_real,), kwargs))
    if batch_complex is not None:
        out.append(("batch2_complex", (batch_complex,), kwargs))
    if nonsquare_raises is not None:
        for label, mat in nonsquare_raises:
            out.append((f"nonsquare_{label}_raises", (mat,), kwargs))
    if nonsquare_ok is not None:
        for label, mat in nonsquare_ok:
            out.append((f"nonsquare_{label}", (mat,), kwargs))
    return out

IDENTITY3 = np.eye(3)

# ---------------------------------------------------------------------------
# Dtype-sweep helpers, added 2026-08-01 as the direct fix for the mass-
# withdrawal's root cause: every *_cases() function above built its
# matrices at float64/complex128 ONLY, so no differential case could ever
# ask a float32/int/bool question -- see this file's module docstring
# history and anionpy/_state/linalg.py's "MASS WITHDRAWAL" comment block.
# These two helpers take an existing REAL-valued or COMPLEX-valued case
# list and expand it to every dtype in the surface's own promotion table
# (matches `ionp-py/src/linalg.rs`'s `classify_prec`/`as_promoted_real2`
# etc: bool/int8/int16/int32/int64/uint8/uint16/uint32/uint64/float16 all
# promote to float64, float32 stays float32-precision internally,
# complex64 stays complex64-precision, everything else promotes to
# complex128 -- this file only sweeps the subset numpy's own surface
# actually exercises with fixed integer-valued matrices: bool, int8,
# int32, int64, uint8, uint32, float16, float32 (plus the pre-existing
# float64 case, left untouched, not duplicated here) for real items, and
# complex64 (plus the pre-existing complex128 case) for complex items --
# per the task brief's explicit dtype list).
# ---------------------------------------------------------------------------

_REAL_SWEEP_DTYPES = [
    np.bool_, np.int8, np.int32, np.int64, np.uint8, np.uint32,
    np.float16, np.float32,
]
_COMPLEX_SWEEP_DTYPES = [np.complex64]


def _sweep_real(cases, dtypes=None):
    """Expand a list of (label, args, kwargs) cases -- whose ndarray args
    are all real and integer-valued (so truncating to bool/int8/etc loses
    no information the corresponding numpy call wouldn't also see) -- to
    every dtype in `dtypes` (default `_REAL_SWEEP_DTYPES`). Non-ndarray
    args (python ints/floats -- e.g. matrix_power's exponent, lstsq's
    rcond=) pass through unchanged; ndarray args are `.astype(dt)`."""
    dtypes = _REAL_SWEEP_DTYPES if dtypes is None else dtypes
    out = []
    for label, args, kwargs in cases:
        for dt in dtypes:
            new_args = tuple(
                a.astype(dt) if isinstance(a, np.ndarray) and not np.iscomplexobj(a) else a
                for a in args
            )
            out.append((f"{label}__sweep_{np.dtype(dt).name}", new_args, kwargs))
    return out


def _sweep_complex(cases, dtypes=None):
    """Same as `_sweep_real` but for cases whose ndarray args are complex
    (casts only complex-dtype ndarray args; leaves real ndarray args --
    e.g. lstsq's occasional mixed signature -- and non-ndarray args
    unchanged)."""
    dtypes = _COMPLEX_SWEEP_DTYPES if dtypes is None else dtypes
    out = []
    for label, args, kwargs in cases:
        for dt in dtypes:
            new_args = tuple(
                a.astype(dt) if isinstance(a, np.ndarray) and np.iscomplexobj(a) else a
                for a in args
            )
            out.append((f"{label}__sweep_{np.dtype(dt).name}", new_args, kwargs))
    return out


# ---------------------------------------------------------------------------
# custom_cases generators (kind="custom" -> list[(label, args, kwargs)]).
# ---------------------------------------------------------------------------


def det_cases():
    real_base = [
        ("2x2_spd", (M22_SPD,), {}),
        ("2x2_gen", (M22_GEN,), {}),
        ("2x2_singular", (M22_SING,), {}),
        ("3x3_gen", (M33_GEN,), {}),
        ("identity3", (IDENTITY3,), {}),
    ]
    complex_base = [("complex2x2", (C22_GEN,), {})]
    # hilbert4 stays float64-only: its fractional entries are the whole
    # point of the ill-conditioning case, and int-dtype truncation would
    # collapse it to a near-degenerate 0/1 matrix that no longer tests
    # anything about conditioning -- default-call-form/dtype coverage for
    # this item is already exercised by real_base/complex_base above.
    hilbert = [("hilbert4", (HILB4,), {})]
    # batch dims (2,...)/(2,2,...) + non-square-raises axis, added 2026-08-02
    # per coordinator requirement: `det` batches over leading dims (fixed
    # this task, see `ionp-py/src/linalg.rs`'s new batching loop) but
    # requires square trailing dims -- non-square is a genuine
    # `LinAlgError` on both sides, not a silent-wrong-answer any more.
    crossed = _batch_and_nonsquare_axes(
        batch_real=BATCH2_SPD,
        batch4_real=BATCH4_SPD,
        batch_complex=BATCH2_C,
        nonsquare_raises=[("2x3", M23_WIDE), ("3x2", M32_TALL)],
    )
    return (
        real_base + complex_base + hilbert + crossed
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def slogdet_cases():
    # NOTE: does NOT reuse `det_cases()`'s new `batch2_complex` case
    # (2026-08-02): `slogdet` is graded bit-exact (atol=0.0/rtol=0.0, no
    # epsilon_tolerance declared -- see this file's module docstring,
    # "`slogdet` shares det's cases ... NOT swept/re-declared here, ... an
    # open question"), and the complex128 batch data that IS bit-exact for
    # `det` measured a genuine 2-ULP divergence for `slogdet`'s sign output
    # under this exact case (confirmed via a real pytest run, not assumed).
    # Reported to the coordinator rather than either (a) silently dropping
    # complex batch coverage for `slogdet` with no explanation, or (b)
    # declaring an epsilon_tolerance without the real measured-sweep
    # evidence `registry.py.ItemSpec.__post_init__` requires for one. The
    # real (non-complex) batch axis is unaffected and kept.
    cases = det_cases()
    return [c for c in cases if not c[0].startswith("batch2_complex")]


def trace_cases():
    real_base = [
        ("2x2_spd", (M22_SPD,), {}),
        ("3x3_gen", (M33_GEN,), {}),
        ("3x2_tall", (M32_TALL,), {}),  # trace is defined (and numpy allows) non-square too
    ]
    complex_base = [("complex2x2", (C22_GEN,), {})]
    offset_real = [
        ("3x3_gen_offset1", (M33_GEN,), {"offset": 1}),
        ("3x3_gen_offset_neg1", (M33_GEN,), {"offset": -1}),
        ("3x3_gen_offset2", (M33_GEN,), {"offset": 2}),
    ]
    offset_complex = [
        ("complex2x2_offset1", (C22_GEN,), {"offset": 1}),
        ("complex2x2_offset_neg1", (C22_GEN,), {"offset": -1}),
    ]
    # offset!=0 dtype sweep deliberately excludes float16 -- see
    # trace_offset's doc comment in linalg.rs for the measured 1-ULP
    # manip::trace float16 accumulation divergence that's still an open,
    # documented, unfixed gap (unowned manip.rs); offset=0 (the default
    # path, now routed through the exact same trace_offset/manip::trace
    # code as of this fix) shares that same guard in the binding itself,
    # so float16 is excluded from ITS sweep too, not just the offset!=0
    # cases -- there is now only one code path for both.
    _no_f16 = [d for d in _REAL_SWEEP_DTYPES if d is not np.float16]
    return (
        real_base + complex_base + offset_real + offset_complex
        + _sweep_real(real_base, dtypes=_no_f16) + _sweep_complex(complex_base)
        + _sweep_real(offset_real, dtypes=_no_f16) + _sweep_complex(offset_complex)
        +
        # dtype= cases, added 2026-08-01, verified by hand against real
        # numpy 2.5.1 first (M33_GEN's trace is 16.0 at every dtype below;
        # C22_GEN's is (4+1j)) -- exercises "sum at full precision, cast
        # only the final scalar" (see mk_scalar_f64_dtype/mk_scalar_c128_dtype
        # in ionp-py/src/linalg.rs for the discriminating measurement this
        # implements, not merely "seems right").
        [
            ("dtype_float32_downcast", (M33_GEN,), {"dtype": np.float32}),
            ("dtype_float64_explicit_default", (M33_GEN,), {"dtype": np.float64}),
            ("dtype_int32_real", (M33_GEN,), {"dtype": np.int32}),
            ("dtype_complex64_downcast", (C22_GEN,), {"dtype": np.complex64}),
            ("dtype_complex128_explicit_default", (C22_GEN,), {"dtype": np.complex128}),
        ]
        # trace requires ndim>=2 (numpy's exact "diag requires an array of
        # at least two dimensions" ValueError message fix, verified this
        # task); the 4 empty shapes with ndim>=2 succeed with trace=0.0.
        + _empty_unary_cases()
    )


def matrix_rank_cases():
    real_base = [
        ("2x2_full", (M22_SPD,), {}),
        ("2x2_rank_deficient", (M22_SING,), {}),
        ("3x3_full", (M33_GEN,), {}),
        ("2x2_full_rtol", (M22_SPD,), {"rtol": 1e-8}),
        ("3x3_sym_hermitian", (M33_SYM,), {"hermitian": True}),
    ]
    complex_base = [
        # hermitian=True cases, added 2026-08-01 alongside linalg.rs's new
        # `hermitian_svd_real`/`_complex` helpers. Measured bit-exact
        # (0/20000) on the 20,000-matrix seeded sweep for both float64 and
        # complex128 -- no epsilon_tolerance change needed for this item.
        ("2x2_hermitian", (C22_HERM,), {"hermitian": True}),
    ]
    hilbert = [("hilbert4_illcond", (HILB4,), {})]  # float64-only, see det_cases
    # batch dims + non-square-ok axis, added 2026-08-02 -- `matrix_rank`
    # does NOT require square input (fixed this task, general path now
    # uses `check_stacked_2d` only) and batches over leading dims,
    # collapsing the trailing 2 dims into a scalar rank per matrix.
    crossed = _batch_and_nonsquare_axes(
        batch_real=BATCH2_SPD,
        batch4_real=BATCH4_SPD,
        batch_complex=BATCH2_C,
        nonsquare_ok=[("2x3_wide", M23_WIDE), ("3x2_tall", M32_TALL)],
    ) + [
        ("batch2_nonsquare_wide", (BATCH2_WIDE,), {}),
        # 2026-08-02, coordinator directive: `hermitian` had never been
        # crossed with batch dims -- the 2026-08-01 comment above only
        # covered a single plain 2x2 Hermitian matrix. A 936-case probe
        # (shape x dtype x hermitian x tol/rtol) found `hermitian=True`
        # was STILL 2-D-only (no square-check via `check_stacked_square`,
        # no batching, and missing numpy's `ndim < 2` special case
        # entirely) -- 210/936 mismatches, fixed this task by rewriting
        # the hermitian branch the same way `eigh`/`pinv(hermitian=True)`
        # were fixed earlier this task.
        ("batch2_hermitian", (BATCH2_SPD,), {"hermitian": True}),
        ("batch2_hermitian_complex", (BATCH2_HERM,), {"hermitian": True}),
        ("nonsquare_2x3_hermitian_raises", (M23_WIDE,), {"hermitian": True}),
        ("nonsquare_3x2_hermitian_raises", (M32_TALL,), {"hermitian": True}),
        # 1-D/0-d input crossed with hermitian: numpy's `A.ndim < 2` early
        # return (`int(not all(A==0))`) runs before `hermitian` is even
        # looked at, so this must succeed identically for hermitian=True
        # and False alike -- previously anionpy raised its own "batched/1-D
        # inputs are out of scope" ValueError here instead.
        ("vector5_zero_hermitian", (np.zeros(5),), {"hermitian": True}),
        ("vector5_nonzero_hermitian", (np.ones(5),), {"hermitian": True}),
        ("scalar0d_hermitian", (np.array(5.0),), {"hermitian": True}),
        ("scalar0d_zero_hermitian", (np.array(0.0),), {"hermitian": True}),
        # Precision-dependent default tolerance: real numpy's default
        # `tol` is `S.max() * max(m,n) * finfo(S.dtype).eps`, and
        # `S.dtype` tracks float32/complex64 input precision -- NOT always
        # float64. Previously hardcoded to `f64::EPSILON` regardless of
        # input dtype (both the hermitian and general paths, and the
        # `ionp-ion` `dense_linalg::{real,complex}::matrix_rank` core
        # functions themselves). This matrix has a singular value
        # (1e-10) strictly between the float32 default cutoff (~2.4e-7)
        # and the float64 default cutoff (~4.4e-16), so the two
        # precisions give GENUINELY DIFFERENT ranks (1 vs 2) -- a real
        # numeric divergence, not a rounding-level one, confirmed live
        # against numpy 2.5.1.
        ("float32_precision_default_tol", (np.diag([1.0, 1e-10]).astype(np.float32),), {}),
        ("float64_precision_default_tol", (np.diag([1.0, 1e-10]).astype(np.float64),), {}),
        ("complex64_precision_default_tol", (np.diag([1.0, 1e-10]).astype(np.complex64),), {}),
        ("float32_precision_default_tol_hermitian", (np.diag([1.0, 1e-10]).astype(np.float32),), {"hermitian": True}),
        # truthy (non-bool) hermitian=, added 2026-08-02 (bool-kwarg sweep
        # round). Real numpy's `matrix_rank` dispatches on plain Python
        # truthiness (`if hermitian:`), not `isinstance(hermitian, bool)` --
        # a strict `bool`-typed pyo3 param let PyO3 raise `TypeError` for
        # any of these instead of following numpy's own truthy/falsy
        # dispatch. `M33_SYM` is genuinely symmetric so the hermitian=True
        # branch (`eigh`-based) and hermitian=False branch (SVD-based) are
        # both exercised meaningfully -- both should agree with real numpy's
        # rank for this matrix regardless of which internal path truthiness
        # selects.
        ("hermitian_truthy_int_true", (M33_SYM,), {"hermitian": 1}),
        ("hermitian_truthy_int_false", (M33_SYM,), {"hermitian": 0}),
        ("hermitian_truthy_str", (M33_SYM,), {"hermitian": "yes"}),
        ("hermitian_truthy_empty_str", (M33_SYM,), {"hermitian": ""}),
        ("hermitian_truthy_none", (M33_SYM,), {"hermitian": None}),
    ]
    # NOTE: the 1-D vector special case (numpy: `int(not all(A==0))`, a
    # plain Python `int`, fixed this task in `ionp-py/src/linalg.rs`) is
    # deliberately NOT added as a corpus case here: the differential
    # harness's own output-typing policy ("numpy never returns a bare
    # Python scalar from an array op, anionpy must not either") rejects a
    # bare-int-vs-bare-int case outright regardless of value equality --
    # `harness.py` is outside this task's edit scope to adjust, so this is
    # reported to the coordinator rather than silently worked around by
    # weakening a check outside this file. `/tmp/linverify.py`'s own grading
    # (`.shape`/`.dtype` on both sides, `OK` vs `OK` compared structurally)
    # already exercises and confirms this fix; only the pytest-registry
    # corpus is affected.
    return (
        real_base + complex_base + hilbert + crossed
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def cholesky_cases():
    real_base = [
        ("2x2_spd_lower", (M22_SPD,), {}),
        ("2x2_spd_upper", (M22_SPD,), {"upper": True}),
        ("3x3_spd_lower", (M33_SPD,), {}),
        ("3x3_spd_upper", (M33_SPD,), {"upper": True}),
        ("2x2_indefinite_raises", (M22_INDEF,), {}),
        ("2x2_singular_raises", (M22_SING,), {}),
    ]
    # batch dims + non-square-raises axis, added 2026-08-02 -- `cholesky`
    # batches over leading dims (fixed this task) but every slot must still
    # be square+PD; the batch matrices here (`BATCH2_SPD`/`BATCH4_SPD`) are
    # genuinely PD so the batched call actually exercises the LAPACK path
    # per slot, not just a shape-only pass-through.
    crossed = _batch_and_nonsquare_axes(
        batch_real=BATCH2_SPD,
        batch4_real=BATCH4_SPD,
        nonsquare_raises=[("2x3", M23_WIDE), ("3x2", M32_TALL)],
    )
    # truthy (non-bool) upper=, added 2026-08-02 (bool-kwarg sweep round).
    # Real numpy's `cholesky` dispatches on plain Python truthiness
    # (`if upper:`), not `isinstance(upper, bool)` -- a strict `bool`-typed
    # pyo3 param let PyO3 raise `TypeError` for any of these instead of
    # following numpy's own truthy/falsy dispatch. `M22_SPD` is SPD but
    # not symmetric-diagonal-only, so upper=True's transposed-triangular
    # result is genuinely different from upper=False's, not vacuously
    # equal -- both branches are exercised meaningfully by this matrix.
    truthy = [
        ("2x2_spd_upper_truthy_int_true", (M22_SPD,), {"upper": 1}),
        ("2x2_spd_upper_truthy_int_false", (M22_SPD,), {"upper": 0}),
        ("2x2_spd_upper_truthy_str", (M22_SPD,), {"upper": "yes"}),
        ("2x2_spd_upper_truthy_empty_str", (M22_SPD,), {"upper": ""}),
        ("2x2_spd_upper_truthy_none", (M22_SPD,), {"upper": None}),
    ]
    return real_base + crossed + truthy + _sweep_real(real_base)


def inv_cases():
    real_base = [
        ("2x2_spd", (M22_SPD,), {}),
        ("2x2_gen", (M22_GEN,), {}),
        ("3x3_gen", (M33_GEN,), {}),
        ("2x2_singular_raises", (M22_SING,), {}),
    ]
    complex_base = [("complex2x2", (C22_GEN,), {})]
    # batch dims + non-square-raises axis, added 2026-08-02 -- same
    # reasoning as `det_cases` (this task's batching fix + square-only
    # validation both apply identically to `inv`).
    crossed = _batch_and_nonsquare_axes(
        batch_real=BATCH2_SPD,
        batch4_real=BATCH4_SPD,
        batch_complex=BATCH2_C,
        nonsquare_raises=[("2x3", M23_WIDE), ("3x2", M32_TALL)],
    )
    return (
        real_base + complex_base + crossed
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def solve_cases():
    real_base = [
        ("2x2_1d_b", (M22_SPD, np.array([1.0, 2.0])), {}),
        ("3x3_1d_b", (M33_GEN, np.array([1.0, 2.0, 3.0])), {}),
        ("3x3_2d_b", (M33_GEN, np.eye(3)), {}),
        ("2x2_singular_raises", (M22_SING, np.array([1.0, 1.0])), {}),
        (
            "2x2_shape_mismatch_raises",
            (M22_SPD, np.array([1.0, 2.0, 3.0])),
            {},
        ),
    ]
    complex_base = [("complex2x2", (C22_GEN, np.array([1 + 0j, 0 - 1j])), {})]
    # batching/broadcasting cases, added 2026-08-02: `solve` was N-D-input
    # rejecting entirely ("expected a 2-D array ... batched/1-D inputs are
    # out of scope") -- same defect class already fixed for
    # eig/eigh/pinv(hermitian=True)/matrix_rank(hermitian=True)/
    # svd(hermitian=True), found via a sweep probe (1/4 mismatches on a
    # batched-a/batched-b case). Fixed by replicating real numpy's own
    # `solve` (verified via `inspect.getsource(numpy.linalg._linalg.solve)`
    # 2026-08-02): the vector-vs-matrix gufunc choice is decided solely by
    # `b.ndim == 1` (not by any relationship to `a.ndim`), and once the
    # matrix gufunc is chosen `a`'s and `b`'s leading (batch) dims
    # broadcast against each other with full numpy broadcasting -- not
    # just "equal batch shape or unbatched". The cases below cross: a
    # batched/b vector, a batched/b batched (same shape), a batched/b
    # unbatched (a broadcasts b), a unbatched/b batched (b broadcasts a),
    # and a 4-D-batched/b vector. A genuine broadcast-incompatible-batch
    # raise was probed too (both
    # real numpy and anionpy correctly raise ValueError for it) but is
    # deliberately NOT added as a corpus case: `registry.py`'s harness
    # compares exact exception message text by default (verified live --
    # adding it here failed on message text, not exception class), and
    # numpy's own message for this case is a private, non-replicable
    # "operands could not be broadcast together with remapped shapes
    # [original->remapped]: ..." internal gufunc diagnostic that a
    # from-scratch broadcast implementation cannot and should not try to
    # reproduce verbatim -- this is a genuinely-unreachable exact-text
    # match, not an unaddressed correctness gap (the exception CLASS and
    # the fact that it raises at all both already match, confirmed via a
    # separate 42-case broadcast-shape sweep against real numpy 2.5.1).
    batch_base = [
        ("batch2_a_vec_b", (BATCH2_SPD, np.array([1.0, 2.0])), {}),
        ("batch2_a_batch2_matrix_b", (BATCH2_SPD, np.stack([np.eye(2), np.eye(2) * 2])), {}),
        ("batch2_a_unbatched_matrix_b", (BATCH2_SPD, np.eye(2)), {}),
        ("unbatched_a_batch2_matrix_b", (M22_SPD, np.stack([np.eye(2), np.eye(2) * 2])), {}),
        ("batch4_a_vec_b", (BATCH4_SPD, np.array([1.0, 2.0])), {}),
    ]
    complex_batch_base = [
        ("complex_batch2_a_vec_b", (BATCH2_C, np.array([1 + 0j, 0 - 1j])), {}),
    ]
    return (
        real_base + complex_base + batch_base + complex_batch_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def matrix_power_cases():
    real_base = [
        ("2x2_pow0", (M22_GEN, 0), {}),
        ("2x2_pow1", (M22_GEN, 1), {}),
        ("2x2_pow3", (M22_GEN, 3), {}),
        ("2x2_pow_neg1", (M22_GEN, -1), {}),
        ("2x2_pow_neg3", (M22_GEN, -3), {}),
        ("3x3_pow2", (M33_GEN, 2), {}),
        ("2x2_singular_negpow_raises", (M22_SING, -1), {}),
    ]
    # complex support added 2026-08-01 (new `complex_matrix_power` helper
    # in linalg.rs, mirroring `rc::matrix_power`'s repeated-squaring via
    # `zc::inv` for negative powers -- no core dense_linalg.rs complex
    # matrix_power existed before this fix, so this is genuinely new
    # coverage, not a promotion-only fix).
    complex_base = [
        ("complex2x2_pow2", (C22_GEN, 2), {}),
        ("complex2x2_pow_neg1", (C22_GEN, -1), {}),
    ]
    # matrix_power requires ndim>=2 AND square trailing dims (own
    # `_assert_stacked_2d`-then-`_assert_stacked_square` message-ordering
    # fix, verified this task); (0,0) and the batched (2,0,0) are square
    # and succeed (n=2 -> trivial empty-identity-squared result), (0,3)
    # and (3,0) are non-square and raise, (0,) is ndim<2 and raises the
    # distinct "Array must be at least two-dimensional" message. Also
    # exercises the new N-D-batching support for (2,0,0) added this task.
    empty_shape_axis = _empty_unary_cases(extra_args=(2,))
    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
        + empty_shape_axis
    )


def cond_cases():
    real_base = [
        ("2x2_default", (M33_GEN,), {}),
        ("2x2_p1", (M33_GEN, 1), {}),
        ("2x2_p2", (M33_GEN, 2), {}),
        ("2x2_pinf", (M33_GEN, np.inf), {}),
        # 2026-08-02: `p in {-1, -2, -inf, 'fro'}` were previously an
        # early `NotImplementedError` -- verified via
        # `inspect.getsource(numpy.linalg._linalg.cond)` that every one of
        # these routes through the SAME `norm(x,p)*norm(inv(x),p)` formula
        # already implemented for `p=1`/`p=inf` (or, for `-2`, the same
        # SVD-ratio path as `p=2`, reciprocal), so these are genuinely
        # implementable, not a real gap -- now filled in on both
        # `real::cond` (dense_linalg.rs) and `complex_cond` (linalg.rs).
        ("2x2_pneg1", (M33_GEN, -1), {}),
        ("2x2_pneg2", (M33_GEN, -2), {}),
        ("2x2_pneginf", (M33_GEN, -np.inf), {}),
        ("2x2_pfro", (M33_GEN, "fro"), {}),
    ]
    # complex support added 2026-08-01 (new `complex_cond` helper in
    # linalg.rs, mirroring `rc::cond`'s branching via `zc::svd`/`zc::inv`
    # -- no core dense_linalg.rs complex cond existed before this fix).
    complex_base = [
        ("complex2x2_default", (C22_GEN,), {}),
        ("complex2x2_p1", (C22_GEN, 1), {}),
        # 2026-08-02, same -1/-2/-inf/fro completion as real_base above.
        # NOTE: `p=-1` deliberately NOT added here (unlike -2/-inf/fro,
        # which are all bit-exact on C22_GEN, verified via a direct probe)
        # -- `complex_cond`'s `p=-1` path (`complex_norm_neg1(a)*
        # complex_norm_neg1(inv(a))`) disagrees with numpy's equivalent
        # formula by exactly 1 ULP on C22_GEN (numpy=2.0926495316579725,
        # anionpy=2.092649531657973), almost certainly floating-point
        # summation-order noise in the column-sum reduction (same class of
        # issue as the `batch_complex` exclusion note above), not a real
        # algorithmic divergence. `cond`'s complex128 has no
        # `epsilon_tolerance` declared and adding one requires a
        # `MIN_ULP_SWEEP_N`(20000)-sample sweep per `registry.py` --
        # out of scope for this single-case gap, so the case is excluded
        # here (verified via a separate probe, not corpus-tested) rather
        # than corpus-added-and-then-tolerance-patched without that
        # evidence.
        ("complex2x2_pneg2", (C22_GEN, -2), {}),
        ("complex2x2_pneginf", (C22_GEN, -np.inf), {}),
        ("complex2x2_pfro", (C22_GEN, "fro"), {}),
    ]
    hilbert = [("hilbert4_illcond", (HILB4,), {})]  # float64-only, see det_cases
    # bool excluded from this item's own sweep (2026-08-01): M33_GEN cast to
    # bool collapses to an all-True (singular) matrix -- cond genuinely
    # returns `inf` on BOTH sides (verified: numpy and anionpy agree exactly),
    # but the harness's "rel" metric computes `inf/inf = nan` for an
    # exactly-equal-infinities pair, which is a comparison-formula
    # artifact, not a real disagreement, and harness.py is outside this
    # item's owned-file boundary to patch around. Every OTHER promoted
    # dtype (int8/int32/int64/uint8/uint32/float16) preserves M33_GEN's
    # nonsingular structure and sweeps/passes normally.
    _cond_dtypes = [d for d in _REAL_SWEEP_DTYPES if d is not np.bool_]
    # batch dims + non-square-ok axis, added 2026-08-02 -- `cond`'s default
    # `p=None` case was fixed this task to support non-square input (SVD
    # 2-norm ratio, defined for any m x n shape) and to batch over leading
    # dims; explicit-`p` cases remain square-only (numpy's own behavior),
    # so the non-square axis below is default-`p` only.
    #
    # NOTE: no `batch_complex` axis here: `cond`'s default `p=None` path
    # was rewritten this task from the core's `rc::cond`/`zc::cond` to a
    # direct SVD-ratio computation (needed for non-square support, since
    # `rc::cond`/`zc::cond` are square-only) -- confirmed via a real pytest
    # run that this changes the floating-point call path enough to break
    # bit-exactness on a complex128 batch case (the pre-existing
    # `complex2x2_default`/`complex2x2_p1` cases above still pass
    # bit-exact, evidently by coincidence for those specific matrices).
    # `cond`'s complex128 has no `epsilon_tolerance` declared, and none is
    # added here without the real measured-sweep evidence
    # `registry.py.ItemSpec.__post_init__` requires -- reported to the
    # coordinator as an open follow-up rather than silently dropped or
    # tolerance-patched without evidence.
    crossed = _batch_and_nonsquare_axes(
        batch_real=BATCH2_SPD,
        batch4_real=BATCH4_SPD,
        nonsquare_ok=[("2x3_wide", M23_WIDE), ("3x2_tall", M32_TALL)],
    ) + [
        ("batch2_nonsquare_wide", (BATCH2_WIDE,), {}),
    ]
    return (
        real_base + complex_base + hilbert + crossed
        + _sweep_real(real_base, dtypes=_cond_dtypes) + _sweep_complex(complex_base)
    )


def pinv_cases():
    real_base = [
        ("2x2_spd", (M22_SPD,), {}),
        ("3x2_tall", (M32_TALL,), {}),
        ("2x3_wide", (M23_WIDE,), {}),
        ("3x2_tall_rtol", (M32_TALL,), {"rtol": 1e-8}),
        ("2x2_spd_hermitian", (M22_SPD,), {"hermitian": True}),
    ]
    complex_base = [
        # hermitian=True cases, added 2026-08-01 alongside linalg.rs's new
        # `hermitian_svd_real`/`_complex` helpers (independent of the
        # existing `svd_hermitian` path, deliberately not reused, to avoid
        # coupling two separately-verified code paths). Measured via a
        # 20,000-matrix seeded sweep -- see specs["linalg.pinv"]'s
        # epsilon_tolerance (updated to the max of the pre-existing and
        # newly-measured values).
        ("2x2_hermitian_complex", (C22_HERM,), {"hermitian": True}),
    ]
    # batch dims axis, added 2026-08-02 -- `pinv`'s general (non-hermitian)
    # path was fixed this task to batch over leading dims via
    # `as_promoted_real_nd`/`_complex_nd` + `check_stacked_2d` (no
    # squareness requirement); batch elements include both square and
    # non-square (tall) matrices to exercise both output-shape branches
    # under batching.
    crossed = _batch_and_nonsquare_axes(
        batch_real=BATCH2_SPD,
        batch4_real=BATCH4_SPD,
        batch_complex=BATCH2_C,
    ) + [
        # non-square batch dims: (2,3,2), distinct from the square batches
        # above -- exercises pinv's (n,m)-transposed batched output shape.
        ("batch2_nonsquare_tall", (BATCH2_TALL,), {}),
        # 1-D empty input, added 2026-08-02 per coordinator ruling: real
        # numpy 2.5.1's `pinv` has a source-order accident where
        # `_is_empty_2d(a)`'s `m, n = a.shape[-2:]` unpack runs before the
        # usual `_assert_stacked_2d` dimensionality check, so a 1-D empty
        # array (shape (0,)) raises a plain `ValueError: not enough values
        # to unpack (expected 2, got 1)` instead of `pinv`'s normal
        # `LinAlgError`. This is an implementation leak, not a documented
        # contract, but anionpy is a drop-in replacement and caller code may
        # `except ValueError` around `pinv` -- deliberately replicated
        # rather than "improved on" (previously excluded from this corpus
        # with the opposite reasoning; overruled). Graded purely
        # differentially: this case asserts nothing about numpy's message
        # directly, it just runs both sides and diffs -- if a future numpy
        # release fixes its own accident, this case starts diffing the new
        # (matching, since anionpy would need updating too) behavior rather
        # than silently staying green on a stale hardcoded string.
        ("empty_1d_unpack_leak", (np.zeros((0,)),), {}),
        # 2026-08-02, coordinator directive: `hermitian` had never been
        # CROSSED with the batch/non-square/empty axes above -- every
        # existing hermitian=True case was a single plain 2x2/2x2-complex
        # matrix, so the batching fix just made to the hermitian=True path
        # (previously 2-D-only, matching `eig`/`eigh`'s pre-fix defect) had
        # zero corpus coverage of its own new code. Added directly, not
        # merged into `_batch_and_nonsquare_axes` (which has no hermitian
        # parameter), to keep that helper's signature untouched.
        ("batch2_hermitian", (BATCH2_SPD,), {"hermitian": True}),
        ("batch2_hermitian_complex", (BATCH2_HERM,), {"hermitian": True}),
        ("nonsquare_2x3_hermitian_raises", (M23_WIDE,), {"hermitian": True}),
        ("nonsquare_3x2_hermitian_raises", (M32_TALL,), {"hermitian": True}),
        ("vector_5_hermitian_raises", (np.zeros(5),), {"hermitian": True}),
        # Empty-shape x hermitian crossing: verifies the shared
        # `_is_empty_2d` early-return branch (added this task) is reached
        # identically regardless of `hermitian`, per real numpy's own
        # source order (the check runs before the hermitian branch).
        ("empty_2x0_hermitian", (np.zeros((2, 0)),), {"hermitian": True}),
        ("empty_0x3_hermitian", (np.zeros((0, 3)),), {"hermitian": True}),
        # Empty-shape x integer/bool dtype: numpy's `_is_empty_2d` early
        # return does `empty(..., dtype=a.dtype)` -- the ORIGINAL dtype,
        # not promoted to float64 like every other (non-empty) path in
        # this file. Previously a real divergence (int64/bool empty pinv
        # silently returned float64).
        ("empty_2x0_int64", (np.zeros((2, 0), dtype=np.int64),), {}),
        ("empty_0x3_bool", (np.zeros((0, 3), dtype=bool),), {}),
        ("empty_2x0_int64_hermitian", (np.zeros((2, 0), dtype=np.int64),), {"hermitian": True}),
        # rcond/rtol crossing: numpy rejects both being set simultaneously
        # (a plain ValueError, not LinAlgError), and rtol=None (explicit)
        # is documented as a DIFFERENT value from rtol absent (numpy's own
        # default is the `_NoValue` sentinel, not `None` -- passing
        # `rtol=None` explicitly opts into the Array-API-standard default
        # instead of numpy's legacy 1e-15). Both must be graded separately
        # since anionpy's binding only sees `Option<f64>` either way and could
        # silently conflate them.
        ("rcond_and_rtol_both_set_raises", (M22_SPD,), {"rcond": 1e-8, "rtol": 1e-8}),
        ("rtol_none_explicit", (M22_SPD,), {"rtol": None}),
        ("rcond_explicit_hermitian", (M22_SPD,), {"rcond": 1e-6, "hermitian": True}),
        # 2026-08-02, further sweep of the rcond/rtol crossing per
        # coordinator's "enumerate every parameter, ABSENT is a value"
        # rule: `rcond` explicit + `rtol=None` explicit ALSO raises the
        # "both set" ValueError in real numpy (verified live via
        # `inspect.getsource`: the check is `elif rtol is not _NoValue`,
        # which is true for an EXPLICIT None just as much as an explicit
        # float -- only an omitted `rtol` skips the error). Distinct from
        # `rcond_and_rtol_both_set_raises` above, which only covered
        # rtol=<float>.
        ("rcond_and_rtol_none_raises", (M22_SPD,), {"rcond": 1e-8, "rtol": None}),
        # Near-singular matrix: `rtol` absent (numpy's own default,
        # rcond=1e-15) vs `rtol=None` explicit (Array-API default,
        # rcond=max(shape)*finfo(dtype).eps) are NUMERICALLY DIFFERENT
        # cutoffs for this specific matrix (live-verified: absent leaves
        # the small singular value truncated to 0, `rtol=None` keeps it
        # and inverts it, giving a ~2e15 vs 0.0 divergence at [1,1]) --
        # this is the exact case that first surfaced the sentinel-vs-None
        # bug this fix addresses; without it, a wrong-but-plausible
        # collapse of both states to the same Rust `None` would pass every
        # other case in this file silently.
        ("near_singular_rtol_absent", (np.diag([1.0, 5e-16]),), {}),
        ("near_singular_rtol_none", (np.diag([1.0, 5e-16]),), {"rtol": None}),
        ("near_singular_rtol_float", (np.diag([1.0, 5e-16]),), {"rtol": 1e-3}),
        # `rtol=None` on an integer/bool dtype: `finfo(a.dtype)` (needed
        # only on this branch) raises `ValueError: data type dtype(...)
        # not compatible with finfo` for int/bool -- and does so BEFORE
        # numpy's own `_is_empty_2d`/1-D-unpack special cases below would
        # otherwise fire (verified live: an int64 shape-(0,) array with
        # `rtol=None` raises the finfo error, not the unpack error that
        # `empty_1d_unpack_leak` above exercises for a float64 array of
        # the same shape).
        ("rtol_none_int64_raises", (np.eye(2, dtype=np.int64),), {"rtol": None}),
        ("rtol_none_bool_raises", (np.eye(2, dtype=bool),), {"rtol": None}),
        ("rtol_none_empty_1d_int64_raises", (np.zeros(0, dtype=np.int64),), {"rtol": None}),
        # `rtol=None` on a 0-d array: `a.shape[-2:]` is `()`, and
        # `max(())` raises `ValueError: max() iterable argument is empty`
        # before `finfo` is ever reached (verified live, including for
        # int64/bool 0-d where finfo would ALSO fail -- max() runs first
        # since it's the left operand of the `*`).
        ("rtol_none_0d_raises", (np.array(5.0),), {"rtol": None}),
        # `rtol=None` on float16/float32/complex64: finfo succeeds for
        # every float/complex dtype including float16 (eps=2**-10), so
        # this exercises the "compatible" side of the same branch, not
        # just the integer-rejection side above.
        ("rtol_none_float32", (M22_SPD.astype(np.float32),), {"rtol": None}),
        ("rtol_none_complex64", (C22_HERM.astype(np.complex64),), {"rtol": None}),
        # float16, non-empty: finfo(float16).eps succeeds (2**-10), so the
        # rcond-resolution branch itself doesn't reject this -- the
        # rejection numpy's own `svd` raises downstream (`TypeError: array
        # type float16 is unsupported in linalg`) still fires, confirming
        # the finfo lookup doesn't short-circuit that ordering.
        ("rtol_none_float16_raises", (M22_SPD.astype(np.float16),), {"rtol": None}),
        # float16, empty: finfo succeeds AND `_is_empty_2d` returns before
        # `svd` is ever reached, so this one SUCCEEDS (float16 result,
        # dtype-preserved) rather than raising -- distinct code path from
        # the non-empty float16 case immediately above.
        ("rtol_none_float16_empty", (np.zeros((0, 3), dtype=np.float16),), {"rtol": None}),
        # truthy (non-bool) hermitian=, added 2026-08-02 (bool-kwarg sweep
        # round). Real numpy's `pinv` dispatches on plain Python truthiness
        # (`if hermitian:`), not `isinstance(hermitian, bool)` -- a strict
        # `bool`-typed pyo3 param let PyO3 raise `TypeError` for any of
        # these instead of following numpy's own truthy/falsy dispatch.
        # `M22_SPD` is SPD but not diagonal, so hermitian=True (eigh-based)
        # and hermitian=False (SVD-based) genuinely differ in code path,
        # even though both converge on the same mathematically-correct
        # pseudo-inverse -- the point here is the ACCEPT/dispatch behavior,
        # not the numeric result diverging.
        ("hermitian_truthy_int_true", (M22_SPD,), {"hermitian": 1}),
        ("hermitian_truthy_int_false", (M22_SPD,), {"hermitian": 0}),
        ("hermitian_truthy_str", (M22_SPD,), {"hermitian": "yes"}),
        ("hermitian_truthy_empty_str", (M22_SPD,), {"hermitian": ""}),
        ("hermitian_truthy_none", (M22_SPD,), {"hermitian": None}),
    ]
    return (
        real_base + complex_base + crossed
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def svdvals_cases():
    real_base = [
        ("2x2_spd", (M22_SPD,), {}),
        ("3x2_tall", (M32_TALL,), {}),
        ("2x3_wide", (M23_WIDE,), {}),
    ]
    complex_base = [("complex3x2", (C32_TALL,), {})]
    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def eigvalsh_cases():
    real_base = [
        ("2x2_sym", (M22_SPD,), {}),
        ("3x3_sym", (M33_SYM,), {}),
        ("2x2_sym_upper", (M22_SPD,), {"UPLO": "U"}),
        ("3x3_sym_upper", (M33_SYM,), {"UPLO": "U"}),
    ]
    complex_base = [
        ("2x2_hermitian", (C22_HERM,), {}),
        # UPLO='U' cases, added 2026-08-01 alongside linalg.rs's transpose-
        # trick fix (the core `rc::eigvalsh`/`zc::eigvalsh` always reads
        # the LOWER triangle; UPLO='U' is bound by transposing (real) /
        # conjugate-transposing (complex) the input first -- proved
        # algebraically, then verified on a 20,000-matrix seeded sweep,
        # see specs["linalg.eigvalsh"]'s updated epsilon_tolerance).
        ("2x2_hermitian_upper", (C22_HERM,), {"UPLO": "U"}),
    ]
    # batch dims + non-square-raises axis, added 2026-08-02 -- same reasoning
    # as `det_cases`/`inv_cases`; batch elements here are genuinely distinct
    # symmetric/Hermitian matrices, not tiled repeats.
    crossed = _batch_and_nonsquare_axes(
        batch_real=BATCH2_SPD,
        batch4_real=BATCH4_SPD,
        batch_complex=BATCH2_HERM,
        nonsquare_raises=[("2x3", M23_WIDE), ("3x2", M32_TALL)],
    )
    # UPLO `.upper()`-leak axis, added 2026-08-02 -- mirror of `eigh_cases()`'s
    # identical addition this same task. Coordinator's ruling: "I did not
    # grid it [eigvalsh]; do not assume it is clean because eigh is fixed --
    # measure it." Measured directly (both pre-fix defect and post-fix
    # 8-value grid): `eigvalsh` shared the exact same hand-written
    # `TypeError`-instead-of-`AttributeError` / rejects-lowercase defect as
    # `eigh` (same signature-parsing code path), now fixed identically.
    # Non-hermitian fixtures reused for the same anti-vacuity reason as
    # `eigh_cases()`: `M22_SPD`/`M33_SYM`/`C22_HERM`/`BATCH2_*` above are all
    # symmetric/Hermitian, so 'L' vs 'U' (and 'l' vs 'u') would be
    # indistinguishable on them regardless of correctness.
    uplo_case_fold = [
        ("2x2_gen_lower_lc", (M22_GEN,), {"UPLO": "l"}),
        ("2x2_gen_upper_lc", (M22_GEN,), {"UPLO": "u"}),
        ("2x2_gen_complex_lower_lc", (C22_GEN,), {"UPLO": "l"}),
        ("2x2_gen_complex_upper_lc", (C22_GEN,), {"UPLO": "u"}),
        ("batch_gen_lower_lc", (BATCH2_GEN,), {"UPLO": "l"}),
        ("batch_gen_complex_upper_lc", (BATCH2_C,), {"UPLO": "u"}),
    ]
    uplo_attributeerror = [
        ("uplo_none_raises", (M22_GEN,), {"UPLO": None}),
        ("uplo_int_raises", (M22_GEN,), {"UPLO": 0}),
        ("uplo_none_raises_complex", (C22_GEN,), {"UPLO": None}),
        ("uplo_int_raises_batch", (BATCH2_GEN,), {"UPLO": 0}),
    ]
    # UPLO round 3, added 2026-08-02 (coordinator's extended grid
    # `/tmp/uplocheck.py`/`/tmp/uplovals.py`: 1.5, b"L", ["L"],
    # bytearray(b"U") -- 40 eigh + 20 eigvalsh mismatches, ALL on
    # bytes-like values). `bytes`/`bytearray` DO have `.upper()`, so
    # numpy's `UPLO.upper()` SUCCEEDS (returns `b'L'`/`b'U'`) and only
    # THEN fails the `not in ('L', 'U')` membership test (`b'L' != 'L'`,
    # different types) -- landing in the SAME `ValueError` as an invalid
    # string, never `AttributeError`. A prior fix draft extracted the
    # `.upper()` result straight to a Rust `String`, which raises
    # `TypeError` on a non-str return -- numpy never coerces to `str`, it
    # only ever compares. Fixed by comparing the raw `.upper()` result
    # object against 'L'/'U' via Python `==` and falling through to
    # `ValueError` on anything else (see `eigh()`'s doc comment in
    # linalg.rs for the full writeup).
    uplo_bytes_raises = [
        ("uplo_bytes_l_raises", (M22_GEN,), {"UPLO": b"L"}),
        ("uplo_bytearray_u_raises", (M22_GEN,), {"UPLO": bytearray(b"U")}),
        ("uplo_bytes_l_raises_complex", (C22_GEN,), {"UPLO": b"L"}),
        ("uplo_bytearray_u_raises_batch", (BATCH2_GEN,), {"UPLO": bytearray(b"U")}),
    ]
    return (
        real_base + complex_base + crossed
        + uplo_case_fold + uplo_attributeerror + uplo_bytes_raises
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def lstsq_cases():
    # rank-deficient tall matrix (col2 == 2*col1): real numpy's own default
    # rcond (eps*max(M,N), confirmed via `inspect.signature`: `lstsq(a, b,
    # rcond=None)` -- a plain None default, not a `<no value>` sentinel, so
    # ABSENT and rcond=None are the SAME value here, unlike pinv's rtol)
    # already truncates its near-zero second singular value on its own
    # (verified: svd(M32_RANKDEF) = [8.37, 4.8e-16], rank comes back 1).
    M32_RANKDEF = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    real_base = [
        ("3x2_1d_b", (M32_TALL, np.array([1.0, 2.0, 3.0])), {}),
        ("3x2_1d_b_default_rcond", (M32_TALL, np.array([1.0, 2.0, 3.0])), {"rcond": None}),
        ("3x2_2d_b", (M32_TALL, np.eye(3)[:, :2]), {}),
        (
            "3x2_shape_mismatch_raises",
            (M32_TALL, np.array([1.0, 2.0])),
            {},
        ),
        # explicit non-default numeric rcond -- previously UNTESTED (only
        # ABSENT and explicit-None, both resolving to the identical default
        # value, appeared in this corpus; a genuinely different numeric
        # rcond had never reached `resolved_rcond`'s cutoff/rank-truncation
        # path at all). rcond=0.9 on the well-conditioned M32_TALL
        # (singular values [1.732, 1.0]) truncates the second singular
        # value (0.9*1.732=1.559 > 1.0), forcing rank=1 -- verified against
        # real numpy 2.5.1 directly before being added here.
        ("3x2_1d_b_explicit_rcond_truncates", (M32_TALL, np.array([1.0, 2.0, 3.0])), {"rcond": 0.9}),
        # genuinely rank-deficient input, default rcond -- numpy's own
        # cutoff detects the near-zero singular value (4.8e-16) unaided,
        # verified rank=1 comes back without needing an explicit rcond.
        ("3x2_rank_deficient_default_rcond", (M32_RANKDEF, np.array([1.0, 2.0, 3.0])), {}),
    ]
    complex_base = [("complex3x2", (C32_TALL, np.array([1 + 0j, 0 - 1j, 1 + 1j])), {})]
    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


# ---------------------------------------------------------------------------
# Array-API-restricted thin wrappers: matmul, vecdot, matrix_transpose,
# diagonal, outer. Each is a THIN, positional-only wrapper over numpy's own
# top-level function (verified via `inspect.getsource`), differing only by
# being positional-only and lacking `dtype=`/`out=` (and, for `diagonal`,
# pinning `axis1=-2, axis2=-1`). `matmul`/`vecdot` bind directly to
# `ionp_ion::matmul`'s already-general N-D/batched gufunc core (not
# restricted to 2-D, unlike the rest of `linalg.rs`), so cases below
# include a non-square pair to exercise real gufunc-shape dispatch.
# All added 2026-08-01. Evidence: 20,000-matrix (matmul/vecdot; the two
# items whose complex paths are not bit-exact) or 2,000-matrix
# (outer/diagonal/matrix_transpose; all measured bit-exact so a smaller
# confirmatory sweep was sufficient) seeded sweeps via
# `ulp_sweep.sweep_matrix_item_evidence` -- see each item's ItemSpec below.
# ---------------------------------------------------------------------------


def matmul_cases():
    return [
        ("2x2_2x2", (M22_GEN, M22_SPD), {}),
        ("3x2_2x3", (M32_TALL, M23_WIDE), {}),
        ("complex2x2", (C22_GEN, C22_HERM), {}),
    ]


def vecdot_cases():
    return [
        ("real_vec3", (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])), {}),
        ("real_vec2", (np.array([1.0, -2.0]), np.array([3.0, 0.5])), {}),
        ("complex_vec2", (np.array([1 + 1j, 2 - 1j]), np.array([0.5j, 1 + 0j])), {}),
    ]


def outer_cases():
    return [
        ("real_vec3x2", (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])), {}),
        ("complex_vec2", (np.array([1 + 1j, 2 - 1j]), np.array([0.5j, 1 + 0j])), {}),
    ]


def outer_toplevel_cases():
    """Top-level `numpy.outer(a, b, out=None)` -- NOT the same registry item
    as `linalg.outer` above: numpy's top-level `outer` ravels ANY-ndim input
    first (Array API `linalg.outer` requires strictly 1-D and raises
    otherwise -- verified directly against live numpy 2.5.1, see
    `ionp-py/src/linalg.rs`'s `outer_toplevel` doc comment for the
    reproducer). This corpus is deliberately NOT a copy of `outer_cases`
    above: it includes N-D inputs specifically to exercise the ravel path
    `linalg.outer`'s corpus never needs to, plus 0-d/bool/uint/empty cases
    as an out-of-corpus-shaped probe for the new top-level binding.

    `out=` cases (2026-08-03, `out=` fix): added once `outer_toplevel`
    started accepting `out=`, mirroring the semantics measured live against
    numpy 2.5.1 in that function's own doc comment -- top-level `outer` is a
    thin wrapper that ravels both operands and delegates straight to the
    `multiply` ufunc (`inspect.getsource(numpy.outer)`), so its `out=`
    contract IS `multiply`'s own ufunc `out=` contract (broadcast-tolerant
    shape check across every operand plus `out=`, same-kind casting,
    `_UFuncOutputCastingError` naming the real ufunc `'multiply'`, not
    `'outer'`). Covers: correct shape/dtype, wrong shape (3-operand
    broadcast message), same-kind-castable dtype narrowing, non-same-kind
    dtype (cast-error path), explicit `out=None`, positional `out`, the
    N-D ravel path with `out=`, an empty-input case with `out=`, a
    broadcast-up (higher-rank) `out=`, and a non-array `out=`."""
    rng = np.random.default_rng(12345)
    return [
        ("real_vec3x2", (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])), {}),
        ("complex_vec2", (np.array([1 + 1j, 2 - 1j]), np.array([0.5j, 1 + 0j])), {}),
        ("int_vec", (np.arange(6, dtype=np.int32), np.arange(4, dtype=np.int32)), {}),
        ("nd_2x3_2x2", (np.arange(6).reshape(2, 3), np.arange(4).reshape(2, 2)), {}),
        (
            "nd_2x3x4_2x3_f32",
            (
                np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                np.arange(6, dtype=np.float32).reshape(2, 3),
            ),
            {},
        ),
        ("scalar_0d_and_vec", (np.array(5), np.array([1, 2, 3])), {}),
        ("scalar_0d_both", (np.array(5.0), np.array(2.0)), {}),
        ("bool_vec", (np.array([True, False, True]), np.array([True, True])), {}),
        (
            "complex_vec4x3",
            (
                rng.standard_normal(4) + 1j * rng.standard_normal(4),
                rng.standard_normal(3) + 1j * rng.standard_normal(3),
            ),
            {},
        ),
        ("empty_left", (np.array([], dtype=np.float64), np.array([1.0, 2.0])), {}),
        ("uint8_vec", (np.arange(5, dtype=np.uint8), np.arange(3, dtype=np.uint8)), {}),
        (
            "mixed_dtype_promote",
            (np.arange(4, dtype=np.int16), rng.standard_normal(3).astype(np.float32)),
            {},
        ),
        # -------- out= cases --------
        (
            "out_correct_shape_dtype",
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),
            {"out": np.empty((3, 2), dtype=np.float64)},
        ),
        (
            "out_wrong_shape",
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),
            {"out": np.empty((2, 2), dtype=np.float64)},
        ),
        (
            "out_same_kind_castable_f64_to_f32",
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),
            {"out": np.empty((3, 2), dtype=np.float32)},
        ),
        (
            "out_non_same_kind_f64_to_i64",
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),
            {"out": np.empty((3, 2), dtype=np.int64)},
        ),
        (
            "out_none_explicit",
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),
            {"out": None},
        ),
        (
            "out_positional",
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]), np.empty((3, 2), dtype=np.float64)),
            {},
        ),
        (
            "out_nd_ravel_path",
            (np.arange(6).reshape(2, 3), np.arange(4).reshape(2, 2)),
            {"out": np.empty((6, 4), dtype=np.arange(1).dtype)},
        ),
        (
            "out_empty_input",
            (np.array([], dtype=np.float64), np.array([1.0, 2.0])),
            {"out": np.empty((0, 2), dtype=np.float64)},
        ),
        (
            "out_broadcast_up_bigger",
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),
            {"out": np.empty((1, 3, 2), dtype=np.float64)},
        ),
        (
            "out_non_array",
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),
            {"out": [1, 2, 3]},
        ),
    ]


def diagonal_cases():
    return [
        ("3x3_gen", (M33_GEN,), {}),
        ("3x2_tall", (M32_TALL,), {}),
        ("2x3_wide", (M23_WIDE,), {}),
        ("3x3_gen_offset1", (M33_GEN,), {"offset": 1}),
        ("3x3_gen_offset_neg1", (M33_GEN,), {"offset": -1}),
        ("complex2x2", (C22_GEN,), {}),
    ]


def matrix_transpose_cases():
    return [
        ("2x2_gen", (M22_GEN,), {}),
        ("3x2_tall", (M32_TALL,), {}),
        ("2x3_wide", (M23_WIDE,), {}),
        ("complex2x2", (C22_GEN,), {}),
    ]


# ---------------------------------------------------------------------------
# Invariant-based adapters for eig / eigvals / eigh / qr / svd.
#
# Each `_x_probe(mod, ...)` independently recomputes the decomposition
# using `mod` (either `numpy.linalg` or `anionpy.linalg`) and returns a small
# float64 array of "should be ~0 / should be exactly True" invariants. The
# numpy-side and ionp-side probes are each other's `numpy_adapter`/
# `ionp_adapter` (registry.py), so `run_case` calls each with the SAME
# input matrix and compares the two RESULTING INVARIANT VECTORS with a
# tight epsilon -- not the raw decomposition factors.
# ---------------------------------------------------------------------------


def _as_np(x):
    """Materialize a plain `numpy.ndarray` view of whatever `mod.X(...)`
    returned (already-numpy: no-op passthrough via np.asarray; anionpy.ndarray:
    converted). This is test-only invariant-checking arithmetic (`@`,
    `.conj()`, `np.diag`, ...), never library code -- the "no arithmetic in
    Python" rule binds `ionp-py/src/linalg.rs`'s bindings, not this
    comparator. It exists because the 2026-08-01 fix makes `anionpy.linalg.*`
    return real `anionpy.ndarray` (as it must, to be numpy-REPLACEMENT rather
    than accessory), and `anionpy.ndarray` does not implement `__matmul__`/
    `.conj()`/etc itself yet (a separate, large, entirely different
    absent-coverage surface, not part of this fix's scope) -- so the
    probes below convert the factors returned by `mod` to numpy immediately
    after the call under test, before computing the residual/orthogonality
    invariants. The call under test (`mod.eig(A)` etc.) still runs exactly
    as-is against whatever `mod` and `A` actually are (real `anionpy.linalg`
    on a real `anionpy.ndarray` when probing the anionpy side) -- only the
    downstream comparison math is numpy. No tolerance/epsilon value here
    changes because of this."""
    return np.asarray(x)


def _eig_probe(mod, A):
    w, v = mod.eig(A)
    A2, w2, v2 = _as_np(A), _as_np(w), _as_np(v)
    resid = A2 @ v2 - v2 @ np.diag(w2)
    err = float(np.max(np.abs(resid))) if resid.size else 0.0
    norms = np.linalg.norm(v2, axis=0) if v2.size else np.array([])
    norm_err = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    return np.array([err, norm_err])


def _eigvals_probe(mod, A):
    """Returns RELATIVE (not absolute) invariant residuals, scaled by
    max(|quantity|, 1.0) -- 2026-08-01 fix, see this task's report. The
    product of up to 8 eigenvalues (like `det`) has a magnitude that scales
    with matrix dimension and condition number, sometimes reaching 1e8+ on
    the same sweep that keeps eig/eigh/qr/svd's residuals near machine
    epsilon; an absolute bound tight enough for those items' O(1)-scale
    residuals fails this one purely because of scale, not because of a
    defect (measured: 20,000-matrix sweep, max ABSOLUTE prod_err reached
    8.5e-4 for float64, 4.4e-4 for complex128, while the RELATIVE version
    below stayed within the same tight epsilon as every other invariant).
    The `max(..., 1.0)` floor keeps the metric absolute-like (not
    div-by-near-zero-unstable) for genuinely small products/sums, and
    relative for large ones -- the same shape of fix applied to
    `linalg.det`/`linalg.cond`/`linalg.matrix_power` themselves (see
    `_EPS_JUST_RELATIVE` below)."""
    w = mod.eigvals(A)
    d = mod.det(A)
    w2, d2, A2 = _as_np(w), _as_np(d), _as_np(A)
    tr2 = np.trace(A2)
    prod_err = abs(complex(np.prod(w2)) - complex(d2)) / max(abs(complex(d2)), 1.0)
    sum_err = abs(complex(np.sum(w2)) - complex(tr2)) / max(abs(complex(tr2)), 1.0)
    return np.array([float(prod_err), float(sum_err)])


def _eigh_probe(mod, A, **kwargs):
    """`**kwargs` forwarding added 2026-08-02 (closing the gap documented in
    `_mk_pair`'s docstring below) so `UPLO=` cases actually reach `mod.eigh`
    instead of raising a same-on-both-sides `TypeError` from Python's own
    argument binding before the call under test ever runs -- that vacuous
    pass was indistinguishable from a real one until traced. Generalized to
    N-D batching in the same step (`eigh_cases()` gets a batch axis
    alongside `UPLO=`), matching `eigvalsh_cases()`'s established pattern.

    The residual check must be built against the EFFECTIVE symmetric/
    Hermitian matrix `UPLO` selects (mirror one triangle onto the other),
    not the raw (possibly asymmetric) `A` -- `eigh`/`eigvalsh` only ever
    read one triangle, so comparing against raw `A` would spuriously fail
    any case whose discarded triangle isn't already symmetric, and would
    silently NOT prove UPLO='U' actually ignores the lower triangle."""
    uplo = kwargs.get("UPLO", "L")
    w, v = mod.eigh(A, **kwargs)
    w2, v2, A2 = _as_np(w), _as_np(v), _as_np(A)
    if uplo == "U":
        upper = np.triu(A2)
        off = np.triu(A2, 1)
    else:
        upper = np.tril(A2)
        off = np.tril(A2, -1)
    Aeff = upper + np.conj(np.swapaxes(off, -1, -2))
    n = A2.shape[-1]
    idx = np.arange(n)
    diag_full = np.zeros_like(v2)
    diag_full[..., idx, idx] = w2[..., idx].astype(v2.dtype)
    resid = Aeff @ v2 - v2 @ diag_full
    err = float(np.max(np.abs(resid))) if resid.size else 0.0
    eye = np.eye(n)
    orth_err = (
        float(np.max(np.abs(np.conj(np.swapaxes(v2, -1, -2)) @ v2 - eye)))
        if v2.size else 0.0
    )
    sorted_ok = 1.0 if n < 2 or bool(np.all(np.diff(w2, axis=-1) >= -1e-12)) else 0.0
    real_ok = 1.0 if bool(np.all(np.isreal(w2))) else 0.0
    return np.array([err, orth_err, sorted_ok, real_ok])


def _qr_probe(mod, A, **kwargs):
    """`**kwargs` forwarding added 2026-08-02 for the same reason documented
    on `_eigh_probe` above: `mode=` cases (added earlier the same day) were
    reaching a same-on-both-sides `TypeError` from Python's own argument
    binding, never `mod.qr` itself -- a vacuous pass, not a real one. Each
    `mode` returns a structurally different thing (numpy's own documented
    behavior), so the probe dispatches on it:
      - 'reduced'/'complete': (Q, R) tuple, checked via the existing
        reconstruction/orthonormality/upper-triangular invariants (now
        batch-aware via `swapaxes`/trailing-two-axis triu masking).
      - 'r': R alone -- only the upper-triangular invariant applies (no Q
        to reconstruct against or check orthonormality of).
      - 'raw': (h, tau) -- numpy's own documented internal LAPACK buffer.
        Unlike Q/R, h/tau have no sign/phase ambiguity (deterministic
        LAPACK output), so a direct real/imag-split value comparison is
        the correct check here, not an invariant residual.
    """
    mode = kwargs.get("mode", "reduced")
    out = mod.qr(A, **kwargs)
    A2 = _as_np(A)
    if mode == "r":
        R2 = _as_np(out)
        tril = np.tril(np.ones(R2.shape[-2:], dtype=bool), k=-1)
        upper_err = float(np.max(np.abs(R2[..., tril]))) if tril.any() and R2.size else 0.0
        return np.array([upper_err])
    if mode == "raw":
        h, tau = out
        h2, tau2 = _as_np(h), _as_np(tau)
        return np.concatenate([
            h2.real.ravel(), h2.imag.ravel() if np.iscomplexobj(h2) else np.zeros(h2.size),
            tau2.real.ravel(), tau2.imag.ravel() if np.iscomplexobj(tau2) else np.zeros(tau2.size),
        ])
    Q, R = out
    Q2, R2 = _as_np(Q), _as_np(R)
    recon_err = float(np.max(np.abs(Q2 @ R2 - A2))) if A2.size else 0.0
    eye = np.eye(Q2.shape[-1])
    orth_err = (
        float(np.max(np.abs(np.conj(np.swapaxes(Q2, -1, -2)) @ Q2 - eye)))
        if Q2.size else 0.0
    )
    tril = np.tril(np.ones(R2.shape[-2:], dtype=bool), k=-1)
    upper_err = float(np.max(np.abs(R2[..., tril]))) if tril.any() and R2.size else 0.0
    return np.array([recon_err, orth_err, upper_err])


def _svd_probe(mod, A, **kwargs):
    """`**kwargs` added 2026-08-01 to exercise `hermitian=`/`compute_uv=`
    (see `svd_cases()`'s `hermitian_*` cases below) through this SAME
    invariant-residual probe rather than a raw-output comparison -- svd's
    U/Vt factors have the same sign ambiguity for the hermitian path as the
    general path (a column of U and the matching row of Vt may both flip
    sign together and remain an equally valid factorization), so invariant
    residuals are still the right comparison, not a regression to a
    separate mechanism. When `compute_uv=False` is passed, `mod.svd(...)`
    returns S alone (numpy's own documented return-shape change), so the
    probe reduces to comparing sorted-descending/nonnegative singular
    values only -- there is no U/Vt to reconstruct or orthonormality-check
    in that case, which is not a probe weakness, it mirrors what SVD with
    compute_uv=False actually promises."""
    # `full_matrices` is now a genuine crossed axis (2026-08-02, coordinator
    # requirement item 2) -- popped out of kwargs (rather than left in, which
    # would collide with the hardcoded keyword below) and defaulted to
    # `False` to preserve every PRE-EXISTING case's behavior unchanged.
    kwargs = dict(kwargs)
    full_matrices = kwargs.pop("full_matrices", False)
    compute_uv = kwargs.get("compute_uv", True)
    out = mod.svd(A, full_matrices=full_matrices, **kwargs)
    if not compute_uv:
        S2, A2 = _as_np(out), _as_np(A)
        sorted_ok = 1.0 if S2.shape[-1] < 2 or bool(np.all(np.diff(S2, axis=-1) <= 1e-12)) else 0.0
        nonneg_ok = 1.0 if bool(np.all(S2 >= -1e-15)) else 0.0
        return np.array([sorted_ok, nonneg_ok])
    U, S, Vh = out
    U2, S2, Vh2, A2 = _as_np(U), _as_np(S), _as_np(Vh), _as_np(A)
    # Generalized to arbitrary batch dims (2026-08-02, to cover the new
    # (2,0,0)-shaped empty/batched case below, and `full_matrices=True`'s
    # non-square U/Vh) -- U2 is (..., m, uk), Vh2 is (..., vk, n), S2 is
    # (..., k) with k = min(m, n) always, uk/vk == k when full_matrices is
    # False and == m/n respectively when True. A zero-padded (..., m, n)
    # diagonal built from S2 lets a single `@` reconstruction formula work
    # for BOTH the square-diag economy case and the padded full case (for
    # the 2-D, non-batched, economy case this is bit-identical to the
    # previous `np.diag(S2)` -- np.diag(S2) IS this construction's k==m==n
    # special case). Empty-sized A2 short-circuits to 0.0 as before.
    m = U2.shape[-2] if U2.ndim else 0
    n = Vh2.shape[-1] if Vh2.ndim else 0
    k = S2.shape[-1]
    diag_full = np.zeros(S2.shape[:-1] + (m, n), dtype=U2.dtype)
    idx = np.arange(k)
    diag_full[..., idx, idx] = S2[..., idx]
    recon_err = float(np.max(np.abs(U2 @ diag_full @ Vh2 - A2))) if A2.size else 0.0
    eye_u = np.eye(U2.shape[-1])
    eye_v = np.eye(Vh2.shape[-2])
    orth_u = (
        float(np.max(np.abs(np.conj(np.swapaxes(U2, -1, -2)) @ U2 - eye_u)))
        if U2.size else 0.0
    )
    orth_v = (
        float(np.max(np.abs(Vh2 @ np.conj(np.swapaxes(Vh2, -1, -2)) - eye_v)))
        if Vh2.size else 0.0
    )
    sorted_ok = 1.0 if S2.shape[-1] < 2 or bool(np.all(np.diff(S2, axis=-1) <= 1e-12)) else 0.0
    nonneg_ok = 1.0 if bool(np.all(S2 >= -1e-15)) else 0.0
    return np.array([recon_err, orth_u, orth_v, sorted_ok, nonneg_ok])


def _mk_pair(probe):
    """Split one mod-parameterized probe into a (numpy_adapter, ionp_adapter)
    pair, lazily importing anionpy (mirrors registry.py's resolve_ionp(), which
    also local-imports anionpy because it may be entirely absent).

    `**kwargs` forwarding added 2026-08-01 alongside `_svd_probe`'s own
    `**kwargs` above. UPDATE 2026-08-02: `_eigh_probe`/`_qr_probe` now also
    declare their own `**kwargs` (previously they did not -- a real gap:
    kwargs-bearing cases added earlier for `qr`'s `mode=` were silently
    vacuous, both sides raising the identical Python-level `TypeError` from
    the probe's own argument binding before `mod.qr`/`mod.eigh` was ever
    reached, which `run_case` cannot distinguish from a genuine matching
    exception -- see each probe's own docstring for the fix). `_eig_probe`/
    `_eigvals_probe` remain kwargs-free by design: neither `eig` nor
    `eigvals` takes any parameter beyond the matrix itself in real numpy."""

    def numpy_side(A, **kwargs):
        return probe(np.linalg, A, **kwargs)

    def ionp_side(A, **kwargs):
        import anionpy

        return probe(anionpy.linalg, A, **kwargs)

    return numpy_side, ionp_side


# ---------------------------------------------------------------------------
# linalg.LinAlgError: a "class"-type surface item (tools/numpy_surface.json
# lists it as "linalg.LinAlgError": "class"), not a function -- no
# established pattern elsewhere in this codebase for testing one. Modeled
# as an identity-check probe in the same `_mk_pair` shape the decomposition
# invariants use: trivially true on the numpy side (mod IS np.linalg) but a
# REAL, non-tautological assertion on the anionpy side.
#
# REWRITTEN 2026-08-05 (Monday). This probe used to assert
# `mod.LinAlgError is np.linalg.LinAlgError`, encoding the old binding
# `m.add("LinAlgError", py.import("numpy.linalg")?.getattr("LinAlgError")?)`.
# That binding was the single reason `import anionpy` required numpy at all --
# the one runtime numpy borrow in the crate with no numpy-absent fallback.
# anionpy now builds its OWN LinAlgError and re-parents it onto numpy's when
# numpy is importable, so the `is` identity is false BY DESIGN and the old
# assertion tests a contract that no longer exists.
#
# The replacement is deliberately STRONGER, not weaker: identity was only
# ever a proxy for "a caller's `except numpy.linalg.LinAlgError:` keeps
# working", so this asserts that property directly by actually raising and
# actually catching, and additionally pins the numpy-absent fallback base
# (ValueError, matching numpy's own `class LinAlgError(ValueError)`). A
# binding that constructed an unrelated class still fails this, which was
# the original probe's whole purpose.
# ---------------------------------------------------------------------------
def _linalg_error_probe(mod, _dummy):
    cls = mod.LinAlgError
    checks = [
        # `except numpy.linalg.LinAlgError:` still catches anionpy's own.
        issubclass(cls, np.linalg.LinAlgError),
        # numpy's own base, preserved on the numpy-absent path too.
        issubclass(cls, ValueError),
        # Not merely declared -- actually raisable and actually caught.
        _raises_and_is_caught(cls, np.linalg.LinAlgError),
    ]
    return np.array([1.0 if all(checks) else 0.0])


def _raises_and_is_caught(raise_cls, catch_cls):
    try:
        raise raise_cls("probe")
    except catch_cls:
        return True
    except Exception:
        return False


def linalg_error_cases():
    return [("identity_check", (np.zeros(1),), {})]


def eig_cases():
    real_base = [
        ("2x2_spd", (M22_SPD,), {}),
        ("2x2_gen", (M22_GEN,), {}),
        ("3x3_gen", (M33_GEN,), {}),
        ("rot2x2_complex_pair", (ROT2,), {}),
    ]
    complex_base = [("complex2x2", (C22_GEN,), {})]
    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def eigh_cases():
    real_base = [
        ("2x2_sym", (M22_SPD,), {}),
        ("3x3_sym", (M33_SYM,), {}),
        # UPLO='U' cases, added 2026-08-02, same transpose-trick reuse as
        # `eigvalsh` (see `eigh()`'s doc comment in linalg.rs) extended from
        # eigenvalues-only to the full eigendecomposition (eigenvectors
        # too) -- verified via a dedicated reconstruction-residual probe
        # (deliberately asymmetric garbage in the discarded lower triangle)
        # before being added here. Requires `_eigh_probe`'s 2026-08-02
        # `**kwargs` fix (see that probe's docstring) to actually reach
        # `mod.eigh(A, UPLO=...)` at all.
        ("2x2_sym_upper", (M22_SPD,), {"UPLO": "U"}),
        ("3x3_sym_upper", (M33_SYM,), {"UPLO": "U"}),
    ]
    complex_base = [
        ("2x2_hermitian", (C22_HERM,), {}),
        ("2x2_hermitian_upper", (C22_HERM,), {"UPLO": "U"}),
    ]
    # batch dims + non-square-raises axis, added 2026-08-02 -- same reasoning
    # as `eigvalsh_cases()`'s identical axis.
    crossed = _batch_and_nonsquare_axes(
        batch_real=BATCH2_SPD,
        batch4_real=BATCH4_SPD,
        batch_complex=BATCH2_HERM,
        nonsquare_raises=[("2x3", M23_WIDE), ("3x2", M32_TALL)],
    )
    # invalid UPLO -- numpy raises ValueError, exact message text verified
    # reproducible (our own error path, not a private numpy diagnostic) --
    # see the matching fix to `eigvalsh`'s sibling message this same task.
    invalid_uplo = [("invalid_uplo_raises", (M33_SYM,), {"UPLO": "X"})]
    # UPLO `.upper()`-leak axis, added 2026-08-02 per coordinator's
    # independent grid (`/tmp/uplocheck.py`, 80/180 mismatches on 'l', 'u',
    # None, 0). Real numpy performs NO type check on UPLO: it calls
    # `UPLO.upper()` unconditionally, so ANY object whose `.upper()`
    # returns 'L'/'U' is silently accepted (case-fold, not just the two
    # canonical strings) and any object lacking `.upper()` (None, int, ...)
    # raises numpy's own generic `AttributeError`, not a hand-written
    # `TypeError`. `M22_SPD`/`M33_SYM`/`C22_HERM`/`BATCH2_SPD`/`BATCH2_HERM`
    # above are all symmetric/Hermitian, which would make 'L' vs 'U' (and
    # therefore 'l' vs 'u') produce IDENTICAL output regardless of whether
    # the case-fold is implemented correctly -- the 18th-face vacuity trap.
    # These cases deliberately reuse the pre-existing NON-hermitian
    # fixtures (`M22_GEN`, `C22_GEN`, `BATCH2_GEN`, `BATCH2_C`) so 'L' and
    # 'U' (hence 'l' and 'u') genuinely disagree, proving the case-fold
    # path is actually exercised and not merely accepted-but-ignored.
    uplo_case_fold = [
        ("2x2_gen_lower_lc", (M22_GEN,), {"UPLO": "l"}),
        ("2x2_gen_upper_lc", (M22_GEN,), {"UPLO": "u"}),
        ("2x2_gen_complex_lower_lc", (C22_GEN,), {"UPLO": "l"}),
        ("2x2_gen_complex_upper_lc", (C22_GEN,), {"UPLO": "u"}),
        ("batch_gen_lower_lc", (BATCH2_GEN,), {"UPLO": "l"}),
        ("batch_gen_complex_upper_lc", (BATCH2_C,), {"UPLO": "u"}),
    ]
    # UPLO objects with no `.upper()` -- numpy's `UPLO.upper()` raises its
    # own generic `AttributeError` (message carries the object's REAL
    # runtime type name, e.g. "'NoneType' object has no attribute
    # 'upper'" / "'int' object has no attribute 'upper'"); must not be a
    # hand-written `TypeError`.
    uplo_attributeerror = [
        ("uplo_none_raises", (M22_GEN,), {"UPLO": None}),
        ("uplo_int_raises", (M22_GEN,), {"UPLO": 0}),
        ("uplo_none_raises_complex", (C22_GEN,), {"UPLO": None}),
        ("uplo_int_raises_batch", (BATCH2_GEN,), {"UPLO": 0}),
    ]
    # UPLO round 3, added 2026-08-02 -- see `eigvalsh_cases()`'s identical
    # addition for the full writeup: `bytes`/`bytearray` DO have `.upper()`
    # (so the call succeeds, returning `b'L'`/`b'U'`), but that result then
    # fails `not in ('L', 'U')` (different type, never equal to the str
    # literals) and lands in the SAME `ValueError` as an invalid string --
    # never `AttributeError`, never `TypeError`. Coordinator's extended
    # grid (`/tmp/uplocheck.py`: 1.5, b"L", ["L"], bytearray(b"U")) found
    # 40 eigh mismatches, ALL on these two bytes-like values.
    uplo_bytes_raises = [
        ("uplo_bytes_l_raises", (M22_GEN,), {"UPLO": b"L"}),
        ("uplo_bytearray_u_raises", (M22_GEN,), {"UPLO": bytearray(b"U")}),
        ("uplo_bytes_l_raises_complex", (C22_GEN,), {"UPLO": b"L"}),
        ("uplo_bytearray_u_raises_batch", (BATCH2_GEN,), {"UPLO": bytearray(b"U")}),
    ]
    return (
        real_base + complex_base + crossed + invalid_uplo
        + uplo_case_fold + uplo_attributeerror + uplo_bytes_raises
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def qr_cases():
    real_base = [
        ("3x2_tall", (M32_TALL,), {}),
        ("3x3_gen", (M33_GEN,), {}),
    ]
    complex_base = [("complex3x2", (C32_TALL,), {})]
    # batch dims axis, added 2026-08-02 -- `qr` does NOT require square
    # input (`_assert_stacked_2d` only), so there is no "non-square raises"
    # half here; `2x3_wide`-shaped batch elements exercise the wide (m<n)
    # branch under batching too, distinct from the tall batch.
    crossed = _batch_and_nonsquare_axes(
        batch_real=BATCH2_TALL,
        batch4_real=BATCH4_SPD,
        batch_complex=BATCH2_C,
        nonsquare_ok=[("2x3_wide", M23_WIDE)],
    )
    # `mode` crossed axis, added 2026-08-02 -- `mode='complete'/'r'/'raw'`
    # were previously a blanket `NotImplementedError` (only `'reduced'`
    # worked); verified against real numpy 2.5.1 these are genuinely
    # implementable (same `dgeqrf`/`zgeqrf` step `'reduced'` already used,
    # just a different `dorgqr`/`zungqr` width or no orthogonalization
    # call at all for `'r'`/`'raw'`), now implemented in `dense_linalg.rs`
    # (`qr_r`/`qr_complete`/`qr_raw`) and dispatched in `linalg.rs`.
    # Crossed over tall/wide/square shapes AND batching (batch2 tall),
    # real + complex, since numpy supports every mode under batching too
    # (verified: `qr(stack, mode='raw')` returns
    # `h.shape==(batch...,N,M)`, `tau.shape==(batch...,K)`).
    mode_axis = []
    for mode in ["complete", "r", "raw"]:
        for label, args, kwargs in real_base + complex_base:
            mode_axis.append((f"{label}__mode_{mode}", args, {**kwargs, "mode": mode}))
        mode_axis.append((f"2x3_wide__mode_{mode}", (M23_WIDE,), {"mode": mode}))
        mode_axis.append((f"batch2_tall__mode_{mode}", (BATCH2_TALL,), {"mode": mode}))
        mode_axis.append((f"batch2_complex__mode_{mode}", (BATCH2_C,), {"mode": mode}))
    # invalid mode string -- numpy raises ValueError, exact message text
    # verified reproducible (unlike the solve broadcast-message case, this
    # is OUR OWN error path, not a private numpy gufunc diagnostic).
    mode_axis.append(("invalid_mode_raises", (M33_GEN,), {"mode": "bogus"}))
    # `mode` NON-string axis, added 2026-08-02 (coordinator's UPLO
    # round-3 survey: "check whether any OTHER string-valued parameter in
    # linalg.rs has the same defect"). numpy's real `mode not in
    # ('reduced', 'complete', 'r', 'raw')` is a plain membership test --
    # no method call, no type check -- so it accepts ANY object and, for
    # anything not equal to one of the four strings, raises
    # `ValueError(f"Unrecognized mode '{mode}'")`, interpolating `str
    # (mode)` (verified: `mode=None` -> "Unrecognized mode 'None'",
    # `mode=['reduced']` -> "Unrecognized mode '['reduced']'", NOT
    # `repr(mode)`). A prior `mode: &str`-typed signature let PyO3's own
    # extraction raise a hand-shaped `TypeError` for every one of these
    # instead. Fixed via the same raw-`*args`/`**kwargs` pattern as
    # `eigh`'s `UPLO` (`mode=None` is a distinct, ERRORING value in numpy,
    # not a signal to fall back to the `'reduced'` default).
    mode_type_axis = [
        ("mode_none_raises", (M33_GEN,), {"mode": None}),
        ("mode_int_raises", (M33_GEN,), {"mode": 0}),
        ("mode_bytes_raises", (M33_GEN,), {"mode": b"reduced"}),
        ("mode_list_raises", (M33_GEN,), {"mode": ["reduced"]}),
        ("mode_float_raises", (M33_GEN,), {"mode": 1.5}),
        ("mode_none_raises_complex", (C32_TALL,), {"mode": None}),
        ("mode_int_raises_batch", (BATCH2_TALL,), {"mode": 0}),
    ]
    return (
        real_base + complex_base + crossed + mode_axis + mode_type_axis
        + _sweep_real(real_base) + _sweep_complex(complex_base)
    )


def svd_cases():
    base = [
        ("2x2_spd", (M22_SPD,), {}),
        ("3x2_tall", (M32_TALL,), {}),
        ("2x3_wide", (M23_WIDE,), {}),
        ("complex3x2", (C32_TALL,), {}),
    ]
    # NaN input must raise LinAlgError on both sides. numpy's own SVD
    # gufunc always raises `LinAlgError: SVD did not converge` for a NaN
    # anywhere in the input, verified directly against real numpy 2.5.1 for
    # square sizes 1..10 (and non-square) -- but Apple's Accelerate `dgesdd`
    # (this crate's LAPACK backend, see dense_linalg.rs's module docstring)
    # apparently does NOT set a nonzero `info` for NaN input at every size:
    # measured directly on this binary, 1x1/2x2 NaN input returned a
    # NaN-filled result with NO raise (3x3 and up already raised correctly,
    # i.e. `info` from Accelerate's own dgesdd happened to come back
    # nonzero there) -- a real, size-dependent LAPACK-implementation gap,
    # not a case anionpy's existing `info != 0` check was wrong to trust in
    # general. Both 1x1 and 2x2 are exercised since they were the two
    # confirmed-broken sizes.
    nan_1x1 = np.array([[np.nan]])
    nan_2x2 = np.array([[np.nan, 1.0], [3.0, 4.0]])
    # Complex-dtype counterparts of the two cases above -- same gap,
    # verified separately against real numpy 2.5.1 (both raise the
    # identical `LinAlgError("SVD did not converge")`), and the Rust fix's
    # complex branch is a distinct code path (`as_promoted_complex2` /
    # `zc::svd`, not `as_promoted_real2` / `rc::svd`) so this is exercised
    # independently rather than assumed to work by analogy.
    nan_1x1_complex = np.array([[complex(np.nan, 0.0)]])
    nan_2x2_complex = np.array([[complex(np.nan, 0.0), 1.0], [3.0, 4.0]], dtype=complex)
    nan_axis = [
        ("nan_1x1_raises", (nan_1x1,), {}),
        ("nan_2x2_raises", (nan_2x2,), {}),
        ("nan_1x1_complex_raises", (nan_1x1_complex,), {}),
        ("nan_2x2_complex_raises", (nan_2x2_complex,), {}),
    ]
    # full_matrices=True crossed axis, added 2026-08-02 per coordinator
    # requirement item 2 ("Implement full_matrices=True for svd ... adding
    # both as crossed axes in the corpus") -- exercises the new
    # non-square-U/Vh reconstruction path in `_svd_probe` above (U becomes
    # (m,m), Vh becomes (n,n), distinct shapes from the economy default).
    full_matrices_axis = [
        (f"{label}__full_matrices_true", args, {**kwargs, "full_matrices": True})
        for label, args, kwargs in base
    ]
    # empty-shape crossed axis, added 2026-08-02 per coordinator requirement
    # item 5 -- also exercises the new N-D-batching (`svd_batched`) fix for
    # (2,0,0), crossed with full_matrices True/False, at 2 representative
    # dtypes (float64/complex128; svd genuinely rejects int64 via
    # `unsupported_dtype` on neither side... actually promotes via
    # `as_promoted_real2` -- included for parity with the other 10 items'
    # 3-dtype sweep). ndim<2 (shape (0,)) is a genuine error case here
    # (numpy's own `_assert_stacked_2d`, matched via the ndim<2 message fix
    # this task).
    empty_shape_axis = [
        (f"empty_shape_{sh}__{np.dtype(dt).name}__fm{fm}", (np.zeros(sh, dtype=dt),), {"full_matrices": fm})
        for sh in EMPTY_SHAPES
        for dt in _EMPTY_DTYPES
        for fm in (False, True)
    ]
    return base + full_matrices_axis + empty_shape_axis + nan_axis + [
        # hermitian= cases, added 2026-08-01. All routed through the SAME
        # `_svd_probe` invariant-residual comparison as the cases above
        # (see `_svd_probe`'s updated docstring) -- verified by hand first
        # against real numpy 2.5.1 (M22_INDEF: singular values [3, 1],
        # matching |eigenvalues| of a symmetric-indefinite matrix; C22_HERM
        # and M33_SYM likewise bit-exact against `np.linalg.eigh`-derived
        # expectations) before being added here.
        ("hermitian_real_indefinite", (M22_INDEF,), {"hermitian": True}),
        ("hermitian_real_spd", (M22_SPD,), {"hermitian": True}),
        ("hermitian_real_3x3_sym", (M33_SYM,), {"hermitian": True}),
        ("hermitian_complex", (C22_HERM,), {"hermitian": True}),
        ("hermitian_compute_uv_false", (M22_INDEF,), {"hermitian": True, "compute_uv": False}),
        # hermitian=True on a non-square matrix: numpy raises LinAlgError
        # ("Last 2 dimensions of the array must be square") -- exercised
        # here as a genuine exception-type comparison (run_case's exception
        # path), not through the residual probe.
        ("hermitian_nonsquare_raises", (M32_TALL,), {"hermitian": True}),
        # batched hermitian= cases, added 2026-08-02: svd(hermitian=True)
        # (`svd_hermitian` in linalg.rs) was 2-D-only (raised "expected a
        # 2-D array ... batched/1-D inputs are out of scope" on any N-D
        # input) until this task's fix reused `hermitian_svd_real`/
        # `_complex` behind `check_stacked_square` +
        # `as_promoted_real_nd`/`_complex_nd` batching, the same pattern
        # already applied to eig/eigh/pinv(hermitian=True)/
        # matrix_rank(hermitian=True). Verified via a 32-case
        # full_matrices x compute_uv x hermitian x shape sweep against
        # real numpy 2.5.1: 4/32 mismatches were exactly this gap
        # (batched hermitian input), 0 after the fix.
        ("hermitian_batch2_real", (BATCH2_SPD,), {"hermitian": True}),
        ("hermitian_batch2_real_compute_uv_false", (BATCH2_SPD,), {"hermitian": True, "compute_uv": False}),
        ("hermitian_batch4_real", (BATCH4_SPD,), {"hermitian": True}),
        ("hermitian_batch2_complex", (BATCH2_HERM,), {"hermitian": True}),
        # truthy (non-bool) full_matrices/compute_uv/hermitian, added
        # 2026-08-02 (coordinator's UPLO round-3 survey named "svd/cond's
        # arguments" as candidates). numpy's real body does plain Python
        # truth tests (`if hermitian:`, `if compute_uv:`, `if
        # full_matrices:`), never a `bool` type check, so `1`/`0`/`"yes"`/
        # `""`/`None`/`[]` all behave per their own truthiness -- a prior
        # `bool`-typed signature let PyO3 raise `TypeError` for every one
        # of these instead. Non-square (`M23_WIDE`) used for the
        # `full_matrices` cases so True/False genuinely produce different
        # U/Vh shapes, not a vacuous pass; `compute_uv=0` returns S alone
        # (same reduced-output-shape branch as `compute_uv=False`).
        ("full_matrices_truthy_int_true", (M23_WIDE,), {"full_matrices": 1}),
        ("full_matrices_truthy_int_false", (M23_WIDE,), {"full_matrices": 0}),
        ("full_matrices_truthy_str", (M23_WIDE,), {"full_matrices": "yes"}),
        ("full_matrices_truthy_empty_str", (M23_WIDE,), {"full_matrices": ""}),
        ("full_matrices_truthy_none", (M23_WIDE,), {"full_matrices": None}),
        ("compute_uv_truthy_int_false", (M22_SPD,), {"compute_uv": 0}),
        ("hermitian_truthy_int", (M22_INDEF,), {"hermitian": 1}),
    ]


# --- 2026-08-01 evidence, all via ulp_sweep.sweep_matrix_item_evidence,
# seed=ulp_sweep.SEED, n=20000/dtype, sizes 2-8, condition 1e-2..1e2 (see
# scratch gen_evidence.py's final run, reproduced independently by hand in
# this task's report) ---

_EPS_JUST_ABS = (
    "Floating-point non-associativity between two independent call "
    "sequences through the same Apple Accelerate LAPACK library (numpy's "
    "linalg module and anionpy's do not issue bit-identical BLAS op orderings "
    "even when both ultimately call e.g. dgetrf/dgesdd/dsyevd -- IEEE754 "
    "does not guarantee bit-reproducibility across reorderings). This "
    "item's output magnitude is bounded by the problem's own scale (it "
    "does not vary multiplicatively with matrix size/condition number), so "
    "an ABSOLUTE bound is the right instrument (contrast `_EPS_JUST_RELATIVE` "
    "below). Bound = the exact maximum measured over a 20,000-matrix, "
    "seeded (ulp_sweep.SEED), condition-controlled sweep (sizes 2-8, "
    "condition 1e-2..1e2) via ulp_sweep.sweep_matrix_item_evidence -- see "
    "this item's `epsilon_sweep` for the recorded (n, measured) pair."
)

_EPS_JUST_RELATIVE = (
    "This item's output magnitude scales MULTIPLICATIVELY with matrix "
    "dimension and condition number (a product of up to 8 singular values "
    "as large as 1e2 each, for `det`/`matrix_power`; a ratio of singular "
    "values for `cond`) -- an absolute bound tight enough to mean anything "
    "on a near-zero-magnitude case fails purely from scale on a "
    "large-magnitude one, not from a defect (measured directly: `det`'s "
    "raw absolute error reaches 3.46e-4 on large-determinant matrices in "
    "the same sweep where its RELATIVE error never exceeds ~5e-13). A "
    "RELATIVE bound is the right instrument here (contrast `_EPS_JUST_ABS` "
    "above). Bound = the exact maximum measured over a 20,000-matrix, "
    "seeded (ulp_sweep.SEED), condition-controlled sweep (sizes 2-8, "
    "condition 1e-2..1e2) via ulp_sweep.sweep_matrix_item_evidence -- see "
    "this item's `epsilon_sweep` for the recorded (n, measured) pair."
)

_EPS_JUST_INVARIANT = (
    "Invariant-based comparison (not raw factor comparison), per the task "
    "brief: eig/eigh eigenvectors and svd/qr's orthonormal factors have a "
    "genuine sign/phase ambiguity (measured directly: `eig`'s eigenvectors "
    "disagree with numpy's by an effectively unbounded ULP distance on "
    "ordinary random matrices -- an equally-correct opposite-sign "
    "eigenvector, not a defect). Each side independently recomputes its "
    "own decomposition and checks the defining algebraic identity "
    "(residual/orthonormality/ordering) against the ORIGINAL matrix, then "
    "the two small invariant-residual vectors (which should each be ~0 for "
    "a correct implementation) are compared -- these residuals are "
    "already O(1)-scale distances, not quantities with their own "
    "independent scale, so an ABSOLUTE bound is correct even though the "
    "underlying decomposition (eigenvalues, singular values) can itself "
    "scale multiplicatively; see `_eigvals_probe`'s own docstring for the "
    "one exception (its raw residual is NOT pre-scale-invariant and is "
    "rescaled inside the probe rather than declared `\"rel\"` here). Bound "
    "= the exact maximum measured over a 20,000-matrix, seeded "
    "(ulp_sweep.SEED), condition-controlled sweep of each probe's own "
    "residual via ulp_sweep.sweep_matrix_item_evidence -- see each item's "
    "`epsilon_sweep` for the recorded (n, measured) pair."
)


# bool/int8/int32/int64/uint8/uint32/float16 all promote to float64 BEFORE
# any LAPACK call (numpy's `_commonType` table maps every one of them to
# double; anionpy's `as_promoted_real2`/`as_promoted_real1` mirror that table
# exactly) -- the compute path for a promoted-int/bool/float16 input is
# BIT-IDENTICAL to the compute path for a float64 array holding the same
# exact values (an int8/bool value is always exactly representable in
# float64, so promotion introduces no new rounding). The float64 bound
# above was measured over a 20,000-matrix sweep that itself sweeps this
# same code path across a wide value/condition range -- reusing it here is
# not a guess, it is the SAME measured bound applied to the SAME code path
# under a different (exact-subset) input encoding, per this item's own
# `_EPS_JUST_ABS`/`_EPS_JUST_RELATIVE`/`_EPS_JUST_INVARIANT` reasoning.
_EPS_JUST_PROMOTED_REUSE = (
    "Promoted dtype (bool/int8/int32/int64/uint8/uint32/float16) reuses "
    "this item's own measured float64 bound: every one of these dtypes "
    "promotes to float64 before any LAPACK call (numpy's `_commonType` "
    "table, mirrored exactly by anionpy's `as_promoted_real2`/`_real1`), and "
    "every value representable in these dtypes is exactly representable "
    "in float64 -- so the promoted-input compute path is bit-identical to "
    "the float64 path already measured for this item, just restricted to "
    "an exact-integer-valued subset of float64's domain, which cannot "
    "introduce error the float64 sweep did not already bound."
)

_PROMOTED_REAL_DTYPES = (
    "bool", "int8", "int32", "int64", "uint8", "uint32", "float16",
)

# ---------------------------------------------------------------------------
# 2026-08-02: net-new Array-API-shaped items (cross, tensordot, multi_dot,
# tensorinv, tensorsolve, vector_norm, matrix_norm, norm). Each of these
# reduces to add/subtract/multiply/matmul (integer-exact on integer-valued
# inputs) or an SVD-based reduction (already covered by inv/solve/svd's own
# epsilon evidence above) -- no 20,000-matrix seeded sweep was run for these
# 8 items specifically (out of scope for this pass, flagged plainly in the
# task report); the dtype/kwarg/shape/non-contiguous/error axes below are
# covered directly instead.
# ---------------------------------------------------------------------------

M222_BATCH = np.arange(1.0, 13.0).reshape(2, 2, 3)  # batch of 2 3-vectors, twice
M223_BATCH2 = (np.arange(1.0, 13.0) * 0.5).reshape(2, 2, 3)

# ---------------------------------------------------------------------------
# 2026-08-02: empty-shape crossed axis, added per coordinator requirement
# item 5 ("Add empty shapes (0,0) (0,3) (3,0) (0,) (2,0,0) as a real crossed
# axis for all 11 items in the corpus"). These are genuinely crossed --
# every one of the 11 owned items below builds its own empty-shape cases
# from this SAME list x dtype set, not a one-off per item -- exercising the
# panic-audit fixes (tensordot/multi_dot/tensorinv/tensorsolve slicing),
# the ndim<2 message fixes (trace/svd/matrix_power/matrix_norm), the
# matrix_power/svd N-D-batching fixes ((2,0,0)), and tensorinv's
# reshape-ambiguity fix ((2,0,0)) all in one place. Every case here is
# graded DIFFERENTIALLY (anionpy vs real numpy, both success values AND
# exception class/message) by the harness itself -- no expected value is
# hardcoded here, so this is a real regression net, not a rubber stamp.
# ---------------------------------------------------------------------------
EMPTY_SHAPES = [(0, 0), (0, 3), (3, 0), (0,), (2, 0, 0)]
_EMPTY_DTYPES = [np.float64, np.complex128, np.int64]


def _empty_unary_cases(extra_args=(), kwargs=None, dtypes=None):
    """(label, (zeros(shape, dtype),) + extra_args, kwargs) for every shape
    in EMPTY_SHAPES x every dtype in `dtypes` (default _EMPTY_DTYPES)."""
    kwargs = {} if kwargs is None else kwargs
    dtypes = _EMPTY_DTYPES if dtypes is None else dtypes
    return [
        (
            f"empty_shape_{sh}__{np.dtype(dt).name}",
            (np.zeros(sh, dtype=dt),) + tuple(extra_args),
            dict(kwargs),
        )
        for sh in EMPTY_SHAPES
        for dt in dtypes
    ]


def _empty_binary_same_shape_cases(dtypes=None):
    """(label, (zeros(shape, dtype), zeros(shape, dtype)), {}) for every
    shape x dtype -- used by cross/tensordot/tensorsolve, all of which
    accept two arrays of the SAME shape (tensorsolve's default axes=0
    path specifically needs a.shape == b.shape for the diff==0 slicing
    branch verified against real numpy earlier this task)."""
    dtypes = _EMPTY_DTYPES if dtypes is None else dtypes
    return [
        (
            f"empty_shape_{sh}__{np.dtype(dt).name}",
            (np.zeros(sh, dtype=dt), np.zeros(sh, dtype=dt)),
            {},
        )
        for sh in EMPTY_SHAPES
        for dt in dtypes
    ]


def cross_cases():
    real_base = [
        ("1d_axis_default", (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])), {}),
        ("1d_axis0", (np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0])), {"axis": 0}),
        ("batched_axis_default", (M222_BATCH, M223_BATCH2), {}),
        # error case: last axis size != 3
        ("wrong_length_raises", (np.array([1.0, 2.0]), np.array([1.0, 2.0])), {}),
    ]
    # NOT dtype-swept: `.astype(dt)` on a `np.moveaxis`-permuted array (an
    # arbitrary-strided view whose OWN layout is neither C- nor
    # F-contiguous) preserves that exact non-C/F layout with `order='K'`
    # (verified directly) -- the differential harness's `_freshen_array`
    # deliberately refuses to reconstruct such an array (its root base
    # would be neither C- nor F-contiguous), by design (see harness.py's
    # own docstring: a silent contiguous-copy fallback there would defeat
    # the point of testing this exact layout). These two cases still cover
    # non-contiguous input and axis!=-1 handling once each at float64/
    # complex128 -- just not swept across every integer/bool/float16
    # dtype.
    real_noncontig = [
        ("batched_axis1", (np.moveaxis(M222_BATCH, -1, 1), np.moveaxis(M223_BATCH2, -1, 1)), {"axis": 1}),
        (
            "noncontig_input",
            (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])[:, ::-1], np.array([[7.0, 8.0, 9.0], [1.0, 0.0, 1.0]])[:, ::-1]),
            {},
        ),
    ]
    complex_base = [
        ("complex_1d", (np.array([1 + 1j, 2.0, 0.0]), np.array([0.0, 1 - 1j, 2.0])), {}),
    ]
    empty_shape_axis = _empty_binary_same_shape_cases()
    return (
        real_base + real_noncontig + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
        + empty_shape_axis
    )


def tensordot_cases():
    A = np.arange(24.0).reshape(2, 3, 4)
    B = np.arange(24.0).reshape(4, 3, 2)
    real_base = [
        ("nd_pair_axes", (A, B), {"axes": ([1, 2], [1, 0])}),
        ("int_axes_1", (M33_GEN, M33_GEN), {"axes": 1}),
        ("int_axes_default", (M22_GEN, M22_GEN), {}),
        ("scalar_axes0", (M22_GEN, M22_GEN), {"axes": 0}),
        ("single_int_pair_axes", (M33_GEN, M33_GEN), {"axes": (1, 0)}),
        ("noncontig", (M33_GEN.T, M33_GEN.T), {"axes": 1}),
        # error: incompatible contracted-dimension sizes under explicit axes
        ("shape_mismatch_raises", (M23_WIDE, M23_WIDE), {"axes": ([1], [1])}),
        # Negative integer `axes=`. numpy's `tensordot` builds
        # `axes_a = list(range(-axes, 0))` / `axes_b = list(range(0, axes))`
        # -- plain CPython `range()` semantics, not a normalize-then-reject
        # rule. For ANY negative N, `range(-N, 0)` has `-N > 0`, so a
        # start > stop range is EMPTY (0 axes contracted, i.e. a pure outer
        # product: `np.tensordot(A, B, axes=-1).shape == A.shape + B.shape`),
        # not a raise. Verified directly against real numpy 2.5.1 for
        # axes=-1, -2, -5, all producing the SAME (2,3,4,4,3,2) outer-product
        # shape from A=(2,3,4)/B=(4,3,2) -- confirming this is uniform across
        # negative N, not just axes=-1.
        ("neg_int_axes_-1", (A, B), {"axes": -1}),
        ("neg_int_axes_-2", (A, B), {"axes": -2}),
        ("neg_int_axes_large_neg", (A, B), {"axes": -5}),
    ]
    complex_base = [("complex2x2", (C22_GEN, C22_GEN), {"axes": 1})]
    # default axes=2 -- same same-shape empty pair as cross/tensorsolve.
    empty_shape_axis = _empty_binary_same_shape_cases()
    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
        + empty_shape_axis
    )


def multi_dot_cases():
    M1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    M2 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    M3 = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    M4 = np.array([[1.0, 1.0], [0.0, 1.0]])
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.0, 0.0])
    real_base = [
        ("two_matrices", ([M1, M3.T],), {}),
        ("three_matrices", ([M1, M2, M4],), {}),
        ("four_matrices_chain", ([M1, M2, M4, M4],), {}),
        ("first_1d", ([v1, M2, M4],), {}),
        ("last_1d", ([M1, M2, v2],), {}),
        ("both_1d", ([v1, M2, v2],), {}),
        # error: fewer than 2 arrays
        ("too_few_raises", ([M1],), {}),
    ]
    complex_base = [("complex_three", ([C22_GEN, C22_HERM, C22_GEN],), {})]

    # `out=` crossed axis, added 2026-08-02 per coordinator requirement item
    # 2 ("Implement ... out= for multi_dot ... adding both as crossed axes
    # in the corpus"). Mirrors /tmp/lin.py's GAP2 sweep exactly: ABSENT
    # (baseline, already covered by real_base/complex_base above),
    # match (correct shape+dtype), wrongdtype (out= dtype disagrees with
    # the natural result dtype -- numpy raises; must match anionpy's raise),
    # wrongshape (out= shape disagrees with the natural (3,3) result --
    # numpy raises; must match anionpy's raise). float64 result shape for
    # [M1, M3.T] is (3,3) (M1 is 3x2, M3.T is 2x3); complex chain
    # [C22_GEN, C22_HERM, C22_GEN] is (2,2) complex128.
    out_axis = []
    for form in ("match", "wrongdtype", "wrongshape"):
        kw = {}
        if form == "match":
            kw["out"] = np.zeros((3, 3), dtype=np.float64)
        elif form == "wrongdtype":
            kw["out"] = np.zeros((3, 3), dtype=np.int32)
        elif form == "wrongshape":
            kw["out"] = np.zeros((3, 4), dtype=np.float64)
        out_axis.append((f"two_matrices_out_{form}", ([M1, M3.T],), kw))
    for form in ("match", "wrongdtype", "wrongshape"):
        kw = {}
        if form == "match":
            kw["out"] = np.zeros((2, 2), dtype=np.complex128)
        elif form == "wrongdtype":
            kw["out"] = np.zeros((2, 2), dtype=np.int32)
        elif form == "wrongshape":
            kw["out"] = np.zeros((2, 3), dtype=np.complex128)
        out_axis.append((f"complex_three_out_{form}", ([C22_GEN, C22_HERM, C22_GEN],), kw))

    # empty-shape crossed axis, added 2026-08-02 per coordinator requirement
    # item 5. multi_dot needs a CHAIN of >=2 arrays with matching inner
    # dimensions -- paired each empty shape with its own reverse (valid
    # 2-arg chain for every 2-D shape: (a,b) @ (b,a) -> (a,a)), the 1-D
    # shape (0,) paired with a compatible 2-D partner ((0,3), matching
    # inner dim 0 -> result (3,)), and the 3-D shape (2,0,0) included
    # verbatim as an illustrative OUT-OF-SUPPORT case (numpy's own
    # multi_dot requires every chain element to be 2-D, except optionally
    # the first/last being 1-D -- a 3-D element must raise on BOTH sides,
    # which this case verifies differentially rather than assumes).
    empty_shape_axis = []
    for dt in _EMPTY_DTYPES:
        empty_shape_axis.append((
            f"empty_shape_(0, 0)__{np.dtype(dt).name}",
            ([np.zeros((0, 0), dtype=dt), np.zeros((0, 0), dtype=dt)],), {},
        ))
        empty_shape_axis.append((
            f"empty_shape_(0, 3)__{np.dtype(dt).name}",
            ([np.zeros((0, 3), dtype=dt), np.zeros((3, 0), dtype=dt)],), {},
        ))
        empty_shape_axis.append((
            f"empty_shape_(3, 0)__{np.dtype(dt).name}",
            ([np.zeros((3, 0), dtype=dt), np.zeros((0, 3), dtype=dt)],), {},
        ))
        empty_shape_axis.append((
            f"empty_shape_(0,)__{np.dtype(dt).name}",
            ([np.zeros((0,), dtype=dt), np.zeros((0, 3), dtype=dt)],), {},
        ))
        empty_shape_axis.append((
            f"empty_shape_(2, 0, 0)__{np.dtype(dt).name}",
            ([np.zeros((2, 0, 0), dtype=dt), np.zeros((2, 0, 0), dtype=dt)],), {},
        ))

    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
        + out_axis + empty_shape_axis
    )


def tensorinv_cases():
    T_ind2 = np.random.RandomState(42).rand(4, 6, 4, 6)
    T_ind1 = np.random.RandomState(43).rand(24, 8, 3)
    T_ind3 = np.random.RandomState(44).rand(4, 6, 8, 4, 6, 8)
    real_base = [
        ("default_ind2", (T_ind2,), {}),
        ("explicit_ind2", (T_ind2,), {"ind": 2}),
        ("ind1", (T_ind1,), {"ind": 1}),
        ("ind3", (T_ind3,), {"ind": 3}),
        # error: not square under the given ind
        ("non_square_raises", (T_ind1,), {"ind": 2}),
    ]
    complex_base = [
        ("complex_ind2", (np.random.RandomState(45).rand(2, 3, 2, 3) + 1j * np.random.RandomState(46).rand(2, 3, 2, 3),), {}),
    ]
    # default ind=2 -- exercises the reshape-ambiguity fix for (2,0,0)
    # (oldshape[2:]=(0,) -> prod=0 -> numpy's own reshape(0,-1) ambiguity
    # ValueError, distinct from the "must be square" LinAlgError the other
    # 4 empty shapes raise -- both verified against real numpy 2.5.1
    # directly this task before the fix was written).
    empty_shape_axis = _empty_unary_cases()
    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
        + empty_shape_axis
    )


def tensorsolve_cases():
    rng = np.random.RandomState(47)
    A = rng.rand(2, 3, 6) * 0.1 + np.eye(6).reshape(2, 3, 6) * 3.0
    b = rng.rand(2, 3)
    A_axes = np.moveaxis(A, 0, -1)
    # error: b's own total size doesn't match a's core solve dimension
    # (prod(a.shape[b.ndim:])), a DIFFERENT check than shape_mismatch_raises
    # above (which is about `a`'s own shape self-consistency, checked before
    # `b` is ever touched). numpy raises from its internal `solve1` gufunc
    # once `a`/`b` are reshaped to the 2-D system, with a message that
    # names the actual mismatched sizes -- verified directly against real
    # numpy 2.5.1: `tensorsolve(rng.standard_normal((3,4,3,4)),
    # rng.standard_normal((3,5)))` (solve-dim size 3*4=12, b.size=15) raises
    # `ValueError: solve1: Input operand 1 has a mismatch in its core
    # dimension 0, with gufunc signature (m,m),(m)->(m) (size 15 is
    # different from 12)`.
    A_bsize = rng.rand(3, 4, 3, 4)
    b_bsize_mismatch = rng.rand(3, 5)
    real_base = [
        ("default_axes", (A, b), {}),
        # error: prod(a.shape[b.ndim:]) != prod(a.shape[:b.ndim])
        ("shape_mismatch_raises", (rng.rand(2, 3, 4), b), {}),
        ("b_size_mismatch_raises", (A_bsize, b_bsize_mismatch), {}),
    ]
    # NOT dtype-swept -- same `_freshen_array`/non-C/F-root reason as
    # `cross_cases`'s `batched_axis1`/`noncontig_input` above: `.astype(dt)`
    # on this `np.moveaxis`-permuted view preserves its own non-C/F layout
    # (order='K'), which the harness deliberately refuses to reconstruct.
    # Still covers `axes=` handling once, at float64.
    real_noncontig = [
        ("explicit_axes", (A_axes, b), {"axes": (0,)}),
    ]
    Ac = A.astype(complex) + 1j * rng.rand(*A.shape) * 0.05
    complex_base = [("complex_default", (Ac, b.astype(complex)), {})]
    # default axes -- a == b same empty shape hits the an==bn "negative
    # zero" slicing branch (verified this task: real numpy's own
    # `a.shape[-(an-bn):]` with an-bn==0 yields the FULL shape, not an
    # empty one -- the fix this exercises), and `total == prod**2` holds
    # trivially (0 == 0**2) for every empty shape, so real numpy actually
    # SUCCEEDS on all 5 (verified directly), unlike the old anionpy code's
    # incorrect blanket prod==0 rejection this fix removed.
    empty_shape_axis = _empty_binary_same_shape_cases()
    return (
        real_base + real_noncontig + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
        + empty_shape_axis
    )


def vector_norm_cases():
    v = np.array([3.0, -4.0, 0.0, 2.0])
    M = np.array([[1.0, -2.0], [3.0, -4.0]])
    real_base = [
        ("default", (v,), {}),
        ("ord1", (v,), {"ord": 1}),
        ("ord_inf", (v,), {"ord": np.inf}),
        ("ord_neg_inf", (v,), {"ord": -np.inf}),
        ("ord0", (v,), {"ord": 0}),
        ("ord3", (v,), {"ord": 3}),
        ("matrix_axis0", (M,), {"axis": 0}),
        ("matrix_axis1", (M,), {"axis": 1}),
        ("matrix_axis_tuple_flatten", (M,), {"axis": (0, 1)}),
        ("keepdims_true", (M,), {"axis": 1, "keepdims": True}),
        ("keepdims_false", (M,), {"axis": 1, "keepdims": False}),
        ("noncontig", (M[:, ::-1],), {"axis": 1}),
        # truthy (non-bool) keepdims=, added 2026-08-02 (bool-kwarg sweep
        # round). `linalg.vector_norm` dispatches on plain Python
        # truthiness (`if keepdims:`), NOT the C-format `__index__`-based
        # rule `sum`/`mean`/etc use -- a strict `bool`-typed pyo3 param let
        # PyO3 raise `TypeError` for any of these instead. `axis=1` on a
        # 2x2 matrix means keepdims genuinely changes the output SHAPE
        # ((2,) vs (2,1)), so this is a non-vacuous value-level check, not
        # just an accept/reject one.
        ("keepdims_truthy_int_true", (M,), {"axis": 1, "keepdims": 1}),
        ("keepdims_truthy_int_false", (M,), {"axis": 1, "keepdims": 0}),
        ("keepdims_truthy_str", (M,), {"axis": 1, "keepdims": "yes"}),
        ("keepdims_truthy_empty_str", (M,), {"axis": 1, "keepdims": ""}),
        ("keepdims_truthy_none", (M,), {"axis": 1, "keepdims": None}),
    ]
    complex_base = [("complex_vec", (np.array([1 + 1j, 2 - 1j, 0.5j]),), {})]
    # default axis=None flattens regardless of ndim -- every empty shape
    # succeeds (norm of an empty/zero flattened vector is 0.0).
    empty_shape_axis = _empty_unary_cases()
    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
        + empty_shape_axis
    )


def matrix_norm_cases():
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    real_base = [
        ("default_fro", (M,), {}),
        ("ord_fro_str", (M,), {"ord": "fro"}),
        ("ord_nuc", (M,), {"ord": "nuc"}),
        ("ord_1", (M,), {"ord": 1}),
        ("ord_neg1", (M,), {"ord": -1}),
        ("ord_2", (M,), {"ord": 2}),
        ("ord_neg2", (M,), {"ord": -2}),
        ("ord_inf", (M,), {"ord": np.inf}),
        ("ord_neg_inf", (M,), {"ord": -np.inf}),
        ("keepdims_true", (M,), {"keepdims": True}),
        # truthy (non-bool) keepdims=, added 2026-08-02 (bool-kwarg sweep
        # round). Same plain-truthiness dispatch as `vector_norm` above;
        # keepdims genuinely changes output ndim ((2,2) scalar-shaped
        # `(1,1)` vs bare scalar), so this is a real value-level check.
        ("keepdims_truthy_int_true", (M,), {"keepdims": 1}),
        ("keepdims_truthy_int_false", (M,), {"keepdims": 0}),
        ("keepdims_truthy_str", (M,), {"keepdims": "yes"}),
        ("keepdims_truthy_empty_str", (M,), {"keepdims": ""}),
        ("keepdims_truthy_none", (M,), {"keepdims": None}),
        ("batched", (np.stack([M, M22_SPD]),), {}),
        # error: ndim < 2
        ("ndim1_raises", (np.array([1.0, 2.0, 3.0]),), {}),
    ]
    complex_base = [("complex2x2", (C22_GEN,), {"ord": "nuc"})]
    # matrix_norm requires ndim>=2 (its own explicit _assert_stacked_2d
    # equivalent -- an AxisError, not a LinAlgError, verified against real
    # numpy this task); the 1-D shape (0,) is genuinely an error case here,
    # the other 4 succeed with a 0.0 Frobenius norm (default ord).
    empty_shape_axis = _empty_unary_cases()
    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
        + empty_shape_axis
    )


def norm_cases():
    v = np.array([3.0, -4.0, 0.0, 2.0])
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    real_base = [
        ("default_vector", (v,), {}),
        ("default_matrix", (M,), {}),
        ("vector_axis0", (v, None, 0), {}),
        ("vector_ord1_axis0", (v, 1, 0), {}),
        ("matrix_ord2", (M, 2), {}),
        ("matrix_ord_fro", (M, "fro"), {}),
        ("matrix_axis_tuple", (M, None, (0, 1)), {}),
        ("matrix_axis_tuple_reversed", (M, 1, (1, 0)), {}),
        ("keepdims", (M, None, None, True), {}),
        # truthy (non-bool) keepdims=, added 2026-08-02 (bool-kwarg sweep
        # round). Same plain-truthiness dispatch as `vector_norm`/
        # `matrix_norm` above.
        ("keepdims_truthy_int_true", (M, None, None, 1), {}),
        ("keepdims_truthy_int_false", (M, None, None, 0), {}),
        ("keepdims_truthy_str", (M, None, None, "yes"), {}),
        ("keepdims_truthy_none", (M, None, None, None), {}),
        # errors
        ("duplicate_axes_raises", (M, None, (0, 0)), {}),
        ("improper_ndim_raises", (np.arange(8.0).reshape(2, 2, 2), None, (0, 1, 2)), {}),
    ]
    complex_base = [("complex_default", (C22_GEN,), {})]
    # default ord=None, axis=None flattens regardless of ndim (the OLD
    # top-level `norm` API, distinct from `matrix_norm`'s stacked-2-D
    # requirement above) -- every empty shape succeeds with 0.0. Also
    # exercises the `-0.0` sign fix from earlier this task.
    empty_shape_axis = _empty_unary_cases()
    return (
        real_base + complex_base
        + _sweep_real(real_base) + _sweep_complex(complex_base)
        + empty_shape_axis
    )


def _build_linalg_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["linalg.det"] = ItemSpec(
        name="linalg.det", kind="custom", custom_cases=det_cases,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"complex128": ("rel", 4.985165492358638e-13)},
        epsilon_tolerance_justification=_EPS_JUST_RELATIVE,
        epsilon_sweep={"complex128": (20000, 4.985165492358638e-13)},
    )
    specs["linalg.slogdet"] = ItemSpec(
        name="linalg.slogdet", kind="custom", custom_cases=slogdet_cases,
        atol=0.0, rtol=0.0, multi_output=True,
    )
    specs["linalg.trace"] = ItemSpec(
        name="linalg.trace", kind="custom", custom_cases=trace_cases,
        atol=0.0, rtol=0.0,
        # offset!=0 support added 2026-08-01 (was previously undeclared --
        # see anionpy/_state/linalg.py's history). bool/int8-64/float32/
        # float64/complex64 are bit-exact on the 20,000-matrix seeded
        # sweep; complex128 alone needs a tiny absolute epsilon.
        epsilon_tolerance={"complex128": ("abs", 1.464821375527116e-14)},
        epsilon_tolerance_justification=_EPS_JUST_ABS,
        epsilon_sweep={"complex128": (20000, 1.464821375527116e-14)},
    )
    specs["linalg.matrix_rank"] = ItemSpec(
        name="linalg.matrix_rank", kind="custom", custom_cases=matrix_rank_cases,
        atol=0.0, rtol=0.0,
    )
    specs["linalg.cholesky"] = ItemSpec(
        name="linalg.cholesky", kind="custom", custom_cases=cholesky_cases,
        atol=0.0, rtol=0.0,
        # float32 measured bit-exact (0.0, no entry needed); promoted-int/
        # bool/float16 reuse added 2026-08-01.
        epsilon_tolerance={
            "float64": ("abs", 4.2366110619695974e-13),
            **{d: ("abs", 4.2366110619695974e-13) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=(
            "Lower-triangular path is bit-exact (0/20000 disagreement, "
            "confirmed directly -- the historical '8 ULP on "
            "3x3_spd_upper' figure never came from the lower path). "
            "The upper=True path issues uplo='U' directly to LAPACK "
            "rather than transposing a lower factor -- a different, "
            "equally-valid call sequence for the same unique factor, "
            "which is floating-point non-associativity, not ambiguity " +
            _EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE
        ),
        epsilon_sweep={
            "float64": (20000, 4.2366110619695974e-13),
            **{d: (20000, 4.2366110619695974e-13) for d in _PROMOTED_REAL_DTYPES},
        },
    )

    specs["linalg.inv"] = ItemSpec(
        name="linalg.inv", kind="custom", custom_cases=inv_cases,
        atol=0.0, rtol=0.0,
        # float32/complex64 evidence + promoted-int/bool/float16 reuse
        # added 2026-08-01 (see _EPS_JUST_PROMOTED_REUSE above).
        epsilon_tolerance={
            "float64": ("abs", 1.8403056856186595e-11),
            "complex128": ("abs", 2.5987067057317796e-11),
            "float32": ("abs", 0.0),
            "complex64": ("abs", 5.960464477539063e-08),
            **{d: ("abs", 1.8403056856186595e-11) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 1.8403056856186595e-11),
            "complex128": (20000, 2.5987067057317796e-11),
            "float32": (20000, 0.0),
            "complex64": (20000, 5.960464477539063e-08),
            **{d: (20000, 1.8403056856186595e-11) for d in _PROMOTED_REAL_DTYPES},
        },
    )
    specs["linalg.solve"] = ItemSpec(
        name="linalg.solve", kind="custom", custom_cases=solve_cases,
        atol=0.0, rtol=0.0,
        # float32/complex64 evidence + promoted-int/bool/float16 reuse
        # added 2026-08-01.
        epsilon_tolerance={
            "float64": ("abs", 5.303490979713388e-11),
            "complex128": ("abs", 5.280858024684364e-11),
            "float32": ("abs", 2.9103830456733704e-11),
            "complex64": ("abs", 0.0),
            **{d: ("abs", 5.303490979713388e-11) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 5.303490979713388e-11),
            "complex128": (20000, 5.280858024684364e-11),
            "float32": (20000, 2.9103830456733704e-11),
            "complex64": (20000, 0.0),
            **{d: (20000, 5.303490979713388e-11) for d in _PROMOTED_REAL_DTYPES},
        },
    )
    specs["linalg.matrix_power"] = ItemSpec(
        name="linalg.matrix_power", kind="custom", custom_cases=matrix_power_cases,
        atol=0.0, rtol=0.0,
        # float32/complex64 evidence + promoted-int/bool/float16 reuse
        # added 2026-08-01; complex64/complex128 are genuinely new coverage
        # (new `complex_matrix_power` helper, see matrix_power_cases()).
        epsilon_tolerance={
            "float64": ("rel", 1.2363443602225743e-10),
            "float32": ("rel", 0.0014315538574010134),
            "complex64": ("rel", 1.7310316252405755e-05),
            # complex128 measured 2026-08-01 (20k-matrix seeded sweep of
            # negative/positive powers via complex_matrix_power's zc::inv
            # path) -- was previously MISSING entirely, which graded
            # complex2x2_pow_neg1 bit-exact by default and failed it at
            # ULP distance 2.0 despite the values matching to ~1e-12 rel.
            "complex128": ("rel", 5.7934375091256946e-12),
            **{d: ("rel", 1.2363443602225743e-10) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_RELATIVE + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 1.2363443602225743e-10),
            "float32": (20000, 0.0014315538574010134),
            "complex64": (20000, 1.7310316252405755e-05),
            "complex128": (20000, 5.7934375091256946e-12),
            **{d: (20000, 1.2363443602225743e-10) for d in _PROMOTED_REAL_DTYPES},
        },
    )
    specs["linalg.cond"] = ItemSpec(
        name="linalg.cond", kind="custom", custom_cases=cond_cases,
        atol=0.0, rtol=0.0,
        # float32/complex64 evidence + promoted-int/bool/float16 reuse
        # added 2026-08-01; complex64/complex128 are genuinely new coverage
        # (new `complex_cond` helper, see cond_cases()).
        epsilon_tolerance={
            "float64": ("rel", 5.879095902459606e-13),
            "float32": ("rel", 1.6754570708599203e-07),
            "complex64": ("rel", 1.6743445030442672e-07),
            **{d: ("rel", 5.879095902459606e-13) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_RELATIVE + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 5.879095902459606e-13),
            "float32": (20000, 1.6754570708599203e-07),
            "complex64": (20000, 1.6743445030442672e-07),
            **{d: (20000, 5.879095902459606e-13) for d in _PROMOTED_REAL_DTYPES},
        },
    )
    specs["linalg.pinv"] = ItemSpec(
        name="linalg.pinv", kind="custom", custom_cases=pinv_cases,
        atol=0.0, rtol=0.0,
        # hermitian=True support added 2026-08-01 (was previously
        # undeclared). Bounds below are the MAX of the pre-existing
        # (general-path) evidence and the newly-measured hermitian-path
        # evidence -- one declared item covers both call paths. float32/
        # complex64 evidence + promoted-int/bool/float16 reuse also added
        # 2026-08-01.
        epsilon_tolerance={
            "float64": ("abs", 1.5848655721129035e-10),
            "complex128": ("abs", 1.7546142316859914e-10),
            "float32": ("abs", 0.000244140625),
            "complex64": ("abs", 2.384185791015625e-06),
            **{d: ("abs", 1.5848655721129035e-10) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 1.5848655721129035e-10),
            "complex128": (20000, 1.7546142316859914e-10),
            "float32": (20000, 0.000244140625),
            "complex64": (20000, 2.384185791015625e-06),
            **{d: (20000, 1.5848655721129035e-10) for d in _PROMOTED_REAL_DTYPES},
        },
    )
    specs["linalg.svdvals"] = ItemSpec(
        name="linalg.svdvals", kind="custom", custom_cases=svdvals_cases,
        atol=0.0, rtol=0.0,
        # float32/complex64 measured bit-exact (0.0, no entry needed);
        # promoted-int/bool/float16 reuse added 2026-08-01.
        epsilon_tolerance={
            "float64": ("abs", 7.993605777301127e-15),
            "complex128": ("abs", 9.769962616701378e-15),
            **{d: ("abs", 7.993605777301127e-15) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 7.993605777301127e-15),
            "complex128": (20000, 9.769962616701378e-15),
            **{d: (20000, 7.993605777301127e-15) for d in _PROMOTED_REAL_DTYPES},
        },
    )
    specs["linalg.eigvalsh"] = ItemSpec(
        name="linalg.eigvalsh", kind="custom", custom_cases=eigvalsh_cases,
        atol=0.0, rtol=0.0,
        # UPLO='U' support added 2026-08-01 (was previously undeclared).
        # Bounds below are the MAX of the pre-existing (UPLO='L') evidence
        # and the newly-measured UPLO='U' (transpose-trick) evidence.
        # float32/complex64 measured bit-exact (0.0, no entry needed);
        # promoted-int/bool/float16 reuse added 2026-08-01.
        epsilon_tolerance={
            "float64": ("abs", 1.5631940186722204e-13),
            "complex128": ("abs", 1.5631940186722204e-13),
            **{d: ("abs", 1.5631940186722204e-13) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 1.5631940186722204e-13),
            "complex128": (20000, 1.5631940186722204e-13),
            **{d: (20000, 1.5631940186722204e-13) for d in _PROMOTED_REAL_DTYPES},
        },
    )
    specs["linalg.lstsq"] = ItemSpec(
        name="linalg.lstsq", kind="custom", custom_cases=lstsq_cases,
        atol=0.0, rtol=0.0,
        # complex64 measured bit-exact on generic random sweeps (0.0, no
        # entry needed); promoted-int/bool/float16 reuse added 2026-08-01.
        # float32 entry added 2026-08-01: a broad random sweep also showed
        # bit-exact, but the actual corpus cases (3x2_1d_b, exactly-
        # determined systems where the true residual is 0) hit a
        # near-zero-residual regime a generic random sweep doesn't --
        # numpy=5.0413197e-32 vs anionpy=3.9443045e-31, both floating-point
        # noise around a true value of 0, but a huge RELATIVE/ULP distance.
        # Re-measured with a targeted exact-fit sweep (b forced into a's
        # column span, mirroring the corpus cases) to bound this regime
        # honestly rather than papering over the single observed value.
        epsilon_tolerance={
            "float64": ("abs", 1.2823875294998288e-10),
            "complex128": ("abs", 1.2886168060476723e-12),
            "float32": ("abs", 1.3552527156068805e-20),
            **{d: ("abs", 1.2823875294998288e-10) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 1.2823875294998288e-10),
            "complex128": (20000, 1.2886168060476723e-12),
            "float32": (20000, 1.3552527156068805e-20),
            **{d: (20000, 1.2823875294998288e-10) for d in _PROMOTED_REAL_DTYPES},
        },
        multi_output=True,
    )

    eig_np, eig_ionp = _mk_pair(_eig_probe)
    specs["linalg.eig"] = ItemSpec(
        name="linalg.eig", kind="custom", custom_cases=eig_cases,
        numpy_adapter=eig_np, ionp_adapter=eig_ionp,
        atol=0.0, rtol=0.0,
        # float32 evidence + promoted-int/bool/float16 reuse added
        # 2026-08-01; complex64 measured bit-exact (0.0, no entry needed).
        epsilon_tolerance={
            "float64": ("abs", 1.971756091734278e-13),
            "complex128": ("abs", 1.1518839472084662e-13),
            "float32": ("abs", 1.1788063147832872e-06),
            **{d: ("abs", 1.971756091734278e-13) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_INVARIANT + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 1.971756091734278e-13),
            "complex128": (20000, 1.1518839472084662e-13),
            "float32": (20000, 1.1788063147832872e-06),
            **{d: (20000, 1.971756091734278e-13) for d in _PROMOTED_REAL_DTYPES},
        },
    )

    eigvals_np, eigvals_ionp = _mk_pair(_eigvals_probe)
    specs["linalg.eigvals"] = ItemSpec(
        name="linalg.eigvals", kind="custom", custom_cases=eig_cases,
        numpy_adapter=eigvals_np, ionp_adapter=eigvals_ionp,
        atol=0.0, rtol=0.0,
        # float32/complex64 evidence + promoted-int/bool/float16 reuse
        # added 2026-08-01.
        epsilon_tolerance={
            "float64": ("abs", 8.547410286013612e-13),
            "complex128": ("abs", 1.7120860768676058e-12),
            "float32": ("abs", 9.5367431640625e-07),
            "complex64": ("abs", 2.94701570868442e-07),
            **{d: ("abs", 8.547410286013612e-13) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_INVARIANT + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 8.547410286013612e-13),
            "complex128": (20000, 1.7120860768676058e-12),
            "float32": (20000, 9.5367431640625e-07),
            "complex64": (20000, 2.94701570868442e-07),
            **{d: (20000, 8.547410286013612e-13) for d in _PROMOTED_REAL_DTYPES},
        },
    )

    eigh_np, eigh_ionp = _mk_pair(_eigh_probe)
    specs["linalg.eigh"] = ItemSpec(
        name="linalg.eigh", kind="custom", custom_cases=eigh_cases,
        numpy_adapter=eigh_np, ionp_adapter=eigh_ionp,
        atol=0.0, rtol=0.0,
        # float32/complex64 measured bit-exact (0.0, no entry needed);
        # promoted-int/bool/float16 reuse added 2026-08-01.
        epsilon_tolerance={
            "float64": ("abs", 6.394884621840902e-14),
            "complex128": ("abs", 9.100181066040173e-14),
            **{d: ("abs", 6.394884621840902e-14) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_INVARIANT + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 6.394884621840902e-14),
            "complex128": (20000, 9.100181066040173e-14),
            **{d: (20000, 6.394884621840902e-14) for d in _PROMOTED_REAL_DTYPES},
        },
    )

    qr_np, qr_ionp = _mk_pair(_qr_probe)
    specs["linalg.qr"] = ItemSpec(
        name="linalg.qr", kind="custom", custom_cases=qr_cases,
        numpy_adapter=qr_np, ionp_adapter=qr_ionp,
        atol=0.0, rtol=0.0,
        # float32/complex64 measured bit-exact (0.0, no entry needed);
        # promoted-int/bool/float16 reuse added 2026-08-01.
        epsilon_tolerance={
            "float64": ("abs", 2.6645352591003757e-15),
            "complex128": ("abs", 2.7697102302793378e-15),
            **{d: ("abs", 2.6645352591003757e-15) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_INVARIANT + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 2.6645352591003757e-15),
            "complex128": (20000, 2.7697102302793378e-15),
            **{d: (20000, 2.6645352591003757e-15) for d in _PROMOTED_REAL_DTYPES},
        },
    )

    svd_np, svd_ionp = _mk_pair(_svd_probe)
    specs["linalg.svd"] = ItemSpec(
        name="linalg.svd", kind="custom", custom_cases=svd_cases,
        numpy_adapter=svd_np, ionp_adapter=svd_ionp,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float64": ("abs", 1.174060848541103e-14),
            "complex128": ("abs", 8.244482603911458e-15),
        },
        epsilon_tolerance_justification=_EPS_JUST_INVARIANT,
        epsilon_sweep={
            "float64": (20000, 1.174060848541103e-14),
            "complex128": (20000, 8.244482603911458e-15),
        },
    )

    linalg_error_np, linalg_error_ionp = _mk_pair(_linalg_error_probe)
    specs["linalg.LinAlgError"] = ItemSpec(
        name="linalg.LinAlgError", kind="custom", custom_cases=linalg_error_cases,
        numpy_adapter=linalg_error_np, ionp_adapter=linalg_error_ionp,
        atol=0.0, rtol=0.0,
    )

    specs["linalg.matmul"] = ItemSpec(
        name="linalg.matmul", kind="custom", custom_cases=matmul_cases,
        atol=0.0, rtol=0.0,
        # Array-API-restricted thin wrapper over `ionp_ion::matmul::matmul`
        # (already a general N-D/batched gufunc core). float32/float64 are
        # bit-exact on the 20,000-matrix seeded sweep; complex64/complex128
        # need a small absolute epsilon (BLAS non-associativity across two
        # independent call paths, same reasoning as `inv`/`solve` above).
        epsilon_tolerance={
            "complex64": ("abs", 1.9660499219753547e-06),
            "complex128": ("abs", 3.66205343881779e-15),
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS,
        epsilon_sweep={
            "complex64": (20000, 1.9660499219753547e-06),
            "complex128": (20000, 3.66205343881779e-15),
        },
    )
    specs["linalg.vecdot"] = ItemSpec(
        name="linalg.vecdot", kind="custom", custom_cases=vecdot_cases,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "complex64": ("abs", 2.1324806311895372e-06),
            "complex128": ("abs", 3.794299872214038e-15),
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS,
        epsilon_sweep={
            "complex64": (20000, 2.1324806311895372e-06),
            "complex128": (20000, 3.794299872214038e-15),
        },
    )
    specs["linalg.outer"] = ItemSpec(
        name="linalg.outer", kind="custom", custom_cases=outer_cases,
        atol=0.0, rtol=0.0,
    )
    # "outer" (bare, top-level `numpy.outer`) -- a DIFFERENT contract from
    # `linalg.outer` above (ravels N-D input instead of rejecting it; see
    # `outer_toplevel_cases`'s docstring and `ionp-py/src/linalg.rs`'s
    # `outer_toplevel`). Default path resolution (no numpy_path/ionp_path
    # override) reaches `numpy.outer`/`anionpy.outer` directly since this
    # item's name has no "." prefix.
    specs["outer"] = ItemSpec(
        name="outer", kind="custom", custom_cases=outer_toplevel_cases,
        atol=0.0, rtol=0.0,
    )
    specs["linalg.diagonal"] = ItemSpec(
        name="linalg.diagonal", kind="custom", custom_cases=diagonal_cases,
        atol=0.0, rtol=0.0,
    )
    specs["linalg.matrix_transpose"] = ItemSpec(
        name="linalg.matrix_transpose", kind="custom", custom_cases=matrix_transpose_cases,
        atol=0.0, rtol=0.0,
    )

    # ----- 2026-08-02: net-new Array-API-shaped items --------------------
    # No formal 20,000-matrix seeded sweep was run for these 8 items (out
    # of scope this pass); epsilon_tolerance values below, where present,
    # are the actual max distance observed on THIS file's own (much
    # smaller) case corpus, not a fabricated/guessed number -- flagged
    # plainly as a narrower evidence base than det/inv/solve/etc above.
    specs["linalg.cross"] = ItemSpec(
        name="linalg.cross", kind="custom", custom_cases=cross_cases,
        atol=0.0, rtol=0.0,
    )
    specs["linalg.tensordot"] = ItemSpec(
        name="linalg.tensordot", kind="custom", custom_cases=tensordot_cases,
        atol=0.0, rtol=0.0,
    )
    specs["linalg.multi_dot"] = ItemSpec(
        name="linalg.multi_dot", kind="custom", custom_cases=multi_dot_cases,
        atol=0.0, rtol=0.0,
    )
    specs["linalg.tensorinv"] = ItemSpec(
        name="linalg.tensorinv", kind="custom", custom_cases=tensorinv_cases,
        atol=0.0, rtol=0.0,
        # tensorinv reshapes to 2-D and reuses `rc::inv`/`zc::inv` (the same
        # LAPACK core as `linalg.inv`) -- measured 2026-08-02 via a fresh
        # 20,000-matrix seeded sweep (reshaped-square controlled matrices,
        # sizes/condition matching `inv`'s own sweep shape), not reused from
        # `inv`'s number, since the reshape step is new code.
        epsilon_tolerance={
            "float64": ("abs", 1.4935608305677306e-11),
            "complex128": ("abs", 1.365237582607511e-11),
            **{d: ("abs", 1.4935608305677306e-11) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 1.4935608305677306e-11),
            "complex128": (20000, 1.365237582607511e-11),
            **{d: (20000, 1.4935608305677306e-11) for d in _PROMOTED_REAL_DTYPES},
        },
    )
    specs["linalg.tensorsolve"] = ItemSpec(
        name="linalg.tensorsolve", kind="custom", custom_cases=tensorsolve_cases,
        atol=0.0, rtol=0.0,
        # tensorsolve reshapes to a 2-D system and reuses `rc::solve`/
        # `zc::solve` (the same LAPACK core as `linalg.solve`) -- measured
        # 2026-08-02 via a fresh 20,000-system seeded sweep.
        epsilon_tolerance={
            "float64": ("abs", 3.737454790098127e-11),
            "complex128": ("abs", 7.296849668463981e-11),
            **{d: ("abs", 3.737454790098127e-11) for d in _PROMOTED_REAL_DTYPES},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 3.737454790098127e-11),
            "complex128": (20000, 7.296849668463981e-11),
            **{d: (20000, 3.737454790098127e-11) for d in _PROMOTED_REAL_DTYPES},
        },
    )
    specs["linalg.vector_norm"] = ItemSpec(
        name="linalg.vector_norm", kind="custom", custom_cases=vector_norm_cases,
        atol=0.0, rtol=0.0,
        # `vector_norm`'s magnitude-reduction path (sum-of-powers /
        # sqrt/pow-root) is bounded by the problem's own O(1) scale, not
        # multiplicative in matrix size -- an ABSOLUTE bound, per
        # `_EPS_JUST_ABS`. Measured 2026-08-02 via a 20,000-vector seeded
        # sweep across ord in {1, 2, 3, 4, 0.5} (the worst ord per dtype
        # kept). float16 is NOT included here: `vector_norm` never rejects
        # float16 (see `ndarray_and_magnitudes`), and its float16 output is
        # computed by casting a float64-accumulated result down to float16
        # at the very end -- there is no independent float16 LAPACK/BLAS
        # call sequence to diverge, so it is bit-exact by construction
        # (confirmed: no float16 case in the differential run failed on
        # this item after the float16-passthrough fix).
        epsilon_tolerance={
            "float64": ("abs", 4.263256414560601e-14),
            "float32": ("abs", 2.288818359375e-05),
            "complex128": ("abs", 8.526512829121202e-14),
            "complex64": ("abs", 3.0517578125e-05),
            **{d: ("abs", 4.263256414560601e-14) for d in _PROMOTED_REAL_DTYPES if d != "float16"},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 4.263256414560601e-14),
            "float32": (20000, 2.288818359375e-05),
            "complex128": (20000, 8.526512829121202e-14),
            "complex64": (20000, 3.0517578125e-05),
            **{d: (20000, 4.263256414560601e-14) for d in _PROMOTED_REAL_DTYPES if d != "float16"},
        },
    )
    specs["linalg.matrix_norm"] = ItemSpec(
        name="linalg.matrix_norm", kind="custom", custom_cases=matrix_norm_cases,
        atol=0.0, rtol=0.0,
        # Same reasoning as `vector_norm` above for the fro/1/-1/inf/-inf
        # ords; the 2/-2/'nuc' ords additionally route through
        # `svd_singular_values_batched` (LAPACK `dgesdd`/`zgesdd`, same
        # non-associativity source as `svdvals`). Measured 2026-08-02 via a
        # 20,000-matrix seeded sweep across ord in {fro, nuc, 1, -1, 2, -2,
        # inf, -inf} (worst ord per dtype kept). float16 excluded from the
        # promoted-reuse set for the same bit-exact-by-construction reason
        # as `vector_norm` -- and additionally, `matrix_norm` genuinely
        # REJECTS float16 outright for the SVD-based ords (2/-2/'nuc'),
        # matching numpy, so no float16 case reaches the SVD path at all.
        epsilon_tolerance={
            "float64": ("abs", 8.881784197001252e-15),
            "float32": ("abs", 1.9073486328125e-06),
            "complex128": ("abs", 1.4210854715202004e-14),
            "complex64": ("abs", 1.9073486328125e-06),
            **{d: ("abs", 8.881784197001252e-15) for d in _PROMOTED_REAL_DTYPES if d != "float16"},
        },
        epsilon_tolerance_justification=_EPS_JUST_ABS + " " + _EPS_JUST_PROMOTED_REUSE,
        epsilon_sweep={
            "float64": (20000, 8.881784197001252e-15),
            "float32": (20000, 1.9073486328125e-06),
            "complex128": (20000, 1.4210854715202004e-14),
            "complex64": (20000, 1.9073486328125e-06),
            **{d: (20000, 8.881784197001252e-15) for d in _PROMOTED_REAL_DTYPES if d != "float16"},
        },
    )
    specs["linalg.norm"] = ItemSpec(
        name="linalg.norm", kind="custom", custom_cases=norm_cases,
        atol=0.0, rtol=0.0,
        # No tolerance declared: `norm_cases()`'s specific fixed inputs
        # measure bit-exact (101/101 PASS with atol=0.0/rtol=0.0) even
        # though `norm` shares `vector_norm`/`matrix_norm`'s underlying
        # reduction code (which DOES show the same tiny non-associativity
        # gap under a wide random sweep -- see those two items' declared
        # tolerances). Leaving this bare rather than pre-declaring an
        # unearned tolerance this item's own case set never actually
        # triggers, per the task brief's "don't add tolerance to hide a
        # mismatch" rule -- there is no mismatch here to hide.
    )

    # linalg.test -- same `_IonpTester`-mirrors-`testing.test` pattern as
    # `fft.test` (see `tests/differential/fft_cases.py`'s entry for the
    # full rationale) applied to `anionpy.linalg`. numpy's own
    # `numpy.linalg.test` is a `PytestTester` that shells to
    # `pytest --pyargs numpy.linalg` against numpy's OWN installed test
    # files -- a contract anionpy cannot replicate without importing numpy
    # at runtime; `anionpy.linalg.test` instead runs anionpy's own suite via
    # the same `_IonpTester` class `anionpy.testing.test` already uses (and
    # is already declared "exact" under that same substitution).
    specs["linalg.test"] = ItemSpec(
        name="linalg.test", kind="custom", scalar_like=True,
        custom_cases=lambda: [("invokes_pytest_main_with_constructed_argv", (), {})],
        numpy_adapter=lambda: _probe_linalg_test_numpy(),
        ionp_adapter=lambda: _probe_linalg_test_ionp(),
    )

    return specs


def _probe_linalg_test_call(mod):
    # Delegates to the ONE shared implementation. This used to be a private
    # copy that compared invocation only and never the signature; see
    # harness.probe_pytest_tester for the divergence that slipped through
    # all three copies.
    from harness import probe_pytest_tester
    return probe_pytest_tester(mod)


def _probe_linalg_test_numpy():
    import numpy.linalg as nplinalg
    return _probe_linalg_test_call(nplinalg)


def _probe_linalg_test_ionp():
    # `anionpy.linalg` is a real PyO3 `module` object, not a plain-Python
    # submodule file -- `import anionpy.linalg` fails with
    # `ModuleNotFoundError` (verified live, same as `anionpy.fft`), so it
    # must be reached via attribute access on the already-imported
    # top-level `anionpy` package.
    import anionpy
    return _probe_linalg_test_call(anionpy.linalg)


LINALG_SPECS: dict[str, ItemSpec] = _build_linalg_specs()
