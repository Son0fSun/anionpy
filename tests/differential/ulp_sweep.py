"""Adequate-sample evidence for `ItemSpec.ulp_tolerance`.

Why this file exists (2026-08-01, sample-size defect): the 4 items that
declared a `ulp_tolerance` (`absolute`/`abs`/`divide`/`true_divide`) were
each measured only against the ~150-case differential corpus in
`corpus.py`. That corpus is built for edge-case *breadth* (empty arrays,
NaN, inf, -0.0, dtype boundaries, broadcasting, ...), not for statistical
*volume* -- it was never designed to bound a worst-case ULP distance, and
a follow-up run against a 20,000-value seeded corpus found the declared
`absolute`/`abs` bound (1.0 ULP) was already wrong (true max 2.0 ULP; see
reports/ionp-ulp-tolerance-decision-2026-08-01.md's correction block).
**The tolerance was a property of the sample, not of the implementation.**

This module is the fix: a declared `ulp_tolerance` must cite evidence
measured over a large, seeded, deterministic sweep -- `MIN_SWEEP_N`
(20,000) values per applicable dtype, same pinned `SEED` the rest of the
differential suite uses (`corpus.SEED`) -- not merely the item's ordinary
differential cases. `registry.ItemSpec.__post_init__` enforces that this
evidence was actually recorded (`ulp_sweep` field, `{dtype_name: (n,
max_ulp)}`) and that the declared `ulp_tolerance` equals (not merely
covers) the measured maximum across every recorded dtype -- so the bound
can neither be under-evidenced (missing/small sweep) nor padded above
what was actually measured (see that module's docstring for why equality,
not `>=`, is the check: padding "to be safe" is exactly how "1 ULP" got
promoted from a single worked example without anyone checking the max).

Sweep values are drawn with the same distribution family `corpus.py`'s
`_fill` uses for float/complex (`standard_normal() * 10`, real and
imaginary parts independent for complex) so the sweep exercises the same
magnitude range as the rest of the suite, just at adequate volume -- this
is not a new, unreviewed input distribution, only more of the existing
one. Binary sweeps draw the second operand from `SEED + 1` so it is an
independent stream, not a shuffle/copy of the first.

Cost: measuring one (name, dtype) pair over 20,000 values is a single
vectorized numpy call plus an O(n) Python loop in
`harness.max_ulp_distance` (that loop is not vectorized -- see its
docstring) -- about 0.1-0.3s per dtype on the reference machine, so
O(1s) per declared item across its 2-4 applicable dtypes. Run once here,
at declaration time (this module has a `__main__` for exactly that), and
the *result* (n, max_ulp) is what gets committed into
`ufunc_registry.py`'s `_ULP_OVERRIDES` -- `run.py`'s normal suite run
never re-executes a sweep, only the ordinary ~150-case corpus, so the
adequacy gate does not add sweep cost to every CI run. If a future item's
sweep becomes expensive enough to matter (e.g. a much larger dtype
catalog), the split to make is exactly this one: sweep at declaration
time, cache the (n, max_ulp) evidence in the registry, never re-sweep at
test time.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

import _bootstrap  # noqa: F401
import corpus
import harness
import registry

SEED = corpus.SEED  # 20260731 -- the same pinned seed every other differential input uses
MIN_SWEEP_N = registry.MIN_ULP_SWEEP_N


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sweep_array(dtype, n: int = MIN_SWEEP_N, seed: int = SEED) -> np.ndarray:
    """`n` seeded values of `dtype`. Same distribution family as
    `corpus.py`'s `_fill` for float/complex (`standard_normal() * 10`), just
    at sweep volume instead of the small fixed-shape corpus. Deterministic:
    same `(dtype, n, seed)` always produces the same array."""
    dt = np.dtype(dtype)
    rng = _rng(seed)
    if dt.kind == "f":
        return (rng.standard_normal(size=n) * 10).astype(dt)
    if dt.kind == "c":
        real = rng.standard_normal(size=n) * 10
        imag = rng.standard_normal(size=n) * 10
        return (real + 1j * imag).astype(dt)
    raise ValueError(f"ULP sweep is only defined for float/complex dtypes, got {dt}")


@dataclasses.dataclass(frozen=True)
class SweepResult:
    dtype: str
    n: int
    max_ulp: float
    disagreeing: int  # count of elements with ULP distance > 0, diagnostic only


def _dispatchers(ufunc_name: str):
    """Build the same numpy/anionpy callables the real registry item would use
    -- via a throwaway, un-toleranced ItemSpec so this module does not
    duplicate `ItemSpec.resolve_numpy`/`resolve_ionp`'s ionp-array-
    conversion logic (see registry.py's `make_ionp_array_converter`)."""
    spec = registry.ItemSpec(name=ufunc_name, kind="ufunc",
                              numpy_path=ufunc_name, ionp_path=ufunc_name)
    return spec.resolve_numpy(), spec.resolve_ionp()


def measure_unary(ufunc_name: str, dtype, n: int = MIN_SWEEP_N, seed: int = SEED) -> SweepResult:
    numpy_fn, ionp_fn = _dispatchers(ufunc_name)
    if numpy_fn is None or ionp_fn is None:
        raise RuntimeError(
            f"{ufunc_name}: cannot sweep -- resolve_numpy/resolve_ionp returned None "
            f"(numpy_fn={numpy_fn!r} ionp_fn={ionp_fn!r}); the item is absent on one "
            f"side, so there is nothing to measure a ULP bound against."
        )
    arr = sweep_array(dtype, n=n, seed=seed)
    np_out = numpy_fn("call", arr)
    ionp_out = ionp_fn("call", arr)
    return _summarize(dtype, n, np_out, ionp_out)


def measure_binary(ufunc_name: str, dtype, n: int = MIN_SWEEP_N, seed: int = SEED) -> SweepResult:
    numpy_fn, ionp_fn = _dispatchers(ufunc_name)
    if numpy_fn is None or ionp_fn is None:
        raise RuntimeError(
            f"{ufunc_name}: cannot sweep -- resolve_numpy/resolve_ionp returned None "
            f"(numpy_fn={numpy_fn!r} ionp_fn={ionp_fn!r}); the item is absent on one "
            f"side, so there is nothing to measure a ULP bound against."
        )
    a = sweep_array(dtype, n=n, seed=seed)
    b = sweep_array(dtype, n=n, seed=seed + 1)
    np_out = numpy_fn("call", a, b)
    ionp_out = ionp_fn("call", a, b)
    return _summarize(dtype, n, np_out, ionp_out)


def _summarize(dtype, n: int, np_out, ionp_out) -> SweepResult:
    dt = np.dtype(dtype)
    a = np.asarray(np_out).ravel()
    b = np.asarray(ionp_out).ravel()
    width = harness._width_for_dtype(dt)  # noqa: SLF001 - same module family, real helper
    disagreeing = 0
    worst = 0.0
    is_complex = dt.kind == "c"
    for i in range(a.size):
        if is_complex:
            d = max(
                harness.ulp_distance(float(a[i].real), float(b[i].real), width),
                harness.ulp_distance(float(a[i].imag), float(b[i].imag), width),
            )
        else:
            d = harness.ulp_distance(float(a[i]), float(b[i]), width)
        if d != 0.0:
            disagreeing += 1
        if d == math.inf:
            worst = math.inf
        elif worst != math.inf and d > worst:
            worst = d
    return SweepResult(dtype=dt.name, n=n, max_ulp=worst, disagreeing=disagreeing)


def measure_reduce_chain(ufunc_name: str, dtype, chain_len: int = 2,
                          n_groups: int = MIN_SWEEP_N, seed: int = SEED) -> SweepResult:
    """Sweep `.reduce(axis=-1)` over `n_groups` independent chains of
    `chain_len` values each, i.e. `chain_len - 1` compounded ufunc calls per
    output element -- the shape of the claim behind `divide`/`true_divide`'s
    original 17.0 ULP figure ("np.divide.reduce chains 2 divisions... error
    compounds"), which was never actually measured at adequate volume (it
    came from one (3,4)-shaped case in the ~150-case corpus). `n_groups` is
    the sweep's sample size here -- each of the `n_groups` reduce outputs is
    one independent measurement, same adequacy bar as `measure_binary`."""
    numpy_fn, ionp_fn = _dispatchers(ufunc_name)
    if numpy_fn is None or ionp_fn is None:
        raise RuntimeError(
            f"{ufunc_name}: cannot sweep -- resolve_numpy/resolve_ionp returned None "
            f"(numpy_fn={numpy_fn!r} ionp_fn={ionp_fn!r}); the item is absent on one "
            f"side, so there is nothing to measure a ULP bound against."
        )
    dt = np.dtype(dtype)
    rng = _rng(seed)
    # (chain_len, n_groups), reduced along axis=0 (the default) rather than
    # (n_groups, chain_len)/axis=-1: anionpy's .reduce() was found, while
    # building this sweep, to only reduce correctly along axis 0 for a
    # multi-thousand-element array (axis=1/axis=-1 silently returns the
    # wrong shape -- an ionp-core bug, out of this module's owned paths,
    # reported separately rather than fixed here). Reducing along axis 0
    # sidesteps that bug entirely rather than working around it, since the
    # measurement only needs SOME array layout that chains chain_len values
    # per output element.
    shape = (chain_len, n_groups)
    if dt.kind == "f":
        arr = (rng.standard_normal(size=shape) * 10).astype(dt)
    elif dt.kind == "c":
        real = rng.standard_normal(size=shape) * 10
        imag = rng.standard_normal(size=shape) * 10
        arr = (real + 1j * imag).astype(dt)
    else:
        raise ValueError(f"ULP sweep is only defined for float/complex dtypes, got {dt}")
    np_out = numpy_fn("reduce", arr)
    ionp_out = ionp_fn("reduce", arr)
    return _summarize(dtype, n_groups, np_out, ionp_out)


def measure_item(ufunc_name: str, dtypes: list, arity: str, n: int = MIN_SWEEP_N,
                  seed: int = SEED) -> dict[str, SweepResult]:
    """Sweep `ufunc_name` over every dtype in `dtypes`, `arity` in
    {"unary", "binary"}. Returns {dtype_name: SweepResult} -- the shape
    `registry.ItemSpec.ulp_sweep` expects is `{dtype_name: (n, max_ulp)}`,
    built from this via `evidence_dict()` below."""
    fn = {"unary": measure_unary, "binary": measure_binary}[arity]
    return {dtype_name: fn(ufunc_name, dtype, n=n, seed=seed)
            for dtype, dtype_name in ((d, np.dtype(d).name) for d in dtypes)}


def evidence_dict(results: dict[str, SweepResult]) -> dict[str, tuple[int, float]]:
    return {name: (r.n, r.max_ulp) for name, r in results.items()}


# -----------------------------------------------------------------------
# Matrix-sweep machinery for `ItemSpec.epsilon_tolerance` evidence
# (2026-08-01 "hardened door / open window" fix, see
# reports/ionp-hardened-door-open-window-2026-08-01.md and
# registry.py's epsilon_tolerance docstring).
#
# The sweeps above (sweep_array/measure_unary/measure_binary/...) only
# generate simple 1-D arrays of scalars, which is enough for `kind="ufunc"`
# items. The `linalg.*` items are `kind="custom"` and take MATRIX
# arguments (square, SPD/Hermitian, rectangular, or a matrix+vector pair),
# with condition number controlling how ill-conditioned the sweep gets --
# the audit's own methodology (sizes 2-8, condition scaled 1e-2...1e2,
# seed 20260801, see the report above) is reproduced here exactly rather
# than inventing a different distribution, so this module's evidence and
# the audit's are directly comparable.
# -----------------------------------------------------------------------

COND_LO = 1e-2
COND_HI = 1e2
MATRIX_SIZE_LO = 2
MATRIX_SIZE_HI = 8


def controlled_matrix(rng: np.random.Generator, n: int, dtype, hermitian: bool = False) -> np.ndarray:
    """One n x n matrix with singular values log-uniform in
    [COND_LO, COND_HI] (condition-number-controlled, via QR of a Gaussian
    matrix -- same construction the audit report's sweep used). Real
    (`dtype.kind == 'f'`) or complex (`'c'`) per `dtype`. `hermitian=True`
    builds A = U @ diag(s) @ U^H (real-eigenvalue, symmetric/Hermitian
    case, needed for cholesky/eigh/eigvalsh); otherwise A = U @ diag(s) @ V^H
    with independent U, V (general case)."""
    dt = np.dtype(dtype)
    is_c = dt.kind == "c"
    s = np.exp(rng.uniform(np.log(COND_LO), np.log(COND_HI), size=n))
    if is_c:
        G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        U, _ = np.linalg.qr(G)
        if hermitian:
            A = (U * s) @ U.conj().T
        else:
            H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            V, _ = np.linalg.qr(H)
            A = (U * s) @ V.conj().T
    else:
        G = rng.standard_normal((n, n))
        U, _ = np.linalg.qr(G)
        if hermitian:
            A = (U * s) @ U.T
        else:
            H = rng.standard_normal((n, n))
            V, _ = np.linalg.qr(H)
            A = (U * s) @ V.T
    return A.astype(dt)


def rectangular_matrix(rng: np.random.Generator, rows: int, cols: int, dtype) -> np.ndarray:
    """One rows x cols matrix, standard-normal entries (real or complex per
    `dtype`) -- unconditioned, for items whose sweep doesn't need a
    controlled condition number (pinv/svdvals/qr/svd/lstsq)."""
    dt = np.dtype(dtype)
    if dt.kind == "c":
        A = rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))
    else:
        A = rng.standard_normal((rows, cols))
    return A.astype(dt)


def random_vector(rng: np.random.Generator, n: int, dtype) -> np.ndarray:
    dt = np.dtype(dtype)
    if dt.kind == "c":
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    else:
        v = rng.standard_normal(n)
    return v.astype(dt)


def sweep_matrix_item(make_args, call_pair, dtype, metric: str,
                       n: int = MIN_SWEEP_N, seed: int = SEED) -> SweepResult:
    """Generic per-dtype matrix sweep for a `kind="custom"` item.

    `make_args(rng, dtype) -> args_tuple` builds one case's arguments (a
    matrix, or a (matrix, vector) pair, ...) using this module's generators
    above. `call_pair(args) -> (np_out, ionp_out)` invokes numpy's and
    anionpy's implementations and returns their raw outputs (arrays or tuples
    of arrays -- tuple outputs, e.g. lstsq, are handled by measuring every
    array-like slot and taking the max across slots). `metric` is `"abs"`
    (via `harness.max_abs_distance`) or `"rel"` (via
    `harness.max_rel_distance`) -- same metric names `ItemSpec.
    epsilon_tolerance` declares, so the sweep and the grading it justifies
    use the identical formula, not two independently-written ones that
    could silently drift apart.

    Returns a `SweepResult` whose `.max_ulp` field is repurposed to hold
    the measured max abs/rel DISTANCE (not an actual ULP count) -- kept as
    the same dataclass/field name as the ULP sweeps above so
    `evidence_dict()` and the reporting code path are shared, not
    duplicated; the `metric` string travels alongside it wherever this
    result is consumed (see `sweep_matrix_item_evidence` below, which is
    the function that actually produces `ItemSpec.epsilon_sweep` entries
    and keeps the metric explicit there)."""
    import harness

    rng = np.random.default_rng(seed)
    dist_fn = harness.max_abs_distance if metric == "abs" else harness.max_rel_distance
    worst = 0.0
    for _ in range(n):
        args = make_args(rng, dtype)
        np_out, ionp_out = call_pair(args)
        if isinstance(np_out, tuple):
            for a, b in zip(np_out, ionp_out):
                a_arr = np.asarray(a)
                if a_arr.dtype.kind not in "fc" or a_arr.size == 0:
                    continue
                worst = max(worst, dist_fn(a, b))
        else:
            worst = max(worst, dist_fn(np_out, ionp_out))
    return SweepResult(dtype=np.dtype(dtype).name, n=n, max_ulp=worst, disagreeing=0)


def sweep_matrix_item_evidence(make_args, call_pair, dtypes_metrics: dict,
                                n: int = MIN_SWEEP_N, seed: int = SEED) -> dict:
    """Sweep `make_args`/`call_pair` over every `{dtype: metric}` entry in
    `dtypes_metrics`; returns `{dtype_name: (n, measured_value)}` --
    exactly the shape `ItemSpec.epsilon_sweep` requires, ready to pair with
    an `epsilon_tolerance` dict built from the same measured values (see
    linalg_cases.py, which calls this at declaration time and pastes the
    resulting numbers into its `ItemSpec` declarations -- same
    "sweep once, commit the (n, measured) evidence" pattern `measure_item`/
    `evidence_dict` already established for the ULP mechanism above)."""
    out = {}
    for dtype, metric in dtypes_metrics.items():
        r = sweep_matrix_item(make_args, call_pair, dtype, metric, n=n, seed=seed)
        out[np.dtype(dtype).name] = (r.n, r.max_ulp)
    return out


if __name__ == "__main__":
    import sys

    targets = [
        ("absolute", [np.float32, np.float64, np.complex64, np.complex128], "unary"),
        ("abs", [np.float32, np.float64, np.complex64, np.complex128], "unary"),
        ("divide", [np.float32, np.float64, np.complex64, np.complex128], "binary"),
        ("true_divide", [np.float32, np.float64, np.complex64, np.complex128], "binary"),
    ]
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        targets = [t for t in targets if t[0] in wanted]

    for name, dtypes, arity in targets:
        print(f"=== {name} ({arity}) ===")
        results = measure_item(name, dtypes, arity)
        for dtype_name, r in results.items():
            print(f"  {dtype_name:12s} n={r.n:6d}  max_ulp={r.max_ulp!r:>8}  "
                  f"disagreeing={r.disagreeing}")
        overall = max(r.max_ulp for r in results.values())
        print(f"  -> overall max_ulp across dtypes = {overall!r}")
