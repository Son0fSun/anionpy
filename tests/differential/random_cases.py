"""anionpy.random.* differential registry entries.

NEW FILE (permitted: the task brief allows adding new files under
tests/differential/, in addition to owning registry.py itself). Builds a
`RANDOM_SPECS: dict[str, ItemSpec]` the same way linalg_cases.py/fft_cases.py
build their own dicts -- imported and merged into `registry.REGISTRY` from
the bottom of registry.py, collision-checked the same way every other
block's merge is.

SCOPE. Only 5 of the 63 `random.*` surface items (`tools/numpy_surface.json`)
are exercised at all: `SeedSequence`, `PCG64`, `PCG64DXSM`, `Generator`,
`default_rng`. Every distribution (`normal`, `beta`, ... 40+ items),
`choice`/`permutation`/`shuffle`/`permuted`, `MT19937`/`Philox`/`SFC64`, and
the legacy `RandomState` surface are NOT implemented in
`ionp-py/src/random.rs` -- calling any of them raises `AttributeError`,
which is not tested here because there is nothing to differentially compare
(absent surface, not a behavioral divergence).

Of the 5 exercised items, `Generator` itself only implements 3 of its ~30
public methods (`random`, `integers`, `bytes`) -- `choice`, `permutation`,
`shuffle`, `permuted`, every distribution method, and the `.bit_generator`
attribute are all missing. `PCG64`/`PCG64DXSM` only implement the
constructor and `.random_raw()` -- `.state` (get/set), `.advance()`,
`.jumped()` are missing. `SeedSequence` only implements the constructor and
`.generate_state()` -- `.entropy`/`.spawn_key`/`.pool_size`/`.pool`/`.state`
(attrs) and `.spawn()` are missing.

DECLARATION. None of these 5 items are declared "exact" in
`anionpy/_state/random.py` despite every case below passing bit-exact --
`tools/numpy_surface.json` only has class-level granularity for these items
(no per-method sub-entries the way `ndarray.reshape` etc. get their own
ledger key), so declaring e.g. `"random.Generator": "exact"` would claim
the WHOLE class matches numpy, which is false given the missing majority of
its method surface. This mirrors the standing precedent already in this
codebase: `char.chararray` is left undeclared for the analogous reason (see
`anionpy/_state/char_strings.py`) rather than declared for a documented
subset. See `anionpy/_state/random.py` for the full writeup.

EVIDENCE. Every case below is SEEDED and compares exact bytes (via
`harness.compare_values`'s dtype+shape+exact-value check -- integers/bool/
bytes have no tolerance mechanism at all to lean on; `random()`'s float
output is graded via the default `atol=0.0, rtol=0.0` bit-exact path, no
ULP slack declared or needed) against real numpy 2.5.1 (`.venv`), confirmed
by hand first via a one-off smoke script before being written here, then
again by running `tests/differential/run.py` itself. Error paths compare
exception TYPE and exact MESSAGE TEXT (`harness.run_case`'s 2026-08-01
message-comparison hardening): negative seed, float seed, bad-string seed,
`low >= high`, out-of-bounds low/high per dtype, negative size.

CRITICAL, empirically-verified correction (see `ionp-core/src/random/
bounded.rs` and `ionp-py/src/random.rs`'s own module docs for the full
derivation): `np.random.default_rng(seed)` in the installed numpy 2.5.1
instantiates plain `PCG64` (XSL-RR), NOT `PCG64DXSM` -- confirmed directly
via `type(np.random.default_rng(42).bit_generator)`. `default_rng` here
binds to `Pcg64`, matching that.

Adapters, not `numpy_path`/`ionp_path` resolution: every case here
constructs a fresh, seeded generator/bit-generator object and calls a
method on it -- there is no single top-level callable with the SAME name
on both `numpy` and `anionpy` these calls resolve to (unlike e.g.
`linalg.det`), so every entry below uses explicit `numpy_adapter`/
`ionp_adapter` callables, the same mechanism `linalg_cases.py`'s
`eig`/`eigvals`/`eigh`/`qr`/`svd` entries use via `_mk_pair`. The
`random.Generator` item uses one generic `(seed, op, opargs, opkwargs)`
dispatcher (`_generator_np`/`_generator_ionp`) rather than one adapter per
method, since a single ledger key can only carry one adapter pair but
needs to exercise three distinct methods (`integers`, `bytes`, and the
`Generator(PCG64(seed))` two-object constructor form, verified via calling
`.random()` on the result).
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec


def _as_comparable(x):
    """Normalizes a call's raw return value into something
    `harness.compare_values` (dtype+shape+exact-value, no `scalar_like`)
    can grade uniformly, REGARDLESS of whether that call form returns an
    array, a bare Python/numpy scalar, or `bytes` -- these items mix all
    three return shapes across their call forms (e.g. `.random()` returns
    an ndarray when `size=` is given but a bare scalar when it is not;
    `.bytes()` always returns `bytes`), and `ItemSpec.scalar_like` is a
    single item-wide flag that cannot express "some cases are arrays, some
    are scalars" -- picking either value would break the other case shape
    (`scalar_like=True` fails array cases: `type(np.ndarray) is not
    type(anionpy.ndarray)`; `scalar_like=False` fails bare-scalar cases: a
    Python/numpy scalar has no `.dtype` numpy itself would ever look for
    on an array result).

    This does NOT change what code path is exercised -- the real
    `.random()`/`.integers()`/`.bytes()` call with the real `size=`/
    `dtype=` arguments still runs exactly as the case declares; only the
    OUTER wrapper applied to whatever it returns is normalized, identically
    on both the numpy and anionpy sides, before handing both to
    `compare_values`. `bytes` -> `np.frombuffer(x, dtype=uint8)` (byte-for-
    byte, verified: a `uint8` view of a `bytes` object preserves every
    byte exactly, which is exactly this task's bit-exactness bar).
    Everything else -> `np.asarray(x)`, which turns e.g. numpy's own
    `numpy.int64` scalar (verified via `type(...)`: `Generator.integers()`
    with `size=None` returns a NUMPY scalar type, not a bare Python int)
    and anionpy's plain Python `int`/`float`/`bool` scalar into 0-d arrays of
    matching dtype -- confirmed directly (`np.asarray(np.int64(45)).dtype
    == np.asarray(45).dtype == int64`) before being relied on here.
    """
    if isinstance(x, bytes):
        return np.frombuffer(x, dtype=np.uint8)
    return np.asarray(x)


# ---------------------------------------------------------------------------
# random.default_rng (+ Generator.random, exercised through it)
# ---------------------------------------------------------------------------


def _default_rng_np(seed, size=None, dtype=None):
    kwargs = {}
    if size is not None:
        kwargs["size"] = size
    if dtype is not None:
        kwargs["dtype"] = dtype
    return _as_comparable(np.random.default_rng(seed).random(**kwargs))


def _default_rng_ionp(seed, size=None, dtype=None):
    import anionpy

    kwargs = {}
    if size is not None:
        kwargs["size"] = size
    if dtype is not None:
        kwargs["dtype"] = dtype
    return _as_comparable(anionpy.random.default_rng(seed).random(**kwargs))


def default_rng_cases():
    return [
        ("scalar_seed42", (42,), {}),
        ("scalar_seed0", (0,), {}),
        ("size10_seed42", (42,), {"size": 10}),
        ("size_3x4_seed42", (42,), {"size": (3, 4)}),
        ("size0_seed42", (42,), {"size": 0}),
        ("size1_seed42", (42,), {"size": 1}),
        ("size_scalar_int_seed7", (7,), {"size": 5}),
        ("dtype_f32_size10_seed42", (42,), {"size": 10, "dtype": np.float32}),
        ("dtype_f64_scalar_seed42", (42,), {"dtype": np.float64}),
        # error paths
        ("negative_size", (42,), {"size": -3}),
        ("negative_seed", (-1,), {}),
        ("float_seed_35", (3.5,), {}),
        ("float_seed_20", (2.0,), {}),
        ("bad_string_seed", ("abc",), {}),
    ]


# ---------------------------------------------------------------------------
# random.Generator: dispatched (seed, op, opargs, opkwargs) form, covering
# `.integers()`, `.bytes()`, and the `Generator(PCG64(seed))` two-object
# constructor call form (verified via `.random()` on the result).
# ---------------------------------------------------------------------------


def _generator_np(seed, op, opargs=(), opkwargs=None):
    opkwargs = opkwargs or {}
    if op == "ctor_random":
        g = np.random.Generator(np.random.PCG64(seed))
        return _as_comparable(g.random(*opargs, **opkwargs))
    g = np.random.default_rng(seed)
    return _as_comparable(getattr(g, op)(*opargs, **opkwargs))


def _generator_ionp(seed, op, opargs=(), opkwargs=None):
    import anionpy

    opkwargs = opkwargs or {}
    if op == "ctor_random":
        g = anionpy.random.Generator(anionpy.random.PCG64(seed))
        return _as_comparable(g.random(*opargs, **opkwargs))
    g = anionpy.random.default_rng(seed)
    return _as_comparable(getattr(g, op)(*opargs, **opkwargs))


def generator_cases():
    cases = [
        # .integers()
        ("integers_default_0_100_size10", (42, "integers"), {"opargs": (0, 100), "opkwargs": {"size": 10}}),
        ("integers_high_none_size5", (42, "integers"), {"opargs": (10,), "opkwargs": {"size": 5}}),
        ("integers_uint8_full_range_size20", (42, "integers"), {"opargs": (0, 256), "opkwargs": {"size": 20, "dtype": np.uint8}}),
        ("integers_int8_neg5_5_size6", (42, "integers"), {"opargs": (-5, 5), "opkwargs": {"size": 6, "dtype": np.int8}}),
        ("integers_bool_size10", (42, "integers"), {"opargs": (0, 2), "opkwargs": {"size": 10, "dtype": bool}}),
        ("integers_endpoint_true_size10", (42, "integers"), {"opargs": (0, 100), "opkwargs": {"size": 10, "endpoint": True}}),
        ("integers_scalar_int8", (42, "integers"), {"opargs": (5,), "opkwargs": {"size": 1, "dtype": np.int8}}),
        ("integers_uint32_scalar_size1", (42, "integers"), {"opargs": (0, 2 ** 32), "opkwargs": {"size": 1, "dtype": np.uint32}}),
        ("integers_int16_range_size8", (3, "integers"), {"opargs": (-1000, 1000), "opkwargs": {"size": 8, "dtype": np.int16}}),
        ("integers_uint16_range_size8", (3, "integers"), {"opargs": (0, 60000), "opkwargs": {"size": 8, "dtype": np.uint16}}),
        ("integers_int64_default_scalar", (99, "integers"), {"opargs": (-50, 50), "opkwargs": {}}),
        ("integers_uint64_size4", (5, "integers"), {"opargs": (0, 2 ** 40), "opkwargs": {"size": 4, "dtype": np.uint64}}),
        # error paths
        ("integers_low_ge_high", (42, "integers"), {"opargs": (5, 3), "opkwargs": {}}),
        ("integers_uint8_out_of_bounds_high", (42, "integers"), {"opargs": (300,), "opkwargs": {"dtype": np.uint8}}),
        ("integers_int8_out_of_bounds_low", (42, "integers"), {"opargs": (-300, 5), "opkwargs": {"dtype": np.int8}}),
        # .bytes()
        ("bytes_len7", (42, "bytes"), {"opargs": (7,), "opkwargs": {}}),
        ("bytes_len4", (42, "bytes"), {"opargs": (4,), "opkwargs": {}}),
        ("bytes_len0", (42, "bytes"), {"opargs": (0,), "opkwargs": {}}),
        ("bytes_len16", (42, "bytes"), {"opargs": (16,), "opkwargs": {}}),
        ("bytes_len1_seed0", (0, "bytes"), {"opargs": (1,), "opkwargs": {}}),
        # Generator(PCG64(seed)) constructor form
        ("ctor_pcg64_seed42_n5", (42, "ctor_random"), {"opargs": (5,), "opkwargs": {}}),
        ("ctor_pcg64_seed0_n3", (0, "ctor_random"), {"opargs": (3,), "opkwargs": {}}),
        # -----------------------------------------------------------------
        # Continuous distributions layered on `ionp_core::random::
        # distributions` (2026-08-02 addition). Each has a bit-exact happy
        # path plus at least one `CONS_NON_NEGATIVE`/`CONS_POSITIVE`
        # error-path case, exercised at several parameter regimes chosen to
        # cross `standard_gamma`'s shape<1 vs shape>=1 branch and `beta`'s
        # Johnk-vs-gamma-ratio branch. See this session's report for the
        # broader out-of-corpus sweep (100 seeds x fixed params, then 40
        # seeds x varied params x size=500) that also passed bit-exact but
        # isn't re-run here every CI pass.
        # -----------------------------------------------------------------
        ("standard_normal_size5", (42, "standard_normal"), {"opargs": (), "opkwargs": {"size": 5}}),
        ("standard_normal_scalar", (7, "standard_normal"), {"opargs": (), "opkwargs": {}}),
        ("normal_loc_scale_size5", (42, "normal"), {"opargs": (2.5, 0.7), "opkwargs": {"size": 5}}),
        ("normal_default_size5", (42, "normal"), {"opargs": (), "opkwargs": {"size": 5}}),
        ("normal_negative_scale", (42, "normal"), {"opargs": (0.0, -1.0), "opkwargs": {}}),
        ("standard_exponential_size5", (42, "standard_exponential"), {"opargs": (), "opkwargs": {"size": 5}}),
        ("exponential_scale3_size5", (42, "exponential"), {"opargs": (3.0,), "opkwargs": {"size": 5}}),
        ("exponential_negative_scale", (42, "exponential"), {"opargs": (-2.0,), "opkwargs": {}}),
        ("standard_gamma_shape_lt1_size5", (42, "standard_gamma"), {"opargs": (0.5,), "opkwargs": {"size": 5}}),
        ("standard_gamma_shape_gt1_size5", (42, "standard_gamma"), {"opargs": (3.5,), "opkwargs": {"size": 5}}),
        ("standard_gamma_negative_shape", (42, "standard_gamma"), {"opargs": (-1.0,), "opkwargs": {}}),
        ("gamma_shape_scale_size5", (42, "gamma"), {"opargs": (2.0, 1.5), "opkwargs": {"size": 5}}),
        ("beta_both_lt1_size5", (42, "beta"), {"opargs": (0.5, 0.5), "opkwargs": {"size": 5}}),
        ("beta_gamma_ratio_size5", (42, "beta"), {"opargs": (3.0, 5.0), "opkwargs": {"size": 5}}),
        ("beta_zero_a", (42, "beta"), {"opargs": (0.0, 1.0), "opkwargs": {}}),
        ("chisquare_df4_size5", (42, "chisquare"), {"opargs": (4.0,), "opkwargs": {"size": 5}}),
        ("chisquare_negative_df", (42, "chisquare"), {"opargs": (-1.0,), "opkwargs": {}}),
        ("f_dfnum_dfden_size5", (42, "f"), {"opargs": (3.0, 5.0), "opkwargs": {"size": 5}}),
        ("uniform_size5", (42, "uniform"), {"opargs": (-1.0, 2.0), "opkwargs": {"size": 5}}),
        ("uniform_default_size5", (42, "uniform"), {"opargs": (), "opkwargs": {"size": 5}}),
        ("uniform_negative_range", (42, "uniform"), {"opargs": (2.0, -1.0), "opkwargs": {}}),
        ("standard_cauchy_size5", (42, "standard_cauchy"), {"opargs": (), "opkwargs": {"size": 5}}),
        ("standard_t_df4_size5", (42, "standard_t"), {"opargs": (4.0,), "opkwargs": {"size": 5}}),
        ("pareto_a_size5", (42, "pareto"), {"opargs": (2.5,), "opkwargs": {"size": 5}}),
        ("pareto_zero_a", (42, "pareto"), {"opargs": (0.0,), "opkwargs": {}}),
        ("weibull_a_size5", (42, "weibull"), {"opargs": (1.5,), "opkwargs": {"size": 5}}),
        ("power_a_size5", (42, "power"), {"opargs": (2.0,), "opkwargs": {"size": 5}}),
        ("laplace_size5", (42, "laplace"), {"opargs": (2.5, 0.7), "opkwargs": {"size": 5}}),
        ("gumbel_size5", (42, "gumbel"), {"opargs": (2.5, 0.7), "opkwargs": {"size": 5}}),
        ("logistic_size5", (42, "logistic"), {"opargs": (2.5, 0.7), "opkwargs": {"size": 5}}),
        ("lognormal_size5", (42, "lognormal"), {"opargs": (0.5, 0.7), "opkwargs": {"size": 5}}),
        ("rayleigh_size5", (42, "rayleigh"), {"opargs": (2.0,), "opkwargs": {"size": 5}}),
        # -----------------------------------------------------------------
        # Declaration-pass additions (2026-08-07 session): out-of-corpus
        # probing (multiple seeds incl. >2^63, size=0/scalar/tuple, every
        # explicit integers() dtype) found two REAL divergences the corpus
        # above never exercised, both now fixed in `ionp-py/src/random.rs`:
        #
        # (1) `Generator.integers(..., size=None, dtype=<non-default,
        #     non-bool>)` returned a bare Python int (dtype-losing) instead
        #     of a width-typed numpy scalar (`numpy.int16`/`numpy.uint32`/
        #     ...) -- only the DEFAULT dtype (int64) coincidentally passed
        #     under `np.asarray(x).dtype` comparison, since a Python int's
        #     `asarray` dtype is always (u)int64. Every explicit narrower
        #     dtype at `size=None` diverged. Fixed via `int_scalar_to_py`
        #     routing through `numpy_scalar_from_0d` (mints a real,
        #     width-typed scalar).
        # (2) `Generator.integers(low, high, size=0, dtype=...)` raised
        #     (bounds/low>=high validation) where real numpy returns an
        #     empty array UNCONDITIONALLY skipping validation whenever the
        #     resolved size is 0 (confirmed:
        #     `np.random.default_rng(1).integers(500, 300, size=0,
        #     dtype=np.int8)` -> `array([], dtype=int8)`, no error, despite
        #     low>high AND both out of int8's range). Fixed by resolving
        #     `size=` to a concrete shape before running bounds validation
        #     and skipping validation whenever the resolved element count
        #     is 0.
        #
        # Both cases below were RED before the fix (confirmed by hand) and
        # are GREEN after -- kept in the corpus, not just the report, so a
        # future revert of either fix goes red here, not silently.
        ("integers_scalar_int16_dtype_fidelity", (12345, "integers"), {"opargs": (0, 1000), "opkwargs": {"dtype": np.int16}}),
        ("integers_scalar_uint32_dtype_fidelity", (12345, "integers"), {"opargs": (0, 1000), "opkwargs": {"dtype": np.uint32}}),
        ("integers_scalar_uint64_dtype_fidelity", (12345, "integers"), {"opargs": (0, 1000), "opkwargs": {"dtype": np.uint64}}),
        ("integers_size0_out_of_range_int8", (12345, "integers"), {"opargs": (500, 300), "opkwargs": {"size": 0, "dtype": np.int8}}),
        ("integers_size0_low_ge_high_default", (12345, "integers"), {"opargs": (5, 3), "opkwargs": {"size": 0}}),
        ("integers_size_0_4_out_of_range_uint8", (12345, "integers"), {"opargs": (-5, 300), "opkwargs": {"size": (0, 4), "dtype": np.uint8}}),
        # >2^63 seed coverage (none of the cases above use one): PCG64's
        # underlying seeding goes through SeedSequence's arbitrary-precision
        # entropy pool, so a seed past int64/uint64 range is a real,
        # distinct code path (int128 handling in `check_integer_bounds`'s
        # caller chain / the Rust `low: i128` extraction), not merely "a
        # bigger number".
        ("integers_seed_gt_2_63_size5", (2 ** 63 + 12345, "integers"), {"opargs": (0, 1000), "opkwargs": {"size": 5}}),
        ("standard_normal_seed_gt_2_63_size5", (2 ** 64 - 7, "standard_normal"), {"opargs": (), "opkwargs": {"size": 5}}),
        ("random_seed_gt_2_63_size5", (2 ** 63 + 999, "random"), {"opargs": (), "opkwargs": {"size": 5}}),
        # size=0 across a representative distribution set (none of the
        # happy-path cases above use size=0; only default_rng_cases does,
        # and only for `.random()`).
        ("normal_size0", (42, "normal"), {"opargs": (0.0, 1.0), "opkwargs": {"size": 0}}),
        ("gamma_size0", (42, "gamma"), {"opargs": (2.0, 1.0), "opkwargs": {"size": 0}}),
        ("integers_size0_default", (42, "integers"), {"opargs": (0, 100), "opkwargs": {"size": 0}}),
        # array-like parameter inputs (e.g. `integers(low=[0,0], high=10)`,
        # `normal(loc=np.array([1.0,2.0]))`) are a documented, explicit,
        # loud-failure scope boundary -- see `anionpy/_state/random.py` for
        # the full writeup, matching `linalg.svd`'s non-batched-only
        # precedent (`anionpy/_state/linalg.py`) of declaring the covered
        # path while explicitly scoping out a documented-but-unsupported
        # one, as long as the unsupported path fails LOUDLY. NOT encoded as
        # a case here: numpy's side SUCCEEDS on array-like input (returns a
        # broadcast array), so this is not a matched-error-pair shape this
        # harness's error-path mode (which asserts both sides raise the
        # SAME exception) can express; verified by hand instead (see
        # `anionpy/_state/random.py`'s declaration comment for the
        # transcript).
        # -----------------------------------------------------------------
        # Discrete distributions + wald/vonmises/triangular/noncentral_*
        # (2026-08-13 addition). Each has a bit-exact happy path at
        # several seeds/parameter regimes chosen to cross the algorithm's
        # own internal branches (poisson mult-vs-ptrs at lam=10, binomial
        # inversion-vs-BTPE at n*p=30, geometric search-vs-inversion at
        # p=1/3, vonmises's four kappa regimes, noncentral_chisquare's
        # df>1-vs-df<=1 split), plus at least one CONS_*-transcribed
        # error-path case per method.
        # -----------------------------------------------------------------
        ("wald_size5", (42, "wald"), {"opargs": (3.0, 2.0), "opkwargs": {"size": 5}}),
        ("wald_seed7_size8", (7, "wald"), {"opargs": (1.0, 4.0), "opkwargs": {"size": 8}}),
        ("wald_nonpositive_mean", (42, "wald"), {"opargs": (0.0, 2.0), "opkwargs": {}}),
        ("wald_nonpositive_scale", (42, "wald"), {"opargs": (3.0, -1.0), "opkwargs": {}}),
        ("vonmises_normal_kappa_size5", (42, "vonmises"), {"opargs": (0.5, 4.0), "opkwargs": {"size": 5}}),
        ("vonmises_tiny_kappa_size5", (42, "vonmises"), {"opargs": (0.0, 1e-9), "opkwargs": {"size": 5}}),
        ("vonmises_taylor_kappa_size5", (42, "vonmises"), {"opargs": (0.0, 1e-6), "opkwargs": {"size": 5}}),
        ("vonmises_huge_kappa_size5", (42, "vonmises"), {"opargs": (1.0, 1e7), "opkwargs": {"size": 5}}),
        ("vonmises_negative_kappa", (42, "vonmises"), {"opargs": (0.0, -1.0), "opkwargs": {}}),
        ("triangular_size5", (42, "triangular"), {"opargs": (0.0, 0.3, 1.0), "opkwargs": {"size": 5}}),
        ("triangular_left_gt_mode", (42, "triangular"), {"opargs": (0.5, 0.3, 1.0), "opkwargs": {}}),
        ("triangular_mode_gt_right", (42, "triangular"), {"opargs": (0.0, 1.5, 1.0), "opkwargs": {}}),
        ("triangular_left_eq_right", (42, "triangular"), {"opargs": (1.0, 1.0, 1.0), "opkwargs": {}}),
        ("noncentral_chisquare_df_gt1_size5", (42, "noncentral_chisquare"), {"opargs": (3.0, 5.0), "opkwargs": {"size": 5}}),
        ("noncentral_chisquare_df_lt1_size5", (42, "noncentral_chisquare"), {"opargs": (0.5, 5.0), "opkwargs": {"size": 5}}),
        ("noncentral_chisquare_nonpositive_df", (42, "noncentral_chisquare"), {"opargs": (0.0, 5.0), "opkwargs": {}}),
        ("noncentral_chisquare_negative_nonc", (42, "noncentral_chisquare"), {"opargs": (3.0, -1.0), "opkwargs": {}}),
        ("noncentral_f_size5", (42, "noncentral_f"), {"opargs": (3.0, 10.0, 5.0), "opkwargs": {"size": 5}}),
        ("noncentral_f_nonpositive_dfnum", (42, "noncentral_f"), {"opargs": (0.0, 10.0, 5.0), "opkwargs": {}}),
        ("noncentral_f_nonpositive_dfden", (42, "noncentral_f"), {"opargs": (3.0, 0.0, 5.0), "opkwargs": {}}),
        ("poisson_small_lam_size5", (42, "poisson"), {"opargs": (5.0,), "opkwargs": {"size": 5}}),
        ("poisson_large_lam_size5", (42, "poisson"), {"opargs": (50.0,), "opkwargs": {"size": 5}}),
        ("poisson_zero_lam", (42, "poisson"), {"opargs": (0.0,), "opkwargs": {"size": 3}}),
        ("poisson_negative_lam", (42, "poisson"), {"opargs": (-1.0,), "opkwargs": {}}),
        ("binomial_inversion_size5", (42, "binomial"), {"opargs": (20, 0.3), "opkwargs": {"size": 5}}),
        ("binomial_btpe_size5", (42, "binomial"), {"opargs": (500, 0.4), "opkwargs": {"size": 5}}),
        ("binomial_negative_n", (42, "binomial"), {"opargs": (-1, 0.5), "opkwargs": {}}),
        ("binomial_p_out_of_range", (42, "binomial"), {"opargs": (10, 1.5), "opkwargs": {}}),
        ("negative_binomial_size5", (42, "negative_binomial"), {"opargs": (5.0, 0.4), "opkwargs": {"size": 5}}),
        ("negative_binomial_nonpositive_n", (42, "negative_binomial"), {"opargs": (0.0, 0.4), "opkwargs": {}}),
        ("negative_binomial_p_out_of_range", (42, "negative_binomial"), {"opargs": (5.0, 0.0), "opkwargs": {}}),
        ("negative_binomial_n_too_large", (42, "negative_binomial"), {"opargs": (1e17, 1e-10), "opkwargs": {}}),
        ("geometric_search_size5", (42, "geometric"), {"opargs": (0.3,), "opkwargs": {"size": 5}}),
        ("geometric_inversion_size5", (42, "geometric"), {"opargs": (0.9,), "opkwargs": {"size": 5}}),
        ("geometric_p_out_of_range", (42, "geometric"), {"opargs": (0.0,), "opkwargs": {}}),
        ("zipf_size5", (42, "zipf"), {"opargs": (2.0,), "opkwargs": {"size": 5}}),
        ("zipf_a_le_1", (42, "zipf"), {"opargs": (1.0,), "opkwargs": {}}),
        ("logseries_size5", (42, "logseries"), {"opargs": (0.6,), "opkwargs": {"size": 5}}),
        ("logseries_p_out_of_range", (42, "logseries"), {"opargs": (1.0,), "opkwargs": {}}),
        ("hypergeometric_size5", (42, "hypergeometric"), {"opargs": (15, 15, 10), "opkwargs": {"size": 5}}),
        ("hypergeometric_hrua_size5", (42, "hypergeometric"), {"opargs": (500, 500, 400), "opkwargs": {"size": 5}}),
        ("hypergeometric_bad_ge_max", (42, "hypergeometric"), {"opargs": (10 ** 9, 5, 3), "opkwargs": {}}),
        ("hypergeometric_nsample_gt_total", (42, "hypergeometric"), {"opargs": (5, 5, 20), "opkwargs": {}}),
        # size=0 across the new discrete/continuous methods.
        ("poisson_size0", (42, "poisson"), {"opargs": (5.0,), "opkwargs": {"size": 0}}),
        ("binomial_size0", (42, "binomial"), {"opargs": (10, 0.5), "opkwargs": {"size": 0}}),
        ("wald_size0", (42, "wald"), {"opargs": (3.0, 2.0), "opkwargs": {"size": 0}}),
        # >2^63 seed coverage for a representative new method.
        ("poisson_seed_gt_2_63_size5", (2 ** 63 + 555, "poisson"), {"opargs": (7.0,), "opkwargs": {"size": 5}}),
    ]
    return cases


# ---------------------------------------------------------------------------
# random.SeedSequence.generate_state
# ---------------------------------------------------------------------------


def _seedseq_np(seed, n_words, dtype=None):
    kwargs = {} if dtype is None else {"dtype": dtype}
    return np.random.SeedSequence(seed).generate_state(n_words, **kwargs)


def _seedseq_ionp(seed, n_words, dtype=None):
    import anionpy

    kwargs = {} if dtype is None else {"dtype": dtype}
    return anionpy.random.SeedSequence(seed).generate_state(n_words, **kwargs)


def seedsequence_cases():
    return [
        ("u32_n4_seed0", (0, 4), {}),
        ("u64_n2_seed42", (42, 2), {"dtype": np.uint64}),
        ("u32_n1_seed7", (7, 1), {}),
        ("u32_n8_seed123456789", (123456789, 8), {}),
        # error paths
        ("negative_seed", (-1, 4), {}),
        ("float_seed", (5.5, 4), {}),
    ]


# ---------------------------------------------------------------------------
# random.PCG64 / random.PCG64DXSM: constructor + .random_raw()
# ---------------------------------------------------------------------------


def _pcg64_random_raw_np(seed, size=None):
    kwargs = {} if size is None else {"size": size}
    return _as_comparable(np.random.PCG64(seed).random_raw(**kwargs))


def _pcg64_random_raw_ionp(seed, size=None):
    import anionpy

    kwargs = {} if size is None else {"size": size}
    return _as_comparable(anionpy.random.PCG64(seed).random_raw(**kwargs))


def pcg64_random_raw_cases():
    return [
        ("n4_seed42", (42,), {"size": 4}),
        ("n1_seed0", (0,), {"size": 1}),
        ("scalar_seed7", (7,), {}),
        # error paths
        ("negative_seed", (-2,), {}),
        ("float_seed", (1.5,), {}),
    ]


def _pcg64dxsm_random_raw_np(seed, size=None):
    kwargs = {} if size is None else {"size": size}
    return _as_comparable(np.random.PCG64DXSM(seed).random_raw(**kwargs))


def _pcg64dxsm_random_raw_ionp(seed, size=None):
    import anionpy

    kwargs = {} if size is None else {"size": size}
    return _as_comparable(anionpy.random.PCG64DXSM(seed).random_raw(**kwargs))


def pcg64dxsm_random_raw_cases():
    return [
        ("n4_seed42", (42,), {"size": 4}),
        ("n1_seed0", (0,), {"size": 1}),
        ("scalar_seed7", (7,), {}),
    ]


RANDOM_SPECS: dict[str, ItemSpec] = {
    "random.default_rng": ItemSpec(
        name="random.default_rng", kind="custom",
        custom_cases=default_rng_cases,
        numpy_adapter=_default_rng_np, ionp_adapter=_default_rng_ionp,
        atol=0.0, rtol=0.0,
    ),
    "random.Generator": ItemSpec(
        name="random.Generator", kind="custom",
        custom_cases=generator_cases,
        numpy_adapter=_generator_np, ionp_adapter=_generator_ionp,
        atol=0.0, rtol=0.0,
    ),
    "random.SeedSequence": ItemSpec(
        name="random.SeedSequence", kind="custom",
        custom_cases=seedsequence_cases,
        numpy_adapter=_seedseq_np, ionp_adapter=_seedseq_ionp,
        atol=0.0, rtol=0.0,
    ),
    "random.PCG64": ItemSpec(
        name="random.PCG64", kind="custom",
        custom_cases=pcg64_random_raw_cases,
        numpy_adapter=_pcg64_random_raw_np, ionp_adapter=_pcg64_random_raw_ionp,
        atol=0.0, rtol=0.0,
    ),
    "random.PCG64DXSM": ItemSpec(
        name="random.PCG64DXSM", kind="custom",
        custom_cases=pcg64dxsm_random_raw_cases,
        numpy_adapter=_pcg64dxsm_random_raw_np, ionp_adapter=_pcg64dxsm_random_raw_ionp,
        atol=0.0, rtol=0.0,
    ),
}
