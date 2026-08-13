"""Builds the 134 kind="ufunc" ItemSpec entries and merges them into the
real registry.REGISTRY.

Names come from `ufunc_introspect.UFUNC_NAMES` -- the same
tools/numpy_surface.json-derived list ufunc_cases.py and ufunc_introspect.py
use -- never hand-typed here either. `multi_output` is set from each
ufunc's own `.nout` (via UfuncInfo), not guessed: only divmod/modf/frexp in
the pinned 134 have nout>1, and that's discovered by asking numpy, not by
this module knowing their names.

Importing this module has ONE side effect: it inserts 134 entries into
registry.REGISTRY (keyed by the exact numpy_surface.json name, e.g. "add",
"char.add", "strings.equal" -- the same key coverage.py's ledger and
run.py's --out report use). It does not touch anionpy.__ion_state__ or any
file outside tests/ -- with zero ufuncs declared there today, every one of
these 134 items resolves ionp_fn=None and reports "absent" via harness.py's
existing short-circuit, which is the correct, honest state pre-
implementation. See GOAL-ionp.md / the ufunc-block task brief.

atol/rtol were originally set uniformly to a non-zero 1e-9 for every ufunc
regardless of whether that specific ufunc's dtype loops are actually
inexact. That was a truthfulness hole (2026-08-01): a non-zero atol/rtol
grades float/complex results via np.allclose -- a looser, relative-epsilon
bar than 1 ULP -- with no recorded justification, and any ufunc credited
under it (i.e. actually implemented and passing) would be silently
mis-reported as "bit-exact" by the coverage ledger. Measured directly
against the six ufuncs that are both implemented on anionpy today and
currently credited (add, subtract, multiply, maximum, minimum, negative):
MAX ULP DISTANCE = 0 across every float/complex case in the differential
corpus (158 cases each, 70 for negative) -- the tolerance was never
needed. atol/rtol are therefore now uniformly 0.0 (exact) instead of
1e-9: this is both accurate for those six and the conservative default for
the other 128 ufuncs, which are not implemented on anionpy at all yet ("absent")
and so have nothing to measure. `registry.ItemSpec.__post_init__` requires
a non-empty `epsilon_justification` for any NON-ZERO atol/rtol (mirroring
the pre-existing ulp_tolerance guard); 0.0/0.0 needs none, since it is not
a tolerance. When a currently-absent ufunc is implemented and turns out to
need real slack, its entry here must be given an explicit ulp_tolerance
(preferred) or a justified non-zero atol/rtol -- never silently reuse this
uniform 0.0.

An exact (int/bool) result never reaches the atol/rtol branch in
harness.compare_values (it takes the np.array_equal path instead), so a
declared 0.0/0.0 on an all-integer ufunc is inert, not wrong.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401

import anionpy
import registry
import ufunc_introspect


# Per-item, PER-DTYPE ULP tolerance overrides.
#
# SUPERSEDES the previous version of this block (2026-08-01, same day, twice
# over):
#
# 1. Sample-size fix: the original version measured `max_ulp_observed`
#    against the ~150/262-case differential corpus only, which is EDGE-CASE
#    BREADTH, not statistical VOLUME -- and a corpus that small can simply
#    not contain the discrepancy a bound needs to cover. It didn't: a
#    follow-up 20,000-value seeded sweep found the declared `absolute`/`abs`
#    bound (1.0 ULP) was already wrong (true max 2.0 ULP), and the declared
#    `divide`/`true_divide` bound (17.0 ULP, attributed to "2 chained
#    divisions compounding float32 rounding") was never actually swept at
#    adequate volume either -- it came from ONE (3,4)-shaped case (4 output
#    elements). See KNOWN-DIFFERENCES.md's "sample-size defect fixed"
#    section and `ulp_sweep.py` (the module that produces the numbers below)
#    for the full account.
# 2. Per-dtype grading fix: even after (1), the re-swept `absolute`/`abs`
#    evidence was ITSELF per-dtype (float32/float64 bit-exact at 0.0 ULP,
#    complex64/complex128 genuinely imprecise at 2.0 ULP) but was declared
#    as one item-wide `ulp_tolerance=2.0` scalar -- giving the bit-exact
#    real dtypes 2 ULP of unearned slack their own measurement showed they
#    never needed. `ulp_tolerance` is now `{dtype_name: bound}`, graded
#    per dtype (harness.compare_values), and `registry.ItemSpec.__post_init__`
#    requires this dict's key set to exactly match `ulp_sweep`'s -- see
#    registry.py's `MIN_ULP_SWEEP_N` docstring for the full account.
#
# `registry.ItemSpec.__post_init__` refuses to construct any item with a
# `ulp_tolerance` unless it also carries `ulp_sweep` evidence measured at
# `registry.MIN_ULP_SWEEP_N` (20,000) seeded values per dtype -- this dict
# is required to satisfy that gate, not just documented prose.
#
#   absolute / abs: FIXED, not just re-measured, 2026-08-02. The 2.0-ULP
#     complex64/complex128 bound recorded immediately below this note
#     (2026-08-01) was a real algorithm defect, not an inherent floor: anionpy's
#     complex `Absolute` was calling the platform libm `hypot`/`cabs`
#     (`f32::hypot`/`f64::hypot` in `ufunc.rs`), which is bit-identical to
#     itself (verified via `ctypes` against `libSystem`, 20,000 seeded pairs,
#     0 mismatches) but NOT what real numpy's complex `np.abs` actually
#     computes for the contiguous arrays this sweep/corpus exercise. Real
#     numpy routes through a SIMD-vectorized ufunc loop
#     (`simd_cabsolute_<sfx>` in `loops_unary_complex.dispatch.c.src`,
#     active on Apple Silicon's NEON `npyv_` backend too) that computes a
#     different scaled-hypot formula: `max(|re|,|im|) *
#     sqrt(fma(ratio,ratio,1.0))` with `ratio = min(|re|,|im|)/max(|re|,|im|)`
#     and explicit inf/nan special-casing. `ufunc.rs`'s `numpy_complex_abs`
#     now reproduces that formula (mul_add, not a separate multiply+add, to
#     match numpy's true FMA) instead of calling libm hypot. Re-swept after
#     the fix, same n=20,000/dtype, SEED=20260731: float32/float64/
#     complex64/complex128 ALL 0.0 ULP (0/20000 disagreeing on every dtype).
#     Since every dtype is now genuinely bit-exact, there is nothing left to
#     declare a `ulp_tolerance` FOR -- these two names carry NO entry in
#     `_ULP_OVERRIDES` below (removed, not zeroed-and-kept: a 0.0 tolerance
#     is not a tolerance, same reasoning the module docstring above gives for
#     the six bit-exact arithmetic ufuncs' 0.0 atol/rtol) and fall back to
#     the harness's ordinary bit-exact byte-level grading, which they now
#     pass on their own. See KNOWN-DIFFERENCES.md's complex-`abs` ULP fix
#     entry and `ufunc.rs`'s `numpy_complex_abs` doc comment for the full
#     account. (The old 1.0-ULP-bound / 2.0-ULP-bound history below is kept
#     for the record, not because either bound is still declared.)
#   divide / true_divide: DELIBERATELY LEFT UNDECLARED (no entry below).
#     The single-call ("call" form, no compounding) sweep is clean: 0.0 ULP
#     float32/float64, 1.0 ULP complex64/complex128 -- but `divide`/
#     `true_divide` are binary (nin=2) ufuncs, so their real corpus also
#     exercises `.reduce`/`.accumulate`/`.outer`/`.at`, which CHAIN multiple
#     divisions. A chained-division sweep (20,000 independent 2-division
#     chains, i.e. the same shape as the retracted 17.0 claim) measured
#     max_ulp = 31228.0 (complex64) / 14581.0 (complex128) -- three orders
#     of magnitude past the old figure -- and grows further with longer
#     chains (5-division chains: 79990.0 / 239968.0), driven by rare
#     near-zero-denominator cancellation rather than a fixed compounding
#     constant. Since `ulp_tolerance` applies uniformly to every case in an
#     item's corpus regardless of call form, and there is no finite,
#     honestly-measured bound that covers the compounding path, no
#     `ulp_tolerance` is declared for either name -- they fall back to the
#     harness's default bit-exact grading. See KNOWN-DIFFERENCES.md for the
#     full measurement and reports/ionp-ulp-tolerance-decision-2026-08-01.md
#     for why padding a "safe-looking" number instead is exactly the
#     tuning-the-ruler failure this policy forbids. (Both names were
#     already failing the differential suite for an unrelated float16 gap
#     before this change -- see KNOWN-DIFFERENCES.md -- so removing the
#     declaration does not newly break anything that was passing.)
_ULP_OVERRIDES: dict[str, tuple[dict, str, dict]] = {
    # `absolute` / `abs`: NO entry -- fixed to genuine 0.0 ULP on every
    # dtype (float32/float64/complex64/complex128) 2026-08-02. See the
    # module comment above for the fix and re-measurement; a bit-exact item
    # needs no `ulp_tolerance` declaration at all.

    # --- 2026-08-01, C99-Annex-G-libm follow-up (commit 12bf939) ---------
    #
    # 12bf939 routed the complex ufunc arms to the platform's C99 Annex G
    # libm, which moved 18 transcendental items from ULP distance = infinity
    # (unfixable by any tolerance -- NaN/inf-ness mismatches, which
    # harness.ulp_distance treats as never-within-tolerance by design) to
    # finite ULP. 6 of those 18 went fully bit-exact and were declared in
    # d93aa09 (arccos/arcsin/arctan/sqrt/tan/log1p). The 12 below are the
    # rest of that list: cos sin tanh cosh sinh exp exp2 expm1 log2 arccosh
    # arcsinh arctanh.
    #
    # Measured via a purpose-built DOMAIN-AWARE sweep (NOT ulp_sweep.py's
    # uniform `standard_normal()*10`), n=24000/dtype (6 buckets x 4000,
    # every bucket >= the 20000 MIN_ULP_SWEEP_N floor), SEED=20260731:
    # generic normal, near branch-cut/pole/domain-edge (multiples of pi/2
    # for cos/sin, tanh's poles at (n+1/2)*i*pi, log2/arccosh/arctanh's
    # branch cuts, arcsinh/arctanh's branch points at +-i/+-1), subnormal,
    # overflow-boundary, and broad-uniform. 0 skipped comparisons on every
    # (item, dtype) pair measured -- every element of every 24000-sample
    # array was graded, none dropped.
    #
    # Real dtypes for arccosh/arcsinh/arctanh, and complex64/complex128 for
    # expm1, are DELIBERATELY NOT declared below -- they hit genuine defects
    # this sweep caught, not borderline precision:
    #   - arccosh float32/float64: ULP=inf. Large x (e.g. x=2.3326e+38 for
    #     float32, x=1.7696e+308 for float64) makes numpy return the correct
    #     finite result (~89.04 / ~710.46, via ln(2x) asymptotics) while
    #     anionpy returns +inf -- an internal overflow in anionpy's formula (not a
    #     rounding difference), so no finite ULP bound can cover it.
    #   - arcsinh float32/float64: same shape of defect. x=1.8508e+38
    #     (float32) -> numpy 88.807, anionpy +inf. x=9.7776e+307 (float64) ->
    #     numpy 709.867, anionpy +inf.
    #   - arctanh float32/float64: finite but absurd ULP (363409.0 float32,
    #     310581023.0 float64) from catastrophic cancellation near the
    #     x=+-1 poles -- e.g. float64 x=-0.9999999998993775: numpy
    #     -11.856396151563919 vs anionpy -11.856396703266643, an ~5.5e-10
    #     absolute gap that is ~3e8 ULP at that magnitude. A padded bound
    #     here would be exactly the "safety margin" this policy forbids
    #     (see `divide`'s retraction above for precedent).
    #   - expm1 complex64/complex128: absurd ULP (24820251.0 / 1.33e16).
    #     Near-subnormal-magnitude complex inputs (e.g. complex64
    #     z=-4.605e-38+7.423e-39j) make anionpy's real component collapse to
    #     exactly 0.0 instead of the correct ~-4.605e-38 -- a real-part
    #     dropout bug, not a rounding difference.
    # These six (item, dtype) pairs are BLOCKED, not declared, per rule 4/5
    # of the 2026-08-01 ULP-declarability task: an inf or absurd measured
    # max is evidence of a defect, not a tolerance to write down. See the
    # task's verification report for the full per-item accounting.
    # Int/bool source-dtype keys below are graded against harness.py's
    # _first_operand_dtype (the SOURCE array's dtype, e.g. "int16"), NOT the
    # promoted float result dtype -- see ufunc_registry.py module comment.
    # Each was measured independently (same n=24000, SEED=20260731) because
    # numpy's ufunc promotion sends bool/int8/uint8 -> float16 results,
    # int16/uint16 -> float32, int32/int64/uint32/uint64 -> float64, and
    # those promoted widths do not always share the corresponding native
    # float dtype's ULP -- e.g. cos(int16) round-trips through float32 with
    # 1 ULP disagreements even where cos(float64) is bit-exact.
    "cos": ({"float32": 1.0, "float64": 0.0, "complex64": 0.0, "complex128": 0.0,
             "bool": 0.0, "int8": 0.0, "int16": 1.0, "int32": 0.0, "int64": 0.0,
             "uint8": 0.0, "uint16": 1.0, "uint32": 0.0, "uint64": 0.0},
            "domain-aware sweep incl. multiples of pi/2, large-magnitude "
            "argument reduction, subnormals; n=24000/dtype, SEED=20260731 "
            "-- float32/int16/uint16 (promote to float32) 1 ULP ordinary "
            "rounding, float64/complex64/complex128 and all other int/bool "
            "dtypes (promote to float16 or float64) bit-exact. See "
            "ufunc_registry.py module comment.",
            {"float32": (24000, 1.0), "float64": (24000, 0.0),
             "complex64": (24000, 0.0), "complex128": (24000, 0.0),
             "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 1.0),
             "int32": (24000, 0.0), "int64": (24000, 0.0), "uint8": (24000, 0.0),
             "uint16": (24000, 1.0), "uint32": (24000, 0.0), "uint64": (24000, 0.0)}),
    "sin": ({"float32": 1.0, "float64": 0.0, "complex64": 0.0, "complex128": 0.0,
             "bool": 0.0, "int8": 0.0, "int16": 1.0, "int32": 0.0, "int64": 0.0,
             "uint8": 0.0, "uint16": 1.0, "uint32": 0.0, "uint64": 0.0},
            "same sweep/methodology as cos (see ufunc_registry.py module "
            "comment); n=24000/dtype, SEED=20260731 -- float32/int16/uint16 "
            "1 ULP ordinary rounding, float64/complex64/complex128 and all "
            "other int/bool dtypes bit-exact.",
            {"float32": (24000, 1.0), "float64": (24000, 0.0),
             "complex64": (24000, 0.0), "complex128": (24000, 0.0),
             "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 1.0),
             "int32": (24000, 0.0), "int64": (24000, 0.0), "uint8": (24000, 0.0),
             "uint16": (24000, 1.0), "uint32": (24000, 0.0), "uint64": (24000, 0.0)}),
    "tanh": ({"float32": 1.0, "float64": 1.0, "complex64": 0.0, "complex128": 0.0,
              "bool": 0.0, "int8": 0.0, "int16": 0.0, "int32": 1.0, "int64": 1.0,
              "uint8": 0.0, "uint16": 0.0, "uint32": 1.0, "uint64": 1.0},
             "domain-aware sweep incl. saturation edge (|x| in [10,30]), "
             "poles at (n+1/2)*i*pi (complex), subnormals; n=24000/dtype, "
             "SEED=20260731 -- float32/float64/int32/int64/uint32/uint64 "
             "(the dtypes that promote to float64 or exercise saturation) "
             "1 ULP ordinary rounding, complex64/complex128/bool/int8/"
             "int16/uint8/uint16 bit-exact. See ufunc_registry.py module "
             "comment.",
             {"float32": (24000, 1.0), "float64": (24000, 1.0),
              "complex64": (24000, 0.0), "complex128": (24000, 0.0),
              "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 0.0),
              "int32": (24000, 1.0), "int64": (24000, 1.0), "uint8": (24000, 0.0),
              "uint16": (24000, 0.0), "uint32": (24000, 1.0), "uint64": (24000, 1.0)}),
    "cosh": ({"float32": 0.0, "float64": 0.0, "complex64": 0.0, "complex128": 0.0,
              "bool": 0.0, "int8": 0.0, "int16": 0.0, "int32": 0.0, "int64": 0.0,
              "uint8": 0.0, "uint16": 0.0, "uint32": 0.0, "uint64": 0.0},
             "domain-aware sweep incl. overflow boundary (|x| near "
             "ln(dtype max)), subnormals; n=24000/dtype, SEED=20260731 -- "
             "bit-exact (0 disagreements) on all 13 dtypes. Note: the item "
             "still fails the differential suite overall on an unrelated "
             "int8 `.at` call-form defect ('exact value mismatch (dtype "
             "int8)') outside this task's ULP scope -- see "
             "KNOWN-DIFFERENCES.md. See ufunc_registry.py module comment.",
             {"float32": (24000, 0.0), "float64": (24000, 0.0),
              "complex64": (24000, 0.0), "complex128": (24000, 0.0),
              "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 0.0),
              "int32": (24000, 0.0), "int64": (24000, 0.0), "uint8": (24000, 0.0),
              "uint16": (24000, 0.0), "uint32": (24000, 0.0), "uint64": (24000, 0.0)}),
    "sinh": ({"float32": 0.0, "float64": 0.0, "complex64": 0.0, "complex128": 0.0,
              "bool": 0.0, "int8": 0.0, "int16": 0.0, "int32": 0.0, "int64": 0.0,
              "uint8": 0.0, "uint16": 0.0, "uint32": 0.0, "uint64": 0.0},
             "same sweep/methodology as cosh (see ufunc_registry.py module "
             "comment); n=24000/dtype, SEED=20260731 -- bit-exact on all "
             "13 dtypes. Same unrelated int8 `.at` defect note as cosh "
             "applies.",
             {"float32": (24000, 0.0), "float64": (24000, 0.0),
              "complex64": (24000, 0.0), "complex128": (24000, 0.0),
              "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 0.0),
              "int32": (24000, 0.0), "int64": (24000, 0.0), "uint8": (24000, 0.0),
              "uint16": (24000, 0.0), "uint32": (24000, 0.0), "uint64": (24000, 0.0)}),
    "exp": ({"float32": 0.0, "float64": 0.0, "complex64": 0.0, "complex128": 0.0,
             "bool": 0.0, "int8": 0.0, "int16": 0.0, "int32": 0.0, "int64": 0.0,
             "uint8": 0.0, "uint16": 0.0, "uint32": 0.0, "uint64": 0.0},
            "domain-aware sweep incl. overflow boundary (near ln(dtype "
            "max)), deep underflow, subnormals; n=24000/dtype, "
            "SEED=20260731 -- bit-exact on all 13 dtypes. Same unrelated "
            "int8 `.at` defect note as cosh applies. See ufunc_registry.py "
            "module comment.",
            {"float32": (24000, 0.0), "float64": (24000, 0.0),
             "complex64": (24000, 0.0), "complex128": (24000, 0.0),
             "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 0.0),
             "int32": (24000, 0.0), "int64": (24000, 0.0), "uint8": (24000, 0.0),
             "uint16": (24000, 0.0), "uint32": (24000, 0.0), "uint64": (24000, 0.0)}),
    "exp2": ({"float32": 0.0, "float64": 0.0, "complex64": 0.0, "complex128": 0.0,
              "bool": 0.0, "int8": 0.0, "int16": 0.0, "int32": 0.0, "int64": 0.0,
              "uint8": 0.0, "uint16": 0.0, "uint32": 0.0, "uint64": 0.0},
             "same sweep/methodology as exp but overflow boundary at "
             "log2(dtype max); n=24000/dtype, SEED=20260731 -- bit-exact "
             "on all 13 dtypes. Same unrelated int8 `.at` defect note as "
             "cosh applies. See ufunc_registry.py module comment.",
             {"float32": (24000, 0.0), "float64": (24000, 0.0),
              "complex64": (24000, 0.0), "complex128": (24000, 0.0),
              "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 0.0),
              "int32": (24000, 0.0), "int64": (24000, 0.0), "uint8": (24000, 0.0),
              "uint16": (24000, 0.0), "uint32": (24000, 0.0), "uint64": (24000, 0.0)}),
    "expm1": ({"float32": 0.0, "float64": 0.0,
               "bool": 0.0, "int8": 0.0, "int16": 0.0, "int32": 0.0, "int64": 0.0,
               "uint8": 0.0, "uint16": 0.0, "uint32": 0.0, "uint64": 0.0},
              "domain-aware sweep incl. near-zero (expm1's precision-"
              "critical region), overflow boundary, subnormals; "
              "n=24000/dtype, SEED=20260731 -- float32/float64 and all "
              "int/bool dtypes (which never reach the subnormal-magnitude "
              "trigger region below) bit-exact (0 disagreements). "
              "complex64/complex128 DELIBERATELY NOT declared -- BLOCKED, "
              "see ufunc_registry.py module comment (real-part dropout "
              "near subnormal-magnitude inputs, 24820251.0 / 1.33e16 "
              "ULP).",
              {"float32": (24000, 0.0), "float64": (24000, 0.0),
               "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 0.0),
               "int32": (24000, 0.0), "int64": (24000, 0.0), "uint8": (24000, 0.0),
               "uint16": (24000, 0.0), "uint32": (24000, 0.0), "uint64": (24000, 0.0)}),
    "log2": ({"float32": 0.0, "float64": 0.0, "complex64": 1.0, "complex128": 1.0,
              "bool": 0.0, "int8": 0.0, "int16": 0.0, "int32": 0.0, "int64": 0.0,
              "uint8": 0.0, "uint16": 0.0, "uint32": 0.0, "uint64": 0.0},
             "domain-aware sweep incl. near x=1 (log2=0 crossing), branch "
             "cut on negative real axis (complex), near-origin, "
             "subnormals; n=24000/dtype, SEED=20260731 -- float32/float64 "
             "and all int/bool dtypes bit-exact, complex64/complex128 1 "
             "ULP ordinary rounding. See ufunc_registry.py module comment.",
             {"float32": (24000, 0.0), "float64": (24000, 0.0),
              "complex64": (24000, 1.0), "complex128": (24000, 1.0),
              "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 0.0),
              "int32": (24000, 0.0), "int64": (24000, 0.0), "uint8": (24000, 0.0),
              "uint16": (24000, 0.0), "uint32": (24000, 0.0), "uint64": (24000, 0.0)}),
    "arccosh": ({"complex64": 0.0, "complex128": 0.0,
                 "bool": 0.0, "int8": 0.0, "int16": 1.0, "int32": 1.0, "int64": 1.0,
                 "uint8": 0.0, "uint16": 1.0, "uint32": 1.0, "uint64": 1.0},
                "domain-aware sweep incl. near branch cut (-inf,1) and "
                "branch point z=1, large magnitude; n=24000/dtype, "
                "SEED=20260731 -- complex64/complex128 bit-exact (0 "
                "disagreements). int/bool dtypes graded separately -- "
                "their largest representable magnitude (int64/uint64 max "
                "~1.8e19-1.8e19) never reaches the overflow trigger region "
                "(~2.3e38 float32 / ~1.77e308 float64) that blocks the "
                "native float dtypes below, so int16/int32/int64/uint16/"
                "uint32/uint64 (which promote to float32/float64) show "
                "only ordinary 1 ULP rounding and bool/int8/uint8 (promote "
                "to float16) are bit-exact. float32/float64 DELIBERATELY "
                "NOT declared -- BLOCKED, see ufunc_registry.py module "
                "comment (large-x overflow to +inf where numpy returns a "
                "finite ln(2x)-asymptotic result, ULP=inf).",
                {"complex64": (24000, 0.0), "complex128": (24000, 0.0),
                 "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 1.0),
                 "int32": (24000, 1.0), "int64": (24000, 1.0), "uint8": (24000, 0.0),
                 "uint16": (24000, 1.0), "uint32": (24000, 1.0), "uint64": (24000, 1.0)}),
    "arcsinh": ({"complex64": 0.0, "complex128": 0.0,
                 "bool": 0.0, "int8": 0.0, "int16": 1.0, "int32": 1.0, "int64": 1.0,
                 "uint8": 1.0, "uint16": 1.0, "uint32": 1.0, "uint64": 1.0},
                "domain-aware sweep incl. near branch cuts +-i[1,inf) and "
                "branch points +-i, large magnitude; n=24000/dtype, "
                "SEED=20260731 -- complex64/complex128 bit-exact (0 "
                "disagreements). int/bool dtypes graded separately, same "
                "overflow-magnitude reasoning as arccosh above -- integer "
                "domains never reach the trigger region, so int16/int32/"
                "int64/uint16/uint32/uint64/uint8 show ordinary 1 ULP "
                "rounding and bool/int8 are bit-exact. float32/float64 "
                "DELIBERATELY NOT declared -- BLOCKED, see "
                "ufunc_registry.py module comment (same large-x "
                "overflow-to-+inf defect as arccosh, ULP=inf).",
                {"complex64": (24000, 0.0), "complex128": (24000, 0.0),
                 "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 1.0),
                 "int32": (24000, 1.0), "int64": (24000, 1.0), "uint8": (24000, 1.0),
                 "uint16": (24000, 1.0), "uint32": (24000, 1.0), "uint64": (24000, 1.0)}),
    "arctanh": ({"complex64": 0.0, "complex128": 0.0,
                 "bool": 0.0, "int8": 0.0, "int16": 0.0, "int32": 0.0, "int64": 0.0,
                 "uint8": 0.0, "uint16": 0.0, "uint32": 0.0, "uint64": 0.0},
                "domain-aware sweep incl. near branch cuts (-inf,-1]/"
                "[1,inf) and branch points +-1, near zero; n=24000/dtype, "
                "SEED=20260731 -- complex64/complex128 bit-exact (0 "
                "disagreements). int/bool dtypes graded separately -- "
                "integer inputs only land on {-1,0,1} within arctanh's "
                "domain (all else is |x|>1, nan on both sides), so all 9 "
                "int/bool dtypes are trivially bit-exact. float32/float64 "
                "DELIBERATELY NOT declared -- BLOCKED, see "
                "ufunc_registry.py module comment (catastrophic "
                "cancellation near x=+-1 poles, absurd finite ULP: "
                "363409.0 float32 / 310581023.0 float64).",
                {"complex64": (24000, 0.0), "complex128": (24000, 0.0),
                 "bool": (24000, 0.0), "int8": (24000, 0.0), "int16": (24000, 0.0),
                 "int32": (24000, 0.0), "int64": (24000, 0.0), "uint8": (24000, 0.0),
                 "uint16": (24000, 0.0), "uint32": (24000, 0.0), "uint64": (24000, 0.0)}),
}


# Comparison ufuncs whose `.reduce`/`.accumulate`/`.reduceat` no-loop shape
# is real numpy's own per-(ufunc, exact input dtype) cold/warm cache
# artifact (plain `TypeError` cold, rich `_UFuncNoLoopError`/`UFuncTypeError`
# warm -- see `registry.py`'s `_COMPARISON_REDUCE_EXC_EQUIV` comment, which
# this set wires in). Exactly the six comparison ufuncs reachable through
# that shape -- not derived from UFUNC_NAMES, hand-listed so this set can
# never silently grow to cover an unrelated ufunc.
_COMPARISON_REDUCE_UFUNCS = frozenset(
    {"equal", "not_equal", "greater", "greater_equal", "less", "less_equal"}
)


def _build_ufunc_specs() -> dict[str, registry.ItemSpec]:
    specs: dict[str, registry.ItemSpec] = {}
    for name in ufunc_introspect.UFUNC_NAMES:
        info = ufunc_introspect.describe(name)
        is_comparison_reduce = name in _COMPARISON_REDUCE_UFUNCS
        exc_equiv = registry._COMPARISON_REDUCE_EXC_EQUIV if is_comparison_reduce else None
        # `message_pair_ok` (see registry.py's `ItemSpec` docstring and the
        # long comment above `_comparison_reduce_message_pair_ok`): the
        # coupled cold/warm class+message fallback, scoped to exactly these
        # six ufuncs for the same reason `exc_equiv` above is -- the shape
        # is only reachable through their `.reduce`/`.accumulate`/
        # `.reduceat` call forms.
        msg_pair_ok = (
            registry._comparison_reduce_message_pair_ok(name)
            if is_comparison_reduce
            else None
        )
        override = _ULP_OVERRIDES.get(name)
        if override is not None:
            ulp_tolerance, ulp_justification, ulp_sweep = override
            specs[name] = registry.ItemSpec(
                name=name,
                kind="ufunc",
                numpy_path=name,
                ionp_path=name,
                atol=0.0,
                rtol=0.0,
                ulp_tolerance=ulp_tolerance,
                ulp_justification=ulp_justification,
                ulp_sweep=ulp_sweep,
                multi_output=(info.nout > 1),
                exception_equivalences=exc_equiv,
                message_pair_ok=msg_pair_ok,
            )
        else:
            specs[name] = registry.ItemSpec(
                name=name,
                kind="ufunc",
                numpy_path=name,
                ionp_path=name,
                atol=0.0,
                rtol=0.0,
                multi_output=(info.nout > 1),
                exception_equivalences=exc_equiv,
                message_pair_ok=msg_pair_ok,
            )
    return specs


UFUNC_SPECS: dict[str, registry.ItemSpec] = _build_ufunc_specs()

assert len(UFUNC_SPECS) == 134, (
    f"expected exactly 134 ufuncs per PLAN-fanout.md / tools/numpy_surface.json, "
    f"got {len(UFUNC_SPECS)} -- the surface manifest or the introspection filter changed; "
    f"do not silently adopt a new number, investigate first"
)

# Merge into the real registry. A name collision here would mean a ufunc
# name also exists as a hand-written REGISTRY entry above -- that's a real
# conflict worth a loud failure, not a silent overwrite either direction.
_collisions = set(UFUNC_SPECS) & set(registry.REGISTRY)
if _collisions:
    raise AssertionError(
        f"ufunc_registry.py: {sorted(_collisions)} already present in registry.REGISTRY "
        f"-- refusing to silently overwrite a hand-written item"
    )
registry.REGISTRY.update(UFUNC_SPECS)


# ---------------------------------------------------------------------------
# `.reduceat` index-bounds regression corpus (2026-08-02).
#
# `_reduceat_indices()` in ufunc_cases.py (the generic per-method case
# generator every kind="ufunc" entry above goes through) only ever produces
# IN-BOUNDS indices (`[0]` / `[0, n//2]`) -- the out-of-bounds/negative/
# float-index axis of `.reduceat`'s argument was never exercised anywhere
# in the corpus. That hole was invisible to the differential gate: real
# numpy's `.reduceat` validates every index against `0 <= idx < n` BEFORE
# any dtype-loop resolution (verified against real numpy 2.5.1, cross-
# checked independently), raising `IndexError` with message `index {i}
# out-of-bounds in {op}.reduceat [0, {n})`; negative indices are NEVER
# valid here (no wraparound, unlike ordinary fancy indexing -- `[-1]` on a
# 4-element array is rejected, not read as the last element); float
# indices are silently TRUNCATED toward zero, not rejected (`[1.5]` reduces
# from index `1`). anionpy's prior behavior on an out-of-range index was an
# unchecked buffer index in `reduceat_generic` (`ionp-core/src/ufunc.rs`)
# that raised an uncatchable Rust panic (`PanicException`, not a
# `BaseException` subclass a `try/except IndexError` can see) instead of
# `IndexError` -- confirmed to affect EVERY binary reduce-family op
# reachable through `.reduceat` (not comparison-specific, not
# complex-specific), including `add`/`multiply`/`subtract`/`maximum`/
# `minimum`/`logical_and`/`bitwise_or`, all seven of which are declared
# `exact` in `anionpy.__ion_state__` and were passing that declaration on a
# corpus that never probed this axis. Fixed by a shared
# `validate_reduceat_indices` bounds check (ufunc.rs) run before any
# dtype-loop error path, plus fixing `extract_index_vec` (ionp-py/src/
# lib.rs) to truncate float indices via numpy's own `asarray(...)
# .astype("int64")` instead of rejecting them with the wrong exception
# class entirely.
#
# Two `.reduceat`-bearing ops are probed here directly (bypassing the
# generic corpus, which has no notion of "give me index 9 on a length-4
# array"): `add` (a declared-exact, non-comparison exemplar -- the seven
# panic-affected declarations all share this one code path, so one
# exemplar covers the shared fix) and `equal` (a comparison ufunc, to
# prove the bounds check fires BEFORE the no-loop `TypeError`/
# `UFuncTypeError` machinery -- including on a COMPLEX empty array, which
# is exactly why real numpy's complex-dtype empty-`.reduceat` raises
# `IndexError` rather than the dtype-mismatch error documented elsewhere
# in this file for `compare_reduce_error`, and was the finding that
# surfaced this whole gap while measuring HOLE 2).
# ---------------------------------------------------------------------------

def _reduceat_bounds_cases_add():
    base = np.array([1, 2, 3, 4], dtype=np.int64)
    empty = np.array([], dtype=np.int64)
    return [
        ("neg_1", (base, [-1]), {}),
        ("neg_4_boundary", (base, [-4]), {}),
        ("neg_5", (base, [-5]), {}),
        ("zero_then_neg_1", (base, [0, -1]), {}),
        ("in_bounds_last_index", (base, [3]), {}),
        ("out_of_bounds_eq_len", (base, [4]), {}),
        ("zero_then_out_of_bounds", (base, [0, 9]), {}),
        ("empty_indices_list", (base, []), {}),
        ("float_index_truncates_in_bounds", (base, [1.5]), {}),
        ("empty_array_index_0", (empty, [0]), {}),
        # Truncate-BEFORE-bounds-check ordering, added after review caught
        # extract_index_vec's numpy round-trip (a vacuous delegation) and
        # measured the sharper ground truth: numpy truncates toward zero
        # FIRST, then bounds-checks the TRUNCATED value -- and the error
        # message names the truncated index, not the original float.
        # `-0.5` is the case that matters most: a negative float that is
        # LEGAL because it truncates to `0`. An implementation that
        # bounds-checks the raw float before truncating (rejecting any
        # negative value outright) would wrongly reject this and every
        # current case above still passes -- only this case is sensitive
        # to that specific ordering bug.
        ("neg_float_truncates_to_valid_zero", (base, [-0.5]), {}),
        ("neg_float_truncates_then_out_of_bounds", (base, [-1.5]), {}),
        ("pos_float_near_boundary_truncates_down", (base, [3.99999]), {}),
        # Non-finite / out-of-`long`-range float indices (2026-08-02
        # follow-up). Measured live against numpy 2.5.1, fresh-process
        # scripts, `except BaseException`: a `nan` index raises `ValueError:
        # cannot convert float NaN to integer` -- anionpy's prior behavior was
        # to silently compute a WRONG ANSWER (Rust's `f64 as isize` cast
        # saturates NaN to `0`, not an error). `inf`/`-inf` raise
        # `OverflowError: cannot convert float infinity to integer` --
        # anionpy's prior behavior was `IndexError` naming the saturated
        # `isize::MAX`/`isize::MIN`, the wrong exception CLASS (and a
        # fabricated in-range-looking index) for a value that was never a
        # legal index at all. A finite float outside representable `long`
        # range (`1e30` / `-1e30`, and a plain Python int like `2**63` that
        # overflows `isize` extraction and falls through to the same `f64`
        # coercion path) raises a THIRD, distinct message: `OverflowError:
        # Python int too large to convert to C long` -- distinguishing this
        # from the inf/-inf `OverflowError` above is the point: numpy uses
        # two different messages for the same exception class depending on
        # whether the offending value is infinite or merely too large, and
        # a fix that collapses them to one text would still fail this
        # corpus. The boundary itself was also measured directly: the
        # largest `f64` strictly below `2**63` (`9223372036854774784.0`,
        # every double at that magnitude has no fractional part and is
        # already `<= isize::MAX`) truncates and bounds-checks normally
        # (`IndexError` naming that exact value); `2**63` itself and
        # anything at or above it always overflows. On the negative side
        # `-(2**63)` exactly (`isize::MIN`) is the last value that is still
        # legal -- it is exactly representable in `f64`, unlike
        # `2**63 - 1` on the positive side, which is why the two boundaries
        # are not symmetric expressions here.
        ("nan_index", (base, [float("nan")]), {}),
        ("nan_index_among_valid", (base, [1, float("nan")]), {}),
        ("pos_inf_index", (base, [float("inf")]), {}),
        ("neg_inf_index", (base, [float("-inf")]), {}),
        ("pos_1e30_overflows_long", (base, [1e30]), {}),
        ("neg_1e30_overflows_long", (base, [-1e30]), {}),
        ("pos_int_2_pow_63_overflows_long", (base, [2**63]), {}),
        ("pos_int_2_pow_63_minus_1_in_bounds_but_oob_idx", (base, [2**63 - 1]), {}),
        ("huge_python_int_2_pow_70_overflows_long", (base, [2**70]), {}),
        ("neg_2_pow_63_exact_float_still_valid", (base, [-9223372036854775808.0]), {}),
        ("pos_2_pow_63_exact_float_overflows_long", (base, [9223372036854775808.0]), {}),
        ("largest_float_below_2_pow_63_out_of_bounds", (base, [9223372036854774784.0]), {}),
    ]


def _reduceat_bounds_cases_equal():
    nonbool = np.array([1, 2, 3, 4], dtype=np.int64)
    boolean = np.array([True, False, True, True], dtype=np.bool_)
    empty_complex = np.array([], dtype=np.complex128)
    return [
        ("nonbool_out_of_bounds", (nonbool, [9]), {}),
        ("nonbool_negative", (nonbool, [-1]), {}),
        ("bool_in_bounds", (boolean, [1, 3]), {}),
        ("bool_out_of_bounds", (boolean, [9]), {}),
        ("complex_empty_index_0", (empty_complex, [0]), {}),
    ]


_REDUCEAT_BOUNDS_SPECS: dict[str, registry.ItemSpec] = {
    "reduceat_bounds.add": registry.ItemSpec(
        name="reduceat_bounds.add",
        kind="custom",
        numpy_adapter=lambda arr, idx: np.add.reduceat(arr, idx),
        ionp_adapter=lambda arr, idx: anionpy.add.reduceat(arr, idx),
        custom_cases=_reduceat_bounds_cases_add,
    ),
    "reduceat_bounds.equal": registry.ItemSpec(
        name="reduceat_bounds.equal",
        kind="custom",
        numpy_adapter=lambda arr, idx: np.equal.reduceat(arr, idx),
        ionp_adapter=lambda arr, idx: anionpy.equal.reduceat(arr, idx),
        custom_cases=_reduceat_bounds_cases_equal,
    ),
}

_bounds_collisions = set(_REDUCEAT_BOUNDS_SPECS) & set(registry.REGISTRY)
if _bounds_collisions:
    raise AssertionError(
        f"ufunc_registry.py: {sorted(_bounds_collisions)} already present in "
        f"registry.REGISTRY -- refusing to silently overwrite"
    )
registry.REGISTRY.update(_REDUCEAT_BOUNDS_SPECS)
