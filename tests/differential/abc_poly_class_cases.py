"""anionpy.polynomial.{Legendre,Laguerre,Hermite,HermiteE,Chebyshev}
(the five new `ABCPolyBase` subclasses) differential registry entries.

NEW FILE (permitted, same pattern as polynomial_cases.py/legendre_cases.py/
linalg_cases.py): builds one `dict[str, ItemSpec]` per class via
`_ClassBinding.build()`, all merged into a single `ABC_POLY_CLASS_SPECS`
dict at the bottom, itself merged into `registry.REGISTRY` in registry.py
(collision-checked, same pattern as every other `*_cases.py` merge there).

Deliberately GENERIC/parametrized rather than five near-duplicate files:
the same "assembly, not new arithmetic" case-generator/adapter shape as
`polynomial_cases.py`'s `Polynomial` section is reused verbatim across all
five classes via `_ClassBinding`, since coefficient corpora are basis-
agnostic (a coefficient array is just floats/complex regardless of which
basis it is interpreted in) -- only the per-class measured epsilon
constants (`fit`/`roots`, and for Legendre only, `mul`/`rmul`/`pow`/
`fromroots` on complex128) and the declare/exclude decision differ.

ITEM SCOPE (see this task's final report for the full per-class,
per-item ledger):
  - `convert`/`cast` are NOT declared for ANY of the five classes here.
    Both route through `_polybase.py::_compose_affine`, which performs a
    naive Horner-loop substitution using the class's own `_mul` -- valid
    ONLY for the power-series basis. For every orthogonal basis here it
    computes the mathematically WRONG value, not merely a non-bit-exact
    one (verified directly: `Legendre([1.,2.,3.]).convert(kind=Legendre)`,
    an identity conversion, returns `[2. 2. 2.]` instead of the required
    `[1. 2. 3.]`; real numpy returns `[1. 2. 3.]` unchanged). This is a bug
    in the shared helper, out of scope to fix here (see legendre.py's
    module docstring, the first place this was found and documented).
  - `__mul__`/`__rmul__`/`__pow__`/`fromroots` are NOT declared for
    Chebyshev: `chebmul` (the underlying kernel) routes through
    `np.convolve`'s unreproduced summation order (z-series multiply),
    already known non-bit-exact at the module-function level -- see
    chebyshev_cases.py / `anionpy/_state/polynomial.py`'s REVOKED
    `chebmul`/`chebfromroots` entries.
  - `__mul__`/`__rmul__`/`__pow__`/`fromroots`: `poly.rs`/`legendre.rs`
    never routed complex arithmetic through `complex_mul_fma`/`complex_div`
    (see `ufunc.rs:2279`, `leg_mul`/`leg_div` in `legendre.rs`) -- every
    epsilon tolerance below used to be papering over that missing routing,
    not a genuine numpy-vs-anionpy accumulation-order difference. Fixed
    2026-08-07 (Monday), routing every `T*T`/`T/T` site in both files
    through the FMA-based primitives (`chebyshev.rs` as exemplar).
    Re-measured post-fix on a 25,000-sample seeded sweep
    (seed=0x5EEDC1A55, /private/tmp/sweep_class.py):
    `Laguerre`/`Hermite`/`HermiteE` `__mul__`/`__rmul__`/`__pow__`/
    `fromroots` and `Legendre.__rmul__` are all 0/25000 (declared, no
    tolerance). `Legendre.__mul__`/`__pow__`/`fromroots` on complex128
    are UNCHANGED by the fix -- still not bit-exact (22244/25000,
    9006/25000, 12242/25000) -- so per this task's hard rule (fix to
    bit-exact or REVOKE, never re-tolerance) those three are REVOKED, not
    epsilon-toleranced. See `anionpy/_state/polynomial.py`'s matching
    REVOKED entries and KNOWN-DIFFERENCES.md. `declare_mul` stays True for
    Legendre so the corpus still exercises and visibly fails the revoked
    three (their complex128 cases are expected-red, same convention as
    `chebfromroots`'s `_DIVERGES` cases) while `__rmul__` stays declared.
  - `__call__` was epsilon-toleranced on complex128 for Legendre and
    Laguerre in an earlier session (`LEGENDRE_CALL_EPS`/`LAGUERRE_CALL_EPS
    = 1e-13`) against a genuine finding: real-coefficient evaluation at
    complex points was majority non-bit-exact for `legval` (12763/20000)
    and `lagval` (1356/20000). That was the SAME missing-FMA-routing bug
    (`lagval`'s real-coefficient/complex-x path additionally needed a
    dedicated hybrid-typed kernel, `lag_eval_real_coef_complex_x`, to
    match numpy's "real until touched by a complex operand" scalar typing
    inside the Clenshaw recursion -- see that function's doc comment in
    `ionp-core/src/laguerre.rs`). Both items are now bit-exact (0/25000
    each on the same sweep) and the tolerance is REMOVED, not kept.
    Hermite/HermiteE/Chebyshev's own `*val` kernels were already clean
    (0/3000) and needed no change. `polynomial.polynomial.polyval` and
    `Polynomial.__call__` show the identical divergence pattern
    (1934/3000, 1937/3000 on the earlier sweep shape) but are out of this
    task's declared scope (the 5 new classes) and were left untouched;
    flagged here and in this task's final report for whoever owns that
    file next.
  - `roots`/`fit` are epsilon-toleranced for all five classes (LAPACK-
    bound eigensolver/lstsq call paths), reusing each class's own
    already-measured module-function-level epsilon constants
    (`LEGROOTS_COMPLEX_EPS`/`LEGFIT_EPS` etc. from each `*_cases.py`
    sibling file) rather than a fresh sweep, since `Class.roots()`/
    `Class.fit()` call the identical underlying kernel plus only exact
    (domain/window remap) arithmetic on top -- same reasoning
    `polynomial_cases.py`'s own `Polynomial.roots`/`Polynomial.fit`
    entries use for `Polynomial`.
"""
from __future__ import annotations

import numpy as np

import anionpy as _anionpy

from registry import ItemSpec
from legendre_cases import (
    _canonical_root_order,
    LEGFIT_EPS, LEGROOTS_COMPLEX_EPS,
)
from laguerre_cases import LAGFIT_EPS, LAGROOTS_COMPLEX_EPS
from hermite_cases import HERMFIT_EPS, HERMROOTS_REAL_EPS, HERMROOTS_COMPLEX_EPS
from hermite_e_cases import HERMEFIT_EPS, HERMEROOTS_REAL_EPS, HERMEROOTS_COMPLEX_EPS
from chebyshev_cases import CHEBFIT_EPS, CHEBROOTS_REAL_EPS, CHEBROOTS_COMPLEX_EPS

# ---------------------------------------------------------------------------
# shared, basis-agnostic coefficient/point fixtures (same values as
# polynomial_cases.py's own -- a coefficient array's bit pattern doesn't
# care which basis it will be interpreted in)
# ---------------------------------------------------------------------------

INT_C = [1, 2, 3]
FLOAT_C = [1.0, 2.0, 3.0]
TRAILING_ZERO_C = [1.0, 2.0, 3.0, 0.0, 0.0]
COMPLEX_C = [1 + 1j, 2 - 1j, 0.5j]
SINGLE_C = [5.0]
C1 = [2.0, -1.0, 3.0]
C2 = [1.0, 0.5]
C1_SHORT = [2.0]
XS = [0.1, 0.5, 1.5, -0.3]
X_SCALAR = 0.4
X_COMPLEX = [0.1 + 0.2j, -0.3j]
CUSTOM_DOMAIN = [-3.0, 3.0]
CUSTOM_WINDOW = [-1.0, 1.0]

# LEGENDRE_CLASS_MUL_EPS, LEGENDRE_CALL_EPS and LAGUERRE_CALL_EPS (all
# 1e-13) used to live here. `ionp-core/src/poly.rs` and `legendre.rs` had
# never been routed through `complex_mul_fma`/`complex_div` (see
# `ufunc.rs:2279`'s doc comment and `leg_mul`/`leg_div` in `legendre.rs`) --
# every one of the three tolerances above was papering over that, not a
# genuine numpy-vs-anionpy accumulation-order difference.
#
# Post-fix (2026-08-07, Monday), re-measured on a fresh 25,000-sample seeded
# sweep (seed=0x5EEDC1A55, /private/tmp/sweep_class.py):
#   Legendre.__call__  (real-coef, complex128-x)   0/25000  -- now bit-exact
#   Laguerre.__call__  (real-coef, complex128-x)   0/25000  -- now bit-exact
#   Legendre.__rmul__  (complex128)                0/25000  -- now bit-exact
#   Legendre.__mul__   (complex128)             22244/25000  -- UNCHANGED
#   Legendre.__pow__   (complex128)              9006/25000  -- UNCHANGED
#   Legendre.fromroots (complex128)             12242/25000  -- UNCHANGED
#
# So `LEGENDRE_CALL_EPS`/`LAGUERRE_CALL_EPS` are gone outright (both
# `__call__` items are declared bit-exact below, no tolerance). Per this
# task's hard rule ("remove tolerance IFF the fix makes it bit-exact,
# otherwise REVOKE -- never re-add tolerance"), `LEGENDRE_CLASS_MUL_EPS`
# is also gone: `Legendre.__mul__`/`__pow__`/`fromroots` on complex128
# are NOT bit-exact after the fix and are REVOKED below (`declare_mul`
# stays True so the corpus still exercises and visibly fails them --
# see `anionpy/_state/polynomial.py`'s matching REVOKED entries and
# KNOWN-DIFFERENCES.md). `Legendre.__rmul__` is unaffected (it reduces to
# a scalar `_mul`, which is bit-exact) and remains declared.
#
# Mechanism for the surviving `legmul` complex128 divergence is NOT
# established -- this looks like the same class of unidentified
# accumulation-order issue as `polymul`/`chebmul` (see those REVOKED
# entries), but that is an untested hypothesis, not a finding, and
# revocation does not depend on identifying it.


def _std(pairs):
    return list(pairs)


def _state(p):
    """Plain-Python, exactly-`==`-comparable snapshot of an ABCPolyBase
    instance: (coef list, domain list, window list, symbol str). Same
    helper as polynomial_cases.py's `_state`."""
    coef = p.coef
    coef = coef.tolist() if hasattr(coef, "tolist") else list(coef)
    domain = p.domain
    domain = domain.tolist() if hasattr(domain, "tolist") else list(domain)
    window = p.window
    window = window.tolist() if hasattr(window, "tolist") else list(window)
    return (coef, domain, window, p.symbol)


class _ClassBinding:
    """Builds the full `dict[str, ItemSpec]` for one `ABCPolyBase`
    subclass, given the (numpy class, anionpy class) pair, this class's
    own already-measured `fit`/`roots` epsilon constants, and whether
    `__mul__`/`__rmul__`/`__pow__`/`fromroots` are declared for it."""

    def __init__(self, prefix, npcls, ioncls, *, fit_eps,
                 roots_real_eps, roots_complex_eps,
                 declare_mul=True, mul_complex_eps=None,
                 call_complex_eps=None):
        self.prefix = prefix
        self.np = npcls
        self.ion = ioncls
        self.fit_eps = fit_eps
        self.roots_real_eps = roots_real_eps
        self.roots_complex_eps = roots_complex_eps
        self.declare_mul = declare_mul
        self.mul_complex_eps = mul_complex_eps
        self.call_complex_eps = call_complex_eps

    def _n(self, item):
        return f"polynomial.{self.prefix}.{item}"

    def build(self):
        np_c, ion_c = self.np, self.ion
        specs = {}

        # -- __init__ --
        def init_cases():
            return _std([
                ("basic_float", (FLOAT_C,), {}),
                ("int_promote", (INT_C,), {}),
                ("complex", (COMPLEX_C,), {}),
                ("trailing_zero_not_trimmed", (TRAILING_ZERO_C,), {}),
                ("single", (SINGLE_C,), {}),
                ("custom_domain_window", (FLOAT_C,), {"domain": CUSTOM_DOMAIN, "window": CUSTOM_WINDOW}),
                ("custom_symbol", (FLOAT_C,), {"symbol": "z"}),
            ])

        specs[self._n("__init__")] = ItemSpec(
            name=self._n("__init__"), kind="custom", custom_cases=init_cases,
            numpy_adapter=lambda coef, **kw: _state(np_c(coef, **kw)),
            ionp_adapter=lambda coef, **kw: _state(ion_c(coef, **kw)),
            scalar_like=True,
        )

        # -- domain / window / basis_name / symbol / maxpower --
        specs[self._n("domain")] = ItemSpec(
            name=self._n("domain"), kind="custom",
            custom_cases=lambda: _std([
                ("default", (FLOAT_C,), {}),
                ("custom", (FLOAT_C,), {"domain": CUSTOM_DOMAIN}),
            ]),
            numpy_adapter=lambda coef, **kw: np_c(coef, **kw).domain,
            ionp_adapter=lambda coef, **kw: ion_c(coef, **kw).domain,
            atol=0.0, rtol=0.0,
        )
        specs[self._n("window")] = ItemSpec(
            name=self._n("window"), kind="custom",
            custom_cases=lambda: _std([
                ("default", (FLOAT_C,), {}),
                ("custom", (FLOAT_C,), {"window": CUSTOM_WINDOW}),
            ]),
            numpy_adapter=lambda coef, **kw: np_c(coef, **kw).window,
            ionp_adapter=lambda coef, **kw: ion_c(coef, **kw).window,
            atol=0.0, rtol=0.0,
        )
        specs[self._n("basis_name")] = ItemSpec(
            name=self._n("basis_name"), kind="custom",
            custom_cases=lambda: _std([("value", (FLOAT_C,), {})]),
            numpy_adapter=lambda coef: np_c(coef).basis_name,
            ionp_adapter=lambda coef: ion_c(coef).basis_name,
            scalar_like=True,
        )
        specs[self._n("symbol")] = ItemSpec(
            name=self._n("symbol"), kind="custom",
            custom_cases=lambda: _std([
                ("default", (FLOAT_C,), {}),
                ("custom", (FLOAT_C,), {"symbol": "z"}),
            ]),
            numpy_adapter=lambda coef, **kw: np_c(coef, **kw).symbol,
            ionp_adapter=lambda coef, **kw: ion_c(coef, **kw).symbol,
            scalar_like=True,
        )
        specs[self._n("maxpower")] = ItemSpec(
            name=self._n("maxpower"), kind="custom",
            custom_cases=lambda: _std([("value", (), {})]),
            numpy_adapter=lambda: np_c.maxpower,
            ionp_adapter=lambda: ion_c.maxpower,
            scalar_like=True,
        )

        # -- __call__ --
        def call_cases():
            cases = []
            for c_label, c in [("float_c", FLOAT_C), ("int_c", INT_C), ("complex_c", COMPLEX_C)]:
                cases.append((f"scalar__{c_label}", (c, X_SCALAR), {}))
                cases.append((f"list__{c_label}", (c, XS), {}))
                cases.append((f"complex_x__{c_label}", (c, X_COMPLEX), {}))
            cases.append(("custom_domain", (FLOAT_C, XS), {"domain": CUSTOM_DOMAIN}))
            return cases

        call_kw = dict(atol=0.0, rtol=0.0)
        if self.call_complex_eps is not None:
            call_kw["epsilon_tolerance"] = {"complex128": ("rel", self.call_complex_eps)}
            call_kw["epsilon_tolerance_justification"] = (
                f"{self.prefix.capitalize()}.__call__ = the class's own "
                f"_val (module-level {self.prefix}val) evaluated via "
                "Clenshaw-style recurrence. Measured directly against "
                "module-level {0}val on a 20,000-sample out-of-corpus "
                "seeded sweep (real coefficients, complex128 x, degrees "
                "0-7): a majority of cases (not a rare edge case) differ "
                "at the last 1-2 ULPs of the imaginary/real accumulator "
                "-- pure summation-order rounding, max relative error "
                "~7.4e-15 (~4.8e-15 for lagval), NOT a correctness bug. "
                "Real-x and complex-coefficient paths remain bit-exact "
                "(0 mismatches on the same sweep); only the real-"
                "coefficient + complex-x combination is affected. This "
                "was NOT anticipated by the task brief and is a pre-"
                "existing gap in the already-declared module-level "
                "{0}val (and polynomial.polynomial.polyval, and "
                "Polynomial.__call__ -- both declared 'exact' in an "
                "earlier session, out of this task's ownership) -- their "
                "differential corpora evidently never exercised a "
                "real-coefficient/complex-x combination at scale. See "
                "/private/tmp for the sweep scripts and this task's final "
                "report.".format(self.prefix)
            )
            call_kw["epsilon_sweep"] = {"complex128": (20000, self.call_complex_eps)}
        specs[self._n("__call__")] = ItemSpec(
            name=self._n("__call__"), kind="custom", custom_cases=call_cases,
            numpy_adapter=lambda coef, x, **kw: np_c(coef, **kw)(x),
            ionp_adapter=lambda coef, x, **kw: ion_c(coef, **kw)(x),
            **call_kw,
        )

        # -- __iter__ / __len__ --
        specs[self._n("__iter__")] = ItemSpec(
            name=self._n("__iter__"), kind="custom",
            custom_cases=lambda: _std([
                ("float_c", (FLOAT_C,), {}), ("complex_c", (COMPLEX_C,), {}),
                ("single", (SINGLE_C,), {}),
            ]),
            numpy_adapter=lambda coef: np.array(list(iter(np_c(coef)))),
            ionp_adapter=lambda coef: _anionpy.array(list(iter(ion_c(coef)))),
            atol=0.0, rtol=0.0,
        )
        specs[self._n("__len__")] = ItemSpec(
            name=self._n("__len__"), kind="custom",
            custom_cases=lambda: _std([
                ("float_c", (FLOAT_C,), {}), ("trailing_zero", (TRAILING_ZERO_C,), {}),
                ("single", (SINGLE_C,), {}),
            ]),
            numpy_adapter=lambda coef: len(np_c(coef)),
            ionp_adapter=lambda coef: len(ion_c(coef)),
            scalar_like=True,
        )

        # -- __hash__ (unhashable -- exercised for the raised exception) --
        specs[self._n("__hash__")] = ItemSpec(
            name=self._n("__hash__"), kind="custom",
            custom_cases=lambda: _std([("value", (FLOAT_C,), {})]),
            numpy_adapter=lambda coef: hash(np_c(coef)),
            ionp_adapter=lambda coef: hash(ion_c(coef)),
            scalar_like=True,
        )

        # -- __array_ufunc__ (blocked -- exercised via TypeError) --
        specs[self._n("__array_ufunc__")] = ItemSpec(
            name=self._n("__array_ufunc__"), kind="custom",
            custom_cases=lambda: _std([("value", (FLOAT_C, C2), {})]),
            numpy_adapter=lambda c1, c2: np.add(np_c(c1), np_c(c2)),
            ionp_adapter=lambda c1, c2: np.add(ion_c(c1), ion_c(c2)),
            scalar_like=True,
        )

        # -- __eq__ / __ne__ --
        def eq_cases():
            return _std([
                ("equal", (FLOAT_C, FLOAT_C), {}),
                ("different_coef", (FLOAT_C, C2), {}),
                ("different_len", (FLOAT_C, C1_SHORT), {}),
                ("different_domain", (FLOAT_C, FLOAT_C), {"domain2": CUSTOM_DOMAIN}),
                ("different_symbol", (FLOAT_C, FLOAT_C), {"symbol2": "z"}),
                ("not_a_poly", (FLOAT_C, None), {}),
            ])

        def _np_eq(c1, c2, domain2=None, symbol2="x"):
            p = np_c(c1)
            q = c2 if c2 is None else np_c(c2, domain=domain2, symbol=symbol2)
            return p == q

        def _ionp_eq(c1, c2, domain2=None, symbol2="x"):
            p = ion_c(c1)
            q = c2 if c2 is None else ion_c(c2, domain=domain2, symbol=symbol2)
            return p == q

        def _np_ne(c1, c2, domain2=None, symbol2="x"):
            p = np_c(c1)
            q = c2 if c2 is None else np_c(c2, domain=domain2, symbol=symbol2)
            return p != q

        def _ionp_ne(c1, c2, domain2=None, symbol2="x"):
            p = ion_c(c1)
            q = c2 if c2 is None else ion_c(c2, domain=domain2, symbol=symbol2)
            return p != q

        specs[self._n("__eq__")] = ItemSpec(
            name=self._n("__eq__"), kind="custom", custom_cases=eq_cases,
            numpy_adapter=_np_eq, ionp_adapter=_ionp_eq, scalar_like=True,
        )
        specs[self._n("__ne__")] = ItemSpec(
            name=self._n("__ne__"), kind="custom", custom_cases=eq_cases,
            numpy_adapter=_np_ne, ionp_adapter=_ionp_ne, scalar_like=True,
        )

        # -- __add__/__sub__/__floordiv__/__mod__/__divmod__ (bit-exact) --
        def binop_cases():
            return _std([
                ("same_len", (C1, C2), {}),
                ("c1_shorter", (C1_SHORT, C1), {}),
                ("int_and_float", (INT_C, FLOAT_C), {}),
                ("complex_and_real", (COMPLEX_C, FLOAT_C), {}),
                ("scalar_other", (C1, 2.0), {}),
            ])

        def mk_binop(name, ionp):
            ctor = ion_c if ionp else np_c

            def adapter(c1, c2):
                p = ctor(c1)
                other = c2 if isinstance(c2, (int, float, complex)) else ctor(c2)
                result = getattr(p, name)(other)
                if isinstance(result, tuple):
                    return tuple(_state(r) for r in result)
                return _state(result)
            return adapter

        for op in ("__add__", "__sub__", "__floordiv__", "__mod__", "__divmod__"):
            specs[self._n(op)] = ItemSpec(
                name=self._n(op), kind="custom", custom_cases=binop_cases,
                numpy_adapter=mk_binop(op, ionp=False),
                ionp_adapter=mk_binop(op, ionp=True),
                scalar_like=True,
            )

        # -- __truediv__ (scalar-rhs only, same reasoning as
        #    polynomial_cases.py's poly_truediv_cases) --
        def truediv_cases():
            return _std([
                ("float_scalar", (C1, 2.0), {}),
                ("int_scalar", (C1_SHORT, 3), {}),
                ("complex_scalar", (FLOAT_C, 1.0 + 2.0j), {}),
            ])

        specs[self._n("__truediv__")] = ItemSpec(
            name=self._n("__truediv__"), kind="custom", custom_cases=truediv_cases,
            numpy_adapter=lambda c1, c2: (np_c(c1) / c2).coef,
            ionp_adapter=lambda c1, c2: (ion_c(c1) / c2).coef,
            atol=0.0, rtol=0.0,
        )

        # -- __mul__/__rmul__/__pow__/fromroots: DECLARED (per-class) --
        if self.declare_mul:
            def np_mul_coef(c1, c2):
                p = np_c(c1)
                other = c2 if isinstance(c2, (int, float, complex)) else np_c(c2)
                return p.__mul__(other).coef

            def ionp_mul_coef(c1, c2):
                p = ion_c(c1)
                other = c2 if isinstance(c2, (int, float, complex)) else ion_c(c2)
                return p.__mul__(other).coef

            mul_kw = dict(atol=0.0, rtol=0.0)
            if self.mul_complex_eps is not None:
                mul_kw["epsilon_tolerance"] = {"complex128": ("rel", self.mul_complex_eps)}
                mul_kw["epsilon_tolerance_justification"] = (
                    f"{self.prefix}mul (Rust three-term-recurrence "
                    "reprojection kernel) measurably diverges from real "
                    "numpy at the last few ULPs for random complex128 "
                    "inputs (float64 remains bit-exact, 0/20000 on a "
                    "seeded sweep) -- see /private/tmp/poly_measure_sweep.py "
                    "and this task's final report. Bound is the max "
                    "relative coefficient error over a 20,000-sample "
                    "seeded complex sweep, independent of this item's own "
                    "corpus."
                )
                mul_kw["epsilon_sweep"] = {"complex128": (20000, self.mul_complex_eps)}
            specs[self._n("__mul__")] = ItemSpec(
                name=self._n("__mul__"), kind="custom", custom_cases=binop_cases,
                numpy_adapter=np_mul_coef, ionp_adapter=ionp_mul_coef, **mul_kw,
            )

        # -- __radd__/__rsub__/__rtruediv__/__rfloordiv__/__rmod__/
        # __rdivmod__: built and declared for ALL FIVE classes unconditionally
        # (not gated on declare_mul). Unlike __rmul__/__pow__/fromroots below,
        # none of these six route through this basis's `_mul` kernel --
        # ABCPolyBase.__radd__/__rsub__ call `self._add`/`self._sub`
        # directly, and __rtruediv__/__rfloordiv__/__rmod__/__rdivmod__ all
        # reduce to `self._div` (see _polybase.py). `_add`/`_sub`/`_div` have
        # no known divergence for any of the five bases (the only known
        # defect in this family -- unreproducible np.convolve/z-series
        # summation order, ticket #45/#48 -- is specific to `_mul`/`_pow`),
        # so bundling these six under `declare_mul` (as a prior session did)
        # was over-conservative for Chebyshev specifically: it left six
        # already-correct, already-implemented dunders (rebound straight
        # from ABCPolyBase onto Chebyshev.__dict__, see chebyshev.py's
        # rebind block) both untested and undeclared. Split out here,
        # verified out-of-corpus before declaring (see this ticket's
        # writeup, docs/TICKET-34-DUNDERS-2026-08-08.md).
        def rbinop_cases():
            return _std([
                ("float_c", (FLOAT_C,), {}), ("int_c", (INT_C,), {}),
                ("complex_c", (COMPLEX_C,), {}),
            ])

        def mk_rbinop(name, ionp):
            ctor = ion_c if ionp else np_c

            def adapter(c):
                p = ctor(c)
                result = getattr(p, name)(2.0)
                if isinstance(result, tuple):
                    return tuple(_state(r) for r in result)
                return _state(result)
            return adapter

        for op in ("__radd__", "__rsub__", "__rtruediv__", "__rfloordiv__",
                   "__rmod__", "__rdivmod__"):
            specs[self._n(op)] = ItemSpec(
                name=self._n(op), kind="custom", custom_cases=rbinop_cases,
                numpy_adapter=mk_rbinop(op, ionp=False),
                ionp_adapter=mk_rbinop(op, ionp=True),
                scalar_like=True,
            )

        if self.declare_mul:
            def np_rmul_coef(c):
                return (2.0 * np_c(c)).coef

            def ionp_rmul_coef(c):
                return (2.0 * ion_c(c)).coef

            rmul_kw = dict(atol=0.0, rtol=0.0)
            if self.mul_complex_eps is not None:
                rmul_kw["epsilon_tolerance"] = {"complex128": ("rel", self.mul_complex_eps)}
                rmul_kw["epsilon_tolerance_justification"] = (
                    f"Same {self.prefix}mul-kernel complex128 divergence as "
                    "__mul__ above -- __rmul__ = self._mul(_as_series1(other), "
                    "self.coef), the identical kernel."
                )
                rmul_kw["epsilon_sweep"] = {"complex128": (20000, self.mul_complex_eps)}
            specs[self._n("__rmul__")] = ItemSpec(
                name=self._n("__rmul__"), kind="custom", custom_cases=rbinop_cases,
                numpy_adapter=np_rmul_coef, ionp_adapter=ionp_rmul_coef, **rmul_kw,
            )

            def pow_cases():
                return _std([
                    ("square", (C1, 2), {}), ("cube", (FLOAT_C, 3), {}),
                    ("power_zero", (C1, 0), {}), ("power_one", (C1, 1), {}),
                ])

            pow_kw = dict(atol=0.0, rtol=0.0)
            if self.mul_complex_eps is not None:
                pow_kw["epsilon_tolerance"] = {"complex128": ("rel", self.mul_complex_eps)}
                pow_kw["epsilon_tolerance_justification"] = (
                    f"__pow__ = repeated {self.prefix}mul under the hood "
                    "(ABCPolyBase.__pow__ -> self._pow -> repeated _mul); "
                    "same complex128 divergence as __mul__ above."
                )
                pow_kw["epsilon_sweep"] = {"complex128": (20000, self.mul_complex_eps)}
            specs[self._n("__pow__")] = ItemSpec(
                name=self._n("__pow__"), kind="custom", custom_cases=pow_cases,
                numpy_adapter=lambda c, n: (np_c(c) ** n).coef,
                ionp_adapter=lambda c, n: (ion_c(c) ** n).coef, **pow_kw,
            )

        # -- __neg__ / __pos__ (always bit-exact: pure negation/copy) --
        specs[self._n("__neg__")] = ItemSpec(
            name=self._n("__neg__"), kind="custom",
            custom_cases=lambda: _std([("float_c", (FLOAT_C,), {}), ("complex_c", (COMPLEX_C,), {})]),
            numpy_adapter=lambda c: _state(-np_c(c)),
            ionp_adapter=lambda c: _state(-ion_c(c)),
            scalar_like=True,
        )
        specs[self._n("__pos__")] = ItemSpec(
            name=self._n("__pos__"), kind="custom",
            custom_cases=lambda: _std([("float_c", (FLOAT_C,), {}), ("complex_c", (COMPLEX_C,), {})]),
            numpy_adapter=lambda c: _state(+np_c(c)),
            ionp_adapter=lambda c: _state(+ion_c(c)),
            scalar_like=True,
        )

        # -- __getstate__ / __setstate__ --
        def state_dict(d):
            return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sorted(d.items())}

        specs[self._n("__getstate__")] = ItemSpec(
            name=self._n("__getstate__"), kind="custom",
            custom_cases=lambda: _std([
                ("basic", (FLOAT_C,), {}),
                ("custom_domain", (FLOAT_C,), {"domain": CUSTOM_DOMAIN, "symbol": "z"}),
            ]),
            numpy_adapter=lambda coef, **kw: state_dict(np_c(coef, **kw).__getstate__()),
            ionp_adapter=lambda coef, **kw: state_dict(ion_c(coef, **kw).__getstate__()),
            scalar_like=True,
        )

        def np_setstate(coef, **kw):
            src = np_c(coef, **kw)
            dst = np_c.__new__(np_c)
            dst.__setstate__(src.__getstate__())
            return _state(dst)

        def ionp_setstate(coef, **kw):
            src = ion_c(coef, **kw)
            dst = ion_c.__new__(ion_c)
            dst.__setstate__(src.__getstate__())
            return _state(dst)

        specs[self._n("__setstate__")] = ItemSpec(
            name=self._n("__setstate__"), kind="custom",
            custom_cases=lambda: _std([
                ("basic", (FLOAT_C,), {}),
                ("custom_domain", (FLOAT_C,), {"domain": CUSTOM_DOMAIN, "symbol": "z"}),
            ]),
            numpy_adapter=np_setstate, ionp_adapter=ionp_setstate, scalar_like=True,
        )

        # -- copy / degree / cutdeg / trim / truncate / mapparms --
        specs[self._n("copy")] = ItemSpec(
            name=self._n("copy"), kind="custom",
            custom_cases=lambda: _std([("basic", (FLOAT_C,), {"domain": CUSTOM_DOMAIN, "symbol": "z"})]),
            numpy_adapter=lambda coef, **kw: _state(np_c(coef, **kw).copy()),
            ionp_adapter=lambda coef, **kw: _state(ion_c(coef, **kw).copy()),
            scalar_like=True,
        )
        specs[self._n("degree")] = ItemSpec(
            name=self._n("degree"), kind="custom",
            custom_cases=lambda: _std([
                ("float_c", (FLOAT_C,), {}), ("trailing_zero", (TRAILING_ZERO_C,), {}),
                ("single", (SINGLE_C,), {}),
            ]),
            numpy_adapter=lambda coef: np_c(coef).degree(),
            ionp_adapter=lambda coef: ion_c(coef).degree(),
            scalar_like=True,
        )
        _CUT_C = [1.0, 2.0, 3.0, 4.0, 5.0]
        specs[self._n("cutdeg")] = ItemSpec(
            name=self._n("cutdeg"), kind="custom",
            custom_cases=lambda: _std([
                ("reduce", (_CUT_C, 2), {}), ("no_change", (_CUT_C, 10), {}),
                ("to_zero", (_CUT_C, 0), {}),
            ]),
            numpy_adapter=lambda coef, deg: _state(np_c(coef).cutdeg(deg)),
            ionp_adapter=lambda coef, deg: _state(ion_c(coef).cutdeg(deg)),
            scalar_like=True,
        )
        _TRIM_C = [1.0, 2.0, 0.0, 1e-10, 0.0]
        specs[self._n("trim")] = ItemSpec(
            name=self._n("trim"), kind="custom",
            custom_cases=lambda: _std([
                ("default_tol", (_TRIM_C,), {}), ("loose_tol", (_TRIM_C,), {"tol": 1e-5}),
                ("no_trim", (FLOAT_C,), {}),
            ]),
            numpy_adapter=lambda coef, **kw: _state(np_c(coef).trim(**kw)),
            ionp_adapter=lambda coef, **kw: _state(ion_c(coef).trim(**kw)),
            scalar_like=True,
        )
        specs[self._n("truncate")] = ItemSpec(
            name=self._n("truncate"), kind="custom",
            custom_cases=lambda: _std([
                ("reduce", (_CUT_C, 2), {}), ("no_change", (_CUT_C, 10), {}),
            ]),
            numpy_adapter=lambda coef, size: _state(np_c(coef).truncate(size)),
            ionp_adapter=lambda coef, size: _state(ion_c(coef).truncate(size)),
            scalar_like=True,
        )
        specs[self._n("mapparms")] = ItemSpec(
            name=self._n("mapparms"), kind="custom",
            custom_cases=lambda: _std([
                ("default", (FLOAT_C,), {}),
                ("custom", (FLOAT_C,), {"domain": CUSTOM_DOMAIN, "window": CUSTOM_WINDOW}),
            ]),
            numpy_adapter=lambda coef, **kw: np_c(coef, **kw).mapparms(),
            ionp_adapter=lambda coef, **kw: ion_c(coef, **kw).mapparms(),
            atol=0.0, rtol=0.0,
        )

        # -- has_samecoef / has_samedomain / has_samewindow / has_sametype --
        def has_same_cases():
            return _std([
                ("same", (FLOAT_C, FLOAT_C), {}),
                ("different_coef", (FLOAT_C, C2), {}),
                ("different_domain", (FLOAT_C, FLOAT_C), {"domain2": CUSTOM_DOMAIN}),
                ("different_window", (FLOAT_C, FLOAT_C), {"window2": CUSTOM_WINDOW}),
            ])

        def mk_has_same(name, ionp):
            ctor = ion_c if ionp else np_c

            def adapter(c1, c2, domain2=None, window2=None):
                p = ctor(c1)
                kw = {}
                if domain2 is not None:
                    kw["domain"] = domain2
                if window2 is not None:
                    kw["window"] = window2
                q = ctor(c2, **kw)
                return getattr(p, name)(q)
            return adapter

        for hname in ("has_samecoef", "has_samedomain", "has_samewindow", "has_sametype"):
            specs[self._n(hname)] = ItemSpec(
                name=self._n(hname), kind="custom", custom_cases=has_same_cases,
                numpy_adapter=mk_has_same(hname, ionp=False),
                ionp_adapter=mk_has_same(hname, ionp=True),
                scalar_like=True,
            )

        # -- integ / deriv --
        specs[self._n("integ")] = ItemSpec(
            name=self._n("integ"), kind="custom",
            custom_cases=lambda: _std([
                ("default", (FLOAT_C,), {}), ("m2", (FLOAT_C,), {"m": 2}),
                ("with_k", (FLOAT_C,), {"k": [1.0]}),
                ("with_lbnd", (FLOAT_C,), {"lbnd": 1.0}),
            ]),
            numpy_adapter=lambda coef, **kw: _state(np_c(coef).integ(**kw)),
            ionp_adapter=lambda coef, **kw: _state(ion_c(coef).integ(**kw)),
            scalar_like=True,
        )
        specs[self._n("deriv")] = ItemSpec(
            name=self._n("deriv"), kind="custom",
            custom_cases=lambda: _std([
                ("default", (FLOAT_C,), {}), ("m2", (FLOAT_C,), {"m": 2}),
            ]),
            numpy_adapter=lambda coef, **kw: _state(np_c(coef).deriv(**kw)),
            ionp_adapter=lambda coef, **kw: _state(ion_c(coef).deriv(**kw)),
            scalar_like=True,
        )

        # -- roots (epsilon-toleranced, LAPACK-bound; sorted comparison) --
        def roots_cases():
            return _std([
                ("float_c", (FLOAT_C,), {}),
                ("five_real_roots", ([1.0, -2.0, 0.5, 3.0, -1.5, 2.5],), {}),
                ("complex_roots_input", (COMPLEX_C,), {}),
            ])

        def np_roots_sorted(coef):
            r = np_c(coef).roots()
            return _canonical_root_order(np.asarray(r))

        def ionp_roots_sorted(coef):
            r = ion_c(coef).roots()
            r_np = np.asarray(r.tolist() if hasattr(r, "tolist") else r)
            return _canonical_root_order(r_np)

        specs[self._n("roots")] = ItemSpec(
            name=self._n("roots"), kind="custom", custom_cases=roots_cases,
            numpy_adapter=np_roots_sorted, ionp_adapter=ionp_roots_sorted,
            atol=0.0, rtol=0.0,
            epsilon_tolerance={
                "float64": ("abs", self.roots_real_eps),
                "complex128": ("abs", self.roots_complex_eps),
            },
            epsilon_tolerance_justification=(
                f"{self.prefix.capitalize()}.roots() = the class's own "
                f"_roots (module-level {self.prefix}roots, already "
                "epsilon-toleranced there for the identical LAPACK-vs-"
                "independent-eigensolver reason) + an exact affine domain/"
                "window remap on top. Reuses that module-function-level "
                "measured bound rather than a fresh sweep -- the remap is "
                "`off + scl * roots` with `scl` derived from this item's "
                "own domain/window (magnitude a few units), which cannot "
                "materially widen an already-generously-measured bound. "
                "Sorted via _canonical_root_order (see legendre_cases.py) "
                "before comparison: roots() makes no ordering guarantee."
            ),
            epsilon_sweep={
                "float64": (20000, self.roots_real_eps),
                "complex128": (20000, self.roots_complex_eps),
            },
        )

        # -- fromroots (declared only alongside mul, same kernel family) --
        if self.declare_mul:
            def fromroots_cases():
                return _std([
                    ("real_roots", ([1.0, -1.0, 2.0],), {}),
                    ("single_root", ([3.0],), {}),
                    ("complex_roots", ([-1j, 0.0, 1j],), {}),
                ])

            fr_kw = dict(atol=0.0, rtol=0.0)
            if self.mul_complex_eps is not None:
                fr_kw["epsilon_tolerance"] = {"complex128": ("rel", self.mul_complex_eps)}
                fr_kw["epsilon_tolerance_justification"] = (
                    f"fromroots = repeated {self.prefix}mul of linear "
                    "factors; same complex128 divergence as __mul__ above."
                )
                fr_kw["epsilon_sweep"] = {"complex128": (20000, self.mul_complex_eps)}
            specs[self._n("fromroots")] = ItemSpec(
                name=self._n("fromroots"), kind="custom", custom_cases=fromroots_cases,
                numpy_adapter=lambda roots: _state(np_c.fromroots(roots)),
                ionp_adapter=lambda roots: _state(ion_c.fromroots(roots)), **fr_kw,
            )

        # -- identity / basis / linspace --
        specs[self._n("identity")] = ItemSpec(
            name=self._n("identity"), kind="custom",
            custom_cases=lambda: _std([
                ("default", (), {}), ("custom_domain", (), {"domain": CUSTOM_DOMAIN}),
            ]),
            numpy_adapter=lambda **kw: _state(np_c.identity(**kw)),
            ionp_adapter=lambda **kw: _state(ion_c.identity(**kw)),
            scalar_like=True,
        )
        specs[self._n("basis")] = ItemSpec(
            name=self._n("basis"), kind="custom",
            custom_cases=lambda: _std([
                ("deg0", (0,), {}), ("deg3", (3,), {}),
            ]),
            numpy_adapter=lambda deg: _state(np_c.basis(deg)),
            ionp_adapter=lambda deg: _state(ion_c.basis(deg)),
            scalar_like=True,
        )
        specs[self._n("linspace")] = ItemSpec(
            name=self._n("linspace"), kind="custom",
            custom_cases=lambda: _std([
                ("default", (FLOAT_C,), {}), ("n10", (FLOAT_C,), {"n": 10}),
            ]),
            numpy_adapter=lambda coef, **kw: np_c(coef).linspace(**kw),
            ionp_adapter=lambda coef, **kw: ion_c(coef).linspace(**kw),
            atol=0.0, rtol=0.0, multi_output=True,
        )

        # -- fit (epsilon-toleranced, LAPACK-bound lstsq) --
        def fit_cases():
            x = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.8, -0.8, 0.3]
            y = [1.0, 0.2, -0.5, 0.1, 1.5, 0.9, 0.4, -0.1]
            return _std([
                ("deg3", (x, y, 3), {}),
                ("deg1", (x, y, 1), {}),
            ])

        specs[self._n("fit")] = ItemSpec(
            name=self._n("fit"), kind="custom", custom_cases=fit_cases,
            numpy_adapter=lambda x, y, deg, **kw: np_c.fit(x, y, deg, **kw).coef,
            ionp_adapter=lambda x, y, deg, **kw: ion_c.fit(x, y, deg, **kw).coef,
            atol=0.0, rtol=0.0,
            epsilon_tolerance={"float64": ("rel", self.fit_eps)},
            epsilon_tolerance_justification=(
                f"{self.prefix.capitalize()}.fit(...).coef IS the module-"
                f"level {self.prefix}fit's own coefficient output (ABCPolyBase"
                ".fit: `cls._fit(xnew, y, deg, ...)` calls it directly) "
                "reused unchanged after an exact domain remap of x -- same "
                "already-measured bound as that module-function-level "
                "item, not a fresh sweep."
            ),
            epsilon_sweep={"float64": (20000, self.fit_eps)},
        )

        return specs


import anionpy.polynomial as _anion_poly  # noqa: E402

_BINDINGS = [
    _ClassBinding(
        "Legendre", np.polynomial.Legendre, _anion_poly.Legendre,
        fit_eps=LEGFIT_EPS, roots_real_eps=LEGROOTS_COMPLEX_EPS,
        roots_complex_eps=LEGROOTS_COMPLEX_EPS,
        declare_mul=True, mul_complex_eps=None,
        call_complex_eps=None,
    ),
    _ClassBinding(
        "Laguerre", np.polynomial.Laguerre, _anion_poly.Laguerre,
        fit_eps=LAGFIT_EPS, roots_real_eps=LAGROOTS_COMPLEX_EPS,
        roots_complex_eps=LAGROOTS_COMPLEX_EPS,
        declare_mul=True, mul_complex_eps=None,
        call_complex_eps=None,
    ),
    _ClassBinding(
        "Hermite", np.polynomial.Hermite, _anion_poly.Hermite,
        fit_eps=HERMFIT_EPS, roots_real_eps=HERMROOTS_REAL_EPS,
        roots_complex_eps=HERMROOTS_COMPLEX_EPS,
        declare_mul=True, mul_complex_eps=None,
    ),
    _ClassBinding(
        "HermiteE", np.polynomial.HermiteE, _anion_poly.HermiteE,
        fit_eps=HERMEFIT_EPS, roots_real_eps=HERMEROOTS_REAL_EPS,
        roots_complex_eps=HERMEROOTS_COMPLEX_EPS,
        declare_mul=True, mul_complex_eps=None,
    ),
    _ClassBinding(
        "Chebyshev", np.polynomial.Chebyshev, _anion_poly.Chebyshev,
        fit_eps=CHEBFIT_EPS, roots_real_eps=CHEBROOTS_REAL_EPS,
        roots_complex_eps=CHEBROOTS_COMPLEX_EPS,
        declare_mul=False, mul_complex_eps=None,
    ),
]


def _build_abc_poly_class_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}
    for binding in _BINDINGS:
        cls_specs = binding.build()
        collisions = set(cls_specs) & set(specs)
        if collisions:
            raise AssertionError(
                f"abc_poly_class_cases.py: {sorted(collisions)} declared "
                f"twice across class bindings -- refusing to silently "
                f"overwrite"
            )
        specs.update(cls_specs)
    return specs


ABC_POLY_CLASS_SPECS = _build_abc_poly_class_specs()
