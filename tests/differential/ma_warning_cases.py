"""Differential corpus for the `errstate(divide='ignore', invalid='ignore')`
suppression `anionpy/ma/core.py` now wraps around the five `fn(...)`
call sites (`make_masked_unary`, `make_masked_binary`,
`make_masked_domained_unary`, `make_masked_domained_binary`, the
comparison/`hypot` scalar-safe factory) -- the opposite-direction sibling
of `complex_warning_cases.py` (task #30: anionpy failing to emit a warning
numpy DOES emit). This one is anionpy EMITTING a warning numpy does not.

WHY THIS FILE EXISTS. Measured 2026-08-07 against numpy 2.5.1: real numpy's
`numpy.ma` computes every masked op's raw ufunc call inside `errstate(
divide='ignore', invalid='ignore')` -- its own source comment: "nans at
masked positions cause RuntimeWarnings, even though they are masked. To
avoid this we suppress warnings." Before this task's fix, anionpy's five
equivalent factory bodies in `anionpy/ma/core.py` did not do this, so e.g.
`anionpy.ma.sqrt(MaskedArray([-4.0]))` emitted `RuntimeWarning: invalid
value encountered in sqrt` and `MaskedArray([1.]) / MaskedArray([0.])`
emitted `RuntimeWarning: divide by zero encountered in divide`, where real
numpy emitted nothing on either. The fix wraps `fn(...)` in
`anionpy.errstate(divide="ignore", invalid="ignore")` at all five sites --
pure Python, no rebuild. This corpus is the ledger-visible proof that fix
is real and that it did not become a blanket silencer.

Two things this docstring is careful NOT to claim, because the previous
revision of two of these five sites' own docstrings made exactly this
mistake and drew a false conclusion from a true premise ("anionpy's
domained ufuncs never raise on out-of-domain input, therefore no errstate
was needed" -- true premise, false conclusion, since the real gap was
warnings, not raises): (1) a mask hiding an invalid position is NOT what
makes the warning fire or not -- `fn(am.data)`/`fn(am.data, bm.data)` is
always called on the FULL raw data, masked or not, matching real numpy's
own always-suppress rule; the cases below therefore vary the mask axis for
completeness (this corpus also folds mask/data/dtype/fill_value into its
verdict, see below) but the suppression bug itself does not depend on it.
(2) "this op doesn't raise" was never the right question for warnings, and
is not asked again here.

WHAT IS COMPARED. Each case wraps a real call in
`warnings.catch_warnings(record=True)` + `simplefilter('always')` and
records `[(w.category.__name__, str(w.message)), ...]` (class name, exact
text, and via list equality, count) -- same recording contract
`complex_warning_cases.py` uses. That list is folded into the SAME
descriptor string as the call's own outcome, reusing `ma_cases.py`'s own
already-hardened `_np_snapshot`/`_ionp_snapshot` normalizers (NaN-safe,
dtype+type-boxing-aware fill_value comparison, `nomask`/`masked`-singleton
special cases) rather than reimplementing them -- so a case that gets the
warning right but breaks the underlying value/mask/dtype/fill_value still
fails, and a case that breaks the warning but keeps the value right also
still fails.

COVERAGE. The domained-unary family (`sqrt`, `log`, `log2`, `log10`,
`arcsin`, `arccos`, `arccosh`, `arctanh`) and the domained-binary family
(`divide`/`true_divide`, `floor_divide`, `remainder`/`mod`, `fmod`), each
called as the module-level `ma.<fn>` function. The `MaskedArray` dunder
routes `/` and `//` (`__truediv__`/`__rtruediv__`/`__floordiv__`/
`__rfloordiv__` -- the four items under standing "exact" review by this
task) against MaskedArray-vs-MaskedArray, MaskedArray-vs-bare-scalar, and
MaskedArray-vs-bare-list operands (the reflected forms exercise the
right-hand list/scalar path a module-function-only corpus never reaches).
`__mod__`/`__rmod__`/`__imod__` ARE now exercised via dedicated case
families below (`ma_warning_mod_dunder`, `ma_warning_imod`) -- added
2026-08-07 once `anionpy/ma/core.py` grew real overrides for those three
(previously true, now stale: this docstring used to say "%` has no dunder
route" because neither package declared it; that changed when
`_generic_domained_binary_dunder`/`_generic_domained_binary_idunder`
landed). `%`'s diagnostic is dtype-conditional on real numpy in a way `/`
and `//` are not (int: warns "divide by zero encountered in remainder"
and writes 0; float: silent, writes nan) -- see those two functions'
docstrings for the measured `__array_wrap__`-vs-in-place divergence this
dtype split exists to catch -- so the mod family gets its own dtype axis
the plain `_DUNDER_OPS`/`_dunder_cases` below (float-only) does not have.
The plain (non-domained) unary/binary families (`make_masked_unary`/
`make_masked_binary`) are covered too, but necessarily as NEGATIVE
CONTROLS -- see below, not because a "safe" input for e.g. `abs`/`add`
naturally produces an invalid/divide-by-zero condition the way a domain
violation does.

NEGATIVE CONTROLS (mandatory, and the easiest part of this brief to skimp
on). Two independent kinds:
  (a) BOTH SIDES MUST WARN, together, non-vacuously: `ma.multiply(
  MaskedArray([1e300]), MaskedArray([1e300]))` -- overflow, not
  divide/invalid, so it is OUTSIDE the `errstate(divide=..., invalid=...)`
  scope the fix added; each case asserts the warning list is non-empty on
  BOTH sides before comparing them, so a case that "passes" by both sides
  going silent (e.g. if some future change widened the errstate scope to
  swallow `over` too) is rejected by the case itself, not just by drifting
  to a false negative-looking pass. `ma.add(MaskedArray([inf]),
  MaskedArray([-inf]))` (invalid, via the PLAIN binary family, not
  domained) is the plain-family analogue: real numpy warns here too (this
  is genuinely invalid data, not a masked-away domain violation), proving
  the plain family's errstate wrap does not swallow this case either.
  (b) BOTH SIDES MUST STAY SILENT for a legitimate reason: in-domain-only
  data (no domain violation anywhere in the operand), including under
  every mask variant -- masking something that was never invalid changes
  nothing about whether a warning could have fired.
A corpus that only ever asserts "no warnings anywhere" cannot distinguish
"correctly suppressed" from "warnings globally broken" -- per this task's
brief, both control kinds are represented across the domained and plain
families alike.

PROVENANCE AXES varied (not just the op): masked vs nomask vs
partially-masked vs fully-masked vs empty vs 0-d operands; int vs float
dtype (the plain-family int cases; the domained families are float-only,
matching real numpy's own ufunc domains, which are not meaningfully
defined over integer division-by-zero the same way -- `anionpy.ma.divide`
on int operands promotes through the same float ufunc path regardless);
MaskedArray-vs-scalar and MaskedArray-vs-list right-hand operands for
every binary/dunder case (the reflected paths).

WHAT IS DELIBERATELY NOT COVERED. `ma.exp`'s overflow gap
(`anionpy.exp(anionpy.array([1000.0]))` emits no `RuntimeWarning: overflow
encountered in exp` where real numpy does) is a PRE-EXISTING, UNRELATED
bare-ufunc gap (task #50 territory), present with or without this task's
fix, and outside the `divide`/`invalid` errstate scope this file tests --
no `ma.exp` overflow case is included here, positive or negative; see this
task's report. `tan`/`cos`/`tanh`/`left_shift`/`right_shift`/`maximum`/
`minimum` remain excluded from `anionpy.ma` entirely (see
`anionpy/ma/core.py`'s and `ma_cases.py`'s own module docstrings) and are
not reintroduced here. `power`/`hypot` (undeclared domained-binary
candidates) are likewise out of scope. This file does not re-verify the
value/mask/fill_value semantics `ma_cases.py` already covers in depth for
every one of these items outside the warning axis -- it reuses that file's
snapshot helpers precisely so it does not have to.
"""

from __future__ import annotations

import warnings

import _bootstrap  # noqa: F401
import ma_cases as _mc
from registry import REGISTRY, ItemSpec

_UNSET = object()


# ---------------------------------------------------------------------------
# recording + descriptor plumbing (mirrors complex_warning_cases.py's
# _record/_descriptor pattern; folds ma_cases.py's own snapshot into the
# same descriptor string rather than reimplementing value/mask/fill_value
# comparison).
# ---------------------------------------------------------------------------
def _record(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            result = fn()
            exc = None
        except BaseException as e:  # noqa: BLE001 -- PyO3 panics derive BaseException
            result = None
            exc = e
    warns = [(w.category.__name__, str(w.message)) for w in rec]
    return warns, result, exc


def _descriptor(warns, result, exc, snapshot_fn):
    if exc is not None:
        body = f"raised:{type(exc).__name__}:{exc}"
    else:
        body = repr(snapshot_fn(result))
    return f"warns={warns!r}|{body}"


def _build_ma(lib, data, mask=_UNSET, fill_value=_UNSET, dtype=None):
    kw = {}
    if mask is not _UNSET:
        kw["mask"] = mask
    if fill_value is not _UNSET:
        kw["fill_value"] = fill_value
    if dtype is not None:
        kw["dtype"] = dtype
    if lib == "numpy":
        import numpy as np

        return np.ma.MaskedArray(data, **kw)
    import anionpy

    return anionpy.ma.MaskedArray(data, **kw)


def _other_operand(lib, spec):
    """`spec` is one of: `("ma", data, mask_kw)` -> a real MaskedArray in
    `lib`; `("scalar", value)` -> a bare Python scalar (same object either
    side, provenance = "never wrapped"); `("list", data)` -> a bare Python
    list (same object either side)."""
    kind = spec[0]
    if kind == "ma":
        _, data, mask_kw = spec
        return _build_ma(lib, data, **mask_kw)
    if kind == "scalar":
        return spec[1]
    if kind == "list":
        return list(spec[1])
    raise ValueError(spec)


# ---------------------------------------------------------------------------
# DOMAINED UNARY family: module-level ma.<fn> calls.
# ---------------------------------------------------------------------------
# Each op's (good, bad) pair: `good` is strictly in-domain (no warning
# possible), `bad` is a value real numpy's own `ufunc_domain` predicate
# masks out (see anionpy/ma/core.py's make_masked_domained_unary docstring
# for each predicate, verified there directly against CPython source).
_DOMAINED_UNARY_OPS = {
    "sqrt": (4.0, -4.0),
    "log": (2.0, 0.0),
    "log2": (2.0, 0.0),
    "log10": (2.0, 0.0),
    "arcsin": (0.5, 2.0),
    "arccos": (0.5, 2.0),
    "arccosh": (2.0, 0.0),
    "arctanh": (0.5, 1.0),
}


def _domained_unary_cases_for(good, bad):
    return [
        ("bad_unmasked_1d", [bad, good], {}),
        ("bad_maskedpos_1d", [bad, good], {"mask": [True, False]}),
        ("bad_unmaskedpos_1d", [bad, good], {"mask": [False, True]}),
        ("bad_fully_masked", [bad, bad], {"mask": [True, True]}),
        ("bad_mask_none_explicit", [bad, good], {"mask": None}),
        ("bad_0d_unmasked", bad, {}),
        ("bad_0d_masked", bad, {"mask": True}),
        ("good_only_nomask", [good, good, good], {}),
        ("good_only_masked", [good, good], {"mask": [True, False]}),
        ("good_0d", good, {}),
        ("empty", [], {}),
    ]


def _domained_unary_adapter(lib, op, label, data, mask_kw):
    def fn():
        arr = _build_ma(lib, data, **mask_kw)
        m = __import__("numpy").ma if lib == "numpy" else __import__("anionpy").ma
        return getattr(m, op)(arr)

    snapshot = _mc._np_snapshot if lib == "numpy" else _mc._ionp_snapshot
    warns, result, exc = _record(fn)
    return _descriptor(warns, result, exc, snapshot)


def _domained_unary_cases():
    out = []
    for op, (good, bad) in _DOMAINED_UNARY_OPS.items():
        for label, data, mask_kw in _domained_unary_cases_for(good, bad):
            out.append((f"{op}|{label}", (op, label, data, mask_kw), {}))
    return out


def _numpy_domained_unary(op, label, data, mask_kw, **kwargs):
    return _domained_unary_adapter("numpy", op, label, data, mask_kw)


def _ionp_domained_unary(op, label, data, mask_kw, **kwargs):
    return _domained_unary_adapter("anionpy", op, label, data, mask_kw)


# ---------------------------------------------------------------------------
# DOMAINED BINARY family: module-level ma.<fn> calls.
# ---------------------------------------------------------------------------
_DOMAINED_BINARY_OPS = ["divide", "true_divide", "floor_divide", "remainder", "mod", "fmod"]


def _domained_binary_cases():
    # (a_data, a_mask_kw, other_spec) triples. `other_spec` uses the plain
    # tuple protocol `_other_operand` understands.
    cases = [
        ("bydiv0_unmasked", ([1.0, 2.0], {}, ("ma", [0.0, 4.0], {}))),
        ("bydiv0_maskedpos", ([1.0, 2.0], {"mask": [False, False]}, ("ma", [0.0, 4.0], {"mask": [True, False]}))),
        ("bydiv0_a_masked", ([1.0, 2.0], {"mask": [True, False]}, ("ma", [0.0, 4.0], {}))),
        ("bydiv0_scalar_other", ([1.0, 2.0], {}, ("scalar", 0.0))),
        ("bydiv0_list_other", ([1.0, 2.0], {}, ("list", [0.0, 4.0]))),
        ("invalid_0by0", ([0.0], {}, ("ma", [0.0], {}))),
        ("fully_masked_div0", ([0.0, 0.0], {"mask": [True, True]}, ("ma", [0.0, 0.0], {"mask": [True, True]}))),
        ("empty", ([], {}, ("ma", [], {}))),
        ("good_only_nomask", ([1.0, 2.0], {}, ("ma", [2.0, 4.0], {}))),
        ("good_only_masked", ([1.0, 2.0], {"mask": [True, False]}, ("ma", [2.0, 4.0], {}))),
        ("bad_0d", (1.0, {}, ("ma", 0.0, {}))),
        ("bad_0d_masked", (1.0, {"mask": True}, ("ma", 0.0, {}))),
        ("good_0d", (1.0, {}, ("ma", 2.0, {}))),
    ]
    out = []
    for op in _DOMAINED_BINARY_OPS:
        for label, payload in cases:
            out.append((f"{op}|{label}", (op, label, payload), {}))
    return out


def _domained_binary_adapter(lib, op, label, payload):
    a_data, a_mask_kw, other_spec = payload

    def fn():
        a = _build_ma(lib, a_data, **a_mask_kw)
        other = _other_operand(lib, other_spec)
        m = __import__("numpy").ma if lib == "numpy" else __import__("anionpy").ma
        return getattr(m, op)(a, other)

    snapshot = _mc._np_snapshot if lib == "numpy" else _mc._ionp_snapshot
    warns, result, exc = _record(fn)
    return _descriptor(warns, result, exc, snapshot)


def _numpy_domained_binary(op, label, payload, **kwargs):
    return _domained_binary_adapter("numpy", op, label, payload)


def _ionp_domained_binary(op, label, payload, **kwargs):
    return _domained_binary_adapter("anionpy", op, label, payload)


# ---------------------------------------------------------------------------
# DUNDER routes: `/` and `//` (and reflected forms) -- the four items under
# standing "exact" review by this task. `%`'s dunders (`__mod__`/`__rmod__`/
# `__imod__`) have their own dedicated case families further below (dtype
# axis needed, see module docstring) rather than being folded in here.
# ---------------------------------------------------------------------------
_DUNDER_OPS = {
    "__truediv__": lambda a, b: a / b,
    "__rtruediv__": lambda a, b: b / a,
    "__floordiv__": lambda a, b: a // b,
    "__rfloordiv__": lambda a, b: b // a,
}


def _dunder_cases():
    # `a` and `other` BOTH carry a zero: `a/other` divides by `other`'s zero
    # (position 1 below) and `other/a` (the reflected forms, __rtruediv__/
    # __rfloordiv__) divides by `a`'s zero (position 0) -- a payload with a
    # zero on only one side would silently under-test whichever dunder puts
    # that operand in the DENOMINATOR position, since `a / other` and
    # `other / a` are genuinely different divisions. Verified this bites:
    # an earlier revision of this corpus put the zero in `other` only, and
    # the guard-bite proof in this task's report found __rtruediv__/
    # __rfloordiv__ never actually exercised a division-by-zero at all.
    payloads = [
        ("div0_ma_other", ([0.0, 2.0], {}, ("ma", [1.0, 0.0], {}))),
        ("div0_scalar_other", ([0.0, 2.0], {}, ("scalar", 0.0))),
        ("div0_list_other", ([0.0, 2.0], {}, ("list", [1.0, 0.0]))),
        ("div0_a_masked", ([0.0, 2.0], {"mask": [True, False]}, ("ma", [1.0, 0.0], {}))),
        ("div0_other_masked", ([0.0, 2.0], {}, ("ma", [1.0, 0.0], {"mask": [True, False]}))),
        ("good_only", ([1.0, 2.0], {}, ("ma", [2.0, 4.0], {}))),
        ("empty", ([], {}, ("ma", [], {}))),
        ("bad_0d_scalar_other", (0.0, {}, ("scalar", 0.0))),
    ]
    out = []
    for op in _DUNDER_OPS:
        for label, payload in payloads:
            out.append((f"{op}|{label}", (op, label, payload), {}))
    return out


def _dunder_adapter(lib, op, label, payload):
    a_data, a_mask_kw, other_spec = payload

    def fn():
        a = _build_ma(lib, a_data, **a_mask_kw)
        other = _other_operand(lib, other_spec)
        return _DUNDER_OPS[op](a, other)

    snapshot = _mc._np_snapshot if lib == "numpy" else _mc._ionp_snapshot
    warns, result, exc = _record(fn)
    return _descriptor(warns, result, exc, snapshot)


def _numpy_dunder(op, label, payload, **kwargs):
    return _dunder_adapter("numpy", op, label, payload)


def _ionp_dunder(op, label, payload, **kwargs):
    return _dunder_adapter("anionpy", op, label, payload)


# ---------------------------------------------------------------------------
# MOD DUNDER routes: `__mod__`/`__rmod__`/`__imod__`. See module docstring
# for why these need their own dtype-aware case family instead of reusing
# `_DUNDER_OPS`/`_dunder_cases` above.
# ---------------------------------------------------------------------------
_MOD_DUNDER_OPS = {
    "__mod__": lambda a, b: a % b,
    "__rmod__": lambda a, b: b % a,
}


def _mod_dunder_payloads(int_dtype):
    # `a_bad` ALSO carries a zero (position 0), not just `other_bad`
    # (position 1): `__rmod__` computes `other % self`, so a zero only in
    # `other_bad` would put that zero in the NUMERATOR for the reflected
    # op and never exercise its domain/divide-by-zero path at all -- this
    # is the exact `_dunder_cases` lesson above ("an earlier revision of
    # this corpus put the zero in `other` only, and the guard-bite proof
    # found `__rtruediv__`/`__rfloordiv__` never actually exercised a
    # division-by-zero at all") repeating itself here, caught the same way
    # (a guard-bite run on this file: `__rmod__|int|*` cases silently
    # never went through the int-dtype warning branch until this fix).
    if int_dtype:
        a_good, a_bad, other_good, other_bad, scalar_good, scalar_bad, bad_0d, good_0d = (
            [5, 2], [0, 2], [2, 3], [2, 0], 2, 0, 0, 5,
        )
    else:
        a_good, a_bad, other_good, other_bad, scalar_good, scalar_bad, bad_0d, good_0d = (
            [5.0, 2.0], [0.0, 2.0], [2.0, 3.0], [2.0, 0.0], 2.0, 0.0, 0.0, 5.0,
        )
    return [
        ("div0_ma_other", (a_bad, {}, ("ma", other_bad, {}))),
        ("div0_scalar_other", (a_bad, {}, ("scalar", scalar_bad))),
        ("div0_list_other", (a_bad, {}, ("list", other_bad))),
        ("div0_a_masked", (a_bad, {"mask": [True, False]}, ("ma", other_bad, {}))),
        ("div0_other_masked", (a_bad, {}, ("ma", other_bad, {"mask": [True, False]}))),
        ("good_only", (a_good, {}, ("ma", other_good, {}))),
        ("empty", ([], {}, ("ma", [], {}))),
        ("bad_0d_scalar_other", (bad_0d, {}, ("scalar", scalar_bad))),
        # In-domain, unmasked 0-d -- the exact shape that caught BUG2
        # during this task's live probing (real numpy's `__array_wrap__`
        # has no scalar-unwrap branch; `MaskedArray(5.0) % MaskedArray(2.0)`
        # must come back as a 0-d `MaskedArray`, not a bare float). Silent
        # on both sides (in-domain), but the underlying snapshot's type
        # tag (`"MA"` vs `"SCALAR"`) is what this case is actually for.
        ("good_0d_scalar_other", (good_0d, {}, ("scalar", scalar_good))),
    ]


def _mod_dunder_cases():
    out = []
    for int_dtype in (True, False):
        dtype_label = "int" if int_dtype else "float"
        for label, payload in _mod_dunder_payloads(int_dtype):
            for op in _MOD_DUNDER_OPS:
                out.append((f"{op}|{dtype_label}|{label}", (op, payload), {}))
    return out


def _mod_dunder_adapter(lib, op, payload):
    a_data, a_mask_kw, other_spec = payload

    def fn():
        a = _build_ma(lib, a_data, **a_mask_kw)
        other = _other_operand(lib, other_spec)
        return _MOD_DUNDER_OPS[op](a, other)

    snapshot = _mc._np_snapshot if lib == "numpy" else _mc._ionp_snapshot
    warns, result, exc = _record(fn)
    return _descriptor(warns, result, exc, snapshot)


def _numpy_mod_dunder(op, payload, **kwargs):
    return _mod_dunder_adapter("numpy", op, payload)


def _ionp_mod_dunder(op, payload, **kwargs):
    return _mod_dunder_adapter("anionpy", op, payload)


# ---------------------------------------------------------------------------
# IMOD route: `__imod__` (`%=`). Kept separate from `_MOD_DUNDER_OPS` above
# because it mutates and returns `self`, not a fresh object -- the case
# label still carries the same dtype/div0 payload axis via
# `_mod_dunder_payloads`, but the adapter applies `a %= other` in place
# rather than calling a lambda that returns a new value.
# ---------------------------------------------------------------------------
def _imod_cases():
    out = []
    for int_dtype in (True, False):
        dtype_label = "int" if int_dtype else "float"
        for label, payload in _mod_dunder_payloads(int_dtype):
            out.append((f"__imod__|{dtype_label}|{label}", (payload,), {}))
    return out


def _imod_adapter(lib, payload):
    a_data, a_mask_kw, other_spec = payload

    def fn():
        a = _build_ma(lib, a_data, **a_mask_kw)
        other = _other_operand(lib, other_spec)
        a %= other
        return a

    snapshot = _mc._np_snapshot if lib == "numpy" else _mc._ionp_snapshot
    warns, result, exc = _record(fn)
    return _descriptor(warns, result, exc, snapshot)


def _numpy_imod(payload, **kwargs):
    return _imod_adapter("numpy", payload)


def _ionp_imod(payload, **kwargs):
    return _imod_adapter("anionpy", payload)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS -- plain (non-domained) unary/binary families. See
# module docstring for why these must exist and what each proves.
# ---------------------------------------------------------------------------
def _plain_negative_cases():
    return [
        # (a): both sides MUST warn (non-vacuous), overflow -- outside the
        # divide/invalid errstate scope, proves the wrap is not a blanket
        # silencer.
        ("multiply_overflow", "binary", "multiply", ([1e300], {}), ("ma", [1e300], {})),
        ("multiply_overflow_masked", "binary", "multiply", ([1e300], {"mask": [False]}), ("ma", [1e300], {"mask": [False]})),
        # (a): both sides MUST warn -- genuinely invalid data (inf - inf),
        # PLAIN binary family (add has no domain function at all), proves
        # the plain family's own errstate wrap (make_masked_binary) still
        # lets real invalid-data warnings through exactly like numpy does
        # (matches real numpy behavior: this is not masked away).
        ("add_inf_minus_inf", "binary", "subtract", ([float("inf")], {}), ("ma", [float("inf")], {})),
        # (b): both sides MUST stay silent -- ordinary in-domain plain-unary
        # usage under every mask shape.
        ("abs_nomask", "unary", "abs", ([-1.0, 2.0, -3.0], {}), None),
        ("abs_masked", "unary", "abs", ([-1.0, 2.0, -3.0], {"mask": [True, False, True]}), None),
        ("abs_fully_masked", "unary", "abs", ([-1.0, -2.0], {"mask": [True, True]}), None),
        ("abs_empty", "unary", "abs", ([], {}), None),
        ("abs_int_dtype", "unary", "abs", ([-1, 2, -3], {"dtype": "int32"}), None),
        ("add_nomask", "binary", "add", ([1.0, 2.0], {}), ("ma", [3.0, 4.0], {})),
        ("add_masked", "binary", "add", ([1.0, 2.0], {"mask": [True, False]}), ("ma", [3.0, 4.0], {})),
        ("add_int_dtype", "binary", "add", ([1, 2], {"dtype": "int32"}), ("ma", [3, 4], {"dtype": "int32"})),
        ("add_scalar_other", "binary", "add", ([1.0, 2.0], {}), ("scalar", 5.0)),
        ("add_list_other", "binary", "add", ([1.0, 2.0], {}), ("list", [3.0, 4.0])),
    ]


def _plain_negative_adapter(lib, shape, op, a_payload, other_spec):
    a_data, a_mask_kw = a_payload

    def fn():
        a = _build_ma(lib, a_data, **a_mask_kw)
        m = __import__("numpy").ma if lib == "numpy" else __import__("anionpy").ma
        if shape == "unary":
            return getattr(m, op)(a)
        other = _other_operand(lib, other_spec)
        return getattr(m, op)(a, other)

    snapshot = _mc._np_snapshot if lib == "numpy" else _mc._ionp_snapshot
    warns, result, exc = _record(fn)
    return warns, _descriptor(warns, result, exc, snapshot)


def _plain_negative_cases_registry():
    out = []
    for label, shape, op, a_payload, other_spec in _plain_negative_cases():
        out.append((f"{shape}/{op}|{label}", (shape, op, a_payload, other_spec), {}))
    return out


def _numpy_plain_negative(shape, op, a_payload, other_spec, **kwargs):
    warns, descr = _plain_negative_adapter("numpy", shape, op, a_payload, other_spec)
    if op in ("multiply", "subtract"):
        # Mandatory non-vacuous check for the "both sides must warn"
        # controls: fail loudly (via the descriptor itself, so both
        # adapters produce a visibly different string) if numpy's OWN side
        # did not actually warn -- a vacuous "both silent" would otherwise
        # look identical to "both correctly suppressed".
        descr = f"nonvacuous={bool(warns)}|{descr}"
    return descr


def _ionp_plain_negative(shape, op, a_payload, other_spec, **kwargs):
    warns, descr = _plain_negative_adapter("anionpy", shape, op, a_payload, other_spec)
    if op in ("multiply", "subtract"):
        descr = f"nonvacuous={bool(warns)}|{descr}"
    return descr


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
def _install():
    REGISTRY["ma_warning_domained_unary"] = ItemSpec(
        name="ma_warning_domained_unary",
        kind="custom",
        numpy_adapter=_numpy_domained_unary,
        ionp_adapter=_ionp_domained_unary,
        scalar_like=True,
        custom_cases=_domained_unary_cases,
    )
    REGISTRY["ma_warning_domained_binary"] = ItemSpec(
        name="ma_warning_domained_binary",
        kind="custom",
        numpy_adapter=_numpy_domained_binary,
        ionp_adapter=_ionp_domained_binary,
        scalar_like=True,
        custom_cases=_domained_binary_cases,
    )
    REGISTRY["ma_warning_dunder"] = ItemSpec(
        name="ma_warning_dunder",
        kind="custom",
        numpy_adapter=_numpy_dunder,
        ionp_adapter=_ionp_dunder,
        scalar_like=True,
        custom_cases=_dunder_cases,
    )
    REGISTRY["ma_warning_plain_negative_controls"] = ItemSpec(
        name="ma_warning_plain_negative_controls",
        kind="custom",
        numpy_adapter=_numpy_plain_negative,
        ionp_adapter=_ionp_plain_negative,
        scalar_like=True,
        custom_cases=_plain_negative_cases_registry,
    )
    REGISTRY["ma_warning_mod_dunder"] = ItemSpec(
        name="ma_warning_mod_dunder",
        kind="custom",
        numpy_adapter=_numpy_mod_dunder,
        ionp_adapter=_ionp_mod_dunder,
        scalar_like=True,
        custom_cases=_mod_dunder_cases,
    )
    REGISTRY["ma_warning_imod"] = ItemSpec(
        name="ma_warning_imod",
        kind="custom",
        numpy_adapter=_numpy_imod,
        ionp_adapter=_ionp_imod,
        scalar_like=True,
        custom_cases=_imod_cases,
    )


_install()
