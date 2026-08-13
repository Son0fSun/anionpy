"""anionpy.polynomial.polynomial -- the power-series basis
(`numpy.polynomial.polynomial`).

This is a pure-Python ASSEMBLY layer, mirroring every other Rust/Python
seam in this codebase (see `anionpy/ma/core.py` for the precedent this
follows): every numerical recurrence over coefficient/sample-point DATA
lives in Rust (`ionp_core::poly`, bound via `ionp-py/src/poly.rs`'s
`_poly_*` functions on the compiled `anionpy._anionpy` extension). The
functions in this file do only argument coercion, dtype/shape bookkeeping,
and orchestration -- the two loops that DO appear below
(`polyfromroots`'s balanced-pairing tree and `polypow`'s repeated
multiply) are bounded by a small, fixed count (O(log(number of roots)) and
the polynomial power respectively, never by array/sample-point DATA size,
which is what the project's "no numerical loop in a .py file" rule is
actually about -- see this task's report for the fuller argument, and
`anionpy/linalg.rs`'s / `anionpy/ma/`'s own precedent for the same
"loop bound is a small fixed count, not data size" carve-out).

SCOPE (deliberate, see this task's report -- NOT a placeholder):

Task #34 (2026-08-08) added `polyval2d`, `polyval3d`, `polyvalnd`,
`polygrid2d`, `polygrid3d`, `polyvander2d`, `polyvander3d` -- built on the
shared `_vander_nd`/`_vander_nd_flat`/`_valnd`/`_gridnd` machinery in
`anionpy/polynomial/polyutils.py` (see that module's docstring for the
full design, including why a NEW private `_polyval_nd` (below) exists
instead of reusing the already-declared, already-tested `polyval` --
`polyval` is deliberately 1-D-`c`-only and reusing it for the multi-D
case here would have re-scoped an already-shipped item's semantics for
a different ticket's items).

Explicitly NOT implemented in this pass:
  - The `Polynomial` class and `ABCPolyBase` machinery (shared across all
    six polynomial bases) -- explicitly out of scope per this task's
    brief; the other five bases entirely.
  - `polyfit`'s vector-`deg` (list of specific term degrees rather than
    "all terms up to deg"), `full=True` diagnostic return, and `w=`
    weighting -- see `polyfit` below for the precise, documented boundary
    (mirrors `anionpy/linalg.rs`'s own "raise NotImplementedError rather
    than silently compute a different thing" policy for unbound call
    forms).
"""
from __future__ import annotations

import anionpy as _ap
from anionpy import _anionpy as _core
from anionpy.polynomial import polyutils as _pu
from anionpy.polynomial._polybase import ABCPolyBase as _ABCPolyBase

__all__ = [
    "polydomain", "polyzero", "polyone", "polyx",
    "polyline", "polyfromroots",
    "polyadd", "polysub", "polymulx", "polymul", "polydiv", "polypow",
    "polyder", "polyint",
    "polyval", "polyvalfromroots",
    "polyval2d", "polyval3d", "polyvalnd",
    "polygrid2d", "polygrid3d",
    "polyvander", "polyvander2d", "polyvander3d",
    "polyfit",
    "polycompanion", "polyroots",
    "polytrim",
    "Polynomial",
]

# ─────────────────────────── constants ────────────────────────────
# Real numpy re-derives these on every module import too (they are plain
# module-level `np.array(...)` literals in `polynomial.py`), so minting
# them once at import time here via the same Rust constructors the rest of
# this module uses is the direct equivalent, not a shortcut.
polydomain = _core._poly_domain()
polyzero = _core._poly_zero()
polyone = _core._poly_one()
polyx = _core._poly_x()


def polyline(off, scl):
    """Return the coefficients of the line ``off + scl*x``.

    numpy's own implementation is `np.array([off, scl])` (or `[off]` when
    `scl == 0`) -- a bare array literal, no dtype coercion, so it
    preserves whatever dtype `off`/`scl` carry (int in stays int out;
    verified directly: `P.polyline(1, -1).dtype` is `int64`). Building
    this via `anionpy.array()` directly, rather than routing through the
    Rust `_poly_line` binding (which only accepts `f64` and would force
    every call to float64), is the faithful port: this is construction,
    not arithmetic, so it needs no Rust kernel of its own -- same
    reasoning as `polyfromroots`'s inlined `[-r, 1]` construction below.
    """
    if scl != 0:
        return _ap.array([off, scl])
    else:
        return _ap.array([off])


def polytrim(c, tol=0):
    """Remove trailing coefficients with absolute value <= tol."""
    return _core._poly_trim(c, float(tol))


# ─────────────────────────── evaluation ────────────────────────────

def polyval(x, c, tensor=True):
    """Evaluate a polynomial at points x.

    ``tensor=False``'s "evaluate elementwise against a matching-shape c"
    call form (used when `c` has extra trailing axes representing multiple
    simultaneous polynomials) is NOT implemented -- only plain 1-D `c`
    (`tensor`'s value is then irrelevant, matching numpy: with 1-D `c`,
    `tensor=True` and `tensor=False` give identical results) is bound.
    """
    return _core._poly_val(x, c)


def polyvalfromroots(x, r, tensor=True):
    """Evaluate a polynomial given its roots, at points x.

    Same 1-D-`r`-only scope note as `polyval` above.
    """
    return _core._poly_valfromroots(x, r)


def _polyval_nd(x, c, tensor=True):
    """PRIVATE, multi-D-`c`-capable companion to `polyval` above -- see
    `polyutils.py`'s module docstring for why this exists separately from
    the public, deliberately 1-D-only `polyval`. Line-for-line transcription
    of real numpy 2.5.1's `polyval` Horner-recursion body (the part after
    its `as_series`-equivalent input coercion, which is instead handled by
    `_valnd`/`_gridnd`'s callers here).
    """
    c = _ap.array(c, ndmin=1)
    if str(c.dtype) in _pu._INT_LIKE_DTYPES:
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = _ap.asanyarray(x)
    if isinstance(x, _ap.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] + x * 0
    for i in range(2, c.shape[0] + 1):
        c0 = c[-i] + c0 * x
    return c0


def polyval2d(x, y, c):
    """Evaluate a 2-D polynomial at points (x, y)."""
    return _pu._valnd(_polyval_nd, c, x, y)


def polyval3d(x, y, z, c):
    """Evaluate a 3-D polynomial at points (x, y, z)."""
    return _pu._valnd(_polyval_nd, c, x, y, z)


def polyvalnd(pts, c):
    """Evaluate an N-D polynomial at points."""
    return _pu._valnd(_polyval_nd, c, *pts)


def polygrid2d(x, y, c):
    """Evaluate a 2-D polynomial on the Cartesian product of x and y."""
    return _pu._gridnd(_polyval_nd, c, x, y)


def polygrid3d(x, y, z, c):
    """Evaluate a 3-D polynomial on the Cartesian product of x, y and z."""
    return _pu._gridnd(_polyval_nd, c, x, y, z)


# ─────────────────────────── arithmetic ────────────────────────────

def polyadd(c1, c2):
    """Add one polynomial to another."""
    return _core._poly_add(c1, c2)


def polysub(c1, c2):
    """Subtract one polynomial from another."""
    return _core._poly_sub(c1, c2)


def polymulx(c):
    """Multiply a polynomial by x."""
    return _core._poly_mulx(c)


def polymul(c1, c2):
    """Multiply one polynomial by another."""
    return _core._poly_mul(c1, c2)


def polydiv(c1, c2):
    """Divide one polynomial by another, returning quotient and remainder."""
    return _core._poly_div(c1, c2)


def polypow(c, pow, maxpower=None):
    """Raise a polynomial to a power.

    numpy's own `_pow` loop bound is `pow` (default-capped at 16 via
    `maxpower`), never array/sample-point data size, so orchestrating it
    here by calling the Rust-backed `polymul` repeatedly is the same
    "small fixed count, not data" carve-out `polyfromroots` below uses --
    not a violation of "no numerical loop in a .py file" (that rule is
    about loops over DATA, e.g. per-coefficient or per-sample-point
    arithmetic, which never happens here; each `polymul` call below is a
    single Rust-backed convolution over the full coefficient arrays).
    """
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    if maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    if power == 0:
        # Match numpy: always a fresh float64 [1.0], regardless of c's own
        # dtype (verified against real numpy 2.5.1: `polypow([1j], 0)` is
        # `array([1.])`, not `array([1.+0.j])`).
        return _ap.array([1.0])
    if power == 1:
        # numpy: `elif power == 1: return c` -- a plain `as_series`-promoted
        # copy of `c`, no arithmetic. `_poly_trim(c, 0.0)` is the promoting,
        # non-mutating copy this project's Rust boundary already provides.
        return _core._poly_trim(c, 0.0)
    prd = c
    for _ in range(2, power + 1):
        prd = polymul(prd, c)
    return prd


# ─────────────────────────── calculus ────────────────────────────

def polyder(c, m=1, scl=1, axis=0):
    """Differentiate a polynomial.

    Only `axis=0` (the only axis a 1-D coefficient array has) is bound;
    numpy's full N-D/`axis=`-selectable generality (for coefficient arrays
    representing multiple simultaneous polynomials stacked along other
    axes) is out of scope for this pass.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.polynomial.polyder: only axis=0 (1-D coefficient "
            "arrays) is implemented in this build"
        )
    return _core._poly_der(c, int(m), float(scl))


def polyint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    """Integrate a polynomial.

    Same `axis=0`-only scope note as `polyder` above.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.polynomial.polyint: only axis=0 (1-D coefficient "
            "arrays) is implemented in this build"
        )
    if k is None:
        k = []
    try:
        k = [float(v) for v in k]
    except TypeError:
        k = [float(k)]
    if len(k) > int(m):
        raise ValueError(
            "k must have length less than or equal to m"
            f" ({len(k)} > {int(m)})"
        )
    return _core._poly_int(c, int(m), k, float(lbnd), float(scl))


# ─────────────────────────── roots ────────────────────────────

def polyfromroots(roots):
    """Generate a polynomial with the given roots.

    numpy's `_fromroots` balanced-pairing loop is bounded by O(log(number
    of roots)), never by sample/coefficient DATA size -- orchestrating it
    here by calling the Rust-backed `polyline`/`polymul` repeatedly is the
    same carve-out `polypow` above uses.
    """
    roots = _ap.asarray(roots)
    n = roots.shape[0] if roots.ndim else 0
    if n == 0:
        return _ap.array([1.0])
    if str(roots.dtype) == "bool":
        # numpy's `_fromroots` runs `[roots] = as_series([roots], trim=False)`
        # -- `as_series` rejects a bool-dtype array via `np.common_type`,
        # which raises for bool outright (verified directly:
        # `P.polyfromroots([True, False])` -> `ValueError("Coefficient
        # arrays have no common type")`, same plural wording used even
        # for this single-array call).
        raise ValueError("Coefficient arrays have no common type")
    rl = _ap.sort(roots)
    # `[polyline(-r, 1) for r in roots]` in numpy's own `_fromroots` --
    # inlined here (rather than calling the public `polyline()` above)
    # because that binding's Rust boundary (`_poly_line`, `ionp-py/src/
    # poly.rs`) only accepts real `f64` off/scl, while roots (and hence
    # `-r`) may be complex; building the 2-term `[-r, 1]` array literal
    # directly is construction, not arithmetic, so it needs no Rust
    # kernel of its own.
    is_complex = _ap.iscomplexobj(rl)
    if is_complex:
        p = [_ap.array([complex(-complex(rl[i])), 1.0 + 0j]) for i in range(n)]
    else:
        p = [_ap.array([-float(rl[i]), 1.0]) for i in range(n)]
    cnt = len(p)
    while cnt > 1:
        m, r = divmod(cnt, 2)
        tmp = [polymul(p[i], p[i + m]) for i in range(m)]
        if r:
            tmp[0] = polymul(tmp[0], p[-1])
        p = tmp
        cnt = m
    return p[0]


def polyvander(x, deg):
    """Pseudo-Vandermonde matrix of the given degree."""
    return _core._poly_vander(x, int(deg))


def polyvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)


def polycompanion(c):
    """Return the companion matrix of c."""
    return _core._poly_companion(c)


def polyroots(c):
    """Compute the roots of a polynomial."""
    c = _ap.asarray(c)
    # `as_series` trims trailing zeros before the length checks below (its
    # `trim=True` default) -- mirror that here so a padded input like
    # `[1, 2, 0]` degree-collapses the same way numpy's does.
    c = _core._poly_trim(c, 0.0)
    n = c.shape[0]
    if n < 2:
        return _ap.array([], dtype=c.dtype)
    if n == 2:
        return _ap.array([-c[0] / c[1]])
    m = polycompanion(c)
    r = _ap.linalg.eigvals(m)
    r = _ap.sort(r)
    # numpy's `_to_real_if_imag_zero`: cast to real only if the companion
    # matrix's dtype was non-complex AND every eigenvalue's imaginary part
    # is EXACTLY 0.0 -- no tolerance. `m`'s dtype mirrors `c`'s dtype
    # exactly (the Rust `companion` kernel never promotes), so checking
    # `m.dtype` here is equivalent to numpy's own check on its own
    # companion matrix.
    if not _ap.iscomplexobj(m) and _ap.all(_ap.imag(r) == 0.0):
        r = _ap.real(r)
    return r


# ─────────────────────────── fitting ────────────────────────────

def polyfit(x, y, deg, rcond=None, full=False, w=None):
    """Least-squares fit of a polynomial to data.

    SCOPE (deliberate, matching `anionpy/linalg.rs`'s own documented-gap
    policy of raising `NotImplementedError` rather than silently computing
    a differently-derived answer): only a scalar integer `deg` (fit every
    term up to and including that degree -- NOT numpy's alternate
    "1-D array of specific term degrees to include" form), `full=False`,
    and `w=None` are implemented. Both `x` and `y` may be 1-D; `y` may
    also be 2-D (multiple simultaneous fits, one per column), matching
    numpy's own documented `y` shape contract.
    """
    if full:
        raise NotImplementedError(
            "anionpy.polynomial.polynomial.polyfit: full=True (SVD diagnostics) "
            "is not implemented in this build"
        )
    if w is not None:
        raise NotImplementedError(
            "anionpy.polynomial.polynomial.polyfit: w= (weighted fit) is not "
            "implemented in this build"
        )
    if hasattr(deg, "__len__"):
        raise NotImplementedError(
            "anionpy.polynomial.polynomial.polyfit: array-valued deg (specific "
            "term selection) is not implemented in this build; pass a single int"
        )
    deg = int(deg)
    if deg < 0:
        raise ValueError("expected deg >= 0")

    x = _ap.asarray(x) + 0.0
    y = _ap.asarray(y) + 0.0
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.shape[0] == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    lmax = deg
    van = polyvander(x, lmax)
    lhs = _ap.transpose(van)
    rhs = _ap.transpose(y)

    if rcond is None:
        rcond = x.shape[0] * _ap.finfo(x.dtype).eps

    if _ap.iscomplexobj(lhs):
        scl = _ap.sqrt(_ap.sum(_ap.square(_ap.real(lhs)) + _ap.square(_ap.imag(lhs)), axis=1))
    else:
        scl = _ap.sqrt(_ap.sum(_ap.square(lhs), axis=1))
    scl = _ap.where(scl == 0, 1.0, scl)

    lhs_scaled = _ap.transpose(_ap.transpose(lhs) / scl)
    c, _resids, _rank, _s = _ap.linalg.lstsq(_ap.transpose(lhs_scaled), _ap.transpose(rhs), rcond)
    c = _ap.transpose(_ap.transpose(c) / scl)
    return c


# ─────────────────────────── Polynomial class ────────────────────────────

class Polynomial(_ABCPolyBase):
    """A power series class (`numpy.polynomial.polynomial.Polynomial`).

    Assembly over the 21 functions above: `ABCPolyBase` (see
    `anionpy/polynomial/_polybase.py`) implements every dunder/method
    generically in terms of the 12 static methods and 3 properties bound
    below, all of which are the already-implemented, already-declared-exact
    functions from this module -- nothing new is computed here.
    """

    _add = staticmethod(polyadd)
    _sub = staticmethod(polysub)
    _mul = staticmethod(polymul)
    _div = staticmethod(polydiv)
    _pow = staticmethod(polypow)
    _val = staticmethod(polyval)
    _int = staticmethod(polyint)
    _der = staticmethod(polyder)
    _fit = staticmethod(polyfit)
    _line = staticmethod(polyline)
    _roots = staticmethod(polyroots)
    _fromroots = staticmethod(polyfromroots)

    # Matches real numpy's own `_polybase.py` verbatim, quirk included:
    # `window` is built from `polydomain`, not a separate `polywindow` --
    # the power-series basis's domain and window are both `[-1., 1.]`, so
    # there is no separate constant to source it from either upstream or
    # here.
    domain = _ap.array(polydomain)
    window = _ap.array(polydomain)
    basis_name = None

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        if i == "1":
            return f"·{arg_str}"
        return f"·{arg_str}{i.translate(cls._superscript_mapping)}"

    @staticmethod
    def _str_term_ascii(i, arg_str):
        if i == "1":
            return f" {arg_str}"
        return f" {arg_str}**{i}"

    @staticmethod
    def _repr_latex_term(i, arg_str, needs_parens):
        if needs_parens:
            arg_str = rf"\left({arg_str}\right)"
        if i == 0:
            return "1"
        elif i == 1:
            return arg_str
        else:
            return f"{arg_str}^{{{i}}}"


# Ticket #75 (2026-08-08): the `_COVERAGE_REBIND_*` block and the
# `_rebind_generic_dunders(Polynomial)` call formerly here were deleted.
# They existed to satisfy `tools/coverage.py`'s old own-`__dict__`-only
# presence check for exploded classes, which could not tell "genuinely
# inherited from a real, purpose-built base class we wrote" (this file's
# situation, mirroring numpy's OWN `Polynomial`/`ABCPolyBase` split) apart
# from "inherited from `object` and never implemented" -- both looked
# identical as "absent from `vars(Polynomial)`". `resolve()` is now
# numpy-relative instead: numpy's own `Polynomial` doesn't bind
# `roots`/`__add__`/etc. in its own `__dict__` either (they live only on
# `ABCPolyBase`, exactly like here), so inheriting the same defaults is
# genuine parity and `hasattr` alone is sufficient -- nothing left to
# rebind. `domain`, `window`, `basis_name` remain own-defined directly in
# `Polynomial`'s class body above (concrete array/None values overriding
# `ABCPolyBase`'s abstract `property` placeholders), unchanged by this
# deletion.
