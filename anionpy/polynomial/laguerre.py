"""anionpy.polynomial.laguerre -- the Laguerre-series basis
(`numpy.polynomial.laguerre`).

Sibling module to `anionpy/polynomial/legendre.py` -- STRUCTURAL template
only, not a relabeled copy. See `ionp_core::laguerre`'s module doc comment
(`ionp-core/src/laguerre.rs`) for the full, verified-against-numpy-source
list of arithmetic-grouping and structural divergences from Legendre this
basis has (multiply-first Clenshaw grouping in `lagval`, no division at
all in `lagmulx`/`lagint`, an unscaled nonzero-diagonal `lagcompanion`,
etc.) -- those live in the Rust core; the divergences that live at THIS
layer (Python orchestration) are:

- `lagline(off, scl)` returns `[off+scl, -scl]`, a genuinely different
  affine map from `legline`'s `[off, scl]`.
- `lagroots` DOES cast its result to real when every eigenvalue's
  imaginary part is exactly `0.0` and the (rotated) companion matrix's own
  dtype is real -- a `_to_real_if_imag_zero`-equivalent downcast that
  `legroots` deliberately does NOT have (verified directly against
  `numpy/polynomial/laguerre.py:1576-1583`: `r = _to_real_if_imag_zero(r,
  m)` is called there but has no counterpart anywhere in
  `legendre.py`'s `legroots`). Real numpy's private helper can't be called
  from shipped code, so its exact documented semantics
  (`numpy.linalg._linalg._to_real_if_imag_zero`, read directly for this
  port) are reimplemented here using only public `anionpy` array ops.
- `laggauss` does NOT symmetrize `x`/`w` (no `(w+w[::-1])/2` /
  `(x-x[::-1])/2` step at all) and normalizes with a bare `w /= w.sum()`
  -- no leading factor of `2.0` (contrast `leggauss`'s
  `w = w * (2.0 / w.sum())`). Both verified directly against
  `numpy/polynomial/laguerre.py`'s `laggauss` body.
- `lagweight(x) = exp(-x)`, not `legweight`'s `x*0 + 1`.
- `lag2poly`/`poly2lag`'s degree-descending loops use Laguerre's own
  subtract-then-divide `c1` grouping and the extra final
  `polysub(c1, polymulx(c1))` term (mirroring `lagmul`'s divergence from
  `legmul` inside the Rust core, but replayed here in terms of
  `polyadd`/`polysub`/`polymulx` since `lag2poly`'s target basis is the
  power series, not Laguerre itself).

Every numerical recurrence over coefficient/sample-point DATA lives in
Rust (`ionp_core::laguerre`, bound via `ionp-py/src/laguerre.rs`'s
`_lag_*` functions). The bounded loops that DO appear below
(`lagfromroots`'s balanced-pairing tree, `lagpow`'s repeated multiply,
`poly2lag`/`lag2poly`'s degree-descending accumulation) are bounded by a
small, fixed count (O(log(number of roots)) or the polynomial
degree/power respectively), never by array/sample-point DATA size -- same
carve-out `legendre.py`/`polynomial.py` document.

SCOPE (deliberate, matching `legendre.py`'s own documented boundary): only
the 24 non-N-D items are implemented. Explicitly NOT implemented in this
pass: `lagval2d`, `lagval3d`, `lagvalnd`, `laggrid2d`, `laggrid3d`,
`lagvander2d`, `lagvander3d` (composition helpers, time-boxed out, not a
measured decline).

The `Laguerre(ABCPolyBase)` class is implemented at the bottom of this
file. NOTE: unlike Legendre/Hermite/HermiteE, Laguerre's `domain` is
`[0, 1]`, NOT `[-1, 1]` -- verified directly against
`numpy/polynomial/laguerre.py`'s `Laguerre.domain = np.array(lagdomain)`.
`lagmul` does not route through `np.convolve`/z-series (same as `legmul`),
so `__mul__`/`__rmul__`/`__pow__`/`fromroots` are declared exact for this
class per this task's measured, per-class sweep. `convert`/`cast` are NOT
declared -- see `legendre.py`'s equivalent note and the rebind-loop
comment at the bottom of this file: the shared `_polybase.py::
_compose_affine` helper is only mathematically valid for the power-series
basis and produces outright wrong values (not merely non-bit-exact ones)
for orthogonal bases like this one.

`lagfit`'s vector-`deg`, `full=True`, and `w=` are NOT implemented, same
documented boundary as `legfit`/`polyfit`.
"""
from __future__ import annotations

import anionpy as _ap
from anionpy import _anionpy as _core
from anionpy.polynomial import polyutils as _pu
from anionpy.polynomial.polynomial import polyadd as _polyadd
from anionpy.polynomial.polynomial import polymulx as _polymulx
from anionpy.polynomial.polynomial import polysub as _polysub
from anionpy.polynomial._polybase import ABCPolyBase as _ABCPolyBase

__all__ = [
    "lagdomain", "lagzero", "lagone", "lagx",
    "lagline", "lagfromroots",
    "lagadd", "lagsub", "lagmulx", "lagmul", "lagdiv", "lagpow",
    "lagder", "lagint",
    "lagval",
    "lagval2d", "lagval3d", "lagvalnd",
    "laggrid2d", "laggrid3d",
    "lagvander", "lagvander2d", "lagvander3d",
    "lagfit",
    "lagcompanion", "lagroots",
    "laggauss", "lagweight",
    "lagtrim",
    "lag2poly", "poly2lag",
    "Laguerre",
]

# ─────────────────────────── constants ────────────────────────────
lagdomain = _core._lag_domain()
lagzero = _core._lag_zero()
lagone = _core._lag_one()
lagx = _core._lag_x()


def lagline(off, scl):
    """Return the coefficients of the Laguerre series for ``off + scl*x``.

    `[off + scl, -scl]`, NOT `legline`'s `[off, scl]` -- built directly via
    `anionpy.array` (not the Rust `_lag_line` binding) for the same
    integer-dtype-preservation reason `legline` documents: `L.lagline(3,
    2).dtype` is `int64` in real numpy, and a Rust binding forcing `f64`
    params would break that.
    """
    if scl != 0:
        return _ap.array([off + scl, -scl])
    else:
        return _ap.array([off])


def lagtrim(c, tol=0):
    """Remove trailing coefficients with absolute value <= tol."""
    return _core._lag_trim(c, float(tol))


# ─────────────────────────── evaluation ────────────────────────────

def lagval(x, c, tensor=True):
    """Evaluate a Laguerre series at points x.

    Only plain 1-D `c` is bound (`tensor`'s value is then irrelevant,
    matching numpy), same scope note as `legendre.legval`.
    """
    return _core._lag_val(x, c)


def _lagval_nd(x, c, tensor=True):
    """PRIVATE, multi-D-`c`-capable companion to `lagval` above -- Task
    #34, see `polyutils.py`'s module docstring. Line-for-line transcription
    of real numpy 2.5.1's `lagval` Clenshaw-recursion body.
    """
    c = _ap.array(c, ndmin=1)
    if str(c.dtype) in _pu._INT_LIKE_DTYPES:
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = _ap.asanyarray(x)
    if isinstance(x, _ap.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if c.shape[0] == 1:
        c0 = c[0]
        c1 = 0
    elif c.shape[0] == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = c.shape[0]
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, c.shape[0] + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
    return c0 + c1 * (1 - x)


def lagval2d(x, y, c):
    """Evaluate a 2-D Laguerre series at points (x, y)."""
    return _pu._valnd(_lagval_nd, c, x, y)


def lagval3d(x, y, z, c):
    """Evaluate a 3-D Laguerre series at points (x, y, z)."""
    return _pu._valnd(_lagval_nd, c, x, y, z)


def lagvalnd(pts, c):
    """Evaluate an N-D Laguerre series at points."""
    return _pu._valnd(_lagval_nd, c, *pts)


def laggrid2d(x, y, c):
    """Evaluate a 2-D Laguerre series on the Cartesian product of x and y."""
    return _pu._gridnd(_lagval_nd, c, x, y)


def laggrid3d(x, y, z, c):
    """Evaluate a 3-D Laguerre series on the Cartesian product of x, y, z."""
    return _pu._gridnd(_lagval_nd, c, x, y, z)


# ─────────────────────────── arithmetic ────────────────────────────

def lagadd(c1, c2):
    """Add one Laguerre series to another."""
    return _core._lag_add(c1, c2)


def lagsub(c1, c2):
    """Subtract one Laguerre series from another."""
    return _core._lag_sub(c1, c2)


def lagmulx(c):
    """Multiply a Laguerre series by x."""
    return _core._lag_mulx(c)


def lagmul(c1, c2):
    """Multiply one Laguerre series by another (with reprojection)."""
    return _core._lag_mul(c1, c2)


def lagdiv(c1, c2):
    """Divide one Laguerre series by another, returning quotient and
    remainder (both reprojected onto the Laguerre basis)."""
    return _core._lag_div(c1, c2)


def lagpow(c, pow, maxpower=16):
    """Raise a Laguerre series to a power.

    Same "loop bound is `pow` (default-capped at 16 via `maxpower`), never
    array/sample-point data size" carve-out as `legendre.legpow`, calling
    the Rust-backed `lagmul` repeatedly instead of `legmul`.
    """
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    if maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    if power == 0:
        # dtype-preserving power==0 branch, same as legpow -- verified
        # live: L.lagpow([1j, 2j], 0) -> array([1.+0.j]) complex128.
        c0 = _core._lag_trim(c, 0.0)
        one = _ap.ones(1, dtype=c0.dtype)
        return one
    if power == 1:
        return _core._lag_trim(c, 0.0)
    prd = c
    for _ in range(2, power + 1):
        prd = lagmul(prd, c)
    return prd


# ─────────────────────────── calculus ────────────────────────────

def lagder(c, m=1, scl=1, axis=0):
    """Differentiate a Laguerre series.

    Only `axis=0` is bound, same scope note as `legendre.legder`.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.laguerre.lagder: only axis=0 (1-D coefficient "
            "arrays) is implemented in this build"
        )
    return _core._lag_der(c, int(m), float(scl))


def lagint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    """Integrate a Laguerre series.

    Same `axis=0`-only scope note as `lagder` above.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.laguerre.lagint: only axis=0 (1-D coefficient "
            "arrays) is implemented in this build"
        )
    if k is None:
        k = []
    try:
        k = list(k)
    except TypeError:
        k = [k]
    if len(k) > int(m):
        raise ValueError(
            "k must have length less than or equal to m"
            f" ({len(k)} > {int(m)})"
        )
    # Split each constant into a (re, im) pair rather than `float()`-casting
    # it -- an earlier version did that and silently discarded the
    # imaginary part of a genuinely complex `k` whenever `c` was complex
    # too (numpy's `lagint` keeps `k` in whatever dtype `c` itself has, no
    # float truncation). Caught by an out-of-corpus random bit-exact sweep,
    # not by the differential corpus. The real/complex promotion decision
    # itself is made in `ionp-py`'s `_lag_int`, which knows `c`'s coerced
    # dtype; this is pure marshaling, not arithmetic.
    kpairs = [(complex(v).real, complex(v).imag) for v in k]
    return _core._lag_int(c, int(m), kpairs, float(lbnd), float(scl))


# ─────────────────────────── roots ────────────────────────────

def lagfromroots(roots):
    """Generate a Laguerre series with the given roots.

    Same O(log(number of roots))-bounded balanced-pairing loop as
    `legendre.legfromroots`, calling `lagline`/`lagmul` instead of
    `legline`/`legmul`.
    """
    roots = _ap.asarray(roots)
    n = roots.shape[0] if roots.ndim else 0
    if n == 0:
        return _ap.array([1.0])
    if str(roots.dtype) == "bool":
        raise ValueError("Coefficient arrays have no common type")
    rl = _ap.sort(roots)
    is_complex = _ap.iscomplexobj(rl)
    # `pu._fromroots` builds each factor via `line_f(-r, 1)`, NOT a
    # hardcoded `[-r, 1]` -- that hardcoding is only numerically valid for
    # Legendre, where `legline(off, scl) == [off, scl]` makes the two the
    # same array. Laguerre's `lagline(off, scl) == [off+scl, -scl]` is a
    # genuinely different affine map, so this MUST go through `lagline`
    # itself (`lagline(-r, 1) == [-r+1, -1]`, verified against
    # `polyutils._fromroots`'s own `p = [line_f(-r, 1) for r in roots]`).
    # `scl` is passed as the plain Python int `1` here -- NOT `1.0 + 0j` --
    # matching `_fromroots`'s own `line_f(-r, 1)` literally (`polyutils.py:
    # 461`). This is load-bearing for signed zero: `lagline`'s `-scl` is a
    # genuine unary negation, and `-(1.0 + 0j)` (negating an already-complex
    # scl) produces `-1.0 - 0.0j` (negative-zero imaginary), while numpy's
    # `-1` (negating a plain int) stays a positive-zero-imaginary `-1.0 +
    # 0.0j` once `np.array([off+scl, -scl])` promotes it to complex
    # alongside `off`. An earlier version passed `1.0 + 0j` here and
    # produced a top-degree coefficient with the wrong imaginary sign after
    # the pairwise `lagmul` reduction -- caught by an out-of-corpus random
    # bit-exact sweep, not the fixed-case corpus.
    if is_complex:
        p = [lagline(complex(-complex(rl[i])), 1) for i in range(n)]
    else:
        p = [lagline(-float(rl[i]), 1.0) for i in range(n)]
    cnt = len(p)
    while cnt > 1:
        m, r = divmod(cnt, 2)
        tmp = [lagmul(p[i], p[i + m]) for i in range(m)]
        if r:
            tmp[0] = lagmul(tmp[0], p[-1])
        p = tmp
        cnt = m
    return p[0]


def lagvander(x, deg):
    """Pseudo-Vandermonde matrix of the given degree."""
    return _core._lag_vander(x, int(deg))


def lagvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((lagvander, lagvander), (x, y), deg)


def lagvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)


def lagcompanion(c):
    """Return the (unscaled) companion matrix of c."""
    return _core._lag_companion(c)


def lagroots(c):
    """Compute the roots of a Laguerre series.

    Uses the ROTATED companion matrix (`lagcompanion(c)[::-1, ::-1]`, same
    "reduces error" comment as `legroots`). UNLIKE `legroots`, real numpy's
    own `lagroots` DOES cast the result to real when possible -- verified
    directly against `numpy/polynomial/laguerre.py:1576-1583`:
    ``r = _to_real_if_imag_zero(r, m)`` is called there, with no
    counterpart in `legroots`. `_to_real_if_imag_zero`'s own documented
    semantics (read directly, `numpy.linalg._linalg`, since it is a
    private helper this build cannot call from shipped code): downcast to
    real if AND ONLY IF the companion matrix `m`'s own dtype is not
    complex AND every returned eigenvalue's imaginary part is EXACTLY
    `0.0` (not epsilon-close) -- reimplemented here with only public
    `anionpy` array ops.
    """
    c = _ap.asarray(c)
    c = _core._lag_trim(c, 0.0)
    n = c.shape[0]
    if n < 2:
        return _ap.array([], dtype=c.dtype)
    if n == 2:
        # len(c) == 2 special case: 1 + c0/c1 -- NOT legroots's -c0/c1.
        return _ap.array([1 + c[0] / c[1]])
    m = lagcompanion(c)
    m = m[::-1, ::-1]
    r = _ap.linalg.eigvals(m)
    r = _ap.sort(r)
    mat_is_complex = _ap.iscomplexobj(m)
    if not mat_is_complex and bool(_ap.all(_ap.imag(r) == 0.0)):
        r = _ap.real(r)
    return r


# ─────────────────────────── fitting ────────────────────────────

def lagfit(x, y, deg, rcond=None, full=False, w=None):
    """Least-squares fit of a Laguerre series to data.

    Same documented scope boundary as `legendre.legfit`: only a scalar
    integer `deg`, `full=False`, and `w=None` are implemented.
    """
    if full:
        raise NotImplementedError(
            "anionpy.polynomial.laguerre.lagfit: full=True (SVD diagnostics) "
            "is not implemented in this build"
        )
    if w is not None:
        raise NotImplementedError(
            "anionpy.polynomial.laguerre.lagfit: w= (weighted fit) is not "
            "implemented in this build"
        )
    if hasattr(deg, "__len__"):
        raise NotImplementedError(
            "anionpy.polynomial.laguerre.lagfit: array-valued deg (specific "
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
    van = lagvander(x, lmax)
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


# ─────────────────────────── quadrature ────────────────────────────

def laggauss(deg):
    """Gauss-Laguerre quadrature: sample points and weights.

    Ported directly from `laguerre.py`'s own body (companion-matrix
    `eigvalsh` first approximation -- the companion matrix for
    `c = [0]*ideg + [1]` is genuinely symmetric here, since `c[:-1]` is all
    zero so the `mat[:, -1] += ...` correction vanishes and `top == bot`
    by construction -- one Newton refinement step, then weight computation
    via `lagval`). UNLIKE `leggauss`, this does NOT symmetrize `x`/`w`
    (no `(w+w[::-1])/2` / `(x-x[::-1])/2` step exists in real numpy's own
    `laggauss` body) and normalizes with a bare `w /= w.sum()` -- no
    leading factor of `2.0`. Both verified directly against
    `numpy/polynomial/laguerre.py`'s `laggauss`.
    """
    ideg = int(deg)
    if ideg != deg or ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = _ap.array([0] * ideg + [1])
    m = lagcompanion(c)
    x = _ap.linalg.eigvalsh(m)

    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x = x - dy / df

    fm = lagval(x, c[1:])
    fm = fm / _ap.abs(fm).max()
    df = df / _ap.abs(df).max()
    w = 1 / (fm * df)

    w = w / w.sum()

    return x, w


def lagweight(x):
    """Weight function of the Laguerre polynomials: ``exp(-x)``.

    Verified live: `NL.lagweight(-1.0)` returns `2.718281828459045`, no
    domain exception -- `exp(-x)` is defined for all real `x`, even though
    the Laguerre polynomials themselves are only orthogonal on `[0, inf)`.
    """
    return _ap.exp(-x)


# ─────────────────────────── basis conversion ────────────────────────────

def poly2lag(pol):
    """Convert a polynomial (power-series basis) to a Laguerre series.

    Same "loop bound is `deg = len(pol) - 1`, never sample/coefficient
    DATA size" carve-out as `legendre.poly2leg`, calling `lagmulx`/`lagadd`
    instead of `legmulx`/`legadd`.
    """
    pol = _core._lag_trim(pol, 0.0)
    deg = pol.shape[0] - 1
    res = _ap.array([0.0])
    for i in range(deg, -1, -1):
        res = lagadd(lagmulx(res), pol[i:i + 1])
    return res


def lag2poly(c):
    """Convert a Laguerre series to a polynomial (power-series basis).

    Same "loop bound is `n = len(c)`, never independent sample/coefficient
    DATA size" carve-out as `legendre.leg2poly`, but Laguerre's OWN
    subtract-then-divide `c1`-update grouping and its extra final
    `polysub(c1, polymulx(c1))` term (mirroring `lagmul`'s divergence from
    `legmul` in `ionp_core::laguerre`, replayed here in `polyadd`/
    `polysub`/`polymulx` terms since the target basis is the power series):
    verified directly against `numpy/polynomial/laguerre.py`'s `lag2poly`,
    NOT copied from `leg2poly`'s grouping.
    """
    c = _ap.asarray(c)
    c = lagtrim(c, 0.0)
    n = c.shape[0]
    if n == 1:
        return c
    c0 = c[-2:-1]
    c1 = c[-1:]
    for i in range(n - 1, 1, -1):
        tmp = c0
        c0 = _polysub(c[i - 2:i - 1], (c1 * (i - 1)) / i)
        c1 = _polyadd(tmp, _polysub((2 * i - 1) * c1, _polymulx(c1)) / i)
    return _polyadd(c0, _polysub(c1, _polymulx(c1)))


class Laguerre(_ABCPolyBase):
    """A Laguerre series class (`numpy.polynomial.laguerre.Laguerre`).

    Assembles the already-implemented `lag*` module functions above
    through the shared `ABCPolyBase` generic machinery in `_polybase.py`.
    NOTE: `domain`/`window` are `[0, 1]`, NOT `[-1, 1]` -- verified
    directly against numpy source, do not copy Legendre's/Hermite's.
    """

    _add = staticmethod(lagadd)
    _sub = staticmethod(lagsub)
    _mul = staticmethod(lagmul)
    _div = staticmethod(lagdiv)
    _pow = staticmethod(lagpow)
    _val = staticmethod(lagval)
    _int = staticmethod(lagint)
    _der = staticmethod(lagder)
    _fit = staticmethod(lagfit)
    _line = staticmethod(lagline)
    _roots = staticmethod(lagroots)
    _fromroots = staticmethod(lagfromroots)

    domain = _ap.array(lagdomain)
    window = _ap.array(lagdomain)
    basis_name = "L"


# Ticket #75 (2026-08-08): the `_LAG_COVERAGE_REBIND_*` block and the
# `_rebind_generic_dunders(Laguerre)` call formerly here were deleted --
# see `chebyshev.py`'s identical comment for the full rationale. numpy's
# own `Laguerre` doesn't bind any of these names in ITS `__dict__` either,
# so inheriting the same defaults from `ABCPolyBase`/`object` is genuine
# parity under `tools/coverage.py`'s numpy-relative check; nothing left to
# rebind.
