"""anionpy.polynomial.legendre -- the Legendre-series basis
(`numpy.polynomial.legendre`).

Sibling module to `anionpy/polynomial/polynomial.py` (the power-series
basis -- read that file's module docstring first, this one follows its
exact seam): every numerical recurrence over coefficient/sample-point DATA
lives in Rust (`ionp_core::legendre`, bound via `ionp-py/src/legendre.rs`'s
`_leg_*` functions on the compiled `anionpy._anionpy` extension). The
functions in this file do only argument coercion, dtype/shape bookkeeping,
and orchestration -- the bounded loops that DO appear below
(`legfromroots`'s balanced-pairing tree, `legpow`'s repeated multiply,
`poly2leg`'s degree-descending accumulation, `leg2poly`'s degree-descending
accumulation) are bounded by a small, fixed count (O(log(number of roots))
or the polynomial degree/power respectively), never by array/sample-point
DATA size -- same "loop bound is a small fixed count, not data size"
carve-out `polynomial.py` documents.

Genuinely NEW here (not power-series-reducible, see this task's report):
`legval`'s Clenshaw recursion (not Horner), `legmulx`/`legmul`'s three-term-
recurrence "reprojection" (not synthetic multiplication/convolution),
`legder`/`legint`'s basis-specific antiderivative fan-out, `legvander`'s
forward three-term recursion, and `legcompanion`'s SCALED (not bare
"negate-and-shift") companion matrix.

SCOPE (deliberate, matching `polynomial.py`'s own documented boundary):
only the 24 non-N-D items are implemented. Explicitly NOT implemented in
this pass: `legval2d`, `legval3d`, `legvalnd`, `leggrid2d`, `leggrid3d`,
`legvander2d`, `legvander3d` (composition helpers over 2-3 fixed axes,
time-boxed out, not a measured decline).

The `Legendre(ABCPolyBase)` class is implemented at the bottom of this
file, same "assembly, not new arithmetic" shape as `polynomial.py`'s
`Polynomial`. UNLIKE `chebyshev.py`'s `Chebyshev` class, `legmul` does
NOT route through `np.convolve`/z-series (it is a genuine three-term-
recurrence reprojection, declared exact above) -- so `__mul__`, `__rmul__`,
`__pow__`, and `fromroots` ARE declared exact for this class, per this
task's report's measured, per-class sweep (do not assume this from
Chebyshev's revocation, and do not assume it FOR this class without
checking that report). `convert`/`cast` are NOT declared for ANY of the
five new classes except (trivially, already-precedented) `Polynomial` --
see the rebind-loop comment at the bottom of this file for why: the
shared `_polybase.py::_compose_affine` helper is only mathematically
valid for the power-series basis, and produces outright wrong values
(not merely non-bit-exact ones) for orthogonal bases like this one.

`legfit`'s vector-`deg`, `full=True`, and `w=` are NOT implemented, same
documented boundary as `polynomial.py`'s `polyfit`.
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
    "legdomain", "legzero", "legone", "legx",
    "legline", "legfromroots",
    "legadd", "legsub", "legmulx", "legmul", "legdiv", "legpow",
    "legder", "legint",
    "legval",
    "legval2d", "legval3d", "legvalnd",
    "leggrid2d", "leggrid3d",
    "legvander", "legvander2d", "legvander3d",
    "legfit",
    "legcompanion", "legroots",
    "leggauss", "legweight",
    "legtrim",
    "leg2poly", "poly2leg",
    "Legendre",
]

# ─────────────────────────── constants ────────────────────────────
legdomain = _core._leg_domain()
legzero = _core._leg_zero()
legone = _core._leg_one()
legx = _core._leg_x()


def legline(off, scl):
    """Return the coefficients of the Legendre series for ``off + scl*x``.

    Same "construction, not arithmetic, preserves off/scl's own dtype"
    reasoning as `polynomial.polyline` -- built directly via `anionpy.array`
    rather than the Rust `_leg_line` binding (which forces `f64` params and
    would break integer-dtype preservation; verified directly:
    `L.legline(3, 2).dtype` is `int64`).
    """
    if scl != 0:
        return _ap.array([off, scl])
    else:
        return _ap.array([off])


def legtrim(c, tol=0):
    """Remove trailing coefficients with absolute value <= tol."""
    return _core._leg_trim(c, float(tol))


# ─────────────────────────── evaluation ────────────────────────────

def legval(x, c, tensor=True):
    """Evaluate a Legendre series at points x.

    Only plain 1-D `c` is bound (`tensor`'s value is then irrelevant,
    matching numpy), same scope note as `polynomial.polyval`.
    """
    return _core._leg_val(x, c)


def _legval_nd(x, c, tensor=True):
    """PRIVATE, multi-D-`c`-capable companion to `legval` above -- Task
    #34, see `polyutils.py`'s module docstring. Line-for-line transcription
    of real numpy 2.5.1's `legval` Clenshaw-recursion body.
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
            c0 = c[-i] - c1 * ((nd - 1) / nd)
            c1 = tmp + c1 * x * ((2 * nd - 1) / nd)
    return c0 + c1 * x


def legval2d(x, y, c):
    """Evaluate a 2-D Legendre series at points (x, y)."""
    return _pu._valnd(_legval_nd, c, x, y)


def legval3d(x, y, z, c):
    """Evaluate a 3-D Legendre series at points (x, y, z)."""
    return _pu._valnd(_legval_nd, c, x, y, z)


def legvalnd(pts, c):
    """Evaluate an N-D Legendre series at points."""
    return _pu._valnd(_legval_nd, c, *pts)


def leggrid2d(x, y, c):
    """Evaluate a 2-D Legendre series on the Cartesian product of x and y."""
    return _pu._gridnd(_legval_nd, c, x, y)


def leggrid3d(x, y, z, c):
    """Evaluate a 3-D Legendre series on the Cartesian product of x, y, z."""
    return _pu._gridnd(_legval_nd, c, x, y, z)


# ─────────────────────────── arithmetic ────────────────────────────

def legadd(c1, c2):
    """Add one Legendre series to another."""
    return _core._leg_add(c1, c2)


def legsub(c1, c2):
    """Subtract one Legendre series from another."""
    return _core._leg_sub(c1, c2)


def legmulx(c):
    """Multiply a Legendre series by x."""
    return _core._leg_mulx(c)


def legmul(c1, c2):
    """Multiply one Legendre series by another (with reprojection)."""
    return _core._leg_mul(c1, c2)


def legdiv(c1, c2):
    """Divide one Legendre series by another, returning quotient and
    remainder (both reprojected onto the Legendre basis)."""
    return _core._leg_div(c1, c2)


def legpow(c, pow, maxpower=16):
    """Raise a Legendre series to a power.

    numpy's own `_pow` loop bound is `pow` (default-capped at 16 via
    `maxpower`), never array/sample-point data size -- same carve-out
    `polynomial.polypow` uses, ported here calling the Rust-backed `legmul`
    repeatedly instead of `polymul`.
    """
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    if maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    if power == 0:
        # `pu._pow`'s power==0 branch is `np.array([1], dtype=c.dtype)` --
        # DTYPE-PRESERVING (unlike `polynomial.polypow`'s always-float64
        # `[1.]`; verified live against real numpy 2.5.1:
        # `L.legpow([1j, 2j], 0)` is `array([1.+0.j])` complex128, not
        # `array([1.])`). `legtrim(c, 0.0)` promotes/copies `c` the same
        # way `_leg_trim` always does elsewhere in this module, then we
        # overwrite its data with a single one-valued entry of the SAME
        # resulting dtype.
        c0 = _core._leg_trim(c, 0.0)
        one = _ap.ones(1, dtype=c0.dtype)
        return one
    if power == 1:
        return _core._leg_trim(c, 0.0)
    prd = c
    for _ in range(2, power + 1):
        prd = legmul(prd, c)
    return prd


# ─────────────────────────── calculus ────────────────────────────

def legder(c, m=1, scl=1, axis=0):
    """Differentiate a Legendre series.

    Only `axis=0` (the only axis a 1-D coefficient array has) is bound,
    same scope note as `polynomial.polyder`.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.legendre.legder: only axis=0 (1-D coefficient "
            "arrays) is implemented in this build"
        )
    return _core._leg_der(c, int(m), float(scl))


def legint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    """Integrate a Legendre series.

    Same `axis=0`-only scope note as `legder` above.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.legendre.legint: only axis=0 (1-D coefficient "
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
    return _core._leg_int(c, int(m), k, float(lbnd), float(scl))


# ─────────────────────────── roots ────────────────────────────

def legfromroots(roots):
    """Generate a Legendre series with the given roots.

    numpy's `_fromroots` balanced-pairing loop is bounded by O(log(number
    of roots)), never by sample/coefficient DATA size -- same carve-out
    `polynomial.polyfromroots` uses, ported here calling `legline`/`legmul`
    instead of `polyline`/`polymul`.
    """
    roots = _ap.asarray(roots)
    n = roots.shape[0] if roots.ndim else 0
    if n == 0:
        return _ap.array([1.0])
    if str(roots.dtype) == "bool":
        # Same `as_series`-rejects-bool wording as `polyfromroots` --
        # verified directly: `L.legfromroots([True, False])` raises the
        # identical `ValueError("Coefficient arrays have no common type")`.
        raise ValueError("Coefficient arrays have no common type")
    rl = _ap.sort(roots)
    is_complex = _ap.iscomplexobj(rl)
    if is_complex:
        p = [_ap.array([complex(-complex(rl[i])), 1.0 + 0j]) for i in range(n)]
    else:
        p = [_ap.array([-float(rl[i]), 1.0]) for i in range(n)]
    cnt = len(p)
    while cnt > 1:
        m, r = divmod(cnt, 2)
        tmp = [legmul(p[i], p[i + m]) for i in range(m)]
        if r:
            tmp[0] = legmul(tmp[0], p[-1])
        p = tmp
        cnt = m
    return p[0]


def legvander(x, deg):
    """Pseudo-Vandermonde matrix of the given degree."""
    return _core._leg_vander(x, int(deg))


def legvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((legvander, legvander), (x, y), deg)


def legvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)


def legcompanion(c):
    """Return the scaled companion matrix of c."""
    return _core._leg_companion(c)


def legroots(c):
    """Compute the roots of a Legendre series.

    Uses the ROTATED companion matrix (`legcompanion(c)[::-1, ::-1]`, per
    numpy's own `legroots` -- "reduces error", its comment says) rather
    than the un-rotated matrix `polyroots` uses for the power basis.

    Unlike `polyroots`, real numpy's own `legroots` does NOT call
    `_to_real_if_imag_zero` -- and, VERIFIED LIVE (2026-08-07,
    `NL.legroots((1,2,3,4))` against real numpy 2.5.1): the result stays
    genuinely COMPLEX (`array([-0.85099543+0.j, -0.11407192+0.j,
    0.51506735+0.j])`), even though every root's imaginary part is exactly
    0 -- `np.linalg.eigvals` does NOT auto-downcast to real either, so this
    is a real, intentional divergence from `polyroots`'s behavior, not an
    oversight in numpy's source. No manual real-cast here, to match.
    """
    c = _ap.asarray(c)
    c = _core._leg_trim(c, 0.0)
    n = c.shape[0]
    if n < 2:
        return _ap.array([], dtype=c.dtype)
    if n == 2:
        return _ap.array([-c[0] / c[1]])
    m = legcompanion(c)
    m = m[::-1, ::-1]
    r = _ap.linalg.eigvals(m)
    r = _ap.sort(r)
    return r


# ─────────────────────────── fitting ────────────────────────────

def legfit(x, y, deg, rcond=None, full=False, w=None):
    """Least-squares fit of a Legendre series to data.

    Same documented scope boundary as `polynomial.polyfit`: only a scalar
    integer `deg`, `full=False`, and `w=None` are implemented.
    """
    if full:
        raise NotImplementedError(
            "anionpy.polynomial.legendre.legfit: full=True (SVD diagnostics) "
            "is not implemented in this build"
        )
    if w is not None:
        raise NotImplementedError(
            "anionpy.polynomial.legendre.legfit: w= (weighted fit) is not "
            "implemented in this build"
        )
    if hasattr(deg, "__len__"):
        raise NotImplementedError(
            "anionpy.polynomial.legendre.legfit: array-valued deg (specific "
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
    van = legvander(x, lmax)
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

def leggauss(deg):
    """Gauss-Legendre quadrature: sample points and weights.

    Ported directly from `legendre.py`'s own body (companion-matrix
    `eigvalsh` first approximation, one Newton refinement step, then
    weight computation via `legval`) -- the loop-free vectorized
    recurrence numpy itself uses, not a `for`-loop over sample points; the
    only "loop" is the fixed handful of vectorized array expressions below,
    matching this file's own "orchestration calling Rust-backed kernels"
    shape, not a per-point Python loop.
    """
    ideg = int(deg)
    if ideg != deg or ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = _ap.array([0] * ideg + [1])
    m = legcompanion(c)
    x = _ap.linalg.eigvalsh(m)

    dy = legval(x, c)
    df = legval(x, legder(c))
    x = x - dy / df

    fm = legval(x, c[1:])
    fm = fm / _ap.abs(fm).max()
    df = df / _ap.abs(df).max()
    w = 1 / (fm * df)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w = w * (2.0 / w.sum())

    return x, w


def legweight(x):
    """Weight function of the Legendre polynomials (identically 1).

    Deliberately does NOT wrap `x` in `_ap.asarray()` first -- real numpy's
    own `legweight` operates on the raw input directly (`x * 0.0 + 1.0`),
    so a plain Python scalar comes back as a plain Python `float` (no
    `.dtype`), and a plain Python `list` raises `TypeError` (lists can't be
    multiplied by a float) rather than silently upgrading to an array.
    Converting first would swallow that error and change scalar output
    from a bare float to a 0-d array -- both confirmed against real numpy
    2.5.1 in this project's venv, not assumed.
    """
    return x * 0.0 + 1.0


# ─────────────────────────── basis conversion ────────────────────────────

def poly2leg(pol):
    """Convert a polynomial (power-series basis) to a Legendre series.

    numpy's own loop bound is `deg = len(pol) - 1` -- the polynomial's OWN
    degree, never sample/coefficient DATA size beyond that (the loop body
    does exactly one `legmulx`+`legadd` Rust-backed call per iteration),
    same "small fixed count" carve-out as `legpow`/`legfromroots` above.
    """
    # `_leg_trim(pol, 0.0)` performs exactly the promotion `pu.as_series`
    # does for a single array (bool rejected, int/float -> float64, complex
    # kept, trailing zeros trimmed) -- the same coercion `legtrim` below
    # exposes publicly, reused here rather than duplicated.
    pol = _core._leg_trim(pol, 0.0)
    deg = pol.shape[0] - 1
    res = _ap.array([0.0])
    # `pol[i:i+1]` (a length-1 1-D slice), not `pol[i]` (a bare scalar):
    # real numpy's own loop body passes the bare scalar `pol[i]` straight
    # into `legadd`, relying on `pu.as_series`'s `ndmin=1` promotion to
    # turn it into a length-1 array internally. This project's own
    # `_leg_add`/`_leg_mulx` Rust bindings reject genuinely 0-d input
    # (`coerce_series_strict` requires `ndim == 1`), so a length-1 SLICE is
    # substituted here -- numerically identical to what numpy's own
    # promotion produces, just constructed on this side of the FFI
    # boundary instead of the other.
    for i in range(deg, -1, -1):
        res = legadd(legmulx(res), pol[i:i + 1])
    return res


def leg2poly(c):
    """Convert a Legendre series to a polynomial (power-series basis).

    numpy's own loop bound is `n = len(c)` -- the Legendre series' OWN
    length, never independent sample/coefficient DATA size, same "small
    fixed count" carve-out as `poly2leg` above. Reuses `polyadd`/`polysub`/
    `polymulx` from the power-series basis module (imported at module top),
    exactly as real numpy's own `leg2poly` does
    (`from .polynomial import polyadd, polymulx, polysub`).
    """
    c = _ap.asarray(c)
    c = legtrim(c, 0.0)
    n = c.shape[0]
    if n < 3:
        return c
    # Length-1 SLICES (`c[-2:-1]`/`c[-1:]`/`c[i-2:i-1]`), not bare scalar
    # indexing (`c[-2]`/`c[-1]`/`c[i-2]`) -- same reasoning as `poly2leg`
    # above: `polyadd`/`polysub`/`polymulx`'s Rust bindings require ndim==1
    # input, while real numpy's own `leg2poly` relies on `as_series`'s
    # `ndmin=1` scalar promotion instead. Numerically identical either way.
    c0 = c[-2:-1]
    c1 = c[-1:]
    for i in range(n - 1, 1, -1):
        tmp = c0
        c0 = _polysub(c[i - 2:i - 1], (c1 * (i - 1)) / i)
        c1 = _polyadd(tmp, (_polymulx(c1) * (2 * i - 1)) / i)
    return _polyadd(c0, _polymulx(c1))


class Legendre(_ABCPolyBase):
    """A Legendre series class (`numpy.polynomial.legendre.Legendre`).

    Assembles the already-implemented `leg*` module functions above
    through the shared `ABCPolyBase` generic machinery in `_polybase.py`
    -- no new arithmetic here, same "assembly, not new arithmetic" shape
    as `polynomial.py`'s `Polynomial` class.
    """

    _add = staticmethod(legadd)
    _sub = staticmethod(legsub)
    _mul = staticmethod(legmul)
    _div = staticmethod(legdiv)
    _pow = staticmethod(legpow)
    _val = staticmethod(legval)
    _int = staticmethod(legint)
    _der = staticmethod(legder)
    _fit = staticmethod(legfit)
    _line = staticmethod(legline)
    _roots = staticmethod(legroots)
    _fromroots = staticmethod(legfromroots)

    domain = _ap.array(legdomain)
    window = _ap.array(legdomain)
    basis_name = "P"


# Ticket #75 (2026-08-08): the `_LEG_COVERAGE_REBIND_*` block and the
# `_rebind_generic_dunders(Legendre)` call formerly here were deleted --
# see `chebyshev.py`'s identical comment for the full rationale. numpy's
# own `Legendre` doesn't bind any of these names in ITS `__dict__` either,
# so inheriting the same defaults from `ABCPolyBase`/`object` is genuine
# parity under `tools/coverage.py`'s numpy-relative check; nothing left to
# rebind. `legmul` does not route through `np.convolve`/z-series, so
# `__mul__`, `__rmul__`, `__pow__`, and `fromroots` are likewise inherited
# unchanged (measured bit-exact: none of them call `_compose_affine`,
# `_mul`/`_pow`/`_fromroots` are called directly).
#
# `cast`/`convert` remain a REAL, separate finding (not a coverage-rebind
# artifact, and not touched by this deletion): both route through the
# shared `_polybase.py::_compose_affine` helper, which performs naive
# Horner substitution (`c0 + x*(c1 + x*(c2 + ...)))`) using the class's own
# `_mul`. This is only mathematically VALID for the power series basis
# (where Horner-on-coefficients IS composition) -- for an orthogonal basis
# like Legendre it computes the WRONG VALUE, not just a non-bit-exact one.
# Measured directly: `Legendre([1.,2.,3.]).convert(kind=Legendre)` (an
# identity conversion, domain==window) returns `[2. 2. 2.]` instead of the
# mathematically-required `[1. 2. 3.]` (real numpy: `[1. 2. 3.]`, confirmed
# identical to input, as expected for x -> x). Real numpy's own `convert`
# instead uses `self(kind.identity(...))`, which dispatches through
# `_val`'s Clenshaw recursion evaluated symbolically with x as an
# `ABCPolyBase` object -- a fundamentally different (and basis-correct)
# algorithm that `_compose_affine` does not implement. This is a bug in
# the SHARED `_compose_affine` helper (not specific to this file), also
# affecting `Laguerre`/`Hermite`/`HermiteE`; `cast`/`convert` remain
# undeclared (absent) for these bases rather than patched over with a
# tolerance.
