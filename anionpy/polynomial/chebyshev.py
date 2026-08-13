"""anionpy.polynomial.chebyshev -- the (first-kind) Chebyshev-series basis
(`numpy.polynomial.chebyshev`).

Sibling module to `anionpy/polynomial/hermite_e.py` -- STRUCTURAL template
only, NOT a relabeled copy. See `ionp_core::chebyshev`'s module doc comment
(`ionp-core/src/chebyshev.rs`) for the full, verified-against-numpy-source
list of arithmetic-grouping divergences from every sibling basis (the
z-series-vs-direct-recurrence split, `chebval`'s no-coefficient-on-`c1`
Clenshaw combine step, `chebmulx`'s direct recurrence, `chebder`'s
order-dependent in-place accumulation, `chebcompanion`'s flat non-cumulative
`scl` vector, ...) -- those live in the Rust core; the divergences that live
at THIS layer (Python orchestration), each verified directly against
`numpy/polynomial/chebyshev.py` (numpy 2.5.1), are:

- `chebline(off, scl)` returns `[off, scl]` -- NO `/2`, same shape as
  `hermeline`/`legline`.
- `chebx = np.array([0, 1])` -- an INT array, same as `hermex`.
- `chebroots`'s `len(c) == 2` special case is `-c[0]/c[1]` -- NO `-0.5`
  factor. Its general branch does **NOT** call any real-downcast helper
  (verified directly: `m = chebcompanion(c)[::-1,::-1]; r =
  np.linalg.eigvals(m); r.sort(); return r`) -- same no-downcast pattern as
  `hermeroots`. `np.linalg.eigvals` always upcasts to complex128
  unconditionally, so `chebroots` on real input with degree >= 3 ALWAYS
  returns complex128, even when every root happens to be real.
- `chebfromroots(roots)` is `pu._fromroots(chebline, chebmul, roots)` -- same
  balanced-pairing-tree shape as `hermefromroots`, built on `chebline`/
  `chebmul` (both Rust-backed).
- `chebweight(x) = 1./(sqrt(1+x)*sqrt(1-x))` -- TWO SEPARATE `sqrt` calls
  multiplied together, NOT `sqrt(1-x**2)` (mathematically equal, but not the
  literal op sequence real numpy executes -- ported as the two-sqrt product
  to preserve numpy's exact rounding).
- `chebgauss(deg)` is CLOSED-FORM, not eigensolver-based (the sharpest
  divergence from every sibling `*gauss`, which all go through a companion-
  matrix `eigvalsh` + Newton-refinement scheme): `x = cos(pi *
  arange(1, 2*ideg, 2) / (2.0*ideg))`, `w = ones(ideg) * (pi/ideg)` --
  verified directly against `chebyshev.py`'s literal source, no
  `chebcompanion`/`eigvalsh`/`_normed_*_n` helper involved at all.
- `chebpts1(npts)`: `x = 0.5*pi/_npts * arange(-_npts+1, _npts+1, 2); return
  sin(x)`. Requires `_npts = int(npts)` to equal `npts` exactly and be
  `>= 1`.
- `chebpts2(npts)`: `x = linspace(-pi, 0, _npts); return cos(x)`. Requires
  `_npts = int(npts)` to equal `npts` exactly and be `>= 2`.
- `chebinterpolate(func, deg, args=())`: `order = deg+1; xcheb =
  chebpts1(order); yfunc = func(xcheb, *args); m = chebvander(xcheb, deg); c
  = dot(m.T, yfunc); c[0] /= order; c[1:] /= 0.5*order`. `np.dot` is ported
  as `m.T @ yfunc` (anionpy's `matmul`/`@`, confirmed live to work for the
  basic positional 2-D-by-1-D case; `_ap.dot` itself does not exist in this
  build). `deg` must be a non-negative integer scalar, matching real numpy's
  own `deg.ndim > 0 or deg.dtype.kind not in 'iu' or deg.size == 0` guard.
- `cheb2poly(c)`: `n < 3` bare `return c` (covers BOTH `n==1` and `n==2` --
  unlike `herme2poly`'s dedicated `n==2` no-doubling special case, Chebyshev
  has no separate `n==2` branch at all). The GENERAL loop DOES include a
  `*2` factor -- `c1 = polyadd(tmp, polymulx(c1) * 2)` -- unlike
  `herme2poly`'s general loop, which has NO `*2` anywhere. The FINAL combine
  step (`return polyadd(c0, polymulx(c1))`), by contrast, has NO `*2` --
  only the in-loop `c1` update carries the doubling factor. This is a
  genuinely basis-specific trap: "does this basis's poly-conversion double"
  must be checked per-function against the literal source, not assumed one
  way for all bases based on `herme2poly`'s precedent either way.
- `poly2cheb(pol)`: `res = 0` (Python int, not an array) initially; `res =
  chebadd(chebmulx(res), pol[i])` each iteration -- same shape as
  `poly2herme`, calling `chebmulx`/`chebadd` (Rust-backed) instead.

Every numerical recurrence over coefficient/sample-point DATA lives in Rust
(`ionp_core::chebyshev`, bound via `ionp-py/src/chebyshev.rs`'s `_cheb_*`
functions), including `chebpow`'s ENTIRE z-series loop (unlike
`hermepow`/`legpow`/`lagpow`'s Python-level repeated-multiply loop):
Chebyshev's `chebpow` must stay in z-series form across every iteration of
its internal loop rather than round-tripping through c-series form each
time (see `ionp_core::chebyshev::chebpow`'s doc comment), so the whole
bounded (by `power`, never by array/data size) loop is a single Rust
function call from this layer. The bounded loops that DO appear below
(`chebfromroots`'s balanced-pairing tree, `poly2cheb`/`cheb2poly`'s
degree-descending accumulation) are bounded by a small, fixed count
(O(log(number of roots)) or the polynomial degree respectively), never by
array/sample-point DATA size -- same carve-out every sibling basis module
documents.

SCOPE (deliberate, matching the sibling bases' own documented boundary):
only the 27 non-N-D items are implemented -- `chebdomain`, `chebzero`,
`chebone`, `chebx`, `chebline`, `chebadd`, `chebsub`, `chebmulx`, `chebmul`,
`chebdiv`, `chebpow`, `chebder`, `chebint`, `chebval`, `chebvander`,
`chebfit`, `chebtrim`, `chebroots`, `chebcompanion`, `chebfromroots`,
`cheb2poly`, `poly2cheb`, `chebweight`, `chebgauss`, `chebpts1`, `chebpts2`,
`chebinterpolate`. Explicitly NOT implemented in this pass: `chebval2d`,
`chebval3d`, `chebvalnd`, `chebgrid2d`, `chebgrid3d`, `chebvander2d`,
`chebvander3d` (composition helpers, time-boxed out, not a measured
decline). The `Chebyshev(ABCPolyBase)` class is implemented at the bottom of
this file, same "assembly, not new arithmetic" shape as `polynomial.py`'s
`Polynomial` -- see that class body's comment block and this task's report
for the measured per-item verdicts. `__mul__`/`__pow__`/`fromroots` are
NOT declared exact for THIS class specifically, unlike `Legendre`/
`Laguerre`/`Hermite`/`HermiteE`: they route through `chebmul`, which is
not bit-exact -- see this module's own `chebmul`/`chebfromroots` REVOKED
entries in `anionpy/_state/polynomial.py`. `convert`/`cast` are NOT
declared exact for ANY of the five new classes (this one doubly so): the
shared `_polybase.py::_compose_affine` helper both basis-multiplies
through the class's own (here non-exact) `_mul`, AND is independently
only mathematically valid for the power-series basis in the first place
-- see `legendre.py`'s module docstring for the measured wrong-value
finding common to all four orthogonal bases.

`chebfit`'s vector-`deg`, `full=True`, and `w=` are NOT implemented, same
documented boundary as `hermefit`/`hermfit`/`lagfit`/`legfit`/`polyfit`.
"""
from __future__ import annotations

import anionpy as _ap
from anionpy import _anionpy as _core
from anionpy.polynomial import polyutils as _pu
from anionpy.polynomial.polynomial import polyadd as _polyadd
from anionpy.polynomial.polynomial import polymulx as _polymulx
from anionpy.polynomial.polynomial import polysub as _polysub
from anionpy.polynomial._polybase import ABCPolyBase as _ABCPolyBase
from anionpy.polynomial._polybase import _mapdomain as _mapdomain

__all__ = [
    "chebdomain", "chebzero", "chebone", "chebx",
    "chebline", "chebfromroots",
    "chebadd", "chebsub", "chebmulx", "chebmul", "chebdiv", "chebpow",
    "chebder", "chebint",
    "chebval",
    "chebval2d", "chebval3d", "chebvalnd",
    "chebgrid2d", "chebgrid3d",
    "chebvander", "chebvander2d", "chebvander3d",
    "chebfit",
    "chebcompanion", "chebroots",
    "chebgauss", "chebweight",
    "chebpts1", "chebpts2", "chebinterpolate",
    "chebtrim",
    "cheb2poly", "poly2cheb",
    "Chebyshev",
]

# ─────────────────────────── constants ────────────────────────────
chebdomain = _core._cheb_domain()
chebzero = _core._cheb_zero()
chebone = _core._cheb_one()
chebx = _core._cheb_x()


def chebline(off, scl):
    """Return the coefficients of the Chebyshev series for ``off + scl*x``.

    `[off, scl]` -- NO `/2`, built directly via `anionpy.array` (not the
    Rust `_cheb_line` binding) for the same integer-dtype-preservation
    reason `hermeline`/`legline` document.
    """
    if scl != 0:
        return _ap.array([off, scl])
    else:
        return _ap.array([off])


def chebtrim(c, tol=0):
    """Remove trailing coefficients with absolute value <= tol."""
    return _core._cheb_trim(c, float(tol))


# ─────────────────────────── evaluation ────────────────────────────

def chebval(x, c, tensor=True):
    """Evaluate a Chebyshev series at points x.

    Only plain 1-D `c` is bound (`tensor`'s value is then irrelevant,
    matching numpy), same scope note as `hermeval`.
    """
    return _core._cheb_val(x, c)


def _chebval_nd(x, c, tensor=True):
    """PRIVATE, multi-D-`c`-capable companion to `chebval` above -- Task
    #34, see `polyutils.py`'s module docstring for why this is separate
    from the public 1-D-only `chebval`. Line-for-line transcription of
    real numpy 2.5.1's `chebval` Clenshaw-recursion body.
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
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, c.shape[0] + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x


def chebval2d(x, y, c):
    """Evaluate a 2-D Chebyshev series at points (x, y)."""
    return _pu._valnd(_chebval_nd, c, x, y)


def chebval3d(x, y, z, c):
    """Evaluate a 3-D Chebyshev series at points (x, y, z)."""
    return _pu._valnd(_chebval_nd, c, x, y, z)


def chebvalnd(pts, c):
    """Evaluate an N-D Chebyshev series at points."""
    return _pu._valnd(_chebval_nd, c, *pts)


def chebgrid2d(x, y, c):
    """Evaluate a 2-D Chebyshev series on the Cartesian product of x and y."""
    return _pu._gridnd(_chebval_nd, c, x, y)


def chebgrid3d(x, y, z, c):
    """Evaluate a 3-D Chebyshev series on the Cartesian product of x, y, z."""
    return _pu._gridnd(_chebval_nd, c, x, y, z)


# ─────────────────────────── arithmetic ────────────────────────────

def chebadd(c1, c2):
    """Add one Chebyshev series to another."""
    return _core._cheb_add(c1, c2)


def chebsub(c1, c2):
    """Subtract one Chebyshev series from another."""
    return _core._cheb_sub(c1, c2)


def chebmulx(c):
    """Multiply a Chebyshev series by x."""
    return _core._cheb_mulx(c)


def chebmul(c1, c2):
    """Multiply one Chebyshev series by another (z-series based, with
    reprojection)."""
    return _core._cheb_mul(c1, c2)


def chebdiv(c1, c2):
    """Divide one Chebyshev series by another, returning quotient and
    remainder (both reprojected onto the Chebyshev basis)."""
    return _core._cheb_div(c1, c2)


def chebpow(c, pow, maxpower=16):
    """Raise a Chebyshev series to a power.

    Same "loop bound is `pow` (default-capped at 16 via `maxpower`), never
    array/sample-point data size" carve-out as `hermepow`/`legpow`/
    `lagpow`, but UNLIKE those siblings the entire z-series repeated-
    convolution loop runs inside the single `_cheb_pow` Rust call rather
    than a Python-level loop calling `chebmul` repeatedly -- z-series form
    must be preserved across iterations, not round-tripped through
    c-series each time (see `ionp_core::chebyshev::chebpow`'s doc
    comment). No trailing trim: real numpy's own `chebpow` does not call
    `pu.trimseq` on its result either.
    """
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    if maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    return _core._cheb_pow(c, power)


# ─────────────────────────── calculus ────────────────────────────

def chebder(c, m=1, scl=1, axis=0):
    """Differentiate a Chebyshev series.

    Only `axis=0` is bound, same scope note as `hermeder`.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.chebyshev.chebder: only axis=0 (1-D "
            "coefficient arrays) is implemented in this build"
        )
    return _core._cheb_der(c, int(m), float(scl))


def chebint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    """Integrate a Chebyshev series.

    Same `axis=0`-only scope note as `chebder` above.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.chebyshev.chebint: only axis=0 (1-D "
            "coefficient arrays) is implemented in this build"
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
    # it -- same rationale as `hermeint`'s identical comment.
    kpairs = [(complex(v).real, complex(v).imag) for v in k]
    return _core._cheb_int(c, int(m), kpairs, float(lbnd), float(scl))


# ─────────────────────────── roots ────────────────────────────

def chebfromroots(roots):
    """Generate a Chebyshev series with the given roots.

    Same O(log(number of roots))-bounded balanced-pairing loop as
    `hermefromroots`, calling `chebline`/`chebmul`.
    """
    roots = _ap.asarray(roots)
    n = roots.shape[0] if roots.ndim else 0
    if n == 0:
        return _ap.array([1.0])
    if str(roots.dtype) == "bool":
        raise ValueError("Coefficient arrays have no common type")
    rl = _ap.sort(roots)
    is_complex = _ap.iscomplexobj(rl)
    if is_complex:
        p = [chebline(complex(-complex(rl[i])), 1) for i in range(n)]
    else:
        p = [chebline(-float(rl[i]), 1.0) for i in range(n)]
    cnt = len(p)
    while cnt > 1:
        m, r = divmod(cnt, 2)
        tmp = [chebmul(p[i], p[i + m]) for i in range(m)]
        if r:
            tmp[0] = chebmul(tmp[0], p[-1])
        p = tmp
        cnt = m
    return p[0]


def chebvander(x, deg):
    """Pseudo-Vandermonde matrix of the given degree."""
    return _core._cheb_vander(x, int(deg))


def chebvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((chebvander, chebvander), (x, y), deg)


def chebvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((chebvander, chebvander, chebvander), (x, y, z), deg)


def chebcompanion(c):
    """Return the (scaled) companion matrix of c."""
    return _core._cheb_companion(c)


def chebroots(c):
    """Compute the roots of a Chebyshev series.

    Uses the ROTATED companion matrix (`chebcompanion(c)[::-1, ::-1]`,
    same "reduces error" comment as `hermeroots`/`hermroots`/`lagroots`/
    `legroots`). Real numpy's `chebroots` does NOT call any real-downcast
    helper -- verified directly: `m = chebcompanion(c)[::-1,::-1]; r =
    np.linalg.eigvals(m); r.sort(); return r`, no downcast step at all.
    Since `np.linalg.eigvals` always upcasts its return to complex128
    regardless of the input's realness, this function's `len(c) > 2`
    branch ALWAYS returns a complex128 array -- no real-downcast logic is
    ported here.
    """
    c = _ap.asarray(c)
    c = _core._cheb_trim(c, 0.0)
    n = c.shape[0]
    if n < 2:
        return _ap.array([], dtype=c.dtype)
    if n == 2:
        # len(c) == 2 special case: -c0/c1 -- NO -0.5 factor.
        return _ap.array([-c[0] / c[1]])
    m = chebcompanion(c)
    m = m[::-1, ::-1]
    r = _ap.linalg.eigvals(m)
    r = _ap.sort(r)
    return r


# ─────────────────────────── fitting ────────────────────────────

def chebfit(x, y, deg, rcond=None, full=False, w=None):
    """Least-squares fit of a Chebyshev series to data.

    Same documented scope boundary as `hermefit`: only a scalar integer
    `deg`, `full=False`, and `w=None` are implemented. Same
    `pu._fit(chebvander, ...)` shape as real numpy's own `chebfit`,
    replayed manually here exactly like `hermefit` replays
    `pu._fit(hermevander, ...)`.
    """
    if full:
        raise NotImplementedError(
            "anionpy.polynomial.chebyshev.chebfit: full=True (SVD "
            "diagnostics) is not implemented in this build"
        )
    if w is not None:
        raise NotImplementedError(
            "anionpy.polynomial.chebyshev.chebfit: w= (weighted fit) is "
            "not implemented in this build"
        )
    if hasattr(deg, "__len__"):
        raise NotImplementedError(
            "anionpy.polynomial.chebyshev.chebfit: array-valued deg "
            "(specific term selection) is not implemented in this build; "
            "pass a single int"
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
    van = chebvander(x, lmax)
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


# ─────────────────────────── quadrature / points ────────────────────────────

def chebgauss(deg):
    """Gauss-Chebyshev quadrature: sample points and weights.

    CLOSED-FORM -- unlike every sibling basis's `*gauss` (companion-matrix
    `eigvalsh` + Newton refinement), Chebyshev's is a direct formula,
    verified directly against `numpy/polynomial/chebyshev.py`'s literal
    source: `x = cos(pi * arange(1, 2*ideg, 2) / (2.0*ideg))`, `w =
    ones(ideg) * (pi/ideg)`. No `chebcompanion`/eigensolver/Newton-
    refinement/symmetrization step at all.
    """
    ideg = int(deg)
    if ideg != deg or ideg <= 0:
        raise ValueError("deg must be a positive integer")

    x = _ap.cos(_ap.pi * _ap.arange(1, 2 * ideg, 2) / (2.0 * ideg))
    w = _ap.ones(ideg) * (_ap.pi / ideg)

    return x, w


def chebweight(x):
    """Weight function of the Chebyshev polynomials: ``1/(sqrt(1+x) *
    sqrt(1-x))``.

    TWO SEPARATE sqrt calls multiplied together (not `sqrt(1-x**2)`) --
    preserves numpy's exact literal op sequence. Domain is `[-1, 1]`.
    """
    return 1.0 / (_ap.sqrt(1.0 + x) * _ap.sqrt(1.0 - x))


def chebpts1(npts):
    """Chebyshev points of the first kind: ``sin(0.5*pi/npts *
    arange(-npts+1, npts+1, 2))``.
    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 1:
        raise ValueError("npts must be >= 1")
    x = 0.5 * _ap.pi / _npts * _ap.arange(-_npts + 1, _npts + 1, 2)
    return _ap.sin(x)


def chebpts2(npts):
    """Chebyshev points of the second kind: ``cos(linspace(-pi, 0,
    npts))``.
    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 2:
        raise ValueError("npts must be >= 2")
    x = _ap.linspace(-_ap.pi, 0, _npts)
    return _ap.cos(x)


def chebinterpolate(func, deg, args=()):
    """Interpolate a function at the Chebyshev points of the first kind.

    `deg` must be a non-negative integer scalar (matches real numpy's
    `deg.ndim > 0 or deg.dtype.kind not in 'iu' or deg.size == 0` guard).
    `np.dot(m.T, yfunc)` is ported as `m.T @ yfunc` -- `_ap.dot` does not
    exist in this build, but the `@`/`matmul` operator does work correctly
    for this basic 2-D-by-1-D positional case.
    """
    if not isinstance(deg, (int,)) and not (hasattr(deg, "ndim") and deg.ndim == 0):
        # allow plain Python ints and 0-d integer arrays/np-scalars; reject
        # everything else the same way real numpy's `deg.ndim > 0 or
        # deg.dtype.kind not in 'iu' or deg.size == 0` guard does.
        try:
            deg_arr = _ap.asarray(deg)
        except Exception as exc:
            raise TypeError("deg must be an int") from exc
        if deg_arr.ndim > 0 or str(deg_arr.dtype)[0] not in ("i", "u") or deg_arr.size == 0:
            raise TypeError("deg must be an int")
        deg = int(deg_arr)
    deg = int(deg)
    if deg < 0:
        raise ValueError("expected deg >= 0")

    order = deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebvander(xcheb, deg)
    c = m.T @ yfunc
    c = c.copy() if hasattr(c, "copy") else _ap.array(c)
    c[0] = c[0] / order
    c[1:] = c[1:] / (0.5 * order)

    return c


# ─────────────────────────── basis conversion ────────────────────────────

def poly2cheb(pol):
    """Convert a polynomial (power-series basis) to a Chebyshev series.

    Same "loop bound is `deg = len(pol) - 1`, never sample/coefficient
    DATA size" carve-out as `poly2herme`, calling `chebmulx`/`chebadd`
    instead of `hermemulx`/`hermeadd`.
    """
    pol = _core._cheb_trim(pol, 0.0)
    deg = pol.shape[0] - 1
    res = _ap.array([0.0])
    for i in range(deg, -1, -1):
        res = chebadd(chebmulx(res), pol[i:i + 1])
    return res


def cheb2poly(c):
    """Convert a Chebyshev series to a polynomial (power-series basis).

    `n < 3` bare `return c` -- covers BOTH `n==1` and `n==2` (unlike
    `herme2poly`'s dedicated `n==2` special case). The general loop's `c1`
    update DOES carry a `*2` factor (`c1 = polyadd(tmp, polymulx(c1) *
    2)`), but the FINAL combine step does NOT (`return polyadd(c0,
    polymulx(c1))`, no `*2`) -- see this module's docstring for why this
    is a genuinely basis-specific trap, not something to assume based on
    `herme2poly`'s (entirely doubling-free) precedent.
    """
    c = _ap.asarray(c)
    c = chebtrim(c, 0.0)
    n = c.shape[0]
    if n < 3:
        return c
    c0 = c[-2:-1]
    c1 = c[-1:]
    for i in range(n - 1, 1, -1):
        tmp = c0
        c0 = _polysub(c[i - 2:i - 1], c1)
        c1 = _polyadd(tmp, _polymulx(c1) * 2)
    return _polyadd(c0, _polymulx(c1))


# ─────────────────────────── Chebyshev class ────────────────────────────

class Chebyshev(_ABCPolyBase):
    """A (first-kind) Chebyshev series class
    (`numpy.polynomial.chebyshev.Chebyshev`).

    Assembly over the 27 functions above, same shape as
    `polynomial.py`'s `Polynomial` -- `ABCPolyBase` implements every
    dunder/method generically in terms of the 12 static methods and 3
    properties bound below, all already-implemented functions from this
    module. See this task's report for the measured per-item verdict:
    `_mul`/`_pow`/`_fromroots` are `chebmul`/`chebpow`/`chebfromroots`,
    all three of which are z-series-based and route through the same
    unreproducible `np.convolve` summation order documented at
    `chebmul`'s REVOKED entry in `anionpy/_state/polynomial.py` -- so
    `__mul__`, `__rmul__`, `__pow__`, and `fromroots` are NOT declared
    exact for this class, unlike the other four bases. `convert`/`cast`
    (built on `_mul` via `_compose_affine`) are ALSO not declared exact,
    same as every other basis class here -- `_compose_affine` is
    independently only valid for the power-series basis (see
    `legendre.py`'s module docstring for the measured wrong-value
    finding shared across all four orthogonal bases).
    """

    _add = staticmethod(chebadd)
    _sub = staticmethod(chebsub)
    _mul = staticmethod(chebmul)
    _div = staticmethod(chebdiv)
    _pow = staticmethod(chebpow)
    _val = staticmethod(chebval)
    _int = staticmethod(chebint)
    _der = staticmethod(chebder)
    _fit = staticmethod(chebfit)
    _line = staticmethod(chebline)
    _roots = staticmethod(chebroots)
    _fromroots = staticmethod(chebfromroots)

    domain = _ap.array(chebdomain)
    window = _ap.array(chebdomain)
    basis_name = "T"

    @classmethod
    def interpolate(cls, func, deg, domain=None, args=()):
        """Interpolate a function at the Chebyshev points of the first
        kind, scaled/shifted to `domain`.

        Direct port of real numpy's own `Chebyshev.interpolate`: `xfunc =
        lambda x: func(mapdomain(x, cls.window, domain), *args)`, then
        `chebinterpolate(xfunc, deg)`. `func` is caller-supplied Python
        code (evaluated once per Chebyshev point, `deg+1` calls -- a
        small, degree-bounded count, not a data-size loop, same carve-out
        as `chebinterpolate` itself), not a numerical loop this file owns.
        """
        if domain is None:
            domain = cls.domain

        def xfunc(x):
            return func(_mapdomain(x, cls.window, domain), *args)

        coef = chebinterpolate(xfunc, deg)
        return cls(coef, domain=domain)


# Ticket #75 (2026-08-08): the `_CHEB_COVERAGE_REBIND_*` block and the
# `_rebind_generic_dunders(Chebyshev)` call formerly here were deleted.
# Both existed only to satisfy `tools/coverage.py`'s old class-`__dict__`
# presence check by copying already-inherited `ABCPolyBase`/`object`
# methods onto `Chebyshev.__dict__` -- a mutation trap the check no longer
# has. `resolve()` is now numpy-relative: numpy's own `Chebyshev` doesn't
# bind any of these names in ITS `__dict__` either (all the rebound names
# above, and the 16 generic dunders, are inherited on real numpy too), so
# `Chebyshev` inheriting the same defaults from `ABCPolyBase`/`object` is
# genuine parity, not a gap -- nothing left to rebind. `domain`, `window`,
# `basis_name`, `interpolate`, `__mul__`, `__rmul__`, `__pow__`,
# `fromroots` remain own-defined above unchanged (Chebyshev-specific,
# never part of any rebind).
