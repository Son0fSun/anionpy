"""anionpy.polynomial.hermite -- the (physicists') Hermite-series basis
(`numpy.polynomial.hermite`).

Sibling module to `anionpy/polynomial/laguerre.py` -- STRUCTURAL template
only, not a relabeled copy. See `ionp_core::hermite`'s module doc comment
(`ionp-core/src/hermite.rs`) for the full, verified-against-numpy-source
list of arithmetic-grouping and structural divergences this basis has
(literal-left-operand-first multiply order throughout, `x2 = x*2`
precomputed-and-reused in `hermval`, DIRECT ASSIGNMENT (not accumulation)
in `hermder`/`hermint`, the trailing `hermmulx(c1) * 2` shape in `hermmul`,
a scaled `hermcompanion` with its own scale-vector construction, etc.) --
those live in the Rust core; the divergences that live at THIS layer
(Python orchestration) are:

- `hermline(off, scl)` returns `[off, scl/2]` -- same shape as `legline`'s
  `[off, scl]` structurally (unlike `lagline`'s `[off+scl, -scl]`), but
  with a genuine `/2` on the slope term, not a relabeled `legline`.
- `hermroots`'s `len(c) == 2` special case is `-.5 * c[0] / c[1]` -- NOT
  `lagroots`'s `1 + c[0]/c[1]` and NOT `legroots`'s `-c0/c1`. Real numpy's
  `hermroots` (`hermite.py:1537-1602`) DOES call
  `_to_real_if_imag_zero(r, m)`, same as `lagroots` -- reimplemented here
  with only public `anionpy` array ops for the same reason `lagroots`
  documents (the real helper is a private `numpy.linalg._linalg` symbol
  this build cannot call from shipped code).
- `hermgauss` DOES symmetrize `x`/`w` (`w = (w+w[::-1])/2`,
  `x = (x-x[::-1])/2`) -- UNLIKE `laggauss`, which has no such step -- and
  scales the final weights by `sqrt(pi)/w.sum()`, not a bare
  `w /= w.sum()`. It also needs a private `_normed_hermite_n(x, n)` helper
  (`hermite.py:1605-1643`) ported here almost verbatim: a degree-bounded
  loop (bounded by `n`, the requested Hermite degree -- same "loop bound is
  a small fixed count, never array/sample-point DATA size" carve-out as
  `lagfromroots`'s pairing tree) using anionpy array ops (`_ap.sqrt`,
  `_ap.pi`), NOT raw Python `math` calls, and decrementing its own loop
  counter `nd` at the END of each iteration (`nd = nd - 1.0` is the LAST
  statement in the loop body) -- the OPPOSITE order from `hermval`'s Rust
  Clenshaw loop, which decrements `nd` near the TOP of each iteration
  (`nd -= 1;` before computing `c0`/`c1`). Both orders are individually
  correct for their own recurrences; this note exists because copying one
  loop's decrement placement into the other would silently produce the
  wrong `nd` value on every iteration but the first.
- `hermweight(x) = exp(-x**2)`, not `lagweight`'s `exp(-x)` or
  `legweight`'s `x*0 + 1`. Domain is `(-inf, inf)`, defined everywhere.
- `herm2poly`'s `len(c) == 2` branch DOUBLES `c[1]` in place
  (`c[1] *= 2; return c`) before returning -- a genuine, easy-to-miss
  divergence from `HermiteE`'s sibling `herme2poly`, which does NOT do
  this (verified directly against `hermite.py:141-216`; pinned as an
  explicit named case in `tests/differential/hermite_cases.py` rather than
  left to chance/random degree sampling, per this task's brief).
- `hermfromroots` goes through `hermline`/`hermmul` via the same balanced
  pairing-tree shape as `lagfromroots`, but since `hermline(-r, 1) ==
  [-r, 0.5]` (nothing analogous to `lagline`'s asymmetric `off+scl`/`-scl`
  pair), the literal-`1`-vs-`1.0+0j` signed-zero trap `lagfromroots`
  documents does not reproduce here in the same shape -- `hermline`'s
  `scl/2` has no unary negation of `scl` itself, so passing plain Python
  `1` vs `1.0 + 0j` as `scl` cannot flip a sign the way `lagline`'s `-scl`
  can. Still passed as the literal `1`/`1.0` matching `pu._fromroots`'s own
  `line_f(-r, 1)`, for parity with the upstream source and out of caution,
  not because a divergence was measured.

Every numerical recurrence over coefficient/sample-point DATA lives in
Rust (`ionp_core::hermite`, bound via `ionp-py/src/hermite.rs`'s `_herm_*`
functions). The bounded loops that DO appear below (`hermfromroots`'s
balanced-pairing tree, `hermpow`'s repeated multiply, `poly2herm`/
`herm2poly`'s degree-descending accumulation, `hermgauss`'s
`_normed_hermite_n` helper) are bounded by a small, fixed count (O(log(number
of roots)), the polynomial degree/power, or the requested Hermite degree
respectively), never by array/sample-point DATA size -- same carve-out
`legendre.py`/`laguerre.py`/`polynomial.py` document.

SCOPE (deliberate, matching `laguerre.py`'s own documented boundary): only
the 24 non-N-D items are implemented. Explicitly NOT implemented in this
pass: `hermval2d`, `hermval3d`, `hermvalnd`, `hermgrid2d`, `hermgrid3d`,
`hermvander2d`, `hermvander3d` (composition helpers, time-boxed out, not a
measured decline).

The `Hermite(ABCPolyBase)` class is implemented at the bottom of this
file. `hermmul` does not route through `np.convolve`/z-series, so
`__mul__`/`__rmul__`/`__pow__`/`fromroots` are declared exact for this
class per this task's measured, per-class sweep. `convert`/`cast` are NOT
declared -- see `legendre.py`'s equivalent note and the rebind-loop
comment at the bottom of this file: the shared `_polybase.py::
_compose_affine` helper is only mathematically valid for the power-series
basis and produces outright wrong values (not merely non-bit-exact ones)
for orthogonal bases like this one.

`hermfit`'s vector-`deg`, `full=True`, and `w=` are NOT implemented, same
documented boundary as `lagfit`/`legfit`/`polyfit`.
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
    "hermdomain", "hermzero", "hermone", "hermx",
    "hermline", "hermfromroots",
    "hermadd", "hermsub", "hermmulx", "hermmul", "hermdiv", "hermpow",
    "hermder", "hermint",
    "hermval",
    "hermval2d", "hermval3d", "hermvalnd",
    "hermgrid2d", "hermgrid3d",
    "hermvander", "hermvander2d", "hermvander3d",
    "hermfit",
    "hermcompanion", "hermroots",
    "hermgauss", "hermweight",
    "hermtrim",
    "herm2poly", "poly2herm",
    "Hermite",
]

# ─────────────────────────── constants ────────────────────────────
hermdomain = _core._herm_domain()
hermzero = _core._herm_zero()
hermone = _core._herm_one()
hermx = _core._herm_x()


def hermline(off, scl):
    """Return the coefficients of the Hermite series for ``off + scl*x``.

    `[off, scl/2]` -- built directly via `anionpy.array` (not the Rust
    `_herm_line` binding) for the same integer-dtype-preservation reason
    `lagline`/`legline` document. Note `scl/2` forces a float result
    whenever `scl` is a nonzero Python int, matching real numpy
    (`H.hermline(3, 2).dtype` is `float64`, not `int64` -- verified live).
    """
    if scl != 0:
        return _ap.array([off, scl / 2])
    else:
        return _ap.array([off])


def hermtrim(c, tol=0):
    """Remove trailing coefficients with absolute value <= tol."""
    return _core._herm_trim(c, float(tol))


# ─────────────────────────── evaluation ────────────────────────────

def hermval(x, c, tensor=True):
    """Evaluate a Hermite series at points x.

    Only plain 1-D `c` is bound (`tensor`'s value is then irrelevant,
    matching numpy), same scope note as `laguerre.lagval`.
    """
    return _core._herm_val(x, c)


def _hermval_nd(x, c, tensor=True):
    """PRIVATE, multi-D-`c`-capable companion to `hermval` above -- Task
    #34, see `polyutils.py`'s module docstring. Line-for-line transcription
    of real numpy 2.5.1's `hermval` (physicists') Clenshaw-recursion body.
    """
    c = _ap.array(c, ndmin=1)
    if str(c.dtype) in _pu._INT_LIKE_DTYPES:
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = _ap.asanyarray(x)
    if isinstance(x, _ap.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    x2 = x * 2
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
            c0 = c[-i] - c1 * (2 * (nd - 1))
            c1 = tmp + c1 * x2
    return c0 + c1 * x2


def hermval2d(x, y, c):
    """Evaluate a 2-D Hermite series at points (x, y)."""
    return _pu._valnd(_hermval_nd, c, x, y)


def hermval3d(x, y, z, c):
    """Evaluate a 3-D Hermite series at points (x, y, z)."""
    return _pu._valnd(_hermval_nd, c, x, y, z)


def hermvalnd(pts, c):
    """Evaluate an N-D Hermite series at points."""
    return _pu._valnd(_hermval_nd, c, *pts)


def hermgrid2d(x, y, c):
    """Evaluate a 2-D Hermite series on the Cartesian product of x and y."""
    return _pu._gridnd(_hermval_nd, c, x, y)


def hermgrid3d(x, y, z, c):
    """Evaluate a 3-D Hermite series on the Cartesian product of x, y, z."""
    return _pu._gridnd(_hermval_nd, c, x, y, z)


# ─────────────────────────── arithmetic ────────────────────────────

def hermadd(c1, c2):
    """Add one Hermite series to another."""
    return _core._herm_add(c1, c2)


def hermsub(c1, c2):
    """Subtract one Hermite series from another."""
    return _core._herm_sub(c1, c2)


def hermmulx(c):
    """Multiply a Hermite series by x."""
    return _core._herm_mulx(c)


def hermmul(c1, c2):
    """Multiply one Hermite series by another (with reprojection)."""
    return _core._herm_mul(c1, c2)


def hermdiv(c1, c2):
    """Divide one Hermite series by another, returning quotient and
    remainder (both reprojected onto the Hermite basis)."""
    return _core._herm_div(c1, c2)


def hermpow(c, pow, maxpower=16):
    """Raise a Hermite series to a power.

    Same "loop bound is `pow` (default-capped at 16 via `maxpower`), never
    array/sample-point data size" carve-out as `lagpow`/`legpow`, calling
    the Rust-backed `hermmul` repeatedly.
    """
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    if maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    if power == 0:
        # dtype-preserving power==0 branch, same as lagpow/legpow.
        c0 = _core._herm_trim(c, 0.0)
        one = _ap.ones(1, dtype=c0.dtype)
        return one
    if power == 1:
        return _core._herm_trim(c, 0.0)
    prd = c
    for _ in range(2, power + 1):
        prd = hermmul(prd, c)
    return prd


# ─────────────────────────── calculus ────────────────────────────

def hermder(c, m=1, scl=1, axis=0):
    """Differentiate a Hermite series.

    Only `axis=0` is bound, same scope note as `laguerre.lagder`.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.hermite.hermder: only axis=0 (1-D coefficient "
            "arrays) is implemented in this build"
        )
    return _core._herm_der(c, int(m), float(scl))


def hermint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    """Integrate a Hermite series.

    Same `axis=0`-only scope note as `hermder` above.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.hermite.hermint: only axis=0 (1-D coefficient "
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
    # it -- same rationale as `laguerre.lagint`'s identical comment: a
    # blind cast would silently discard the imaginary part of a genuinely
    # complex `k` whenever `c` is complex too.
    kpairs = [(complex(v).real, complex(v).imag) for v in k]
    return _core._herm_int(c, int(m), kpairs, float(lbnd), float(scl))


# ─────────────────────────── roots ────────────────────────────

def hermfromroots(roots):
    """Generate a Hermite series with the given roots.

    Same O(log(number of roots))-bounded balanced-pairing loop as
    `laguerre.lagfromroots`, calling `hermline`/`hermmul`. See this
    module's docstring for why the literal-`1`-vs-`1.0+0j` signed-zero
    trap `lagfromroots` documents does not reproduce here in the same
    shape (still passed literally, out of caution/parity with
    `pu._fromroots`'s own `line_f(-r, 1)`).
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
        p = [hermline(complex(-complex(rl[i])), 1) for i in range(n)]
    else:
        p = [hermline(-float(rl[i]), 1.0) for i in range(n)]
    cnt = len(p)
    while cnt > 1:
        m, r = divmod(cnt, 2)
        tmp = [hermmul(p[i], p[i + m]) for i in range(m)]
        if r:
            tmp[0] = hermmul(tmp[0], p[-1])
        p = tmp
        cnt = m
    return p[0]


def hermvander(x, deg):
    """Pseudo-Vandermonde matrix of the given degree."""
    return _core._herm_vander(x, int(deg))


def hermvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((hermvander, hermvander), (x, y), deg)


def hermvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((hermvander, hermvander, hermvander), (x, y, z), deg)


def hermcompanion(c):
    """Return the (scaled) companion matrix of c."""
    return _core._herm_companion(c)


def hermroots(c):
    """Compute the roots of a Hermite series.

    Uses the ROTATED companion matrix (`hermcompanion(c)[::-1, ::-1]`,
    same "reduces error" comment as `lagroots`/`legroots`). Real numpy's
    own `hermroots` DOES cast the result to real when possible -- verified
    directly against `numpy/polynomial/hermite.py:1587-1602`:
    ``r = _to_real_if_imag_zero(r, m)`` is called there, same shape as
    `lagroots`. `_to_real_if_imag_zero`'s own documented semantics
    (read directly, `numpy.linalg._linalg`, since it is a private helper
    this build cannot call from shipped code): downcast to real if AND
    ONLY IF the companion matrix `m`'s own dtype is not complex AND every
    returned eigenvalue's imaginary part is EXACTLY `0.0` (not
    epsilon-close) -- reimplemented here with only public `anionpy` array
    ops.
    """
    c = _ap.asarray(c)
    c = _core._herm_trim(c, 0.0)
    n = c.shape[0]
    if n < 2:
        return _ap.array([], dtype=c.dtype)
    if n == 2:
        # len(c) == 2 special case: -.5*c0/c1 -- NOT lagroots's 1+c0/c1,
        # NOT legroots's -c0/c1.
        return _ap.array([-0.5 * c[0] / c[1]])
    m = hermcompanion(c)
    m = m[::-1, ::-1]
    r = _ap.linalg.eigvals(m)
    r = _ap.sort(r)
    mat_is_complex = _ap.iscomplexobj(m)
    if not mat_is_complex and bool(_ap.all(_ap.imag(r) == 0.0)):
        r = _ap.real(r)
    return r


# ─────────────────────────── fitting ────────────────────────────

def hermfit(x, y, deg, rcond=None, full=False, w=None):
    """Least-squares fit of a Hermite series to data.

    Same documented scope boundary as `laguerre.lagfit`: only a scalar
    integer `deg`, `full=False`, and `w=None` are implemented. Same
    `pu._fit(hermvander, ...)` shape as real numpy's own `hermfit`
    (`hermite.py:1487`), replayed manually here exactly like `lagfit`
    replays `pu._fit(lagvander, ...)`.
    """
    if full:
        raise NotImplementedError(
            "anionpy.polynomial.hermite.hermfit: full=True (SVD diagnostics) "
            "is not implemented in this build"
        )
    if w is not None:
        raise NotImplementedError(
            "anionpy.polynomial.hermite.hermfit: w= (weighted fit) is not "
            "implemented in this build"
        )
    if hasattr(deg, "__len__"):
        raise NotImplementedError(
            "anionpy.polynomial.hermite.hermfit: array-valued deg (specific "
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
    van = hermvander(x, lmax)
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

def _normed_hermite_n(x, n):
    """Evaluate a normalized Hermite polynomial, ported almost verbatim
    from `hermite.py:1605-1643`. Loop is bounded by `n` (the requested
    Hermite degree), never by `x`'s own size -- same carve-out as
    `hermfromroots`'s pairing tree. `nd` is decremented at the END of the
    loop body (`nd = nd - 1.0` is the LAST statement), the opposite
    placement from the Rust Clenshaw loop in `herm_eval` -- see this
    module's docstring for why that distinction is load-bearing.
    """
    if n == 0:
        return _ap.full(x.shape, 1.0 / _ap.sqrt(_ap.sqrt(_ap.pi)))

    c0 = 0.0
    c1 = 1.0 / _ap.sqrt(_ap.sqrt(_ap.pi))
    nd = float(n)
    for _ in range(n - 1):
        tmp = c0
        c0 = -c1 * _ap.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * _ap.sqrt(2.0 / nd)
        nd = nd - 1.0
    return c0 + c1 * x * _ap.sqrt(2.0)


def hermgauss(deg):
    """Gauss-Hermite quadrature: sample points and weights.

    Ported directly from `hermite.py`'s own body (companion-matrix
    `eigvalsh` first approximation -- the companion matrix for
    `c = [0]*ideg + [1]` is genuinely symmetric here, same reasoning as
    `laggauss` -- one Newton refinement step via `_normed_hermite_n`, then
    weight computation). UNLIKE `laggauss`, this DOES symmetrize `x`/`w`
    (`w = (w+w[::-1])/2`, `x = (x-x[::-1])/2`) and scales the final
    weights by `sqrt(pi)/w.sum()`, not a bare `w /= w.sum()`. Both
    verified directly against `numpy/polynomial/hermite.py`'s `hermgauss`.
    """
    ideg = int(deg)
    if ideg != deg or ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = _ap.array([0] * ideg + [1], dtype=_ap.float64)
    m = hermcompanion(c)
    x = _ap.linalg.eigvalsh(m)

    dy = _normed_hermite_n(x, ideg)
    df = _normed_hermite_n(x, ideg - 1) * _ap.sqrt(2 * ideg)
    x = x - dy / df

    fm = _normed_hermite_n(x, ideg - 1)
    fm = fm / _ap.abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w = w * (_ap.sqrt(_ap.pi) / w.sum())

    return x, w


def hermweight(x):
    """Weight function of the Hermite polynomials: ``exp(-x**2)``.

    Verified live: `H.hermweight(2.0)` returns `0.01831563889...`, no
    domain exception -- `exp(-x**2)` is defined for all real `x`, and the
    interval of integration is `(-inf, inf)`, unlike `lagweight`'s
    `[0, inf)`.
    """
    return _ap.exp(-(x**2))


# ─────────────────────────── basis conversion ────────────────────────────

def poly2herm(pol):
    """Convert a polynomial (power-series basis) to a Hermite series.

    Same "loop bound is `deg = len(pol) - 1`, never sample/coefficient
    DATA size" carve-out as `laguerre.poly2lag`, calling
    `hermmulx`/`hermadd` instead of `lagmulx`/`lagadd`.
    """
    pol = _core._herm_trim(pol, 0.0)
    deg = pol.shape[0] - 1
    res = _ap.array([0.0])
    for i in range(deg, -1, -1):
        res = hermadd(hermmulx(res), pol[i:i + 1])
    return res


def herm2poly(c):
    """Convert a Hermite series to a polynomial (power-series basis).

    Same "loop bound is `n = len(c)`, never independent sample/coefficient
    DATA size" carve-out as `laguerre.lag2poly`, but note the `n == 2`
    special case DOUBLES `c[1]` in place before returning -- see this
    module's docstring for why that is a genuine, easy-to-miss divergence
    from `HermiteE`'s `herme2poly` (which has no such doubling), verified
    directly against `numpy/polynomial/hermite.py:141-216`, NOT copied
    from `lag2poly`'s or `leg2poly`'s shape (neither has an `n == 2`
    special case at all).
    """
    c = _ap.asarray(c)
    c = hermtrim(c, 0.0)
    n = c.shape[0]
    if n == 1:
        return c
    if n == 2:
        out = _ap.array(c)
        out[1] = out[1] * 2
        return out
    c0 = c[-2:-1]
    c1 = c[-1:]
    for i in range(n - 1, 1, -1):
        tmp = c0
        c0 = _polysub(c[i - 2:i - 1], c1 * (2 * (i - 1)))
        c1 = _polyadd(tmp, _polymulx(c1) * 2)
    return _polyadd(c0, _polymulx(c1) * 2)


class Hermite(_ABCPolyBase):
    """A (physicists') Hermite series class
    (`numpy.polynomial.hermite.Hermite`).

    Assembles the already-implemented `herm*` module functions above
    through the shared `ABCPolyBase` generic machinery in `_polybase.py`.
    """

    _add = staticmethod(hermadd)
    _sub = staticmethod(hermsub)
    _mul = staticmethod(hermmul)
    _div = staticmethod(hermdiv)
    _pow = staticmethod(hermpow)
    _val = staticmethod(hermval)
    _int = staticmethod(hermint)
    _der = staticmethod(hermder)
    _fit = staticmethod(hermfit)
    _line = staticmethod(hermline)
    _roots = staticmethod(hermroots)
    _fromroots = staticmethod(hermfromroots)

    domain = _ap.array(hermdomain)
    window = _ap.array(hermdomain)
    basis_name = "H"


# Ticket #75 (2026-08-08): the `_HERM_COVERAGE_REBIND_*` block and the
# `_rebind_generic_dunders(Hermite)` call formerly here were deleted -- see
# `chebyshev.py`'s identical comment for the full rationale. numpy's own
# `Hermite` doesn't bind any of these names in ITS `__dict__` either, so
# inheriting the same defaults from `ABCPolyBase`/`object` is genuine
# parity under `tools/coverage.py`'s numpy-relative check; nothing left to
# rebind.
