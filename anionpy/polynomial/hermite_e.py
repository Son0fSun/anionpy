"""anionpy.polynomial.hermite_e -- the (probabilists') HermiteE-series
basis (`numpy.polynomial.hermite_e`).

Sibling module to `anionpy/polynomial/hermite.py` -- STRUCTURAL template
only, NOT a relabeled copy with the 2s deleted. See
`ionp_core::hermite_e`'s module doc comment (`ionp-core/src/hermite_e.rs`)
for the full, verified-against-numpy-source list of arithmetic-grouping
divergences from `hermite.py` (no `x2` precompute in `hermeval`, no
division in `hermemulx`, no trailing `*2` in `hermemul`, direct-assignment
`hermeder`/`hermeint` with no leading `2*`, `hermevander`'s plain-`x`
recurrence, `hermecompanion`'s `-0.5`-free scaling, ...) -- those live in
the Rust core; the divergences that live at THIS layer (Python
orchestration), each verified directly against `hermite_e.py`, are:

- `hermeline(off, scl)` returns `[off, scl]` -- NO `/2`, same shape as
  `legline`'s `[off, scl]`, unlike `hermline`'s `[off, scl/2]`.
- `hermex = np.array([0, 1])` -- an INT array (unlike `hermx`'s
  `[0, 0.5]`, forced to float by the `0.5`).
- `hermecompanion`'s `len(c) == 2` special case is `[[-c[0] / c[1]]]` --
  NO `-0.5` factor at all (contrast `hermcompanion`'s `[[-.5*c0/c1]]`).
- `hermeroots`'s `len(c) == 2` special case is `-c[0] / c[1]` -- same
  no-`-0.5`-factor pattern. More importantly: real numpy's `hermeroots`
  (`hermite_e.py:1505-1516`) does **NOT** call `_to_real_if_imag_zero` (or
  anything equivalent) at all -- verified directly against the literal
  source, which is just `m = hermecompanion(c)[::-1,::-1]; r =
  np.linalg.eigvals(m); r.sort(); return r`, no downcast step. Since
  `np.linalg.eigvals` always upcasts its return to complex
  (`numpy/linalg/_linalg.py`'s own `eigvals`: `return
  w.astype(_complexType(result_t), copy=False)`, unconditional on the
  input's realness), `hermeroots` on real input with degree >= 3 ALWAYS
  returns a complex128 array, even when every root is real -- a genuine
  dtype divergence from `hermroots`, which explicitly downcasts. Do NOT
  port `hermroots`'s `_to_real_if_imag_zero`-equivalent logic here.
- `hermegauss` DOES symmetrize `x`/`w` (`w = (w+w[::-1])/2`, `x =
  (x-x[::-1])/2`) -- same as `hermgauss` -- but its `_normed_hermite_e_n`
  helper differs from `_normed_hermite_n` in THREE places, not just the
  commonly-cited loop-step constant:
    1. the `n == 0` base case is `1/sqrt(sqrt(2*pi))`, NOT
       `hermite.py`'s `1/sqrt(sqrt(pi))` -- an extra factor of 2 inside
       the SAME nested sqrt expression that is easy to miss if only the
       general-loop constants are checked;
    2. the loop step is `c1 * x * sqrt(1./nd)`, NOT `sqrt(2./nd)`;
    3. the FINAL return line is the bare `c0 + c1 * x` -- NO trailing
       `* sqrt(2)` factor at all, unlike `_normed_hermite_n`'s `c0 + c1 *
       x * sqrt(2)`. This third point is NOT explicitly called out as its
       own row in `docs/POLY-BASIS-SOURCE-AUDIT.md`'s normed-n-helper
       comparison table (which only mentions the loop-step constant) --
       flagged here, and in this task's final report, as an audit-doc gap.
  `nd` is decremented at the END of the loop body (`nd = nd - 1.0` is the
  LAST statement), same placement discipline as `hermite.py`'s helper.
  `hermegauss`'s own `df` scale is `sqrt(ideg)` (NOT `sqrt(2*ideg)`), and
  the final weight scale is `sqrt(2*pi)/w.sum()` (NOT `hermgauss`'s bare
  `sqrt(pi)/w.sum()`). All verified directly against
  `hermite_e.py:1519-1621`.
- `hermeweight(x) = exp(-0.5*x**2)`, NOT `hermweight`'s `exp(-x**2)`.
  Domain is `(-inf, inf)`, defined everywhere, same as `hermweight`.
- `herme2poly`'s `len(c) == 2` branch is a BARE `return c` -- it does
  **NOT** double `c[1]` in place, unlike `herm2poly`'s `c[1] *= 2; return
  c`. This is the single sharpest, most implementer-hostile divergence in
  the whole pair: copying `herm2poly`'s doubling line here by mistake
  produces a WRONG VALUE (not a rounding difference), and only at degree
  exactly 2. Pinned as an explicit named case
  (`len2_pinned_no_doubling`) in `tests/differential/hermite_e_cases.py`
  rather than left to chance/random degree sampling, per this task's
  brief. The general (`n > 2`) loop also drops every `*2` present in
  `herm2poly`'s: `c1 * (i - 1)` (not `c1 * (2*(i-1))`), and
  `_polymulx(c1)` (not `_polymulx(c1) * 2`), both in the loop body and
  the final `_polyadd(c0, _polymulx(c1))` combine step.
- `hermefromroots` goes through `hermeline`/`hermemul` via the same
  balanced pairing-tree shape as `hermfromroots`, but since `hermeline(-r,
  1) == [-r, 1]` (no `/2` at all, unlike `hermline`'s `[-r, 0.5]`), the
  literal-`1`-vs-`1.0+0j` signed-zero consideration `hermite.py`'s
  docstring discusses does not arise here in the same shape either --
  still passed as literal `1`/`1.0`, matching `pu._fromroots`'s own
  `line_f(-r, 1)`, for parity with the upstream source.

Every numerical recurrence over coefficient/sample-point DATA lives in
Rust (`ionp_core::hermite_e`, bound via `ionp-py/src/hermite_e.rs`'s
`_herme_*` functions). The bounded loops that DO appear below
(`hermefromroots`'s balanced-pairing tree, `hermepow`'s repeated multiply,
`poly2herme`/`herme2poly`'s degree-descending accumulation, `hermegauss`'s
`_normed_hermite_e_n` helper) are bounded by a small, fixed count
(O(log(number of roots)), the polynomial degree/power, or the requested
HermiteE degree respectively), never by array/sample-point DATA size --
same carve-out `hermite.py`/`legendre.py`/`laguerre.py`/`polynomial.py`
document.

SCOPE (deliberate, matching `hermite.py`'s own documented boundary): only
the 24 non-N-D items are implemented. Explicitly NOT implemented in this
pass: `hermeval2d`, `hermeval3d`, `hermevalnd`, `hermegrid2d`,
`hermegrid3d`, `hermevander2d`, `hermevander3d` (composition helpers,
time-boxed out, not a measured decline).

The `HermiteE(ABCPolyBase)` class is implemented at the bottom of this
file. `hermemul` does not route through `np.convolve`/z-series, so
`__mul__`/`__rmul__`/`__pow__`/`fromroots` are declared exact for this
class per this task's measured, per-class sweep. `convert`/`cast` are NOT
declared -- see `legendre.py`'s equivalent note and the rebind-loop
comment at the bottom of this file: the shared `_polybase.py::
_compose_affine` helper is only mathematically valid for the power-series
basis and produces outright wrong values (not merely non-bit-exact ones)
for orthogonal bases like this one.

`hermefit`'s vector-`deg`, `full=True`, and `w=` are NOT implemented, same
documented boundary as `hermfit`/`lagfit`/`legfit`/`polyfit`.
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
    "hermedomain", "hermezero", "hermeone", "hermex",
    "hermeline", "hermefromroots",
    "hermeadd", "hermesub", "hermemulx", "hermemul", "hermediv", "hermepow",
    "hermeder", "hermeint",
    "hermeval",
    "hermeval2d", "hermeval3d", "hermevalnd",
    "hermegrid2d", "hermegrid3d",
    "hermevander", "hermevander2d", "hermevander3d",
    "hermefit",
    "hermecompanion", "hermeroots",
    "hermegauss", "hermeweight",
    "hermetrim",
    "herme2poly", "poly2herme",
    "HermiteE",
]

# ─────────────────────────── constants ────────────────────────────
hermedomain = _core._herme_domain()
hermezero = _core._herme_zero()
hermeone = _core._herme_one()
hermex = _core._herme_x()


def hermeline(off, scl):
    """Return the coefficients of the HermiteE series for ``off + scl*x``.

    `[off, scl]` -- NO `/2`, built directly via `anionpy.array` (not the
    Rust `_herme_line` binding) for the same integer-dtype-preservation
    reason `hermline`/`legline` document.
    """
    if scl != 0:
        return _ap.array([off, scl])
    else:
        return _ap.array([off])


def hermetrim(c, tol=0):
    """Remove trailing coefficients with absolute value <= tol."""
    return _core._herme_trim(c, float(tol))


# ─────────────────────────── evaluation ────────────────────────────

def hermeval(x, c, tensor=True):
    """Evaluate a HermiteE series at points x.

    Only plain 1-D `c` is bound (`tensor`'s value is then irrelevant,
    matching numpy), same scope note as `hermite.hermval`.
    """
    return _core._herme_val(x, c)


def _hermeval_nd(x, c, tensor=True):
    """PRIVATE, multi-D-`c`-capable companion to `hermeval` above -- Task
    #34, see `polyutils.py`'s module docstring. Line-for-line transcription
    of real numpy 2.5.1's `hermeval` (probabilists') Clenshaw-recursion
    body.
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
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * x
    return c0 + c1 * x


def hermeval2d(x, y, c):
    """Evaluate a 2-D HermiteE series at points (x, y)."""
    return _pu._valnd(_hermeval_nd, c, x, y)


def hermeval3d(x, y, z, c):
    """Evaluate a 3-D HermiteE series at points (x, y, z)."""
    return _pu._valnd(_hermeval_nd, c, x, y, z)


def hermevalnd(pts, c):
    """Evaluate an N-D HermiteE series at points."""
    return _pu._valnd(_hermeval_nd, c, *pts)


def hermegrid2d(x, y, c):
    """Evaluate a 2-D HermiteE series on the Cartesian product of x and y."""
    return _pu._gridnd(_hermeval_nd, c, x, y)


def hermegrid3d(x, y, z, c):
    """Evaluate a 3-D HermiteE series on the Cartesian product of x, y, z."""
    return _pu._gridnd(_hermeval_nd, c, x, y, z)


# ─────────────────────────── arithmetic ────────────────────────────

def hermeadd(c1, c2):
    """Add one HermiteE series to another."""
    return _core._herme_add(c1, c2)


def hermesub(c1, c2):
    """Subtract one HermiteE series from another."""
    return _core._herme_sub(c1, c2)


def hermemulx(c):
    """Multiply a HermiteE series by x."""
    return _core._herme_mulx(c)


def hermemul(c1, c2):
    """Multiply one HermiteE series by another (with reprojection)."""
    return _core._herme_mul(c1, c2)


def hermediv(c1, c2):
    """Divide one HermiteE series by another, returning quotient and
    remainder (both reprojected onto the HermiteE basis)."""
    return _core._herme_div(c1, c2)


def hermepow(c, pow, maxpower=16):
    """Raise a HermiteE series to a power.

    Same "loop bound is `pow` (default-capped at 16 via `maxpower`), never
    array/sample-point data size" carve-out as `hermpow`/`lagpow`/`legpow`,
    calling the Rust-backed `hermemul` repeatedly.
    """
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    if maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    if power == 0:
        # dtype-preserving power==0 branch, same as hermpow/lagpow/legpow.
        c0 = _core._herme_trim(c, 0.0)
        one = _ap.ones(1, dtype=c0.dtype)
        return one
    if power == 1:
        return _core._herme_trim(c, 0.0)
    prd = c
    for _ in range(2, power + 1):
        prd = hermemul(prd, c)
    return prd


# ─────────────────────────── calculus ────────────────────────────

def hermeder(c, m=1, scl=1, axis=0):
    """Differentiate a HermiteE series.

    Only `axis=0` is bound, same scope note as `hermite.hermder`.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.hermite_e.hermeder: only axis=0 (1-D "
            "coefficient arrays) is implemented in this build"
        )
    return _core._herme_der(c, int(m), float(scl))


def hermeint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    """Integrate a HermiteE series.

    Same `axis=0`-only scope note as `hermeder` above.
    """
    if axis != 0:
        raise NotImplementedError(
            "anionpy.polynomial.hermite_e.hermeint: only axis=0 (1-D "
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
    # it -- same rationale as `hermite.hermint`'s identical comment.
    kpairs = [(complex(v).real, complex(v).imag) for v in k]
    return _core._herme_int(c, int(m), kpairs, float(lbnd), float(scl))


# ─────────────────────────── roots ────────────────────────────

def hermefromroots(roots):
    """Generate a HermiteE series with the given roots.

    Same O(log(number of roots))-bounded balanced-pairing loop as
    `hermite.hermfromroots`, calling `hermeline`/`hermemul`.
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
        p = [hermeline(complex(-complex(rl[i])), 1) for i in range(n)]
    else:
        p = [hermeline(-float(rl[i]), 1.0) for i in range(n)]
    cnt = len(p)
    while cnt > 1:
        m, r = divmod(cnt, 2)
        tmp = [hermemul(p[i], p[i + m]) for i in range(m)]
        if r:
            tmp[0] = hermemul(tmp[0], p[-1])
        p = tmp
        cnt = m
    return p[0]


def hermevander(x, deg):
    """Pseudo-Vandermonde matrix of the given degree."""
    return _core._herme_vander(x, int(deg))


def hermevander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((hermevander, hermevander), (x, y), deg)


def hermevander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees."""
    return _pu._vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)


def hermecompanion(c):
    """Return the (scaled) companion matrix of c."""
    return _core._herme_companion(c)


def hermeroots(c):
    """Compute the roots of a HermiteE series.

    Uses the ROTATED companion matrix (`hermecompanion(c)[::-1, ::-1]`,
    same "reduces error" comment as `hermroots`/`lagroots`/`legroots`).
    UNLIKE `hermroots`, real numpy's `hermeroots`
    (`hermite_e.py:1505-1516`) does **NOT** call `_to_real_if_imag_zero`
    or anything equivalent -- verified directly against the literal
    source: `m = hermecompanion(c)[::-1,::-1]; r = np.linalg.eigvals(m);
    r.sort(); return r`, no downcast step at all. Since
    `np.linalg.eigvals` always upcasts its return to complex128
    regardless of the input's realness, this function's `len(c) > 2`
    branch ALWAYS returns a complex128 array -- no real-downcast logic is
    ported here, unlike `hermroots`'s reimplementation of
    `_to_real_if_imag_zero`.
    """
    c = _ap.asarray(c)
    c = _core._herme_trim(c, 0.0)
    n = c.shape[0]
    if n < 2:
        return _ap.array([], dtype=c.dtype)
    if n == 2:
        # len(c) == 2 special case: -c0/c1 -- NO -0.5 factor, unlike
        # hermroots's -.5*c0/c1.
        return _ap.array([-c[0] / c[1]])
    m = hermecompanion(c)
    m = m[::-1, ::-1]
    r = _ap.linalg.eigvals(m)
    r = _ap.sort(r)
    return r


# ─────────────────────────── fitting ────────────────────────────

def hermefit(x, y, deg, rcond=None, full=False, w=None):
    """Least-squares fit of a HermiteE series to data.

    Same documented scope boundary as `hermite.hermfit`: only a scalar
    integer `deg`, `full=False`, and `w=None` are implemented. Same
    `pu._fit(hermevander, ...)` shape as real numpy's own `hermefit`
    (`hermite_e.py:1487`), replayed manually here exactly like `hermfit`
    replays `pu._fit(hermvander, ...)`.
    """
    if full:
        raise NotImplementedError(
            "anionpy.polynomial.hermite_e.hermefit: full=True (SVD "
            "diagnostics) is not implemented in this build"
        )
    if w is not None:
        raise NotImplementedError(
            "anionpy.polynomial.hermite_e.hermefit: w= (weighted fit) is "
            "not implemented in this build"
        )
    if hasattr(deg, "__len__"):
        raise NotImplementedError(
            "anionpy.polynomial.hermite_e.hermefit: array-valued deg "
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
    van = hermevander(x, lmax)
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

def _normed_hermite_e_n(x, n):
    """Evaluate a normalized HermiteE polynomial, ported almost verbatim
    from `hermite_e.py:1554-1575`. Loop is bounded by `n` (the requested
    HermiteE degree), never by `x`'s own size -- same carve-out as
    `hermefromroots`'s pairing tree. `nd` is decremented at the END of the
    loop body, same placement as `_normed_hermite_n`.

    THREE divergences from `_normed_hermite_n`, not just the commonly-cited
    loop-step constant -- see this module's docstring:
      1. `n == 0` base case is `1/sqrt(sqrt(2*pi))`, NOT `1/sqrt(sqrt(pi))`.
      2. loop step uses `sqrt(1./nd)`, NOT `sqrt(2./nd)`.
      3. final return is the bare `c0 + c1*x` -- NO trailing `*sqrt(2)`.
    """
    if n == 0:
        return _ap.full(x.shape, 1.0 / _ap.sqrt(_ap.sqrt(2.0 * _ap.pi)))

    c0 = 0.0
    c1 = 1.0 / _ap.sqrt(_ap.sqrt(2.0 * _ap.pi))
    nd = float(n)
    for _ in range(n - 1):
        tmp = c0
        c0 = -c1 * _ap.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * _ap.sqrt(1.0 / nd)
        nd = nd - 1.0
    return c0 + c1 * x


def hermegauss(deg):
    """Gauss-HermiteE quadrature: sample points and weights.

    Ported directly from `hermite_e.py`'s own body (companion-matrix
    `eigvalsh` first approximation, one Newton refinement step via
    `_normed_hermite_e_n`, then weight computation). Symmetrizes `x`/`w`
    (`w = (w+w[::-1])/2`, `x = (x-x[::-1])/2`), same as `hermgauss`. `df`'s
    scale is `sqrt(ideg)`, NOT `hermgauss`'s `sqrt(2*ideg)`; the final
    weight scale is `sqrt(2*pi)/w.sum()`, NOT `hermgauss`'s bare
    `sqrt(pi)/w.sum()`. Both verified directly against
    `numpy/polynomial/hermite_e.py`'s `hermegauss`.
    """
    ideg = int(deg)
    if ideg != deg or ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = _ap.array([0] * ideg + [1], dtype=_ap.float64)
    m = hermecompanion(c)
    x = _ap.linalg.eigvalsh(m)

    dy = _normed_hermite_e_n(x, ideg)
    df = _normed_hermite_e_n(x, ideg - 1) * _ap.sqrt(ideg)
    x = x - dy / df

    fm = _normed_hermite_e_n(x, ideg - 1)
    fm = fm / _ap.abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w = w * (_ap.sqrt(2.0 * _ap.pi) / w.sum())

    return x, w


def hermeweight(x):
    """Weight function of the HermiteE polynomials: ``exp(-0.5*x**2)``.

    Domain is `(-inf, inf)`, defined everywhere, unlike `lagweight`'s
    `[0, inf)`.
    """
    return _ap.exp(-0.5 * (x**2))


# ─────────────────────────── basis conversion ────────────────────────────

def poly2herme(pol):
    """Convert a polynomial (power-series basis) to a HermiteE series.

    Same "loop bound is `deg = len(pol) - 1`, never sample/coefficient
    DATA size" carve-out as `hermite.poly2herm`, calling
    `hermemulx`/`hermeadd` instead of `hermmulx`/`hermadd`.
    """
    pol = _core._herme_trim(pol, 0.0)
    deg = pol.shape[0] - 1
    res = _ap.array([0.0])
    for i in range(deg, -1, -1):
        res = hermeadd(hermemulx(res), pol[i:i + 1])
    return res


def herme2poly(c):
    """Convert a HermiteE series to a polynomial (power-series basis).

    Same "loop bound is `n = len(c)`, never independent sample/coefficient
    DATA size" carve-out as `hermite.herm2poly`, but note the `n == 2`
    special case is a BARE `return c` -- it does NOT double `c[1]`, unlike
    `herm2poly`'s `c[1] *= 2; return c`. See this module's docstring for
    why that is the sharpest divergence in the whole pair, verified
    directly against `numpy/polynomial/hermite_e.py:143-196`. The general
    loop also drops every `*2` present in `herm2poly`'s: `c1 * (i-1)` (not
    `c1 * (2*(i-1))`), `_polymulx(c1)` (not `_polymulx(c1) * 2`), in both
    the loop body and the final combine step.
    """
    c = _ap.asarray(c)
    c = hermetrim(c, 0.0)
    n = c.shape[0]
    if n == 1:
        return c
    if n == 2:
        return c
    c0 = c[-2:-1]
    c1 = c[-1:]
    for i in range(n - 1, 1, -1):
        tmp = c0
        c0 = _polysub(c[i - 2:i - 1], c1 * (i - 1))
        c1 = _polyadd(tmp, _polymulx(c1))
    return _polyadd(c0, _polymulx(c1))


class HermiteE(_ABCPolyBase):
    """A (probabilists') HermiteE series class
    (`numpy.polynomial.hermite_e.HermiteE`).

    Assembles the already-implemented `herme*` module functions above
    through the shared `ABCPolyBase` generic machinery in `_polybase.py`.
    """

    _add = staticmethod(hermeadd)
    _sub = staticmethod(hermesub)
    _mul = staticmethod(hermemul)
    _div = staticmethod(hermediv)
    _pow = staticmethod(hermepow)
    _val = staticmethod(hermeval)
    _int = staticmethod(hermeint)
    _der = staticmethod(hermeder)
    _fit = staticmethod(hermefit)
    _line = staticmethod(hermeline)
    _roots = staticmethod(hermeroots)
    _fromroots = staticmethod(hermefromroots)

    domain = _ap.array(hermedomain)
    window = _ap.array(hermedomain)
    basis_name = "He"


# Ticket #75 (2026-08-08): the `_HERMEE_COVERAGE_REBIND_*` block and the
# `_rebind_generic_dunders(HermiteE)` call formerly here were deleted --
# see `chebyshev.py`'s identical comment for the full rationale. numpy's
# own `HermiteE` doesn't bind any of these names in ITS `__dict__` either,
# so inheriting the same defaults from `ABCPolyBase`/`object` is genuine
# parity under `tools/coverage.py`'s numpy-relative check; nothing left to
# rebind.
