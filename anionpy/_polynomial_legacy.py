"""numpy's LEGACY polynomial API (`np.poly1d`, `np.polyval`, `np.polyadd`,
`np.polysub`, `np.polyder`, `np.polyint`, `np.polydiv`, `np.poly`,
`np.roots`, `np.polyfit`).

This is NOT `anionpy.polynomial` (the modern chebyshev/legendre/hermite/
power-series package). It is the original, pre-1.4 "highest degree
first" coefficient convention:

    p[0]*x**N + p[1]*x**(N-1) + ... + p[N-1]*x + p[N]

Architecture rule: this file is dispatch/validation/formatting ONLY.
Every actual floating-point/complex arithmetic operation here is either
(a) delegated to a Rust kernel in `ionp-core::poly_legacy` via the
`anionpy._anionpy._poly*_legacy` PyO3 bindings, or (b) composed purely
from *other* already-Rust-backed anionpy primitives (`concatenate`,
`zeros`, array `+`/`-`/`/`, `diag`, 2-D slice assignment,
`anionpy.linalg.eigvals`, `anionpy.linalg.lstsq`, ...) -- exactly the
same style real numpy's OWN legacy implementation uses (its `polyadd`/
`polysub`/`roots`/`polyfit` are themselves just `concatenate`/`zeros`/
`diag`/slice-assignment/`eigvals`/`lstsq` calls, no bespoke loop). See
`ionp/docs/TICKET-72-POLY1D-2026-08-08.md` for the full reasoning.

DECLINED (not implemented here, and not silently omitted):
  - `polymul` (free function) and `poly1d.__mul__`/`__rmul__` /
    `__pow__` for the poly-times-poly case, and the free function
    `convolve`/`correlate`.  All of these need numpy's `convolve`
    summation order, which ticket #45 has NOT reproduced. Declining to
    declare them is treated as a success condition per that ticket, not
    a shortfall of this one.  Scalar multiplication
    (`poly1d * scalar`) does NOT need convolve and IS implemented.
  - `array_str` is not implemented as a standalone top-level function
    (it is not needed by anything below: `poly1d.__str__`'s per-scalar
    `.4g` formatting is self-contained and does not call it).
"""

import warnings

from . import _anionpy as _core
from . import (
    array,
    asarray,
    atleast_1d,
    concatenate,
    diag,
    finfo,
    hstack,
    iscomplex,
    isscalar,
    logical_or,
    mintypecode,
    newaxis,
    nonzero,
    ones,
    outer,
    real,
    imag,
    sort,
    trim_zeros,
    zeros,
    zeros_like,
)
from . import linalg as _linalg
from .polynomial.polynomial import polyvander as _modern_polyvander

__all__ = [
    "poly1d",
    "poly",
    "roots",
    "polyval",
    "polyadd",
    "polysub",
    "polyder",
    "polyint",
    "polydiv",
    "polyfit",
    "RankWarning",
]


class RankWarning(UserWarning):
    """Issued by `polyfit` when the Vandermonde matrix is rank deficient.

    Own class, not borrowed from numpy at runtime -- numpy's own
    `numpy.exceptions.RankWarning` text/class identity is not imported;
    this is a same-named, same-behaved local definition.
    """


# ---------------------------------------------------------------------
# polyval / polyadd / polysub / polyder / polyint / polydiv
# ---------------------------------------------------------------------


def _is_poly1d(x):
    return isinstance(x, poly1d)


def polyval(p, x):
    """Evaluate polynomial `p` (highest degree first) at `x`."""
    if _is_poly1d(p):
        p = p.coeffs
    p = asarray(p)

    if _is_poly1d(x):
        # p(x(t)): composite polynomial. Built purely from poly1d
        # arithmetic (which itself only needs polyadd/scalar-mul, not
        # convolve, EXCEPT when a term's power > 1 requires poly-poly
        # multiply). Implemented via repeated Horner step using
        # poly1d.__mul__ (scalar) / poly1d ** (int) is NOT available
        # here since ** on a poly1d needs polymul (declined). So the
        # poly1d-x case is declined too, consistently with #45.
        raise NotImplementedError(
            "polyval(p, x) with x a poly1d (polynomial composition) "
            "is declined: it needs poly-poly multiplication, which "
            "needs np.convolve's summation order (ticket #45, not yet "
            "reproduced)."
        )

    # Real numpy places NO ndim restriction on `p` at all: `for pv in p:
    # y = y*x + pv` Horners over p's FIRST axis regardless of p.ndim
    # (ticket #76 defect A). Mirrored here exactly, branch for branch:
    #   - p.ndim == 0: `for pv in p` on a 0-d array raises numpy's own
    #     `TypeError: iteration over a 0-d array` -- reproduced by
    #     literally iterating p ourselves (dispatch, not arithmetic);
    #     the `for _ in p: break` below never executes its body.
    #   - p.shape[0] == 0 (any ndim): the loop body never runs, so `y`
    #     stays `zeros_like(x)` -- pre-existing, unchanged branch, now
    #     reached for N-D empty-first-axis `p` too (matches numpy: `np.
    #     polyval(np.ones((0,3)), 1)` is also the zeros_like(x) case).
    #   - p.ndim == 1, shape[0] > 0: existing dtype-specialized Rust
    #     fast path, unchanged (already independently verified exact).
    #   - p.ndim >= 2, shape[0] > 0: new generic Horner-over-first-axis
    #     Rust kernel (`_polyval_legacy_nd`), see `ionp-core/src/
    #     poly_legacy.rs`.
    if p.ndim == 0:
        for _ in p:
            break

    if p.shape[0] == 0:
        # Real numpy: y = zeros_like(x) and the loop body never runs.
        xarr = asarray(x)
        return zeros_like(xarr)

    if p.ndim == 1:
        return _core._polyval_legacy(p, x)
    return _core._polyval_legacy_nd(p, x)


def polyadd(a1, a2):
    """Sum of two polynomials (no trimming, matches real numpy)."""
    truepoly = _is_poly1d(a1) or _is_poly1d(a2)
    a1c = a1.coeffs if _is_poly1d(a1) else atleast_1d(a1)
    a2c = a2.coeffs if _is_poly1d(a2) else atleast_1d(a2)
    diff = len(a2c) - len(a1c)
    if diff == 0:
        val = a1c + a2c
    elif diff > 0:
        zr = zeros(diff, dtype=a1c.dtype)
        val = concatenate((zr, a1c)) + a2c
    else:
        zr = zeros(-diff, dtype=a2c.dtype)
        val = a1c + concatenate((zr, a2c))
    if truepoly:
        val = poly1d(val)
    return val


def polysub(a1, a2):
    """Difference of two polynomials (no trimming, matches real numpy)."""
    truepoly = _is_poly1d(a1) or _is_poly1d(a2)
    a1c = a1.coeffs if _is_poly1d(a1) else atleast_1d(a1)
    a2c = a2.coeffs if _is_poly1d(a2) else atleast_1d(a2)
    diff = len(a2c) - len(a1c)
    if diff == 0:
        val = a1c - a2c
    elif diff > 0:
        zr = zeros(diff, dtype=a1c.dtype)
        val = concatenate((zr, a1c)) - a2c
    else:
        zr = zeros(-diff, dtype=a2c.dtype)
        val = a1c - concatenate((zr, a2c))
    if truepoly:
        val = poly1d(val)
    return val


def polyder(p, m=1):
    """Derivative of order `m` of polynomial `p`."""
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")
    truepoly = _is_poly1d(p)
    pc = p.coeffs if truepoly else asarray(p)
    if pc.ndim == 1:
        val = _core._polyder_legacy(pc, m)
    else:
        # Real numpy's polyder computes `n = len(p) - 1` UNCONDITIONALLY
        # on every call, including the `m == 0` base case (ticket #76
        # defects B/C) -- so `len(pc)` is called here too, purely for
        # its side effect of raising numpy's own `TypeError: len() of
        # unsized object` on a 0-d `pc` (anionpy's array wrapper already
        # raises that exact message for `len()` of a 0-d array; this is
        # dispatch/validation, not arithmetic). For ndim >= 2, the
        # actual N-D slice/broadcast-multiply recursion lives in Rust
        # (`_polyder_legacy_nd`, `ionp-core/src/poly_legacy.rs`).
        len(pc)
        val = _core._polyder_legacy_nd(pc, m)
    if truepoly:
        val = poly1d(val)
    return val


def polyint(p, m=1, k=None):
    """Antiderivative of order `m` of polynomial `p`."""
    m = int(m)
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    if k is None:
        kk = [0.0] * m
    else:
        karr = atleast_1d(k)
        if len(karr) == 1 and m > 1:
            kk = [float(karr[0])] * m
        elif len(karr) < m:
            raise ValueError(
                "k must be a scalar or a rank-1 array of length 1 or >m."
            )
        else:
            kk = [float(v) for v in karr]

    truepoly = _is_poly1d(p)
    pc = p.coeffs if truepoly else asarray(p)

    if pc.ndim == 1:
        if m == 0:
            val = pc
        else:
            val = _core._polyint_legacy(pc, kk)
    else:
        # Real numpy's polyint checks `m == 0` FIRST and returns `p`
        # unchanged with no computation at all in that case -- unlike
        # polyder above, a 0-d `pc` at `m == 0` genuinely succeeds
        # (`np.polyint(np.array(5), 0)` returns a 0-d ndarray, measured
        # directly against numpy 2.5.1; ticket #76 defect B only
        # applies here when `m > 0`). `len(pc)` is called only in that
        # `m > 0` case, purely for its side effect of raising numpy's
        # own `TypeError: len() of unsized object` on a 0-d `pc` --
        # dispatch/validation, not arithmetic. For ndim >= 2 (or `m ==
        # 0` at any ndim), the actual N-D slice/divide/concatenate
        # recursion lives in Rust (`_polyint_legacy_nd`, `ionp-core/
        # src/poly_legacy.rs`).
        if m > 0:
            len(pc)
        val = _core._polyint_legacy_nd(pc, m, kk)
    if truepoly:
        val = poly1d(val)
    return val


def polydiv(u, v):
    """Quotient and remainder of polynomial division `u / v`."""
    truepoly = _is_poly1d(u) or _is_poly1d(v)
    uc = u.coeffs if _is_poly1d(u) else atleast_1d(u)
    vc = v.coeffs if _is_poly1d(v) else atleast_1d(v)
    if uc.ndim != 1 or vc.ndim != 1:
        raise ValueError("u and v must be 1d (or 0d) arrays")
    q, r = _core._polydiv_legacy(uc, vc)
    if truepoly:
        return poly1d(q), poly1d(r)
    return q, r


# ---------------------------------------------------------------------
# poly / roots
# ---------------------------------------------------------------------


def poly(seq_of_zeros):
    """Coefficients of a monic polynomial with the given roots (or the
    characteristic polynomial of a square matrix)."""
    seq = atleast_1d(seq_of_zeros)
    sh = seq.shape

    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        seq = _linalg.eigvals(seq)
    elif len(sh) == 1:
        dt = seq.dtype
        seq = seq.astype(mintypecode(dt.char))
    else:
        raise ValueError("input must be 1d or non-empty square 2d array.")

    if len(seq) == 0:
        return 1.0

    a = _core._poly_from_roots_legacy(seq)

    if a.dtype.kind == "c":
        rr = seq.astype(complex)
        if (sort(rr) == sort(rr.conjugate())).all():
            a = real(a).copy()

    return a


def roots(p):
    """Roots of polynomial `p` (highest degree first)."""
    p = atleast_1d(p)
    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")

    nz = nonzero(p.ravel())[0]
    if len(nz) == 0:
        return array([])

    trailing_zeros = len(p) - int(nz[-1]) - 1
    p = p[int(nz[0]) : int(nz[-1]) + 1]

    if not (p.dtype.kind in "fc"):
        p = p.astype(float)

    n = len(p)
    if n > 1:
        a = diag(ones(n - 2, dtype=p.dtype), -1)
        a[0, :] = -p[1:] / p[0]
        rts = _linalg.eigvals(a)
        rts = _to_real_if_imag_zero(rts)
    else:
        rts = array([])

    rts = hstack((rts, zeros(trailing_zeros, dtype=rts.dtype)))
    return rts


def _to_real_if_imag_zero(z):
    """Local equivalent of `numpy.linalg._linalg._to_real_if_imag_zero`:
    if every imaginary part is exactly zero, return the real part
    (real dtype); otherwise return `z` unchanged."""
    if z.dtype.kind != "c":
        return z
    if (imag(z) == 0).all():
        return real(z)
    return z


# ---------------------------------------------------------------------
# polyfit
# ---------------------------------------------------------------------


def _legacy_vander(x, order):
    """`np.vander(x, order)` (decreasing powers, the legacy default) via
    the already-shipped modern `polyvander` (increasing powers) with its
    columns reversed -- a pure data rearrangement, zero new arithmetic."""
    deg = order - 1
    v = _modern_polyvander(x, deg)
    return v[:, ::-1]


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """Least squares polynomial fit, highest degree first."""
    order = int(deg) + 1
    x = asarray(x) + 0.0
    y = asarray(y) + 0.0

    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    if rcond is None:
        rcond = len(x) * finfo(x.dtype).eps

    lhs = _legacy_vander(x, order)
    rhs = y

    if w is not None:
        w = asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs = lhs * w[:, newaxis]
        if rhs.ndim == 2:
            rhs = rhs * w[:, newaxis]
        else:
            rhs = rhs * w

    scale = ((lhs * lhs).sum(axis=0)) ** 0.5
    lhs = lhs / scale
    c, resids, rank, s = _linalg.lstsq(lhs, rhs, rcond)
    c = (c.T / scale).T

    if rank != order and not full:
        warnings.warn("Polyfit may be poorly conditioned", RankWarning, stacklevel=2)

    if full:
        return c, resids, rank, s, rcond
    elif cov:
        vbase = _linalg.inv(lhs.T @ lhs)
        vbase = vbase / outer(scale, scale)
        if cov == "unscaled":
            fac = 1
        else:
            if len(x) <= order:
                raise ValueError(
                    "the number of data points must exceed order "
                    "to scale the covariance matrix"
                )
            fac = resids / (len(x) - order)
        if y.ndim == 1:
            return c, vbase * fac
        else:
            return c, vbase[:, :, newaxis] * fac
    else:
        return c


# ---------------------------------------------------------------------
# poly1d
# ---------------------------------------------------------------------


import re as _re

_poly_mat = _re.compile(r"\*\*([0-9]*)")


def _raise_power(astr, wrap=70):
    n = 0
    line1 = ""
    line2 = ""
    output = " "
    while True:
        mat = _poly_mat.search(astr, n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + " " * (len(power) - 1)
        toadd1 = " " * (len(partstr) - 1) + power
        if (len(line2) + len(toadd2) > wrap) or (len(line1) + len(toadd1) > wrap):
            output += line1 + "\n" + line2 + "\n "
            line1 = toadd1
            line2 = toadd2
        else:
            line2 += partstr + " " * (len(power) - 1)
            line1 += " " * (len(partstr) - 1) + power
    output += line1 + "\n" + line2
    return output + astr[n:]


class poly1d:
    """A one-dimensional polynomial class (legacy numpy API).

    `poly1d([1, 2, 3])` represents ``1*x**2 + 2*x + 3``.
    """

    __hash__ = None

    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            self._variable = c_or_r._variable
            self._coeffs_ = c_or_r._coeffs_
            if variable is not None:
                self._variable = variable
            return
        if r:
            c_or_r = poly(c_or_r)
        c_or_r = atleast_1d(c_or_r)
        if c_or_r.ndim > 1:
            raise ValueError("Polynomial must be 1d only.")
        c_or_r = trim_zeros(c_or_r, trim="f")
        if len(c_or_r) == 0:
            c_or_r = array([0], dtype=c_or_r.dtype)
        self._coeffs_ = c_or_r
        self._variable = variable if variable is not None else "x"

    # -- properties --------------------------------------------------
    @property
    def coeffs(self):
        return self._coeffs_

    @property
    def variable(self):
        return self._variable

    @property
    def order(self):
        return len(self._coeffs_) - 1

    @property
    def roots(self):
        return roots(self._coeffs_)

    c = coef = coefficients = coeffs
    r = roots
    o = order

    def __array__(self, t=None, copy=None):
        if t:
            return asarray(self.coeffs).astype(t)
        return asarray(self.coeffs)

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return f"poly1d({vals})"

    def __len__(self):
        return self.order

    def __str__(self):
        thestr = "0"
        var = self.variable

        mask = logical_or.accumulate(self.coeffs != 0)
        coeffs = self.coeffs[mask]
        n = len(coeffs) - 1

        def fmt_float(q):
            s = f"{q:.4g}"
            if s.endswith(".0000"):
                s = s[: -len(".0000")]
            return s

        for k in range(len(coeffs)):
            coeff = coeffs[k]
            if not bool(iscomplex(coeff)):
                coefstr = fmt_float(float(real(coeff)))
            elif float(real(coeff)) == 0:
                coefstr = f"{fmt_float(float(imag(coeff)))}j"
            else:
                coefstr = (
                    f"({fmt_float(float(real(coeff)))} + "
                    f"{fmt_float(float(imag(coeff)))}j)"
                )

            power = n - k
            if power == 0:
                if coefstr != "0":
                    newstr = f"{coefstr}"
                elif k == 0:
                    newstr = "0"
                else:
                    newstr = ""
            elif power == 1:
                if coefstr == "0":
                    newstr = ""
                elif coefstr == "b":
                    newstr = var
                else:
                    newstr = f"{coefstr} {var}"
            elif coefstr == "0":
                newstr = ""
            elif coefstr == "b":
                newstr = f"{var}**{power}"
            else:
                newstr = f"{coefstr} {var}**{power}"

            if k > 0:
                if newstr != "":
                    if newstr.startswith("-"):
                        thestr = f"{thestr} - {newstr[1:]}"
                    else:
                        thestr = f"{thestr} + {newstr}"
            else:
                thestr = newstr
        return _raise_power(thestr)

    def __call__(self, val):
        return polyval(self.coeffs, val)

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs * other)
        raise NotImplementedError(
            "poly1d * poly1d (non-scalar) is declined: it needs "
            "polymul/np.convolve's summation order (ticket #45, not "
            "yet reproduced)."
        )

    def __rmul__(self, other):
        if isscalar(other):
            return poly1d(other * self.coeffs)
        raise NotImplementedError(
            "poly1d * poly1d (non-scalar) is declined: it needs "
            "polymul/np.convolve's summation order (ticket #45, not "
            "yet reproduced)."
        )

    def __add__(self, other):
        other = other if _is_poly1d(other) else poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __radd__(self, other):
        other = other if _is_poly1d(other) else poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __pow__(self, val):
        if not isscalar(val) or int(val) != val or val < 0:
            raise ValueError("Power to non-negative integers only.")
        if val > 1:
            raise NotImplementedError(
                "poly1d ** n for n > 1 is declined: it needs "
                "polymul/np.convolve's summation order (ticket #45, "
                "not yet reproduced)."
            )
        if val == 0:
            return poly1d([1])
        return poly1d(self.coeffs)

    def __sub__(self, other):
        other = other if _is_poly1d(other) else poly1d(other)
        return poly1d(polysub(self.coeffs, other.coeffs))

    def __rsub__(self, other):
        other = other if _is_poly1d(other) else poly1d(other)
        return poly1d(polysub(other.coeffs, self.coeffs))

    def __truediv__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs / other)
        other = other if _is_poly1d(other) else poly1d(other)
        return polydiv(self, other)

    def __rtruediv__(self, other):
        if isscalar(other):
            return poly1d(other / self.coeffs)
        other = other if _is_poly1d(other) else poly1d(other)
        return polydiv(other, self)

    def __eq__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        if self.coeffs.shape != other.coeffs.shape:
            return False
        return bool((self.coeffs == other.coeffs).all())

    def __ne__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        return not self.__eq__(other)

    def __getitem__(self, val):
        ind = self.order - val
        if val > self.order or val < 0:
            return asarray(0, dtype=self.coeffs.dtype).item()
        return self.coeffs[ind]

    def __setitem__(self, key, val):
        ind = self.order - key
        if key < 0:
            raise ValueError("Does not support negative powers.")
        if key > self.order:
            zr = zeros(key - self.order, dtype=self.coeffs.dtype)
            self._coeffs_ = concatenate((zr, self.coeffs))
            ind = 0
        self._coeffs_[ind] = val

    def __iter__(self):
        return iter(self.coeffs)

    def integ(self, m=1, k=0):
        return poly1d(polyint(self.coeffs, m=m, k=k))

    def deriv(self, m=1):
        return poly1d(polyder(self.coeffs, m=m))
