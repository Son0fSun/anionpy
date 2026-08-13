"""anionpy.polynomial._polybase -- `ABCPolyBase`
(`numpy.polynomial._polybase.ABCPolyBase`).

This is a pure-Python ASSEMBLY layer, same seam as
`anionpy/polynomial/polynomial.py` (see that file's module docstring for
the fuller precedent this follows): `ABCPolyBase` itself does no per-element
numerical work of its own in real numpy either -- it is a mixin that
implements every dunder/method generically in terms of 12 abstract static
methods (`_add _sub _mul _div _pow _val _int _der _fit _line _roots
_fromroots`) plus 3 abstract properties (`domain`, `window`, `basis_name`)
that each concrete subclass (`Polynomial`, `Chebyshev`, ...) supplies. Only
`Polynomial` (see `anionpy/polynomial/polynomial.py`) supplies those here;
the other five bases are out of scope for this pass (see that file's
existing module docstring).

Two small helpers below (`_compose_affine`) DO contain a Python `for` loop.
Its bound is `len(coef)` -- the polynomial's OWN degree, plugged directly
into the recurrence real numpy's own `polyval` uses internally (Horner's
method: `c0 = c[-1]; c0 = c[-i] + c0*x` for `i in range(2, len(c)+1)`) --
never sample-point/array DATA size. This is the same "small fixed count,
not data size" carve-out `polynomial.py`'s `polypow`/`polyfromroots` already
use and document; see those functions' comments for the fuller argument.
It exists here (rather than reusing `_val`/`polyval` directly) because real
numpy's `convert`/`cast` reach this same recurrence a different way --
by boxing the substituted polynomial in a 0-d object array and letting
Python's `+`/`*` operator protocol dispatch back into `ABCPolyBase`'s own
arithmetic dunders (`self(kind.identity(...))` in numpy's source) -- and
that object-array-of-a-Python-instance trick is not something the
Rust-backed `_poly_val` kernel (a numeric array kernel, not a generic
Python-object one) can be asked to do. `_compose_affine` performs the
IDENTICAL recurrence directly over coefficient arrays instead, which is
provably the same arithmetic in the same order (see this task's report),
not a different algorithm standing in for numpy's.

Utility helpers (`_trimseq`, `_as_series1`, `_trimcoef`, `_getdomain`,
`_mapparms`, `_mapdomain`) are this module's private port of the handful of
`numpy.polynomial.polyutils` functions `ABCPolyBase` itself calls
(`as_series`, `trimcoef`, `getdomain`, `mapparms`, `mapdomain`) -- `polyutils`
is not part of this task's scope as a public module, so these live here,
underscore-prefixed, sized to exactly what `ABCPolyBase` needs (a single
array at a time -- `ABCPolyBase.__init__` always calls `as_series` on a
ONE-element list, `[coef]`/`[domain]`/`[window]`, so the multi-array
`np.common_type(*arrays)` promotion real `as_series` supports for a list of
SEVERAL differently-typed arrays is not needed and not implemented).

`format_float` (real numpy's `dragon4`-backed float formatter, used by
`__repr__`/`__str__`/`__format__`) has a from-scratch, non-dragon4
approximation below (`_format_float`, ticket #34 Part B) matching its
outward contract (precision=8, exp-notation threshold, trim-to-one-
trailing-zero) via Python's own shortest-round-trip float formatting --
see that function's docstring for exactly why it cannot be promised
bit-exact against real dragon4 output. `__str__`/`__format__` now produce
the same basis-notation structure as real numpy (unicode superscripts/
subscripts + `set_default_printstyle`, ported from `numpy.polynomial.
_polybase`'s `_generate_string`/`_format_term`/`_str_term_unicode`/
`_str_term_ascii`, paraphrased not copied) but remain NOT bit-exact and
NOT declared in `anionpy/_state/polynomial.py`, entirely because of
`_format_float`'s float-rendering gap -- see that file's docstring for
the full accounting of what is and is not declared.
"""
from __future__ import annotations

import abc

import anionpy as _ap

__all__ = ["ABCPolyBase", "set_default_printstyle"]


# ─────────────────────────── polyutils-equivalent private helpers ────────

def _trimseq(arr):
    """Port of `numpy.polynomial.polyutils.trimseq`: drop trailing zeros.

    Vectorized (a single `nonzero` call, not a per-element Python loop) --
    same character as the rest of this module's array-level bookkeeping.
    """
    if arr.shape[0] == 0 or arr[-1] != 0:
        return arr
    nz = _ap.nonzero(arr)[0]
    if nz.shape[0] == 0:
        return arr[:1]
    last = int(nz[-1])
    return arr[: last + 1]


def _as_series1(a, trim):
    """Port of `numpy.polynomial.polyutils.as_series` for a SINGLE
    array-like (`ABCPolyBase.__init__`'s only call shape -- see module
    docstring). Coerces to a 1-D array, rejects bool (mirrors the
    `as_series`-family "no common type" split already documented in
    `polynomial.py`'s `polyfromroots`/`polyroots`), and promotes integer
    dtypes to float64 while leaving float/complex dtypes at their own
    precision (matching `np.common_type` applied to a single array).
    """
    arr = _ap.atleast_1d(_ap.asarray(a))
    if arr.size == 0:
        raise ValueError("Coefficient array is empty")
    if arr.ndim != 1:
        raise ValueError("Coefficient array is not 1-d")
    if trim:
        arr = _trimseq(arr)
    dt = str(arr.dtype)
    if dt == "bool":
        raise ValueError("Coefficient arrays have no common type")
    if dt.startswith("int") or dt.startswith("uint"):
        return arr.astype(_ap.float64)
    return arr.astype(arr.dtype)  # copy, dtype unchanged (float*/complex*)


def _trimcoef(c, tol=0):
    """Port of `numpy.polynomial.polyutils.trimcoef`."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    carr = _as_series1(c, trim=True)
    idx = _ap.nonzero(_ap.abs(carr) > tol)[0]
    if idx.shape[0] == 0:
        return carr[:1] * 0
    last = int(idx[-1])
    return carr[: last + 1].astype(carr.dtype)


def _getdomain(x):
    """Port of `numpy.polynomial.polyutils.getdomain`."""
    xarr = _as_series1(x, trim=False)
    if _ap.iscomplexobj(xarr):
        r = _ap.real(xarr)
        i = _ap.imag(xarr)
        return _ap.array([
            complex(float(r.min()), float(i.min())),
            complex(float(r.max()), float(i.max())),
        ])
    return _ap.array([float(xarr.min()), float(xarr.max())])


def _mapparms(old, new):
    """Port of `numpy.polynomial.polyutils.mapparms` -- plain scalar
    arithmetic on the two 2-element domain/window sequences, no array work
    at all (matches real numpy's own implementation exactly).
    """
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl = newlen / oldlen
    return off, scl


def _mapdomain(x, old, new):
    """Port of `numpy.polynomial.polyutils.mapdomain`."""
    if not isinstance(x, (int, float, complex)):
        x = _ap.asanyarray(x)
    off, scl = _mapparms(old, new)
    return off + scl * x


def _compose_affine(add_fn, mul_fn, line_fn, coef, line_off, line_scl):
    """Substitute the degree-<=1 polynomial `line_fn(line_off, line_scl)`
    for `x` in the polynomial `coef`, via Horner's method over coefficient
    arrays.

    See the module docstring for why this exists (real numpy reaches the
    same arithmetic via an object-array composition trick that the
    Rust-backed `_val` kernel here cannot be handed) and why its loop bound
    (`len(coef)`, the polynomial's own degree) is not a "data size" loop.
    `line_fn` is the class's own `_line` (`polyline`) -- the same
    constructor `ABCPolyBase.identity()` uses -- so the substituted "x" is
    built identically here and there.
    """
    n = coef.shape[0]
    line = line_fn(line_off, line_scl)
    c0 = coef[n - 1 : n]
    for i in range(2, n + 1):
        idx = n - i
        c0 = add_fn(coef[idx : idx + 1], mul_fn(c0, line))
    return c0


def _format_float(x, parens=False):
    """Ticket #34 Part B: an approximation of real numpy's
    `numpy.polynomial.polyutils.format_float` (2.5.1) for a SINGLE
    already-computed Python/anionpy scalar -- pure string formatting of a
    value, not array arithmetic, same "Python is dispatch, not math" seam
    every other pure-formatting helper in this module already sits on.

    Real `format_float` is backed by `numpy._core.multiarray.
    dragon4_positional`/`dragon4_scientific` -- calling either would
    violate this project's "never call real numpy to produce a value"
    rule, so this is a from-scratch reimplementation of the OUTWARD
    CONTRACT (shortest round-tripping decimal string, capped at
    `precision=8` fractional/significant digits, trailing zeros trimmed
    to one, switching to `m.mmmmmmmme[+-]NN` scientific form when
    `abs(x) >= 1.e8 or abs(x) < 1.e-4` -- numpy's default printoptions:
    precision=8, floatmode='maxprec', sign='-'), built on Python's own
    `repr`/`f"{x:.8f}"`, which is ALSO a shortest-round-tripping
    algorithm (Python floats have used one since 3.1) -- NOT a port of
    dragon4's actual digit-generation algorithm.

    NOT declared bit-exact, NOT covered by a differential test claiming
    exactness -- see docs/TICKET-34-DUNDERS-CORRECTION-2026-08-08.md's
    Part B section for the measured match rate and the concrete reason
    bit-exactness cannot be promised: two independently-implemented
    shortest-round-trip algorithms are not guaranteed to pick the same
    string when more than one string of the minimal length round-trips
    to the same float, and dragon4's precision-capping rounding step
    (for values whose shortest round-trip form needs MORE than 8
    fractional digits) is not guaranteed to round identically to
    Python's own `.8f` rounding in the final displayed digit.
    """
    import math

    if isinstance(x, complex):
        return str(x)
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "nan"
    if math.isinf(xf):
        # Matches real numpy's `format_float`: it returns the *printoption*
        # `infstr` (default "inf") verbatim for ANY infinite value -- sign
        # included -- because `dragon4_*` is never invoked for inf/nan; the
        # sign of `-inf` is silently dropped by real numpy itself. Confirmed
        # via `inspect.getsource(numpy.polynomial.polyutils.format_float)`
        # and `numpy.get_printoptions()['infstr'] == 'inf'` (measurement
        # only, not shipped code). We match this quirk rather than "fix"
        # it, since the goal is byte-identical output, not a nicer numpy.
        return "inf"

    precision = 8
    a = abs(xf)
    exp_format = xf != 0 and (a >= 1.0e8 or a < 10 ** (-(precision - 1) // 2))

    if exp_format:
        mantissa, _, exponent = f"{xf:.{precision}e}".partition("e")
        if "." in mantissa:
            mantissa = mantissa.rstrip("0")
            if mantissa.endswith("."):
                mantissa += "0"
        exp_sign = exponent[0]
        exp_digits = exponent[1:].lstrip("0") or "0"
        if len(exp_digits) < 2:
            exp_digits = exp_digits.zfill(2)
        s = f"{mantissa}e{exp_sign}{exp_digits}"
        return f"({s})" if parens else s

    s = f"{xf:.{precision}f}"
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s += "0"
    return s


# ─────────────────────────── ABCPolyBase ────────────────────────────

class ABCPolyBase(abc.ABC):
    """Port of `numpy.polynomial._polybase.ABCPolyBase`.

    See the module docstring for the overall shape and this file's scope.
    Method bodies below are a direct, line-by-line translation of real
    numpy's own `_polybase.py` (2.5.1) onto anionpy's array API, substituting
    this module's private `_as_series1`/`_trimcoef`/`_getdomain`/
    `_mapparms`/`_mapdomain` for `polyutils`'s versions and anionpy's own
    `_ap.all`/`_ap.array`/etc. for numpy's -- not a reinterpretation.
    """

    __hash__ = None

    __array_ufunc__ = None

    maxpower = 100

    # Used by `_str_term_unicode` (below, and by `Polynomial`'s own
    # override) to render term powers/basis-function indices in unicode
    # superscript/subscript digits -- ticket #34 Part B, matching real
    # numpy's `ABCPolyBase._superscript_mapping`/`_subscript_mapping`.
    # `__str__`/`__format__` built on these are functional and structurally
    # matched to real numpy but NOT bit-exact (see module docstring/
    # `_format_float`'s docstring) so this mapping is not itself declared
    # in `anionpy/_state/polynomial.py` -- it's an internal formatting
    # detail, not a public surface item.
    _superscript_mapping = str.maketrans({
        "0": "⁰", "1": "¹", "2": "²", "3": "³",
        "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷",
        "8": "⁸", "9": "⁹",
    })
    _subscript_mapping = str.maketrans({
        "0": "₀", "1": "₁", "2": "₂", "3": "₃",
        "4": "₄", "5": "₅", "6": "₆", "7": "₇",
        "8": "₈", "9": "₉",
    })

    # Real numpy defaults unicode printing on except on Windows (`os.name
    # == 'nt'`) -- this project has no Windows target, so this is
    # unconditionally `True`, matching real numpy's behavior on every
    # platform this project actually runs on. `set_default_printstyle`
    # (module-level function, bottom of this file) flips it at the CLASS
    # level, same mechanism as real numpy.
    _use_unicode = True

    @property
    def symbol(self):
        return self._symbol

    @property
    @abc.abstractmethod
    def domain(self):
        pass

    @property
    @abc.abstractmethod
    def window(self):
        pass

    @property
    @abc.abstractmethod
    def basis_name(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def _add(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _sub(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _mul(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _div(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _pow(c, pow, maxpower=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def _val(x, c):
        pass

    @staticmethod
    @abc.abstractmethod
    def _int(c, m, k, lbnd, scl):
        pass

    @staticmethod
    @abc.abstractmethod
    def _der(c, m, scl):
        pass

    @staticmethod
    @abc.abstractmethod
    def _fit(x, y, deg, rcond, full):
        pass

    @staticmethod
    @abc.abstractmethod
    def _line(off, scl):
        pass

    @staticmethod
    @abc.abstractmethod
    def _roots(c):
        pass

    @staticmethod
    @abc.abstractmethod
    def _fromroots(r):
        pass

    def has_samecoef(self, other):
        return (
            len(self.coef) == len(other.coef)
            and _ap.all(self.coef == other.coef)
        )

    def has_samedomain(self, other):
        return _ap.all(self.domain == other.domain)

    def has_samewindow(self, other):
        return _ap.all(self.window == other.window)

    def has_sametype(self, other):
        return isinstance(other, self.__class__)

    def _get_coefficients(self, other):
        if isinstance(other, ABCPolyBase):
            if not isinstance(other, self.__class__):
                raise TypeError("Polynomial types differ")
            elif not _ap.all(self.domain == other.domain):
                raise TypeError("Domains differ")
            elif not _ap.all(self.window == other.window):
                raise TypeError("Windows differ")
            elif self.symbol != other.symbol:
                raise ValueError("Polynomial symbols differ")
            return other.coef
        return other

    def __init__(self, coef, domain=None, window=None, symbol="x"):
        self.coef = _as_series1(coef, trim=False)

        if domain is not None:
            domain = _as_series1(domain, trim=False)
            if len(domain) != 2:
                raise ValueError("Domain has wrong number of elements.")
            self.domain = domain

        if window is not None:
            window = _as_series1(window, trim=False)
            if len(window) != 2:
                raise ValueError("Window has wrong number of elements.")
            self.window = window

        try:
            if not symbol.isidentifier():
                raise ValueError(
                    "Symbol string must be a valid Python identifier"
                )
        except AttributeError:
            raise TypeError("Symbol must be a non-empty string")

        self._symbol = symbol

    # -- formatting (structurally matched to real numpy, ticket #34 Part B;
    # NOT bit-exact and NOT declared -- see module docstring and
    # `_format_float`'s docstring for exactly why) --

    def __repr__(self):
        # `repr(self.coef)` on an anionpy ndarray renders as
        # `"array([1., 2., 3.])"`, same shape as real numpy's own
        # `ndarray.__repr__` -- stripping the leading `"array("` (6 chars)
        # and trailing `")"` (1 char) is real numpy's own `ABCPolyBase.
        # __repr__` technique (`repr(self.coef)[6:-1]`), directly portable
        # since both reprs share that exact prefix/suffix shape. This
        # replaces the previous plain `self.coef!r`, which left the
        # `array(...)` wrapper in (the "bare-vs-array(...)" gap this
        # ticket's brief flagged).
        coef = repr(self.coef)[6:-1]
        domain = repr(self.domain)[6:-1]
        window = repr(self.window)[6:-1]
        name = self.__class__.__name__
        return (
            f"{name}({coef}, domain={domain}, window={window}, "
            f"symbol='{self.symbol}')"
        )

    def __str__(self):
        if self._use_unicode:
            return self._generate_string(self._str_term_unicode)
        return self._generate_string(self._str_term_ascii)

    def _generate_string(self, term_method):
        """Port of real numpy's `ABCPolyBase._generate_string` (paraphrased,
        not copied) -- assembles the full `"1.0 + 2.0*x + ..."`-shaped
        string, one term at a time, via `term_method` for basis-specific
        power/subscript notation and `_format_float` for each coefficient.
        Line-wrapping (real numpy wraps at `linewidth`, default 75, reading
        `np.get_printoptions()`) is NOT ported: this project has no
        anionpy-level printoptions surface to read a `linewidth` from, and
        wrapping is a cosmetic nicety outside this ticket's "basis notation
        + set_default_printstyle" scope -- long polynomials render on one
        line here. This is a disclosed, deliberate scope cut, not silent.
        """
        out = _format_float(self.coef[0])

        off, scale = self.mapparms()
        scaled_symbol, needs_parens = self._format_term(off, scale)
        if needs_parens:
            scaled_symbol = "(" + scaled_symbol + ")"

        for i, coef in enumerate(self.coef[1:]):
            out += " "
            power = str(i + 1)
            try:
                if coef >= 0:
                    next_term = "+ " + _format_float(coef, parens=True)
                else:
                    next_term = "- " + _format_float(-coef, parens=True)
            except TypeError:
                next_term = f"+ {coef}"
            next_term += term_method(power, scaled_symbol)
            out += next_term
        return out

    def _format_term(self, off, scale):
        """Port of real numpy's `ABCPolyBase._format_term`: renders the
        (possibly domain/window-affine-mapped) symbol argument shared by
        every term, e.g. plain `x` when `off == 0 and scale == 1`, or
        `(2.0 + 3.0x)` when the domain/window mapping is non-trivial.
        """
        if off == 0 and scale == 1:
            return self.symbol, False
        elif scale == 1:
            return f"{_format_float(off)} + {self.symbol}", True
        elif off == 0:
            return f"{_format_float(scale)}{self.symbol}", True
        else:
            return (
                f"{_format_float(off)} + {_format_float(scale)}{self.symbol}",
                True,
            )

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        """Port of real numpy's generic `ABCPolyBase._str_term_unicode`:
        `·<basis_name><subscript i>(<arg_str>)`, e.g. `·T₂(x)` for
        Chebyshev. `Polynomial` overrides this (see `polynomial.py`) to
        match real numpy's own `Polynomial`-specific override (plain
        `·x²`, no basis-name/subscript/parens).
        """
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis_name, or override "
                "_str_term_unicode(cls, i, arg_str)"
            )
        return f"·{cls.basis_name}{i.translate(cls._subscript_mapping)}({arg_str})"

    @classmethod
    def _str_term_ascii(cls, i, arg_str):
        """Port of real numpy's generic `ABCPolyBase._str_term_ascii`:
        ` <basis_name>_<i>(<arg_str>)`, e.g. ` T_2(x)` for Chebyshev.
        `Polynomial` overrides this (see `polynomial.py`), same reasoning
        as `_str_term_unicode` above.
        """
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis_name, or override "
                "_str_term_ascii(cls, i, arg_str)"
            )
        return f" {cls.basis_name}_{i}({arg_str})"

    def __format__(self, fmt_str):
        if fmt_str == "":
            return self.__str__()
        if fmt_str not in ("ascii", "unicode"):
            raise ValueError(
                f"Unsupported format string '{fmt_str}' passed to "
                f"{self.__class__}.__format__. Valid options are "
                f"'ascii' and 'unicode'"
            )
        # `__format__`'s explicit `fmt_str` supersedes the class-level
        # `_use_unicode` default (matching real numpy's own documented
        # behavior: "Formatting supersedes all class/package-level
        # defaults").
        term_method = (
            self._str_term_unicode if fmt_str == "unicode" else self._str_term_ascii
        )
        return self._generate_string(term_method)

    # -- pickle and copy --

    def __getstate__(self):
        ret = self.__dict__.copy()
        ret["coef"] = self.coef.copy()
        ret["domain"] = self.domain.copy()
        ret["window"] = self.window.copy()
        ret["symbol"] = self.symbol
        return ret

    def __setstate__(self, dict):
        self.__dict__ = dict

    # -- call --

    def __call__(self, arg):
        arg = _mapdomain(arg, self.domain, self.window)
        return self._val(arg, self.coef)

    def __iter__(self):
        return iter(self.coef)

    def __len__(self):
        return len(self.coef)

    # -- numeric protocol --

    def __neg__(self):
        return self.__class__(-self.coef, self.domain, self.window, self.symbol)

    def __pos__(self):
        return self

    def __add__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            coef = self._add(self.coef, _as_series1(othercoef, trim=False))
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __sub__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            coef = self._sub(self.coef, _as_series1(othercoef, trim=False))
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __mul__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            coef = self._mul(self.coef, _as_series1(othercoef, trim=False))
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __truediv__(self, other):
        import numbers
        if not isinstance(other, numbers.Number) or isinstance(other, bool):
            raise TypeError(
                f"unsupported types for true division: "
                f"'{type(self)}', '{type(other)}'"
            )
        return self.__floordiv__(other)

    def __floordiv__(self, other):
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    def __mod__(self, other):
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[1]

    def __divmod__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            quo, rem = self._div(self.coef, _as_series1(othercoef, trim=False))
        except ZeroDivisionError:
            raise
        except Exception:
            return NotImplemented
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return quo, rem

    def __pow__(self, other):
        coef = self._pow(self.coef, other, maxpower=self.maxpower)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __radd__(self, other):
        try:
            coef = self._add(_as_series1(other, trim=False), self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rsub__(self, other):
        try:
            coef = self._sub(_as_series1(other, trim=False), self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rmul__(self, other):
        try:
            coef = self._mul(_as_series1(other, trim=False), self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rtruediv__(self, other):
        return NotImplemented

    def __rfloordiv__(self, other):
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    def __rmod__(self, other):
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[1]

    def __rdivmod__(self, other):
        try:
            quo, rem = self._div(_as_series1(other, trim=False), self.coef)
        except ZeroDivisionError:
            raise
        except Exception:
            return NotImplemented
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return quo, rem

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and _ap.all(self.domain == other.domain)
            and _ap.all(self.window == other.window)
            and (self.coef.shape == other.coef.shape)
            and _ap.all(self.coef == other.coef)
            and (self.symbol == other.symbol)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    # -- extra methods --

    def copy(self):
        return self.__class__(self.coef, self.domain, self.window, self.symbol)

    def degree(self):
        return len(self) - 1

    def cutdeg(self, deg):
        return self.truncate(deg + 1)

    def trim(self, tol=0):
        coef = _trimcoef(self.coef, tol)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def truncate(self, size):
        isize = int(size)
        if isize != size or isize < 1:
            raise ValueError("size must be a positive integer")
        if isize >= len(self.coef):
            coef = self.coef
        else:
            coef = self.coef[:isize]
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def convert(self, domain=None, kind=None, window=None):
        if kind is None:
            kind = self.__class__
        if domain is None:
            domain = kind.domain
        if window is None:
            window = kind.window
        # Mathematically == `self(kind.identity(domain, window=window,
        # symbol=self.symbol))` -- see module docstring for why this is
        # computed as one composed affine substitution instead.
        ident_off, ident_scl = _mapparms(window, domain)
        call_off, call_scl = _mapparms(self.domain, self.window)
        line_off = call_off + call_scl * ident_off
        line_scl = call_scl * ident_scl
        coef = _compose_affine(self._add, self._mul, self._line, self.coef, line_off, line_scl)
        return self.__class__(coef, domain, window, self.symbol)

    def mapparms(self):
        return _mapparms(self.domain, self.window)

    def integ(self, m=1, k=[], lbnd=None):
        off, scl = self.mapparms()
        if lbnd is None:
            lbnd = 0
        else:
            lbnd = off + scl * lbnd
        coef = self._int(self.coef, m, k, lbnd, 1.0 / scl)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def deriv(self, m=1):
        off, scl = self.mapparms()
        coef = self._der(self.coef, m, scl)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def roots(self):
        roots = self._roots(self.coef)
        return _mapdomain(roots, self.window, self.domain)

    def linspace(self, n=100, domain=None):
        if domain is None:
            domain = self.domain
        x = _ap.linspace(domain[0], domain[1], n)
        y = self(x)
        return x, y

    @classmethod
    def fit(cls, x, y, deg, domain=None, rcond=None, full=False, w=None,
            window=None, symbol="x"):
        if domain is None:
            domain = _getdomain(x)
            if domain[0] == domain[1]:
                domain[0] -= 1
                domain[1] += 1
        elif isinstance(domain, list) and len(domain) == 0:
            domain = cls.domain

        if window is None:
            window = cls.window

        xnew = _mapdomain(x, domain, window)
        res = cls._fit(xnew, y, deg, w=w, rcond=rcond, full=full)
        if full:
            [coef, status] = res
            return (
                cls(coef, domain=domain, window=window, symbol=symbol), status
            )
        else:
            coef = res
            return cls(coef, domain=domain, window=window, symbol=symbol)

    @classmethod
    def fromroots(cls, roots, domain=[], window=None, symbol="x"):
        roots = _as_series1(roots, trim=False)
        if domain is None:
            domain = _getdomain(roots)
        elif isinstance(domain, list) and len(domain) == 0:
            domain = cls.domain

        if window is None:
            window = cls.window

        deg = len(roots)
        off, scl = _mapparms(domain, window)
        rnew = off + scl * roots
        coef = cls._fromroots(rnew) / scl ** deg
        return cls(coef, domain=domain, window=window, symbol=symbol)

    @classmethod
    def identity(cls, domain=None, window=None, symbol="x"):
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        off, scl = _mapparms(window, domain)
        coef = cls._line(off, scl)
        return cls(coef, domain, window, symbol)

    @classmethod
    def basis(cls, deg, domain=None, window=None, symbol="x"):
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        ideg = int(deg)

        if ideg != deg or ideg < 0:
            raise ValueError("deg must be non-negative integer")
        return cls([0] * ideg + [1], domain, window, symbol)

    @classmethod
    def cast(cls, series, domain=None, window=None):
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        return series.convert(domain, cls, window)


# -----------------------------------------------------------------------------
# Ticket #34 correction (2026-08-08, Monday): 195107b's docs/TICKET-34-DUNDERS-
# 2026-08-08.md wrongly classified 120 of the 155 absent class-dunder items
# across the six ABCPolyBase subclasses as "fundamentally unmatchable by
# construction". Direct out-of-corpus measurement (see
# docs/TICKET-34-DUNDERS-CORRECTION-2026-08-08.md) showed 18 of the 20 names
# it lumped in there -- `__lt__ __le__ __gt__ __ge__ __setattr__ __delattr__
# __new__ __dir__ __reduce__ __reduce_ex__ __subclasshook__ __getattribute__
# __sizeof__ __weakref__ __slots__ __abstractmethods__ __static_attributes__
# __init_subclass__` -- already match real numpy bit-for-bit, because
# `ABCPolyBase` never overrides them: they're pure `object`/`abc.ABC`
# mechanics inherited unchanged. (Only `__module__` and `__firstlineno__`
# are genuinely unmatchable by construction -- see Part C in the correction
# doc.) That measurement still stands.
#
# The mechanism this correction used to CREDIT that measurement --
# `_rebind_generic_dunders`, copying each already-correct inherited
# function/descriptor object onto every subclass's own `__dict__` -- is
# superseded by ticket #75 and has been deleted. It rested on the same
# premise as the per-basis `_COVERAGE_REBIND_*` blocks it explicitly
# mirrored ("following the EXACT same precedent"): that `tools/coverage.py`
# checked `attr in vars(cls)` unconditionally for exploded classes. Ticket
# #75 replaced that with a check relative to real numpy's OWN class dict --
# numpy's `Chebyshev`/etc. don't bind these 16 names in their own `__dict__`
# either (verified: none of them are in `vars(numpy.polynomial.Chebyshev)`),
# so inheriting the identical default is genuine parity and `hasattr` alone
# is sufficient. No rebind, generic or per-basis, is needed to make the
# ledger see what was already true.
# -----------------------------------------------------------------------------


def set_default_printstyle(style):
    """Ticket #34 Part B. Port of real numpy's `numpy.polynomial.
    set_default_printstyle` (`numpy.polynomial._polybase.
    set_default_printstyle`, paraphrased not copied): sets the class-level
    `ABCPolyBase._use_unicode` default that `__str__` reads, for every
    basis class at once (they all share this one base class attribute,
    same mechanism as real numpy). `__format__`'s explicit `fmt_str`
    argument always overrides this, matching real numpy's own documented
    behavior ("Formatting supersedes all class/package-level defaults").
    """
    if style not in ("unicode", "ascii"):
        raise ValueError(
            f"Unsupported format string '{style}'. Valid options are "
            f"'ascii' and 'unicode'"
        )
    ABCPolyBase._use_unicode = style == "unicode"
