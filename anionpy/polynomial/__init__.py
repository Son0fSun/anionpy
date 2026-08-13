"""anionpy.polynomial -- `numpy.polynomial` namespace.

The power-series basis (`numpy.polynomial.polynomial`) and its `Polynomial`
class (+ the shared `ABCPolyBase` machinery it is built on, see
`anionpy/polynomial/_polybase.py`) are implemented. `Polynomial` itself is
re-exported here as `anionpy.polynomial.Polynomial`, matching
`numpy.polynomial.Polynomial`'s own top-of-namespace re-export.

The Legendre basis (`numpy.polynomial.legendre`), the Laguerre basis
(`numpy.polynomial.laguerre`), the (physicists') Hermite basis
(`numpy.polynomial.hermite`), and the (probabilists') HermiteE basis
(`numpy.polynomial.hermite_e`) are each implemented at the function level
AND at the `ABCPolyBase` subclass level (`Legendre`, `Laguerre`, `Hermite`,
`HermiteE` -- see `anionpy/polynomial/legendre.py`'s, `laguerre.py`'s,
`hermite.py`'s, and `hermite_e.py`'s own module docstrings for exact
per-item declared/undeclared scope, in particular that `convert`/`cast`
are undeclared for all four due to the shared `_compose_affine` bug
documented in those files).

The (first-kind) Chebyshev basis (`numpy.polynomial.chebyshev`) is also
implemented at the function level and at the `ABCPolyBase` subclass level
(`Chebyshev`) -- see `anionpy/polynomial/chebyshev.py`'s own module
docstring for its exact scope, in particular that `__mul__`, `__rmul__`,
`__pow__`, `fromroots`, `convert`, and `cast` are all undeclared for this
class (routes through the non-bit-exact `chebmul`, and doubly so for
`convert`/`cast` via the shared `_compose_affine` bug).

`numpy.polynomial.polyutils` -- the shared, basis-agnostic helper module
all six bases above are themselves built on -- is implemented at
`anionpy/polynomial/polyutils.py` (see its own module docstring for exact
scope: `format_float` is the one deliberately undeclared name).
"""
from anionpy.polynomial import chebyshev
from anionpy.polynomial import hermite
from anionpy.polynomial import hermite_e
from anionpy.polynomial import laguerre
from anionpy.polynomial import legendre
from anionpy.polynomial import polynomial
from anionpy.polynomial import polyutils
from anionpy.polynomial.chebyshev import Chebyshev
from anionpy.polynomial.hermite import Hermite
from anionpy.polynomial.hermite_e import HermiteE
from anionpy.polynomial.laguerre import Laguerre
from anionpy.polynomial.legendre import Legendre
from anionpy.polynomial.polynomial import Polynomial

__all__ = [
    "polynomial", "Polynomial",
    "legendre", "Legendre",
    "laguerre", "Laguerre",
    "hermite", "Hermite",
    "hermite_e", "HermiteE",
    "chebyshev", "Chebyshev",
    "polyutils",
]
