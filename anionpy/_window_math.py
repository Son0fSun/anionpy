"""Pure-composition implementations for a handful of `numpy` top-level
items that need no new Rust: the five window functions (`bartlett`,
`blackman`, `hamming`, `hanning`, `kaiser`), `kaiser`'s dependency `i0`
(modified Bessel function of the first kind, order 0), `angle`, `sinc`,
and `unwrap`.

Every function below is Python-level *composition only* -- it wires
together already-Rust-backed, already-exact `anionpy._anionpy` calls
(`arange`, `where`, `cos`, `sin`, `arctan2`, `mod`, `diff`, `cumsum`,
`concatenate`, `abs`, `sqrt`, `exp`, ...). No arithmetic is performed by
Python: every elementwise operation happens inside one of those Rust
calls. See this task's brief / `anionpy/__init__.py`'s module docstring for
the rule.

`_chbevl` below has a Python `for` loop, but it is a loop over a FIXED,
small list of Chebyshev coefficients (29 or 25 of them -- Horner's-
method-style polynomial evaluation), not a loop over array elements:
each iteration is one vectorised (whole-array) multiply/add pair
performed by `_anionpy`'s ufuncs. This mirrors numpy's own pure-Python
`_chbevl` (`numpy/lib/_function_base_impl.py`) exactly, coefficient for
coefficient -- see the module-level `_I0A`/`_I0B` tables, copied
verbatim from numpy 2.5.1's `_i0A`/`_i0B`.

Formulas transcribed verbatim from numpy 2.5.1's
`numpy/lib/_function_base_impl.py` (`bartlett`, `blackman`, `hamming`,
`hanning`, `kaiser`, `i0`, `_chbevl`, `_i0_1`, `_i0_2`, `sinc`, `unwrap`)
and `numpy/lib/_type_check_impl.py` (`angle`). Verified bit-exact
(atol=0, rtol=0) against real numpy 2.5.1 across out-of-corpus sweeps
including negative/zero/one/non-integer `M`, `beta` in
{0, negative, huge}, boundary `|x|==8.0` for `i0`, NaN/Inf/int/float32/
complex input for `sinc`, and multi-axis/int-period/length-1-axis input
for `unwrap` -- see this task's report for the exact probes run.

`real`/`imag`/`real_if_close` are deliberately NOT here: numpy's own
`real`/`imag` are literally `val.real`/`val.imag` (a VIEW into the
input's storage, aliasing it -- `np.shares_memory(a, a.real)` is True).
`anionpy.ndarray.real`/`.imag` (not owned by this task; a different,
`ndarray.`-prefixed manifest item) return a fresh COPY instead
(`anionpy.shares_memory(a, a.real)` is False, `.base` is None where numpy's
is the original array) -- confirmed directly, both for complex and for
real dtype input. Composing `real()`/`imag()` on top of that copy would
silently ship the same aliasing defect one level up. `real_if_close`
inherits the same problem on its "close enough" branch. All three are
declined; see this task's report for the measured before/after.
"""

from anionpy._anionpy import (
    abs as _abs,
    arange as _arange,
    arctan2 as _arctan2,
    array as _array,
    concatenate as _concatenate,
    cos as _cos,
    cumsum as _cumsum,
    diff as _diff,
    equal as _equal,
    exp as _exp,
    float64 as _float64,
    greater as _greater,
    less as _less,
    less_equal as _less_equal,
    logical_and as _logical_and,
    mod as _mod,
    ndarray as _ndarray,
    ones as _ones,
    pi as _pi,
    result_type as _result_type,
    sin as _sin,
    sqrt as _sqrt,
    where as _where,
    zeros as _zeros,
)

__all__ = [
    "bartlett", "blackman", "hamming", "hanning", "kaiser", "i0",
    "angle", "sinc", "unwrap",
]


# Machine epsilon per float width. Literal table, NOT a numpy call -- see the
# note in `sinc`. Verified live against numpy 2.5.1's `finfo(d).eps` 2026-08-03.
_EPS = {
    "float16": 0.0009765625,
    "float32": 1.1920928955078125e-07,
    "float64": 2.220446049250313e-16,
}


def _as_ionp(x):
    return x if isinstance(x, _ndarray) else _array(x)


def _window_common(M):
    """Shared `values = np.array([0.0, M]); M = values[1]` prelude used
    by all five window functions (numpy source, verbatim): forces `M`
    through at-least-float64 array construction, so int/float/numpy-
    scalar `M` all land on the same float64 comparison/arithmetic path
    real numpy uses. Returns (M_as_python_float, values_dtype)."""
    values = _array([0.0, M])
    return values[1].item(), values.dtype


def bartlett(M):
    """Bartlett (triangular) window, `M` samples."""
    Mf, dt = _window_common(M)
    if Mf < 1:
        return _array([], dtype=dt)
    if Mf == 1:
        return _ones(1, dtype=dt)
    n = _arange(1 - Mf, Mf, 2)
    return _where(_less_equal(n, 0), 1 + n / (Mf - 1), 1 - n / (Mf - 1))


def hanning(M):
    """Hanning (Hann) window, `M` samples."""
    Mf, dt = _window_common(M)
    if Mf < 1:
        return _array([], dtype=dt)
    if Mf == 1:
        return _ones(1, dtype=dt)
    n = _arange(1 - Mf, Mf, 2)
    return 0.5 + 0.5 * _cos(_pi * n / (Mf - 1))


def hamming(M):
    """Hamming window, `M` samples."""
    Mf, dt = _window_common(M)
    if Mf < 1:
        return _array([], dtype=dt)
    if Mf == 1:
        return _ones(1, dtype=dt)
    n = _arange(1 - Mf, Mf, 2)
    return 0.54 + 0.46 * _cos(_pi * n / (Mf - 1))


def blackman(M):
    """Blackman window, `M` samples."""
    Mf, dt = _window_common(M)
    if Mf < 1:
        return _array([], dtype=dt)
    if Mf == 1:
        return _ones(1, dtype=dt)
    n = _arange(1 - Mf, Mf, 2)
    return (
        0.42
        + 0.5 * _cos(_pi * n / (Mf - 1))
        + 0.08 * _cos(2.0 * _pi * n / (Mf - 1))
    )


# Chebyshev coefficients for i0, copied verbatim from numpy 2.5.1's
# `numpy/lib/_function_base_impl.py` (`_i0A`/`_i0B`, in turn from the
# Cephes math library). Domain [0, 8] uses _I0A, domain (8, inf) uses
# _I0B -- see `i0` below.
_I0A = [
    -4.4153416464793395e-18, 3.3307945188222384e-17, -2.431279846547955e-16,
    1.715391285555133e-15, -1.1685332877993451e-14, 7.676185498604936e-14,
    -4.856446783111929e-13, 2.95505266312964e-12, -1.726826291441556e-11,
    9.675809035373237e-11, -5.189795601635263e-10, 2.6598237246823866e-09,
    -1.300025009986248e-08, 6.046995022541919e-08, -2.670793853940612e-07,
    1.1173875391201037e-06, -4.4167383584587505e-06, 1.6448448070728896e-05,
    -5.754195010082104e-05, 0.00018850288509584165, -0.0005763755745385824,
    0.0016394756169413357, -0.004324309995050576, 0.010546460394594998,
    -0.02373741480589947, 0.04930528423967071, -0.09490109704804764,
    0.17162090152220877, -0.3046826723431984, 0.6767952744094761,
]
_I0B = [
    -7.233180487874754e-18, -4.830504485944182e-18, 4.46562142029676e-17,
    3.461222867697461e-17, -2.8276239805165836e-16, -3.425485619677219e-16,
    1.7725601330565263e-15, 3.8116806693526224e-15, -9.554846698828307e-15,
    -4.150569347287222e-14, 1.54008621752141e-14, 3.8527783827421426e-13,
    7.180124451383666e-13, -1.7941785315068062e-12, -1.3215811840447713e-11,
    -3.1499165279632416e-11, 1.1889147107846439e-11, 4.94060238822497e-10,
    3.3962320257083865e-09, 2.266668990498178e-08, 2.0489185894690638e-07,
    2.8913705208347567e-06, 6.889758346916825e-05, 0.0033691164782556943,
    0.8044904110141088,
]


def _chbevl(x, vals):
    """Clenshaw/Chebyshev polynomial evaluation, numpy's `_chbevl`
    verbatim. The `for` loop below runs a FIXED `len(vals)` times (29 or
    25 -- a compile-time-known coefficient count), not once per array
    element; every iteration is one whole-array `_anionpy` multiply/add."""
    b0 = _zeros(x.shape, dtype=x.dtype) + vals[0]
    b1 = _zeros(x.shape, dtype=x.dtype)
    b2 = b1
    for i in range(1, len(vals)):
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + vals[i]
    return 0.5 * (b0 - b2)


def _i0_1(x):
    return _exp(x) * _chbevl(x / 2.0 - 2, _I0A)


def _i0_2(x):
    return _exp(x) * _chbevl(32.0 / x - 2.0, _I0B) / _sqrt(x)


def i0(x):
    """Modified Bessel function of the first kind, order 0."""
    x = _as_ionp(x)
    if x.dtype.name in ("complex64", "complex128"):
        raise TypeError("i0 not supported for complex values")
    if x.dtype.name not in ("float16", "float32", "float64"):
        x = x.astype(_float64)
    x = _abs(x)
    return _where(_less_equal(x, 8.0), _i0_1(x), _i0_2(x))


def kaiser(M, beta):
    """Kaiser window, `M` samples, shape parameter `beta`."""
    values = _array([0.0, M, beta])
    Mf = values[1].item()
    betaf = values[2].item()
    if Mf == 1:
        return _ones(1, dtype=values.dtype)
    n = _arange(0, Mf)
    alpha = (Mf - 1) / 2.0
    t = (n - alpha) / alpha
    return i0(betaf * _sqrt(1 - t * t)) / i0(betaf)


def angle(z, deg=False):
    """Return the angle (in radians, or degrees if `deg=True`) of a
    complex (or real) argument."""
    z = _as_ionp(z)
    if z.dtype.name in ("complex64", "complex128"):
        zimag = z.imag
        zreal = z.real
    else:
        # BUG FOUND + FIXED (2026-08-04): this was
        # `_zeros(z.shape, dtype=z.dtype)`, a STRONG zero array at the
        # operand's own dtype. numpy's `_type_check_impl.angle` writes the
        # bare literal `zimag = 0` -- a NEP 50 WEAK Python int, which
        # promotes differently. Measured 2026-08-04, numpy 2.5.1,
        # `np.array([1, 0, 1], dtype=bool)`:
        #     np.arctan2(0, bool_arr)               -> float64   (numpy's path)
        #     np.arctan2(zeros(3, bool), bool_arr)  -> float16   (ours)
        # so `np.angle(bool_arr)` was float64 while anionpy's was float16 --
        # values agreed, only the dtype was wrong, which is what revoked
        # this item's declaration on 2026-08-03. Transcribing numpy's
        # literal instead of re-deriving it.
        zimag = 0
        zreal = z
    a = _arctan2(zimag, zreal)
    if deg:
        a = a * (180.0 / _pi)
    return a


def sinc(x):
    """Normalized sinc function, `sin(pi*x) / (pi*x)`, `sinc(0) == 1`."""
    # CORRECTED 2026-08-03 (Monday): this shipped as `import numpy as _np` /
    # `_np.finfo(x.dtype.name).eps`, which violates the standing rule -- anionpy
    # must never call into real numpy to PRODUCE a value; numpy may hand us
    # input bytes, never answers. Replaced with a literal table.
    #
    # Two traps found while fixing this, both worth recording:
    #  1. `anionpy._anionpy.finfo` is NOT `anionpy.finfo`. The Rust-level symbol returns
    #     a plain tuple; `.eps` on it raises AttributeError. The `.eps` attribute
    #     lives on the Python wrapper in `anionpy/__init__.py`. Importing `finfo`
    #     from `._anionpy` alongside the ufuncs looks right and is not.
    #  2. The value alone is not sufficient -- numpy's `finfo().eps` is a TYPED
    #     scalar (`np.float32(1.19e-07)`), and its dtype participates in the
    #     `where` promotion. A bare Python float is a weak scalar. The table
    #     below is therefore materialized at `x`'s own dtype, not passed raw.
    # eps values verified live against numpy 2.5.1 on 2026-08-03.
    x = _as_ionp(x)
    x = _pi * x
    name = x.dtype.name
    if name.startswith("float"):
        eps = _array(_EPS[name], dtype=name)
    else:
        eps = 1e-20
    y = _where(_equal(x, 0), eps, x)
    return _sin(y) / y


def unwrap(p, discont=None, axis=-1, period=2 * _pi):
    """Unwrap by taking the complement of large deltas w.r.t. `period`."""
    p = _as_ionp(p)
    nd = p.ndim
    ax = axis + nd if axis < 0 else axis
    dd = _diff(p, axis=ax)
    if discont is None:
        discont = period / 2.0
    # BUG FOUND + FIXED (2026-08-04): this was `dtype = dd.dtype`, which is
    # NOT what numpy's `_function_base_impl.unwrap` computes. numpy's line is
    #     dtype = np.result_type(dd, period)
    # i.e. the period participates in the promotion. The two differ whenever
    # `period` is not the default `2*pi`. Measured 2026-08-04, numpy 2.5.1,
    # `np.unwrap(np.array([0, 3, 6, 9, 12], dtype=D), period=4)`:
    #     D=bool   np -> int64   anionpy -> float64   (dd is bool, so the
    #                                               `is_int` test below said
    #                                               "not an integer" and took
    #                                               the float branch)
    #     D=int8   np -> int8    anionpy -> int64
    #     D=int16  np -> int16   anionpy -> int64
    #     D=int32  np -> int32   anionpy -> int64
    # Found by the widened `dtype/*/period4` differential cases added in this
    # same change, NOT by the original probe (which only ever ran the default
    # period, where `result_type(dd, 2*pi)` and `dd.dtype` happen to agree for
    # every float input -- a control that could not fail).
    dtype = _result_type(dd, period)
    is_int = dtype.name.startswith("int") or dtype.name.startswith("uint")
    if is_int:
        interval_high = period // 2
        boundary_ambiguous = (period % 2) == 0
    else:
        interval_high = period / 2.0
        boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = _mod(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        cond = _logical_and(_equal(ddmod, interval_low), _greater(dd, 0))
        ddmod = _where(cond, interval_high, ddmod)
    ph_correct = ddmod - dd
    small = _less(_abs(dd), discont)
    ph_correct = _where(small, _zeros(ph_correct.shape, dtype=ph_correct.dtype), ph_correct)
    cs = _cumsum(ph_correct, axis=ax)
    first_idx = tuple(slice(None) if i != ax else slice(0, 1) for i in range(nd))
    rest_idx = tuple(slice(None) if i != ax else slice(1, None) for i in range(nd))
    first_part = p[first_idx]
    if first_part.dtype != dtype:
        first_part = first_part.astype(dtype)
    rest_part = p[rest_idx]
    if rest_part.dtype != dtype:
        rest_part = rest_part.astype(dtype)
    tail = rest_part + cs
    # numpy assigns the tail INTO `up = array(p, copy=True, dtype=dtype)` via
    # `up[slice1] = ...`, and that assignment casts back DOWN to `dtype`. We
    # have no `__setitem__`, so the concatenate below would instead let the
    # cumsum's own promotion win: `cumsum` on a narrow signed integer widens
    # to the default platform int (numpy does this too -- `np.cumsum(
    # np.array([1, 2], dtype=np.int8)).dtype` is int64), and numpy simply
    # throws that widening away on store. Casting explicitly is what
    # reproduces the store, not an extra liberty.
    if tail.dtype != dtype:
        tail = tail.astype(dtype)
    return _concatenate([first_part, tail], axis=ax)
