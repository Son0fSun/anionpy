"""anionpy: NumPy replacement, Python skin over a Rust core.

Python here does dispatch, dunders, exceptions, and dtype rules — never
arithmetic. Every number in this package is computed by the compiled
Rust extension `_anionpy` (see ionp-py/, ionp-core/). See GOAL-ionp.md for
the coverage ledger and the rule this file must never break.

`ndarray` and `dtype` are the Rust-backed types defined in `_anionpy`
(ionp-py/src/lib.rs, wrapping ionp_core::NdArray) — this module does not
define its own array class or reimplement any of their behavior; it only
re-exports them and declares `__ion_state__` for the coverage ledger.
"""

from collections import namedtuple as _namedtuple
from anionpy._anionpy import (
    array,
    ndarray,
    dtype,
    _reconstruct_ndarray,
    sum_f64,
    seterr,
    geterr,
    seterrcall,
    geterrcall,
    errstate,
    add,
    subtract,
    multiply,
    divide,
    true_divide,
    maximum,
    minimum,
    greater,
    greater_equal,
    less,
    less_equal,
    equal,
    not_equal,
    logical_and,
    logical_or,
    logical_xor,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    negative,
    absolute,
    abs,
    invert,
    bitwise_not,
    logical_not,
    sqrt,
    cbrt,
    square,
    reciprocal,
    exp,
    exp2,
    expm1,
    log,
    log2,
    log10,
    log1p,
    sin,
    cos,
    tan,
    arcsin,
    arccos,
    arctan,
    sinh,
    cosh,
    tanh,
    arcsinh,
    arccosh,
    arctanh,
    sign,
    signbit,
    isnan,
    isinf,
    isfinite,
    positive,
    conj,
    conjugate,
    floor,
    ceil,
    trunc,
    rint,
    fabs,
    degrees,
    radians,
    hypot,
    arctan2,
    power,
    copysign,
    fmod,
    remainder,
    nextafter,
    logaddexp,
    logaddexp2,
    heaviside,
    fmax,
    fmin,
    gcd,
    lcm,
    spacing,
    bitwise_count,
    float_power,
    ldexp,
    divmod,
    frexp,
    modf,
    isnat,
    acos,
    asin,
    atan,
    asinh,
    acosh,
    atanh,
    bitwise_invert,
    atan2,
    mod,
    pow,
    bitwise_left_shift,
    bitwise_right_shift,
    deg2rad,
    rad2deg,
    floor_divide,
    left_shift,
    right_shift,
    matmul,
    vecdot,
    matvec,
    vecmat,
    outer,
    dot,
    vdot,
    inner,
    tensordot,
    cross,
    save,
    load,
    savez,
    savez_compressed,
    frombuffer,
    linalg,
    fft,
    random,
    zeros,
    ones,
    empty,
    full,
    zeros_like,
    ones_like,
    empty_like,
    full_like,
    arange,
    linspace,
    eye,
    identity,
    asarray,
    copy,
    copyto,
    nan_to_num,
    ascontiguousarray,
    reshape,
    ravel,
    transpose,
    swapaxes,
    moveaxis,
    squeeze,
    expand_dims,
    broadcast_to,
    sum,
    prod,
    all,
    any,
    min,
    max,
    amin,
    amax,
    argmin,
    argmax,
    cumsum,
    cumprod,
    mean,
    ptp,
    count_nonzero,
    nansum,
    nanprod,
    nancumsum,
    nancumprod,
    nanmin,
    nanmax,
    nanmean,
    var,
    std,
    nanvar,
    nanstd,
    average,
    median,
    nanmedian,
    percentile,
    quantile,
    nanpercentile,
    nanquantile,
    sort,
    argsort,
    sort_complex,
    lexsort,
    nonzero,
    flatnonzero,
    argwhere,
    extract,
    where,
    searchsorted,
    nanargmax,
    nanargmin,
    concatenate,
    stack,
    hstack,
    vstack,
    row_stack,
    dstack,
    column_stack,
    flip,
    fliplr,
    flipud,
    roll,
    tile,
    repeat,
    broadcast_shapes,
    broadcast_arrays,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    diag,
    diagflat,
    diagonal,
    tril,
    triu,
    trace,
    unique,
    unique_values,
    unique_counts as _unique_counts_raw,
    unique_inverse as _unique_inverse_raw,
    unique_all as _unique_all_raw,
    diff,
    ediff1d,
    trim_zeros,
    intersect1d,
    union1d,
    setdiff1d,
    setxor1d,
    isin,
    split,
    array_split,
    hsplit,
    vsplit,
    dsplit,
    insert,
    delete,
    append,
    resize,
    rot90,
    rollaxis,
    diag_indices,
    diag_indices_from,
    tril_indices,
    triu_indices,
    ix_,
    indices,
    meshgrid,
    ravel_multi_index,
    unravel_index,
    fill_diagonal,
    take,
    put,
    take_along_axis,
    put_along_axis,
    compress,
    # Scalar-type hierarchy + module constants (ionp-py/src/scalars.rs).
    # Abstract, non-instantiable bases:
    generic,
    number,
    integer,
    signedinteger,
    unsignedinteger,
    inexact,
    floating,
    complexfloating,
    flexible,
    character,
    # Concrete scalar types:
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    complex64,
    complex128,
    # `longdouble`/`clongdouble` are NOT imported here -- see the
    # TICKET #90a guarded import below `_anionpy`'s scalar-type block,
    # right after this statement: they are compile-time-gated OUT of the
    # `_anionpy` extension entirely on a target where C `long double` is
    # wider than `double` (see scalars.rs), so an unconditional import here
    # would make the whole `anionpy` package fail to import on that target.
    # Same-object aliases (verified `is`-identical to a concrete type above
    # on this platform -- see scalars.rs's `register` for the evidence):
    intp,
    uintp,
    int_,
    long,
    uint,
    ulong,
    intc,
    uintc,
    short,
    ushort,
    byte,
    ubyte,
    half,
    single,
    double,
    csingle,
    cdouble,
    # Module constants:
    nan,
    inf,
    pi,
    e,
    euler_gamma,
    newaxis,
    little_endian,
    True_,
    False_,
    can_cast,
    promote_types,
    result_type,
    min_scalar_type,
    typename,
    mintypecode,
    isdtype,
    shares_memory,
    may_share_memory,
)

# `iinfo`/`finfo` are classes in real numpy (`np.iinfo('int8').max`); the
# Rust side (`_anionpy.iinfo`/`_anionpy.finfo`) hands back plain tuples because
# pyo3 0.29 has no ergonomic way to return a rich object with that many
# fields, so these two names are shadowed here with the thin Python
# wrapper classes from `anionpy._dtypeinfo` rather than imported directly
# above. See `anionpy/_dtypeinfo.py`.
# ---------------------------------------------------------------------------
# `unique_counts` / `unique_inverse` / `unique_all` result containers.
#
# The Array-API spec (which numpy 2.x follows here) says these return a NAMED
# tuple, not a bare one: `np.unique_counts(a)` is a `UniqueCountsResult` with
# `.values`/`.counts`, and `.values` is how the documented API is meant to be
# read. anionpy's Rust layer returned a plain `tuple`, which is positionally
# identical and attribute-wise useless -- code written against the documented
# field names raises AttributeError. That is a real API divergence and it is
# why these three items stood undeclared.
#
# Packaging only: the arrays themselves are computed in Rust and passed
# through untouched, so this adds a container and performs no arithmetic --
# it stays inside "Python where needed, Rust in the core". The class names
# match numpy's exactly (`UniqueCountsResult`, ...), because `type(r).__name__`
# and `repr(r)` are both observable and both compared by the differential
# suite.
# ---------------------------------------------------------------------------
UniqueCountsResult = _namedtuple("UniqueCountsResult", ["values", "counts"])
UniqueInverseResult = _namedtuple("UniqueInverseResult", ["values", "inverse_indices"])
UniqueAllResult = _namedtuple(
    "UniqueAllResult", ["values", "indices", "inverse_indices", "counts"]
)


def unique_counts(x):
    """Return the unique values of `x` and how many times each occurs."""
    return UniqueCountsResult(*_unique_counts_raw(x))


def unique_inverse(x):
    """Return the unique values of `x` and the indices reconstructing it."""
    return UniqueInverseResult(*_unique_inverse_raw(x))


def unique_all(x):
    """Return unique values, first indices, inverse indices, and counts."""
    return UniqueAllResult(*_unique_all_raw(x))


from anionpy._dtypeinfo import iinfo, finfo  # noqa: E402

# Submodules. Imported eagerly so `anionpy.testing` resolves in a fresh process
# without the caller first doing `import anionpy.testing` -- numpy behaves this
# way, and the differential harness resolves items by attribute lookup on the
# top-level module, so a lazily-imported submodule reads as absent.
from anionpy import testing  # noqa: E402,F401

# `fft.test`/`linalg.test`: real numpy's `numpy.fft.test`/`numpy.linalg.test`
# are `numpy._pytesttester.PytestTester` instances that shell out to
# `pytest --pyargs numpy.fft`/`numpy.linalg` against NUMPY'S OWN installed
# test files -- a contract anionpy cannot replicate without importing numpy
# at runtime (banned) or bundling numpy's test files (nonsensical). Mirrors
# `anionpy.testing`'s own `test` attribute instead (`_IonpTester`, defined
# in `anionpy/testing.py`): a same-SHAPE callable (label/verbose/extra_argv/
# tests kwargs, returns bool, internally calls `pytest.main`) that runs
# ANIONPY'S OWN test suite rather than numpy's -- the same substitution
# already accepted and declared for `testing.test` (see that item's ledger
# entry). `anionpy.fft`/`anionpy.linalg` are real PyO3-backed `module`
# objects (not plain-Python files), so `.test` is attached here via plain
# `setattr` after import, rather than via a module-level assignment inside
# a `.py` file the way `testing.py`'s own `test = _IonpTester(...)` works.
from anionpy.testing import _IonpTester as _IonpTester_for_submodules  # noqa: E402
fft.test = _IonpTester_for_submodules("anionpy.fft")
linalg.test = _IonpTester_for_submodules("anionpy.linalg")
del _IonpTester_for_submodules

from anionpy import char  # noqa: E402,F401
from anionpy import strings  # noqa: E402,F401
from anionpy._anionpy import emath  # noqa: E402,F401

# TICKET #90a: `longdouble`/`clongdouble` (scalar types) and the `dtypes`
# submodule (`LongDoubleDType`/`CLongDoubleDType`) exist in the compiled
# `_anionpy` extension ONLY on a target where C `long double` is the same
# width as `double` (see `ionp-py/src/scalars.rs`'s `#[cfg(...)]` gate on
# `LongDouble`/`CLongDouble`, and `dtypes_module.rs`'s matching gate) --
# on a target where `long double` is genuinely wider (e.g. glibc/aarch64,
# x86-64), the Rust extension does not export these names at all, so this
# import raises `ImportError` there and the names simply stay absent from
# `anionpy`, matching `getattr(anionpy, 'longdouble', None) is None`. This
# is NOT a runtime Python decision about correctness (that decision was
# already made at COMPILE time, in Rust, per the ticket's hard constraint)
# -- it is only forwarding whichever symbols the already-built extension
# happens to export on this specific interpreter's target.
try:
    from anionpy._anionpy import longdouble, clongdouble, dtypes  # noqa: E402,F401
except ImportError:
    # Absent by compile-time gate on this target (see the comment above);
    # `__all__` below still lists these three names (this project's static
    # published surface, matching what THIS platform's build actually
    # exports) since every build this project ships from is Apple
    # aarch64 -- if that ever changes, `__all__` would need the same
    # existence filter this `try` already applies to the import itself.
    pass

from anionpy import lib  # noqa: E402,F401
from anionpy import ctypeslib  # noqa: E402,F401
from anionpy import ma  # noqa: E402,F401
from anionpy import polynomial  # noqa: E402,F401
from anionpy.matrix import matrix, asmatrix  # noqa: E402,F401
from anionpy.memmap import memmap  # noqa: E402,F401

# Window functions + a couple of near-free toplevel math items composed
# purely from the calls already imported above -- no new Rust, no Python
# arithmetic loop over array elements. See anionpy/_window_math.py's module
# docstring for the numpy sources these were transcribed from and the
# out-of-corpus bit-exactness sweeps run against them.
from anionpy._window_math import (  # noqa: E402
    bartlett, blackman, hamming, hanning, kaiser, i0, angle, sinc, unwrap,
)

# asanyarray / astype / unstack: thin array-API-alignment wrappers over
# existing anionpy primitives (asarray, ndarray.astype, moveaxis) -- no Rust
# changes, no arithmetic of their own. See anionpy/_manip_compose.py's module
# docstring for the deliberately narrowed scope on each (the open
# identity/layout-contract question) and anionpy/_state/toplevel.py for the
# differential evidence behind each declaration.
from anionpy._manip_compose import asanyarray, astype, unstack  # noqa: E402

# ndarray.mean / ndarray.std / ndarray.var / ndarray.take / ndarray.compress:
# METHOD forms attached onto `ndarray` as thin pass-throughs to the
# already-exact top-level functions of the same name -- no Rust changes.
# See anionpy/_ndarray_methods.py's module docstring for the signature
# derivation (mirrors each top-level anionpy function's own parameter list,
# dropping the leading array argument in favor of `self`) and
# anionpy/_state/ndarray.py for the differential evidence behind each
# declaration.
from anionpy import _ndarray_methods as _ndarray_methods  # noqa: E402
for _name in ("mean", "std", "var", "take", "compress"):
    setattr(ndarray, _name, getattr(_ndarray_methods, _name))
del _name

# isclose: thin Python composition over already-declared-exact primitives
# (asarray, multiply, subtract, absolute, add, less_equal, equal, isfinite,
# isnan, logical_and, logical_or) -- no Rust changes. See
# anionpy/_compare_compose.py's module docstring for the numpy source this was
# transcribed from and the one deliberate substitution (multiply-by-1.0
# in place of the known-broken result_type/promote_types/can_cast trio).
from anionpy._compare_compose import isclose  # noqa: E402

# asarray_chkfinite: thin Python composition over already-declared-exact
# primitives (asarray, isfinite, .all()) -- no Rust changes, no arithmetic.
# See anionpy/_compare_compose.py's module docstring for the numpy source this
# was transcribed from and the one disclosed, pre-existing, out-of-scope gap
# it inherits from asarray itself (invalid `order=` values are silently
# accepted instead of raising).
from anionpy._compare_compose import asarray_chkfinite  # noqa: E402

# round / around / ndarray.round: a transcription of numpy's `PyArray_Round`
# C driver as a sequence of whole-array anionpy calls (multiply/divide/rint plus
# four private Rust primitives for the two things no ufunc call can express).
# `numpy.round` is itself only `_wrapfunc(a, 'round', ...)`, so the method is
# the real implementation and the two module functions delegate to it. See
# anionpy/_round_compose.py's module docstring for the transcribed C flow, the
# 133/143-cell pre-implementation model check, the observable branch ORDER,
# and why `fix` is deliberately NOT part of this change.
from anionpy._round_compose import around, round  # noqa: E402, A004
from anionpy import _round_compose as _round_compose  # noqa: E402
setattr(ndarray, "round", _round_compose._round_method)

# real_if_close: a transcription of numpy's own PYTHON implementation in
# numpy/lib/_type_check_impl.py -- there is no C driver and no ufunc behind
# it, so this is the same shape numpy has, not a shortcut around Rust. Every
# numeric step is one whole-array anionpy call. See
# anionpy/_typecheck_compose.py's module docstring for the transcribed source,
# the four load-bearing details behind it (Python's own `>` supplies the
# bad-`tol` error surface; the `tol > 1` gate is strict; eps DTYPE decides the
# comparison dtype under NEP 50, which is why anionpy.finfo is deliberately not
# used; `a.real` is a view in numpy and a copy here), and why nan_to_num is
# NOT part of this change.
from anionpy._typecheck_compose import real_if_close  # noqa: E402

# kron: another transcription of numpy's OWN Python implementation
# (numpy/lib/_shape_base_impl.py). np.kron has no loop and no accumulation --
# it is a broadcast `multiply` between two arrays with alternating length-1
# axes inserted, then one `reshape` -- so every output element is exactly one
# a[i]*b[j] product and bit-exactness comes for free. See
# anionpy/_shape_compose.py's module docstring for the transcribed source and
# the four measured notes on it (copy=None is not copy=False; ndmin=b.ndim is
# asymmetric ON PURPOSE and both operand orders are tested; the np.matrix
# branch is dead code here, dropped deliberately; the flags.contiguous
# reshape is a layout no-op kept so output layout does not silently depend
# on input layout).
from anionpy._shape_compose import kron  # noqa: E402

# binary_repr / base_repr: transcriptions of numpy's OWN Python bodies
# (numpy/_core/numeric.py). Both are integer-to-STRING formatting, not array
# arithmetic -- and they are correct in numpy precisely because they run on
# CPython's arbitrary-precision `int`, which no fixed-width Rust integer
# could reproduce (`base_repr(2**200)`). They were blocked until
# ndarray.__index__/__int__ existed, since numpy opens them with
# `operator.index(num)` and `int(number)` respectively. See
# anionpy/_intrepr_compose.py's module docstring for the six measured details,
# including binary_repr's zero branch that never checks width and
# base_repr(-0.5) == '-'.
from anionpy._intrepr_compose import base_repr, binary_repr  # noqa: E402

# ---------------------------------------------------------------------------
# Pure-Python wrappers for the comparison/predicate/introspection cluster.
#
# These are NOT ufuncs and have no Rust counterpart -- real numpy implements
# every one of them in Python too (`_core/numeric.py`, `_core/fromnumeric.py`,
# `lib/_function_base_impl.py`, `lib/_type_check_impl.py`), each a thin shim
# over an existing ndarray attribute or an existing ufunc. They are written
# here the same way: no arithmetic of their own, only attribute access and
# calls into the Rust-backed ufuncs/methods already imported above. `.item()`
# is used to unwrap any internal 0-d `anionpy.ndarray` intermediate (`.all()`
# etc. currently return a 0-d array where real numpy returns a numpy scalar --
# a known, separately-tracked defect, see the coverage ledger) into a real
# Python `bool` before it is returned, so these wrappers' own return types are
# correct regardless of that defect. Evidence for each declaration lives in
# `anionpy/_state/toplevel.py`, not here.
# ---------------------------------------------------------------------------
import numbers as _numbers
import operator as _operator
import warnings as _warnings


class _NoValueType:
    """Stand-in for `numpy._NoValue`. Needed wherever `None` is itself a
    meaningful argument value and so cannot double as "not passed" --
    `clip`'s `a_min`/`a_max` are the current case, where `None` means
    "leave this edge unclipped". Its `__repr__` matches numpy's so that a
    TypeError text quoting the default reads identically."""

    __slots__ = ()

    def __repr__(self):
        return "<no value>"


_NoValue = _NoValueType()


def ndim(a):
    """Return the number of dimensions of `a`."""
    try:
        return a.ndim
    except AttributeError:
        return array(a).ndim


def shape(a):
    """Return the shape of `a`."""
    try:
        return a.shape
    except AttributeError:
        return array(a).shape


def size(a, axis=None):
    """Return the number of elements along the given axis (or in total)."""
    if axis is None:
        try:
            return a.size
        except AttributeError:
            return array(a).size
    _shape = shape(a)
    nd = len(_shape)
    axes = axis if isinstance(axis, tuple) else (axis,)
    normalized = []
    for ax in axes:
        n = ax + nd if ax < 0 else ax
        if n < 0 or n >= nd:
            # Reuse anionpy's own axis-validated reduction path so the raised
            # exception is byte-identical to what an out-of-range axis on a
            # real anionpy call already produces instead of hand-rolling a
            # second, possibly-divergent error message here.
            #
            # BUG FOUND + FIXED 2026-08-03 (Monday): this delegated to
            # `sum` and therefore stopped raising at all on a 0-d operand.
            # Real numpy's `sum` grants a 0-d array a courtesy no-op for a
            # scalar `axis=0`/`axis=-1` while `size` does NOT (both
            # measured, /tmp/mg_0d_table.py) -- so the borrowed exception
            # has to be borrowed from a function that shares `size`'s rule,
            # not merely one that happens to validate axes. `mean` is
            # exactly that function: like `size`, and unlike `sum`, it
            # raises `AxisError` for a scalar axis on a 0-d operand, and
            # for every out-of-range axis on an n-d operand it raises the
            # identical error `sum` did. Falling through instead produced
            # a bare `IndexError: tuple index out of range` from the
            # `normalized` lookup below.
            mean(array(a), axis=ax)
        normalized.append(n)
    if len(set(normalized)) != len(normalized):
        # Message verified byte-for-byte against real numpy's
        # `normalize_axis_tuple(..., allow_duplicate=False)`, which is what
        # `np.size` delegates to for its own duplicate-axis check.
        raise ValueError("repeated axis")
    result = 1
    for n in normalized:
        result *= _shape[n]
    return result


# Mirrors real numpy's `np.ScalarType`: builtin Python scalar types plus every
# concrete (instantiable) scalar class. Abstract bases (`generic`, `number`,
# `integer`, ...) are deliberately excluded here -- they are covered by the
# `isinstance(element, generic)` branch in `isscalar` below, exactly as real
# numpy's own `isinstance(element, generic)` branch covers them.
_scalar_types = (
    int, float, complex, bool, bytes, str, memoryview,
    bool_, complex64, complex128, float16, float32, float64,
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
)


def isscalar(element):
    """True if `element` is a scalar (Python builtin or anionpy scalar) type."""
    return (
        isinstance(element, generic)
        or type(element) in _scalar_types
        or isinstance(element, _numbers.Number)
    )


def iterable(y):
    """True if `y` can be iterated over."""
    # Real numpy ndarray defines an explicit `__iter__` that raises
    # `TypeError: iteration over a 0-d array` (verified directly:
    # `np.iterable(np.array(1.0))` is False even though
    # `isinstance(a, collections.abc.Iterable)` is True -- see numpy's own
    # docstring note for `iterable`). `anionpy.ndarray` has no `__iter__` of
    # its own, so Python falls back to the legacy `__getitem__`-based
    # sequence protocol, under which a 0-d array's `iter()` call does NOT
    # raise -- it silently yields an empty sequence (verified directly:
    # `list(iter(anionpy.array(5)))` is `[]`, not a TypeError) -- a real,
    # separate anionpy gap (missing `__iter__`), out of scope to fix in
    # `ionp-py/src/lib.rs` for this cluster. Guarded here explicitly so
    # THIS function's own return value stays correct regardless.
    if isinstance(y, ndarray) and y.ndim == 0:
        return False
    try:
        iter(y)
    except TypeError:
        return False
    return True


def isfortran(a):
    """True if `a` is Fortran-contiguous but not C-contiguous."""
    return a.flags.fnc


def iscomplexobj(x):
    """True if the dtype of `x` is a complex floating type."""
    try:
        dt = x.dtype
    except AttributeError:
        dt = array(x).dtype
    # `anionpy.dtype` has no `.type`/`.kind` (only `.name`/`.itemsize`, verified
    # directly via `dir()`) unlike real numpy's dtype, so dtype family is
    # read off the canonical `.name` string instead.
    return dt.name in ("complex64", "complex128")


def isrealobj(x):
    """True if the dtype of `x` is not a complex floating type."""
    return not iscomplexobj(x)


# -----------------------------------------------------------------------
# real/imag/iscomplex/isreal/isposinf/isneginf  (2026-08-03)
#
# All six live on the Python import surface and contain NO arithmetic: every
# elementwise pass below is a Rust kernel (`ndarray.real`/`.imag` getters,
# `isinf`, `signbit`, `logical_and`, `__ne__`, `__eq__`, `__invert__`,
# `zeros`). What Python contributes is dispatch and result packaging, which
# is exactly what the "Python where needed, Rust in the core" split is for.
#
# They are written as numpy writes them -- same composition, same order --
# because numpy's OBSERVABLE behaviour here is a consequence of the
# composition rather than a specification laid over it, and three of the
# quirks below cannot be reproduced any other way. Measured against numpy
# 2.5.1 before writing a line:
#
#   * `isreal(3.0)` returns a PYTHON `bool`, not `np.bool_`, because it is
#     `imag(x) == 0` and `imag(3.0)` is the Python float `0.0`. Meanwhile
#     `isreal(array(3.0))` returns `np.True_`. Same function, two result
#     types, decided by whether the input had a `.real` attribute.
#   * `real(3.0)` returns the Python float `3.0` for the same reason, while
#     `real(array(3.0))` returns a 0-d ARRAY -- not the scalar that most of
#     the rest of numpy's 0-d surface returns.
#   * `iscomplex` on a non-complex input returns `zeros(shape, bool)[()]`,
#     so it is a 0-d-to-scalar unwrap on 0-d and an array otherwise -- and
#     it never looks at the values at all.
#
# A "cleaner" implementation that normalised any of those would be wrong.
# ------------------------------------------------------------------------


def real(val):
    """Return the real part of `val`, as numpy does: via the `.real`
    attribute when there is one, falling back to `asarray`.

    The `try`/`except AttributeError` is not defensive style, it is the
    dispatch: it is what makes `real(3.0)` a Python float and
    `real([3.0])` an array."""
    try:
        return val.real
    except AttributeError:
        return asarray(val).real


def imag(val):
    """Return the imaginary part of `val`. See `real` for why this is
    written as attribute-access-with-fallback rather than a conversion."""
    try:
        return val.imag
    except AttributeError:
        return asarray(val).imag


def iscomplex(x):
    """True where `x` has a nonzero imaginary part.

    Note what this is NOT: it is not "has a complex dtype" (that is
    `iscomplexobj`). A complex array of purely-real values reports False
    everywhere. And the zero test is `!= 0`, so `complex(1, -0.0)` is False
    (negative zero compares equal to zero) while `complex(1, nan)` is True
    (nan compares unequal to everything, including zero). Both measured
    against numpy 2.5.1 rather than reasoned about.

    For a non-complex input the values are never inspected -- numpy
    allocates a `zeros(shape, bool)` and returns it, so this short-circuit
    is the behaviour, not an optimisation of it."""
    ax = asarray(x)
    if iscomplexobj(ax):
        return ax.imag != 0
    res = zeros(ax.shape, dtype=bool_)
    # `[()]` unwraps a 0-d result to a scalar and is a no-op on any other
    # rank. Dropping it would make `iscomplex(array(1.0))` return a 0-d
    # array where numpy returns `np.False_`.
    return res[()]


def isreal(x):
    """True where `x` has a zero imaginary part.

    Deliberately NOT `logical_not(iscomplex(x))`: it is `imag(x) == 0`,
    which is what makes `isreal(3.0)` a Python `bool` (see this section's
    header). It also means a complex array with a nan imaginary part
    reports False here AND True from `iscomplex` -- consistent, since nan
    is neither equal nor unequal-by-negation to zero."""
    return imag(x) == 0


def _ambiguous_for_complex(x, exc):
    """numpy's TypeError for `isposinf`/`isneginf` on complex input, raised
    from the underlying `signbit` failure. The message names the dtype and
    is reproduced literally (measured on complex64 and complex128:
    "This operation is not supported for complex128 values because it would
    be ambiguous.")."""
    dt = asarray(x).dtype
    raise TypeError(
        f"This operation is not supported for {dt.name} values because "
        f"it would be ambiguous."
    ) from exc


def isposinf(x, out=None):
    """True where `x` is `+inf`.

    Complex input raises TypeError, and it does so because `signbit`
    rejects complex -- the guard is placed on that call rather than on a
    dtype check up front so that the failure keeps numpy's cause chain.
    Integer and boolean input are accepted and report False everywhere,
    since `isinf` is defined for them."""
    is_inf = isinf(x)
    try:
        not_negative = ~signbit(x)
    except TypeError as exc:
        _ambiguous_for_complex(x, exc)
    return logical_and(is_inf, not_negative, out)


def isneginf(x, out=None):
    """True where `x` is `-inf`. See `isposinf`."""
    is_inf = isinf(x)
    try:
        negative = signbit(x)
    except TypeError as exc:
        _ambiguous_for_complex(x, exc)
    return logical_and(is_inf, negative, out)


# numpy's LEGACY polynomial API (poly1d, polyval, polyadd, polysub, polyder,
# polyint, polydiv, poly, roots, polyfit) -- the pre-1.4, highest-degree-
# first convention. NOT `anionpy.polynomial` (imported above), which is the
# modern chebyshev/legendre/hermite/power-series package. Every arithmetic
# op here is either a Rust kernel in `ionp_core::poly_legacy` or composed
# from already-Rust-backed anionpy primitives (including `iscomplex`/`real`/
# `imag`/`isscalar` defined just above, which is why this import sits here
# rather than earlier in the file). `polymul`/`convolve`/`correlate` are
# deliberately NOT exported: they need np.convolve's summation order,
# unreproduced (ticket #45). See anionpy/_polynomial_legacy.py's module
# docstring and ionp/docs/TICKET-72-POLY1D-2026-08-08.md.
from anionpy._polynomial_legacy import (  # noqa: E402
    poly1d, poly, roots, polyval, polyadd, polysub, polyder, polyint,
    polydiv, polyfit, RankWarning,
)


# ---------------------------------------------------------------------------
# permute_dims / clip / cumulative_sum / cumulative_prod / tri
# (added 2026-08-03)
#
# No arithmetic in any of them: `transpose`, `ndarray.clip`, `cumsum`,
# `cumprod`, `arange`, `>=`, `astype`, `concatenate` and `full_like` are all
# Rust-backed. Structured exactly as numpy 2.5.1's own
# `_core/numeric.py` / `_core/fromnumeric.py` / `lib/_twodim_base_impl.py`
# write them, source read (not called).
#
# THREE SIBLINGS FROM THE SAME NUMPY SOURCE FILES ARE DELIBERATELY *NOT*
# HERE, each for a measured reason, not for lack of time:
#   - `geomspace`: numpy's implementation is not a composition, it MUTATES
#     (`result[0] = start`, `result[-1] = stop`, `result *= out_sign`) to
#     force the endpoints, precisely because `exp(log(x)) != x`. anionpy has
#     no `__setitem__` -- the architectural absence already tracked -- so
#     there is no faithful way to write it, and an endpoint-drifting
#     rewrite would be a different function wearing the name.
#   - `logspace`: it is `power(base, linspace(...))`, and anionpy's `power`
#     FAILS its own differential test today. Building on a failing
#     primitive to claim a passing item is exactly the lie the ledger
#     exists to prevent. Blocked on `power`, not on logspace.
#   - `concat`: in real numpy `concat is concatenate` (verified by
#     identity, not by docs). anionpy's `concatenate` passes its tests but is
#     undeclared for its `out=`/`casting='unsafe'` gaps -- an alias
#     inherits its target's gaps exactly, so declaring `concat` while
#     `concatenate` stays undeclared would be laundering.
# ---------------------------------------------------------------------------

def permute_dims(a, axes=None):
    """Array-API spelling of `transpose`. In real numpy this is not a
    wrapper but the SAME function object (`np.permute_dims is np.transpose`
    -- verified by identity), so it is bound, not redefined: a wrapper
    could drift from its target, a binding cannot."""
    return transpose(a, axes)


def matrix_transpose(x, /):
    """Array-API spelling of "transpose the last two axes".

    Unlike `permute_dims`/`transpose` above, real numpy's top-level
    `matrix_transpose` is NOT the same object as `linalg.matrix_transpose`
    (`np.matrix_transpose is np.linalg.matrix_transpose` -> False,
    verified; their `__module__`s differ, `numpy` vs `numpy.linalg`), so
    this is a delegating wrapper rather than a binding. Both spellings
    were measured to produce the identical result AND the identical
    `ValueError: Input array must be at least 2-dimensional, but it is
    {ndim}` on 0-d and 1-d input, so delegation is a transcription of that
    measured equivalence, not an assumption that two same-named functions
    must agree.

    `x` is positional-only in numpy (`inspect.signature` -> `(x, /)`), and
    the `/` here is what reproduces its `TypeError: matrix_transpose() got
    some positional-only arguments passed as keyword arguments: 'x'`.
    """
    return linalg.matrix_transpose(x)


def clip(a, a_min=_NoValue, a_max=_NoValue, out=None, *,
         min=_NoValue, max=_NoValue, **kwargs):
    """Clip values to an interval.

    The four-way argument dance is numpy's, reproduced literally including
    the two distinct error types: omitting exactly one of the positional
    pair is a TypeError naming the missing one, while mixing the
    positional pair with the newer `min=`/`max=` keywords is a ValueError.
    A sentinel is required (not `None`) because `None` is a MEANINGFUL
    value here -- it means "do not clip this edge" -- so `None` and
    "not passed" cannot share a spelling."""
    if a_min is _NoValue and a_max is _NoValue:
        a_min = None if min is _NoValue else min
        a_max = None if max is _NoValue else max
    elif a_min is _NoValue:
        raise TypeError("clip() missing 1 required positional "
                        "argument: 'a_min'")
    elif a_max is _NoValue:
        raise TypeError("clip() missing 1 required positional "
                        "argument: 'a_max'")
    elif min is not _NoValue or max is not _NoValue:
        raise ValueError("Passing `min` or `max` keyword argument when "
                         "`a_min` and `a_max` are provided is forbidden.")

    arr = a if isinstance(a, ndarray) else asarray(a)
    return arr.clip(a_min, a_max, out=out, **kwargs)


def _cumulative(x, accumulate, identity, axis, dtype, out, include_initial):
    """Shared body of `cumulative_sum`/`cumulative_prod`, mirroring numpy's
    own `_cumulative_func`. The `axis=None`-only-for-1-D rule is numpy's
    and is the one behaviour that distinguishes these from `cumsum`/
    `cumprod`, which silently ravel instead."""
    arr = atleast_1d(x)
    ndim = arr.ndim
    if axis is None:
        if ndim >= 2:
            raise ValueError("For arrays which have more than one dimension "
                             "``axis`` argument is required.")
        axis = 0

    if out is not None and include_initial:
        # numpy writes the accumulation into a slice of `out` and then
        # stamps the identity into position 0. Both halves need
        # `__setitem__`, which anionpy does not have.
        raise NotImplementedError(
            "anionpy.cumulative_sum/cumulative_prod: out= together with "
            "include_initial=True requires assigning into a slice of `out`, "
            "and anionpy arrays have no __setitem__ (tracked architectural "
            "absence). Either pass out= alone or include_initial= alone."
        )

    res = accumulate(arr, axis=axis, dtype=dtype, out=out)
    if include_initial:
        initial_shape = list(arr.shape)
        initial_shape[axis] = 1
        res = concatenate(
            [full_like(res, identity, shape=initial_shape), res],
            axis=axis,
        )
    return res


def cumulative_sum(x, /, *, axis=None, dtype=None, out=None,
                   include_initial=False):
    """Array-API cumulative sum. Identity for the prepended initial value
    is `add.identity`, i.e. 0."""
    return _cumulative(x, cumsum, 0, axis, dtype, out, include_initial)


def cumulative_prod(x, /, *, axis=None, dtype=None, out=None,
                    include_initial=False):
    """Array-API cumulative product. Identity is `multiply.identity`, 1."""
    return _cumulative(x, cumprod, 1, axis, dtype, out, include_initial)


def _min_int(low, high):
    """numpy's own `lib/_twodim_base_impl._min_int` -- the narrowest signed
    integer dtype spanning [low, high]. Reproduced rather than replaced with
    a blunt int64, because the `<=`/`>=` comparisons here are load-bearing
    for ERROR behaviour, not just for storage: `tri("3")` must fail with
    Python's own `"'<=' not supported between instances of 'str' and 'int'"`,
    which is exactly what these comparisons raise. An int64 shortcut would
    have produced a different, wrong TypeError."""
    if high <= iinfo(int8).max and low >= iinfo(int8).min:
        return int8
    if high <= iinfo(int16).max and low >= iinfo(int16).min:
        return int16
    if high <= iinfo(int32).max and low >= iinfo(int32).min:
        return int32
    return int64


def tri(N, M=None, k=0, dtype=float, *, like=None):
    """Ones at and below the k-th diagonal, zeros elsewhere.

    numpy builds this as `greater_equal.outer(arange(N), arange(-k, M-k))`
    then `astype(dtype, copy=False)`; anionpy has no `ufunc.outer`, so the
    same outer product is spelled with an explicit trailing axis on the
    left operand -- identical broadcast, identical values.

    numpy's own version selects a minimum-width integer dtype for the two
    `arange`s (`_min_int`); the result is a bool comparison either way, so
    the choice is invisible in the output for every N/M/k that fits in
    int64. Using int64 unconditionally is therefore a difference in
    intermediate storage, not in behaviour, and is noted rather than
    hidden.

    `like=` is rejected rather than ignored: silently dropping a
    dispatch-protocol argument would hand back an anionpy array to a caller
    who asked for something else entirely."""
    warning_for_type = None
    try:
        N = _operator.index(N)
    except TypeError:
        warning_for_type = warning_for_type or type(N)

    if like is not None:
        raise NotImplementedError(
            "anionpy.tri: like= (the __array_function__ dispatch protocol) is "
            "not supported; anionpy cannot construct a foreign array type."
        )

    if M is None:
        M = N
    else:
        try:
            M = _operator.index(M)
        except TypeError:
            warning_for_type = warning_for_type or type(M)

    try:
        k = _operator.index(k)
    except TypeError:
        warning_for_type = warning_for_type or type(k)

    # reshape(-1, 1), NOT reshape(N, 1). numpy's own tri builds this with
    # `greater_equal.outer(...)`, which never needs N as a shape at all --
    # so a non-integer N (which numpy DEPRECATES but still honours, see
    # warning_for_type below) flows straight through. Writing reshape(N, 1)
    # re-introduced an integer requirement numpy does not have and made
    # tri(3.0) raise "'float' object cannot be interpreted as an integer".
    # Worth recording why that took so long to see: it was misfiled for a
    # day as an `arange(3.0, dtype=int8)` defect, on the strength of the
    # traceback's neighbourhood rather than a probe. arange handles every
    # float/int dtype combination correctly; the bug was always in this
    # line. A blocker you have not reproduced in isolation is a guess.
    rows = arange(N, dtype=_min_int(0, N)).reshape(-1, 1)
    cols = arange(-k, M - k, dtype=_min_int(-k, M - k))
    m = greater_equal(rows, cols)
    m = m.astype(dtype, copy=False)

    if warning_for_type:
        _warnings.warn(
            (f"Cannot convert {(warning_for_type).__name__} safely to an integer."
             "This will raise an error in future versions (Deprecated NumPy 2.5)"),
            DeprecationWarning,
            stacklevel=2,
        )

    return m


def array_equal(a1, a2, equal_nan=False):
    """True if two array_likes have the same shape and all elements equal."""
    try:
        arr1 = a1 if isinstance(a1, ndarray) else array(a1)
        arr2 = a2 if isinstance(a2, ndarray) else array(a2)
    except Exception:
        return False
    if arr1.shape != arr2.shape:
        return False
    if not equal_nan:
        return equal(arr1, arr2).all().item()
    if arr1 is arr2:
        return True
    _nan_capable = ("float16", "float32", "float64", "complex64", "complex128")
    if arr1.dtype.name not in _nan_capable and arr2.dtype.name not in _nan_capable:
        # Neither dtype can hold a NaN -- same fast path as real numpy's
        # `_dtype_cannot_hold_nan` check.
        return equal(arr1, arr2).all().item()
    a1nan = isnan(arr1)
    if a1nan.all().item():
        return isnan(arr2).all().item()
    eq_or_both_nan = logical_or(equal(arr1, arr2), logical_and(a1nan, isnan(arr2)))
    return eq_or_both_nan.all().item()


def array_equiv(a1, a2):
    """True if `a1` and `a2` are broadcast-shape-consistent and all-equal."""
    try:
        arr1 = a1 if isinstance(a1, ndarray) else array(a1)
        arr2 = a2 if isinstance(a2, ndarray) else array(a2)
    except Exception:
        return False
    try:
        eq = equal(arr1, arr2)
    except Exception:
        # Real numpy probes broadcastability with `multiarray.broadcast`
        # first and returns False on failure; anionpy has no standalone
        # broadcast-shape-check primitive, so `equal` (which already
        # implements the same broadcasting rules -- verified directly:
        # shape-mismatched, non-broadcastable operands raise `ValueError`
        # from the same code path a bare `a1 == a2` would use) is used as
        # the probe instead. Either way the outcome is the same: any
        # failure to combine the two operands means "not equivalent".
        return False
    return eq.all().item()


# `isneginf`/`isposinf`/`isclose`/`iscomplex`/`isreal` are deliberately NOT
# implemented here. Real numpy's actual contract for all five is dual-typed
# on the input: scalar input returns `numpy.bool_`, array input returns
# `numpy.ndarray` (verified directly against real numpy 2.5.1, see
# `/tmp/probe_iscomplex_types.py` in the 2026-08-02 session log). `numpy.bool_`
# is a specific class belonging to real numpy; no ionp-side type can ever be
# `is`-identical to it (`anionpy.bool_` is a different class), and manufacturing
# one would mean importing and calling into real numpy from shipping code,
# which is out of bounds. This is the same root cause a live measurement
# already identified across 22 reduction-style anionpy operations (0-d
# `anionpy.ndarray` where real numpy returns a scalar). Left undeclared and
# unimplemented rather than shipped with a silently-wrong return type.


_INEXACT_DTYPE_NAMES = ("float16", "float32", "float64", "complex64", "complex128")


def _isclose_arr(a, b, rtol, atol, equal_nan):
    x = a if isinstance(a, ndarray) else array(a)
    y = b if isinstance(b, ndarray) else array(b)
    # Real numpy's own `isclose` explicitly promotes `y` to an inexact dtype
    # before subtracting (`_core/numeric.py`: `dt = result_type(y, 1.)`,
    # then `y = asanyarray(y, dtype=dt)`) -- "to avoid bad behavior on
    # abs(MIN_INT)" per its own comment, but it also sidesteps a genuine
    # anionpy gap found directly via the differential corpus: `subtract` on
    # two bool arrays raises `TypeError` in anionpy (matching real numpy's own
    # ufunc restriction -- bool subtraction is banned in both), so a bare
    # `x - y` on two bool inputs would raise here where real
    # `np.allclose([True, False], [True, True])` returns `False` cleanly.
    # Promoting `y` to float64 up front (mirroring numpy) sidesteps the
    # same case here, since anionpy's `subtract` supports bool-vs-float64.
    if y.dtype.name not in _INEXACT_DTYPE_NAMES:
        y = y.astype(float64)
    close = logical_and(
        less_equal(absolute(subtract(x, y)), atol + rtol * absolute(y)),
        isfinite(y),
    )
    close = logical_or(close, equal(x, y))
    if equal_nan:
        close = logical_or(close, logical_and(isnan(x), isnan(y)))
    return close


def allclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    """True if every element of `a` is close to the matching element of `b`."""
    return _isclose_arr(a, b, rtol, atol, equal_nan).all().item()







def _ndarray_buffer(self, flags=0):
    """PEP 688 buffer export.

    Returns a typed, shaped memoryview of a C-contiguous copy (`tobytes()`).
    Complex and any empty N-D array raise TypeError so ``np.asarray`` falls
    back to ``__array__`` (verified). A naive ``memoryview(tobytes())`` is
    uint8-1d and *steals* asarray; do not regress to that.
    """
    _ = flags
    name = self.dtype.name
    fmt = {
        "bool": "?",
        "int8": "b",
        "int16": "h",
        "int32": "i",
        "int64": "q",
        "uint8": "B",
        "uint16": "H",
        "uint32": "I",
        "uint64": "Q",
        "float16": "e",
        "float32": "f",
        "float64": "d",
    }.get(name)
    if fmt is None:
        raise TypeError(
            f"anionpy.ndarray.__buffer__: dtype {name!r} has no native "
            "single-character memoryview format"
        )
    shape = tuple(int(s) for s in self.shape)
    raw = self.tobytes()
    mv = memoryview(raw)
    if __import__("builtins").any(s == 0 for s in shape):
        if self.ndim == 1:
            return mv.cast(fmt)
        raise TypeError(
            "anionpy.ndarray.__buffer__: empty N-D arrays cannot be cast "
            "to a shaped memoryview"
        )
    if self.ndim == 0:
        return mv.cast(fmt, ())
    return mv.cast(fmt, shape)


ndarray.__buffer__ = _ndarray_buffer

__all__ = [
    "testing",
    "char",
    "strings",
    "emath",
    "lib",
    "ctypeslib",
    "ma",
    "array",
    "ndarray",
    "dtype",
    "sum_f64",
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "maximum",
    "minimum",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "negative",
    "absolute",
    "abs",
    "invert",
    "bitwise_not",
    "logical_not",
    "sqrt",
    "cbrt",
    "square",
    "reciprocal",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sign",
    "signbit",
    "isnan",
    "isinf",
    "isfinite",
    "positive",
    "conj",
    "conjugate",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "round",
    "around",
    "fabs",
    "degrees",
    "radians",
    "hypot",
    "arctan2",
    "power",
    "copysign",
    "fmod",
    "remainder",
    "nextafter",
    "logaddexp",
    "logaddexp2",
    "heaviside",
    "fmax",
    "fmin",
    "gcd",
    "lcm",
    "spacing",
    "bitwise_count",
    "float_power",
    "ldexp",
    "divmod",
    "frexp",
    "modf",
    "isnat",
    "acos",
    "asin",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "bitwise_invert",
    "atan2",
    "mod",
    "pow",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "deg2rad",
    "rad2deg",
    "floor_divide",
    "left_shift",
    "right_shift",
    "matmul",
    "vecdot",
    "matvec",
    "vecmat",
    "outer",
    "dot",
    "vdot",
    "inner",
    "tensordot",
    "cross",
    "save",
    "load",
    "savez",
    "savez_compressed",
    "frombuffer",
    "linalg",
    "fft",
    "random",
    "dtypes",
    "zeros",
    "ones",
    "empty",
    "full",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
    "arange",
    "linspace",
    "eye",
    "identity",
    "asarray",
    "copy",
    "copyto",
    "nan_to_num",
    "ascontiguousarray",
    "reshape",
    "ravel",
    "transpose",
    "swapaxes",
    "moveaxis",
    "squeeze",
    "expand_dims",
    "broadcast_to",
    "sum",
    "prod",
    "all",
    "any",
    "min",
    "max",
    "amin",
    "amax",
    "argmin",
    "argmax",
    "cumsum",
    "cumprod",
    "mean",
    "ptp",
    "count_nonzero",
    "nansum",
    "nanprod",
    "nancumsum",
    "nancumprod",
    "nanmin",
    "nanmax",
    "nanmean",
    "var",
    "std",
    "nanvar",
    "nanstd",
    "average",
    "median",
    "nanmedian",
    "percentile",
    "quantile",
    "nanpercentile",
    "nanquantile",
    "sort",
    "argsort",
    "sort_complex",
    "lexsort",
    "nonzero",
    "flatnonzero",
    "argwhere",
    "extract",
    "where",
    "searchsorted",
    "nanargmax",
    "nanargmin",
    "concatenate",
    "stack",
    "hstack",
    "vstack",
    "row_stack",
    "dstack",
    "column_stack",
    "flip",
    "fliplr",
    "flipud",
    "roll",
    "tile",
    "repeat",
    "broadcast_shapes",
    "broadcast_arrays",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "diag",
    "diagflat",
    "diagonal",
    "tril",
    "triu",
    "trace",
    "unique",
    "unique_values",
    "unique_counts",
    "unique_inverse",
    "unique_all",
    "diff",
    "ediff1d",
    "trim_zeros",
    "intersect1d",
    "union1d",
    "setdiff1d",
    "setxor1d",
    "isin",
    "split",
    "array_split",
    "hsplit",
    "vsplit",
    "dsplit",
    "insert",
    "delete",
    "append",
    "resize",
    "rot90",
    "rollaxis",
    "diag_indices",
    "diag_indices_from",
    "tril_indices",
    "triu_indices",
    "ix_",
    "indices",
    "meshgrid",
    "ravel_multi_index",
    "unravel_index",
    "fill_diagonal",
    "take",
    "put",
    "take_along_axis",
    "put_along_axis",
    "compress",
    "generic",
    "number",
    "integer",
    "signedinteger",
    "unsignedinteger",
    "inexact",
    "floating",
    "complexfloating",
    "flexible",
    "character",
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "longdouble",
    "clongdouble",
    "intp",
    "uintp",
    "int_",
    "long",
    "uint",
    "ulong",
    "intc",
    "uintc",
    "short",
    "ushort",
    "byte",
    "ubyte",
    "half",
    "single",
    "double",
    "csingle",
    "cdouble",
    "nan",
    "inf",
    "pi",
    "e",
    "euler_gamma",
    "newaxis",
    "little_endian",
    "True_",
    "False_",
    "can_cast",
    "promote_types",
    "result_type",
    "min_scalar_type",
    "typename",
    "mintypecode",
    "isdtype",
    "iinfo",
    "finfo",
    "shares_memory",
    "may_share_memory",
    "ndim",
    "shape",
    "size",
    "isscalar",
    "iterable",
    "isfortran",
    "iscomplexobj",
    "isrealobj",
    "matrix_transpose",
    "permute_dims",
    "clip",
    "cumulative_sum",
    "cumulative_prod",
    "tri",
    "real",
    "imag",
    "iscomplex",
    "isreal",
    "isposinf",
    "isneginf",
    "array_equal",
    "array_equiv",
    "allclose",
    "bartlett",
    "blackman",
    "hamming",
    "hanning",
    "kaiser",
    "i0",
    "angle",
    "sinc",
    "unwrap",
    "asanyarray",
    "astype",
    "unstack",
    "isclose",
    "asarray_chkfinite",
    "real_if_close",
    "kron",
    "base_repr",
    "binary_repr",
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

# ---------------------------------------------------------------------------
# Coverage ledger declarations (tools/coverage.py). ONLY items that are both
# implemented AND verified against real numpy belong in the ledger — declaring
# more than is proven is exactly the "phantom" failure mode GOAL-ionp.md calls
# out. A passing differential test is necessary but has twice proven NOT
# sufficient (matmul, and ndarray.__repr__/__getitem__, both declared on green
# tests and both false outside the test corpus), so an item is only as good as
# the evidence recorded beside it.
#
# Everything currently declared is at state 'exact': correct, Rust-computed,
# general-purpose plumbing, with no structural/asymptotic win over numpy
# claimed (that's the planned acceleration layer's job, ordering steps 3-5 in
# GOAL-ionp.md, not yet started).
#
# Coverage declarations live in `anionpy/_state/`, one module per surface block
# (toplevel / ndarray / linalg / testing), and are merged here with a hard
# collision check. Split out of this file on 2026-08-01 -- it had grown to a
# single 565-line dict that every declaring agent had to serialise through,
# which is where five separate blocks of finished work went uncounted. See
# `anionpy/_state/__init__.py` and
# `reports/ionp-throughput-bottleneck-2026-08-01.md`.
#
# To declare an item: edit the module for its block, and bring the measurement
# that justifies it. Do not add entries here.
# ---------------------------------------------------------------------------
from anionpy._state import build_ion_state  # noqa: E402

__ion_state__ = build_ion_state()
