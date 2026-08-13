"""Python-facing wrapper classes for `anionpy.iinfo` / `anionpy.finfo`.

`_anionpy.iinfo(dtype)` / `_anionpy.finfo(dtype)` (the Rust-side functions in
`ionp-py/src/dtypeinfo.rs`) return plain tuples -- pyo3 0.29 tuple-arity
limits meant the Rust side hands back raw values rather than an object.
Real `numpy.iinfo`/`numpy.finfo` are classes accessed via attributes
(`np.iinfo('int8').max`), so this module wraps the tuples the same way
`anionpy.lib.NumpyVersion` wraps a parsed version string -- a thin Python
class over Rust-derived data, per that established precedent.

`IInfo` mirrors numpy's `iinfo` exactly (verified field-by-field and by
repr against numpy 2.5.1) and is declared in `anionpy/_state/toplevel.py`.

`finfo` mirrors numpy's `finfo` field-by-field: numpy's finfo attributes
(`.eps`, `.max`, ...) are numpy scalar objects (e.g. `numpy.float32(...)`),
not plain Python floats. Every float-valued attribute below is therefore
minted as a dtype-typed scalar via `_finfo_scalar()`, which routes through
anionpy's established 0-d-array scalar contract (`numpy_scalar_from_0d`) --
read that function's docstring before changing it, and in particular do NOT
"simplify" it to `_core.float32(...)`: that constructs anionpy's own lookalike
class, which is exactly what the contract exists to avoid.

The value itself is computed in Rust by `_core.finfo`; nothing here does
arithmetic. See `toplevel.py`'s finfo comment for the declaration history.
"""

from anionpy import _anionpy as _core

#: dtype name -> anionpy scalar type constructor, for wrapping `_core.finfo`'s
#: raw float values the same way numpy's `finfo` attributes are numpy-
#: scalar-typed rather than plain Python floats. `_core.finfo`'s `name`
#: field is always a float dtype name (float16/32/64) even for a
#: complex64/complex128 receiver -- the Rust side already redirects
#: complex dtypes to their component float type, matching numpy's own
#: `finfo(complex64).dtype == dtype('float32')` redirection (verified).
_FINFO_SCALAR_DTYPES = frozenset({"float16", "float32", "float64"})


def _finfo_scalar(value, name):
    """Mint the dtype-typed scalar for a `finfo` attribute.

    CORRECTED 2026-08-06 (Monday). This deliberately does NOT call
    `_core.float16`/`float32`/`float64` directly. Those construct anionpy's
    OWN scalar classes, and `numpy_scalar_from_0d`'s doc comment in
    `ionp-py/src/lib.rs` names the reason that is wrong here:

        the differential harness asserts `type(ionp_out) is type(np_out)`
        exactly, so this branch must keep minting REAL numpy scalars
        whenever numpy is importable, never anionpy's own lookalikes.

    MEASURED: `type(anionpy.float32(1.5)) is numpy.float32` -> False, while
    `type(_core.array(1.5, dtype="float32")[()]) is numpy.float32` -> True.
    Routing through a 0-d array and indexing it with `[()]` reaches the
    established `numpy_scalar_from_0d` contract that array indexing already
    uses, so `finfo` returns the same scalar objects the rest of anionpy does
    instead of a second, divergent kind.

    This is NOT a new numpy borrow and does not weaken the standing rule
    that anionpy never calls numpy for an answer. The VALUE is computed in
    Rust by `_core.finfo`; numpy only mints the wrapper TYPE, which is the
    same already-blessed division `numpy_scalar_from_0d` makes at every one
    of its existing call sites. When numpy is absent that helper falls back
    to anionpy's own same-named scalar class on its own, so this path degrades
    gracefully without a second code path here.

    Side effect worth recording: this also makes float16 `str()` match
    numpy (`'6.104e-05'`, not `'6.097555e-05'`), because with numpy present
    the object IS a real `numpy.float16`. That does NOT fix the underlying
    `anionpy.float16.__str__` defect, which is still open and still applies
    whenever numpy is absent -- it only means `finfo` no longer depends on
    it. Do not read this as clearing that item.
    """
    return _core.array(value, dtype=name)[()]

#: dtype name -> the `%`-style format numpy's `finfo.__repr__` uses for
#: `.min`/`.max` (real numpy 2.5.1's `numpy._core.getlimits._MACHAR_PARAMS`
#: table, read directly and reproduced here as anionpy's own constants -- not
#: computed, not imported from numpy). `.resolution` is never run through
#: this format in numpy's own repr (it always uses plain `str()`), so it
#: has no entry here on purpose.
_FINFO_REPR_FMT = {
    "float16": "%12.5e",
    "float32": "%15.7e",
    "float64": "%24.16e",
}


class iinfo:
    """Machine limits for integer dtypes. Mirrors `numpy.iinfo`.

    RENAMED from `IInfo` 2026-08-08 (ticket #71): real numpy's class is
    literally named `iinfo` (lowercase; `type(np.iinfo('i1')).__name__ ==
    'iinfo'`, live-verified against numpy 2.5.1), same axis `finfo` was
    already fixed on in task #28 -- this was the "asymmetric survivor" that
    fix's own doc comment flagged as a separate, not-fixed-there gap.
    `anionpy/__init__.py` already imported this class AS `iinfo` (so
    `anionpy.iinfo` the CALLABLE was already correctly named), but the class
    object's OWN identity was not -- `type(anionpy.iinfo(x)).__name__` was
    `'IInfo'`, which fails any check on `x.__class__.__name__` or a class-
    name-embedding repr, not just a cosmetic label.
    """

    __module__ = "anionpy"

    def __init__(self, dtype):
        min_, max_, bits, name = _core.iinfo(dtype)
        # STORED vs COMPUTED split, matching real numpy's own class shape
        # (2026-08-08, ticket #71). numpy's `iinfo.min`/`.max` are
        # `@property` descriptors defined on the CLASS
        # (`numpy/_core/getlimits.py`), not plain instance attributes set in
        # `__init__` -- confirmed by direct inspection of
        # `numpy._core.getlimits.iinfo.__dict__`, which contains `'min'`/
        # `'max'` as `property` objects. Storing the raw values under
        # private names and exposing them through `@property` below (see
        # `min`/`max` immediately after this method) reproduces that same
        # class-level shape. This is not cosmetic: `anionpy._dtypeinfo`
        # used to set `self.min`/`self.max` directly, which makes `'min' in
        # vars(iinfo)` (the CLASS, not an instance) False where numpy's is
        # True -- exactly the check `tools/coverage.py`'s `resolve()` uses
        # to decide whether an exploded-class ledger item even RESOLVES
        # (`present = attr in vars(cls)`), so the old shape made
        # `iinfo.min`/`iinfo.max` register as structurally `absent` no
        # matter how correct the computed value was. The values themselves
        # are unchanged by this -- still the same Rust-computed ints from
        # `_core.iinfo`, no new arithmetic added here.
        self._min = min_
        self._max = max_
        self.bits = bits
        # `_anionpy.dtype` has no public string constructor; `promote_types`
        # applied to a dtype name against itself is a no-op that yields the
        # dtype object for that name -- reused here as the name -> PyDType
        # lookup.
        self.dtype = _core.promote_types(name, name)

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(min={self.min}, max={self.max}, dtype={self.dtype})"

    def __str__(self):
        # Verbatim transcription of real numpy 2.5.1's `iinfo.__str__`
        # (`numpy/_core/getlimits.py`), read directly, not reconstructed
        # from the repr. `%(dtype)s` calls `str()` on `self.dtype`, which
        # for both numpy's and anionpy's dtype objects is the bare name
        # ("int8"), not the `dtype('int8')` repr form -- verified live,
        # byte-for-byte, for all 8 concrete integer dtypes.
        fmt = (
            "Machine parameters for %(dtype)s\n"
            "---------------------------------------------------------------\n"
            "min = %(min)s\n"
            "max = %(max)s\n"
            "---------------------------------------------------------------\n"
        )
        return fmt % {"dtype": self.dtype, "min": self.min, "max": self.max}


class finfo:
    """Machine limits for floating-point dtypes. Mirrors `numpy.finfo`
    field-by-field, including attribute TYPE (see module docstring).

    RENAMED from `FInfo` 2026-08-06 (task #28): real numpy's class is
    literally named `finfo` (lowercase; `type(np.finfo('f4')).__name__ ==
    'finfo'`). `iinfo`'s matching mismatch on this same axis (class was
    `IInfo`) was fixed later, 2026-08-08, ticket #71 -- see that class's
    docstring; both are now correctly named.

    Also mirrors numpy's per-RESOLVED-dtype INSTANCE CACHING
    (`np.finfo('f4') is np.finfo('f4')` is True, and -- measured live --
    every spelling that resolves to the same concrete dtype shares the
    SAME instance: `np.finfo('f4') is np.finfo(np.float32) is
    np.finfo(np.dtype('float32'))`, and a complex dtype's finfo IS its
    component float's cached instance, `np.finfo('complex64') is
    np.finfo('float32')`, both True). `__new__` below keys the cache by
    the RESOLVED float dtype name (`_core.finfo`'s own last tuple field,
    already component-redirected on the Rust side), which reproduces all
    of the above for free -- no dtype-string normalization needed here.

    numpy's `finfo` does NOT define a custom `__eq__`/`__hash__`
    (confirmed: `'__eq__' not in np.finfo.__dict__` on a live check) --
    `==` on two finfo instances is plain `object.__eq__`, i.e. identity.
    The caching above is what makes two calls for the SAME resolved dtype
    compare `==` to each other; this class deliberately does NOT add its
    own `__eq__` either, for the same reason.
    """

    __module__ = "anionpy"

    #: dtype name -> the single cached instance for that resolved float
    #: dtype, mirroring numpy's own `finfo._finfo_cache` (a plain class-
    #: level dict, not thread-safe there either -- matched, not improved
    #: on).
    _cache: dict = {}

    def __new__(cls, dtype):
        raw = _core.finfo(dtype)
        name = raw[-1]
        cached = cls._cache.get(name)
        if cached is not None:
            return cached
        obj = super().__new__(cls)
        # Stashed for `__init__` (which Python calls unconditionally on
        # the object `__new__` returns, cache hit or not) to consume and
        # delete. A cache-hit `__init__` call sees no `_pending_raw` and
        # returns immediately, leaving the already-initialized cached
        # instance untouched.
        obj._pending_raw = raw
        cls._cache[name] = obj
        return obj

    def __init__(self, dtype):
        raw = self.__dict__.pop("_pending_raw", None)
        if raw is None:
            return
        (
            eps,
            epsneg,
            max_,
            min_,
            tiny,
            smallest_normal,
            smallest_subnormal,
            resolution,
            precision,
            bits,
            iexp,
            nexp,
            nmant,
            machep,
            negep,
            minexp,
            maxexp,
            name,
        ) = _core.finfo(dtype)
        # `name` is always a float dtype name (float16/32/64) -- see
        # `_FINFO_SCALAR_DTYPES`' comment above. A name outside that set
        # would mean `_core.finfo` broke its Rust-side contract, which is
        # something to surface loudly, not paper over with a silent
        # `.get(..., float)` fallback that would quietly reintroduce the
        # plain-`float` divergence this whole module exists to fix.
        if name not in _FINFO_SCALAR_DTYPES:
            raise KeyError(name)

        def ctor(v):
            return _finfo_scalar(v, name)
        # STORED vs COMPUTED split, matching real numpy's own class shape
        # (2026-08-08, ticket #71). numpy's `finfo` defines `epsneg`,
        # `iexp`, `machep`, `negep`, `nexp`, `resolution`, and `tiny` as
        # `@cached_property` on the CLASS (`numpy/_core/getlimits.py`);
        # every other float-valued attribute here (`eps`, `max`, `min`,
        # `smallest_normal`, `smallest_subnormal`) is a plain instance
        # attribute set directly in `_populate_finfo_constants`, same as
        # `precision`/`bits`/`nmant`/`minexp`/`maxexp`/`dtype` below --
        # confirmed by direct inspection of
        # `numpy._core.getlimits.finfo.__dict__` (contains exactly those
        # seven names as descriptors; nothing else). anionpy previously set
        # ALL seventeen fields as plain `self.<name> = ...` instance
        # attributes, which makes e.g. `'epsneg' in vars(finfo)` (the
        # CLASS) False where numpy's is True -- the exact check
        # `tools/coverage.py`'s `resolve()` uses to decide whether an
        # exploded-class ledger item resolves at all (`present = attr in
        # vars(cls)`), so all seven registered as structurally `absent`
        # regardless of value correctness. Fixed by storing the seven
        # under private names and exposing them via `@property` below (see
        # immediately after this method) -- values unchanged, still the
        # same Rust-computed constants from `_core.finfo`, no new Python
        # arithmetic.
        #
        # `tiny` specifically: numpy's `tiny` cached_property does not hold
        # its own value at all -- it literally `return self.smallest_normal`
        # (an alias, per its own docstring: "alias of smallest_normal").
        # Mirrored exactly below rather than exposing the separately-read
        # `tiny` tuple field (`_core.finfo` still returns it identical to
        # `smallest_normal` -- Rust's `FInfo` table already sets both to the
        # same constant -- but the PROPERTY reads through `smallest_normal`,
        # matching numpy's actual alias structure, not merely its value).
        self.eps = ctor(eps)
        self._epsneg = ctor(epsneg)
        self.max = ctor(max_)
        self.min = ctor(min_)
        self.smallest_normal = ctor(smallest_normal)
        self.smallest_subnormal = ctor(smallest_subnormal)
        self._resolution = ctor(resolution)
        self.precision = precision
        self.bits = bits
        self._iexp = iexp
        self._nexp = nexp
        self.nmant = nmant
        self._machep = machep
        self._negep = negep
        self.minexp = minexp
        self.maxexp = maxexp
        self.dtype = _core.promote_types(name, name)

    @property
    def epsneg(self):
        return self._epsneg

    @property
    def resolution(self):
        return self._resolution

    @property
    def iexp(self):
        return self._iexp

    @property
    def nexp(self):
        return self._nexp

    @property
    def machep(self):
        return self._machep

    @property
    def negep(self):
        return self._negep

    @property
    def tiny(self):
        return self.smallest_normal

    def __repr__(self):
        # Mirrors real numpy 2.5.1's `finfo.__repr__` (`getlimits.py`):
        # `.min`/`.max` are formatted with the dtype's `_MACHAR_PARAMS` `%`
        # spec and `.strip()`ped; `.resolution` always goes through plain
        # `str()`, never the `%` spec, even though it is float-dtype-typed
        # too -- that asymmetry is numpy's own, reproduced here verbatim,
        # not a shortcut of ours.
        #
        # numpy's repr computes `c = self.__class__.__name__`, which for
        # numpy's own class is literally the string "finfo" (numpy's class
        # is `numpy.finfo`, lowercase). This class is ALSO named `finfo`
        # now (renamed from `FInfo` 2026-08-06, task #28 -- see the class
        # docstring), so `self.__class__.__name__` is used directly rather
        # than the literal `"finfo("` text a prior version of this file
        # hardcoded specifically to paper over the `FInfo`/`finfo` name
        # mismatch -- that workaround is gone along with the mismatch it
        # existed for.
        fmt = _FINFO_REPR_FMT.get(self.dtype.name)
        if fmt is not None:
            max_str = (fmt % self.max).strip()
            min_str = (fmt % self.min).strip()
        else:
            max_str = str(self.max)
            min_str = str(self.min)
        resolution_str = str(self.resolution)
        return (
            f"{self.__class__.__name__}(resolution={resolution_str}, "
            f"min={min_str}, max={max_str}, dtype={self.dtype})"
        )

    def __str__(self):
        # Verbatim transcription of real numpy 2.5.1's `finfo.__str__`
        # (`numpy/_core/getlimits.py`), added 2026-08-08 (ticket #71) --
        # anionpy had no `__str__` at all before this, so `str(finfo_obj)`
        # fell back to `object.__str__`, which for a class defining
        # `__repr__` but not `__str__` calls `__repr__` -- i.e. `str()` and
        # `repr()` returned the SAME short one-line text, where real numpy's
        # `str()` is a completely different multi-line machine-parameters
        # table. Confirmed live this was the actual prior behaviour, not
        # assumed.
        #
        # NOTE on a subtlety in numpy's own source: `get_str(name, pad)`
        # computes `s = str(val).ljust(pad)` when `pad is not None` but then
        # unconditionally `return str(val)` -- the padded `s` is dead code,
        # so real numpy's `__str__` output is UNPADDED plain `str()` of
        # every field despite the `pad=6`/`pad=3` arguments littered through
        # the call sites. Reproduced here as plain `str()` throughout,
        # matching that actual (not documented-looking) behaviour, verified
        # byte-for-byte against `str(np.finfo(...))` for float16/32/64.
        def s(name):
            val = getattr(self, name, None)
            return "<undefined>" if val is None else str(val)

        min_ = "-max" if -self.min == self.max else s("min")
        return (
            f"Machine parameters for {self.dtype}\n"
            f"---------------------------------------------------------------\n"
            f"precision = {s('precision')}   resolution = {s('resolution')}\n"
            f"machep = {s('machep')}   eps =        {s('eps')}\n"
            f"negep =  {s('negep')}   epsneg =     {s('epsneg')}\n"
            f"minexp = {s('minexp')}   tiny =       {s('tiny')}\n"
            f"maxexp = {s('maxexp')}   max =        {s('max')}\n"
            f"nexp =   {s('nexp')}   min =        {min_}\n"
            f"smallest_normal = {s('smallest_normal')}   "
            f"smallest_subnormal = {s('smallest_subnormal')}\n"
            f"---------------------------------------------------------------\n"
        )
