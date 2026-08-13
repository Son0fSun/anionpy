"""anionpy.ma.core -- the `MaskedArray` core type (Phase 0) and the mechanical
wrapper families (Phase 1 / Phase 2) built on top of it.

Scope note (see /Users/rabite/Monday/ionp/MA-DESIGN.md): this module covers
Phases 0-2 only. Mask CONSTRUCTION helpers (`masked_where`, `make_mask*`,
fill-value machinery, contiguity utilities) and the five mutating functions
are OUT OF SCOPE here and are not implemented.

THE ARCHITECTURAL RULE: Python here does dispatch and attribute plumbing
only. Every actual number is produced by an existing, already-Rust-backed
anionpy function/ufunc (`anionpy.sin`, `anionpy.add`, `anionpy.logical_or`,
`anionpy.where`, `anionpy.full`, `anionpy.repeat`, ...) -- see each wrapper factory's
docstring for exactly which Rust-backed call does the work. No Python `for`
loop over array elements exists anywhere in this file.

A `MaskedArray` is a PAIR of ordinary `anionpy.ndarray`s (data, mask), never a
special buffer -- `mask` does not share memory with `data`. `nomask` is a
dedicated singleton distinct from an explicit all-False mask array: passing
no `mask=` argument (or `mask=None`) produces `nomask`; passing an explicit
mask (even `False` or all-zeros) produces a real boolean array. This was
verified directly against real numpy 2.5.1 (see this task's probe scripts),
not assumed -- MA-DESIGN.md does not spell out this distinction and it is
easy to get backwards.
"""
from __future__ import annotations

import math as _math

import anionpy as _anionpy
from anionpy import array as _array, ndarray as _ndarray, bool_ as _bool_dtype

_UNSET = object()


def _is_bool_dtype(dt) -> bool:
    return dt.name == "bool"


def _box_typed_scalar(value, dtype):
    """Box `value` as a scalar of `dtype`, returning a GENUINE numpy-family
    scalar object (`numpy.float64`, `numpy.uint64`, `numpy.bool`, ...), not
    one of anionpy's OWN scalar-hierarchy objects (`anionpy.float64`, ...).

    ROOT CAUSE this fixes (measured live, not assumed -- this task's own
    probe): every other scalar-returning path off a `MaskedArray`
    (`.mean()`, `.max()`, `anionpy.sum(...)` on a 0-d result, plain
    indexing `anionpy.array([1.0])[0]`) already returns a genuine
    `numpy.float64`-family object -- confirmed live these come straight out
    of the already-Rust-backed `anionpy` extension itself (indexing/
    reduction), never through Python. The PREVIOUS revision of this helper
    (`_scalar_type_for`, now removed) instead boxed through anionpy's own
    top-level scalar namespace (`anionpy.float64(value)`), which
    constructs anionpy's OWN scalar class -- verified live
    `issubclass(anionpy.float64, numpy.generic)` is `False`, so
    `type(np_out) is type(ionp_out)` (the contract
    `harness._compare_scalar_like` enforces, see tests/differential/
    harness.py) can never pass for a `fill_value` read, even when the
    VALUE matched exactly. This was the actual, measured failure --  not a
    value bug at all.

    Fix: route the same single-element-array-then-index path every other
    scalar-producing call in this file already goes through (no Python
    arithmetic, no numpy import -- `_array`/`__getitem__` are the existing
    Rust-backed primitives): `_array([value], dtype=dtype)[0]`. Verified
    live this returns `numpy.<dtype>` for every dtype kind this module
    touches (bool/int*/uint*/float*/complex*), with numeric casting
    (float->int truncation, etc.) matching real numpy's own array
    construction (`anionpy.array([-7.0], dtype='int32')[0]` is `-7`, same
    as `numpy.array([-7.0], dtype='int32')[0]`).

    INGESTION GAP FOUND WHILE VERIFYING THIS FIX OUT OF CORPUS (not by
    inspection): `anionpy.array()` only ingests bare Python
    bool/int/float/complex (or a numpy.ndarray) -- a `list` containing a
    NUMPY or ANIONPY scalar OBJECT raises `TypeError` (verified live:
    `anionpy.array([numpy.int64(999999)], dtype='int64')` raises
    `"anionpy.array() only supports (possibly nested) lists/tuples of
    bool/int/float/complex, or a numpy.ndarray"`). This bites for real
    once THIS fix is in place: `.fill_value` now returns a genuine
    `numpy.<dtype>` scalar, and several existing callers
    (`masked_invalid`/`vstack`/`hstack`/`dstack`/`column_stack`/
    `diagflat`/`alltrue`/`sometrue`/...) construct a new `MaskedArray`
    with `fill_value=<an existing array's already-read .fill_value>` --
    which used to be one of anionpy's OWN scalar objects (accepted
    unremarked by the old `_scalar_type_for(dtype)(value)` boxing) and is
    now a numpy scalar hitting this exact ingestion gap. `.item()`
    unwraps EITHER a numpy or an anionpy scalar (both types implement it,
    verified live) down to a bare Python bool/int/float/complex that
    `_array` accepts, with no precision cost -- the actual width/kind cast
    still happens via this function's own `dtype=` argument below,
    unaffected by what native-Python type the unwrapped value lands as."""
    if hasattr(value, "item"):
        value = value.item()
    return _array([value], dtype=dtype)[0]


def _default_fill_value(dtype):
    """Minimal default-fill-value table, verified live against real numpy
    2.5.1 for exactly the dtype kinds this task's declared items touch
    (bool/int*/uint*/float*/complex*). This is NOT `ma.default_fill_value`
    (that item, and the rest of the fill-value machinery, is Phase 4 --
    explicitly out of scope for this task) -- it exists only so
    `MaskedArray.fill_value` has a correct default to report on
    construction/lazy recompute. Verified: bool_->True, every SIGNED
    integer width->999999 typed int64 (real numpy reports this as its own
    int64 scalar regardless of the source array's integer width --
    confirmed live: `np.ma.masked_array([1], dtype=np.int32).fill_value` is
    `999999` typed `int64`, not `int32`), every UNSIGNED integer
    width->999999 typed uint64 (a DIFFERENT default width than the signed
    case -- confirmed live `np.ma.masked_array([1],
    dtype=np.uint8).fill_value` is `999999` typed `uint64`, not `int64`;
    FOUND AND FIXED in this task's own pass -- an earlier revision of this
    function used the same 999999-typed-int64 default for BOTH int and
    uint dtype families, a real, pre-existing divergence on unsigned input
    this task's own "vary dtypes including uint" verification mandate
    would otherwise have silently sailed past), every float width->1e20
    typed float64 (never narrowed to the source's own float32/float16
    width -- confirmed live), every complex width->(1e20+0j) typed
    complex128 (same never-narrowed rule). Returns a GENUINE numpy-family
    scalar (via `_box_typed_scalar`, see its docstring for the boxing-bug
    root cause this fixes), not one of anionpy's own scalar-hierarchy
    objects and not a bare Python bool/int/float/complex -- needed so
    `np.asarray(fill_value).dtype.name` reports the correct default width
    for every one of these cases, including the uint one a bare Python
    `int` could never represent (`np.asarray(999999).dtype` is always
    `int64`), AND so `type(fill_value)` matches real numpy's own
    `numpy.<dtype>` scalar class, not anionpy's.
    """
    name = dtype.name
    if name == "bool":
        return _box_typed_scalar(True, "bool")
    if name.startswith("uint"):
        return _box_typed_scalar(999999, "uint64")
    if name.startswith("int"):
        return _box_typed_scalar(999999, "int64")
    if name.startswith("float"):
        return _box_typed_scalar(1e20, "float64")
    if name.startswith("complex"):
        return _box_typed_scalar(complex(1e20, 0.0), "complex128")
    raise TypeError(f"anionpy.ma: no default fill_value known for dtype {name!r}")


def _cast_fill_value(value, dtype):
    """Cast an explicitly-given `fill_value=` scalar to the array's OWN
    dtype, matching real numpy (verified live: `masked_array([1,2,3],
    dtype=int32, fill_value=-7.0).fill_value` reports `-7` typed `int32`,
    NOT forced to `int64` the way the untyped DEFAULT table's int entry is
    -- that int64-regardless-of-width quirk is `_default_fill_value`'s own,
    documented there, and does not apply here). This is a single scalar
    cast, not a per-element array loop -- the "no arithmetic in Python over
    array elements" rule concerns vectorized data, not one bookkeeping
    scalar.

    Returns a GENUINE numpy-family scalar (via `_box_typed_scalar`), e.g.
    `numpy.float32(-7.0)` for a `float32` array -- NOT a bare Python
    `float`/`int`, and NOT one of anionpy's own scalar-hierarchy objects
    either (see `_box_typed_scalar`'s docstring for that root cause). This
    is what fixes the CONSTRUCTOR GAP
    this task's report documents: `masked_array(float32_arr,
    fill_value=999999.0).fill_value` must report `(999999.0, 'float32')`
    like real numpy, not `(999999.0, 'float64')` -- a bare Python `float`
    always reads back as `float64` through `np.asarray`, regardless of
    which array dtype it was "cast" to by a plain `float(value)` call (the
    previous revision's bug: `float(value)`/`int(value)` discard width
    information a bare Python scalar cannot carry).

    Bug this fixes (earlier, still-valid finding): an earlier revision
    stored `fill_value` uncast, so e.g. `ma.logical_not(masked_array([1,0],
    fill_value=-7.0))`'s output fill_value stayed the raw Python `-7.0`
    instead of round-tripping through the INPUT array's int dtype first
    (`-7`) the way real numpy's does before the unary/binary families ever
    inherit it -- caught via live differential probing
    (`ma.logical_not`/`ma.logical_and`/etc. fill_value mismatches on
    `fv_custom` cases). Still fixed here: `_box_typed_scalar(value, dtype)`
    performs the same narrowing cast `int(value)`/`float(value)` did, just
    via the width-preserving, genuine-numpy-scalar-producing helper above
    instead of a bare Python type (or, as of this task's fix, instead of
    anionpy's own scalar constructor -- see `_box_typed_scalar`'s
    docstring).
    """
    return _box_typed_scalar(value, dtype)


class MaskedArray:
    """Phase 0 core type: a (data, mask, fill_value) triple.

    `data` is always a real `anionpy.ndarray`. `mask` is either the `nomask`
    singleton (identity-preserved: `a.mask is nomask`) or a real boolean
    `anionpy.ndarray` of the same shape as `data` -- never the same buffer as
    `data` (verified: `mask` is constructed independently, see
    `ionp-py`/`ionp-core`'s own buffer model -- MaskedArray never touches
    that layer at all, it only holds two independent `anionpy.ndarray`s).
    """

    def __init__(self, data, mask=_UNSET, fill_value=None, dtype=None):
        # EXPLICITNESS TRACKING (this task): a `fill_value=` kwarg genuinely
        # given to THIS construction call marks the result explicit --
        # verified live this is real numpy's own rule (`MaskedArray.
        # _fill_value` stays `None` internally until a user EXPLICITLY sets
        # one, constructor `fill_value=` or the `.fill_value` setter; an
        # untouched array's `.fill_value` property computes
        # `default_fill_value(self.dtype)` freshly on EVERY access -- see
        # this task's report for the live probes establishing this). The
        # NESTED-MaskedArray-input case below is the one place a `None`
        # `fill_value=` argument can still end up explicit: constructing
        # FROM an already-explicit source re-applies that source's stored
        # value through the ordinary cast-to-THIS-instance's-dtype path
        # below (verified live this recasts to a PROMOTED dtype, e.g.
        # `masked_array(int64_source_with_explicit_fv, dtype='float64')`
        # reports the new float64-cast value -- unlike the raw,
        # never-recast carry `_update_from` below does for ufunc-style
        # propagation, see its own docstring for why those two paths
        # genuinely differ).
        explicit_fv = fill_value is not None
        if isinstance(data, MaskedArray):
            inner = data
            data = inner.data
            if mask is _UNSET:
                mask = inner.mask
            if fill_value is None and inner._fill_value_explicit:
                fill_value = inner._fill_value
                explicit_fv = True

        if not isinstance(data, _ndarray):
            data = _array(data) if dtype is None else _array(data, dtype=dtype)
        elif dtype is not None:
            data = data.astype(dtype)
        self._data = data

        if mask is _UNSET or mask is nomask:
            # Only a genuinely OMITTED mask= kwarg (or nomask passed back in,
            # e.g. from another MaskedArray's .mask) collapses to the
            # `nomask` singleton. Verified live against real numpy 2.5.1:
            # `MaskedArray.__new__`'s own default parameter is `mask=np.False_`
            # (a bare scalar sentinel, not None) and only THAT default path
            # produces `nomask` identity -- explicitly passing `mask=None`
            # measurably does NOT (`np.ma.masked_array([1.,2.], mask=None).mask
            # is np.ma.nomask` is False; it materializes a real all-False
            # array, `array([False, False])`), exactly like `mask=False` and
            # an explicit all-False array already were known to. `None` is
            # therefore handled by the general fallback below, not treated as
            # "no mask" -- getting this backwards was an uncaught bug in an
            # earlier revision of this file.
            self._mask = nomask
        elif isinstance(mask, bool):
            # Explicit scalar mask -- still NOT nomask (verified live: real
            # numpy's `mask=False` produces a real all-False array, not the
            # `nomask` singleton). Broadcast via `anionpy.full`, a single
            # Rust-backed call, not a Python fill loop.
            self._mask = _anionpy.full(data.shape, mask, dtype=_bool_dtype)
        elif mask is None:
            # Verified live (see the block comment above): `mask=None`
            # produces a real all-False array, the same content `mask=False`
            # would, just reached via a different input value.
            self._mask = _anionpy.full(data.shape, False, dtype=_bool_dtype)
        elif isinstance(mask, _ndarray):
            self._mask = mask if _is_bool_dtype(mask.dtype) else mask.astype(_bool_dtype)
        else:
            self._mask = _array(mask, dtype=_bool_dtype)

        self._fill_value_explicit = explicit_fv
        if explicit_fv:
            self._fill_value = _cast_fill_value(fill_value, self._data.dtype)
        else:
            # NOT eagerly computed: `self._fill_value` stays `None`, exactly
            # like real numpy's own internal `_fill_value` field before any
            # explicit set -- the `.fill_value` PROPERTY below computes the
            # default fresh, every access, off whatever `self._data.dtype`
            # is AT THAT MOMENT (verified live this is what makes the
            # property "lazy" -- see the property's own docstring).
            self._fill_value = None

    # -- attributes (Phase 0) ------------------------------------------------
    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        return self._mask

    @property
    def fill_value(self):
        # LAZY BY DESIGN (this task's core fix): an explicitly-set
        # `_fill_value` is returned verbatim (whatever dtype it was cast to
        # at SET time, which may now differ from `self._data.dtype` if it
        # was carried across a promoting op via `_update_from` below --
        # verified live real numpy does the same, an explicit fill_value
        # set on an int64 array survives a promoting `ma.log` call still
        # reporting `int64`, not recast to the result's own float64). A
        # NEVER-explicitly-set instance instead computes
        # `_default_fill_value(self._data.dtype)` FRESH on every single
        # access -- NOT cached at construction time -- so a promoted-dtype
        # result of an op on an untouched source correctly reports ITS OWN
        # dtype's default (e.g. float64's 1e20), never a stale pre-
        # promotion default (e.g. int64's 999999) -- this is the exact bug
        # this task's report documents and fixes (previously
        # `anionpy.ma.log(int64_masked_array).fill_value` was `999999.0`
        # instead of numpy's `1e+20`).
        if self._fill_value_explicit:
            return self._fill_value
        return _default_fill_value(self._data.dtype)

    @fill_value.setter
    def fill_value(self, value):
        # Backs `ma.set_fill_value` (Phase 4) -- a plain Python attribute
        # write on this wrapper, casting the new scalar to the array's own
        # dtype exactly like the constructor's explicit `fill_value=` path
        # (`_cast_fill_value`, already used above). Never touches the
        # underlying Rust `anionpy.ndarray` buffer -- see `set_fill_value`'s
        # docstring in `anionpy/ma/core.py` for why this is NOT gated by the
        # Phase-6 in-place-mutation blocker. Marks this instance explicit,
        # same as if the value had been passed to the constructor.
        self._fill_value = _cast_fill_value(value, self._data.dtype)
        self._fill_value_explicit = True

    def _update_from(self, source):
        """Internal-only propagation helper, used by the ufunc-style
        wrapper families (`make_masked_unary`/`make_masked_binary`/
        `make_masked_domained_unary`/`make_masked_domained_binary`) instead
        of passing a computed `fill_value=` value through the public
        constructor. Mirrors real numpy's `MaskedArray._update_from`
        (CPython source): `self._fill_value = obj._fill_value` -- a DIRECT
        raw-attribute copy, NOT a call through the `.fill_value` SETTER
        (which would `_cast_fill_value` to `self`'s -- possibly promoted --
        dtype). This is the mechanism that makes an inherited-explicit
        fill_value survive a promoting op UNCHANGED, dtype and all
        (verified live, see the `fill_value` property's own docstring for
        the exact reproducing case), while an inherited-NON-explicit source
        contributes nothing (`source._fill_value` is `None`, `self`
        becomes non-explicit too, and `self.fill_value` computes its OWN
        dtype's fresh default on next access) -- exactly the propagation
        rule real numpy's own `_update_from` implements. Returns `self` so
        call sites can `return MaskedArray(...)._update_from(source)` in
        one expression."""
        self._fill_value_explicit = source._fill_value_explicit
        self._fill_value = source._fill_value
        return self

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    # -- ma-phase5-2026-08-08: layout/metadata attrs -----------------------
    # `itemsize`/`nbytes`/`strides` verified live (real numpy 2.5.1) to
    # depend ONLY on `self._data` (the mask array's own itemsize/nbytes are
    # never consulted for these three -- checked against a partially-masked
    # `int32` receiver where `.nbytes` reported the DATA-only byte count,
    # not data+mask). All three delegate to `anionpy.ndarray.itemsize`/
    # `.nbytes`/`.strides`, which are already declared exact toplevel (see
    # `anionpy/_state/ndarray.py`) -- no new arithmetic, pure passthrough.
    @property
    def itemsize(self):
        return self._data.itemsize

    @property
    def nbytes(self):
        return self._data.nbytes

    @property
    def strides(self):
        return self._data.strides

    # `.flags` (the top-level MaskedArray attribute, not `.data.flags`) is
    # DELIBERATELY NOT implemented here -- see `anionpy/_state/ma.py`'s
    # "ma.MaskedArray.flags" decline note. Real numpy's `MaskedArray` is an
    # `ndarray` SUBCLASS, so `a.flags.owndata` is `False` for EVERY masked
    # array (verified live: fresh construction, view construction, all
    # report `owndata=False` uniformly -- an artifact of `.data` always
    # being reached via `.view(ndarray)` in real numpy's implementation,
    # not of provenance). anionpy's `MaskedArray` is a plain (data, mask)
    # pair, not an ndarray subclass -- `self._data` is genuinely OWNED, so
    # `self._data.flags.owndata` is `True` for a fresh construction. A naive
    # `return self._data.flags` would silently mismatch `owndata` on every
    # single case (a uniform mismatch, which is itself the finding here,
    # not a reason to paper over it with a hand-built flags-like value).
    # `.iscontiguous()` below sidesteps this: it only reads `C_CONTIGUOUS`,
    # which real numpy verified live does NOT depend on the owndata/view
    # distinction (fortran-order and sliced-view receivers both correctly
    # report `False` on real numpy, matching a direct C_CONTIGUOUS read).
    def iscontiguous(self):
        return bool(self._data.flags["C_CONTIGUOUS"])

    # `get_fill_value()` is a plain zero-arg alias for the already-exact
    # `.fill_value` property (verified live against real numpy's own
    # `MaskedArray.get_fill_value` source: `return self.fill_value`,
    # nothing else -- no new fill-value logic here).
    def get_fill_value(self):
        return self.fill_value

    def __repr__(self):
        mask_repr = "False" if self._mask is nomask else repr(self._mask)
        return (
            f"MaskedArray(data={self._data!r}, mask={mask_repr}, "
            f"fill_value={self._fill_value!r})"
        )

    # -- Dunders (this task): indexing/sequence protocol ------------------
    # Verified live against real numpy 2.5.1 (`MaskedArray.__getitem__`,
    # CPython source): a masked SCALAR position (result of indexing down to
    # a single element, e.g. `a[1]` on a 1-d array) returns the `masked`
    # singleton itself, NOT a fresh `MaskedConstant` -- confirmed live
    # (`a[1] is np.ma.masked` is `True`). An UNmasked scalar position
    # returns the bare underlying scalar (whatever `self._data[idx]` itself
    # produces -- already a genuine numpy-family scalar, verified live this
    # task). A sub-array result (slice/fancy/multi-dim indexing that still
    # has `.ndim > 0`) wraps in a fresh `MaskedArray`, inheriting
    # `fill_value` via `_update_from(self)` (verified live: `a[0:2
    # ].fill_value` equals `a.fill_value`, not a freshly-defaulted one) --
    # and if `self._mask is nomask`, the result's mask stays `nomask` too
    # (identity preserved, verified live), never materialized just because
    # indexing happened.
    def __getitem__(self, idx):
        d = self._data[idx]
        if self._mask is nomask:
            if not isinstance(d, _ndarray):
                return d
            return MaskedArray(d, mask=nomask)._update_from(self)
        m = self._mask[idx]
        if not isinstance(d, _ndarray):
            return masked if bool(m) else d
        return MaskedArray(d, mask=m)._update_from(self)

    # `__len__`/`__bool__`/`.T` are verified live to depend ONLY on
    # `self._data` -- the mask plays no role at all (a masked-but-nonzero
    # element is still truthy, `bool()` raises the identical
    # `ValueError`/`TypeError` real numpy's plain `ndarray.__bool__` raises
    # for empty/multi-element/0-d inputs, since anionpy's own `ndarray`
    # already reproduces those exact messages -- verified live this task,
    # see this task's report). `.T` reuses this file's own already-declared
    # `transpose` (Phase 3), which already handles nomask-identity/mask-
    # transpose/fill_value propagation correctly.
    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return bool(self._data)

    @property
    def T(self):
        return transpose(self)

    # -- ma batch 5: shape/layout METHODS ----------------------------------
    # Mirrors real numpy's own `numpy.ma.core._arraymethod('<name>')`
    # factory (CPython source, `numpy/ma/core.py`): `copy`/`diagonal`/
    # `flatten`/`repeat`/`squeeze`/`swapaxes`/`transpose` are ALL built
    # from the SAME shape -- call the identically-named method on
    # `self._data`, then (mask permitting) the identically-named method on
    # `self._mask`, wrap the pair in a fresh `MaskedArray`, and carry
    # `fill_value` across via `_update_from`. `reshape`/`ravel` are NOT
    # built on `_arraymethod` in real numpy either (confirmed:
    # `inspect.getsource(np.ma.core.MaskedArray.reshape)` is a bespoke
    # method, not a `_arraymethod(...)` assignment) but their body is
    # observably the identical shape, so they route through this same
    # helper too -- `reshape`'s only difference is accepting `*args`
    # (shape given as separate ints OR one tuple), which this helper
    # already handles since it forwards `*args, **kwargs` verbatim to
    # `getattr(self._data, funcname)`.
    #
    # Verified live (this task) that `MaskedArray.__init__` does NOT copy
    # an already-`anionpy.ndarray` `data=`/`mask=` argument (`self._data =
    # data`, no `.copy()` call -- read directly from the constructor
    # source above), so whatever view-vs-copy relationship
    # `getattr(self._data, funcname)(...)` itself produces (each of these
    # funcnames is ALREADY declared exact at the plain-`ndarray` level in
    # `anionpy/_state/ndarray.py`, view semantics included) survives
    # unchanged through this wrap -- this helper does not need to (and
    # does not) re-derive view/copy behavior itself.
    #
    # `diagonal` is DELIBERATELY NOT included here: `anionpy/_state/
    # ndarray.py`'s `ndarray.diagonal` entry is REVOKED (2026-08-02,
    # measured wrong strides / a genuine copy where real numpy returns an
    # aliasing view) -- a `MaskedArray.diagonal` built on this same helper
    # would silently inherit that same defect. No `diagonal` method is
    # defined on this class; `ma.MaskedArray.diagonal` stays absent.
    def _masked_arraymethod(self, funcname, *args, **kwargs):
        data = getattr(self._data, funcname)(*args, **kwargs)
        if self._mask is nomask:
            mask = nomask
        else:
            mask = getattr(self._mask, funcname)(*args, **kwargs)
        return MaskedArray(data, mask=mask)._update_from(self)

    def copy(self, *args, **kwargs):
        return self._masked_arraymethod("copy", *args, **kwargs)

    def flatten(self, *args, **kwargs):
        return self._masked_arraymethod("flatten", *args, **kwargs)

    def squeeze(self, *args, **kwargs):
        return self._masked_arraymethod("squeeze", *args, **kwargs)

    def swapaxes(self, axis1, axis2):
        return self._masked_arraymethod("swapaxes", axis1, axis2)

    def transpose(self, *axes):
        return self._masked_arraymethod("transpose", *axes)

    def reshape(self, *shape, **kwargs):
        return self._masked_arraymethod("reshape", *shape, **kwargs)

    def ravel(self, order="C"):
        # Verified live against `inspect.getsource(np.ma.core.
        # MaskedArray.ravel)`: 'K'/'A'/'k'/'a' all normalize to 'F' if
        # `self._data.flags.fnc` (F-contiguous AND NOT C-contiguous) else
        # 'C', BEFORE either data or mask is raveled -- applied identically
        # to both (real numpy's own comment: "The order of _data and _mask
        # could be different ... So we ignore the mask memory order").
        if order in ("K", "k", "A", "a"):
            order = "F" if self._data.flags.fnc else "C"
        return self._masked_arraymethod("ravel", order=order)

    @property
    def mT(self):
        # Bespoke property, NOT built on `_masked_arraymethod`: verified
        # live against `inspect.getsource(np.ma.core.MaskedArray.mT.fget)`
        # that real numpy's `mT` does NOT call `_update_from` on either
        # branch -- a measured, genuine exception to this family's usual
        # fill_value-inheritance rule (confirmed live: `x.mT.fill_value`
        # reports the plain DEFAULT even when `x.fill_value` was set
        # explicitly, while `x.T.fill_value`/`x.reshape(n).fill_value`
        # both correctly carry the explicit value through). The ndim<2
        # guard's message is `self._data.mT`'s own (already-declared-exact
        # `ndarray.mT`) error, reached by just delegating -- not
        # reimplemented here.
        data = self._data.mT
        mask = nomask if self._mask is nomask else self._mask.mT
        return MaskedArray(data, mask=mask)

    def conj(self):
        return self._conjugate_method()

    def conjugate(self):
        return self._conjugate_method()

    def _conjugate_method(self):
        # Bespoke, NOT `_arraymethod`-based: confirmed live
        # `numpy.ma.core` source has no `conj`/`conjugate =
        # _arraymethod(...)` assignment. Verified live: a non-complex
        # receiver's `.conj()` returns `self` UNCHANGED (`x.conj() is x`,
        # real numpy) -- no copy, no rewrap. A complex receiver conjugates
        # via `self._data.conj()`, the ndarray METHOD (already K-order-
        # fixed for non-contiguous/transposed inputs, see commit
        # `4b2d1bc`) -- NOT the top-level `_anionpy.conjugate` ufunc this
        # file's unrelated module-level `ma.conjugate` (Phase 1) function
        # uses. The mask is always materialized (`getmaskarray`, matching
        # real numpy's own `conjugate = _MaskedUnaryOperation`-adjacent
        # convention of never leaving a computed result on `nomask` for
        # this family) and fill_value inherits via `_update_from`. A 0-d
        # result collapses to the `masked` singleton ONLY when it is
        # itself genuinely masked (verified live: an UNMASKED 0-d complex
        # receiver's `.conj()` stays a genuine 0-d `MaskedArray`; a MASKED
        # one collapses -- `r2 is ma.masked` is `True` on real numpy).
        if self._data.dtype.kind != "c":
            return self
        data = self._data.conj()
        mask = getmaskarray(self)
        if data.ndim == 0 and bool(mask.item()):
            return masked
        return MaskedArray(data, mask=mask)._update_from(self)

    # -- Dunders (this task): arithmetic -----------------------------------
    # Verified DIRECTLY against `numpy.ma.core.MaskedArray`'s CPython
    # source (`__add__`/`__radd__`/.../`__rfloordiv__`, lines ~4306-4380):
    # every one of these is LITERALLY a thin call to this file's own
    # already-declared, already-verified module-level function
    # (`add`/`subtract`/`multiply`/`true_divide`/`floor_divide`) -- not a
    # new algorithm. `__pow__`/`__rpow__` are DELIBERATELY excluded here:
    # they call `power`, which this file does not declare (see
    # `anionpy/_state/ma.py`'s module docstring for why), so the dunder
    # stays undeclared too rather than being built on an undeclared base.
    # `_delegate_binop`'s NotImplemented short-circuit (real numpy: used
    # for interop with foreign ndarray subclasses carrying a higher
    # `__array_priority__`) is not reproduced -- out of this task's
    # verification budget, and anionpy has no `__array_priority__`
    # machinery for it to matter against.
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __floordiv__(self, other):
        return floor_divide(self, other)

    def __rfloordiv__(self, other):
        return floor_divide(other, self)

    # -- Dunders (this task): comparisons ----------------------------------
    # Verified DIRECTLY against `MaskedArray._comparison`'s CPython source
    # (lines ~4193-4266) -- a GENUINELY DIFFERENT algorithm from this
    # file's already-declared `ma.equal`/`ma.not_equal`/etc module
    # functions (`_make_masked_binary_scalar_safe`): no data-revert at all
    # (`compare_fn` runs on the full raw data, always), mask is
    # `mask_or(self.mask, getmask(other))` (NOT a plain `logical_or` --
    # `mask_or` shrinks an all-False combination back to `nomask`,
    # verified live this task), and ONLY `__eq__`/`__ne__` additionally
    # overwrite each MASKED position's boolean with "both sides masked ->
    # equal / one side masked -> unequal" (`eqne_fn(smask, omask)`) --
    # every other comparison leaves the raw (unreverted) compare result at
    # masked positions untouched. `fill_value` inherits via
    # `_update_from(self)` same as everywhere else, then is CAST TO BOOL
    # (falling back to bool's own default fill value, `True`, if the cast
    # fails) -- but ONLY when the inherited fill_value was already
    # explicit (verified live: real numpy's own guard is `if check.
    # _fill_value is not None`, which is exactly this file's
    # `_fill_value_explicit` flag).
    def __eq__(self, other):
        return _comparison_dunder(self, other, _anionpy.equal, eqne_fn=_anionpy.equal)

    def __ne__(self, other):
        return _comparison_dunder(self, other, _anionpy.not_equal, eqne_fn=_anionpy.not_equal)

    def __lt__(self, other):
        return _comparison_dunder(self, other, _anionpy.less)

    def __le__(self, other):
        return _comparison_dunder(self, other, _anionpy.less_equal)

    def __gt__(self, other):
        return _comparison_dunder(self, other, _anionpy.greater)

    def __ge__(self, other):
        return _comparison_dunder(self, other, _anionpy.greater_equal)

    # `__eq__`/`__ne__` overrides make this class unhashable by Python's own
    # default rule (defining `__eq__` without `__hash__` sets `__hash__` to
    # `None`) -- verified live this MATCHES real numpy (`hash(np.ma.
    # masked_array([1.]))` raises `TypeError: unhashable type`), so no
    # `__hash__` is defined here either; this is not an oversight.
    __hash__ = None

    # -- Dunders (this task): "generic ufunc dispatch" family -------------
    # Verified live against real numpy 2.5.1: `__neg__`/`__pos__`/
    # `__abs__`/`__invert__`/`__and__`/`__or__`/`__xor__` (and their
    # reflected binary forms) are NOT explicitly defined anywhere in
    # `numpy.ma.core.MaskedArray`'s own source (confirmed by direct grep --
    # zero matches) -- they are inherited from `ndarray`'s generic
    # subclass ufunc-dispatch machinery, which this task's report
    # establishes has YET ANOTHER distinct shape from both the plain
    # `ma.*` module functions (Phase 1's copyto-revert family) AND the
    # `_comparison`-based family just above:
    #   * UNARY (`neg`/`pos`/`abs`/`invert`): the computed value is kept
    #     EVERYWHERE (no revert at masked positions, verified live:
    #     `abs(masked_array([-2.], mask=[True])).data` is `[2.]`, not the
    #     original `[-2.]`), and the mask is ALWAYS materialized via
    #     `getmaskarray` -- even a `nomask` input's result mask is a real
    #     all-False array, NEVER `nomask` identity (verified live,
    #     asymmetric with the binary case below; this task's report
    #     documents the live measurement establishing this, root
    #     mechanism not chased further given this task's effort budget --
    #     the MEASURED behavior is what is reproduced here, trusted
    #     per this task's own "verified live, not guessed" bar).
    #   * BINARY (`and`/`or`/`xor`, reflected forms identical -- see
    #     `__rand__` etc below): mask is `mask_or(self.mask, other.mask)`
    #     (verified live: a real live probe combining two REAL all-False
    #     mask arrays via `&` collapses back to `nomask` identity, which a
    #     naive unshrunk `logical_or` would NOT do -- `mask_or`'s own
    #     shrink is what reproduces this). No data-revert (matches the
    #     unary case -- verified live).
    #   Both families: `fill_value` ALWAYS inherits from `self` alone
    #   (`_update_from(self)`) -- NOT the smart `_fv_source` "prefer
    #   whichever original operand was a MaskedArray" rule the Phase 1
    #   binary family uses. Verified live for the reflected forms too:
    #   `5 & a` (`a.__rand__(5)`) and `a & 5` (`a.__and__(5)`) both report
    #   `a`'s own fill_value, confirming the rule is "whichever instance's
    #   dunder METHOD is executing", not the left/right textual operand --
    #   which is also why `__rand__`/`__ror__`/`__rxor__` below are
    #   literally identical code to their non-reflected counterparts (AND/
    #   OR/XOR are commutative, and the fill_value rule does not depend on
    #   operand order either).
    def __neg__(self):
        return _generic_unary_dunder(self, _anionpy.negative)

    def __pos__(self):
        return _generic_unary_dunder(self, _anionpy.positive)

    def __abs__(self):
        return _generic_unary_dunder(self, _anionpy.absolute)

    def __invert__(self):
        return _generic_unary_dunder(self, _anionpy.invert)

    def __and__(self, other):
        return _generic_binary_dunder(self, other, _anionpy.bitwise_and)

    def __rand__(self, other):
        return _generic_binary_dunder(self, other, _anionpy.bitwise_and)

    def __or__(self, other):
        return _generic_binary_dunder(self, other, _anionpy.bitwise_or)

    def __ror__(self, other):
        return _generic_binary_dunder(self, other, _anionpy.bitwise_or)

    def __xor__(self, other):
        return _generic_binary_dunder(self, other, _anionpy.bitwise_xor)

    def __rxor__(self, other):
        return _generic_binary_dunder(self, other, _anionpy.bitwise_xor)

    # -- ma batch 4 (this task): __mod__/__rmod__/__imod__ -- also
    # inherited-unoverridden `ndarray` slot wrappers (verified live, same
    # way as `__and__`/`__or__`/`__xor__` above were confirmed), but
    # `np.remainder` carries a `_DomainSafeDivide` domain the bitwise ops
    # above do not -- see `_generic_domained_binary_dunder`'s own docstring
    # (below `_generic_binary_dunder`, further down this file) for the full
    # measured algorithm, why it is a genuinely different fill rule from
    # the module-level `ma.remainder`/`ma.mod` a few thousand lines below,
    # and why `__imod__` needed ITS OWN separate function rather than
    # reusing this one (a real, measured float-vs-int divergence in how
    # in-place mutation interacts with the domain check, not a shortcut).
    def __mod__(self, other):
        return _generic_domained_binary_dunder(self, other, _anionpy.remainder, _domain_safe_divide, 1)

    def __rmod__(self, other):
        return _generic_domained_binary_dunder(self, other, _anionpy.remainder, _domain_safe_divide, 1, reflect=True)

    def __imod__(self, other):
        return _generic_domained_binary_idunder(self, other, _domain_safe_divide, 1)

    # -- Phase 7: reduction methods -------------------------------------
    # Thin instance-method forwarders onto the module-level functions
    # defined in the "Phase 7" section near the end of this file (below
    # `MaskedArray`'s definition). This is a genuine forward reference:
    # Python resolves a bare name used INSIDE a method body against the
    # enclosing MODULE's globals at CALL time, not at class-definition
    # time, so this works even though `count`/`sum`/`any`/`all`/`min`/
    # `max`/`mean` are not yet defined when this class body executes --
    # verified by the fact this file already relies on the same principle
    # every existing method already does (`_default_fill_value`,
    # `_cast_fill_value`, etc. are all called before this class's own
    # `__init__` returns, from functions defined earlier in the file,
    # which is the ordinary/uncontroversial direction; this is simply the
    # same mechanism used in the other temporal direction). Every one of
    # these module-level functions is verified against real numpy 2.5.1
    # live -- see each one's own docstring below for the exact evidence.
    def count(self, axis=None, keepdims=False):
        return count(self, axis=axis, keepdims=keepdims)

    def sum(self, axis=None, dtype=None, keepdims=False):
        return sum(self, axis=axis, dtype=dtype, keepdims=keepdims)

    def any(self, axis=None, keepdims=False):
        return any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        return all(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return min(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        return max(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, dtype=None, keepdims=False):
        return mean(self, axis=axis, dtype=dtype, keepdims=keepdims)

    # -- Phase 8: additional method forms --------------------------------
    # Same forward-reference mechanism as the Phase 7 block above -- these
    # module-level functions are defined further down this file.
    def argmax(self, axis=None, fill_value=None, keepdims=False):
        return argmax(self, axis=axis, fill_value=fill_value, keepdims=keepdims)

    def argmin(self, axis=None, fill_value=None, keepdims=False):
        return argmin(self, axis=axis, fill_value=fill_value, keepdims=keepdims)

    def cumsum(self, axis=None, dtype=None):
        return cumsum(self, axis=axis, dtype=dtype)

    def cumprod(self, axis=None, dtype=None):
        return cumprod(self, axis=axis, dtype=dtype)

    def ptp(self, axis=None, keepdims=False):
        return ptp(self, axis=axis, keepdims=keepdims)

    # -- ma batch 3: thin forwarders onto already-declared module-level
    # `filled`/`compressed` (same forward-reference mechanism as Phase 7/8
    # above -- both are defined further down this file). Verified live
    # against real numpy 2.5.1: `inspect.signature(np.ma.MaskedArray.filled)`
    # -> `(self, fill_value=None)`; `inspect.signature(np.ma.MaskedArray.
    # compressed)` -> `(self)`.
    def filled(self, fill_value=None):
        return filled(self, fill_value=fill_value)

    def compressed(self):
        return compressed(self)

    # -- ma batch 3: dunders reverse-engineered live against real numpy
    # 2.5.1 (none of these are documented in MA-DESIGN.md; every claim
    # below was independently probed, not assumed from the module
    # docstring).
    def __iter__(self):
        # `np.ndarray.__iter__` (inherited, unoverridden by MaskedArray)
        # dispatches to `self[i]` for each `i` -- verified live: iterating
        # a partially-masked 1-D array yields the real `masked` singleton
        # at masked positions, exactly like `arr[i]` does; a 0-d array
        # raises `TypeError: iteration over a 0-d array` (a DIFFERENT
        # message from `len()`'s `"len() of unsized object"`, so this
        # cannot be implemented by delegating to `len(self)` first). This
        # loop calls the already-Rust-backed, already-declared-exact
        # `__getitem__` once per index -- it is dispatch/protocol, not a
        # numerical loop; no arithmetic happens in this function.
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        for i in range(self.shape[0]):
            yield self[i]

    def __contains__(self, item):
        # `np.ndarray.__contains__` (inherited) is documented as `bool(key
        # in self)` == `bool((self == key).any())` -- verified live this is
        # mask-aware ONLY because `__eq__`/`.any()` already are (both
        # already declared exact above/in Phase 7): a masked position's
        # comparison result is itself masked and therefore excluded by
        # `.any()`, and `bool(<the masked singleton>)` is `False` (matches
        # real numpy: `2 in masked_array([1,2,3], mask=[F,T,F])` is
        # `False` even though the raw, masked-out data at index 1 IS `2`).
        # No new mask logic here at all -- this is a pure composition of
        # two already-verified primitives.
        return bool((self == item).any())

    def __float__(self):
        # Real `MaskedArray.__float__` (CPython source, not inherited):
        # size>1 raises the same `TypeError` message as plain `float()` on
        # a multi-element ndarray; a masked scalar warns
        # ("Warning: converting a masked element to nan.") and returns
        # `nan`; otherwise returns `float(self.item())`. Verified live
        # against real numpy 2.5.1 (both the warning text and the nan
        # return, and that the warning is skipped entirely on unmasked/
        # nomask receivers).
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted to Python scalars")
        m = self.mask
        is_masked = m is not nomask and bool(m.item())
        if is_masked:
            import warnings

            warnings.warn("Warning: converting a masked element to nan.", stacklevel=2)
            return float("nan")
        return float(self.data.item())

    def __int__(self):
        # Real `MaskedArray.__int__` (CPython source, not inherited):
        # size>1 raises the same message as `__float__` above; a masked
        # scalar raises `MaskError('Cannot convert masked element to a
        # Python int.')` (no warning, unlike `__float__`); otherwise
        # returns `int(self.item())`. Verified live.
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted to Python scalars")
        m = self.mask
        is_masked = m is not nomask and bool(m.item())
        if is_masked:
            raise MaskError("Cannot convert masked element to a Python int.")
        return int(self.data.item())

    def __complex__(self):
        # `np.ndarray.__complex__` (inherited, unoverridden by MaskedArray)
        # -- verified live this is genuinely MASK-BLIND: `complex(<masked
        # scalar>)` returns the underlying data value untouched, no warning,
        # no exception, mask ignored entirely (confirmed: a masked `5+2j`
        # still converts to `(5+2j)`). Delegating straight to
        # `anionpy.ndarray.__complex__` (already declared exact) reproduces
        # this exactly, size-mismatch error message included, for free.
        return complex(self.data)

    def __index__(self):
        # `np.ndarray.__index__` (inherited, unoverridden) -- verified live,
        # same mask-blind story as `__complex__` above: a masked integer
        # scalar's `__index__()` returns its raw underlying int, mask
        # ignored (confirmed: `masked_array(5, mask=True).__index__()` is
        # `5`, not an error). Delegates straight to `anionpy.ndarray.
        # __index__` (already declared exact), which already owns the
        # correct error messages for non-integer dtype / size != 1.
        return self.data.__index__()

    def __copy__(self):
        # `np.ndarray.__copy__` on a MaskedArray subclass copies the WHOLE
        # object -- data AND mask AND fill_value, into independent memory
        # (verified live: mutating the copy's `.data` does not touch the
        # original's). Built from `.copy()` on each already-Rust-backed
        # `anionpy.ndarray` (data, and mask when it is a real array --
        # `nomask` is a singleton, not a buffer, so it is passed through
        # unchanged rather than "copied"), plus `_update_from` to carry the
        # explicit/non-explicit fill_value state, mirroring real numpy's
        # own `_update_from`-based copy path exactly.
        m = self.mask
        new_mask = nomask if m is nomask else m.copy()
        new = MaskedArray(self.data.copy(), mask=new_mask)
        new._update_from(self)
        return new

    def __deepcopy__(self, memo=None):
        # Verified live: real `MaskedArray.__deepcopy__` produces the same
        # observable result as `__copy__` above (independent data/mask
        # buffers, fill_value preserved) -- there is no nested-object graph
        # inside a MaskedArray's own `data`/`mask` (both are flat
        # `anionpy.ndarray`s) for "deep" to mean anything beyond what
        # `__copy__` already does, so this simply reuses it. `memo` is
        # accepted (the `copy.deepcopy` protocol always passes one) and
        # otherwise unused, matching real numpy's own signature shape.
        return self.__copy__()


class _NoMaskType:
    """Singleton sentinel for "no mask at all" -- distinct from any real
    all-False boolean array. `bool(nomask)` is False, matching real numpy's
    `nomask` (which is itself a falsy `numpy.bool_(False)`), but identity
    (`is`) is what every declared item's differential test actually checks
    (see MA-DESIGN.md section 7 / this task's brief): this class exists so
    that identity check has exactly one object to be true of.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "False"

    def __bool__(self):
        return False


nomask = _NoMaskType()


class MaskedConstant(MaskedArray):
    """`ma.masked` / `ma.masked_singleton` -- a singleton 0-d MaskedArray
    with `mask=True`. Verified live: `np.ma.masked is np.ma.masked_singleton`
    is True in real numpy; modeled here the same way, as one object bound to
    two names, not two instances that merely compare equal.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            obj = MaskedArray.__new__(cls)
            MaskedArray.__init__(obj, _array(0.0), mask=True)
            cls._instance = obj
        return cls._instance

    def __init__(self):
        # Real initialization happens once in __new__; __init__ still runs
        # every time `MaskedConstant()` is called (Python always calls
        # __init__ after __new__ returns an instance of the class), so this
        # is intentionally a no-op rather than re-running MaskedArray's
        # constructor on the singleton.
        pass

    def __repr__(self):
        return "masked"


masked = MaskedConstant()
masked_singleton = masked


# ---------------------------------------------------------------------------
# Phase 0 plumbing functions (getdata/getmask/getmaskarray/filled).
# ---------------------------------------------------------------------------
def getdata(a):
    """Return the underlying data array of a MaskedArray (or `anionpy.array(a)`
    for anything else)."""
    return a.data if isinstance(a, MaskedArray) else _array(a)


def getmask(a):
    """Return `a`'s mask -- `nomask` if `a` is unmasked or not a
    MaskedArray at all."""
    return a.mask if isinstance(a, MaskedArray) else nomask


def getmaskarray(a):
    """Like `getmask`, but never returns `nomask`: an unmasked input gets a
    real, freshly materialized all-False array (via `anionpy.full`, one
    Rust-backed call) of the correct shape."""
    m = getmask(a)
    if m is nomask:
        d = getdata(a)
        return _anionpy.full(d.shape, False, dtype=_bool_dtype)
    return m


def filled(a, fill_value=None):
    """Return a plain `anionpy.ndarray`: `a`'s data with every masked position
    replaced by `fill_value` (or `a.fill_value` if not given). Computed via
    `anionpy.where` (a single Rust-backed elementwise select), never a Python
    loop over positions."""
    if not isinstance(a, MaskedArray):
        return _array(a)
    m = a.mask
    if m is nomask:
        return a.data.copy()
    fv = fill_value if fill_value is not None else a.fill_value
    fv_arr = _anionpy.full(a.data.shape, fv, dtype=a.data.dtype)
    return _anionpy.where(m, fv_arr, a.data)


# ---------------------------------------------------------------------------
# Phase 1: the generic wrapper families.
#
# Every one of these is "call an already-Rust-backed anionpy function on the
# data, combine/propagate the mask via anionpy.logical_or (also Rust-backed)".
# No new Rust, no arithmetic in Python -- the mask combination IS elementwise
# boolean algebra and is delegated to the existing kernel exactly like
# MA-DESIGN.md requires.
# ---------------------------------------------------------------------------
def _as_masked(x):
    return x if isinstance(x, MaskedArray) else MaskedArray(x)


def _fv_source(a, b, am, bm):
    """Real numpy's `_update_from`-style SOURCE-SELECTION rule, shared by
    every binary-shaped family (plain `_MaskedBinaryOperation` and
    `_DomainedBinaryOperation` alike): prefer whichever ORIGINAL argument
    (the raw `a`/`b` BEFORE `_as_masked` wrapping) was already a
    `MaskedArray`, preferring `a` when both were; when NEITHER original
    argument was a `MaskedArray`, there is nothing genuine to prefer (both
    `am`/`bm` are freshly `_as_masked`-wrapped and non-explicit by
    construction anyway) and `am` is returned as a harmless default.

    THIS IS THE FIX for the "Phase 1 plain binary family's `fill_value=
    am.fill_value` unconditional inheritance asymmetry" this task's report
    documents: the previous `make_masked_binary` always inherited from
    `am` (`_as_masked(a)`) regardless of whether the ORIGINAL `a` was ever
    a `MaskedArray` to begin with -- confirmed live this diverges from real
    numpy whenever `a` is plain data and `b` is the actual masked operand
    (`np.ma.add([1.,2.,3.], np.ma.masked_array([1.,2.,3.],
    fill_value=-9.0)).fill_value` is `-9.0`, inherited from `b`, NOT the
    freshly-defaulted `1e20` a naive `am.fill_value`-always rule would
    produce since `am` there is just `[1.,2.,3.]` freshly wrapped with no
    explicit fill_value at all) -- verified live for both the plain binary
    family (`ma.add`) and this phase's own domained binary family
    (`ma.divide`), same rule, same fix, shared here rather than
    duplicated.
    """
    if isinstance(a, MaskedArray):
        return am
    if isinstance(b, MaskedArray):
        return bm
    return am


def make_masked_unary(fn):
    """_MaskedUnaryOperation family (non-domained members only -- `tan` is
    EXCLUDED from this family: real numpy's `ma.tan` carries a
    `_DomainTan` domain check the plain pass-through model below does not
    reproduce, so it is not declared at all, see `anionpy/_state/ma.py`).

    Verified directly against `numpy.ma.core._MaskedUnaryOperation.__call__`
    (CPython source, not guessed): `fn` is called on the FULL raw data
    (every position, masked or not), and the mask is simply the INPUT's
    mask, unchanged -- BUT the source then does
    `np.copyto(result, d, where=m)`: every position where the mask is True
    has its *data* reverted back to the ORIGINAL input value, discarding
    the just-computed result there. A first implementation of this file
    got this wrong (kept the computed value everywhere, mask pass-through
    only) -- caught by live differential probes against real numpy 2.5.1
    showing e.g. `ma.ceil(masked_array([0.1, ...], mask=[1,1,...])).data`
    equal to the ORIGINAL `[0.1, ...]`, not `[1.0, ...]`. Reproduced here
    via `anionpy.where(mask, original_data, computed)`, a single vectorized
    Rust-backed select -- not a Python loop over positions.
    """

    def wrapped(a):
        am = _as_masked(a)
        with _anionpy.errstate(divide="ignore", invalid="ignore"):
            computed = fn(am.data)
        m = am.mask
        if not isinstance(computed, _ndarray):
            # 0-d path -- verified directly against real numpy's CPython
            # source (`_MaskedUnaryOperation.__call__`, "Case 2.1: The
            # result is scalar"): `if not result.ndim: return masked if m
            # else result` -- a 0-d operand's result is NEVER wrapped in a
            # MaskedArray, it is either the `masked` singleton or a bare
            # scalar, same collapse rule `make_masked_domained_unary` above
            # already implements (see that factory's docstring and the
            # module comment on the 0-d quirk). `anionpy.<fn>` already
            # collapses a 0-d `anionpy.ndarray` operand to a bare scalar the
            # same way real numpy's ufuncs do (verified live), which is what
            # makes `isinstance(computed, _ndarray)` a reliable signal here,
            # with no separate shape check needed.
            is_masked = m is not nomask and bool(m.item())
            return masked if is_masked else computed
        if m is nomask:
            data = computed
        else:
            data = _anionpy.where(m, am.data, computed)
        # NOTE: must NOT do `fill_value=am.fill_value` here -- that reads
        # the (possibly lazily-defaulted) property on the INPUT array and
        # feeds it into the constructor as an EXPLICIT fill_value for the
        # OUTPUT array. When `am` was never given an explicit fill_value
        # and this op promotes dtype (e.g. `sin`/`cosh`/`exp` on an int
        # array -> float64), that bakes in the INPUT dtype's default
        # (e.g. int64's 999999) cast to the OUTPUT dtype, instead of
        # letting the OUTPUT dtype compute its OWN fresh default (float64's
        # 1e20) the way real numpy's `_update_from`-based propagation does.
        # `_update_from` carries the raw (possibly-None) `_fill_value` and
        # its explicitness flag verbatim, exactly like real numpy's
        # `_MaskedUnaryOperation.__call__` -> `result._update_from(a)`.
        return MaskedArray(data, mask=m)._update_from(am)

    wrapped.__name__ = getattr(fn, "__name__", "masked_unary")
    return wrapped


def make_masked_binary(fn):
    """_MaskedBinaryOperation family (verified directly against
    `numpy.ma.core._MaskedBinaryOperation.__call__`'s CPython source, not
    guessed -- and against live `type(np.ma.<name>)` checks that confirm
    exactly which candidate names are real instances of this class, since
    two names that visually look like ordinary binary ufunc wrappers
    (`left_shift`, `right_shift`) turned out to be plain hand-written
    Python functions with DIFFERENT, unreproduced masked-data behavior and
    were excluded from this family entirely rather than guessed at; see
    `anionpy/_state/ma.py`. `maximum`/`minimum` were similarly excluded: they
    are real numpy instances of the unrelated `_extrema_operation` class
    (`where(compare(a, b), a, b)`), not this one.)

    `fn` is called on the FULL raw data of both operands (real op
    everywhere, no domain substitution). The mask is `mask_a | mask_b`
    (via `anionpy.logical_or`; `nomask | nomask` short-circuits to `nomask`,
    identity preserved; `nomask` on one side contributes no extra masking
    without ever materializing a throwaway all-False array). Then, exactly
    like the unary family above, `np.copyto(result, da, where=m)`: every
    combined-masked position has its data reverted to operand A's
    (`am`'s) RAW value, discarding the just-computed result there --
    verified live: `ma.add(masked_a, masked_b).data` at a jointly-masked
    position equals `a`'s raw data, not `a+b`, not `b`'s raw data.
    Reproduced via `anionpy.where(mask, am.data, computed)`, one vectorized
    Rust-backed select.

    fill_value is inherited via `_fv_source`'s `_update_from`-style rule
    (SEE THAT FUNCTION'S DOCSTRING -- CORRECTED in this task's pass): prefer
    whichever ORIGINAL argument (`a` or `b`, before `_as_masked` wrapping)
    was already a `MaskedArray`, preferring `a`, and propagate its
    EXPLICITNESS (not just a computed value) via `MaskedArray._update_from`
    -- an original operand that never had a fill_value explicitly set
    contributes nothing, and the result gets its own (possibly promoted)
    dtype's fresh default instead, matching real numpy exactly (verified
    live, see `_fv_source`'s docstring and this task's report). A PREVIOUS
    revision of this factory unconditionally inherited `am.fill_value`
    (always `_as_masked(a)`, regardless of whether the original `a` was
    ever actually a `MaskedArray`) -- that asymmetry is what this pass
    fixes.
    """

    def wrapped(a, b):
        am = _as_masked(a)
        bm = _as_masked(b)
        with _anionpy.errstate(divide="ignore", invalid="ignore"):
            computed = fn(am.data, bm.data)
        if not isinstance(computed, _ndarray):
            # 0-d path -- verified directly against real numpy's CPython
            # source (`_MaskedBinaryOperation.__call__`, "Case 1: scalar"):
            # `if not result.ndim: return masked if m else result` -- same
            # collapse rule `make_masked_domained_binary` above already
            # implements (see that factory's docstring and the module
            # comment on the 0-d quirk). `anionpy.<fn>` already collapses a
            # 0-d/0-d operand pair to a bare scalar the same way real
            # numpy's ufuncs do (verified live), so `isinstance(computed,
            # _ndarray)` is a reliable signal here too.
            a_masked = am.mask is not nomask and bool(am.mask.item())
            b_masked = bm.mask is not nomask and bool(bm.mask.item())
            return masked if (a_masked or b_masked) else computed
        if am.mask is nomask and bm.mask is nomask:
            mask = nomask
        elif am.mask is nomask:
            mask = bm.mask
        elif bm.mask is nomask:
            mask = am.mask
        else:
            mask = _anionpy.logical_or(am.mask, bm.mask)
        if mask is nomask:
            data = computed
        else:
            data = _anionpy.where(mask, am.data, computed)
        return MaskedArray(data, mask=mask)._update_from(_fv_source(a, b, am, bm))

    wrapped.__name__ = getattr(fn, "__name__", "masked_binary")
    return wrapped


def make_masked_shape_op(fn):
    """Family for shape-transforming ops (`repeat`, `take`, ...): the SAME
    transform is applied to `data` and, when present, to the mask too (via
    the identical already-declared-exact anionpy function -- e.g.
    `anionpy.repeat` called a second time on the boolean mask array). Verified
    live against real numpy: `ma.repeat(a, 2).mask` is `anionpy.repeat`'s
    output applied to the mask, not the mask left at its old shape."""

    def wrapped(a, *args, **kwargs):
        am = _as_masked(a)
        data = fn(am.data, *args, **kwargs)
        if am.mask is nomask:
            mask = nomask
        else:
            mask = fn(am.mask, *args, **kwargs)
        return MaskedArray(data, mask=mask, fill_value=am.fill_value)

    wrapped.__name__ = getattr(fn, "__name__", "masked_shape_op")
    return wrapped


def make_masked_like(fn):
    """`zeros_like`/`ones_like`/`empty_like` family -- NOT a member of
    `make_masked_unary` despite the superficial resemblance (all three take
    one masked-array argument and return one). Verified live: `type(np.ma.
    zeros_like)` (and `ones_like`/`empty_like`) is a plain `function`, not
    `_MaskedUnaryOperation`, and their masked-position DATA is the genuinely
    computed value everywhere (`ma.zeros_like(masked_array([1,2,3,4],
    mask=[T,F,T,F])).data == [0,0,0,0]`, not `[1,2,3,4]`) -- i.e. there is
    NO copyto-revert-to-raw-input here, unlike the real `_MaskedUnaryOperation`
    family (`abs`/`sin`/`ceil`/...). Only the MASK is carried over from the
    input, and as an independent copy (verified live: `zeros_like(a).mask is
    a.mask` is False) -- caught by live differential probing after
    `make_masked_unary`'s copyto-revert fix was (incorrectly) reused here in
    an earlier revision of this file, which put the ORIGINAL raw data back
    at masked positions instead of the genuine zeros/ones."""

    def wrapped(a):
        am = _as_masked(a)
        data = fn(am.data)
        mask = nomask if am.mask is nomask else _anionpy.copy(am.mask)
        return MaskedArray(data, mask=mask, fill_value=am.fill_value)

    wrapped.__name__ = getattr(fn, "__name__", "masked_like")
    return wrapped


# -- Phase 1 instances: unary (mask pass-through) ----------------------------
# `tan` is DELIBERATELY EXCLUDED: verified live `type(np.ma.tan) is
# numpy.ma.core._MaskedUnaryOperation` with a non-None `.domain`
# (`_DomainTan`), i.e. real numpy's `ma.tan` does extra domain masking
# (near cos(x)==0) this family's plain model does not reproduce.
# `cos`/`tanh` are ALSO DELIBERATELY EXCLUDED, for a different reason: live
# differential probing against real numpy 2.5.1 found the wrapper logic
# itself is correct (structurally identical to `sin`/`cosh`/etc., which DO
# pass bit-exact), but the underlying BASE function `anionpy.cos`/`anionpy.tanh`
# -- already declared "exact" in `anionpy/_state/toplevel.py` -- has a
# measurable float32 ULP mismatch on some inputs not exercised by that
# declaration's own test corpus (e.g. `anionpy.cos(array([-2.0],
# dtype=float32))` gives `-0.416146844625473`, real numpy gives
# `-0.41614681482315063`). Building a masked wrapper on top of a base whose
# correctness is not fully verified for the exact inputs this task
# exercises would be declaring correctness nobody has actually checked --
# same discipline as excluding the domained/undeclared-base items above.
# This is a finding about `anionpy.cos`/`anionpy.tanh` for the base ledger's
# maintainers, not something this ma-only task should paper over with a
# tolerance band.
abs = make_masked_unary(_anionpy.abs)
absolute = make_masked_unary(_anionpy.absolute)
fabs = make_masked_unary(_anionpy.fabs)
exp = make_masked_unary(_anionpy.exp)
sin = make_masked_unary(_anionpy.sin)
sinh = make_masked_unary(_anionpy.sinh)
cosh = make_masked_unary(_anionpy.cosh)
arctan = make_masked_unary(_anionpy.arctan)
arcsinh = make_masked_unary(_anionpy.arcsinh)
negative = make_masked_unary(_anionpy.negative)
conjugate = make_masked_unary(_anionpy.conjugate)
ceil = make_masked_unary(_anionpy.ceil)
floor = make_masked_unary(_anionpy.floor)
logical_not = make_masked_unary(_anionpy.logical_not)

# -- Phase 1 instances: binary (mask = mask_a | mask_b) ----------------------
# `left_shift`/`right_shift` are DELIBERATELY EXCLUDED: verified live
# `type(np.ma.left_shift)`/`type(np.ma.right_shift)` are plain `function`,
# NOT `_MaskedBinaryOperation` instances like every other name below --
# their masked-data behavior at combined-masked positions measurably does
# NOT follow the copyto-revert-to-A rule this family implements (probed
# live: shifted result kept, not reverted), so they are a different,
# unreproduced implementation, not a member of this family.
# `maximum`/`minimum` are DELIBERATELY EXCLUDED: verified live
# `type(np.ma.maximum) is numpy.ma.core._extrema_operation`, a distinct
# class implemented as `where(compare(a, b), a, b)` over masked-aware
# `ma.where` -- not this family's copyto-revert model either.
add = make_masked_binary(_anionpy.add)
subtract = make_masked_binary(_anionpy.subtract)
multiply = make_masked_binary(_anionpy.multiply)
bitwise_and = make_masked_binary(_anionpy.bitwise_and)
bitwise_or = make_masked_binary(_anionpy.bitwise_or)
bitwise_xor = make_masked_binary(_anionpy.bitwise_xor)
arctan2 = make_masked_binary(_anionpy.arctan2)
logical_and = make_masked_binary(_anionpy.logical_and)
logical_or = make_masked_binary(_anionpy.logical_or)
logical_xor = make_masked_binary(_anionpy.logical_xor)

# ---------------------------------------------------------------------------
# Phase 2: wrappers whose non-masked anionpy counterpart is already declared
# exact. Each comment below names the exact top-level anionpy function reused.
# ---------------------------------------------------------------------------

# _like variants: verified live against real numpy that these preserve the
# INPUT's mask (as an independent copy) but do NOT revert masked-position
# data to the raw input -- a genuinely different family from
# make_masked_unary, see make_masked_like's docstring.
zeros_like = make_masked_like(_anionpy.zeros_like)
ones_like = make_masked_like(_anionpy.ones_like)
empty_like = make_masked_like(_anionpy.empty_like)

# Shape-based creation (no existing masked array to inherit a mask from):
# always nomask, per real numpy.
def zeros(*args, **kwargs):
    return MaskedArray(_anionpy.zeros(*args, **kwargs))


def ones(*args, **kwargs):
    return MaskedArray(_anionpy.ones(*args, **kwargs))


def empty(*args, **kwargs):
    return MaskedArray(_anionpy.empty(*args, **kwargs))


def identity(n, dtype=None, *, fill_value=_UNSET, hardmask=_UNSET):
    """`ma.identity`: verified live (`inspect.signature` ->
    `(n, dtype=None, *, like=None, fill_value=None, hardmask=False)`) and via
    `inspect.getsource(np.ma.core._convert2ma)`, the generic factory real
    numpy builds `identity`/`arange`/`clip`/`empty`/`indices`/`ones`/`zeros`
    from: `result = np.identity(n, dtype=...).view(MaskedArray)`, THEN, only
    for the keyword-only extras that were ACTUALLY PASSED (checked via
    `kwargs.keys() & params.keys()`, not "is not None" -- so passing
    `fill_value=None` explicitly is a distinct case from omitting it),
    `result.fill_value = fill_value` / `result._hardmask =
    bool(hardmask)`. Omitting `fill_value` entirely leaves the fresh
    `.view(MaskedArray)`'s own default (per-dtype) fill_value untouched --
    verified live, matching this file's `_UNSET`-sentinel convention
    (distinct from `default=None`) used elsewhere for exactly this
    "omitted vs. explicitly None" distinction (see `MaskedArray.__init__`'s
    own `mask=_UNSET`).

    `hardmask` is NOT supported: this codebase has no hardmask/harden_mask
    mechanism anywhere (`MaskedArray` has no `_hardmask` attribute, no
    `harden_mask`/`soften_mask` methods, `__setitem__` never consults such a
    flag) -- same parameter-blindness gap already reported (not hidden) for
    `squeeze`/`expand_dims`'s undeclared `fill_value=`/`hardmask=` override
    kwargs above. Passing `hardmask=True` here is silently accepted (no
    observable effect either way, matching real numpy's OWN observable
    behavior for every operation this codebase implements, since nothing
    reads `_hardmask` on either side) -- reported here rather than raising,
    consistent with the rest of this module's practice for undeclared
    kwargs that have no test-visible consequence.

    Mask is always `nomask` (a fresh `.view(MaskedArray)` with no explicit
    mask kwarg, verified live: `np.ma.identity(3).mask is np.ma.nomask` is
    `True`), never a materialized all-False array -- distinct from the
    stack/join family's always-materialize rule documented above.

    `fill_value=None` EXPLICIT is its own sub-case, caught live by this
    item's own differential probe: real numpy's fill_value SETTER (not the
    constructor) routes an explicit `None` through `_check_fill_value`,
    which resolves it to the plain per-dtype default and stores THAT --
    observably identical to the omitted case's value, but no longer "lazy"
    on numpy's side either way, so reproduced here by resolving `None` to
    `_default_fill_value(...)` before handing it to this wrapper's own
    setter, rather than passing `None` through unchanged (which this file's
    `_cast_fill_value` cannot cast to an array element and raises
    `TypeError` -- confirmed live, the first version of this function did
    exactly that and failed the `fill_value_explicit_None` case).
    """
    data = _anionpy.identity(n, dtype=dtype)
    result = MaskedArray(data)
    if fill_value is not _UNSET:
        result.fill_value = fill_value if fill_value is not None else _default_fill_value(result.dtype)
    return result


# Shape-transforming: mask transformed the same way as data (verified live,
# see make_masked_shape_op's docstring).
repeat = make_masked_shape_op(_anionpy.repeat)
take = make_masked_shape_op(_anionpy.take)


def copy(a):
    """`ma.copy`: real numpy makes an independent copy of BOTH data and mask
    (verified live: `ma.copy(a).mask is a.mask` is False, unlike the
    make_masked_unary pass-through family) -- via `anionpy.copy`, applied to
    each array independently."""
    am = _as_masked(a)
    data = _anionpy.copy(am.data)
    mask = nomask if am.mask is nomask else _anionpy.copy(am.mask)
    return MaskedArray(data, mask=mask, fill_value=am.fill_value)


def size(a, axis=None):
    """`ma.size`: total element count, ignoring the mask entirely (verified
    live against real numpy) -- delegates straight to `anionpy.size` on the
    underlying data."""
    return _anionpy.size(getdata(a), axis) if axis is not None else _anionpy.size(getdata(a))


def ndim(a):
    """`ma.ndim`: delegates to `anionpy.ndim` on the underlying data."""
    return _anionpy.ndim(getdata(a))


def shape(a):
    """`ma.shape`: delegates to `anionpy.shape` on the underlying data."""
    return _anionpy.shape(getdata(a))


# ---------------------------------------------------------------------------
# Phase 3: mask construction / testers.
#
# MASK-DESIGN.md CORRECTION #6: the doc's section 6 groups `MaskType` with
# `mvoid`/`bool_`/`frombuffer`/`fromflex`/`flatten_structured_array`/
# `make_mask_descr` as "touch structured/void dtypes ... blocked on the DType
# enum decision". That is wrong for `MaskType` specifically: verified live
# `np.ma.MaskType is np.bool_` -- it is plainly the scalar bool dtype used for
# every mask array in this whole module (the same `_bool_dtype` already
# imported at the top of this file), not a structured/void type at all. It
# needs no DType-enum work; it is declared here.
MaskType = _bool_dtype

# Same correction applies to `bool_` itself, which MASK-DESIGN.md's section 6
# hazard list also (wrongly, for this specific name) grouped with the
# structured/void-dtype-blocked set. Verified live: `np.ma.bool_ is np.bool_`
# -- `np.ma.bool_` is not a `ma`-specific type at all, just numpy's own
# scalar bool dtype re-exported under the `ma` namespace (same object
# `MaskType` above already aliases). No DType-enum work needed.
bool_ = _bool_dtype


class MAError(Exception):
    """Base class for masked-array errors -- verified live
    `np.ma.MAError.__mro__` is `(MAError, Exception, BaseException, object)`,
    i.e. a plain `Exception` subclass with no special state."""


class MaskError(MAError):
    """Verified live `np.ma.MaskError.__mro__` is `(MaskError, MAError,
    Exception, BaseException, object)` -- `MaskError` is a `MAError`."""


def is_mask(m):
    """Verified live against real numpy: True for the `nomask` singleton and
    for any real boolean-dtype `anionpy.ndarray` (any shape, including 0-d);
    False for anything else (a plain Python list, a non-bool-dtype array).
    Pure attribute/identity check -- no arithmetic."""
    if m is nomask:
        return True
    if not isinstance(m, _ndarray):
        return False
    return _is_bool_dtype(m.dtype)


def is_masked(x):
    """Verified live: False when `x`'s mask is `nomask`; False when `x` has
    an explicit mask but every entry is False (`mask=[0,0,0]`); True only
    when at least one entry of the mask is actually True. Computed via
    `anionpy.any` (one Rust-backed reduction), not a Python loop."""
    m = getmask(x)
    if m is nomask:
        return False
    return bool(_anionpy.any(m))


def isMaskedArray(x):
    """True iff `x` is an instance of this module's `MaskedArray` (or a
    subclass, e.g. `MaskedConstant`)."""
    return isinstance(x, MaskedArray)


# `isMA` / `isarray` are numpy's own aliases for `isMaskedArray` -- verified
# live `np.ma.isMA is np.ma.isMaskedArray` and `np.ma.isarray is
# np.ma.isMaskedArray` are both True (real numpy binds three names to one
# function object). MA-DESIGN.md section 6 flags `isarray` as a "deprecated
# alias" needing a `DeprecationWarning` reproduced -- CORRECTION: verified
# live against real numpy 2.5.1 with `warnings.catch_warnings(record=True)`,
# calling `np.ma.isarray(...)` and `np.ma.isMA(...)` raises no warning at
# all. The doc's claim does not hold for this numpy version; modeled as
# plain aliases, no warning machinery.
isMA = isMaskedArray
isarray = isMaskedArray


def make_mask(m, copy=False, shrink=True, dtype=None):
    """Verified live against real numpy 2.5.1 (`inspect.signature`, plus
    behavioral probes):
      - `m is nomask` (or `m is None`) always returns `nomask` immediately,
        regardless of `shrink`/`copy`.
      - a non-`anionpy.ndarray` input is materialized fresh via `anionpy.array`
        (always a new object -- there is nothing to alias yet).
      - an `anionpy.ndarray` input already at the target dtype is returned
        AS-IS when `copy=False` (identity preserved -- verified live
        `make_mask(bool_src, copy=False) is bool_src`), and copied via
        `anionpy.copy` when `copy=True` (verified `is bool_src` is False).
      - a dtype mismatch always produces a new array via `.astype`
        (unavoidable copy, matches real numpy's own behavior there).
      - when `shrink` is True and the resulting mask has no True entries
        anywhere (checked via `anionpy.any`, not a loop), the singleton
        `nomask` is returned instead of the all-False array (verified live:
        `make_mask([0,0,0])` returns `nomask`, `make_mask([0,0,0],
        shrink=False)` returns the real all-False array).
    """
    if m is nomask or m is None:
        return nomask
    dt = dtype if dtype is not None else MaskType
    if not isinstance(m, _ndarray):
        out = _array(m, dtype=dt)
    elif not _is_bool_dtype(m.dtype) or (dtype is not None and m.dtype.name != dt.name):
        out = m.astype(dt)
    elif copy:
        out = _anionpy.copy(m)
    else:
        out = m
    if shrink and not bool(_anionpy.any(out)):
        return nomask
    return out


def make_mask_none(shape, dtype=None):
    """Verified live: with `dtype=None` (the overwhelmingly common case, and
    the only one anionpy's dtype system can represent) this is exactly
    `anionpy.full(shape, False, dtype=MaskType)` -- a fresh all-False mask
    array of the given shape. Real numpy's `dtype=` parameter is for
    STRUCTURED mask descriptors (`make_mask_descr`'s domain) -- anionpy has no
    structured-dtype support (verified: `anionpy.dtype([('a','i4')])` raises),
    so a non-trivial `dtype` here raises rather than silently ignoring it or
    faking structured support.
    """
    if dtype is not None and getattr(dtype, "name", None) != "bool":
        raise NotImplementedError(
            "anionpy.ma.make_mask_none: structured mask dtypes are not "
            "supported (anionpy has no structured-dtype support)"
        )
    return _anionpy.full(shape, False, dtype=MaskType)


def mask_or(m1, m2, copy=False, shrink=True):
    """Verified live: `mask_or(nomask, nomask)` is `nomask`; `mask_or(nomask,
    m)` / `mask_or(m, nomask)` is (a shrink/copy-processed) `m` itself, no
    `anionpy.logical_or` call needed since ORing with an all-False identity is
    a no-op; otherwise the real elementwise OR via `anionpy.logical_or`
    (Rust-backed). The combined result is passed through the exact same
    `make_mask` shrink/copy contract real numpy uses (verified live: an
    all-False combination collapses to `nomask` when `shrink=True`)."""
    if m1 is nomask and m2 is nomask:
        return nomask
    if m1 is nomask:
        combined = m2
    elif m2 is nomask:
        combined = m1
    else:
        combined = _anionpy.logical_or(m1, m2)
    return make_mask(combined, copy=copy, shrink=shrink)


def _coerce_mask_arg(m):
    """Turn a `masked_where`/`fix_invalid`-style extra `mask=` argument into
    `nomask` or a real boolean `anionpy.ndarray` -- shared by the functions
    below, not a new family of its own."""
    if m is nomask or m is None:
        return nomask
    if isinstance(m, MaskedArray):
        m = m.data
    if not isinstance(m, _ndarray):
        m = _array(m, dtype=MaskType)
    elif not _is_bool_dtype(m.dtype):
        m = m.astype(MaskType)
    return m


def _masked_where_impl(cond, am, copy, shrink):
    """Shared body for `masked_where` and every `masked_<comparator>`/
    `masked_object` wrapper below -- verified live against real numpy: the
    DATA is never recomputed or reverted here (unlike the Phase 1
    unary/binary families), only the mask changes; `am`'s existing mask (if
    any) is combined with `cond` via `mask_or` (so calling `masked_where`
    again on an already-masked array widens the mask rather than
    replacing it -- verified live).

    SHRINK IS ASYMMETRIC, verified live against real numpy's own
    `MaskedArray.__setmask__` (CPython source): a fresh, all-False combined
    mask collapses to the `nomask` singleton ONLY when `am`'s mask was
    ALREADY `nomask` going in. If `am` already carried a real (materialized)
    mask array -- even one that is currently empty or all-False, e.g. an
    array built with the explicit spelling `mask=None` -- the result mask
    stays a real array; it never re-collapses to `nomask`. `__setmask__`'s
    own logic is exactly this: `if current_mask is nomask: (stay nomask iff
    the new mask is also nomask)`, but when `current_mask` is already a real
    array, incoming `nomask`/all-False is applied via `current_mask[...] =
    mask` -- a plain in-place fill that keeps the array object, not a
    replace-with-singleton. Caught live: `ma.masked_equal([], 2.0)` on an
    empty array built via `mask=None` (or `mask=[]`, or `mask=True`+shrink)
    keeps `mask == []` (a real empty array) in real numpy, not `nomask`, and
    the differential harness's `masked_equal/empty/mask_none_explicit` case
    (and its `fully_masked`/`mask_all_false` siblings) caught the naive
    always-shrink version of this function reporting `nomask` instead."""
    data = _anionpy.copy(am.data) if copy else am.data
    effective_shrink = shrink and (am.mask is nomask)
    combined = mask_or(am.mask, cond, shrink=effective_shrink)
    return MaskedArray(data, mask=combined, fill_value=am.fill_value)


def masked_where(condition, a, copy=True):
    """Verified live (`inspect.signature` -> `(condition, a, copy=True)`):
    masks every position where `condition` is true, in ADDITION to any mask
    `a` already carries (verified live: calling `masked_where` on an
    already-partially-masked input widens the mask, does not replace it).
    `condition` may be a plain `anionpy.ndarray`/list/MaskedArray -- coerced to
    a bool mask via `_coerce_mask_arg`. `fill_value` is inherited from `a`
    unchanged (verified live)."""
    am = _as_masked(a)
    cond = _coerce_mask_arg(condition)
    return _masked_where_impl(cond, am, copy, shrink=True)


def _make_masked_comparator(ionp_cmp, fill_value_from_arg):
    """Factory for the ten thin `masked_<comparator>(x, value, copy=True)`
    wrappers -- each is `masked_where(<comparator>(x, value), x)`, verified
    live against real numpy's own CPython source (`numpy/ma/core.py`'s
    `masked_equal`/`masked_greater`/etc. bodies are exactly this one-liner
    pattern). `fill_value_from_arg`: verified live only `masked_equal` (and,
    separately, `masked_values`/`masked_object` below) override the result's
    `fill_value` to the comparison value itself; the other nine
    (`not_equal`, `greater`, `greater_equal`, `less`, `less_equal`) leave
    `fill_value` at `x`'s own default/explicit value, confirmed live for
    every one of them.
    """

    def wrapped(x, value, copy=True):
        xm = _as_masked(x)
        cond = ionp_cmp(xm.data, value)
        result = _masked_where_impl(cond, xm, copy, shrink=True)
        if fill_value_from_arg:
            return MaskedArray(
                result.data, mask=result.mask,
                fill_value=_cast_fill_value(value, xm.data.dtype),
            )
        return result

    return wrapped


masked_equal = _make_masked_comparator(_anionpy.equal, True)
masked_not_equal = _make_masked_comparator(_anionpy.not_equal, False)
masked_greater = _make_masked_comparator(_anionpy.greater, False)
masked_greater_equal = _make_masked_comparator(_anionpy.greater_equal, False)
masked_less = _make_masked_comparator(_anionpy.less, False)
masked_less_equal = _make_masked_comparator(_anionpy.less_equal, False)


def masked_inside(x, v1, v2, copy=True):
    """Verified live (`inspect.signature` -> `(x, v1, v2, copy=True)`):
    masks every position where `v1 <= x <= v2` (real numpy swaps `v1`/`v2`
    if given in reverse order; reproduced here via `min`/`max` on the two
    Python scalars -- bookkeeping on two scalars, not an array loop).
    `fill_value` verified live to stay at `x`'s own default, NOT the
    boundary values."""
    lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
    xm = _as_masked(x)
    cond = _anionpy.logical_and(_anionpy.greater_equal(xm.data, lo), _anionpy.less_equal(xm.data, hi))
    return _masked_where_impl(cond, xm, copy, shrink=True)


def masked_outside(x, v1, v2, copy=True):
    """Verified live: masks every position where `x < v1` or `x > v2`
    (`v1`/`v2` order-normalized the same way as `masked_inside`).
    `fill_value` stays at `x`'s own default, verified live."""
    lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
    xm = _as_masked(x)
    cond = _anionpy.logical_or(_anionpy.less(xm.data, lo), _anionpy.greater(xm.data, hi))
    return _masked_where_impl(cond, xm, copy, shrink=True)


def masked_invalid(a, copy=True):
    """Verified live: masks NaN and +/-inf positions (`not isfinite`, built
    from `anionpy.isfinite`/`anionpy.logical_not`, both Rust-backed). `fill_value`
    verified live to stay at `a`'s own default (e.g. `1e20` for float64),
    NOT overridden.

    DOES NOT SHRINK -- corrected 2026-08-03 (Monday). This was `shrink=True`
    and was wrong. Unlike the `masked_where`/`masked_equal`/`masked_values`
    comparator family, real numpy's `masked_invalid` materializes a full
    boolean mask array even when NOTHING is invalid; it does not collapse to
    `nomask`. Verified live against numpy 2.5.1:
        np.ma.masked_invalid(np.array([1.,2.,3.])).mask -> array([False,False,False])
        np.ma.masked_invalid(np.array([1,2],dtype='int64')).mask -> array([False,False])
        np.ma.masked_invalid(np.array([],dtype='float64')).mask -> array([], dtype=bool)
    anionpy returned `nomask` in all three. That is a real divergence, not a
    cosmetic one: numpy's ma API distinguishes `nomask` from an all-False
    mask and callers branch on `mask is nomask`.

    Why the original test corpus missed it: `_masked_invalid_cases` used a
    SINGLE data vector that always contained nan/inf, so the no-invalid path
    -- the only path where shrinking is observable -- was never sampled. The
    declaration's "verified live" claim was true of the defect's centre and
    silent about its boundary."""
    am = _as_masked(a)
    cond = _anionpy.logical_not(_anionpy.isfinite(am.data))
    return _masked_where_impl(cond, am, copy, shrink=False)


def masked_object(x, value, copy=True, shrink=True):
    """Verified live against real numpy: on the non-object dtypes anionpy
    actually supports (anionpy has no object dtype -- verified:
    `anionpy.array([1], dtype=object)` raises `TypeError` -- so this item is
    declared only for anionpy's real dtypes, matching what `ma.masked_equal`
    would do on the same input, confirmed live bit-for-bit identical mask
    AND `fill_value` output between `masked_object(a, v)` and
    `masked_equal(a, v)` for the same `a`/`v`), `fill_value` IS overridden
    to `value` (verified live, same as `masked_equal`, unlike the plain
    comparator family above)."""
    xm = _as_masked(x)
    cond = _anionpy.equal(xm.data, value)
    result = _masked_where_impl(cond, xm, copy, shrink)
    return MaskedArray(
        result.data, mask=result.mask,
        fill_value=_cast_fill_value(value, xm.data.dtype),
    )


def masked_values(x, value, rtol=1e-5, atol=1e-8, copy=True, shrink=True):
    """Verified live (`inspect.signature` ->
    `(x, value, rtol=1e-05, atol=1e-08, copy=True, shrink=True)`): on
    floating-point data the comparison is a TOLERANCE match
    (`|x - value| <= atol + rtol * |value|`, i.e. `isclose`'s formula --
    anionpy has no standalone `isclose`, so it is inlined here from
    already-Rust-backed `anionpy.abs`/`anionpy.subtract`/`anionpy.less_equal`, and
    `atol + rtol * abs(value)` is a single Python-scalar computation on the
    scalar `value`, not an array loop, exactly like `_cast_fill_value`'s
    existing scalar bookkeeping elsewhere in this file); on integer/bool
    data it is exact `anionpy.equal`, verified live real numpy does NOT apply a
    tolerance there. `fill_value` is overridden to `value`, verified live.

    NOT built on `_masked_where_impl` (unlike every other `masked_*`
    comparator in this family), and that is a deliberate, verified
    departure: real numpy's CPython source (`numpy/ma/core.py`) does
    `xnew = filled(x, value)` FIRST -- any position `x` already had masked
    is overwritten with `value` -- and only THEN computes `mask` against
    `xnew`, replacing the mask outright (no `mask_or` with `x`'s prior
    mask). This is the one comparator where a pre-existing masked position's
    DATA is genuinely rewritten, not left alone and not reverted to its raw
    input value -- caught live: `masked_values` on an already-fully-masked
    input reproducibly returned real numpy's own placeholder data
    (`value`) at every position, not the original unmasked data, which the
    naive `_masked_where_impl`-based version (computing `cond` against
    `xm.data` directly, leaving already-masked positions' underlying data
    untouched) failed to reproduce."""
    xm = _as_masked(x)
    dt = xm.data.dtype
    xnew = filled(xm, value)
    if dt.name.startswith("float") or dt.name.startswith("complex"):
        # NOTE: `abs` at module scope is this file's OWN masked-unary `abs`
        # (`abs = make_masked_unary(_anionpy.abs)` above), not the Python
        # builtin -- using it here on a bare scalar `value` would shadow
        # into the wrong function. `value.__abs__()` (a single Python
        # scalar operation, not an array loop) sidesteps the shadowing.
        tol = atol + rtol * value.__abs__()
        cond = _anionpy.less_equal(_anionpy.abs(_anionpy.subtract(xnew, value)), tol)
    else:
        cond = _anionpy.equal(xnew, value)
    mask = make_mask(cond, shrink=shrink)
    return MaskedArray(
        xnew, mask=mask,
        fill_value=_cast_fill_value(value, dt),
    )


# `ma.masked_array` -- verified live `np.ma.masked_array is np.ma.MaskedArray`
# (real numpy binds the constructor function name and the class to the same
# object). Modeled identically: one more name for the same class.
masked_array = MaskedArray


def masked_all(shape, dtype=None):
    """Verified live (`inspect.signature` -> `(shape, dtype=<class
    'float'>)`): an uninitialized (`anionpy.empty` -- genuinely arbitrary
    garbage data, verified live real numpy's own values are the raw
    unwritten buffer contents too, not zeros) array, fully masked
    (`mask=True` broadcast via `anionpy.full`, both Rust-backed calls)."""
    dt = dtype if dtype is not None else _anionpy.float64
    data = _anionpy.empty(shape, dtype=dt)
    mask = _anionpy.full(shape, True, dtype=MaskType)
    return MaskedArray(data, mask=mask)


def masked_all_like(arr):
    """Verified live (`inspect.signature` -> `(arr)`): same shape/dtype as
    `arr`'s data (via `anionpy.empty_like`, Rust-backed, again genuinely
    uninitialized data, not zeros or a copy of `arr`'s own data), fully
    masked."""
    d = getdata(arr)
    data = _anionpy.empty_like(d)
    mask = _anionpy.full(d.shape, True, dtype=MaskType)
    return MaskedArray(data, mask=mask)


class _MaskedPrintOption:
    """`ma.masked_print_option` -- verified live against real numpy: a
    single global object controlling what string represents a masked
    value when printed (`_display`, default `'--'`) and whether that
    substitution is `_enabled` (default True). Pure bookkeeping state, no
    numeric computation of any kind -- modeled as a plain Python object,
    not built on any anionpy/Rust call."""

    def __init__(self, display="--"):
        self._display = display
        self._enabled = True

    def display(self):
        return self._display

    def set_display(self, s):
        self._display = s

    def enabled(self):
        return self._enabled

    def enable(self, shrink=1):
        self._enabled = bool(shrink)

    def __str__(self):
        return self._display

    def __repr__(self):
        return self._display


masked_print_option = _MaskedPrintOption()


# ---------------------------------------------------------------------------
# Phase 4: fill-value machinery.
# ---------------------------------------------------------------------------
def _resolve_dtype(obj):
    """Shared dtype-resolution for the four `default_fill_value`-family
    functions below: accepts a `MaskedArray`, a plain `anionpy.ndarray`, or a
    bare Python scalar (materialized via `anionpy.array` to discover its
    dtype, verified live real numpy does the same for a bare scalar
    argument -- e.g. `np.ma.default_fill_value(1.0)` reports the float
    default)."""
    if isinstance(obj, MaskedArray):
        return obj.data.dtype
    if isinstance(obj, _ndarray):
        return obj.dtype
    return _array(obj).dtype


def default_fill_value(obj):
    """Verified live for every dtype kind anionpy actually supports
    (bool/int*/uint*/float*/complex*, both array and bare-scalar `obj`
    forms).

    NOT a simple delegation to `_default_fill_value` (the internal table
    used by `MaskedArray.fill_value`'s property) -- real numpy's two paths
    genuinely disagree on unsigned-int width, confirmed live:
    `np.ma.default_fill_value(np.dtype('uint8'))` reports `999999` typed
    `int64`, while `np.ma.masked_array([1], dtype='uint8').fill_value`
    (the property, going through `_check_fill_value` which special-cases
    `ndtype.kind == 'u'` to `np.uint(fill_value)`) reports `999999` typed
    `uint64`. The standalone function has no such special case -- it goes
    through `_recursive_fill_value` -> `_scalar_fill_value`, which returns
    the bare Python `999999` for both the `'i'` and `'u'` dtype kinds
    un-narrowed, and `np.asarray(999999).dtype` is always `int64`. So this
    function intentionally does NOT call `_default_fill_value` for the
    unsigned case.

    BUG FOUND AND FIXED 2026-08-07 (Monday, grind-continuation session),
    NOT while working this item directly -- found by chance while reading
    this function's source as prep for an unrelated grind item, which is
    exactly the "re-measure rather than trust the comment" discipline the
    coordinator flagged after the `ma.mean` finding. The docstring above
    already correctly states real numpy's `default_fill_value` returns BARE
    PYTHON `True`/`999999`/`1e20`/`complex(1e20, 0j)` (`_scalar_fill_value`
    never boxes), but the CODE below it did not follow its own docstring --
    it called `_anionpy.bool_(True)` / `_anionpy.int64(999999)` /
    `_anionpy.float64(1e20)` / `_anionpy.complex128(...)` directly, which
    constructs anionpy's OWN scalar-hierarchy objects (`anionpy.bool_`,
    `anionpy.int64`, ...) -- the SAME failure shape as the
    `MaskedArray.fill_value`-property boxing bug fixed elsewhere in this
    file, but a DIFFERENT instance of it, in a sibling function, still
    declared `exact` in `anionpy/_state/ma.py` the whole time it was wrong.
    Verified live before fixing: `type(anionpy.ma.default_fill_value(...))`
    was `anionpy.bool_`/`anionpy.int64`/`anionpy.float64`/`anionpy.complex128`
    against real numpy's `bool`/`int`/`float`/`complex`, across bool/int32/
    uint16/float32/complex64. Fixed by returning the bare Python literals
    directly (no `_box_typed_scalar` needed here -- unlike the fill_value
    property, this function's contract per the docstring above is bare
    Python types, not even genuine numpy scalars). The differential corpus's
    `_fv_norm_scalar` helper (`tests/differential/ma_cases.py`) had the same
    type-blind-normalization gap `_fv_norm` had before its own fix; also
    corrected there, in the same pass.
    """
    dtype = _resolve_dtype(obj)
    name = dtype.name
    if name == "bool":
        return True
    if name.startswith("uint") or name.startswith("int"):
        return 999999
    if name.startswith("float"):
        return 1e20
    if name.startswith("complex"):
        return complex(1e20, 0.0)
    raise TypeError(f"anionpy.ma: no default fill_value known for dtype {name!r}")


def _int_bounds(name):
    """Fixed bit-width table for signed/unsigned integer dtype bounds --
    scalar bookkeeping constants (there are exactly 8 integer dtype names
    anionpy supports), not a per-element computation."""
    if name.startswith("uint"):
        bits = int(name[4:])
        return 0, (1 << bits) - 1
    bits = int(name[3:])
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def minimum_fill_value(obj):
    """Verified live against real numpy for bool/int*/uint*/float*/
    complex*: the value used to fill masked slots so they never win a
    MINIMUM reduction, i.e. the dtype's largest representable value (the
    dtype's integer max, `+inf` for float, `inf+infj` for complex -- every
    one of these confirmed live, not guessed).

    CORRECTED DURING DIFFERENTIAL TESTING: an earlier draft of this
    docstring/return claimed the bool case returns `True`. Live re-check
    (`np.ma.minimum_fill_value(np.array([1], dtype=bool))`) returns the
    plain Python `int` `1`, NOT a `bool`/`np.bool_` -- real numpy's bool
    branch of its own `_extremum_fill_value`-style table reuses the integer
    path (`1`/`0`), it does not special-case bool with `True`/`False`. The
    differential harness's `minimum_fill_value/bool` case caught this."""
    name = _resolve_dtype(obj).name
    if name == "bool":
        return 1
    if name.startswith("uint") or name.startswith("int"):
        return _int_bounds(name)[1]
    if name.startswith("float"):
        return float("inf")
    if name.startswith("complex"):
        return complex(float("inf"), float("inf"))
    raise TypeError(f"anionpy.ma: no minimum fill value known for dtype {name!r}")


def maximum_fill_value(obj):
    """Verified live: the dual of `minimum_fill_value` -- the dtype's
    smallest representable value, so a masked slot never wins a MAXIMUM
    reduction (the dtype's integer min -- `0` for unsigned -- `-inf` for
    float, `-inf-infj` for complex; every one confirmed live).

    Same bool-branch correction as `minimum_fill_value` above: real numpy
    returns the plain Python `int` `0` for bool dtype here, not `False`."""
    name = _resolve_dtype(obj).name
    if name == "bool":
        return 0
    if name.startswith("uint"):
        return 0
    if name.startswith("int"):
        return _int_bounds(name)[0]
    if name.startswith("float"):
        return float("-inf")
    if name.startswith("complex"):
        return complex(float("-inf"), float("-inf"))
    raise TypeError(f"anionpy.ma: no maximum fill value known for dtype {name!r}")


def common_fill_value(a, b):
    """Verified live (`inspect.signature` -> `(a, b)`): returns the shared
    `fill_value` if both operands have the SAME one, else `None` -- a
    two-scalar `==` check, not an array operation."""
    fva = getattr(a, "fill_value", None)
    fvb = getattr(b, "fill_value", None)
    if fva is None or fvb is None or fva != fvb:
        return None
    return fva


def set_fill_value(a, fill_value):
    """MASK-DESIGN.md CORRECTION: section 6's mutation table (line ~41)
    grouped `set_fill_value` with `put`/`putmask`/`soften_mask`/
    `harden_mask` as one of the "5 mutating items ... gated on the same
    buffer decision as `out=`". That grouping is wrong for this one item:
    `fill_value` is bookkeeping state on THIS Python `MaskedArray` wrapper
    object (`self._fill_value`), never touching the underlying Rust
    `anionpy.ndarray` buffer at all -- unlike `put`/`putmask` (which write into
    `.data`/`.mask`'s actual buffer) or `soften_mask`/`harden_mask` (which
    gate later in-place `__setitem__` behavior). Mutating a plain Python
    attribute on our own wrapper class needs no Rust interior-mutability
    change; verified live real numpy's own `set_fill_value` is exactly
    `a.fill_value = fill_value` (a property setter), reproduced here via
    `MaskedArray.fill_value`'s new setter (see above) plus the same
    dtype-cast the constructor already applies."""
    if isinstance(a, MaskedArray):
        a.fill_value = fill_value


# ---------------------------------------------------------------------------
# Phase 5: contiguity / structure / joining.
#
# Two genuinely distinct families live here, verified live against real
# numpy 2.5.1 -- do NOT assume one's shrink/fill_value contract for the
# other (this task's brief's own warning, borne out by probing):
#
#   (a) the STACK/JOIN family (`vstack`/`hstack`/`dstack`/`column_stack`/
#       `row_stack`/`append`/`diagflat`): real numpy's `numpy/ma/extras.py`
#       generates these via `_fromnxfunction`-style wrappers that call the
#       plain top-level anionpy function on `.data` AND SEPARATELY on
#       `getmaskarray(...)` (never `.mask` directly -- every operand's mask
#       is materialized to a real all-False array first if it was `nomask`).
#       Verified live: the result mask is a REAL array even when every
#       input was `nomask` (`ma.vstack([nomask_a, nomask_b]).mask` is a real
#       all-False 2-d array, `is nomask` is False) -- these never shrink.
#       `fill_value` is ALWAYS the plain per-dtype DEFAULT
#       (`_default_fill_value`), verified live to NOT inherit from any
#       operand even when every operand shares the identical custom
#       `fill_value` (`ma.vstack([a_fv=-5, b_fv=-5]).fill_value` is `1e20`,
#       not `-5.0`) -- a genuinely different rule from every family above.
#       Data at masked positions is the plain juxtaposed/rearranged
#       underlying value (no computation happens here at all, so there is
#       nothing to "revert").
#
#   (b) the SHAPE-TRANSFORM family (`transpose`/`swapaxes`/`reshape`/
#       `ravel`/`squeeze`/`expand_dims`/`compress`): reuses
#       `make_masked_shape_op`'s existing contract (nomask stays nomask,
#       `fill_value` is INHERITED from the input, verified live for every
#       one of these six/seven with a custom `fill_value` operand) --
#       `resize` is the one exception in this family (see its own docstring
#       below): it is NOT built on `make_masked_shape_op` because its
#       `fill_value` does NOT inherit (verified live, real numpy's own
#       `ma.resize` source does `np.resize(x, new_shape).view(subclass)`,
#       which gets a fresh default `fill_value`, then separately assigns
#       `result._mask = np.resize(getmask(x), new_shape)` -- mask stays
#       `nomask` when the input's was, but `fill_value` resets to default).
# ---------------------------------------------------------------------------


def _make_masked_stack(fn):
    """Factory for `vstack`/`hstack`/`dstack`/`column_stack`: `fn` takes a
    single sequence argument (verified live: real `anionpy.vstack`/`hstack`/
    `dstack`/`column_stack` all accept a list/tuple of arrays, not
    varargs). Mask is ALWAYS materialized via `getmaskarray` on every
    operand (never left as `nomask`, verified live -- see module comment
    above), then `fn` is called on the mask list too -- both calls are the
    same already-declared-exact Rust-backed function, just applied twice.
    `fill_value` is the plain per-dtype default, not inherited (verified
    live, see module comment above)."""

    def wrapped(arrays):
        arrs = [_as_masked(a) for a in arrays]
        data = fn([a.data for a in arrs])
        mask = fn([getmaskarray(a) for a in arrs])
        return MaskedArray(data, mask=mask, fill_value=_default_fill_value(data.dtype))

    wrapped.__name__ = getattr(fn, "__name__", "masked_stack")
    return wrapped


vstack = _make_masked_stack(_anionpy.vstack)
hstack = _make_masked_stack(_anionpy.hstack)
dstack = _make_masked_stack(_anionpy.dstack)
column_stack = _make_masked_stack(_anionpy.column_stack)
# row_stack: verified live `np.ma.row_stack is np.ma.vstack` -- real numpy
# binds the same function object to both names (like `masked_array`/
# `MaskedArray` and `isMA`/`isMaskedArray` elsewhere in this file).
row_stack = vstack


def append(a, b, axis=None):
    """Verified live (`inspect.signature` -> `(a, b, axis=None)`), then
    CORRECTED DURING DIFFERENTIAL TESTING: an earlier draft only collapsed
    to `nomask` when BOTH operands were `nomask` by object identity, else
    materialized the combined mask unconditionally -- that is `vstack`'s
    rule, not `append`'s. Real `numpy.ma.append` is literally
    `return concatenate([a, b], axis)` (verified via
    `inspect.getsource(np.ma.append)`), and `numpy.ma.concatenate`'s own
    source (`inspect.getsource(np.ma.concatenate)`) shows it ALWAYS
    shrinks the combined mask via `data._mask = _shrink_mask(dm)` on the
    `np.concatenate([getmaskarray(a) for a in arrays], axis)` result --
    the only fast path it skips this on is when EVERY operand's mask `is
    nomask` by identity (`for x in arrays: if getmask(x) is not nomask:
    break else: return data`), which is just an optimization: shrinking an
    all-False array built from all-nomask operands would collapse to
    `nomask` anyway, so the two paths are observationally identical.
    Reproduced live: `ma.append(masked_array([1.,2.,3.], mask=None),
    masked_array([4.,5.,6.]))` (left operand has a REAL, not identity-
    nomask, all-False mask array) still returns `.mask is np.ma.nomask ==
    True` -- caught by this task's `a_none_b_nomask`/`both_all_false`
    differential cases, which the identity-only check above failed.
    `fill_value` is the plain per-dtype default, not inherited from either
    operand (verified live) -- `concatenate`'s `data = d.view(rcls)` never
    touches fill_value, so the fresh view gets the default."""
    am = _as_masked(a)
    bm = _as_masked(b)
    kwargs = {} if axis is None else {"axis": axis}
    data = _anionpy.append(am.data, bm.data, **kwargs)
    combined = _anionpy.append(getmaskarray(am), getmaskarray(bm), **kwargs)
    mask = make_mask(combined, shrink=True)
    return MaskedArray(data, mask=mask, fill_value=_default_fill_value(data.dtype))


def diagflat(v, k=0):
    """Verified live (`inspect.signature` -> `(v, k=0)`): builds a 2-d
    matrix with `v`'s (flattened) values on the `k`-th diagonal via
    `anionpy.diagflat` (already declared exact), applied identically to the
    mask (materialized via `getmaskarray`, same always-materialize /
    default-fill_value rule as the rest of the stack family above --
    verified live: `ma.diagflat(nomask_v).mask is nomask` is False, a real
    all-False 2-d array; `ma.diagflat(v, fill_value=custom).fill_value` is
    the plain dtype default, not `custom`)."""
    vm = _as_masked(v)
    data = _anionpy.diagflat(vm.data, k)
    mask = _anionpy.diagflat(getmaskarray(vm), k)
    return MaskedArray(data, mask=mask, fill_value=_default_fill_value(data.dtype))


# -- shape-transform family (reuses make_masked_shape_op's existing
# contract: nomask stays nomask, fill_value inherited from the input,
# verified live for every member below with a custom-fill_value operand) --
transpose = make_masked_shape_op(_anionpy.transpose)
swapaxes = make_masked_shape_op(_anionpy.swapaxes)
reshape = make_masked_shape_op(_anionpy.reshape)
ravel = make_masked_shape_op(_anionpy.ravel)
# squeeze: real numpy's signature is `(a, axis=None, *, fill_value=None,
# hardmask=None)` -- verified live the two extra keyword-only params are
# rarely-used OVERRIDES for the result's own fill_value/hardmask state, not
# something that changes the default (kwargs-omitted) behavior verified
# here (`ma.squeeze(a).fill_value` with no `fill_value=` kwarg still
# inherits `a`'s own fill_value, verified live). NOT supporting those two
# override kwargs is a parameter-blindness gap, same shape as
# `anionpy.concatenate`'s undeclared `out=`/`casting=` -- reported, not hidden.
squeeze = make_masked_shape_op(_anionpy.squeeze)
expand_dims = make_masked_shape_op(_anionpy.expand_dims)


def _atleast_nd_one(a, npfunc):
    """Shared step for `atleast_1d`/`atleast_2d`/`atleast_3d`, one array.

    Real numpy's `ma.atleast_1d`/`2d`/`3d` (numpy/ma/extras.py) are NOT
    the mask-preserving subclass-dispatch style used by the shape-
    transform family (`reshape`/`transpose`/etc, `make_masked_shape_op`).
    They are built via `extras._fromnxfunction_allargs(np.atleast_Nd)`,
    whose body (read via `inspect.getsource`) is exactly:

        masked_array(
            data=npfunc(np.asarray(a), **kwargs),
            mask=npfunc(getmaskarray(a), **kwargs),
        )

    i.e. the PLAIN top-level `atleast_Nd` is applied independently to the
    data array AND to the fully-materialized boolean mask array
    (`getmaskarray`, which forces `nomask` into an explicit all-False
    array of the input's shape), and the two results are combined into a
    BRAND NEW `masked_array(...)`. Two consequences verified live against
    real numpy 2.5.1, both deliberately reproduced here rather than
    "fixed" (matching real numpy's behavior, however surprising, is the
    goal of this differential suite):

      1. The result's mask is NEVER `nomask`, even when the input was
         `nomask` -- confirmed live: `np.ma.atleast_1d(masked_array([1.,
         2.,3.])).mask` is `array([False, False, False])`, not `nomask`
         (`is np.ma.nomask` is False), because `getmaskarray` (not
         `.mask`) feeds the second `npfunc` call.
      2. The result's `fill_value` is NOT inherited from the input --
         `masked_array(data=..., mask=...)` with no `fill_value=` kwarg
         gets numpy's dtype-default fill value. Confirmed live:
         `masked_array([1.,2.,3.], fill_value=99.0)` round-tripped
         through `np.ma.atleast_1d` comes back with `fill_value ==
         1e+20` (the float default), not `99.0`.

    Both anionpy's plain `atleast_1d`/`2d`/`3d` (already declared exact
    at toplevel) and `getmaskarray` are reused directly, so this
    reproduces the exact same two-independent-calls-plus-fresh-wrap
    shape as real numpy, not just its net effect."""
    am = _as_masked(a)
    new_data = npfunc(am.data)
    new_mask = npfunc(getmaskarray(am))
    return MaskedArray(new_data, mask=new_mask)


def atleast_1d(*arys):
    """`ma.atleast_1d`: see `_atleast_nd_one`'s docstring -- built from
    real numpy's actual `extras._fromnxfunction_allargs` algorithm, not a
    reuse of the mask-preserving shape-transform family."""
    if len(arys) == 1:
        return _atleast_nd_one(arys[0], _anionpy.atleast_1d)
    return tuple(_atleast_nd_one(a, _anionpy.atleast_1d) for a in arys)


def atleast_2d(*arys):
    """`ma.atleast_2d`: see `_atleast_nd_one`'s docstring."""
    if len(arys) == 1:
        return _atleast_nd_one(arys[0], _anionpy.atleast_2d)
    return tuple(_atleast_nd_one(a, _anionpy.atleast_2d) for a in arys)


def atleast_3d(*arys):
    """`ma.atleast_3d`: see `_atleast_nd_one`'s docstring."""
    if len(arys) == 1:
        return _atleast_nd_one(arys[0], _anionpy.atleast_3d)
    return tuple(_atleast_nd_one(a, _anionpy.atleast_3d) for a in arys)


def resize(x, new_shape):
    """Verified live (`inspect.signature` -> `(x, new_shape)`, and against
    real numpy's own CPython source for `numpy.ma.core.resize`): mask is
    resized the SAME way as data (`anionpy.resize`, already declared exact) --
    staying `nomask` when the input's was (verified live) -- but
    `fill_value` resets to the plain per-dtype DEFAULT, NOT inherited from
    `x` (verified live: `ma.resize(x_with_custom_fill_value,
    new_shape).fill_value` is the dtype default, not the custom value) --
    this is the one shape-transform-family member that does NOT reuse
    `make_masked_shape_op`, because real numpy's own implementation
    constructs a fresh `.view(subclass)` result rather than propagating the
    input's `fill_value` the way every sibling above does."""
    xm = _as_masked(x)
    data = _anionpy.resize(xm.data, new_shape)
    mask = nomask if xm.mask is nomask else _anionpy.resize(xm.mask, new_shape)
    return MaskedArray(data, mask=mask, fill_value=_default_fill_value(data.dtype))


def compress(condition, a, axis=None):
    """Verified live (`inspect.signature` -> `(condition, a, axis=None,
    out=None)`; `out=` not supported here, same parameter-blindness gap
    noted elsewhere in this file): unlike every other shape-transform
    member, `condition` is the FIRST argument, not `a` -- a dedicated
    wrapper rather than a `make_masked_shape_op` instance. `fill_value` IS
    inherited from `a` (verified live with a custom `fill_value` operand --
    this member does NOT follow the stack family's default-fill_value
    rule), and `nomask` stays `nomask` when `a`'s mask was already `nomask`
    (verified live). `condition` is coerced to a real boolean `anionpy.ndarray`
    via the already-shared `_coerce_mask_arg` helper -- reused here for its
    exact "materialize to bool dtype" behavior, not because `condition` is
    a mask semantically."""
    am = _as_masked(a)
    cond = _coerce_mask_arg(condition)
    kwargs = {} if axis is None else {"axis": axis}
    data = _anionpy.compress(cond, am.data, **kwargs)
    mask = nomask if am.mask is nomask else _anionpy.compress(cond, am.mask, **kwargs)
    return MaskedArray(data, mask=mask, fill_value=am.fill_value)


def compressed(x):
    """Verified live (`inspect.signature` -> `(x)`): returns a PLAIN
    `anionpy.ndarray` (never a `MaskedArray`) of every unmasked value, in
    flattened (C-order) order -- `nomask` input just ravels `.data`
    (`anionpy.ravel`, already declared exact); a real mask flattens both data
    and mask (`anionpy.ravel` on each) and selects via `anionpy.compress` against
    the raveled `logical_not(mask)`, both already-declared-exact
    Rust-backed calls, no Python loop over positions. Verified live on
    0-d, empty, and fully-masked (all-empty-result) inputs -- `masked_all`'s
    UNINITIALIZED underlying data means a fully-masked `compressed()` result
    is compared for shape/dtype only in this task's differential cases, per
    this task's brief, never for data content."""
    xm = _as_masked(x)
    if xm.mask is nomask:
        return _anionpy.ravel(xm.data)
    cond = _anionpy.ravel(_anionpy.logical_not(xm.mask))
    flat_data = _anionpy.ravel(xm.data)
    return _anionpy.compress(cond, flat_data)


def nonzero(a):
    """Verified live (`inspect.signature` -> `(a)`): returns the same
    tuple-of-index-arrays shape as `anionpy.nonzero`, but a MASKED position is
    treated as excluded regardless of its underlying data value (verified
    live: `ma.nonzero(masked_array([0,5,0,7], mask=[1,0,0,0]))` excludes
    index 0 even though real numpy's plain `nonzero` semantics only ever
    look at "is this masked" for that exclusion, not the raw `0` value
    already sitting there -- i.e. the mask bit dominates, it is not merely
    OR'd with the "value is truthy" test). Computed as `(data cast to bool)
    AND NOT mask`, via `.astype(MaskType)` (already relied on elsewhere in
    this file, e.g. the constructor's own dtype coercion -- verified live
    against real numpy's own truthiness-cast semantics: NaN casts True,
    `0+0j` casts False, `0+1j` casts True) combined with `anionpy.logical_and`/
    `anionpy.logical_not`, then `anionpy.nonzero` on the combined boolean array --
    three already-declared-exact Rust-backed calls, no Python loop over
    positions."""
    am = _as_masked(a)
    nz = am.data.astype(MaskType)
    if am.mask is not nomask:
        nz = _anionpy.logical_and(nz, _anionpy.logical_not(am.mask))
    return _anionpy.nonzero(nz)


def fix_invalid(a, mask=nomask, copy=True, fill_value=None):
    """Verified live (`inspect.signature` ->
    `(a, mask=nomask, copy=True, fill_value=None)`): UNLIKE every other
    function in this file, the DATA at newly-invalid positions is actually
    OVERWRITTEN with the fill value (verified live: `fix_invalid([1., nan,
    3.]).data` is `[1., 1e20, 3.]`, not `[1., nan, 3.]`) -- computed via
    `anionpy.where`, one vectorized Rust-backed select, same mechanism the
    Phase 1 families use, just selecting the OPPOSITE way (fill value at
    invalid positions, computed data everywhere else) since here the
    computed data is already correct and only invalid positions need
    replacing. The resulting mask is `a`'s existing mask OR both the
    caller-supplied extra `mask=` OR the newly-discovered invalid
    positions -- combined via `mask_or`, verified live."""
    am = _as_masked(a)
    invalid = _anionpy.logical_not(_anionpy.isfinite(am.data))
    extra = _coerce_mask_arg(mask)
    if extra is not nomask:
        invalid = _anionpy.logical_or(invalid, extra)
    combined = mask_or(am.mask, invalid, shrink=True)
    fv = fill_value if fill_value is not None else am.fill_value
    fv_arr = _anionpy.full(am.data.shape, fv, dtype=am.data.dtype)
    data = _anionpy.where(invalid, fv_arr, am.data)
    if copy:
        data = _anionpy.copy(data)
    return MaskedArray(data, mask=combined, fill_value=am.fill_value)


# ---------------------------------------------------------------------------
# Phase 6: the DOMAINED unary/binary families (`_MaskedUnaryOperation` with a
# non-None `.domain`, and `_DomainedBinaryOperation`) -- verified directly
# against real numpy 2.5.1's own CPython source
# (`numpy/ma/core.py`'s `_MaskedUnaryOperation.__call__` "Case 1.1: Domained
# function" branch, and `_DomainedBinaryOperation.__call__`), not guessed.
#
# THE 0-d QUIRK (found while building this phase, applies to BOTH families):
# real numpy's own source ends every one of these `__call__`s with
# `if not result.ndim: (return masked if the combined mask bit else the bare
# computed scalar)` -- for a 0-d operand, the function returns EITHER the
# `masked` singleton OR a bare Python/numpy scalar, NEVER a `MaskedArray`.
# Verified live: `type(np.ma.sqrt(np.ma.masked_array(-4.0)))` is
# `numpy.ma.core.MaskedConstant`; `type(np.ma.sqrt(np.ma.masked_array(4.0)))`
# is `numpy.float64`. anionpy's own ufuncs already collapse a 0-d
# `anionpy.ndarray` operand to a bare scalar the same way real numpy's do
# (verified live: `anionpy.sqrt(anionpy.array(4.0))` is already a bare
# `numpy.float64`, not an `anionpy.ndarray`) -- `isinstance(computed, _ndarray)`
# is therefore a reliable, already-available signal for "did this call
# collapse to the 0-d scalar path", no separate shape check needed.
#
# THIS IS A PRE-EXISTING GAP IN THE ALREADY-DECLARED (Phase 1) PLAIN
# unary/binary families above (`make_masked_unary`/`make_masked_binary`,
# e.g. `sin`/`add`): neither one special-cases 0-d input at all --
# `MaskedArray.__init__`'s own `not isinstance(data, _ndarray): data =
# _array(data)` fallback silently re-wraps the bare scalar `fn(am.data)`
# already returns back into a 0-d MaskedArray, which measurably diverges
# from real numpy's bare-scalar/`masked`-singleton return (verified live:
# `anionpy.ma.sin(anionpy.ma.MaskedArray(2.0)).__class__` is `MaskedArray`, real
# numpy's is `numpy.float64`). That family's own declared items are outside
# this phase's scope (already shipped, already covered by a 1-d/2-d/empty
# corpus that never samples 0-d, per this task's own "vary input data" brief
# -- this is exactly the ragged, boundary-shaped defect class the brief
# warns about) and are NOT touched here to avoid destabilizing 40+ already-
# passing declarations under this task's own no-scope-creep discipline; see
# this task's report for the exact file:line and a recommendation for a
# follow-up task.
#
# THE FILL_VALUE ASYMMETRY (found while building this phase) -- FIXED, a
# LATER PASS, along with the deeper EXPLICITNESS bug it was tangled up
# with (see `MaskedArray._update_from`'s docstring near the top of this
# file for the full mechanism). Real numpy's `_DomainedBinaryOperation.
# __call__` (like `_MaskedBinaryOperation`'s) ends with
# `masked_result._update_from(a) if isinstance(a, MaskedArray) else (...
# elif isinstance(b, MaskedArray): masked_result._update_from(b))` -- i.e.
# `fill_value` is inherited from whichever ORIGINAL argument was ALREADY a
# `MaskedArray` (preferring `a`), not unconditionally from `_as_masked(a)`,
# AND that inheritance is EXPLICITNESS-aware (an original operand that
# never had a fill_value explicitly set contributes nothing -- the result
# just gets its own, possibly-promoted, dtype's fresh default). Verified
# live: `np.ma.divide([1.,2.,3.], np.ma.masked_array([1.,2.,3.],
# fill_value=-9.0)).fill_value` is `-9.0` (inherited from the plain list's
# masked PARTNER, `b`), NOT the per-dtype default a naive
# `_as_masked(a).fill_value` would produce. `_fv_source` (defined next to
# `make_masked_binary` above) implements the shared source-selection half
# of this rule; `MaskedArray._update_from` implements the explicitness-
# propagation half. Both `make_masked_binary` (Phase 1) and
# `make_masked_domained_binary` (this phase, below) now use them -- see
# each factory's own docstring.
# ---------------------------------------------------------------------------


def _scalar_isfinite(x):
    """`math.isfinite` extended to a bare Python/numpy complex scalar (real
    AND imaginary parts both finite) -- a single scalar check, not an array
    loop. None of this phase's declared domained items actually accept
    complex data (see each factory instance's own comment), but the check
    is written generically rather than assuming that in a way a future
    caller could silently violate."""
    if isinstance(x, complex):
        return _math.isfinite(x.real) and _math.isfinite(x.imag)
    return _math.isfinite(float(x))


# `_domained_binary_fill_value` (the dtype-promotion HEURISTIC that used to
# live here -- "carry a source's fill_value across a promoting op UNLESS it
# equals that source's own dtype default, in which case recompute fresh")
# is GONE, replaced by the real mechanism it was standing in for: explicit-
# vs-defaulted tracking on `MaskedArray` itself (`_fill_value_explicit`,
# `MaskedArray._update_from` -- see their docstrings near the top of this
# file). That heuristic was KNOWN-WRONG on a case its own docstring named
# as an accepted gap: a user who explicitly sets a fill_value EQUAL to
# their array's own dtype default (e.g. `fill_value=999999` on an int64
# array) is indistinguishable from "never set" by VALUE alone, yet real
# numpy carries the explicit one through a promoting op verbatim, dtype and
# all (confirmed live: `ma.divide` on an int64 `MaskedArray` with
# EXPLICIT `fill_value=999999` -- the exact int64 default value, set
# on purpose -- keeps reporting `(999999, 'int64')` after promotion to
# float64 data, not the naive heuristic's `(1e20, 'float64')`). The
# binary family's fill_value propagation is now just `_fv_source` (source
# selection, shared with `make_masked_binary` above) + `_update_from`
# (explicitness-aware raw carry) -- see `make_masked_domained_binary`
# below.


def make_masked_domained_unary(fn, domain_fn):
    """`_MaskedUnaryOperation` family WITH a non-None `.domain` (verified
    directly against real numpy's CPython source, "Case 1.1: Domained
    function" branch of `_MaskedUnaryOperation.__call__`): `fn` is called on
    the FULL raw data first inside `anionpy.errstate(divide='ignore',
    invalid='ignore')`, matching real numpy's own wrapper verbatim (numpy's
    source comment: "nans at masked positions cause RuntimeWarnings, even
    though they are masked. To avoid this we suppress warnings").

    CORRECTION 2026-08-07. This docstring previously argued no errstate call
    was needed on anionpy's side, because `anionpy.sqrt(anionpy.array([-4.0]))`
    is `[nan]` rather than an exception. That premise is true and the
    conclusion drawn from it was false: it reasoned about RAISING and said
    nothing about WARNING. Measured live, with the suppression absent,
    `anionpy.ma.sqrt(MaskedArray([-4.0]))` emitted `RuntimeWarning: invalid
    value encountered in sqrt` and `anionpy.ma.log(MaskedArray([0.0]))`
    emitted `RuntimeWarning: divide by zero encountered in log` where real
    numpy emitted nothing at all -- the mask made the position invisible on
    numpy's side but not on ours. Not every member showed it (`log10(0)`,
    `arcsin(2)`, `arccosh(0)` agreed) because anionpy's FP-error detection
    only covers some ops (task #50), which is precisely why "we never
    observed a warning" was not evidence that none could fire. The mask
    is `~isfinite(computed) | domain_fn(am.data) | am.mask` -- a STRICT
    superset of the plain (non-domained) family's mask (input mask alone):
    verified live the domained mask is ALWAYS a real, fully materialized
    array (never `nomask`), even when `am.mask` itself was `nomask` and
    nothing is actually invalid, because `~isfinite(computed)` and
    `domain_fn(am.data)` are themselves full boolean arrays the moment
    `computed`/`am.data` have ndim >= 1. At masked positions, data is
    copyto-reverted to the original input (`anionpy.where`, same mechanism as
    the plain unary family's `make_masked_unary`) -- verified live.

    Declared instances (all bases already declared `exact` in
    `anionpy/_state/toplevel.py`, verified via `anionpy.__ion_state__` directly,
    not assumed): `sqrt` (`_DomainGreaterEqual(0.0)` -- masks `x < 0`),
    `log`/`log2`/`log10` (`_DomainGreater(0.0)` -- masks `x <= 0`),
    `arcsin`/`arccos` (`_DomainCheckInterval(-1.0, 1.0)`), `arccosh`
    (`_DomainGreaterEqual(1.0)` -- masks `x < 1`), `arctanh`
    (`_DomainCheckInterval(-0.999999999999999, 0.999999999999999)` -- the
    exact, NOT-quite-1.0 boundary real numpy's own source uses, confirmed
    live via `vars(numpy.ma.core.ufunc_domain[numpy.arctanh])`, not
    guessed). `tan` is DELIBERATELY EXCLUDED despite `anionpy.tan` itself being
    declared exact: its domain (`_DomainTan`, masks where `abs(cos(x)) <
    eps`) depends on `cos(x)`, and `anionpy.cos` has a known, documented
    float32 ULP mismatch on some inputs (see the existing `sin =
    make_masked_unary(...)` block comment above) -- building this phase's
    domain check on that same unverified `cos` would poison the MASK, not
    just risk a data ULP difference, so it is declined here for the same
    "never build on an unverified base" discipline as every other decline
    in this phase.
    """

    def wrapped(a):
        am = _as_masked(a)
        with _anionpy.errstate(divide="ignore", invalid="ignore"):
            computed = fn(am.data)
        if not isinstance(computed, _ndarray):
            # 0-d path -- see the module comment above. `domain_fn(am.data)`
            # on a 0-d `anionpy.ndarray` operand collapses to a bare scalar the
            # same way `fn` itself just did (verified live), so `bool(...)`
            # is enough, no `.item()` needed.
            dom_bad = bool(domain_fn(am.data))
            is_masked = am.mask is not nomask and bool(am.mask.item())
            if dom_bad or not _scalar_isfinite(computed) or is_masked:
                return masked
            return computed
        dom = domain_fn(am.data)
        bad = _anionpy.logical_or(_anionpy.logical_not(_anionpy.isfinite(computed)), dom)
        if am.mask is not nomask:
            bad = _anionpy.logical_or(bad, am.mask)
        data = _anionpy.where(bad, am.data, computed)
        # fill_value propagation via `_update_from` (this task's fix), NOT
        # a plain `fill_value=am.fill_value` constructor kwarg: the latter
        # always marks the result explicit (the property never returns
        # `None`), which is exactly the bug this task's report documents
        # and the whole reason all 8 of this family's declarations were
        # revoked -- see `MaskedArray._update_from`'s docstring.
        return MaskedArray(data, mask=bad)._update_from(am)

    wrapped.__name__ = getattr(fn, "__name__", "masked_domained_unary")
    return wrapped


def _domain_ge(critical):
    """`_DomainGreaterEqual(v)`: true (masked) where `x < v` -- verified
    directly against real numpy's CPython source."""

    def f(x):
        return _anionpy.less(x, critical)

    return f


def _domain_gt(critical):
    """`_DomainGreater(v)`: true (masked) where `x <= v` -- verified
    directly against real numpy's CPython source."""

    def f(x):
        return _anionpy.less_equal(x, critical)

    return f


def _domain_interval(lo, hi):
    """`_DomainCheckInterval(a, b)`: true (masked) where `x > b` or `x < a`
    -- verified directly against real numpy's CPython source."""

    def f(x):
        return _anionpy.logical_or(_anionpy.greater(x, hi), _anionpy.less(x, lo))

    return f


sqrt = make_masked_domained_unary(_anionpy.sqrt, _domain_ge(0.0))
log = make_masked_domained_unary(_anionpy.log, _domain_gt(0.0))
log2 = make_masked_domained_unary(_anionpy.log2, _domain_gt(0.0))
log10 = make_masked_domained_unary(_anionpy.log10, _domain_gt(0.0))
arcsin = make_masked_domained_unary(_anionpy.arcsin, _domain_interval(-1.0, 1.0))
arccos = make_masked_domained_unary(_anionpy.arccos, _domain_interval(-1.0, 1.0))
arccosh = make_masked_domained_unary(_anionpy.arccosh, _domain_ge(1.0))
arctanh = make_masked_domained_unary(
    _anionpy.arctanh, _domain_interval(-0.999999999999999, 0.999999999999999)
)


def make_masked_domained_binary(fn, domain_fn):
    """`_DomainedBinaryOperation` (verified directly against real numpy's
    CPython source, `_DomainedBinaryOperation.__call__`): `fn` is called on
    the FULL raw data of both operands first inside `anionpy.errstate(
    divide='ignore', invalid='ignore')`, matching real numpy's own wrapper
    (its `_DomainedBinaryOperation.__call__` spells it `with np.errstate():
    np.seterr(divide='ignore', invalid='ignore')`, which is the same
    suppression by a different idiom).

    CORRECTION 2026-08-07. As with the domained UNARY family above, this
    docstring previously argued the suppression was unnecessary because
    anionpy's own binary domained ufuncs never RAISE on out-of-domain input
    (`anionpy.divide([1.0], [0.0])` is `[inf]`, `anionpy.remainder(..., 0.0)`
    is `[nan]`). Both facts are still true; the conclusion was still false.
    Measured live with the suppression absent, `MaskedArray([1.])/
    MaskedArray([0.])` emitted `RuntimeWarning: divide by zero encountered
    in divide` and the `//` form emitted the `floor_divide` equivalent,
    while real numpy emitted nothing -- a divergence of exactly the same
    class as task #30 (anionpy silently NOT emitting `ComplexWarning`) but
    in the opposite direction: emitting a warning numpy does not. It was
    invisible to the corpus because the corpus compared values, masks,
    dtypes and fill_values, and never captured warnings.

    The mask is `~isfinite(computed) | domain_fn(da,
    db) | mask_a | mask_b` -- ALWAYS a real materialized array once ndim >=
    1, same "strict superset, never nomask" property as the domained unary
    family above (verified live). At masked positions, data is
    copyto-reverted to operand A's raw value (`anionpy.where`, same mechanism
    `make_masked_binary` already uses) -- verified live real numpy's own
    source does exactly this (`np.copyto(result, 0, where=m)` then `result
    += m * da`, which is `da` at every masked position since `m` is 1 there
    and 0 elsewhere -- reproduced here directly via `anionpy.where` rather than
    replaying the zero-then-add arithmetic).

    `fill_value` uses `_fv_source` + `MaskedArray._update_from` (same
    mechanism `make_masked_binary` above now uses, module comment above) --
    prefer whichever ORIGINAL argument was already a `MaskedArray`
    (preferring `a`), propagating its EXPLICITNESS, not just a computed
    value -- verified live this phase's own domained family needs exactly
    this rule (see this task's report for the live probes).

    Declared instances: `divide` (and `true_divide`, a plain alias --
    verified live `np.ma.divide is np.ma.true_divide`), `floor_divide`,
    `remainder` (and `mod`, a plain alias -- verified live `np.ma.mod is
    np.ma.remainder`), `fmod`. All four distinct domain predicates share the
    identical `_DomainSafeDivide` class/tolerance (verified live via
    `vars(numpy.ma.core.ufunc_domain[...])` for all four ufuncs: each is its
    own `_DomainSafeDivide` object but with the SAME lazily-defaulted
    `tolerance` = `finfo(float64).tiny`, i.e. functionally one predicate),
    so `_domain_safe_divide` below is shared across all four rather than
    reimplemented per item. `power`/`hypot` are DELIBERATELY EXCLUDED:
    verified live `type(np.ma.power)` is a plain `function` (not this
    class at all -- it has its own, unreproduced implementation), and
    `type(np.ma.hypot)` is `_MaskedBinaryOperation` (the PLAIN family, not
    domained) but `anionpy.hypot` itself is not yet declared exact
    (`anionpy.__ion_state__["hypot"] is None`) -- building either would be
    declaring correctness nobody has verified.
    """

    def wrapped(a, b):
        am = _as_masked(a)
        bm = _as_masked(b)
        with _anionpy.errstate(divide="ignore", invalid="ignore"):
            computed = fn(am.data, bm.data)
        if not isinstance(computed, _ndarray):
            # 0-d path -- see the module comment above.
            dom_bad = bool(domain_fn(am.data, bm.data))
            a_masked = am.mask is not nomask and bool(am.mask.item())
            b_masked = bm.mask is not nomask and bool(bm.mask.item())
            if dom_bad or not _scalar_isfinite(computed) or a_masked or b_masked:
                return masked
            return computed
        dom = domain_fn(am.data, bm.data)
        bad = _anionpy.logical_or(_anionpy.logical_not(_anionpy.isfinite(computed)), dom)
        if am.mask is not nomask:
            bad = _anionpy.logical_or(bad, am.mask)
        if bm.mask is not nomask:
            bad = _anionpy.logical_or(bad, bm.mask)
        data = _anionpy.where(bad, am.data, computed)
        return MaskedArray(data, mask=bad)._update_from(_fv_source(a, b, am, bm))

    wrapped.__name__ = getattr(fn, "__name__", "masked_domained_binary")
    return wrapped


_DIVIDE_TOL = _anionpy.finfo(_anionpy.float64).tiny


def _domain_safe_divide(da, db):
    """`_DomainSafeDivide`: true (masked) where `abs(da) * tolerance >=
    abs(db)` (`tolerance` = `finfo(float64).tiny`, verified directly against
    real numpy's CPython source) -- in practice this is true almost exactly
    where `db == 0` for ordinary-magnitude operands, plus a vanishing-safety
    margin near it. Computed via `anionpy.abs`/`anionpy.multiply`/
    `anionpy.greater_equal`, three already-in-use Rust-backed calls."""
    return _anionpy.greater_equal(_anionpy.multiply(_anionpy.abs(da), _DIVIDE_TOL), _anionpy.abs(db))


divide = make_masked_domained_binary(_anionpy.divide, _domain_safe_divide)
true_divide = divide
floor_divide = make_masked_domained_binary(_anionpy.floor_divide, _domain_safe_divide)
remainder = make_masked_domained_binary(_anionpy.remainder, _domain_safe_divide)
mod = remainder
fmod = make_masked_domained_binary(_anionpy.fmod, _domain_safe_divide)


# ---------------------------------------------------------------------------
# Phase 7: reduction METHODS (`count`, `sum`, `any`, `all`, `min`, `max`,
# `mean`), plus their `MaskedArray` instance-method counterparts defined
# earlier in this file. All 7 base primitives this phase builds on
# (`anionpy.sum`/`any`/`all`/`min`/`max`/`mean`/`count_nonzero`) are declared
# `"exact"` in `anionpy.__ion_state__` -- verified live before writing a single
# line here (`anionpy.prod` is NOT declared, which is why `prod`/`product` are
# excluded from this phase entirely: never build a reduction method on an
# unverified base).
#
# Every non-trivial edge case below was checked against real numpy 2.5.1
# live, not guessed -- see each function's own docstring for the specific
# reproducing probe. The four cross-cutting rules discovered:
#
#  1. `sum`/`any`/`all`/`min`/`max` on the NOMASK branch NEVER inherit the
#     input's fill_value (fresh default only), but `mean`'s NOMASK branch
#     DOES inherit it (`._update_from(am)`), because real numpy's nomask
#     branch calls `super().mean()` directly on `self` (an ndarray-style
#     view operation that carries `_fill_value` across via
#     `__array_finalize__`), while the other four go through
#     `self.filled(<identity>).<op>()` (a fresh MaskedArray-wrapped
#     construction with no such inheritance).
#  2. `min`/`max` overwrite the OUTPUT's masked positions with the OUTPUT's
#     OWN default fill_value (a real `anionpy.where` select, not a Python
#     loop); `sum`/`any`/`all` do not overwrite anything -- the identity-
#     filled reduction result is used as-is.
#  3. A full reduction to a single value collapses to `masked` (the
#     singleton) whenever every contributing position was masked, for
#     every one of `sum`/`any`/`all`/`min`/`max`/`mean` -- detected via the
#     "is `computed` still an `anionpy.ndarray`?" test already used by the
#     Phase 6 domained families: when reduction fully collapses (axis=None
#     with no keepdims, or an axis argument that empties every remaining
#     dimension), `anionpy.sum`/`any`/`all`/`min`/`max` themselves return a
#     bare numpy/anionpy scalar type, not an `anionpy.ndarray` -- verified live,
#     see e.g. `anionpy.sum(anionpy.array([1,2,3,4]), axis=0)` returning
#     `numpy.int64(10)`, not an `anionpy.ndarray`.
#  4. `count()` is the one outlier: it NEVER returns `masked`, and on its
#     NOMASK branch a full reduction genuinely stays an `anionpy.ndarray`
#     (type, not value) even when the reduction fully collapses -- verified
#     live (`np.ma.masked_array([1,2,3]).count(axis=0)` is
#     `type(...) is numpy.ndarray`, shape `()`, NOT a bare `numpy.int64`,
#     unlike every sibling method here). See `count`'s own docstring for
#     how this is reproduced with zero Python arithmetic.
# ---------------------------------------------------------------------------


def _check_mask_axis(mask, axis, keepdims):
    """Shared helper: reduce a MaskedArray's own `.mask` along `axis` via
    `anionpy.all` (a single Rust-backed call, `mask.all(axis, keepdims)` in
    real numpy's own CPython source for this exact purpose -- a position in
    the OUTPUT is masked iff EVERY contributing input position was masked).
    Returns `nomask` unchanged when the input never had a real mask at all
    (no reduction needed -- there is nothing to combine)."""
    if mask is nomask:
        return nomask
    return _anionpy.all(mask, axis=axis, keepdims=keepdims)


def _count_0d_axis_error(axis):
    """Build the `AxisError` `ma.count`'s 0-d nomask branch must raise.

    The CLASS is captured from a real anionpy call rather than imported, for
    two reasons: production code in `anionpy/` must never `import numpy`, and
    hard-coding a class would let anionpy's own exception type drift away from
    what the rest of the surface raises without any test noticing.

    The MESSAGE cannot be borrowed the way `anionpy.size` borrows `mean`'s,
    because real numpy renders the axis VERBATIM here -- measured
    (/tmp/mg_ma_count_msg.py): `axis (0,) is out of bounds for array of
    dimension 0`, tuple parens and all -- whereas every donor candidate
    (`mean`/`var`/`average`) normalizes `(0,)` down to `0` in its own
    message, and `median` raises `TypeError` for a tuple. No anionpy function
    shares this rule exactly, so the message is reconstructed; `{axis!s}`
    reproduces both spellings (`-1`, and `(0,)` for the tuple forms).
    """
    try:
        _anionpy.mean(_anionpy.full((), 0.0), axis=1)
    except BaseException as exc:  # noqa: BLE001 - capturing a class, not flow
        cls = type(exc)
    else:  # pragma: no cover - guard: donor stopped validating axes
        raise AssertionError(
            "anionpy.mean no longer rejects an out-of-range axis on a 0-d "
            "operand; _count_0d_axis_error can no longer borrow its class")
    return cls(f"axis {axis} is out of bounds for array of dimension 0")


def count(a, axis=None, keepdims=False):
    """`ma.count`: number of UNMASKED elements along `axis`.

    Two genuinely different branches, verified live (real numpy 2.5.1):

    NOMASK branch: `axis=None` and no `keepdims` returns a bare Python
    `int` (verified: `type(np.ma.masked_array([1,2,3]).count())` is
    `int`, not `numpy.int64`) -- everything else (an `axis` given, or
    `keepdims=True`) returns an `anionpy.ndarray` of `intp`, broadcasting the
    SAME per-position count (the product of the reduced axes' lengths, a
    constant since nothing is masked) to every output position, and
    critically STAYS an `anionpy.ndarray` even when the reduction fully
    collapses to a single value (verified live:
    `np.ma.masked_array([1,2,3]).count(axis=0)` is `type(...) is
    numpy.ndarray`, shape `()` -- NOT a bare `numpy.int64`, the one place
    this function's collapse behavior differs from every other reduction
    method in this phase). Reproduced here with ZERO Python arithmetic: a
    same-shaped all-True boolean array is reduced with `anionpy.count_nonzero`
    (a Rust-backed call) to get the correct per-position broadcast value,
    then re-wrapped via `anionpy.full` only in the one case that needs to be
    forced back into `anionpy.ndarray` form (a bare-scalar collapse).

    MASKED branch: always delegates to `anionpy.count_nonzero` on the
    logically-negated mask (`~mask`, via `anionpy.logical_not`) -- this
    naturally reproduces real numpy's own type behavior here too (a
    `numpy.int64` scalar on full collapse, an `anionpy.ndarray` otherwise),
    verified live it matches without any extra collapse-forcing logic.
    """
    am = _as_masked(a)
    m = am.mask
    if m is nomask:
        if am.ndim == 0:
            # FIXED 2026-08-03 (Monday). Real numpy's nomask branch tests
            # the RAW axis for membership in a literal `(None, 0)` -- no
            # negative-axis normalization -- then returns the literal int
            # 1. That unnormalized membership test is the only thing that
            # explains the measured asymmetry (/tmp/mg_ma_count_rule.py):
            # `axis=0` is accepted while `axis=-1` raises, even though the
            # two denote the same axis everywhere else in the library, and
            # `keepdims=True` does NOT promote the result to an array the
            # way it does at every rank >= 1. Note this branch is stricter
            # than the MASKED branch below, which reaches `count_nonzero`
            # and accepts both `0` and `-1`: on a 0-d operand the accept/
            # reject answer genuinely depends on whether a mask is present.
            # Reproducing that inconsistency is the contract; normalizing
            # it away would be a different library.
            if axis is not None and axis != 0:
                raise _count_0d_axis_error(axis)
            return 1
        if axis is None and not keepdims:
            return int(am.data.size)
        ones_mask = _anionpy.full(am.shape, True, dtype=_bool_dtype)
        result = _anionpy.count_nonzero(ones_mask, axis=axis, keepdims=keepdims)
        if isinstance(result, _ndarray):
            return result
        return _anionpy.full((), result, dtype=_anionpy.intp)
    unmasked = _anionpy.logical_not(m)
    return _anionpy.count_nonzero(unmasked, axis=axis, keepdims=keepdims)


def sum(a, axis=None, dtype=None, keepdims=False):
    """`ma.sum`: identity-fill (masked slots -> `0`) then `anionpy.sum`,
    exactly mirroring real numpy's own `self.filled(0).sum(...)` (CPython
    source). Unlike `min`/`max` below, the computed data is used AS-IS --
    masked output positions are NOT overwritten with a fresh fill_value
    (verified live: `masked_array([[1,2],[3,4]],
    mask=[[1,1],[1,1]]).sum(axis=0)` reports `data=[0, 0]`, the raw
    identity-filled sum, not `[999999, 999999]`). Fresh default fill_value
    only on the array-output path (never inherited from the input, even on
    the nomask branch -- verified live, see this phase's module docstring,
    rule 1). A full collapse to a single value returns the `masked`
    singleton if every contributing position was masked, else the bare
    scalar `anionpy.sum` itself already produced (rule 3)."""
    am = _as_masked(a)
    newmask = _check_mask_axis(am.mask, axis, keepdims)
    computed = _anionpy.sum(filled(am, 0), axis=axis, dtype=dtype, keepdims=keepdims)
    if not isinstance(computed, _ndarray):
        if newmask is not nomask and bool(newmask):
            return masked
        return computed
    return MaskedArray(computed, mask=newmask)


def any(a, axis=None, keepdims=False):
    """`ma.any`: identity-fill (masked slots -> `False`) then `anionpy.any`.
    Same shape/collapse/fresh-fill_value contract as `sum` above (verified
    live: `masked_array([True, False],
    mask=[True, True]).any()` is the `masked` singleton;
    `masked_array([True, False], mask=[False, True]).any()` is the bare
    `numpy.bool_(True)`, not wrapped)."""
    am = _as_masked(a)
    newmask = _check_mask_axis(am.mask, axis, keepdims)
    computed = _anionpy.any(filled(am, False), axis=axis, keepdims=keepdims)
    if not isinstance(computed, _ndarray):
        if newmask is not nomask and bool(newmask):
            return masked
        return computed
    return MaskedArray(computed, mask=newmask)


def all(a, axis=None, keepdims=False):
    """`ma.all`: identity-fill (masked slots -> `True`) then `anionpy.all`.
    Same contract as `any`/`sum` above (verified live: fully-masked ->
    `masked` singleton; partially-masked scalar collapse -> bare
    `numpy.bool_`, unwrapped)."""
    am = _as_masked(a)
    newmask = _check_mask_axis(am.mask, axis, keepdims)
    computed = _anionpy.all(filled(am, True), axis=axis, keepdims=keepdims)
    if not isinstance(computed, _ndarray):
        if newmask is not nomask and bool(newmask):
            return masked
        return computed
    return MaskedArray(computed, mask=newmask)


def _minmax(a, axis, keepdims, fill_fn, reduce_fn):
    """Shared body for `min`/`max`: identity-fill via `fill_fn`
    (`minimum_fill_value`/`maximum_fill_value`, Phase 4 -- a value that can
    never WIN the reduction), then `reduce_fn` (`anionpy.min`/`anionpy.max`).
    UNLIKE `sum`/`any`/`all`, the array-output path overwrites every masked
    OUTPUT position with the OUTPUT's own default fill_value (`anionpy.where`,
    one Rust-backed select) -- verified live: `masked_array([[1,2],[3,4]],
    mask=[[1,1],[1,1]]).min(axis=0)` reports `data=[999999, 999999]` (the
    dtype's default fill_value), not the raw identity-filled reduction
    result the way `sum` leaves it."""
    am = _as_masked(a)
    newmask = _check_mask_axis(am.mask, axis, keepdims)
    fv = fill_fn(am)
    computed = reduce_fn(filled(am, fv), axis=axis, keepdims=keepdims)
    if not isinstance(computed, _ndarray):
        if newmask is not nomask and bool(newmask):
            return masked
        return computed
    if newmask is nomask:
        return MaskedArray(computed, mask=nomask)
    default_fv = _default_fill_value(computed.dtype)
    fv_arr = _anionpy.full(computed.shape, default_fv, dtype=computed.dtype)
    data = _anionpy.where(newmask, fv_arr, computed)
    return MaskedArray(data, mask=newmask)


def min(a, axis=None, keepdims=False):
    """`ma.min`: see `_minmax` above for the shared identity-fill +
    masked-output-overwrite contract. Identity fill is
    `minimum_fill_value(a)` (Phase 4) -- the dtype's LARGEST representable
    value, so a masked slot can never win a MINIMUM reduction."""
    return _minmax(a, axis, keepdims, minimum_fill_value, _anionpy.min)


def max(a, axis=None, keepdims=False):
    """`ma.max`: see `_minmax` above. Identity fill is
    `maximum_fill_value(a)` (Phase 4) -- the dtype's SMALLEST representable
    value, so a masked slot can never win a MAXIMUM reduction."""
    return _minmax(a, axis, keepdims, maximum_fill_value, _anionpy.max)


_MEAN_INT_PROMOTE_NAMES = (
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
)


def mean(a, axis=None, dtype=None, keepdims=False):
    """`ma.mean`: two genuinely different branches, verified live.

    NOMASK branch: computed directly via `anionpy.mean(am.data, ...)` (no
    identity-fill detour -- there is nothing to mask out) and, UNLIKE every
    other method in this phase, the array-output result INHERITS the
    input's fill_value verbatim (`._update_from(am)`) -- verified live:
    `masked_array([[1,2],[3,4]], fill_value=-55.0).mean(axis=0).fill_value`
    reports the inherited `-55`, not float64's fresh default `1e20`, while
    `masked_array([1,2,3,4], fill_value=-55).sum().fill_value`-equivalent
    array-output case does NOT inherit (see this phase's module docstring,
    rule 1) -- real numpy's nomask `mean` branch is a `super().mean()`
    view-style call, not a fresh `MaskedArray(...)` construction, which is
    what carries the fill_value across.

    MASKED branch: composed entirely from THIS phase's own `sum`/`count`
    plus the already-declared, already-exact `divide` (Phase 6 domained
    binary) -- `divide(sum(...), count(...))`. This single composition,
    verified live across nomask/masked/fully-masked/int/bool/float16/
    custom-fill_value/axis-given variants, reproduces THREE separate real
    numpy special cases for free, with no bespoke code of their own:
    (a) `count()==0` at a given output position -> `masked`, because
    `divide`'s own domain-safe-divide check (Phase 6) already masks a
    near-zero denominator; (b) the result's fresh-default fill_value on the
    array-output path, via `divide`'s own `_fv_source`/`_update_from`
    machinery (which prefers `sum(...)`'s own freshly-defaulted
    fill_value, since `sum(...)`'s result -- not `count(...)`'s bare
    int/array -- is the only genuine `MaskedArray` operand passed in); and
    (c) integer-division correctness, since `divide` always computes in
    floating point.

    dtype promotion (verified live, matches real numpy's own `_mean`):
    when the caller does NOT pass an explicit `dtype=`, bool/int*/uint*
    inputs are summed in `float64`, and `float16` inputs are summed in
    `float32` with the FINAL divided result cast back down to `float16`
    (`.astype`/scalar-cast, a single width-cast, not arithmetic) -- an
    explicit caller-supplied `dtype=` bypasses this promotion entirely,
    exactly like real numpy.
    """
    am = _as_masked(a)
    if am.mask is nomask:
        computed = _anionpy.mean(am.data, axis=axis, dtype=dtype, keepdims=keepdims)
        if not isinstance(computed, _ndarray):
            return computed
        return MaskedArray(computed, mask=nomask)._update_from(am)

    eff_dtype = dtype
    is_f16 = False
    if eff_dtype is None:
        name = am.data.dtype.name
        if name in _MEAN_INT_PROMOTE_NAMES:
            eff_dtype = _anionpy.float64
        elif name == "float16":
            eff_dtype = _anionpy.float32
            is_f16 = True

    dsum = sum(am, axis=axis, dtype=eff_dtype, keepdims=keepdims)
    cnt = count(am, axis=axis, keepdims=keepdims)
    result = divide(dsum, cnt)

    if is_f16:
        if result is masked:
            return masked
        if isinstance(result, MaskedArray):
            return MaskedArray(result.data.astype(_anionpy.float16), mask=result.mask)._update_from(result)
        return _anionpy.float16(result)
    return result


# ---------------------------------------------------------------------------
# Phase 8: comparison family, a few `_frommethod`-style aliases, and a
# handful of independent reduction-shaped functions. Verified directly
# against `numpy.ma.core`/`numpy.ma.extras`'s own CPython source (not
# guessed), and live against real numpy 2.5.1, same discipline as every
# earlier phase.
#
# THE ONE CROSS-CUTTING BUG THIS PHASE FOUND AND FIXED: `make_masked_binary`
# (Phase 1, above) does not reproduce real numpy's `_MaskedBinaryOperation`
# "scalar case" (CPython source: `if not result.ndim: ...`) -- when BOTH
# operands are 0-d, anionpy's own binary ufuncs (verified live, e.g.
# `anionpy.add(anionpy.array(3.0), anionpy.array(4.0))`) already collapse to a bare
# Python/numpy scalar, exactly like every OTHER full-reduction collapse
# this file already special-cases (Phase 6/7's `isinstance(computed,
# _ndarray)` checks) -- but `make_masked_binary` never checks for this, so
# it always wraps the bare scalar into a full `MaskedArray`, which is wrong
# on two counts: an unmasked 0-d result should come back as the bare scalar
# anionpy itself produced (not a `MaskedArray`), and a masked 0-d result should
# come back as the `masked` singleton (not a `MaskedArray` with mask=True).
# Verified live this is a REAL, PRE-EXISTING divergence already present in
# already-declared `ma.add` (`anionpy.ma.add(MaskedArray(3.0, mask=True),
# MaskedArray(4.0))` returns a `MaskedArray`, real numpy returns the
# `masked` singleton) -- NOT introduced by this phase. This phase does NOT
# touch `make_masked_binary` itself (that would risk every ALREADY-declared
# item built on it -- `add`/`subtract`/`multiply`/... -- outside this
# phase's scope and verification budget); instead, every NEW binary-shaped
# item declared below (`equal`/`not_equal`/.../`hypot`) is built on a new,
# separate factory (`_make_masked_binary_scalar_safe`) that gets this right
# from the start. The pre-existing `make_masked_binary` 0-d gap belongs in
# RUST-QUEUE.md / a future lane's scope, not this one's.
# ---------------------------------------------------------------------------


def _bool_cast(data):
    """Reproduce `np.copyto(bool_result, da, casting='unsafe', where=m)`'s
    effect for a COMPARISON's masked positions (verified directly against
    `_MaskedBinaryOperation.__call__`'s CPython source, and live: real
    numpy's `ma.equal` at a masked position reports `bool(da)` -- e.g. `1`
    at that raw input position reads back as `True`, `0` as `False` --
    NOT a plain revert of the raw numeric value, because the comparator's
    own result array is bool-dtyped and the revert-copy unsafe-casts INTO
    that dtype). A numeric-to-bool unsafe cast is exactly a nonzero test
    (true for any nonzero real, any inf, any NaN -- `bool(nan)` is `True`
    in Python too; true for any complex with a nonzero real OR imaginary
    part), computed here via anionpy's own (undeclared only for an unrelated
    casting=/order=/subok= kwarg gap on the generic ufunc dispatcher --
    verified correct for a plain two-positional-argument call, see
    `anionpy/_state/toplevel.py`'s own `not_equal` comment) `not_equal` ufunc
    against a same-dtype, same-shape zero array -- no numpy involved, no
    Python loop over elements.
    """
    zero = _anionpy.full(data.shape, 0, dtype=data.dtype)
    return _anionpy.not_equal(data, zero)


def _scalar_truthy(x):
    """`bool(x)` for either a bare Python/numpy scalar OR a 0-d
    `anionpy.ndarray` -- plain `bool()` on a 0-d `anionpy.ndarray` raises
    (verified live: `TypeError: len() of unsized object`, an existing
    ndarray-method gap outside this phase's edit scope), so route through
    `.item()` whenever `x` is still an array."""
    return bool(x.item()) if isinstance(x, _ndarray) else bool(x)


def _make_masked_binary_scalar_safe(fn, revert_fn=None):
    """Like Phase 1's `make_masked_binary`, but additionally reproduces
    `_MaskedBinaryOperation.__call__`'s scalar-case branch (see this
    phase's module docstring above for the bug this fixes and why it is a
    NEW factory rather than a fix to the shared one). `revert_fn`, when
    given, transforms operand `a`'s raw data before it is used to revert
    computed-but-masked positions (`_bool_cast` for the comparison family
    below, whose result dtype is bool and needs the unsafe-cast semantics;
    `None` -- the default, used for `hypot` -- means "use `a`'s raw data
    unchanged", matching the plain `_MaskedBinaryOperation` revert
    `make_masked_binary` already implements for `add`/`multiply`/...).
    """

    def wrapped(a, b):
        am = _as_masked(a)
        bm = _as_masked(b)
        da, db = am.data, bm.data
        with _anionpy.errstate(divide="ignore", invalid="ignore"):
            computed = fn(da, db)
        ma_, mb_ = am.mask, bm.mask
        if ma_ is nomask and mb_ is nomask:
            mask = nomask
        elif ma_ is nomask:
            mask = mb_
        elif mb_ is nomask:
            mask = ma_
        else:
            mask = _anionpy.logical_or(ma_, mb_)
        if not isinstance(computed, _ndarray):
            # Scalar case (both operands were 0-d): mirror real numpy's
            # `if not result.ndim: return masked if m else result` exactly.
            # `mask` here is either a bare bool (when both `ma_`/`mb_` were
            # nomask-or-collapsed via `logical_or`, which itself collapses
            # 0-d-vs-0-d to a bare bool, verified live) or -- when only ONE
            # side carried a mask -- still a 0-d `anionpy.ndarray` (`mask =
            # ma_`/`mask = mb_` directly, never routed through
            # `logical_or`). `bool()` on a 0-d anionpy array raises (verified
            # live: `TypeError: len() of unsized object`, a known ndarray
            # gap outside this phase's edit scope), so route through
            # `.item()` whenever `mask` is still an array.
            if mask is not nomask and _scalar_truthy(mask):
                return masked
            return computed
        if mask is nomask:
            data = computed
        else:
            revert = revert_fn(da) if revert_fn is not None else da
            data = _anionpy.where(mask, revert, computed)
        return MaskedArray(data, mask=mask)._update_from(_fv_source(a, b, am, bm))

    wrapped.__name__ = getattr(fn, "__name__", "masked_binary_scalar_safe")
    return wrapped


# Comparison family: real numpy instances of `_MaskedBinaryOperation`
# wrapping `umath.equal`/`not_equal`/`less`/`less_equal`/`greater`/
# `greater_equal` (verified directly against CPython source, `numpy/ma/
# core.py` lines ~1283-1294) -- module-function-only, no `MaskedArray`
# method form exists in real numpy (verified live: `hasattr(masked_array(
# [1]), 'equal')` is `False`).
equal = _make_masked_binary_scalar_safe(_anionpy.equal, revert_fn=_bool_cast)
not_equal = _make_masked_binary_scalar_safe(_anionpy.not_equal, revert_fn=_bool_cast)
less = _make_masked_binary_scalar_safe(_anionpy.less, revert_fn=_bool_cast)
less_equal = _make_masked_binary_scalar_safe(_anionpy.less_equal, revert_fn=_bool_cast)
greater = _make_masked_binary_scalar_safe(_anionpy.greater, revert_fn=_bool_cast)
greater_equal = _make_masked_binary_scalar_safe(_anionpy.greater_equal, revert_fn=_bool_cast)

# `hypot`: also a real `_MaskedBinaryOperation(umath.hypot)` instance
# (CPython source line ~1303), same shape as `add`/`multiply` in Phase 1
# but built on the scalar-safe factory above instead of `make_masked_binary`
# (see this phase's module docstring).
hypot = _make_masked_binary_scalar_safe(_anionpy.hypot)


_ALLTRUE_ERR = "No loop matching the specified signature and casting was found for ufunc logical_and"
_SOMETRUE_ERR = "No loop matching the specified signature and casting was found for ufunc logical_or"


def _dtype_is_bool_like(dtype):
    """Resolve an arbitrary `dtype=` spec (a string like `'bool'`, a
    Python `bool`, an `anionpy.bool_` scalar type, or an already-resolved
    `anionpy.dtype` object) down to "is this the bool dtype". `anionpy.dtype(...)`
    itself is not constructible from a spec (verified live: `TypeError:
    cannot create 'anionpy.dtype' instances`), so resolution goes through
    `anionpy.empty(0, dtype=dtype).dtype` instead -- the same array-construction
    path every OTHER dtype-accepting item in this file already relies on to
    turn a spec into a real dtype object.
    """
    try:
        return _anionpy.empty(0, dtype=dtype).dtype.name == "bool"
    except Exception:
        return False


def _reduce_ravel_0d(tm):
    """Ravel a 0-d `MaskedArray` to shape `(1,)` for the `alltrue`/`sometrue`
    `ufunc.reduce` path; return any other rank unchanged.

    FIXED 2026-08-03 (Monday). `alltrue`/`sometrue` are NOT aliases of
    `all`/`any` (the docstrings below already record the `axis=0`-default
    half of that); they are `logical_{and,or}.reduce`, and `reduce` PROMOTES
    a 0-d operand to shape `(1,)` before reducing. That single mechanism --
    not a table of special cases -- explains every measured cell of the 0-d
    axis boundary, where `alltrue` and `all` genuinely disagree:

        axis        ma.all(0d)     ma.alltrue(0d)
        ()          bool           MaskedArray shape (1,)   <- not a scalar
        (0,) (-1,)  AxisError      collapses like axis=0
        (0,0)       AxisError      ValueError (duplicate)
        None 0 -1   collapse       collapse    (these two agree)
        1 -2 (1,)   AxisError      AxisError   (these two agree)

    Verified against real numpy 2.5.1 by /tmp/mg_ma_ravel_hypo.py, which
    asserts `alltrue(0d, axis=X) == alltrue(1d_len1, axis=X)` for all 11
    axis forms x 2 mask states: 22/22 cells identical, ERROR TYPES INCLUDED
    (AxisError vs ValueError distinguish correctly). The boundary is rank 0
    only -- ranks 1 and 2 already agreed on every form before this fix
    (/tmp/mg_ma_emptyaxis.py), so this deliberately does not touch them.

    numpy's inconsistency between `all` and its own deprecated alias is
    reproduced here, not normalized away: matching numpy is the contract.
    """
    if tm.ndim != 0:
        return tm
    mask = tm.mask if tm.mask is nomask else _anionpy.reshape(tm.mask, (1,))
    return MaskedArray(_anionpy.reshape(tm.data, (1,)), mask=mask,
                       fill_value=tm.fill_value)


def alltrue(target, axis=0, dtype=None):
    """`ma.alltrue`: deprecated alias of `all`, but NOT a pure alias --
    verified live real numpy's own `alltrue = _MaskedBinaryOperation(
    logical_and, 1, 1).reduce` defaults to `axis=0` (NOT `axis=None`, the
    default `ma.all` uses): `np.ma.alltrue(2d_array)` reduces only the
    FIRST axis, while `np.ma.all(2d_array)` fully flattens -- confirmed
    live, see this phase's report.

    `dtype`'s error-parity story is genuinely stranger than a simple
    "non-bool dtype always raises" rule, confirmed directly against
    `_MaskedBinaryOperation.reduce`'s own CPython source AND live
    behavior: `dtype` is passed through to the underlying
    `logical_and.reduce` call ONLY when the target actually carries a mask
    (`m is not nomask`); when the target has NO mask at all, the source's
    `if m is nomask: tr = self.f.reduce(t, axis)` branch never even looks
    at `dtype`, so an incompatible `dtype` is silently ignored rather than
    raising -- verified live: `nma.alltrue(masked_array([True,False,True]),
    dtype='float64')` (no mask) returns `False` with no error, while the
    SAME call on a target carrying an explicit `mask=[True,False,False]`
    raises the `TypeError` below. `dtype=None` (the default) never raises
    either way and behaves identically to `all`.
    """
    tm = _as_masked(target)
    if dtype is not None and tm.mask is not nomask and not _dtype_is_bool_like(dtype):
        raise TypeError(_ALLTRUE_ERR)
    return all(_reduce_ravel_0d(tm), axis=axis)


def sometrue(target, axis=0, dtype=None):
    """`ma.sometrue`: deprecated alias of `any`, same `axis=0` default and
    mask-gated `dtype` error-parity story as `alltrue` above (verified
    live: real numpy's `sometrue = logical_or.reduce`, same
    `_MaskedBinaryOperation.reduce` default-axis AND
    dtype-only-forwarded-when-masked behavior, different underlying op)."""
    tm = _as_masked(target)
    if dtype is not None and tm.mask is not nomask and not _dtype_is_bool_like(dtype):
        raise TypeError(_SOMETRUE_ERR)
    return any(_reduce_ravel_0d(tm), axis=axis)


def count_masked(arr, axis=None):
    """`ma.count_masked`: number of MASKED elements along `axis` (the
    complement of `count` above). Verified directly against
    `numpy.ma.extras.count_masked`'s CPython source: `getmaskarray(arr)
    .sum(axis)` -- a real bool array (via Phase 0's `getmaskarray`, which
    materializes an all-False array for `nomask` input rather than
    special-casing it) summed with the already-declared-exact `anionpy.sum`
    (bool -> int, matching real numpy's own bool-sum-counts-True rule).
    Returns whatever shape/type `anionpy.sum` itself produces (bare int on
    full collapse, `anionpy.ndarray` otherwise) -- no masked-output wrapping
    at all, matching real numpy (the RESULT of `count_masked` is a plain
    count, never itself a `MaskedArray`).
    """
    am = _as_masked(arr)
    m = getmaskarray(am)
    return _anionpy.sum(m, axis=axis)


def allequal(a, b, fill_value=True):
    """`ma.allequal`: verified directly against `numpy.ma.core.allequal`'s
    CPython source. `fill_value=True` (the default) means "a masked
    position counts as equal, whichever side or sides it came from";
    `fill_value=False` means the two arrays are considered UNEQUAL as soon
    as the COMBINED mask is non-trivial (source: `elif fill_value: ...
    else: return False`).

    Critically, "combined mask exists" is judged after the exact same
    `mask_or(..., shrink=True)` (Phase 3, above) every other mask-combine
    in this file already goes through -- an all-False combined mask (e.g.
    both inputs are 0-size, or both masks happen to carry no `True` at
    all) SHRINKS back down to `nomask` rather than counting as "a mask
    exists", verified live: `allequal` on two empty (`shape=(0,)`) masked
    arrays with `fill_value=False` returns `True`, not `False`, because
    `mask_or` of two empty masks shrinks to `nomask` before the
    `fill_value` branch is ever reached. An earlier draft of this function
    combined masks with a bare `logical_or` (skipping the shrink step) and
    got exactly this case wrong -- caught by this phase's own differential
    probe, not by inspection.
    """
    # NOTE 2026-08-07 (Monday, boxing sweep): this function used to wrap
    # every non-`False`-literal return in a bare `bool(...)`, silently
    # downgrading `_anionpy.all(...)`'s genuine `numpy.bool_` return (and
    # `_anionpy.equal`'s own already-correct 0-d `numpy.bool_` scalar)
    # back to a plain Python `bool` -- the same scalar-boxing defect
    # class as `.fill_value`/`ma.mean`/`ma.default_fill_value`, caught
    # here by the standing type-identity sweep, not by a fresh symptom.
    # Real numpy's `allequal` (CPython source, `numpy/ma/core.py`)
    # returns `numpy.bool_` from `d.all()` / `dm.filled(True).all(None)`
    # on every path except the literal `else: return False` (a genuine
    # bare Python `bool`, verified live) -- reproduced exactly below by
    # never re-wrapping an already-`numpy.bool_`-typed value in `bool()`.
    am = _as_masked(a)
    bm = _as_masked(b)
    mask = mask_or(am.mask, bm.mask)
    d = _anionpy.equal(am.data, bm.data)
    if mask is nomask:
        if isinstance(d, _ndarray):
            return _anionpy.all(d)
        return d
    if not fill_value:
        return False
    if isinstance(d, _ndarray):
        true_arr = _anionpy.full(d.shape, True, dtype=_bool_dtype)
        d_filled = _anionpy.where(mask, true_arr, d)
        return _anionpy.all(d_filled)
    # 0-d scalar collapse with a truthy mask: the single masked position
    # counts as equal (fill_value=True got here, already checked above).
    # `mask`/`d` land here as 0-d `_ndarray`s (verified live: `mask_or`/
    # `_anionpy.equal` on 0-d `MaskedArray` operands return a 0-d
    # `anionpy.ndarray`, NOT an already-boxed scalar) -- `[()]`-indexing a
    # 0-d array is this file's existing, already-verified path to a
    # genuine `numpy.bool_` scalar (same mechanism `_box_typed_scalar`'s
    # docstring documents for indexing/reduction generally), so index
    # rather than re-wrap in `bool()`.
    winner = mask if _scalar_truthy(mask) else d
    return winner[()] if isinstance(winner, _ndarray) else winner


# `amax`/`amin`: verified directly against CPython source
# (`numpy/ma/core.py`: `from numpy import ... amax, amin`) -- real numpy's
# `ma.amax`/`ma.amin` are LITERALLY the plain top-level `numpy.amax`/
# `numpy.amin` functions, re-exported into the `numpy.ma` namespace
# unchanged (verified live: `np.ma.amax is np.amax`); calling
# `numpy.amax(masked_array)` dispatches to the array's own bound `.max()`
# method (numpy's generic reduction protocol), which for a `MaskedArray`
# IS `ma.max` (Phase 7, above) -- so the faithful anionpy composition is a
# direct name alias onto this file's own already-declared-exact `max`/
# `min`, not a new wrapper.
amax = max
amin = min


def argmax(a, axis=None, fill_value=None, keepdims=False):
    """`ma.argmax`: verified directly against `MaskedArray.argmax`'s
    CPython source (`numpy/ma/core.py`): identity-fill masked positions
    with `fill_value` (defaulting to `maximum_fill_value(a)`, Phase 4 --
    the dtype's SMALLEST representable value, so a masked slot can never
    win an ARGMAX), then a plain unmasked `argmax` on the filled data --
    the RESULT is never itself masked (a plain int or `anionpy.ndarray` of
    indices, exactly what `anionpy.argmax` itself already produces, matching
    real numpy: `ma.argmax` never returns the `masked` singleton, even for
    an all-masked input -- verified live). `out=` is not supported here
    (no declared item in this file supports it, see Phase 7's module
    docstring); the module-level function form here is `_frommethod`-
    equivalent to `MaskedArray.argmax` in real numpy, not a separate
    implementation -- both call sites below share this one function.
    """
    am = _as_masked(a)
    fv = maximum_fill_value(am) if fill_value is None else fill_value
    d = filled(am, fv)
    return _anionpy.argmax(d, axis=axis, keepdims=keepdims)


def argmin(a, axis=None, fill_value=None, keepdims=False):
    """`ma.argmin`: see `argmax` above -- identical shape, mirrored
    identity fill (`minimum_fill_value`, Phase 4 -- the dtype's LARGEST
    representable value, so a masked slot can never win an ARGMIN)."""
    am = _as_masked(a)
    fv = minimum_fill_value(am) if fill_value is None else fill_value
    d = filled(am, fv)
    return _anionpy.argmin(d, axis=axis, keepdims=keepdims)


def cumsum(a, axis=None, dtype=None):
    """`ma.cumsum`: verified directly against `MaskedArray.cumsum`'s
    CPython source: masked positions are identity-filled with `0` for the
    computation, but UNLIKE the Phase 7 reductions, the mask is NOT
    reduced along `axis` -- every output position corresponds 1:1 to an
    input position (a cumulative op, not a collapsing one), so the output
    mask is simply the input's mask, reshaped to match `axis=None`'s
    flattening when that default is used (verified live: `MaskedArray(
    [[1,2],[3,4]], mask=[[T,F],[F,T]]).cumsum()` -- no `axis` given --
    returns a 1-D length-4 result whose mask is the 2-D input mask
    RAVELED, not left 2-D).

    CORRECTED 2026-08-03 (Monday). This docstring previously continued "an
    explicit `axis=` leaves the mask's shape alone, since the output shape
    then still matches the input's", and the code implemented exactly that
    -- reshaping the mask ONLY on the `axis is None` branch. True at rank
    >= 1, false at rank 0: a 0-d input with an explicit `axis=0`/`-1`
    yields DATA of shape `(1,)` (numpy's cumulative ops never return 0-d)
    while the mask stayed `()`, producing a MaskedArray whose mask did not
    match its data. Measured against real numpy 2.5.1 by
    /tmp/mg_ma_rank_scope.py: np `data(1,)/mask(1,)` vs anionpy
    `data(1,)/mask()`, at ranks 1 and 2 no divergence in any axis form.
    The rule that covers every rank uniformly is "reshape the mask to the
    OUTPUT's shape", which is what the code now does -- at rank >= 1 that
    reshape is an identity, so no existing behavior moves.

    This one hid behind a shape-only check: an earlier pass confirmed the
    DATA shape was `(1,)`, called cumsum fixed, and never looked at the
    mask. Verifying half a container is not verifying it.
    `fill_value` is NEVER inherited here either (verified live: an
    explicit input `fill_value` does not survive `.cumsum()` -- real
    numpy's own implementation reaches it via `result.view(type(self))`,
    an `__array_finalize__`-based VIEW that -- verified live -- does NOT
    carry `_fill_value` across, unlike the `_update_from`-based propagation
    every OTHER family in this file uses), so this composition correctly
    omits any `_update_from` call, unlike almost every other item here.
    """
    am = _as_masked(a)
    data = _anionpy.cumsum(filled(am, 0), axis=axis, dtype=dtype)
    mask = nomask if am.mask is nomask else _anionpy.reshape(am.mask, data.shape)
    return MaskedArray(data, mask=mask)


def cumprod(a, axis=None, dtype=None):
    """`ma.cumprod`: see `cumsum` above for the shared contract (mask
    reshaped-not-reduced, fresh non-inherited fill_value). Identity fill is
    `1` (the multiplicative identity) instead of `0`."""
    am = _as_masked(a)
    data = _anionpy.cumprod(filled(am, 1), axis=axis, dtype=dtype)
    mask = nomask if am.mask is nomask else _anionpy.reshape(am.mask, data.shape)
    return MaskedArray(data, mask=mask)


def ptp(a, axis=None, keepdims=False):
    """`ma.ptp`: verified directly against `MaskedArray.ptp`'s CPython
    source: `self.max(axis, keepdims=keepdims) - self.min(axis,
    keepdims=keepdims)` (the `out=`/custom-`fill_value` parameters real
    numpy's `ptp` also accepts are not supported here, matching every
    other reduction in this file). Composed entirely from THIS file's own
    already-declared `max`/`min` (Phase 7) and `subtract` (Phase 1) --
    handles every collapse combination those two can produce: if either
    side fully collapses to the `masked` singleton, the whole result is
    `masked` (`masked - anything` is `masked` in real numpy too, verified
    live); if both sides are bare (unmasked full-collapse) scalars, plain
    Python subtraction is exact (same values `anionpy.subtract` would give,
    since there is nothing left to mask); otherwise at least one side is a
    real `MaskedArray` and `subtract`'s own already-verified composition
    takes over.
    """
    hi = max(a, axis=axis, keepdims=keepdims)
    lo = min(a, axis=axis, keepdims=keepdims)
    if hi is masked or lo is masked:
        return masked
    if isinstance(hi, MaskedArray) or isinstance(lo, MaskedArray):
        return subtract(hi, lo)
    return hi - lo


# ---------------------------------------------------------------------------
# Dunder support (this task): the three genuinely-new algorithm shapes
# `MaskedArray`'s magic methods (defined directly on the class, above) need
# -- none of them is a re-shape of an existing `make_masked_*` family, see
# each dunder's own docstring on the class body for the live-numpy-source
# evidence establishing why. Defined here (module scope, after every
# function these three call by name -- `mask_or`, `getmask`, `getdata`,
# `filled` is not needed here) so the ordinary forward-reference mechanism
# already used throughout this file (Phase 7/8 methods calling module-level
# functions not yet defined at class-body-execution time) applies uniformly.
# ---------------------------------------------------------------------------
def _comparison_dunder(self, other, compare_fn, eqne_fn=None):
    """Shared body for `__eq__`/`__ne__`/`__lt__`/`__le__`/`__gt__`/
    `__ge__` -- see each one's own docstring on the class body above for
    the CPython-source citation this reproduces (`MaskedArray._comparison`,
    `numpy/ma/core.py` lines ~4193-4266). `eqne_fn` is the SAME raw
    comparator as `compare_fn`, passed again ONLY for `__eq__`/`__ne__` (to
    apply at masked positions -- "both masked -> equal / one masked ->
    unequal" is exactly what re-running `eqne_fn` on the two MASKS
    themselves computes), `None` for every other comparison (no masked-
    position override -- real numpy's own `if compare in (operator.eq,
    operator.ne):` guard, reproduced here as "was an eqne_fn even given").
    """
    omask = getmask(other)
    smask = self.mask
    mask = mask_or(smask, omask, copy=True)
    odata = getdata(other)
    sdata = self.data
    check = compare_fn(sdata, odata)
    if not isinstance(check, _ndarray):
        # Scalar case (both operands 0-d): real numpy's own
        # `if isinstance(check, (np.bool, bool)): return masked if mask
        # else check` -- reproduced via `_scalar_truthy` for the same
        # "`mask` may still be a bare 0-d anionpy.ndarray" reason
        # `_make_masked_binary_scalar_safe` above already documents.
        return masked if (mask is not nomask and _scalar_truthy(mask)) else check
    if mask is not nomask and eqne_fn is not None and mask.shape == check.shape:
        # Shape-mismatch guard: real numpy additionally broadcasts `mask`
        # to `check`'s shape here when they differ (self/other genuinely
        # different shapes needing numpy-style broadcasting). Not
        # reproduced -- out of this task's verification budget/corpus
        # (every case this task declares keeps self/other same-shaped) --
        # so a genuinely shape-mismatched comparison simply skips this
        # masked-position override rather than risk a wrong broadcast; the
        # comparison as a whole is NOT declared for that combination (see
        # anionpy/_state/ma.py's own scope note for these six items).
        sm_arr = smask if smask is not nomask else _anionpy.full(mask.shape, False, dtype=_bool_dtype)
        om_arr = omask if omask is not nomask else _anionpy.full(mask.shape, False, dtype=_bool_dtype)
        mask_check = eqne_fn(sm_arr, om_arr)
        check = _anionpy.where(mask, mask_check, check)
    result = MaskedArray(check, mask=mask)._update_from(self)
    if result._fill_value_explicit:
        try:
            result._fill_value = _cast_fill_value(result._fill_value, _bool_dtype)
        except Exception:
            result._fill_value = _cast_fill_value(True, _bool_dtype)
    return result


def _is_weak_pyscalar(x):
    """True iff `x` is a bare Python `bool`/`int`/`float`/`complex` -- NOT a
    `numpy`/`anionpy` scalar (`anionpy.int64(2)` etc, none of which are
    `int`/`float`/`complex` subclasses -- verified live, see this task's
    report), not an `ndarray`, and not a `MaskedArray`.

    This is the NEP-50 "weak" operand category: real numpy's ufunc
    machinery keeps such an operand's promotion contribution weak (it
    defers entirely to the OTHER operand's dtype) only when the raw Python
    object reaches the ufunc unconverted. `getdata`'s `_array(a)` fallback
    materializes ANY non-MaskedArray into a concrete `anionpy.ndarray`
    first, which -- verified live this task -- destroys that weakness and
    forces the strong `int64`/`float64` promotion result instead, even
    though the VALUE, mask, and fill_value all remain correct (this is
    exactly why the corpus this task extended did not previously catch it:
    every prior binary-dunder case compared dtype but never against a
    narrower-than-int64/float64 array operand). `type(x) in (...)` (not
    `isinstance`) is deliberate: `bool` is an `int` subclass, so
    `isinstance(x, int)` would also accept it (fine), but `isinstance`
    against `numpy.bool_`/`anionpy.bool_` could accept THOSE too if numpy
    ever changed their base classes -- an exact `type(x) in {...}` set
    cannot silently widen that way.
    """
    return type(x) in (bool, int, float, complex)


def _generic_unary_dunder(self, fn):
    """Shared body for `__neg__`/`__pos__`/`__abs__`/`__invert__` -- see
    each one's own docstring on the class body above for the live-numpy
    evidence this reproduces: computed everywhere (no revert), mask ALWAYS
    materialized via `getmaskarray` (never `nomask` identity, even from a
    `nomask` input -- verified live, asymmetric with the binary family
    below), `fill_value` from `self` alone (`_update_from(self)`).
    """
    computed = fn(self.data)
    mask = getmaskarray(self)
    if not isinstance(computed, _ndarray):
        return masked if _scalar_truthy(mask) else computed
    return MaskedArray(computed, mask=mask)._update_from(self)


def _generic_binary_dunder(self, other, fn):
    """Shared body for `__and__`/`__rand__`/`__or__`/`__ror__`/`__xor__`/
    `__rxor__` -- see each one's own docstring on the class body above for
    the live-numpy evidence this reproduces: computed everywhere (no
    revert), mask is `mask_or(self.mask, getmask(other))` (shrinks an
    all-False combination back to `nomask` identity -- verified live this
    is NOT the same as a raw unshrunk `logical_or`), `fill_value` from
    `self` alone regardless of dunder direction (verified live for both
    `a & 5` and `5 & a`, see the class body docstring).

    2026-08-07 weak-scalar fix (this task): `other` is passed to `fn`
    UNMATERIALIZED when it is a bare Python `bool`/`int`/`float`/`complex`
    (`_is_weak_pyscalar`), not funneled through `getdata` -- `getdata`'s
    `_array(a)` fallback strips NEP-50 weak-scalar promotion, which
    real numpy's raw `ndarray.__and__`/`__mod__` slot wrappers never do
    (they never materialize the scalar at all; see
    `_generic_domained_binary_dunder`'s docstring for the fuller measured
    picture). `getmask(other)` is untouched: it already returns `nomask`
    for anything that is not a `MaskedArray`, scalar or not.
    """
    other_data = other if _is_weak_pyscalar(other) else getdata(other)
    computed = fn(self.data, other_data)
    mask = mask_or(self.mask, getmask(other))
    if not isinstance(computed, _ndarray):
        return masked if (mask is not nomask and _scalar_truthy(mask)) else computed
    return MaskedArray(computed, mask=mask)._update_from(self)


def _generic_domained_binary_dunder(self, other, fn, domain_fn, fill_const, reflect=False):
    """Shared body for `__mod__`/`__rmod__` (ma batch 4, this task).

    `MaskedArray.__mod__`/`__rmod__` are NOT defined in `numpy.ma.core` at
    all (verified live: `numpy.ma.core.MaskedArray.__mod__` is a literal
    `<slot wrapper '__mod__' of 'numpy.ndarray' objects>`) -- they dispatch
    through the SAME generic `__array_wrap__` ufunc-context hook
    `_generic_binary_dunder` above already reproduces for `__and__`/`__or__`
    /`__xor__`, but `np.remainder` additionally carries a domain
    (`numpy.ma.core.ufunc_domain[np.remainder]` is a `_DomainSafeDivide`,
    the identical predicate `_domain_safe_divide` above already implements
    for the module-level `ma.remainder`/`ma.mod` family), and
    `__array_wrap__`'s `if domain is not None:` branch does something a
    plain domain-less dunder never does: at every domain-violating
    position it OVERWRITES the computed value with a literal constant
    (`ufunc_fills[np.remainder][-1]`, `1` -- read directly off real numpy's
    own `ufunc_fills` dict, not a guess) and ORs the domain mask into the
    result mask. Read directly from `numpy.ma.core.MaskedArray.
    __array_wrap__`'s CPython source (`d = domain(*input_args)...
    np.copyto(result, fill_value, where=d); m = m | d`), then verified live
    end-to-end this task: `MaskedArray([1.,2.], mask=[False,False]) %
    MaskedArray([0.,4.])` gives data `[1., 2.]` -> `[1., 2.]`? -- concretely,
    `[7.] % [0.]` gives data `[1.]` (the constant, not `nan`) and mask
    `[True]`, where the SAME inputs through the module-level `ma.remainder`
    (`make_masked_domained_binary`) instead revert to operand A's raw value
    (`7.0`) -- a genuinely different fill rule, confirmed side-by-side on
    identical inputs, not assumed from the source read alone.

    THIS FUNCTION IS NOT REUSABLE FOR `__imod__`: measured live (see this
    task's report) that in-place `a %= b` computes the domain check against
    `a`'s data AFTER the raw in-place remainder has already overwritten it
    -- for float dtype this means the domain check runs against a `nan`
    (IEEE754 `remainder(x, 0.0)`), and `nan >= anything` is `False`,
    silently DEFEATING the fill/mask step entirely (the raw `nan` is left
    in place, unmasked) -- while for int dtype the raw in-place remainder
    writes `0` at that position (verified live, `anionpy`'s own `%=`
    already matches this exactly), and `abs(0)*tiny >= abs(0)` IS `True`,
    so the domain fill/mask step fires normally. This float-vs-int
    divergence is a genuine, deterministic, reproducible artifact of the
    C-level "compute in place, THEN run `__array_wrap__` against the
    (already-mutated) buffer" ordering -- not noise -- and is reproduced
    exactly by `_generic_domained_binary_idunder` below, which computes
    `domain_fn` against `self.data` AFTER mutating it, not before.

    `reflect=True` (for `__rmod__`) swaps which operand is the dividend
    (`da`) vs divisor (`db`) fed to both `fn` and `domain_fn` -- `other %
    self` semantically, matching real numpy's own `b.__rmod__(a) == a % b`
    contract -- while `mask`/`fill_value` still come from `self` alone
    regardless of `reflect` (verified live: `5.0 % b` and `[list] % b`
    both report `b`'s own fill_value, matching `_generic_binary_dunder`'s
    already-established "whichever instance's dunder METHOD is executing"
    rule, which does not depend on operand order for a commutative op but
    DOES still apply here even though `%` itself is not commutative --
    only the VALUE computation is order-sensitive, not the fill_value
    source).

    2026-08-07 weak-scalar fix (this task): same `_is_weak_pyscalar` guard
    as `_generic_binary_dunder` -- `other` reaches both `fn` and
    `domain_fn` unmaterialized when it is a bare Python scalar, on
    WHICHEVER side of `da`/`db` the `reflect` swap puts it. `domain_fn`
    (`_domain_safe_divide`) is itself built from `anionpy.abs`/`multiply`/
    `greater_equal`, the same Rust-backed ufunc entry points `fn` uses, so
    it accepts a raw Python scalar operand exactly as `fn` does -- verified
    live this task, not assumed. The domain predicate's OWN result is
    always a bool value/array regardless of which side promoted weak, so
    this cannot change the domain outcome, only which dtype the surviving
    `computed` values are re-promoted to.
    """
    sdata = self.data
    odata = other if _is_weak_pyscalar(other) else getdata(other)
    da, db = (odata, sdata) if reflect else (sdata, odata)
    # NOT suppressed: verified live real numpy's raw dunder path emits
    # `RuntimeWarning: divide by zero encountered in remainder` for int
    # dtype div-by-zero (no errstate anywhere around the raw `fn(...)`
    # call in `__array_wrap__` -- only the DOMAIN check itself is wrapped,
    # per its own CPython source, `with np.errstate(divide='ignore',
    # invalid='ignore'): d = domain(*input_args)`). Wrapping `fn` here too,
    # as an earlier revision of this function did (copied from the
    # module-level `make_masked_domained_binary`'s DELIBERATE suppression),
    # was a measured bug: it silently swallowed that warning where real
    # numpy emits it.
    computed = fn(da, db)
    with _anionpy.errstate(divide="ignore", invalid="ignore"):
        dom = domain_fn(da, db)
    mask = mask_or(self.mask, getmask(other))
    if not isinstance(computed, _ndarray):
        # 0-d path: `anionpy.remainder` degenerates a 0-d/0-d call to a bare
        # scalar, but real numpy's `__array_wrap__` has NO scalar-unwrap
        # branch at all -- `result = obj.view(type(self))` is ALWAYS a
        # (0-d) MaskedArray, and the ONLY early-return is the masked-
        # singleton collapse (`result.shape == () and m`). Verified live:
        # `MaskedArray(5.0) % MaskedArray(2.0)` (in-domain, unmasked) is a
        # 0-d `MaskedArray`, NOT a bare `float`/`numpy.float64` -- an
        # earlier revision of this function returned the bare scalar here
        # (copying the module-level domained-binary family's OWN,
        # genuinely different, 0-d contract) and was caught by this
        # exact live probe.
        dom_bad = _scalar_truthy(dom)
        if dom_bad:
            computed = fill_const
            mask = mask_or(mask, dom)
        if mask is not nomask and _scalar_truthy(mask):
            return masked
        return MaskedArray(computed, mask=mask)._update_from(self)
    if bool(_anionpy.any(dom)):
        _anionpy.copyto(computed, fill_const, where=dom)
        mask = mask_or(mask, dom)
    return MaskedArray(computed, mask=mask)._update_from(self)


def _generic_domained_binary_idunder(self, other, domain_fn, fill_const):
    """Shared body for `__imod__` (ma batch 4, this task) -- see
    `_generic_domained_binary_dunder`'s docstring above for why this is a
    SEPARATE, not-shared, function: the domain check here runs against
    `self.data` AFTER the raw in-place `%=` has already overwritten it
    (matching real numpy's measured behavior exactly, float-vs-int
    divergence included), not before. `other`'s data is captured up front
    since `other` is never mutated by this operation (verified live both
    sides: `b` is unchanged after `a %= b`).

    Mutates `self` in place and returns `self` (verified live: `id(a %= b
    ; a) == id(a_before)` is `True` on both sides, same object -- this is
    what makes it `__imod__` rather than a disguised `__mod__`). The raw
    in-place remainder (`self.data %= odata`) is NOT wrapped in
    `errstate`: verified live real numpy's raw in-place int `%=` by zero
    DOES emit `RuntimeWarning: divide by zero encountered in remainder`
    (unlike the module-level `_DomainedBinaryOperation` family, which
    explicitly suppresses this around its own internal `fn(...)` call --
    `make_masked_domained_binary`'s own docstring) -- there is no
    suppression anywhere in the inherited-`ndarray` in-place dunder path,
    so none is added here either; the warning capture is part of this
    item's differential corpus (`tests/differential/ma_warning_cases.py`).
    """
    odata = getdata(other)
    mask = mask_or(self.mask, getmask(other))
    self._data %= odata
    with _anionpy.errstate(divide="ignore", invalid="ignore"):
        dom = domain_fn(self._data, odata)
    if bool(_anionpy.any(dom)):
        _anionpy.copyto(self._data, fill_const, where=dom)
        mask = mask_or(mask, dom)
    self._mask = mask
    return self
