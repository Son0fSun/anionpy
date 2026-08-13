"""NEW test-cases file: the array-protocol dunders on `anionpy.ndarray`
(`__array_finalize__`, `__array_wrap__`, `__array_priority__`,
`__array_namespace__`, `__array_function__`, `__array_ufunc__`,
`__dlpack_device__`, `__array_interface__`, `__array_struct__`), 2026-08-13
(Monday, array-protocol-dunders ticket).

WHY THIS NEEDED A NEW FILE
---------------------------------------------------------------------------
All nine items had ZERO differential coverage before this task (confirmed
by grep across the whole `tests/differential/` tree) -- they did not exist
on `anionpy.ndarray` at all until `ionp-py/src/array_protocol.rs` (this same
task) added them as a separate `#[pymethods] impl PyArray` block (pyo3's
`multiple-pymethods` feature, same pattern as `ndarray_attrs.rs`). None of
these fit `kind="unary"`/`kind="binary_op"`/`kind="method"` cleanly: several
return non-ndarray objects (a dict, a `PyCapsule`, a module, `None`, a bare
float), and two (`__array_function__`, `__array_ufunc__`) are dispatch
protocols whose whole point is to be called directly with a *callable*
argument, not through the ordinary arg-corpus machinery. `kind="custom"`
with a dedicated adapter per item, following `ndarray_float_cases.py`'s
shape, is the correct fit for all nine.

`__array_prepare__` is deliberately ABSENT from this file: removed from
numpy in the 2.x line (`hasattr(numpy.ndarray, "__array_prepare__")` is
`False` on numpy 2.5.1, confirmed directly) -- not ours to implement, and
`tools/numpy_surface.json`'s own `"exploded"` manifest for `ndarray`
correctly omits it (12 entries, not 13).

`__buffer__` and full `__dlpack__` (as opposed to `__dlpack_device__`) are
ALSO deliberately absent -- not implemented in `array_protocol.rs` either.
`__buffer__` needs real PEP 688 `tp_as_buffer` C-level FFI wiring in PyO3
that overlaps another agent's in-flight "no views" work; full `__dlpack__`
needs a named ("dltensor") `PyCapsule` wrapping a `DLManagedTensor` struct
with a producer-side rename-on-consume deleter contract. Both are
substantially more machinery than every other item here and were reported,
not built, per this ticket's explicit "report genuine impossibilities and
move on" instruction. `__dlpack_device__` alone (no capsule, no FFI) IS
implemented and covered below.

MEASUREMENT NOTES PER ITEM (see each adapter for the invocation this was
checked against; every adapter below independently repeats the same
invocation this file's differential run performs, out-of-corpus probing was
also done directly against a live build at /tmp/probe_protocol.py before
any of this was declared -- see that script's real-numpy-2.5.1 output for
the semantics this Rust implementation was built to match):

- `__array_finalize__`: real numpy's version is a no-op returning `None`
  for a base `ndarray` (its real work is in subclasses, which `anionpy`
  cannot have -- `PyArray` is not subclassable from Python). Verified: both
  sides return `None` for `obj=None` and `obj=self`.
- `__array_wrap__`: verified with `return_scalar=True` on a 0-d wrapped
  array (must return a numpy/anionpy SCALAR, not a 0-d array) and
  `return_scalar=False` (must return the wrapped array object unchanged --
  identity-preserving passthrough, checked via `is`, not just `==`, inside
  the adapter itself since the harness's own equality-based comparator
  cannot express an identity requirement).
- `__array_priority__`: real numpy's base `ndarray.__array_priority__` is
  `0.0`. `anionpy.ndarray` matches. (Note `anionpy.memmap` -- a composition
  wrapper, not a real subclass -- separately overrides this to `-100.0` on
  itself; irrelevant here, this item is about `ndarray` only.)
- `__array_namespace__`: cannot be compared by identity or `==` across two
  different libraries' namespace modules -- there is no numpy object that
  "equals" the anionpy module. Verified structurally instead: the returned
  object must expose the small set of Array-API-standard names both
  `numpy`'s own namespace return and `anionpy`'s `__array_namespace__`
  return are required to carry (`asarray`, `arange`, `sum`, `equal`,
  `astype`) -- same functional bar as calling it for its intended purpose
  (an Array-API consumer library asking "what can I call on this").
- `__array_function__` / `__array_ufunc__`: dispatch protocols, verified by
  actually dispatching a real callable through them (`np.sum`/`anionpy.sum`
  for the former; `np.add`/anionpy's own `add` ufunc object, both
  `__call__` and `reduce`, for the latter) and comparing the numeric result.
- `__dlpack_device__`: real numpy returns `(1, 0)` (`kDLCPU`, device 0) for
  any in-memory array. `anionpy` matches (CPU-only, always device 0).
- `__array_interface__`: real numpy's dict has keys `data` (address,
  readonly), `strides`, `typestr`, `descr`, `shape`, `version`. The `data`
  address is obviously never equal between two independent arrays holding
  independent buffers, so it is excluded from the cross-implementation
  structural comparison -- but it is NOT left unverified: the adapter reads
  the raw bytes back from that exact address via `ctypes.string_at` and
  byte-compares them against `arr.tobytes()`, which is the actual
  correctness property the address exists to carry. `strides=None`
  (numpy's own convention for C-contiguous) is exercised alongside an
  explicit non-contiguous case (transposed 2-D) that must produce real byte
  strides. String dtypes (`S`/`U`) are verified separately to raise
  `NotImplementedError` on both... no -- see the note in
  `_interface_custom_cases` below: numpy DOES expose `__array_interface__`
  for `S`/`U` (fixed-width, still a flat buffer in real numpy's own
  storage), so this is a genuine, measured anionpy gap (ragged
  `Buffer::S`/`Buffer::U` storage, one heap allocation per element, no flat
  pointer to expose) -- NOT declared "exact", reported in the final report
  instead.
- `__array_struct__`: same data-pointer situation as `__array_interface__`,
  verified the same way but through the raw C struct instead of a Python
  dict -- `ctypes` reads the `PyArrayInterface` struct out of the capsule
  (`PyCapsule_GetPointer` via `ctypes.pythonapi`, matching the unnamed-
  capsule convention numpy itself uses) on BOTH sides and structurally
  compares `nd`/`typekind`/`itemsize`/`flags`(masked to the bits this
  implementation sets)/`shape`/byte-`strides`, then independently confirms
  the `data` field on each side points at bytes matching that side's own
  `arr.tobytes()`.

NOT verified: `__array_wrap__`'s `context` parameter (numpy's own signature
still accepts it but real numpy 2.5.1 never populates it with anything this
adapter can trigger -- confirmed by reading numpy's `_wrapit`/ufunc dispatch
source, it is always `None` in every code path reachable from pure Python
call sites as of 2.x); anything about `__array_prepare__` (removed, not
ours); `__buffer__`/full `__dlpack__` (not implemented, see above).
"""
from __future__ import annotations

import ctypes

import numpy as np

from registry import ItemSpec

import anionpy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_case(label, factory):
    return (label, (factory,), {})


def _err(e: BaseException):
    return ("err", type(e).__name__, str(e))


# ---------------------------------------------------------------------------
# __array_finalize__
# ---------------------------------------------------------------------------

def _finalize_numpy_adapter(factory):
    a = factory()
    try:
        r1 = np.ndarray.__array_finalize__(a, None)
        r2 = np.ndarray.__array_finalize__(a, a)
        return ("ok", r1 is None, r2 is None)
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _finalize_ionp_adapter(factory):
    a = factory()
    ia = anionpy.array(a)
    try:
        r1 = ia.__array_finalize__(None)
        r2 = ia.__array_finalize__(ia)
        return ("ok", r1 is None, r2 is None)
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _finalize_cases():
    return [
        _make_case("f64_1d", lambda: np.array([1.0, 2.0, 3.0])),
        _make_case("i32_2d", lambda: np.arange(6, dtype=np.int32).reshape(2, 3)),
        _make_case("0d", lambda: np.array(5.0)),
    ]


# ---------------------------------------------------------------------------
# __array_wrap__
# ---------------------------------------------------------------------------

def _wrap_numpy_adapter(factory):
    a, b = factory()
    r = a + b
    try:
        w_default = a.__array_wrap__(r)
        default_is_r = w_default is r
        w_no_scalar = a.__array_wrap__(r, None, False)
        no_scalar_is_r = w_no_scalar is r
        scalar0d = np.array(float(r.reshape(-1)[0]) if r.ndim else float(r))
        sw_true = a.__array_wrap__(scalar0d, None, True)
        sw_false = a.__array_wrap__(scalar0d, None, False)
        return (
            "ok", default_is_r, no_scalar_is_r,
            type(sw_true).__name__, float(sw_true),
            type(sw_false).__name__, float(sw_false),
        )
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _wrap_ionp_adapter(factory):
    a_np, b_np = factory()
    ia, ib = anionpy.array(a_np), anionpy.array(b_np)
    r = ia + ib
    try:
        w_default = ia.__array_wrap__(r)
        default_is_r = w_default is r
        w_no_scalar = ia.__array_wrap__(r, None, False)
        no_scalar_is_r = w_no_scalar is r
        flat0 = float(anionpy.reshape(r, (-1,))[0]) if r.ndim else float(r)
        scalar0d = anionpy.array(np.array(flat0))
        sw_true = ia.__array_wrap__(scalar0d, None, True)
        sw_false = ia.__array_wrap__(scalar0d, None, False)
        return (
            "ok", default_is_r, no_scalar_is_r,
            type(sw_true).__name__, float(sw_true),
            type(sw_false).__name__, float(sw_false),
        )
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _wrap_cases():
    return [
        _make_case("f64_1d", lambda: (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))),
        _make_case("i32_2d", lambda: (
            np.arange(6, dtype=np.int32).reshape(2, 3),
            np.arange(6, dtype=np.int32).reshape(2, 3) * 2,
        )),
    ]


# ---------------------------------------------------------------------------
# __array_priority__
# ---------------------------------------------------------------------------

def _priority_numpy_adapter(factory):
    a = factory()
    return float(np.ndarray.__array_priority__.__get__(a))


def _priority_ionp_adapter(factory):
    a = factory()
    ia = anionpy.array(a)
    return float(ia.__array_priority__)


def _priority_cases():
    return [_make_case("any", lambda: np.array([1.0]))]


# ---------------------------------------------------------------------------
# __array_namespace__
# ---------------------------------------------------------------------------

_NAMESPACE_PROBE_NAMES = ("asarray", "arange", "sum", "equal", "astype")


def _namespace_numpy_adapter(factory):
    a = factory()
    try:
        ns = np.ndarray.__array_namespace__(a)
        return ("ok", tuple(hasattr(ns, n) for n in _NAMESPACE_PROBE_NAMES))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _namespace_ionp_adapter(factory):
    a = factory()
    ia = anionpy.array(a)
    try:
        ns = ia.__array_namespace__()
        return ("ok", tuple(hasattr(ns, n) for n in _NAMESPACE_PROBE_NAMES))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _namespace_cases():
    return [_make_case("any", lambda: np.array([1.0]))]


# ---------------------------------------------------------------------------
# __array_function__
# ---------------------------------------------------------------------------

def _array_function_numpy_adapter(factory):
    a = factory()
    try:
        res = np.ndarray.__array_function__(a, np.sum, (np.ndarray,), (a,), {})
        return ("ok", float(res))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _array_function_ionp_adapter(factory):
    a = factory()
    ia = anionpy.array(a)
    try:
        res = ia.__array_function__(anionpy.sum, (anionpy.ndarray,), (ia,), {})
        return ("ok", float(res))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _array_function_cases():
    return [
        _make_case("f64_1d", lambda: np.array([1.0, 2.0, 3.0, 4.0])),
        _make_case("i32_2d", lambda: np.arange(6, dtype=np.int32).reshape(2, 3)),
    ]


# Added 2026-08-13: the two adapters above call `__array_function__` DIRECTLY
# with anionpy's own `anionpy.sum` (ionp side) / real numpy's own dispatcher
# already holding a base-`ndarray` `self` (numpy side) -- neither ever
# exercises numpy's REAL `__array_function__` protocol dispatch actually
# calling INTO this method with a genuine numpy public dispatcher function
# (e.g. `np.sum`) while `self` is an `anionpy.ndarray`. That specific path is
# exactly where a naive "just call func(*args, **kwargs)" implementation
# recurses forever, because `func` IS the same public dispatcher that is
# about to notice `self` again and re-enter this method (caught live via
# `/private/tmp/probe_array_function_recursion.py`: `np.sum(anionpy_array)`
# stack-overflowed before the fix in `array_protocol.rs` that resolves the
# equivalent callable from anionpy's own namespace by name instead of
# calling `func` back). This pair closes that gap by going through the
# genuine `np.sum(...)` entry point on both sides.
def _array_function_dispatch_numpy_adapter(factory):
    a = factory()
    try:
        return ("ok", float(np.sum(a)))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _array_function_dispatch_ionp_adapter(factory):
    a = factory()
    ia = anionpy.array(a)
    try:
        return ("ok", float(np.sum(ia)))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# __array_ufunc__
# ---------------------------------------------------------------------------

def _array_ufunc_numpy_adapter(factory):
    a, b = factory()
    try:
        call_res = np.ndarray.__array_ufunc__(a, np.add, "__call__", a, b)
        reduce_res = np.ndarray.__array_ufunc__(a, np.add, "reduce", a)
        return ("ok", call_res.tolist(), float(reduce_res))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _array_ufunc_ionp_adapter(factory):
    a_np, b_np = factory()
    ia, ib = anionpy.array(a_np), anionpy.array(b_np)
    try:
        call_res = ia.__array_ufunc__(anionpy.add, "__call__", ia, ib)
        reduce_res = ia.__array_ufunc__(anionpy.add, "reduce", ia)
        return ("ok", call_res.tolist(), float(reduce_res))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _array_ufunc_cases():
    return [
        _make_case("f64_1d", lambda: (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))),
        _make_case("i32_2d", lambda: (
            np.arange(6, dtype=np.int32).reshape(2, 3),
            np.arange(6, dtype=np.int32).reshape(2, 3) * 3,
        )),
    ]


# Added 2026-08-13, per explicit coordinator directive: "__array_ufunc__ is
# the one worth getting exactly right: returning NotImplemented vs a value
# vs raising changes what numpy does on the OTHER operand, so test it with a
# real numpy array on both the left and the right side of the operator, not
# just ours on the left." The two adapters above call `__array_ufunc__`
# directly with anionpy's OWN `add` ufunc object on both sides -- that path
# never engages real numpy's ufunc dispatch machinery at all, and so never
# caught the recursion bug this file's module docstring update describes: a
# naive "forward to `ufunc`/`getattr(ufunc, method)`" implementation
# re-invokes the SAME real numpy ufunc that is already mid-dispatch on
# `self`, which notices `self` again and calls back into this method --
# infinite recursion (caught live via `/private/tmp/probe_mixed_ufunc.py`:
# `np.add(numpy_arr, anionpy_arr)` and `np.add(anionpy_arr, numpy_arr)` both
# stack-overflowed before the `array_protocol.rs` fix that resolves
# `ufunc.__name__` against anionpy's own namespace instead of calling back
# through the original `ufunc`). This pair exercises the genuine mixed-
# operand path numpy's dispatcher takes in real life, with anionpy on each
# side in turn.
def _array_ufunc_mixed_numpy_adapter(factory):
    a, b = factory()
    try:
        left = np.add(a, b)
        right = np.add(b, a)
        return ("ok", left.tolist(), right.tolist())
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _array_ufunc_mixed_ionp_adapter(factory):
    a_np, b_np = factory()
    ib = anionpy.array(b_np)
    try:
        # numpy array on the LEFT, anionpy array on the RIGHT: numpy's own
        # `np.add.__call__` inspects both operands, finds `ib` doesn't
        # subclass `ndarray`/lack a compatible `__array_ufunc__`, and defers
        # to `ib.__array_ufunc__(np.add, "__call__", a_np, ib)`.
        left = np.add(a_np, ib)
        # anionpy on the LEFT, numpy on the RIGHT -- same deferral, mirrored.
        ia = anionpy.array(a_np)
        right = np.add(ia, b_np)
        return ("ok", left.tolist(), right.tolist())
    except BaseException as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# __dlpack_device__
# ---------------------------------------------------------------------------

def _dlpack_device_numpy_adapter(factory):
    a = factory()
    return tuple(int(x) for x in a.__dlpack_device__())


def _dlpack_device_ionp_adapter(factory):
    a = factory()
    ia = anionpy.array(a)
    return tuple(int(x) for x in ia.__dlpack_device__())


def _dlpack_device_cases():
    return [
        _make_case("f64_1d", lambda: np.array([1.0, 2.0])),
        _make_case("0d", lambda: np.array(1.0)),
    ]


# ---------------------------------------------------------------------------
# __array_interface__
# ---------------------------------------------------------------------------

_STRING_KINDS = ("S", "U")


def _normalize_interface(arr, d: dict):
    data_addr, readonly = d["data"]
    nbytes = arr.nbytes
    live_bytes = ctypes.string_at(data_addr, nbytes)
    bytes_ok = live_bytes == arr.tobytes()
    return (
        bool(readonly),
        d["strides"],
        d["typestr"],
        d["descr"],
        d["shape"],
        d["version"],
        bytes_ok,
    )


def _interface_numpy_adapter(factory):
    a = factory()
    try:
        d = a.__array_interface__
        return ("ok", *_normalize_interface(a, d))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _interface_ionp_adapter(factory):
    a = factory()
    ia = anionpy.array(a)
    try:
        d = ia.__array_interface__
        return ("ok", *_normalize_interface(ia, d))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _interface_cases():
    cases = []
    dtypes = [
        np.bool_, np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128,
    ]
    for dt in dtypes:
        cases.append(_make_case(
            f"1d/{dt.__name__}",
            lambda dt=dt: np.arange(6, dtype=dt) if dt not in (np.complex64, np.complex128)
            else (np.arange(6, dtype=np.float64) + 1j * np.arange(6, dtype=np.float64)).astype(dt),
        ))
    cases.append(_make_case("c_contig_2d", lambda: np.arange(12, dtype=np.float64).reshape(3, 4)))
    cases.append(_make_case("transposed_2d_noncontig", lambda: np.arange(12, dtype=np.float64).reshape(3, 4).T))
    cases.append(_make_case("f_order_2d", lambda: np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))))
    cases.append(_make_case("0d", lambda: np.array(7.0)))
    cases.append(_make_case("empty", lambda: np.array([], dtype=np.float64)))
    return cases


# NOTE: `S`/`U` (string) dtypes are deliberately NOT exercised through
# `_interface_cases`/`_struct_cases` above, and are NOT registered as a
# differential item at all here. Real numpy DOES support
# `__array_interface__`/`__array_struct__` on `S`/`U` (fixed-width, still a
# flat buffer in numpy's own storage); anionpy's `Buffer::S`/`Buffer::U` are
# ragged (`Vec<Vec<u8>>`/`Vec<Vec<u32>>`, one heap allocation per element,
# confirmed by reading `ionp-core/src/buffer.rs`), so there is no flat
# pointer to expose and `array_protocol.rs` raises `NotImplementedError`
# there by design rather than fabricate one. The two sides are asymmetric
# by construction (numpy succeeds, anionpy correctly refuses), so this is
# NOT registered as a passing/failing differential item -- doing so would
# either always fail (polluting the baseline with a "failure" that is
# actually a documented, correct refusal) or require papering over a real
# capability gap with `exception_equivalences`, which this task's rules
# forbid. Measured directly instead (out-of-corpus, not shipped):
# `np.array([b"ab", b"cd"], dtype="S2").__array_interface__` returns a real
# dict on real numpy; `anionpy.array(...).__array_interface__` on the
# equivalent anionpy array raises `NotImplementedError` with a message
# naming the ragged-storage reason. Reported in the final report as a named
# gap, not silently dropped.


# ---------------------------------------------------------------------------
# __array_struct__
# ---------------------------------------------------------------------------

class _RawArrayInterface(ctypes.Structure):
    _fields_ = [
        ("two", ctypes.c_int),
        ("nd", ctypes.c_int),
        ("typekind", ctypes.c_char),
        ("itemsize", ctypes.c_int),
        ("flags", ctypes.c_int),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("data", ctypes.c_void_p),
        ("descr", ctypes.c_void_p),
    ]


_FLAG_MASK = 0x1 | 0x2 | 0x100 | 0x200 | 0x400  # CONTIGUOUS|FORTRAN|ALIGNED|NOTSWAPPED|WRITEABLE

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _read_struct(arr, capsule):
    raw_ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, None)
    st = ctypes.cast(raw_ptr, ctypes.POINTER(_RawArrayInterface))[0]
    nd = st.nd
    shape = tuple(st.shape[i] for i in range(nd))
    strides = tuple(st.strides[i] for i in range(nd)) if st.strides else None
    live_bytes = ctypes.string_at(st.data, arr.nbytes)
    bytes_ok = live_bytes == arr.tobytes()
    return (
        st.two, nd, st.typekind, st.itemsize, st.flags & _FLAG_MASK,
        shape, strides, bytes_ok,
    )


def _struct_numpy_adapter(factory):
    a = factory()
    try:
        return ("ok", *_read_struct(a, a.__array_struct__))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _struct_ionp_adapter(factory):
    a = factory()
    ia = anionpy.array(a)
    try:
        return ("ok", *_read_struct(ia, ia.__array_struct__))
    except BaseException as e:  # noqa: BLE001
        return _err(e)


def _struct_cases():
    return _interface_cases()  # same corpus; same underlying data


ARRAY_PROTOCOL_SPECS: dict[str, ItemSpec] = {
    "ndarray.__array_finalize__": ItemSpec(
        name="ndarray.__array_finalize__", kind="custom", scalar_like=True,
        custom_cases=_finalize_cases,
        numpy_adapter=_finalize_numpy_adapter, ionp_adapter=_finalize_ionp_adapter,
    ),
    "ndarray.__array_wrap__": ItemSpec(
        name="ndarray.__array_wrap__", kind="custom", scalar_like=True,
        custom_cases=_wrap_cases,
        numpy_adapter=_wrap_numpy_adapter, ionp_adapter=_wrap_ionp_adapter,
    ),
    "ndarray.__array_priority__": ItemSpec(
        name="ndarray.__array_priority__", kind="custom", scalar_like=True,
        custom_cases=_priority_cases,
        numpy_adapter=_priority_numpy_adapter, ionp_adapter=_priority_ionp_adapter,
    ),
    "ndarray.__array_namespace__": ItemSpec(
        name="ndarray.__array_namespace__", kind="custom", scalar_like=True,
        custom_cases=_namespace_cases,
        numpy_adapter=_namespace_numpy_adapter, ionp_adapter=_namespace_ionp_adapter,
    ),
    "ndarray.__array_function__": ItemSpec(
        name="ndarray.__array_function__", kind="custom", scalar_like=True,
        custom_cases=_array_function_cases,
        numpy_adapter=_array_function_numpy_adapter, ionp_adapter=_array_function_ionp_adapter,
    ),
    "ndarray.__array_ufunc__": ItemSpec(
        name="ndarray.__array_ufunc__", kind="custom", scalar_like=True,
        custom_cases=_array_ufunc_cases,
        numpy_adapter=_array_ufunc_numpy_adapter, ionp_adapter=_array_ufunc_ionp_adapter,
    ),
    "ndarray.__array_ufunc__.mixed_dispatch": ItemSpec(
        name="ndarray.__array_ufunc__.mixed_dispatch", kind="custom", scalar_like=True,
        custom_cases=_array_ufunc_cases,
        numpy_adapter=_array_ufunc_mixed_numpy_adapter, ionp_adapter=_array_ufunc_mixed_ionp_adapter,
    ),
    "ndarray.__array_function__.mixed_dispatch": ItemSpec(
        name="ndarray.__array_function__.mixed_dispatch", kind="custom", scalar_like=True,
        custom_cases=_array_function_cases,
        numpy_adapter=_array_function_dispatch_numpy_adapter, ionp_adapter=_array_function_dispatch_ionp_adapter,
    ),
    "ndarray.__dlpack_device__": ItemSpec(
        name="ndarray.__dlpack_device__", kind="custom", scalar_like=True,
        custom_cases=_dlpack_device_cases,
        numpy_adapter=_dlpack_device_numpy_adapter, ionp_adapter=_dlpack_device_ionp_adapter,
    ),
    "ndarray.__array_interface__": ItemSpec(
        name="ndarray.__array_interface__", kind="custom", scalar_like=True,
        custom_cases=_interface_cases,
        numpy_adapter=_interface_numpy_adapter, ionp_adapter=_interface_ionp_adapter,
    ),
    "ndarray.__array_struct__": ItemSpec(
        name="ndarray.__array_struct__", kind="custom", scalar_like=True,
        custom_cases=_struct_cases,
        numpy_adapter=_struct_numpy_adapter, ionp_adapter=_struct_ionp_adapter,
    ),
}
