"""NEW test-cases file: self-aliasing (`a //= a`, and the whole
`ndarray.__iXXX__` family) + partial/overlapping-slice-aliasing corpus for
the 2026-08-02 "fix the double-mutable-borrow" task.

THE BUG THIS CORPUS TARGETS
---------------------------------------------------------------------------
Before this task's `ionp-py/src/lib.rs` fix, every in-place dunder was
`fn __ixxx__(&mut self, other: &Bound<'_, PyAny>)`. `&mut self` holds a
mutable PyO3 `PyCell` borrow for the ENTIRE call. When `other` is the same
Python object as `self` (`a //= a`, or the direct call `a.__ifloordiv__(a)`
this corpus mostly uses so it does not depend on `ndarray.__setitem__`,
which anionpy does not implement at all -- see the "PARTIAL/OVERLAPPING-SLICE
ALIASING" section below), the coercion helper
(`coerce_operand`/`coerce_operand_for_truediv`/`matmul::coerce_matmul_operand`)
tried to ALSO borrow `other` (== the same PyCell `self` already holds
mutably borrowed) via `other.extract::<PyRef<PyArray>>()`, and PyO3's own
runtime borrow-checker raised `RuntimeError: Already mutably borrowed`.
numpy has no such problem: `ndarray.__ifloordiv__` reads `other`'s buffer
and writes `self`'s buffer as two ordinary C-level pointer reads/writes,
with no borrow-tracking layer in between.

The fix (see `lib.rs`'s `coerce_inplace_operand`/
`coerce_inplace_operand_for_truediv` and the `__imatmul__` inline check):
switch every in-place dunder's receiver from `&mut self` to
`mut slf: PyRefMut<'_, Self>`, which exposes `.as_ptr()` (a raw FFI pointer,
available on `PyRef`/`PyRefMut` per `pyo3-0.29.0/src/pycell.rs`), and
short-circuit with `slf.inner.clone()` (an `Arc`-bump, not a data copy --
see `ionp_core::NdArray`'s `Arc<Buffer>` field) whenever
`other.as_ptr() == slf.as_ptr()`, instead of ever re-borrowing the same
PyCell. This is safe because every one of these ops already computes a
fresh output buffer before reassigning `slf.inner` -- self-aliasing was
never a data race, only a spurious PyO3 borrow-tracking false positive.

WHY THIS IS A SEPARATE FILE/NAMESPACE, NOT MORE CASES ON
`inplace_cases.py`'s EXISTING 7-ITEM `INPLACE_SPECS`
---------------------------------------------------------------------------
`inplace_cases.py` (pre-existing, not written by this task, not modified by
it) already declares `ndarray.__iadd__ __isub__ __imul__ __itruediv__
__iand__ __ior__ __ixor__` and its `_probe(op)` adapter already checks
identity-preservation + buffer-aliasing for a NON-aliased `(a, b)` pair
(`b` is an independently-filled array, never `a` itself, by construction --
`_value_cases`/`_dtype_promotion_cases`/`_broadcast_mismatch_cases` all
build two separate arrays). It genuinely never exercises `other is self`,
so it cannot be reused, and per this task's scope `inplace_cases.py`/
`registry.py` are not touched. Merged into `registry.REGISTRY` here, at
this file's own bottom, with a collision check, imported by `run.py` for
its side effect only (the run.py-import pattern `floordiv_crossing_cases.py`
established) -- fresh `"alias/<op>"` keys, colliding with nothing.

BLAST-RADIUS FINDING #1 (fixed by this task): SELF-ALIASING
---------------------------------------------------------------------------
`_self_alias_cases` builds ONE array `a` per case and the probe calls
`getattr(a, op)(a)` -- literally the same Python object passed as both
receiver and operand, the exact shape of `a //= a`. Verified via a `/tmp`
probe (13/13 in-place dunders) that every one of these raised
`RuntimeError: Already mutably borrowed` before the fix and computes the
correct values after it.

BLAST-RADIUS FINDING #2 (found, NOT fixed by this task, honestly left
undeclared): PARTIAL/OVERLAPPING-SLICE ALIASING DOES NOT WRITE BACK
---------------------------------------------------------------------------
`_view_alias_cases`' probe takes ONE base array `a`, derives two DIFFERENT
Python view objects `v1 = a[1:]` / `v2 = a[:-1]` (different PyCells, so this
never hits the borrow bug at all -- two distinct objects, two distinct
`.as_ptr()`s, even though they share the same underlying `Arc<Buffer>`),
and calls `v1.__ifloordiv__(v2)` directly, NOT `a[1:] //= a[:-1]`.

[CORRECTED 2026-08-03] The original reason given here was that
`anionpy.ndarray` "has no `__setitem__` at all", so the `a[1:] //= a[:-1]`
spelling was unreachable. `ndarray.__setitem__` landed 2026-08-02 and that
is no longer true. Re-measured: `a[1:] //= a[:-1]` now works and MATCHES
numpy (`[8,4,2,1]` -> `[8,0,0,0]` on both sides), because Python expands it
to an explicit `a.__setitem__(...)` that copies the result back into the
base.

The direct-dunder call below is therefore no longer a workaround for an
unreachable syntax -- it is now the ONLY spelling that still reaches the
write-back gap, which makes this probe MORE valuable than when it was
written, not less. Keeping it, and keeping it calling the dunder directly.
Manually verified with `/tmp/ionp_inplace/probe_slice_alias2.py`: `v1`'s OWN
computed values are byte-correct (matches numpy's `v_np.__ifloordiv__(...)`
exactly), but numpy's base array reflects the mutation (`[2,2,2,2,2]`)
while anionpy's base array is UNCHANGED (`[2,4,8,16,32]`, still the original).
Root cause: every `__iXXX__` does `slf.inner = out.cast_to(...)` -- a full
REASSIGNMENT of the view's own `NdArray` struct to a freshly computed
buffer, not a strided write into the buffer the view and its base still
share. This is a genuine, separate, pre-existing architectural gap
(orthogonal to the self-aliasing borrow fix -- fixing it would mean
rewriting every in-place op's write path to write element-by-element
through the view's existing strides into the shared buffer, not "restructure
the borrow"). `_view_alias_cases`' probe returns BOTH the view's own values
AND the base array's post-call values so this divergence is visible in the
ledger rather than hidden by a corpus that only checks the view's own
slot -- these cases are EXPECTED to fail on the base-array slot for every
op, and nothing in `anionpy/_state/ndarray.py` claims otherwise (see that
file's comments on this task's declarations).

BLAST-RADIUS FINDING #3: non-aliased self-alias-corpus siblings
---------------------------------------------------------------------------
`_nonaliased_baseline_cases` is a small sanity corpus (same op, `b`
independently filled, no aliasing at all) for the 6 dunders
`inplace_cases.py` does not cover (`__ifloordiv__ __imod__ __ipow__
__ilshift__ __irshift__ __imatmul__`) -- confirming the self-aliasing fix
did not regress the ordinary non-aliased path for exactly the ops this
task's fix touched. `__ifloordiv__ __imod__ __ilshift__ __irshift__`
already have a `kind="binary_op"` value-only item in `registry.py`
(pre-existing, undisturbed); this corpus additionally checks
identity-preservation the way `inplace_cases.py`'s probe does, which that
`kind="binary_op"` item never has.
"""
from __future__ import annotations

import math

import numpy as np

import _bootstrap  # noqa: F401
import registry
from registry import ItemSpec

ALIAS_SEED = 20260802


def _nan_safe_value(v):
    """`harness.compare_values`' `scalar_like=True` path is plain Python
    `==` on the whole returned tuple -- correct for every value these
    probes return EXCEPT float `nan`, where `nan == nan` is `False` by
    IEEE 754 definition, which would flag a byte-identical
    nan-vs-nan pair (e.g. both sides' `0.0 // 0.0`) as a spurious mismatch.
    This is a property of the comparison, not of the values under test (the
    div-by-zero cases below deliberately construct nan on purpose to prove
    aliasing doesn't change numpy's own 0-divisor behavior) -- so nan gets
    mapped to a distinct string sentinel before the tuple leaves the probe,
    the same nan-vs-nan-should-compare-equal intent
    `np.array_equal(..., equal_nan=True)` expresses elsewhere in this
    corpus family, just implemented for the plain-`==` path this module's
    probes use instead."""
    if isinstance(v, complex):
        re = "nan" if math.isnan(v.real) else v.real
        im = "nan" if math.isnan(v.imag) else v.imag
        return (re, im)
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return v


def _nan_safe_tuple(arr: np.ndarray) -> tuple:
    return tuple(_nan_safe_value(v) for v in arr.ravel().tolist())


def _rng() -> np.random.Generator:
    return np.random.default_rng(ALIAS_SEED)


def _fill(rng: np.random.Generator, shape: tuple[int, ...], dtype) -> np.ndarray:
    dt = np.dtype(dtype)
    n = int(np.prod(shape)) if shape else 1
    if dt.kind == "b":
        data = rng.integers(0, 2, size=n).astype(bool)
    elif dt.kind in "iu":
        info = np.iinfo(dt)
        lo = max(info.min, 1)  # >=1: every self-alias case doubles as its
        hi = min(info.max, 50)  # own "nonzero"/"no-negative-shift" filler,
        data = rng.integers(lo, hi, size=n, dtype=np.int64).astype(dt)  # see module docstring cases below for the deliberate negative/zero ones
    elif dt.kind == "f":
        data = (rng.uniform(1.0, 9.0, size=n)).astype(dt)
    elif dt.kind == "c":
        real = rng.uniform(1.0, 9.0, size=n)
        imag = rng.uniform(1.0, 9.0, size=n)
        data = (real + 1j * imag).astype(dt)
    else:  # pragma: no cover
        raise ValueError(f"unhandled dtype kind {dt.kind!r}")
    return data.reshape(shape)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _self_alias_probe(op: str):
    """`a`/`b` are the SAME Python object -- literally `getattr(a, op)(a)`,
    the exact call shape `a //= a` makes. Flat tuple of plain Python values
    only, see `inplace_cases.py`'s `_probe` docstring for why."""
    def adapter(a):
        before_id = id(a)
        ret = getattr(a, op)(a)
        id_preserved = (id(a) == before_id)
        ret_is_self = (ret is a)
        arr = np.asarray(a)
        return (
            _nan_safe_tuple(arr),
            arr.shape,
            str(arr.dtype),
            id_preserved,
            ret_is_self,
        )
    return adapter


def _view_alias_probe(op: str):
    """`v1 = a[1:]`, `v2 = a[:-1]` -- two DIFFERENT view objects sharing
    `a`'s buffer, overlapping in memory. Calls `v1.__iXXX__(v2)` directly,
    which since 2026-08-02 deliberately BYPASSES the now-implemented
    `ndarray.__setitem__` -- the `a[1:] //= a[:-1]` spelling routes through
    it and passes, so only the direct call still exercises the view
    write-back gap (see module docstring). Returns the view's own post-call values AND the base
    array's post-call values, so a write-back divergence shows up as a
    mismatch on the LAST slot specifically, distinguishable from a
    wrong-value bug on the view's own computation (the first two slots)."""
    def adapter(a):
        v1 = a[1:]
        v2 = a[:-1]
        getattr(v1, op)(v2)
        v1_arr = np.asarray(v1)
        base_arr = np.asarray(a)
        return (
            _nan_safe_tuple(v1_arr),
            str(v1_arr.dtype),
            _nan_safe_tuple(base_arr),
        )
    return adapter


def _nonaliased_probe(op: str):
    """Same shape as `inplace_cases.py`'s `_probe`, minus the buffer-
    aliasing check (that property is already what `_view_alias_probe`
    above measures, deliberately, for the divergence it is expected to
    find) -- just value/identity, for the 6 dunders that probe doesn't
    cover."""
    def adapter(a, b):
        before_id = id(a)
        ret = getattr(a, op)(b)
        id_preserved = (id(a) == before_id)
        ret_is_self = (ret is a)
        arr = np.asarray(a)
        return (
            _nan_safe_tuple(arr),
            arr.shape,
            str(arr.dtype),
            id_preserved,
            ret_is_self,
        )
    return adapter


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

def _self_alias_cases(dtypes: list, shape=(6,)) -> list:
    rng = _rng()
    cases = []
    for dtype in dtypes:
        dt = np.dtype(dtype)
        a = _fill(rng, shape, dtype)
        cases.append((f"self_alias/{dt.name}", (a,), {}))
    return cases


def _view_alias_cases(dtypes: list) -> list:
    rng = _rng()
    cases = []
    for dtype in dtypes:
        dt = np.dtype(dtype)
        a = _fill(rng, (5,), dtype)
        cases.append((f"view_alias/{dt.name}", (a,), {}))
    return cases


def _nonaliased_cases(dtypes: list) -> list:
    rng = _rng()
    cases = []
    for dtype in dtypes:
        dt = np.dtype(dtype)
        a = _fill(rng, (6,), dtype)
        b = _fill(rng, (6,), dtype)
        cases.append((f"nonaliased/{dt.name}", (a, b), {}))
    return cases


_ARITH_DTYPES = [np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
_TRUEDIV_DTYPES = [np.float32, np.float64, np.complex64, np.complex128]
_BITWISE_DTYPES = [np.bool_, np.int8, np.int16, np.int32, np.uint8, np.uint32]
_SHIFT_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32]
_FLOORMOD_DTYPES = [np.int32, np.int64, np.float32, np.float64]
_POW_DTYPES = [np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]


def _zero_containing(dtype):
    dt = np.dtype(dtype)
    if dt.kind in "iu":
        return np.array([0, 3, -4, 7], dtype=dt)
    if dt.kind == "f":
        return np.array([0.0, 3.5, -4.25, 7.0], dtype=dt)
    raise ValueError(dt)


def _build_arith(op: str):
    def build():
        cases = _self_alias_cases(_ARITH_DTYPES)
        # scalar-operand self-alias sibling: `a op= a` where `a` is a
        # 0-d array (numpy's own scalar-array form) -- still true identity
        # aliasing, just at ndim==0, an axis the (6,)-shape cases above
        # never cross.
        cases += _self_alias_cases(_ARITH_DTYPES, shape=())
        return cases
    return build


def _build_truediv(op: str):
    def build():
        cases = _self_alias_cases(_TRUEDIV_DTYPES)
        cases += _self_alias_cases(_TRUEDIV_DTYPES, shape=())
        # int self-alias must-raise: true division always produces a float
        # result, so `a.__itruediv__(a)` for an int `a` must raise the same
        # UFuncTypeError the non-aliased int64/=int64 case in
        # inplace_cases.py raises -- aliasing doesn't change that rule.
        a_int = np.array([4, 8, 2], dtype=np.int64)
        cases.append(("self_alias_must_raise/int64_truediv", (a_int,), {}))
        return cases
    return build


def _build_bitwise(op: str):
    def build():
        return _self_alias_cases(_BITWISE_DTYPES)
    return build


def _build_floordiv():
    def build():
        cases = _self_alias_cases(_FLOORMOD_DTYPES)
        # division-by-zero self-alias: a //= a where a contains a zero.
        # numpy's own 0 // 0 (int -> 0 with a RuntimeWarning, not an
        # exception; float -> nan) behavior applies identically whether or
        # not the divisor happens to be the same object as the dividend --
        # this case exists to prove aliasing doesn't change that.
        for dtype in _FLOORMOD_DTYPES:
            cases.append((f"self_alias_divzero/{np.dtype(dtype).name}",
                          (_zero_containing(dtype),), {}))
        return cases
    return build


def _build_mod():
    def build():
        cases = _self_alias_cases(_FLOORMOD_DTYPES)
        for dtype in _FLOORMOD_DTYPES:
            cases.append((f"self_alias_divzero/{np.dtype(dtype).name}",
                          (_zero_containing(dtype),), {}))
        return cases
    return build


def _build_pow():
    def build():
        cases = _self_alias_cases(_POW_DTYPES)
        # self-alias must-raise: a**a where a is int and contains a
        # negative value -- numpy raises ValueError("Integers to negative
        # integer powers are not allowed.") because the exponent (== a
        # itself here) has a negative entry.
        a_neg = np.array([-2, 3, -1], dtype=np.int32)
        cases.append(("self_alias_must_raise/int32_negative_exponent", (a_neg,), {}))
        return cases
    return build


def _build_shift(op: str):
    def build():
        cases = _self_alias_cases(_SHIFT_DTYPES)
        # self-alias must-raise: a <<= a / a >>= a where a contains a
        # negative value -- numpy raises ValueError("negative shift count")
        # because the shift-amount operand (== a itself here) has a
        # negative entry. Signed dtypes only (unsigned _SHIFT_DTYPES
        # entries can't hold a negative literal).
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            a_neg = np.array([-3, 2, 5], dtype=dtype)
            cases.append((f"self_alias_must_raise/{np.dtype(dtype).name}_negative_shift",
                          (a_neg,), {}))
        return cases
    return build


def _build_matmul():
    def build():
        rng = _rng()
        cases = []
        for dtype in (np.float32, np.float64, np.int64):
            a = _fill(rng, (3, 3), dtype)
            cases.append((f"self_alias/square/{np.dtype(dtype).name}", (a,), {}))
        # self-alias must-raise: a @= a where a is non-square -- matmul's
        # own shape rule refuses this regardless of aliasing (a (2,3) can
        # never be its own matmul partner in-place: result would be (2,2),
        # which cannot be written into a's existing (2,3) buffer).
        a_nonsquare = _fill(rng, (2, 3), np.float64)
        cases.append(("self_alias_must_raise/nonsquare", (a_nonsquare,), {}))
        return cases
    return build


def _view_build(dtypes):
    def build():
        return _view_alias_cases(dtypes)
    return build


def _nonaliased_build(dtypes):
    def build():
        return _nonaliased_cases(dtypes)
    return build


# ---------------------------------------------------------------------------
# ItemSpecs
# ---------------------------------------------------------------------------

def _self_spec(op: str, build) -> ItemSpec:
    return ItemSpec(
        name=f"alias/self/{op}", kind="custom", atol=0.0, rtol=0.0,
        scalar_like=True,
        numpy_adapter=_self_alias_probe(op), ionp_adapter=_self_alias_probe(op),
        custom_cases=build,
    )


def _view_spec(op: str, dtypes) -> ItemSpec:
    return ItemSpec(
        name=f"alias/view/{op}", kind="custom", atol=0.0, rtol=0.0,
        scalar_like=True,
        numpy_adapter=_view_alias_probe(op), ionp_adapter=_view_alias_probe(op),
        custom_cases=_view_build(dtypes),
    )


def _nonaliased_spec(op: str, dtypes) -> ItemSpec:
    return ItemSpec(
        name=f"alias/nonaliased/{op}", kind="custom", atol=0.0, rtol=0.0,
        scalar_like=True,
        numpy_adapter=_nonaliased_probe(op), ionp_adapter=_nonaliased_probe(op),
        custom_cases=_nonaliased_build(dtypes),
    )


_OPS_WITH_VIEW_CASES = [
    ("__iadd__", _ARITH_DTYPES), ("__isub__", _ARITH_DTYPES), ("__imul__", _ARITH_DTYPES),
    ("__itruediv__", _TRUEDIV_DTYPES),
    ("__ifloordiv__", _FLOORMOD_DTYPES), ("__imod__", _FLOORMOD_DTYPES),
    ("__ipow__", _POW_DTYPES),
    ("__iand__", _BITWISE_DTYPES), ("__ior__", _BITWISE_DTYPES), ("__ixor__", _BITWISE_DTYPES),
    ("__ilshift__", _SHIFT_DTYPES), ("__irshift__", _SHIFT_DTYPES),
]

INPLACE_ALIAS_SPECS: dict[str, ItemSpec] = {
    "alias/self/__iadd__": _self_spec("__iadd__", _build_arith("__iadd__")),
    "alias/self/__isub__": _self_spec("__isub__", _build_arith("__isub__")),
    "alias/self/__imul__": _self_spec("__imul__", _build_arith("__imul__")),
    "alias/self/__itruediv__": _self_spec("__itruediv__", _build_truediv("__itruediv__")),
    "alias/self/__ifloordiv__": _self_spec("__ifloordiv__", _build_floordiv()),
    "alias/self/__imod__": _self_spec("__imod__", _build_mod()),
    "alias/self/__ipow__": _self_spec("__ipow__", _build_pow()),
    "alias/self/__iand__": _self_spec("__iand__", _build_bitwise("__iand__")),
    "alias/self/__ior__": _self_spec("__ior__", _build_bitwise("__ior__")),
    "alias/self/__ixor__": _self_spec("__ixor__", _build_bitwise("__ixor__")),
    "alias/self/__ilshift__": _self_spec("__ilshift__", _build_shift("__ilshift__")),
    "alias/self/__irshift__": _self_spec("__irshift__", _build_shift("__irshift__")),
    "alias/self/__imatmul__": _self_spec("__imatmul__", _build_matmul()),
}
for _op, _dtypes in _OPS_WITH_VIEW_CASES:
    INPLACE_ALIAS_SPECS[f"alias/view/{_op}"] = _view_spec(_op, _dtypes)

for _op, _dtypes in [
    ("__ifloordiv__", _FLOORMOD_DTYPES), ("__imod__", _FLOORMOD_DTYPES),
    ("__ipow__", _POW_DTYPES), ("__ilshift__", _SHIFT_DTYPES), ("__irshift__", _SHIFT_DTYPES),
]:
    INPLACE_ALIAS_SPECS[f"alias/nonaliased/{_op}"] = _nonaliased_spec(_op, _dtypes)

_alias_collisions = set(INPLACE_ALIAS_SPECS) & set(registry.REGISTRY)
if _alias_collisions:
    raise AssertionError(
        f"inplace_alias_cases.py: {sorted(_alias_collisions)} already "
        f"present in registry.REGISTRY -- refusing to silently overwrite an "
        f"existing item"
    )
registry.REGISTRY.update(INPLACE_ALIAS_SPECS)
