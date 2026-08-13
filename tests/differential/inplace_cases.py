"""Custom differential test cases for the seven `ndarray.__i*__` in-place
dunders (`__iadd__ __isub__ __imul__ __itruediv__ __iand__ __ior__
__ixor__`) -- see GOAL-ionp.md / the 2026-08-01 "declare the in-place
operators" task brief.

Why these need their OWN case builder instead of reusing kind="binary_op"
(the machinery `ndarray.__add__` etc. already use, see registry.py's module
docstring): a value-only comparison of the final array is not the thing
that distinguishes an in-place operator from its binary sibling. `a.__add__
= lambda a, b: a + b` (returning a brand-new array instead of mutating `a`)
would pass every value check `__add__`'s own corpus runs, and so would
`ndarray.__iadd__` if it silently returned a *new* object with the right
numbers in it. What actually makes an operator "in-place" -- and therefore
what a test claiming to cover it must check -- is:

  1. object identity is preserved (`id(a)` unchanged across the call, and
     the call's return value on the numpy side is `a` itself -- Python's
     augmented-assignment protocol rebinds the name to whatever
     `__iadd__` returns, so a correct in-place implementation must return
     `self`, not a new object, or `a += b` would silently swap `a` for a
     copy).
  2. the SAME buffer is mutated: a view taken of `a` *before* the in-place
     call must observe the mutation afterward (numpy views alias memory;
     an implementation that computes a new result and reassigns it to a
     Python-level attribute, rather than writing into the existing
     buffer, breaks this even while (1) still holds -- these are
     independent properties, see `_probe`'s docstring below and the
     'ndarray.__iadd__: FOUND A REAL BUG' report this module's grading
     surfaces).
  3. dtype is not silently promoted: `int_array += float_array` (or,
     for `__itruediv__`, even `int_array /= int_array` -- true division
     ALWAYS produces a float result in numpy, so it always needs an
     output cast back to the int self, which numpy's 'same_kind' casting
     rule always refuses) must raise the exact exception real numpy
     raises, not silently upcast the array's dtype. This is the single
     most common in-place bug and a value-only test misses it completely
     -- it only ever notices if it happens to check `.dtype` after a
     successful call, and a same-dtype-only corpus (the common case for a
     lazily-written test) never even exercises the code path.
  4. broadcasting that would change `a`'s shape must raise, matching
     numpy -- an in-place op can only ever write into `a`'s existing
     buffer, so a result shape larger than `a.shape` is a contradiction
     numpy detects and refuses, not a case where numpy silently grows the
     array.
  5. in-place on a non-contiguous/negative-stride view of `a` must still
     write the mathematically correct values through `a`'s existing
     strides.

The single `_probe(op)` adapter below (used as BOTH `numpy_adapter` and
`ionp_adapter` on every ItemSpec this module builds -- registry.py's
`ItemSpec.numpy_adapter`/`ionp_adapter` mechanism, the same one
linalg_cases.py uses for invariant-based probes) exercises all five in one
pass per case and returns a single flat tuple of PLAIN PYTHON VALUES (never
raw ndarrays -- see `_probe`'s docstring for why): values-as-a-list,
shape, dtype name, identity-preserved, return-is-self, and
buffer-aliasing-observed. harness.compare_values' `scalar_like=True` path
then does ordinary Python tuple `==`, which is exact for every element
(list-of-python-scalars from `.tolist()`, a str, three bools) -- no
allclose/ULP slack of any kind, matching this task's "declare bit-exact"
requirement. Cases built to deliberately trigger numpy's own dtype-refusal
or broadcast-refusal never reach that return at all: the exception raised
by `getattr(a, op)(b)` propagates straight out of the adapter, and
harness.run_case's PRE-EXISTING exception-type-matching logic (unchanged by
this module) is what actually grades those cases -- there is exactly one
place in this codebase that decides "did numpy and anionpy raise the same
thing", and it is not duplicated here.

This module supplies cases only; REGISTRY is not owned by this file (it is
merged in by registry.py, at the bottom, the same way registry.py already
merges linalg_cases.py's LINALG_SPECS and ufunc_registry.py's UFUNC_SPECS --
see registry.py's collision-checked `REGISTRY.update(...)` call).
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)

from registry import ItemSpec

# Independent of corpus.py's SEED (20260731) -- this module builds its own
# small, purpose-built corpus rather than reusing corpus.py's general-purpose
# one (that corpus has no notion of "a dtype pair numpy is guaranteed to
# refuse", which is the entire point of several cases here), but is pinned
# the same deliberate way: one fixed seed, reproducible across machines and
# runs.
INPLACE_SEED = 20260801


def _rng() -> np.random.Generator:
    return np.random.default_rng(INPLACE_SEED)


def _fill(rng: np.random.Generator, shape: tuple[int, ...], dtype) -> np.ndarray:
    """Same-spirit fill as corpus.py's `_fill` (not imported from there --
    that one is tied to corpus.py's own SEED/RNG stream, and reusing it here
    would make this module's cases depend on draw-order coupling with an
    unrelated corpus; a small independent copy is clearer than a shared
    helper for five call sites)."""
    dt = np.dtype(dtype)
    n = int(np.prod(shape)) if shape else 1
    if dt.kind == "b":
        data = rng.integers(0, 2, size=n).astype(bool)
    elif dt.kind in "iu":
        info = np.iinfo(dt)
        lo = max(info.min, -1000)
        hi = min(info.max, 1000)
        data = rng.integers(lo, hi, size=n, dtype=np.int64).astype(dt)
    elif dt.kind == "f":
        data = (rng.standard_normal(size=n) * 5).astype(dt)
    elif dt.kind == "c":
        real = rng.standard_normal(size=n)
        imag = rng.standard_normal(size=n)
        data = (real + 1j * imag).astype(dt)
    else:  # pragma: no cover - not reachable with our dtype catalog
        raise ValueError(f"unhandled dtype kind {dt.kind!r}")
    return data.reshape(shape)


def _nonzero_fill(rng: np.random.Generator, shape: tuple[int, ...], dtype) -> np.ndarray:
    """Like `_fill`, but guaranteed to contain no zero/zero-magnitude
    elements -- for `__itruediv__` divisor operands, where a seeded zero
    would make the case's numpy-side result NaN/inf for reasons unrelated to
    anything this module is testing (dtype promotion / identity / aliasing),
    and could turn a would-be exact-value case into a spurious NaN mismatch
    depending on draw. Kept separate from `_fill` rather than adding a
    `nonzero=` flag to it, since only the divisor cases need this."""
    dt = np.dtype(dtype)
    if dt.kind in "iu":
        arr = _fill(rng, shape, dtype)
        arr = np.where(arr == 0, np.array(1, dtype=dt), arr)
        return arr
    if dt.kind == "f":
        arr = _fill(rng, shape, dtype)
        return np.where(arr == 0, np.array(1.0, dtype=dt), arr).astype(dt)
    if dt.kind == "c":
        arr = _fill(rng, shape, dtype)
        zero_mask = arr == 0
        if np.any(zero_mask):
            arr = np.where(zero_mask, np.array(1 + 1j, dtype=dt), arr).astype(dt)
        return arr
    raise ValueError(f"_nonzero_fill: unsupported dtype kind {dt.kind!r}")


# ---------------------------------------------------------------------------
# The probe: one adapter, used as both the numpy-side and ionp-side callable
# for every item this module declares. See module docstring for why a
# single flat tuple-of-plain-values return type is what lets ONE adapter
# uniformly cover the value/identity/aliasing/dtype/broadcast/stride
# properties across every case, and why the exception-raising cases don't
# need any special handling here at all.
# ---------------------------------------------------------------------------

def _probe(op: str):
    """Build the (identical) adapter used for both `numpy_adapter` and
    `ionp_adapter` of the ItemSpec for dunder `op`. `a`/`b` are whatever
    `run_case` hands the adapter for this call -- a real `numpy.ndarray`
    pair on the numpy side (via `ItemSpec.numpy_adapter`, called directly
    with the freshened case args -- see harness.py's `_freshen`), a real
    `anionpy.ndarray` pair on the anionpy side (via `ItemSpec._wrap_custom_conversion`,
    which converts every numpy.ndarray argument before calling
    `ionp_adapter` -- registry.py's existing, unmodified mechanism). Nothing
    in this function's body cares which one it got: both expose `.ndim`,
    `.shape`, `__getitem__`, `getattr(a, op)`, and are accepted by
    `np.asarray()` (numpy natively; anionpy via its `__array__` interop seam),
    so this is genuinely the same code path measuring the same properties on
    both sides, not two parallel hand-written checks that could quietly
    drift apart.

    Returns a flat tuple of PLAIN PYTHON VALUES, never a raw ndarray:
    `harness.compare_multi_output`/`compare_values` has no path for a tuple
    that MIXES an array slot (needing dtype/shape/ULP-aware comparison) with
    plain-scalar slots (needing exact `==`) in one item -- `scalar_like`
    applies to the whole comparison, not per-slot. Converting the array to
    `.tolist()` + `str(dtype)` up front sidesteps that entirely: every
    element of the returned tuple is a `str`/`bool`/`tuple`, so plain Python
    `==` (which is what `harness._compare_scalar_like` uses under
    `scalar_like=True`) is already exact -- `.tolist()` is loss-free for
    every dtype this module's corpus uses (bool/int/uint/float/complex all
    round-trip through Python's own bool/int/float/complex exactly; no
    numpy-internal precision is lost the way it could be for, say, a
    truncating cast).
    """
    def adapter(a, b):
        can_check_alias = a.ndim >= 1 and a.shape[0] >= 4
        view_before = a[1:4] if can_check_alias else None

        before_id = id(a)
        ret = getattr(a, op)(b)

        id_preserved = (id(a) == before_id)
        ret_is_self = (ret is a)

        alias_observed = None
        if view_before is not None:
            alias_observed = bool(np.array_equal(
                np.asarray(view_before), np.asarray(a)[1:4], equal_nan=True,
            ))

        arr = np.asarray(a)
        return (
            tuple(arr.ravel().tolist()),
            arr.shape,
            str(arr.dtype),
            id_preserved,
            ret_is_self,
            alias_observed,
        )
    return adapter


# ---------------------------------------------------------------------------
# Case builders. Each returns list[(label, (a, b), {})] -- `a`/`b` are real
# numpy.ndarray, freshened independently per side by harness.run_case before
# either adapter sees them (see the 2026-08-01 argument-reuse-contract fix
# this task's harness.py change makes; this module is exactly the intended
# beneficiary -- every case below hands the SAME `a` object to a mutating
# call, and the whole point is that the numpy call's mutation of ITS copy
# must never be visible to the anionpy call's copy, or vice versa).
# ---------------------------------------------------------------------------

def _value_cases(same_dtypes: list, broadcast_dtype, allow_truediv_int: bool) -> list:
    """Ordinary "does the op compute the right numbers, preserve identity,
    and preserve aliasing" cases: same dtype on both operands (the case
    where numpy itself never raises), a broadcastable-but-not-equal-shape
    pair, a non-contiguous (strided, non-unit-stride) view as `a`, and a
    negative-stride view as `a` -- see property (5) in the module
    docstring. `allow_truediv_int=False` (only `__itruediv__` sets this)
    skips integer dtypes entirely: true division of two integer arrays
    ALWAYS produces a float result in real numpy, which ALWAYS fails the
    output-cast back to an integer self under numpy's own 'same_kind' rule
    -- that is not a same-dtype "does it compute the right answer" case at
    all, it is a guaranteed-raise case, covered separately by
    `_dtype_promotion_cases`.
    """
    rng = _rng()
    cases = []
    # allow_truediv_int is True for every op except __itruediv__, and for
    # those ops a zero operand is perfectly fine (add/sub/mul/bitwise all
    # handle zero uneventfully) -- only __itruediv__'s divisor needs the
    # nonzero guarantee, so pick the filler once, up front, rather than
    # re-deriving it at every call site.
    fill_b = _fill if allow_truediv_int else _nonzero_fill

    for dtype in same_dtypes:
        dt = np.dtype(dtype)
        if dt.kind in "iu" and not allow_truediv_int:
            continue
        a = _fill(rng, (6,), dtype)
        b = fill_b(rng, (6,), dtype)
        cases.append((f"value/same_dtype/{dt.name}", (a, b), {}))

    # broadcastable-but-not-equal-shape: b broadcasts onto a's shape, a's
    # own shape is unchanged by the result (the case an in-place op CAN
    # satisfy, as opposed to the shape-grows case in
    # `_broadcast_mismatch_cases`, which it cannot). `broadcast_dtype` is
    # always float64 (arithmetic/truediv callers) or int32 (bitwise
    # callers) at every call site below -- never an integer dtype paired
    # with allow_truediv_int=False -- so no skip is needed here.
    a2 = _fill(rng, (3, 4), broadcast_dtype)
    b2 = fill_b(rng, (4,), broadcast_dtype)
    cases.append((f"value/broadcast_compatible/{np.dtype(broadcast_dtype).name}", (a2, b2), {}))

    # non-contiguous (stride-2) view as `a` itself: property (5).
    base = np.arange(20, dtype=broadcast_dtype).reshape(4, 5)
    col_view = base[:, 1::2]
    b_noncontig = fill_b(rng, col_view.shape, broadcast_dtype)
    cases.append((f"value/noncontig_view_a/{np.dtype(broadcast_dtype).name}", (col_view, b_noncontig), {}))

    # negative-stride 1-D view as `a`.
    rev = np.arange(10, dtype=broadcast_dtype)[::-1]
    b_rev = fill_b(rng, rev.shape, broadcast_dtype)
    cases.append((f"value/negstride_view_a/{np.dtype(broadcast_dtype).name}", (rev, b_rev), {}))

    return cases


def _dtype_promotion_cases(pairs: list) -> list:
    """`(label, a, b)` triples where real numpy is VERIFIED (by direct
    measurement against numpy 2.5.1, see this task's report) to raise --
    property (3). Each entry in `pairs` is `(label, a_dtype, b_dtype,
    a_vals, b_vals)`; kept as literal small arrays (not the seeded filler)
    because what matters here is the DTYPE pairing, not the values -- using
    tiny, readable literals also means a human reading a failure diagnostic
    doesn't have to cross-reference the RNG stream to see what was actually
    compared."""
    cases = []
    for label, a_dtype, b_dtype, a_vals, b_vals in pairs:
        a = np.array(a_vals, dtype=a_dtype)
        b = np.array(b_vals, dtype=b_dtype)
        cases.append((f"dtype_promotion_must_raise/{label}", (a, b), {}))
    return cases


def _broadcast_mismatch_cases(dtype) -> list:
    """Shapes where the broadcast RESULT would differ from `a`'s own shape
    -- property (4). An in-place op can only ever write into `a`'s existing
    buffer, so numpy always refuses these; `a`'s shape is never grown."""
    dt = np.dtype(dtype)
    a1 = _fill(_rng(), (3,), dtype)
    b1 = _fill(_rng(), (3, 4), dtype)
    a2 = _fill(_rng(), (3, 4), dtype)
    b2 = _fill(_rng(), (2, 5), dtype)
    return [
        (f"broadcast_mismatch_must_raise/shape_grows/{dt.name}", (a1, b1), {}),
        (f"broadcast_mismatch_must_raise/incompatible/{dt.name}", (a2, b2), {}),
    ]


# ---------------------------------------------------------------------------
# Per-op case lists.
#
# Arithmetic (`__iadd__ __isub__ __imul__`): same-dtype corpus spans
# int32/int64/float32/float64/complex64/complex128; dtype-promotion-must-
# raise corpus pairs int64<-float64 and int32<-complex128 (both VERIFIED to
# raise numpy.exceptions.UFuncTypeError, see module docstring / task
# report).
#
# `__itruediv__` is the same shape MINUS integer self dtypes in the
# same-dtype corpus (see `_value_cases`'s `allow_truediv_int` docstring),
# PLUS its own dedicated must-raise case: int64 /= int64, which raises even
# though both operands share a dtype -- true division's result is always
# float, so the output-cast-back-to-int64 is what numpy refuses, not a
# cross-dtype mismatch.
#
# Bitwise (`__iand__ __ior__ __ixor__`): same-dtype corpus spans
# bool/int8/int16/int32/uint8/uint32; dtype-promotion-must-raise corpus
# pairs uint8<-int8 and bool<-int32 (both VERIFIED to raise
# UFuncTypeError -- note int8<-int32, the mirror-image narrowing, does NOT
# raise in real numpy: same-signedness integer narrowing is permitted under
# 'same_kind' casting, which is exactly why `uint8<-int8` -- a SIGNEDNESS
# mismatch, not just a width one -- was chosen instead).
# ---------------------------------------------------------------------------

_ARITH_SAME_DTYPES = [np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
_TRUEDIV_SAME_DTYPES = [np.float32, np.float64, np.complex64, np.complex128]
_BITWISE_SAME_DTYPES = [np.bool_, np.int8, np.int16, np.int32, np.uint8, np.uint32]

_ARITH_DTYPE_PROMOTION_PAIRS = [
    ("int64_lhs_float64_rhs", np.int64, np.float64, [1, 2, 3], [1.5, 2.5, 3.5]),
    ("int32_lhs_complex128_rhs", np.int32, np.complex128, [1, 2, 3], [1 + 1j, 2 + 2j, 3 + 3j]),
]
_TRUEDIV_DTYPE_PROMOTION_PAIRS = [
    ("int64_lhs_int64_rhs_truediv_always_floats", np.int64, np.int64, [4, 8, 2], [2, 4, 1]),
    ("int32_lhs_float64_rhs", np.int32, np.float64, [4, 8, 2], [2.0, 4.0, 1.0]),
]
_BITWISE_DTYPE_PROMOTION_PAIRS = [
    ("uint8_lhs_int8_rhs_signedness_mismatch", np.uint8, np.int8, [1, 2, 3], [1, 2, 3]),
    ("bool_lhs_int32_rhs", np.bool_, np.int32, [True, False, True], [1, 2, 3]),
]


def _arith_custom_cases(op: str):
    def build():
        cases = _value_cases(_ARITH_SAME_DTYPES, np.float64, allow_truediv_int=True)
        cases += _dtype_promotion_cases(_ARITH_DTYPE_PROMOTION_PAIRS)
        cases += _broadcast_mismatch_cases(np.float64)
        return cases
    return build


def _truediv_custom_cases():
    def build():
        cases = _value_cases(_TRUEDIV_SAME_DTYPES, np.float64, allow_truediv_int=False)
        cases += _dtype_promotion_cases(_TRUEDIV_DTYPE_PROMOTION_PAIRS)
        cases += _broadcast_mismatch_cases(np.float64)
        return cases
    return build


def _bitwise_custom_cases(op: str):
    def build():
        cases = _value_cases(_BITWISE_SAME_DTYPES, np.int32, allow_truediv_int=True)
        cases += _dtype_promotion_cases(_BITWISE_DTYPE_PROMOTION_PAIRS)
        cases += _broadcast_mismatch_cases(np.int32)
        return cases
    return build


def _make_spec(name: str, op: str, custom_cases) -> ItemSpec:
    return ItemSpec(
        name=name,
        kind="custom",
        atol=0.0,
        rtol=0.0,
        scalar_like=True,
        numpy_adapter=_probe(op),
        ionp_adapter=_probe(op),
        custom_cases=custom_cases,
    )


INPLACE_SPECS: dict[str, ItemSpec] = {
    "ndarray.__iadd__": _make_spec("ndarray.__iadd__", "__iadd__", _arith_custom_cases("__iadd__")),
    "ndarray.__isub__": _make_spec("ndarray.__isub__", "__isub__", _arith_custom_cases("__isub__")),
    "ndarray.__imul__": _make_spec("ndarray.__imul__", "__imul__", _arith_custom_cases("__imul__")),
    "ndarray.__itruediv__": _make_spec("ndarray.__itruediv__", "__itruediv__", _truediv_custom_cases()),
    "ndarray.__iand__": _make_spec("ndarray.__iand__", "__iand__", _bitwise_custom_cases("__iand__")),
    "ndarray.__ior__": _make_spec("ndarray.__ior__", "__ior__", _bitwise_custom_cases("__ior__")),
    "ndarray.__ixor__": _make_spec("ndarray.__ixor__", "__ixor__", _bitwise_custom_cases("__ixor__")),
}
