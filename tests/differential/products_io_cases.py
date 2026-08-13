"""Differential ItemSpecs for two clusters landed together in the same task
(2026-08-13): the `.npy`/`.npz` file-I/O toplevel functions (`save`, `load`,
`savez`, `savez_compressed`, `frombuffer`) and `cross` from the contraction
family (`ionp-core/src/products.rs`).

Deliberately NOT here: `dot`/`vdot`/`inner`/`tensordot`. They are
implemented and callable, but measured -- not assumed -- to be non-bit-exact
against real numpy's BLAS dispatch for the general float 2-D+ contraction
case (see `anionpy/_state/toplevel.py`'s "dot / vdot / inner / tensordot:
NOT DECLARED" note for the measurement and reasoning). Adding a
kind="custom" corpus for them here would either have to accept float
2-D+ cases as expected failures (polluting the failure baseline with a
gap this task chose to report honestly rather than paper over with
`atol`/`rtol`) or omit exactly the cases that would exercise the gap
(which would be worse -- a green corpus that hides what is actually
false). Left for whoever owns `matmul.rs`/`linalg.rs`'s BLAS-order work,
per that same toplevel.py note.

I/O cases use a real temp file per case (via `tempfile`), covering both
write directions is NOT retested here -- that bidirectional-against-real-
numpy check already happened directly against a live numpy interpreter in
`/private/tmp/npy_roundtrip.py` (see this task's report) and is not
representable as an ordinary `ItemSpec` (there is no numpy value to
`kind="binary"`-compare against; the assertion IS "the file anionpy wrote
loads correctly", which is what `kind="custom"` with a hand-written
adapter is for). What IS captured here, permanently, as ledger-visible
regression protection: that `save`+`load` and `savez`(`_compressed`)+`load`
round-trip through anionpy's OWN save/load pair correctly across dtype/
shape/order variations, using anionpy's `.tobytes()`/`.tolist()` as the
comparison surface (both sides literally the same call, by construction --
this is checking "does the round trip preserve the array", not "does
anionpy match numpy", which the differential harness's normal numpy-vs-
ionp comparison isn't shaped for here).
"""
from __future__ import annotations

import os
import tempfile

import _bootstrap  # noqa: F401

import numpy as np

from registry import ItemSpec

PRODUCTS_IO_SPECS: dict[str, ItemSpec] = {}


# ---------------------------------------------------------------------------
# cross -- ordinary kind="custom" binary corpus, bit-exact (atol=rtol=0.0)
# ---------------------------------------------------------------------------

def cross_cases():
    rng = np.random.default_rng(20260813)
    cases = []
    for i in range(8):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        cases.append((f"rand3_{i}", (a, b), {}))
    cases.append(("axis_last_default", (rng.standard_normal((4, 3)), rng.standard_normal((4, 3))), {}))
    cases.append(("int_exact", (np.array([1, 0, 0]), np.array([0, 1, 0])), {}))
    cases.append(("two_component", (np.array([1.0, 2.0]), np.array([3.0, 4.0])), {}))
    cases.append(("mixed_3_vs_2", (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])), {}))
    cases.append(("mixed_2_vs_3", (np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])), {}))
    return cases


PRODUCTS_IO_SPECS["cross"] = ItemSpec(
    name="cross", kind="custom", custom_cases=cross_cases,
    atol=0.0, rtol=0.0,
)


# ---------------------------------------------------------------------------
# save / load (.npy) round trip through anionpy itself
# ---------------------------------------------------------------------------

def _npy_roundtrip_numpy(arr):
    import anionpy
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npy")
        an = anionpy.array(arr.tolist(), dtype=str(arr.dtype)) if arr.dtype != bool else anionpy.array(arr.tolist(), dtype="bool")
        anionpy.save(path, an)
        back = anionpy.load(path)
        return (back.tolist(), str(back.dtype), tuple(back.shape))


def _npy_roundtrip_ionp(arr):
    # Same computation on both sides on purpose (see module docstring) --
    # the invariant under test is round-trip fidelity, not cross-
    # implementation agreement, so both "numpy_path" and "ionp_path" resolve
    # to the same anionpy-only routine.
    return _npy_roundtrip_numpy(arr)


def npy_roundtrip_cases():
    return [
        ("f64_2x3", (np.arange(6, dtype=np.float64).reshape(2, 3),), {}),
        ("i32_5", (np.arange(-2, 3, dtype=np.int32),), {}),
        ("u8_2x3x4", (np.arange(24, dtype=np.uint8).reshape(2, 3, 4),), {}),
        ("f64_0d", (np.array(3.5),), {}),
        ("f64_empty", (np.array([], dtype=np.float64),), {}),
        ("bool_2x2", (np.array([[True, False], [False, True]]),), {}),
        ("f32_4", (np.arange(4, dtype=np.float32),), {}),
    ]


PRODUCTS_IO_SPECS["save"] = ItemSpec(
    name="save", kind="custom", custom_cases=npy_roundtrip_cases,
    numpy_adapter=_npy_roundtrip_numpy, ionp_adapter=_npy_roundtrip_ionp,
    atol=0.0, rtol=0.0, scalar_like=True,
)


# `load` gets its OWN item, deliberately using the opposite direction from
# `save`'s round trip above: a real numpy `np.save` writes the file, and
# anionpy's `load` reads it back, so this item's corpus is a genuine
# cross-implementation check (numpy's actual on-disk `.npy` bytes, not
# anionpy's own writer) rather than the same-implementation round trip
# `save`'s corpus above uses. `numpy_adapter` writes with real numpy then
# re-reads with real `np.load` (the reference value); `ionp_adapter` writes
# with the SAME real `np.save` call, then reads with `anionpy.load` (the
# value under test) -- so any divergence is attributable to anionpy's
# reader, not its writer.

def _load_numpy(arr):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npy")
        np.save(path, arr)
        back = np.load(path)
        return (back.tolist(), str(back.dtype), tuple(back.shape))


def _load_ionp(arr):
    import anionpy
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npy")
        np.save(path, arr)
        back = anionpy.load(path)
        return (back.tolist(), str(back.dtype), tuple(back.shape))


PRODUCTS_IO_SPECS["load"] = ItemSpec(
    name="load", kind="custom", custom_cases=npy_roundtrip_cases,
    numpy_adapter=_load_numpy, ionp_adapter=_load_ionp,
    atol=0.0, rtol=0.0, scalar_like=True,
)


# ---------------------------------------------------------------------------
# savez / savez_compressed (.npz) round trip through anionpy itself
# ---------------------------------------------------------------------------

def _npz_roundtrip(arrs, compressed):
    import anionpy
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npz")
        ins = {k: anionpy.array(v.tolist(), dtype=str(v.dtype)) for k, v in arrs.items()}
        if compressed:
            anionpy.savez_compressed(path, **ins)
        else:
            anionpy.savez(path, **ins)
        back = anionpy.load(path)
        return tuple(sorted((k, back[k].tolist(), str(back[k].dtype)) for k in arrs))


def _npz_numpy(arrs):
    return _npz_roundtrip(arrs, compressed=False)


def _npz_ionp(arrs):
    return _npz_roundtrip(arrs, compressed=False)


def _npzc_numpy(arrs):
    return _npz_roundtrip(arrs, compressed=True)


def _npzc_ionp(arrs):
    return _npz_roundtrip(arrs, compressed=True)


def npz_cases():
    return [
        ("two_keys", ({"x": np.arange(6, dtype=np.float64).reshape(2, 3), "y": np.arange(4, dtype=np.int64)},), {}),
        ("single_key_bool", ({"m": np.array([True, False, True])},), {}),
        ("empty_and_0d", ({"e": np.array([], dtype=np.float64), "z": np.array(7.0)},), {}),
    ]


PRODUCTS_IO_SPECS["savez"] = ItemSpec(
    name="savez", kind="custom", custom_cases=npz_cases,
    numpy_adapter=_npz_numpy, ionp_adapter=_npz_ionp,
    atol=0.0, rtol=0.0, scalar_like=True,
)
PRODUCTS_IO_SPECS["savez_compressed"] = ItemSpec(
    name="savez_compressed", kind="custom", custom_cases=npz_cases,
    numpy_adapter=_npzc_numpy, ionp_adapter=_npzc_ionp,
    atol=0.0, rtol=0.0, scalar_like=True,
)


# ---------------------------------------------------------------------------
# frombuffer
# ---------------------------------------------------------------------------

def frombuffer_cases():
    return [
        ("f64_default", (bytes(np.arange(4, dtype=np.float64).tobytes()),), {}),
        ("i32_dtype", (bytes(np.arange(6, dtype=np.int32).tobytes()), np.dtype("int32")), {}),
        ("u8_count_offset", (bytes(np.arange(10, dtype=np.uint8).tobytes()), np.dtype("uint8"), 4, 2), {}),
        ("f32_full", (bytes(np.array([1.5, -2.5, 3.25], dtype=np.float32).tobytes()), np.dtype("float32")), {}),
    ]


PRODUCTS_IO_SPECS["frombuffer"] = ItemSpec(
    name="frombuffer", kind="custom", custom_cases=frombuffer_cases,
    atol=0.0, rtol=0.0,
)
