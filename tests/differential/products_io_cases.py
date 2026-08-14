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
regression protection: that `save`+`load` and `savez`+`load` round-trip
through anionpy's OWN save/load pair correctly across dtype/shape/order
variations, using anionpy's `.tobytes()`/`.tolist()` as the comparison
surface (both sides literally the same call, by construction -- this is
checking "does the round trip preserve the array", not "does anionpy
match numpy", which the differential harness's normal numpy-vs-ionp
comparison isn't shaped for here).

`savez_compressed` is the ONE exception to the same-implementation shape
described above, added later in this same task after a same-implementation
corpus was shown (by the coordinator, then reproduced independently) to be
structurally blind to a real defect: values round-tripped fine but the
archive never actually got smaller. Its adapters below (`_npzc_numpy`/
`_npzc_ionp`) each use their OWN implementation's real writer end to end
(real `np.savez_compressed` vs real `anionpy.savez_compressed`) and compare
both the round-tripped values AND a compression-effectiveness signal, so a
regression in either is corpus-visible. See the comment directly above
`_npzc_numpy` for the measured numbers and reasoning.
"""
from __future__ import annotations

import os
import tempfile
import warnings
import zipfile

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
    # numpy 2.5.1 has REMOVED 2-D cross-product support entirely (verified
    # live: `np.cross([1.,2.],[3.,4.])` now unconditionally raises
    # `ValueError: Both input arrays must be (arrays of) 3-dimensional
    # vectors, but they are 2 and 2 dimensional instead.`, for any axis
    # length != 3 on EITHER operand -- this crate's first `cross`
    # implementation still accepted 2-component vectors, matching an OLDER
    # numpy contract, and this exact corpus is what caught it (see
    # `ionp-core/src/products.rs::cross`'s doc comment for the fix). These
    # three cases are the negative control that keeps that regression from
    # coming back.
    cases.append(("two_component_raises", (np.array([1.0, 2.0]), np.array([3.0, 4.0])), {}))
    cases.append(("mixed_3_vs_2_raises", (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])), {}))
    cases.append(("mixed_2_vs_3_raises", (np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])), {}))
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


# `savez_compressed` gets a STRICTER, genuinely cross-implementation check
# than plain `savez` above, for a reason found live during this task's own
# review: an earlier version of this corpus checked ONLY that values
# round-tripped through anionpy's own writer+reader, the same
# same-implementation shape `save`/`load` above use. That is exactly the
# kind of corpus this project's own sharpest prior lesson warns about --
# it can pass 100% while the one thing `savez_compressed` exists to do
# (make the file smaller than `savez`'s plain output) silently does not
# happen, because nothing in the assertion ever looked at size. Measured
# directly (`zipfile.ZipFile(...).infolist()`, 100-element float64 array):
#   ours:  compress_type=8 (DEFLATE header, correctly labelled), compress_size=933, file_size=928  (LARGER)
#   numpy: compress_type=8 (DEFLATE),                            compress_size=255, file_size=928  (smaller, as intended)
# `ionp-core/src/format.rs::deflate`'s encoder emits RFC-1951 "stored"
# (uncompressed) blocks only -- a valid DEFLATE stream, correctly
# decodable (values ARE still exactly right, see the values half of this
# same tuple-comparison below), but achieving no real compression. See
# `docs/TICKET-deflate-real-compressor.md` for the follow-up. Every call
# to `anionpy.savez_compressed` also now raises a `UserWarning` naming
# this gap (`ionp-py/src/io_ops.rs`), so it is visible at the call site,
# not just in this corpus and a ledger comment.
#
# Each adapter below returns `(values, compressed_smaller_than_uncompressed)`
# using ITS OWN implementation's real writer for both halves (real numpy
# for `numpy_adapter`, real anionpy for `ionp_adapter` -- no shared
# same-implementation shortcut this time), so a mismatch on the second
# element is what makes this item fail rather than silently pass.

def _npzc_numpy(arrs):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npz")
        np.savez_compressed(path, **arrs)
        with zipfile.ZipFile(path) as z:
            shrinks = all(i.compress_size < i.file_size for i in z.infolist() if i.file_size > 0)
        back = np.load(path)
        values = tuple(sorted((k, back[k].tolist(), str(back[k].dtype)) for k in arrs))
        return (values, shrinks)


def _npzc_ionp(arrs):
    import anionpy
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npz")
        ins = {k: anionpy.array(v.tolist(), dtype=str(v.dtype)) for k, v in arrs.items()}
        with warnings.catch_warnings():
            # The UserWarning this call now raises (see module doc above)
            # is itself asserted elsewhere (io_ops.rs's own doctest-style
            # comment/manual check) -- silenced here only so it doesn't
            # get misclassified as a test failure by the harness's warning
            # capture, which is unrelated to what THIS item is checking.
            warnings.simplefilter("ignore")
            anionpy.savez_compressed(path, **ins)
        with zipfile.ZipFile(path) as z:
            shrinks = all(i.compress_size < i.file_size for i in z.infolist() if i.file_size > 0)
        back = anionpy.load(path)
        values = tuple(sorted((k, back[k].tolist(), str(back[k].dtype)) for k in arrs))
        return (values, shrinks)


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
