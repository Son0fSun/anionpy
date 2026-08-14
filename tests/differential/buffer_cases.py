"""Differential specs for ndarray.__buffer__ (PEP 688).

Compares a typed memoryview export against numpy's buffer: format-compatible
bytes and recovered ndarray values. Complex and empty N-D are out of this
corpus (anionpy raises so np.asarray can fall back to __array__).
"""
import numpy as np

import anionpy

from registry import ItemSpec


def _mv_np(arr):
    a = np.asarray(arr)
    mv = memoryview(a)
    return np.asarray(mv)


def _mv_ionp(arr):
    mv = memoryview(arr)
    return np.asarray(mv)


def _cases():
    return [
        ("vec_f64", (anionpy.array([1.0, 2.0, 3.0]),), {}),
        ("mat_i32", (anionpy.array([[1, 2], [3, 4]], dtype=anionpy.int32),), {}),
        ("bools", (anionpy.array([True, False, True]),), {}),
        ("scalar_f64", (anionpy.array(3.5),), {}),
        ("empty_1d", (anionpy.array([], dtype=anionpy.float64),), {}),
        ("uint8", (anionpy.array([1, 2, 3], dtype=anionpy.uint8),), {}),
    ]


BUFFER_SPECS = {
    "ndarray.__buffer__": ItemSpec(
        name="ndarray.__buffer__",
        kind="custom",
        custom_cases=_cases,
        numpy_adapter=_mv_np,
        ionp_adapter=_mv_ionp,
        atol=0.0,
        rtol=0.0,
    ),
}
