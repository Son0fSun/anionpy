"""Differential specs for ndarray pickle dunders.

Compares *values* after pickle.dumps/loads, not pickle byte identity.
anionpy reconstructs via anionpy._reconstruct_ndarray, not numpy._reconstruct.
"""
import pickle

import numpy as np

import anionpy

from registry import ItemSpec


def _roundtrip_np(arr):
    return pickle.loads(pickle.dumps(np.asarray(arr)))


def _roundtrip_ionp(arr):
    out = pickle.loads(pickle.dumps(arr))
    return np.asarray(out)


def _cases():
    return [
        ("vec_f64", (anionpy.array([1.0, 2.0, 3.0]),), {}),
        ("mat_i32", (anionpy.array([[1, 2], [3, 4]], dtype=anionpy.int32),), {}),
        ("empty", (anionpy.array([], dtype=anionpy.float64),), {}),
        ("bools", (anionpy.array([True, False, True]),), {}),
    ]


PICKLE_SPECS = {
    "ndarray.__reduce__": ItemSpec(
        name="ndarray.__reduce__",
        kind="custom",
        custom_cases=_cases,
        numpy_adapter=lambda a: _roundtrip_np(a),
        ionp_adapter=lambda a: _roundtrip_ionp(a),
        atol=0.0,
        rtol=0.0,
    ),
    "ndarray.__reduce_ex__": ItemSpec(
        name="ndarray.__reduce_ex__",
        kind="custom",
        custom_cases=_cases,
        numpy_adapter=lambda a: _roundtrip_np(a),
        ionp_adapter=lambda a: _roundtrip_ionp(a),
        atol=0.0,
        rtol=0.0,
    ),
    "ndarray.__getstate__": ItemSpec(
        name="ndarray.__getstate__",
        kind="custom",
        custom_cases=_cases,
        numpy_adapter=lambda a: _roundtrip_np(a),
        ionp_adapter=lambda a: _roundtrip_ionp(a),
        atol=0.0,
        rtol=0.0,
    ),
    "ndarray.__setstate__": ItemSpec(
        name="ndarray.__setstate__",
        kind="custom",
        custom_cases=_cases,
        numpy_adapter=lambda a: _roundtrip_np(a),
        ionp_adapter=lambda a: _roundtrip_ionp(a),
        atol=0.0,
        rtol=0.0,
    ),
}
