"""anionpy.ctypeslib -- the portable slice of the `numpy.ctypeslib.*` block.

Two of the six `ctypeslib.*` surface items are implemented here;
`anionpy/_state/misc_namespaces.py` documents in detail why the other four
(`as_array`, `as_ctypes`, `as_ctypes_type`, `ndpointer`) are NOT here.

- `load_library` (ported verbatim from `numpy.ctypeslib.load_library`,
  `numpy/ctypeslib/_ctypeslib.py`) never touches an ndarray at all -- it is
  pure `os.path`/`ctypes.cdll` lookup logic, finding and loading a shared
  library by name. It has zero dependency on anionpy's array internals, which
  is exactly why it is safe to port unmodified.
- `c_intp` is numpy's ctypes type alias for a platform's pointer-sized
  signed integer (numpy computes this from its own `dtype('n')` char via
  `numpy._core._internal._getintp_ctype`; anionpy has no equivalent internal
  dtype-char table to consult, so this reimplements the SAME platform rule
  numpy's own function encodes -- "the C integer type whose size matches
  `ctypes.c_void_p`'s" -- directly from `ctypes.sizeof`, with zero
  numpy dependency and zero array arithmetic). Verified to match numpy's
  own `np.ctypeslib.c_intp` byte-for-byte on this platform (both resolve
  to `ctypes.c_long`, 8 bytes, matching `sizeof(void*)`).
"""
from __future__ import annotations

import ctypes
import os

__all__ = ["load_library", "c_intp"]


def load_library(libname, loader_path):
    """Load a shared library, cross-platform. Ported verbatim from
    `numpy.ctypeslib.load_library`.
    """
    # Convert path-like objects into strings
    libname = os.fsdecode(libname)
    loader_path = os.fsdecode(loader_path)

    ext = os.path.splitext(libname)[1]
    if not ext:
        import sys
        import sysconfig
        # Try to load library with platform-specific name, otherwise
        # default to libname.[so|dll|dylib].  Sometimes, these files are
        # built erroneously on non-linux platforms.
        base_ext = ".so"
        if sys.platform.startswith("darwin"):
            base_ext = ".dylib"
        elif sys.platform.startswith("win"):
            base_ext = ".dll"
        libname_ext = [libname + base_ext]
        so_ext = sysconfig.get_config_var("EXT_SUFFIX")
        if not so_ext == base_ext:
            libname_ext.insert(0, libname + so_ext)
    else:
        libname_ext = [libname]

    loader_path = os.path.abspath(loader_path)
    if not os.path.isdir(loader_path):
        libdir = os.path.dirname(loader_path)
    else:
        libdir = loader_path

    for ln in libname_ext:
        libpath = os.path.join(libdir, ln)
        if os.path.exists(libpath):
            try:
                return ctypes.cdll[libpath]
            except OSError:
                # defective lib file
                raise
    # if no successful return in the libname_ext loop:
    raise OSError("no file with expected extension")


def _compute_c_intp():
    """The ctypes integer type matching this platform's pointer width --
    same rule numpy._core._internal._getintp_ctype applies (match
    `dtype('n')`'s char), expressed directly against `ctypes.sizeof`
    instead of consulting a numpy-internal dtype-char table anionpy has no
    equivalent of.
    """
    void_p_size = ctypes.sizeof(ctypes.c_void_p)
    if ctypes.sizeof(ctypes.c_int) == void_p_size:
        return ctypes.c_int
    if ctypes.sizeof(ctypes.c_long) == void_p_size:
        return ctypes.c_long
    if ctypes.sizeof(ctypes.c_longlong) == void_p_size:
        return ctypes.c_longlong
    return ctypes.c_long


c_intp = _compute_c_intp()
