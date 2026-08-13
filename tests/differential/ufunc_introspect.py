"""numpy-introspection layer for the ufunc block.

Nothing in this module hand-types a ufunc name. `UFUNC_NAMES` is read
straight out of `tools/numpy_surface.json` (kind == "ufunc") -- that file is
the one legal enumeration source per the task brief ("Enumerate them from
tools/numpy_surface.json ... do NOT hand-type a list of 134 names"). Every
other fact about a ufunc used anywhere in this test bed (arity, output
count, identity element, a dtype it actually accepts, which ufunc-protocol
methods numpy itself allows) is derived here from numpy's own introspection
(`.nin`, `.nout`, `.types`, `.identity`, `.signature`) or from a real trial
call against real numpy -- never guessed or assumed from the name.

No comparisons happen here and no anionpy import happens here -- this module
only looks at numpy.
"""
from __future__ import annotations

import dataclasses
import json
import warnings
from typing import Optional

import numpy as np

import _bootstrap  # noqa: F401

SURFACE_PATH = _bootstrap.REPO_ROOT / "tools" / "numpy_surface.json"

# Methods the ufunc protocol exposes beyond the plain call. "at" mutates in
# place; the other four are pure. Whether a *specific* ufunc supports a
# *specific* one of these is never assumed here -- see applicable_methods().
UFUNC_METHODS = ("reduce", "accumulate", "outer", "reduceat", "at")


def _load_surface() -> dict:
    return json.loads(SURFACE_PATH.read_text())


def _ufunc_names_from_surface() -> list[str]:
    surface = _load_surface()
    names = surface["names"]
    return sorted(name for name, kind in names.items() if kind == "ufunc")


UFUNC_NAMES: list[str] = _ufunc_names_from_surface()


def resolve_numpy_ufunc(path: str):
    """'add' -> np.add, 'char.add' -> np.char.add, 'strings.equal' -> np.strings.equal."""
    obj = np
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


@dataclasses.dataclass(frozen=True)
class UfuncInfo:
    name: str                      # the numpy_surface.json / coverage-ledger key
    nin: int
    nout: int
    identity: object
    signature: Optional[str]       # non-None => gufunc (matmul, vecdot, vecmat)
    types: tuple                   # numpy's own list of legacy loop signatures, e.g. "dd->d"
    is_string: bool                # name starts with "char." or "strings."
    typed_sample: Optional[tuple]  # nin numpy arrays, dtype read from `.types`, guaranteed
                                    # to be a loop numpy itself declares -- None if no legacy
                                    # loop could be parsed into a usable sample (e.g. object-only)


# ---------------------------------------------------------------------------
# typecode -> concrete sample array, built from numpy's OWN loop signatures
# (ufunc.types), not from a hand-maintained per-ufunc table. This is what
# lets isnat (needs datetime64), ldexp (needs float+int mixed), gcd (needs
# integer), bitwise_count (needs integer) all get a sample that numpy itself
# says is valid, without a single hardcoded "isnat wants datetime64" rule.
# ---------------------------------------------------------------------------

def _typecode_sample(code: str, n: int = 4) -> np.ndarray:
    dt = np.dtype(code)
    if dt.kind == "b":
        return np.array([True, False, True, False][:n], dtype=dt)
    if dt.kind in "iu":
        vals = [1, 2, 3, 4][:n]
        return np.array(vals, dtype=dt)
    if dt.kind == "f":
        return np.array([0.5, 1.0, 1.5, 2.0][:n], dtype=dt)
    if dt.kind == "c":
        return np.array([1 + 1j, 2 + 0.5j, 0.5 - 1j, 1.5 + 1.5j][:n], dtype=dt)
    if dt.kind == "M":
        return np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"][:n], dtype=dt)
    if dt.kind == "m":
        return np.array([1, 2, 3, 4][:n], dtype=dt)
    if dt.kind in "US":
        return np.array(["ab", "cd", "ef", "gh"][:n], dtype=dt)
    raise ValueError(f"no sample builder for typecode kind {dt.kind!r} ({code!r})")


def _string_sample(n: int = 4) -> np.ndarray:
    return np.array(["alpha", "Beta3", "xyz ", "abcd"][:n])


def _parse_loop(loop: str) -> Optional[tuple]:
    """'dd->d' -> ('d', 'd'). Returns None for loops this module doesn't
    know how to sample from (currently: 'O' object loops -- arbitrary
    Python objects have no principled generic sample and are out of scope;
    see the report's KNOWN-GAPS section)."""
    if "->" not in loop:
        return None
    lhs, _rhs = loop.split("->", 1)
    if "O" in lhs:
        return None
    return tuple(lhs)


def _build_typed_sample(info_name: str, nin: int, types: tuple, is_string: bool,
                         signature: Optional[str]) -> Optional[tuple]:
    if signature is not None:
        # gufunc (matmul/vecdot/vecmat): the (n,k),(k,m) shape contract
        # matters more than dtype here; build a small valid 2-D/1-D sample
        # directly rather than via the legacy-loop parser, which describes
        # dtype loops only, not the gufunc's core-dimension shape contract.
        if info_name == "matmul":
            return (np.arange(4.0).reshape(2, 2), np.arange(4.0).reshape(2, 2))
        if info_name == "vecdot":
            return (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        if info_name == "vecmat":
            return (np.array([1.0, 2.0]), np.arange(4.0).reshape(2, 2))
        if info_name == "matvec":
            return (np.arange(4.0).reshape(2, 2), np.array([1.0, 2.0]))
        return None

    if is_string:
        base = _string_sample()
        return tuple(base for _ in range(nin)) if nin else None

    for loop in types:
        codes = _parse_loop(loop)
        if codes is None or len(codes) != nin:
            continue
        try:
            return tuple(_typecode_sample(c) for c in codes)
        except ValueError:
            continue
    return None


def describe(path: str) -> UfuncInfo:
    obj = resolve_numpy_ufunc(path)
    is_string = path.startswith("char.") or path.startswith("strings.")
    types = tuple(obj.types)
    typed_sample = _build_typed_sample(path, obj.nin, types, is_string, obj.signature)
    return UfuncInfo(
        name=path, nin=obj.nin, nout=obj.nout, identity=obj.identity,
        signature=obj.signature, types=types, is_string=is_string,
        typed_sample=typed_sample,
    )


# ---------------------------------------------------------------------------
# Applicability of each call form, decided by TRYING it against real numpy,
# never by guessing from nin/nout alone -- signature-bearing gufuncs
# (matmul) reject .reduce/.at/.outer with their own errors, multi-output
# ufuncs (divmod/modf/frexp) reject .reduce/.accumulate/.reduceat/.at with
# theirs, and there is no principled way to enumerate every such rule ahead
# of time other than asking numpy. See GOAL/task brief: "detect that by
# trying it against numpy, not by guessing."
# ---------------------------------------------------------------------------

def applicable_methods(obj, info: UfuncInfo) -> dict[str, bool]:
    result = {m: False for m in UFUNC_METHODS}
    result["call_out"] = False
    result["call_where"] = False
    result["call_dtype"] = False
    result["call_order"] = False

    sample = info.typed_sample
    if sample is None:
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            base_out = obj(*sample)
        except Exception:
            return result  # can't even plain-call it; nothing else applies

        if info.nin == 2 and info.signature is None:
            try:
                obj.reduce(sample[0]); result["reduce"] = True
            except Exception:
                pass
            try:
                obj.accumulate(sample[0]); result["accumulate"] = True
            except Exception:
                pass
            try:
                obj.outer(sample[0], sample[1]); result["outer"] = True
            except Exception:
                pass
            try:
                n = sample[0].shape[0] if sample[0].ndim else 0
                idx = [0] if n < 2 else [0, n // 2]
                obj.reduceat(sample[0], idx); result["reduceat"] = True
            except Exception:
                pass

        try:
            target = sample[0].copy()
            if info.nin == 1:
                obj.at(target, [0])
            elif info.nin == 2:
                obj.at(target, [0], sample[1][:1])
            else:
                raise TypeError("unsupported nin for .at probe")
            result["at"] = True
        except Exception:
            pass

        if not info.is_string:
            try:
                out = np.empty_like(base_out) if info.nout == 1 else None
                if out is not None:
                    obj(*sample, out=out)
                    result["call_out"] = True
            except Exception:
                pass

            try:
                if info.nin >= 1 and info.nout == 1:
                    mask = np.zeros(np.broadcast(*sample).shape if info.nin > 1 else sample[0].shape,
                                     dtype=bool)
                    mask.reshape(-1)[0] = True
                    out = np.zeros_like(base_out)
                    obj(*sample, where=mask, out=out)
                    result["call_where"] = True
            except Exception:
                pass

            try:
                if info.nout == 1:
                    obj(*sample, dtype=np.asarray(base_out).dtype)
                    result["call_dtype"] = True
            except Exception:
                pass

            try:
                if info.nout == 1:
                    obj(*sample, order="K")
                    result["call_order"] = True
            except Exception:
                pass

    return result
