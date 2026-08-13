"""Differential corpus for `numpy.nan_to_num`.

WHY THIS FILE EXISTS. `nan_to_num` was absent from anionpy until 2026-08-04.
It is the first item built on top of `copyto` -- real numpy implements it as
three `copyto(dest, value, where=mask)` calls -- and it is the reason
`copyto` was implemented first. A composition can be wrong when every
primitive in it is right, so it gets its own corpus rather than leaning on
`copyto_cases.py`.

WHAT IS COMPARED. A descriptor, not just the returned values:

    r=<values> | shape | dtype | ndim | root=<values of the ROOT owner> | id=same/new

Four of those six fields have caught something during development and none
is decoration:

  * `root` is read back off the ORIGINAL base array, not the (possibly
    view) argument. `nan_to_num(a[2:6], copy=False)` must write THROUGH to
    `a`; an implementation that quietly substitutes into a private copy
    leaves the returned view looking perfect and the root untouched.
  * `id` distinguishes `copy=False` (numpy returns the SAME object) from
    `copy=True`. Values alone cannot tell those apart.
  * `ndim` catches the 0-d rule: numpy's trailing `x[()]` means a 0-d input
    returns a numpy SCALAR, not a 0-d array.
  * `dtype` catches the layout/width questions -- including that a float16
    array keeps float16 rather than being widened by an f64 substitution
    value.

Errors are compared with full type and message.

THE RULES THIS CORPUS PINS. Each measured against real numpy 2.5.1; none
follows from the others.

1. THE THREE MASKS ARE DECIDED FROM THE ORIGINAL VALUE, ALL AT ONCE. numpy
   computes `isnan`/`isposinf`/`isneginf` up front and only then runs the
   three `copyto` calls, so a substitution cannot cascade:

       nan_to_num([nan, inf], nan=inf, posinf=7.0)  ->  [inf, 7.0]

   A naive sequential implementation gives `[7.0, 7.0]` -- the NaN becomes
   `inf` and is then caught by the posinf pass. Pinned across every float
   width.

2. COMPLEX REAL AND IMAGINARY PARTS ARE INDEPENDENT, AND THE SUBSTITUTION
   DTYPE IS THE COMPONENT'S.

       nan_to_num([complex(nan, inf)])  ->  0 + 1.7976931348623157e+308j

   and an array-valued `nan=[5.0]` lands on BOTH components (`5+5j`), so it
   is one substitution element per ARRAY element, not per component. The
   component dtype is also what error messages name: `nan=2+3j` on a
   complex128 array says "to dtype('float64')", NOT complex128. An
   implementation that resolved against the array's own dtype would accept
   that call instead of raising.

3. DEFAULTS ARE PER-WIDTH, AND THE FLOAT16 ONE IS NOT AN f64 MAX ROUNDED
   DOWN. `posinf` defaults to `finfo(component).max`: 65504.0 for float16,
   3.4028234663852886e+38 for float32/complex64,
   1.7976931348623157e+308 for float64/complex128. Building the float16
   default by casting the f64 max would give `inf`, which is a legal float16
   and therefore fails silently -- hence a float16 case for every default.

4. INTEGER AND BOOL INPUT IGNORES EVERY KEYWORD, INCLUDING INVALID ONES.
   `nan_to_num(int_array, nan=None)` returns the array unchanged and raises
   nothing, even though `nan=None` on a float array is a TypeError. numpy
   returns before the substitutions are examined at all. Every substitution
   spelling is therefore replayed against the integer dtypes, where the
   correct answer is always "no error, no change".

5. `nan=None` AND `posinf=None` MEAN OPPOSITE THINGS. `posinf=None` /
   `neginf=None` are the DEFAULTS; `nan=None` is an object-dtype source
   handed to `copyto` and raises
   `Cannot cast scalar from dtype('O') to dtype('float64') ...`. Same
   spelling, decided entirely by which keyword it lands on.

6. `copy=` IS THREE-WAY. `True`/`1` copy; `False`/`0` never copy and raise
   on an input that would require one; `None` copies only if needed. A `str`
   is rejected with a dedicated sentence rather than being taken as truthy.
   The copy is LAYOUT-PRESERVING (numpy's `order='K'`), so an F-ordered
   input comes back F-ordered -- an earlier draft returned C and this corpus
   is why that was caught before it shipped.

7. SUBSTITUTION VALUES INHERIT COPYTO'S WHOLE CONTRACT. They broadcast
   (`nan=[1,2,3]`), they take an UNCHECKED narrowing cast (float16
   `nan=1e30` stores `inf`), and their casting/broadcast errors are copyto's
   errors verbatim.

WHAT IS DELIBERATELY NOT HERE.

  * The missing overflow `RuntimeWarning` on a narrowing substitution
    (float16 `nan=1e30`). That is anionpy's already-recorded library-wide
    warning gap, not a `nan_to_num` difference; the stored VALUE is
    compared and matches.
  * `casting='unsafe'`-style string/object sources. `nan_to_num` always
    passes `same_kind`, so the seam where numpy starts doing real string
    conversion is unreachable from here (see `copyto`'s
    `foreign_scalar_dtype_name`).
  * `flags.writeable = False`. anionpy has no such setter (a separate,
    already-recorded gap), so read-only inputs are built with
    `broadcast_to`, whose result is read-only in both libraries.
"""

import numpy as np

from registry import REGISTRY, ItemSpec

INF = float("inf")
NAN = float("nan")

_FLOAT_DTYPES = ("float16", "float32", "float64", "complex64", "complex128")
_INT_DTYPES = ("bool", "int8", "int64", "uint8", "uint32")


class NanProbe:
    """One `nan_to_num` call, spelled module-independently."""

    __slots__ = ("kind", "dtype", "subs", "copy")

    def __init__(self, kind, dtype, subs=(), copy="omit"):
        self.kind = kind
        self.dtype = dtype
        self.subs = tuple(subs)
        self.copy = copy

    def __repr__(self):
        return (
            f"NanProbe(kind={self.kind!r}, dtype={self.dtype!r}, "
            f"subs={self.subs!r}, copy={self.copy!r})"
        )


def _seed(dtype):
    """Eight elements covering every branch of the substitution.

    Both signed zeros are present on purpose: they are the one pair of
    values that a comparison-based `isneginf` implementation can confuse
    with `-inf` if it reaches for the sign bit before the magnitude.
    """
    if dtype.startswith("complex"):
        return [
            complex(NAN, 1.0), complex(1.0, NAN), complex(INF, -INF),
            complex(NAN, INF), complex(-INF, NAN), complex(2.0, 3.0),
            complex(0.0, -0.0), complex(-INF, -INF),
        ]
    if dtype in _INT_DTYPES:
        return [1, 2, 3, 4, 5, 6, 7, 8]
    return [NAN, INF, -INF, 1.5, -0.0, 0.0, 2.0, -7.25]


def _build(m, kind, dtype):
    """Return `(array_to_pass, root_to_read_back)`.

    The two differ for every view kind. `root` is what proves the write
    landed in the caller's memory rather than in a private copy.
    """
    base = m.array(_seed(dtype), dtype=dtype)
    if kind == "1d":
        return base, base
    if kind == "0d":
        a = base[0:1].reshape(())
        return a, a
    if kind == "empty":
        a = base[0:0]
        return a, a
    if kind == "2d":
        return base.reshape(2, 4), base
    if kind == "3d":
        return base.reshape(2, 2, 2), base
    if kind == "offset":
        # A CONTIGUOUS view carrying a nonzero offset -- the shape that hid
        # the `cast_to` offset bug from every other corpus in this suite
        # until 2026-08-03. Cheap to build, easy to omit, and the only
        # view kind whose breakage the exotic ones cannot stand in for.
        return base[3:], base
    if kind == "offset_mid":
        return base[2:6], base
    if kind == "step":
        return base[::2], base
    if kind == "rev":
        return base[::-1], base
    if kind == "T":
        return base.reshape(2, 4).T, base
    if kind == "viewview":
        return base[1:][::2], base
    if kind == "row":
        return base.reshape(2, 4)[1], base
    if kind == "col":
        return base.reshape(2, 4)[:, 2], base
    if kind == "fortran":
        # Built from `_seed`, NOT from float literals: a literal `NAN` in
        # an integer-dtype constructor is a ValueError in numpy and a
        # silent cast in anionpy, which is a real divergence but belongs to
        # `array()`, not here. An earlier draft hardcoded the literals and
        # the resulting 11 "failures" were all that unrelated seam.
        s = _seed(dtype)
        a = m.array([s[:4], s[4:]], dtype=dtype, order="F")
        return a, a
    if kind == "readonly":
        a = m.broadcast_to(base[0:2], (3, 2))
        return a, a
    if kind == "list":
        return _seed(dtype), None
    if kind == "nested":
        s = _seed(dtype)
        return [s[:4], s[4:]], None
    if kind == "pyscalar":
        return _seed(dtype)[0], None
    if kind == "empty_list":
        return [], None
    raise AssertionError(kind)


_KINDS = (
    "1d", "0d", "empty", "2d", "3d", "offset", "offset_mid", "step", "rev",
    "T", "viewview", "row", "col", "fortran", "readonly", "list", "nested",
    "pyscalar", "empty_list",
)

# Substitution spellings. `omit` is absence of the keyword, which for
# `posinf`/`neginf` is the same as `None` and for `nan` is not.
_SUBS = {
    "omit": None,
    "pyfloat": lambda m: 1.25,
    "pyint": lambda m: 3,
    "pybool": lambda m: True,
    "pycomplex": lambda m: 2 + 3j,
    "none": lambda m: None,
    "str": lambda m: "ab",
    "bytes": lambda m: b"abc",
    "npf32": lambda m: np.float32(5.0),
    "npf64": lambda m: np.float64(5.0),
    "0darr": lambda m: m.array(9.0),
    "list1": lambda m: [4.0],
    "list8": lambda m: [1.0, 2, 3, 4, 5, 6, 7, 8],
    "arr8": lambda m: m.array([1.0, 2, 3, 4, 5, 6, 7, 8]),
    # Substitution arrays whose PROVENANCE differs from a fresh
    # allocation: an offset slice and a reversed view. Same values, same
    # dtype, same shape -- different strides and a different base offset.
    # That distinction is what the 2026-08-03 `cast_to` incident turned on.
    "arr8_off": lambda m: m.arange(16, dtype="float64")[8:],
    "arr8_rev": lambda m: m.arange(8, dtype="float64")[::-1],
    "arr_i8": lambda m: m.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="int8"),
    "arr_c": lambda m: m.array([1 + 1j]),
    "wrongshape": lambda m: [1.0, 2.0, 3.0],
    "huge": lambda m: 1e30,
    "huge2": lambda m: 1e300,
    "bigint": lambda m: 2 ** 70,
    "inf": lambda m: INF,
    "neginf": lambda m: -INF,
    "nan": lambda m: NAN,
    "negzero": lambda m: -0.0,
    "tuple2": lambda m: (1.0, 2.0),
}

_COPIES = {
    "omit": None,
    "true": lambda m: True,
    "false": lambda m: False,
    "none": lambda m: None,
    "zero": lambda m: 0,
    "one": lambda m: 1,
    "npbool": lambda m: np.True_,
    "str": lambda m: "x",
}


def _flat(a):
    """Values as plain Python scalars -- `tolist()` and nothing else.

    Never `np.asarray(...).tolist()`: that would run anionpy's answers back
    through real numpy, and numpy may supply this suite's inputs but never
    its answers.
    """
    if a is None:
        return None
    try:
        return a.tolist()
    except AttributeError:
        return a


def _layout(r):
    """`C`/`F`/`CF`/`-`, from the result's own contiguity flags.

    Load-bearing, not decoration. `copy=True` uses numpy's `order='K'`,
    which PRESERVES layout: an F-ordered input comes back F-ordered. Nothing
    else in this descriptor can see that -- `tolist()`, shape, dtype and
    ndim are all identical for a C copy and an F copy of the same array.
    Confirmed by mutation: with the implementation switched from
    `to_contiguous_order("K")` to plain `to_contiguous()`, every other
    field still matched and the corpus passed. This field is the only
    thing that fails.

    A numpy scalar (the 0-d return) has no `.flags`, hence the fallback.
    """
    try:
        f = r.flags
        c, o = bool(f["C_CONTIGUOUS"]), bool(f["F_CONTIGUOUS"])
    except (AttributeError, KeyError, TypeError):
        return "scalar"
    return ("C" if c else "") + ("F" if o else "") or "-"


def _probe(m, p):
    try:
        arg, root = _build(m, p.kind, p.dtype)
        kwargs = {}
        for name, key in p.subs:
            if key == "omit":
                continue
            kwargs[name] = _SUBS[key](m)
        if p.copy != "omit":
            kwargs["copy"] = _COPIES[p.copy](m)
    except BaseException as exc:  # noqa: BLE001
        return f"setup:{type(exc).__name__}:{exc}"
    try:
        r = m.nan_to_num(arg, **kwargs)
    except BaseException as exc:  # noqa: BLE001
        # `BaseException`, not `Exception`: PyO3's `PanicException` derives
        # from `BaseException`, and a panic must be graded as the
        # divergence it is rather than crash the harness.
        return f"raised:{type(exc).__name__}:{exc}"
    ident = "same" if r is arg else "new"
    return (
        f"r={_flat(r)}|shape={tuple(r.shape)}|dt={r.dtype.name}|nd={r.ndim}"
        f"|root={_flat(root)}|id={ident}|layout={_layout(r)}"
    )


def _numpy_adapter(arg, *rest, **kwargs):
    assert isinstance(arg, NanProbe), arg
    return _probe(np, arg)


def _ionp_adapter(arg, *rest, **kwargs):
    import anionpy

    assert isinstance(arg, NanProbe), arg
    return _probe(anionpy, arg)


def _cases():
    out = []
    seen = set()

    def add(kind, dtype, subs=(), copy="omit"):
        subs = tuple(subs)
        label = f"{kind}|{dtype}|{','.join(f'{n}={k}' for n, k in subs) or '-'}|{copy}"
        if label in seen:
            return
        seen.add(label)
        out.append((label, (NanProbe(kind, dtype, subs, copy),), {}))

    # --- Rules 1/3: defaults across every dtype x every provenance. This
    # is the block that pins the per-width `finfo.max` and the
    # write-through behaviour; the view kinds are the ones that matter.
    for dtype in _FLOAT_DTYPES + _INT_DTYPES:
        for kind in _KINDS:
            add(kind, dtype)

    # --- Rule 7 + rule 2: every substitution spelling, on each keyword
    # separately, across every float width. Keeping the keywords separate
    # (rather than only testing them together) is what makes rule 5's
    # `nan=None` vs `posinf=None` asymmetry visible.
    for dtype in _FLOAT_DTYPES:
        for key in _SUBS:
            for name in ("nan", "posinf", "neginf"):
                add("1d", dtype, ((name, key),))

    # --- Rule 4: the same spellings against integer/bool input, where the
    # answer is always "unchanged, no error" -- including for the
    # spellings that are errors on a float array.
    for dtype in _INT_DTYPES:
        for key in _SUBS:
            add("1d", dtype, (("nan", key),))
            add("1d", dtype, (("posinf", key),))

    # --- Rule 1: the cascade cases. `nan=inf` with an explicit `posinf`
    # is the pair that separates "masks decided up front" from "three
    # sequential passes"; the rest of the product is here so a fix that
    # special-cased only the obvious spelling still fails.
    for dtype in _FLOAT_DTYPES:
        for a_ in ("omit", "pyfloat", "inf", "neginf", "nan", "none"):
            for b_ in ("omit", "pyfloat", "inf", "none"):
                for c_ in ("omit", "pyfloat", "neginf", "none"):
                    add("1d", dtype, (("nan", a_), ("posinf", b_), ("neginf", c_)))

    # --- Rule 7's broadcast/provenance half: substitutions whose shape or
    # stride pattern must survive being pushed through copyto, applied to
    # destinations that are themselves views.
    for kind in ("0d", "empty", "2d", "offset", "offset_mid", "rev", "T",
                 "col", "viewview", "readonly", "fortran"):
        for key in ("pyfloat", "list1", "list8", "arr8", "arr8_off",
                    "arr8_rev", "0darr", "wrongshape", "none", "str",
                    "inf", "nan", "negzero"):
            for dtype in ("float64", "float32", "float16", "complex128"):
                add(kind, dtype, (("nan", key),))

    # --- Rule 6: the copy flag, against every provenance that can
    # distinguish its three states.
    for dtype in ("float64", "float16", "complex64", "int64", "bool"):
        for kind in ("1d", "0d", "empty", "offset", "rev", "T", "fortran",
                     "readonly", "list", "pyscalar", "nested"):
            for cs in _COPIES:
                add(kind, dtype, (), cs)

    # --- Rule 6 x rule 5: `copy=False` on a read-only input must lose to
    # nothing and win over a bad substitution value, matching copyto's own
    # tier ordering (read-only is tier 4, src casting is tier 5).
    for dtype in ("float64", "complex128", "int64"):
        for key in ("pyfloat", "none", "str", "wrongshape"):
            for cs in ("false", "true", "none"):
                add("readonly", dtype, (("nan", key),), cs)

    return out


def _install():
    REGISTRY["nan_to_num"] = ItemSpec(
        name="nan_to_num",
        kind="custom",
        numpy_adapter=_numpy_adapter,
        ionp_adapter=_ionp_adapter,
        scalar_like=True,
        custom_cases=_cases,
    )


_install()
