"""Differential corpus for `numpy.copyto`.

WHY THIS FILE EXISTS. `copyto` was entirely absent from anionpy until
2026-08-03. It is the keystone under the whole `out=` family and under
`nan_to_num`/`place`/`putmask`, so it was implemented first and this file is
its corpus.

WHAT IS COMPARED. `copyto` returns `None` on both sides, always, so the
return value grades nothing. What is compared is a DESCRIPTOR built
identically on each side:

    ret | dst values | dst dtype | dst shape      (on success)
    raised:<ExcType>:<message>                    (on failure)

The values are read back off `dst` AFTER the call, because `copyto`'s entire
observable effect is the mutation. Where a view is written through, the
descriptor reads the ROOT owner, not the view -- a `copyto` that writes into
a private copy leaves the view looking right and the root untouched.

Errors are compared with FULL message and exception type. That is not
belt-and-braces: of the divergences this corpus and its sweeps found, every
single one was a wrong message, a wrong exception TYPE, or a raise-vs-no-
raise -- never a wrong stored value. Grading values alone would have called
this implementation finished while it was still wrong in five distinct ways.

THE RULES THIS CORPUS PINS. Each was measured against real numpy 2.5.1, and
none follows from the others.

1. SEVEN-TIER ERROR ORDERING. Simultaneous faults do not race. Confirmed by
   building cases that fail two tiers at once and seeing which wins:

       casting= string  >  dst-is-array  >  where= mask dtype
         >  dst read-only  >  src cast/value  >  src broadcast
         >  where= mask broadcast

   Two of those are genuinely counterintuitive and are pinned here because
   an implementation gets them wrong by default: the casting-STRING check
   beats the dst-type check (`copyto([1,2], 1, casting='bogus')` complains
   about the string, not the list), and the mask-DTYPE check beats the
   read-only check.

2. WEAK PYTHON SCALARS ARE NOT STRONG ARRAY SOURCES. A bare Python
   int/float/complex follows NEP 50 weak rules against dst's dtype; an
   array, a numpy scalar, a list and a tuple all follow the ordinary casting
   table. The two disagree in BOTH directions, so neither can stand in for
   the other:

       copyto(f4_dst, 2.5,             casting='safe')   ->  writes 2.5
       copyto(f4_dst, np.array(2.5),   casting='safe')   ->  TypeError

3. VALUE CHECKING APPLIES TO THE SCALAR, NEVER TO A SEQUENCE.

       copyto(i1_dst, 300)     ->  OverflowError (out of bounds for int8)
       copyto(i1_dst, [300])   ->  silently stores 44

   The list is a strong source and takes an unchecked narrowing cast. The
   scalar check also fires when nothing would be written at all --
   `copyto(i1_dst, 300, where=False)` still raises.

4. `src` AND `where=` BROADCAST BY DIFFERENT RULES. `src` may shed LEADING
   size-1 axes to fit a lower-rank dst; the mask may not.

       src  (1,3) -> dst (3,)   OK          mask (1,3) -> dst (3,)   raises
       src  (1,1) -> dst ()     OK          mask (1,1) -> dst ()     raises
       src  (3,1) -> dst (3,)   raises

5. OVERLAP IS BUFFERED. `copyto(a[1:], a[:-1])` gives `[1,1,2,3]` and
   `copyto(a[:-1], a[1:])` gives `[2,3,4,4]`. Neither smears, in either
   direction, so the read must be fully materialized before the write
   starts. This is the one rule a "compose it out of existing primitives"
   implementation is most likely to break while every primitive it uses
   stays correct.

6. `where=` COERCION IS SOURCE-DEPENDENT. An ARRAY mask must already be
   boolean (numpy demands a 'safe' cast to bool, which only bool passes); a
   list/tuple/int is coerced instead. `where=None` is NOT "default" -- it
   writes nothing, unlike an omitted `where=`.

WHAT IS DELIBERATELY NOT COVERED.

  * A non-array `dst`. anionpy raises the same sentence numpy does but names
    its own type (`anionpy.ndarray`, not `numpy.ndarray`), consistent with
    every other type-naming message in the library. Recorded in
    KNOWN-DIFFERENCES.md rather than tested green here or papered over by
    claiming to be numpy.
  * Object and string dtypes, which anionpy does not have at all
    (`copyto(f8_dst, 'abc')` names `dtype('<U3')` on numpy's side).
  * The overflow `RuntimeWarning` on `copyto(f4_dst, 1e40)`. anionpy stores
    `inf` and matches the VALUE; the missing warning is a separate,
    already-recorded library-wide gap.
"""

import numpy as np

from registry import REGISTRY, ItemSpec

_DTYPES = (
    "bool", "uint8", "int8", "int16", "int32", "int64", "uint32", "uint64",
    "float16", "float32", "float64", "complex64", "complex128",
)

_RULES = ("no", "equiv", "safe", "same_kind", "unsafe")


class CopyProbe:
    """One `copyto` call, spelled module-independently."""

    __slots__ = ("dst", "src", "casting", "where")

    def __init__(self, dst, src, casting=None, where=None):
        self.dst = dst
        self.src = src
        self.casting = casting
        self.where = where

    def __repr__(self):
        return (
            f"CopyProbe(dst={self.dst!r}, src={self.src!r}, "
            f"casting={self.casting!r}, where={self.where!r})"
        )


# --------------------------------------------------------------------------
# Destinations. Each entry builds (view_to_write, root_to_read_back). They
# differ only for the view kinds, which is exactly where a write into a
# private copy would otherwise go unnoticed.
# --------------------------------------------------------------------------
def _dst_builder(kind, dtype):
    def build(m):
        if kind == "1d":
            r = m.zeros(3, dtype=dtype)
            return r, r
        if kind == "2d":
            r = m.zeros((2, 3), dtype=dtype)
            return r, r
        if kind == "3d":
            r = m.zeros((2, 2, 2), dtype=dtype)
            return r, r
        if kind == "0d":
            r = m.zeros((), dtype=dtype)
            return r, r
        if kind == "empty":
            r = m.zeros(0, dtype=dtype)
            return r, r
        if kind == "1x3":
            r = m.zeros((1, 3), dtype=dtype)
            return r, r
        if kind == "step":
            r = m.zeros(6, dtype=dtype)
            return r[::2], r
        if kind == "rev":
            r = m.zeros(4, dtype=dtype)
            return r[::-1], r
        if kind == "T":
            r = m.zeros((2, 3), dtype=dtype)
            return r.T, r
        if kind == "slice2d":
            r = m.zeros((3, 4), dtype=dtype)
            return r[1:, 1:], r
        raise AssertionError(f"unknown dst kind {kind}")

    return build


_DST_KINDS = ("1d", "2d", "3d", "0d", "empty", "1x3", "step", "rev", "T", "slice2d")

# --------------------------------------------------------------------------
# Sources. The weak/strong split is the point: `pyint_5` and `arr_i8_5` hold
# the same NUMBER and must not behave the same way.
# --------------------------------------------------------------------------
_SRCS = {
    # Weak (Python) scalars.
    "pyint_5": lambda m: 5,
    "pyint_0": lambda m: 0,
    "pyint_300": lambda m: 300,
    "pyint_neg1": lambda m: -1,
    "pyint_neg300": lambda m: -300,
    "pyint_70000": lambda m: 70000,
    "pyint_2p63": lambda m: 2 ** 63,
    "pyint_2p64m1": lambda m: 2 ** 64 - 1,
    "pyint_2p70": lambda m: 2 ** 70,
    "pyint_neg2p63": lambda m: -(2 ** 63),
    "pyfloat": lambda m: 2.5,
    "pyfloat_neg": lambda m: -0.5,
    "pyfloat_whole": lambda m: 3.0,
    "pycomplex": lambda m: 1 + 2j,
    "pycomplex_real": lambda m: 3 + 0j,
    "pybool_T": lambda m: True,
    "pybool_F": lambda m: False,
    # Strong sources.
    "arr_i64": lambda m: m.array([5], dtype="int64"),
    "arr_i64_vec": lambda m: m.array([1, 2, 3], dtype="int64"),
    "arr_u8": lambda m: m.array([1, 2, 3], dtype="uint8"),
    "arr_f64": lambda m: m.array([1.5, 2.5, 3.5]),
    "arr_f32": lambda m: m.array([1.5, 2.5, 3.5], dtype="float32"),
    "arr_c128": lambda m: m.array([1 + 2j, 3 + 4j, 5 + 6j]),
    "arr_bool": lambda m: m.array([True, False, True]),
    "arr_0d_f64": lambda m: m.array(2.5),
    "arr_0d_i64": lambda m: m.array(7),
    "list_int": lambda m: [1, 2, 3],
    "list_1": lambda m: [5],
    "list_300": lambda m: [300],
    "list_float": lambda m: [1.5, 2.5, 3.5],
    "list_bool": lambda m: [True, False, True],
    "tuple_int": lambda m: (1, 2, 3),
    "nested": lambda m: [[1, 2, 3], [4, 5, 6]],
}

# Sources whose SHAPE is the point (rule 4).
_SHAPE_SRCS = {
    "sh_1x3": lambda m: m.ones((1, 3)),
    "sh_3x1": lambda m: m.ones((3, 1)),
    "sh_1x1": lambda m: m.ones((1, 1)),
    "sh_1": lambda m: m.ones(1),
    "sh_3": lambda m: m.ones(3),
    "sh_2x3": lambda m: m.ones((2, 3)),
    "sh_7": lambda m: m.ones(7),
    "sh_2x1": lambda m: m.ones((2, 1)),
    "sh_1x1x3": lambda m: m.ones((1, 1, 3)),
    "sh_0": lambda m: m.ones(0),
}

_WHERES = {
    "omitted": None,  # sentinel handled in _probe
    "True": lambda m: True,
    "False": lambda m: False,
    "None": lambda m: None,
    "list_bool": lambda m: [True, False, True],
    "tuple_bool": lambda m: (True, False, True),
    "list_int": lambda m: [1, 0, 1],
    "int_1": lambda m: 1,
    "int_0": lambda m: 0,
    "arr_bool": lambda m: m.array([True, False, True]),
    "arr_bool_1": lambda m: m.array([True]),
    "arr_bool_0d": lambda m: m.array(True),
    "arr_bool_all": lambda m: m.array([True, True, True]),
    "arr_i64": lambda m: m.array([1, 0, 1], dtype="int64"),
    "arr_f64": lambda m: m.array([1.0, 0.0, 1.0]),
    "arr_u8": lambda m: m.array([1, 0, 1], dtype="uint8"),
    "arr_bool_2x3": lambda m: m.ones((2, 3), dtype=bool),
    "arr_bool_1x3": lambda m: m.ones((1, 3), dtype=bool),
    "arr_bool_5": lambda m: m.ones(5, dtype=bool),
    "arr_bool_2": lambda m: m.ones(2, dtype=bool),
}


def _flat(a):
    """Values as a plain nested list of Python scalars.

    `tolist()` and nothing else. An earlier draft round-tripped through
    `np.asarray(...).tolist()` to "normalize spelling"; that would have run
    anionpy's results through real numpy, which is exactly the thing this suite
    must never do -- numpy may supply inputs, never answers.
    """
    return a.tolist()


def _probe(m, p):
    try:
        build = p.dst if callable(p.dst) else _dst_builder(*p.dst)
        dst, root = build(m)
        src = _SRCS.get(p.src, _SHAPE_SRCS.get(p.src))
        src = src(m) if src is not None else p.src
        kwargs = {}
        if p.casting is not None:
            kwargs["casting"] = p.casting
        if p.where != "omitted":
            kwargs["where"] = _WHERES[p.where](m)
    except BaseException as exc:  # noqa: BLE001
        return f"setup:{type(exc).__name__}:{exc}"
    try:
        ret = m.copyto(dst, src, **kwargs)
    except BaseException as exc:  # noqa: BLE001
        # `BaseException`, not `Exception`: PyO3 raises `PanicException`,
        # which derives from `BaseException`. Catching only `Exception`
        # would let an anionpy panic escape as a harness crash instead of
        # being graded as the divergence it is.
        return f"raised:{type(exc).__name__}:{exc}"
    # The ROOT, not the view that was written through.
    return f"ret={ret!r}|root={_flat(root)}|dt={root.dtype.name}|shape={tuple(root.shape)}"


def _numpy_adapter(arg, *rest, **kwargs):
    assert isinstance(arg, CopyProbe), arg
    return _probe(np, arg)


def _ionp_adapter(arg, *rest, **kwargs):
    import anionpy

    assert isinstance(arg, CopyProbe), arg
    return _probe(anionpy, arg)


def _cases():
    out = []

    def add(dst, src, casting=None, where="omitted"):
        label = f"{dst}|{src}|{casting or '-'}|{where}"
        out.append((label, (CopyProbe(dst, src, casting, where),), {}))

    # --- Rule 2 + 3: the full weak/strong x dtype x casting-rule table. This
    # is the heart of the corpus. Every cell is a distinct answer: OK, one of
    # two different casting sentences, or one of three different
    # OverflowError wordings.
    _CAST_SRCS = (
        "pyint_5", "pyint_300", "pyint_2p63", "pyint_2p70", "pyint_neg1",
        "pyint_2p64m1", "pyint_neg2p63", "pyint_70000",
        "pyfloat", "pycomplex", "pybool_T", "pybool_F",
        "arr_i64", "arr_f64", "arr_c128", "arr_bool", "arr_0d_f64",
        "arr_0d_i64", "arr_f32", "arr_u8", "list_1", "list_300", "tuple_int",
    )
    for dtype in _DTYPES:
        for src in _CAST_SRCS:
            for rule in _RULES:
                add(("1d", dtype), src, rule)

    # The same table with `casting` OMITTED, which must behave as
    # 'same_kind' -- a default that is easy to get right and easy to lose.
    for dtype in _DTYPES:
        for src in _CAST_SRCS:
            add(("1d", dtype), src)

    # --- Rule 3: value checking is scalar-only, and fires even when the
    # mask would suppress the write entirely.
    for dtype in ("bool", "uint8", "int8", "int16", "uint32", "uint64", "int64"):
        for src in ("pyint_300", "pyint_neg1", "pyint_neg300", "pyint_70000",
                    "pyint_2p63", "pyint_2p70", "list_300", "arr_i64"):
            for where in ("omitted", "False", "None", "list_bool"):
                add(("1d", dtype), src, None, where)

    # --- Rule 4: broadcast, both policies, against every dst rank.
    for kind in ("1d", "2d", "3d", "0d", "empty", "1x3"):
        for src in _SHAPE_SRCS:
            add((kind, "float64"), src)

    # --- Rule 6 + tier 3: `where=` across every spelling, on dsts whose
    # shape makes the mask's broadcast policy observable.
    for kind in ("1d", "2d", "0d", "1x3", "empty"):
        for where in _WHERES:
            add((kind, "float64"), "pyfloat", None, where)
            add((kind, "float64"), "arr_f64", None, where)
    # A mask on an integer dst too: the mask-dtype error quotes dtypes and
    # must not accidentally quote the DST's.
    for where in ("arr_i64", "arr_f64", "arr_u8", "arr_bool", "list_int"):
        add(("1d", "int64"), "pyint_5", None, where)
        add(("1d", "int64"), "pyint_5", "unsafe", where)

    # --- Rule 5 + view write-through: every non-contiguous destination,
    # crossed with sources that are plain, overlapping, or shaped to
    # broadcast. If `copyto` writes into a private copy, the root reads back
    # unchanged and these are the cases that say so.
    for kind in ("step", "rev", "T", "slice2d", "2d", "3d"):
        for dtype in ("float64", "int64", "complex128", "bool", "uint8"):
            for src in ("pyint_5", "pyfloat", "arr_f64", "list_int", "nested",
                        "arr_0d_f64", "pybool_T"):
                add((kind, dtype), src)
        for where in ("True", "False", "None", "arr_bool_2x3", "list_bool"):
            add((kind, "float64"), "pyfloat", None, where)

    # --- Tier ordering: cases that fail two tiers at once. The label says
    # which two; the descriptor says which numpy picked.
    for rule in ("bogus", "BOGUS", "", "Safe"):
        add(("1d", "float64"), "pyfloat", rule)
        add(("1d", "int64"), "arr_f64", rule, "arr_i64")  # + cast + mask faults
    for where in ("arr_i64", "arr_f64"):
        add(("1d", "int64"), "arr_f64", None, where)      # mask dtype vs src cast
        add(("1d", "int64"), "sh_7", None, where)         # mask dtype vs broadcast
        add(("1d", "int64"), "pyint_300", None, where)    # mask dtype vs overflow
    add(("1d", "int8"), "sh_7", None, "arr_bool_5")       # src bcast vs mask bcast
    add(("1d", "int8"), "pyint_300", None, "arr_bool_5")  # overflow vs mask bcast
    add(("1d", "int8"), "arr_f64", None, "arr_bool_5")    # src cast vs mask bcast

    # --- `castoff/`: REGRESSION GUARD for the ionp-core bug this item's
    # out-of-corpus sweep found on 2026-08-03.
    #
    # `NdArray::cast_to` hardcoded `offset: 0` in its result while
    # `Buffer::cast_to` returns a buffer of the FULL original length, so a
    # plain offset slice -- `a[2:5]`, which is contiguous and carries only
    # `offset: 2` -- was re-read from element 0 whenever a cast was needed.
    # `copyto(int8_dst, arange(9.)[2:5])` stored `[0,1,2]` instead of
    # `[2,3,4]`: right dtype, right shape, wrong data.
    #
    # This was never a `copyto` bug. `astype`, `asarray(dtype=)` and
    # `array(dtype=)` -- all long-declared "exact" -- were wrong the same
    # way and had no case that caught it. These guard the fix from BELOW,
    # through copyto, in addition to the ionp-core unit test.
    #
    # The pairing is the whole point: the SAME-dtype spelling of each case
    # takes cast_to's early-return and was always correct, so a fix that
    # regressed would show up as the two halves disagreeing. Cases whose
    # src is non-contiguous (`co_rev`, `co_t`) were also always correct --
    # `to_contiguous()` rebases honestly -- and are here so the guard
    # cannot be satisfied by a change that merely re-breaks those instead.
    _OFFSET_SRCS = {
        "co_mid": lambda m: m.arange(9, dtype="float64")[2:5],
        "co_tail": lambda m: m.arange(9, dtype="float64")[6:],
        "co_last": lambda m: m.arange(9, dtype="float64")[8:],
        "co_head": lambda m: m.arange(9, dtype="float64")[0:3],
        "co_2d": lambda m: m.arange(12, dtype="float64").reshape(3, 4)[1:, :],
        "co_2drow": lambda m: m.arange(12, dtype="float64").reshape(3, 4)[2:3, :],
        "co_int": lambda m: m.arange(9, dtype="int64")[2:5],
        "co_rev": lambda m: m.arange(9, dtype="float64")[::-1][2:5],
        "co_t": lambda m: m.arange(6, dtype="float64").reshape(2, 3).T,
    }
    _SHAPE_SRCS.update(_OFFSET_SRCS)
    for src in _OFFSET_SRCS:
        for dtype in _DTYPES:
            for rule in ("unsafe", "same_kind"):
                add(("1d", dtype), src, rule)
                add(("2d", dtype), src, rule)
        # Through a strided dst and a mask too, so a fix that repaired the
        # value but broke the write path would still be caught.
        for kind in ("step", "rev", "T"):
            add((kind, "int8"), src, "unsafe")
        add(("1d", "int8"), src, "unsafe", "list_bool")

    # --- Empty and 0-d destinations, where "nothing to write" must still
    # perform every check.
    for dtype in ("float64", "int8", "bool"):
        for src in ("pyint_300", "pyfloat", "arr_f64", "sh_1", "sh_3", "sh_0"):
            add(("empty", dtype), src)
            add(("0d", dtype), src)

    return out


def _install():
    REGISTRY["copyto"] = ItemSpec(
        name="copyto",
        kind="custom",
        numpy_adapter=_numpy_adapter,
        ionp_adapter=_ionp_adapter,
        scalar_like=True,
        custom_cases=_cases,
    )


_install()
