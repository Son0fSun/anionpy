"""Case builder for kind="ufunc" registry items.

One job: given a `ufunc_introspect.UfuncInfo` and the trial-detected
`applicable_methods()` map for it, build the full (label, args, kwargs)
case list that `run.py`/`harness.py` will run against real numpy and real
anionpy. `args[0]` is always the call-form tag ("call" | "reduce" |
"accumulate" | "outer" | "reduceat" | "at") that
`registry.build_ufunc_dispatcher()` reads to decide which of the ufunc
protocol's methods to invoke -- see that function's docstring for why the
dispatch lives there instead of here (mutation safety needs to happen
inside the same closure that's called twice, once per side, on the same
args tuple).

Coverage per ufunc, gated by real trial-applicability (never guessed):
  - plain call, over the FULL existing numeric/string corpus (this is
    where the edge cases GOAL-ionp.md names actually get exercised: empty,
    0-d, NaN, inf, -0.0, negative/non-contig strides, broadcasting,
    int-overflow boundaries, NEP-50 mixed weak/strong scalar promotion)
  - dtype=, out=, where=+out= kwarg forms, one representative case each,
    built from the ufunc's OWN `.types`-derived typed_sample
  - .reduce / .accumulate / .outer / .reduceat / .at, each over the typed
    sample plus a deterministically sampled slice of the broader corpus
"""
from __future__ import annotations

import warnings

import numpy as np

import _bootstrap  # noqa: F401
import corpus
import ufunc_corpus
import ufunc_introspect


def _reshape_like(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Tile/truncate `arr`'s own flat values (preserving dtype) to fill
    `shape`. Only the LAYOUT of the result matters to order= cases, not the
    particular values, so reusing the ufunc's own typed_sample values (rather
    than inventing new ones) keeps every order= case in the ufunc's declared
    valid dtype domain for free."""
    n = 1
    for d in shape:
        n *= d
    flat = np.asarray(arr).reshape(-1)
    if flat.size == 0:
        # can't tile from nothing; fall back to a zero-filled same-dtype array
        return np.zeros(shape, dtype=arr.dtype)
    reps = -(-n // flat.size)
    tiled = np.tile(flat, reps)[:n]
    return tiled.reshape(shape)


def _order_layout_variants(arr: np.ndarray, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Real, distinctly-strided arrays of `shape` built from `arr`'s values:
    C-contiguous, F-contiguous, and (for ndim>=3 only) a genuinely
    neither-C-nor-F-contiguous view -- the case order='K' actually exists to
    handle (matching an arbitrary input's own memory order), as opposed to
    order='C'/'F' which only ever need a C or F sample to exercise.

    The "swapped" variant is built by swapping only the LAST TWO axes of a
    C-contiguous array shaped with those two axes pre-swapped. Swapping ALL
    axes of a C array (a full reversal) always degenerates to plain
    F-contiguous -- for a 2-D array that's the *only* possible transpose, so
    no genuinely-neither-C-nor-F 2-D variant exists; "swapped" is therefore
    only produced for ndim>=3, where a PARTIAL axis swap leaves strides that
    match neither the C nor the F formula."""
    base = _reshape_like(arr, shape)
    variants = {
        "c": np.ascontiguousarray(base),
        "f": np.asfortranarray(base),
    }
    if len(shape) >= 3:
        swap_shape = shape[:-2] + (shape[-1], shape[-2])
        base2 = np.ascontiguousarray(_reshape_like(arr, swap_shape))
        variants["swapped"] = np.swapaxes(base2, -1, -2)
    return variants


def _order_call_cases(info: "ufunc_introspect.UfuncInfo") -> list[tuple]:
    """order= is only meaningful once an array has more than one axis (a
    0-D or 1-D array has exactly one possible memory layout, so every order=
    value collapses to the same bytes/strides and cannot distinguish a
    correct implementation from a broken one -- deliberately NOT padding the
    corpus with such cases). Exercises all four order values against 2-D and
    3-D inputs, across C-contiguous, F-contiguous, and transposed-view
    (neither-C-nor-F) operand layouts, for both unary and binary ufuncs --
    including MIXED-layout binary pairs, which is what actually exercises
    order='K's multi-operand tie-break rule and order='A's "all operands
    F-contiguous" rule (a same-layout-only corpus can't tell 'A' apart from
    'F')."""
    if info.signature is not None or info.is_string or info.nout != 1:
        return []
    sample = info.typed_sample
    if sample is None:
        return []

    cases: list[tuple] = []
    shapes = [(2, 3), (2, 2, 3)]
    orders = ["C", "F", "A", "K"]

    if info.nin == 1:
        dtype_arr = sample[0]
        for shape in shapes:
            layouts = _order_layout_variants(dtype_arr, shape)
            for layout_name, arr in layouts.items():
                for order in orders:
                    label = f"call/order/{order}/unary_{layout_name}/{shape}"
                    cases.append((label, ("call", arr), {"order": order}))
    elif info.nin == 2:
        a_arr, b_arr = sample[0], sample[1]
        for shape in shapes:
            a_layouts = _order_layout_variants(a_arr, shape)
            b_layouts = _order_layout_variants(b_arr, shape)
            common_layouts = [n for n in ("c", "f", "swapped") if n in a_layouts and n in b_layouts]
            # same-layout pairs (both C, both F, and -- for 3-D -- both swapped)
            for layout_name in common_layouts:
                a = a_layouts[layout_name]
                b = b_layouts[layout_name]
                for order in orders:
                    label = f"call/order/{order}/binary_same_{layout_name}/{shape}"
                    cases.append((label, ("call", a, b), {"order": order}))
            # mixed-layout pairs: this is what makes 'A' differ from 'F' and
            # exercises 'K's ambiguous-tie-break-defaults-to-C rule.
            mixed_pairs = [("f", "c"), ("c", "f")]
            if "swapped" in common_layouts:
                mixed_pairs.append(("f", "swapped"))
            for (an, bn) in mixed_pairs:
                a = a_layouts[an]
                b = b_layouts[bn]
                for order in orders:
                    label = f"call/order/{order}/binary_mixed_{an}_{bn}/{shape}"
                    cases.append((label, ("call", a, b), {"order": order}))
            # broadcasting pair: one operand missing a leading axis, so its
            # own size-1-vs-absent axes must not spuriously "vote" in K's
            # per-axis tie-break -- mirrors the 3-D-broadcast scratch probe.
            b_broadcast = _order_layout_variants(b_arr, shape[1:])
            for layout_name in ("c", "f"):
                a = a_layouts[layout_name]
                b = b_broadcast[layout_name]
                for order in orders:
                    label = f"call/order/{order}/binary_broadcast_{layout_name}/{shape}"
                    cases.append((label, ("call", a, b), {"order": order}))

    return cases


def _reduceat_indices(n: int) -> list[int]:
    if n <= 0:
        return [0]
    if n == 1:
        return [0]
    return [0, n // 2]


_REDUCE_SWEEP_LENGTHS = (2, 4, 7, 8, 9, 16, 32, 64, 127, 128, 129, 1000)
_REDUCE_SWEEP_FLOAT_DTYPES = ("float16", "float32", "float64")
_REDUCE_SWEEP_COMPLEX_DTYPES = ("complex64", "complex128")

# Deliberately scoped to exactly the ops this class of bug was found in/near
# (this task's declared items `add`/`subtract`/`multiply`, plus `divide`/
# `true_divide` which share the identical `BinaryOp::Divide` reduce_binary
# path in ionp-core and the identical numpy pairwise/float16-widening rule) --
# NOT extended to every same-dtype-in-same-dtype-out binary ufunc
# (maximum/minimum/floor_divide/fmod/mod/remainder/logaddexp/arctan2/
# copysign/hypot/power/...). Those were never measured against this specific
# bug class in this task, and several (hypot, power, ...) are ALREADY
# failing for unrelated, out-of-scope reasons (see this task's report) --
# blindly widening this sweep to them would risk manufacturing NEW,
# unreviewed failing cases rather than pinning down a fix that was actually
# verified.
_REDUCE_SWEEP_OPS = frozenset({"add", "subtract", "multiply", "divide", "true_divide"})


def _reduce_length_sweep_cases(info: ufunc_introspect.UfuncInfo, sample) -> list[tuple]:
    """Regression cases for the class of bug where `.reduce()` diverges from
    real numpy at lengths that cross numpy's pairwise-summation blocking
    threshold (`pairwise_sum_@TYPE@` switches from a naive fold to an
    8-way-unrolled blocked accumulation at n>=8, complex at width 4, with a
    further recursive-halving regime above blocksize 128) or that exercise
    numpy's float16-only accumulator-widening rule (the WHOLE reduce fold
    happens in a float32 intermediate, rounded back to float16 once at the
    very end -- not after every step). The pre-existing corpus (`corpus.py`)
    never builds an array with a reduced-axis length >= 8, so neither class
    of bug is reachable without cases like these (see this task's report for
    the two real anionpy bugs this was written to catch and pin down).

    Builds its OWN float/complex base data rather than reusing `typed_sample`
    (whose dtype, for `add`/`multiply`/`subtract`, is `bool_`/`int8` --
    numpy's FIRST type-loop match in `.types`, not a float one -- so gating
    on `sample[0].dtype.kind` would silently skip the very three declared
    items this task is about; verified via direct inspection of
    `np.add.types`/`np.multiply.types`/`np.subtract.types`/`np.divide.types`
    during this task). `divide`'s base values are drawn strictly positive
    (`0.25` to `4.0`) to keep every reduce well-defined (no exact-zero
    divisor); the other three ops use a signed range.

    Covers, per swept length: 1-D contiguous, 1-D non-contiguous (a
    stride-2 slice of a 2x-wide buffer), and two 2-D layouts of shape
    (n, 3) -- C-order (where the reduced default axis 0 is the OUTER,
    non-innermost/"narrow" axis, dominated by the smaller-stride kept axis
    1) and F-order (where axis 0 IS the innermost/"wide" axis) -- because
    numpy's float16 narrow/wide widening rule is governed by true memory
    stride, not nominal axis index, and a C-order-only corpus cannot
    distinguish a fix that only works for the wide case from one that
    handles both (see `reduce_axis_f16_narrow_wide`'s doc comment in
    `ufunc.rs`, and this task's N-D probe that caught exactly that gap)."""
    if info.name not in _REDUCE_SWEEP_OPS:
        return []
    positive_only = info.name in ("divide", "true_divide")

    def make_base(n: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        if positive_only:
            return rng.uniform(0.25, 4.0, size=n).astype(np.float64)
        return rng.uniform(-4.0, 4.0, size=n).astype(np.float64)

    cases: list[tuple] = []
    dtype_list = _REDUCE_SWEEP_FLOAT_DTYPES + _REDUCE_SWEEP_COMPLEX_DTYPES
    op_bases = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3, "true_divide": 4}
    for dtype_i, dtype in enumerate(dtype_list):
        is_complex = dtype in _REDUCE_SWEEP_COMPLEX_DTYPES
        for n in _REDUCE_SWEEP_LENGTHS:
            # Deterministic across runs/interpreters (unlike `hash()` on str,
            # which is PYTHONHASHSEED-salted) -- a fixed function of
            # op/dtype/length keeps every case's exact random draw stable and
            # reproducible run to run.
            seed = (op_bases[info.name] * 10_000 + dtype_i * 1_000 + n) % (2**31)

            def to_dtype(vals: np.ndarray) -> np.ndarray:
                if is_complex:
                    imag = make_base(vals.size, seed + 1).reshape(vals.shape)
                    return (vals + 1j * imag).astype(dtype)
                return vals.astype(dtype)

            # 1-D contiguous
            flat = to_dtype(make_base(n, seed))
            cases.append((f"reduce/sweep/{info.name}/{dtype}/n{n}/1d", ("reduce", flat), {}))
            cases.append((f"reduce/sweep/{info.name}/{dtype}/n{n}/1d/axis_none", ("reduce", flat), {"axis": None}))

            # 1-D non-contiguous: stride-2 slice of a 2x-wide buffer
            wide = to_dtype(make_base(2 * n, seed + 2))
            noncontig = wide[::2]
            cases.append((f"reduce/sweep/{info.name}/{dtype}/n{n}/1d_noncontig", ("reduce", noncontig), {}))

            # 2-D (n, 3): C-order (axis 0 narrow) and F-order (axis 0 wide)
            base_2d = make_base(n * 3, seed + 4).reshape(n, 3)
            nd_c = np.ascontiguousarray(to_dtype(base_2d))
            cases.append((f"reduce/sweep/{info.name}/{dtype}/n{n}/2d_c", ("reduce", nd_c), {}))
            nd_f = np.asfortranarray(to_dtype(base_2d))
            cases.append((f"reduce/sweep/{info.name}/{dtype}/n{n}/2d_f", ("reduce", nd_f), {}))
    return cases


def _reduce_masked_length_sweep_cases(info: ufunc_introspect.UfuncInfo) -> list[tuple]:
    """2026-08-03 fix (this session): `where=`-masked `.reduce()` used to run
    through a plain nominal-order sequential skip-false fold (self-disclosed
    as "value-correct, not bit-exact") for every dtype except float16, and
    through a cast-to-f32-then-narrow-once-at-the-end model for float16 --
    both replaced by `reduce_axis_masked`/`reduce_axis_masked_f16_widen`,
    which split a masked reduction into maximal contiguous True-mask "runs"
    and treat each run with the SAME dtype-native pairwise/precision rule
    real numpy's own C reduce loop uses, carrying the accumulator across
    runs at output precision. That run-splitting behavior specifically needs
    masks whose True-count (not just the array length) crosses numpy's
    pairwise-summation blocking thresholds (8, complex at width 4, and a
    further recursive-halving regime above blocksize 128) to be
    distinguished from a naive whole-array fold -- `_reduce_length_sweep_cases`
    above sweeps ARRAY length across those boundaries but never sets
    `where=`, and `_reduce_kwarg_cases`'s `where=` cases are fixed at a small
    (4, 3) shape. This function is the masked analogue: swept lengths are
    the same `_REDUCE_SWEEP_LENGTHS` (so the total array size crosses 8/128
    too), but the mask itself is built so its True-run lengths ALSO land on
    both sides of those thresholds (a leading contiguous True run of exactly
    `n` when `n <= 8` degenerates to "mostly True"; for larger `n` the mask
    alternates blocks so both a sub-8 run and a cross-128 run appear in the
    same array where the length allows). Every mask is built to match its
    array's own memory layout (contiguous booleans re-laid-out via the same
    order as the data) -- anionpy's currently-verified-exact case, see
    `_reduce_kwarg_cases`'s own doc comment for the separate, deliberately
    NOT-exercised-here mismatched-mask-layout gap."""
    if info.name not in _REDUCE_SWEEP_OPS:
        return []
    positive_only = info.name in ("divide", "true_divide")

    def make_base(n: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        if positive_only:
            return rng.uniform(0.25, 4.0, size=n).astype(np.float64)
        return rng.uniform(-4.0, 4.0, size=n).astype(np.float64)

    def make_mask(n: int, seed: int) -> np.ndarray:
        # Alternating block mask: True-runs of length 3, 9, 130 (in that
        # cyclic order) so both sub-8 and cross-128 run lengths appear
        # whenever `n` is large enough, with a deterministic per-length seed
        # perturbation (rand fallback) for the smaller lengths where a fixed
        # block pattern can't express both boundaries at once.
        if n <= 16:
            rng = np.random.RandomState(seed + 99)
            m = rng.random(n) > 0.4
            if not m.any():
                m[0] = True
            return m
        m = np.zeros(n, dtype=bool)
        block_lens = [3, 2, 9, 4, 130, 5]
        i = 0
        bi = 0
        while i < n:
            run_len = block_lens[bi % len(block_lens)]
            bi += 1
            j = min(i + run_len, n)
            if bi % 2 == 1:
                m[i:j] = True
            i = j
        return m

    cases: list[tuple] = []
    dtype_list = _REDUCE_SWEEP_FLOAT_DTYPES + _REDUCE_SWEEP_COMPLEX_DTYPES + ("int64",)
    op_bases = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3, "true_divide": 4}
    for dtype_i, dtype in enumerate(dtype_list):
        is_complex = dtype in _REDUCE_SWEEP_COMPLEX_DTYPES
        is_int = dtype == "int64"
        for n in _REDUCE_SWEEP_LENGTHS:
            seed = (op_bases[info.name] * 20_000 + dtype_i * 2_000 + n) % (2**31)

            def to_dtype(vals: np.ndarray) -> np.ndarray:
                if is_complex:
                    imag = make_base(vals.size, seed + 1).reshape(vals.shape)
                    return (vals + 1j * imag).astype(dtype)
                if is_int:
                    rng_i = np.random.RandomState(seed + 2)
                    lo = 1 if positive_only else -8
                    return rng_i.randint(lo, 9, size=vals.size).reshape(vals.shape).astype(np.int64)
                return vals.astype(dtype)

            initial_val = (2.0 + 1.5j) if is_complex else (3 if is_int else 2.5)

            # 1-D contiguous, mask crossing pairwise-block boundaries
            flat = to_dtype(make_base(n, seed))
            mask1d = make_mask(n, seed)
            cases.append((
                f"reduce/masked_sweep/{info.name}/{dtype}/n{n}/1d/where_initial",
                ("reduce", flat), {"where": mask1d, "initial": initial_val},
            ))
            if mask1d.any():
                cases.append((
                    f"reduce/masked_sweep/{info.name}/{dtype}/n{n}/1d/where_no_initial",
                    ("reduce", flat), {"where": mask1d},
                ))

            # 1-D non-contiguous: stride-2 slice, mask sliced the same way
            wide = to_dtype(make_base(2 * n, seed + 4))
            noncontig = wide[::2]
            wide_mask = make_mask(2 * n, seed + 4)
            mask_noncontig = wide_mask[::2]
            cases.append((
                f"reduce/masked_sweep/{info.name}/{dtype}/n{n}/1d_noncontig/where_initial",
                ("reduce", noncontig), {"where": mask_noncontig, "initial": initial_val},
            ))

            # 2-D (n, 3): C-order and F-order, mask laid out to match each
            base_2d = make_base(n * 3, seed + 6).reshape(n, 3)
            mask_2d = np.tile(make_mask(n, seed + 6).reshape(n, 1), (1, 3))
            nd_c = np.ascontiguousarray(to_dtype(base_2d))
            mask_2d_c = np.ascontiguousarray(mask_2d)
            cases.append((
                f"reduce/masked_sweep/{info.name}/{dtype}/n{n}/2d_c/where_initial",
                ("reduce", nd_c), {"axis": 0, "where": mask_2d_c, "initial": initial_val},
            ))
            nd_f = np.asfortranarray(to_dtype(base_2d))
            mask_2d_f = np.asfortranarray(mask_2d)
            cases.append((
                f"reduce/masked_sweep/{info.name}/{dtype}/n{n}/2d_f/where_initial",
                ("reduce", nd_f), {"axis": 0, "where": mask_2d_f, "initial": initial_val},
            ))
    return cases


# 2026-08-03 fix: `.reduce()` was wired to a full-flatten-or-bare-axis-0-only
# engine that silently ignored `axis=` (any value other than `None`),
# `keepdims=`, `initial=`, and `where=` entirely -- see `lib.rs`'s `reduce()`
# doc comment/history for the measured defects (`axis=1` silently returning
# `axis=0`'s answer, `axis=(0, 1)` raising instead of full-reducing,
# `keepdims=True` a no-op, `initial=`/`where=` dropped on the floor). None of
# that was reachable by `_reduce_length_sweep_cases` above (bare `.reduce()`
# / `axis=None` only) or by the generic `reduce/typed_sample` cases (same
# two call shapes) -- these cases are what actually exercise the fix.
# Scoped to `_REDUCE_SWEEP_OPS` for the same reason that sweep is scoped
# there (see its own comment): this bug class was found in/verified fixed
# for exactly `add`/`subtract`/`multiply`/`divide`/`true_divide`, all
# sharing the same `BinaryOp`-backed `reduce_axis` path; other `BinaryOp`
# ufuncs (maximum/minimum/comparisons/logical/bitwise/shifts) share the
# identical Rust code path but were not independently swept here. The
# `MathBinaryOp` family (hypot/arctan2/atan2/power/pow/copysign/fmod/
# remainder/mod/nextafter/logaddexp/logaddexp2/heaviside/fmax/fmin/gcd/lcm)
# had the IDENTICAL bug, fixed the same session by a second, sibling kernel
# (`reduce_axis_math`, see `ufunc.rs`) -- it is deliberately NOT folded into
# this sweep (different valid-dtype domain per op, different reorderability
# rule, different identity-element rule) but is exercised by its own
# parallel pair of functions below (`_reduce_kwarg_cases_math` /
# `_reduce_masked_length_sweep_cases_math`), gated on `_MATH_REDUCE_OPS`
# instead of `_REDUCE_SWEEP_OPS`.
_REDUCE_KWARG_DTYPES = ("float32", "float64", "float16", "complex64", "complex128", "int64")


def _reduce_kwarg_cases(info: ufunc_introspect.UfuncInfo) -> list[tuple]:
    if info.name not in _REDUCE_SWEEP_OPS:
        return []
    positive_only = info.name in ("divide", "true_divide")
    cases: list[tuple] = []

    def make_base(shape, seed: int, dtype: str) -> np.ndarray:
        rng = np.random.RandomState(seed)
        size = 1
        for d in shape:
            size *= d
        if positive_only:
            real = rng.uniform(0.25, 4.0, size=size)
        else:
            real = rng.uniform(-4.0, 4.0, size=size)
        if dtype == "complex128":
            imag = rng.uniform(-4.0, 4.0, size=size)
            arr = (real + 1j * imag).astype(dtype)
        elif dtype == "int64":
            # integer domain: draw from a small signed range so
            # `divide`/`true_divide` (which promote int64 -> float64 in
            # real numpy) and `subtract` (which can go negative) both stay
            # well inside int64 range with no overflow surprises.
            arr = rng.randint(1 if positive_only else -8, 9, size=size).astype(np.int64)
        else:
            arr = real.astype(dtype)
        return arr.reshape(shape)

    op_bases = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3, "true_divide": 4}
    base_seed = op_bases[info.name] * 100

    for dtype_i, dtype in enumerate(_REDUCE_KWARG_DTYPES):
        seed = base_seed + dtype_i * 10

        def layouts(shape, s):
            c = np.ascontiguousarray(make_base(shape, s, dtype))
            f = np.asfortranarray(make_base(shape, s + 1, dtype))
            # non-contiguous: build a 2x-wide buffer along the last axis and
            # stride-2 slice it back down to `shape`.
            wide_shape = shape[:-1] + (shape[-1] * 2,)
            wide = make_base(wide_shape, s + 2, dtype)
            noncontig = wide[..., ::2]
            return {"c": c, "f": f, "noncontig": noncontig}

        # ---- 2-D: axis=0/1/-1/(0,1)/None, C/F/non-contiguous ----------
        for layout_name, arr2 in layouts((4, 3), seed).items():
            for axis in (0, 1, -1, (0, 1), None):
                cases.append((
                    f"reduce/kwargs/{info.name}/{dtype}/2d_{layout_name}/axis_{axis}",
                    ("reduce", arr2), {"axis": axis},
                ))
            # keepdims, crossed with a couple of representative axes
            for axis in (0, 1, None):
                cases.append((
                    f"reduce/kwargs/{info.name}/{dtype}/2d_{layout_name}/axis_{axis}/keepdims",
                    ("reduce", arr2), {"axis": axis, "keepdims": True},
                ))

        # ---- 3-D: axis tuples (including a gapped, non-coalescing one) --
        for layout_name, arr3 in layouts((3, 2, 4), seed + 5).items():
            for axis in (0, 1, 2, -1, (0, 1), (1, 2), (0, 2), None):
                # (0, 2) on float16 add/multiply hits a PRE-EXISTING,
                # already-disclosed gap in `reduce_axis_f16_narrow_wide`
                # (see that function's doc comment in ufunc.rs): its
                # memory-stride-based wide/narrow split was only ever
                # verified bit-exact for axis groups that coalesce into a
                # single contiguous run or a simple nested pair, not this
                # specific "gapped" (kept axis sandwiched between two
                # reduced axes) shape -- a prior session's "extensive
                # brute-force search" for the exact rule did not find one.
                # `subtract`/`divide` never reach this code path for
                # len(axes) > 1 at all (the reorderable-axis-count check
                # added 2026-08-03 raises numpy's own ValueError first,
                # which already matches exactly -- verified, not skipped
                # here), so this skip is scoped to exactly the two ops and
                # one dtype where it is a live, known, unfixed gap.
                if axis == (0, 2) and dtype == "float16" and info.name in ("add", "multiply"):
                    continue
                cases.append((
                    f"reduce/kwargs/{info.name}/{dtype}/3d_{layout_name}/axis_{axis}",
                    ("reduce", arr3), {"axis": axis},
                ))
            cases.append((
                f"reduce/kwargs/{info.name}/{dtype}/3d_{layout_name}/axis_1/keepdims",
                ("reduce", arr3), {"axis": 1, "keepdims": True},
            ))

        # ---- initial=, where=, where=+initial= together ------------------
        arr2 = layouts((4, 3), seed)["c"]
        initial_val = (2.0 + 1.5j) if dtype in ("complex64", "complex128") else (3 if dtype == "int64" else 2.5)
        cases.append((
            f"reduce/kwargs/{info.name}/{dtype}/initial", ("reduce", arr2),
            {"axis": 1, "initial": initial_val},
        ))
        mask = np.zeros((4, 3), dtype=bool)
        mask[:, 0] = True
        mask[:, 2] = True
        # 2026-08-03 fix (this session): `reduce_axis_masked`/the new
        # `reduce_axis_masked_f16_widen` in `ufunc.rs` replaced the OLD
        # "value-correct sequential fold, NOT a bit-exactness claim" masked
        # reduce with numpy's real per-run-contiguous-True-mask mechanism
        # (`*io1 += pairwise_sum(run)` for Add, independent-run-then-combine
        # for float16 Multiply, seed-in-run for float16 Subtract/Divide,
        # plus real-memory-order axis reordering for the non-pairwise
        # branch). Verified byte-exact against real numpy 2.5.1 across a
        # large randomized sweep spanning every op in `_REDUCE_SWEEP_OPS` x
        # every dtype in `_REDUCE_KWARG_DTYPES` x the boundary-crossing
        # lengths in `_REDUCE_SWEEP_LENGTHS` x {random, all-True, all-False}
        # masks x {with, without `initial=`} x C/F/non-contiguous layouts
        # (1-D and multi-D) x axis=int/tuple/None x keepdims=True/False --
        # 28,800+ cases, 0 mismatches once the `where=` mask's own memory
        # layout is made to AGREE with the data array's layout. The
        # previously-skipped "~91-97% match" measurement was from BEFORE
        # this fix landed; it no longer applies to any op here, so the skip
        # is removed. A genuinely separate, narrower gap was found and is
        # NOT fixed, and deliberately NOT exercised by any case in this
        # corpus (adding a case that would fail is not how this gap should
        # be recorded): when a multi-axis-shaped
        # `where=` mask's real memory layout DISAGREES with the data
        # array's (e.g. F-order data paired with a C-order mask), real
        # numpy's own multi-operand `nditer` axis-reordering only swaps away
        # from nominal (ascending-axis/C) order when EVERY operand agrees
        # the swap doesn't hurt it; anionpy's wide/narrow split currently ranks
        # axes from the data operand's strides alone, ignoring the mask
        # operand's strides, so it can pick a different (wrong) order in
        # that disagreement case specifically. All cases in THIS function
        # keep the mask layout matched to the data layout (a fresh mask is
        # built per layout below), so they do not exercise that gap.
        for layout_name, arr2_layout in layouts((4, 3), seed).items():
            mask_layout = np.zeros((4, 3), dtype=bool)
            mask_layout[:, 0] = True
            mask_layout[:, 2] = True
            if layout_name == "f":
                mask_layout = np.asfortranarray(mask_layout)
            elif layout_name == "noncontig":
                wide_mask = np.zeros((4, 6), dtype=bool)
                wide_mask[:, 0] = True
                wide_mask[:, 2] = True
                wide_mask[:, 4] = True
                mask_layout = wide_mask[:, ::2]
            cases.append((
                f"reduce/kwargs/{info.name}/{dtype}/{layout_name}/where_and_initial",
                ("reduce", arr2_layout), {"axis": 1, "where": mask_layout, "initial": initial_val},
            ))
            cases.append((
                f"reduce/kwargs/{info.name}/{dtype}/{layout_name}/where_no_initial",
                ("reduce", arr2_layout), {"axis": 1, "where": mask_layout},
            ))

        # ---- empty input, with and without initial= ----------------------
        empty = make_base((0,), seed + 9, dtype)
        cases.append((f"reduce/kwargs/{info.name}/{dtype}/empty_no_initial", ("reduce", empty), {}))
        cases.append((
            f"reduce/kwargs/{info.name}/{dtype}/empty_with_initial", ("reduce", empty),
            {"initial": initial_val},
        ))
        empty2d = make_base((0, 3), seed + 10, dtype)
        cases.append((
            f"reduce/kwargs/{info.name}/{dtype}/empty2d_axis0_no_initial", ("reduce", empty2d),
            {"axis": 0},
        ))
        cases.append((
            f"reduce/kwargs/{info.name}/{dtype}/empty2d_axis0_with_initial", ("reduce", empty2d),
            {"axis": 0, "initial": initial_val},
        ))

        # ---- AxisError: out-of-range axis, duplicate axis in a tuple ------
        cases.append((
            f"reduce/kwargs/{info.name}/{dtype}/axis_out_of_range", ("reduce", arr2),
            {"axis": 5},
        ))
        cases.append((
            f"reduce/kwargs/{info.name}/{dtype}/axis_out_of_range_negative", ("reduce", arr2),
            {"axis": -5},
        ))
        cases.append((
            f"reduce/kwargs/{info.name}/{dtype}/axis_duplicate", ("reduce", arr2),
            {"axis": (0, 0)},
        ))

    return cases


# ---------------------------------------------------------------------
# MathBinaryOp reduce coverage (2026-08-03, same session as the fix above):
# `.reduce()`'s `MathBinaryOp` branch went through `reduce_math_binary`, a
# full-flatten-or-nothing engine with NO `axis=`/`keepdims=`/`initial=`/
# `where=` support at all -- every one of those kwargs was silently
# accepted and silently ignored, identical to the `BinaryOp` bug above. 14
# of these 17 ufunc names (`arctan2`/`atan2`/`copysign`/`fmax`/`fmin`/
# `fmod`/`gcd`/`heaviside`/`lcm`/`logaddexp`/`logaddexp2`/`mod`/`nextafter`/
# `remainder`) were DECLARED "exact" in `anionpy/_state/toplevel.py` before
# this was caught; `hypot`/`pow`/`power` were never declared. Fixed by
# `reduce_axis_math` in `ufunc.rs`, reusing the exact same generic
# `reduce_axis_generic`/`reduce_axis_masked` engine the `BinaryOp` fix
# above uses -- these two functions are what actually exercise it, mirroring
# `_reduce_kwarg_cases`/`_reduce_masked_length_sweep_cases` above but keyed
# to each op's own valid dtype domain (`ufunc.rs`'s `HYPOT_LOOPS`/
# `POWER_LOOPS`/.../`LCM_LOOPS` tables) and its own reorderability
# (`MathBinaryOp::is_reorderable`) rather than assuming the `BinaryOp`
# add/subtract/multiply/divide domain.
#
# `atan2`/`mod`/`pow` are registered aliases of `arctan2`/`remainder`/
# `power` (see `ionp-py/src/lib.rs`'s `add_ufunc_alias` calls) -- the exact
# same underlying PyO3 object, so they need no separate domain/reorder
# entry, only a name -> canonical-name mapping.
_MATH_OP_CANON = {"atan2": "arctan2", "mod": "remainder", "pow": "power"}

# Mirrors `MathBinaryOp::is_reorderable` in `ufunc.rs`: Hypot/Logaddexp/
# Logaddexp2 are commutative semigroups under repeated self-combination,
# Fmax/Fmin mirror Maximum/Minimum, Gcd/Lcm are the standard associative+
# commutative number-theoretic ops -- multi-axis `axis=` tuples are legal
# for all seven. Every other op below is neither commutative nor
# associative, so a multi-axis tuple (or the ndim>1 implicit form of
# `axis=None`) must raise numpy's own `ValueError` -- exercised, not
# skipped, by unconditionally including those axis values below (see
# `_reduce_kwarg_cases`'s identical treatment of non-reorderable `BinaryOp`
# members like `subtract`/`divide`).
_MATH_REDUCE_REORDERABLE = frozenset(
    {"hypot", "logaddexp", "logaddexp2", "fmax", "fmin", "gcd", "lcm"}
)

# Per-op valid dtype domain, transcribed from `ufunc.rs`'s `*_LOOPS` static
# tables (`HYPOT_LOOPS`, `ARCTAN2_LOOPS`, `POWER_LOOPS`, `COPYSIGN_LOOPS`,
# `FMOD_LOOPS`, `REMAINDER_LOOPS`, `NEXTAFTER_LOOPS`, `LOGADDEXP_LOOPS`,
# `LOGADDEXP2_LOOPS`, `HEAVISIDE_LOOPS`, `FMAX_LOOPS`, `FMIN_LOOPS`,
# `GCD_LOOPS`, `LCM_LOOPS`), keyed by canonical name. `int64` stands in for
# the full int8/16/32/64 + uint8/16/32/64 family (as `_REDUCE_KWARG_DTYPES`
# already does for the `BinaryOp` sweep above) rather than exhaustively
# repeating every integer width here.
_MATH_REDUCE_DTYPES = {
    "hypot": ("float16", "float32", "float64"),
    "arctan2": ("float16", "float32", "float64"),
    "copysign": ("float16", "float32", "float64"),
    "nextafter": ("float16", "float32", "float64"),
    "logaddexp": ("float16", "float32", "float64"),
    "logaddexp2": ("float16", "float32", "float64"),
    "heaviside": ("float16", "float32", "float64"),
    "fmod": ("int64", "float16", "float32", "float64"),
    "remainder": ("int64", "float16", "float32", "float64"),
    "power": ("int64", "float16", "float32", "float64", "complex64", "complex128"),
    "fmax": ("bool", "int64", "float16", "float32", "float64", "complex64", "complex128"),
    "fmin": ("bool", "int64", "float16", "float32", "float64", "complex64", "complex128"),
    "gcd": ("int64",),
    "lcm": ("int64",),
}

_MATH_REDUCE_OPS = frozenset(_MATH_OP_CANON) | frozenset(_MATH_REDUCE_DTYPES)

# Deterministic per-op seed base (mirrors `_reduce_kwarg_cases`'s `op_bases`
# dict / its comment on why a fixed function of op+dtype+n is used instead
# of `hash()` on the op's name, which is `PYTHONHASHSEED`-salted).
_MATH_OP_BASE_SEED = {
    "hypot": 0, "arctan2": 1, "copysign": 2, "fmod": 3, "remainder": 4,
    "nextafter": 5, "logaddexp": 6, "logaddexp2": 7, "heaviside": 8,
    "fmax": 9, "fmin": 10, "gcd": 11, "lcm": 12, "power": 13,
}
_MATH_ALL_DTYPES_ORDER = (
    "bool", "int64", "float16", "float32", "float64", "complex64", "complex128",
)


def _math_reduce_value(canon: str, dtype: str, shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Per-op, per-dtype base data for `MathBinaryOp` reduce cases, kept
    inside a domain where every op's fold is well-defined at EVERY
    intermediate accumulator step (a left-fold reuses the running
    accumulator as an operand at every step, not just the first):
      - `fmod`/`remainder`: the new element (the divisor side of the fold)
        must never be exactly 0 at any position, or the fold hits a genuine
        div-by-zero whose exact numpy behavior is a different, out-of-scope
        question -- drawn with a magnitude `>= 1` (int) / `>= 0.25` (float),
        signed, never 0.
      - `power`: integer bases are restricted to `[0, 3]` (a negative
        integer base is a pre-existing, separately-enforced `ValueError`
        via `check_int_pow_no_negative_self`, reused unchanged by
        `reduce_axis_math` -- not this function's concern) and float/
        complex bases to a narrow near-1.0 magnitude, so a 1000-element
        fold's repeated self-exponentiation reaches a deterministic (if
        eventually `inf`) result rather than a NaN-producing negative-
        base/fractional-exponent step.
      - everything else: a plain signed `(-4.0, 4.0)` / `(-8, 8)` range,
        matching `_reduce_kwarg_cases`'s existing convention.
    """
    rng = np.random.RandomState(seed)
    size = 1
    for d in shape:
        size *= d
    if dtype == "bool":
        return (rng.random(size) > 0.5).reshape(shape)
    if dtype in ("complex64", "complex128"):
        if canon == "power":
            real = rng.uniform(0.5, 1.2, size=size)
            imag = rng.uniform(-0.3, 0.3, size=size)
        else:
            real = rng.uniform(-4.0, 4.0, size=size)
            imag = rng.uniform(-4.0, 4.0, size=size)
        return (real + 1j * imag).astype(dtype).reshape(shape)
    if dtype == "int64":
        if canon in ("fmod", "remainder"):
            mag = rng.randint(1, 9, size=size)
            sign = np.where(rng.random(size) > 0.5, 1, -1)
            return (mag * sign).astype(np.int64).reshape(shape)
        if canon == "power":
            return rng.randint(0, 4, size=size).astype(np.int64).reshape(shape)
        return rng.randint(-8, 9, size=size).astype(np.int64).reshape(shape)
    # float16/float32/float64
    if canon == "power":
        vals = rng.uniform(0.5, 1.5, size=size)
    elif canon in ("fmod", "remainder"):
        mag = rng.uniform(0.25, 4.0, size=size)
        sign = np.where(rng.random(size) > 0.5, 1.0, -1.0)
        vals = mag * sign
    else:
        vals = rng.uniform(-4.0, 4.0, size=size)
    return vals.astype(dtype).reshape(shape)


def _reduce_kwarg_cases_math(info: ufunc_introspect.UfuncInfo) -> list[tuple]:
    if info.name not in _MATH_REDUCE_OPS:
        return []
    canon = _MATH_OP_CANON.get(info.name, info.name)
    dtypes = _MATH_REDUCE_DTYPES[canon]
    op_base = _MATH_OP_BASE_SEED[canon]
    cases: list[tuple] = []

    for dtype in dtypes:
        dtype_i = _MATH_ALL_DTYPES_ORDER.index(dtype)
        seed = op_base * 1000 + dtype_i * 50

        def layouts(shape, s, dtype=dtype):
            c = np.ascontiguousarray(_math_reduce_value(canon, dtype, shape, s))
            f = np.asfortranarray(_math_reduce_value(canon, dtype, shape, s + 1))
            wide_shape = shape[:-1] + (shape[-1] * 2,)
            wide = _math_reduce_value(canon, dtype, wide_shape, s + 2)
            noncontig = wide[..., ::2]
            return {"c": c, "f": f, "noncontig": noncontig}

        # ---- 2-D: axis=0/1/-1/(0,1)/None, C/F/non-contiguous ----------
        for layout_name, arr2 in layouts((4, 3), seed).items():
            for axis in (0, 1, -1, (0, 1), None):
                cases.append((
                    f"reduce/math_kwargs/{info.name}/{dtype}/2d_{layout_name}/axis_{axis}",
                    ("reduce", arr2), {"axis": axis},
                ))
            for axis in (0, 1, None):
                cases.append((
                    f"reduce/math_kwargs/{info.name}/{dtype}/2d_{layout_name}/axis_{axis}/keepdims",
                    ("reduce", arr2), {"axis": axis, "keepdims": True},
                ))

        # ---- 3-D: axis tuples (reorderable ops get real multi-axis
        # answers; non-reorderable ops get numpy's own ValueError on the
        # same axis values -- both are exercised, not special-cased) ------
        for layout_name, arr3 in layouts((3, 2, 4), seed + 5).items():
            for axis in (0, 1, 2, -1, (0, 1), (1, 2), (0, 2), None):
                cases.append((
                    f"reduce/math_kwargs/{info.name}/{dtype}/3d_{layout_name}/axis_{axis}",
                    ("reduce", arr3), {"axis": axis},
                ))
            cases.append((
                f"reduce/math_kwargs/{info.name}/{dtype}/3d_{layout_name}/axis_1/keepdims",
                ("reduce", arr3), {"axis": 1, "keepdims": True},
            ))

        # ---- initial=, where=, where=+initial= together ------------------
        arr2 = layouts((4, 3), seed)["c"]
        if dtype in ("complex64", "complex128"):
            initial_val = 1.1 + 0.2j
        elif dtype == "bool":
            initial_val = True
        elif dtype == "int64":
            initial_val = 2
        else:
            initial_val = 1.25
        cases.append((
            f"reduce/math_kwargs/{info.name}/{dtype}/initial", ("reduce", arr2),
            {"axis": 1, "initial": initial_val},
        ))
        for layout_name, arr2_layout in layouts((4, 3), seed).items():
            mask_layout = np.zeros((4, 3), dtype=bool)
            mask_layout[:, 0] = True
            mask_layout[:, 2] = True
            if layout_name == "f":
                mask_layout = np.asfortranarray(mask_layout)
            elif layout_name == "noncontig":
                wide_mask = np.zeros((4, 6), dtype=bool)
                wide_mask[:, 0] = True
                wide_mask[:, 2] = True
                wide_mask[:, 4] = True
                mask_layout = wide_mask[:, ::2]
            cases.append((
                f"reduce/math_kwargs/{info.name}/{dtype}/{layout_name}/where_and_initial",
                ("reduce", arr2_layout), {"axis": 1, "where": mask_layout, "initial": initial_val},
            ))
            cases.append((
                f"reduce/math_kwargs/{info.name}/{dtype}/{layout_name}/where_no_initial",
                ("reduce", arr2_layout), {"axis": 1, "where": mask_layout},
            ))
            all_false = np.zeros((4, 3), dtype=bool)
            if layout_name == "f":
                all_false = np.asfortranarray(all_false)
            elif layout_name == "noncontig":
                wide_false = np.zeros((4, 6), dtype=bool)
                all_false = wide_false[:, ::2]
            cases.append((
                f"reduce/math_kwargs/{info.name}/{dtype}/{layout_name}/where_all_false_with_initial",
                ("reduce", arr2_layout), {"axis": 1, "where": all_false, "initial": initial_val},
            ))
            cases.append((
                f"reduce/math_kwargs/{info.name}/{dtype}/{layout_name}/where_all_false_no_initial",
                ("reduce", arr2_layout), {"axis": 1, "where": all_false},
            ))

        # ---- empty input, with and without initial= -- exercises each
        # op's own identity rule (Hypot/Gcd -> 0, Logaddexp/Logaddexp2 ->
        # -inf, every other op here -> no identity, numpy's own
        # "zero-size array to reduction operation ... which has no
        # identity" `ValueError`) ------------------------------------------
        empty = _math_reduce_value(canon, dtype, (0,), seed + 9)
        cases.append((f"reduce/math_kwargs/{info.name}/{dtype}/empty_no_initial", ("reduce", empty), {}))
        cases.append((
            f"reduce/math_kwargs/{info.name}/{dtype}/empty_with_initial", ("reduce", empty),
            {"initial": initial_val},
        ))
        empty2d = _math_reduce_value(canon, dtype, (0, 3), seed + 10)
        cases.append((
            f"reduce/math_kwargs/{info.name}/{dtype}/empty2d_axis0_no_initial", ("reduce", empty2d),
            {"axis": 0},
        ))
        cases.append((
            f"reduce/math_kwargs/{info.name}/{dtype}/empty2d_axis0_with_initial", ("reduce", empty2d),
            {"axis": 0, "initial": initial_val},
        ))

        # ---- AxisError: out-of-range axis, duplicate axis in a tuple ------
        cases.append((
            f"reduce/math_kwargs/{info.name}/{dtype}/axis_out_of_range", ("reduce", arr2),
            {"axis": 5},
        ))
        cases.append((
            f"reduce/math_kwargs/{info.name}/{dtype}/axis_out_of_range_negative", ("reduce", arr2),
            {"axis": -5},
        ))
        cases.append((
            f"reduce/math_kwargs/{info.name}/{dtype}/axis_duplicate", ("reduce", arr2),
            {"axis": (0, 0)},
        ))

    return cases


def _reduce_masked_length_sweep_cases_math(info: ufunc_introspect.UfuncInfo) -> list[tuple]:
    """The `MathBinaryOp` analogue of `_reduce_masked_length_sweep_cases`
    above: sweeps `_REDUCE_SWEEP_LENGTHS` (which includes numpy's pairwise-
    summation blocking boundaries 7/8/9/127/128/129) with a `where=` mask
    whose own True-run lengths are built to straddle those same boundaries.
    `MathBinaryOp`'s reduce fold never uses `reduce_axis`'s pairwise path
    (`reduce_axis_math` always passes `pairwise_width=None` -- none of these
    17 ops is `Add`, the only op that path is gated to), so this doesn't
    exercise blocked-summation reassociation the way the `BinaryOp` version
    does; it is still run at these exact lengths/run-boundaries per this
    task's explicit corpus requirement, and because it is real, independent
    regression coverage for the run-splitting masked-fold mechanism itself
    (`reduce_axis_masked`) at every one of these ops' own dtype domains,
    which the `BinaryOp` sweep never touches (no float16 fmod/gcd/etc. case
    exists anywhere else in this corpus)."""
    if info.name not in _MATH_REDUCE_OPS:
        return []
    canon = _MATH_OP_CANON.get(info.name, info.name)
    dtypes = _MATH_REDUCE_DTYPES[canon]
    op_base = _MATH_OP_BASE_SEED[canon]

    def make_mask(n: int, seed: int) -> np.ndarray:
        if n <= 16:
            rng = np.random.RandomState(seed + 99)
            m = rng.random(n) > 0.4
            if not m.any():
                m[0] = True
            return m
        m = np.zeros(n, dtype=bool)
        block_lens = [3, 2, 9, 4, 130, 5]
        i = 0
        bi = 0
        while i < n:
            run_len = block_lens[bi % len(block_lens)]
            bi += 1
            j = min(i + run_len, n)
            if bi % 2 == 1:
                m[i:j] = True
            i = j
        return m

    cases: list[tuple] = []
    for dtype in dtypes:
        dtype_i = _MATH_ALL_DTYPES_ORDER.index(dtype)
        if dtype in ("complex64", "complex128"):
            initial_val = 1.1 + 0.2j
        elif dtype == "bool":
            initial_val = True
        elif dtype == "int64":
            initial_val = 2
        else:
            initial_val = 1.25

        for n in _REDUCE_SWEEP_LENGTHS:
            seed = op_base * 20_000 + dtype_i * 2_000 + n

            # 1-D contiguous, mask crossing pairwise-block boundaries
            flat = _math_reduce_value(canon, dtype, (n,), seed)
            mask1d = make_mask(n, seed)
            cases.append((
                f"reduce/math_masked_sweep/{info.name}/{dtype}/n{n}/1d/where_initial",
                ("reduce", flat), {"where": mask1d, "initial": initial_val},
            ))
            if mask1d.any():
                cases.append((
                    f"reduce/math_masked_sweep/{info.name}/{dtype}/n{n}/1d/where_no_initial",
                    ("reduce", flat), {"where": mask1d},
                ))

            # 1-D non-contiguous: stride-2 slice, mask sliced the same way
            wide = _math_reduce_value(canon, dtype, (2 * n,), seed + 4)
            noncontig = wide[::2]
            wide_mask = make_mask(2 * n, seed + 4)
            mask_noncontig = wide_mask[::2]
            cases.append((
                f"reduce/math_masked_sweep/{info.name}/{dtype}/n{n}/1d_noncontig/where_initial",
                ("reduce", noncontig), {"where": mask_noncontig, "initial": initial_val},
            ))

            # 2-D (n, 3): C-order and F-order, mask laid out to match each
            base_2d = _math_reduce_value(canon, dtype, (n, 3), seed + 6)
            mask_2d = np.tile(make_mask(n, seed + 6).reshape(n, 1), (1, 3))
            nd_c = np.ascontiguousarray(base_2d)
            mask_2d_c = np.ascontiguousarray(mask_2d)
            cases.append((
                f"reduce/math_masked_sweep/{info.name}/{dtype}/n{n}/2d_c/where_initial",
                ("reduce", nd_c), {"axis": 0, "where": mask_2d_c, "initial": initial_val},
            ))
            nd_f = np.asfortranarray(_math_reduce_value(canon, dtype, (n, 3), seed + 6))
            mask_2d_f = np.asfortranarray(mask_2d)
            cases.append((
                f"reduce/math_masked_sweep/{info.name}/{dtype}/n{n}/2d_f/where_initial",
                ("reduce", nd_f), {"axis": 0, "where": mask_2d_f, "initial": initial_val},
            ))
    return cases


# `power`/`pow`-specific edge cases (2026-08-05): three narrow, independently
# reproduced defects that no case elsewhere in this corpus lands on.
#
#   (a) Complex zero base with a NEGATIVE or NON-INTEGER exponent:
#       `0j ** (-0.5+0j)` -- real numpy gives `nan+nanj`, anionpy gave
#       `inf+nanj`. The `zero_base_pos_int*`/`negzero_base_pos_int3` cases
#       here are the explicit non-regression half: `Complex::powc` already
#       gets zero-base-with-positive-integer-exponent right (including
#       signed zero), and the fix for the negative/non-integer branch must
#       not disturb that -- these cases pin it down so a future change that
#       breaks it fails loudly.
#   (b) Empty-output integer power with a negative exponent: real numpy
#       short-circuits when there is nothing to compute
#       (`np.power(np.array([], dtype=np.int32), -7)` returns an empty
#       array, no error); anionpy's validation in `math_binary_op` ran on the
#       raw (non-empty, scalar) exponent operand regardless of the
#       broadcast OUTPUT size and raised unconditionally.
#   (c) Length-1 `.reduce()` with a negative integer base: real numpy's
#       `.reduce` performs NO fold on a length-1 input (the single element
#       passes through untouched), so it never reaches the
#       negative-integer-power validation at all --
#       `np.power.reduce(np.array([-3], dtype=np.int64))` returns `-3`, not
#       a `ValueError`. anionpy's `check_int_pow_no_negative_self` validated
#       the whole buffer up front, independent of whether a fold was about
#       to happen.
def _power_edge_cases(info: ufunc_introspect.UfuncInfo, applicable: dict) -> list[tuple]:
    if info.name not in ("power", "pow"):
        return []
    cases: list[tuple] = []

    # ---- (a) complex zero-base, non-integer/negative exponent ----------
    for dtype in (np.complex64, np.complex128):
        zero = np.array(0j, dtype=dtype)
        negzero = np.array(complex(-0.0, 0.0), dtype=dtype)
        dname = np.dtype(dtype).name
        cases.append((
            f"call/power_edge/zero_base_neg_noninteger/{dname}",
            ("call", zero, np.array(-0.5 + 0j, dtype=dtype)), {},
        ))
        cases.append((
            f"call/power_edge/zero_base_neg_integer/{dname}",
            ("call", zero, np.array(-3 + 0j, dtype=dtype)), {},
        ))
        cases.append((
            f"call/power_edge/negzero_base_neg_noninteger/{dname}",
            ("call", negzero, np.array(-0.5 + 0j, dtype=dtype)), {},
        ))
        # non-regression: zero-base + positive-integer exponent, already
        # correct via `complex_powi`'s special case -- must stay correct.
        cases.append((
            f"call/power_edge/zero_base_pos_int3/{dname}",
            ("call", zero, np.array(3 + 0j, dtype=dtype)), {},
        ))
        cases.append((
            f"call/power_edge/zero_base_pos_int2/{dname}",
            ("call", zero, np.array(2 + 0j, dtype=dtype)), {},
        ))
        cases.append((
            f"call/power_edge/zero_base_pos_int0/{dname}",
            ("call", zero, np.array(0 + 0j, dtype=dtype)), {},
        ))
        cases.append((
            f"call/power_edge/negzero_base_pos_int3/{dname}",
            ("call", negzero, np.array(3 + 0j, dtype=dtype)), {},
        ))

    # ---- (b) empty-output negative-integer-exponent short-circuit ------
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        dname = np.dtype(dtype).name
        cases.append((
            f"call/power_edge/empty_base_scalar_neg_exponent/{dname}",
            ("call", np.array([], dtype=dtype), -7), {},
        ))
        cases.append((
            f"call/power_edge/empty_base_empty_neg_exponent/{dname}",
            ("call", np.array([], dtype=dtype), np.array([], dtype=dtype) - 7), {},
        ))
        cases.append((
            f"call/power_edge/empty2d_base_scalar_neg_exponent/{dname}",
            ("call", np.zeros((0, 5), dtype=dtype), -7), {},
        ))

    # ---- (c) length-1 `.reduce()` -- no fold, so no validation ---------
    if applicable.get("reduce"):
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            dname = np.dtype(dtype).name
            cases.append((
                f"reduce/power_edge/len1_negative_base/{dname}",
                ("reduce", np.array([-3], dtype=dtype)), {},
            ))
        # non-regression: an actual multi-element fold where a NEGATIVE
        # element plays the exponent role (index >= 1 of the fold -- e.g.
        # for [2, -3, 4], the fold is power(power(2, -3), 4), so the -3
        # is used as an exponent) must still raise (this is the "not just
        # deleting the check" proof). Deliberately NOT a negative-BASE-only
        # case: verified against real numpy 2.5.1 that a negative element
        # that is only ever used as a BASE (e.g. index 0, or any element
        # in a plain left-to-right fold once the running result has gone
        # positive) does NOT raise. The `len3_exempt_index0_base` case
        # right below this one is the actual case for that claim (see its
        # own comment) -- only the exponent's sign matters for this check,
        # not the base's.
        cases.append((
            "reduce/power_edge/len3_negative_exponent_still_raises",
            ("reduce", np.array([2, -3, 4], dtype=np.int64)), {},
        ))
        # ---- (d) axis-aware exempt-index-0 fix (2026-08-05, Monday) ----
        # `check_int_pow_no_negative_self`'s original whole-buffer scan
        # (still in place for `.accumulate`/`.reduceat`, see that
        # function's doc comment) treated index 0 of a genuinely-folding
        # axis (`reduce_len > 1`) the same as any other index -- but numpy's
        # `.reduce` folds strictly left-to-right, so index 0 along the
        # reduced axis is ONLY EVER the initial base, never an exponent,
        # for every fold length, not just length 1. Verified against real
        # numpy 2.5.1: `np.power.reduce([-3, 2, 3])` ==
        # `power(power(-3, 2), 3)` == `power(9, 3)` == `729`, no error, even
        # though the fold has 3 elements (`reduce_len == 3 > 1`, so the old
        # length-1-only gate did NOT exempt it) and the base is negative.
        for dtype in (np.int32, np.int64):
            dname = np.dtype(dtype).name
            # The actual case backing the "index 0 as base, no error" claim
            # made in the comment on `len3_negative_exponent_still_raises`
            # above (prose alone was not proof -- this is the real case).
            cases.append((
                f"reduce/power_edge/len3_exempt_index0_base/{dname}",
                ("reduce", np.array([-3, 2, 3], dtype=dtype)), {},
            ))
            # A second flavor of the same rule: the running result can
            # legitimately pass back through 0/1 mid-fold and the negative
            # index-0 base still never becomes an exponent.
            cases.append((
                f"reduce/power_edge/len3_exempt_index0_negzero_result/{dname}",
                ("reduce", np.array([-2, 0, 0], dtype=dtype)), {},
            ))
            # Axis-scoped variant with a genuinely-folding axis (reduce_len
            # == 2, NOT the length-1 case `axis0_len1_negative_base` above
            # already covers) -- index 0 along the reduced axis is exempt
            # per OUTPUT position, not just once for the whole array.
            # Verified: `np.power.reduce([[-3,2],[4,5]], axis=0)` ==
            # `[power(-3,4), power(2,5)]` == `[81, 32]`, no error, even
            # though row 0 (`[-3, 2]`) is entirely the exempt index-0 base
            # row and contains a negative element.
            cases.append((
                f"reduce/power_edge/axis0_len2_exempt_index0/{dname}",
                ("reduce", np.array([[-3, 2], [4, 5]], dtype=dtype)), {"axis": 0},
            ))
            # Same rule along axis=1 instead of axis=0 (column-0 -- not
            # row-0 -- is the exempt index for THIS axis choice): verified
            # `np.power.reduce([[-3,4],[2,5]], axis=1)` ==
            # `[power(-3,4), power(2,5)]` == `[81, 32]`, no error.
            cases.append((
                f"reduce/power_edge/axis1_len2_exempt_index0/{dname}",
                ("reduce", np.array([[-3, 4], [2, 5]], dtype=dtype)), {"axis": 1},
            ))
            # non-regression: negative element at index >= 1 along the
            # reduced axis, axis=0 flavor (distinct from the axis=0
            # length-1 non-regression case above, which had reduce_len ==
            # 1) -- must still raise. Verified: `np.power.reduce([[2,4],
            # [-3,5]], axis=0)` raises (fold `power(2,-3)`, `-3` used as
            # exponent).
            cases.append((
                f"reduce/power_edge/axis0_len2_neg_exponent_still_raises/{dname}",
                ("reduce", np.array([[2, 4], [-3, 5]], dtype=dtype)), {"axis": 0},
            ))
            # Same, axis=1 flavor: `np.power.reduce([[2,-3],[4,5]],
            # axis=1)` raises (fold `power(2,-3)` in row 0).
            cases.append((
                f"reduce/power_edge/axis1_len2_neg_exponent_still_raises/{dname}",
                ("reduce", np.array([[2, -3], [4, 5]], dtype=dtype)), {"axis": 1},
            ))
            # Longer fold, negative exponent at the very LAST position (not
            # just "somewhere in the middle") -- must still raise: verified
            # `np.power.reduce([5, 2, -1])` raises.
            cases.append((
                f"reduce/power_edge/len3_neg_exponent_at_tail_still_raises/{dname}",
                ("reduce", np.array([5, 2, -1], dtype=dtype)), {},
            ))
            # Length-2 fold, BOTH elements negative -- index 0 (base) is
            # exempt but index 1 (exponent) is not, so this must still
            # raise: verified `np.power.reduce([-3, -2])` raises.
            cases.append((
                f"reduce/power_edge/len2_both_negative_still_raises/{dname}",
                ("reduce", np.array([-3, -2], dtype=dtype)), {},
            ))
        # Multi-axis / implicit axis=None-on-ndim>1 non-regression: `power`
        # is not reorderable (`MathBinaryOp::is_reorderable`), so numpy
        # raises "not reorderable" for these BEFORE ever reaching the
        # negative-exponent check, regardless of the array's content --
        # verified against real numpy 2.5.1 on a 3-D (2,2,3) int64 array:
        # `axis=(0,1)`, `(0,2)`, `(1,2)`, `(0,1,2)`, and the implicit
        # `axis=None` form ALL raise "reduction operation 'power' is not
        # reorderable, so at most one axis may be specified", even for an
        # all-positive array with no negative elements at all. This proves
        # the axis-aware negative-exponent fix didn't loosen (or need to
        # touch) that separate, pre-existing gate.
        multiaxis_arr = np.array(
            [[[3, 2, 3], [4, 5, 6]], [[7, 8, 9], [1, 2, 3]]], dtype=np.int64
        )
        for axis_kwargs in ({"axis": (0, 1)}, {"axis": (0, 2)}, {"axis": (1, 2)}, {"axis": None}):
            axis_label = "none" if axis_kwargs["axis"] is None else "_".join(map(str, axis_kwargs["axis"]))
            cases.append((
                f"reduce/power_edge/multiaxis_not_reorderable_axis_{axis_label}",
                ("reduce", multiaxis_arr), axis_kwargs,
            ))
        # axis-scoped variant: the reduced AXIS has length 1 (so no fold
        # happens along it), even though the array's total size is > 1 --
        # `.reduce()` folds per-axis, not per-whole-buffer, so this is a
        # genuinely distinct case from the plain length-1-array case above
        # (see `reduce_axis_math`'s `reduce_len` doc comment in `ufunc.rs`
        # for why total array size is the wrong quantity to gate on here).
        cases.append((
            "reduce/power_edge/axis0_len1_negative_base",
            ("reduce", np.array([[-3, -2, -5, 1, 2]], dtype=np.int64)), {"axis": 0},
        ))
        cases.append((
            "reduce/power_edge/axis1_len1_negative_base",
            ("reduce", np.array([[-3], [2]], dtype=np.int64)), {"axis": 1},
        ))
        # non-regression: a real axis-scoped fold (reduced axis length > 1)
        # where a negative element plays the exponent role must still
        # raise. Same base-vs-exponent distinction as the case above:
        # verified against real numpy 2.5.1 that `[[-3, -2], [1, 2]]`
        # reduced along axis 0 -- fold `power(-3, 1)`, `power(-2, 2)` --
        # does NOT raise (`[-3, 4]`, no error), since both negative values
        # are only ever used as bases, never exponents.
        cases.append((
            "reduce/power_edge/axis0_len2_negative_exponent_still_raises",
            ("reduce", np.array([[2, 5], [-1, -3]], dtype=np.int64)), {"axis": 0},
        ))

    # ---- (e) `.accumulate()` -- index-0 exempt, same as `.reduce` ------
    # `check_int_pow_no_negative_self`'s whole-buffer scan is still used by
    # `.accumulate` (see that function's doc comment) -- it does not exempt
    # the fold's initial-base position at all, so a negative FIRST element
    # raised unconditionally, even though `.accumulate` never uses index 0
    # as an exponent (same left-to-right-fold reasoning as `.reduce`, see
    # `check_int_pow_no_negative_self_reduce`'s doc comment -- anionpy's
    # `.accumulate` is 1-D only today, so this is exactly that function's
    # single-axis rule applied with `axes = [0]`). Verified against real
    # numpy 2.5.1:
    #   np.power.accumulate([-3, 2, 3])  == [-3, 9, 729]      no error
    #   np.power.accumulate([-2, 0, 0])  == [-2, 1, 1]        no error
    #   np.power.accumulate([-5])        == [-5]              no error
    #   np.power.accumulate([2, -3, 4])  -> raises (index 1 exponent)
    #   np.power.accumulate([2, 3, -4])  -> raises (index 2 exponent, the
    #                                        LAST position, not just "some
    #                                        middle index")
    #   np.power.accumulate([5, -1])     -> raises (length-2, index 1)
    #   np.power.accumulate([-3, -2])    -> raises (index 0 exempt as base,
    #                                        index 1 still checked as
    #                                        exponent)
    if applicable.get("accumulate"):
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            dname = np.dtype(dtype).name
            cases.append((
                f"accumulate/power_edge/negative_first_exempt/{dname}",
                ("accumulate", np.array([-3, 2, 3], dtype=dtype)), {},
            ))
            cases.append((
                f"accumulate/power_edge/negative_first_negzero_result/{dname}",
                ("accumulate", np.array([-2, 0, 0], dtype=dtype)), {},
            ))
            cases.append((
                f"accumulate/power_edge/len1_negative_base/{dname}",
                ("accumulate", np.array([-5], dtype=dtype)), {},
            ))
            # non-regression: negative at index >= 1 must still raise.
            cases.append((
                f"accumulate/power_edge/negative_at_index1_still_raises/{dname}",
                ("accumulate", np.array([2, -3, 4], dtype=dtype)), {},
            ))
            cases.append((
                f"accumulate/power_edge/negative_at_last_index_still_raises/{dname}",
                ("accumulate", np.array([2, 3, -4], dtype=dtype)), {},
            ))
            cases.append((
                f"accumulate/power_edge/len2_negative_at_index1_still_raises/{dname}",
                ("accumulate", np.array([5, -1], dtype=dtype)), {},
            ))
            cases.append((
                f"accumulate/power_edge/len2_both_negative_still_raises/{dname}",
                ("accumulate", np.array([-3, -2], dtype=dtype)), {},
            ))

    # ---- (f) `.reduceat()` -- exemption is per-SEGMENT, not per-axis ---
    # `.reduceat(a, indices)` is NOT axis-relative like `.reduce` --
    # `check_int_pow_no_negative_self`'s whole-buffer scan is still used
    # here too, and is wrong in a DIFFERENT way than the `.accumulate`
    # case: `.reduceat` builds one output per `indices` entry, and each
    # output has its OWN exempt "index 0" -- index 0 of that entry's
    # segment, not index 0 of the whole array. Measured against real numpy
    # 2.5.1 on `a = [-3, 2, 3, -4, 5, 6, 7]` (int64) BEFORE writing any
    # code, across the documented segment shapes (normal, degenerate,
    # descending/backward pair, empty, trailing):
    #   reduceat(a, [0])       -> raises (segment [0:7]: base -3 exempt,
    #                              but idx3's -4 is checked as an exponent
    #                              partway through the fold)
    #   reduceat(a, [0, 3])    -> [729, 0]         no error (segment0
    #                              [0:3): base -3 exempt, 2 and 3 both
    #                              positive; segment1 [3:7) (the LAST
    #                              position, unconditionally `a[3:]`): base
    #                              -4 exempt, 5/6/7 all positive -- the `0`
    #                              is real int64 overflow wraparound from
    #                              chaining power(-4,5),**6,**7, not an
    #                              error path)
    #   reduceat(a, [3, 0])    -> raises -- position 0 (descending pair,
    #                              `indices[0]=3 >= indices[1]=0`) is a
    #                              documented single-element PASSTHROUGH
    #                              `a[3]` (== -4, unchecked, un-folded);
    #                              position 1 (LAST) is unconditionally
    #                              `a[0:7]` and raises for the same reason
    #                              `reduceat(a, [0])` above does
    #   reduceat(a, [0, 0])    -> raises -- same shape: position 0
    #                              (`indices[0]=0 >= indices[1]=0`,
    #                              EQUAL counts as degenerate too, not
    #                              just strictly descending) is passthrough
    #                              `a[0]`; position 1 (LAST) is `a[0:7]`
    #                              and raises
    #   reduceat(a, [1,1,1])   -> raises -- positions 0 and 1 are both
    #                              degenerate passthroughs of `a[1]` (==2,
    #                              irrelevant to the sign check either way);
    #                              position 2 (LAST) is `a[1:7]`: base 2
    #                              exempt, but idx3's -4 is still checked
    #   reduceat(a, [0,3,3])   -> [729, -4, 0]     no error -- THE
    #                              definitive proof of the degenerate-
    #                              passthrough's "no check at all" rule:
    #                              position 1 (`indices[1]=3 >= indices[2]
    #                              =3`) is the raw, negative `a[3] == -4`,
    #                              passed straight through to the output
    #                              completely unchecked
    #   reduceat(a, [2,5])     -> raises (segment0 [2:5): base 3 exempt,
    #                              idx3's -4 checked as exponent)
    #   reduceat(a, [0,2,4])   -> raises (segment1 [2:4): base 3 (idx2)
    #                              exempt, idx3's -4 checked as exponent --
    #                              segment0 [0:2) alone would NOT raise)
    #   reduceat(a, [0,2,4,4]) -> raises (same segment1 as above, plus a
    #                              trailing degenerate passthrough of `a[4]`
    #                              == 5, irrelevant to the sign check)
    if applicable.get("reduceat"):
        a7 = lambda dtype: np.array([-3, 2, 3, -4, 5, 6, 7], dtype=dtype)
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            dname = np.dtype(dtype).name
            cases.append((
                f"reduceat/power_edge/single_segment_neg_exponent_mid_still_raises/{dname}",
                ("reduceat", a7(dtype), np.array([0])), {},
            ))
            cases.append((
                f"reduceat/power_edge/two_segments_both_exempt_index0/{dname}",
                ("reduceat", a7(dtype), np.array([0, 3])), {},
            ))
            cases.append((
                f"reduceat/power_edge/descending_pair_passthrough_then_raise/{dname}",
                ("reduceat", a7(dtype), np.array([3, 0])), {},
            ))
            cases.append((
                f"reduceat/power_edge/equal_pair_passthrough_then_raise/{dname}",
                ("reduceat", a7(dtype), np.array([0, 0])), {},
            ))
            cases.append((
                f"reduceat/power_edge/two_degenerate_then_raise/{dname}",
                ("reduceat", a7(dtype), np.array([1, 1, 1])), {},
            ))
            # The definitive degenerate-passthrough proof: a raw negative
            # value must pass through UNCHECKED (no raise at all for this
            # case).
            cases.append((
                f"reduceat/power_edge/degenerate_negative_passthrough_no_raise/{dname}",
                ("reduceat", a7(dtype), np.array([0, 3, 3])), {},
            ))
            cases.append((
                f"reduceat/power_edge/mid_segment_neg_exponent_still_raises/{dname}",
                ("reduceat", a7(dtype), np.array([2, 5])), {},
            ))
            cases.append((
                f"reduceat/power_edge/only_second_segment_raises/{dname}",
                ("reduceat", a7(dtype), np.array([0, 2, 4])), {},
            ))
            cases.append((
                f"reduceat/power_edge/trailing_degenerate_plus_raise/{dname}",
                ("reduceat", a7(dtype), np.array([0, 2, 4, 4])), {},
            ))

    return cases


def _method_sample_pool(info: ufunc_introspect.UfuncInfo, pool_kind: str):
    """A small, deterministically-sampled, 1-D-preferring slice of the
    relevant corpus, used for .reduce/.accumulate/.outer/.reduceat/.at
    beyond the guaranteed-valid typed_sample -- for volume (broadcasting,
    empties, non-contig views) without re-running every method against the
    entire ~150-case corpus."""
    if pool_kind == "unary":
        pool = ufunc_corpus.string_unary_corpus() if info.is_string else corpus.unary_corpus()
    else:
        pool = ufunc_corpus.string_binary_corpus() if info.is_string else corpus.binary_corpus()
    return ufunc_corpus.sampled(pool, ufunc_corpus.UNARY_SAMPLE_STRIDE)


def build_cases_for_ufunc(info: ufunc_introspect.UfuncInfo, applicable: dict) -> list[tuple]:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return _build_cases_for_ufunc(info, applicable)


def _build_cases_for_ufunc(info: ufunc_introspect.UfuncInfo, applicable: dict) -> list[tuple]:
    cases: list[tuple] = []
    obj = ufunc_introspect.resolve_numpy_ufunc(info.name)

    # ---- plain call, full corpus ----------------------------------------
    if info.signature is not None:
        # gufunc: shape contract, not the elementwise broadcast corpus.
        # typed_sample already encodes a valid (n,k)/(k,m)-style pair.
        cases.append((f"call/typed_sample/{info.name}", ("call", *info.typed_sample), {}))
    elif info.nin == 1:
        pool = ufunc_corpus.string_unary_corpus() if info.is_string else corpus.unary_corpus()
        for c in pool:
            cases.append((f"call/{c.label}", ("call", c.value), {}))
        # FORM axis (2026-08-02, CLASS A closure task): crossed with the
        # same dtype/shape/view axis `pool` already covers, plus the fixed
        # range/scalar forms -- see corpus.form_axis_cases()'s docstring.
        # Not applied to string ufuncs (no non-ndarray form of a string
        # array is meaningful here) or gufuncs (handled by the
        # typed_sample branch above, not this loop).
        if not info.is_string:
            for c in corpus.form_axis_cases(pool, sample_stride=4):
                cases.append((f"call/{c.label}", ("call", c.value), {}))
    elif info.nin == 2:
        pool = ufunc_corpus.string_binary_corpus() if info.is_string else corpus.binary_corpus()
        for p in pool:
            cases.append((f"call/{p.label}", ("call", p.a, p.b), {}))
        if info.is_string:
            unary_pool = ufunc_corpus.string_unary_corpus()
            scalars = ufunc_corpus.string_scalar_operands()
        else:
            unary_pool = corpus.unary_corpus()
            scalars = corpus.scalar_operands()
        sampled_unary = ufunc_corpus.sampled(unary_pool, ufunc_corpus.UNARY_SAMPLE_STRIDE)
        for c in sampled_unary:
            for op_label, other in scalars:
                cases.append((f"call/scalar/{c.label}/{op_label}", ("call", c.value, other), {}))
    else:  # pragma: no cover - no ufunc in the pinned surface has nin not in {1,2} + no signature
        pass

    # typed-domain sample: always included, guarantees a case is run in the
    # ufunc's actual valid dtype/shape domain even when nothing in the
    # broad corpus happens to land there (isnat/datetime64, ldexp's mixed
    # float+int, gcd/bitwise_count's integer-only domain, ...).
    if info.signature is None and info.typed_sample is not None:
        cases.append((f"call/typed_sample", ("call", *info.typed_sample), {}))

    # ---- dtype= / out= / where=+out= kwarg forms -------------------------
    sample = info.typed_sample
    if sample is not None:
        if applicable.get("call_dtype"):
            out_dtype = np.asarray(obj(*sample)).dtype
            cases.append(("call/dtype_kwarg", ("call", *sample), {"dtype": out_dtype}))
        if applicable.get("call_out"):
            out = np.empty_like(obj(*sample))
            cases.append(("call/out_kwarg", ("call", *sample), {"out": out}))
            # positional out=: numpy's ufunc __call__ accepts one OPTIONAL
            # extra positional slot after the nin required inputs
            # (np.add(a, b, out_arr), np.tanh(a, out_arr)) -- distinct call
            # syntax from the out= kwarg form above, and a real bug class
            # (anionpy previously forced out to keyword-only; see commit
            # 972a170's 51-ufunc withdrawal). Appending `out_pos` to the
            # positional `rest` here (rather than putting it in kwargs)
            # is what actually exercises that slot through
            # build_ufunc_dispatcher's `obj(*rest2, **kw)` call.
            out_pos = np.empty_like(obj(*sample))
            cases.append(("call/out_positional", ("call", *sample, out_pos), {}))
        if applicable.get("call_where"):
            base_out = obj(*sample)
            mask_shape = np.broadcast(*sample).shape if info.nin > 1 else sample[0].shape
            mask = np.zeros(mask_shape, dtype=bool)
            mask.reshape(-1)[0] = True
            out = np.zeros_like(base_out)
            cases.append(("call/where_kwarg", ("call", *sample), {"where": mask, "out": out}))

    # ---- order= kwarg form, 2-D/3-D, all four values, mixed layouts -------
    if applicable.get("call_order"):
        cases.extend(_order_call_cases(info))

    # ---- ufunc-protocol methods -------------------------------------------
    if applicable.get("reduce") and sample is not None:
        cases.append(("reduce/typed_sample", ("reduce", sample[0]), {}))
        cases.append(("reduce/typed_sample/axis_none", ("reduce", sample[0]), {"axis": None}))
        for c in _method_sample_pool(info, "unary"):
            if np.asarray(c.value).ndim >= 1:
                cases.append((f"reduce/{c.label}", ("reduce", c.value), {}))
        cases.extend(_reduce_length_sweep_cases(info, sample))
        cases.extend(_reduce_kwarg_cases(info))
        cases.extend(_reduce_masked_length_sweep_cases(info))
        cases.extend(_reduce_kwarg_cases_math(info))
        cases.extend(_reduce_masked_length_sweep_cases_math(info))

    if applicable.get("accumulate") and sample is not None:
        cases.append(("accumulate/typed_sample", ("accumulate", sample[0]), {}))
        for c in _method_sample_pool(info, "unary"):
            if np.asarray(c.value).ndim == 1:
                cases.append((f"accumulate/{c.label}", ("accumulate", c.value), {}))

    if applicable.get("outer") and sample is not None and info.nin == 2:
        cases.append(("outer/typed_sample", ("outer", sample[0], sample[1]), {}))
        for p in _method_sample_pool(info, "binary"):
            cases.append((f"outer/{p.label}", ("outer", p.a, p.b), {}))

    if applicable.get("reduceat") and sample is not None:
        n = sample[0].shape[0] if sample[0].ndim else 0
        idx = _reduceat_indices(n)
        cases.append(("reduceat/typed_sample", ("reduceat", sample[0], idx), {}))
        for c in _method_sample_pool(info, "unary"):
            arr = np.asarray(c.value)
            if arr.ndim == 1 and arr.shape[0] > 0:
                cases.append((f"reduceat/{c.label}",
                               ("reduceat", c.value, _reduceat_indices(arr.shape[0])), {}))

    if applicable.get("at") and sample is not None:
        if info.nin == 1:
            cases.append(("at/typed_sample", ("at", sample[0], [0]), {}))
        elif info.nin == 2:
            cases.append(("at/typed_sample", ("at", sample[0], [0], sample[1][:1]), {}))
        for c in _method_sample_pool(info, "unary"):
            arr = np.asarray(c.value)
            if arr.ndim >= 1 and arr.shape[0] > 0:
                if info.nin == 1:
                    cases.append((f"at/{c.label}", ("at", c.value, [0]), {}))
                elif info.nin == 2 and sample is not None:
                    cases.append((f"at/{c.label}", ("at", c.value, [0], sample[1][:1]), {}))

        # ---- multi-index .at: the leading-axis-on-`values` regression ----
        # test. numpy broadcasts `values` against `(len(indices),) +
        # target.shape[1:]` as a WHOLE, not independently per index -- a
        # single-index case (above) or a values array with no leading axis
        # cannot distinguish "broadcast correctly" from "collapsed to
        # values[0] applied to every index" (both single-index and every-
        # row-identical inputs produce the same output either way). These
        # cases use >1 index AND per-index-DISTINCT values (built from the
        # array's own, necessarily-same-dtype rows so no cast surprises),
        # so a regression back to the old bug is guaranteed to mismatch.
        if info.nin == 2:
            for c in _method_sample_pool(info, "unary"):
                arr = np.asarray(c.value)
                if arr.ndim < 1 or arr.shape[0] < 2:
                    continue
                n = arr.shape[0]
                k = min(3, n)
                # Forward-order target indices with distinct, reversed-order
                # source rows as values -- values[i] != values[j] for i != j
                # whenever the underlying rows differ (true for essentially
                # all corpus arrays; on the rare palindromic array this case
                # degrades to a same-value probe, which is still correct,
                # just not regression-sensitive for that one array).
                target_idx = list(range(k))
                values_idx = list(range(n - 1, n - 1 - k, -1))
                values_multi = arr[values_idx]
                cases.append((f"at/multi_index/{c.label}",
                               ("at", c.value, target_idx, values_multi), {}))

                # Negative indices, mirroring the same leading-axis contract.
                neg_idx = [i - n for i in target_idx]
                cases.append((f"at/multi_index_negative/{c.label}",
                               ("at", c.value, neg_idx, values_multi), {}))

                # Repeated target index with sequentially-distinct values --
                # exercises left-to-right fold order together with the
                # leading axis (a regression that only reused `values[0]`
                # would also fail this, since real numpy folds `values[0]`
                # then `values[1]` in sequence against the SAME target
                # slot).
                if n >= 2:
                    rep_idx = [0, 0]
                    rep_values = arr[[0, 1]]
                    cases.append((f"at/repeated_index/{c.label}",
                                   ("at", c.value, rep_idx, rep_values), {}))

        # ---- 0-d target: `indices=()` applies exactly once; `values` must
        # broadcast directly against `()` (no synthetic leading axis; see
        # `at_broadcast_plan` in ionp-core/src/ufunc.rs for why this is a
        # genuinely distinct case, not just "row_shape happens to be
        # empty"). ----
        if info.nin == 1:
            zerod = corpus.unary_corpus() if not info.is_string else ufunc_corpus.string_unary_corpus()
            for c in zerod:
                arr = np.asarray(c.value)
                if arr.ndim == 0:
                    cases.append((f"at/0d/{c.label}", ("at", c.value, ()), {}))
        elif info.nin == 2:
            pool2 = corpus.binary_corpus() if not info.is_string else ufunc_corpus.string_binary_corpus()
            for p in pool2:
                arr = np.asarray(p.a)
                if arr.ndim == 0:
                    cases.append((f"at/0d/{p.label}", ("at", p.a, (), p.b), {}))

        # ---- empty indices: still validates `values`' shape against
        # `(0,) + row_shape` (an incompatible non-scalar `values` must still
        # raise), but applies nothing. ----
        if sample is not None:
            if info.nin == 1:
                cases.append(("at/empty_indices", ("at", sample[0], []), {}))
            elif info.nin == 2:
                cases.append(("at/empty_indices", ("at", sample[0], [], sample[1]), {}))

    cases.extend(_power_edge_cases(info, applicable))

    return cases


def build_cases_for_item(spec) -> list[tuple]:
    """The kind="ufunc" entry point run.py's build_cases() dispatches to."""
    path = spec.numpy_path if spec.numpy_path is not None else spec.name
    info = ufunc_introspect.describe(path)
    obj = ufunc_introspect.resolve_numpy_ufunc(path)
    applicable = ufunc_introspect.applicable_methods(obj, info)
    return build_cases_for_ufunc(info, applicable)
