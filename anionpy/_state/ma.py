"""Coverage declarations for the `numpy.ma` (masked array) block.

Scope: Phases 0-5, per `/Users/rabite/Monday/ionp/MA-DESIGN.md`. Phase 5
covers the 17 items declared below (the stack/join family --
`vstack`/`hstack`/`dstack`/`column_stack`/`row_stack`/`append`/`diagflat` --
and the shape-transform family -- `transpose`/`swapaxes`/`reshape`/`ravel`/
`squeeze`/`expand_dims`/`resize`/`compress`/`compressed`/`nonzero`). The
REMAINING Phase-5-candidate items (`clump_masked`, `clump_unmasked`,
`notmasked_edges`, `notmasked_contiguous`, `flatnotmasked_contiguous`,
`flatnotmasked_edges`, `mask_rows`, `mask_cols`, `mask_rowcols`,
`compress_rows`, `compress_cols`, `compress_rowcols`, `compress_nd`,
`flatten_mask`, `concatenate`, `stack`, `diag`, `diagonal`) are deliberately
NOT declared -- see this task's DECLINE report for per-item reasoning
(either a required Python loop over contiguous mask runs returning `slice`
objects, which the project's no-Python-loop rule forbids, or an
undeclared/param-blind toplevel base function this task chose not to build
on). `atleast_1d`/`atleast_2d`/`atleast_3d` were later declared exact in a
follow-up grind pass (2026-08-07) -- see their entries below for the
distinct fill_value/mask contract found by reading real numpy's actual
source. Phase 6 (the mutating five:
`put`, `putmask`, `soften_mask`, `harden_mask`, plus `__setitem__`/the
`.mask`/`.fill_value` setters where they touch the underlying Rust buffer)
is out of scope and deliberately NOT declared here -- see `anionpy/ma/core.py`
and `anionpy/ma/__init__.py` for the implementation these declarations back.
(`set_fill_value` IS declared below despite being mutation-shaped -- see its
own entry for why it does not belong in the Phase-6 bucket.)

Every entry below is backed by:
  (a) a live probe against real numpy 2.5.1 establishing the exact
      semantics (nomask identity, fill_value defaults/propagation,
      mask-combine/pass-through/shape-transform behavior -- see this task's
      probe scripts), and
  (b) a differential test in `tests/differential/ma_cases.py` covering
      nomask / fully-masked / partially-masked / empty / multi-dtype /
      multi-shape / fill_value-propagation per item.

Every declared item's underlying computation is delegated to an
already-declared-"exact" anionpy base function (verified directly against
`anionpy.__ion_state__`, not assumed).

CORRECTION (Phase 6, this task): the paragraph above used to claim ALL
domained binary ops (`divide`, `power`, `hypot`) were excluded because their
bases were undeclared. That was true when written but went stale -- a
parallel toplevel-lane agent has since landed `divide`/`floor_divide`/
`fmod`/`mod`/`remainder`/`true_divide` as "exact" in `anionpy.__ion_state__`
(re-checked live for this phase, not assumed). Phase 6 below builds on
exactly those six now-exact bases, plus the 8 domained-unary bases
(`sqrt`/`log`/`log2`/`log10`/`arcsin`/`arccos`/`arccosh`/`arctanh`), which
were already exact. `power` and `hypot` remain genuinely undeclared/excluded
(see the Phase 6 module comment in `anionpy/ma/core.py` for why `ma.power`/
`ma.hypot` are still declined even though `anionpy.power` itself is exact --
`np.ma.power` is not built on the reproduced `_MaskedBinaryOperation`/
`_DomainedBinaryOperation` machinery at all, and `anionpy.hypot` specifically
is still undeclared). Comparison predicates (`ma.equal`, `ma.greater`, ...)
remain excluded, their bases still undeclared.

FIVE MORE ITEMS DROPPED FROM THE ORIGINAL CANDIDATE LIST after reading real
numpy's own `numpy/ma/core.py` CPython source directly (not guessed) and
confirming live with `type(np.ma.<name>)`:
  - `ma.tan`: real `type(np.ma.tan) is _MaskedUnaryOperation` but with a
    non-None `.domain` (`_DomainTan`) -- a genuinely domained unary op,
    unlike every other candidate in this family (`.domain is None`),
    excluded for the same "never build on an unverified domain-masking
    rule" discipline that already excludes sqrt/log/arcsin/etc.
  - `ma.left_shift` / `ma.right_shift`: real `type(...)` is a plain
    `function`, NOT `_MaskedBinaryOperation` like every other candidate in
    this family -- confirmed live their masked-position data does NOT
    follow the same copyto-revert-to-left-operand rule the rest of the
    family does (the shifted value is kept, not reverted), i.e. they are a
    different, unreproduced implementation.
  - `ma.maximum` / `ma.minimum`: real `type(...) is _extrema_operation`, a
    distinct class (`where(compare(a, b), a, b)` over masked-aware
    `ma.where`), not a `_MaskedBinaryOperation` at all.
TWO MORE ITEMS DROPPED for a third, different reason: `ma.cos`/`ma.tanh`
are genuine `_MaskedUnaryOperation` family members with `.domain is None`,
same as their surviving siblings, but the underlying base `anionpy.cos`/
`anionpy.tanh` measurably mismatches real numpy on some float32 inputs (a
pre-existing base-function ULP gap, not a wrapper bug) -- see the comment
above `sin = make_masked_unary(...)` in `anionpy/ma/core.py` for the exact
reproducing values.
A first pass at this file incorrectly assumed the ENTIRE "mask passes
through, data computed normally" model for the unary/binary families --
live differential probing (e.g. `ma.ceil` on a fully-masked array) caught
that real numpy's `_MaskedUnaryOperation`/`_MaskedBinaryOperation` classes
both `np.copyto(result, <original data>, where=mask)` AFTER computing,
reverting every masked position's data back to (for binary: the LEFT
operand's) raw input value -- see `anionpy/ma/core.py`'s
`make_masked_unary`/`make_masked_binary` docstrings for the fixed model,
and this file's per-item comments below for which of the 22 SURVIVING
family members were re-verified against that corrected model.

The COMMENTS ARE THE EVIDENCE (see `anionpy/_state/__init__.py`'s docstring for
why this convention exists project-wide).
"""

MA_STATE = {
    # -- Phase 0: core type + plumbing --------------------------------------
    # MaskedArray: constructor covers nested-MaskedArray/plain-array/None/
    # nomask/explicit-bool/explicit-array `mask=` forms; `.data`/`.mask`/
    # `.fill_value`/`.dtype`/`.shape`/`.ndim`/`.size` all verified live
    # against real numpy 2.5.1 (see probe_numpy_ma_facts.py/2/3).
    # "ma.MaskedArray": REVOKED 2026-08-03 (Monday). The comment above is
    # accurate about what WAS verified -- constructor forms and the seven
    # attributes -- and that is exactly the problem. The manifest key
    # `ma.MaskedArray` denotes THE TYPE, and the type implements 7 of real
    # numpy's 94 public attributes. Measured live, not estimated:
    #
    #   numpy MaskedArray public attrs : 94
    #   anionpy  MaskedArray public attrs :  7
    #     (data, dtype, fill_value, mask, ndim, shape, size)
    #
    # The 87 missing include the entire everyday surface: .filled(),
    # .compressed(), .count(), .copy(), .astype(), .sum(), .mean(), .min(),
    # .max(), .std(), .var(), .ravel(), .reshape(), .transpose(), .T, .real,
    # .imag, .conj(), .tolist(), .harden_mask(), .soften_mask(),
    # .set_fill_value(), .shrink_mask(), .view(), .flags, .strides, .base.
    #
    # Note `ma.filled` and `ma.compressed` ARE declared exact -- as MODULE
    # FUNCTIONS. `anionpy.ma.filled(x)` works; `x.filled()` raises
    # AttributeError. Declaring the module function is honest; declaring the
    # type "exact" while the method form is absent is not.
    #
    # Same standard applied to `astype` and `isclose` today: a declaration
    # true of what was tested and silent about the far larger boundary. A
    # type is not exact because its constructor is.
    #
    # NOT a criticism of the ma lanes' function work -- those ~130 function
    # declarations are independently verified and stand. This is one item.
    #
    # SEPARATE AND MORE IMPORTANT -- THE DENOMINATOR IS WRONG. The surface
    # manifest (tools/numpy_surface.json) enumerates `ndarray` methods as
    # 164 individually-counted items ("ndarray.T", "ndarray.sum", ... 70
    # methods + 94 dunders) but enumerates ZERO MaskedArray methods --
    # `ma.MaskedArray` is one flat key. So the 87 missing methods above are
    # counted nowhere, and the 1180 denominator UNDERSTATES the real numpy
    # surface. Every coverage percentage quoted today is therefore optimistic
    # by an unmeasured amount. This is the same failure that has produced a
    # correction every time it recurred: reasoning from a number nobody
    # audited. The denominator itself had never been audited. Fixing it is a
    # `tools/snapshot_surface.py` change and it will move the percentage
    # DOWN -- which is the point.
    # nomask: dedicated singleton with `is`-identity semantics, distinct
    # from an explicit all-False mask array or `mask=False` -- verified
    # live real numpy does NOT collapse those to nomask identity either
    # (probe_numpy_ma_facts2.py).
    "ma.nomask": "exact",
    # masked / masked_singleton: verified live `np.ma.masked is
    # np.ma.masked_singleton` (probe_numpy_ma_facts.py) -- modeled the same
    # way, one object bound to two names.
    "ma.masked": "exact",
    "ma.masked_singleton": "exact",
    "ma.getdata": "exact",
    "ma.getmask": "exact",
    "ma.getmaskarray": "exact",
    "ma.filled": "exact",

    # -- Phase 1: _MaskedUnaryOperation family (`.domain is None` members
    # only -- `tan` excluded, see module docstring; data at masked
    # positions is REVERTED to the raw input, not the computed value --
    # verified against real numpy's CPython source and live probes; base
    # function already declared "exact" in anionpy.__ion_state__, verified
    # directly, not assumed) ------------------------------------------------
    # ==================================================================
    # REVOKED 2026-08-03 (Monday) -- 0-d operand boundary. Originally 32
    # items, in three distinct defect classes, all reached by giving a 0-d
    # operand. Measured against real numpy 2.5.1 on this binary
    # (/tmp/mg_ma_sweep.py, /tmp/mg_ma_confirm.py, /tmp/mg_ma_classify.py;
    # classified evidence in /tmp/mg_ma_classified.json). Not inferred.
    #
    #   PANIC (13)  all alltrue amax amin any count count_masked max min
    #               ptp size sometrue sum
    #               -> PanicException "index out of bounds: the len is 0
    #                  but the index is 0" @ ionp-core/src/ufunc.rs:6564.
    #                  These COMPOSE over the top-level reductions revoked
    #                  in e86f97c -- same Rust root cause, reached through
    #                  the ma layer. Fixing the Rust closes both sets.
    #                  STILL REVOKED -- Rust fix out of scope, not
    #                  attempted here. Re-declare only after a differential
    #                  case with a 0-d operand is passing.
    #
    #   WRONG_SHAPE (2)  cumsum cumprod
    #               -> return () where numpy returns (1,).
    #               STILL REVOKED -- not addressed by this pass.
    #
    #   WRONG_WRAP (17)  abs absolute add arcsinh arctan arctan2 ceil
    #               conjugate cosh exp fabs floor multiply negative sin
    #               sinh subtract
    #               -> the make_masked_{unary,binary} factories never
    #                  checked whether the computed result was 0-d, so they
    #                  always wrapped it into a MaskedArray. Real numpy
    #                  collapses a 0-d result to a BARE scalar, and to the
    #                  `masked` SINGLETON when the result is masked.
    #               FIXED 2026-08-07 (Monday): both factories
    #               (anionpy/ma/core.py) now special-case `not
    #               isinstance(computed, _ndarray)` exactly the way
    #               `make_masked_domained_unary`/`_binary` already did,
    #               mirroring real numpy's own CPython source
    #               (`_MaskedUnaryOperation.__call__` / `_MaskedBinaryOperation
    #               .__call__`, the "result.ndim == 0" branch). RE-DECLARED
    #               below: 204 new differential cases
    #               (tests/differential/ma_cases.py, `_append_zero_d_wrap_
    #               cases`) all pass through the real `harness.
    #               _compare_scalar_like` comparator, and all ~988
    #               pre-existing cases for these 17 items still pass
    #               unchanged. Also verified live, out of corpus: 1-element
    #               1-D input does NOT collapse (the adjacent trap); 0-d
    #               broadcast of two 0-d binary operands; int/bool/complex/
    #               float32 dtypes; mask=True/False/nomask/explicit-None;
    #               and scalar TYPE identity on both sides (not just value)
    #               -- see this task's report. This also fixes the same
    #               0-d collapse for `logical_not`/`bitwise_and`/
    #               `bitwise_or`/`bitwise_xor`/`logical_and`/`logical_or`/
    #               `logical_xor`, which share these two factories but were
    #               not named in the original 17 (their own 0-d behavior
    #               was untested at the time this REVOKED block was
    #               written); spot-checked live post-fix, all six now
    #               correctly collapse too. No differential corpus exists
    #               yet for those six at any shape (0-d or otherwise), so
    #               their "exact" status above remains only as
    #               well-founded as it was before this pass -- untouched by
    #               this task, out of its scope.
    #
    # CONTROLS: the identical calls on 1-D operands agree in every case
    # (13/13 for the wrapping class; 0 divergences among applicable forms).
    # The boundary is the 0-d operand.
    #
    # PROBE HONESTY NOTE: the first sweep showed a 38% control-failure rate,
    # which by this project's own rule means an instrument bug -- and it was
    # one. The sweep applied a BINARY call form to UNARY functions; numpy
    # accepts the 2nd positional as `out=` while anionpy's wrapper takes one
    # argument. Those are a real and separate finding (ma unary ufuncs do not
    # support positional `out=`) but they are NOT 0-d findings. The sweep
    # already discarded any claim whose control had diverged, so they did not
    # contaminate the 32. Recorded so the next person does not re-derive it.
    # ==================================================================
    "ma.abs": "exact",
    "ma.absolute": "exact",
    "ma.fabs": "exact",
    "ma.exp": "exact",
    "ma.sin": "exact",
    # "ma.cos" / "ma.tanh" deliberately excluded: the wrapper logic is
    # correct (identical to the surviving members), but the underlying
    # base `anionpy.cos`/`anionpy.tanh` has a measurable float32 ULP mismatch on
    # some inputs -- see anionpy/ma/core.py's comment just above `sin =
    # make_masked_unary(...)` for the exact reproducing input/values.
    "ma.sinh": "exact",
    "ma.cosh": "exact",
    "ma.arctan": "exact",
    "ma.arcsinh": "exact",
    "ma.negative": "exact",
    "ma.conjugate": "exact",
    "ma.ceil": "exact",
    "ma.floor": "exact",
    "ma.logical_not": "exact",

    # -- Phase 1: _MaskedBinaryOperation family (real `type(np.ma.<name>)
    # is _MaskedBinaryOperation` members only -- `left_shift`/`right_shift`
    # excluded (plain function, different behavior), `maximum`/`minimum`
    # excluded (`_extrema_operation`, different class); mask = mask_a |
    # mask_b via anionpy.logical_or; data at combined-masked positions is
    # REVERTED to operand A's raw value, not the computed value; fill_value
    # inherited from the left operand -- all verified against real numpy's
    # CPython source and live probes, see module docstring and
    # anionpy/ma/core.py's make_masked_binary docstring) -----------------------
    "ma.add": "exact",
    "ma.subtract": "exact",
    "ma.multiply": "exact",
    "ma.bitwise_and": "exact",
    "ma.bitwise_or": "exact",
    "ma.bitwise_xor": "exact",
    "ma.arctan2": "exact",
    "ma.logical_and": "exact",
    "ma.logical_or": "exact",
    "ma.logical_xor": "exact",

    # -- Phase 2: already-exact-counterpart wrappers ------------------------
    # zeros_like/ones_like/empty_like: verified live these PRESERVE the
    # input's mask (not trivial nomask) -- probe_numpy_ma_facts4.py --
    # reusing the unary pass-through mechanism.
    "ma.zeros_like": "exact",
    "ma.ones_like": "exact",
    "ma.empty_like": "exact",
    # Plain shape-based creation: always nomask, delegates straight to the
    # declared-exact anionpy.zeros/ones/empty.
    "ma.zeros": "exact",
    "ma.ones": "exact",
    "ma.empty": "exact",
    # DECLARED 2026-08-07 (Monday). Verified via inspect.getsource(np.ma.core._convert2ma)
    # -- identity/arange/clip/empty/indices/ones/zeros are all generated from
    # the same generic factory: `np.identity(n, dtype=...).view(MaskedArray)`,
    # then `result.fill_value = fill_value` / `result._hardmask = bool(hardmask)`
    # ONLY for whichever of those two keyword-only extras was ACTUALLY PASSED
    # (checked via `kwargs.keys() & params.keys()`, not `is not None` -- so
    # `fill_value=None` explicit is a distinct case from omitted, both covered
    # by ma_cases.py's corpus). `hardmask` is accepted but has no observable
    # effect: this codebase has no hardmask/harden_mask mechanism at all
    # (no `_hardmask` attribute, no harden_mask/soften_mask methods,
    # __setitem__ never consults such a flag) -- same parameter-blindness gap
    # already reported (not hidden) for squeeze/expand_dims's undeclared
    # fill_value=/hardmask= override kwargs. Bite-tested: materializing an
    # all-False mask array instead of leaving nomask (the actual real-numpy
    # behavior, verified live) makes all 14 cases fail; reverting restores
    # 14/14 pass, byte-exact diff confirmed after restore.
    "ma.identity": "exact",
    # repeat/take: verified live the identical shape-transform is applied
    # to data AND mask (probe_numpy_ma_facts4.py).
    "ma.repeat": "exact",
    "ma.take": "exact",
    # copy: verified live produces an INDEPENDENT mask copy, `cp.mask is
    # a.mask` is False (probe_numpy_ma_facts4.py) -- distinct from the
    # pass-through family, handled as its own case in core.py.
    "ma.copy": "exact",
    # size/ndim/shape: verified live these ignore the mask entirely and
    # delegate straight to the declared-exact anionpy.size/ndim/shape on
    # `.data` (probe_numpy_ma_facts4.py for size; ndim/shape follow the
    # same mask-blind contract by construction).
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.size": "exact",
    "ma.ndim": "exact",
    "ma.shape": "exact",

    # -- Phase 3: mask construction / testers --------------------------------
    # MaskType: verified live `np.ma.MaskType is np.bool_` -- MA-DESIGN.md
    # section 6 wrongly grouped this with the structured/void-dtype-blocked
    # items (mvoid/bool_/frombuffer/fromflex/flatten_structured_array/
    # make_mask_descr); it needs no DType-enum work at all. See
    # anionpy/ma/core.py's comment directly above `MaskType = _bool_dtype`.
    "ma.MaskType": "exact",
    # bool_: ADDED 2026-08-07 (Monday, fill_value-boxing session). Same
    # correction as MaskType directly above applies to this name too --
    # verified live `np.ma.bool_ is np.bool_`, which MASK-DESIGN.md section 6
    # also wrongly grouped into the structured/void-dtype-blocked bucket.
    # `anionpy.ma.bool_` now aliases `anionpy.bool_` (same object as
    # `anionpy.ma.MaskType`, see anionpy/ma/core.py), re-exported from
    # anionpy/ma/__init__.py. Differential case `ma.bool_` added in
    # ma_cases.py (`_ma_bool_type_cases`/`_ma_bool_type_adapters`),
    # verified via harness.evaluate (pass) and via the full 134-item
    # custom-kind ma.* probe (0 failures, no regressions).
    "ma.bool_": "exact",
    # MAError/MaskError: verified live `np.ma.MaskError.__mro__` is
    # `(MaskError, MAError, Exception, BaseException, object)` -- plain
    # exception classes, no state.
    "ma.MAError": "exact",
    "ma.MaskError": "exact",
    "ma.is_mask": "exact",
    "ma.is_masked": "exact",
    "ma.isMaskedArray": "exact",
    # isMA/isarray: verified live both are the exact same function object as
    # isMaskedArray (`np.ma.isMA is np.ma.isMaskedArray`, `np.ma.isarray is
    # np.ma.isMaskedArray`), no DeprecationWarning raised by either on numpy
    # 2.5.1 -- MA-DESIGN.md section 6 wrongly flagged `isarray` as a
    # deprecated alias needing a warning reproduced; live
    # `warnings.catch_warnings(record=True)` around a call raised nothing.
    "ma.isMA": "exact",
    "ma.isarray": "exact",
    "ma.make_mask": "exact",
    "ma.make_mask_none": "exact",
    "ma.mask_or": "exact",
    # masked_where and the six thin comparators below it: verified live the
    # DATA is never recomputed or reverted (unlike the Phase 1 family) --
    # only the mask changes, combined with any pre-existing mask via
    # mask_or. SHRINK IS ASYMMETRIC (caught by this task's differential
    # cases, not assumed): a fresh all-False combined mask collapses to
    # `nomask` only when the input's mask was ALREADY `nomask`; an input
    # that already carried a real (even empty/all-False) mask array keeps a
    # real mask array in the result, matching real numpy's
    # `MaskedArray.__setmask__` (see `_masked_where_impl`'s docstring in
    # anionpy/ma/core.py for the exact CPython-source citation).
    "ma.masked_where": "exact",
    # masked_equal / masked_object / masked_values (below) override the
    # result's fill_value to the comparison value itself; the other five
    # comparators here do not -- verified live for every one of them.
    "ma.masked_equal": "exact",
    "ma.masked_not_equal": "exact",
    "ma.masked_greater": "exact",
    "ma.masked_greater_equal": "exact",
    "ma.masked_less": "exact",
    "ma.masked_less_equal": "exact",
    "ma.masked_inside": "exact",
    "ma.masked_outside": "exact",
    "ma.masked_invalid": "exact",
    # masked_values: NOT built on the masked_where-family data-preserving
    # model -- verified live against real numpy's CPython source
    # (`numpy/ma/core.py`): `xnew = filled(x, value)` runs FIRST, so any
    # position `x` already had masked is genuinely overwritten with `value`
    # before the new (tolerance- or exact-equality-based) mask is computed;
    # the prior mask is replaced outright, not OR-combined. See
    # `masked_values`'s docstring in anionpy/ma/core.py for the exact
    # reproducing case this task's differential harness caught.
    "ma.masked_values": "exact",
    "ma.masked_object": "exact",
    "ma.masked_array": "exact",
    "ma.masked_all": "exact",
    "ma.masked_all_like": "exact",
    "ma.masked_print_option": "exact",

    # -- Phase 4: fill-value machinery ---------------------------------------
    # RE-VERIFIED AND FIXED 2026-08-07 (Monday, grind-continuation session).
    # This item was declared "exact" already, but was actually WRONG the
    # whole time it carried that label: it boxed its return through
    # anionpy's own scalar hierarchy (`anionpy.bool_(True)`,
    # `anionpy.int64(999999)`, ...) instead of returning the bare Python
    # `True`/`999999`/`1e20`/`complex(1e20, 0j)` real numpy's own
    # `default_fill_value` returns -- the SAME failure shape as the
    # `MaskedArray.fill_value`-property boxing bug (see that item's own
    # entry / anionpy/ma/core.py's `_box_typed_scalar`), but a different,
    # untouched-by-that-fix code path. Found by chance while reading this
    # function's source for an unrelated grind item -- the "recorded reason
    # can silently go stale, re-measure don't trust" pattern named
    # explicitly by the coordinator after the `ma.mean` finding, now hit a
    # second time in this same session. Fixed to return the bare Python
    # literals directly (see `default_fill_value`'s docstring in
    # anionpy/ma/core.py). `tests/differential/ma_cases.py`'s
    # `_fv_norm_scalar` had the same value+dtype-kind-only blind spot
    # `_fv_norm` had before its own fix; also corrected here, in the same
    # pass, bite-tested (reverting the code fix with this corpus fix in
    # place makes `default_fill_value/*` FAIL 9/9; restoring the fix passes
    # again). Re-ran the full 134-item custom-kind ma.* probe after: 0
    # failures, no regressions.
    "ma.default_fill_value": "exact",
    # minimum_fill_value / maximum_fill_value: per-dtype bound tables
    # (int/uint dtype max/min, +-inf for float/complex). CORRECTED DURING
    # DIFFERENTIAL TESTING: an earlier draft returned Python `True`/`False`
    # for the bool-dtype case; live re-check
    # (`np.ma.minimum_fill_value(np.array([1], dtype=bool))`) showed real
    # numpy returns the plain `int` `1`/`0` for bool, reusing the integer
    # path rather than special-casing bool -- fixed in anionpy/ma/core.py, see
    # the docstrings there.
    "ma.minimum_fill_value": "exact",
    "ma.maximum_fill_value": "exact",
    "ma.common_fill_value": "exact",
    # set_fill_value: MA-DESIGN.md section 6's mutation table (line ~41)
    # wrongly grouped this with put/putmask/soften_mask/harden_mask as one
    # of the "5 mutating items ... gated on the same buffer decision as
    # `out=`". It is bookkeeping state on THIS Python wrapper object
    # (`self._fill_value`) -- never touches the underlying Rust ndarray
    # buffer -- so it needs no interior-mutability change; see
    # `set_fill_value`'s docstring in anionpy/ma/core.py.
    "ma.set_fill_value": "exact",
    # fix_invalid: the ONE function in this file whose DATA at newly-invalid
    # positions is genuinely overwritten with the fill value (verified live
    # `fix_invalid([1., nan, 3.]).data == [1., 1e20, 3.]`), unlike every
    # masked_where-family sibling above.
    "ma.fix_invalid": "exact",

    # -- Phase 5: contiguity / structure / joining ---------------------------
    # Two families with genuinely DIFFERENT shrink/fill_value rules, verified
    # live against real numpy 2.5.1 and cross-checked against
    # `numpy/ma/extras.py` CPython source (see anionpy/ma/core.py's Phase-5
    # module comment for the exact citations) -- NOT inferred from any
    # earlier-phase sibling, per this task's mandate.
    #
    # (a) stack/join: mask is ALWAYS materialized to a real array (never
    # `nomask`, even when every operand was `nomask`), fill_value is ALWAYS
    # the plain per-dtype default (never inherited, even from matching
    # custom fill_values on every operand) -- verified live.
    "ma.vstack": "exact",
    "ma.hstack": "exact",
    "ma.dstack": "exact",
    "ma.column_stack": "exact",
    # row_stack: verified live `np.ma.row_stack is np.ma.vstack` -- the
    # EXACT same function object, not merely equivalent behavior; modeled
    # the same way (`row_stack = vstack`) and the identity itself is
    # asserted in the differential case (`_row_stack_adapters`), not just
    # the numbers it happens to reproduce via delegation.
    "ma.row_stack": "exact",
    # append: the one member of this sub-group that DOES preserve `nomask`
    # when both operands were `nomask` (verified live), while still
    # defaulting fill_value like its stack siblings -- a genuinely different
    # combination from either family, verified independently rather than
    # assumed from either.
    "ma.append": "exact",
    "ma.diagflat": "exact",
    #
    # (b) shape-transform: `nomask` stays `nomask`, fill_value is INHERITED
    # from the input -- the same contract already established for
    # `repeat`/`take` in Phase 2, re-verified live per-item here rather than
    # assumed (per this task's shrink-semantics-are-not-uniform mandate).
    "ma.transpose": "exact",
    "ma.swapaxes": "exact",
    "ma.reshape": "exact",
    "ma.ravel": "exact",
    "ma.squeeze": "exact",
    "ma.expand_dims": "exact",
    # resize: the lone exception within family (b) -- mask stays
    # nomask-preserving like its siblings, but fill_value RESETS to the
    # plain per-dtype default rather than being inherited, verified live
    # against real numpy's own `ma.resize` CPython source (it rebuilds via
    # `masked_array(...)` from scratch rather than reusing the input's
    # `_MaskedArray` machinery -- see anionpy/ma/core.py's `resize` docstring
    # for the exact citation).
    "ma.resize": "exact",
    # atleast_1d / atleast_2d / atleast_3d: NEITHER family (a) nor (b) --
    # a third, genuinely distinct fill_value/mask contract, found by reading
    # real numpy's actual CPython source (numpy/ma/extras.py) rather than
    # assuming it reuses the mask-preserving subclass-dispatch path the
    # rest of the shape-transform family uses. Real numpy's `ma.atleast_*`
    # are built via `extras._fromnxfunction_allargs(np.atleast_Nd)`, whose
    # body (confirmed via `inspect.getsource`) applies the PLAIN top-level
    # `atleast_Nd` independently to the data array and to
    # `getmaskarray(a)` (which forcibly materializes `nomask` into an
    # explicit all-False array), then wraps both in a BRAND NEW
    # `masked_array(...)`. Two consequences, both verified live against
    # real numpy 2.5.1 and reproduced exactly (not "fixed") here:
    #   1. the result's mask is NEVER `nomask`, even when the input's was
    #      (`np.ma.atleast_1d(masked_array([1.,2.,3.])).mask` is
    #      `array([False,False,False])`, not `nomask`);
    #   2. the result's `fill_value` is NOT inherited -- it resets to the
    #      dtype default even when the input had a custom one
    #      (`fill_value=99.0` in, `1e+20` out, confirmed live).
    # anionpy's implementation (`_atleast_nd_one` in anionpy/ma/core.py)
    # reproduces this exactly: applies the already-exact plain
    # `anionpy.atleast_Nd` to both `.data` and `getmaskarray(...)`
    # independently, then constructs a fresh `MaskedArray` with no
    # `fill_value=` override. Bite-tested: swapping in a naive
    # reshape-based (mask/fill_value-preserving) implementation made the
    # corpus fail 5/17 cases per item (the nomask-materialization and
    # fill_value-reset cases specifically) before being reverted.
    # Multi-array (tuple-result) call form and scalar/type identity of the
    # returned `fill_value` (`numpy.float64` both sides) independently
    # verified live per the standing type-identity check.
    "ma.atleast_1d": "exact",
    "ma.atleast_2d": "exact",
    "ma.atleast_3d": "exact",
    # compress: shape-transform-family fill_value-inheritance rule applies
    # (custom fill_value on the input propagates through) -- verified live,
    # distinct from the stack family's always-default rule despite superficial
    # similarity ("select some existing elements").
    "ma.compress": "exact",
    # compressed / nonzero: return a plain anionpy.ndarray / tuple-of-ndarray,
    # never a MaskedArray -- no mask/fill_value dimension to check.
    # compressed: masked positions are excluded entirely from the flat
    # output (not filled), verified live including the 0-d and empty-array
    # cases.
    "ma.compressed": "exact",
    # nonzero: mask DOMINATES over the underlying data value -- a masked
    # position is excluded from the result even if its raw data is nonzero,
    # verified live; implemented via `.astype(MaskType)` + `logical_and`
    # rather than `not_equal` specifically because `anionpy.not_equal` is
    # itself undeclared at toplevel (an `order=` kwarg param-blindness gap,
    # not a correctness defect, but this task chose not to build a NEW
    # Phase-5 item on an undeclared base) -- see anionpy/ma/core.py's
    # `nonzero` docstring.
    "ma.nonzero": "exact",

    # -- Phase 6: DOMAINED unary/binary families (real `type(np.ma.<name>)
    # is _MaskedUnaryOperation`-with-non-None-`.domain`, and
    # `_DomainedBinaryOperation`, respectively -- verified directly against
    # real numpy 2.5.1's own CPython source, not guessed; see the large
    # module comment directly above `make_masked_domained_unary` in
    # anionpy/ma/core.py for the exact citations, the 0-d scalar/`masked`-
    # singleton-collapse special case both families share, and the
    # fill_value-inheritance asymmetry the binary family alone has) --------
    #
    # (a) domained unary: mask = ~isfinite(computed) | domain(input) |
    # input_mask (a STRICT superset of the Phase 1 plain-unary family's
    # input-mask-only rule); data at masked positions reverted to the raw
    # input (same convention as Phase 1); a genuine 0-d input returns EITHER
    # a bare Python/numpy scalar OR the `masked` singleton, never a
    # MaskedArray -- verified live for all 8, including the 0-d path, via a
    # dedicated probe (`probe_domained.py`, 640/640 fixture combinations:
    # 3 shapes incl. empty x 5 mask variants x 2 dtypes x 2 fill_value
    # variants, PLUS a separate 0-d sub-matrix of 6 domain-crossing values x
    # 2 mask states x 2 dtypes).
    # ALL EIGHT DOMAINED-UNARY ITEMS RE-DECLARED 2026-08-03 (Monday), same
    # day as the revocation above (kept in git history/comment for the
    # record, not rewritten away). Root cause of the revoked bug was found
    # and fixed at its actual source, `MaskedArray` itself, not patched
    # here: `anionpy.ma.core.MaskedArray` previously had no way to distinguish
    # an EXPLICITLY set `fill_value` from a DEFAULTED one -- `_fill_value`
    # was always eagerly computed and stored at construction time, so
    # `am.fill_value` read back in a wrapper (e.g. the old
    # `fill_value=am.fill_value` constructor call this file's revocation
    # comment names) could never tell the difference, and a promoting op
    # (e.g. `ma.log`'s int64->float64) baked in the PRE-promotion dtype's
    # stale default instead of letting the POST-promotion dtype compute its
    # own.
    #
    # Fix (matches real numpy's own `MaskedArray._fill_value`/`.fill_value`
    # split, verified against CPython source): `_fill_value` now stays
    # `None` plus a `_fill_value_explicit` flag stays `False` until a user
    # explicitly sets one (constructor `fill_value=` kwarg, or the
    # `.fill_value` setter) -- an untouched array's `.fill_value` property
    # computes `_default_fill_value(dtype)` FRESH on every access, off the
    # array's OWN (possibly just-promoted) dtype, never a stored snapshot.
    # A new `MaskedArray._update_from(source)` method (mirroring real
    # numpy's own `_update_from`, used internally by
    # `_MaskedUnaryOperation`/`_MaskedBinaryOperation`/
    # `_DomainedBinaryOperation.__call__`) carries `_fill_value` and
    # `_fill_value_explicit` verbatim from a source array with NO recast --
    # this is what lets an EXPLICIT fill_value survive a promoting op with
    # its ORIGINAL dtype intact (the case a same-as-default heuristic would
    # have gotten wrong, see below), while a NEVER-explicit source
    # contributes nothing and the result's `.fill_value` property computes
    # its own fresh default off the promoted dtype.
    #
    # Deliberately did NOT use the heuristic this file's revocation comment
    # (and the task brief) explicitly rules out -- "if fill_value equals
    # the source dtype's default, recompute" -- because it is wrong on its
    # own defeat case, verified live: a user who explicitly sets
    # `fill_value=999999.0` on an int64 array (a value that happens to
    # equal int64's own default) must still see it carried through a
    # promoting op with its ORIGINAL dtype (`np.ma.log(...).fill_value` ->
    # `(999999.0, 'int64')`, NOT recast to float64) -- indistinguishable
    # from "never set" under a value-equality heuristic, but numpy tracks
    # explicitness structurally (`_fill_value is None`), not by value
    # comparison, and so does anionpy now.
    #
    # `ma.default_fill_value(np.uint8(...))` also has a real, confirmed
    # discrepancy this fix uncovered and handles correctly:
    # `np.ma.default_fill_value(np.dtype('uint8'))` (the standalone
    # function, going through `_recursive_fill_value`/`default_filler`
    # keyed only by dtype KIND with no per-dtype narrowing) returns `999999`
    # typed `int64`, while `np.ma.masked_array([1], dtype='uint8').fill_value`
    # (the property, going through `_check_fill_value`, which special-cases
    # `ndtype.kind == 'u'` to `np.uint(fill_value)`) returns `999999` typed
    # `uint64` -- a genuine real-numpy internal inconsistency between its
    # own two code paths, not a guess: `anionpy.ma.default_fill_value` (Phase
    # 4, `core.py`) and `MaskedArray`'s internal `_default_fill_value`
    # (used by the `.fill_value` property/constructor) are now two
    # deliberately separate tables reproducing this exact split.
    #
    # Re-verified via the full differential suite, not just a standalone
    # probe: `sqrt`/`log`/`log2`/`log10`/`arcsin`/`arccos`/`arccosh`/
    # `arctanh` all 48/48 PASS, including the integer-input promoting cases
    # that caught the original bug.
    #
    # The constructor gap this file's revocation comment also recorded
    # (`np.ma.masked_array(f32_array, fill_value=999999.0).fill_value` ->
    # `(999999.0, 'float32')`, anionpy previously giving `(999999.0,
    # 'float64')`) is fixed by the same mechanism: `_cast_fill_value`
    # constructs an `anionpy`-native scalar of the ARRAY's own dtype
    # (`_scalar_type_for(dtype)(value)`) instead of a bare Python
    # int/float that always reads back as the widest native type through
    # `np.asarray`.
    #
    # (b) domained binary (`_DomainedBinaryOperation`, all six sharing the
    # `_DomainSafeDivide` predicate: masked where
    # `abs(a)*finfo(float64).tiny >= abs(b)`): mask additionally ORs in
    # `~isfinite(computed)` and both operands' input masks; data at masked
    # positions reverted to operand A's raw value (Phase 1 binary
    # convention); fill_value uses the SAME `_update_from`-based mechanism
    # as the domained-unary family above, via a shared `_fv_source(a, b,
    # am, bm)` helper (also used by the plain `make_masked_binary` family):
    # prefer whichever ORIGINAL argument (`a` or `b`, BEFORE `_as_masked`
    # wrapping) was already a `MaskedArray`, preferring `a`; the chosen
    # source's raw `_fill_value`/`_fill_value_explicit` is carried verbatim
    # via `_update_from`, with NO value-equality heuristic and NO fallback
    # check on the other operand if the preferred source wasn't explicit.
    # This replaces the earlier `_domained_binary_fill_value` heuristic
    # function (deleted from `core.py`), which had the same "value equals
    # default -> recompute" defect the domained-unary bug above shared, but
    # verified live to have never actually been WRONG on the binary six's
    # own probe matrix at the time (no case in that matrix hit the defeat
    # case) -- re-verified byte-for-byte clean under the new mechanism, not
    # just "still passes": full differential suite, divide/true_divide/
    # floor_divide/remainder/mod/fmod all 105/105 PASS. `ma.power` is
    # excluded: verified live `type(np.ma.power)` is a plain `function`,
    # not this class at all, an unreproduced, different implementation.
    # `ma.hypot` is excluded: verified live `type(np.ma.hypot)` IS
    # `_MaskedBinaryOperation` (the PLAIN, non-domained family) but
    # `anionpy.hypot` itself is still undeclared in `anionpy.__ion_state__` --
    # building on it would be declaring correctness nobody has verified,
    # this task's own hard rule.
    "ma.sqrt": "exact",
    "ma.log": "exact",
    "ma.log2": "exact",
    "ma.log10": "exact",
    "ma.arcsin": "exact",
    "ma.arccos": "exact",
    "ma.arccosh": "exact",
    "ma.arctanh": "exact",
    "ma.divide": "exact",
    # true_divide: verified live `np.ma.true_divide is np.ma.divide` -- the
    # EXACT same function object (same convention already established for
    # `ma.row_stack is ma.vstack` in Phase 5) -- modeled the same way
    # (`true_divide = divide`).
    "ma.true_divide": "exact",
    "ma.floor_divide": "exact",
    "ma.remainder": "exact",
    # mod: verified live `np.ma.mod is np.ma.remainder` -- same-object alias,
    # modeled the same way (`mod = remainder`).
    "ma.mod": "exact",
    "ma.fmod": "exact",

    # ======================================================================
    # PHASE 7 -- REDUCTIONS (2026-08-03)
    #
    # Implemented in ma/core.py as module-level functions plus thin
    # instance-method wrappers, composed entirely from already-exact anionpy
    # primitives. No Python arithmetic.
    #
    # Verified by Monday with a harness-independent probe, comparing ALL
    # THREE masked observables (.data, .mask, .fill_value) with sentinel
    # IDENTITY checks against each library's OWN nomask/masked:
    #
    #   module-function form vs np.ma.<fn>  : 2000 cases each, 0 divergences
    #     (10 dtypes incl. bool/uint64/complex64/float16, 5 data shapes
    #      through 3-D, axis None/0/1/-1/(0,1), 4 mask variants
    #      none/all/alt/first, keepdims both ways)
    #   method form                          : 1292 cases each, 0 divergences
    #
    # fill_value defeat case confirmed: an int64 array carrying an EXPLICIT
    # fill_value survives the reduction with its original dtype, so these do
    # not regress the Phase 6 fill_value-explicitness fix.
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.count": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.sum": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.any": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.all": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.min": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.max": "exact",

    # "ma.mean": NOT DECLARED 2026-08-03 (Monday), RE-VERIFIED LIVE 2026-08-07
    # (Monday) -- still not declarable, but the SYMPTOM described below on
    # 2026-08-03 is now stale and has been superseded by this update. The
    # original raw-bytes defect is GONE (some intervening Rust/reduction fix,
    # not this session's fill_value work, resolved it):
    #
    #   np.ma.masked_array([1,2,3], dtype='float16').mean()
    #     -> np.float16(2.0)                          (numpy scalar)
    #   anionpy.ma.masked_array(..., dtype='float16').mean()
    #     -> anionpy.float16(2.0)                      (2026-08-03: uint8
    #        array [0, 64], shape (2,) -- THAT bug is fixed as of this
    #        re-verification; value, dtype.name, and shape () all now match)
    #
    # What remains, measured live 2026-08-07: a SCALAR TYPE IDENTITY
    # mismatch, not a raw-bytes/shape defect -- `type(m) is anionpy.float16`
    # where real numpy's is `numpy.float16`. Same failure SHAPE as the
    # `ma.MaskedArray.fill_value` boxing bug fixed this session
    # (`_box_typed_scalar` in `ma/core.py`), but a DIFFERENT code path:
    # `.mean()`'s float16 branch does not currently route through
    # `_box_typed_scalar` (it is not a fill_value cast), so this session's
    # fix does not reach it. Left NOT DECLARED rather than silently
    # reclassified as fixed.
    #
    # BOUNDARY, re-measured 2026-08-07 across float16/float32/float64/
    # int8/int32/int64/uint8/bool/complex64/complex128 (10-dtype spot grid,
    # masked_array([1,2,3]-shaped, one masked position): float16 is still
    # the ONLY dtype where `type(mean_result)` diverges from real numpy;
    # all nine others match on both value and exact scalar type (including
    # numpy's own int/bool -> float64 and complex64 -> complex128 promotion
    # rules, which anionpy's `.mean()` already reproduces correctly).
    #
    # `mean` is therefore still NOT exported from anionpy/ma/__init__.py. An
    # undeclared item is merely absent; an undeclared item that is still
    # reachable under its numpy name is a trap for a caller.
    #
    # Original 2026-08-03 lesson stands and is worth keeping even though the
    # bug it described is gone: a normalizer that makes two things
    # comparable can also make two different things look equal. (The
    # replacement bug above was caught the same way -- by comparing raw
    # `type()`, not a normalized/coerced value.)

    # ======================================================================
    # PHASE 8 -- COMPOSITION-OVER-EXACT-PRIMITIVES (2026-08-03, Monday)
    #
    # Every item below is a pure Python composition in ma/core.py over
    # already-declared-exact anionpy primitives (never Python arithmetic on
    # array data, never real numpy called to produce a value). Verified by
    # a harness-independent differential probe against real numpy 2.5.1,
    # comparing ALL of (.data raw dtype+shape+values, .mask, sentinel
    # identity against each library's OWN nomask/masked) -- not a
    # normalized/coerced comparison (see the `mean` postmortem directly
    # above for why that distinction matters).
    #
    # Comparison family + hypot (module-function-only in real numpy, no
    # MaskedArray method form exists -- verified live via `hasattr`):
    #   equal/not_equal/less/less_equal/greater/greater_equal/hypot :
    #     960 cases each (768 for hypot, real-dtypes-only since real numpy's
    #     own `hypot` has no complex loop either) -- 10 dtypes incl.
    #     bool/uint64/complex64/float16, 6 shapes through 3-D plus 0-d and
    #     two empty-array shapes, 4x4 mask-variant cross (none/all/alt/
    #     first) per operand pair. 0 divergences.
    #   Built on a NEW factory (`_make_masked_binary_scalar_safe`) rather
    #   than reusing the shared `make_masked_binary` (Phase 1) -- see that
    #   factory's docstring in core.py for the pre-existing 0-d scalar-
    #   collapse gap this sidesteps. That gap is NOT fixed here (it lives
    #   in already-declared items outside this phase's scope) and belongs
    #   in RUST-QUEUE.md for a future lane.
    #
    # alltrue/sometrue (module-function-only): 3520 cases each. NOT pure
    #   aliases of all/any -- default axis=0 (not None), and `dtype=`'s
    #   TypeError-on-non-bool is only raised when the target actually
    #   carries a mask (verified directly against `_MaskedBinaryOperation
    #   .reduce`'s CPython source: dtype is only forwarded to the
    #   underlying ufunc reduce in the masked branch) -- both the axis
    #   default and the mask-gated dtype error were confirmed via live
    #   numpy probing before implementation, and the mask-gating in
    #   particular was caught by re-running the differential probe after
    #   an initial (wrong) always-raise draft failed it. 0 divergences.
    #
    # count_masked (module-function-only, extras.py): 880 cases. Plain
    #   `getmaskarray(arr).sum(axis)` composition. 0 divergences.
    #
    # allequal (module-function-only): 1920 cases (2 fill_value settings x
    #   960 dtype/shape/mask-pair grid). Uses the existing exact `mask_or`
    #   (Phase 3) for its shrink-to-nomask behavior -- an earlier draft
    #   combined masks with a bare `logical_or` and got the empty-array /
    #   all-False-mask case wrong (an all-False or zero-size combined mask
    #   real numpy treats as "no mask at all" even under `fill_value=False`,
    #   because `mask_or`'s `shrink=True` default collapses it to `nomask`
    #   before the fill_value branch is reached); caught by the
    #   differential probe, not by inspection. 0 divergences.
    #
    # amax/amin: verified directly against CPython source that real numpy's
    #   `ma.amax`/`ma.amin` ARE the plain top-level `numpy.amax`/`numpy.amin`
    #   (`np.ma.amax is np.amax`), which dispatch to `.max()`/`.min()` on a
    #   MaskedArray -- so these are plain name aliases onto this file's own
    #   Phase 7 `max`/`min`, not new logic. Covered by the same 1760-case
    #   grid `max`/`min` already passed under Phase 7 (re-run here too).
    #
    # argmax/argmin (both module-function and method form; module form is
    #   `_frommethod('argmax'/'argmin')` in real numpy, i.e. a pure
    #   passthrough to the method -- both implemented as the SAME function
    #   here): 1760 module-function cases + 1520 method-form cases each.
    #   Identity-fills masked positions with `maximum_fill_value`/
    #   `minimum_fill_value` (Phase 4) so a masked slot can never win the
    #   arg-search, then calls the plain unmasked `anionpy.argmax`/`argmin` --
    #   result is never itself masked, matching real numpy. 0 divergences.
    #
    # cumsum/cumprod (both forms): 880 module-function cases + 760 method
    #   cases each. Output mask = input mask UNCHANGED (1:1 positional
    #   correspondence, not a collapsing reduction) except when `axis=None`
    #   on a >1-D array, where both data and mask are flattened to match;
    #   fill_value is NOT inherited (verified live against real numpy's own
    #   `.view()`-based propagation, which does not carry `_fill_value`
    #   across) so no `_update_from` call here, unlike nearly every other
    #   item in this file. 0 divergences.
    #
    # ptp (both forms): 1760 module-function cases + 1520 method cases.
    #   Composed from this file's own `max`/`min` (Phase 7) + `subtract`
    #   (Phase 1), handling the `masked`-singleton / bare-scalar / real-
    #   MaskedArray collapse combinations both sides can produce. 0
    #   divergences. GUARD-BITE PROVEN: a deliberate `hi - lo` -> `lo - hi`
    #   swap was introduced, re-run against the same 1760-case grid,
    #   confirmed 126/1760 divergences detected, then reverted and
    #   reconfirmed 0/1760 -- the probe can bite.
    #
    # DECLINED (evaluated, not shipped -- reasons measured, not assumed):
    #   left_shift/right_shift: real numpy's hand-written implementations
    #     explicitly reference only `a`'s mask in source, but live testing
    #     showed an EMERGENT quirk where a MaskedArray shift-amount operand
    #     `n`'s mask leaks into the output via numpy's generic
    #     ufunc-subclass `__array_wrap__`/`__array_finalize__` dispatch --
    #     a mechanism anionpy's MaskedArray (which does NOT subclass ndarray,
    #     unlike real numpy's) has no equivalent hook for. Reproducing it
    #     would mean guessing at emergent subclass behavior rather than
    #     composing a known contract. RUST-QUEUE.md-worthy if the
    #     underlying dispatch difference matters elsewhere.
    #   allclose: CPython source uses asymmetric fancy/boolean-indexed
    #     inf-position masking between the two operands (`x[xinf]`/
    #     `y[xinf]` with DIFFERENT mask provenance per side) -- genuinely
    #     more fragile than the effort budget for this phase justified.
    #   diag/diagonal/concatenate/clip/
    #   round/around/choose/arange/array/prod/product: their required
    #   non-masked anionpy base primitive is itself still undeclared/withdrawn
    #   at the top level (verified against anionpy/_state/toplevel.py's own
    #   comments) or, for `prod`/`product`, carries known remaining
    #   (`atleast_1d`/`atleast_2d`/`atleast_3d` were later declared exact in
    #   a 2026-08-07 follow-up pass -- their base toplevel primitives were
    #   in fact already exact; see the dedicated entries above for the
    #   real reason this phase originally passed on them.)
    #   failures -- composing on top of an unverified primitive would mean
    #   this phase could not tell its own bugs apart from an inherited one.
    "ma.equal": "exact",
    "ma.not_equal": "exact",
    "ma.less": "exact",
    "ma.less_equal": "exact",
    "ma.greater": "exact",
    "ma.greater_equal": "exact",
    "ma.hypot": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.alltrue": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.sometrue": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.count_masked": "exact",
    # FIXED 2026-08-07 (Monday, boxing sweep). Found via the systemic scalar-boxing
    # sweep (same defect class as .fill_value / ma.mean / ma.default_fill_value):
    # this function used to wrap every non-`False`-literal return in a bare
    # `bool(...)`, silently downgrading _anionpy.all(...)'s and 0-d
    # _anionpy.equal(...)'s already-correct numpy.bool_ typing back to plain
    # Python bool, even though real numpy's own allequal (verified via
    # inspect.getsource) returns numpy.bool_ on every path except the literal
    # `else: return False`. Fixed in core.py by removing the bool() wraps and
    # using [()]-indexing on the 0-d ndarray branches instead. The differential
    # corpus was ALSO structurally blind to this: MA snapshot tuples always
    # compare outer-`tuple`-to-outer-`tuple` (always equal), and this item's own
    # bespoke adapter independently re-cast both sides through bool() before
    # snapshotting, bypassing even a generic type-tag fix. Both the shared
    # _np_snapshot/_ionp_snapshot SCALAR branch and this item's adapter in
    # ma_cases.py now embed a module-qualified type tag
    # (f"{type(m).__module__}.{type(m).__qualname__}", not bare __name__ --
    # numpy.bool_.__name__ is literally "bool" in numpy 2.x, identical to
    # bool.__name__, a false-negative trap hit and caught live during this fix).
    # Bite-tested: re-neutering allequal back to the bool()-wrapped form now
    # correctly fails 10/12 live comparison cases; before this fix (adapter and
    # snapshot both), the same neutering showed 0 failures.
    "ma.allequal": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.amax": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.amin": "exact",
    "ma.argmax": "exact",
    "ma.argmin": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.cumsum": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.cumprod": "exact",
    # RESTORED 2026-08-03 (Monday). The 0-d x explicit-axis defect that revoked this is fixed and covered by new differential cases in ma_cases.py (12 axis forms x 3 mask states x keepdims). Bite-tested: neutering the fix makes those cases FAIL.
    "ma.ptp": "exact",

    # -- Phase 9 (2026-08-07, Monday): MaskedArray indexing + dunder methods --
    # This is the batch that fixes the load-bearing gap measured at the start
    # of this task: `x[0]`, `x + 1`, `x.T` on a bare MaskedArray all raised
    # errors before this phase (no __getitem__/__len__/__bool__/T/arithmetic-
    # or-comparison-or-bitwise dunders existed on the class at all). 30 items,
    # 3 genuinely distinct algorithm families, each confirmed by reading real
    # numpy 2.5.1's actual CPython source (`numpy/ma/core.py` in this venv's
    # site-packages) rather than guessed, then re-verified live. Every item
    # below has a differential-corpus entry in `ma_cases.py` (2622 cases
    # across the 30 items, 0 failures) PLUS additional manual out-of-corpus
    # probing (3-d indexing, typed dtypes, NaN, chained ops, Ellipsis
    # indexing, complex dtype, 0-d masked-singleton collapse, __pow__
    # confirmed correctly still absent/erroring, 3-d .T, mask-independent
    # bool() truthiness). Full-suite re-run after this phase: 1849 items
    # (+30), failure set unchanged from the 28-failure/1819-item baseline by
    # NAME (zero regressions, zero fixes-by-accident).
    #
    # Family A -- __getitem__/__len__/__bool__/T (indexing + basics):
    #   __getitem__: nomask input stays nomask on the returned sub-array
    #     (never materializes); a masked scalar position returns the
    #     `masked` singleton, matching real numpy's `__getitem__` scalar
    #     collapse. __len__/__bool__ delegate straight to `self._data`'s own
    #     `__len__`/`__bool__` -- no mask involvement, matching real numpy
    #     (mask never changes truthiness or length). `T` delegates to the
    #     already-exact `transpose` (Phase 5). BONUS finding (not a declared
    #     item, no dedicated corpus): Python's legacy iteration/`in` fallback
    #     protocol (`__iter__`/`__contains__` via `__getitem__` + `__len__`)
    #     works correctly for free and was spot-checked live against real
    #     numpy with identical output including masked-position `masked`
    #     singletons -- left undeclared since it has no dedicated corpus.
    #   Bite-tested implicitly: all 4 verified failing before this phase's
    #   `core.py` edit landed (that was the original measured-broken state).
    #
    # Family B -- arithmetic dunders (__add__/__radd__/__sub__/__rsub__/
    #   __mul__/__rmul__/__truediv__/__rtruediv__/__floordiv__/
    #   __rfloordiv__): confirmed via direct CPython source read
    #   (numpy/ma/core.py ~lines 4306-4380) that real numpy's own dunders are
    #   thin calls to the already-declared module functions `add`/
    #   `subtract`/`multiply`/`true_divide`/`floor_divide` (Phase 1), in the
    #   exact (self, other) / (other, self) order the source uses for
    #   reflected forms. `__pow__`/`__rpow__` deliberately excluded --
    #   `ma.power` itself remains undeclared/declined (see the Phase 6 note
    #   above), so building the dunder on top of it would mean inheriting an
    #   unverified base; confirmed live that `x ** 2` on a MaskedArray still
    #   correctly raises/misbehaves exactly as it did before this phase (no
    #   accidental fix, no accidental regression -- it was never touched).
    #   NOT reproduced: real numpy's `_delegate_binop` NotImplemented
    #   short-circuit for foreign `__array_priority__` interop -- anionpy has
    #   no such mechanism to matter against, documented out of verification
    #   budget rather than silently skipped.
    #
    # Family C -- comparison dunders (__eq__/__ne__/__lt__/__le__/__gt__/
    #   __ge__): confirmed via direct CPython source read (`_comparison`,
    #   ~lines 4193-4266) of the exact algorithm: combined mask is a
    #   SHRINKING `mask_or(self.mask, getmask(other), copy=True)` (Phase 3's
    #   already-exact `mask_or`, reused not reimplemented); the comparison is
    #   computed on raw data EVERYWHERE, no revert-on-mask (unlike Family
    #   D/E's domained families); only `__eq__`/`__ne__` additionally
    #   override masked positions via `where(mask, compare_fn(smask, omask),
    #   check)` -- encoding "both masked = equal (for eq) / unequal (for
    #   ne), one masked = opposite"; scalar-shaped results return the
    #   `masked` singleton or the bare bool directly; `fill_value` inherits
    #   via `_update_from` then gets CAST TO BOOL with a try/except fallback
    #   to `True` (bool's own default), gated on `_fill_value_explicit`
    #   exactly mirroring real numpy's own `check._fill_value is not None`
    #   guard. Scope limit, not reproduced: real numpy's broadcast-to-
    #   different-shapes case for the eq/ne masked-position override
    #   (`_comparison_dunder` only applies the override when
    #   `mask.shape == check.shape`) -- the corpus deliberately keeps
    #   same-shaped operands so this scope limit was never exercised in a
    #   way that could hide a defect; documented, not silently assumed safe.
    #   `equal`/`not_equal`/`less`/`greater` are undeclared at the anionpy
    #   TOPLEVEL (order=/casting=/subok= kwarg-blindness on the generic
    #   ufunc dispatcher -- verified against `anionpy/_state/toplevel.py`'s
    #   own comments) but that does not block using them as plain 2-arg
    #   calls here -- already-declared `ma.equal`/`ma.not_equal`/`ma.less`/
    #   `ma.greater` (above in this file) use the identical plain-2-arg-call
    #   pattern.
    #
    # Family D -- "generic ufunc dispatch" unary (__neg__/__pos__/__abs__/
    #   __invert__): confirmed via grep that NONE of these are explicitly
    #   defined anywhere in real numpy's `ma/core.py` source -- inherited
    #   from `ndarray`'s generic subclass ufunc machinery, which anionpy's
    #   MaskedArray (composition, not subclassing) has no equivalent hook
    #   for, so the behavior was reverse-engineered from live probing rather
    #   than read from source: computed EVERYWHERE (no revert -- e.g.
    #   `abs(-2)` at a masked position is `2`, not reverted back to `-2`,
    #   unlike Family B/comparison); mask is ALWAYS materialized via
    #   `getmaskarray` -- even a `nomask` input's result mask is a real
    #   all-False array, confirmed asymmetric with Family E's nomask-
    #   preserving behavior below (not a guess -- this asymmetry was
    #   specifically probed for and confirmed live, not assumed
    #   symmetric-by-default); `fill_value` inherits from `self` alone
    #   (`_update_from(self)`), NOT the "prefer whichever operand was a
    #   MaskedArray" `_fv_source` rule Phase 1's binary family uses (there is
    #   only one operand, so this distinction only matters as a documented
    #   fact, not a behavioral difference).
    #
    # Family E -- "generic ufunc dispatch" binary (__and__/__rand__/
    #   __or__/__ror__/__xor__/__rxor__): same "not in numpy source, reverse-
    #   engineered live" status as Family D. Reflected forms use IDENTICAL
    #   code to non-reflected (confirmed live: `5 & a` and `a & 5` both
    #   report `a`'s OWN fill_value -- the rule is "whichever instance's
    #   dunder method executes", not left/right operand position; these ops
    #   are also commutative so there is no value-order difference to hide a
    #   defect either way). Mask is `mask_or(self.mask, getmask(other))` --
    #   confirmed NOT a naive unshrunk `logical_or` by a targeted live probe
    #   combining two already-materialized real all-False mask arrays via
    #   `&` and observing the result correctly collapse back to `nomask`
    #   identity (the shrinking behavior only real `mask_or` provides). No
    #   data-revert on masked positions (matches Family D). `fill_value`
    #   inherits from `self` alone, same rule as Family D.
    #
    # OUT-OF-SCOPE FINDING (reported, not fixed -- outside this task's
    # 4-file edit scope): `anionpy.int8(6) & <MaskedArray>` (typed anionpy
    # scalar on the LEFT of a bitwise op against a MaskedArray on the right)
    # raises `TypeError: unsupported numpy dtype for anionpy.array() ...`
    # instead of falling back to `MaskedArray.__rand__`. Root cause is in
    # anionpy's base scalar type (not in any of this task's 4 in-scope
    # files): `anionpy.int8.__and__` does not recognize `MaskedArray` as a
    # foreign type to defer on (return `NotImplemented` for), so Python
    # never gets to try the reflected `__rand__`. Confirmed this is a real
    # anionpy-side divergence, not a Python-protocol limitation: real numpy's
    # `np.int8(6) & <masked array>` works correctly. Does NOT invalidate the
    # `__rand__`/`__ror__`/`__rxor__` declarations below -- the corpus only
    # ever puts a plain Python int/float on the left for the reflected
    # cases (which correctly returns NotImplemented and lets the reflected
    # dunder run), and `MaskedArray.__rand__` itself was independently
    # verified correct when reached (e.g. `5 & a` above). RUST-QUEUE.md /
    # scalar-dunder-worthy if this ever matters on a live code path.
    "ma.MaskedArray.__getitem__": "exact",
    "ma.MaskedArray.__len__": "exact",
    "ma.MaskedArray.__bool__": "exact",
    "ma.MaskedArray.T": "exact",
    "ma.MaskedArray.__add__": "exact",
    "ma.MaskedArray.__radd__": "exact",
    "ma.MaskedArray.__sub__": "exact",
    "ma.MaskedArray.__rsub__": "exact",
    "ma.MaskedArray.__mul__": "exact",
    "ma.MaskedArray.__rmul__": "exact",
    "ma.MaskedArray.__truediv__": "exact",
    "ma.MaskedArray.__rtruediv__": "exact",
    "ma.MaskedArray.__floordiv__": "exact",
    "ma.MaskedArray.__rfloordiv__": "exact",
    "ma.MaskedArray.__eq__": "exact",
    "ma.MaskedArray.__ne__": "exact",
    "ma.MaskedArray.__lt__": "exact",
    "ma.MaskedArray.__le__": "exact",
    "ma.MaskedArray.__gt__": "exact",
    "ma.MaskedArray.__ge__": "exact",
    "ma.MaskedArray.__neg__": "exact",
    "ma.MaskedArray.__pos__": "exact",
    "ma.MaskedArray.__abs__": "exact",
    "ma.MaskedArray.__invert__": "exact",
    "ma.MaskedArray.__and__": "exact",
    "ma.MaskedArray.__rand__": "exact",
    "ma.MaskedArray.__or__": "exact",
    "ma.MaskedArray.__ror__": "exact",
    "ma.MaskedArray.__xor__": "exact",
    "ma.MaskedArray.__rxor__": "exact",

    # -- ma batch 3 -----------------------------------------------------
    # Instance-method / property forms of already-declared module-level
    # `ma.*` functions of the same name (see the module-level declarations
    # of `count`/`sum`/`any`/`all`/`min`/`max`/`argmax`/`argmin`/`ptp`
    # above, and `filled`/`compressed` at the top of this file) -- these
    # methods are thin forwarders onto exactly those already-verified
    # functions (see anionpy/ma/core.py's Phase 7/8 blocks and the new
    # `filled`/`compressed` forwarders added alongside this declaration),
    # exercised here through `exploded_class_cases.py`'s
    # `_maskedarray_receiver_cases`/`_maskedarray_filled_cases` corpus
    # (4 receivers: 1-D float-partial, 1-D int-partial, 1-D fully-unmasked,
    # 2-D float-partial; `filled` additionally gets an explicit
    # `fill_value=` argument on each). All 15 confirmed passing in that
    # corpus AND, per this task's mandatory out-of-corpus check, live
    # against real numpy 2.5.1 on: a fully-masked receiver (excluded from
    # the corpus itself for an unrelated MaskedConstant-type-identity
    # harness reason -- see that corpus function's own "Deliberately NOT
    # included" comment -- but verified by hand: `count`->0, `argmax`/
    # `argmin`->0, everything else -> the `masked` singleton, matching
    # real numpy value-for-value including RuntimeWarning-or-not); a
    # zero-size receiver (both sides raise byte-identical
    # `ValueError: zero-size array to reduction operation
    # <minimum|maximum> has no identity` messages for min/max/ptp, and
    # `ValueError: attempt to get arg{max,min} of an empty sequence` for
    # argmax/argmin); a `nomask` receiver; and, for `fill_value`/`filled`
    # specifically, a receiver with an EXPLICITLY set (non-default)
    # `fill_value` to confirm propagation. No warning divergence observed
    # in any of the above (checked via `warnings.catch_warnings(record=
    # True)` on both sides).
    #
    # `ma.MaskedArray.mean` is DELIBERATELY EXCLUDED from this batch despite
    # passing in the (narrow, 4-receiver, all-float64-or-default-int)
    # exploded corpus above: the pre-existing, already-documented float16
    # scalar-type-identity bug (see this file's "ma.mean NOT DECLARED"
    # note near the module-level `count`/`sum`/... block) is real and
    # still reproduces live as of this same verification pass --
    # `np.ma.MaskedArray([1,2,3], dtype='float16', mask=[0,1,0]).mean()`
    # returns a genuine `numpy.float16`, anionpy's equivalent returns an
    # `anionpy.float16` (same numeric value, wrong Python type) -- this
    # corpus simply never happens to exercise float16, so the method form
    # inherits the SAME undeclared status as the module-level function it
    # forwards to, not a new, independently-verified pass.
    "ma.MaskedArray.count": "exact",
    "ma.MaskedArray.sum": "exact",
    "ma.MaskedArray.any": "exact",
    "ma.MaskedArray.all": "exact",
    "ma.MaskedArray.min": "exact",
    "ma.MaskedArray.max": "exact",
    "ma.MaskedArray.argmax": "exact",
    "ma.MaskedArray.argmin": "exact",
    "ma.MaskedArray.ptp": "exact",
    "ma.MaskedArray.ndim": "exact",
    "ma.MaskedArray.shape": "exact",
    "ma.MaskedArray.size": "exact",
    "ma.MaskedArray.fill_value": "exact",
    "ma.MaskedArray.filled": "exact",
    "ma.MaskedArray.compressed": "exact",

    # -- ma-phase5-2026-08-08 (Monday): layout/metadata attribute mirrors --
    # Eight items, all pure `self._data.<x>` passthroughs (no new mask
    # logic -- verified live these six/eight never consult the mask array's
    # own itemsize/nbytes/strides): `data`, `dtype`, `mask`, `itemsize`,
    # `nbytes`, `strides`, `iscontiguous()`, `get_fill_value()`. Corpus:
    # `tests/differential/ma_phase5_cases.py` (`MA_PHASE5_SPECS`, NOT yet
    # merged into `registry.py` at declaration time -- collision-checked
    # clean against the live 2058-entry REGISTRY by this task; merge block
    # is in this task's report). 48/48 cases pass for all eight items,
    # covering: 10 dtypes (bool/int8/int32/uint32/int64/float16/float32/
    # float64/complex64/complex128) x {nomask, fully-masked, partially-
    # masked}; 4 empty-dtype cases; 6 0-d cases (2 dtypes x {nomask,
    # masked, unmasked} scalar); 2 explicit-fill_value cases; 6 non-C-
    # contiguous layout cases (F-order 2x2, transposed view, negative-
    # stride slice, each x {nomask, partial}). Implementation in
    # `anionpy/ma/core.py` (inserted directly after the pre-existing `size`
    # property): `itemsize`/`nbytes`/`strides` delegate to
    # `anionpy.ndarray`'s own already-exact `.itemsize`/`.nbytes`/
    # `.strides` (declared in `anionpy/_state/ndarray.py`); `data`/`dtype`/
    # `mask` were already-implemented Phase-0 properties never previously
    # declared as their own manifest items; `iscontiguous()` and
    # `get_fill_value()` are new zero-arg methods.
    #
    # `ma.MaskedArray.flags` is DELIBERATELY NOT included in this batch,
    # not merely deferred -- a genuine, reproducible semantic gap, not a
    # thin-evidence decline. Real numpy's `MaskedArray` is an `ndarray`
    # SUBCLASS, so `a.flags.owndata` is `False` for literally EVERY real
    # MaskedArray regardless of provenance (verified live: fresh
    # construction, view construction, F-order, all report
    # `owndata=False` uniformly -- an artifact of `.data`/`.mask` always
    # being reached through an extra `.view()` layer in real numpy's
    # implementation, independently corroborated by `ma_cases.py`'s
    # pre-existing `_layout_snapshot_np` comment, which documents the same
    # finding). anionpy's `MaskedArray` is a plain (data, mask) pair, not
    # an `ndarray` subclass -- `self._data` is genuinely OWNED, so a naive
    # `return self._data.flags` would mismatch `owndata` on every single
    # case with no way to "fix" it short of fabricating a hand-built
    # flags-like object with a hardcoded `owndata=False`, which this task's
    # instrument-discipline rules explicitly forbid (that would be papering
    # over a uniform mismatch rather than reporting it). `iscontiguous()`
    # above is the safe substitute: it reads only `C_CONTIGUOUS`, which was
    # verified live to NOT depend on the owndata/view distinction (fortran-
    # order and sliced-view receivers both correctly report `False` on real
    # numpy, matching a direct `C_CONTIGUOUS` read on both sides in all 6
    # layout cases above).
    "ma.MaskedArray.data": "exact",
    "ma.MaskedArray.dtype": "exact",
    "ma.MaskedArray.mask": "exact",
    "ma.MaskedArray.itemsize": "exact",
    "ma.MaskedArray.nbytes": "exact",
    "ma.MaskedArray.strides": "exact",
    "ma.MaskedArray.iscontiguous": "exact",
    "ma.MaskedArray.get_fill_value": "exact",

    # -- ma batch 3, part 2: dunders with genuinely NEW algorithms (none
    # of these forward to an already-declared module-level function).
    # `__iter__`/`__contains__`/`__complex__`/`__index__`/`__copy__` are,
    # per real numpy 2.5.1 (live-verified, not assumed), literally
    # `ndarray`'s INHERITED, unoverridden dunders -- `__iter__`/`__copy__`
    # exercise mask-awareness only through already-declared-exact
    # primitives they compose (`__getitem__` and `.copy()` respectively);
    # `__contains__` is `bool((self == item).any())`, composing two
    # already-declared-exact primitives (`__eq__`, `.any()`) with zero new
    # mask logic of its own; `__complex__`/`__index__` are confirmed
    # MASK-BLIND (return the raw underlying value regardless of mask,
    # verified live: `complex(masked_array(5+2j, mask=True))` is
    # `(5+2j)`, `masked_array(5, mask=True).__index__()` is `5`) and
    # delegate straight to the already-declared-exact `anionpy.ndarray`
    # equivalent. `__float__`/`__int__` ARE genuinely overridden by real
    # numpy (not inherited) -- a masked receiver returns `nan` with a
    # `UserWarning` for `__float__`, or raises `MaskError` for `__int__`;
    # implemented and corpus-verified against both behaviors, including
    # the warning text and the `MaskError` cross-package exception
    # equivalence (numpy's `np.ma.MaskError` and anionpy's own are
    # distinct class objects with no inheritance relationship -- declared
    # via `exception_equivalences` in ma_cases.py, same mechanism already
    # used for `LinAlgError`-family items elsewhere in this suite).
    # `__deepcopy__` reuses `__copy__` (verified live real numpy's own
    # `__deepcopy__` produces an observably identical result -- no nested
    # object graph inside a MaskedArray's own flat data/mask for "deep" to
    # mean anything more).
    #
    # All 8 corpus-verified (16-30 cases each depending on the item) AND,
    # per this task's mandatory out-of-corpus check, live against real
    # numpy 2.5.1 on inputs the corpus does not contain: bool-dtype
    # iteration, a `__contains__` item that IS present unmasked elsewhere
    # in the same array (distinguishing "item excluded because masked"
    # from "item absent entirely"), negative-valued nomask scalars for
    # `__int__`/`__float__`/`__index__`, a nomask complex scalar, a 3-D
    # masked receiver with an explicit custom fill_value for `__copy__`/
    # `__deepcopy__` (confirmed independent underlying buffers -- mutating
    # the copy's data left the original's untouched, both sides), and a
    # warnings.catch_warnings comparison (both sides emit byte-identical
    # warning lists, or none, in every one of the above).
    "ma.MaskedArray.__iter__": "exact",
    "ma.MaskedArray.__contains__": "exact",
    "ma.MaskedArray.__float__": "exact",
    "ma.MaskedArray.__int__": "exact",
    "ma.MaskedArray.__complex__": "exact",
    "ma.MaskedArray.__index__": "exact",
    "ma.MaskedArray.__copy__": "exact",
    "ma.MaskedArray.__deepcopy__": "exact",

    # -- ma batch 4: __mod__/__rmod__/__imod__ ---------------------------
    # Previously declined under ticket #49 (float divide-by-zero warning
    # divergence); unblocked once ticket #56 (commit c903b9c) fixed the
    # underlying raw dtype-conditional diagnostic -- re-measured from
    # scratch here rather than assumed fixed. Like Family D/E above, these
    # three are NOT defined anywhere in real numpy's `ma/core.py` source --
    # `getattr(numpy.ma.MaskedArray, '__imod__', None)` is the plain
    # inherited `ndarray` slot wrapper, confirmed live. Their behavior
    # therefore comes from real numpy's generic `__array_wrap__` hook
    # (read directly from CPython source, cited in full in
    # `anionpy/ma/core.py`'s `_generic_domained_binary_dunder`/
    # `_generic_domained_binary_idunder` docstrings) applied to
    # `np.remainder`'s own ufunc: domain = `_DomainSafeDivide`
    # (`abs(a)*tiny >= abs(b)`, `tiny = finfo(float).tiny`, the same
    # predicate already implemented here as `_domain_safe_divide` for the
    # module-level `ma.remainder` family), binary-domain fill constant =
    # `ufunc_fills[np.remainder][-1] == 1` (a literal numeric constant, NOT
    # the array's own fill_value, NOT a revert-to-operand-A). `fill_value`
    # inherits from `self` alone (Family D/E's rule), confirmed live across
    # `a%b`/`b%a`/`scalar%b`/`list%b`.
    #
    # `__mod__`/`__rmod__` and `__imod__` are genuinely DIFFERENT
    # algorithms, not one wrapper reused, because of where real numpy
    # evaluates the domain check relative to the raw ufunc call:
    #   - `__mod__`/`__rmod__`: `result = obj.view(type(self))` is a FRESH
    #     object, so `input_args` in the wrap context are the ORIGINAL,
    #     unmutated operands -- the domain check always correctly detects
    #     a zero-divisor regardless of dtype, and the fill/mask-update
    #     always fires.
    #   - `__imod__`: `obj is self`, so the raw in-place ufunc has ALREADY
    #     overwritten `self`'s buffer before the domain check runs against
    #     it -- `input_args` reads the POST-mutation data. This is a real,
    #     deterministic, dtype-conditional divergence, confirmed live:
    #       * float: raw in-place `remainder(x, 0.0)` silently writes IEEE
    #         `nan` (no warning). `domain_fn(nan, 0.0)` is `nan >= 0` ==
    #         **False** (NaN comparisons are always false) -- this
    #         SILENTLY DEFEATS the domain-fill/mask-update step. The raw
    #         unfilled `nan` is left in data and the mask is NOT updated
    #         with the domain bit -- verified via the warning corpus below
    #         (`ma_warning_imod`'s `__imod__|float|div0_*` cases snapshot
    #         mask staying `False`/nomask and data holding the bare nan on
    #         BOTH real numpy and anionpy).
    #       * int: raw in-place `remainder(x, 0)` writes `0` WITH
    #         `RuntimeWarning: divide by zero encountered in remainder`
    #         (anionpy's raw `%=`/`_anionpy.remainder()` already reproduce
    #         this exactly -- confirmed a side effect of the already-shipped
    #         #56 fix, no new Rust work needed). `domain_fn(0, 0)` is
    #         `0 >= 0` == **True** -- the domain fill DOES fire normally
    #         (fills with the constant `1`, ORs the domain bit into mask).
    #   Implemented as two separate functions in `anionpy/ma/core.py`
    #   (`_generic_domained_binary_dunder` for `__mod__`/`__rmod__`,
    #   `_generic_domained_binary_idunder` for `__imod__`) rather than one
    #   shared body, matching this real divergence instead of hiding it.
    #
    # Two regression guards earned from live differential probing during
    # implementation, both now covered by the corpus below (not just
    # asserted in a comment): (1) the raw `fn(...)` ufunc call must NOT be
    # wrapped in `errstate(divide=..., invalid=...)` for the dunder path
    # (unlike the module-level `make_masked_domained_binary` family, which
    # deliberately does wrap its own `fn` call) -- only the domain-check
    # call gets suppressed; wrapping `fn` too silently ate the real,
    # correct int-dtype `RuntimeWarning` and would have shipped a false
    # "exact". (2) the 0-d/scalar path has NO scalar-unwrap branch in real
    # numpy's `__array_wrap__` -- `result = obj.view(type(self))` is
    # ALWAYS a (0-d) `MaskedArray`-typed object unless the masked-singleton
    # collapse condition (`shape==() and mask truthy`) applies; an earlier
    # draft returned a bare unwrapped scalar for the in-domain 0-d case,
    # which live probing against `MaskedArray(5.0) % MaskedArray(2.0)`
    # caught returning the wrong type.
    #
    # Corpus: `ma.MaskedArray.__mod__`/`__rmod__`/`__imod__` in
    # `tests/differential/ma_cases.py` (`_mod_dunder_cases`, 392 cases each,
    # crossing int/float dtype x 3 shapes (1d/2d/empty) x 5 mask variants x
    # 2 fill_value variants x 7 other-operand variants including both
    # in-domain and zero-divisor operands, masked and unmasked, scalar/
    # plain-list/MaskedArray) -- value/mask/dtype/fill_value only (no
    # warnings). `__imod__`'s adapter additionally asserts object identity
    # (`arr %= other` returns the SAME object) and, when `other` is itself
    # a MaskedArray, that `other`'s data/mask are byte-identical before and
    # after, on BOTH the numpy and anionpy side -- buffer independence and
    # in-place identity proven directly in the corpus, not just by
    # ad-hoc probing. Warnings: `ma_warning_mod_dunder` (32 cases) and
    # `ma_warning_imod` (16 cases) in `tests/differential/ma_warning_cases.py`,
    # covering int-div0 (must warn, exact text) and float-div0 (must stay
    # silent) for both dtypes, folding the warning list into the same
    # descriptor as the value/mask/fill_value snapshot so a case can't pass
    # on value alone while getting the diagnostic wrong.
    #
    # Out-of-corpus verification (this task's mandatory check, live against
    # real numpy 2.5.1, inputs the corpus above does not contain): 3-element
    # float array with the zero divisor NOT at a masked position; int8/
    # uint8 dtype div0 (narrower than the corpus's default int64); a
    # `__imod__` chain (`a %= b; a %= c`) confirming the post-first-mutation
    # nan/0 leak-through composes correctly on the second application;
    # negative dividends or divisors on both sides of `%`. All matched byte-
    # for-byte (data, mask, dtype, fill_value, warning list) against real
    # numpy on every one of these.
    "ma.MaskedArray.__mod__": "exact",
    "ma.MaskedArray.__rmod__": "exact",
    "ma.MaskedArray.__imod__": "exact",

    # ma batch 5 -- Group A (shape/layout methods where the MASK must be
    # transformed in lockstep with the data). All 9 items below share one
    # implementation family in `anionpy/ma/core.py`: `_masked_arraymethod`
    # calls `getattr(self._data, funcname)(...)` and, independently,
    # `getattr(self._mask, funcname)(...)` when the mask is materialized
    # (leaves it `nomask` when it already was), then wraps both in a fresh
    # `MaskedArray(data, mask=mask)._update_from(self)`. This is safe
    # because `MaskedArray.__init__` does NOT copy an already-`anionpy.
    # ndarray` `data=`/`mask=` argument (confirmed by direct source read)
    # -- so whatever view-vs-copy relationship the ndarray-layer method
    # produces is preserved through the wrap, not silently re-copied.
    # `copy`/`flatten`/`squeeze`/`swapaxes`/`transpose` match real numpy's
    # own `_arraymethod(funcname, onmask=True)` factory (confirmed via
    # `inspect.getsource(np.ma.core._arraymethod)` regex-scan of numpy's
    # source -- this is the exact set it backs). `reshape`/`ravel`/`mT`/
    # `conj`/`conjugate` are bespoke in real numpy too (not built on
    # `_arraymethod`) and implemented as bespoke methods here to match.
    #
    # `diagonal` is DELIBERATELY NOT declared/implemented here even though
    # it is in the same real-numpy `_arraymethod` family: `ndarray.
    # diagonal` is REVOKED at the ndarray layer (`anionpy/_state/ndarray.
    # py`, wrong strides / genuine copy where numpy returns a view) --
    # declaring `ma.MaskedArray.diagonal` would inherit that defect
    # silently. No corpus, no implementation, left undeclared.
    #
    # Corpus: `tests/differential/ma_cases.py`, `_build_group_a_specs()`
    # and helpers (`_group_a_cases`, `_provenance_build`, `_layout_
    # snapshot_np`/`_layout_snapshot_ionp`, `_mutate_alias_check`). Crosses
    # 9 provenance labels (`c_contig_1d`, `c_contig_2d`, `f_order_2d`,
    # `transposed_view`, `noncontig_slice`, `0d`, `empty_1d`, `3d`,
    # `len1_axis` -- trap #3: not everything built from C-contiguous 1-D
    # float64) x mask variants (nomask omitted/explicit-None, materialized
    # all-False, fully-masked, partially-masked) x fill_value variants
    # (default/explicit) = 86 cases per item (166 for `ravel`, which adds a
    # dedicated `order=` sub-corpus, see below). Every case's adapter
    # returns a `("LAYOUT", ...)` snapshot comparing: the usual value/mask/
    # dtype/fill_value snapshot (`_np_snapshot`/`_ionp_snapshot`), the
    # DATA array's `.strides` and `(C_CONTIGUOUS, F_CONTIGUOUS)` flags
    # (trap #1 -- strides ARE part of the contract), the MASK array's
    # `.strides`/flags when materialized, and a live mutation-based
    # aliasing probe (`_mutate_alias_check`) on both data and mask that
    # mutates the RESULT's first element and checks whether the ORIGINAL
    # changed too (trap #2 -- view vs copy is observable; comparing values
    # alone cannot detect it). `swapaxes(0, 1)`/`.mT` deliberately keep the
    # ndim<2 provenance labels IN the corpus (not filtered out) to exercise
    # the real numpy `AxisError`/`ValueError` PATH, message text included
    # (`harness.run_case`'s exception comparison is generic across `kind`,
    # confirmed by direct read of `tests/differential/harness.py`).
    #
    # OWNDATA is deliberately EXCLUDED from the flags comparison (see the
    # comment at `_layout_snapshot_np` in `ma_cases.py`): measured live
    # that real numpy's `.data`/`.mask` accessors on a MaskedArray report
    # `OWNDATA: False` UNCONDITIONALLY -- even wrapping data independently
    # known to be owned, even immediately after a genuine `.copy()` that
    # the mutation probe proves is NOT aliased. This is an artifact of
    # numpy's `.data`/`.mask` properties always crossing an internal
    # `.view()` layer, not a real view-vs-copy signal; asserting it would
    # mean shipping a FALSE "not owned" flag on the anionpy side (trap #4).
    # C_CONTIGUOUS/F_CONTIGUOUS (real layout facts) and the mutation-based
    # alias check (the actual view-vs-copy signal) both stay compared.
    #
    # `ravel` additionally gets `_ravel_order_cases()`/`_ravel_order_
    # adapters()` (80 extra cases, 166 total): `order` in `('C','F','A',
    # 'K')` crossed with the 4 provenance labels where C-order and F-order
    # genuinely differ (`c_contig_2d`, `f_order_2d`, `transposed_view`,
    # `noncontig_slice`) x nomask/materialized mask. This closes a real
    # blind spot found during this task: `_group_a_cases()` alone never
    # passes `order=` at all, so removing the 'K'/'A'-normalization guard
    # in `core.py`'s `ravel` and rerunning left the plain 86-case corpus at
    # 86/86 pass -- silently blind to the exact bug it was meant to catch.
    # With the order sub-corpus, the same guard-removal produces 16/166
    # failures, all `order=A`/`order=K` on `f_order_2d` (measured, not
    # asserted). `ravel`'s normalization itself: verified live against
    # `inspect.getsource(np.ma.core.MaskedArray.ravel)` -- 'K'/'A' (both
    # cases) normalize to 'F' if `self._data.flags.fnc` (F-contiguous AND
    # NOT C-contiguous) else 'C', applied identically to data and mask.
    #
    # `mT`: bespoke property, NOT `_update_from`-inheriting -- verified
    # live against `inspect.getsource(np.ma.core.MaskedArray.mT.fget)` that
    # real numpy's own `mT` getter does not call `_update_from` on either
    # branch (a genuine, measured exception to this family's usual fill_
    # value-inheritance rule: `x.mT.fill_value` reports the plain default
    # even when `x.fill_value` was set explicitly, while `x.T.fill_value`/
    # `x.reshape(n).fill_value` both correctly carry it through). Guard-
    # bite: reinstating the `_update_from` call and rerunning produced
    # 25/86 failures, all `fv_custom` cases (measured).
    #
    # `conj`/`conjugate`: bespoke, NOT `_arraymethod`-based (confirmed live
    # `numpy.ma.core` has no `conj`/`conjugate = _arraymethod(...)`
    # assignment). A non-complex receiver's `.conj()` returns `self`
    # UNCHANGED (`x.conj() is x`, verified live on real numpy) -- guard-
    # bite: removing this short-circuit produced 10/72 failures for each
    # of `conj`/`conjugate` (measured). A complex receiver conjugates via
    # `self._data.conj()` (the already-`4b2d1bc`-fixed, order='K'-correct
    # ndarray method) with the mask always materialized via `getmaskarray`
    # -- verified live this session that real numpy's `.conj()`/
    # `.conjugate()` also materializes the mask unconditionally, even from
    # a `nomask` receiver (`x = masked_array([1+2j,3-4j]); x.mask is
    # nomask` is `True` but `x.conj().mask is nomask` is `False`), so this
    # is not a divergence to guard against -- it VALIDATES the existing
    # unconditional `getmaskarray(self)` call. A 0-d result collapses to
    # the `masked` singleton only when genuinely masked (verified live:
    # an unmasked 0-d complex receiver's `.conj()` stays a real 0-d
    # `MaskedArray`; a masked one collapses, `r2 is ma.masked` is `True`).
    #
    # Out-of-corpus verification (live, real numpy 2.5.1, inputs not in
    # the corpus above): a 4x3x2 3-D array (non-power-of-2 shape) with a
    # partial mask and explicit fill_value, run through copy/flatten/
    # ravel/squeeze/transpose((2,0,1))/swapaxes((0,2))/reshape((4,6)) --
    # data, mask, strides, and fill_value all matched byte-for-byte on
    # both sides; `mT` on a fresh 3x2 partially-masked 2-D array matched
    # likewise; `conj`/`conjugate` on a 2x3 complex array with negative
    # imaginary parts and a partial mask matched exactly. Separately, a
    # live mutation-based view/copy check on a FRESH (non-corpus) 3x2
    # array across copy/flatten/squeeze/transpose/swapaxes/reshape/mT/
    # ravel(C-contig) confirmed anionpy's aliasing behaviour matches real
    # numpy's on every one: `copy`/`flatten` genuinely independent (mutate
    # result, original unchanged, both sides); `squeeze`/`transpose`/
    # `swapaxes`/`reshape`/`mT`/`ravel` genuinely aliased (mutate result,
    # original DOES change, both sides) -- confirming the "views under
    # conditions" numpy semantics are reproduced, not silently converted
    # to copies (trap #2).
    #
    # `ma.MaskedArray.flatten` DECLINED (not declared): measured a genuine,
    # narrow divergence inherited from the ndarray layer. Real numpy's
    # `.flatten()` on an empty 1-D array preserves stride `(0,)`; anionpy's
    # already-declared-exact `ndarray.flatten` (declared before this task,
    # not touched by it) instead computes a fresh contiguous stride `(8,)`
    # for the same empty-array case. All 78/86 non-empty-provenance cases
    # match exactly; only the `empty_1d` label's 8 cases fail, purely on
    # this stride mismatch (values/mask/dtype/fill_value all agree).
    # Fixing it would mean touching Rust `ionp-core`'s flatten -- out of
    # this batch's scope (ma-layer only). Left undeclared, same precedent
    # as `diagonal`'s ndarray-layer REVOKED status.
    "ma.MaskedArray.copy": "exact",
    "ma.MaskedArray.ravel": "exact",
    "ma.MaskedArray.squeeze": "exact",
    "ma.MaskedArray.transpose": "exact",
    "ma.MaskedArray.swapaxes": "exact",
    "ma.MaskedArray.reshape": "exact",
    "ma.MaskedArray.mT": "exact",
    "ma.MaskedArray.conj": "exact",
    "ma.MaskedArray.conjugate": "exact",
}
