# Umbrella-bucket ledger gap (2026-08-13)

## The mechanism

`tools/coverage.py::load_test_results` credits an item purely by **exact
string match** between a differential-suite JSON key and the item's name
(`RANDOM_STATE`/`__ion_state__` key) -- see `build_ledger`'s
`elif item not in passed: verdict = "untested"` branch. There is no check
that the JSON key's underlying cases actually isolate *that specific*
surface item.

Most of `tests/differential/*_cases.py` avoids this by construction: a
loop builds one `ItemSpec` per real surface item (see
`exploded_class_cases.py`'s `for _method, _args in
_GENERATOR_METHOD_ARGS.items(): EXPLODED_CLASS_SPECS[f"random.Generator.
{_method}"] = ItemSpec(...)` -- one JSON key per method, each key's cases
exercise only that method).

A different, narrower pattern exists in a handful of files: **one shared
dispatch function, one JSON key, many conceptually-different sub-behaviors
multiplexed inside it via an `op`/method-name string carried in each
case tuple.** `tests/differential/random_cases.py`'s `generator_cases()`
is the example that surfaced this: it returns `(seed, op, opargs,
opkwargs)` tuples covering ~20 different `Generator` methods
(`op="beta"`, `op="wald"`, `op="poisson"`, ...), all routed through
`_generator_np`/`_generator_ionp`'s `getattr(g, op)(*opargs, **opkwargs)`,
and all reported under **one** JSON key: `"random.Generator"`.

**The gap this creates is bidirectional, and the dangerous direction is
not the one anyone was looking for:**

- *Under-crediting* (what actually happened this session): declaring
  `random.Generator.wald` etc. in `RANDOM_STATE` while their only
  passing cases lived inside the `"random.Generator"` umbrella key meant
  `coverage.py` correctly, if confusingly, showed them `untested` --
  `"random.Generator.wald" not in passed` is true even though cases
  for `wald` ran and passed, because they were filed under a
  differently-spelled key. Annoying, but safe: nothing was falsely
  credited.
- *Over-crediting* (the failure mode worth guarding against): if the
  umbrella key's literal string **is itself ever declared** as an
  `__ion_state__` item (nothing stops `RANDOM_STATE["random.Generator"]
  = "exact"` from being written), `coverage.py` would credit it "exact"
  on the strength of `generator_cases()`'s aggregate pass, which is a
  looser bar than a dedicated per-item corpus: a single case regression
  inside one method could be masked by the bucket rule (`fold_axis_
  failures`/pass-set membership is whole-key, not per-case), and there
  would be no per-method regression signal at all -- exactly the
  structure the coordinator's stream-position and variable-trip-count
  warnings are about, just at the bookkeeping layer instead of the
  algorithm layer.

## What was checked, and the result

Cross-referenced every currently-declared `__ion_state__` entry (all
`anionpy/_state/*.py` `*_STATE` dicts feeding `anionpy/_state/__init__.py`)
against every differential-suite file using this multi-op-under-one-key
shape (found by grepping for `getattr(<obj>, op)` / `def _..._np(op` /
`, op,` argument patterns across `tests/differential/*.py`):

- **`random_cases.py`** (`generator_cases()` -> key `"random.Generator"`):
  the umbrella key itself is intentionally kept **undeclared** (see
  `anionpy/_state/random.py`'s "bare class item stays undeclared"
  section) -- currently harmless, but was the closest live example of the
  dangerous shape, which is why this got caught at all.
- **`ma_warning_cases.py`**: also multiplexes several dunder/domain
  cases per bucket, but the bucket *names* are synthetic composites
  (`"ma_warning_domained_unary"`, `"ma_warning_dunder"`,
  `"ma_warning_mod_dunder"`, ...) that do not collide with any real
  numpy surface item string, so no `__ion_state__` entry can ever
  accidentally rest on one.
- **`inplace_cases.py`/`inplace_alias_cases.py`**: same shape
  (`getattr(a, op)(b)`), same synthetic composite-key pattern
  (`"alias/self/__ipow__"` etc., seen directly in this session's own
  FAILURE-BASELINE diff work) -- not a numpy surface string, no
  collision risk.
- **`view_semantics_cases.py`**: has an `op` parameter (`_probe(mod,
  probe, op, multi)`), but `make_adapters(op, ...)` is called once per
  real item and *augments that item's own existing adapter* rather than
  filing multiple items under one shared key -- this is architecturally
  the same safe one-key-per-item shape as `exploded_class_cases.py`, not
  the umbrella shape.
- **`array_protocol_cases.py`, `arrayapi_cases.py`, `floordiv_crossing_
  cases.py`, `fperr_cases.py`, `linalg_cases.py`, `ma_cases.py`,
  `ndarray_float_cases.py`, `manip_cases.py`, `memmap_cases.py`,
  `poly1d_legacy_cases.py`, `ndarray_attrs_cases.py`, `order_cases.py`,
  `products_io_cases.py`, `reduction_cases.py`, `setops_cases.py`,
  `reshape_order_cases.py`, `setitem_cases.py`, `strings_cases.py`,
  `strparam_cases.py`, `ufunc_cases.py`**: grepped for the same
  `getattr(_, op)` / `def _fn(op` shapes; every hit resolved to a
  per-item factory function (e.g. `memmap_cases.py`'s `_unary_spec(name,
  op, ...)`, called once per real item to build one `ItemSpec` each),
  not a shared multi-item bucket. **Not individually verified case-by-
  case beyond confirming the factory-per-item shape** -- a deeper audit
  of each file's full case list was out of scope for this pass.

**Conclusion at time of writing: no currently-declared `__ion_state__`
item is resting on an umbrella bucket that doesn't feed the per-item
ledger.** The one bucket with this shape (`"random.Generator"`) is
correctly undeclared, and the fix applied this session
(`exploded_class_cases.py`'s `_GENERATOR_METHOD_ARGS` gaining per-method
entries) is the same pattern the rest of the suite already uses --
this was a gap in *this one file's* coverage, not a structural hole in
`coverage.py` itself.

## Recommendation

`coverage.py` has no built-in guard against a future `__ion_state__`
declaration whose only backing JSON key is a multi-op umbrella bucket --
the check above was manual and file names must exercise it again before
believing a fresh audit. A cheap structural guard would be: at the point
a `*_cases.py` file defines a shared `op`-dispatch function, assert (in
a `test_selftest.py`-style check) that its bucket key is not also a
literal key in any `__ion_state__` dict, or, if it must be, that the
bucket's own differential harness entry is marked so `coverage.py`
refuses to credit it without a per-op breakdown. Not implemented here --
flagging for whoever owns `coverage.py`/`registry.py`'s invariants next.
