# The manifest does not descend into submodules — Phase 2's denominator is still incomplete

**Date:** 2026-08-06
**Measured against:** numpy 2.5.1 live, and `tools/numpy_surface.json` as committed.
**Status:** CLOSED for the `np.polynomial.*` layer — see
`docs/DENOMINATOR-SUBMODULE-FIX-2026-08-06.md` for the fix, the measured
before/after numbers, and the re-adjudication of every other candidate
namespace named below (`ma.core`, `random.mtrand`, `lib.*`, ... — all
EXCLUDED, not silently dropped: see that doc for why each one is a re-export
or was left unadjudicated for lack of evidence).
**Bearing:** Phase 2 ("fix the manifest to a true denominator") is finished
for the submodule-descent bias identified here. A blind full-numpy recursive
crawl was and remains explicitly out of scope (see "What must NOT be
inferred" below, unchanged).

## The finding

`tools/snapshot_surface.py`'s `surface()` skips module objects outright
(`if inspect.ismodule(o): continue`), and `SUBMODULES` treats `polynomial` as
a single flat level. So the manifest never descends into
`numpy.polynomial.chebyshev` and its siblings.

Measured, per submodule — public names vs. how many the manifest tracks:

```
np.polynomial.chebyshev   public= 36  tracked= 0
np.polynomial.legendre    public= 33  tracked= 0
np.polynomial.laguerre    public= 33  tracked= 0
np.polynomial.hermite     public= 33  tracked= 0
np.polynomial.hermite_e   public= 33  tracked= 0
np.polynomial.polynomial  public= 30  tracked= 8
np.polynomial.polyutils   public=  7  tracked= 0
                          -----------------------
                          public=205  tracked= 8
```

**~197 real public names are invisible to the denominator.** They are not
counted as `absent`; they are not counted at all. Concretely these include
`chebval`, `chebfit`, `chebroots`, `chebcompanion`, `chebgauss`, `chebweight`,
`chebvander`, `cheb2poly`, `poly2cheb`, `chebinterpolate`, `chebgrid2d/3d`,
`chebline`, `chebdomain`, `chebder`, `chebint`, `chebdiv`, `chebadd`,
`chebfromroots` — and the same shape again for each of the other five bases.
**ILLUSTRATIVE, NOT EXHAUSTIVE** — one basis enumerated, six exist.

The 453 `polynomial` items currently in the ledger are the six **classes**
exploded into methods/dunders. The module-level free-function API each basis
also exposes is a separate, untracked layer.

## Why this matters more than 197 items

Phase 2 was taken specifically to stop the denominator flattering us, on the
stated principle from `RUST-QUEUE.md`: *"A coverage number that only ever
rises is not being audited."* It fixed the class-explosion bias. It did **not**
fix the submodule-descent bias, and nothing in the Phase 2 record claims it
did — this is an unnoticed gap, not a broken promise.

Coverage today reads 669/2866 = 23.343%. Adding the polynomial submodule layer
alone moves the denominator to ~3063 and the same 669 to ~21.8%. That is the
number getting more honest, not the project regressing, and whoever lands the
fix must say so in the same commit.

## My own sweep was over-broad — recording the instrument bug, not its output

I ran a recursive crawl for this class of hole across all of numpy and it
returned a headline "3104 untracked names." **That number is wrong and must
not be quoted.** It is contaminated by two things:

1. **Import aliases counted as API.** `numpy.polynomial.chebyshev.np` is that
   module's own `import numpy as np`, re-crawled as if it were a namespace —
   it reports 477 public names because it *is* numpy. Same for `.pu`
   (`polyutils`), `ma.core.mu` / `lib.mixins.um` (`umath`),
   `ma.core.ntypes`, `ma.extras.ma`. Nine of the top rows are this artifact.
2. **`f2py` internals.** `crackfortran`, `capi_maps`, `auxfuncs`, `rules`,
   `cb_rules`, `func2subr`, `cfuncs` are Fortran-wrapper build machinery, not
   numeric API a numpy replacement owes anyone. They dominate the raw list.

The tell was the shape: a large total, arrived at uniformly, with the biggest
contributors all being things I had not deliberately chosen to include. That
is the same signature as the standing rule *"a uniform failure rate across a
cross-product is an instrument bug, not a finding."* It applied to my own
measurement this time.

**Only the polynomial figure above is curated and defensible.** The other
candidate namespaces (`ma.core` 84, `ma.extras` 41, `random.mtrand` 53,
`lib.format` 20, `lib.stride_tricks`, `lib.npyio`, `lib.array_utils`,
`lib.scimath`, `testing.overrides`, `numpy.typing`) each need individual
adjudication for re-export overlap before a single one is added to the
denominator. I have NOT done that adjudication. Anyone who does must check
whether the name is already tracked at its re-exported location — `numpy.ma`
is tracked with 225 items, so much of `ma.core` is very likely double-counting.

## What must NOT be inferred from this

That the fix is "make `surface()` recurse." A blind recursion produces exactly
the 3104 garbage above. The fix is a **curated submodule list**, matching the
curated-explosion decision Phase 2 already took for classes and for the same
reason: curation is what keeps the count honest in both directions.

## Not checked

- Whether `poly1d` (53 items per `DENOMINATOR-AUDIT.md`) overlaps the
  polynomial submodule surface.
- Whether any currently-declared item would be RENAMED by adding submodule
  prefixes. Phase 2's curated-class fix renamed 0 declarations; that property
  must be re-verified for this change before it lands, not assumed to carry.
- numpy versions other than 2.5.1.
