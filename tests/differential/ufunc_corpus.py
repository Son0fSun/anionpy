"""Additional corpus material for the ufunc block.

Numeric ufuncs (106 of the 134) reuse `corpus.py`'s existing seeded corpus
unchanged -- it already covers every edge case GOAL-ionp.md names (empty,
0-d, NaN, inf, -0.0, negative/non-contiguous strides, broadcasting pairs,
integer overflow boundaries) and every mixed-dtype/NEP-50 pair via
`corpus.scalar_operands()`. Re-deriving that here would be exactly the kind
of duplicated, driftable corpus the module docstring in corpus.py warns
about.

The 28 `char.*`/`strings.*` ufuncs need a corpus `corpus.py` does not have:
numpy string arrays. That corpus lives here, seeded off the SAME pinned
`corpus.SEED` (see corpus.py) so it is deterministic for the same reason.

This module also owns the deterministic *sampling* helpers used to keep the
binary-ufunc x scalar-operand cross product (144 unary cases x 16 scalar
forms = 2304 per binary ufunc, x ~90 binary ufuncs) bounded. The stride is a
module-level constant, printed by run.py's ufunc summary, so the sampling
is visible rather than a silent truncation -- see the task brief.
"""
from __future__ import annotations

import dataclasses

import numpy as np

import _bootstrap  # noqa: F401
import corpus

SEED = corpus.SEED


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# string corpus
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Case:
    label: str
    value: object


@dataclasses.dataclass(frozen=True)
class Pair:
    label: str
    a: object
    b: object


_WORDS_UNICODE = ["alpha", "Beta3", "  spaced  ", "", "CAFÉ", "日本語", "MiXeD_Case9", "z"]
_WORDS_ASCII = ["alpha", "Beta3", "  spaced  ", "", "MIXED_case9", "z", "TITLE Case", "9numbers9"]


def string_unary_corpus() -> list[Case]:
    """Every input a unary (or the receiver of a binary) char/strings ufunc
    is run against. Deterministic and literal where the edge case is the
    point (empty string, empty array, all-whitespace, non-ASCII), seeded
    random for volume."""
    out: list[Case] = []

    out.append(Case("string/1d_unicode", np.array(_WORDS_UNICODE, dtype="<U16")))
    out.append(Case("string/1d_ascii_bytes", np.array([w.encode("ascii") for w in _WORDS_ASCII],
                                                        dtype="S16")))
    out.append(Case("string/0d", np.array("solo", dtype="<U8")))
    out.append(Case("string/empty_array", np.array([], dtype="<U8")))
    out.append(Case("string/empty_string_elements",
                     np.array(["", "", "x", ""], dtype="<U4")))
    out.append(Case("string/2d", np.array(_WORDS_UNICODE, dtype="<U16").reshape(2, 4)))
    base = np.array(_WORDS_UNICODE, dtype="<U16")
    out.append(Case("string/view_reversed", base[::-1]))
    out.append(Case("string/view_stride2", base[::2]))
    out.append(Case("string/all_same", np.array(["dup"] * 5, dtype="<U8")))

    rng = _rng()
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFG "))
    for shape, label in (((6,), "random/1d"), ((3, 2), "random/2d")):
        n = int(np.prod(shape))
        lengths = rng.integers(0, 10, size=n)
        words = []
        for length in lengths:
            idx = rng.integers(0, len(alphabet), size=int(length))
            words.append("".join(alphabet[idx]))
        arr = np.array(words, dtype="<U16").reshape(shape)
        out.append(Case(f"string/{label}", arr))

    return out


def string_binary_corpus() -> list[Pair]:
    rng = _rng()
    unary = string_unary_corpus()
    out: list[Pair] = []

    same_shape_labels = ["string/1d_unicode", "string/2d", "string/random/1d"]
    for label in same_shape_labels:
        a = next(c.value for c in unary if c.label == label)
        b = np.array(a)  # independent copy
        rng.shuffle(b.reshape(-1))  # deterministic (same rng stream every run)
        out.append(Pair(f"same_shape/{label}", a, b))

    # broadcasting
    out.append(Pair("broadcast/col_row",
                     np.array([["a"], ["bb"], ["ccc"]], dtype="<U8"),
                     np.array([["x", "yy", "zzz", ""]], dtype="<U8")))
    out.append(Pair("broadcast/0d_scalar",
                     np.array("pre", dtype="<U8"),
                     np.array(["a", "bb", "ccc"], dtype="<U8")))
    out.append(Pair("broadcast/scalar_empty",
                     np.array("x", dtype="<U8"),
                     np.array([], dtype="<U8")))

    # incompatible shapes -- numpy MUST raise, anionpy must match
    out.append(Pair("broadcast_fail/mismatched_1d",
                     np.array(["a", "b", "c"], dtype="<U8"),
                     np.array(["x", "y"], dtype="<U8")))

    return out


def string_scalar_operands() -> list[tuple[str, object]]:
    """Weak-python-string vs numpy-string-scalar vs 0-d-array call forms,
    the string analogue of corpus.scalar_operands()."""
    return [
        ("python_str", "suffix"),
        ("python_str_empty", ""),
        ("numpy_str_scalar", np.str_("nsfx")),
        ("zero_d_array", np.array("zd", dtype="<U8")),
    ]


# ---------------------------------------------------------------------------
# deterministic sampling -- keeps the (unary corpus) x (scalar operands)
# cross product bounded for the ~90 binary numeric ufuncs. A fixed stride
# over the corpus's own deterministic ordering, not a random subset -- two
# runs pick exactly the same cases.
# ---------------------------------------------------------------------------

UNARY_SAMPLE_STRIDE = 12  # 144 unary cases / 12 ~= 12 kept per binary ufunc


def sampled(pool: list, stride: int) -> list:
    """Deterministic 'every Nth' sample, always including index 0."""
    if not pool:
        return []
    return pool[::stride]
