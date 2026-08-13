"""anionpy.char -- pure import surface for the `numpy.char` block.

No logic lives here. Every function is a straight re-export of a compiled
Rust symbol from `anionpy._anionpy` (see `ionp-py/src/strings.rs` /
`ionp-core/src/strings.rs`), renamed to its public `numpy.char.*` name. The
Rust-side names are prefixed `strings_*` to avoid colliding with the
numeric ufuncs already installed flat on `_anionpy` (e.g. `_anionpy.add` is the
numeric add; the string add is `_anionpy.strings_add`).

`strings_multiply` (imported below WITHOUT a rename, unlike everything else
in this file) is a deliberate exception to the "renamed to its public name"
rule above: real numpy's `numpy._core.defchararray` does `from numpy.strings
import (multiply as strings_multiply, ...)`, i.e. `numpy.char.
strings_multiply IS numpy.strings.multiply` -- the literal same object,
confirmed directly (`np.char.strings_multiply is np.strings.multiply ==
True`), NOT a numpy.char-specific reimplementation. `numpy.char.multiply` is
a separate, different object (a thin wrapper that catches `TypeError` and
re-raises `ValueError` -- see `anionpy/_anionpy`'s `char_multiply`, already
exported above as `multiply`), so this is not redundant with that entry.
Since `anionpy.strings.multiply` (in `anionpy/strings.py`) already imports this
exact same Rust symbol under that public name and is already declared
exact, importing the identical, unrenamed symbol here and exporting it
under its `strings_*` name makes `anionpy.char.strings_multiply is anionpy.
strings.multiply` True by construction -- the same identity real numpy has
-- with zero new logic and zero new probing risk.

NOTE: `numpy.char.strings_partition`/`strings_rpartition` are the identical
kind of alias (same relationship to `numpy.strings.partition`/
`rpartition`) but are DELIBERATELY NOT exposed here -- an out-of-corpus
probe during this round found that the underlying shared Rust
`strings_partition`/`strings_rpartition` (already imported into `anionpy.
strings` and declared exact) has a real, previously-undiscovered defect:
real numpy silently truncates the `sep` argument to the array's declared
per-element dtype width before searching (confirmed:
`np.strings.partition(np.array(['a'], dtype='<U1'), 'ab')` matches as if
`sep` were truncated to `'a'`, for both `U` and `S` dtype), while anionpy's
Rust implementation searches with the untruncated `sep` and misses the
match. This affects `anionpy.strings.partition`/`rpartition` too, not just
this alias -- flagged for the coordinator/RUST-QUEUE, not fixed here (Rust
edits are out of this task's scope). See `anionpy/_state/char_strings.py`'s
round-6 docstring for the full probe detail.
"""

from anionpy._anionpy import (
    strings_add as add,
    strings_isalpha as isalpha,
    strings_isalnum as isalnum,
    strings_isdecimal as isdecimal,
    strings_isdigit as isdigit,
    strings_isnumeric as isnumeric,
    strings_isspace as isspace,
    strings_islower as islower,
    strings_isupper as isupper,
    strings_istitle as istitle,
    strings_str_len as str_len,
    strings_upper as upper,
    strings_lower as lower,
    strings_swapcase as swapcase,
    strings_title as title,
    strings_capitalize as capitalize,
    strings_strip as strip,
    strings_lstrip as lstrip,
    strings_rstrip as rstrip,
    strings_center as center,
    strings_ljust as ljust,
    strings_rjust as rjust,
    strings_zfill as zfill,
    strings_find as find,
    strings_rfind as rfind,
    strings_count as count,
    strings_startswith as startswith,
    strings_endswith as endswith,
    strings_index as index,
    strings_rindex as rindex,
    strings_replace as replace,
    char_multiply as multiply,
    strings_encode as encode,
    strings_decode as decode,
    char_equal as equal,
    char_not_equal as not_equal,
    char_less as less,
    char_less_equal as less_equal,
    char_greater as greater,
    char_greater_equal as greater_equal,
    char_compare_chararrays as compare_chararrays,
    strings_multiply,
)

__all__ = [
    "add",
    "isalpha",
    "isalnum",
    "isdecimal",
    "isdigit",
    "isnumeric",
    "isspace",
    "islower",
    "isupper",
    "istitle",
    "str_len",
    "upper",
    "lower",
    "swapcase",
    "title",
    "capitalize",
    "strip",
    "lstrip",
    "rstrip",
    "center",
    "ljust",
    "rjust",
    "zfill",
    "find",
    "rfind",
    "count",
    "startswith",
    "endswith",
    "index",
    "rindex",
    "replace",
    "multiply",
    "encode",
    "decode",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "compare_chararrays",
    "strings_multiply",
]
