# TICKET: real DEFLATE (Huffman/LZ77) compressor for `savez_compressed`

Filed 2026-08-13, during the I/O-cluster task, at the coordinator's
explicit request ("file a ticket-style note in docs/ for the real DEFLATE
implementation so it is not lost"). Not implemented as part of that task --
out of scope for the time available, and a real codec is a standalone job.

## Current state

`ionp-core/src/format.rs::deflate` is a hand-rolled RFC 1951 DEFLATE
encoder. Per its own pre-existing module doc comment, it emits ONLY
"stored" (uncompressed) blocks. This is:

- Valid DEFLATE. Any conformant decoder, including real numpy's `np.load`,
  reads it back correctly.
- Not a value/correctness bug. Every array round-trips exactly.
- Not actually compression. Stored blocks carry ~5 bytes of block-header
  overhead per block plus the raw payload, so output is never smaller than
  input and is typically slightly larger.

`ionp-py/src/io_ops.rs::savez_compressed` is the only caller that exposes
this to users, via `np.savez_compressed`'s Python-level equivalent. It now
(as of this task) emits a `UserWarning` on every call naming this gap, and
is NOT declared `"exact"` in `anionpy/_state/toplevel.py` -- see that
file's `savez_compressed` note for the full reasoning and measurements.

## Measured impact

100-element `float64` array, single-key `.npz`, measured independently
twice (coordinator + this task), consistent both times:

| | file size | compress_type | compress_size | file_size (uncompressed) |
|---|---|---|---|---|
| anionpy (stored-blocks-only) | 1041 B | 8 (DEFLATE) | 933 B | 928 B |
| real numpy (genuine Huffman/LZ77) | 383 B | 8 (DEFLATE) | 255 B | 928 B |

anionpy's `compress_size` (933) exceeds its own `file_size` (928): net
expansion, not compression, for this input. Real numpy achieves ~3.6x
shrink on the same array. The gap will vary by input compressibility but
is structural -- it does not narrow for more-compressible inputs, since
stored blocks never compress anything regardless of input redundancy.

## What's needed

A genuine DEFLATE encoder in `ionp-core/src/format.rs`, replacing (or
added alongside, with the stored-blocks path kept as a fallback/reference)
the current `deflate` function:

1. **LZ77 matching**: a sliding window (32 KiB per RFC 1951) over the
   input, finding back-references (length, distance) instead of emitting
   literal bytes for repeated sequences. Even a simple greedy/lazy matcher
   (no need to match zlib's exact match-finding heuristics) would produce
   real compression; matching zlib's specific ratios is not required for
   correctness, only for competitive file size.
2. **Huffman coding**: either fixed Huffman codes (RFC 1951 §3.2.6, no
   extra bookkeeping, simpler first milestone) or dynamic Huffman codes
   (§3.2.7, better ratio, requires emitting the code-length tree itself).
   Fixed-code DEFLATE alone would already close most of the gap above
   stored-blocks-only.
3. **Bitstream writer**: DEFLATE packs Huffman codes and back-reference
   codes at the bit level (not byte-aligned) -- the existing stored-block
   path is presumably byte-aligned throughout, so this is new plumbing,
   not a reuse of existing code.
4. **Decoder side check**: `format.rs` presumably already has a decoder
   (since anionpy's own `load()` can read real numpy's genuinely-compressed
   `.npz` files today -- that direction already works). Confirm the
   existing decoder handles both fixed and dynamic Huffman blocks, not
   just stored blocks, before assuming only the encoder needs work.

## Acceptance bar for closing this ticket

Per this project's `.tobytes()`-based, no-tolerance comparison standard:

- `compress_size < file_size` for realistic (non-adversarial,
  non-incompressible-random) inputs -- the property
  `tests/differential/products_io_cases.py::_npzc_ionp` now asserts and
  that currently correctly FAILS for this reason.
- Round-trip values must remain exactly correct (already true; must not
  regress).
- Once real compression lands, revisit `savez_compressed`'s ledger entry
  in `anionpy/_state/toplevel.py` -- it should move from "NOT DECLARED"
  back to a declared state (with fresh measurements, not by just deleting
  the caveat) and the `UserWarning` in `ionp-py/src/io_ops.rs` should be
  removed or narrowed to only fire if compression is somehow still
  ineffective for a given input.
- Achieving byte-for-byte identical compressed OUTPUT to real numpy's
  zlib-backed encoder is explicitly NOT required (different encoders,
  even both "correct" DEFLATE, do not need to produce identical bytes) --
  only genuine size reduction and correct round-trip.

## Non-goals for this ticket

- Matching real numpy's exact compression ratio bit-for-bit.
- Fixing the separate, unrelated, already-disclosed `.npz` whole-archive
  metadata divergence (`create_system`/`create_version`/`external_attr`
  ZIP fields differ between anionpy's writer and numpy's Unix-flavored
  one) -- that affects `savez` too, is purely cosmetic/non-functional, and
  is out of scope here. See the `savez` note in
  `anionpy/_state/toplevel.py` for that finding.
