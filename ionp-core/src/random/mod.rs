//! `ionp.random`: numpy-bit-exact random number generation.
//!
//! Every algorithm in this module is transcribed directly from numpy
//! 2.5.1's own Cython/C sources (`numpy/random/bit_generator.pyx`,
//! `numpy/random/_pcg64.pyx`, `numpy/random/src/pcg64/pcg64.{h,c}`,
//! `numpy/random/src/distributions/distributions.c`,
//! `numpy/random/_bounded_integers.pyx.in`), fetched read-only from the
//! numpy GitHub mirror at the `v2.5.1` tag and read offline -- never
//! executed at runtime. numpy is the REFERENCE side of the differential
//! only (see the task brief); nothing in this module calls into numpy.
//!
//! Module map:
//!   - `seed_sequence` -- `SeedSequence`'s entropy-mixing algorithm
//!     (`hashmix`/`mix`/pool assembly), the seeding root every
//!     BitGenerator below is built on.
//!   - `pcg64` -- `PCG64` (PCG-XSL-RR 128/64, numpy's legacy `PCG64`) and
//!     `PCG64DXSM` (the DXSM output function, numpy's `default_rng`
//!     default since 1.25).
//!   - `bounded` -- Lemire's rejection-sampling algorithm for
//!     `Generator.integers()`, one function per output width
//!     (8/16/32/64-bit, matching numpy's per-width buffering behavior
//!     exactly: 16/8-bit draws pull from a shared 32-bit buffer, 32/64-bit
//!     draws do not).
//!
//! NOT implemented here (see the task's final report for the honest
//! breakdown): MT19937, Philox (measured 2026-08-13, deferred -- see
//! `docs`/task report, not attempted this pass), legacy RandomState, and
//! every distribution beyond the direct uniform-double/bounded-integer paths
//! (`random`, `integers`, `bytes`). Depth over breadth, per the task
//! brief -- a bit-exact PCG64/PCG64DXSM + SeedSequence + the `Generator`
//! surface that sits directly on `next_uint64`/`next_uint32` beats a
//! larger set of approximately-right distributions.

pub mod bounded;
pub mod discrete;
pub mod distributions;
mod logfactorial;
pub mod pcg64;
pub mod seed_sequence;
pub mod sfc64;
mod ziggurat_tables;

pub use pcg64::{Pcg64, Pcg64Dxsm};
pub use seed_sequence::SeedSequence;
pub use sfc64::Sfc64;

/// Shared interface every numpy-compatible bit generator implements.
/// `next_uint32` MUST cache the unused half of a `next_uint64` draw
/// exactly as numpy's `pcg64_next32`/`pcg64_cm_next32` do (`has_uint32`/
/// `uinteger` fields) -- `bounded.rs`'s buffered 16/8-bit draws and
/// `Generator.bytes()`'s raw uint32 path both depend on this cache being
/// live across calls on the SAME bit generator instance.
pub trait BitGen64 {
    fn next_u64(&mut self) -> u64;
    fn next_u32(&mut self) -> u32;

    /// `numpy/random/_common.pxd`'s `uint64_to_double`: `(rnd >> 11) * 2^-53`.
    /// Identical for every bit generator -- the BitGenerator interface
    /// only ever hands `Generator` a `next_double` built this way from its
    /// own `next_uint64`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// `numpy/random/src/distributions/distributions.c`'s `next_float`:
    /// `(next_uint32(state) >> 8) * 2^-24`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0f32 / 16777216.0f32)
    }
}
