//! Lemire's rejection-sampling algorithm for `Generator.integers()`,
//! transcribed from `numpy/random/src/distributions/distributions.c`
//! (`bounded_lemire_uint64`, `buffered_bounded_lemire_uint{32,16,8}`,
//! `random_bounded_uint64`, `random_buffered_bounded_uint{32,16,8}`,
//! `random_buffered_bounded_bool`) at the numpy `v2.5.1` git tag.
//!
//! `Generator.integers()` ALWAYS calls these with `use_masked=False`
//! (confirmed by reading `_generator.pyx`'s `Generator.integers` --
//! `use_masked` is hardcoded `False` there; the legacy `RandomState`
//! masked-rejection path (`buffered_bounded_masked_*`) is a different,
//! out-of-scope code path for `RandomState.randint`), so only the Lemire
//! functions are ported here.
//!
//! Buffering: 16-bit and 8-bit draws pull from a shared 32-bit buffer
//! (`bcnt`/`buf`) that numpy declares LOCAL to each fill call
//! (`int bcnt = 0; uint32_t buf;` inside `random_bounded_uint16_fill` /
//! `..._uint8_fill`) -- i.e. it resets every `Generator.integers()` call,
//! never persists on the `Generator`/BitGenerator across calls. Callers of
//! this module must create a fresh `(0i32, 0u32)` buffer pair per fill and
//! thread it through every draw in that one fill.

use super::BitGen64;

/// `buffered_uint16`: draws a fresh `next_uint32` every other call, serving
/// the cached upper half on the alternating call.
fn buffered_u16(bg: &mut dyn BitGen64, bcnt: &mut i32, buf: &mut u32) -> u16 {
    if *bcnt == 0 {
        *buf = bg.next_u32();
        *bcnt = 1;
    } else {
        *buf >>= 16;
        *bcnt -= 1;
    }
    *buf as u16
}

/// `buffered_uint8`: draws a fresh `next_uint32` every fourth call.
fn buffered_u8(bg: &mut dyn BitGen64, bcnt: &mut i32, buf: &mut u32) -> u8 {
    if *bcnt == 0 {
        *buf = bg.next_u32();
        *bcnt = 3;
    } else {
        *buf >>= 8;
        *bcnt -= 1;
    }
    *buf as u8
}

/// `buffered_bounded_bool`: one bit per draw, from a 31-bit-deep buffer.
fn buffered_bool(bg: &mut dyn BitGen64, bcnt: &mut i32, buf: &mut u32) -> bool {
    if *bcnt == 0 {
        *buf = bg.next_u32();
        *bcnt = 31;
    } else {
        *buf >>= 1;
        *bcnt -= 1;
    }
    (*buf & 1) != 0
}

/// `bounded_lemire_uint64`. `rng` is the INCLUSIVE range width (`high -
/// low` in the closed-interval sense); caller must never pass
/// `rng == u64::MAX` (that path bypasses Lemire entirely, see
/// `bounded_u64`).
fn lemire_u64(bg: &mut dyn BitGen64, rng: u64) -> u64 {
    let rng_excl = rng + 1;
    let mut m = (bg.next_u64() as u128) * (rng_excl as u128);
    let mut leftover = m as u64;
    if leftover < rng_excl {
        let threshold = (u64::MAX - rng) % rng_excl;
        while leftover < threshold {
            m = (bg.next_u64() as u128) * (rng_excl as u128);
            leftover = m as u64;
        }
    }
    (m >> 64) as u64
}

/// `buffered_bounded_lemire_uint32` (unbuffered in practice -- draws raw
/// `next_uint32` directly, `bcnt`/`buf` exist in numpy only for templating
/// symmetry with the 16/8-bit variants). Caller must never pass
/// `rng == u32::MAX`.
fn lemire_u32(bg: &mut dyn BitGen64, rng: u32) -> u32 {
    let rng_excl = rng + 1;
    let mut m = (bg.next_u32() as u64) * (rng_excl as u64);
    let mut leftover = m as u32;
    if leftover < rng_excl {
        let threshold = (u32::MAX - rng) % rng_excl;
        while leftover < threshold {
            m = (bg.next_u32() as u64) * (rng_excl as u64);
            leftover = m as u32;
        }
    }
    (m >> 32) as u32
}

/// `buffered_bounded_lemire_uint16`, drawing via `buffered_u16`. Caller
/// must never pass `rng == u16::MAX`.
fn lemire_u16(bg: &mut dyn BitGen64, rng: u16, bcnt: &mut i32, buf: &mut u32) -> u16 {
    let rng_excl = rng + 1;
    let mut m = (buffered_u16(bg, bcnt, buf) as u32) * (rng_excl as u32);
    let mut leftover = m as u16;
    if leftover < rng_excl {
        let threshold = (u16::MAX - rng) % rng_excl;
        while leftover < threshold {
            m = (buffered_u16(bg, bcnt, buf) as u32) * (rng_excl as u32);
            leftover = m as u16;
        }
    }
    (m >> 16) as u16
}

/// `buffered_bounded_lemire_uint8`, drawing via `buffered_u8`. Caller must
/// never pass `rng == u8::MAX`.
fn lemire_u8(bg: &mut dyn BitGen64, rng: u8, bcnt: &mut i32, buf: &mut u32) -> u8 {
    let rng_excl = rng + 1;
    let mut m = (buffered_u8(bg, bcnt, buf) as u16) * (rng_excl as u16);
    let mut leftover = m as u8;
    if leftover < rng_excl {
        let threshold = (u8::MAX - rng) % rng_excl;
        while leftover < threshold {
            m = (buffered_u8(bg, bcnt, buf) as u16) * (rng_excl as u16);
            leftover = m as u8;
        }
    }
    (m >> 8) as u8
}

/// `random_bounded_uint64` with `use_masked = False`. `off` is the low
/// bound, `rng` is the INCLUSIVE range width (`high_inclusive - low`).
pub fn bounded_u64(bg: &mut dyn BitGen64, off: u64, rng: u64) -> u64 {
    if rng == 0 {
        off
    } else if rng <= 0xFFFF_FFFF {
        if rng == 0xFFFF_FFFF {
            off.wrapping_add(bg.next_u32() as u64)
        } else {
            off.wrapping_add(lemire_u32(bg, rng as u32) as u64)
        }
    } else if rng == u64::MAX {
        off.wrapping_add(bg.next_u64())
    } else {
        off.wrapping_add(lemire_u64(bg, rng))
    }
}

/// `random_buffered_bounded_uint32` with `use_masked = False`.
pub fn bounded_u32(bg: &mut dyn BitGen64, off: u32, rng: u32) -> u32 {
    if rng == 0 {
        off
    } else if rng == 0xFFFF_FFFF {
        off.wrapping_add(bg.next_u32())
    } else {
        off.wrapping_add(lemire_u32(bg, rng))
    }
}

/// `random_buffered_bounded_uint16` with `use_masked = False`. `bcnt`/`buf`
/// must be a FRESH `(0, 0)` pair per `Generator.integers()` fill call (see
/// module docs).
pub fn bounded_u16(bg: &mut dyn BitGen64, off: u16, rng: u16, bcnt: &mut i32, buf: &mut u32) -> u16 {
    if rng == 0 {
        off
    } else if rng == 0xFFFF {
        off.wrapping_add(buffered_u16(bg, bcnt, buf))
    } else {
        off.wrapping_add(lemire_u16(bg, rng, bcnt, buf))
    }
}

/// `random_buffered_bounded_uint8` with `use_masked = False`. `bcnt`/`buf`
/// must be a FRESH `(0, 0)` pair per `Generator.integers()` fill call.
pub fn bounded_u8(bg: &mut dyn BitGen64, off: u8, rng: u8, bcnt: &mut i32, buf: &mut u32) -> u8 {
    if rng == 0 {
        off
    } else if rng == 0xFF {
        off.wrapping_add(buffered_u8(bg, bcnt, buf))
    } else {
        off.wrapping_add(lemire_u8(bg, rng, bcnt, buf))
    }
}

/// `random_buffered_bounded_bool`. `off`/`rng` are 0/1-valued (numpy's
/// `npy_bool` bounded generator); `rng == 0` short-circuits to `off`
/// without consuming any bits, matching `buffered_bounded_bool`'s own
/// early return.
pub fn bounded_bool(bg: &mut dyn BitGen64, off: bool, rng: bool, bcnt: &mut i32, buf: &mut u32) -> bool {
    if !rng {
        off
    } else {
        buffered_bool(bg, bcnt, buf)
    }
}

/// `random_interval` (`distributions.c`): masked rejection, returning a
/// uniform value in `[0, max]` INCLUSIVE. Distinct from the Lemire family
/// above -- this is the function `hypergeometric_sample`/`Generator.choice`/
/// `.shuffle`/`.permutation` call, not `Generator.integers()`. `max == 0`
/// short-circuits to `0` without consuming any bits, matching numpy's own
/// early return.
pub fn random_interval(bg: &mut dyn BitGen64, max: u64) -> u64 {
    if max == 0 {
        return 0;
    }
    let mut mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    if max <= 0xFFFF_FFFF {
        loop {
            let value = (bg.next_u32() as u64) & mask;
            if value <= max {
                return value;
            }
        }
    } else {
        loop {
            let value = bg.next_u64() & mask;
            if value <= max {
                return value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Pcg64, SeedSequence};

    // IMPORTANT correction vs this task's original assumption: empirically
    // verified (`.venv`, real numpy 2.5.1) that `np.random.default_rng(seed)`
    // instantiates plain `PCG64` (XSL-RR), NOT `PCG64DXSM` --
    // `type(np.random.default_rng(42).bit_generator)` is
    // `numpy.random._pcg64.PCG64`, and its `.state` matches a freshly
    // constructed `PCG64(42)` exactly. `PCG64DXSM` is implemented and
    // bit-exact-verified in `pcg64.rs` but is NOT what `default_rng` uses in
    // this numpy version. `Generator`'s BitGenerator binding below must use
    // `Pcg64`, not `Pcg64Dxsm`.
    fn pcg64(seed: u32) -> Pcg64 {
        let seq = SeedSequence::new(&[seed], &[], 4);
        let v = seq.generate_state_u64(4);
        Pcg64::from_seed_words([v[0], v[1], v[2], v[3]])
    }

    // Verified against real numpy 2.5.1 (`.venv`, offline reference only):
    //   np.random.default_rng(42).integers(0, 100, size=10) ->
    //   array([ 8, 77, 65, 43, 43, 85,  8, 69, 20,  9])
    #[test]
    fn integers_0_100_seed42_matches_numpy() {
        let mut bg = pcg64(42);
        let draws: Vec<u64> = (0..10).map(|_| bounded_u64(&mut bg, 0, 99)).collect();
        assert_eq!(draws, vec![8, 77, 65, 43, 43, 85, 8, 69, 20, 9]);
    }
}
