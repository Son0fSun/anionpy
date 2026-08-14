//! `SFC64` (Chris Doty-Humphrey's Small Fast Chaotic PRNG), transcribed
//! constant-for-constant from `numpy/random/src/sfc64/sfc64.{h,c}` and
//! `numpy/random/_sfc64.pyx` at the numpy `v2.5.1` git tag.
//!
//! State is `[u64; 4]`: the first 3 words come from
//! `SeedSequence::generate_state(3, np.uint64)`, the 4th is a counter that
//! `sfc64_set_seed` initializes to `1` (NOT drawn from the seed sequence)
//! before iterating the generator 12 times to mix the seed into the other
//! three words -- this discard loop is REQUIRED for numpy-matching output;
//! skipping it produces a stream that still "looks random" but diverges on
//! the very first draw.

use super::BitGen64;

fn rotl(value: u64, rot: u32) -> u64 {
    value.rotate_left(rot)
}

/// `sfc64_next`: advances `s` in place and returns the drawn word.
fn sfc64_next(s: &mut [u64; 4]) -> u64 {
    let tmp = s[0].wrapping_add(s[1]).wrapping_add(s[3]);
    s[3] = s[3].wrapping_add(1);

    s[0] = s[1] ^ (s[1] >> 11);
    s[1] = s[2].wrapping_add(s[2] << 3);
    s[2] = rotl(s[2], 24).wrapping_add(tmp);

    tmp
}

#[derive(Clone)]
pub struct Sfc64 {
    s: [u64; 4],
    has_uint32: bool,
    uinteger: u32,
}

impl Sfc64 {
    /// `words` is `SeedSequence::generate_state_u64(3)` in draw order
    /// `[w0, w1, w2]`. `sfc64_set_seed`: `state->s = {w0, w1, w2, 1}`, then
    /// `sfc64_next` is called 12 times and discarded (`_reset_state_variables`
    /// separately zeroes `has_uint32`/`uinteger`, which our struct literal
    /// already starts at).
    pub fn from_seed_words(words: [u64; 3]) -> Self {
        let mut s = [words[0], words[1], words[2], 1];
        for _ in 0..12 {
            sfc64_next(&mut s);
        }
        Self { s, has_uint32: false, uinteger: 0 }
    }
}

impl BitGen64 for Sfc64 {
    fn next_u64(&mut self) -> u64 {
        sfc64_next(&mut self.s)
    }

    /// `sfc64_next32`: caches the unused half of a `next64` draw, same
    /// shape as `Pcg64`/`Pcg64Dxsm`.
    fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        (next & 0xffff_ffff) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::SeedSequence;

    fn seed_words(seed: u32) -> [u64; 3] {
        let seq = SeedSequence::new(&[seed], &[], 4);
        let v = seq.generate_state_u64(3);
        [v[0], v[1], v[2]]
    }

    // Verified against real numpy 2.5.1 (`.venv`, offline reference only):
    //   bg = np.random.SFC64(0)
    //   bg.state -> {'state': array([9957867060933711493,
    //     532597980065565856, 14769588338631205282, 13], dtype=uint64),
    //     'has_uint32': 0, 'uinteger': 0}
    //   [bg.random_raw() for _ in range(4)] ->
    //     [10490465040999277362, 4331856608414834465,
    //      7312684695965765022, 1874867651408945186]
    #[test]
    fn sfc64_zero_matches_numpy() {
        let mut bg = Sfc64::from_seed_words(seed_words(0));
        assert_eq!(bg.s, [9957867060933711493u64, 532597980065565856, 14769588338631205282, 13]);
        let draws: Vec<u64> = (0..4).map(|_| bg.next_u64()).collect();
        assert_eq!(
            draws,
            vec![
                10490465040999277362,
                4331856608414834465,
                7312684695965765022,
                1874867651408945186,
            ]
        );
    }

    // np.random.SFC64(42).state ->
    //   state=[9143715722600539226, 631878879238184246,
    //          4804307118799948943, 13]
    // random_raw() x4 -> [9775594601838723485, 6977463094773878866,
    //                     17439770048677797496, 7768405669198076140]
    #[test]
    fn sfc64_42_matches_numpy() {
        let mut bg = Sfc64::from_seed_words(seed_words(42));
        assert_eq!(bg.s, [9143715722600539226u64, 631878879238184246, 4804307118799948943, 13]);
        let draws: Vec<u64> = (0..4).map(|_| bg.next_u64()).collect();
        assert_eq!(
            draws,
            vec![
                9775594601838723485,
                6977463094773878866,
                17439770048677797496,
                7768405669198076140,
            ]
        );
    }
}
