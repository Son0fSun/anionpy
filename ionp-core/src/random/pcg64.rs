//! `PCG64` (PCG-XSL-RR 128/64) and `PCG64DXSM` (PCG-CM-DXSM 128/64),
//! transcribed constant-for-constant from `numpy/random/src/pcg64/pcg64.h`
//! (the `__SIZEOF_INT128__` / native-128-bit-math branch, which is what
//! every real build of numpy on a 64-bit host actually compiles -- Rust's
//! native `u128` mirrors that fast path exactly, so there is no need to
//! port the emulated-struct branch) and `numpy/random/_pcg64.pyx`
//! (`PCG64.__init__`/`PCG64DXSM.__init__`), at the numpy `v2.5.1` git tag.
//!
//! Confirmed by reading `_pcg64.pyx` directly (not guessed): both
//! `PCG64.__init__` and `PCG64DXSM.__init__` seed via the exact same C
//! function, `pcg64_set_seed` -> `pcg64_srandom_r` ==
//! `pcg_setseq_128_srandom_r` -- the DEFAULT-multiplier LCG step, not the
//! DXSM variant's own "cheap multiplier" step. `PCG64DXSM` only diverges
//! from `PCG64` in its PER-DRAW step/output functions (`pcg_cm_step_r` /
//! `pcg_output_cm_128_64`), never in seeding. Getting this backwards (i.e.
//! seeding `PCG64DXSM` with the cheap-multiplier step) would silently
//! produce a wrong-but-plausible-looking stream.
//!
//! Both bit generators are seeded from `SeedSequence::generate_state_u64(4)`
//! `[w0, w1, w2, w3]`: `initstate = (w0 << 64) | w1`, `initseq = (w2 << 64)
//! | w3` (`_pcg64.pyx`: `pcg64_set_seed(&rng_state, val, val + 2)`, and
//! `pcg64_set_seed` reads `seed[0..2]` as `initstate.{high,low}` and
//! `inc[0..2]` as `initseq.{high,low}`).

use super::BitGen64;

/// `PCG_DEFAULT_MULTIPLIER_128 = PCG_128BIT_CONSTANT(2549297995355413924ULL,
/// 4865540595714422341ULL)`.
const PCG_DEFAULT_MULTIPLIER_128: u128 =
    (2549297995355413924u128 << 64) | 4865540595714422341u128;

/// `PCG_CHEAP_MULTIPLIER_128 = 0xda942042e4dd58b5ULL` -- a 64-bit constant
/// numpy multiplies against the full 128-bit state via `pcg128_mult_64`
/// (state * 64-bit-scalar, truncated mod 2^128); native `u128` arithmetic
/// with the constant widened to `u128` does the same truncation for free.
const PCG_CHEAP_MULTIPLIER: u64 = 0xda942042e4dd58b5;

/// `pcg_setseq_128_srandom_r` / `pcg_cm_srandom_r`: BOTH use the DEFAULT
/// (non-cheap) step, confirmed identical between `PCG64` and `PCG64DXSM`
/// seeding (see module docs). `initseq` is folded into an odd increment
/// (`(initseq << 1) | 1`) before two default LCG steps sandwich the state
/// addition.
fn seed_state(initstate: u128, initseq: u128) -> (u128, u128) {
    let inc = (initseq << 1) | 1;
    let mut state: u128 = 0;
    state = step_default(state, inc);
    state = state.wrapping_add(initstate);
    state = step_default(state, inc);
    (state, inc)
}

fn step_default(state: u128, inc: u128) -> u128 {
    state
        .wrapping_mul(PCG_DEFAULT_MULTIPLIER_128)
        .wrapping_add(inc)
}

/// `PCG64`: PCG-XSL-RR 128/64. Advance-then-output: `pcg_setseq_128_xsl_rr_64_random_r`
/// steps the state with the DEFAULT multiplier, then reads
/// `pcg_output_xsl_rr_128_64` off the POST-step state.
#[derive(Clone)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    /// `words` is `SeedSequence::generate_state_u64(4)` in draw order
    /// `[w0, w1, w2, w3]`.
    pub fn from_seed_words(words: [u64; 4]) -> Self {
        let initstate = ((words[0] as u128) << 64) | (words[1] as u128);
        let initseq = ((words[2] as u128) << 64) | (words[3] as u128);
        let (state, inc) = seed_state(initstate, initseq);
        Self {
            state,
            inc,
            has_uint32: false,
            uinteger: 0,
        }
    }

    fn step(&mut self) {
        self.state = step_default(self.state, self.inc);
    }

    /// `pcg_output_xsl_rr_128_64`: `rotr64(hi ^ lo, hi >> 58)`.
    fn output(state: u128) -> u64 {
        let hi = (state >> 64) as u64;
        let lo = state as u64;
        (hi ^ lo).rotate_right((hi >> 58) as u32)
    }
}

impl BitGen64 for Pcg64 {
    fn next_u64(&mut self) -> u64 {
        self.step();
        Self::output(self.state)
    }

    /// `pcg64_next32`: caches the unused half of a `next64` draw.
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

/// `PCG64DXSM`: PCG-CM-DXSM 128/64, `default_rng`'s default BitGenerator
/// since numpy 1.25. Output-then-step (the inverse order vs `Pcg64`):
/// `pcg_cm_random_r` computes the DXSM output from the PRE-step state, THEN
/// steps the state with the cheap multiplier.
#[derive(Clone)]
pub struct Pcg64Dxsm {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64Dxsm {
    pub fn from_seed_words(words: [u64; 4]) -> Self {
        // Same seeding function as `Pcg64` -- see module docs.
        let initstate = ((words[0] as u128) << 64) | (words[1] as u128);
        let initseq = ((words[2] as u128) << 64) | (words[3] as u128);
        let (state, inc) = seed_state(initstate, initseq);
        Self {
            state,
            inc,
            has_uint32: false,
            uinteger: 0,
        }
    }

    /// `pcg_output_cm_128_64`, run on the state BEFORE this draw's step.
    fn output_cm(state: u128) -> u64 {
        let mut hi = (state >> 64) as u64;
        let mut lo = state as u64;
        lo |= 1;
        hi ^= hi >> 32;
        hi = hi.wrapping_mul(PCG_CHEAP_MULTIPLIER);
        hi ^= hi >> 48;
        hi = hi.wrapping_mul(lo);
        hi
    }

    /// `pcg_cm_step_r`: state * cheap_multiplier + inc, mod 2^128.
    fn step_cm(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_CHEAP_MULTIPLIER as u128)
            .wrapping_add(self.inc);
    }
}

impl BitGen64 for Pcg64Dxsm {
    fn next_u64(&mut self) -> u64 {
        let ret = Self::output_cm(self.state);
        self.step_cm();
        ret
    }

    /// `pcg64_cm_next32`: identical caching shape to `Pcg64::next_u32`.
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

    fn seed_words(seed: u32) -> [u64; 4] {
        let seq = SeedSequence::new(&[seed], &[], 4);
        let v = seq.generate_state_u64(4);
        [v[0], v[1], v[2], v[3]]
    }

    // Verified against real numpy 2.5.1 (`.venv`, offline reference only):
    //   bg = np.random.PCG64(0)
    //   bg.state ->
    //     {'state': 35399562948360463058890781895381311971,
    //      'inc':   87136372517582989555478159403783844777}
    //   [bg.random_raw() for _ in range(4)] ->
    //     [11749869230777074271, 4976686463289251617,
    //      755828109848996024, 304881062738325533]
    #[test]
    fn pcg64_zero_matches_numpy() {
        let mut bg = Pcg64::from_seed_words(seed_words(0));
        assert_eq!(bg.state, 35399562948360463058890781895381311971u128);
        assert_eq!(bg.inc, 87136372517582989555478159403783844777u128);
        let draws: Vec<u64> = (0..4).map(|_| bg.next_u64()).collect();
        assert_eq!(
            draws,
            vec![
                11749869230777074271,
                4976686463289251617,
                755828109848996024,
                304881062738325533,
            ]
        );
    }

    // np.random.PCG64(42).random_raw() x4 ->
    //   [14276969152011380360, 8095878257575067585,
    //    15838336090824644132, 12864169557245331597]
    #[test]
    fn pcg64_42_matches_numpy() {
        let mut bg = Pcg64::from_seed_words(seed_words(42));
        let draws: Vec<u64> = (0..4).map(|_| bg.next_u64()).collect();
        assert_eq!(
            draws,
            vec![
                14276969152011380360,
                8095878257575067585,
                15838336090824644132,
                12864169557245331597,
            ]
        );
    }

    // np.random.PCG64DXSM(0).state is identical to PCG64(0).state (same
    // seeding function); random_raw() x4 ->
    //   [15672045205194312304, 10230625629676741203,
    //    1393141542142426128, 6186804329743392408]
    #[test]
    fn pcg64dxsm_zero_matches_numpy() {
        let mut bg = Pcg64Dxsm::from_seed_words(seed_words(0));
        assert_eq!(bg.state, 35399562948360463058890781895381311971u128);
        assert_eq!(bg.inc, 87136372517582989555478159403783844777u128);
        let draws: Vec<u64> = (0..4).map(|_| bg.next_u64()).collect();
        assert_eq!(
            draws,
            vec![
                15672045205194312304,
                10230625629676741203,
                1393141542142426128,
                6186804329743392408,
            ]
        );
    }

    // np.random.PCG64DXSM(42).random_raw() x4 ->
    //   [12329818062196000797, 125530269004142706,
    //    12137922674892001441, 6848431486601849532]
    #[test]
    fn pcg64dxsm_42_matches_numpy() {
        let mut bg = Pcg64Dxsm::from_seed_words(seed_words(42));
        let draws: Vec<u64> = (0..4).map(|_| bg.next_u64()).collect();
        assert_eq!(
            draws,
            vec![
                12329818062196000797,
                125530269004142706,
                12137922674892001441,
                6848431486601849532,
            ]
        );
    }
}
