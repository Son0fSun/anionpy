//! `numpy.random.SeedSequence`'s entropy-mixing algorithm, transcribed
//! verbatim (constant-for-constant, loop-for-loop) from
//! `numpy/random/bit_generator.pyx` (`hashmix`/`mix`/`SeedSequence.mix_entropy`/
//! `SeedSequence.generate_state`) at the numpy `v2.5.1` git tag. Derived from
//! Melissa E. O'Neill's C++11 `std::seed_seq` alternative -- see that file's
//! own header comment for the algorithm's provenance.
//!
//! This is the seeding root every BitGenerator in this module sits on:
//! `PCG64`/`PCG64DXSM` both seed via `SeedSequence::generate_state(4,
//! Uint64)`. Get this wrong and every downstream stream is wrong in a way
//! that would still look "random" -- which is exactly why the task brief
//! calls out getting this right FIRST and verifying it in isolation.

const DEFAULT_POOL_SIZE: usize = 4;
const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u32 = 16; // np.dtype(np.uint32).itemsize * 8 // 2

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^= value >> XSHIFT;
    value
}

fn mix(x: u32, y: u32) -> u32 {
    let mut result = (MIX_MULT_L.wrapping_mul(x)).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

/// Word-for-word port of `SeedSequence.mix_entropy`.
fn mix_entropy(pool_size: usize, entropy_array: &[u32]) -> Vec<u32> {
    let mut mixer = vec![0u32; pool_size];
    let mut hash_const = INIT_A;

    for i in 0..mixer.len() {
        if i < entropy_array.len() {
            mixer[i] = hashmix(entropy_array[i], &mut hash_const);
        } else {
            mixer[i] = hashmix(0, &mut hash_const);
        }
    }

    for i_src in 0..mixer.len() {
        for i_dst in 0..mixer.len() {
            if i_src != i_dst {
                let src_val = mixer[i_src];
                let hashed = hashmix(src_val, &mut hash_const);
                mixer[i_dst] = mix(mixer[i_dst], hashed);
            }
        }
    }

    // BUG FIX (2026-08-13, this session, found via an SFC64-corpus seed
    // >2**128, which is the first case in this repo's history to push
    // entropy past the 4-word pool and actually run this loop): numpy's
    // `hashmix(entropy_array[i_src], hash_const)` is called INSIDE the
    // `i_dst` inner loop, not once per `i_src` -- `hashmix` mutates
    // `hash_const` and returns a DIFFERENT value on every call even for
    // the same input word, so each of the `len(mixer)` inner iterations
    // must get its own fresh `hashmix` call. The previous version hoisted
    // the `hashmix` call outside the inner loop and reused one value for
    // all `i_dst`, silently wrong only when `entropy_array.len() >
    // mixer.len()` (i.e. seeds needing more than 4 uint32 words, roughly
    // seeds >= 2**128) -- every previously-committed SeedSequence/PCG64/
    // PCG64DXSM test used a single-word (small int) seed and never
    // reached this branch. Verified against real numpy 2.5.1 directly:
    // `SeedSequence(2**130+7).generate_state(4, dtype=np.uint64)`.
    for i_src in mixer.len()..entropy_array.len() {
        for i_dst in 0..mixer.len() {
            let hashed = hashmix(entropy_array[i_src], &mut hash_const);
            mixer[i_dst] = mix(mixer[i_dst], hashed);
        }
    }

    mixer
}

/// A single `SeedSequence`: owns its mixed entropy pool and can be asked
/// to `generate_state` any number of 32- or 64-bit words from it.
#[derive(Clone, Debug)]
pub struct SeedSequence {
    pool: Vec<u32>,
}

impl SeedSequence {
    /// `entropy_words` is the caller's already-flattened run-entropy
    /// (`_coerce_to_uint32_array(entropy)`, little-endian 32-bit words of
    /// the (possibly arbitrary-precision) input integer/sequence --
    /// numpy's own int-splitting is Python bigint arithmetic done in the
    /// PyO3 boundary, not here, since this crate has no bigint type).
    /// `spawn_key_words` is `_coerce_to_uint32_array(spawn_key)`, empty
    /// for a non-spawned `SeedSequence`.
    pub fn new(entropy_words: &[u32], spawn_key_words: &[u32], pool_size: usize) -> Self {
        let pool_size = pool_size.max(DEFAULT_POOL_SIZE);
        // get_assembled_entropy's padding rule: if a spawn_key is present
        // AND run-entropy is shorter than the pool, zero-pad run-entropy
        // to pool_size first so it can never collide with the spawn key's
        // own words (gh-16539's fix, numpy 1.19.0+).
        let mut assembled: Vec<u32> = if !spawn_key_words.is_empty() && entropy_words.len() < pool_size {
            let mut padded = entropy_words.to_vec();
            padded.resize(pool_size, 0);
            padded
        } else {
            entropy_words.to_vec()
        };
        assembled.extend_from_slice(spawn_key_words);

        let pool = mix_entropy(pool_size, &assembled);
        Self { pool }
    }

    /// `SeedSequence.generate_state(n_words, dtype=np.uint32)`.
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        let mut state = vec![0u32; n_words];
        for (i_dst, slot) in state.iter_mut().enumerate() {
            let mut data_val = self.pool[i_dst % self.pool.len()];
            data_val ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data_val = data_val.wrapping_mul(hash_const);
            data_val ^= data_val >> XSHIFT;
            *slot = data_val;
        }
        state
    }

    /// `SeedSequence.generate_state(n_words, dtype=np.uint64)`: draws
    /// `2 * n_words` uint32 words, then reinterprets each little-endian
    /// PAIR as one uint64 (`state.astype('<u4').view('<u8')` --
    /// endianness-explicit in numpy specifically so the result is the
    /// same on big- and little-endian hosts; word `2i` is the LOW 32
    /// bits, word `2i+1` is the HIGH 32 bits).
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let words = self.generate_state_u32(n_words * 2);
        words
            .chunks_exact(2)
            .map(|pair| (pair[0] as u64) | ((pair[1] as u64) << 32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verified against real numpy 2.5.1 (this task's `.venv`, numpy used
    // only to derive/verify this fixture offline, never at ionp runtime):
    //   np.random.SeedSequence(0).generate_state(4)
    //     -> array([2968811710, 3677149159,  745650761, 2884920346], uint32)
    //   np.random.SeedSequence(0).generate_state(2, dtype=np.uint64)
    //     -> array([15793235383387715774, 12390638538380655177], uint64)
    #[test]
    fn seed_sequence_zero_matches_numpy() {
        let seq = SeedSequence::new(&[0], &[], 4);
        let state32 = seq.generate_state_u32(4);
        assert_eq!(state32, vec![2968811710, 3677149159, 745650761, 2884920346]);

        let state64 = seq.generate_state_u64(2);
        assert_eq!(state64, vec![15793235383387715774, 12390638538380655177]);
    }

    // np.random.SeedSequence(42).generate_state(4)
    //   -> array([3444837047, 2669555309, 2046530742, 3581440988], uint32)
    #[test]
    fn seed_sequence_42_matches_numpy() {
        let seq = SeedSequence::new(&[42], &[], 4);
        let state32 = seq.generate_state_u32(4);
        assert_eq!(state32, vec![3444837047, 2669555309, 2046530742, 3581440988]);
    }

    // Regression test for the mix_entropy "remaining entropy" loop bug
    // (2026-08-13, this session): entropy words for seed `2**130 + 7`
    // (`_int_to_uint32_array(2**130+7)` == `[7, 0, 0, 0, 4]`, 5 words --
    // one MORE than the default pool_size=4, so this is the first case in
    // this crate's history to actually execute mix_entropy's third loop).
    // Verified against real numpy 2.5.1 directly:
    //   np.random.SeedSequence(2**130+7).generate_state(4)
    //     -> [1956387801, 4266865393, 498201352, 1035958608]
    //   np.random.SeedSequence(2**130+7).generate_state(2, dtype=np.uint64)
    //     -> [18326047321325575129, 4449408341867885320]
    // Before the fix (hashmix hoisted outside the i_dst loop instead of
    // called fresh per i_dst), this produced a DIFFERENT, wrong pool.
    #[test]
    fn seed_sequence_entropy_beyond_pool_matches_numpy() {
        let seq = SeedSequence::new(&[7, 0, 0, 0, 4], &[], 4);
        let state32 = seq.generate_state_u32(4);
        assert_eq!(state32, vec![1956387801, 4266865393, 498201352, 1035958608]);

        let state64 = seq.generate_state_u64(2);
        assert_eq!(state64, vec![18326047321325575129, 4449408341867885320]);
    }
}
