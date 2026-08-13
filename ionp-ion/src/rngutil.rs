//! Minimal, dependency-free deterministic PRNG for the randomized range
//! finder in `detect.rs`. SplitMix64 (Vigna) + Box-Muller, seeded
//! explicitly so detection is reproducible run-to-run — no `rand` crate
//! pulled in just for a handful of Gaussian probe vectors.

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 significant bits
        let v = (bits as f64) * (1.0 / (1u64 << 53) as f64);
        v.max(f64::MIN_POSITIVE)
    }

    /// Standard normal via Box-Muller.
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_is_roughly_standard() {
        let mut rng = SplitMix64::new(42);
        let n = 20_000;
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for _ in 0..n {
            let g = rng.next_gaussian();
            sum += g;
            sumsq += g * g;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean={mean}");
        assert!((var - 1.0).abs() < 0.1, "var={var}");
    }
}
