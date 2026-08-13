//! Continuous distribution algorithms transcribed from
//! `numpy/random/src/distributions/distributions.c` at the numpy `v2.5.1`
//! git tag (fetched read-only from the numpy GitHub mirror, read offline --
//! see `ionp-core/src/random/mod.rs`'s module docs for the standing "never
//! call numpy at runtime" rule this whole crate follows).
//!
//! Scope of this pass: `standard_normal`/`standard_exponential`/
//! `standard_gamma` (the three primitives everything else in
//! `distributions.c` is built from) plus the direct wrappers around them
//! that `Generator`'s Python surface exposes: `normal`, `exponential`,
//! `gamma`, `beta`, `chisquare`, `f`, `uniform`, `standard_cauchy`,
//! `pareto`, `weibull`, `power`, `laplace`, `gumbel`, `logistic`,
//! `lognormal`, `rayleigh`, `standard_t`. Every function here takes `&mut
//! dyn BitGen64` and is float64-only (`dtype=float32` is a documented,
//! out-of-scope gap matching the existing `Generator.random`/`.integers`
//! precedent of leaving float32 support out where it would require a
//! second, separately-verified table/constant set -- see
//! `ziggurat_tables.rs`'s own module docs for why the `_float`/`_f` tables
//! were not ported).
//!
//! NOT ported here (out of scope for this pass): `noncentral_chisquare`,
//! `noncentral_f`, `standard_t`'s multivariate cousins, `wald`,
//! `triangular`, `vonmises`, `zipf`, `geometric`, `binomial`, `poisson`,
//! `negative_binomial`, `logseries`, `multivariate_*`, `dirichlet` -- all
//! either need their own additional rejection-sampling machinery
//! (`poisson`'s PTRS, `binomial`'s BTPE) or their own lookup tables this
//! pass did not transcribe. Calling any of them on `ionp`'s `Generator`
//! still raises `AttributeError` (absent surface).

use super::BitGen64;
use super::ziggurat_tables::{
    FE_DOUBLE, FI_DOUBLE, KE_DOUBLE, KI_DOUBLE, WE_DOUBLE, WI_DOUBLE, ZIGGURAT_EXP_R,
    ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R,
};

/// `standard_exponential_unlikely` (`distributions.c`), the rejection-loop
/// tail `random_standard_exponential` falls into on the ~1.1% slow path.
fn standard_exponential_unlikely(bg: &mut dyn BitGen64, idx: usize, x: f64) -> f64 {
    if idx == 0 {
        // "Switch to 1.0 - U to avoid log(0.0)" (GH 13361).
        ZIGGURAT_EXP_R - (-bg.next_f64()).ln_1p()
    } else if (FE_DOUBLE[idx - 1] - FE_DOUBLE[idx]) * bg.next_f64() + FE_DOUBLE[idx] < (-x).exp() {
        x
    } else {
        standard_exponential(bg)
    }
}

/// `random_standard_exponential` (`distributions.c`): ziggurat method.
pub fn standard_exponential(bg: &mut dyn BitGen64) -> f64 {
    let mut ri = bg.next_u64();
    ri >>= 3;
    let idx = (ri & 0xFF) as usize;
    ri >>= 8;
    let x = ri as f64 * WE_DOUBLE[idx];
    if ri < KE_DOUBLE[idx] {
        return x; // 98.9% of the time we return here 1st try.
    }
    standard_exponential_unlikely(bg, idx, x)
}

/// `random_standard_normal` (`distributions.c`): ziggurat method.
pub fn standard_normal(bg: &mut dyn BitGen64) -> f64 {
    loop {
        let r = bg.next_u64();
        let idx = (r & 0xff) as usize;
        let mut r = r >> 8;
        let sign = r & 0x1;
        r >>= 1;
        let rabs = r & 0x000f_ffff_ffff_ffff;
        let mut x = rabs as f64 * WI_DOUBLE[idx];
        if sign & 0x1 != 0 {
            x = -x;
        }
        if rabs < KI_DOUBLE[idx] {
            return x; // 99.3% of the time return here.
        }
        if idx == 0 {
            loop {
                // "Switch to 1.0 - U to avoid log(0.0)" (GH 13361).
                let xx = -ZIGGURAT_NOR_INV_R * (-bg.next_f64()).ln_1p();
                let yy = -(-bg.next_f64()).ln_1p();
                if yy + yy > xx * xx {
                    return if (rabs >> 8) & 0x1 != 0 { -(ZIGGURAT_NOR_R + xx) } else { ZIGGURAT_NOR_R + xx };
                }
            }
        } else if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * bg.next_f64() + FI_DOUBLE[idx] < (-0.5 * x * x).exp() {
            return x;
        }
    }
}

/// `random_standard_gamma` (`distributions.c`): Marsaglia-Tsang (`shape >=
/// 1`) / Ahrens-Dieter GD boost (`shape < 1`, via `pow`+`standard_exponential`
/// rejection). Caller must ensure `shape >= 0.0` (numpy's own
/// `CONS_NON_NEGATIVE` check on `Generator.standard_gamma`'s Python side --
/// see `ionp-py/src/random.rs` for the matching Rust-side check + exact
/// error text).
pub fn standard_gamma(bg: &mut dyn BitGen64, shape: f64) -> f64 {
    if shape == 1.0 {
        return standard_exponential(bg);
    } else if shape == 0.0 {
        return 0.0;
    } else if shape < 1.0 {
        loop {
            let u = bg.next_f64();
            let v = standard_exponential(bg);
            if u <= 1.0 - shape {
                let x = u.powf(1.0 / shape);
                if x <= v {
                    return x;
                }
            } else {
                let y = -((1.0 - u) / shape).ln();
                // FMA contraction, same class as the `x.mul_add(c, 1.0)` fix
                // in the `shape >= 1` branch below -- and missed here.
                // numpy's C is `pow(1.0 - shape + shape*Y, 1./shape)`, which
                // the compiler contracts to `fma(shape, Y, 1.0 - shape)`:
                // ONE rounding. Naive Rust `1.0 - shape + shape * y` rounds
                // TWICE and diverges by ~1 ULP.
                //
                // Measured before the fix, standard_gamma vs numpy 2.5.1
                // over 600 draws (3 seeds x 200):
                //     shape 0.05 ->  0/600 differ
                //     shape 0.3  -> 20/600
                //     shape 0.7  -> 72/600
                //     shape 0.999-> 96/600
                //     shape >= 1 ->  0/600 (already correct)
                // The `u <= 1.0 - shape` branch above needs no fix: it is a
                // pure comparison and a `powf`, with no sum-of-product.
                //
                // This propagated into every consumer of the boost path --
                // `beta` (both params < 1) and `standard_t` (df < 2) were
                // also failing, and are fixed by this one line.
                let x = shape.mul_add(y, 1.0 - shape).powf(1.0 / shape);
                if x <= v + y {
                    return x;
                }
            }
        }
    } else {
        let b = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * b).sqrt();
        loop {
            // `V = 1.0 + c * X` (`distributions.c`) is a single-expression
            // sum-of-product that numpy's compiled build contracts into a
            // genuine fused multiply-add -- empirically confirmed (see this
            // function's module docs / this task's report): the naive
            // two-rounding `1.0 + c * x` does NOT reproduce real numpy
            // 2.5.1's `standard_gamma(shape >= 1)` bit-for-bit, but
            // `x.mul_add(c, 1.0)` does.
            let (x, v) = loop {
                let x = standard_normal(bg);
                let v = x.mul_add(c, 1.0);
                if v > 0.0 {
                    break (x, v);
                }
            };
            let v = v * v * v;
            let u = bg.next_f64();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return b * v;
            }
            // log(0.0) ok here.
            if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
                return b * v;
            }
        }
    }
}

/// `random_uniform`: `lower + range * next_double`. numpy's compiled build
/// contracts this single-expression sum-of-product into a genuine fused
/// multiply-add (empirically confirmed: `math.fma(range, next_double, low)`
/// reproduces real numpy 2.5.1's `Generator.uniform()` bit-for-bit for
/// cases where the naive two-rounding `low + range * next_double()` does
/// NOT -- see `ionp-core`'s test module docs / this task's report for the
/// discovery). `f64::mul_add` is Rust's fused multiply-add (single
/// rounding), matching the contracted C exactly.
pub fn uniform(bg: &mut dyn BitGen64, low: f64, range: f64) -> f64 {
    bg.next_f64().mul_add(range, low)
}

/// `random_normal`: `loc + scale * random_standard_normal`. Same FMA
/// contraction as `uniform` (see its doc comment) -- empirically confirmed
/// against real numpy.
pub fn normal(bg: &mut dyn BitGen64, loc: f64, scale: f64) -> f64 {
    standard_normal(bg).mul_add(scale, loc)
}

/// `random_exponential`: `scale * random_standard_exponential`.
pub fn exponential(bg: &mut dyn BitGen64, scale: f64) -> f64 {
    scale * standard_exponential(bg)
}

/// `random_gamma`: `scale * random_standard_gamma`.
pub fn gamma(bg: &mut dyn BitGen64, shape: f64, scale: f64) -> f64 {
    scale * standard_gamma(bg, shape)
}

const BETA_TINY_THRESHOLD: f64 = 3e-103;

/// `random_beta`. Caller must ensure `a > 0.0 && b > 0.0` (numpy's own
/// `CONS_POSITIVE` on both -- `random_beta` itself assumes `a != 0 && b !=
/// 0`, per its own C comment).
pub fn beta(bg: &mut dyn BitGen64, a: f64, b: f64) -> f64 {
    if a <= 1.0 && b <= 1.0 {
        if a < BETA_TINY_THRESHOLD && b < BETA_TINY_THRESHOLD {
            let u = bg.next_f64();
            return if (a + b) * u < a { 1.0 } else { 0.0 };
        }
        loop {
            let u = bg.next_f64();
            let v = bg.next_f64();
            let x = u.powf(1.0 / a);
            let y = v.powf(1.0 / b);
            let xpy = x + y;
            if xpy <= 1.0 && u + v > 0.0 {
                if x > 0.0 && y > 0.0 {
                    return x / xpy;
                } else {
                    let log_x = u.ln() / a;
                    let log_y = v.ln() / b;
                    let delta = log_x - log_y;
                    return if delta > 0.0 {
                        (-(-delta).exp().ln_1p()).exp()
                    } else {
                        (delta - delta.exp().ln_1p()).exp()
                    };
                }
            }
        }
    } else {
        let ga = standard_gamma(bg, a);
        let gb = standard_gamma(bg, b);
        ga / (ga + gb)
    }
}

/// `random_chisquare`: `2.0 * random_standard_gamma(df / 2.0)`. Caller
/// must ensure `df > 0.0` (`CONS_POSITIVE`).
pub fn chisquare(bg: &mut dyn BitGen64, df: f64) -> f64 {
    2.0 * standard_gamma(bg, df / 2.0)
}

/// `random_f`. Caller must ensure `dfnum > 0.0 && dfden > 0.0`
/// (`CONS_POSITIVE` on both).
pub fn f(bg: &mut dyn BitGen64, dfnum: f64, dfden: f64) -> f64 {
    let subexpr1 = chisquare(bg, dfnum) * dfden;
    let subexpr2 = chisquare(bg, dfden) * dfnum;
    subexpr1 / subexpr2
}

/// `random_standard_cauchy`: ratio of two independent standard normals.
pub fn standard_cauchy(bg: &mut dyn BitGen64) -> f64 {
    standard_normal(bg) / standard_normal(bg)
}

/// `random_pareto`. Caller must ensure `a > 0.0` (`CONS_POSITIVE`).
pub fn pareto(bg: &mut dyn BitGen64, a: f64) -> f64 {
    (standard_exponential(bg) / a).exp_m1()
}

/// `random_weibull`. Caller must ensure `a >= 0.0` (`CONS_NON_NEGATIVE`).
pub fn weibull(bg: &mut dyn BitGen64, a: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    standard_exponential(bg).powf(1.0 / a)
}

/// `random_power`. Caller must ensure `a > 0.0` (`CONS_POSITIVE`).
pub fn power(bg: &mut dyn BitGen64, a: f64) -> f64 {
    (-(-standard_exponential(bg)).exp_m1()).powf(1.0 / a)
}

/// `random_laplace`. Caller must ensure `scale >= 0.0` (`CONS_NON_NEGATIVE`).
pub fn laplace(bg: &mut dyn BitGen64, loc: f64, scale: f64) -> f64 {
    loop {
        let u = bg.next_f64();
        if u >= 0.5 {
            // `loc - scale * log(2.0 - U - U)` (`distributions.c`) is a
            // single-expression sum-of-product that numpy's compiled build
            // contracts into a genuine FMA -- confirmed empirically against
            // real numpy 2.5.1 across 300 seeds x 4 (loc, scale) pairs: the
            // naive `loc - scale * x` form mismatched ~16% of draws, the
            // `mul_add` form matched all of them (same methodology/finding
            // as `uniform`/`normal`/`standard_gamma`).
            return (2.0 - u - u).ln().mul_add(-scale, loc);
        } else if u > 0.0 {
            return (u + u).ln().mul_add(scale, loc);
        }
        // Reject U == 0.0 and draw again.
    }
}

/// `random_gumbel`. Caller must ensure `scale >= 0.0` (`CONS_NON_NEGATIVE`).
pub fn gumbel(bg: &mut dyn BitGen64, loc: f64, scale: f64) -> f64 {
    loop {
        let u = 1.0 - bg.next_f64();
        if u < 1.0 {
            // `loc - scale * log(-log(U))`: same FMA-contraction pattern as
            // `laplace` above, confirmed empirically the same way.
            return (-u.ln()).ln().mul_add(-scale, loc);
        }
        // Reject U == 1.0 (i.e. the original draw was 0.0) and draw again.
    }
}

/// `random_logistic`. Caller must ensure `scale >= 0.0` (`CONS_NON_NEGATIVE`).
pub fn logistic(bg: &mut dyn BitGen64, loc: f64, scale: f64) -> f64 {
    loop {
        let u = bg.next_f64();
        if u > 0.0 {
            // `loc + scale * log(U / (1.0 - U))`: same FMA-contraction
            // pattern as `laplace`/`gumbel` above, confirmed empirically
            // the same way.
            return (u / (1.0 - u)).ln().mul_add(scale, loc);
        }
        // Reject U == 0.0 and draw again.
    }
}

/// `random_lognormal`: `exp(random_normal(mean, sigma))`. Caller must
/// ensure `sigma >= 0.0` (`CONS_NON_NEGATIVE`).
pub fn lognormal(bg: &mut dyn BitGen64, mean: f64, sigma: f64) -> f64 {
    normal(bg, mean, sigma).exp()
}

/// `random_rayleigh`. Caller must ensure `mode >= 0.0` (`CONS_NON_NEGATIVE`).
pub fn rayleigh(bg: &mut dyn BitGen64, mode: f64) -> f64 {
    mode * (2.0 * standard_exponential(bg)).sqrt()
}

/// `random_standard_t`. Caller must ensure `df > 0.0` (`CONS_POSITIVE`).
pub fn standard_t(bg: &mut dyn BitGen64, df: f64) -> f64 {
    let num = standard_normal(bg);
    let denom = standard_gamma(bg, df / 2.0);
    (df / 2.0).sqrt() * num / denom.sqrt()
}

/// `random_wald`. Caller must ensure `mean > 0.0 && scale > 0.0`
/// (`CONS_POSITIVE` on both).
pub fn wald(bg: &mut dyn BitGen64, mean: f64, scale: f64) -> f64 {
    let mut y = standard_normal(bg);
    y = mean * y * y;
    let d = 1.0 + (1.0 + 4.0 * scale / y).sqrt();
    let x = mean * (1.0 - 2.0 / d);
    let u = bg.next_f64();
    if u <= mean / (mean + x) {
        x
    } else {
        mean * mean / x
    }
}

/// `random_vonmises`. Caller must ensure `kappa >= 0.0`
/// (`CONS_NON_NEGATIVE`); `mu` is unconstrained (`CONS_NONE`).
pub fn vonmises(bg: &mut dyn BitGen64, mu: f64, kappa: f64) -> f64 {
    if kappa.is_nan() {
        return f64::NAN;
    }
    if kappa < 1e-8 {
        return std::f64::consts::PI * (2.0 * bg.next_f64() - 1.0);
    }
    let s: f64;
    if kappa < 1e-5 {
        s = 1.0 / kappa + kappa;
    } else if kappa <= 1e6 {
        let r = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
        let rho = (r - (2.0 * r).sqrt()) / (2.0 * kappa);
        s = (1.0 + rho * rho) / (2.0 * rho);
    } else {
        let mut result = mu + (1.0 / kappa).sqrt() * standard_normal(bg);
        if result < -std::f64::consts::PI {
            result += 2.0 * std::f64::consts::PI;
        }
        if result > std::f64::consts::PI {
            result -= 2.0 * std::f64::consts::PI;
        }
        return result;
    }

    let w;
    loop {
        let u = bg.next_f64();
        let z = (std::f64::consts::PI * u).cos();
        // C: `(1 + s * Z)` -- single expression, FMA-contracted.
        let ww = s.mul_add(z, 1.0) / (s + z);
        let y = kappa * (s - ww);
        let v = bg.next_f64();
        // C: `Y * (2 - Y) - V` -- single expression, FMA-contracted
        // (`fma(Y, 2-Y, -V)`).
        if y.mul_add(2.0 - y, -v) >= 0.0 || (y / v).ln() + 1.0 - y >= 0.0 {
            w = ww;
            break;
        }
    }

    let u = bg.next_f64();
    let mut result = w.acos();
    if u < 0.5 {
        result = -result;
    }
    result += mu;
    let neg = result < 0.0;
    let mut m = result.abs();
    // `fmod` (C) truncates toward zero, same as Rust's `%` -- both operands
    // here are always non-negative so this is exact, not merely close.
    m = (m + std::f64::consts::PI) % (2.0 * std::f64::consts::PI) - std::f64::consts::PI;
    if neg {
        m = -m;
    }
    m
}

/// `random_triangular`. Caller must ensure `left <= mode <= right` and
/// `left != right` (numpy's Cython-level checks -- see
/// `ionp-py/src/random.rs`).
pub fn triangular(bg: &mut dyn BitGen64, left: f64, mode: f64, right: f64) -> f64 {
    let base = right - left;
    let leftbase = mode - left;
    let ratio = leftbase / base;
    let leftprod = leftbase * base;
    let rightprod = (right - mode) * base;

    let u = bg.next_f64();
    if u <= ratio {
        left + (u * leftprod).sqrt()
    } else {
        right - ((1.0 - u) * rightprod).sqrt()
    }
}

/// `random_noncentral_chisquare`. Caller must ensure `df > 0.0` and
/// `nonc >= 0.0` (numpy's `CONS_POSITIVE`/`CONS_NON_NEGATIVE`).
pub fn noncentral_chisquare(bg: &mut dyn BitGen64, df: f64, nonc: f64) -> f64 {
    if nonc.is_nan() {
        return f64::NAN;
    }
    if nonc == 0.0 {
        return chisquare(bg, df);
    }
    if df > 1.0 {
        let chi2 = chisquare(bg, df - 1.0);
        let n = standard_normal(bg) + nonc.sqrt();
        // C: `Chi2 + n * n` -- single expression, FMA-contracted
        // (`fma(n, n, Chi2)`).
        n.mul_add(n, chi2)
    } else {
        let i = super::discrete::poisson(bg, nonc / 2.0);
        chisquare(bg, df + 2.0 * (i as f64))
    }
}

/// `random_noncentral_f`. Caller must ensure `dfnum > 0.0 && dfden > 0.0
/// && nonc >= 0.0`.
pub fn noncentral_f(bg: &mut dyn BitGen64, dfnum: f64, dfden: f64, nonc: f64) -> f64 {
    let t = noncentral_chisquare(bg, dfnum, nonc) * dfden;
    t / (chisquare(bg, dfden) * dfnum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Pcg64, SeedSequence};

    /// Every reference vector below was captured OUTSIDE this test file, by
    /// hand-running real numpy 2.5.1 (`.venv`) with `default_rng(42)` and
    /// printing `.tolist()` of each call (see this task's own session
    /// commands, not transcribed from memory or from numpy docs). PCG64
    /// seeding itself is separately verified bit-exact in `pcg64.rs`'s own
    /// tests, so any mismatch here isolates to THIS file's distribution
    /// algorithm, not to seeding.
    fn pcg64(seed: u32) -> Pcg64 {
        let seq = SeedSequence::new(&[seed], &[], 4);
        let v = seq.generate_state_u64(4);
        Pcg64::from_seed_words([v[0], v[1], v[2], v[3]])
    }

    // np.random.default_rng(42).standard_normal(5) ->
    // [0.30471707975443135, -1.0399841062404955, 0.7504511958064572,
    //  0.9405647163912139, -1.9510351886538364]
    #[test]
    fn standard_normal_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| standard_normal(&mut bg)).collect();
        assert_eq!(
            got,
            vec![0.30471707975443135, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139, -1.9510351886538364]
        );
    }

    // np.random.default_rng(42).standard_exponential(5) ->
    // [2.4042086039659947, 2.3361896558244535, 2.384760999874255,
    //  0.2797942898515832, 0.08643739969837702]
    #[test]
    fn standard_exponential_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| standard_exponential(&mut bg)).collect();
        assert_eq!(
            got,
            vec![2.4042086039659947, 2.3361896558244535, 2.384760999874255, 0.2797942898515832, 0.08643739969837702]
        );
    }

    // np.random.default_rng(42).standard_gamma(0.5, 5) ->
    // [0.8045001461315131, 1.2802929678223112, 0.008869372855151365,
    //  0.7557948464629789, 0.016413102877324694]
    #[test]
    fn standard_gamma_shape_lt_1_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| standard_gamma(&mut bg, 0.5)).collect();
        assert_eq!(
            got,
            vec![0.8045001461315131, 1.2802929678223112, 0.008869372855151365, 0.7557948464629789, 0.016413102877324694]
        );
    }

    // np.random.default_rng(42).standard_gamma(3.5, 5) ->
    // [3.7404543952771747, 4.69862637115865, 3.39965153929014,
    //  3.136862782023423, 5.003501497445522]
    #[test]
    fn standard_gamma_shape_gt_1_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| standard_gamma(&mut bg, 3.5)).collect();
        assert_eq!(
            got,
            vec![3.7404543952771747, 4.69862637115865, 3.39965153929014, 3.136862782023423, 5.003501497445522]
        );
    }

    // np.random.default_rng(42).normal(2.0, 3.0, 5) ->
    // [2.914151239263294, -1.1199523187214866, 4.251353587419372,
    //  4.821694149173641, -3.853105565961509]
    #[test]
    fn normal_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| normal(&mut bg, 2.0, 3.0)).collect();
        assert_eq!(
            got,
            vec![2.914151239263294, -1.1199523187214866, 4.251353587419372, 4.821694149173641, -3.853105565961509]
        );
    }

    // np.random.default_rng(42).exponential(2.5, 5) ->
    // [6.0105215099149865, 5.840474139561134, 5.961902499685637,
    //  0.6994857246289581, 0.21609349924594257]
    #[test]
    fn exponential_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| exponential(&mut bg, 2.5)).collect();
        assert_eq!(
            got,
            vec![6.0105215099149865, 5.840474139561134, 5.961902499685637, 0.6994857246289581, 0.21609349924594257]
        );
    }

    // np.random.default_rng(42).gamma(2.0, 1.5, 5) ->
    // [3.137725905749924, 4.253018384678798, 2.7558233704607065,
    //  2.467605633751651, 4.618882994332351]
    #[test]
    fn gamma_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| gamma(&mut bg, 2.0, 1.5)).collect();
        assert_eq!(
            got,
            vec![3.137725905749924, 4.253018384678798, 2.7558233704607065, 2.467605633751651, 4.618882994332351]
        );
    }

    // np.random.default_rng(42).beta(0.5, 0.5, 5) ->
    // [0.7566840941053092, 0.009232118044248207, 0.07485660069290462,
    //  0.1379900752243652, 0.791997219411125]  (Johnk's algorithm path)
    #[test]
    fn beta_both_lt_1_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| beta(&mut bg, 0.5, 0.5)).collect();
        assert_eq!(
            got,
            vec![0.7566840941053092, 0.009232118044248207, 0.07485660069290462, 0.1379900752243652, 0.791997219411125]
        );
    }

    // np.random.default_rng(42).beta(3.0, 5.0, 5) ->
    // [0.3301966042213875, 0.383540750965424, 0.4763328596774486,
    //  0.38883929463672495, 0.5054995443255691]  (gamma-ratio path)
    #[test]
    fn beta_gamma_ratio_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| beta(&mut bg, 3.0, 5.0)).collect();
        assert_eq!(
            got,
            vec![0.3301966042213875, 0.383540750965424, 0.4763328596774486, 0.38883929463672495, 0.5054995443255691]
        );
    }

    // np.random.default_rng(42).chisquare(4.0, 5) ->
    // [4.183634540999899, 5.6706911795717305, 3.6744311606142754,
    //  3.290140845002201, 6.158510659109801]
    #[test]
    fn chisquare_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| chisquare(&mut bg, 4.0)).collect();
        assert_eq!(
            got,
            vec![4.183634540999899, 5.6706911795717305, 3.6744311606142754, 3.290140845002201, 6.158510659109801]
        );
    }

    // np.random.default_rng(42).f(3.0, 5.0, 5) ->
    // [0.7338468182700809, 1.0194919811685932, 1.764006029769655,
    //  1.0570679372181275, 2.095344372199731]
    #[test]
    fn f_dist_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| f(&mut bg, 3.0, 5.0)).collect();
        assert_eq!(
            got,
            vec![0.7338468182700809, 1.0194919811685932, 1.764006029769655, 1.0570679372181275, 2.095344372199731]
        );
    }

    // np.random.default_rng(42).uniform(-1.0, 2.0, 5) ->
    // [1.32186814566789, 0.31663531925615696, 1.5757937597341474,
    //  1.0921040871780918, -0.7174679563370514]
    #[test]
    fn uniform_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| uniform(&mut bg, -1.0, 3.0)).collect();
        assert_eq!(
            got,
            vec![1.32186814566789, 0.31663531925615696, 1.5757937597341474, 1.0921040871780918, -0.7174679563370514]
        );
    }

    // np.random.default_rng(42).standard_cauchy(5) ->
    // [-0.2930016698581793, 0.7978730040882357, 1.4982843597001279,
    //  -0.40424789785555826, 0.01969553613971374]
    #[test]
    fn standard_cauchy_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| standard_cauchy(&mut bg)).collect();
        assert_eq!(
            got,
            vec![-0.2930016698581793, 0.7978730040882357, 1.4982843597001279, -0.40424789785555826, 0.01969553613971374]
        );
    }

    // np.random.default_rng(42).pareto(2.0, 5) ->
    // [2.327110807402737, 2.215860031777119, 2.2949154249682957,
    //  0.15015549344454615, 0.044166228878177186]
    #[test]
    fn pareto_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| pareto(&mut bg, 2.0)).collect();
        assert_eq!(
            got,
            vec![2.327110807402737, 2.215860031777119, 2.2949154249682957, 0.15015549344454615, 0.044166228878177186]
        );
    }

    // np.random.default_rng(42).weibull(1.5, 5) ->
    // [1.794656893050126, 1.7606460812380325, 1.7849658315895707,
    //  0.42778534056545525, 0.19549446343791463]
    #[test]
    fn weibull_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| weibull(&mut bg, 1.5)).collect();
        assert_eq!(
            got,
            vec![1.794656893050126, 1.7606460812380325, 1.7849658315895707, 0.42778534056545525, 0.19549446343791463]
        );
    }

    // np.random.default_rng(42).power(2.0, 5) ->
    // [0.9537625702241296, 0.9504233899416021, 0.9528320988047685,
    //  0.4940250703824942, 0.28776211782692906]
    #[test]
    fn power_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| power(&mut bg, 2.0)).collect();
        assert_eq!(
            got,
            vec![0.9537625702241296, 0.9504233899416021, 0.9528320988047685, 0.4940250703824942, 0.28776211782692906]
        );
    }

    // np.random.default_rng(42).laplace(1.0, 2.0, 5) ->
    // [2.5877572852834136, 0.7392287473869138, 3.526001268797544,
    //  2.0041812970319457, -2.3388568280819664]
    #[test]
    fn laplace_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| laplace(&mut bg, 1.0, 2.0)).collect();
        assert_eq!(
            got,
            vec![2.5877572852834136, 0.7392287473869138, 3.526001268797544, 2.0041812970319457, -2.3388568280819664]
        );
    }

    // np.random.default_rng(42).gumbel(1.0, 2.0, 5) ->
    // [0.20644393339980482, 2.0969936744344135, -0.3419542775542994,
    //  0.6433096285423923, 5.627054676785422]
    #[test]
    fn gumbel_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| gumbel(&mut bg, 1.0, 2.0)).collect();
        assert_eq!(
            got,
            vec![0.20644393339980482, 2.0969936744344135, -0.3419542775542994, 0.6433096285423923, 5.627054676785422]
        );
    }

    // np.random.default_rng(42).logistic(1.0, 2.0, 5) ->
    // [3.4615712627710136, 0.5085698102635776, 4.60738653822512,
    //  2.669591680475995, -3.5273277085604473]
    #[test]
    fn logistic_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| logistic(&mut bg, 1.0, 2.0)).collect();
        assert_eq!(
            got,
            vec![3.4615712627710136, 0.5085698102635776, 4.60738653822512, 2.669591680475995, -3.5273277085604473]
        );
    }

    // np.random.default_rng(42).lognormal(0.5, 1.0, 5) ->
    // [2.2360637816057536, 0.5827575145081694, 3.4919181408987034,
    //  4.223079986233102, 0.23432758923426197]
    #[test]
    fn lognormal_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| lognormal(&mut bg, 0.5, 1.0)).collect();
        assert_eq!(
            got,
            vec![2.2360637816057536, 0.5827575145081694, 3.4919181408987034, 4.223079986233102, 0.23432758923426197]
        );
    }

    // np.random.default_rng(42).rayleigh(2.0, 5) ->
    // [4.385620689449551, 4.323137430916999, 4.367847066804656,
    //  1.496113070196456, 0.8315643075475379]
    #[test]
    fn rayleigh_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| rayleigh(&mut bg, 2.0)).collect();
        assert_eq!(
            got,
            vec![4.385620689449551, 4.323137430916999, 4.367847066804656, 1.496113070196456, 0.8315643075475379]
        );
    }

    // np.random.default_rng(42).standard_t(5.0, 5) ->
    // [0.48968077961378614, 0.9679905861688154, -0.024902633373061672,
    //  0.8170873490395495, 0.6947681050011598]
    #[test]
    fn standard_t_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| standard_t(&mut bg, 5.0)).collect();
        assert_eq!(
            got,
            vec![0.48968077961378614, 0.9679905861688154, -0.024902633373061672, 0.8170873490395495, 0.6947681050011598]
        );
    }

    // np.random.default_rng(42).wald(3.0, 2.0, 5) ->
    // [2.069990769639344, 1.2325882864415432, 22.733531404311037,
    //  3.5079249356569857, 2.9389004658607654]
    #[test]
    fn wald_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| wald(&mut bg, 3.0, 2.0)).collect();
        assert_eq!(
            got,
            vec![2.069990769639344, 1.2325882864415432, 22.733531404311037, 3.5079249356569857, 2.9389004658607654]
        );
    }

    // np.random.default_rng(42).vonmises(0.5, 4.0, 5) ->
    // [0.3982847795528137, 0.31443609227022407, 1.6167872858198393,
    //  0.3428134398551954, 0.6232124440455369]
    #[test]
    fn vonmises_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| vonmises(&mut bg, 0.5, 4.0)).collect();
        assert_eq!(
            got,
            vec![0.3982847795528137, 0.31443609227022407, 1.6167872858198393, 0.3428134398551954, 0.6232124440455369]
        );
    }

    // np.random.default_rng(42).vonmises(0.0, 1e-9, 5) -- kappa < 1e-8,
    // exercises the near-uniform-fallback branch.
    // -> [1.7213166190998062, -0.3840380893017967, 2.2531371815723604,
    //     1.2400999002927888, -2.5498589250729733]
    #[test]
    fn vonmises_tiny_kappa_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| vonmises(&mut bg, 0.0, 1e-9)).collect();
        assert_eq!(
            got,
            vec![1.7213166190998062, -0.3840380893017967, 2.2531371815723604, 1.2400999002927888, -2.5498589250729733]
        );
    }

    // np.random.default_rng(42).vonmises(1.0, 1e7, 5) -- kappa > 1e6,
    // exercises the wrapped-normal-fallback branch.
    // -> [1.000096360001398, 0.9996711281493905, 1.0002373135051545,
    //     1.0002974326790586, 0.9993830285008718]
    #[test]
    fn vonmises_huge_kappa_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| vonmises(&mut bg, 1.0, 1e7)).collect();
        assert_eq!(
            got,
            vec![1.000096360001398, 0.9996711281493905, 1.0002373135051545, 1.0002974326790586, 0.9993830285008718]
        );
    }

    // np.random.default_rng(42).triangular(0.0, 0.3, 1.0, 5) ->
    // [0.6022176901736005, 0.37327430866960354, 0.685386815180876,
    //  0.5397366192510583, 0.1680868952842394]
    #[test]
    fn triangular_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| triangular(&mut bg, 0.0, 0.3, 1.0)).collect();
        assert_eq!(
            got,
            vec![0.6022176901736005, 0.37327430866960354, 0.685386815180876, 0.5397366192510583, 0.1680868952842394]
        );
    }

    // np.random.default_rng(42).noncentral_chisquare(3.0, 5.0, 5) --
    // df > 1 branch.
    // -> [6.239033835018609, 14.860517271645726, 1.04502247498635,
    //     6.505650898005328, 2.071343917041313]
    #[test]
    fn noncentral_chisquare_df_gt_1_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| noncentral_chisquare(&mut bg, 3.0, 5.0)).collect();
        assert_eq!(
            got,
            vec![6.239033835018609, 14.860517271645726, 1.04502247498635, 6.505650898005328, 2.071343917041313]
        );
    }

    // np.random.default_rng(42).noncentral_chisquare(0.5, 5.0, 5) --
    // df <= 1 branch (routes through discrete::poisson).
    // -> [3.7269602601566976, 6.820223856948423, 7.185683282153251,
    //     16.82638271486298, 2.81454483504597]
    #[test]
    fn noncentral_chisquare_df_lt_1_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| noncentral_chisquare(&mut bg, 0.5, 5.0)).collect();
        assert_eq!(
            got,
            vec![3.7269602601566976, 6.820223856948423, 7.185683282153251, 16.82638271486298, 2.81454483504597]
        );
    }

    // np.random.default_rng(42).noncentral_f(3.0, 10.0, 5.0, 5) ->
    // [1.6039971246646807, 0.351979088871238, 0.5050218011845795,
    //  4.281564392349691, 0.44253582044002177]
    #[test]
    fn noncentral_f_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<f64> = (0..5).map(|_| noncentral_f(&mut bg, 3.0, 10.0, 5.0)).collect();
        assert_eq!(
            got,
            vec![1.6039971246646807, 0.351979088871238, 0.5050218011845795, 4.281564392349691, 0.44253582044002177]
        );
    }
}
