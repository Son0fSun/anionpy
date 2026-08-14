//! Discrete-distribution algorithms transcribed from
//! `numpy/random/src/distributions/distributions.c` and
//! `numpy/random/src/distributions/random_hypergeometric.c` at the numpy
//! `v2.5.1` git tag (fetched read-only from the numpy GitHub mirror, read
//! offline -- see `ionp-core/src/random/mod.rs`'s module docs for the
//! standing "never call numpy at runtime" rule this crate follows).
//!
//! Scope of this pass: `poisson` (both `random_poisson_mult`, `lam < 10`,
//! and `random_poisson_ptrs`, `lam >= 10`), `binomial` (both
//! `random_binomial_inversion` and the BTPE rejection algorithm
//! `random_binomial_btpe`), `negative_binomial` (gamma-then-Poisson
//! mixture, reuses `distributions::standard_gamma`), `geometric` (both
//! the direct-search and inversion paths), `zipf`, `logseries`, and
//! `hypergeometric` (both the direct-sample path and the
//! ratio-of-uniforms `hypergeometric_hrua` path, via `logfactorial.rs`).
//!
//! numpy's `binomial_t` cache (`self._binomial` on the Cython
//! `Generator`, keyed on `(n, p)` across consecutive calls to skip
//! recomputing `r`/`q`/`fm`/... when unchanged) is deliberately NOT
//! reproduced here: it is a pure performance optimization -- every
//! `binomial_t` field is a **function of `(n, p)` alone**, so recomputing
//! it on every call (as this module does) produces bit-identical output
//! to a cache hit, just slower. No caller of this module needs to thread
//! cache state through.

use super::bounded;
use super::distributions::standard_gamma;
use super::logfactorial::logfactorial;
use super::BitGen64;

// ---------------------------------------------------------------------------
// Poisson
// ---------------------------------------------------------------------------

/// `random_poisson_mult` (`lam < 10`).
fn poisson_mult(bg: &mut dyn BitGen64, lam: f64) -> i64 {
    let enlam = (-lam).exp();
    let mut x: i64 = 0;
    let mut prod = 1.0;
    loop {
        let u = bg.next_f64();
        prod *= u;
        if prod > enlam {
            x += 1;
        } else {
            return x;
        }
    }
}

const LS2PI: f64 = 0.918_938_533_204_672_67;
const TWELFTH: f64 = 0.083_333_333_333_333_33;

/// `random_poisson_ptrs` (`lam >= 10`): Hoermann's transformed rejection
/// method. `LS2PI`/`TWELFTH` are unused by the transcribed body (numpy
/// defines them but the actual `random_loggam` call below supersedes the
/// `LS2PI`/Stirling inline it once used) -- kept as named constants purely
/// to mirror the C source 1:1 for auditability.
fn poisson_ptrs(bg: &mut dyn BitGen64, lam: f64) -> i64 {
    let _ = (LS2PI, TWELFTH); // silence "unused" while keeping the C-source names visible.
    let slam = lam.sqrt();
    let loglam = lam.ln();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let invalpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);

    loop {
        let u = bg.next_f64() - 0.5;
        let v = bg.next_f64();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k as i64;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if (v.ln() + invalpha.ln() - (a / (us * us) + b).ln())
            <= (-lam + k * loglam - random_loggam(k + 1.0))
        {
            return k as i64;
        }
    }
}

/// `random_loggam` (`distributions.c`): a hand-transcribed log-gamma
/// series used only by `poisson_ptrs` above (matching numpy's own scope
/// for this helper -- it is NOT numpy's general `scipy.special.gammaln`,
/// just this one function, transcribed 1:1 including its exact
/// iteration/shift structure).
fn random_loggam(x: f64) -> f64 {
    const A: [f64; 10] = [
        8.333_333_333_333_333e-2,
        -2.777_777_777_777_778e-3,
        7.936_507_936_507_937e-4,
        -5.952_380_952_380_952e-4,
        8.417_508_417_508_418e-4,
        -1.917_526_917_526_918e-3,
        6.410_256_410_256_41e-3,
        -2.955_065_359_477_124e-2,
        1.796_443_723_688_307e-1,
        -1.392_432_216_905_9,
    ];
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let n: i64 = if x < 7.0 { (7.0 - x) as i64 } else { 0 };
    let mut x0 = x + (n as f64);
    let x2 = (1.0 / x0) * (1.0 / x0);
    const LG2PI: f64 = 1.837_877_066_409_345_3;
    let mut gl0 = A[9];
    for &c in A[..9].iter().rev() {
        gl0 = gl0 * x2 + c;
    }
    let mut gl = gl0 / x0 + 0.5 * LG2PI + (x0 - 0.5) * x0.ln() - x0;
    if x < 7.0 {
        for _ in 1..=n {
            gl -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    gl
}

/// `random_poisson`. Caller must ensure `0.0 <= lam <= POISSON_LAM_MAX`
/// (numpy's own `CONS_POISSON`).
pub fn poisson(bg: &mut dyn BitGen64, lam: f64) -> i64 {
    if lam >= 10.0 {
        poisson_ptrs(bg, lam)
    } else if lam == 0.0 {
        0
    } else {
        poisson_mult(bg, lam)
    }
}

// ---------------------------------------------------------------------------
// Binomial
// ---------------------------------------------------------------------------

/// `random_binomial_btpe`.
fn binomial_btpe(bg: &mut dyn BitGen64, n: i64, p: f64) -> i64 {
    let r = p.min(1.0 - p);
    let q = 1.0 - r;
    let fm = (n as f64) * r + r;
    let m = fm.floor() as i64;
    let p1 = (2.195 * ((n as f64) * r * q).sqrt() - 4.6 * q).floor() + 0.5;
    let xm = (m as f64) + 0.5;
    let xl = xm - p1;
    let xr = xm + p1;
    let c = 0.134 + 20.5 / (15.3 + (m as f64));
    let a1 = (fm - xl) / (fm - xl * r);
    let laml = a1 * (1.0 + a1 / 2.0);
    let a2 = (xr - fm) / (xr * q);
    let lamr = a2 * (1.0 + a2 / 2.0);
    let p2 = p1 * (1.0 + 2.0 * c);
    let p3 = p2 + c / laml;
    let p4 = p3 + c / lamr;
    let nrq = (n as f64) * r * q;

    let y = 'outer: loop {
        let u = bg.next_f64() * p4;
        let mut v = bg.next_f64();
        let y: i64;
        if u <= p1 {
            y = (xm - p1 * v + u).floor() as i64;
            break 'outer y;
        } else if u <= p2 {
            let x = xl + (u - p1) / c;
            v = v * c + 1.0 - ((m as f64) - x + 0.5).abs() / p1;
            if v > 1.0 {
                continue;
            }
            y = x.floor() as i64;
            // Step50
            if step50_accept(y, m, n, r, q, nrq, v) {
                break 'outer y;
            }
            continue;
        } else if u <= p3 {
            let yy = (xl + v.ln() / laml).floor();
            if yy < 0.0 || v == 0.0 {
                continue;
            }
            let y2 = yy as i64;
            let v2 = v * (u - p2) * laml;
            if step50_accept(y2, m, n, r, q, nrq, v2) {
                break 'outer y2;
            }
            continue;
        } else {
            let yy = (xr - v.ln() / lamr).floor();
            if yy > (n as f64) || v == 0.0 {
                continue;
            }
            let y2 = yy as i64;
            let v2 = v * (u - p3) * lamr;
            if step50_accept(y2, m, n, r, q, nrq, v2) {
                break 'outer y2;
            }
            continue;
        }
    };
    if p > 0.5 {
        n - y
    } else {
        y
    }
}

/// Step50/Step52/Step60 acceptance test shared by all three BTPE regions
/// that reach it (u<=p1 skips straight to acceptance without this test).
fn step50_accept(y: i64, m: i64, n: i64, r: f64, q: f64, nrq: f64, v: f64) -> bool {
    let k = (y - m).abs();
    if !(k > 20 && (k as f64) < nrq / 2.0 - 1.0) {
        let s = r / q;
        let a = s * ((n + 1) as f64);
        let mut f = 1.0;
        if m < y {
            for i in (m + 1)..=y {
                f *= a / (i as f64) - s;
            }
        } else if m > y {
            for i in (y + 1)..=m {
                f /= a / (i as f64) - s;
            }
        }
        return v <= f;
    }
    // Step52: squeeze approximation.
    let kf = k as f64;
    let rho = (kf / nrq) * ((kf * (kf / 3.0 + 0.625) + 0.166_666_666_666_666_66) / nrq + 0.5);
    let t = -kf * kf / (2.0 * nrq);
    let a_ln = v.ln();
    if a_ln < t - rho {
        return true;
    }
    if a_ln > t + rho {
        return false;
    }
    let x1 = (y as f64) + 1.0;
    let f1 = (m as f64) + 1.0;
    let z = (n as f64) + 1.0 - (m as f64);
    let w = (n as f64) - (y as f64) + 1.0;
    let x2 = x1 * x1;
    let f2 = f1 * f1;
    let z2 = z * z;
    let w2 = w * w;
    let xm = (m as f64) + 0.5;
    let bound = xm * (f1 / x1).ln()
        + ((n as f64) - (m as f64) + 0.5) * (z / w).ln()
        + ((y as f64) - (m as f64)) * (w * r / (x1 * q)).ln()
        + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0
        + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0
        - (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0
        - (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0;
    a_ln <= bound
}

/// `random_binomial_inversion`.
fn binomial_inversion(bg: &mut dyn BitGen64, n: i64, p: f64) -> i64 {
    let q = 1.0 - p;
    let qn = ((n as f64) * (-p).ln_1p()).exp();
    let np = (n as f64) * p;
    let bound = (n as f64).min(np + 10.0 * (np * q + 1.0).sqrt()) as i64;

    let mut x: i64 = 0;
    let mut px = qn;
    let mut u = bg.next_f64();
    while u > px {
        x += 1;
        if x > bound {
            x = 0;
            px = qn;
            u = bg.next_f64();
        } else {
            u -= px;
            px = (((n - x + 1) as f64) * p * px) / ((x as f64) * q);
        }
    }
    x
}

/// `random_binomial`. Caller must ensure `n >= 0` and `0.0 <= p <= 1.0`
/// (numpy's `CONS_NON_NEGATIVE` on `n`, `CONS_BOUNDED_0_1` on `p`).
pub fn binomial(bg: &mut dyn BitGen64, p: f64, n: i64) -> i64 {
    if n == 0 || p == 0.0 {
        return 0;
    }
    if p <= 0.5 {
        if p * (n as f64) <= 30.0 {
            binomial_inversion(bg, n, p)
        } else {
            binomial_btpe(bg, n, p)
        }
    } else {
        let q = 1.0 - p;
        if q * (n as f64) <= 30.0 {
            n - binomial_inversion(bg, n, q)
        } else {
            n - binomial_btpe(bg, n, q)
        }
    }
}

// ---------------------------------------------------------------------------
// Negative binomial (built on `standard_gamma` + `poisson`).
// ---------------------------------------------------------------------------

/// `random_negative_binomial`. Caller must ensure `n > 0`, `0.0 < p <=
/// 1.0` (numpy's `CONS_POSITIVE_NOT_NAN` on `n`, `CONS_BOUNDED_GT_0_1` on
/// `p`), plus the "n too large or p too small" `POISSON_LAM_MAX` guard on
/// the derived Poisson `lam` (see `ionp-py/src/random.rs` for that check's
/// exact transcription).
pub fn negative_binomial(bg: &mut dyn BitGen64, n: f64, p: f64) -> i64 {
    let y = standard_gamma(bg, n) * (1.0 - p) / p;
    poisson(bg, y)
}

// ---------------------------------------------------------------------------
// Geometric
// ---------------------------------------------------------------------------

fn geometric_search(bg: &mut dyn BitGen64, p: f64) -> i64 {
    let mut x: i64 = 1;
    let mut prod = p;
    let mut sum = p;
    let q = 1.0 - p;
    let u = bg.next_f64();
    while u > sum {
        prod *= q;
        sum += prod;
        x += 1;
    }
    x
}

fn geometric_inversion(bg: &mut dyn BitGen64, p: f64) -> i64 {
    use super::distributions::standard_exponential;
    let z = (-standard_exponential(bg) / (-p).ln_1p()).ceil();
    if z >= 9.223_372_036_854_776e18 {
        i64::MAX
    } else {
        z as i64
    }
}

/// `random_geometric`. Caller must ensure `0.0 < p < 1.0`
/// (`CONS_BOUNDED_GT_0_1` per `_generator.pyx`'s `geometric`, which is
/// actually the SAME check as logseries's despite the docstring's `(0,
/// 1]` wording -- see `ionp-py/src/random.rs` for the exact transcription
/// of numpy's own bound).
pub fn geometric(bg: &mut dyn BitGen64, p: f64) -> i64 {
    if p >= 0.333_333_333_333_333_33 {
        geometric_search(bg, p)
    } else {
        geometric_inversion(bg, p)
    }
}

// ---------------------------------------------------------------------------
// Zipf
// ---------------------------------------------------------------------------

/// `random_zipf`. Caller must ensure `a > 1.0` (`CONS_GT_1`).
pub fn zipf(bg: &mut dyn BitGen64, a: f64) -> i64 {
    const RAND_INT_MAX: f64 = i64::MAX as f64;
    if a >= 1025.0 {
        return 1;
    }
    let am1 = a - 1.0;
    let b = 2.0f64.powf(am1);
    let umin = RAND_INT_MAX.powf(-am1);
    loop {
        let u01 = bg.next_f64();
        let u = u01 * umin + (1.0 - u01);
        let v = bg.next_f64();
        let x = u.powf(-1.0 / am1).floor();
        if x > RAND_INT_MAX || x < 1.0 {
            continue;
        }
        let t = (1.0 + 1.0 / x).powf(am1);
        if v * x * (t - 1.0) / (b - 1.0) <= t / b {
            return x as i64;
        }
    }
}

// ---------------------------------------------------------------------------
// Logseries
// ---------------------------------------------------------------------------

/// `random_logseries`. Caller must ensure `0.0 < p < 1.0`
/// (`CONS_BOUNDED_LT_0_1`).
pub fn logseries(bg: &mut dyn BitGen64, p: f64) -> i64 {
    let r = (-p).ln_1p();
    loop {
        let v = bg.next_f64();
        if v >= p {
            return 1;
        }
        let u = bg.next_f64();
        let q = -(r * u).exp_m1();
        if v <= q * q {
            let result = (1.0 + v.ln() / q.ln()).floor();
            if result < 1.0 || v == 0.0 {
                continue;
            }
            return result as i64;
        }
        if v >= q {
            return 1;
        }
        return 2;
    }
}

// ---------------------------------------------------------------------------
// Hypergeometric
// ---------------------------------------------------------------------------

fn hypergeometric_sample(
    bg: &mut dyn BitGen64,
    good: i64,
    bad: i64,
    sample: i64,
) -> i64 {
    use super::bounded::random_interval;
    let total = good + bad;
    let mut computed_sample = if sample > total / 2 { total - sample } else { sample };
    let mut remaining_total = total;
    let mut remaining_good = good;

    while computed_sample > 0 && remaining_good > 0 && remaining_total > remaining_good {
        remaining_total -= 1;
        if (random_interval(bg, remaining_total as u64) as i64) < remaining_good {
            remaining_good -= 1;
        }
        computed_sample -= 1;
    }

    if remaining_total == remaining_good {
        remaining_good -= computed_sample;
    }

    if sample > total / 2 {
        remaining_good
    } else {
        good - remaining_good
    }
}

const HRUA_D1: f64 = 1.715_527_769_921_413_5;
const HRUA_D2: f64 = 0.898_916_162_058_898_8;

fn hypergeometric_hrua(bg: &mut dyn BitGen64, good: i64, bad: i64, sample: i64) -> i64 {
    let popsize = good + bad;
    let computed_sample = sample.min(popsize - sample);
    let mingoodbad = good.min(bad);
    let maxgoodbad = good.max(bad);

    let p = (mingoodbad as f64) / (popsize as f64);
    let q = (maxgoodbad as f64) / (popsize as f64);

    let mu = (computed_sample as f64) * p;
    let a = mu + 0.5;

    let var = ((popsize - computed_sample) as f64) * (computed_sample as f64) * p * q / ((popsize - 1) as f64);
    let c = (var + 0.5).sqrt();
    let h = HRUA_D1 * c + HRUA_D2;

    let m = (((computed_sample + 1) as f64) * ((mingoodbad + 1) as f64) / ((popsize + 2) as f64)).floor() as i64;

    let g = logfactorial(m)
        + logfactorial(mingoodbad - m)
        + logfactorial(computed_sample - m)
        + logfactorial(maxgoodbad - computed_sample + m);

    let b = (computed_sample.min(mingoodbad) + 1) as f64;
    let b = b.min((a + 16.0 * c).floor());

    let mut k: i64;
    loop {
        let u = bg.next_f64();
        let v = bg.next_f64();
        let x = a + h * (v - 0.5) / u;
        if x < 0.0 || x >= b {
            continue;
        }
        k = x.floor() as i64;
        let gp = logfactorial(k)
            + logfactorial(mingoodbad - k)
            + logfactorial(computed_sample - k)
            + logfactorial(maxgoodbad - computed_sample + k);
        let t = g - gp;
        if u * (4.0 - u) - 3.0 <= t {
            break;
        }
        if u * (u - t) >= 1.0 {
            continue;
        }
        if 2.0 * u.ln() <= t {
            break;
        }
    }

    if good > bad {
        k = computed_sample - k;
    }
    if computed_sample < sample {
        k = good - k;
    }
    k
}

/// `random_hypergeometric`. Caller must ensure `good >= 0 && bad >= 0 &&
/// 0 <= sample <= good + bad` (numpy's `CONS_NON_NEGATIVE` on all three
/// plus the Cython-level `ngood + nbad < nsample` check -- see
/// `ionp-py/src/random.rs` for the exact transcription).
pub fn hypergeometric(bg: &mut dyn BitGen64, good: i64, bad: i64, sample: i64) -> i64 {
    if sample >= 10 && sample <= good + bad - 10 {
        hypergeometric_hrua(bg, good, bad, sample)
    } else {
        hypergeometric_sample(bg, good, bad, sample)
    }
}

// ---------------------------------------------------------------------------
// Multinomial (built on `binomial`).
// ---------------------------------------------------------------------------

/// `random_multinomial`. `pix` is the probability vector (length `d`,
/// summing to <= 1.0, caller-validated); writes `d` counts into `mnix`
/// summing to `n`. Caller must ensure `n >= 0` and `pix` passes numpy's
/// own `CONS_BOUNDED_0_1`/simplex-sum checks (see
/// `ionp-py/src/random.rs`).
pub fn multinomial(bg: &mut dyn BitGen64, n: i64, pix: &[f64], mnix: &mut [i64]) {
    let d = pix.len();
    let mut remaining_p = 1.0;
    let mut dn = n;
    for j in 0..(d.saturating_sub(1)) {
        let draw = binomial(bg, pix[j] / remaining_p, dn);
        mnix[j] = draw;
        dn -= draw;
        if dn <= 0 {
            return;
        }
        remaining_p -= pix[j];
    }
    if dn > 0 {
        mnix[d - 1] = dn;
    }
}

// ---------------------------------------------------------------------------
// Multivariate hypergeometric, "marginals" method (built on `hypergeometric`).
// ---------------------------------------------------------------------------

/// `Generator.multivariate_hypergeometric(colors, nsample, size=None,
/// method='marginals')`'s DEFAULT `method='marginals'` algorithm,
/// transcribed from numpy's `random_mvhg_marginals.c`
/// `random_multivariate_hypergeometric_marginals`. `method='count'` (an
/// alternate, statistically-different-but-equally-valid algorithm using a
/// `total`-sized temp array and a distinct draw order) is NOT implemented
/// -- see `ionp-py/src/random.rs`'s doc comment on the `#[pymethods]`
/// wrapper for why only the default is in scope.
///
/// Writes one length-`colors.len()` variate into `out`
/// (`out.len() == colors.len()`, caller must zero-init -- this function
/// does not touch trailing/skipped entries when the `num_to_sample > 0`
/// loop-exit condition is hit early, exactly mirroring the C source's own
/// "variates is not initialized in the function" contract). `total` is
/// `colors.iter().sum()` (caller-computed, exactly mirroring the C
/// function's own redundant-total-as-parameter shape -- NOT recomputed
/// here).
///
/// Algorithm: draws `num_colors - 1` correlated univariate
/// `hypergeometric` samples in sequence (color `j`'s draw uses `remaining
/// = total - sum(colors[..=j])` as its `bad` parameter and `num_to_sample
/// = nsample - sum(previous draws)` as its `sample` parameter), then
/// assigns whatever's left of `num_to_sample` to the LAST color
/// unconditionally (no further draw). A "more than half" symmetry
/// optimization runs FIRST: if `nsample > total / 2`, the function draws
/// for `total - nsample` (numerically cheaper -- `hypergeometric`'s own
/// cost scales with the smaller side) and then complements every entry
/// (`out[k] = colors[k] - out[k]`) at the end -- this changes the ORDER
/// and VALUES of the underlying draws relative to the naive nsample, not
/// just a final cosmetic transform, so it must run exactly where numpy
/// runs it (before the draw loop, not folded into the loop itself).
pub fn multivariate_hypergeometric_marginals(
    bg: &mut dyn BitGen64,
    total: i64,
    colors: &[i64],
    nsample: i64,
    out: &mut [i64],
) {
    let num_colors = colors.len();
    debug_assert_eq!(out.len(), num_colors);
    if total == 0 || nsample == 0 || num_colors == 0 {
        return;
    }

    let more_than_half = nsample > total / 2;
    let mut num_to_sample = if more_than_half { total - nsample } else { nsample };
    let mut remaining = total;

    for j in 0..num_colors.saturating_sub(1) {
        if num_to_sample <= 0 {
            break;
        }
        remaining -= colors[j];
        let r = hypergeometric(bg, colors[j], remaining, num_to_sample);
        out[j] = r;
        num_to_sample -= r;
    }
    if num_to_sample > 0 {
        out[num_colors - 1] = num_to_sample;
    }
    if more_than_half {
        for k in 0..num_colors {
            out[k] = colors[k] - out[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Sequence operations: `shuffle`/`permutation`/`permuted`/`choice`.
// Transcribed from `_generator.pyx`'s `_shuffle_raw`/`_shuffle_int` free
// functions and `Generator.choice`'s replace/no-replace branches. Scoped
// to 1-D contiguous data (any dtype, generic over `T: Copy`) -- N-D
// `axis=`-navigated shuffling and object-array (`dtype=object`) handling
// are out of scope (ionp has no Python-object dtype at all, so the
// numpy source's GIL-holding "hasobject" branch is moot here).
//
// TWO DISTINCT bounded-draw primitives are used by different callers in
// numpy's own source, and mixing them up is exactly the kind of
// bit-stream-consumption bug the task's stream-position-probe mandate
// exists to catch:
//   - `shuffle`/`permuted`/`permutation` (via `shuffle`) all bottom out
//     in `_shuffle_raw`, which draws via `random_interval` (masked
//     rejection, smallest power-of-2-minus-1 mask >= max, next_uint32 if
//     max <= 0xFFFFFFFF else next_uint64) -- see `shuffle_masked` below.
//   - `choice`'s replace=False/p=None Floyd's-algorithm AND tail-shuffle
//     branches instead call `random_bounded_uint64(bitgen, 0, j, 0, 0)`
//     (`use_masked=false` -> Lemire's algorithm, NOT masked rejection) --
//     see `shuffle_lemire`/`choice_no_replace_no_p` below. This is NOT a
//     numpy inconsistency to paper over: it is numpy's actual, intentional
//     source, and reproducing it exactly (rather than "simplifying" both
//     call sites to one primitive) is required for bit-exactness.
// ---------------------------------------------------------------------------

/// `_shuffle_raw`'s Fisher-Yates loop: `for i in reversed(range(first,
/// n))`, `j = random_interval(bitgen, i)`, swap `data[i]`/`data[j]`
/// (skipped when `i == j`, matching the C source's own
/// "memcpy is undefined when i==j" guard -- a no-op either way, kept for
/// parity rather than correctness). `first = 1` (numpy's own default for
/// whole-array shuffling); with `n == 0` the loop range is empty and
/// nothing is drawn, matching numpy's own no-op-on-empty behavior.
pub fn shuffle_masked<T>(bg: &mut dyn BitGen64, data: &mut [T]) {
    let n = data.len();
    if n == 0 {
        return;
    }
    for i in (1..n).rev() {
        let j = bounded::random_interval(bg, i as u64) as usize;
        if i != j {
            data.swap(i, j);
        }
    }
}

/// `_shuffle_int`'s Fisher-Yates loop: `for i in reversed(range(first,
/// n))`, `j = random_bounded_uint64(bitgen, 0, i, 0, 0)` (Lemire, NOT
/// masked -- see module doc above), ALWAYS swaps (no `i == j` skip in the
/// C source, since the swap there is only 3 word-moves, cheap enough that
/// numpy doesn't bother branching around it -- functionally identical
/// output and identical bit-stream consumption either way). `first` is
/// exposed as a parameter because `choice`'s tail-shuffle branch calls
/// this with `first = max(pop_size - size, 1)`, not always `1`.
pub fn shuffle_lemire<T>(bg: &mut dyn BitGen64, data: &mut [T], first: usize) {
    let n = data.len();
    if first >= n {
        return;
    }
    for i in (first..n).rev() {
        let j = bounded::bounded_u64(bg, 0, i as u64) as usize;
        data.swap(i, j);
    }
}

/// Smallest bit-mask `>= max`, matching `_generator.pyx`'s `_gen_mask`
/// (identical bit-smearing trick to `bounded::random_interval`'s own
/// internal mask computation, duplicated here since `choice`'s hash-set
/// sizing needs the mask value itself, not just a bounded draw).
fn gen_mask(max: u64) -> u64 {
    let mut mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    mask
}

/// `Generator.choice(..., replace=False, p=None)`'s index-selection
/// algorithm, returning the `size`-length array of selected indices into
/// `0..pop_size` (NOT yet shuffled into final `idx.reshape(shape)` order
/// by the caller -- this function's own internal `shuffle` bool controls
/// only the FINAL in-function shuffle step, matching numpy's own
/// `shuffle=True` default).
///
/// Two sub-algorithms, matching `_generator.pyx`'s own heuristic branch
/// EXACTLY (both consume the bit stream differently, so picking the
/// "simpler" one unconditionally would silently diverge from numpy
/// whenever `pop_size > 10000` and `size` is large relative to
/// `pop_size`):
///
///   - `pop_size > 10000 && size > pop_size / cutoff` (`cutoff = 50` if
///     `shuffle` else `20`): "tail shuffle" -- build `idx = 0..pop_size`,
///     Lemire-Fisher-Yates-shuffle (`shuffle_lemire`) only its tail
///     (`first = max(pop_size - size, 1)`), return the last `size`
///     entries.
///   - otherwise: Floyd's algorithm with an open-addressing uint64 hash
///     set (capacity = smallest power of 2 >= `1.2 * size`, linear
///     probing on collision), `for j in (pop_size - size)..pop_size`:
///     draw `val = bounded_u64(bg, 0, j)` (Lemire); if `val` is already
///     in the hash set, insert `j` itself as the selected index instead
///     (the classic in-place Floyd trick); the newly-selected index for
///     slot `j` is always appended in order of `j`, THEN (only if
///     `shuffle` is true) the whole `size`-length output is
///     Lemire-Fisher-Yates-shuffled in place (`shuffle_lemire`, `first =
///     1`) as a separate final pass.
pub fn choice_no_replace_no_p(bg: &mut dyn BitGen64, pop_size: i64, size: i64, shuffle: bool) -> Vec<i64> {
    if size == 0 {
        return Vec::new();
    }
    let cutoff: i64 = if shuffle { 50 } else { 20 };
    if pop_size > 10000 && size > pop_size / cutoff {
        let mut idx: Vec<i64> = (0..pop_size).collect();
        let first = std::cmp::max(pop_size - size, 1) as usize;
        shuffle_lemire(bg, &mut idx, first);
        idx[(pop_size - size) as usize..].to_vec()
    } else {
        let mut out = vec![0i64; size as usize];
        let set_size_hint = (1.2 * size as f64) as u64;
        let mask = gen_mask(set_size_hint);
        let set_size = 1 + mask;
        let mut hash_set = vec![u64::MAX; set_size as usize];
        for j in (pop_size - size)..pop_size {
            let val = bounded::bounded_u64(bg, 0, j as u64);
            let mut loc = (val & mask) as usize;
            while hash_set[loc] != u64::MAX && hash_set[loc] != val {
                loc = ((loc as u64 + 1) & mask) as usize;
            }
            let out_idx = (j - (pop_size - size)) as usize;
            if hash_set[loc] == u64::MAX {
                hash_set[loc] = val;
                out[out_idx] = val as i64;
            } else {
                let mut loc2 = (j as u64 & mask) as usize;
                while hash_set[loc2] != u64::MAX {
                    loc2 = ((loc2 as u64 + 1) & mask) as usize;
                }
                hash_set[loc2] = j as u64;
                out[out_idx] = j;
            }
        }
        if shuffle {
            shuffle_lemire(bg, &mut out, 1);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Pcg64, SeedSequence};

    /// Every reference vector below was captured OUTSIDE this test file, by
    /// hand-running real numpy 2.5.1 (`.venv`) with `default_rng(42)` and
    /// printing `.tolist()` of each call -- same methodology as
    /// `distributions.rs`'s own test module.
    fn pcg64(seed: u32) -> Pcg64 {
        let seq = SeedSequence::new(&[seed], &[], 4);
        let v = seq.generate_state_u64(4);
        Pcg64::from_seed_words([v[0], v[1], v[2], v[3]])
    }

    // np.random.default_rng(42).poisson(5.0, 5) -> [8, 7, 7, 2, 6]
    // (lam < 10, exercises `poisson_mult`).
    #[test]
    fn poisson_small_lam_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<i64> = (0..5).map(|_| poisson(&mut bg, 5.0)).collect();
        assert_eq!(got, vec![8, 7, 7, 2, 6]);
    }

    // np.random.default_rng(42).poisson(50.0, 5) -> [56, 59, 56, 41, 47]
    // (lam >= 10, exercises `poisson_ptrs`).
    #[test]
    fn poisson_large_lam_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<i64> = (0..5).map(|_| poisson(&mut bg, 50.0)).collect();
        assert_eq!(got, vec![56, 59, 56, 41, 47]);
    }

    // np.random.default_rng(42).binomial(20, 0.3, 5) -> [8, 6, 8, 7, 3]
    #[test]
    fn binomial_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<i64> = (0..5).map(|_| binomial(&mut bg, 0.3, 20)).collect();
        assert_eq!(got, vec![8, 6, 8, 7, 3]);
    }

    // np.random.default_rng(42).negative_binomial(5.0, 0.4, 5) ->
    // [12, 4, 15, 8, 8]
    #[test]
    fn negative_binomial_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<i64> = (0..5).map(|_| negative_binomial(&mut bg, 5.0, 0.4)).collect();
        assert_eq!(got, vec![12, 4, 15, 8, 8]);
    }

    // np.random.default_rng(42).geometric(0.3, 5) -> [7, 7, 7, 1, 1]
    // (p < 1/3 -> search branch).
    #[test]
    fn geometric_search_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<i64> = (0..5).map(|_| geometric(&mut bg, 0.3)).collect();
        assert_eq!(got, vec![7, 7, 7, 1, 1]);
    }

    // np.random.default_rng(42).geometric(0.9, 5) -> [1, 1, 1, 1, 1]
    // (p >= 1/3 -> inversion branch).
    #[test]
    fn geometric_inversion_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<i64> = (0..5).map(|_| geometric(&mut bg, 0.9)).collect();
        assert_eq!(got, vec![1, 1, 1, 1, 1]);
    }

    // np.random.default_rng(42).zipf(2.0, 5) -> [4, 1, 1, 1, 1]
    #[test]
    fn zipf_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<i64> = (0..5).map(|_| zipf(&mut bg, 2.0)).collect();
        assert_eq!(got, vec![4, 1, 1, 1, 1]);
    }

    // np.random.default_rng(42).logseries(0.6, 5) -> [1, 2, 1, 5, 1]
    #[test]
    fn logseries_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<i64> = (0..5).map(|_| logseries(&mut bg, 0.6)).collect();
        assert_eq!(got, vec![1, 2, 1, 5, 1]);
    }

    // np.random.default_rng(42).hypergeometric(15, 15, 10, 5) ->
    // [5, 6, 6, 4, 3]
    #[test]
    fn hypergeometric_seed42() {
        let mut bg = pcg64(42);
        let got: Vec<i64> = (0..5).map(|_| hypergeometric(&mut bg, 15, 15, 10)).collect();
        assert_eq!(got, vec![5, 6, 6, 4, 3]);
    }

    // np.random.default_rng(42).multinomial(20, [0.2, 0.3, 0.5]) ->
    // [5, 5, 10]  (single draw, d=3 -> d-1=2 binomial trips)
    #[test]
    fn multinomial_single_draw_seed42() {
        let mut bg = pcg64(42);
        let pvals = [0.2, 0.3, 0.5];
        let mut out = vec![0i64; 3];
        multinomial(&mut bg, 20, &pvals, &mut out);
        assert_eq!(out, vec![5, 5, 10]);
    }

    // np.random.default_rng(42).multinomial(20, [0.2, 0.3, 0.5], size=7) ->
    // [[5, 5, 10], [6, 6, 8], [2, 11, 7], [5, 7, 8], [2, 6, 12], [3, 9, 8],
    //  [5, 7, 8]]  -- variable data-dependent trip count guard: this is
    // the SAME (seed, n, pvals) as the size=1 case above, re-drawn 7
    // times in a row from one continuing bit stream, so the first row
    // must reproduce exactly and the stream must stay in lockstep for
    // all 7 (the "loop trip count depends on the data" trap the ticket's
    // coordinator warned about -- this corpus exercises trip counts
    // 1 and 7 explicitly, not just one fixed shape).
    #[test]
    fn multinomial_many_draws_seed42() {
        let mut bg = pcg64(42);
        let pvals = [0.2, 0.3, 0.5];
        let mut got = Vec::new();
        for _ in 0..7 {
            let mut row = vec![0i64; 3];
            multinomial(&mut bg, 20, &pvals, &mut row);
            got.push(row);
        }
        assert_eq!(
            got,
            vec![
                vec![5, 5, 10],
                vec![6, 6, 8],
                vec![2, 11, 7],
                vec![5, 7, 8],
                vec![2, 6, 12],
                vec![3, 9, 8],
                vec![5, 7, 8],
            ]
        );
    }

    // np.random.default_rng(42).multinomial(0, [0.2, 0.3, 0.5], size=2) ->
    // [[0, 0, 0], [0, 0, 0]]  -- n=0 short-circuits `dn <= 0` on the FIRST
    // binomial draw (draw=0 always when n=0), so mnix[d-1] never gets its
    // explicit `if dn > 0` assignment either -- confirms the all-zero
    // array pre-fill assumption `multinomial`'s doc comment relies on.
    #[test]
    fn multinomial_n_zero_seed42() {
        let mut bg = pcg64(42);
        let pvals = [0.2, 0.3, 0.5];
        let mut got = Vec::new();
        for _ in 0..2 {
            let mut row = vec![0i64; 3];
            multinomial(&mut bg, 0, &pvals, &mut row);
            got.push(row);
        }
        assert_eq!(got, vec![vec![0, 0, 0], vec![0, 0, 0]]);
    }

    // np.random.default_rng(42).multivariate_hypergeometric([16, 8, 4], 6)
    // -> [6, 0, 0]  (single draw, num_colors=3 -> 2 hypergeometric trips)
    #[test]
    fn mvhg_marginals_single_draw_seed42() {
        let mut bg = pcg64(42);
        let colors = [16i64, 8, 4];
        let total: i64 = colors.iter().sum();
        let mut out = vec![0i64; 3];
        multivariate_hypergeometric_marginals(&mut bg, total, &colors, 6, &mut out);
        assert_eq!(out, vec![6, 0, 0]);
    }

    // np.random.default_rng(42).multivariate_hypergeometric([16, 8, 4], 6,
    // size=7) -> [[6,0,0],[4,2,0],[4,1,1],[4,1,1],[3,3,0],[2,2,2],[3,2,1]]
    // -- same (seed, colors, nsample) as the size=1 case above, re-drawn 7
    // times from one continuing bit stream (trip counts 1 and 7 in one
    // corpus, per the data-dependent-loop-trip-count trap).
    #[test]
    fn mvhg_marginals_many_draws_seed42() {
        let mut bg = pcg64(42);
        let colors = [16i64, 8, 4];
        let total: i64 = colors.iter().sum();
        let mut got = Vec::new();
        for _ in 0..7 {
            let mut row = vec![0i64; 3];
            multivariate_hypergeometric_marginals(&mut bg, total, &colors, 6, &mut row);
            got.push(row);
        }
        assert_eq!(
            got,
            vec![
                vec![6, 0, 0],
                vec![4, 2, 0],
                vec![4, 1, 1],
                vec![4, 1, 1],
                vec![3, 3, 0],
                vec![2, 2, 2],
                vec![3, 2, 1],
            ]
        );
    }

    // np.random.default_rng(42).multivariate_hypergeometric([16, 8, 4], 25,
    // size=5) -- nsample=25 > total/2=14, exercises the `more_than_half`
    // complement branch (draws for total-nsample=3 internally, then
    // out[k] = colors[k] - out[k]).
    #[test]
    fn mvhg_marginals_more_than_half_seed42() {
        let mut bg = pcg64(42);
        let colors = [16i64, 8, 4];
        let total: i64 = colors.iter().sum();
        let mut got = Vec::new();
        for _ in 0..5 {
            let mut row = vec![0i64; 3];
            multivariate_hypergeometric_marginals(&mut bg, total, &colors, 25, &mut row);
            got.push(row);
        }
        assert_eq!(
            got,
            vec![
                vec![13, 8, 4],
                vec![13, 8, 4],
                vec![13, 8, 4],
                vec![14, 7, 4],
                vec![13, 8, 4],
            ]
        );
    }

    // np.random.default_rng(42).multivariate_hypergeometric([16, 8, 4], 0,
    // size=3) -> all-zero rows -- `nsample == 0` early-return.
    #[test]
    fn mvhg_marginals_nsample_zero_seed42() {
        let mut bg = pcg64(42);
        let colors = [16i64, 8, 4];
        let total: i64 = colors.iter().sum();
        let mut got = Vec::new();
        for _ in 0..3 {
            let mut row = vec![0i64; 3];
            multivariate_hypergeometric_marginals(&mut bg, total, &colors, 0, &mut row);
            got.push(row);
        }
        assert_eq!(got, vec![vec![0, 0, 0]; 3]);
    }

    // np.random.default_rng(42).multivariate_hypergeometric([10], 5,
    // size=3) -> [[5],[5],[5]]  -- num_colors=1: the `for j in
    // 0..num_colors-1` draw loop never runs (0 iterations), the entire
    // nsample goes straight to `out[num_colors-1]` unconditionally.
    #[test]
    fn mvhg_marginals_single_color_seed42() {
        let mut bg = pcg64(42);
        let colors = [10i64];
        let total: i64 = colors.iter().sum();
        let mut got = Vec::new();
        for _ in 0..3 {
            let mut row = vec![0i64; 1];
            multivariate_hypergeometric_marginals(&mut bg, total, &colors, 5, &mut row);
            got.push(row);
        }
        assert_eq!(got, vec![vec![5]; 3]);
    }

    // np.random.default_rng(42).multivariate_hypergeometric([12, 20], 15,
    // size=5) -> [[5,10],[6,9],[7,8],[4,11],[7,8]]  -- num_colors=2.
    #[test]
    fn mvhg_marginals_two_color_seed42() {
        let mut bg = pcg64(42);
        let colors = [12i64, 20];
        let total: i64 = colors.iter().sum();
        let mut got = Vec::new();
        for _ in 0..5 {
            let mut row = vec![0i64; 2];
            multivariate_hypergeometric_marginals(&mut bg, total, &colors, 15, &mut row);
            got.push(row);
        }
        assert_eq!(got, vec![vec![5, 10], vec![6, 9], vec![7, 8], vec![4, 11], vec![7, 8]]);
    }

    // np.random.default_rng(42): arr = arange(10); g.shuffle(arr) ->
    // [5, 6, 0, 7, 3, 2, 4, 9, 1, 8]. `_shuffle_raw`'s masked-rejection
    // primitive (`random_interval`), NOT Lemire.
    #[test]
    fn shuffle_masked_len10_seed42() {
        let mut bg = pcg64(42);
        let mut data: Vec<i64> = (0..10).collect();
        shuffle_masked(&mut bg, &mut data);
        assert_eq!(data, vec![5, 6, 0, 7, 3, 2, 4, 9, 1, 8]);
    }

    // np.random.default_rng(42): arange(1), shuffle -> [0] (n=1: loop
    // range (1..1) is empty, zero draws, no panic).
    #[test]
    fn shuffle_masked_len1_seed42() {
        let mut bg = pcg64(42);
        let mut data: Vec<i64> = vec![0];
        shuffle_masked(&mut bg, &mut data);
        assert_eq!(data, vec![0]);
    }

    // np.random.default_rng(42): arange(0), shuffle -> [] (empty, no-op,
    // no panic on the `n == 0` early return).
    #[test]
    fn shuffle_masked_len0_seed42() {
        let mut bg = pcg64(42);
        let mut data: Vec<i64> = vec![];
        shuffle_masked(&mut bg, &mut data);
        assert_eq!(data, Vec::<i64>::new());
    }

    // np.random.default_rng(7): arange(6), shuffle -> [5, 2, 0, 4, 1, 3].
    // Different seed from the len10/len1/len0 cases above so this isn't
    // just re-testing the same PCG64 stream position.
    #[test]
    fn shuffle_masked_len6_seed7() {
        let mut bg = pcg64(7);
        let mut data: Vec<i64> = (0..6).collect();
        shuffle_masked(&mut bg, &mut data);
        assert_eq!(data, vec![5, 2, 0, 4, 1, 3]);
    }

    // np.random.default_rng(42).choice(20, size=5, replace=False,
    // shuffle=False) -> [1, 13, 11, 8, 19]. Floyd's-algorithm branch
    // (pop_size=20 is far below the 10000 tail-shuffle threshold),
    // shuffle=False so the FINAL `shuffle_lemire(out, 1)` pass is
    // skipped -- this is the raw Floyd insertion order.
    #[test]
    fn choice_no_replace_no_p_floyd_noshuffle_seed42() {
        let mut bg = pcg64(42);
        let out = choice_no_replace_no_p(&mut bg, 20, 5, false);
        assert_eq!(out, vec![1, 13, 11, 8, 19]);
    }

    // np.random.default_rng(42).choice(20, size=5, replace=False) (default
    // shuffle=True) -> [13, 8, 11, 1, 19]. Same Floyd draws as the
    // no-shuffle case above, but with the extra `shuffle_lemire(out, 1)`
    // final pass applied -- confirms that pass is wired in and consumes
    // its OWN additional bit-stream draws (a value-only comparison against
    // the noshuffle case could pass by coincidence; this checks the
    // actual numpy-captured permutation of the same 5 values).
    #[test]
    fn choice_no_replace_no_p_floyd_shuffle_seed42() {
        let mut bg = pcg64(42);
        let out = choice_no_replace_no_p(&mut bg, 20, 5, true);
        assert_eq!(out, vec![13, 8, 11, 1, 19]);
    }

    // np.random.default_rng(42).choice(20000, size=1000,
    // replace=False)[:10] -> [9742, 18275, 6853, 10003, 11897, 10066,
    // 8199, 3744, 10438, 6514]. pop_size=20000 > 10000 AND size=1000 >
    // 20000/50=400 -- the "tail shuffle" branch, NOT Floyd's algorithm.
    // If the branch condition were wrong (e.g. off-by-one on the cutoff),
    // this would silently fall through to Floyd's algorithm instead and
    // produce different values from a different draw sequence.
    #[test]
    fn choice_no_replace_no_p_tail_shuffle_seed42() {
        let mut bg = pcg64(42);
        let out = choice_no_replace_no_p(&mut bg, 20000, 1000, true);
        assert_eq!(&out[..10], &[9742, 18275, 6853, 10003, 11897, 10066, 8199, 3744, 10438, 6514]);
        assert_eq!(out.len(), 1000);
    }
}
