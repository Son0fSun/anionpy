//! `numpy_ref` — the baseline-competitiveness gate's numpy side.
//!
//! Context: `reports/ion-gate-baseline-audit-2026-08-01.md` found that
//! `bench`'s speedup numbers were computed against our own Rust dense
//! baseline (`baseline::dense_matvec`/`dense_matmul`), and that baseline is
//! itself ~2x slower than numpy's BLAS on the identical op/dtype/machine.
//! A speedup vs a baseline that loses to BLAS is not a speedup vs the thing
//! anyone would actually compare to. This module measures numpy's BLAS on
//! the SAME op, SAME dtype, SAME machine, at bench time, so `bench.rs` can
//! refuse to print a headline number built on an unfair baseline.
//!
//! ## Why shell-out-and-parse, not a committed timings JSON
//! Two ways to get this number were on the table: (a) shell out to the
//! canonical venv python at bench run time and parse its stdout, or
//! (b) commit a numpy-timings JSON next to `reference_gate.json`, produced
//! by a small script, and have `bench` read the file.
//!
//! (a) was chosen. The whole point of this gate is a *same-machine,
//! same-run* comparison — that's what the audit measured and what makes
//! the 2.07x finding meaningful. A committed file is a snapshot: run the
//! gate on different hardware (or after a numpy/BLAS upgrade) and a stale
//! committed number would silently certify a baseline that is no longer
//! competitive, which is exactly the failure mode this whole change exists
//! to close. Shelling out re-measures numpy every single gate run, so
//! "competitive" always means "competitive right now, on this box" — and we
//! still print+record which interpreter and numpy version produced the
//! number (see [`NumpyMeasurement`]), so every printed ratio is traceable
//! without being a frozen, driftable artifact.
//!
//! Uses ONLY the project's canonical interpreter — see [`CANONICAL_PYTHON`].
//! System `python3` (numpy 2.4.2 on this machine, per the task spec) is
//! explicitly NOT the reference. If the canonical interpreter is missing or
//! misbehaves, this returns `Err` and `bench.rs` treats that case's
//! competitiveness as UNKNOWN — which suppresses the headline exactly like
//! a proven-inflated baseline would (an unmeasurable baseline is not a
//! fair one, it's an unverified one).

use std::process::Command;

/// The canonical numpy reference interpreter for this machine, per the task
/// spec: "/Users/rabite/Monday/ionp/.venv/bin/python3, numpy 2.5.1. System
/// python3 is 2.4.2 and is NOT the reference." Transcribed from that spec,
/// not from memory of any benchmark result.
pub const CANONICAL_PYTHON: &str = "/Users/rabite/Monday/ionp/.venv/bin/python3";

#[derive(Debug, Clone)]
pub struct NumpyMeasurement {
    /// Best-of-N wall-clock seconds for the measured op.
    pub seconds: f64,
    pub interpreter: String,
    pub numpy_version: String,
}

#[derive(Debug)]
pub enum NumpyRefError {
    Spawn(String),
    BadOutput(String),
}

impl std::fmt::Display for NumpyRefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumpyRefError::Spawn(e) => write!(f, "could not run canonical numpy reference ({CANONICAL_PYTHON}): {e}"),
            NumpyRefError::BadOutput(e) => write!(f, "canonical numpy reference produced unparseable output: {e}"),
        }
    }
}

fn run_script(script: &str) -> Result<NumpyMeasurement, NumpyRefError> {
    let out = Command::new(CANONICAL_PYTHON)
        .arg("-c")
        .arg(script)
        .output()
        .map_err(|e| NumpyRefError::Spawn(e.to_string()))?;
    if !out.status.success() {
        return Err(NumpyRefError::Spawn(format!(
            "exit status {:?}, stderr: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines();
    let seconds: f64 = lines
        .next()
        .ok_or_else(|| NumpyRefError::BadOutput("no seconds line".into()))?
        .trim()
        .parse()
        .map_err(|e| NumpyRefError::BadOutput(format!("seconds parse: {e}")))?;
    let interpreter = lines.next().ok_or_else(|| NumpyRefError::BadOutput("no interpreter line".into()))?.trim().to_string();
    let numpy_version = lines.next().ok_or_else(|| NumpyRefError::BadOutput("no numpy version line".into()))?.trim().to_string();
    Ok(NumpyMeasurement { seconds, interpreter, numpy_version })
}

/// Time `A @ x`: real float64 dense matvec, n x n by n. This is numpy's
/// BLAS `dgemv` path — the honest real-dtype comparator for our
/// `baseline::dense_matvec`.
pub fn time_matvec_f64(n: usize, repeats: usize) -> Result<NumpyMeasurement, NumpyRefError> {
    let script = format!(
        "import numpy as np, sys, time\n\
         n = {n}\n\
         rng = np.random.default_rng(12345)\n\
         a = rng.standard_normal((n, n))\n\
         x = rng.standard_normal(n)\n\
         _ = a @ x\n\
         best = float('inf')\n\
         for _ in range({repeats}):\n\
         \tt0 = time.perf_counter()\n\
         \ty = a @ x\n\
         \tdt = time.perf_counter() - t0\n\
         \tif dt < best: best = dt\n\
         print(best)\n\
         print(sys.executable)\n\
         print(np.__version__)\n"
    );
    run_script(&script)
}

/// Time `(A @ B) @ x`: real float64, n x n throughout. This is numpy's
/// BLAS `dgemm` (forming the product) then `dgemv` — the honest real-dtype
/// comparator for `baseline::dense_matmul` + `dense_matvec` chained exactly
/// as `bench.rs`'s showcase case chains them.
pub fn time_matmul_then_matvec_f64(n: usize, repeats: usize) -> Result<NumpyMeasurement, NumpyRefError> {
    let script = format!(
        "import numpy as np, sys, time\n\
         n = {n}\n\
         rng = np.random.default_rng(12345)\n\
         a = rng.standard_normal((n, n))\n\
         b = rng.standard_normal((n, n))\n\
         x = rng.standard_normal(n)\n\
         _p = a @ b\n\
         _ = _p @ x\n\
         best = float('inf')\n\
         for _ in range({repeats}):\n\
         \tt0 = time.perf_counter()\n\
         \tp = a @ b\n\
         \ty = p @ x\n\
         \tdt = time.perf_counter() - t0\n\
         \tif dt < best: best = dt\n\
         print(best)\n\
         print(sys.executable)\n\
         print(np.__version__)\n"
    );
    run_script(&script)
}
