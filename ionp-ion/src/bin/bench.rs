//! bench — reproduces `bench/reference_gate.json`'s cases against the Rust
//! Ion layer, in the same two phases the Python reference's
//! `bench/matmul_vs_numpy.py` uses, in the same order, always:
//!
//!   1. CORRECTNESS GATE. Every operator's `apply`/`apply_mat` must match a
//!      dense reference to < 1e-9. This is enforced *structurally*: the
//!      timing functions below only accept `ionp_ion::Certified`, which only
//!      `gate::certify()` can construct, which only succeeds below tol. There
//!      is no code path from "operator exists" to "speed number printed"
//!      that skips this.
//!   2. TIMING, only for what passed phase 1: wall-clock the Ion action
//!      against a real, multi-core dense baseline (`ionp_ion::baseline`,
//!      NOT a naive single-threaded triple loop), across the exact cases
//!      `reference_gate.json` measured (n=4096 matvecs, n=2048 showcase
//!      product), printing an honest table plus machine-readable JSON.
//!
//! The dense baseline's own achieved GFLOP/s is sanity-checked before any
//! speedup number is trusted — see `assert_plausible_gflops`. A prior
//! benchmark in this project reported a 123x "win" that was entirely an
//! artifact of a 0.58 GFLOP/s baseline; this binary refuses to repeat that.
//!
//! ## Two target kinds (2026-08-01)
//! Per `reports/ion-gate-target-semantics-2026-08-01.md`: every case in
//! `reference_gate.json` now carries a `floor` (the original reference
//! implementation's ratio, measured on its own machine — a hard lower bound;
//! falling below it means our Rust port lost to the Python it reimplements,
//! which is a real regression) and a `reproduction` (a `[0.5x, 2.0x]` band
//! recorded ON THIS MACHINE FROM THIS IMPLEMENTATION — the only band it is
//! valid to compare a same-machine measurement against). A case passes only
//! if it clears the floor AND sits inside its reproduction band. The old
//! scheme band-checked a Rust/M4/Accelerate measurement against a Python
//! reference-machine ratio, which is a category error, not a correctness
//! check — see the report for the full argument.
//!
//! Run: `cargo run -p ionp-ion --release --bin bench`
//! Refresh the reproduction bands deliberately (never as a side effect of a
//! normal run): `cargo run -p ionp-ion --release --bin bench -- --record-reproduction`

/// Minimal, self-contained JSON reader/splicer for `bench/reference_gate.json`.
/// No `serde_json` (or any new crate) is available to this file — adding a
/// dependency means touching `Cargo.toml`/`Cargo.lock`, which are off limits
/// while other agents have concurrent crate work in flight in this tree (see
/// the module doc above). This is small on purpose: a real recursive-descent
/// parser for reading (`parse`/`Json::get`/`as_f64`/`as_bool`), plus a
/// string-and-brace-aware splice (`replace_object_value`) for writing back
/// just the one `reproduction` object `--record-reproduction` touches,
/// without reformatting or reparsing the rest of the file.
mod json_min {
    #[derive(Debug, Clone)]
    pub enum Json {
        Null,
        Bool(bool),
        Num(f64),
        #[allow(dead_code)]
        Str(String),
        #[allow(dead_code)]
        Arr(Vec<Json>),
        Obj(Vec<(String, Json)>),
    }

    impl Json {
        pub fn get(&self, key: &str) -> Option<&Json> {
            match self {
                Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
        pub fn as_f64(&self) -> Option<f64> {
            match self {
                Json::Num(n) => Some(*n),
                _ => None,
            }
        }
        pub fn as_bool(&self) -> Option<bool> {
            match self {
                Json::Bool(b) => Some(*b),
                _ => None,
            }
        }
    }

    pub fn parse(text: &str) -> Result<Json, String> {
        let chars: Vec<char> = text.chars().collect();
        let mut pos = 0usize;
        let v = parse_value(&chars, &mut pos)?;
        Ok(v)
    }

    fn skip_ws(c: &[char], pos: &mut usize) {
        while *pos < c.len() && c[*pos].is_whitespace() {
            *pos += 1;
        }
    }

    fn parse_value(c: &[char], pos: &mut usize) -> Result<Json, String> {
        skip_ws(c, pos);
        if *pos >= c.len() {
            return Err("unexpected end of input".into());
        }
        match c[*pos] {
            '{' => parse_obj(c, pos),
            '[' => parse_arr(c, pos),
            '"' => Ok(Json::Str(parse_str(c, pos)?)),
            't' => {
                expect_lit(c, pos, "true")?;
                Ok(Json::Bool(true))
            }
            'f' => {
                expect_lit(c, pos, "false")?;
                Ok(Json::Bool(false))
            }
            'n' => {
                expect_lit(c, pos, "null")?;
                Ok(Json::Null)
            }
            _ => parse_num(c, pos),
        }
    }

    fn expect_lit(c: &[char], pos: &mut usize, lit: &str) -> Result<(), String> {
        for ch in lit.chars() {
            if *pos >= c.len() || c[*pos] != ch {
                return Err(format!("expected literal {lit}"));
            }
            *pos += 1;
        }
        Ok(())
    }

    fn parse_obj(c: &[char], pos: &mut usize) -> Result<Json, String> {
        *pos += 1; // {
        let mut out = Vec::new();
        skip_ws(c, pos);
        if *pos < c.len() && c[*pos] == '}' {
            *pos += 1;
            return Ok(Json::Obj(out));
        }
        loop {
            skip_ws(c, pos);
            let key = parse_str(c, pos)?;
            skip_ws(c, pos);
            if *pos >= c.len() || c[*pos] != ':' {
                return Err("expected ':'".into());
            }
            *pos += 1;
            let val = parse_value(c, pos)?;
            out.push((key, val));
            skip_ws(c, pos);
            if *pos >= c.len() {
                return Err("unexpected end in object".into());
            }
            match c[*pos] {
                ',' => {
                    *pos += 1;
                }
                '}' => {
                    *pos += 1;
                    break;
                }
                other => return Err(format!("expected ',' or '}}', got '{other}'")),
            }
        }
        Ok(Json::Obj(out))
    }

    fn parse_arr(c: &[char], pos: &mut usize) -> Result<Json, String> {
        *pos += 1; // [
        let mut out = Vec::new();
        skip_ws(c, pos);
        if *pos < c.len() && c[*pos] == ']' {
            *pos += 1;
            return Ok(Json::Arr(out));
        }
        loop {
            let val = parse_value(c, pos)?;
            out.push(val);
            skip_ws(c, pos);
            if *pos >= c.len() {
                return Err("unexpected end in array".into());
            }
            match c[*pos] {
                ',' => {
                    *pos += 1;
                }
                ']' => {
                    *pos += 1;
                    break;
                }
                other => return Err(format!("expected ',' or ']', got '{other}'")),
            }
        }
        Ok(Json::Arr(out))
    }

    fn parse_str(c: &[char], pos: &mut usize) -> Result<String, String> {
        if *pos >= c.len() || c[*pos] != '"' {
            return Err("expected string".into());
        }
        *pos += 1;
        let mut s = String::new();
        loop {
            if *pos >= c.len() {
                return Err("unterminated string".into());
            }
            let ch = c[*pos];
            if ch == '"' {
                *pos += 1;
                break;
            }
            if ch == '\\' {
                *pos += 1;
                if *pos >= c.len() {
                    return Err("unterminated escape".into());
                }
                match c[*pos] {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    'b' => s.push('\u{8}'),
                    'f' => s.push('\u{c}'),
                    'u' => {
                        if *pos + 4 >= c.len() {
                            return Err("truncated \\u escape".into());
                        }
                        let hex: String = c[*pos + 1..*pos + 5].iter().collect();
                        let cp = u32::from_str_radix(&hex, 16).map_err(|e| e.to_string())?;
                        if let Some(uc) = char::from_u32(cp) {
                            s.push(uc);
                        }
                        *pos += 4;
                    }
                    other => return Err(format!("bad escape \\{other}")),
                }
                *pos += 1;
            } else {
                s.push(ch);
                *pos += 1;
            }
        }
        Ok(s)
    }

    fn parse_num(c: &[char], pos: &mut usize) -> Result<Json, String> {
        let start = *pos;
        if *pos < c.len() && (c[*pos] == '-' || c[*pos] == '+') {
            *pos += 1;
        }
        while *pos < c.len() && (c[*pos].is_ascii_digit() || c[*pos] == '.' || c[*pos] == 'e' || c[*pos] == 'E' || c[*pos] == '+' || c[*pos] == '-') {
            *pos += 1;
        }
        let s: String = c[start..*pos].iter().collect();
        s.parse::<f64>().map(Json::Num).map_err(|e| format!("bad number '{s}': {e}"))
    }

    /// Locate `"case_name": { ... "key": { ... } ... }` in raw JSON text and
    /// replace the VALUE of `key` (must itself be a `{...}` object) with
    /// `replacement`. Used only by `--record-reproduction` to splice a
    /// fresh `reproduction` block into `reference_gate.json` without
    /// disturbing anything else in the file (its `_comment`/`_note` keys,
    /// formatting, or unrelated sections). String-and-brace-aware text
    /// surgery, not a full reparse-and-reserialize, on purpose: reprinting
    /// the whole file would reformat scientific notation and hand-authored
    /// layout elsewhere in it for no reason.
    pub fn replace_object_value(text: &str, case_name: &str, key: &str, replacement: &str) -> Result<String, String> {
        let case_needle = format!("\"{case_name}\"");
        let case_idx = text.find(&case_needle).ok_or_else(|| format!("case '{case_name}' not found"))?;
        let key_needle = format!("\"{key}\"");
        let key_idx_rel = text[case_idx..].find(&key_needle).ok_or_else(|| format!("key '{key}' not found after case '{case_name}'"))?;
        let key_idx = case_idx + key_idx_rel;
        let after_key = &text[key_idx..];
        let colon_rel = after_key.find(':').ok_or("expected ':' after key")?;
        let mut i = key_idx + colon_rel + 1;
        let bytes = text.as_bytes();
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'{' {
            return Err(format!("value of '{key}' is not an object"));
        }
        let open = i;
        let close = find_matching_brace(text, open)?;
        let mut out = String::with_capacity(text.len() + replacement.len());
        out.push_str(&text[..open]);
        out.push_str(replacement);
        out.push_str(&text[close + 1..]);
        Ok(out)
    }

    fn find_matching_brace(text: &str, open_idx: usize) -> Result<usize, String> {
        let bytes = text.as_bytes();
        let mut depth = 0i32;
        let mut in_str = false;
        let mut escape = false;
        let mut i = open_idx;
        while i < bytes.len() {
            let ch = bytes[i] as char;
            if in_str {
                if escape {
                    escape = false;
                } else if ch == '\\' {
                    escape = true;
                } else if ch == '"' {
                    in_str = false;
                }
            } else {
                match ch {
                    '"' => in_str = true,
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(i);
                        }
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        Err("unbalanced braces".into())
    }
}

use ionp_ion::baseline::{dense_matmul, dense_matvec, matmul_gflops};
use ionp_ion::numpy_ref;
use ionp_ion::{certify, matmul, Certified, Circulant, Dense, Diagonal, LowRank, Operator, SplitMix64, Toeplitz};
use json_min::Json;
use num_complex::Complex64;
use std::process::Command;
use std::time::Instant;

const TOL: f64 = 1e-9;

// ── band + competitiveness policy constants ─────────────────────────────
// These are GATE POLICY thresholds (decisions about what counts as a pass),
// not benchmark measurements — same category as TOL above. Transcribed
// from bench/reference_gate.json's "speedup_gate" / "baseline_competitiveness_gate"
// blocks and from reports/ion-gate-target-semantics-2026-08-01.md, cross-checked
// against both files in this repo.
//
// REPRODUCTION BAND: a measured speedup outside [recorded*BAND_LO, recorded*BAND_HI]
// is DISCREPANT and fails the gate, whether it undershoots OR overshoots. These
// multipliers are also written into reference_gate.json's "band" blocks at
// record time so the file is self-describing; the constants here are the
// same numbers used to build that band, not an independent second opinion.
const BAND_LO_MULT: f64 = 0.5;
const BAND_HI_MULT: f64 = 2.0;
// COMPETITIVENESS: if our Rust dense baseline is more than this multiple
// slower than numpy's BLAS on the identical op/dtype/machine, the baseline
// is INFLATED and no headline speedup derived from it may print — and
// `--record-reproduction` refuses to bake that case's number into the gate.
const INFLATED_BASELINE_MULT: f64 = 1.5;
// best-of-N repeats for the numpy-side timing, matching bench_best's use
// on the Rust side so both numbers are "best observed", not "first observed".
const NUMPY_REPEATS: usize = 5;

const GATE_JSON_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/reference_gate.json");
const CASE_NAMES: [&str; 4] =
    ["circulant_matvec_n4096", "toeplitz_matvec_n4096", "lowrank_r16_matvec_n4096", "circulant_product_n2048"];

// ── gate targets, loaded from bench/reference_gate.json at run time ────────
// No target numbers are hardcoded here (that duplication is exactly how a
// stale/hand-typed number could drift from the file this whole scheme exists
// to keep honest). `floor` is the reference implementation's number, retained verbatim from the file
// (transcribed once, by the agent who wrote it, with provenance attached —
// not re-typed here). `reproduction` is populated only by
// `bench --record-reproduction`, which writes measured numbers straight from
// this binary's own timing back into the file; nothing about it is ever
// hand-typed either.

/// One case's `floor` block, parsed straight out of `reference_gate.json`.
struct FloorTarget {
    speedup: f64,
}

/// One case's `reproduction` block. `None` fields mean "not recorded yet" —
/// `bench --record-reproduction` has never run since this schema landed, and
/// the gate correctly refuses to band-check against nothing.
struct ReproductionTarget {
    recorded: bool,
    band_lo: Option<f64>,
    band_hi: Option<f64>,
}

struct CaseGate {
    floor: FloorTarget,
    reproduction: ReproductionTarget,
}

fn load_gate_json() -> Json {
    let text = std::fs::read_to_string(GATE_JSON_PATH)
        .unwrap_or_else(|e| panic!("could not read gate file {GATE_JSON_PATH}: {e}"));
    json_min::parse(&text).unwrap_or_else(|e| panic!("could not parse {GATE_JSON_PATH} as JSON: {e}"))
}

fn case_gate(gate_json: &Json, name: &str) -> CaseGate {
    let case = gate_json
        .get("speedup_gate")
        .and_then(|g| g.get(name))
        .unwrap_or_else(|| panic!("speedup_gate.{name} missing from {GATE_JSON_PATH}"));
    let floor = case.get("floor").unwrap_or_else(|| panic!("speedup_gate.{name}.floor missing"));
    let floor_speedup = floor.get("speedup").and_then(Json::as_f64).unwrap_or_else(|| panic!("speedup_gate.{name}.floor.speedup missing/not a number"));

    let repro = case.get("reproduction").unwrap_or_else(|| panic!("speedup_gate.{name}.reproduction missing"));
    let recorded = repro.get("recorded").and_then(Json::as_bool).unwrap_or(false);
    let band = repro.get("band");
    let band_lo = band.and_then(|b| b.get("lo")).and_then(Json::as_f64);
    let band_hi = band.and_then(|b| b.get("hi")).and_then(Json::as_f64);

    CaseGate {
        floor: FloorTarget { speedup: floor_speedup },
        reproduction: ReproductionTarget { recorded: recorded && band_lo.is_some() && band_hi.is_some(), band_lo, band_hi },
    }
}

/// Whether the measured speedup clears the floor. This is the ONE side of
/// the old band that survives band semantics: a lower bound is always a
/// valid check regardless of what machine produced it — a Rust port being
/// slower than the Python original it reimplements is a real regression on
/// any hardware.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FloorStatus {
    Cleared,
    Below,
}

impl FloorStatus {
    fn classify(speedup: f64, floor: f64) -> Self {
        if speedup >= floor {
            FloorStatus::Cleared
        } else {
            FloorStatus::Below
        }
    }
    fn passes(self) -> bool {
        self == FloorStatus::Cleared
    }
    fn label(self) -> &'static str {
        match self {
            FloorStatus::Cleared => "CLEARS FLOOR",
            FloorStatus::Below => "BELOW FLOOR",
        }
    }
}

/// Outcome of comparing measured speedup to the `[recorded*0.5, recorded*2.0]`
/// reproduction band recorded on THIS machine from THIS implementation.
/// `Meets` is the only passing case; both directions of `Discrepant` fail the
/// gate, same as `Unrecorded` (nothing to band-check against yet).
#[derive(Debug, Clone, Copy, PartialEq)]
enum ReproductionStatus {
    Meets,
    DiscrepantLow,
    DiscrepantHigh,
    Unrecorded,
}

impl ReproductionStatus {
    fn classify(speedup: f64, target: &ReproductionTarget) -> Self {
        match (target.recorded, target.band_lo, target.band_hi) {
            (true, Some(lo), Some(hi)) => {
                if speedup < lo {
                    ReproductionStatus::DiscrepantLow
                } else if speedup > hi {
                    ReproductionStatus::DiscrepantHigh
                } else {
                    ReproductionStatus::Meets
                }
            }
            _ => ReproductionStatus::Unrecorded,
        }
    }
    fn passes(self) -> bool {
        self == ReproductionStatus::Meets
    }
    fn label(self) -> &'static str {
        match self {
            ReproductionStatus::Meets => "MEETS",
            ReproductionStatus::DiscrepantLow => "DISCREPANT (undershoot)",
            ReproductionStatus::DiscrepantHigh => "DISCREPANT (overshoot)",
            ReproductionStatus::Unrecorded => "UNRECORDED (run --record-reproduction first)",
        }
    }
}

/// Outcome of comparing our Rust dense baseline's wall time to numpy's BLAS
/// on the identical op/dtype/n, measured on this machine, this run.
enum Competitiveness {
    Fair { ratio: f64, numpy: numpy_ref::NumpyMeasurement },
    Inflated { ratio: f64, numpy: numpy_ref::NumpyMeasurement },
    Unmeasurable(String),
}

impl Competitiveness {
    fn check(rust_seconds: f64, numpy_result: Result<numpy_ref::NumpyMeasurement, numpy_ref::NumpyRefError>) -> Self {
        match numpy_result {
            Err(e) => Competitiveness::Unmeasurable(e.to_string()),
            Ok(numpy) => {
                let ratio = rust_seconds / numpy.seconds;
                if ratio > INFLATED_BASELINE_MULT {
                    Competitiveness::Inflated { ratio, numpy }
                } else {
                    Competitiveness::Fair { ratio, numpy }
                }
            }
        }
    }

    /// A baseline whose competitiveness could not be established is treated
    /// the same as a proven-inflated one: unverified is not the same thing
    /// as fair, and this gate must not get easier by failing open.
    fn ok(&self) -> bool {
        matches!(self, Competitiveness::Fair { .. })
    }
}

fn to_c64(x: &[f64]) -> Vec<Complex64> {
    x.iter().map(|&v| Complex64::new(v, 0.0)).collect()
}

fn rand_vec(rng: &mut SplitMix64, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.next_gaussian()).collect()
}

/// row-major f64 (n x n or n x m) -> column-major Complex64 Dense.
fn dense_from_rowmajor(a: &[f64], rows: usize, cols: usize) -> Dense {
    let mut data = vec![Complex64::new(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            data[j * rows + i] = Complex64::new(a[i * cols + j], 0.0);
        }
    }
    Dense::new(rows, cols, data, true)
}

fn circulant_dense_rowmajor(c: &[f64], n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = (i as isize - j as isize).rem_euclid(n as isize) as usize;
            m[i * n + j] = c[idx];
        }
    }
    m
}

fn toeplitz_dense_rowmajor(col: &[f64], row: &[f64], n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = if i >= j { col[i - j] } else { row[j - i] };
        }
    }
    m
}

fn lowrank_dense_rowmajor(u: &[f64], v: &[f64], n: usize, r: usize) -> Vec<f64> {
    // U (n x r) @ V^T (r x n), both row-major
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for t in 0..r {
                acc += u[i * r + t] * v[j * r + t];
            }
            m[i * n + j] = acc;
        }
    }
    m
}

fn bench_best(mut f: impl FnMut(), repeats: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    best
}

fn fmt_time(seconds: f64) -> String {
    if seconds < 1e-3 {
        format!("{:8.1} us", seconds * 1e6)
    } else if seconds < 1.0 {
        format!("{:8.2} ms", seconds * 1e3)
    } else {
        format!("{:8.3} s ", seconds)
    }
}

struct JsonBuf(String);
impl JsonBuf {
    fn new() -> Self {
        Self(String::from("{\n"))
    }
    fn kv_num(&mut self, key: &str, val: f64) {
        self.0.push_str(&format!("  \"{key}\": {val},\n"));
    }
    fn kv_bool(&mut self, key: &str, val: bool) {
        self.0.push_str(&format!("  \"{key}\": {val},\n"));
    }
    fn kv_str(&mut self, key: &str, val: &str) {
        let escaped = val.replace('\\', "\\\\").replace('"', "\\\"");
        self.0.push_str(&format!("  \"{key}\": \"{escaped}\",\n"));
    }
    fn finish(mut self) -> String {
        // trim trailing comma+newline, close brace
        if self.0.ends_with(",\n") {
            self.0.truncate(self.0.len() - 2);
            self.0.push('\n');
        }
        self.0.push('}');
        self.0
    }
}

/// Measured numbers for one case this run, kept around only long enough to
/// either (a) get compared against the gate, in normal mode, or (b) get
/// written back into `reference_gate.json`'s `reproduction` block, in
/// `--record-reproduction` mode. Never hand-typed: every field here is
/// either a direct timer read or derived arithmetic on one.
struct Measured {
    name: &'static str,
    speedup: f64,
    numpy_us: f64,
    ion_us: f64,
    numpy_interpreter: String,
    numpy_version: String,
}

/// Host machine identity for provenance, via `hostname` — same "shell out,
/// don't hand-type" discipline as `numpy_ref.rs` uses for interpreter/numpy
/// version. Never cached across runs; always re-asked.
fn host_string() -> String {
    Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
}

/// Today's date via `date -u`, for provenance stamps — not computed from
/// `SystemTime` by hand (no calendar-date crate in this workspace's
/// dependency set, and this file is not permitted to add one).
fn today_utc() -> String {
    Command::new("date")
        .args(["-u", "+%Y-%m-%d"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-date".to_string())
}

const BLAS_BACKEND: &str = "Apple Accelerate (blas-src, accelerate feature; cblas_dgemm/cblas_dgemv)";
const RUST_IMPLEMENTATION: &str = "Rust ionp-ion (this crate, this commit)";

fn main() {
    let record_mode = std::env::args().skip(1).any(|a| a == "--record-reproduction");

    println!("{}", "=".repeat(78));
    println!("PHASE 1 -- CORRECTNESS GATE  (Ion apply() must match dense to < {TOL:.0e})");
    println!("{}", "=".repeat(78));

    let mut rng = SplitMix64::new(20260731);
    let n0 = 512;
    let x0 = to_c64(&rand_vec(&mut rng, n0));
    let xm0 = to_c64(&rand_vec(&mut rng, n0 * 4));

    let mut gate_ok = true;

    // circulant
    let c0 = rand_vec(&mut rng, n0);
    let circ0 = Circulant::from_real(c0.clone());
    let circ0_dense = dense_from_rowmajor(&circulant_dense_rowmajor(&c0, n0), n0, n0);
    let cert_circ = certify("circulant", &circ0, &circ0_dense, &x0, &xm0, 4, TOL);
    report_gate("circulant", &cert_circ);
    gate_ok &= cert_circ.is_ok();

    // circulant @ circulant
    let c2 = rand_vec(&mut rng, n0);
    let circ2 = Circulant::from_real(c2.clone());
    let prod: Box<dyn Operator> = matmul(Box::new(Circulant::from_real(c0.clone())), Box::new(Circulant::from_real(c2.clone())));
    let d1 = circulant_dense_rowmajor(&c0, n0);
    let d2 = circulant_dense_rowmajor(&c2, n0);
    let dprod = dense_matmul(&d1, n0, n0, &d2, n0);
    let dprod_op = dense_from_rowmajor(&dprod, n0, n0);
    let cert_prod = certify("circulant@circulant", prod.as_ref(), &dprod_op, &x0, &xm0, 4, TOL);
    report_gate("circulant @ circulant", &cert_prod);
    gate_ok &= cert_prod.is_ok();
    let _ = circ2;

    // toeplitz
    let col0 = rand_vec(&mut rng, n0);
    let mut row0 = rand_vec(&mut rng, n0);
    row0[0] = col0[0];
    let toep0 = Toeplitz::from_real(col0.clone(), Some(row0.clone()));
    let toep0_dense = dense_from_rowmajor(&toeplitz_dense_rowmajor(&col0, &row0, n0), n0, n0);
    let cert_toep = certify("toeplitz", &toep0, &toep0_dense, &x0, &xm0, 4, TOL);
    report_gate("toeplitz", &cert_toep);
    gate_ok &= cert_toep.is_ok();

    // lowrank r=16
    let r = 16;
    let u0 = rand_vec(&mut rng, n0 * r);
    let v0 = rand_vec(&mut rng, n0 * r);
    let lr0 = LowRank::from_real(to_f64_flat_colmajor(&u0, n0, r), to_f64_flat_colmajor(&v0, n0, r), n0, n0, r);
    let lr0_dense = dense_from_rowmajor(&lowrank_dense_rowmajor(&u0, &v0, n0, r), n0, n0);
    let cert_lr = certify("lowrank_r16", &lr0, &lr0_dense, &x0, &xm0, 4, TOL);
    report_gate("lowrank (r=16)", &cert_lr);
    gate_ok &= cert_lr.is_ok();

    // diagonal
    let d0 = rand_vec(&mut rng, n0);
    let diag0 = Diagonal::from_real(d0.clone());
    let mut diag_dense_row = vec![0.0f64; n0 * n0];
    for i in 0..n0 {
        diag_dense_row[i * n0 + i] = d0[i];
    }
    let diag0_dense = dense_from_rowmajor(&diag_dense_row, n0, n0);
    let cert_diag = certify("diagonal", &diag0, &diag0_dense, &x0, &xm0, 4, TOL);
    report_gate("diagonal", &cert_diag);
    gate_ok &= cert_diag.is_ok();

    println!(
        "\n  GATE: {}",
        if gate_ok { "ALL PASS -- proceeding to timing" } else { "FAILED -- aborting, no speed numbers will be printed" }
    );
    if !gate_ok {
        std::process::exit(1);
    }

    // ── phase 2: timing ─────────────────────────────────────────────────
    println!("\n{}", "=".repeat(78));
    println!("PHASE 2 -- TIMING  (Ion vs a real multi-core dense baseline)");
    println!("{}", "=".repeat(78));
    if record_mode {
        println!("\n  *** --record-reproduction: measuring for the record ***");
        println!("  Every case must clear its floor AND run on a competitive baseline, or");
        println!("  nothing gets written and reference_gate.json is left untouched.");
    }

    let gate_json = load_gate_json();

    let mut json = JsonBuf::new();
    let mut all_gate_pass = true;
    let mut record_ok = true; // AND of floor+competitiveness across all 4 cases, record mode only
    let mut measured: Vec<Measured> = Vec::new();

    // -- dense baseline plausibility sanity check (matmul, n=1024) --
    {
        let n = 1024;
        let mut rng2 = SplitMix64::new(1);
        let a = rand_vec(&mut rng2, n * n);
        let b = rand_vec(&mut rng2, n * n);
        let _ = dense_matmul(&a, n, n, &b, n); // warm
        let t = bench_best(|| { let _ = dense_matmul(&a, n, n, &b, n); }, 3);
        let gflops = matmul_gflops(n, n, n, t);
        println!("\n  dense baseline sanity: {n}x{n}x{n} matmul achieves {gflops:.2} GFLOP/s ({} for {t:.3}s)", fmt_time(t));
        json.kv_num("baseline_matmul_gflops_n1024", gflops);
        if gflops < 2.0 {
            println!("  WARNING: {gflops:.2} GFLOP/s is implausibly slow for Apple M4 -- any speedup number");
            println!("           measured against this baseline should be treated as SUSPECT, not trusted.");
            json.kv_bool("baseline_plausible", false);
        } else {
            println!("  plausible for Apple M4 (dispatches to Accelerate cblas_dgemm -- BLAS-parity by construction).");
            json.kv_bool("baseline_plausible", true);
        }
    }

    // -- circulant matvec, n=4096 --
    {
        let n = 4096;
        let mut rng2 = SplitMix64::new(2);
        let c = rand_vec(&mut rng2, n);
        let x = rand_vec(&mut rng2, n);
        let circ = Circulant::from_real(c.clone());
        let dense_row = circulant_dense_rowmajor(&c, n);
        let dense_op = dense_from_rowmajor(&dense_row, n, n);
        let cx = to_c64(&x);
        let xm = to_c64(&x);
        let cert = certify("circulant_matvec_n4096", &circ, &dense_op, &cx, &xm, 1, TOL).unwrap_or_else(|e| panic!("{e}"));

        let _ = dense_matvec(&dense_row, n, n, &x); // warm
        let _ = time_certified(&cert, &cx); // warm
        let t_np = bench_best(|| { let _ = dense_matvec(&dense_row, n, n, &x); }, 5);
        let t_ion = bench_best(|| { let _ = time_certified(&cert, &cx); }, 5);
        report_timing(CASE_NAMES[0], t_np, t_ion, n, &gate_json, &mut json, &mut all_gate_pass, record_mode, &mut record_ok, &mut measured);
    }

    // -- toeplitz matvec, n=4096 --
    {
        let n = 4096;
        let mut rng2 = SplitMix64::new(3);
        let col = rand_vec(&mut rng2, n);
        let mut row = rand_vec(&mut rng2, n);
        row[0] = col[0];
        let x = rand_vec(&mut rng2, n);
        let toep = Toeplitz::from_real(col.clone(), Some(row.clone()));
        let dense_row = toeplitz_dense_rowmajor(&col, &row, n);
        let dense_op = dense_from_rowmajor(&dense_row, n, n);
        let cx = to_c64(&x);
        let cert = certify("toeplitz_matvec_n4096", &toep, &dense_op, &cx, &cx, 1, TOL).unwrap_or_else(|e| panic!("{e}"));

        let _ = dense_matvec(&dense_row, n, n, &x);
        let _ = time_certified(&cert, &cx);
        let t_np = bench_best(|| { let _ = dense_matvec(&dense_row, n, n, &x); }, 5);
        let t_ion = bench_best(|| { let _ = time_certified(&cert, &cx); }, 5);
        report_timing(CASE_NAMES[1], t_np, t_ion, n, &gate_json, &mut json, &mut all_gate_pass, record_mode, &mut record_ok, &mut measured);
    }

    // -- lowrank r=16 matvec, n=4096 --
    {
        let n = 4096;
        let r = 16;
        let mut rng2 = SplitMix64::new(4);
        let u = rand_vec(&mut rng2, n * r);
        let v = rand_vec(&mut rng2, n * r);
        let x = rand_vec(&mut rng2, n);
        let lr = LowRank::from_real(to_f64_flat_colmajor(&u, n, r), to_f64_flat_colmajor(&v, n, r), n, n, r);
        let dense_row = lowrank_dense_rowmajor(&u, &v, n, r);
        let dense_op = dense_from_rowmajor(&dense_row, n, n);
        let cx = to_c64(&x);
        let cert = certify("lowrank_r16_matvec_n4096", &lr, &dense_op, &cx, &cx, 1, TOL).unwrap_or_else(|e| panic!("{e}"));

        let _ = dense_matvec(&dense_row, n, n, &x);
        let _ = time_certified(&cert, &cx);
        let t_np = bench_best(|| { let _ = dense_matvec(&dense_row, n, n, &x); }, 5);
        let t_ion = bench_best(|| { let _ = time_certified(&cert, &cx); }, 5);
        report_timing(CASE_NAMES[2], t_np, t_ion, n, &gate_json, &mut json, &mut all_gate_pass, record_mode, &mut record_ok, &mut measured);
    }

    // -- showcase: circulant @ circulant, n=2048 --
    {
        let n = 2048;
        let mut rng2 = SplitMix64::new(5);
        let c1 = rand_vec(&mut rng2, n);
        let c2 = rand_vec(&mut rng2, n);
        let x = rand_vec(&mut rng2, n);

        let prod: Box<dyn Operator> = matmul(Box::new(Circulant::from_real(c1.clone())), Box::new(Circulant::from_real(c2.clone())));
        let d1 = circulant_dense_rowmajor(&c1, n);
        let d2 = circulant_dense_rowmajor(&c2, n);

        let cx = to_c64(&x);
        // certify against the TRUE dense product (never formed by Ion's path)
        let dprod = dense_matmul(&d1, n, n, &d2, n);
        let dprod_op = dense_from_rowmajor(&dprod, n, n);
        let cert = certify("circulant_product_n2048", prod.as_ref(), &dprod_op, &cx, &cx, 1, TOL).unwrap_or_else(|e| panic!("{e}"));

        // numpy-equivalent path: form D1@D2 (O(n^3)) THEN matvec -- timed as
        // one unit, matching the reference bench's `(D1 @ D2) @ x`.
        let dense_showcase = || {
            let p = dense_matmul(&d1, n, n, &d2, n);
            let _ = dense_matvec(&p, n, n, &x);
        };
        dense_showcase(); // warm
        let t_np = bench_best(dense_showcase, 3);

        let _ = time_certified(&cert, &cx); // warm
        let t_ion = bench_best(|| { let _ = time_certified(&cert, &cx); }, 5);

        println!("\n  circulant_product_n2048 (showcase: dense forms A@B at O(n^3); Ion multiplies");
        println!("  spectra elementwise + one ifft at O(n log n), dense product NEVER formed):");

        let numpy_result = numpy_ref::time_matmul_then_matvec_f64(n, NUMPY_REPEATS.min(3));
        report_case_body(
            CASE_NAMES[3],
            t_np,
            t_ion,
            numpy_result,
            &gate_json,
            &mut json,
            &mut all_gate_pass,
            record_mode,
            &mut record_ok,
            &mut measured,
        );
    }

    println!("\n{}", "=".repeat(78));
    if record_mode {
        finish_record_mode(record_ok, measured);
    } else {
        println!(
            "OVERALL: {}",
            if all_gate_pass {
                "all measured speedups clear their floor and sit inside their reproduction band"
            } else {
                "one or more cases BELOW FLOOR, DISCREPANT, UNRECORDED, or INFLATED_BASELINE -- see above, reported honestly"
            }
        );
        println!("{}", "=".repeat(78));

        json.kv_bool("all_targets_met", all_gate_pass);
        println!("\n--- JSON ---\n{}", json.finish());

        if !all_gate_pass {
            std::process::exit(1);
        }
    }
}

/// After all 4 cases have been measured in `--record-reproduction` mode:
/// write the new reproduction bands back to `reference_gate.json`, but only
/// if every single case cleared its floor on a competitive baseline. This is
/// the guard against a gate silently re-baselining itself to a regression —
/// see the module doc and `reports/ion-gate-target-semantics-2026-08-01.md`.
fn finish_record_mode(record_ok: bool, measured: Vec<Measured>) {
    if !record_ok || measured.len() != CASE_NAMES.len() {
        println!("RECORDING ABORTED: at least one case failed its floor or its baseline was not");
        println!("competitive this run (see above). reference_gate.json was NOT modified --");
        println!("a gate must never silently re-baseline itself to a regression.");
        println!("{}", "=".repeat(78));
        std::process::exit(1);
    }

    let host = host_string();
    let date = today_utc();
    println!("RECORDING reproduction bands to {GATE_JSON_PATH}");
    println!("  host: {host}   blas: {BLAS_BACKEND}   date: {date}");

    let mut text = std::fs::read_to_string(GATE_JSON_PATH).unwrap_or_else(|e| panic!("re-reading {GATE_JSON_PATH}: {e}"));
    for m in &measured {
        let band_lo = m.speedup * BAND_LO_MULT;
        let band_hi = m.speedup * BAND_HI_MULT;
        let replacement = format!(
            "{{
      \"recorded\": true,
      \"speedup\": {speedup},
      \"numpy_us\": {numpy_us},
      \"ion_us\": {ion_us},
      \"band\": {{
        \"lo\": {band_lo},
        \"hi\": {band_hi},
        \"lo_mult\": {BAND_LO_MULT},
        \"hi_mult\": {BAND_HI_MULT}
      }},
      \"provenance\": {{
        \"implementation\": \"{RUST_IMPLEMENTATION}\",
        \"host\": \"{host}\",
        \"blas\": \"{BLAS_BACKEND}\",
        \"numpy_version\": \"{numpy_interpreter} v{numpy_version}\",
        \"date\": \"{date}\"
      }}
    }}",
            speedup = m.speedup,
            numpy_us = m.numpy_us,
            ion_us = m.ion_us,
            numpy_interpreter = m.numpy_interpreter,
            numpy_version = m.numpy_version,
        );
        text = json_min::replace_object_value(&text, m.name, "reproduction", &replacement)
            .unwrap_or_else(|e| panic!("could not splice reproduction block for {}: {e}", m.name));
        println!("  {:28} recorded speedup={:.1}x -> band [{:.1}x, {:.1}x]", m.name, m.speedup, band_lo, band_hi);
    }
    std::fs::write(GATE_JSON_PATH, &text).unwrap_or_else(|e| panic!("writing {GATE_JSON_PATH}: {e}"));

    // Verify what we just wrote is still valid JSON with the values we intended.
    let reparsed = json_min::parse(&text).unwrap_or_else(|e| panic!("BUG: wrote invalid JSON to {GATE_JSON_PATH}: {e}"));
    for m in &measured {
        let got = case_gate(&reparsed, m.name);
        assert!(got.reproduction.recorded, "BUG: {} not marked recorded after write", m.name);
    }
    println!("\nreference_gate.json updated and re-parsed clean. Run bench again without");
    println!("--record-reproduction to certify against the bands just recorded.");
    println!("{}", "=".repeat(78));
}

fn report_gate(name: &str, cert: &Result<Certified, ionp_ion::GateFailure>) {
    match cert {
        Ok(c) => println!("  {name:28} matvec={:.2e}  matmat={:.2e}  PASS", c.matvec_err, c.matmat_err),
        Err(e) => println!("  {name:28} {e}"),
    }
}

fn time_certified(cert: &Certified, x: &[Complex64]) -> Vec<Complex64> {
    cert.op().apply(x)
}

/// `n` is the problem size, used to shell out to the canonical numpy
/// reference for the baseline-competitiveness check (same op, same dtype,
/// same n, same machine, this run — see `numpy_ref`).
#[allow(clippy::too_many_arguments)]
fn report_timing(
    name: &'static str,
    t_dense: f64,
    t_ion: f64,
    n: usize,
    gate_json: &Json,
    json: &mut JsonBuf,
    all_gate_pass: &mut bool,
    record_mode: bool,
    record_ok: &mut bool,
    measured: &mut Vec<Measured>,
) {
    println!("\n  {name}:");
    let numpy_result = numpy_ref::time_matvec_f64(n, NUMPY_REPEATS);
    report_case_body(name, t_dense, t_ion, numpy_result, gate_json, json, all_gate_pass, record_mode, record_ok, measured);
}

/// Shared gating/printing/JSON-emission body for both the three matvec
/// cases and the n=2048 showcase product. Splitting this out means the
/// floor+reproduction+competitiveness logic exists in exactly one place.
#[allow(clippy::too_many_arguments)]
fn report_case_body(
    name: &'static str,
    t_dense: f64,
    t_ion: f64,
    numpy_result: Result<numpy_ref::NumpyMeasurement, numpy_ref::NumpyRefError>,
    gate_json: &Json,
    json: &mut JsonBuf,
    all_gate_pass: &mut bool,
    record_mode: bool,
    record_ok: &mut bool,
    measured: &mut Vec<Measured>,
) {
    let speedup = t_dense / t_ion;
    let gate = case_gate(gate_json, name);
    let floor_status = FloorStatus::classify(speedup, gate.floor.speedup);
    let repro_status = ReproductionStatus::classify(speedup, &gate.reproduction);

    println!("    dense baseline: {}", fmt_time(t_dense));
    println!("    ion:            {}", fmt_time(t_ion));
    println!("    speedup: {speedup:.1}x   (floor: {:.1}x -- {})", gate.floor.speedup, floor_status.label());
    match (gate.reproduction.band_lo, gate.reproduction.band_hi) {
        (Some(lo), Some(hi)) => println!("    reproduction band: [{lo:.1}x, {hi:.1}x] -- {}", repro_status.label()),
        _ => println!("    reproduction band: {}", repro_status.label()),
    }

    let competitiveness = Competitiveness::check(t_dense, numpy_result);
    print_competitiveness(&competitiveness);

    if record_mode {
        // In record mode we don't gate on the (possibly not-yet-recorded)
        // reproduction band -- that's what we're here to set. We DO still
        // require the floor and baseline competitiveness, both of which are
        // independent of reproduction and gate whether we're willing to
        // write anything at all (see finish_record_mode).
        let this_ok = floor_status.passes() && competitiveness.ok();
        *record_ok &= this_ok;
        println!(
            "    {}",
            if this_ok {
                "eligible to record (floor cleared, baseline competitive)"
            } else {
                "NOT eligible to record -- see floor/competitiveness above"
            }
        );
        if let Competitiveness::Fair { numpy, .. } = &competitiveness {
            measured.push(Measured {
                name,
                speedup,
                numpy_us: t_dense * 1e6,
                ion_us: t_ion * 1e6,
                numpy_interpreter: numpy.interpreter.clone(),
                numpy_version: numpy.numpy_version.clone(),
            });
        }
    } else {
        let pass = floor_status.passes() && repro_status.passes() && competitiveness.ok();
        println!(
            "    {}",
            if !competitiveness.ok() {
                "INFLATED_BASELINE -- headline speedup suppressed, not reported as a win"
            } else if pass {
                "MEETS gate (floor cleared + inside reproduction band)"
            } else {
                "FAILS gate -- reporting honestly, not fabricating"
            }
        );
        *all_gate_pass &= pass;
    }

    json.kv_num(&format!("{name}_dense_us"), t_dense * 1e6);
    json.kv_num(&format!("{name}_ion_us"), t_ion * 1e6);
    if competitiveness.ok() {
        json.kv_num(&format!("{name}_speedup"), speedup);
    } else {
        json.kv_str(&format!("{name}_speedup"), "SUPPRESSED_INFLATED_BASELINE");
    }
    json.kv_num(&format!("{name}_floor_speedup"), gate.floor.speedup);
    json.kv_str(&format!("{name}_floor_status"), floor_status.label());
    json.kv_str(&format!("{name}_reproduction_status"), repro_status.label());
    if let (Some(lo), Some(hi)) = (gate.reproduction.band_lo, gate.reproduction.band_hi) {
        json.kv_num(&format!("{name}_reproduction_band_lo"), lo);
        json.kv_num(&format!("{name}_reproduction_band_hi"), hi);
    }
    write_competitiveness_json(json, name, &competitiveness);
    if !record_mode {
        let pass = floor_status.passes() && repro_status.passes() && competitiveness.ok();
        json.kv_bool(&format!("{name}_meets_target"), pass);
    }
}

fn print_competitiveness(c: &Competitiveness) {
    match c {
        Competitiveness::Fair { ratio, numpy } => {
            println!(
                "    baseline competitiveness: rust/numpy = {ratio:.2}x  (numpy: {} v{})  -- FAIR (<= {INFLATED_BASELINE_MULT:.1}x)",
                numpy.interpreter, numpy.numpy_version
            );
        }
        Competitiveness::Inflated { ratio, numpy } => {
            println!(
                "    baseline competitiveness: rust/numpy = {ratio:.2}x  (numpy: {} v{})  -- INFLATED_BASELINE (> {INFLATED_BASELINE_MULT:.1}x)",
                numpy.interpreter, numpy.numpy_version
            );
        }
        Competitiveness::Unmeasurable(e) => {
            println!("    baseline competitiveness: UNMEASURABLE ({e}) -- treated as not-fair, gate fails");
        }
    }
}

fn write_competitiveness_json(json: &mut JsonBuf, name: &str, c: &Competitiveness) {
    match c {
        Competitiveness::Fair { ratio, numpy } => {
            json.kv_num(&format!("{name}_baseline_vs_numpy_ratio"), *ratio);
            json.kv_str(&format!("{name}_baseline_status"), "fair");
            json.kv_str(&format!("{name}_numpy_interpreter"), &numpy.interpreter);
            json.kv_str(&format!("{name}_numpy_version"), &numpy.numpy_version);
        }
        Competitiveness::Inflated { ratio, numpy } => {
            json.kv_num(&format!("{name}_baseline_vs_numpy_ratio"), *ratio);
            json.kv_str(&format!("{name}_baseline_status"), "inflated");
            json.kv_str(&format!("{name}_numpy_interpreter"), &numpy.interpreter);
            json.kv_str(&format!("{name}_numpy_version"), &numpy.numpy_version);
        }
        Competitiveness::Unmeasurable(e) => {
            json.kv_str(&format!("{name}_baseline_status"), "unmeasurable");
            json.kv_str(&format!("{name}_baseline_error"), e);
        }
    }
}

/// U/V for LowRank are stored column-major (n x r); rand_vec produces a flat
/// row-major-ish stream, so re-lay it out column-major before handing to
/// LowRank::from_real (which expects the same column-major convention as
/// every other Operator's to_dense()).
fn to_f64_flat_colmajor(row_major_n_by_r: &[f64], n: usize, r: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * r];
    for i in 0..n {
        for t in 0..r {
            out[t * n + i] = row_major_n_by_r[i * r + t];
        }
    }
    out
}
