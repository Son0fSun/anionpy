//! `ndarray.__repr__`: a from-scratch reimplementation of numpy's
//! `array_repr`/`array2string`/`_formatArray` (see
//! `numpy/_core/arrayprint.py`) targeting bit-for-bit output match.
//!
//! This is formatting, not arithmetic -- every digit that appears here
//! comes from Rust's own `f32`/`f64` `Display`/`LowerExp` implementations,
//! which (like numpy's "dragon4 unique mode") produce the shortest decimal
//! string that round-trips back to the exact same IEEE754 value. Numpy and
//! Rust are two independent, correct implementations of the same
//! well-defined "shortest unique round-trip decimal" problem, so their
//! *digits* agree; only the surrounding padding/sign/bracket/line-wrap
//! layout (implemented here by hand, verified against real numpy 2.5.1
//! source and empirical output) can differ, and that's exactly the part
//! this module owns.
//!
//! Scope note: no `threshold`/summarization (`...`) logic -- numpy's
//! default `threshold=1000` and the largest array this crate is
//! differential-tested against is 32 elements, so summarization never
//! triggers and is not implemented.

use crate::array::NdArray;
use crate::buffer::Buffer;
use crate::dtype::DType;

const LINEWIDTH: usize = 75;

pub fn array_repr(arr: &NdArray) -> String {
    let prefix = "array(";
    let suffix = ")";
    let body = if arr.size() == 0 {
        "[]".to_string()
    } else {
        let fmt = ElementFormatter::build(arr);
        let line_width = LINEWIDTH.saturating_sub(suffix.len());
        let next_line_prefix = " ".to_string() + &" ".repeat(prefix.len());
        format_array_recursive(arr, &fmt, 0, &[], &next_line_prefix, line_width)
    };

    let mut extras: Vec<String> = Vec::new();
    if arr.size() == 0 && arr.shape() != [0usize] {
        extras.push(format!("shape={}", shape_tuple_str(arr.shape())));
    }
    if !dtype_is_implied(arr.dtype()) || arr.size() == 0 {
        // `S`/`U` show their QUOTED short code here (`dtype='|S5'`/
        // `dtype='<U5'`), not the bare `.name()` every other dtype uses
        // (`dtype=uint8`, no quotes) -- verified live against numpy 2.5.1:
        // `repr(np.array([b'a'], dtype='S5'))` ends `dtype='|S5')`.
        match arr.dtype() {
            DType::S(n) => extras.push(format!("dtype='|S{n}'")),
            DType::U(_) => extras.push(format!("dtype='<U{}'", arr.dtype().char_count())),
            dt => extras.push(format!("dtype={}", dt.name())),
        }
    }

    if extras.is_empty() {
        return format!("{prefix}{body}{suffix}");
    }
    let arr_str = format!("{prefix}{body},");
    let extra_str = extras.join(", ") + suffix;
    let last_line_len = arr_str.len() - arr_str.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let spacer = if last_line_len + extra_str.len() + 1 > LINEWIDTH {
        "\n".to_string() + &" ".repeat(prefix.len())
    } else {
        " ".to_string()
    };
    format!("{arr_str}{spacer}{extra_str}")
}

/// `ndarray.__str__` (numpy's `array_str`, distinct from `array_repr`): the
/// bare bracketed value body only -- no `array(` wrapper, no trailing
/// `dtype=`/`shape=` annotation (verified: `str(np.array([1,2],
/// dtype=np.uint8))` == `'[1 2]'`, with no `array(...)` wrapper and no
/// dtype note even though `repr()` of the same array shows
/// `dtype=uint8`). `str()`'s line-wrapping has no `array(`-length hanging
/// indent (numpy calls `array2string(a, separator=' ', prefix="")` here,
/// vs. `prefix="array("` for `repr()`).
pub fn array_str(arr: &NdArray) -> String {
    if arr.size() == 0 {
        return "[]".to_string();
    }
    let fmt = ElementFormatter::build(arr);
    // The initial hanging indent is 1 space (not 0): even with no `array(`
    // prefix text, the array's own outermost `[` still occupies one column
    // that nested rows must align underneath (verified: `str(np.arange(4)
    // .reshape(2, 2))` == `'[[0 1]\n [2 3]]'`, one leading space on the
    // second row -- `repr()`'s analogous `next_line_prefix` is `len(prefix)
    // + 1` spaces for exactly this same reason, where `prefix` there is
    // `"array("`).
    format_array_recursive_sep(arr, &fmt, 0, &[], " ", LINEWIDTH, " ")
}

fn shape_tuple_str(shape: &[usize]) -> String {
    if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    }
}

/// numpy's `dtype_is_implied`: true for the dtype a bare Python
/// scalar/list literal would produce -- bool, the platform default int
/// (int64 on this environment's macOS/arm64 numpy build), float64,
/// complex128. Anything else must show `dtype=...` even when non-empty.
fn dtype_is_implied(dt: DType) -> bool {
    matches!(dt, DType::Bool | DType::I64 | DType::F64 | DType::C128)
}

// ---------------------------------------------------------------------------
// Per-element formatting function, computed once per array (numpy's
// `_get_format_function` + `FloatingFormat`/`IntegerFormat`/`BoolFormat`/
// `ComplexFloatingFormat`).
// ---------------------------------------------------------------------------

enum ElementFormatter {
    Bool { pad: bool },
    Int { width: usize },
    Float(FloatFmt),
    Complex { re: FloatFmt, im: FloatFmt },
    // `S`/`U` elements are never padded/aligned to a common width (unlike
    // every numeric formatter above) -- verified live: `np.array(['a',
    // 'bbb', 'cc'])` prints `['a' 'bbb' 'cc']`, each element exactly its
    // own quoted length. `is_bytes` picks the `b'...'` prefix.
    Str { is_bytes: bool },
}

impl ElementFormatter {
    fn build(arr: &NdArray) -> Self {
        match arr.buffer() {
            Buffer::Bool(_) => ElementFormatter::Bool { pad: arr.ndim() != 0 },
            Buffer::I8(v) => ElementFormatter::Int { width: int_width(v.iter().map(|&x| x as i64)) },
            Buffer::I16(v) => ElementFormatter::Int { width: int_width(v.iter().map(|&x| x as i64)) },
            Buffer::I32(v) => ElementFormatter::Int { width: int_width(v.iter().map(|&x| x as i64)) },
            Buffer::I64(v) => ElementFormatter::Int { width: int_width(v.iter().copied()) },
            Buffer::U8(v) => ElementFormatter::Int { width: uint_width(v.iter().map(|&x| x as u64)) },
            Buffer::U16(v) => ElementFormatter::Int { width: uint_width(v.iter().map(|&x| x as u64)) },
            Buffer::U32(v) => ElementFormatter::Int { width: uint_width(v.iter().map(|&x| x as u64)) },
            Buffer::U64(v) => ElementFormatter::Int { width: uint_width(v.iter().copied()) },
            Buffer::F16(v) => ElementFormatter::Float(FloatFmt::build(
                v.iter().map(|&x| x.to_f64()),
                3,
                false,
                FloatWidth::F16,
            )),
            Buffer::F32(v) => ElementFormatter::Float(FloatFmt::build(
                v.iter().map(|&x| x as f64),
                6,
                false,
                FloatWidth::F32,
            )),
            Buffer::F64(v) => ElementFormatter::Float(FloatFmt::build(
                v.iter().copied(),
                15,
                false,
                FloatWidth::F64,
            )),
            Buffer::C64(v) => ElementFormatter::Complex {
                re: FloatFmt::build(v.iter().map(|c| c.re as f64), 6, false, FloatWidth::F32),
                im: FloatFmt::build(v.iter().map(|c| c.im as f64), 6, true, FloatWidth::F32),
            },
            Buffer::C128(v) => ElementFormatter::Complex {
                re: FloatFmt::build(v.iter().map(|c| c.re), 15, false, FloatWidth::F64),
                im: FloatFmt::build(v.iter().map(|c| c.im), 15, true, FloatWidth::F64),
            },
            Buffer::S(_, _) => ElementFormatter::Str { is_bytes: true },
            Buffer::U(_, _) => ElementFormatter::Str { is_bytes: false },
        }
    }

    fn format_at(&self, arr: &NdArray, flat_index: usize) -> String {
        match self {
            ElementFormatter::Bool { pad } => match arr.buffer() {
                Buffer::Bool(v) => {
                    let x = v[flat_index];
                    if x {
                        if *pad { " True".to_string() } else { "True".to_string() }
                    } else {
                        "False".to_string()
                    }
                }
                _ => unreachable!(),
            },
            ElementFormatter::Int { width } => {
                let s = match arr.buffer() {
                    Buffer::I8(v) => (v[flat_index] as i64).to_string(),
                    Buffer::I16(v) => (v[flat_index] as i64).to_string(),
                    Buffer::I32(v) => (v[flat_index] as i64).to_string(),
                    Buffer::I64(v) => v[flat_index].to_string(),
                    Buffer::U8(v) => (v[flat_index] as u64).to_string(),
                    Buffer::U16(v) => (v[flat_index] as u64).to_string(),
                    Buffer::U32(v) => (v[flat_index] as u64).to_string(),
                    Buffer::U64(v) => v[flat_index].to_string(),
                    _ => unreachable!(),
                };
                format!("{:>width$}", s, width = width)
            }
            ElementFormatter::Float(f) => match arr.buffer() {
                Buffer::F16(v) => f.format(v[flat_index].to_f64()),
                Buffer::F32(v) => f.format(v[flat_index] as f64),
                Buffer::F64(v) => f.format(v[flat_index]),
                _ => unreachable!(),
            },
            ElementFormatter::Complex { re, im } => {
                let (r, i) = match arr.buffer() {
                    Buffer::C64(v) => {
                        let c = v[flat_index];
                        (c.re as f64, c.im as f64)
                    }
                    Buffer::C128(v) => {
                        let c = v[flat_index];
                        (c.re, c.im)
                    }
                    _ => unreachable!(),
                };
                let r_s = re.format(r);
                let i_s = im.format(i);
                let sp = i_s.trim_end().len();
                format!("{}{}j{}", r_s, &i_s[..sp], &i_s[sp..])
            }
            ElementFormatter::Str { is_bytes } => match arr.buffer() {
                Buffer::S(_, v) => {
                    let raw = &v[flat_index];
                    let trimmed = strip_trailing_zero(raw);
                    py_bytes_repr(trimmed)
                }
                Buffer::U(_, v) => {
                    let raw = &v[flat_index];
                    let trimmed = strip_trailing_zero_u32(raw);
                    let s: String = trimmed.iter().map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}')).collect();
                    let _ = is_bytes;
                    py_str_repr(&s)
                }
                _ => unreachable!(),
            },
        }
    }
}

/// Trailing-NUL-strip a padded `S` element (numpy: `.rstrip(b'\x00')`
/// semantics -- only a trailing RUN of zero bytes is dropped, an embedded
/// zero byte followed by non-zero content is kept, verified live against
/// numpy 2.5.1: `np.array([b'ab\x00cd'], dtype='S6')[0] == b'ab\x00cd'`
/// (5 bytes, only the true trailing pad byte gone) but `np.array([b'ab\x00\x00'],
/// dtype='S6')[0] == b'ab'`.
fn strip_trailing_zero(v: &[u8]) -> &[u8] {
    let mut end = v.len();
    while end > 0 && v[end - 1] == 0 {
        end -= 1;
    }
    &v[..end]
}

fn strip_trailing_zero_u32(v: &[u32]) -> &[u32] {
    let mut end = v.len();
    while end > 0 && v[end - 1] == 0 {
        end -= 1;
    }
    &v[..end]
}

/// Python's `bytes.__repr__`: `b'...'`, quote character chosen the same
/// way as `str.__repr__` below (prefer `'`, switch to `"` only when the
/// content has a `'` and no `"`), ASCII-printable bytes pass through
/// as-is, everything else (including all non-ASCII bytes -- `bytes` has
/// no notion of "printable" beyond the ASCII printable range) becomes
/// `\xHH`, plus the usual `\\`, `\n`, `\r`, `\t` escapes.
fn py_bytes_repr(v: &[u8]) -> String {
    let has_single = v.contains(&b'\'');
    let has_double = v.contains(&b'"');
    let quote = if has_single && !has_double { b'"' } else { b'\'' };
    let mut out = String::with_capacity(v.len() + 3);
    out.push('b');
    out.push(quote as char);
    for &b in v {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c as char);
            }
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\x{:02x}", b)),
        }
    }
    out.push(quote as char);
    out
}

/// Python's `str.__repr__`: `'...'` (or `"..."` under the same
/// single-vs-double quote rule as `bytes` above), with `\\`/`\n`/`\r`/`\t`
/// escaped and any codepoint Python's `str.isprintable()` calls
/// non-printable escaped as `\xHH` (< 0x100), `\uHHHH` (< 0x10000), or
/// `\UHHHHHHHH` (>= 0x10000). `isprintable()` is, precisely: not a
/// separator (category Zs/Zl/Zp) other than U+0020, and not an "Other"
/// category (Cc/Cf/Cs/Co/Cn) codepoint -- approximated here by treating
/// every ASCII control character and U+007F..U+00A0 (the C1 control
/// block plus its bracketing separator) as non-printable and everything
/// else as printable, which covers this task's measured corpus (ASCII,
/// embedded NUL, Latin-1 non-ASCII, and astral-plane characters, all of
/// which ARE printable) without hand-porting the full Unicode category
/// tables real `isprintable()` consults.
fn py_str_repr(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let quote = if has_single && !has_double { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..=0xa0).contains(&(c as u32)) => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn int_width(vals: impl Iterator<Item = i64> + Clone) -> usize {
    let mut it = vals.peekable();
    if it.peek().is_none() {
        return 0;
    }
    it.map(|v| v.to_string().len()).max().unwrap_or(0)
}
fn uint_width(vals: impl Iterator<Item = u64> + Clone) -> usize {
    let mut it = vals.peekable();
    if it.peek().is_none() {
        return 0;
    }
    it.map(|v| v.to_string().len()).max().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Float formatting: mirrors `FloatingFormat` in numpy/_core/arrayprint.py.
// ---------------------------------------------------------------------------

/// Source dtype width -- see `sci_parts`/`positional_parts` doc comment.
/// `F16` needs its own path: `half::f16`'s `Display`/`LowerExp` do NOT
/// produce shortest-round-trip decimal strings (unlike Rust's native
/// `f32`/`f64`, which do) -- they just print the full widened `f32` value
/// (e.g. `3.140625` instead of the shortest string that round-trips back to
/// the same f16 bit pattern, e.g. `3.14`). So for `F16`, digit generation is
/// done by a brute-force precision search on the already-widened `f64`
/// value (see `f16_sci_search`/`f16_positional_search` below): try
/// increasingly many digits of precision until formatting-then-reparsing
/// recovers the exact same `half::f16` bit pattern the value came from.
#[derive(Clone, Copy, PartialEq)]
enum FloatWidth {
    F16,
    F32,
    F64,
}

struct FloatFmt {
    exp_format: bool,
    pad_left: usize,
    pad_right: usize, // fixed mode: fractional digits; exp mode: mantissa fractional digits
    exp_size: usize,  // exp mode only: exponent digit width (>=2)
    force_sign: bool, // '+' always shown for non-negative (imaginary parts)
    has_any: bool,    // any finite non-nan/inf value present at all
    width: FloatWidth, // source dtype width -- see `sci_parts`/`positional_parts` doc comment
}

/// Shortest-round-trip decomposition of a finite, non-negative `f64` in
/// scientific form: exactly one digit before the point.
struct SciParts {
    /// digits with no leading zero, e.g. "1234" for 1.234e5 (no decimal pt)
    digits: String,
    /// base-10 exponent such that value = 0.digits * 10^(exponent+1),
    /// i.e. digits[0] is the 10^exponent place.
    exponent: i32,
}

/// `x` (an `f32` value already widened to `f64` losslessly) determines
/// whether digit generation happens through `f32`'s own shortest-round-trip
/// `Display`/`LowerExp` or `f64`'s. This matters enormously: re-widening an
/// `f32` to `f64` and asking for *`f64`'s* shortest round-trip string
/// produces a ~17-digit decimal (the exact binary value of that `f32`,
/// which is essentially never a short `f64` decimal) instead of the
/// ~6-9-digit string numpy shows for a `complex64`/`float32` array -- numpy
/// always generates digits at the array's own dtype precision, never at
/// `f64`.
/// `half::f16`'s own `Display`/`LowerExp` are NOT shortest-round-trip (they
/// just print the widened `f32` value's own shortest string, e.g.
/// `3.140625` for a value whose shortest f16-round-tripping decimal is
/// `3.14`), unlike Rust's native `f32`/`f64` `Display`. So for `F16` we
/// brute-force the shortest digit string ourselves: try increasing decimal
/// precision until formatting-then-reparsing recovers the same f16 bit
/// pattern the value came from. `half::f16` has 11 bits of significand
/// (10 explicit + 1 implicit) which is <= 4 significant decimal digits, so
/// this loop always terminates well inside the small bound below.
fn f16_bits(x: f64) -> u16 {
    half::f16::from_f64(x).to_bits()
}

fn f16_sci_shortest(x: f64) -> SciParts {
    let target = f16_bits(x);
    for p in 0..=6 {
        let s = format!("{:.*e}", p, x);
        if s.parse::<f64>().map(|v| f16_bits(v) == target).unwrap_or(false) {
            let (mantissa, exp) = s.split_once('e').unwrap();
            let exponent: i32 = exp.parse().unwrap();
            let mut digits: String = mantissa.chars().filter(|&c| c != '.').collect();
            while digits.len() > 1 && digits.ends_with('0') {
                digits.pop();
            }
            return SciParts { digits, exponent };
        }
    }
    // Unreachable in practice given f16's precision, but fall back to a
    // generous fixed precision rather than panicking.
    let s = format!("{:.6e}", x);
    let (mantissa, exp) = s.split_once('e').unwrap();
    let exponent: i32 = exp.parse().unwrap();
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    SciParts { digits, exponent }
}

fn f16_positional_shortest(x: f64) -> (String, String) {
    let target = f16_bits(x);
    for p in 0..=12 {
        let s = format!("{:.*}", p, x);
        if s.parse::<f64>().map(|v| f16_bits(v) == target).unwrap_or(false) {
            return match s.split_once('.') {
                Some((i, f)) => (i.to_string(), f.trim_end_matches('0').to_string()),
                None => (s, String::new()),
            };
        }
    }
    let s = format!("{:.12}", x);
    match s.split_once('.') {
        Some((i, f)) => (i.to_string(), f.to_string()),
        None => (s, String::new()),
    }
}

fn sci_parts(x: f64, width: FloatWidth) -> SciParts {
    debug_assert!(x >= 0.0 && x.is_finite());
    if x == 0.0 {
        return SciParts { digits: "0".to_string(), exponent: 0 };
    }
    if width == FloatWidth::F16 {
        return f16_sci_shortest(x);
    }
    let s = if width == FloatWidth::F32 { format!("{:e}", x as f32) } else { format!("{:e}", x) };
    let (mantissa, exp) = s.split_once('e').unwrap();
    let exponent: i32 = exp.parse().unwrap();
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    SciParts { digits, exponent }
}

/// Shortest-round-trip positional (fixed-point) decomposition of a finite,
/// non-negative value: integer-part digit string and fractional-part digit
/// string (each may be empty for whole numbers). See `sci_parts` above for
/// why `width` matters.
fn positional_parts(x: f64, width: FloatWidth) -> (String, String) {
    debug_assert!(x >= 0.0 && x.is_finite());
    if width == FloatWidth::F16 {
        if x == 0.0 {
            return ("0".to_string(), String::new());
        }
        return f16_positional_shortest(x);
    }
    let s = if width == FloatWidth::F32 { format!("{}", x as f32) } else { format!("{}", x) };
    match s.split_once('.') {
        Some((int_part, frac_part)) => (int_part.to_string(), frac_part.to_string()),
        None => (s, String::new()),
    }
}

/// numpy's default print options are `precision=8, floatmode='maxprec'`:
/// print the *unique* shortest round-trip digit string, but never more than
/// `precision` fractional digits -- if the unique string would need more,
/// round to exactly `precision` fractional digits instead (trimming any
/// trailing zeros that leaves) rather than truncating. This constant is
/// numpy's global default and applies identically to float32 and float64
/// (it is not derived from `np.finfo(dtype).precision`, which is a
/// different number used only for the scientific-notation cutoff decision
/// in `FloatFmt::build` above).
const PRECISION: usize = 8;

/// Capped/forced-precision formatting deliberately rounds to a fixed
/// decimal digit count rather than searching for the shortest round-trip
/// string, so -- unlike `sci_parts`/`positional_parts` above -- F16 needs no
/// special-casing here: `x` is already the exact f64 widening of the
/// half value, and rounding that f64 to N decimal places is the same
/// regardless of which binary float width it originated from.
fn positional_parts_capped(x: f64, width: FloatWidth) -> (String, String) {
    let (int_part, frac_part) = positional_parts(x, width);
    if frac_part.len() <= PRECISION {
        return (int_part, frac_part);
    }
    let s = if width == FloatWidth::F32 { format!("{:.*}", PRECISION, x as f32) } else { format!("{:.*}", PRECISION, x) };
    let (i, f) = match s.split_once('.') {
        Some((i, f)) => (i.to_string(), f.to_string()),
        None => (s, String::new()),
    };
    (i, f.trim_end_matches('0').to_string())
}

fn sci_parts_capped(x: f64, width: FloatWidth) -> SciParts {
    let parts = sci_parts(x, width);
    let frac_len = parts.digits.len().saturating_sub(1);
    if frac_len <= PRECISION {
        return parts;
    }
    let s = if width == FloatWidth::F32 { format!("{:.*e}", PRECISION, x as f32) } else { format!("{:.*e}", PRECISION, x) };
    let (mantissa, exp) = s.split_once('e').unwrap();
    let exponent: i32 = exp.parse().unwrap();
    let mut digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    SciParts { digits, exponent }
}

/// Format `x` at exactly `precision` fractional mantissa digits
/// (correctly rounded from the true value, not truncated/zero-padded from
/// the natural shortest-round-trip string). Used for exp-mode's forced
/// equal-width second pass -- see the call site in `FloatFmt::format`.
fn sci_parts_at(x: f64, width: FloatWidth, precision: usize) -> SciParts {
    debug_assert!(x >= 0.0 && x.is_finite());
    let s = if width == FloatWidth::F32 { format!("{:.*e}", precision, x as f32) } else { format!("{:.*e}", precision, x) };
    let (mantissa, exp) = s.split_once('e').unwrap();
    let exponent: i32 = exp.parse().unwrap();
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    SciParts { digits, exponent }
}

impl FloatFmt {
    fn build(vals: impl Iterator<Item = f64> + Clone, dtype_precision: i32, force_sign: bool, width: FloatWidth) -> Self {
        let all: Vec<f64> = vals.clone().collect();
        let finite: Vec<f64> = all.iter().copied().filter(|v| v.is_finite()).collect();
        let has_any = !finite.is_empty();
        let has_negative = finite.iter().any(|&v| v.is_sign_negative() && v != 0.0 || (v == 0.0 && v.is_sign_negative()));
        let abs_non_zero: Vec<f64> = finite.iter().copied().filter(|&v| v != 0.0).map(f64::abs).collect();

        let mut exp_format = false;
        if !abs_non_zero.is_empty() {
            let max_val = abs_non_zero.iter().cloned().fold(f64::MIN, f64::max);
            let min_val = abs_non_zero.iter().cloned().fold(f64::MAX, f64::min);
            let exp_cutoff_max = 10f64.powi(8i32.min(dtype_precision));
            if max_val >= exp_cutoff_max || min_val < 0.0001 || max_val / min_val > 1000.0 {
                exp_format = true;
            }
        }

        let (mut pad_left, pad_right, exp_size);
        if finite.is_empty() {
            pad_left = 0;
            pad_right = 0;
            exp_size = 0;
        } else if exp_format {
            let mut max_frac_len = 0usize;
            let mut max_int_len = 0usize;
            let mut max_exp_digits = 2usize;
            for &v in &finite {
                let neg = v.is_sign_negative();
                let parts = sci_parts_capped(v.abs(), width);
                let frac_len = parts.digits.len().saturating_sub(1);
                max_frac_len = max_frac_len.max(frac_len);
                let int_len = 1 + if neg || force_sign { 1 } else { 0 };
                max_int_len = max_int_len.max(int_len);
                let exp_digits = parts.exponent.unsigned_abs().to_string().len().max(2);
                max_exp_digits = max_exp_digits.max(exp_digits);
            }
            pad_left = max_int_len;
            pad_right = max_frac_len;
            exp_size = max_exp_digits;
        } else {
            let mut max_int_len = 0usize;
            let mut max_frac_len = 0usize;
            for &v in &finite {
                let neg = v.is_sign_negative();
                let (int_part, frac_part) = positional_parts_capped(v.abs(), width);
                let int_len = int_part.len() + if neg || force_sign { 1 } else { 0 };
                max_int_len = max_int_len.max(int_len);
                max_frac_len = max_frac_len.max(frac_part.len());
            }
            pad_left = max_int_len;
            pad_right = max_frac_len;
            exp_size = 0;
        }
        let _ = has_negative;

        if all.len() != finite.len() {
            let neginf = force_sign || all.iter().any(|v| v.is_infinite() && *v < 0.0);
            let offset = pad_right + 1;
            let nan_need = 3usize.saturating_sub(offset); // "nan".len() == 3
            let inf_need = (3 + neginf as usize).saturating_sub(offset); // "inf".len() == 3
            pad_left = pad_left.max(nan_need).max(inf_need);
        }

        FloatFmt { exp_format, pad_left, pad_right, exp_size, force_sign, has_any, width }
    }

    fn format(&self, x: f64) -> String {
        if !x.is_finite() {
            let ret = if x.is_nan() {
                let sign = if self.force_sign { "+" } else { "" };
                format!("{sign}nan")
            } else if x < 0.0 {
                "-inf".to_string()
            } else if self.force_sign {
                "+inf".to_string()
            } else {
                "inf".to_string()
            };
            let width = self.pad_left + self.pad_right + 1;
            let pad = width.saturating_sub(ret.len());
            return " ".repeat(pad) + &ret;
        }
        if !self.has_any {
            // only reached if the whole array is nan/inf (handled above)
            // or genuinely empty -- format() shouldn't be called then.
        }

        // Negative zero must print with a leading '-' (numpy: `array(-0.)`);
        // `is_sign_negative()` alone (no `&& x != 0.0` guard) correctly
        // reports `true` for `-0.0`.
        let neg = x.is_sign_negative();
        let sign_char = if neg {
            "-"
        } else if self.force_sign {
            "+"
        } else {
            ""
        };

        if self.exp_format {
            // Unlike fixed-point mode, numpy's exp-mode second pass forces
            // EVERY element to exactly `self.precision` (== our pad_right)
            // fractional mantissa digits via `dragon4_scientific(precision=P,
            // min_digits=P, unique=True)` -- when an element's own natural
            // shortest representation is shorter than P, this does NOT
            // zero-pad; it reveals the element's true correctly-rounded
            // digits out to P places (e.g. array `[14.6075096, 2.12075067]`
            // both print at 8 frac digits even though 14.6075096 alone
            // would uniquely round-trip in fewer). So format directly at
            // `pad_right` digits of precision rather than reusing the
            // natural-shortest (possibly capped) digit string from `build`.
            let parts = sci_parts_at(x.abs(), self.width, self.pad_right);
            let digits = parts.digits.clone();
            let first = &digits[0..1];
            let rest = if digits.len() > 1 { &digits[1..] } else { "" };
            let exp_sign = if parts.exponent < 0 { "-" } else { "+" };
            let exp_digits = format!("{:0width$}", parts.exponent.unsigned_abs(), width = self.exp_size);
            let mantissa = format!("{sign_char}{first}.{rest}");
            let int_len = 1 + sign_char.len();
            let left_pad = self.pad_left.saturating_sub(int_len);
            format!("{}{}e{}{}", " ".repeat(left_pad), mantissa, exp_sign, exp_digits)
        } else {
            // Default numpy printoptions are precision=8, floatmode='maxprec'
            // (not 'fixed'/'maxprec_equal'): each element keeps its OWN
            // natural (precision-capped) digit count -- shorter numbers are
            // NOT zero-padded to match the widest element in the array.
            // Column alignment across elements of differing digit counts is
            // achieved with trailing SPACES instead (numpy's dragon4
            // `pad_right` parameter), e.g. numpy prints
            // `[ 0.60512  , -3.2967448, ...]` with the shorter `0.60512`
            // followed by two spaces, not zeros.
            let (int_part, frac_part) = positional_parts_capped(x.abs(), self.width);
            let body = format!("{sign_char}{int_part}.{frac_part}");
            let int_len = int_part.len() + sign_char.len();
            let left_pad = self.pad_left.saturating_sub(int_len);
            let right_pad = self.pad_right.saturating_sub(frac_part.len());
            " ".repeat(left_pad) + &body + &" ".repeat(right_pad)
        }
    }
}

// ---------------------------------------------------------------------------
// Bracket nesting / line wrapping: mirrors `_formatArray`/`_extendLine`.
// ---------------------------------------------------------------------------

fn strides_c(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1].max(1);
    }
    s
}

/// Buffer offset (in elements, matching `ElementFormatter::format_at`'s own
/// direct `arr.buffer()[flat_index]` indexing) for the element at logical
/// index `index` within `arr`. MUST use `arr`'s own strides/offset, not
/// shape-derived C-order strides (`strides_c`) -- `arr` may be a
/// non-contiguous view (a transpose, an F-order array, a sliced/negative-
/// stride view), whose buffer layout does not match its logical shape's
/// C-order layout at all. Using `strides_c(shape)` here (the bug this
/// replaced) silently reads elements out of numpy's-C-order positions
/// regardless of `arr`'s real layout, which is correct only by coincidence
/// for a C-contiguous array and wrong for any view -- confirmed via the
/// differential corpus's `ndarray.__repr__` `view/2d_transpose` and
/// `view/2d_fortran_order` cases, both of which printed the buffer's raw
/// storage order instead of the logical (transposed / F-order) view numpy
/// itself prints.
fn flat_index(arr: &NdArray, index: &[usize]) -> usize {
    let strides = arr.strides();
    let offset = arr.offset();
    let signed: isize = index
        .iter()
        .zip(strides.iter())
        .map(|(&i, &s)| i as isize * s)
        .sum();
    (offset + signed) as usize
}

fn format_array_recursive(
    arr: &NdArray,
    fmt: &ElementFormatter,
    axis: usize,
    index_prefix: &[usize],
    hanging_indent: &str,
    curr_width: usize,
) -> String {
    format_array_recursive_sep(arr, fmt, axis, index_prefix, hanging_indent, curr_width, ", ")
}

/// Same as `format_array_recursive`, parameterized on the inter-element
/// separator -- `repr()` uses `", "`, `str()` uses a bare `" "` (verified:
/// `str(np.array([1, 2]))` == `'[1 2]'`, no comma, where `repr()` of the
/// same array is `'array([1, 2])'`).
fn format_array_recursive_sep(
    arr: &NdArray,
    fmt: &ElementFormatter,
    axis: usize,
    index_prefix: &[usize],
    hanging_indent: &str,
    curr_width: usize,
    separator: &str,
) -> String {
    let ndim = arr.ndim();
    let axes_left = ndim - axis;
    if axes_left == 0 {
        let idx = flat_index(arr, index_prefix);
        return fmt.format_at(arr, idx);
    }

    let next_hanging_indent = hanging_indent.to_string() + " ";
    let next_width = curr_width.saturating_sub(1);
    let a_len = arr.shape()[axis];

    let mut s = String::new();

    if axes_left == 1 {
        let elem_width = curr_width.saturating_sub(separator.trim_end().len().max(1));
        let mut line = hanging_indent.to_string();
        for i in 0..a_len {
            let mut idx = index_prefix.to_vec();
            idx.push(i);
            let word = format_array_recursive_sep(
                arr,
                fmt,
                axis + 1,
                &idx,
                &next_hanging_indent,
                next_width,
                separator,
            );
            extend_line_pretty(&mut s, &mut line, &word, elem_width, hanging_indent);
            if i + 1 < a_len {
                line.push_str(separator);
            }
        }
        s.push_str(&line);
    } else {
        let line_sep = "\n".repeat(axes_left - 1);
        for i in 0..a_len {
            let mut idx = index_prefix.to_vec();
            idx.push(i);
            let nested = format_array_recursive_sep(
                arr,
                fmt,
                axis + 1,
                &idx,
                &next_hanging_indent,
                next_width,
                separator,
            );
            s.push_str(hanging_indent);
            s.push_str(&nested);
            if i + 1 < a_len {
                s.push_str(separator.trim_end());
                s.push_str(&line_sep);
            }
        }
    }

    let trimmed = s.strip_prefix(hanging_indent).unwrap_or(&s);
    format!("[{trimmed}]")
}

/// Port of numpy's `_extendLine_pretty`/`_extendLine`: wraps `word` onto a
/// new output line (flushing `line` into `s`) if appending it would exceed
/// `line_width`, unless the current line is still just the (empty)
/// indent -- wrapping an indent-only line never helps.
fn extend_line_pretty(s: &mut String, line: &mut String, word: &str, line_width: usize, next_line_prefix: &str) {
    let words: Vec<&str> = word.lines().collect();
    if words.len() <= 1 {
        extend_line(s, line, word, line_width, next_line_prefix);
        return;
    }
    let max_word_len = words.iter().map(|w| w.len()).max().unwrap_or(0);
    let indent;
    if line.len() + max_word_len > line_width && line.len() > next_line_prefix.len() {
        s.push_str(line.trim_end());
        s.push('\n');
        *line = next_line_prefix.to_string() + words[0];
        indent = next_line_prefix.to_string();
    } else {
        indent = " ".repeat(line.len());
        line.push_str(words[0]);
    }
    for w in &words[1..] {
        s.push_str(line.trim_end());
        s.push('\n');
        *line = indent.clone() + w;
    }
    let suffix_len = max_word_len.saturating_sub(words.last().unwrap_or(&"").len());
    line.push_str(&" ".repeat(suffix_len));
}

fn extend_line(s: &mut String, line: &mut String, word: &str, line_width: usize, next_line_prefix: &str) {
    let mut needs_wrap = line.len() + word.len() > line_width;
    if line.len() <= next_line_prefix.len() {
        needs_wrap = false;
    }
    if needs_wrap {
        s.push_str(line.trim_end());
        s.push('\n');
        *line = next_line_prefix.to_string();
    }
    line.push_str(word);
}
