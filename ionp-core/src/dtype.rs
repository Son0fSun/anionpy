//! The dtype system: the 13 concrete scalar types ionp supports, plus
//! `promote_dtype`, which implements numpy's `result_type`/`promote_types`
//! table for array-array binary operations (NEP 50, "strong" dtype side of
//! the rules — see KNOWN-DIFFERENCES.md for what is punted: weak-scalar
//! promotion for bare Python `int`/`float`/`complex` operands).
//!
//! Every other piece of the crate (buffer, array, ufunc) is generic over
//! this enum. Get this file wrong and everything downstream is wrong in a
//! way that is easy to miss and hard to trust.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    C64,  // complex64  = 2x f32
    C128, // complex128 = 2x f64
    /// numpy `S<n>` (`bytes_`): fixed-width, NUL-padded byte string. The
    /// `u32` payload is the itemsize **in bytes** (numpy's own `S` itemsize
    /// unit — `S5` carries `5`), unlike `U` below. THIS IS THE ONE
    /// STRUCTURAL DIFFERENCE from every other `DType` variant: itemsize is
    /// part of the *instance*, not a pure function of the variant tag, so
    /// `self as usize` (used by `SAFE_CAST_LT`'s old discriminant trick)
    /// no longer compiles for a data-carrying variant — see
    /// `safe_cast_lt`'s rewritten body. Two `S(n)` values with different
    /// `n` are different dtypes (`S3 != S5`), which `#[derive(PartialEq,
    /// Eq, Hash)]` already gets right for free since `u32` itself is
    /// `PartialEq + Eq + Hash`.
    S(u32),
    /// numpy `U<n>` (`str_`): fixed-width UCS-4 string, 4 bytes/char. The
    /// `u32` payload is the itemsize **in bytes** (so `U5` carries `20`,
    /// NOT `5`) — chosen so `itemsize()` stays a trivial projection for
    /// both variants and every byte-counting caller (buffer sizing,
    /// `.nbytes`, `tobytes()`) doesn't need a per-variant branch. Character
    /// count is `itemsize() / 4`, exposed via `DType::char_count`.
    U(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Bool,
    Int,
    UInt,
    Float,
    Complex,
    Bytes,
    Str,
}

impl DType {
    /// numpy-style short name, as printed by `dtype.name`. `Cow` rather than
    /// `&'static str` ONLY because the two flexible dtypes (`S`/`U`) carry a
    /// per-instance itemsize baked into the name string (`'bytes40'`,
    /// `'str160'`) that can't be a `'static` literal; the 14 fixed dtypes
    /// still return zero-cost `Cow::Borrowed` literals exactly as before.
    pub fn name(self) -> std::borrow::Cow<'static, str> {
        use std::borrow::Cow;
        match self {
            DType::Bool => Cow::Borrowed("bool"),
            DType::I8 => Cow::Borrowed("int8"),
            DType::I16 => Cow::Borrowed("int16"),
            DType::I32 => Cow::Borrowed("int32"),
            DType::I64 => Cow::Borrowed("int64"),
            DType::U8 => Cow::Borrowed("uint8"),
            DType::U16 => Cow::Borrowed("uint16"),
            DType::U32 => Cow::Borrowed("uint32"),
            DType::U64 => Cow::Borrowed("uint64"),
            DType::F16 => Cow::Borrowed("float16"),
            DType::F32 => Cow::Borrowed("float32"),
            DType::F64 => Cow::Borrowed("float64"),
            DType::C64 => Cow::Borrowed("complex64"),
            DType::C128 => Cow::Borrowed("complex128"),
            // Verified against real numpy 2.5.1: `np.dtype('S5').name ==
            // 'bytes40'`, `np.dtype('U5').name == 'str160'` — numpy's
            // `.name` for the flexible dtypes is the kind word plus the
            // itemsize in BITS (`itemsize * 8`), not bytes and not the
            // `'S5'`/`'U5'` short code (that spelling is `.str`/`repr`,
            // produced elsewhere — see `ionp-py`'s `dtype_repr`).
            DType::S(n) => Cow::Owned(format!("bytes{}", n * 8)),
            DType::U(n) => Cow::Owned(format!("str{}", n * 8)),
        }
    }

    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool | DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 | DType::F16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 | DType::C64 => 8,
            DType::C128 => 16,
            DType::S(n) => n as usize,
            DType::U(n) => n as usize,
        }
    }

    /// `U<n>`'s character count (`itemsize() / 4`, UCS-4). Panics if called
    /// on a non-`U` dtype — callers must check `matches!(self, DType::U(_))`
    /// (or just use `self.kind_char() == 'U'`) first; there is no sane
    /// "character count" of a `Bool`/`S` dtype to fall back to.
    pub fn char_count(self) -> usize {
        match self {
            DType::U(n) => (n / 4) as usize,
            _ => panic!("char_count() called on non-U dtype {self}"),
        }
    }

    /// numpy-style `dtype.alignment`: equal to `itemsize()` for every real
    /// (non-complex) dtype, but for the two complex dtypes it is the
    /// alignment of the underlying COMPONENT float, not the full packed
    /// itemsize -- `complex64` (itemsize 8, two float32 components) aligns
    /// like `float32` (4), and `complex128` (itemsize 16, two float64
    /// components) aligns like `float64` (8). Verified against real numpy
    /// 2.5.1's `np.dtype(name).alignment` for all 14 dtypes ionp supports;
    /// this is x86_64/arm64 native alignment, which is what numpy reports
    /// on both of ionp's supported platforms (no packed/unaligned dtype
    /// exists in ionp to report anything else for).
    pub fn alignment(self) -> usize {
        match self {
            DType::C64 => 4,
            DType::C128 => 8,
            // `S` is a raw byte string -- alignment 1, like `np.dtype('S5').alignment == 1`.
            DType::S(_) => 1,
            // `U` is UCS-4 (4-byte code units) -- `np.dtype('U5').alignment == 4`,
            // regardless of the declared char count. Verified against real numpy 2.5.1.
            DType::U(_) => 4,
            _ => self.itemsize(),
        }
    }

    /// numpy-style `dtype.kind` single-character code: `'b'` bool, `'i'`
    /// signed int, `'u'` unsigned int, `'f'` float, `'c'` complex, `'S'`
    /// bytes, `'U'` unicode. Verified against real numpy 2.5.1's
    /// `np.dtype(name).kind` for all 14 fixed dtypes ionp supports plus the
    /// two flexible ones -- exactly the same partition `Kind` already
    /// encodes, just given numpy's own printable spelling.
    pub fn kind_char(self) -> char {
        match self.kind() {
            Kind::Bool => 'b',
            Kind::Int => 'i',
            Kind::UInt => 'u',
            Kind::Float => 'f',
            Kind::Complex => 'c',
            Kind::Bytes => 'S',
            Kind::Str => 'U',
        }
    }

    pub fn is_string(self) -> bool {
        matches!(self.kind(), Kind::Bytes | Kind::Str)
    }
    pub fn is_bytes_dtype(self) -> bool {
        matches!(self.kind(), Kind::Bytes)
    }
    pub fn is_unicode_dtype(self) -> bool {
        matches!(self.kind(), Kind::Str)
    }

    pub fn is_bool(self) -> bool {
        matches!(self, DType::Bool)
    }
    pub fn is_integer(self) -> bool {
        matches!(self.kind(), Kind::Int | Kind::UInt)
    }
    pub fn is_signed_integer(self) -> bool {
        matches!(self.kind(), Kind::Int)
    }
    pub fn is_unsigned_integer(self) -> bool {
        matches!(self.kind(), Kind::UInt)
    }
    pub fn is_floating(self) -> bool {
        matches!(self.kind(), Kind::Float)
    }
    pub fn is_complex(self) -> bool {
        matches!(self.kind(), Kind::Complex)
    }
    /// numpy's `np.inexact` category: the float and complex widths, i.e.
    /// exactly the dtypes that can hold a NaN. Bool and every integer
    /// width are EXACT and answer `false`.
    ///
    /// This is not cosmetic sugar over `is_floating() || is_complex()`:
    /// numpy branches real, observable behaviour on this category. See
    /// `reduce_axes_from_pyobj` in `ionp-py/src/reductions.rs`, where
    /// `nanmean`/`nanvar`/`nanstd` accept a 0-d `axis=0` for inexact
    /// operands and raise `AxisError` for exact ones.
    pub fn is_inexact(self) -> bool {
        matches!(self.kind(), Kind::Float | Kind::Complex)
    }

    fn kind(self) -> Kind {
        match self {
            DType::Bool => Kind::Bool,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => Kind::Int,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => Kind::UInt,
            DType::F16 | DType::F32 | DType::F64 => Kind::Float,
            DType::C64 | DType::C128 => Kind::Complex,
            DType::S(_) => Kind::Bytes,
            DType::U(_) => Kind::Str,
        }
    }

    /// Bit width of the *representation*. For complex types this is the
    /// total width (2x the component float width) so that "bigger bits"
    /// comparisons stay meaningful within same-kind promotion.
    fn bits(self) -> u32 {
        (self.itemsize() * 8) as u32
    }

    fn int_of(bits: u32) -> DType {
        match bits {
            8 => DType::I8,
            16 => DType::I16,
            32 => DType::I32,
            _ => DType::I64,
        }
    }
    fn uint_of(bits: u32) -> DType {
        match bits {
            8 => DType::U8,
            16 => DType::U16,
            32 => DType::U32,
            _ => DType::U64,
        }
    }
    fn float_of(bits: u32) -> DType {
        if bits <= 16 {
            DType::F16
        } else if bits <= 32 {
            DType::F32
        } else {
            DType::F64
        }
    }
    fn complex_of(total_bits: u32) -> DType {
        if total_bits <= 64 {
            DType::C64
        } else {
            DType::C128
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// numpy's fixed int/uint cross-promotion table. This is NOT "pick the
/// bigger one" — a same-width int/uint pair has no int type that can hold
/// both ranges, so numpy promotes to float64, and asymmetric widths only
/// upgrade to the next size that provably holds both ranges. Table
/// reproduced from `numpy.promote_types` (verified against numpy 2.4.2):
///
/// ```text
///        u8    u16   u32   u64
///  i8    i16   i32   i64   f64
///  i16   i16   i32   i64   f64
///  i32   i32   i32   i64   f64
///  i64   i64   i64   i64   f64
/// ```
fn int_uint_table(ibits: u32, ubits: u32) -> DType {
    match (ibits, ubits) {
        (8, 8) => DType::I16,
        (8, 16) => DType::I32,
        (8, 32) => DType::I64,
        (8, 64) => DType::F64,
        (16, 8) => DType::I16,
        (16, 16) => DType::I32,
        (16, 32) => DType::I64,
        (16, 64) => DType::F64,
        (32, 8) => DType::I32,
        (32, 16) => DType::I32,
        (32, 32) => DType::I64,
        (32, 64) => DType::F64,
        (64, 8) => DType::I64,
        (64, 16) => DType::I64,
        (64, 32) => DType::I64,
        (64, 64) => DType::F64,
        _ => unreachable!("int/uint bit widths are always in {{8,16,32,64}}"),
    }
}

/// numpy's `result_type(a, b)` for two *array* dtypes (not scalars — see
/// module docs on the weak-scalar punt). Both operand orders give the same
/// result: promotion is commutative.
pub fn promote_dtype(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }
    use Kind::*;
    let (ka, kb) = (a.kind(), b.kind());

    // Flexible (`S`/`U`) dtypes checked BEFORE the `Bool` absorption
    // wildcards below: bool is NOT simply absorbed into a string dtype the
    // way it is into every numeric kind -- `np.promote_types(bool_, 'S2')
    // == 'S5'` (bool's own minimum-safe-string-width of 5 wins over the
    // narrower operand), verified live against numpy 2.5.1, so a naive
    // `(Bool, _) => b` here would be wrong whenever `b` is narrower than
    // that minimum. See `string_min_itemsize` below for the per-numeric-
    // dtype width table (same table `safe_cast_lt`'s numeric<->string arm
    // uses -- kept as one function, not forked).
    if ka == Bytes || kb == Bytes || ka == Str || kb == Str {
        return promote_dtype_with_string(a, ka, b, kb);
    }

    match (ka, kb) {
        (Bool, _) => b,
        (_, Bool) => a,

        (Int, Int) => DType::int_of(a.bits().max(b.bits())),
        (UInt, UInt) => DType::uint_of(a.bits().max(b.bits())),
        (Int, UInt) => int_uint_table(a.bits(), b.bits()),
        (UInt, Int) => int_uint_table(b.bits(), a.bits()),

        (Float, Float) => DType::float_of(a.bits().max(b.bits())),

        (Int, Float) | (UInt, Float) => DType::float_of(int_equiv_float_bits(a.bits()).max(b.bits())),
        (Float, Int) | (Float, UInt) => DType::float_of(a.bits().max(int_equiv_float_bits(b.bits()))),

        (Complex, Complex) => DType::complex_of(a.bits().max(b.bits())),

        (Float, Complex) => {
            let real_bits = a.bits().max(b.bits() / 2);
            DType::complex_of(real_bits * 2)
        }
        (Complex, Float) => {
            let real_bits = b.bits().max(a.bits() / 2);
            DType::complex_of(real_bits * 2)
        }
        (Int, Complex) | (UInt, Complex) => {
            let real_bits = int_equiv_float_bits(a.bits()).max(b.bits() / 2);
            DType::complex_of(real_bits * 2)
        }
        (Complex, Int) | (Complex, UInt) => {
            let real_bits = int_equiv_float_bits(b.bits()).max(a.bits() / 2);
            DType::complex_of(real_bits * 2)
        }

        // Unreachable: every `(Bytes, _)`/`(Str, _)` pairing already
        // returned above via `promote_dtype_with_string`, so this arm only
        // exists to keep the match exhaustive over `Kind`'s now-7 variants.
        (Bytes, _) | (_, Bytes) | (Str, _) | (_, Str) => {
            unreachable!("string kinds are handled by the early return above")
        }
    }
}

/// Per-numeric-dtype minimum `S`/`U` width numpy requires to safely hold
/// every value of that dtype as a decimal (int/uint/bool) or general
/// (float/complex) string representation -- e.g. `int64`'s min/max need up
/// to 20 digits plus a sign, and numpy's own table rounds that up to 21.
/// Units are "characters" (== `S` itemsize in bytes == `U` itemsize in
/// bytes/4). Verified live against real numpy 2.5.1 by binary-searching
/// `np.can_cast(dtype, f'S{n}', casting='safe')` for the smallest `n` that
/// returns `True`, for all 14 fixed dtypes; `U` gave the identical numbers
/// (in chars, not bytes) at every width tried. Used by both
/// `promote_dtype_with_string` (promotion) and `safe_cast_lt` (casting) --
/// the same underlying numpy rule drives both, so this is the one place it
/// is encoded.
fn string_min_itemsize(d: DType) -> u32 {
    match d {
        DType::Bool => 5,
        DType::I8 => 4,
        DType::I16 => 6,
        DType::I32 => 11,
        DType::I64 => 21,
        DType::U8 => 3,
        DType::U16 => 5,
        DType::U32 => 10,
        DType::U64 => 20,
        DType::F16 | DType::F32 | DType::F64 => 32,
        DType::C64 | DType::C128 => 64,
        DType::S(_) | DType::U(_) => {
            unreachable!("string_min_itemsize is only called with a numeric operand")
        }
    }
}

/// `promote_dtype`'s flexible-dtype half: `a`/`b` where at least one of
/// `ka`/`kb` is `Kind::Bytes` or `Kind::Str`. All four sub-cases (string x
/// string, string x numeric, and the symmetric flips) verified live against
/// real numpy 2.5.1's `np.promote_types`:
///   - `S x S`, `U x U`: same-kind, result itemsize is the max of the two.
///   - `S x U`: result is ALWAYS `U` (never `S`) -- numpy has no
///     "promote to bytes" outcome once unicode is involved -- with char
///     count `max(S's byte-count, U's char-count)` (an `S` byte and a `U`
///     char are treated as equal-weight units here, confirmed at 5 separate
///     width pairs, not just one).
///   - `numeric x S`/`numeric x U` (either order): result is the string
///     kind already present, widened (never narrowed) to
///     `string_min_itemsize(numeric)` if the given string dtype's own
///     width is smaller -- e.g. `promote_types(bool_, 'S2') == 'S5'`, NOT
///     `'S2'`, because bool's own minimum-safe width (5) wins.
fn promote_dtype_with_string(a: DType, ka: Kind, b: DType, kb: Kind) -> DType {
    use Kind::*;
    match (ka, kb) {
        (Bytes, Bytes) => DType::S(a.itemsize().max(b.itemsize()) as u32),
        (Str, Str) => DType::U(a.itemsize().max(b.itemsize()) as u32),
        (Bytes, Str) => DType::U((4 * a.itemsize()).max(b.itemsize()) as u32),
        (Str, Bytes) => DType::U(a.itemsize().max(4 * b.itemsize()) as u32),
        (Bytes, _) => DType::S(a.itemsize().max(string_min_itemsize(b) as usize) as u32),
        (_, Bytes) => DType::S(b.itemsize().max(string_min_itemsize(a) as usize) as u32),
        (Str, _) => DType::U((a.itemsize().max(4 * string_min_itemsize(b) as usize)) as u32),
        (_, Str) => DType::U((b.itemsize().max(4 * string_min_itemsize(a) as usize)) as u32),
        _ => unreachable!("caller guarantees at least one of ka/kb is Bytes or Str"),
    }
}

impl DType {
    /// Inclusive `(min, max)` range of `self` as `i128`, for bounds-checking
    /// a Python `int` before it is narrowed into this dtype (NEP 50
    /// weak-scalar promotion: out-of-range Python ints must raise
    /// `OverflowError`, not silently wrap the way an already-array-typed
    /// operand does). `None` for non-integer dtypes, which have no such
    /// fixed range.
    pub fn int_bounds(self) -> Option<(i128, i128)> {
        match self {
            DType::Bool => Some((0, 1)),
            DType::I8 => Some((i8::MIN as i128, i8::MAX as i128)),
            DType::I16 => Some((i16::MIN as i128, i16::MAX as i128)),
            DType::I32 => Some((i32::MIN as i128, i32::MAX as i128)),
            DType::I64 => Some((i64::MIN as i128, i64::MAX as i128)),
            DType::U8 => Some((0, u8::MAX as i128)),
            DType::U16 => Some((0, u16::MAX as i128)),
            DType::U32 => Some((0, u32::MAX as i128)),
            DType::U64 => Some((0, u64::MAX as i128)),
            DType::F16 | DType::F32 | DType::F64 | DType::C64 | DType::C128 => None,
            DType::S(_) | DType::U(_) => None,
        }
    }

    /// Rank of `self`'s promotion "kind" for NEP 50 weak-scalar comparison:
    /// `Bool < {Int,UInt} < Float < Complex`. A weak Python scalar whose own
    /// kind-rank is `<=` the array's does not widen the array's dtype;
    /// higher ranks trigger a kind-appropriate default (see `ionp-py`'s
    /// `weak_target_dtype`, the only caller — this is array-dtype-side rank
    /// only, the scalar side has no `DType` to rank until after this
    /// decision).
    pub fn kind_rank(self) -> u8 {
        if self.is_bool() {
            0
        } else if self.is_integer() {
            1
        } else if self.is_floating() {
            2
        } else {
            3
        }
    }
}

/// Strict partial order used by `dtype.__lt__`/`__le__`/`__gt__`/`__ge__`:
/// `a < b` iff `a` numpy-casts to `b` under `casting='safe'` and `a != b`.
/// Measured live against real numpy 2.5.1, `np.can_cast(a, b, 'safe')`, for
/// ALL 196 ordered pairs of the 14 dtypes this crate supports (14x14, self
/// pairs included) -- zero mismatches. This is genuinely a PARTIAL order,
/// not a total one: e.g. `int8`/`uint8` are incomparable in both
/// directions (`int8 < uint8` and `uint8 < int8` are both `False` against
/// real numpy, not one `True`/one raise), and this table reproduces that
/// exactly rather than approximating with e.g. itemsize. Indexed
/// `[self as usize][other as usize]` using `DType`'s declaration order
/// (`Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, C64,
/// C128`) as the discriminant -- do not reorder the enum without
/// regenerating this table.
///
/// NOT the same relation as `kind_rank`/`ScalarKind::rank` above (NEP 50
/// weak-scalar promotion, a 4-bucket total preorder over {Bool, Int/UInt,
/// Float, Complex} kinds) -- this is the full safe-casting partial order
/// over all 14 concrete dtypes, unrelated in purpose and shape.
#[rustfmt::skip]
const SAFE_CAST_LT: [[bool; 14]; 14] = [
    // to:    Bool   I8     I16    I32    I64    U8     U16    U32    U64    F16    F32    F64    C64    C128
    /*Bool*/ [false, true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true ],
    /*I8  */ [false, false, true,  true,  true,  false, false, false, false, true,  true,  true,  true,  true ],
    /*I16 */ [false, false, false, true,  true,  false, false, false, false, false, true,  true,  true,  true ],
    /*I32 */ [false, false, false, false, true,  false, false, false, false, false, false, true,  false, true ],
    /*I64 */ [false, false, false, false, false, false, false, false, false, false, false, true,  false, true ],
    /*U8  */ [false, false, true,  true,  true,  false, true,  true,  true,  true,  true,  true,  true,  true ],
    /*U16 */ [false, false, false, true,  true,  false, false, true,  true,  false, true,  true,  true,  true ],
    /*U32 */ [false, false, false, false, true,  false, false, false, true,  false, false, true,  false, true ],
    /*U64 */ [false, false, false, false, false, false, false, false, false, false, false, true,  false, true ],
    /*F16 */ [false, false, false, false, false, false, false, false, false, false, true,  true,  true,  true ],
    /*F32 */ [false, false, false, false, false, false, false, false, false, false, false, true,  true,  true ],
    /*F64 */ [false, false, false, false, false, false, false, false, false, false, false, false, false, true ],
    /*C64 */ [false, false, false, false, false, false, false, false, false, false, false, false, false, true ],
    /*C128*/ [false, false, false, false, false, false, false, false, false, false, false, false, false, false],
];

/// `SAFE_CAST_LT`'s row/col index for the 14 fixed-width dtypes, `None` for
/// `S`/`U` -- replaces the old `self as usize` discriminant cast, which
/// stopped compiling the moment `DType` grew data-carrying variants (Rust's
/// `as` numeric cast on an enum is only defined for field-less/"C-like"
/// enums). Same declaration order as the table's own doc comment.
fn fixed_discriminant(d: DType) -> Option<usize> {
    match d {
        DType::Bool => Some(0),
        DType::I8 => Some(1),
        DType::I16 => Some(2),
        DType::I32 => Some(3),
        DType::I64 => Some(4),
        DType::U8 => Some(5),
        DType::U16 => Some(6),
        DType::U32 => Some(7),
        DType::U64 => Some(8),
        DType::F16 => Some(9),
        DType::F32 => Some(10),
        DType::F64 => Some(11),
        DType::C64 => Some(12),
        DType::C128 => Some(13),
        DType::S(_) | DType::U(_) => None,
    }
}

/// `safe_cast_lt`'s string half: `self`/`other` where at least one side is
/// `S`/`U`. Verified live against real numpy 2.5.1's
/// `np.can_cast(a, b, casting='safe')`:
///   - `S(n) < S(m)` iff `n < m` (byte-for-byte, same kind).
///   - `U(n) < U(m)` iff `n < m` (byte-for-byte -- both sides already in
///     the same 4-bytes/char unit, so no /4 needed to compare).
///   - `S(n bytes) < U(m bytes)` iff `m >= 4*n` (an `S` byte safely widens
///     into an `U` char one-for-one; confirmed at 5 width pairs, not just
///     the equal-width one) -- non-strict on this cross-kind pairing since
///     `S(n) != U(m)` always (different kind), so equality of the
///     underlying capacity still counts as `<`.
///   - `U(_) < S(_)`: ALWAYS `false` -- unicode never safely narrows to
///     bytes at any width (verified: `U1 -> S4` is `False`, not just the
///     equal-width case).
///   - numeric `d < S(n)` iff `n >= string_min_itemsize(d)` (numpy's fixed
///     per-dtype "always fits" width table -- see that function).
///   - numeric `d < U(n bytes)` iff `n >= 4*string_min_itemsize(d)`.
///   - `S(_) < numeric` / `U(_) < numeric`: ALWAYS `false` (verified:
///     `S5 -> int8`/`bool`/`float32`/`int64` are all `False` -- a string
///     dtype, however narrow, never safely casts INTO a numeric one).
fn string_safe_cast_lt(from: DType, to: DType) -> bool {
    match (from, to) {
        (DType::S(n), DType::S(m)) => n < m,
        (DType::U(n), DType::U(m)) => n < m,
        (DType::S(n), DType::U(m)) => m >= 4 * n,
        (DType::U(_), DType::S(_)) => false,
        (_, DType::S(m)) => m >= string_min_itemsize(from),
        (_, DType::U(m)) => m >= 4 * string_min_itemsize(from),
        (DType::S(_), _) | (DType::U(_), _) => false,
        _ => unreachable!(
            "safe_cast_lt only routes here when at least one of from/to is S/U; \
             a from=numeric,to=numeric pair never reaches string_safe_cast_lt \
             (fixed_discriminant would have matched both sides instead)"
        ),
    }
}

impl DType {
    /// `self < other` under the safe-casting partial order (see
    /// `SAFE_CAST_LT` above). Total function -- always returns a `bool`,
    /// never panics or needs an "incomparable" third state, because a
    /// partial order's incomparable pairs are represented by BOTH
    /// `lt(a,b)` and `lt(b,a)` returning `false`, which is exactly what a
    /// dense truth table already gives for free. `S`/`U` are NOT part of
    /// `SAFE_CAST_LT`'s dense table (their itemsize is unbounded, so no
    /// fixed 14x14-style table can enumerate them) -- routed instead to
    /// `string_safe_cast_lt`, which encodes the same real-numpy-verified
    /// relation as a formula instead of a lookup.
    pub fn safe_cast_lt(self, other: DType) -> bool {
        match (fixed_discriminant(self), fixed_discriminant(other)) {
            (Some(i), Some(j)) => SAFE_CAST_LT[i][j],
            _ => string_safe_cast_lt(self, other),
        }
    }
}

/// Which of the four NEP 50 "weak scalar" kinds a bare Python
/// `bool`/`int`/`float`/`complex` operand belongs to. `Bool` is ranked
/// below `Int` (Python's `bool` is a subtype of `int`, but a bare Python
/// `True`/`False` is weaker than a bare Python `int` for promotion
/// purposes, matching numpy). This type has no PyO3 dependency — it is the
/// pure classification/promotion-rule half of weak-scalar handling; the
/// impure half (recognizing a `Bound<'_, PyAny>`'s Python type and
/// extracting/marshaling its value into a `Buffer`) lives in `ionp-py`,
/// which has no logic of its own left to get wrong once it calls
/// `weak_target_dtype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    Bool,
    Int,
    Float,
    Complex,
}

impl ScalarKind {
    /// NEP 50 kind rank, directly comparable against `DType::kind_rank`
    /// (`Bool=0 < {Int,UInt}=1 < Float=2 < Complex=3`).
    pub fn rank(self) -> u8 {
        match self {
            ScalarKind::Bool => 0,
            ScalarKind::Int => 1,
            ScalarKind::Float => 2,
            ScalarKind::Complex => 3,
        }
    }
}

/// NEP 50 weak-scalar promotion: the target dtype of `array_dtype op
/// scalar`, where `scalar` is a bare Python scalar of the given `kind`
/// (never a numpy scalar or 0-d array — those are "strong" and use ordinary
/// `promote_dtype` array-array promotion instead, unrelated to this
/// function).
///
/// Rule, verified against a real numpy 2.5.1 install across all 13 ionp
/// dtypes x all 4 scalar kinds:
///   - if the scalar's kind-rank is `<=` the array's, the scalar does NOT
///     widen the array's dtype at all — this is the defining "weak"
///     behavior (`float32_array + 1.0` stays `float32`; `int8_array + 1000`
///     stays `int8`, subject to `int_bounds` raising `OverflowError` first).
///   - otherwise the scalar promotes to the *default* dtype for its kind
///     (`int` -> int64, `float` -> float64, `complex` -> complex128) —
///     EXCEPT a `complex` scalar against a `float`-kind array, which numpy
///     promotes to the *width-matched* complex type instead of the default
///     (`float32_array + 1j` -> `complex64`, not `complex128`).
pub fn weak_target_dtype(array_dtype: DType, kind: ScalarKind) -> DType {
    if kind.rank() <= array_dtype.kind_rank() {
        return array_dtype;
    }
    match kind {
        ScalarKind::Bool => {
            unreachable!("bool has rank 0, never greater than any array kind_rank")
        }
        ScalarKind::Int => DType::I64,
        ScalarKind::Float => DType::F64,
        ScalarKind::Complex => {
            if array_dtype.is_floating() {
                match array_dtype {
                    // Verified against real numpy 2.5.1:
                    // `np.array([1], dtype=np.float16) + 1j` -> complex64,
                    // the same width-matched (not kind-default) special
                    // case float32 gets, not complex128.
                    DType::F16 => DType::C64,
                    DType::F32 => DType::C64,
                    DType::F64 => DType::C128,
                    _ => unreachable!("is_floating() guarantees F16, F32, or F64"),
                }
            } else {
                DType::C128
            }
        }
    }
}

/// Bit width of the float type needed to represent an integer type's range
/// without immediately truncating, matching numpy's `promote_types` table
/// exactly (verified against real numpy 2.5.1, e.g.
/// `np.promote_types('int8','float16') == float16` but
/// `np.promote_types('int16','float16') == float32`): 8-bit int/uint fits
/// float16's 11-bit significand exactly; 16-bit int/uint needs float32;
/// 32/64-bit int/uint need float64.
fn int_equiv_float_bits(int_bits: u32) -> u32 {
    if int_bits <= 8 {
        16
    } else if int_bits <= 16 {
        32
    } else {
        64
    }
}

/// numpy's `casting=` kwarg (`astype`, ufunc `out=`, etc.): the five modes
/// are `'no'`/`'equiv'` (identical dtype only -- ionp is native-order-only,
/// so `'equiv'` collapses to `'no'`), `'safe'` (value-preserving only),
/// `'same_kind'` (`'safe'`, plus a documented set of narrowing/kind-
/// crossing casts numpy still considers acceptable), and `'unsafe'`
/// (anything).
///
/// `dtype_index` maps `DType` onto a row/col in `SAFE_CAST` below, in the
/// exact order `bool, int8, int16, int32, int64, uint8, uint16, uint32,
/// uint64, float16, float32, float64, complex64, complex128`.
/// `np.can_cast(from, to, casting='safe')` for every one of the 14x14 = 196
/// `(from, to)` pairs across bool/int8..int64/uint8..uint64/float16/
/// float32/float64/complex64/complex128, verified live against real numpy
/// 2.5.1 (`np.can_cast(a, b, casting='safe')` for all 196 pairs, 0
/// mismatches against this table).
///
/// This is NOT a derivable formula in the naive "does the mantissa have
/// enough bits" sense: numpy's OWN safe-casting table treats `int64` ->
/// `float64` and `uint64` -> `float64` (also `-> complex128`) as safe even
/// though a `float64` mantissa (52 bits) cannot exactly represent every
/// `int64`/`uint64` value (which need up to 63/64 bits) -- this is a
/// documented, long-standing accepted imprecision in numpy's own safe-cast
/// table (same-itemsize int/uint <-> float is treated as safe at every
/// width, including the one width where it is not actually lossless), not
/// a bug in ionp's reproduction of it. Reproducing numpy's ACTUAL table
/// (including this quirk) is the correct behavior here, not "fixing" it to
/// what pure precision analysis would say -- a `casting='safe'` cast that
/// numpy accepts must NOT raise in ionp, or every differential case
/// exercising it fails for the opposite reason `ndarray.astype` was
/// withdrawn.
#[rustfmt::skip]
const SAFE_CAST: [[bool; 14]; 14] = [
    // bool
    [true , true , true , true , true , true , true , true , true , true , true , true , true , true ],
    // int8
    [false, true , true , true , true , false, false, false, false, true , true , true , true , true ],
    // int16
    [false, false, true , true , true , false, false, false, false, false, true , true , true , true ],
    // int32
    [false, false, false, true , true , false, false, false, false, false, false, true , false, true ],
    // int64
    [false, false, false, false, true , false, false, false, false, false, false, true , false, true ],
    // uint8
    [false, false, true , true , true , true , true , true , true , true , true , true , true , true ],
    // uint16
    [false, false, false, true , true , false, true , true , true , false, true , true , true , true ],
    // uint32
    [false, false, false, false, true , false, false, true , true , false, false, true , false, true ],
    // uint64
    [false, false, false, false, false, false, false, false, true , false, false, true , false, true ],
    // float16
    [false, false, false, false, false, false, false, false, false, true , true , true , true , true ],
    // float32
    [false, false, false, false, false, false, false, false, false, false, true , true , true , true ],
    // float64
    [false, false, false, false, false, false, false, false, false, false, false, true , false, true ],
    // complex64
    [false, false, false, false, false, false, false, false, false, false, false, false, true , true ],
    // complex128
    [false, false, false, false, false, false, false, false, false, false, false, false, false, true ],
];

/// `casting='safe'`: table lookup for the 14 fixed dtypes (see `SAFE_CAST`
/// doc comment), or `from == to || from.safe_cast_lt(to)` -- the same
/// relation `SAFE_CAST`'s diagonal-inclusive table already encodes for the
/// fixed dtypes -- for any pair touching `S`/`U`, since a `u32`-parameterised
/// dtype has no fixed row/col to look up.
pub fn is_safe_cast(from: DType, to: DType) -> bool {
    match (fixed_discriminant(from), fixed_discriminant(to)) {
        (Some(i), Some(j)) => SAFE_CAST[i][j],
        _ => from == to || from.safe_cast_lt(to),
    }
}

/// The additional pairs `casting='same_kind'` accepts beyond `'safe'`.
/// Derived (not guessed) from numpy 2.5.1's live `can_cast(..,
/// casting='same_kind')` truth table for all 196 pairs by computing, for
/// every pair NOT already `'safe'`, whether `'same_kind'` still accepts
/// it, then generalizing by `Kind` -- verified to reproduce all 196
/// `'same_kind'` answers with ZERO mismatches (script used to verify is
/// throwaway, not part of this crate; the rule below is what was checked,
/// not the script's incidental behavior):
///
/// - a cast INTO `Bool` is never `same_kind`-extra (only `bool->bool`,
///   already `'safe'`).
/// - `{Bool,Int,UInt} -> {Int,UInt}` is `same_kind`-extra in every
///   direction and every width EXCEPT signed `Int -> UInt` ever (verified:
///   `int8->uint8` through `int64->uint64` are ALL rejected under
///   `same_kind`, even `int8->uint64` which is a pure widening in range --
///   numpy treats crossing from signed to unsigned as unsafe at every
///   width, full stop, because a negative source value has no
///   representation on the unsigned side). The reverse, `UInt -> Int`, is
///   accepted at every width pair including the most extreme narrowing
///   (`uint64->int8`).
/// - `{Bool,Int,UInt,Float} -> {Float,Complex}` is `same_kind`-extra
///   (any integer or float source may narrow into any float or complex
///   destination).
/// - `Complex -> Complex` is `same_kind`-extra (narrowing precision only,
///   e.g. `complex128->complex64`); `Complex -> Float` and `Complex ->
///   {Bool,Int,UInt}` are NEVER `same_kind`-extra (verified: `complex64
///   -> float32` stays `same_kind=False` despite being a same-total-width
///   truncation of the imaginary part -- numpy never lets a cast that
///   drops the imaginary component through below `'unsafe'`).
/// `Bytes`/`Str` destination arms verified live against real numpy 2.5.1
/// the same way as the rest of this function: a cast INTO `S`, from ANY
/// numeric kind or from `S` itself, is `same_kind`-extra regardless of
/// width (`bool -> S1` is `True` under `same_kind` despite needing `S5` to
/// be `safe`; `S5 -> S1` narrowing is also `True`) -- except FROM `U`,
/// which is never `same_kind`-extra into `S` at any width (`U1 -> S10` is
/// `False`, matching `U -> S` being permanently unsafe). A cast INTO `U`
/// is `same_kind`-extra from anything on the numeric/`Bytes`/`Str` side
/// (every one of those, any width, verified `True`). A cast FROM `Bytes`/
/// `Str` INTO a numeric kind is never `same_kind`-extra (`S1 -> bool` and
/// `U1 -> bool` both `False`) -- already `false` under the existing
/// `Int|UInt`/`Float`/`Complex` arms above since none of their `matches!`
/// patterns list `Bytes`/`Str`, so no change needed there.
fn same_kind_extra(from: DType, to: DType) -> bool {
    use Kind::*;
    let (kf, kt) = (from.kind(), to.kind());
    match kt {
        Bool => false,
        Int | UInt => matches!(kf, Bool | Int | UInt) && !(kf == Int && kt == UInt),
        Float => matches!(kf, Bool | Int | UInt | Float),
        Complex => matches!(kf, Bool | Int | UInt | Float | Complex),
        Bytes => matches!(kf, Bool | Int | UInt | Float | Complex | Bytes),
        Str => matches!(kf, Bool | Int | UInt | Float | Complex | Bytes | Str),
    }
}

/// `casting='same_kind'`: `'safe'` plus `same_kind_extra`'s additions.
pub fn is_same_kind_cast(from: DType, to: DType) -> bool {
    is_safe_cast(from, to) || same_kind_extra(from, to)
}

/// The full `can_cast(from, to, casting=rule)` relation for all five of
/// numpy's documented `casting=` modes. `'no'` and `'equiv'` collapse to
/// the same check here because ionp has no non-native byte order for
/// `'equiv'` to additionally permit (verified against real numpy 2.5.1:
/// for every one of the 196 pairs, `can_cast(.., 'no') ==
/// can_cast(.., 'equiv') == (from == to)`).
///
/// `rule` is assumed already validated to be one of the five legal
/// spellings (`check_casting_kwarg`-equivalent validation is the caller's
/// job, same division of labor `casting_rule_type_error`/
/// `check_casting_kwarg` already use in `ionp-py`). An unrecognized rule
/// falls back to `false` rather than panicking, so a caller that somehow
/// skips validation fails closed (raises) instead of silently allowing
/// an unsafe cast.
pub fn can_cast(from: DType, to: DType, rule: &str) -> bool {
    match rule {
        "no" | "equiv" => from == to,
        "safe" => is_safe_cast(from, to),
        "same_kind" => is_same_kind_cast(from, to),
        "unsafe" => true,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Dtype introspection/promotion toplevel support: `min_scalar_type`,
// `finfo`/`iinfo` constant tables, and the static `typename`/`mintypecode`
// lookup tables. `promote_types`/`result_type`'s NEP 50 orchestration lives
// in `ionp-py` (it needs `classify_scalar`/`weak_target_dtype`, which are
// PyO3-side and core-side respectively) -- only the pieces that are pure
// value/dtype logic with no PyO3 dependency live here.
// ---------------------------------------------------------------------------

/// Smallest *unsigned* dtype that can hold `v` (`v` assumed `>= 0`), matching
/// `np.min_scalar_type`'s table for non-negative Python ints. `None` means
/// the value exceeds `u64::MAX` (18446744073709551615) -- real numpy falls
/// back to `dtype('O')` there, which ionp has no representation for at all
/// (ionp has no object dtype, full stop; this is an architectural absence,
/// not a bug in this function).
pub fn min_scalar_type_unsigned(v: u128) -> Option<DType> {
    if v <= u8::MAX as u128 {
        Some(DType::U8)
    } else if v <= u16::MAX as u128 {
        Some(DType::U16)
    } else if v <= u32::MAX as u128 {
        Some(DType::U32)
    } else if v <= u64::MAX as u128 {
        Some(DType::U64)
    } else {
        None
    }
}

/// Smallest *signed* dtype that can hold `v` (`v` assumed `< 0`), matching
/// `np.min_scalar_type`'s table for negative Python ints. `None` means the
/// value is below `i64::MIN` (-9223372036854775808) -- same object-dtype
/// fallback caveat as `min_scalar_type_unsigned`.
pub fn min_scalar_type_signed(v: i128) -> Option<DType> {
    if v >= i8::MIN as i128 {
        Some(DType::I8)
    } else if v >= i16::MIN as i128 {
        Some(DType::I16)
    } else if v >= i32::MIN as i128 {
        Some(DType::I32)
    } else if v >= i64::MIN as i128 {
        Some(DType::I64)
    } else {
        None
    }
}

/// Smallest float dtype that can hold `v`, matching `np.min_scalar_type`'s
/// table for Python floats: float16 if it fits, else float32, else float64
/// (+-inf/nan always "fit" in float16). Verified against real numpy 2.5.1
/// by bisection -- **the boundaries are NOT the types' true finite MAX**
/// (float16 max is 65504.0, float32 max is ~3.4028235e38): numpy's actual
/// cutoffs are the round decimal constants 65000.0 and 3.4e38. E.g.
/// `np.min_scalar_type(65504.0)` (float16's own exact max) is `float32`,
/// and `np.min_scalar_type(64999.9)` is `float16` while `65000.0` is
/// `float32` -- confirmed by bisecting the exact flip point to
/// `65000.0` and `3.4e38` respectively. This is numpy's real, empirically-
/// measured behavior, not a "corrected" precision analysis -- do not
/// change these to the mathematically-obvious `f16::MAX`/`f32::MAX`
/// without re-measuring against real numpy first.
pub fn min_scalar_type_float(v: f64) -> DType {
    const F16_BOUND: f64 = 65000.0;
    const F32_BOUND: f64 = 3.4e38;
    if v.is_nan() || v.is_infinite() || v.abs() < F16_BOUND {
        DType::F16
    } else if v.abs() < F32_BOUND {
        DType::F32
    } else {
        DType::F64
    }
}

/// Smallest complex dtype that can hold `(re, im)`: complex64 if both parts
/// fit in the same rounded 3.4e38 bound `min_scalar_type_float` uses for
/// float32, else complex128. Verified against real numpy 2.5.1
/// (`min_scalar_type` never returns complex32 -- ionp has no such type
/// either, matching numpy exactly here).
pub fn min_scalar_type_complex(re: f64, im: f64) -> DType {
    const F32_BOUND: f64 = 3.4e38;
    let fits32 = |x: f64| x.is_nan() || x.is_infinite() || x.abs() < F32_BOUND;
    if fits32(re) && fits32(im) {
        DType::C64
    } else {
        DType::C128
    }
}

/// `np.finfo` machine parameters for a floating dtype. Every field is a
/// per-dtype IEEE-754 constant, hardcoded here and cross-checked against
/// real numpy 2.5.1's `np.finfo(float16|float32|float64)` output (sanctioned
/// per the mission: "these are constants per dtype -- derive them and
/// hardcode the derived table").
pub struct FInfo {
    pub eps: f64,
    pub epsneg: f64,
    pub max: f64,
    pub min: f64,
    pub tiny: f64,
    pub smallest_normal: f64,
    pub smallest_subnormal: f64,
    pub resolution: f64,
    pub precision: i32,
    pub bits: i32,
    pub iexp: i32,
    pub nexp: i32,
    pub nmant: i32,
    pub machep: i32,
    pub negep: i32,
    pub minexp: i32,
    pub maxexp: i32,
}

/// Returns `None` for non-floating dtypes (`np.finfo` on an integer dtype
/// raises `ValueError`; the PyO3 layer is responsible for turning `None`
/// into that exact exception).
pub fn finfo_for(dtype: DType) -> Option<FInfo> {
    match dtype {
        DType::F16 => Some(FInfo {
            // Every value here is the EXACT f64 widening of the true f16/f32
            // bit pattern (`half::f16::from_bits(...).to_f64()` /
            // `(x as f32) as f64`), NOT a short decimal literal rounded to
            // ~7 significant digits -- verified against real numpy 2.5.1's
            // `np.finfo(...).eps.item()` etc., which prints the FULL,
            // exact-widened decimal expansion (e.g. `eps ==
            // 0.0009765625`, not `0.000977`). A first pass at this table
            // used short literals and passed its own unit tests (which
            // compared against the same wrong literals) but silently
            // diverged from real numpy at 3+ significant digits of
            // precision -- caught only by a direct differential probe
            // against real numpy, not by the original unit tests. Do not
            // "simplify" these back to short decimals.
            eps: 0.0009765625_f64,
            epsneg: 0.00048828125_f64,
            max: 65504.0,
            min: -65504.0,
            tiny: 6.103515625e-05,
            smallest_normal: 6.103515625e-05,
            smallest_subnormal: 5.960464477539063e-08,
            resolution: 0.0010004043579101562,
            precision: 3,
            bits: 16,
            iexp: 5,
            nexp: 5,
            nmant: 10,
            machep: -10,
            negep: -11,
            minexp: -14,
            maxexp: 16,
        }),
        DType::F32 => Some(FInfo {
            // See the F16 arm's comment: exact f32->f64 widened values, not
            // short decimal literals.
            eps: 1.1920928955078125e-07,
            epsneg: 5.960464477539063e-08,
            max: 3.4028234663852886e+38,
            min: -3.4028234663852886e+38,
            tiny: 1.1754943508222875e-38,
            smallest_normal: 1.1754943508222875e-38,
            smallest_subnormal: 1.401298464324817e-45,
            resolution: 9.999999974752427e-07,
            precision: 6,
            bits: 32,
            iexp: 8,
            nexp: 8,
            nmant: 23,
            machep: -23,
            negep: -24,
            minexp: -126,
            maxexp: 128,
        }),
        DType::F64 => Some(FInfo {
            eps: 2.220446049250313e-16,
            epsneg: 1.1102230246251565e-16,
            max: 1.7976931348623157e+308,
            min: -1.7976931348623157e+308,
            tiny: 2.2250738585072014e-308,
            smallest_normal: 2.2250738585072014e-308,
            smallest_subnormal: 5e-324,
            resolution: 1e-15,
            precision: 15,
            bits: 64,
            iexp: 11,
            nexp: 11,
            nmant: 52,
            machep: -52,
            negep: -53,
            minexp: -1022,
            maxexp: 1024,
        }),
        _ => None,
    }
}

/// `np.iinfo` machine parameters for an integer dtype: just `min`/`max`/
/// `bits`, all exact -- no derived-constant subtlety the way `finfo` has.
pub struct IInfo {
    pub min: i128,
    pub max: i128,
    pub bits: i32,
}

/// Returns `None` for non-integer dtypes (`np.iinfo` on a float dtype raises
/// `ValueError`; PyO3 layer turns that into the exact exception/message).
pub fn iinfo_for(dtype: DType) -> Option<IInfo> {
    // Bool has `int_bounds() == Some((0, 1))` (needed for weak-scalar range
    // checks elsewhere in this file), but real numpy 2.5.1's `np.iinfo(bool)`
    // raises `ValueError: Invalid integer data type 'b'.` -- bool is
    // deliberately excluded here even though it's "integer-shaped" by
    // `int_bounds`'s own contract.
    if dtype == DType::Bool {
        return None;
    }
    let bounds = dtype.int_bounds()?;
    let bits = (dtype.itemsize() * 8) as i32;
    Some(IInfo { min: bounds.0, max: bounds.1, bits })
}

/// `np.typename(char)`'s exact lookup table (verified against real numpy
/// 2.5.1's source for `numpy._core.numerictypes.typename`, a bare dict
/// lookup keyed on typecode strings, value-independent and dtype-independent
/// -- it does not require an actual dtype to exist for a code, which is why
/// `'g'`/`'G'` (long double / complex long double, dtypes ionp does not
/// implement) are still valid keys: numpy's table has them regardless of
/// what ionp itself supports numerically). Unknown codes are `None`; the
/// PyO3 layer raises `KeyError` with `repr(char)` as its message, matching
/// numpy exactly (numpy's `typename` is a literal `dict.__getitem__`).
pub fn typename_for(code: &str) -> Option<&'static str> {
    Some(match code {
        "?" => "bool",
        "b" => "signed char",
        "B" => "unsigned char",
        "h" => "short",
        "H" => "unsigned short",
        "i" => "integer",
        "I" => "unsigned integer",
        "l" => "long integer",
        "L" => "unsigned long integer",
        "q" => "long long integer",
        "Q" => "unsigned long long integer",
        "f" => "single precision",
        "d" => "double precision",
        "g" => "long precision",
        "F" => "complex single precision",
        "D" => "complex double precision",
        "G" => "complex long double precision",
        "S1" => "character",
        "S" => "string",
        "U" => "unicode",
        "V" => "void",
        "O" => "object",
        _ => return None,
    })
}

/// `np.mintypecode(typechars, typeset='GDFgdf', default='d')`'s core
/// selection logic, given the already-collected list of single-character
/// typecodes (the PyO3 layer handles turning each element of `typechars`
/// into a typecode string, including the string-is-iterable-of-chars and
/// array_like-uses-`.dtype.char` cases numpy's Python implementation
/// special-cases). Verified against real numpy 2.5.1's
/// `numpy.lib._type_check_impl.mintypecode` source: intersect the given
/// codes with `typeset`; if empty, return `default`; the `'F' and 'd' ->
/// 'D'` special case predates and overrides the general
/// smallest-index-wins rule; otherwise pick the code with the smallest
/// index in the fixed precedence string `"GDFgdf"` (restricted to the
/// float/complex codes ionp's `typeset` default covers -- ionp does not
/// implement long double ('g'/'G'), so callers passing a non-default
/// `typeset` including them are the PyO3 layer's responsibility to reject
/// or pass through untouched, not this function's).
pub fn mintypecode_select(typecodes: &[&str], typeset: &str, default: &str) -> String {
    const PRECEDENCE: &str = "GDFgdf";
    let intersection: Vec<&str> = typecodes
        .iter()
        .copied()
        .filter(|t| typeset.contains(t))
        .collect();
    if intersection.is_empty() {
        return default.to_string();
    }
    if intersection.contains(&"F") && intersection.contains(&"d") {
        return "D".to_string();
    }
    intersection
        .into_iter()
        .min_by_key(|t| PRECEDENCE.find(t).unwrap_or(usize::MAX))
        .unwrap_or(default)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use DType::*;

    #[test]
    fn identity_promotion() {
        for d in [Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, C64, C128] {
            assert_eq!(promote_dtype(d, d), d);
        }
    }

    #[test]
    fn bool_is_absorbed() {
        assert_eq!(promote_dtype(Bool, I32), I32);
        assert_eq!(promote_dtype(F64, Bool), F64);
        assert_eq!(promote_dtype(Bool, C64), C64);
    }

    #[test]
    fn same_kind_int_widens() {
        assert_eq!(promote_dtype(I8, I32), I32);
        assert_eq!(promote_dtype(U16, U8), U16);
    }

    #[test]
    fn int_uint_cross_table_matches_numpy() {
        // np.promote_types('int8','uint8')  == int16
        assert_eq!(promote_dtype(I8, U8), I16);
        // np.promote_types('int16','uint8') == int16
        assert_eq!(promote_dtype(I16, U8), I16);
        // np.promote_types('int32','uint16') == int32
        assert_eq!(promote_dtype(I32, U16), I32);
        // np.promote_types('int8','uint32')  == int64
        assert_eq!(promote_dtype(I8, U32), I64);
        // np.promote_types('int64','uint64') == float64  (the famous one)
        assert_eq!(promote_dtype(I64, U64), F64);
        assert_eq!(promote_dtype(U64, I64), F64);
        // np.promote_types('int32','uint64')  == float64
        assert_eq!(promote_dtype(I32, U64), F64);
    }

    #[test]
    fn int_float_promotion_matches_numpy() {
        // np.promote_types('int8','float32')  == float32
        assert_eq!(promote_dtype(I8, F32), F32);
        // np.promote_types('int32','float32')  == float64
        assert_eq!(promote_dtype(I32, F32), F64);
        // np.promote_types('int64','float32')  == float64
        assert_eq!(promote_dtype(I64, F32), F64);
        // np.promote_types('int16','float64')  == float64
        assert_eq!(promote_dtype(I16, F64), F64);
    }

    #[test]
    fn float16_promotion_matches_numpy() {
        // Verified against real numpy 2.5.1 (`np.promote_types`):
        // bool/int8/uint8 fit float16's significand exactly and promote
        // there; int16/uint16 need float32; int32/int64/uint32/uint64 need
        // float64.
        assert_eq!(promote_dtype(Bool, F16), F16);
        assert_eq!(promote_dtype(I8, F16), F16);
        assert_eq!(promote_dtype(U8, F16), F16);
        assert_eq!(promote_dtype(I16, F16), F32);
        assert_eq!(promote_dtype(U16, F16), F32);
        assert_eq!(promote_dtype(I32, F16), F64);
        assert_eq!(promote_dtype(I64, F16), F64);
        assert_eq!(promote_dtype(U32, F16), F64);
        assert_eq!(promote_dtype(U64, F16), F64);
        // np.promote_types('float16','float32') == float32
        assert_eq!(promote_dtype(F16, F32), F32);
        // np.promote_types('float16','float64') == float64
        assert_eq!(promote_dtype(F16, F64), F64);
        // np.promote_types('float16','complex64')  == complex64
        assert_eq!(promote_dtype(F16, C64), C64);
        // np.promote_types('float16','complex128') == complex128
        assert_eq!(promote_dtype(F16, C128), C128);
        // int8/float32 unaffected by the int_equiv_float_bits fix (still F32)
        assert_eq!(promote_dtype(I8, F32), F32);
    }

    #[test]
    fn float_complex_promotion_matches_numpy() {
        // np.promote_types('float32','complex64')   == complex64
        assert_eq!(promote_dtype(F32, C64), C64);
        // np.promote_types('float64','complex64')   == complex128
        assert_eq!(promote_dtype(F64, C64), C128);
        // np.promote_types('float32','complex128')  == complex128
        assert_eq!(promote_dtype(F32, C128), C128);
        assert_eq!(promote_dtype(C64, C128), C128);
    }

    #[test]
    fn int_complex_promotion_matches_numpy() {
        // np.promote_types('int8','complex64')   == complex64
        assert_eq!(promote_dtype(I8, C64), C64);
        // np.promote_types('int64','complex64')  == complex128
        assert_eq!(promote_dtype(I64, C64), C128);
    }

    #[test]
    fn promotion_is_commutative_exhaustive() {
        let all = [Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, C64, C128];
        for &a in &all {
            for &b in &all {
                assert_eq!(promote_dtype(a, b), promote_dtype(b, a), "{a} vs {b} not commutative");
            }
        }
    }

    #[test]
    fn int_bounds_matches_rust_native_ranges() {
        assert_eq!(I8.int_bounds(), Some((-128, 127)));
        assert_eq!(U8.int_bounds(), Some((0, 255)));
        assert_eq!(Bool.int_bounds(), Some((0, 1)));
        assert_eq!(I64.int_bounds(), Some((i64::MIN as i128, i64::MAX as i128)));
        assert_eq!(U64.int_bounds(), Some((0, u64::MAX as i128)));
        assert_eq!(F32.int_bounds(), None);
        assert_eq!(F64.int_bounds(), None);
        assert_eq!(C64.int_bounds(), None);
        assert_eq!(C128.int_bounds(), None);
    }

    #[test]
    fn kind_rank_orders_bool_int_float_complex() {
        assert_eq!(Bool.kind_rank(), 0);
        for d in [I8, I16, I32, I64, U8, U16, U32, U64] {
            assert_eq!(d.kind_rank(), 1, "{d}");
        }
        for d in [F16, F32, F64] {
            assert_eq!(d.kind_rank(), 2, "{d}");
        }
        for d in [C64, C128] {
            assert_eq!(d.kind_rank(), 3, "{d}");
        }
    }

    #[test]
    fn weak_scalar_rank_le_array_rank_never_widens() {
        // np.array([1], dtype=np.int8) + 1000  -> still int8 (value out of
        // range is a separate OverflowError concern, covered by ionp-py's
        // own check_int_bounds test; this test is purely about dtype).
        assert_eq!(weak_target_dtype(I8, ScalarKind::Int), I8);
        // np.array([1,2,3], dtype=np.float32) + 1.0  -> stays float32.
        assert_eq!(weak_target_dtype(F32, ScalarKind::Float), F32);
        // np.array([1], dtype=np.uint8) + True  -> stays uint8 (bool is
        // weaker than every array kind).
        assert_eq!(weak_target_dtype(U8, ScalarKind::Bool), U8);
        assert_eq!(weak_target_dtype(C64, ScalarKind::Bool), C64);
        assert_eq!(weak_target_dtype(C128, ScalarKind::Float), C128);
        // int array + int scalar: still weak, stays put.
        assert_eq!(weak_target_dtype(I16, ScalarKind::Int), I16);
        assert_eq!(weak_target_dtype(U64, ScalarKind::Int), U64);
    }

    #[test]
    fn weak_scalar_rank_gt_array_rank_promotes_to_kind_default() {
        // np.array([1], dtype=np.int8) + 1.0  -> float64 (int array, float
        // scalar: scalar rank 2 > array rank 1 -> kind-default float64).
        assert_eq!(weak_target_dtype(I8, ScalarKind::Float), F64);
        assert_eq!(weak_target_dtype(U32, ScalarKind::Float), F64);
        // np.array([True]) + 1  -> int64 (bool array, int scalar).
        assert_eq!(weak_target_dtype(Bool, ScalarKind::Int), I64);
        // np.array([True]) + 1.0  -> float64.
        assert_eq!(weak_target_dtype(Bool, ScalarKind::Float), F64);
        // np.array([1], dtype=np.int8) + 1j  -> complex128 (int array,
        // complex scalar: no width-match special case for int arrays).
        assert_eq!(weak_target_dtype(I8, ScalarKind::Complex), C128);
        assert_eq!(weak_target_dtype(Bool, ScalarKind::Complex), C128);
    }

    #[test]
    fn weak_complex_scalar_against_float_array_width_matches_not_default() {
        // np.array([1], dtype=np.float32) + 1j  -> complex64, NOT the
        // kind-default complex128 -- the one special case in NEP 50 weak
        // promotion, verified against real numpy 2.5.1.
        assert_eq!(weak_target_dtype(F32, ScalarKind::Complex), C64);
        // np.array([1], dtype=np.float64) + 1j  -> complex128 (which is
        // also the kind-default here, so this doesn't by itself prove the
        // width-match rule, but confirms the other width is still right).
        assert_eq!(weak_target_dtype(F64, ScalarKind::Complex), C128);
        // np.array([1], dtype=np.float16) + 1j  -> complex64, same
        // width-match special case as float32 (verified against numpy
        // 2.5.1).
        assert_eq!(weak_target_dtype(F16, ScalarKind::Complex), C64);
    }

    #[test]
    fn can_cast_matches_numpy_2_5_1_exhaustively() {
        // Exhaustive ground truth: `np.can_cast(from, to, casting=mode)`
        // for all 14 dtypes x 14 dtypes x 5 modes (980 checks), captured
        // live against real numpy 2.5.1 and pasted here verbatim -- this
        // is the actual verification behind `SAFE_CAST`/`same_kind_extra`
        // above, not just a handful of hand-picked spot checks. If this
        // test ever needs to change, it means either numpy's table
        // changed or the capture above was wrong -- re-derive from numpy,
        // do not edit an assertion to make it pass.
        assert_eq!(can_cast(Bool, Bool, "no"), true);
        assert_eq!(can_cast(Bool, I8, "no"), false);
        assert_eq!(can_cast(Bool, I16, "no"), false);
        assert_eq!(can_cast(Bool, I32, "no"), false);
        assert_eq!(can_cast(Bool, I64, "no"), false);
        assert_eq!(can_cast(Bool, U8, "no"), false);
        assert_eq!(can_cast(Bool, U16, "no"), false);
        assert_eq!(can_cast(Bool, U32, "no"), false);
        assert_eq!(can_cast(Bool, U64, "no"), false);
        assert_eq!(can_cast(Bool, F16, "no"), false);
        assert_eq!(can_cast(Bool, F32, "no"), false);
        assert_eq!(can_cast(Bool, F64, "no"), false);
        assert_eq!(can_cast(Bool, C64, "no"), false);
        assert_eq!(can_cast(Bool, C128, "no"), false);
        assert_eq!(can_cast(I8, Bool, "no"), false);
        assert_eq!(can_cast(I8, I8, "no"), true);
        assert_eq!(can_cast(I8, I16, "no"), false);
        assert_eq!(can_cast(I8, I32, "no"), false);
        assert_eq!(can_cast(I8, I64, "no"), false);
        assert_eq!(can_cast(I8, U8, "no"), false);
        assert_eq!(can_cast(I8, U16, "no"), false);
        assert_eq!(can_cast(I8, U32, "no"), false);
        assert_eq!(can_cast(I8, U64, "no"), false);
        assert_eq!(can_cast(I8, F16, "no"), false);
        assert_eq!(can_cast(I8, F32, "no"), false);
        assert_eq!(can_cast(I8, F64, "no"), false);
        assert_eq!(can_cast(I8, C64, "no"), false);
        assert_eq!(can_cast(I8, C128, "no"), false);
        assert_eq!(can_cast(I16, Bool, "no"), false);
        assert_eq!(can_cast(I16, I8, "no"), false);
        assert_eq!(can_cast(I16, I16, "no"), true);
        assert_eq!(can_cast(I16, I32, "no"), false);
        assert_eq!(can_cast(I16, I64, "no"), false);
        assert_eq!(can_cast(I16, U8, "no"), false);
        assert_eq!(can_cast(I16, U16, "no"), false);
        assert_eq!(can_cast(I16, U32, "no"), false);
        assert_eq!(can_cast(I16, U64, "no"), false);
        assert_eq!(can_cast(I16, F16, "no"), false);
        assert_eq!(can_cast(I16, F32, "no"), false);
        assert_eq!(can_cast(I16, F64, "no"), false);
        assert_eq!(can_cast(I16, C64, "no"), false);
        assert_eq!(can_cast(I16, C128, "no"), false);
        assert_eq!(can_cast(I32, Bool, "no"), false);
        assert_eq!(can_cast(I32, I8, "no"), false);
        assert_eq!(can_cast(I32, I16, "no"), false);
        assert_eq!(can_cast(I32, I32, "no"), true);
        assert_eq!(can_cast(I32, I64, "no"), false);
        assert_eq!(can_cast(I32, U8, "no"), false);
        assert_eq!(can_cast(I32, U16, "no"), false);
        assert_eq!(can_cast(I32, U32, "no"), false);
        assert_eq!(can_cast(I32, U64, "no"), false);
        assert_eq!(can_cast(I32, F16, "no"), false);
        assert_eq!(can_cast(I32, F32, "no"), false);
        assert_eq!(can_cast(I32, F64, "no"), false);
        assert_eq!(can_cast(I32, C64, "no"), false);
        assert_eq!(can_cast(I32, C128, "no"), false);
        assert_eq!(can_cast(I64, Bool, "no"), false);
        assert_eq!(can_cast(I64, I8, "no"), false);
        assert_eq!(can_cast(I64, I16, "no"), false);
        assert_eq!(can_cast(I64, I32, "no"), false);
        assert_eq!(can_cast(I64, I64, "no"), true);
        assert_eq!(can_cast(I64, U8, "no"), false);
        assert_eq!(can_cast(I64, U16, "no"), false);
        assert_eq!(can_cast(I64, U32, "no"), false);
        assert_eq!(can_cast(I64, U64, "no"), false);
        assert_eq!(can_cast(I64, F16, "no"), false);
        assert_eq!(can_cast(I64, F32, "no"), false);
        assert_eq!(can_cast(I64, F64, "no"), false);
        assert_eq!(can_cast(I64, C64, "no"), false);
        assert_eq!(can_cast(I64, C128, "no"), false);
        assert_eq!(can_cast(U8, Bool, "no"), false);
        assert_eq!(can_cast(U8, I8, "no"), false);
        assert_eq!(can_cast(U8, I16, "no"), false);
        assert_eq!(can_cast(U8, I32, "no"), false);
        assert_eq!(can_cast(U8, I64, "no"), false);
        assert_eq!(can_cast(U8, U8, "no"), true);
        assert_eq!(can_cast(U8, U16, "no"), false);
        assert_eq!(can_cast(U8, U32, "no"), false);
        assert_eq!(can_cast(U8, U64, "no"), false);
        assert_eq!(can_cast(U8, F16, "no"), false);
        assert_eq!(can_cast(U8, F32, "no"), false);
        assert_eq!(can_cast(U8, F64, "no"), false);
        assert_eq!(can_cast(U8, C64, "no"), false);
        assert_eq!(can_cast(U8, C128, "no"), false);
        assert_eq!(can_cast(U16, Bool, "no"), false);
        assert_eq!(can_cast(U16, I8, "no"), false);
        assert_eq!(can_cast(U16, I16, "no"), false);
        assert_eq!(can_cast(U16, I32, "no"), false);
        assert_eq!(can_cast(U16, I64, "no"), false);
        assert_eq!(can_cast(U16, U8, "no"), false);
        assert_eq!(can_cast(U16, U16, "no"), true);
        assert_eq!(can_cast(U16, U32, "no"), false);
        assert_eq!(can_cast(U16, U64, "no"), false);
        assert_eq!(can_cast(U16, F16, "no"), false);
        assert_eq!(can_cast(U16, F32, "no"), false);
        assert_eq!(can_cast(U16, F64, "no"), false);
        assert_eq!(can_cast(U16, C64, "no"), false);
        assert_eq!(can_cast(U16, C128, "no"), false);
        assert_eq!(can_cast(U32, Bool, "no"), false);
        assert_eq!(can_cast(U32, I8, "no"), false);
        assert_eq!(can_cast(U32, I16, "no"), false);
        assert_eq!(can_cast(U32, I32, "no"), false);
        assert_eq!(can_cast(U32, I64, "no"), false);
        assert_eq!(can_cast(U32, U8, "no"), false);
        assert_eq!(can_cast(U32, U16, "no"), false);
        assert_eq!(can_cast(U32, U32, "no"), true);
        assert_eq!(can_cast(U32, U64, "no"), false);
        assert_eq!(can_cast(U32, F16, "no"), false);
        assert_eq!(can_cast(U32, F32, "no"), false);
        assert_eq!(can_cast(U32, F64, "no"), false);
        assert_eq!(can_cast(U32, C64, "no"), false);
        assert_eq!(can_cast(U32, C128, "no"), false);
        assert_eq!(can_cast(U64, Bool, "no"), false);
        assert_eq!(can_cast(U64, I8, "no"), false);
        assert_eq!(can_cast(U64, I16, "no"), false);
        assert_eq!(can_cast(U64, I32, "no"), false);
        assert_eq!(can_cast(U64, I64, "no"), false);
        assert_eq!(can_cast(U64, U8, "no"), false);
        assert_eq!(can_cast(U64, U16, "no"), false);
        assert_eq!(can_cast(U64, U32, "no"), false);
        assert_eq!(can_cast(U64, U64, "no"), true);
        assert_eq!(can_cast(U64, F16, "no"), false);
        assert_eq!(can_cast(U64, F32, "no"), false);
        assert_eq!(can_cast(U64, F64, "no"), false);
        assert_eq!(can_cast(U64, C64, "no"), false);
        assert_eq!(can_cast(U64, C128, "no"), false);
        assert_eq!(can_cast(F16, Bool, "no"), false);
        assert_eq!(can_cast(F16, I8, "no"), false);
        assert_eq!(can_cast(F16, I16, "no"), false);
        assert_eq!(can_cast(F16, I32, "no"), false);
        assert_eq!(can_cast(F16, I64, "no"), false);
        assert_eq!(can_cast(F16, U8, "no"), false);
        assert_eq!(can_cast(F16, U16, "no"), false);
        assert_eq!(can_cast(F16, U32, "no"), false);
        assert_eq!(can_cast(F16, U64, "no"), false);
        assert_eq!(can_cast(F16, F16, "no"), true);
        assert_eq!(can_cast(F16, F32, "no"), false);
        assert_eq!(can_cast(F16, F64, "no"), false);
        assert_eq!(can_cast(F16, C64, "no"), false);
        assert_eq!(can_cast(F16, C128, "no"), false);
        assert_eq!(can_cast(F32, Bool, "no"), false);
        assert_eq!(can_cast(F32, I8, "no"), false);
        assert_eq!(can_cast(F32, I16, "no"), false);
        assert_eq!(can_cast(F32, I32, "no"), false);
        assert_eq!(can_cast(F32, I64, "no"), false);
        assert_eq!(can_cast(F32, U8, "no"), false);
        assert_eq!(can_cast(F32, U16, "no"), false);
        assert_eq!(can_cast(F32, U32, "no"), false);
        assert_eq!(can_cast(F32, U64, "no"), false);
        assert_eq!(can_cast(F32, F16, "no"), false);
        assert_eq!(can_cast(F32, F32, "no"), true);
        assert_eq!(can_cast(F32, F64, "no"), false);
        assert_eq!(can_cast(F32, C64, "no"), false);
        assert_eq!(can_cast(F32, C128, "no"), false);
        assert_eq!(can_cast(F64, Bool, "no"), false);
        assert_eq!(can_cast(F64, I8, "no"), false);
        assert_eq!(can_cast(F64, I16, "no"), false);
        assert_eq!(can_cast(F64, I32, "no"), false);
        assert_eq!(can_cast(F64, I64, "no"), false);
        assert_eq!(can_cast(F64, U8, "no"), false);
        assert_eq!(can_cast(F64, U16, "no"), false);
        assert_eq!(can_cast(F64, U32, "no"), false);
        assert_eq!(can_cast(F64, U64, "no"), false);
        assert_eq!(can_cast(F64, F16, "no"), false);
        assert_eq!(can_cast(F64, F32, "no"), false);
        assert_eq!(can_cast(F64, F64, "no"), true);
        assert_eq!(can_cast(F64, C64, "no"), false);
        assert_eq!(can_cast(F64, C128, "no"), false);
        assert_eq!(can_cast(C64, Bool, "no"), false);
        assert_eq!(can_cast(C64, I8, "no"), false);
        assert_eq!(can_cast(C64, I16, "no"), false);
        assert_eq!(can_cast(C64, I32, "no"), false);
        assert_eq!(can_cast(C64, I64, "no"), false);
        assert_eq!(can_cast(C64, U8, "no"), false);
        assert_eq!(can_cast(C64, U16, "no"), false);
        assert_eq!(can_cast(C64, U32, "no"), false);
        assert_eq!(can_cast(C64, U64, "no"), false);
        assert_eq!(can_cast(C64, F16, "no"), false);
        assert_eq!(can_cast(C64, F32, "no"), false);
        assert_eq!(can_cast(C64, F64, "no"), false);
        assert_eq!(can_cast(C64, C64, "no"), true);
        assert_eq!(can_cast(C64, C128, "no"), false);
        assert_eq!(can_cast(C128, Bool, "no"), false);
        assert_eq!(can_cast(C128, I8, "no"), false);
        assert_eq!(can_cast(C128, I16, "no"), false);
        assert_eq!(can_cast(C128, I32, "no"), false);
        assert_eq!(can_cast(C128, I64, "no"), false);
        assert_eq!(can_cast(C128, U8, "no"), false);
        assert_eq!(can_cast(C128, U16, "no"), false);
        assert_eq!(can_cast(C128, U32, "no"), false);
        assert_eq!(can_cast(C128, U64, "no"), false);
        assert_eq!(can_cast(C128, F16, "no"), false);
        assert_eq!(can_cast(C128, F32, "no"), false);
        assert_eq!(can_cast(C128, F64, "no"), false);
        assert_eq!(can_cast(C128, C64, "no"), false);
        assert_eq!(can_cast(C128, C128, "no"), true);
        assert_eq!(can_cast(Bool, Bool, "equiv"), true);
        assert_eq!(can_cast(Bool, I8, "equiv"), false);
        assert_eq!(can_cast(Bool, I16, "equiv"), false);
        assert_eq!(can_cast(Bool, I32, "equiv"), false);
        assert_eq!(can_cast(Bool, I64, "equiv"), false);
        assert_eq!(can_cast(Bool, U8, "equiv"), false);
        assert_eq!(can_cast(Bool, U16, "equiv"), false);
        assert_eq!(can_cast(Bool, U32, "equiv"), false);
        assert_eq!(can_cast(Bool, U64, "equiv"), false);
        assert_eq!(can_cast(Bool, F16, "equiv"), false);
        assert_eq!(can_cast(Bool, F32, "equiv"), false);
        assert_eq!(can_cast(Bool, F64, "equiv"), false);
        assert_eq!(can_cast(Bool, C64, "equiv"), false);
        assert_eq!(can_cast(Bool, C128, "equiv"), false);
        assert_eq!(can_cast(I8, Bool, "equiv"), false);
        assert_eq!(can_cast(I8, I8, "equiv"), true);
        assert_eq!(can_cast(I8, I16, "equiv"), false);
        assert_eq!(can_cast(I8, I32, "equiv"), false);
        assert_eq!(can_cast(I8, I64, "equiv"), false);
        assert_eq!(can_cast(I8, U8, "equiv"), false);
        assert_eq!(can_cast(I8, U16, "equiv"), false);
        assert_eq!(can_cast(I8, U32, "equiv"), false);
        assert_eq!(can_cast(I8, U64, "equiv"), false);
        assert_eq!(can_cast(I8, F16, "equiv"), false);
        assert_eq!(can_cast(I8, F32, "equiv"), false);
        assert_eq!(can_cast(I8, F64, "equiv"), false);
        assert_eq!(can_cast(I8, C64, "equiv"), false);
        assert_eq!(can_cast(I8, C128, "equiv"), false);
        assert_eq!(can_cast(I16, Bool, "equiv"), false);
        assert_eq!(can_cast(I16, I8, "equiv"), false);
        assert_eq!(can_cast(I16, I16, "equiv"), true);
        assert_eq!(can_cast(I16, I32, "equiv"), false);
        assert_eq!(can_cast(I16, I64, "equiv"), false);
        assert_eq!(can_cast(I16, U8, "equiv"), false);
        assert_eq!(can_cast(I16, U16, "equiv"), false);
        assert_eq!(can_cast(I16, U32, "equiv"), false);
        assert_eq!(can_cast(I16, U64, "equiv"), false);
        assert_eq!(can_cast(I16, F16, "equiv"), false);
        assert_eq!(can_cast(I16, F32, "equiv"), false);
        assert_eq!(can_cast(I16, F64, "equiv"), false);
        assert_eq!(can_cast(I16, C64, "equiv"), false);
        assert_eq!(can_cast(I16, C128, "equiv"), false);
        assert_eq!(can_cast(I32, Bool, "equiv"), false);
        assert_eq!(can_cast(I32, I8, "equiv"), false);
        assert_eq!(can_cast(I32, I16, "equiv"), false);
        assert_eq!(can_cast(I32, I32, "equiv"), true);
        assert_eq!(can_cast(I32, I64, "equiv"), false);
        assert_eq!(can_cast(I32, U8, "equiv"), false);
        assert_eq!(can_cast(I32, U16, "equiv"), false);
        assert_eq!(can_cast(I32, U32, "equiv"), false);
        assert_eq!(can_cast(I32, U64, "equiv"), false);
        assert_eq!(can_cast(I32, F16, "equiv"), false);
        assert_eq!(can_cast(I32, F32, "equiv"), false);
        assert_eq!(can_cast(I32, F64, "equiv"), false);
        assert_eq!(can_cast(I32, C64, "equiv"), false);
        assert_eq!(can_cast(I32, C128, "equiv"), false);
        assert_eq!(can_cast(I64, Bool, "equiv"), false);
        assert_eq!(can_cast(I64, I8, "equiv"), false);
        assert_eq!(can_cast(I64, I16, "equiv"), false);
        assert_eq!(can_cast(I64, I32, "equiv"), false);
        assert_eq!(can_cast(I64, I64, "equiv"), true);
        assert_eq!(can_cast(I64, U8, "equiv"), false);
        assert_eq!(can_cast(I64, U16, "equiv"), false);
        assert_eq!(can_cast(I64, U32, "equiv"), false);
        assert_eq!(can_cast(I64, U64, "equiv"), false);
        assert_eq!(can_cast(I64, F16, "equiv"), false);
        assert_eq!(can_cast(I64, F32, "equiv"), false);
        assert_eq!(can_cast(I64, F64, "equiv"), false);
        assert_eq!(can_cast(I64, C64, "equiv"), false);
        assert_eq!(can_cast(I64, C128, "equiv"), false);
        assert_eq!(can_cast(U8, Bool, "equiv"), false);
        assert_eq!(can_cast(U8, I8, "equiv"), false);
        assert_eq!(can_cast(U8, I16, "equiv"), false);
        assert_eq!(can_cast(U8, I32, "equiv"), false);
        assert_eq!(can_cast(U8, I64, "equiv"), false);
        assert_eq!(can_cast(U8, U8, "equiv"), true);
        assert_eq!(can_cast(U8, U16, "equiv"), false);
        assert_eq!(can_cast(U8, U32, "equiv"), false);
        assert_eq!(can_cast(U8, U64, "equiv"), false);
        assert_eq!(can_cast(U8, F16, "equiv"), false);
        assert_eq!(can_cast(U8, F32, "equiv"), false);
        assert_eq!(can_cast(U8, F64, "equiv"), false);
        assert_eq!(can_cast(U8, C64, "equiv"), false);
        assert_eq!(can_cast(U8, C128, "equiv"), false);
        assert_eq!(can_cast(U16, Bool, "equiv"), false);
        assert_eq!(can_cast(U16, I8, "equiv"), false);
        assert_eq!(can_cast(U16, I16, "equiv"), false);
        assert_eq!(can_cast(U16, I32, "equiv"), false);
        assert_eq!(can_cast(U16, I64, "equiv"), false);
        assert_eq!(can_cast(U16, U8, "equiv"), false);
        assert_eq!(can_cast(U16, U16, "equiv"), true);
        assert_eq!(can_cast(U16, U32, "equiv"), false);
        assert_eq!(can_cast(U16, U64, "equiv"), false);
        assert_eq!(can_cast(U16, F16, "equiv"), false);
        assert_eq!(can_cast(U16, F32, "equiv"), false);
        assert_eq!(can_cast(U16, F64, "equiv"), false);
        assert_eq!(can_cast(U16, C64, "equiv"), false);
        assert_eq!(can_cast(U16, C128, "equiv"), false);
        assert_eq!(can_cast(U32, Bool, "equiv"), false);
        assert_eq!(can_cast(U32, I8, "equiv"), false);
        assert_eq!(can_cast(U32, I16, "equiv"), false);
        assert_eq!(can_cast(U32, I32, "equiv"), false);
        assert_eq!(can_cast(U32, I64, "equiv"), false);
        assert_eq!(can_cast(U32, U8, "equiv"), false);
        assert_eq!(can_cast(U32, U16, "equiv"), false);
        assert_eq!(can_cast(U32, U32, "equiv"), true);
        assert_eq!(can_cast(U32, U64, "equiv"), false);
        assert_eq!(can_cast(U32, F16, "equiv"), false);
        assert_eq!(can_cast(U32, F32, "equiv"), false);
        assert_eq!(can_cast(U32, F64, "equiv"), false);
        assert_eq!(can_cast(U32, C64, "equiv"), false);
        assert_eq!(can_cast(U32, C128, "equiv"), false);
        assert_eq!(can_cast(U64, Bool, "equiv"), false);
        assert_eq!(can_cast(U64, I8, "equiv"), false);
        assert_eq!(can_cast(U64, I16, "equiv"), false);
        assert_eq!(can_cast(U64, I32, "equiv"), false);
        assert_eq!(can_cast(U64, I64, "equiv"), false);
        assert_eq!(can_cast(U64, U8, "equiv"), false);
        assert_eq!(can_cast(U64, U16, "equiv"), false);
        assert_eq!(can_cast(U64, U32, "equiv"), false);
        assert_eq!(can_cast(U64, U64, "equiv"), true);
        assert_eq!(can_cast(U64, F16, "equiv"), false);
        assert_eq!(can_cast(U64, F32, "equiv"), false);
        assert_eq!(can_cast(U64, F64, "equiv"), false);
        assert_eq!(can_cast(U64, C64, "equiv"), false);
        assert_eq!(can_cast(U64, C128, "equiv"), false);
        assert_eq!(can_cast(F16, Bool, "equiv"), false);
        assert_eq!(can_cast(F16, I8, "equiv"), false);
        assert_eq!(can_cast(F16, I16, "equiv"), false);
        assert_eq!(can_cast(F16, I32, "equiv"), false);
        assert_eq!(can_cast(F16, I64, "equiv"), false);
        assert_eq!(can_cast(F16, U8, "equiv"), false);
        assert_eq!(can_cast(F16, U16, "equiv"), false);
        assert_eq!(can_cast(F16, U32, "equiv"), false);
        assert_eq!(can_cast(F16, U64, "equiv"), false);
        assert_eq!(can_cast(F16, F16, "equiv"), true);
        assert_eq!(can_cast(F16, F32, "equiv"), false);
        assert_eq!(can_cast(F16, F64, "equiv"), false);
        assert_eq!(can_cast(F16, C64, "equiv"), false);
        assert_eq!(can_cast(F16, C128, "equiv"), false);
        assert_eq!(can_cast(F32, Bool, "equiv"), false);
        assert_eq!(can_cast(F32, I8, "equiv"), false);
        assert_eq!(can_cast(F32, I16, "equiv"), false);
        assert_eq!(can_cast(F32, I32, "equiv"), false);
        assert_eq!(can_cast(F32, I64, "equiv"), false);
        assert_eq!(can_cast(F32, U8, "equiv"), false);
        assert_eq!(can_cast(F32, U16, "equiv"), false);
        assert_eq!(can_cast(F32, U32, "equiv"), false);
        assert_eq!(can_cast(F32, U64, "equiv"), false);
        assert_eq!(can_cast(F32, F16, "equiv"), false);
        assert_eq!(can_cast(F32, F32, "equiv"), true);
        assert_eq!(can_cast(F32, F64, "equiv"), false);
        assert_eq!(can_cast(F32, C64, "equiv"), false);
        assert_eq!(can_cast(F32, C128, "equiv"), false);
        assert_eq!(can_cast(F64, Bool, "equiv"), false);
        assert_eq!(can_cast(F64, I8, "equiv"), false);
        assert_eq!(can_cast(F64, I16, "equiv"), false);
        assert_eq!(can_cast(F64, I32, "equiv"), false);
        assert_eq!(can_cast(F64, I64, "equiv"), false);
        assert_eq!(can_cast(F64, U8, "equiv"), false);
        assert_eq!(can_cast(F64, U16, "equiv"), false);
        assert_eq!(can_cast(F64, U32, "equiv"), false);
        assert_eq!(can_cast(F64, U64, "equiv"), false);
        assert_eq!(can_cast(F64, F16, "equiv"), false);
        assert_eq!(can_cast(F64, F32, "equiv"), false);
        assert_eq!(can_cast(F64, F64, "equiv"), true);
        assert_eq!(can_cast(F64, C64, "equiv"), false);
        assert_eq!(can_cast(F64, C128, "equiv"), false);
        assert_eq!(can_cast(C64, Bool, "equiv"), false);
        assert_eq!(can_cast(C64, I8, "equiv"), false);
        assert_eq!(can_cast(C64, I16, "equiv"), false);
        assert_eq!(can_cast(C64, I32, "equiv"), false);
        assert_eq!(can_cast(C64, I64, "equiv"), false);
        assert_eq!(can_cast(C64, U8, "equiv"), false);
        assert_eq!(can_cast(C64, U16, "equiv"), false);
        assert_eq!(can_cast(C64, U32, "equiv"), false);
        assert_eq!(can_cast(C64, U64, "equiv"), false);
        assert_eq!(can_cast(C64, F16, "equiv"), false);
        assert_eq!(can_cast(C64, F32, "equiv"), false);
        assert_eq!(can_cast(C64, F64, "equiv"), false);
        assert_eq!(can_cast(C64, C64, "equiv"), true);
        assert_eq!(can_cast(C64, C128, "equiv"), false);
        assert_eq!(can_cast(C128, Bool, "equiv"), false);
        assert_eq!(can_cast(C128, I8, "equiv"), false);
        assert_eq!(can_cast(C128, I16, "equiv"), false);
        assert_eq!(can_cast(C128, I32, "equiv"), false);
        assert_eq!(can_cast(C128, I64, "equiv"), false);
        assert_eq!(can_cast(C128, U8, "equiv"), false);
        assert_eq!(can_cast(C128, U16, "equiv"), false);
        assert_eq!(can_cast(C128, U32, "equiv"), false);
        assert_eq!(can_cast(C128, U64, "equiv"), false);
        assert_eq!(can_cast(C128, F16, "equiv"), false);
        assert_eq!(can_cast(C128, F32, "equiv"), false);
        assert_eq!(can_cast(C128, F64, "equiv"), false);
        assert_eq!(can_cast(C128, C64, "equiv"), false);
        assert_eq!(can_cast(C128, C128, "equiv"), true);
        assert_eq!(can_cast(Bool, Bool, "safe"), true);
        assert_eq!(can_cast(Bool, I8, "safe"), true);
        assert_eq!(can_cast(Bool, I16, "safe"), true);
        assert_eq!(can_cast(Bool, I32, "safe"), true);
        assert_eq!(can_cast(Bool, I64, "safe"), true);
        assert_eq!(can_cast(Bool, U8, "safe"), true);
        assert_eq!(can_cast(Bool, U16, "safe"), true);
        assert_eq!(can_cast(Bool, U32, "safe"), true);
        assert_eq!(can_cast(Bool, U64, "safe"), true);
        assert_eq!(can_cast(Bool, F16, "safe"), true);
        assert_eq!(can_cast(Bool, F32, "safe"), true);
        assert_eq!(can_cast(Bool, F64, "safe"), true);
        assert_eq!(can_cast(Bool, C64, "safe"), true);
        assert_eq!(can_cast(Bool, C128, "safe"), true);
        assert_eq!(can_cast(I8, Bool, "safe"), false);
        assert_eq!(can_cast(I8, I8, "safe"), true);
        assert_eq!(can_cast(I8, I16, "safe"), true);
        assert_eq!(can_cast(I8, I32, "safe"), true);
        assert_eq!(can_cast(I8, I64, "safe"), true);
        assert_eq!(can_cast(I8, U8, "safe"), false);
        assert_eq!(can_cast(I8, U16, "safe"), false);
        assert_eq!(can_cast(I8, U32, "safe"), false);
        assert_eq!(can_cast(I8, U64, "safe"), false);
        assert_eq!(can_cast(I8, F16, "safe"), true);
        assert_eq!(can_cast(I8, F32, "safe"), true);
        assert_eq!(can_cast(I8, F64, "safe"), true);
        assert_eq!(can_cast(I8, C64, "safe"), true);
        assert_eq!(can_cast(I8, C128, "safe"), true);
        assert_eq!(can_cast(I16, Bool, "safe"), false);
        assert_eq!(can_cast(I16, I8, "safe"), false);
        assert_eq!(can_cast(I16, I16, "safe"), true);
        assert_eq!(can_cast(I16, I32, "safe"), true);
        assert_eq!(can_cast(I16, I64, "safe"), true);
        assert_eq!(can_cast(I16, U8, "safe"), false);
        assert_eq!(can_cast(I16, U16, "safe"), false);
        assert_eq!(can_cast(I16, U32, "safe"), false);
        assert_eq!(can_cast(I16, U64, "safe"), false);
        assert_eq!(can_cast(I16, F16, "safe"), false);
        assert_eq!(can_cast(I16, F32, "safe"), true);
        assert_eq!(can_cast(I16, F64, "safe"), true);
        assert_eq!(can_cast(I16, C64, "safe"), true);
        assert_eq!(can_cast(I16, C128, "safe"), true);
        assert_eq!(can_cast(I32, Bool, "safe"), false);
        assert_eq!(can_cast(I32, I8, "safe"), false);
        assert_eq!(can_cast(I32, I16, "safe"), false);
        assert_eq!(can_cast(I32, I32, "safe"), true);
        assert_eq!(can_cast(I32, I64, "safe"), true);
        assert_eq!(can_cast(I32, U8, "safe"), false);
        assert_eq!(can_cast(I32, U16, "safe"), false);
        assert_eq!(can_cast(I32, U32, "safe"), false);
        assert_eq!(can_cast(I32, U64, "safe"), false);
        assert_eq!(can_cast(I32, F16, "safe"), false);
        assert_eq!(can_cast(I32, F32, "safe"), false);
        assert_eq!(can_cast(I32, F64, "safe"), true);
        assert_eq!(can_cast(I32, C64, "safe"), false);
        assert_eq!(can_cast(I32, C128, "safe"), true);
        assert_eq!(can_cast(I64, Bool, "safe"), false);
        assert_eq!(can_cast(I64, I8, "safe"), false);
        assert_eq!(can_cast(I64, I16, "safe"), false);
        assert_eq!(can_cast(I64, I32, "safe"), false);
        assert_eq!(can_cast(I64, I64, "safe"), true);
        assert_eq!(can_cast(I64, U8, "safe"), false);
        assert_eq!(can_cast(I64, U16, "safe"), false);
        assert_eq!(can_cast(I64, U32, "safe"), false);
        assert_eq!(can_cast(I64, U64, "safe"), false);
        assert_eq!(can_cast(I64, F16, "safe"), false);
        assert_eq!(can_cast(I64, F32, "safe"), false);
        assert_eq!(can_cast(I64, F64, "safe"), true);
        assert_eq!(can_cast(I64, C64, "safe"), false);
        assert_eq!(can_cast(I64, C128, "safe"), true);
        assert_eq!(can_cast(U8, Bool, "safe"), false);
        assert_eq!(can_cast(U8, I8, "safe"), false);
        assert_eq!(can_cast(U8, I16, "safe"), true);
        assert_eq!(can_cast(U8, I32, "safe"), true);
        assert_eq!(can_cast(U8, I64, "safe"), true);
        assert_eq!(can_cast(U8, U8, "safe"), true);
        assert_eq!(can_cast(U8, U16, "safe"), true);
        assert_eq!(can_cast(U8, U32, "safe"), true);
        assert_eq!(can_cast(U8, U64, "safe"), true);
        assert_eq!(can_cast(U8, F16, "safe"), true);
        assert_eq!(can_cast(U8, F32, "safe"), true);
        assert_eq!(can_cast(U8, F64, "safe"), true);
        assert_eq!(can_cast(U8, C64, "safe"), true);
        assert_eq!(can_cast(U8, C128, "safe"), true);
        assert_eq!(can_cast(U16, Bool, "safe"), false);
        assert_eq!(can_cast(U16, I8, "safe"), false);
        assert_eq!(can_cast(U16, I16, "safe"), false);
        assert_eq!(can_cast(U16, I32, "safe"), true);
        assert_eq!(can_cast(U16, I64, "safe"), true);
        assert_eq!(can_cast(U16, U8, "safe"), false);
        assert_eq!(can_cast(U16, U16, "safe"), true);
        assert_eq!(can_cast(U16, U32, "safe"), true);
        assert_eq!(can_cast(U16, U64, "safe"), true);
        assert_eq!(can_cast(U16, F16, "safe"), false);
        assert_eq!(can_cast(U16, F32, "safe"), true);
        assert_eq!(can_cast(U16, F64, "safe"), true);
        assert_eq!(can_cast(U16, C64, "safe"), true);
        assert_eq!(can_cast(U16, C128, "safe"), true);
        assert_eq!(can_cast(U32, Bool, "safe"), false);
        assert_eq!(can_cast(U32, I8, "safe"), false);
        assert_eq!(can_cast(U32, I16, "safe"), false);
        assert_eq!(can_cast(U32, I32, "safe"), false);
        assert_eq!(can_cast(U32, I64, "safe"), true);
        assert_eq!(can_cast(U32, U8, "safe"), false);
        assert_eq!(can_cast(U32, U16, "safe"), false);
        assert_eq!(can_cast(U32, U32, "safe"), true);
        assert_eq!(can_cast(U32, U64, "safe"), true);
        assert_eq!(can_cast(U32, F16, "safe"), false);
        assert_eq!(can_cast(U32, F32, "safe"), false);
        assert_eq!(can_cast(U32, F64, "safe"), true);
        assert_eq!(can_cast(U32, C64, "safe"), false);
        assert_eq!(can_cast(U32, C128, "safe"), true);
        assert_eq!(can_cast(U64, Bool, "safe"), false);
        assert_eq!(can_cast(U64, I8, "safe"), false);
        assert_eq!(can_cast(U64, I16, "safe"), false);
        assert_eq!(can_cast(U64, I32, "safe"), false);
        assert_eq!(can_cast(U64, I64, "safe"), false);
        assert_eq!(can_cast(U64, U8, "safe"), false);
        assert_eq!(can_cast(U64, U16, "safe"), false);
        assert_eq!(can_cast(U64, U32, "safe"), false);
        assert_eq!(can_cast(U64, U64, "safe"), true);
        assert_eq!(can_cast(U64, F16, "safe"), false);
        assert_eq!(can_cast(U64, F32, "safe"), false);
        assert_eq!(can_cast(U64, F64, "safe"), true);
        assert_eq!(can_cast(U64, C64, "safe"), false);
        assert_eq!(can_cast(U64, C128, "safe"), true);
        assert_eq!(can_cast(F16, Bool, "safe"), false);
        assert_eq!(can_cast(F16, I8, "safe"), false);
        assert_eq!(can_cast(F16, I16, "safe"), false);
        assert_eq!(can_cast(F16, I32, "safe"), false);
        assert_eq!(can_cast(F16, I64, "safe"), false);
        assert_eq!(can_cast(F16, U8, "safe"), false);
        assert_eq!(can_cast(F16, U16, "safe"), false);
        assert_eq!(can_cast(F16, U32, "safe"), false);
        assert_eq!(can_cast(F16, U64, "safe"), false);
        assert_eq!(can_cast(F16, F16, "safe"), true);
        assert_eq!(can_cast(F16, F32, "safe"), true);
        assert_eq!(can_cast(F16, F64, "safe"), true);
        assert_eq!(can_cast(F16, C64, "safe"), true);
        assert_eq!(can_cast(F16, C128, "safe"), true);
        assert_eq!(can_cast(F32, Bool, "safe"), false);
        assert_eq!(can_cast(F32, I8, "safe"), false);
        assert_eq!(can_cast(F32, I16, "safe"), false);
        assert_eq!(can_cast(F32, I32, "safe"), false);
        assert_eq!(can_cast(F32, I64, "safe"), false);
        assert_eq!(can_cast(F32, U8, "safe"), false);
        assert_eq!(can_cast(F32, U16, "safe"), false);
        assert_eq!(can_cast(F32, U32, "safe"), false);
        assert_eq!(can_cast(F32, U64, "safe"), false);
        assert_eq!(can_cast(F32, F16, "safe"), false);
        assert_eq!(can_cast(F32, F32, "safe"), true);
        assert_eq!(can_cast(F32, F64, "safe"), true);
        assert_eq!(can_cast(F32, C64, "safe"), true);
        assert_eq!(can_cast(F32, C128, "safe"), true);
        assert_eq!(can_cast(F64, Bool, "safe"), false);
        assert_eq!(can_cast(F64, I8, "safe"), false);
        assert_eq!(can_cast(F64, I16, "safe"), false);
        assert_eq!(can_cast(F64, I32, "safe"), false);
        assert_eq!(can_cast(F64, I64, "safe"), false);
        assert_eq!(can_cast(F64, U8, "safe"), false);
        assert_eq!(can_cast(F64, U16, "safe"), false);
        assert_eq!(can_cast(F64, U32, "safe"), false);
        assert_eq!(can_cast(F64, U64, "safe"), false);
        assert_eq!(can_cast(F64, F16, "safe"), false);
        assert_eq!(can_cast(F64, F32, "safe"), false);
        assert_eq!(can_cast(F64, F64, "safe"), true);
        assert_eq!(can_cast(F64, C64, "safe"), false);
        assert_eq!(can_cast(F64, C128, "safe"), true);
        assert_eq!(can_cast(C64, Bool, "safe"), false);
        assert_eq!(can_cast(C64, I8, "safe"), false);
        assert_eq!(can_cast(C64, I16, "safe"), false);
        assert_eq!(can_cast(C64, I32, "safe"), false);
        assert_eq!(can_cast(C64, I64, "safe"), false);
        assert_eq!(can_cast(C64, U8, "safe"), false);
        assert_eq!(can_cast(C64, U16, "safe"), false);
        assert_eq!(can_cast(C64, U32, "safe"), false);
        assert_eq!(can_cast(C64, U64, "safe"), false);
        assert_eq!(can_cast(C64, F16, "safe"), false);
        assert_eq!(can_cast(C64, F32, "safe"), false);
        assert_eq!(can_cast(C64, F64, "safe"), false);
        assert_eq!(can_cast(C64, C64, "safe"), true);
        assert_eq!(can_cast(C64, C128, "safe"), true);
        assert_eq!(can_cast(C128, Bool, "safe"), false);
        assert_eq!(can_cast(C128, I8, "safe"), false);
        assert_eq!(can_cast(C128, I16, "safe"), false);
        assert_eq!(can_cast(C128, I32, "safe"), false);
        assert_eq!(can_cast(C128, I64, "safe"), false);
        assert_eq!(can_cast(C128, U8, "safe"), false);
        assert_eq!(can_cast(C128, U16, "safe"), false);
        assert_eq!(can_cast(C128, U32, "safe"), false);
        assert_eq!(can_cast(C128, U64, "safe"), false);
        assert_eq!(can_cast(C128, F16, "safe"), false);
        assert_eq!(can_cast(C128, F32, "safe"), false);
        assert_eq!(can_cast(C128, F64, "safe"), false);
        assert_eq!(can_cast(C128, C64, "safe"), false);
        assert_eq!(can_cast(C128, C128, "safe"), true);
        assert_eq!(can_cast(Bool, Bool, "same_kind"), true);
        assert_eq!(can_cast(Bool, I8, "same_kind"), true);
        assert_eq!(can_cast(Bool, I16, "same_kind"), true);
        assert_eq!(can_cast(Bool, I32, "same_kind"), true);
        assert_eq!(can_cast(Bool, I64, "same_kind"), true);
        assert_eq!(can_cast(Bool, U8, "same_kind"), true);
        assert_eq!(can_cast(Bool, U16, "same_kind"), true);
        assert_eq!(can_cast(Bool, U32, "same_kind"), true);
        assert_eq!(can_cast(Bool, U64, "same_kind"), true);
        assert_eq!(can_cast(Bool, F16, "same_kind"), true);
        assert_eq!(can_cast(Bool, F32, "same_kind"), true);
        assert_eq!(can_cast(Bool, F64, "same_kind"), true);
        assert_eq!(can_cast(Bool, C64, "same_kind"), true);
        assert_eq!(can_cast(Bool, C128, "same_kind"), true);
        assert_eq!(can_cast(I8, Bool, "same_kind"), false);
        assert_eq!(can_cast(I8, I8, "same_kind"), true);
        assert_eq!(can_cast(I8, I16, "same_kind"), true);
        assert_eq!(can_cast(I8, I32, "same_kind"), true);
        assert_eq!(can_cast(I8, I64, "same_kind"), true);
        assert_eq!(can_cast(I8, U8, "same_kind"), false);
        assert_eq!(can_cast(I8, U16, "same_kind"), false);
        assert_eq!(can_cast(I8, U32, "same_kind"), false);
        assert_eq!(can_cast(I8, U64, "same_kind"), false);
        assert_eq!(can_cast(I8, F16, "same_kind"), true);
        assert_eq!(can_cast(I8, F32, "same_kind"), true);
        assert_eq!(can_cast(I8, F64, "same_kind"), true);
        assert_eq!(can_cast(I8, C64, "same_kind"), true);
        assert_eq!(can_cast(I8, C128, "same_kind"), true);
        assert_eq!(can_cast(I16, Bool, "same_kind"), false);
        assert_eq!(can_cast(I16, I8, "same_kind"), true);
        assert_eq!(can_cast(I16, I16, "same_kind"), true);
        assert_eq!(can_cast(I16, I32, "same_kind"), true);
        assert_eq!(can_cast(I16, I64, "same_kind"), true);
        assert_eq!(can_cast(I16, U8, "same_kind"), false);
        assert_eq!(can_cast(I16, U16, "same_kind"), false);
        assert_eq!(can_cast(I16, U32, "same_kind"), false);
        assert_eq!(can_cast(I16, U64, "same_kind"), false);
        assert_eq!(can_cast(I16, F16, "same_kind"), true);
        assert_eq!(can_cast(I16, F32, "same_kind"), true);
        assert_eq!(can_cast(I16, F64, "same_kind"), true);
        assert_eq!(can_cast(I16, C64, "same_kind"), true);
        assert_eq!(can_cast(I16, C128, "same_kind"), true);
        assert_eq!(can_cast(I32, Bool, "same_kind"), false);
        assert_eq!(can_cast(I32, I8, "same_kind"), true);
        assert_eq!(can_cast(I32, I16, "same_kind"), true);
        assert_eq!(can_cast(I32, I32, "same_kind"), true);
        assert_eq!(can_cast(I32, I64, "same_kind"), true);
        assert_eq!(can_cast(I32, U8, "same_kind"), false);
        assert_eq!(can_cast(I32, U16, "same_kind"), false);
        assert_eq!(can_cast(I32, U32, "same_kind"), false);
        assert_eq!(can_cast(I32, U64, "same_kind"), false);
        assert_eq!(can_cast(I32, F16, "same_kind"), true);
        assert_eq!(can_cast(I32, F32, "same_kind"), true);
        assert_eq!(can_cast(I32, F64, "same_kind"), true);
        assert_eq!(can_cast(I32, C64, "same_kind"), true);
        assert_eq!(can_cast(I32, C128, "same_kind"), true);
        assert_eq!(can_cast(I64, Bool, "same_kind"), false);
        assert_eq!(can_cast(I64, I8, "same_kind"), true);
        assert_eq!(can_cast(I64, I16, "same_kind"), true);
        assert_eq!(can_cast(I64, I32, "same_kind"), true);
        assert_eq!(can_cast(I64, I64, "same_kind"), true);
        assert_eq!(can_cast(I64, U8, "same_kind"), false);
        assert_eq!(can_cast(I64, U16, "same_kind"), false);
        assert_eq!(can_cast(I64, U32, "same_kind"), false);
        assert_eq!(can_cast(I64, U64, "same_kind"), false);
        assert_eq!(can_cast(I64, F16, "same_kind"), true);
        assert_eq!(can_cast(I64, F32, "same_kind"), true);
        assert_eq!(can_cast(I64, F64, "same_kind"), true);
        assert_eq!(can_cast(I64, C64, "same_kind"), true);
        assert_eq!(can_cast(I64, C128, "same_kind"), true);
        assert_eq!(can_cast(U8, Bool, "same_kind"), false);
        assert_eq!(can_cast(U8, I8, "same_kind"), true);
        assert_eq!(can_cast(U8, I16, "same_kind"), true);
        assert_eq!(can_cast(U8, I32, "same_kind"), true);
        assert_eq!(can_cast(U8, I64, "same_kind"), true);
        assert_eq!(can_cast(U8, U8, "same_kind"), true);
        assert_eq!(can_cast(U8, U16, "same_kind"), true);
        assert_eq!(can_cast(U8, U32, "same_kind"), true);
        assert_eq!(can_cast(U8, U64, "same_kind"), true);
        assert_eq!(can_cast(U8, F16, "same_kind"), true);
        assert_eq!(can_cast(U8, F32, "same_kind"), true);
        assert_eq!(can_cast(U8, F64, "same_kind"), true);
        assert_eq!(can_cast(U8, C64, "same_kind"), true);
        assert_eq!(can_cast(U8, C128, "same_kind"), true);
        assert_eq!(can_cast(U16, Bool, "same_kind"), false);
        assert_eq!(can_cast(U16, I8, "same_kind"), true);
        assert_eq!(can_cast(U16, I16, "same_kind"), true);
        assert_eq!(can_cast(U16, I32, "same_kind"), true);
        assert_eq!(can_cast(U16, I64, "same_kind"), true);
        assert_eq!(can_cast(U16, U8, "same_kind"), true);
        assert_eq!(can_cast(U16, U16, "same_kind"), true);
        assert_eq!(can_cast(U16, U32, "same_kind"), true);
        assert_eq!(can_cast(U16, U64, "same_kind"), true);
        assert_eq!(can_cast(U16, F16, "same_kind"), true);
        assert_eq!(can_cast(U16, F32, "same_kind"), true);
        assert_eq!(can_cast(U16, F64, "same_kind"), true);
        assert_eq!(can_cast(U16, C64, "same_kind"), true);
        assert_eq!(can_cast(U16, C128, "same_kind"), true);
        assert_eq!(can_cast(U32, Bool, "same_kind"), false);
        assert_eq!(can_cast(U32, I8, "same_kind"), true);
        assert_eq!(can_cast(U32, I16, "same_kind"), true);
        assert_eq!(can_cast(U32, I32, "same_kind"), true);
        assert_eq!(can_cast(U32, I64, "same_kind"), true);
        assert_eq!(can_cast(U32, U8, "same_kind"), true);
        assert_eq!(can_cast(U32, U16, "same_kind"), true);
        assert_eq!(can_cast(U32, U32, "same_kind"), true);
        assert_eq!(can_cast(U32, U64, "same_kind"), true);
        assert_eq!(can_cast(U32, F16, "same_kind"), true);
        assert_eq!(can_cast(U32, F32, "same_kind"), true);
        assert_eq!(can_cast(U32, F64, "same_kind"), true);
        assert_eq!(can_cast(U32, C64, "same_kind"), true);
        assert_eq!(can_cast(U32, C128, "same_kind"), true);
        assert_eq!(can_cast(U64, Bool, "same_kind"), false);
        assert_eq!(can_cast(U64, I8, "same_kind"), true);
        assert_eq!(can_cast(U64, I16, "same_kind"), true);
        assert_eq!(can_cast(U64, I32, "same_kind"), true);
        assert_eq!(can_cast(U64, I64, "same_kind"), true);
        assert_eq!(can_cast(U64, U8, "same_kind"), true);
        assert_eq!(can_cast(U64, U16, "same_kind"), true);
        assert_eq!(can_cast(U64, U32, "same_kind"), true);
        assert_eq!(can_cast(U64, U64, "same_kind"), true);
        assert_eq!(can_cast(U64, F16, "same_kind"), true);
        assert_eq!(can_cast(U64, F32, "same_kind"), true);
        assert_eq!(can_cast(U64, F64, "same_kind"), true);
        assert_eq!(can_cast(U64, C64, "same_kind"), true);
        assert_eq!(can_cast(U64, C128, "same_kind"), true);
        assert_eq!(can_cast(F16, Bool, "same_kind"), false);
        assert_eq!(can_cast(F16, I8, "same_kind"), false);
        assert_eq!(can_cast(F16, I16, "same_kind"), false);
        assert_eq!(can_cast(F16, I32, "same_kind"), false);
        assert_eq!(can_cast(F16, I64, "same_kind"), false);
        assert_eq!(can_cast(F16, U8, "same_kind"), false);
        assert_eq!(can_cast(F16, U16, "same_kind"), false);
        assert_eq!(can_cast(F16, U32, "same_kind"), false);
        assert_eq!(can_cast(F16, U64, "same_kind"), false);
        assert_eq!(can_cast(F16, F16, "same_kind"), true);
        assert_eq!(can_cast(F16, F32, "same_kind"), true);
        assert_eq!(can_cast(F16, F64, "same_kind"), true);
        assert_eq!(can_cast(F16, C64, "same_kind"), true);
        assert_eq!(can_cast(F16, C128, "same_kind"), true);
        assert_eq!(can_cast(F32, Bool, "same_kind"), false);
        assert_eq!(can_cast(F32, I8, "same_kind"), false);
        assert_eq!(can_cast(F32, I16, "same_kind"), false);
        assert_eq!(can_cast(F32, I32, "same_kind"), false);
        assert_eq!(can_cast(F32, I64, "same_kind"), false);
        assert_eq!(can_cast(F32, U8, "same_kind"), false);
        assert_eq!(can_cast(F32, U16, "same_kind"), false);
        assert_eq!(can_cast(F32, U32, "same_kind"), false);
        assert_eq!(can_cast(F32, U64, "same_kind"), false);
        assert_eq!(can_cast(F32, F16, "same_kind"), true);
        assert_eq!(can_cast(F32, F32, "same_kind"), true);
        assert_eq!(can_cast(F32, F64, "same_kind"), true);
        assert_eq!(can_cast(F32, C64, "same_kind"), true);
        assert_eq!(can_cast(F32, C128, "same_kind"), true);
        assert_eq!(can_cast(F64, Bool, "same_kind"), false);
        assert_eq!(can_cast(F64, I8, "same_kind"), false);
        assert_eq!(can_cast(F64, I16, "same_kind"), false);
        assert_eq!(can_cast(F64, I32, "same_kind"), false);
        assert_eq!(can_cast(F64, I64, "same_kind"), false);
        assert_eq!(can_cast(F64, U8, "same_kind"), false);
        assert_eq!(can_cast(F64, U16, "same_kind"), false);
        assert_eq!(can_cast(F64, U32, "same_kind"), false);
        assert_eq!(can_cast(F64, U64, "same_kind"), false);
        assert_eq!(can_cast(F64, F16, "same_kind"), true);
        assert_eq!(can_cast(F64, F32, "same_kind"), true);
        assert_eq!(can_cast(F64, F64, "same_kind"), true);
        assert_eq!(can_cast(F64, C64, "same_kind"), true);
        assert_eq!(can_cast(F64, C128, "same_kind"), true);
        assert_eq!(can_cast(C64, Bool, "same_kind"), false);
        assert_eq!(can_cast(C64, I8, "same_kind"), false);
        assert_eq!(can_cast(C64, I16, "same_kind"), false);
        assert_eq!(can_cast(C64, I32, "same_kind"), false);
        assert_eq!(can_cast(C64, I64, "same_kind"), false);
        assert_eq!(can_cast(C64, U8, "same_kind"), false);
        assert_eq!(can_cast(C64, U16, "same_kind"), false);
        assert_eq!(can_cast(C64, U32, "same_kind"), false);
        assert_eq!(can_cast(C64, U64, "same_kind"), false);
        assert_eq!(can_cast(C64, F16, "same_kind"), false);
        assert_eq!(can_cast(C64, F32, "same_kind"), false);
        assert_eq!(can_cast(C64, F64, "same_kind"), false);
        assert_eq!(can_cast(C64, C64, "same_kind"), true);
        assert_eq!(can_cast(C64, C128, "same_kind"), true);
        assert_eq!(can_cast(C128, Bool, "same_kind"), false);
        assert_eq!(can_cast(C128, I8, "same_kind"), false);
        assert_eq!(can_cast(C128, I16, "same_kind"), false);
        assert_eq!(can_cast(C128, I32, "same_kind"), false);
        assert_eq!(can_cast(C128, I64, "same_kind"), false);
        assert_eq!(can_cast(C128, U8, "same_kind"), false);
        assert_eq!(can_cast(C128, U16, "same_kind"), false);
        assert_eq!(can_cast(C128, U32, "same_kind"), false);
        assert_eq!(can_cast(C128, U64, "same_kind"), false);
        assert_eq!(can_cast(C128, F16, "same_kind"), false);
        assert_eq!(can_cast(C128, F32, "same_kind"), false);
        assert_eq!(can_cast(C128, F64, "same_kind"), false);
        assert_eq!(can_cast(C128, C64, "same_kind"), true);
        assert_eq!(can_cast(C128, C128, "same_kind"), true);
        assert_eq!(can_cast(Bool, Bool, "unsafe"), true);
        assert_eq!(can_cast(Bool, I8, "unsafe"), true);
        assert_eq!(can_cast(Bool, I16, "unsafe"), true);
        assert_eq!(can_cast(Bool, I32, "unsafe"), true);
        assert_eq!(can_cast(Bool, I64, "unsafe"), true);
        assert_eq!(can_cast(Bool, U8, "unsafe"), true);
        assert_eq!(can_cast(Bool, U16, "unsafe"), true);
        assert_eq!(can_cast(Bool, U32, "unsafe"), true);
        assert_eq!(can_cast(Bool, U64, "unsafe"), true);
        assert_eq!(can_cast(Bool, F16, "unsafe"), true);
        assert_eq!(can_cast(Bool, F32, "unsafe"), true);
        assert_eq!(can_cast(Bool, F64, "unsafe"), true);
        assert_eq!(can_cast(Bool, C64, "unsafe"), true);
        assert_eq!(can_cast(Bool, C128, "unsafe"), true);
        assert_eq!(can_cast(I8, Bool, "unsafe"), true);
        assert_eq!(can_cast(I8, I8, "unsafe"), true);
        assert_eq!(can_cast(I8, I16, "unsafe"), true);
        assert_eq!(can_cast(I8, I32, "unsafe"), true);
        assert_eq!(can_cast(I8, I64, "unsafe"), true);
        assert_eq!(can_cast(I8, U8, "unsafe"), true);
        assert_eq!(can_cast(I8, U16, "unsafe"), true);
        assert_eq!(can_cast(I8, U32, "unsafe"), true);
        assert_eq!(can_cast(I8, U64, "unsafe"), true);
        assert_eq!(can_cast(I8, F16, "unsafe"), true);
        assert_eq!(can_cast(I8, F32, "unsafe"), true);
        assert_eq!(can_cast(I8, F64, "unsafe"), true);
        assert_eq!(can_cast(I8, C64, "unsafe"), true);
        assert_eq!(can_cast(I8, C128, "unsafe"), true);
        assert_eq!(can_cast(I16, Bool, "unsafe"), true);
        assert_eq!(can_cast(I16, I8, "unsafe"), true);
        assert_eq!(can_cast(I16, I16, "unsafe"), true);
        assert_eq!(can_cast(I16, I32, "unsafe"), true);
        assert_eq!(can_cast(I16, I64, "unsafe"), true);
        assert_eq!(can_cast(I16, U8, "unsafe"), true);
        assert_eq!(can_cast(I16, U16, "unsafe"), true);
        assert_eq!(can_cast(I16, U32, "unsafe"), true);
        assert_eq!(can_cast(I16, U64, "unsafe"), true);
        assert_eq!(can_cast(I16, F16, "unsafe"), true);
        assert_eq!(can_cast(I16, F32, "unsafe"), true);
        assert_eq!(can_cast(I16, F64, "unsafe"), true);
        assert_eq!(can_cast(I16, C64, "unsafe"), true);
        assert_eq!(can_cast(I16, C128, "unsafe"), true);
        assert_eq!(can_cast(I32, Bool, "unsafe"), true);
        assert_eq!(can_cast(I32, I8, "unsafe"), true);
        assert_eq!(can_cast(I32, I16, "unsafe"), true);
        assert_eq!(can_cast(I32, I32, "unsafe"), true);
        assert_eq!(can_cast(I32, I64, "unsafe"), true);
        assert_eq!(can_cast(I32, U8, "unsafe"), true);
        assert_eq!(can_cast(I32, U16, "unsafe"), true);
        assert_eq!(can_cast(I32, U32, "unsafe"), true);
        assert_eq!(can_cast(I32, U64, "unsafe"), true);
        assert_eq!(can_cast(I32, F16, "unsafe"), true);
        assert_eq!(can_cast(I32, F32, "unsafe"), true);
        assert_eq!(can_cast(I32, F64, "unsafe"), true);
        assert_eq!(can_cast(I32, C64, "unsafe"), true);
        assert_eq!(can_cast(I32, C128, "unsafe"), true);
        assert_eq!(can_cast(I64, Bool, "unsafe"), true);
        assert_eq!(can_cast(I64, I8, "unsafe"), true);
        assert_eq!(can_cast(I64, I16, "unsafe"), true);
        assert_eq!(can_cast(I64, I32, "unsafe"), true);
        assert_eq!(can_cast(I64, I64, "unsafe"), true);
        assert_eq!(can_cast(I64, U8, "unsafe"), true);
        assert_eq!(can_cast(I64, U16, "unsafe"), true);
        assert_eq!(can_cast(I64, U32, "unsafe"), true);
        assert_eq!(can_cast(I64, U64, "unsafe"), true);
        assert_eq!(can_cast(I64, F16, "unsafe"), true);
        assert_eq!(can_cast(I64, F32, "unsafe"), true);
        assert_eq!(can_cast(I64, F64, "unsafe"), true);
        assert_eq!(can_cast(I64, C64, "unsafe"), true);
        assert_eq!(can_cast(I64, C128, "unsafe"), true);
        assert_eq!(can_cast(U8, Bool, "unsafe"), true);
        assert_eq!(can_cast(U8, I8, "unsafe"), true);
        assert_eq!(can_cast(U8, I16, "unsafe"), true);
        assert_eq!(can_cast(U8, I32, "unsafe"), true);
        assert_eq!(can_cast(U8, I64, "unsafe"), true);
        assert_eq!(can_cast(U8, U8, "unsafe"), true);
        assert_eq!(can_cast(U8, U16, "unsafe"), true);
        assert_eq!(can_cast(U8, U32, "unsafe"), true);
        assert_eq!(can_cast(U8, U64, "unsafe"), true);
        assert_eq!(can_cast(U8, F16, "unsafe"), true);
        assert_eq!(can_cast(U8, F32, "unsafe"), true);
        assert_eq!(can_cast(U8, F64, "unsafe"), true);
        assert_eq!(can_cast(U8, C64, "unsafe"), true);
        assert_eq!(can_cast(U8, C128, "unsafe"), true);
        assert_eq!(can_cast(U16, Bool, "unsafe"), true);
        assert_eq!(can_cast(U16, I8, "unsafe"), true);
        assert_eq!(can_cast(U16, I16, "unsafe"), true);
        assert_eq!(can_cast(U16, I32, "unsafe"), true);
        assert_eq!(can_cast(U16, I64, "unsafe"), true);
        assert_eq!(can_cast(U16, U8, "unsafe"), true);
        assert_eq!(can_cast(U16, U16, "unsafe"), true);
        assert_eq!(can_cast(U16, U32, "unsafe"), true);
        assert_eq!(can_cast(U16, U64, "unsafe"), true);
        assert_eq!(can_cast(U16, F16, "unsafe"), true);
        assert_eq!(can_cast(U16, F32, "unsafe"), true);
        assert_eq!(can_cast(U16, F64, "unsafe"), true);
        assert_eq!(can_cast(U16, C64, "unsafe"), true);
        assert_eq!(can_cast(U16, C128, "unsafe"), true);
        assert_eq!(can_cast(U32, Bool, "unsafe"), true);
        assert_eq!(can_cast(U32, I8, "unsafe"), true);
        assert_eq!(can_cast(U32, I16, "unsafe"), true);
        assert_eq!(can_cast(U32, I32, "unsafe"), true);
        assert_eq!(can_cast(U32, I64, "unsafe"), true);
        assert_eq!(can_cast(U32, U8, "unsafe"), true);
        assert_eq!(can_cast(U32, U16, "unsafe"), true);
        assert_eq!(can_cast(U32, U32, "unsafe"), true);
        assert_eq!(can_cast(U32, U64, "unsafe"), true);
        assert_eq!(can_cast(U32, F16, "unsafe"), true);
        assert_eq!(can_cast(U32, F32, "unsafe"), true);
        assert_eq!(can_cast(U32, F64, "unsafe"), true);
        assert_eq!(can_cast(U32, C64, "unsafe"), true);
        assert_eq!(can_cast(U32, C128, "unsafe"), true);
        assert_eq!(can_cast(U64, Bool, "unsafe"), true);
        assert_eq!(can_cast(U64, I8, "unsafe"), true);
        assert_eq!(can_cast(U64, I16, "unsafe"), true);
        assert_eq!(can_cast(U64, I32, "unsafe"), true);
        assert_eq!(can_cast(U64, I64, "unsafe"), true);
        assert_eq!(can_cast(U64, U8, "unsafe"), true);
        assert_eq!(can_cast(U64, U16, "unsafe"), true);
        assert_eq!(can_cast(U64, U32, "unsafe"), true);
        assert_eq!(can_cast(U64, U64, "unsafe"), true);
        assert_eq!(can_cast(U64, F16, "unsafe"), true);
        assert_eq!(can_cast(U64, F32, "unsafe"), true);
        assert_eq!(can_cast(U64, F64, "unsafe"), true);
        assert_eq!(can_cast(U64, C64, "unsafe"), true);
        assert_eq!(can_cast(U64, C128, "unsafe"), true);
        assert_eq!(can_cast(F16, Bool, "unsafe"), true);
        assert_eq!(can_cast(F16, I8, "unsafe"), true);
        assert_eq!(can_cast(F16, I16, "unsafe"), true);
        assert_eq!(can_cast(F16, I32, "unsafe"), true);
        assert_eq!(can_cast(F16, I64, "unsafe"), true);
        assert_eq!(can_cast(F16, U8, "unsafe"), true);
        assert_eq!(can_cast(F16, U16, "unsafe"), true);
        assert_eq!(can_cast(F16, U32, "unsafe"), true);
        assert_eq!(can_cast(F16, U64, "unsafe"), true);
        assert_eq!(can_cast(F16, F16, "unsafe"), true);
        assert_eq!(can_cast(F16, F32, "unsafe"), true);
        assert_eq!(can_cast(F16, F64, "unsafe"), true);
        assert_eq!(can_cast(F16, C64, "unsafe"), true);
        assert_eq!(can_cast(F16, C128, "unsafe"), true);
        assert_eq!(can_cast(F32, Bool, "unsafe"), true);
        assert_eq!(can_cast(F32, I8, "unsafe"), true);
        assert_eq!(can_cast(F32, I16, "unsafe"), true);
        assert_eq!(can_cast(F32, I32, "unsafe"), true);
        assert_eq!(can_cast(F32, I64, "unsafe"), true);
        assert_eq!(can_cast(F32, U8, "unsafe"), true);
        assert_eq!(can_cast(F32, U16, "unsafe"), true);
        assert_eq!(can_cast(F32, U32, "unsafe"), true);
        assert_eq!(can_cast(F32, U64, "unsafe"), true);
        assert_eq!(can_cast(F32, F16, "unsafe"), true);
        assert_eq!(can_cast(F32, F32, "unsafe"), true);
        assert_eq!(can_cast(F32, F64, "unsafe"), true);
        assert_eq!(can_cast(F32, C64, "unsafe"), true);
        assert_eq!(can_cast(F32, C128, "unsafe"), true);
        assert_eq!(can_cast(F64, Bool, "unsafe"), true);
        assert_eq!(can_cast(F64, I8, "unsafe"), true);
        assert_eq!(can_cast(F64, I16, "unsafe"), true);
        assert_eq!(can_cast(F64, I32, "unsafe"), true);
        assert_eq!(can_cast(F64, I64, "unsafe"), true);
        assert_eq!(can_cast(F64, U8, "unsafe"), true);
        assert_eq!(can_cast(F64, U16, "unsafe"), true);
        assert_eq!(can_cast(F64, U32, "unsafe"), true);
        assert_eq!(can_cast(F64, U64, "unsafe"), true);
        assert_eq!(can_cast(F64, F16, "unsafe"), true);
        assert_eq!(can_cast(F64, F32, "unsafe"), true);
        assert_eq!(can_cast(F64, F64, "unsafe"), true);
        assert_eq!(can_cast(F64, C64, "unsafe"), true);
        assert_eq!(can_cast(F64, C128, "unsafe"), true);
        assert_eq!(can_cast(C64, Bool, "unsafe"), true);
        assert_eq!(can_cast(C64, I8, "unsafe"), true);
        assert_eq!(can_cast(C64, I16, "unsafe"), true);
        assert_eq!(can_cast(C64, I32, "unsafe"), true);
        assert_eq!(can_cast(C64, I64, "unsafe"), true);
        assert_eq!(can_cast(C64, U8, "unsafe"), true);
        assert_eq!(can_cast(C64, U16, "unsafe"), true);
        assert_eq!(can_cast(C64, U32, "unsafe"), true);
        assert_eq!(can_cast(C64, U64, "unsafe"), true);
        assert_eq!(can_cast(C64, F16, "unsafe"), true);
        assert_eq!(can_cast(C64, F32, "unsafe"), true);
        assert_eq!(can_cast(C64, F64, "unsafe"), true);
        assert_eq!(can_cast(C64, C64, "unsafe"), true);
        assert_eq!(can_cast(C64, C128, "unsafe"), true);
        assert_eq!(can_cast(C128, Bool, "unsafe"), true);
        assert_eq!(can_cast(C128, I8, "unsafe"), true);
        assert_eq!(can_cast(C128, I16, "unsafe"), true);
        assert_eq!(can_cast(C128, I32, "unsafe"), true);
        assert_eq!(can_cast(C128, I64, "unsafe"), true);
        assert_eq!(can_cast(C128, U8, "unsafe"), true);
        assert_eq!(can_cast(C128, U16, "unsafe"), true);
        assert_eq!(can_cast(C128, U32, "unsafe"), true);
        assert_eq!(can_cast(C128, U64, "unsafe"), true);
        assert_eq!(can_cast(C128, F16, "unsafe"), true);
        assert_eq!(can_cast(C128, F32, "unsafe"), true);
        assert_eq!(can_cast(C128, F64, "unsafe"), true);
        assert_eq!(can_cast(C128, C64, "unsafe"), true);
        assert_eq!(can_cast(C128, C128, "unsafe"), true);
    }

    #[test]
    fn min_scalar_type_unsigned_matches_numpy() {
        assert_eq!(min_scalar_type_unsigned(0), Some(U8));
        assert_eq!(min_scalar_type_unsigned(255), Some(U8));
        assert_eq!(min_scalar_type_unsigned(256), Some(U16));
        assert_eq!(min_scalar_type_unsigned(65535), Some(U16));
        assert_eq!(min_scalar_type_unsigned(65536), Some(U32));
        assert_eq!(min_scalar_type_unsigned(4294967295), Some(U32));
        assert_eq!(min_scalar_type_unsigned(4294967296), Some(U64));
        assert_eq!(min_scalar_type_unsigned(18446744073709551615), Some(U64));
        assert_eq!(min_scalar_type_unsigned(18446744073709551616), None);
    }

    #[test]
    fn min_scalar_type_signed_matches_numpy() {
        assert_eq!(min_scalar_type_signed(-1), Some(I8));
        assert_eq!(min_scalar_type_signed(-128), Some(I8));
        assert_eq!(min_scalar_type_signed(-129), Some(I16));
        assert_eq!(min_scalar_type_signed(-32768), Some(I16));
        assert_eq!(min_scalar_type_signed(-32769), Some(I32));
        assert_eq!(min_scalar_type_signed(-2147483648), Some(I32));
        assert_eq!(min_scalar_type_signed(-2147483649), Some(I64));
        assert_eq!(min_scalar_type_signed(-9223372036854775808), Some(I64));
        assert_eq!(min_scalar_type_signed(-9223372036854775809), None);
    }

    #[test]
    fn min_scalar_type_float_matches_numpy() {
        assert_eq!(min_scalar_type_float(0.0), F16);
        assert_eq!(min_scalar_type_float(1.5), F16);
        assert_eq!(min_scalar_type_float(-1.5), F16);
        assert_eq!(min_scalar_type_float(64999.9), F16);
        assert_eq!(min_scalar_type_float(65000.0), F32);
        assert_eq!(min_scalar_type_float(65504.0), F32);
        assert_eq!(min_scalar_type_float(70000.0), F32);
        assert_eq!(min_scalar_type_float(3.39e38), F32);
        assert_eq!(min_scalar_type_float(3.4e38), F64);
        assert_eq!(min_scalar_type_float(1e300), F64);
        assert_eq!(min_scalar_type_float(-1e300), F64);
        assert_eq!(min_scalar_type_float(f64::INFINITY), F16);
        assert_eq!(min_scalar_type_float(f64::NAN), F16);
    }

    #[test]
    fn min_scalar_type_complex_matches_numpy() {
        assert_eq!(min_scalar_type_complex(1.0, 2.0), C64);
        assert_eq!(min_scalar_type_complex(3.3687953e38, 0.0), C64);
        assert_eq!(min_scalar_type_complex(3.4e38, 1.0), C128);
        assert_eq!(min_scalar_type_complex(1.0, 3.4e38), C128);
    }

    #[test]
    fn finfo_matches_numpy_2_5_1() {
        // Pinned against real numpy 2.5.1's `np.finfo(...).<field>.item()`
        // output directly (full-precision decimal, not the ~7-sig-fig
        // `repr()` numpy prints by default) -- a first pass at this table
        // used short decimal literals for f16/f32 that matched numpy's
        // truncated *display* but NOT its actual stored f64 value (the
        // exact widening of the true f16/f32 bit pattern), and this
        // narrower test (checking only `max`/`bits`/`precision`) did not
        // catch it. Caught by a differential probe against real numpy;
        // these assertions now pin every field to prevent recurrence.
        let f16 = finfo_for(F16).unwrap();
        assert_eq!(f16.eps, 0.0009765625);
        assert_eq!(f16.epsneg, 0.00048828125);
        assert_eq!(f16.max, 65504.0);
        assert_eq!(f16.min, -65504.0);
        assert_eq!(f16.tiny, 6.103515625e-05);
        assert_eq!(f16.smallest_normal, 6.103515625e-05);
        assert_eq!(f16.smallest_subnormal, 5.960464477539063e-08);
        assert_eq!(f16.resolution, 0.0010004043579101562);
        assert_eq!(f16.bits, 16);
        assert_eq!(f16.precision, 3);

        let f32 = finfo_for(F32).unwrap();
        assert_eq!(f32.eps, 1.1920928955078125e-07);
        assert_eq!(f32.epsneg, 5.960464477539063e-08);
        assert_eq!(f32.max, 3.4028234663852886e+38);
        assert_eq!(f32.min, -3.4028234663852886e+38);
        assert_eq!(f32.tiny, 1.1754943508222875e-38);
        assert_eq!(f32.smallest_normal, 1.1754943508222875e-38);
        assert_eq!(f32.smallest_subnormal, 1.401298464324817e-45);
        assert_eq!(f32.resolution, 9.999999974752427e-07);
        assert_eq!(f32.bits, 32);
        assert_eq!(f32.precision, 6);

        let f64_ = finfo_for(F64).unwrap();
        assert_eq!(f64_.eps, 2.220446049250313e-16);
        assert_eq!(f64_.epsneg, 1.1102230246251565e-16);
        assert_eq!(f64_.max, 1.7976931348623157e+308);
        assert_eq!(f64_.min, -1.7976931348623157e+308);
        assert_eq!(f64_.tiny, 2.2250738585072014e-308);
        assert_eq!(f64_.smallest_normal, 2.2250738585072014e-308);
        assert_eq!(f64_.smallest_subnormal, 5e-324);
        assert_eq!(f64_.resolution, 1e-15);
        assert_eq!(f64_.bits, 64);
        assert_eq!(f64_.precision, 15);

        assert!(finfo_for(I32).is_none());
        assert!(finfo_for(C64).is_none());
    }

    #[test]
    fn iinfo_matches_numpy_2_5_1() {
        let i8_ = iinfo_for(I8).unwrap();
        assert_eq!((i8_.min, i8_.max, i8_.bits), (-128, 127, 8));
        let u64_ = iinfo_for(U64).unwrap();
        assert_eq!((u64_.min, u64_.max, u64_.bits), (0, 18446744073709551615, 64));
        assert!(iinfo_for(F32).is_none());
        assert!(iinfo_for(Bool).is_none());
    }

    #[test]
    fn typename_for_matches_numpy() {
        assert_eq!(typename_for("?"), Some("bool"));
        assert_eq!(typename_for("S1"), Some("character"));
        assert_eq!(typename_for("G"), Some("complex long double precision"));
        assert_eq!(typename_for("e"), None);
        assert_eq!(typename_for("U1"), None);
    }

    #[test]
    fn mintypecode_select_matches_numpy() {
        assert_eq!(mintypecode_select(&["d", "f", "S"], "GDFgdf", "d"), "d");
        assert_eq!(mintypecode_select(&["F", "d"], "GDFgdf", "d"), "D");
        assert_eq!(mintypecode_select(&["f", "F"], "GDFgdf", "d"), "F");
        assert_eq!(mintypecode_select(&["i"], "GDFgdf", "d"), "d");
    }

    // -- S/U (flexible string dtype) tests: dtype-layer only, no Buffer/
    // Python reachability yet (phase 2). Every assertion below mirrors a
    // fact hand-verified live against real numpy 2.5.1 during this
    // session, cited in the doc comments of the functions under test.

    #[test]
    fn string_name_itemsize_kind() {
        // np.dtype('S5').name == 'bytes40'; np.dtype('U5').name == 'str160'
        assert_eq!(S(5).name(), "bytes40");
        assert_eq!(U(20).name(), "str160");
        assert_eq!(S(5).itemsize(), 5);
        assert_eq!(U(20).itemsize(), 20);
        assert_eq!(U(20).char_count(), 5);
        assert_eq!(S(5).kind_char(), 'S');
        assert_eq!(U(20).kind_char(), 'U');
        // np.dtype('S5').alignment == 1; np.dtype('U5').alignment == 4
        assert_eq!(S(5).alignment(), 1);
        assert_eq!(U(20).alignment(), 4);
        assert!(S(5).is_string() && S(5).is_bytes_dtype() && !S(5).is_unicode_dtype());
        assert!(U(20).is_string() && U(20).is_unicode_dtype() && !U(20).is_bytes_dtype());
        assert!(!Bool.is_string());
    }

    #[test]
    fn string_distinct_widths_are_distinct_dtypes() {
        // S3 != S5, and #[derive(PartialEq, Eq, Hash)] must get this right
        // "for free" from the u32 payload.
        assert_ne!(S(3), S(5));
        assert_ne!(U(4), U(20));
        assert_ne!(S(5), U(5)); // different kind entirely, even same n
    }

    #[test]
    fn string_promotion_matches_numpy() {
        // np.promote_types('S3','S5') == 'S5'  (max width, same kind)
        assert_eq!(promote_dtype(S(3), S(5)), S(5));
        // np.promote_types('U3','U5') == 'U5'
        assert_eq!(promote_dtype(U(12), U(20)), U(20));
        // np.promote_types('S3','U5') == 'U5' (3 bytes -> 3 chars = 12
        // bytes, still < 5-char U's 20 bytes, so U5 wins)
        assert_eq!(promote_dtype(S(3), U(20)), U(20));
        // np.promote_types('S6','U5') == 'U6' (6-byte S needs 6 chars =
        // 24 bytes > U5's 20, so widens to U6)
        assert_eq!(promote_dtype(S(6), U(20)), U(24));
        // np.promote_types(bool_, 'S2') == 'S5' (bool's own minimum-safe
        // width of 5 wins over the narrower S2 operand)
        assert_eq!(promote_dtype(Bool, S(2)), S(5));
        assert_eq!(promote_dtype(S(2), Bool), S(5));
        // np.promote_types('int64', 'U1') == 'U21' (int64 needs 21 chars)
        assert_eq!(promote_dtype(I64, U(4)), U(84));
        assert_eq!(promote_dtype(U(4), I64), U(84));
        // a wide-enough string operand is left untouched by a narrow
        // numeric partner: np.promote_types('int8','S10') == 'S10'
        assert_eq!(promote_dtype(I8, S(10)), S(10));
    }

    #[test]
    fn string_safe_cast_matches_numpy() {
        // S3 -> S5 safe (True); S5 -> S3 unsafe (False)
        assert!(is_safe_cast(S(3), S(5)));
        assert!(!is_safe_cast(S(5), S(3)));
        // U3 -> U5 safe; U5 -> U3 unsafe
        assert!(is_safe_cast(U(12), U(20)));
        assert!(!is_safe_cast(U(20), U(12)));
        // np.can_cast('S3','U5',casting='safe') == True (3 bytes fits in
        // 5 chars = 20 bytes); np.can_cast('S6','U5','safe') == False
        assert!(is_safe_cast(S(3), U(20)));
        assert!(!is_safe_cast(S(6), U(20)));
        // np.can_cast('U1','S4','safe') == False -- unicode never safely
        // narrows to bytes, at ANY width, not just equal-width.
        assert!(!is_safe_cast(U(4), S(4)));
        assert!(!is_safe_cast(U(4), S(100)));
        // np.can_cast(bool_, 'S5', 'safe') == True; np.can_cast(bool_,
        // 'S4', 'safe') == False (needs the full min width of 5)
        assert!(is_safe_cast(Bool, S(5)));
        assert!(!is_safe_cast(Bool, S(4)));
        // np.can_cast('int64','U21','safe') == True; 'U20' == False
        assert!(is_safe_cast(I64, U(84)));
        assert!(!is_safe_cast(I64, U(80)));
        // np.can_cast('S5','int8','safe') / .../'bool'/'float32'/'int64'
        // are all False -- a string dtype never safely casts into numeric.
        assert!(!is_safe_cast(S(5), I8));
        assert!(!is_safe_cast(S(5), Bool));
        assert!(!is_safe_cast(S(5), F32));
        assert!(!is_safe_cast(S(5), I64));
    }

    #[test]
    fn string_same_kind_cast_matches_numpy() {
        // bool -> S1 is True under same_kind despite needing S5 to be safe
        assert!(same_kind_extra(Bool, S(1)));
        // S5 -> S1 narrowing is True under same_kind
        assert!(same_kind_extra(S(5), S(1)));
        // U1 -> S10 is False at any width -- U never same_kind-narrows to S
        assert!(!same_kind_extra(U(4), S(10)));
        // anything numeric/Bytes/Str -> U is True under same_kind
        assert!(same_kind_extra(I64, U(4)));
        assert!(same_kind_extra(S(1), U(4)));
        assert!(same_kind_extra(U(100), U(4)));
        // S1 -> bool / U1 -> bool are both False (string -> numeric is
        // never same_kind, matching the existing Int/UInt/Float/Complex
        // arms that already exclude Bytes/Str from their kf patterns)
        assert!(!same_kind_extra(S(4), Bool));
        assert!(!same_kind_extra(U(4), Bool));
    }
}
