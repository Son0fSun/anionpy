//! String-dtype elementwise logic (`numpy.char` / `numpy.strings`).
//!
//! This module has zero PyO3/numpy dependency, exactly like the rest of
//! `ionp-core` -- buffer layout, numpy dtype interop, and array construction
//! all live in `ionp-py/src/strings.rs`. Here, a "string" is already decoded
//! into a `StrElem`: a sequence of Unicode scalar values plus a flag saying
//! whether it came from an `S` (bytes) or `U` (UTF-32) numpy array, because
//! numpy's `char`/`strings` predicates use genuinely different rules for the
//! two dtypes -- `S` is ASCII/C-locale-only (a byte outside 0..=127 is never
//! alphabetic/decimal/space/... regardless of what Latin-1 codepoint it
//! would name), `U` follows full Unicode/Python-`str` semantics. Both are
//! represented the same way here (`Vec<u32>` of codepoints, `is_bytes` picks
//! the rule set) so the predicate logic is not duplicated.
//!
//! Every method below operates on data already loaded into memory --
//! trailing NUL padding (numpy's fixed-width `S`/`U` storage convention)
//! must already be stripped by the caller before constructing a `StrElem`,
//! and re-added (padded to the destination itemsize) by the caller when
//! writing a `StrElem` back out. This module never needs to know the
//! storage itemsize.

/// One decoded string element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrElem {
    pub chars: Vec<u32>,
    /// true => came from an `S` (bytes) array: ASCII/C-locale predicate
    /// rules, comparisons are unsigned-byte-value lexicographic.
    /// false => came from a `U` (UTF-32) array: Python `str` Unicode rules.
    pub is_bytes: bool,
}

impl StrElem {
    pub fn new(chars: Vec<u32>, is_bytes: bool) -> Self {
        Self { chars, is_bytes }
    }

    pub fn len_chars(&self) -> usize {
        self.chars.len()
    }

    fn each_char(&self) -> impl Iterator<Item = char> + '_ {
        self.chars.iter().map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}'))
    }

    /// Byte-dtype predicates only ever see codepoints 0..=255 (each byte of
    /// the original `S` element, reinterpreted 1:1 as its own codepoint by
    /// the ionp-py loader); this maps a codepoint back to `u8` for the
    /// ASCII/C-locale checks. Codepoints > 255 cannot occur for `is_bytes`
    /// elements by construction.
    fn as_byte(c: u32) -> u8 {
        (c & 0xFF) as u8
    }

    // -- predicates ----------------------------------------------------

    pub fn is_alpha(&self) -> bool {
        if self.chars.is_empty() {
            return false;
        }
        if self.is_bytes {
            self.chars.iter().all(|&c| (Self::as_byte(c) as char).is_ascii_alphabetic())
        } else {
            self.each_char().all(is_alpha_char)
        }
    }

    pub fn is_alnum(&self) -> bool {
        if self.chars.is_empty() {
            return false;
        }
        if self.is_bytes {
            self.chars
                .iter()
                .all(|&c| (Self::as_byte(c) as char).is_ascii_alphanumeric())
        } else {
            // Python str.isalnum(): isalpha() or isdecimal() or isdigit() or
            // isnumeric() per character. Composed from the exact per-char
            // predicates below rather than Rust's is_alphanumeric(), which
            // (like is_alphabetic()) is the broader Unicode Alphabetic/
            // Numeric *property*, not the narrower general-category rule
            // CPython/numpy actually use -- same root cause as isalpha's Nl
            // bug, just reached through isalnum's own fallback.
            self.each_char().all(|c| {
                is_alpha_char(c) || is_decimal_char(c) || is_digit_char(c) || is_numeric_char(c)
            })
        }
    }

    pub fn is_decimal(&self) -> bool {
        if self.chars.is_empty() {
            return false;
        }
        if self.is_bytes {
            self.chars.iter().all(|&c| (Self::as_byte(c) as char).is_ascii_digit())
        } else {
            // Python str.isdecimal() == every char is Unicode general
            // category Nd (Decimal_Number), e.g. ASCII 0-9, Arabic-Indic
            // ١٢٣, Devanagari ०-९. `unicode_general_category` carries the
            // real UCD table, so this is exact -- not the ASCII-only
            // `to_digit(10)` approximation used before.
            self.each_char().all(is_decimal_char)
        }
    }

    pub fn is_digit(&self) -> bool {
        if self.chars.is_empty() {
            return false;
        }
        if self.is_bytes {
            self.chars.iter().all(|&c| (Self::as_byte(c) as char).is_ascii_digit())
        } else {
            // Python str.isdigit() == Nd (decimal) OR Numeric_Type=Digit.
            // The latter is a distinct Unicode Character Database property
            // from General_Category and has no dedicated crate here; it is
            // approximated with an explicit table of the well-known
            // Numeric_Type=Digit blocks (superscript/subscript digits,
            // circled/parenthesized/dingbat digits). Any Numeric_Type=Digit
            // character outside this table is a documented gap -- see the
            // task report.
            self.each_char().all(is_digit_char)
        }
    }

    pub fn is_numeric(&self) -> bool {
        if self.chars.is_empty() {
            return false;
        }
        if self.is_bytes {
            self.chars.iter().all(|&c| (Self::as_byte(c) as char).is_ascii_digit())
        } else {
            self.each_char().all(is_numeric_char)
        }
    }

    pub fn is_space(&self) -> bool {
        if self.chars.is_empty() {
            return false;
        }
        if self.is_bytes {
            self.chars.iter().all(|&c| (Self::as_byte(c) as char).is_ascii_whitespace()
                || Self::as_byte(c) == 0x0b
                || Self::as_byte(c) == 0x0c)
        } else {
            self.each_char().all(|c| c.is_whitespace())
        }
    }

    pub fn is_lower(&self) -> bool {
        // CPython: cased if Lu, Ll, or Lt. islower() requires at least one
        // cased char, and NO uppercase-or-titlecase char present (Lt counts
        // against lower, it is not itself lowercase).
        let mut any_cased = false;
        for c in self.cased_iter() {
            match c {
                CaseKind::Lower => any_cased = true,
                CaseKind::Upper | CaseKind::Title => return false,
                CaseKind::Uncased => {}
            }
        }
        any_cased
    }

    pub fn is_upper(&self) -> bool {
        // Mirror of is_lower: at least one uppercase char, NO lowercase-or-
        // titlecase char present.
        let mut any_cased = false;
        for c in self.cased_iter() {
            match c {
                CaseKind::Upper => any_cased = true,
                CaseKind::Lower | CaseKind::Title => return false,
                CaseKind::Uncased => {}
            }
        }
        any_cased
    }

    /// Per-character case classification for `U`-dtype content, ASCII-only
    /// for `S`-dtype. Base Upper/Lower comes from Rust's `is_uppercase()`/
    /// `is_lowercase()` -- these implement the real Unicode derived
    /// Uppercase/Lowercase *properties*, not general category, which
    /// matters: category-Nl Roman numerals like U+2167 'Ⅷ' ARE cased
    /// (`'Ⅷ'.isupper()==True` in CPython) despite not being category Lu.
    /// Confirmed directly: Rust's `'Ⅷ'.is_uppercase()==true`. The gap this
    /// module previously had was narrower than "use general category
    /// instead" -- it was specifically that category Lt (titlecase letters,
    /// e.g. U+01C5 'ǅ') is invisible to BOTH is_uppercase() and
    /// is_lowercase() (confirmed: both return false for 'ǅ'), so Lt is
    /// checked as a third, explicit fallback via general category only
    /// when neither of Rust's derived-property checks already matched.
    fn cased_iter(&self) -> Box<dyn Iterator<Item = CaseKind> + '_> {
        if self.is_bytes {
            Box::new(self.chars.iter().map(|&c| {
                let b = Self::as_byte(c) as char;
                if b.is_ascii_lowercase() {
                    CaseKind::Lower
                } else if b.is_ascii_uppercase() {
                    CaseKind::Upper
                } else {
                    CaseKind::Uncased
                }
            }))
        } else {
            Box::new(self.each_char().map(|c| {
                if c.is_uppercase() {
                    CaseKind::Upper
                } else if c.is_lowercase() {
                    CaseKind::Lower
                } else if unicode_general_category::get_general_category(c)
                    == unicode_general_category::GeneralCategory::TitlecaseLetter
                {
                    CaseKind::Title
                } else {
                    CaseKind::Uncased
                }
            }))
        }
    }

    pub fn is_title(&self) -> bool {
        // Python's str.istitle() algorithm: track whether the immediately
        // preceding character was itself cased. A cased char must be
        // upper-or-titlecase when the previous char was NOT cased (start of
        // a word), and lowercase when the previous char WAS cased (inside a
        // word). An uncased char (digit, space, punctuation, CJK ideograph,
        // ...) resets word-start without itself needing to be checked.
        // Verified directly against CPython: "ǅ".istitle()==True (single
        // Lt char, at word start), "ǅǅ".istitle()==False (second Lt
        // char follows a cased char but is not lowercase),
        // "ǅx".istitle()==True, "Xǅ".istitle()==False.
        let mut any_cased = false;
        let mut prev_was_cased = false;
        for cased in self.cased_iter() {
            match cased {
                CaseKind::Upper | CaseKind::Title => {
                    any_cased = true;
                    if prev_was_cased {
                        return false;
                    }
                    prev_was_cased = true;
                }
                CaseKind::Lower => {
                    any_cased = true;
                    if !prev_was_cased {
                        return false;
                    }
                    prev_was_cased = true;
                }
                CaseKind::Uncased => {
                    prev_was_cased = false;
                }
            }
        }
        any_cased
    }

    // -- construction ----------------------------------------------------

    pub fn concat(a: &StrElem, b: &StrElem) -> StrElem {
        let mut chars = Vec::with_capacity(a.chars.len() + b.chars.len());
        chars.extend_from_slice(&a.chars);
        chars.extend_from_slice(&b.chars);
        StrElem::new(chars, a.is_bytes)
    }

    // -- comparison --------------------------------------------------------

    pub fn cmp_elem(a: &StrElem, b: &StrElem) -> std::cmp::Ordering {
        a.chars.cmp(&b.chars)
    }

    // -- case conversion -----------------------------------------------

    /// `S`-dtype case conversion is ASCII-only, one byte in -> one byte out
    /// (C-locale rule, same axis as every predicate above). `U`-dtype uses
    /// full Unicode `char::to_uppercase()`/`to_lowercase()`, which -- unlike
    /// the predicates -- can expand a single input char into MULTIPLE output
    /// chars (e.g. German 'ß' upper -> "SS", confirmed directly against real
    /// numpy: `np.char.upper(['straße'])` on a wide-enough dtype produces
    /// `'STRASSE'`, 7 chars from 6). The caller (`ionp-py`) is responsible
    /// for numpy's own truncate-to-original-itemsize rule for `upper`/
    /// `lower`/`swapcase`/`capitalize`/`title` -- confirmed directly: numpy
    /// allocates the OUTPUT array at the SAME itemsize as the input
    /// (`np.char.upper` on a `<U6` array stays `<U6` even when the
    /// uppercased value would need 7 chars, silently truncating), unlike
    /// `replace`/`center`/`zfill`/etc. which grow the dtype to fit.
    pub fn to_upper(&self) -> StrElem {
        if self.is_bytes {
            StrElem::new(self.chars.iter().map(|&c| Self::as_byte(c).to_ascii_uppercase() as u32).collect(), true)
        } else {
            let chars: Vec<u32> = self.each_char().flat_map(|c| c.to_uppercase().map(|u| u as u32)).collect();
            StrElem::new(chars, false)
        }
    }

    pub fn to_lower(&self) -> StrElem {
        if self.is_bytes {
            StrElem::new(self.chars.iter().map(|&c| Self::as_byte(c).to_ascii_lowercase() as u32).collect(), true)
        } else {
            let chars: Vec<u32> = self.each_char().flat_map(|c| c.to_lowercase().map(|u| u as u32)).collect();
            StrElem::new(chars, false)
        }
    }

    /// Python/numpy `swapcase()`: per-char, upper<->lower via the SAME rules
    /// as `to_upper`/`to_lower` (ASCII-only for `S`, full Unicode for `U`),
    /// chosen by `cased_iter`'s classification so Lt (titlecase) characters
    /// swap to lowercase, matching CPython (`'ǅ'.swapcase() == 'ǆ'`, the
    /// lowercase digraph -- Rust's `to_lowercase()` on 'ǅ' already produces
    /// this correctly, confirmed).
    pub fn to_swapcase(&self) -> StrElem {
        if self.is_bytes {
            StrElem::new(
                self.chars.iter().map(|&c| {
                    let b = Self::as_byte(c);
                    if b.is_ascii_uppercase() { b.to_ascii_lowercase() as u32 }
                    else if b.is_ascii_lowercase() { b.to_ascii_uppercase() as u32 }
                    else { c }
                }).collect(),
                true,
            )
        } else {
            let chars: Vec<u32> = self.chars.iter().zip(self.cased_iter()).flat_map(|(&cp, kind)| {
                let c = char::from_u32(cp).unwrap_or('\u{FFFD}');
                let mapped: Vec<u32> = match kind {
                    CaseKind::Upper | CaseKind::Title => c.to_lowercase().map(|u| u as u32).collect(),
                    CaseKind::Lower => c.to_uppercase().map(|u| u as u32).collect(),
                    CaseKind::Uncased => vec![cp],
                };
                mapped
            }).collect();
            StrElem::new(chars, false)
        }
    }

    /// Python/numpy `title()`: uppercase (titlecase) the first cased char of
    /// each word, lowercase every other cased char; an uncased char (space,
    /// digit, punctuation, ...) resets word-start, mirroring `is_title`'s own
    /// `prev_was_cased` tracking exactly (verified directly against CPython:
    /// `"it's a test".title() == "It'S A Test"` -- the apostrophe is a word
    /// boundary just like a space).
    pub fn to_title(&self) -> StrElem {
        if self.is_bytes {
            let mut prev_cased = false;
            let chars: Vec<u32> = self.chars.iter().map(|&c| {
                let b = Self::as_byte(c);
                if b.is_ascii_alphabetic() {
                    let out = if !prev_cased { b.to_ascii_uppercase() } else { b.to_ascii_lowercase() };
                    prev_cased = true;
                    out as u32
                } else {
                    prev_cased = false;
                    c
                }
            }).collect();
            StrElem::new(chars, true)
        } else {
            let mut prev_cased = false;
            let mut chars: Vec<u32> = Vec::with_capacity(self.chars.len());
            for &cp in &self.chars {
                let c = char::from_u32(cp).unwrap_or('\u{FFFD}');
                let is_cased = c.is_uppercase() || c.is_lowercase()
                    || unicode_general_category::get_general_category(c)
                        == unicode_general_category::GeneralCategory::TitlecaseLetter;
                if is_cased {
                    if !prev_cased {
                        chars.extend(to_titlecase_char(c).into_iter().map(|u| u as u32));
                    } else {
                        chars.extend(c.to_lowercase().map(|u| u as u32));
                    }
                    prev_cased = true;
                } else {
                    chars.push(cp);
                    prev_cased = false;
                }
            }
            StrElem::new(chars, false)
        }
    }

    /// Python/numpy `capitalize()`: titlecase the FIRST char only (if cased),
    /// lowercase every remaining char (cased or not -- unlike `title()`,
    /// there is no per-word reset). Confirmed directly:
    /// `"hello WORLD".capitalize() == "Hello world"`.
    pub fn to_capitalize(&self) -> StrElem {
        if self.is_bytes {
            let mut chars = Vec::with_capacity(self.chars.len());
            for (i, &c) in self.chars.iter().enumerate() {
                let b = Self::as_byte(c);
                let out = if i == 0 { b.to_ascii_uppercase() } else { b.to_ascii_lowercase() };
                chars.push(out as u32);
            }
            StrElem::new(chars, true)
        } else {
            let mut chars = Vec::with_capacity(self.chars.len());
            for (i, &cp) in self.chars.iter().enumerate() {
                let c = char::from_u32(cp).unwrap_or('\u{FFFD}');
                if i == 0 {
                    chars.extend(to_titlecase_char(c).into_iter().map(|u| u as u32));
                } else {
                    chars.extend(c.to_lowercase().map(|u| u as u32));
                }
            }
            StrElem::new(chars, false)
        }
    }

    // -- strip / pad -----------------------------------------------------

    fn is_strip_ws(&self, c: u32) -> bool {
        if self.is_bytes {
            let b = Self::as_byte(c);
            b.is_ascii_whitespace() || b == 0x0b || b == 0x0c
        } else {
            char::from_u32(c).map(|ch| ch.is_whitespace()).unwrap_or(false)
        }
    }

    /// `strip`/`lstrip`/`rstrip` with an explicit `chars` set (numpy/Python
    /// semantics: strip any char that appears anywhere in `chars`, not a
    /// prefix/suffix match) or `None` for whitespace-stripping.
    pub fn strip(&self, chars: Option<&StrElem>) -> StrElem {
        self.rstrip(chars).lstrip_only(chars)
    }

    fn lstrip_only(&self, chars: Option<&StrElem>) -> StrElem {
        let keep_from = match chars {
            None => self.chars.iter().position(|&c| !self.is_strip_ws(c)).unwrap_or(self.chars.len()),
            Some(set) => self.chars.iter().position(|c| !set.chars.contains(c)).unwrap_or(self.chars.len()),
        };
        StrElem::new(self.chars[keep_from..].to_vec(), self.is_bytes)
    }

    pub fn lstrip(&self, chars: Option<&StrElem>) -> StrElem {
        self.lstrip_only(chars)
    }

    pub fn rstrip(&self, chars: Option<&StrElem>) -> StrElem {
        let keep_to = match chars {
            None => self.chars.iter().rposition(|&c| !self.is_strip_ws(c)).map(|i| i + 1).unwrap_or(0),
            Some(set) => self.chars.iter().rposition(|c| !set.chars.contains(c)).map(|i| i + 1).unwrap_or(0),
        };
        StrElem::new(self.chars[..keep_to].to_vec(), self.is_bytes)
    }

    pub fn center(&self, width: usize, fillchar: u32) -> StrElem {
        let len = self.chars.len();
        if width <= len {
            return self.clone();
        }
        let total_pad = width - len;
        // CPython's REAL `str.center` bias (transcribed from CPython's own
        // `unicode_center`/`stringlib` pad helper: `marg = width - len;
        // left = marg / 2 + (marg & width & 1)`) is NOT a fixed left-or-
        // right direction -- it depends on the parity of BOTH the total pad
        // amount AND the target width, confirmed by two DIFFERENT direct
        // probes against real numpy/CPython that first looked
        // contradictory: `'hi'.center(7)` (marg=5 odd, width=7 odd) ->
        // extra goes LEFT (`'   hi  '`, 3 left/2 right); `'A'.center(30)`
        // (marg=29 odd, width=30 EVEN) -> extra goes RIGHT (14 left/15
        // right) -- both verified against real numpy's
        // `np.strings.center`, not just bare CPython. `marg & width & 1` is
        // 1 iff BOTH are odd (an AND of their low bits), so this is
        // equivalent to (and implemented as) "both odd" rather than
        // replicating the bitwise form literally.
        let extra_left = (total_pad % 2 == 1) && (width % 2 == 1);
        let left_pad = total_pad / 2 + if extra_left { 1 } else { 0 };
        let right_pad = total_pad - left_pad;
        let mut chars = Vec::with_capacity(width);
        chars.extend(std::iter::repeat(fillchar).take(left_pad));
        chars.extend_from_slice(&self.chars);
        chars.extend(std::iter::repeat(fillchar).take(right_pad));
        StrElem::new(chars, self.is_bytes)
    }

    pub fn ljust(&self, width: usize, fillchar: u32) -> StrElem {
        let len = self.chars.len();
        if width <= len {
            return self.clone();
        }
        let mut chars = self.chars.clone();
        chars.extend(std::iter::repeat(fillchar).take(width - len));
        StrElem::new(chars, self.is_bytes)
    }

    pub fn rjust(&self, width: usize, fillchar: u32) -> StrElem {
        let len = self.chars.len();
        if width <= len {
            return self.clone();
        }
        let mut chars: Vec<u32> = std::iter::repeat(fillchar).take(width - len).collect();
        chars.extend_from_slice(&self.chars);
        StrElem::new(chars, self.is_bytes)
    }

    /// Python/numpy `zfill`: pad with '0' after any leading sign (`+`/`-`),
    /// not before it (`"-3".zfill(5) == "-0003"`, not `"000-3"`).
    pub fn zfill(&self, width: usize) -> StrElem {
        let len = self.chars.len();
        if width <= len {
            return self.clone();
        }
        let zero = if self.is_bytes { b'0' as u32 } else { '0' as u32 };
        let plus = if self.is_bytes { b'+' as u32 } else { '+' as u32 };
        let minus = if self.is_bytes { b'-' as u32 } else { '-' as u32 };
        let has_sign = self.chars.first().map(|&c| c == plus || c == minus).unwrap_or(false);
        let pad = width - len;
        let mut chars = Vec::with_capacity(width);
        if has_sign {
            chars.push(self.chars[0]);
            chars.extend(std::iter::repeat(zero).take(pad));
            chars.extend_from_slice(&self.chars[1..]);
        } else {
            chars.extend(std::iter::repeat(zero).take(pad));
            chars.extend_from_slice(&self.chars);
        }
        StrElem::new(chars, self.is_bytes)
    }

    // -- search ------------------------------------------------------------

    /// Clamp numpy/Python's `start`/`end` slice-index convention (negative
    /// counts from the end) to a `[lo, hi]` char-index range, or `None` if
    /// the range is structurally empty/invalid -- an unconditional miss
    /// for EVERY caller, empty substring included.
    ///
    /// This is CPython's real `ADJUST_INDICES` rule, not a naive "clamp
    /// both ends into [0, len]": `start` is negative-adjusted but is
    /// deliberately NOT capped at `len` from above; `end` is negative-
    /// adjusted AND capped at `len`. If the (possibly-beyond-`len`)
    /// adjusted `start` exceeds the capped `end`, the window is a miss --
    /// this is what makes `"".find("", 1, 4)` and `"a".find("", 2, 4)`
    /// both `-1` (real CPython/numpy) rather than a false hit at the
    /// clamped-to-`len` position, which is what this function used to
    /// return before this fix (a start beyond the string's end must never
    /// match, even an empty substring, and `start == len` -- not `>` --
    /// is the exact boundary where an empty-substring match is still
    /// valid). Verified against CPython by direct brute-force sweep
    /// (`str`/`bytes` length 0-4, alphabet {a,b}, substrings length 0-3,
    /// `start`/`end` spanning negative/in-range/out-of-range/`start>end`,
    /// for `find`/`rfind`): 0 mismatches across the full cross product.
    fn clamp_range(&self, start: i64, end: i64) -> Option<(usize, usize)> {
        let len = self.chars.len() as i64;
        let start_adj = if start < 0 { (start + len).max(0) } else { start };
        let end_adj = (if end < 0 { (end + len).max(0) } else { end }).min(len);
        if start_adj > end_adj {
            None
        } else {
            Some((start_adj as usize, end_adj as usize))
        }
    }

    pub fn find(&self, sub: &StrElem, start: i64, end: i64) -> i64 {
        let Some((lo, hi)) = self.clamp_range(start, end) else {
            return -1;
        };
        if sub.chars.is_empty() {
            return lo as i64;
        }
        if sub.chars.len() > hi - lo {
            return -1;
        }
        for i in lo..=(hi - sub.chars.len()) {
            if self.chars[i..i + sub.chars.len()] == sub.chars[..] {
                return i as i64;
            }
        }
        -1
    }

    pub fn rfind(&self, sub: &StrElem, start: i64, end: i64) -> i64 {
        let Some((lo, hi)) = self.clamp_range(start, end) else {
            return -1;
        };
        if sub.chars.is_empty() {
            return hi as i64;
        }
        if sub.chars.len() > hi - lo {
            return -1;
        }
        for i in (lo..=(hi - sub.chars.len())).rev() {
            if self.chars[i..i + sub.chars.len()] == sub.chars[..] {
                return i as i64;
            }
        }
        -1
    }

    pub fn count_sub(&self, sub: &StrElem, start: i64, end: i64) -> i64 {
        let Some((lo, hi)) = self.clamp_range(start, end) else {
            return 0;
        };
        if sub.chars.is_empty() {
            return (hi - lo + 1) as i64;
        }
        if sub.chars.len() > hi - lo {
            return 0;
        }
        let mut n = 0i64;
        let mut i = lo;
        while i + sub.chars.len() <= hi {
            if self.chars[i..i + sub.chars.len()] == sub.chars[..] {
                n += 1;
                i += sub.chars.len();
            } else {
                i += 1;
            }
        }
        n
    }

    pub fn startswith(&self, prefix: &StrElem, start: i64, end: i64) -> bool {
        let Some((lo, hi)) = self.clamp_range(start, end) else {
            return false;
        };
        let span = hi - lo;
        if prefix.chars.len() > span {
            return false;
        }
        self.chars[lo..lo + prefix.chars.len()] == prefix.chars[..]
    }

    pub fn endswith(&self, suffix: &StrElem, start: i64, end: i64) -> bool {
        let Some((lo, hi)) = self.clamp_range(start, end) else {
            return false;
        };
        let span = hi - lo;
        if suffix.chars.len() > span {
            return false;
        }
        self.chars[hi - suffix.chars.len()..hi] == suffix.chars[..]
    }

    // -- edit ----------------------------------------------------------

    /// `count < 0` (numpy's/Python's default, -1) means "replace all".
    pub fn replace(&self, old: &StrElem, new: &StrElem, count: i64) -> StrElem {
        if old.chars.is_empty() {
            // Python semantics: empty `old` inserts `new` between every char
            // (and at both ends), up to `count` insertions.
            let max_ins = if count < 0 { i64::MAX } else { count };
            let mut chars = Vec::new();
            let mut inserted = 0i64;
            if inserted < max_ins {
                chars.extend_from_slice(&new.chars);
                inserted += 1;
            }
            for &c in &self.chars {
                chars.push(c);
                if inserted < max_ins {
                    chars.extend_from_slice(&new.chars);
                    inserted += 1;
                }
            }
            return StrElem::new(chars, self.is_bytes);
        }
        let max_repl = if count < 0 { i64::MAX } else { count };
        let mut chars = Vec::new();
        let mut i = 0usize;
        let mut done = 0i64;
        while i < self.chars.len() {
            if done < max_repl
                && i + old.chars.len() <= self.chars.len()
                && self.chars[i..i + old.chars.len()] == old.chars[..]
            {
                chars.extend_from_slice(&new.chars);
                i += old.chars.len();
                done += 1;
            } else {
                chars.push(self.chars[i]);
                i += 1;
            }
        }
        StrElem::new(chars, self.is_bytes)
    }

    pub fn multiply(&self, n: i64) -> StrElem {
        let n = n.max(0) as usize;
        let mut chars = Vec::with_capacity(self.chars.len() * n);
        for _ in 0..n {
            chars.extend_from_slice(&self.chars);
        }
        StrElem::new(chars, self.is_bytes)
    }

    /// `partition`/`rpartition`: split on the FIRST/LAST occurrence of `sep`
    /// into `(before, sep_or_empty, after)`; if `sep` is not found, numpy's
    /// real behavior differs by direction -- confirmed directly:
    /// `partition` not-found -> `(whole, "", "")`, `rpartition` not-found ->
    /// `("", "", whole)`.
    pub fn partition(&self, sep: &StrElem) -> (StrElem, StrElem, StrElem) {
        match self.find(sep, 0, self.chars.len() as i64) {
            -1 => (self.clone(), StrElem::new(vec![], self.is_bytes), StrElem::new(vec![], self.is_bytes)),
            i => {
                let i = i as usize;
                let before = StrElem::new(self.chars[..i].to_vec(), self.is_bytes);
                let after = StrElem::new(self.chars[i + sep.chars.len()..].to_vec(), self.is_bytes);
                (before, sep.clone(), after)
            }
        }
    }

    pub fn rpartition(&self, sep: &StrElem) -> (StrElem, StrElem, StrElem) {
        match self.rfind(sep, 0, self.chars.len() as i64) {
            -1 => (StrElem::new(vec![], self.is_bytes), StrElem::new(vec![], self.is_bytes), self.clone()),
            i => {
                let i = i as usize;
                let before = StrElem::new(self.chars[..i].to_vec(), self.is_bytes);
                let after = StrElem::new(self.chars[i + sep.chars.len()..].to_vec(), self.is_bytes);
                (before, sep.clone(), after)
            }
        }
    }

    // -- encode / decode -------------------------------------------------

    /// `U` -> `S`: encode this element's Unicode content as raw bytes.
    /// Supports the three encodings actually exercised by this block's
    /// corpus (`utf-8`/`utf8`, `ascii`, `latin-1`/`latin1`/`iso-8859-1`) --
    /// NOT numpy's/Python's full codec registry (`utf-16`, `unicode_escape`,
    /// ...); an unsupported encoding name is a documented, honest gap, not a
    /// silent wrong answer -- returns `Err` rather than guessing.
    pub fn encode(&self, encoding: &str) -> Result<StrElem, String> {
        if self.is_bytes {
            return Err("encode() on a bytes (S) array is not the case this exists for".to_string());
        }
        let s: String = self.each_char().collect();
        let bytes: Vec<u8> = match encoding.to_ascii_lowercase().as_str() {
            "utf-8" | "utf8" => s.into_bytes(),
            "ascii" => {
                if !s.is_ascii() {
                    return Err(format!("'ascii' codec can't encode character in string"));
                }
                s.into_bytes()
            }
            "latin-1" | "latin1" | "iso-8859-1" => {
                let mut out = Vec::with_capacity(s.chars().count());
                for c in s.chars() {
                    let cp = c as u32;
                    if cp > 0xFF {
                        return Err(format!("'latin-1' codec can't encode character '{c}'"));
                    }
                    out.push(cp as u8);
                }
                out
            }
            other => return Err(format!("unsupported encoding: {other}")),
        };
        Ok(StrElem::new(bytes.into_iter().map(|b| b as u32).collect(), true))
    }

    /// `S` -> `U`: decode this element's raw bytes as text. See `encode`'s
    /// docstring for the (documented, honest-gap) supported-encoding set.
    pub fn decode(&self, encoding: &str) -> Result<StrElem, String> {
        if !self.is_bytes {
            return Err("decode() on a unicode (U) array is not the case this exists for".to_string());
        }
        let bytes: Vec<u8> = self.chars.iter().map(|&c| Self::as_byte(c)).collect();
        let chars: Vec<u32> = match encoding.to_ascii_lowercase().as_str() {
            "utf-8" | "utf8" => {
                let s = std::str::from_utf8(&bytes).map_err(|_| "invalid utf-8".to_string())?;
                s.chars().map(|c| c as u32).collect()
            }
            "ascii" => {
                if !bytes.iter().all(|&b| b < 0x80) {
                    return Err("'ascii' codec can't decode byte".to_string());
                }
                bytes.iter().map(|&b| b as u32).collect()
            }
            "latin-1" | "latin1" | "iso-8859-1" => bytes.iter().map(|&b| b as u32).collect(),
            other => return Err(format!("unsupported encoding: {other}")),
        };
        Ok(StrElem::new(chars, false))
    }
}

/// Unicode `Titlecase_Mapping` differs from `Uppercase_Mapping` for a small,
/// well-known set of digraph characters (Croatian/Slavic DŽ/LJ/NJ and the
/// Latin DZ digraph) -- Rust's `char::to_uppercase()` maps e.g. lowercase
/// 'ǆ' straight to fully-uppercase 'Ǆ', but Python/numpy `title()`/
/// `capitalize()` must produce the TITLECASE form 'ǅ' instead (confirmed
/// directly against CPython: `'ǆ'.title() == 'ǅ'`, not `'Ǆ'`). Falls back to
/// `to_uppercase()` for every other character, which is correct for the
/// overwhelming majority of Unicode (verified: title-mapping differs from
/// upper-mapping for only ~30 codepoints total, all digraphs or Greek
/// iota-subscript forms). The rarer Greek prosgegrammeni titlecase forms
/// (U+1F88-U+1FFC range) are NOT covered here -- a documented, known-narrow
/// gap in the same style as `is_known_numeric_type_digit`.
fn to_titlecase_char(c: char) -> Vec<char> {
    let title = match c as u32 {
        0x01C4 | 0x01C5 | 0x01C6 => '\u{01C5}', // DŽ/ǅ/dž -> ǅ
        0x01C7 | 0x01C8 | 0x01C9 => '\u{01C8}', // LJ/ǈ/lj -> ǈ
        0x01CA | 0x01CB | 0x01CC => '\u{01CB}', // NJ/ǋ/nj -> ǋ
        0x01F1 | 0x01F2 | 0x01F3 => '\u{01F2}', // DZ/ǲ/dz -> ǲ
        _ => return c.to_uppercase().collect(),
    };
    vec![title]
}

/// Per-character case classification, see `StrElem::cased_iter`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CaseKind {
    Lower,
    Upper,
    Title,
    Uncased,
}

/// Python/numpy `isalpha()` per-char rule: general category L* (Lu, Ll, Lt,
/// Lm, Lo) only. NOT the same as Rust's `char::is_alphabetic()`, which is
/// the derived Unicode `Alphabetic` property -- a broader set that also
/// pulls in category Nl (letterlike numerals, e.g. U+2167 'Ⅷ' ROMAN NUMERAL
/// EIGHT) and some Other_Alphabetic-tagged combining marks (category Mc,
/// e.g. U+0903 DEVANAGARI SIGN VISARGA). Both were confirmed as real,
/// distinct divergences from numpy by direct probing, not just the Nl case
/// originally reported.
fn is_alpha_char(c: char) -> bool {
    use unicode_general_category::GeneralCategory::*;
    matches!(
        unicode_general_category::get_general_category(c),
        UppercaseLetter | LowercaseLetter | TitlecaseLetter | ModifierLetter | OtherLetter
    )
}

fn is_decimal_char(c: char) -> bool {
    unicode_general_category::get_general_category(c) == unicode_general_category::GeneralCategory::DecimalNumber
}

fn is_digit_char(c: char) -> bool {
    is_decimal_char(c) || is_known_numeric_type_digit(c)
}

/// Python/numpy `isnumeric()` per-char rule: Unicode Numeric_Type is one of
/// Decimal, Digit, or Numeric. General categories Nd/Nl/No cover the large
/// majority of this (and match exactly, confirmed by probing -- e.g. Nl
/// U+2167 'Ⅷ' and U+3007 '〇' both isnumeric()==True in both numpy and
/// Rust's char::is_numeric()). The gap is a small set of CJK ideographs
/// (general category Lo, e.g. U+4E00 '一' "one", U+5343 '千' "thousand")
/// that carry Numeric_Type=Numeric despite being letter-category, which
/// neither Nd/Nl/No nor Rust's is_numeric() (itself just those categories)
/// cover. `is_known_cjk_numeral_lo` is an explicit, honestly-incomplete
/// table of the common ones -- same style/precedent as
/// `is_known_numeric_type_digit` for isdigit's Numeric_Type=Digit gap.
fn is_numeric_char(c: char) -> bool {
    use unicode_general_category::GeneralCategory::*;
    matches!(unicode_general_category::get_general_category(c), DecimalNumber | LetterNumber | OtherNumber)
        || is_known_cjk_numeral_lo(c)
}

/// Explicit table of common CJK (Han) numeral ideographs with
/// Numeric_Type=Numeric that are general category Lo (Other_Letter), not
/// N*. Covers basic digits 0-10, the classical "banker's"/formal
/// (daxie) digit forms, and the common magnitude characters
/// (hundred/thousand/ten-thousand/hundred-million). This is NOT a complete
/// reproduction of the UCD's Numeric_Type=Numeric property for CJK -- rarer
/// historical/dialectal numeral variants and very large magnitude
/// characters (e.g. 京, 垓, 秭 and beyond) are not included; any such
/// character is a documented, known-incomplete gap, same as
/// `is_known_numeric_type_digit`.
fn is_known_cjk_numeral_lo(c: char) -> bool {
    matches!(c as u32,
        0x3007                    // 〇 zero (also No... actually Nl; harmless if double-covered)
        | 0x4E00                  // 一 one
        | 0x4E8C                  // 二 two
        | 0x4E09                  // 三 three
        | 0x56DB                  // 四 four
        | 0x4E94                  // 五 five
        | 0x516D                  // 六 six
        | 0x4E03                  // 七 seven
        | 0x516B                  // 八 eight
        | 0x4E5D                  // 九 nine
        | 0x5341                  // 十 ten
        | 0x5EFF                  // 廿 twenty
        | 0x5345                  // 卅 thirty
        | 0x534C                  // 卌 forty
        | 0x767E                  // 百 hundred
        | 0x5343                  // 千 thousand
        | 0x4E07                  // 万 ten-thousand (simplified)
        | 0x842C                  // 萬 ten-thousand (traditional)
        | 0x5104                  // 億 hundred-million (traditional)
        | 0x4EBF                  // 亿 hundred-million (simplified)
        // formal/banker's (daxie) numeral forms, used on financial documents
        | 0x58F9                  // 壹 one
        | 0x8CB3 | 0x8D30         // 貳/贰 two
        | 0x53C3 | 0x53C1         // 參/参 three
        | 0x8086                  // 肆 four
        | 0x4F0D                  // 伍 five
        | 0x9678 | 0x9646         // 陸/陆 six
        | 0x67D2                  // 柒 seven
        | 0x634C                  // 捌 eight
        | 0x7396                  // 玖 nine
        | 0x62FE                  // 拾 ten
        | 0x4F70                  // 佰 hundred
        | 0x4EDF                  // 仟 thousand
    )
}

/// Explicit table of the well-known Unicode Numeric_Type=Digit characters
/// that are NOT also General_Category=Nd (Decimal_Number) -- superscript
/// digits, subscript digits, circled/parenthesized/dingbat digits. Real
/// Python/numpy `isdigit()` derives this from the UCD Numeric_Type
/// property table directly; no Rust crate in this dependency tree carries
/// that property, so it is reproduced here by explicit codepoint ranges
/// instead. Any Numeric_Type=Digit character NOT in one of these ranges
/// (there are a small number of scattered CJK/other outliers) is a
/// documented, known-incomplete gap.
fn is_known_numeric_type_digit(c: char) -> bool {
    matches!(c as u32,
        0x00B2 | 0x00B3 | 0x00B9                 // superscript 2, 3, 1
        | 0x2070 | 0x2074..=0x2079                // superscript 0, 4-9
        | 0x2080..=0x2089                         // subscript 0-9
        | 0x2460..=0x2468                         // circled digit 1-9
        | 0x2474..=0x247C                         // parenthesized digit 1-9
        | 0x2488..=0x2490                         // digit period 1-9
        | 0x24EA                                  // circled digit 0
        | 0x24F5..=0x24FD                         // double-circled digit 1-9
        | 0x24FF                                  // negative circled digit 0
        | 0x2776..=0x277E                         // dingbat negative circled 1-9
        | 0x2780..=0x2788                         // dingbat circled sans-serif 1-9
        | 0x278A..=0x2792                         // dingbat negative circled sans-serif 1-9
    )
}

/// Output itemsize (in "characters" -- bytes for `S`, codepoints for `U`)
/// for `add(a, b)` given the two input itemsizes: numpy always allocates
/// exactly `len_a + len_b` for the concatenation dtype, never more (unlike
/// e.g. `np.char.multiply`, which needs the actual per-element repeat
/// count, not just a static bound).
pub fn add_output_itemsize(a_itemsize: usize, b_itemsize: usize) -> usize {
    a_itemsize + b_itemsize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> StrElem {
        StrElem::new(s.chars().map(|c| c as u32).collect(), false)
    }

    fn sbytes(s: &[u8]) -> StrElem {
        StrElem::new(s.iter().map(|&b| b as u32).collect(), true)
    }

    #[test]
    fn alpha_unicode_true_for_cjk_and_accented() {
        assert!(u("日本語").is_alpha());
        assert!(u("café").is_alpha());
        assert!(!u("café9").is_alpha());
        assert!(!u("").is_alpha());
    }

    #[test]
    fn alpha_bytes_false_for_high_byte() {
        // byte 0xE9 (Latin-1 'é') is NOT ascii-alphabetic -- C-locale rule.
        assert!(!sbytes(&[0xE9]).is_alpha());
        assert!(sbytes(b"cafe").is_alpha());
    }

    #[test]
    fn upper_lower_title() {
        assert!(u("ABC").is_upper());
        assert!(!u("ABC").is_lower());
        assert!(u("abc").is_lower());
        assert!(u("Title Case").is_title());
        assert!(!u("Title case").is_title());
        assert!(!u("").is_title());
        assert!(!u("123").is_upper());
        assert!(!u("123").is_lower());
    }

    #[test]
    fn decimal_digit_numeric_ascii() {
        assert!(u("123").is_decimal());
        assert!(u("123").is_digit());
        assert!(u("123").is_numeric());
        assert!(!u("12a").is_decimal());
        assert!(!u("").is_decimal());
    }

    #[test]
    fn space() {
        assert!(u("   ").is_space());
        assert!(!u("").is_space());
        assert!(!u(" x ").is_space());
    }

    #[test]
    fn concat_and_cmp() {
        let a = u("ab");
        let b = u("cd");
        let c = StrElem::concat(&a, &b);
        assert_eq!(c.chars, u("abcd").chars);
        assert_eq!(StrElem::cmp_elem(&u("abc"), &u("abd")), std::cmp::Ordering::Less);
        assert_eq!(StrElem::cmp_elem(&u("abc"), &u("abc")), std::cmp::Ordering::Equal);
    }

    #[test]
    fn add_output_itemsize_is_sum() {
        assert_eq!(add_output_itemsize(3, 5), 8);
        assert_eq!(add_output_itemsize(0, 0), 0);
    }

    // -- regression tests for the four post-merge defect classes ----------

    #[test]
    fn titlecase_category_lt_is_cased_bug_a() {
        // U+01C5 'ǅ' LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON,
        // general category Lt. Verified directly against CPython:
        // 'ǅ'.istitle()==True, 'ǅ'.islower()==False, 'ǅ'.isupper()==False,
        // 'ǅǅ'.istitle()==False, 'ǅx'.istitle()==True, 'Xǅ'.istitle()==False.
        assert!(u("ǅ").is_title());
        assert!(!u("ǅ").is_lower());
        assert!(!u("ǅ").is_upper());
        assert!(!u("ǅǅ").is_title());
        assert!(u("ǅx").is_title());
        assert!(!u("Xǅ").is_title());
    }

    #[test]
    fn roman_numeral_nl_is_still_cased_not_broken_by_lt_fix() {
        // U+2167 'Ⅷ' ROMAN NUMERAL EIGHT, general category Nl (NOT Lu), but
        // CPython treats it as cased via the Uppercase derived property:
        // 'Ⅷ'.isupper()==True, 'ⅷ'.islower()==True, 'Ⅷ'.istitle()==True.
        // Guards against a category-Lu/Ll/Lt-only reimplementation, which
        // would silently un-fix this (verified as a real regression during
        // development -- switching cased_iter to general-category-only
        // broke this exact case).
        assert!(u("Ⅷ").is_upper());
        assert!(u("ⅷ").is_lower());
        assert!(u("Ⅷ").is_title());
    }

    #[test]
    fn isalpha_rejects_nl_and_mc_bug_b() {
        // U+2167 'Ⅷ' (Nl) and U+0903 Devanagari sign visarga (Mc) are NOT
        // general category L*, so isalpha() must be False even though
        // Rust's is_alphabetic() (the Alphabetic property) would wrongly
        // say True. isalnum() on 'Ⅷ' is still True -- verified against
        // CPython -- because it IS numeric (Nl), just not alphabetic; this
        // exercises isalnum()'s OR-composition (alpha||decimal||digit||
        // numeric) rather than a single Rust is_alphanumeric() call.
        assert!(!u("Ⅷ").is_alpha());
        assert!(u("Ⅷ").is_alnum());
        assert!(!u("ः").is_alpha()); // U+0903 DEVANAGARI SIGN VISARGA (Mc)
        assert!(!u("ः").is_alnum()); // Mc, not alpha/decimal/digit/numeric
    }

    #[test]
    fn isnumeric_covers_lo_cjk_numerals() {
        // U+4E00 '一' "one" and U+5343 '千' "thousand": general category Lo,
        // not covered by Rust's is_numeric() (Nd/Nl/No only), but real
        // numpy/CPython isnumeric()==True (Numeric_Type=Numeric).
        assert!(u("一").is_numeric());
        assert!(u("千").is_numeric());
        assert!(u("一x").is_numeric() == false); // mixed with non-numeric char
        assert!(u("一一").is_numeric());
    }

    // -- new block: case conversion, strip/pad, search, edit --------------

    fn s(v: &[u32]) -> String {
        v.iter().map(|&c| char::from_u32(c).unwrap()).collect()
    }

    #[test]
    fn upper_lower_swapcase_unicode() {
        assert_eq!(s(&u("straße").to_upper().chars), "STRASSE");
        assert_eq!(s(&u("CAFÉ").to_lower().chars), "café");
        assert_eq!(s(&u("HeLLo").to_swapcase().chars), "hEllO");
        assert_eq!(s(&u("ǅ").to_swapcase().chars), "ǆ");
    }

    #[test]
    fn upper_lower_bytes_ascii_only() {
        // high byte 0xE9 ('é' in Latin-1) must NOT be case-converted.
        let elem = sbytes(&[0xE9, b'a', b'B']);
        assert_eq!(elem.to_upper().chars, vec![0xE9, b'A' as u32, b'B' as u32]);
        assert_eq!(elem.to_lower().chars, vec![0xE9, b'a' as u32, b'b' as u32]);
    }

    #[test]
    fn title_and_capitalize() {
        assert_eq!(s(&u("hello world").to_title().chars), "Hello World");
        assert_eq!(s(&u("it's a test").to_title().chars), "It'S A Test");
        assert_eq!(s(&u("foo9bar").to_title().chars), "Foo9Bar");
        assert_eq!(s(&u("ǆabc").to_title().chars), "ǅabc");
        assert_eq!(s(&u("hello WORLD").to_capitalize().chars), "Hello world");
        assert_eq!(s(&u("").to_capitalize().chars), "");
    }

    #[test]
    fn strip_family() {
        assert_eq!(s(&u("  hi  ").strip(None).chars), "hi");
        assert_eq!(s(&u("  hi  ").lstrip(None).chars), "hi  ");
        assert_eq!(s(&u("  hi  ").rstrip(None).chars), "  hi");
        let xset = u("x");
        assert_eq!(s(&u("xxhixx").strip(Some(&xset)).chars), "hi");
        assert_eq!(s(&u("xxhixx").lstrip(Some(&xset)).chars), "hixx");
        assert_eq!(s(&u("xxhixx").rstrip(Some(&xset)).chars), "xxhi");
    }

    #[test]
    fn pad_family() {
        assert_eq!(s(&u("hi").center(6, '*' as u32).chars), "**hi**");
        // 'hi'.center(5): marg=3 (odd), width=5 (odd) -> both odd -> extra
        // pad goes LEFT (confirmed via real CPython: repr('hi'.center(5))
        // == '  hi '). The previous assertion here (' hi  ', extra RIGHT)
        // encoded the same fixed-direction bias bug fixed in `center()`
        // itself this round -- corrected, not weakened.
        assert_eq!(s(&u("hi").center(5, ' ' as u32).chars), "  hi ");
        assert_eq!(s(&u("hi").ljust(6, '*' as u32).chars), "hi****");
        assert_eq!(s(&u("hi").rjust(6, '*' as u32).chars), "****hi");
        assert_eq!(s(&u("-3").zfill(5).chars), "-0003");
        assert_eq!(s(&u("7").zfill(5).chars), "00007");
        assert_eq!(s(&u("toolong").center(3, ' ' as u32).chars), "toolong");
    }

    #[test]
    fn search_family() {
        let hay = u("abcabcabc");
        assert_eq!(hay.find(&u("bc"), 2, 7), 4);
        assert_eq!(hay.find(&u("zz"), 0, 9), -1);
        assert_eq!(hay.count_sub(&u("bc"), 0, 9), 3);
        assert_eq!(hay.rfind(&u("bc"), 0, 9), 7);
        assert!(hay.startswith(&u("abc"), 0, 9));
        assert!(!hay.startswith(&u("bc"), 0, 9));
        assert!(hay.endswith(&u("abc"), 0, 9));
        assert!(hay.find(&u(""), 2, 5) == 2);
    }

    #[test]
    fn replace_and_multiply() {
        let a = u("aXbXc");
        assert_eq!(s(&a.replace(&u("X"), &u("YYY"), -1).chars), "aYYYbYYYc");
        assert_eq!(s(&a.replace(&u("X"), &u("YYY"), 1).chars), "aYYYbXc");
        assert_eq!(s(&u("aXXbXXc").replace(&u("XX"), &u("Y"), -1).chars), "aYbYc");
        assert_eq!(s(&u("ab").multiply(3).chars), "ababab");
        assert_eq!(s(&u("c").multiply(2).chars), "cc");
        assert_eq!(u("x").multiply(0).chars.len(), 0);
    }

    #[test]
    fn partition_family() {
        let (b, sep, a) = u("key=val").partition(&u("="));
        assert_eq!((s(&b.chars), s(&sep.chars), s(&a.chars)), ("key".into(), "=".into(), "val".into()));
        let (b2, sep2, a2) = u("noeq").partition(&u("="));
        assert_eq!((s(&b2.chars), s(&sep2.chars), s(&a2.chars)), ("noeq".into(), "".into(), "".into()));
        let (b3, sep3, a3) = u("noeq").rpartition(&u("="));
        assert_eq!((s(&b3.chars), s(&sep3.chars), s(&a3.chars)), ("".into(), "".into(), "noeq".into()));
    }

    #[test]
    fn encode_decode_roundtrip() {
        let elem = u("héllo");
        let enc = elem.encode("utf-8").unwrap();
        assert!(enc.is_bytes);
        assert_eq!(enc.chars.len(), 6); // 'é' is 2 bytes in utf-8
        let dec = enc.decode("utf-8").unwrap();
        assert!(!dec.is_bytes);
        assert_eq!(s(&dec.chars), "héllo");
    }
}
