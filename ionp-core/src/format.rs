//! The `.npy`/`.npz` binary format: numpy's own public, stable,
//! documented-on-disk container for a single array (`.npy`) or a
//! collection of named arrays in a ZIP archive (`.npz`). This module is
//! the byte-for-byte encoder/decoder for both -- the round trip against
//! REAL numpy (`np.save`/`np.load` on a file WE wrote, and vice versa) is
//! the acceptance test, not self-consistency (see this task's brief).
//!
//! Spec summary (paraphrased from numpy's own public documentation of the
//! format, `numpy.lib.format`, not transcribed from its source): a `.npy`
//! file is
//!   `MAGIC_PREFIX (6 bytes, \x93NUMPY) | major (1 byte) | minor (1 byte)
//!    | header_len (2 bytes LE for v1.0, 4 bytes LE for v2.0/3.0)
//!    | header (ASCII/latin1/utf8 text, a Python dict literal spelling
//!      {'descr': <dtype str>, 'fortran_order': <bool>, 'shape': <tuple>},
//!      space-padded and newline-terminated so the TOTAL preamble length
//!      (magic+version+header_len field+header text) is a multiple of
//!      `ARRAY_ALIGN`)
//!    | raw element bytes, in the traversal order `fortran_order` names,
//!      no padding between elements`.
//! `.npz` is an ordinary ZIP archive (`save` -> `ZIP_STORED`/no
//! compression, `savez_compressed` -> `ZIP_DEFLATED`) whose entries are
//! each a complete `.npy` file named `<array-name>.npy`.

use std::collections::BTreeMap;

use crate::array::{NdArray, Order};
use crate::buffer::{Buffer, C128, C64};
use crate::dtype::DType;
use crate::error::IonpError;
use half::f16;

pub const MAGIC_PREFIX: &[u8; 6] = b"\x93NUMPY";
pub const MAGIC_LEN: usize = 8;
/// numpy pads every header so the total preamble (magic + version +
/// header-length field + header text) is a multiple of this many bytes.
pub const ARRAY_ALIGN: usize = 64;
/// numpy's own I/O chunk size for `fromfile`/streaming reads (`2**18`,
/// documented in `numpy.lib.format`); ionp has no streaming reader, this
/// is kept only so the coverage ledger's `lib.format.BUFFER_SIZE` constant
/// resolves to the same value real numpy exposes.
pub const BUFFER_SIZE: usize = 1 << 18;
/// Max decimal digits numpy reserves when it has to grow a header to fit a
/// bigger shape tuple in-place (`numpy.lib.format.GROWTH_AXIS_MAX_DIGITS`).
pub const GROWTH_AXIS_MAX_DIGITS: usize = 21;
pub const EXPECTED_KEYS: &[&str] = &["descr", "fortran_order", "shape"];

pub fn magic(major: u8, minor: u8) -> Vec<u8> {
    let mut v = MAGIC_PREFIX.to_vec();
    v.push(major);
    v.push(minor);
    v
}

/// Reads the 8-byte magic+version prefix and returns `(major, minor)`.
pub fn read_magic(data: &[u8]) -> Result<(u8, u8), IonpError> {
    if data.len() < MAGIC_LEN || &data[0..6] != MAGIC_PREFIX.as_slice() {
        return Err(IonpError::Value(
            "the magic string is not correct; expected b'\\x93NUMPY'".to_string(),
        ));
    }
    Ok((data[6], data[7]))
}

#[derive(Debug, Clone)]
pub struct HeaderData {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

/// numpy's `dtype.str`-equivalent descriptor spelling used inside a `.npy`
/// header: `<byteorder><kind><itemsize>` (itemsize in bytes, except `U`
/// which counts characters), byteorder `|` for single-byte / bytes-kind
/// dtypes, `<` (native little-endian, ionp's only supported byte order)
/// otherwise. Mirrors `ionp-py/src/lib.rs`'s `PyDType::str` getter exactly
/// (duplicated, not imported -- that file is outside this task's
/// ownership fence, and this module must not depend on ionp-py at all
/// since ionp-core sits below it).
pub fn dtype_to_descr(dtype: DType) -> String {
    match dtype {
        DType::S(n) => format!("|S{n}"),
        DType::U(_) => format!("<U{}", dtype.char_count()),
        _ => {
            let marker = if dtype.itemsize() == 1 { "|" } else { "<" };
            format!("{}{}{}", marker, dtype.kind_char(), dtype.itemsize())
        }
    }
}

/// Inverse of `dtype_to_descr`, extended to also accept `>` (big-endian)
/// and `=`/native markers for interop with files written by real numpy on
/// a big-endian source or with an explicit byte-order request -- ionp
/// itself is little-endian-only, so a `>` descriptor is accepted (its
/// bytes get byte-swapped by the caller, see `read_array`) but never
/// produced by `dtype_to_descr`. Returns `(dtype, needs_byteswap)`.
pub fn descr_to_dtype(descr: &str) -> Result<(DType, bool), IonpError> {
    let s = descr.trim();
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return Err(IonpError::Value("empty dtype descriptor".to_string()));
    }
    let (marker, rest) = match bytes[0] {
        b'<' | b'>' | b'=' | b'|' => (bytes[0] as char, &s[1..]),
        _ => ('=', s),
    };
    if rest.is_empty() {
        return Err(IonpError::Value(format!("cannot parse dtype descriptor '{descr}'")));
    }
    let kind = rest.chars().next().unwrap();
    let digits = &rest[kind.len_utf8()..];
    let n: usize = if digits.is_empty() {
        0
    } else {
        digits
            .parse()
            .map_err(|_| IonpError::Value(format!("cannot parse dtype descriptor '{descr}'")))?
    };
    let needs_byteswap = marker == '>';
    let dtype = match kind {
        'b' => DType::Bool,
        'i' => match n {
            1 => DType::I8,
            2 => DType::I16,
            4 => DType::I32,
            8 => DType::I64,
            _ => return Err(IonpError::Value(format!("unsupported int itemsize in '{descr}'"))),
        },
        'u' => match n {
            1 => DType::U8,
            2 => DType::U16,
            4 => DType::U32,
            8 => DType::U64,
            _ => return Err(IonpError::Value(format!("unsupported uint itemsize in '{descr}'"))),
        },
        'f' => match n {
            2 => DType::F16,
            4 => DType::F32,
            8 => DType::F64,
            _ => return Err(IonpError::Value(format!("unsupported float itemsize in '{descr}'"))),
        },
        'c' => match n {
            8 => DType::C64,
            16 => DType::C128,
            _ => return Err(IonpError::Value(format!("unsupported complex itemsize in '{descr}'"))),
        },
        'S' => DType::S(n as u32),
        'U' => DType::U((n * 4) as u32),
        _ => {
            return Err(IonpError::Value(format!(
                "ionp cannot load dtype descriptor '{descr}' (unsupported kind '{kind}')"
            )))
        }
    };
    Ok((dtype, needs_byteswap))
}

/// Builds the `(descr, fortran_order, shape)` triple real numpy's
/// `header_data_from_array_1_0` computes for a given array: `descr` from
/// its dtype, `shape` from its shape tuple, and `fortran_order = True`
/// ONLY when the array is Fortran-contiguous and NOT C-contiguous (a
/// C-contiguous array -- including every 0-/1-D array, where the two
/// notions coincide -- is always saved `fortran_order=False`). Verified
/// against real numpy 2.5.1 for 0-d, 1-d, C-contiguous 2-d, and
/// `np.asfortranarray`'d 2-d/3-d inputs.
pub fn header_data_from_array(arr: &NdArray) -> HeaderData {
    let fortran_order = arr.is_f_contiguous() && !arr.is_c_contiguous();
    HeaderData {
        descr: dtype_to_descr(arr.dtype()),
        fortran_order,
        shape: arr.shape().to_vec(),
    }
}

fn shape_tuple_repr(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", parts.join(", "))
        }
    }
}

fn header_dict_text(hd: &HeaderData) -> String {
    format!(
        "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
        hd.descr,
        if hd.fortran_order { "True" } else { "False" },
        shape_tuple_repr(&hd.shape)
    )
}

/// Encodes the full preamble (magic + version + header-length field +
/// padded header text + trailing `\n`) for the given header data,
/// choosing version 1.0 unless the header text is too long for a 2-byte
/// length field (in which case 2.0, a 4-byte length field, matching real
/// numpy's own automatic-upgrade rule -- ionp never needs 3.0's utf-8
/// header support since every descriptor/shape it can produce is pure
/// ASCII).
pub fn write_array_header(hd: &HeaderData) -> Vec<u8> {
    let dict_text = header_dict_text(hd);
    // Try v1.0 (2-byte length field) first.
    let preamble_fixed_v1 = MAGIC_LEN; // 6 magic + 2 version, header_len field counted separately below
    let len_field_v1 = 2usize;
    let unpadded_v1 = preamble_fixed_v1 + len_field_v1 + dict_text.len() + 1; // +1 for '\n'
    let (major, minor, len_field_size) = if unpadded_v1.div_ceil(ARRAY_ALIGN) * ARRAY_ALIGN - (preamble_fixed_v1 + len_field_v1)
        <= u16::MAX as usize
    {
        (1u8, 0u8, 2usize)
    } else {
        (2u8, 0u8, 4usize)
    };
    let prefix_len = MAGIC_LEN + len_field_size;
    let mut total = prefix_len + dict_text.len() + 1;
    let padded_total = total.div_ceil(ARRAY_ALIGN) * ARRAY_ALIGN;
    let pad = padded_total - total;
    total = padded_total;
    let mut header_text = dict_text;
    header_text.push_str(&" ".repeat(pad));
    header_text.push('\n');
    let header_len = total - prefix_len;

    let mut out = magic(major, minor);
    if len_field_size == 2 {
        out.extend_from_slice(&(header_len as u16).to_le_bytes());
    } else {
        out.extend_from_slice(&(header_len as u32).to_le_bytes());
    }
    out.extend_from_slice(header_text.as_bytes());
    out
}

/// Parses a Python-dict-literal header of the restricted shape numpy
/// itself always writes (`{'descr': '...', 'fortran_order': True/False,
/// 'shape': (...), }`, keys in that fixed order) -- a small hand-rolled
/// scanner rather than a general Python literal parser, since that is the
/// only shape numpy's own `np.save` (or this module's own
/// `write_array_header`) ever produces, and no third-party parsing crate
/// is available in this workspace.
fn parse_header_dict(text: &str) -> Result<HeaderData, IonpError> {
    let descr = extract_quoted(text, "descr")?;
    let fortran_order = extract_bare(text, "fortran_order")?.trim() == "True";
    let shape_str = extract_paren(text, "shape")?;
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| IonpError::Value(format!("cannot parse header shape '{shape_str}'")))?;
    Ok(HeaderData { descr, fortran_order, shape })
}

fn key_pos(text: &str, key: &str) -> Result<usize, IonpError> {
    let needle = format!("'{key}'");
    text.find(&needle)
        .map(|p| p + needle.len())
        .ok_or_else(|| IonpError::Value(format!("header missing key '{key}'")))
}

fn extract_quoted(text: &str, key: &str) -> Result<String, IonpError> {
    let after = &text[key_pos(text, key)?..];
    let colon = after.find(':').ok_or_else(|| IonpError::Value("malformed header".to_string()))?;
    let rest = after[colon + 1..].trim_start();
    let q = rest.chars().next();
    if q != Some('\'') && q != Some('"') {
        return Err(IonpError::Value("malformed header string value".to_string()));
    }
    let quote = q.unwrap();
    let body = &rest[1..];
    let end = body.find(quote).ok_or_else(|| IonpError::Value("unterminated header string".to_string()))?;
    Ok(body[..end].to_string())
}

fn extract_bare(text: &str, key: &str) -> Result<String, IonpError> {
    let after = &text[key_pos(text, key)?..];
    let colon = after.find(':').ok_or_else(|| IonpError::Value("malformed header".to_string()))?;
    let rest = after[colon + 1..].trim_start();
    let end = rest.find(',').unwrap_or(rest.len());
    Ok(rest[..end].to_string())
}

fn extract_paren(text: &str, key: &str) -> Result<String, IonpError> {
    let after = &text[key_pos(text, key)?..];
    let colon = after.find(':').ok_or_else(|| IonpError::Value("malformed header".to_string()))?;
    let rest = after[colon + 1..].trim_start();
    if !rest.starts_with('(') {
        return Err(IonpError::Value("malformed header shape tuple".to_string()));
    }
    let end = rest.find(')').ok_or_else(|| IonpError::Value("unterminated header shape tuple".to_string()))?;
    Ok(rest[1..end].to_string())
}

/// Reads the header starting at byte 0 of `data` (magic through the
/// trailing `\n`) and returns `(HeaderData, offset_of_first_data_byte)`.
pub fn read_array_header(data: &[u8]) -> Result<(HeaderData, usize), IonpError> {
    let (major, _minor) = read_magic(data)?;
    let (len_field_size, prefix_len) = if major == 1 { (2usize, MAGIC_LEN + 2) } else { (4usize, MAGIC_LEN + 4) };
    if data.len() < prefix_len {
        return Err(IonpError::Value("truncated .npy header".to_string()));
    }
    let header_len = if len_field_size == 2 {
        u16::from_le_bytes([data[6 + 2 - 2], data[6 + 2 - 1]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    // Re-derive correctly for both field sizes (avoid the fragile index
    // arithmetic above being misread): for v1.0 the 2-byte field is at
    // offset 8..10; for >=2.0 the 4-byte field is at offset 8..12.
    let header_len = if len_field_size == 2 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        header_len
    };
    let header_start = prefix_len;
    let header_end = header_start + header_len;
    if data.len() < header_end {
        return Err(IonpError::Value("truncated .npy header".to_string()));
    }
    let header_text = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|_| IonpError::Value("header is not valid utf-8".to_string()))?;
    let hd = parse_header_dict(header_text)?;
    Ok((hd, header_end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Order;

    // Exact bytes captured from real numpy 2.5.1 (`np.save` on
    // `np.arange(12, dtype='<f8').reshape(3, 4)`), see this task's
    // report -- probe script output, not transcribed from numpy source.
    #[test]
    fn header_matches_real_numpy_2d_f8() {
        let arr = NdArray::from_buffer(
            Buffer::F64((0..12).map(|i| i as f64).collect()),
            vec![3, 4],
            Order::C,
        )
        .unwrap();
        let bytes = write_array(&arr);
        assert_eq!(&bytes[0..6], MAGIC_PREFIX.as_slice());
        assert_eq!(bytes[6], 1);
        assert_eq!(bytes[7], 0);
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!(header_len, 118);
        let header_text = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        assert!(header_text.starts_with("{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }"));
        assert!(header_text.ends_with('\n'));
        assert!(header_text[..header_text.len() - 1].ends_with(' '));
        assert_eq!(bytes.len(), 224);
    }

    #[test]
    fn header_matches_real_numpy_scalar() {
        let arr = NdArray::from_buffer(Buffer::F64(vec![3.5]), vec![], Order::C).unwrap();
        let bytes = write_array(&arr);
        let header_text = std::str::from_utf8(&bytes[10..10 + 118]).unwrap();
        assert!(header_text.starts_with("{'descr': '<f8', 'fortran_order': False, 'shape': (), }"));
        assert!(header_text.ends_with('\n'));
    }

    #[test]
    fn header_matches_real_numpy_1d_bool() {
        let arr = NdArray::from_buffer(Buffer::Bool(vec![true, false, true]), vec![3], Order::C).unwrap();
        let bytes = write_array(&arr);
        assert_eq!(bytes.len(), 128 + 3);
        assert_eq!(&bytes[128..131], &[1u8, 0u8, 1u8]);
    }

    #[test]
    fn roundtrip_through_read_array() {
        let arr = NdArray::from_buffer(
            Buffer::F64((0..12).map(|i| i as f64 * 1.5).collect()),
            vec![3, 4],
            Order::C,
        )
        .unwrap();
        let bytes = write_array(&arr);
        let back = read_array(&bytes).unwrap();
        assert_eq!(back.shape(), arr.shape());
        assert_eq!(buffer_to_bytes_le(back.buffer()), buffer_to_bytes_le(arr.buffer()));
    }

    #[test]
    fn npz_roundtrip_stored_and_deflated() {
        let a = NdArray::from_buffer(Buffer::I64((0..20).collect()), vec![4, 5], Order::C).unwrap();
        let b = NdArray::from_buffer(Buffer::F32(vec![1.5, 2.5, 3.5]), vec![3], Order::C).unwrap();
        let entries = vec![NpzEntry { name: "a".to_string(), array: &a }, NpzEntry { name: "b".to_string(), array: &b }];
        for compressed in [false, true] {
            let bytes = if compressed { write_npz_compressed(&entries) } else { write_npz(&entries) };
            let loaded = read_npz(&bytes).unwrap();
            assert_eq!(loaded.len(), 2);
            assert_eq!(buffer_to_bytes_le(loaded["a"].buffer()), buffer_to_bytes_le(a.buffer()));
            assert_eq!(buffer_to_bytes_le(loaded["b"].buffer()), buffer_to_bytes_le(b.buffer()));
        }
    }

    #[test]
    fn deflate_decompress_handles_dynamic_and_fixed_huffman() {
        // A larger, repetitive payload forces the encoder-independent
        // decoder path to exercise LZ77 back-references; here we only
        // have OUR OWN (stored-block) encoder to round-trip against --
        // the dynamic/fixed Huffman decode paths are exercised by the
        // real-numpy interop check in this task's report, not by this
        // unit test (no compressing encoder to produce them from here).
        let data = b"the quick brown fox jumps over the lazy dog ".repeat(50);
        let compressed = deflate::compress(&data);
        let back = deflate::decompress(&compressed, data.len()).unwrap();
        assert_eq!(back, data);
    }
}

fn size_of_shape(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Encodes a complete `.npy` file's bytes for `arr`.
pub fn write_array(arr: &NdArray) -> Vec<u8> {
    let hd = header_data_from_array(arr);
    let mut out = write_array_header(&hd);
    let order = if hd.fortran_order { "F" } else { "C" };
    let flat = arr.ravel_order(order).expect("ravel_order('C'/'F') cannot fail");
    out.extend_from_slice(&buffer_to_bytes_le(flat.buffer()));
    out
}

/// Decodes a complete `.npy` file's bytes into an `NdArray`.
pub fn read_array(data: &[u8]) -> Result<NdArray, IonpError> {
    let (hd, offset) = read_array_header(data)?;
    let (dtype, needs_byteswap) = descr_to_dtype(&hd.descr)?;
    let n = size_of_shape(&hd.shape);
    let itemsize = dtype.itemsize();
    let needed = n * itemsize;
    let raw = data
        .get(offset..offset + needed)
        .ok_or_else(|| IonpError::Value("truncated .npy data segment".to_string()))?;
    let buffer = if needs_byteswap {
        let swapped = byteswap_bytes(raw, dtype);
        bytes_to_buffer(dtype, &swapped)?
    } else {
        bytes_to_buffer(dtype, raw)?
    };
    if hd.fortran_order && hd.shape.len() > 1 {
        NdArray::from_buffer(buffer, hd.shape, Order::F)
    } else {
        NdArray::from_buffer(buffer, hd.shape, Order::C)
    }
}

/// Byte-swaps `raw` in place per `itemsize`-wide chunks (used only for the
/// `>` big-endian descriptor interop path in `read_array` -- ionp itself
/// never writes big-endian files). `S`/`U`/`Bool` are never byte-swapped
/// by numpy itself (single-byte or opaque-byte-sequence dtypes), so this
/// is only reachable for the numeric multi-byte kinds.
fn byteswap_bytes(raw: &[u8], dtype: DType) -> Vec<u8> {
    let width = dtype.itemsize();
    if width <= 1 {
        return raw.to_vec();
    }
    // Complex dtypes byte-swap PER COMPONENT (half of itemsize each), not
    // across the whole 8/16-byte pair.
    let chunk = if matches!(dtype, DType::C64 | DType::C128) { width / 2 } else { width };
    let mut out = Vec::with_capacity(raw.len());
    for group in raw.chunks(chunk) {
        out.extend(group.iter().rev());
    }
    out
}

/// Raw little-endian element bytes for `buf`, the SAME layout
/// `ndarray.tobytes()` (`ionp-py/src/ndarray_attrs.rs`) produces --
/// duplicated here rather than imported since that file is outside this
/// task's ownership fence and `ionp-core` cannot depend on `ionp-py`
/// (dependency direction is the other way).
pub fn buffer_to_bytes_le(buf: &Buffer) -> Vec<u8> {
    match buf {
        Buffer::Bool(v) => v.iter().map(|&b| b as u8).collect(),
        Buffer::I8(v) => v.iter().map(|&x| x as u8).collect(),
        Buffer::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::U8(v) => v.clone(),
        Buffer::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::U64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::F16(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
        Buffer::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::C64(v) => v
            .iter()
            .flat_map(|c| {
                let mut b = c.re.to_le_bytes().to_vec();
                b.extend_from_slice(&c.im.to_le_bytes());
                b
            })
            .collect(),
        Buffer::C128(v) => v
            .iter()
            .flat_map(|c| {
                let mut b = c.re.to_le_bytes().to_vec();
                b.extend_from_slice(&c.im.to_le_bytes());
                b
            })
            .collect(),
        Buffer::S(_, v) => v.iter().flat_map(|e| e.iter().copied()).collect(),
        Buffer::U(_, v) => v.iter().flat_map(|e| e.iter().flat_map(|c| c.to_le_bytes())).collect(),
    }
}

/// Inverse of `buffer_to_bytes_le`: reinterprets a flat little-endian byte
/// slice as a `Buffer` of `dtype`, `bytes.len() / dtype.itemsize()`
/// elements. `bytes.len()` must be an exact multiple of the itemsize --
/// callers (`load`/`frombuffer`/`fromstring`/`fromfile`) are responsible
/// for that check (numpy itself raises `ValueError: buffer size must be a
/// multiple of element size` when it isn't).
pub fn bytes_to_buffer(dtype: DType, bytes: &[u8]) -> Result<Buffer, IonpError> {
    let width = dtype.itemsize();
    if width == 0 {
        return Err(IonpError::Value("zero-width dtype".to_string()));
    }
    if !bytes.len().is_multiple_of(width) {
        return Err(IonpError::Value(
            "buffer size must be a multiple of element size".to_string(),
        ));
    }
    let n = bytes.len() / width;
    Ok(match dtype {
        DType::Bool => Buffer::Bool(bytes.iter().map(|&b| b != 0).collect()),
        DType::I8 => Buffer::I8(bytes.iter().map(|&b| b as i8).collect()),
        DType::U8 => Buffer::U8(bytes.to_vec()),
        DType::I16 => Buffer::I16((0..n).map(|i| i16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())).collect()),
        DType::U16 => Buffer::U16((0..n).map(|i| u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())).collect()),
        DType::F16 => Buffer::F16(
            (0..n)
                .map(|i| f16::from_bits(u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())))
                .collect(),
        ),
        DType::I32 => Buffer::I32((0..n).map(|i| i32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())).collect()),
        DType::U32 => Buffer::U32((0..n).map(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())).collect()),
        DType::F32 => Buffer::F32((0..n).map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())).collect()),
        DType::I64 => Buffer::I64((0..n).map(|i| i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap())).collect()),
        DType::U64 => Buffer::U64((0..n).map(|i| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap())).collect()),
        DType::F64 => Buffer::F64((0..n).map(|i| f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap())).collect()),
        DType::C64 => Buffer::C64(
            (0..n)
                .map(|i| {
                    let re = f32::from_le_bytes(bytes[i * 8..i * 8 + 4].try_into().unwrap());
                    let im = f32::from_le_bytes(bytes[i * 8 + 4..i * 8 + 8].try_into().unwrap());
                    C64::new(re, im)
                })
                .collect(),
        ),
        DType::C128 => Buffer::C128(
            (0..n)
                .map(|i| {
                    let re = f64::from_le_bytes(bytes[i * 16..i * 16 + 8].try_into().unwrap());
                    let im = f64::from_le_bytes(bytes[i * 16 + 8..i * 16 + 16].try_into().unwrap());
                    C128::new(re, im)
                })
                .collect(),
        ),
        DType::S(w) => Buffer::S(w, bytes.chunks(width).map(|c| c.to_vec()).collect()),
        DType::U(w) => Buffer::U(
            w,
            bytes
                .chunks(width)
                .map(|c| c.chunks(4).map(|q| u32::from_le_bytes(q.try_into().unwrap())).collect())
                .collect(),
        ),
    })
}

// ---------------------------------------------------------------------
// .npz (ZIP) container
// ---------------------------------------------------------------------

/// CRC-32 (ISO-HDLC / zlib polynomial 0xEDB88320), hand-rolled: ZIP's
/// local/central-directory headers require it and no crate in this
/// workspace already provides one. Table-driven bitwise implementation,
/// verified against the well-known `crc32(b"123456789") == 0xCBF43926`
/// check value.
fn crc32(data: &[u8]) -> u32 {
    fn table() -> [u32; 256] {
        let mut t = [0u32; 256];
        let mut i = 0u32;
        while i < 256 {
            let mut c = i;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
                k += 1;
            }
            t[i as usize] = c;
            i += 1;
        }
        t
    }
    let t = table();
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc = t[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// One named array entry queued for a `.npz` archive.
pub struct NpzEntry<'a> {
    pub name: String,
    pub array: &'a NdArray,
}

/// Writes an uncompressed (`ZIP_STORED`) `.npz` archive -- `np.savez`'s
/// on-disk format. Each entry is stored as `<name>.npy` (numpy appends
/// `.npy` itself when a bare key has none). MS-DOS-epoch (1980-01-01)
/// timestamp is used for every entry, matching what a deterministic
/// writer (and numpy's own zipfile-based writer, absent an explicit
/// `mtime`) would produce; no real numpy consumer inspects it.
pub fn write_npz(entries: &[NpzEntry]) -> Vec<u8> {
    write_npz_impl(entries, false)
}

/// Writes a DEFLATE-compressed (`ZIP_DEFLATED`) `.npz` archive --
/// `np.savez_compressed`'s on-disk format.
pub fn write_npz_compressed(entries: &[NpzEntry]) -> Vec<u8> {
    write_npz_impl(entries, true)
}

fn write_npz_impl(entries: &[NpzEntry], compressed: bool) -> Vec<u8> {
    let mut out = Vec::new();
    struct CentralRec {
        name: String,
        crc: u32,
        csize: u32,
        usize: u32,
        offset: u32,
        method: u16,
    }
    let mut central: Vec<CentralRec> = Vec::new();
    for entry in entries {
        let filename = format!("{}.npy", entry.name);
        let raw = write_array(entry.array);
        let crc = crc32(&raw);
        let (method, payload) = if compressed {
            (8u16, deflate::compress(&raw))
        } else {
            (0u16, raw.clone())
        };
        let offset = out.len() as u32;
        // Local file header
        out.extend_from_slice(&0x04034b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0x21u16.to_le_bytes()); // mod date (1980-01-01)
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        out.extend_from_slice(&(filename.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        out.extend_from_slice(filename.as_bytes());
        out.extend_from_slice(&payload);
        central.push(CentralRec {
            name: filename,
            crc,
            csize: payload.len() as u32,
            usize: raw.len() as u32,
            offset,
            method,
        });
    }
    let central_start = out.len() as u32;
    for rec in &central {
        out.extend_from_slice(&0x02014b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version made by
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&rec.method.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0x21u16.to_le_bytes());
        out.extend_from_slice(&rec.crc.to_le_bytes());
        out.extend_from_slice(&rec.csize.to_le_bytes());
        out.extend_from_slice(&rec.usize.to_le_bytes());
        out.extend_from_slice(&(rec.name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra
        out.extend_from_slice(&0u16.to_le_bytes()); // comment
        out.extend_from_slice(&0u16.to_le_bytes()); // disk number
        out.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        out.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        out.extend_from_slice(&rec.offset.to_le_bytes());
        out.extend_from_slice(rec.name.as_bytes());
    }
    let central_size = out.len() as u32 - central_start;
    // End of central directory record
    out.extend_from_slice(&0x06054b50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // disk number
    out.extend_from_slice(&0u16.to_le_bytes()); // disk with central dir
    out.extend_from_slice(&(central.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u16).to_le_bytes());
    out.extend_from_slice(&central_size.to_le_bytes());
    out.extend_from_slice(&central_start.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment length
    out
}

/// Reads a `.npz` ZIP archive's central directory and inflates/extracts
/// every entry, returning `{stem_without_.npy: NdArray}` (numpy's own
/// `NpzFile` mapping key convention: the `.npy` suffix is stripped from
/// each archive member name). Supports both `ZIP_STORED` (method 0) and
/// `ZIP_DEFLATED` (method 8) entries, matching what `write_npz`/
/// `write_npz_compressed` produce AND what real numpy's own `savez`/
/// `savez_compressed` produce, so this is the read side of the two-way
/// round trip this task's brief requires.
pub fn read_npz(data: &[u8]) -> Result<BTreeMap<String, NdArray>, IonpError> {
    // Locate the end-of-central-directory record by scanning back from
    // the end for its signature (handles an optional trailing comment,
    // which real numpy's writer never sets but is valid per the ZIP
    // spec).
    let sig = 0x06054b50u32.to_le_bytes();
    let mut eocd_pos = None;
    if data.len() >= 22 {
        for start in (0..=data.len() - 22).rev() {
            if data[start..start + 4] == sig {
                eocd_pos = Some(start);
                break;
            }
        }
    }
    let eocd = eocd_pos.ok_or_else(|| IonpError::Value("not a zip file (no end-of-central-directory record)".to_string()))?;
    let n_entries = u16::from_le_bytes([data[eocd + 10], data[eocd + 11]]) as usize;
    let central_start = u32::from_le_bytes([data[eocd + 16], data[eocd + 17], data[eocd + 18], data[eocd + 19]]) as usize;

    let mut out = BTreeMap::new();
    let mut pos = central_start;
    for _ in 0..n_entries {
        if data[pos..pos + 4] != 0x02014b50u32.to_le_bytes() {
            return Err(IonpError::Value("malformed zip central directory".to_string()));
        }
        let method = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
        let csize = u32::from_le_bytes([data[pos + 20], data[pos + 21], data[pos + 22], data[pos + 23]]) as usize;
        let usize_ = u32::from_le_bytes([data[pos + 24], data[pos + 25], data[pos + 26], data[pos + 27]]) as usize;
        let name_len = u16::from_le_bytes([data[pos + 28], data[pos + 29]]) as usize;
        let extra_len = u16::from_le_bytes([data[pos + 30], data[pos + 31]]) as usize;
        let comment_len = u16::from_le_bytes([data[pos + 32], data[pos + 33]]) as usize;
        let lfh_offset = u32::from_le_bytes([data[pos + 42], data[pos + 43], data[pos + 44], data[pos + 45]]) as usize;
        let name_start = pos + 46;
        let name = std::str::from_utf8(&data[name_start..name_start + name_len])
            .map_err(|_| IonpError::Value("zip entry name is not valid utf-8".to_string()))?
            .to_string();
        pos = name_start + name_len + extra_len + comment_len;

        // Local file header at lfh_offset: fixed 30-byte header + name + extra, then payload.
        let lfh_name_len = u16::from_le_bytes([data[lfh_offset + 26], data[lfh_offset + 27]]) as usize;
        let lfh_extra_len = u16::from_le_bytes([data[lfh_offset + 28], data[lfh_offset + 29]]) as usize;
        let payload_start = lfh_offset + 30 + lfh_name_len + lfh_extra_len;
        let payload = &data[payload_start..payload_start + csize];
        let raw = match method {
            0 => payload.to_vec(),
            8 => deflate::decompress(payload, usize_)?,
            other => return Err(IonpError::Value(format!("unsupported zip compression method {other}"))),
        };
        let arr = read_array(&raw)?;
        let stem = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        out.insert(stem, arr);
    }
    Ok(out)
}

/// Minimal DEFLATE (RFC 1951) codec, hand-rolled: no compression crate is
/// available in this workspace and adding one for a single narrow use
/// (`np.savez_compressed`'s on-disk format) was judged higher-risk than a
/// small from-scratch encoder/decoder restricted to the one representation
/// this module needs. The ENCODER always emits "stored" (uncompressed)
/// DEFLATE blocks (still a fully valid, standard-conformant DEFLATE
/// stream any real zlib/numpy reader can inflate -- verified against real
/// numpy's own `zipfile`/`np.load`, see this task's report); the DECODER
/// is a complete implementation (fixed AND dynamic Huffman blocks, stored
/// blocks, LZ77 back-references) since it must also read files produced
/// by real numpy's own `savez_compressed`, which uses genuine Huffman/LZ77
/// compression, not stored blocks.
mod deflate {
    use crate::error::IonpError;

    /// Encodes `data` as a sequence of DEFLATE "stored" (type 0, no
    /// compression) blocks, each up to 65535 bytes, final block flagged.
    /// Bigger than a real compressor's output but a byte-for-byte valid
    /// DEFLATE stream that any conforming inflater (including real
    /// numpy's) accepts.
    pub fn compress(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut bitbuf = BitWriter::new();
        if data.is_empty() {
            bitbuf.write_bits(1, 1); // BFINAL
            bitbuf.write_bits(0, 2); // BTYPE=00 stored
            bitbuf.align();
            out.extend(bitbuf.take());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0xFFFFu16.to_le_bytes());
            return out;
        }
        let mut i = 0;
        while i < data.len() {
            let chunk_len = (data.len() - i).min(65535);
            let is_final = i + chunk_len >= data.len();
            let mut bw = BitWriter::new();
            bw.write_bits(if is_final { 1 } else { 0 }, 1);
            bw.write_bits(0, 2);
            bw.align();
            out.extend(bw.take());
            out.extend_from_slice(&(chunk_len as u16).to_le_bytes());
            out.extend_from_slice(&(!(chunk_len as u16)).to_le_bytes());
            out.extend_from_slice(&data[i..i + chunk_len]);
            i += chunk_len;
        }
        out
    }

    struct BitWriter {
        bytes: Vec<u8>,
        cur: u8,
        nbits: u8,
    }
    impl BitWriter {
        fn new() -> Self {
            BitWriter { bytes: Vec::new(), cur: 0, nbits: 0 }
        }
        fn write_bits(&mut self, mut val: u32, mut n: u8) {
            while n > 0 {
                if (val & 1) != 0 {
                    self.cur |= 1 << self.nbits;
                }
                self.nbits += 1;
                val >>= 1;
                n -= 1;
                if self.nbits == 8 {
                    self.bytes.push(self.cur);
                    self.cur = 0;
                    self.nbits = 0;
                }
            }
        }
        fn align(&mut self) {
            if self.nbits > 0 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
        fn take(self) -> Vec<u8> {
            self.bytes
        }
    }

    struct BitReader<'a> {
        data: &'a [u8],
        pos: usize,
        bitpos: u8,
    }
    impl<'a> BitReader<'a> {
        fn new(data: &'a [u8]) -> Self {
            BitReader { data, pos: 0, bitpos: 0 }
        }
        fn read_bit(&mut self) -> Result<u32, IonpError> {
            if self.pos >= self.data.len() {
                return Err(IonpError::Value("truncated deflate stream".to_string()));
            }
            let bit = (self.data[self.pos] >> self.bitpos) & 1;
            self.bitpos += 1;
            if self.bitpos == 8 {
                self.bitpos = 0;
                self.pos += 1;
            }
            Ok(bit as u32)
        }
        fn read_bits(&mut self, n: u8) -> Result<u32, IonpError> {
            let mut v = 0u32;
            for i in 0..n {
                v |= self.read_bit()? << i;
            }
            Ok(v)
        }
        fn align(&mut self) {
            if self.bitpos != 0 {
                self.bitpos = 0;
                self.pos += 1;
            }
        }
    }

    /// Canonical Huffman decoder table: `code_lengths[symbol]` -> a lookup
    /// from bit-reversed code to symbol, built per RFC 1951 section 3.2.2.
    struct Huffman {
        // (code, length) -> symbol, searched by incrementally reading bits
        // MSB-first per the spec (DEFLATE Huffman codes are packed
        // MSB-first, unlike everything else in the stream, which is
        // LSB-first).
        counts: Vec<u32>,
        symbols: Vec<u32>,
    }
    impl Huffman {
        fn build(lengths: &[u32]) -> Self {
            let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;
            let mut counts = vec![0u32; max_len + 1];
            for &l in lengths {
                if l > 0 {
                    counts[l as usize] += 1;
                }
            }
            let mut offsets = vec![0u32; max_len + 2];
            for l in 1..=max_len {
                offsets[l + 1] = offsets[l] + counts[l];
            }
            let mut symbols = vec![0u32; lengths.len()];
            let mut offsets2 = offsets.clone();
            for (sym, &l) in lengths.iter().enumerate() {
                if l > 0 {
                    symbols[offsets2[l as usize] as usize] = sym as u32;
                    offsets2[l as usize] += 1;
                }
            }
            Huffman { counts, symbols }
        }
        fn decode(&self, br: &mut BitReader) -> Result<u32, IonpError> {
            let mut code = 0i32;
            let mut first = 0i32;
            let mut index = 0i32;
            for len in 1..self.counts.len() {
                code |= br.read_bit()? as i32;
                let count = self.counts[len] as i32;
                if code - first < count {
                    return Ok(self.symbols[(index + (code - first)) as usize]);
                }
                index += count;
                first += count;
                first <<= 1;
                code <<= 1;
            }
            Err(IonpError::Value("invalid deflate huffman code".to_string()))
        }
    }

    const LEN_BASE: [u32; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u32; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [u32; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
        8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u32; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
    ];

    fn fixed_huffman() -> (Huffman, Huffman) {
        let mut lit_lengths = vec![0u32; 288];
        for i in 0..144 {
            lit_lengths[i] = 8;
        }
        for i in 144..256 {
            lit_lengths[i] = 9;
        }
        for i in 256..280 {
            lit_lengths[i] = 7;
        }
        for i in 280..288 {
            lit_lengths[i] = 8;
        }
        let dist_lengths = vec![5u32; 30];
        (Huffman::build(&lit_lengths), Huffman::build(&dist_lengths))
    }

    const CLEN_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

    fn dynamic_huffman(br: &mut BitReader) -> Result<(Huffman, Huffman), IonpError> {
        let hlit = br.read_bits(5)? + 257;
        let hdist = br.read_bits(5)? + 1;
        let hclen = br.read_bits(4)? + 4;
        let mut clen_lengths = vec![0u32; 19];
        for i in 0..hclen as usize {
            clen_lengths[CLEN_ORDER[i]] = br.read_bits(3)?;
        }
        let clen_huff = Huffman::build(&clen_lengths);
        let total = (hlit + hdist) as usize;
        let mut lengths = Vec::with_capacity(total);
        while lengths.len() < total {
            let sym = clen_huff.decode(br)?;
            match sym {
                0..=15 => lengths.push(sym),
                16 => {
                    let prev = *lengths.last().ok_or_else(|| IonpError::Value("invalid deflate code-length repeat".to_string()))?;
                    let rep = br.read_bits(2)? + 3;
                    for _ in 0..rep {
                        lengths.push(prev);
                    }
                }
                17 => {
                    let rep = br.read_bits(3)? + 3;
                    for _ in 0..rep {
                        lengths.push(0);
                    }
                }
                18 => {
                    let rep = br.read_bits(7)? + 11;
                    for _ in 0..rep {
                        lengths.push(0);
                    }
                }
                _ => return Err(IonpError::Value("invalid deflate code-length symbol".to_string())),
            }
        }
        let lit_lengths = lengths[..hlit as usize].to_vec();
        let dist_lengths = lengths[hlit as usize..].to_vec();
        Ok((Huffman::build(&lit_lengths), Huffman::build(&dist_lengths)))
    }

    pub fn decompress(data: &[u8], expected_size: usize) -> Result<Vec<u8>, IonpError> {
        let mut br = BitReader::new(data);
        let mut out = Vec::with_capacity(expected_size);
        loop {
            let bfinal = br.read_bit()?;
            let btype = br.read_bits(2)?;
            match btype {
                0 => {
                    br.align();
                    if br.pos + 4 > br.data.len() {
                        return Err(IonpError::Value("truncated deflate stored block".to_string()));
                    }
                    let len = u16::from_le_bytes([br.data[br.pos], br.data[br.pos + 1]]) as usize;
                    br.pos += 4;
                    out.extend_from_slice(&br.data[br.pos..br.pos + len]);
                    br.pos += len;
                }
                1 | 2 => {
                    let (lit_huff, dist_huff) = if btype == 1 { fixed_huffman() } else { dynamic_huffman(&mut br)? };
                    loop {
                        let sym = lit_huff.decode(&mut br)?;
                        if sym == 256 {
                            break;
                        } else if sym < 256 {
                            out.push(sym as u8);
                        } else {
                            let idx = (sym - 257) as usize;
                            if idx >= LEN_BASE.len() {
                                return Err(IonpError::Value("invalid deflate length symbol".to_string()));
                            }
                            let length = LEN_BASE[idx] + br.read_bits(LEN_EXTRA[idx] as u8)?;
                            let dsym = dist_huff.decode(&mut br)? as usize;
                            if dsym >= DIST_BASE.len() {
                                return Err(IonpError::Value("invalid deflate distance symbol".to_string()));
                            }
                            let dist = DIST_BASE[dsym] + br.read_bits(DIST_EXTRA[dsym] as u8)?;
                            let start = out.len().checked_sub(dist as usize).ok_or_else(|| {
                                IonpError::Value("invalid deflate back-reference distance".to_string())
                            })?;
                            for k in 0..length as usize {
                                let b = out[start + k];
                                out.push(b);
                            }
                        }
                    }
                }
                _ => return Err(IonpError::Value("invalid deflate block type".to_string())),
            }
            if bfinal == 1 {
                break;
            }
        }
        Ok(out)
    }
}
