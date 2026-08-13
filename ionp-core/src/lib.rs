//! ionp-core: ndarray semantics, dtypes, strides, broadcasting, ufunc engine.
//!
//! This is the Rust foundation every one of the 1,169 numpy API items in
//! GOAL-ionp.md sits on. It is a real, owned, Rust-backed array
//! implementation (buffer + shape + strides + dtype + offset) — not a
//! wrapper around `numpy.ndarray`. Only `ionp-py` ever imports `pyo3` or
//! `numpy`; this crate has zero Python dependency and zero Python
//! arithmetic, by construction (there is no Python type in scope here to
//! do arithmetic with).
//!
//! Module map:
//!   - `dtype`  — the 13 scalar dtypes and `promote_dtype` (NEP 50 array
//!                promotion rules).
//!   - `buffer` — owned typed storage (`Buffer`, one `Vec<T>` variant per
//!                dtype) and elementwise casting between dtypes.
//!   - `shape`  — strides, broadcasting, and `NdIter`, the one iteration
//!                primitive the rest of the crate is built on.
//!   - `array`  — `NdArray`: views, slicing, transpose, reshape.
//!   - `ufunc`  — the ufunc engine: broadcast + promote + generic scalar
//!                kernel, shared by every binary ufunc.
//!   - `error`  — `IonpError`, numpy-shaped exception text.

pub mod array;
pub mod buffer;
pub mod chebyshev;
pub mod creation;
pub mod dtype;
pub mod emath;
pub mod error;
pub mod fft;
pub mod format;
pub mod fpe;
pub mod hermite;
pub mod hermite_e;
pub mod indexing;
pub mod laguerre;
pub mod legendre;
pub mod manip;
pub mod poly;
pub mod poly_legacy;
pub mod products;
pub mod random;
pub mod repr;
pub mod round;
pub mod setops;
pub mod shape;
pub mod sort;
pub mod stats;
pub mod strings;
pub mod ufunc;

pub use array::{NdArray, Order, SliceItem};
pub use buffer::Buffer;
pub use dtype::{promote_dtype, weak_target_dtype, DType, ScalarKind};
pub use error::IonpError;

/// Sum a slice of f64. Kept from the toolchain-proving stage: still a
/// real, minimal, standalone piece of Rust arithmetic, now dwarfed by (but
/// not superseded by, since it needs no NdArray machinery) the rest of the
/// crate below.
pub fn sum_f64(data: &[f64]) -> f64 {
    data.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_matches_expected() {
        let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert_eq!(sum_f64(&v), 45.0);
    }

    #[test]
    fn sum_empty_is_zero() {
        let v: Vec<f64> = vec![];
        assert_eq!(sum_f64(&v), 0.0);
    }
}
