//! Error types, with messages shaped to match numpy's own exception text
//! where practical, since PyO3 bindings map these onto the matching
//! Python exception type (ValueError, IndexError, TypeError).

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum IonpError {
    /// numpy: `ValueError: operands could not be broadcast together with
    /// shapes ... `
    Broadcast { shape_a: Vec<usize>, shape_b: Vec<usize> },
    /// numpy: `ValueError: cannot reshape array of size N into shape (...)`
    Reshape { size: usize, shape: Vec<usize> },
    /// numpy: `IndexError: ...`
    Index(String),
    /// numpy: `TypeError: ...`
    Type(String),
    /// numpy: `ValueError: ...` (catch-all for shape/value problems that
    /// don't fit the more specific variants above)
    Value(String),
    /// numpy: `numpy._core._exceptions._UFuncNoLoopError` (a `TypeError`
    /// subclass whose `__name__` numpy overrides to display as
    /// `UFuncTypeError`) -- raised e.g. by `.reduce`/`.accumulate` on a
    /// comparison ufunc applied to a non-bool-dtype array (comparison
    /// ufuncs only have a `(bool, bool) -> bool` reduce loop; every other
    /// dtype has no matching loop). Kept distinct from the generic `Type`
    /// variant so the PyO3 boundary (`ionp-py`) can raise numpy's own
    /// exception CLASS, not just a same-shaped message -- the differential
    /// harness matches on exact exception type, not message content.
    NoUfuncLoop { ufunc_name: String },
    /// numpy: `numpy.exceptions.AxisError` -- raised for an axis argument
    /// out of range for the array's `ndim`. Kept distinct from `Index` so
    /// the PyO3 boundary can raise numpy's REAL `AxisError` class (which
    /// subclasses both `ValueError` and `IndexError` -- a plain
    /// `IndexError`/`ValueError` would only satisfy one of the two
    /// `except` clauses a caller might use). `ndim` is `None` for the
    /// 1-argument form (`AxisError(axis)`, used when no ndim context is
    /// available), `Some(ndim)` for the 2-argument form used everywhere
    /// else here.
    AxisError { axis: isize, ndim: Option<usize> },
}

// BUG FOUND AND FIXED 2026-08-01: this joined multi-element shapes with
// `", "` (comma-space), producing e.g. `(4, 5)` -- but real numpy's own
// C-level shape formatter for these two exact messages uses a bare comma
// with NO space: `(4,5)`. Verified char-for-char against real numpy 2.5.1:
// `np.reshape(np.arange(2), (1, 3))` raises `'cannot reshape array of size
// 2 into shape (1,3)'`, and `np.zeros((3,)) + np.zeros((4,5))` raises
// `'operands could not be broadcast together with shapes (3,) (4,5) '` --
// both no-space. (This is NOT how Python's own tuple `repr()` formats a
// shape -- that would be `(4, 5)` -- these are numpy's own bespoke error
// strings.) Found via independent out-of-corpus exception-message probing
// of `reshape`, not part of the originally assigned gap list; fixing the
// shared formatter here (rather than duplicating it per call site) also
// corrects every other caller of `IonpError::Broadcast`/`::Reshape`
// (e.g. binary-op shape-mismatch errors raised from `ufunc.rs`) to match
// real numpy exactly, with no existing test anywhere in this repo
// asserting the old (wrong) comma-space wording.
fn fmt_shape(shape: &[usize]) -> String {
    if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")
        )
    }
}

impl fmt::Display for IonpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IonpError::Broadcast { shape_a, shape_b } => write!(
                f,
                "operands could not be broadcast together with shapes {} {} ",
                fmt_shape(shape_a),
                fmt_shape(shape_b)
            ),
            IonpError::Reshape { size, shape } => write!(
                f,
                "cannot reshape array of size {} into shape {}",
                size,
                fmt_shape(shape)
            ),
            IonpError::Index(msg) => write!(f, "{msg}"),
            IonpError::Type(msg) => write!(f, "{msg}"),
            IonpError::Value(msg) => write!(f, "{msg}"),
            IonpError::NoUfuncLoop { ufunc_name } => {
                write!(f, "ufunc '{ufunc_name}' did not contain a loop with signature matching types")
            }
            IonpError::AxisError { axis, ndim: Some(ndim) } => write!(
                f,
                "axis {axis} is out of bounds for array of dimension {ndim}"
            ),
            IonpError::AxisError { axis, ndim: None } => write!(f, "axis {axis} is out of bounds"),
        }
    }
}

impl std::error::Error for IonpError {}
