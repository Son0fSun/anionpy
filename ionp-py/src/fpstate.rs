//! The floating-point-error subsystem's Python-facing half: per-thread
//! `seterr`/`geterr`/`errstate`/`seterrcall`/`geterrcall` state, and what
//! to DO once `ionp_core::fpe` has already decided which category (if any)
//! fired for a given op -- warn, raise, print, or call/log a registrant.
//! `ionp_core::fpe` (pure Rust, no Python) answers "did it fire"; this
//! module (the only place in the crate allowed to touch `pyo3::exceptions`/
//! `warnings`/a Python callback) answers "now what".
//!
//! State is genuinely PER-THREAD (`std::thread_local!`), matching numpy's
//! own measured behavior (spec doc section 4.4) -- CPython threads map 1:1
//! onto OS threads under the GIL, so a plain Rust `thread_local!` gets this
//! property for free, no extra bookkeeping needed.
//!
//! SCOPE (this task, see its scope-control directive): implements all five
//! entry points and all six modes (`ignore`/`warn`/`raise`/`print`/`call`/
//! `log`), `errstate` as both context manager and decorator, and nesting.
//! `'print'` mode's real behavior (fd-2 stderr, bypassing `sys.stderr`) is
//! implemented via `PySys_WriteStderr`-equivalent (`eprint!` on the
//! process's real stderr, which for a still-attached Python process is the
//! same fd-2 numpy's own C `fprintf` targets) -- the differential corpus
//! itself does not exercise `'print'` (declared out of scope in its own
//! module doc, "needs OS-level fd redirection... not attempted"), so this
//! is implemented for completeness/honesty but is NOT corpus-verified.

use std::cell::RefCell;
use std::ffi::CString;

use pyo3::exceptions::{PyFloatingPointError, PyRuntimeWarning, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ionp_core::fpe::{FpCategory, ALL_CATEGORIES};

// ---------------------------------------------------------------------------
// Mode + per-thread state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Ignore,
    Warn,
    Raise,
    Print,
    Call,
    Log,
}

impl Mode {
    fn parse(s: &str) -> Option<Mode> {
        match s {
            "ignore" => Some(Mode::Ignore),
            "warn" => Some(Mode::Warn),
            "raise" => Some(Mode::Raise),
            "print" => Some(Mode::Print),
            "call" => Some(Mode::Call),
            "log" => Some(Mode::Log),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Mode::Ignore => "ignore",
            Mode::Warn => "warn",
            Mode::Raise => "raise",
            Mode::Print => "print",
            Mode::Call => "call",
            Mode::Log => "log",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ErrState {
    divide: Mode,
    over: Mode,
    under: Mode,
    invalid: Mode,
}

/// Measured default (spec doc section 6): three `'warn'`, `under` alone
/// `'ignore'`.
const DEFAULT_STATE: ErrState = ErrState { divide: Mode::Warn, over: Mode::Warn, under: Mode::Ignore, invalid: Mode::Warn };

impl ErrState {
    fn get(&self, cat: FpCategory) -> Mode {
        match cat {
            FpCategory::Divide => self.divide,
            FpCategory::Over => self.over,
            FpCategory::Under => self.under,
            FpCategory::Invalid => self.invalid,
        }
    }
    fn set(&mut self, cat: FpCategory, mode: Mode) {
        match cat {
            FpCategory::Divide => self.divide = mode,
            FpCategory::Over => self.over = mode,
            FpCategory::Under => self.under = mode,
            FpCategory::Invalid => self.invalid = mode,
        }
    }
}

thread_local! {
    static STATE: RefCell<ErrState> = const { RefCell::new(DEFAULT_STATE) };
    /// `seterrcall`'s registrant. `None` is the measured default
    /// (`geterrcall()` on an untouched thread, spec doc section 4).
    static CALLBACK: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
}

fn current_state() -> ErrState {
    STATE.with(|s| *s.borrow())
}

fn state_to_dict<'py>(py: Python<'py>, st: ErrState) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for cat in ALL_CATEGORIES {
        d.set_item(cat.key(), st.get(cat).name())?;
    }
    Ok(d)
}

// ---------------------------------------------------------------------------
// seterr / geterr
// ---------------------------------------------------------------------------

fn parse_mode_kwarg(name: &str, val: &Bound<'_, PyAny>) -> PyResult<Mode> {
    let s: String = val.extract().map_err(|_| PyTypeError::new_err(format!(
        "seterr() argument '{name}' must be str"
    )))?;
    Mode::parse(&s).ok_or_else(|| PyValueError::new_err(format!("invalid error mode {s:?}")))
}

/// `numpy.seterr(divide=None, over=None, under=None, invalid=None,
/// all=None) -> dict`. Unspecified (`None`) categories are left unchanged.
/// `all=<mode>` is a shorthand that sets every category to the same mode
/// (applied first, then any per-category kwarg overrides it -- matches
/// numpy's own measured behavior of `all` being a base layer, not an
/// exclusive alternative). Returns the PRE-call state, a fresh `dict` --
/// mutating the returned dict never affects internal state (it is a brand
/// new `PyDict`, not a live view).
#[pyfunction]
#[pyo3(signature = (divide=None, over=None, under=None, invalid=None, all=None))]
pub fn seterr<'py>(
    py: Python<'py>,
    divide: Option<Bound<'py, PyAny>>,
    over: Option<Bound<'py, PyAny>>,
    under: Option<Bound<'py, PyAny>>,
    invalid: Option<Bound<'py, PyAny>>,
    all: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let previous = current_state();
    let mut new_state = previous;
    if let Some(a) = &all {
        let m = parse_mode_kwarg("all", a)?;
        for cat in ALL_CATEGORIES {
            new_state.set(cat, m);
        }
    }
    if let Some(v) = &divide {
        new_state.divide = parse_mode_kwarg("divide", v)?;
    }
    if let Some(v) = &over {
        new_state.over = parse_mode_kwarg("over", v)?;
    }
    if let Some(v) = &under {
        new_state.under = parse_mode_kwarg("under", v)?;
    }
    if let Some(v) = &invalid {
        new_state.invalid = parse_mode_kwarg("invalid", v)?;
    }
    STATE.with(|s| *s.borrow_mut() = new_state);
    state_to_dict(py, previous)
}

/// `numpy.geterr() -> dict`, always all four keys.
#[pyfunction]
pub fn geterr(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    state_to_dict(py, current_state())
}

/// `numpy.seterrcall(func_or_log_object) -> previous`.
#[pyfunction]
pub fn seterrcall(py: Python<'_>, handler: Option<Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let previous = CALLBACK.with(|c| c.borrow().as_ref().map(|p| p.clone_ref(py)));
    CALLBACK.with(|c| *c.borrow_mut() = handler.map(|h| h.unbind()));
    Ok(previous.unwrap_or_else(|| py.None()))
}

/// `numpy.geterrcall() -> current registrant or None`.
#[pyfunction]
pub fn geterrcall(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(CALLBACK.with(|c| c.borrow().as_ref().map(|p| p.clone_ref(py))).unwrap_or_else(|| py.None()))
}

// ---------------------------------------------------------------------------
// The action dispatcher -- called after `ionp_core::fpe` has already
// decided a category fired. `ufunc_name` is the exact text numpy's own
// long-form message names (`"divide"`, `"floor_divide"`, `"remainder"`,
// `"multiply"`, `"sqrt"`, `"log"`, `"cast"`).
// ---------------------------------------------------------------------------

fn long_message(cat: FpCategory, ufunc_name: &str) -> String {
    format!("{} encountered in {}", cat.short_phrase(), ufunc_name)
}

/// Executes the current thread's configured mode for `cat`, having already
/// been told (by the caller, via `ionp_core::fpe`'s pure detection
/// functions) that `cat` fired for the operation named `ufunc_name`.
/// `stacklevel`: passed straight through to `PyErr::warn` for `'warn'`
/// mode -- see call sites for the exact value each one measured/matched
/// against real numpy (spec doc section 2's stacklevel finding).
pub fn signal(py: Python<'_>, cat: FpCategory, ufunc_name: &str, stacklevel: u32) -> PyResult<()> {
    let mode = STATE.with(|s| s.borrow().get(cat));
    match mode {
        Mode::Ignore => Ok(()),
        Mode::Warn => {
            let msg = long_message(cat, ufunc_name);
            let c_msg = CString::new(msg).expect("no interior NUL");
            let warn_cat = py.get_type::<PyRuntimeWarning>();
            PyErr::warn(py, &warn_cat, c_msg.as_c_str(), stacklevel as i32)
        }
        Mode::Raise => {
            let msg = long_message(cat, ufunc_name);
            Err(PyFloatingPointError::new_err(msg))
        }
        Mode::Print => {
            // Real numpy writes directly to fd-2, bypassing `sys.stderr`
            // (spec doc section 4.3). `eprintln!` targets the process's
            // real stderr file descriptor, which is the closest a Rust
            // extension can get to the same C-level `fprintf` numpy uses,
            // without a Python-level `sys.stderr` redirect being able to
            // intercept it -- matching the measured "not `sys.stderr`"
            // property. Not corpus-verified (see module doc).
            let msg = long_message(cat, ufunc_name);
            eprintln!("Warning: {msg}");
            Ok(())
        }
        Mode::Call => {
            let cb = CALLBACK.with(|c| c.borrow().as_ref().map(|p| p.clone_ref(py)));
            match cb {
                Some(f) => {
                    let _: Py<PyAny> = f.call1(py, (cat.short_phrase(), cat.bit() as i64))?;
                    Ok(())
                }
                // Measured live: numpy's own behavior with NO registrant
                // under 'call' mode was not part of this task's
                // measurement pass; the analogous 'log' no-registrant case
                // (below) IS measured (NameError) -- 'call' with no
                // registrant is treated the same way on the strength of
                // sharing the same `seterrcall` registrant slot, not on a
                // separate live measurement.
                None => Err(pyo3::exceptions::PyNameError::new_err(format!(
                    "call specified for {} (in {}) but no callable function found",
                    cat.short_phrase(),
                    ufunc_name
                ))),
            }
        }
        Mode::Log => {
            let cb = CALLBACK.with(|c| c.borrow().as_ref().map(|p| p.clone_ref(py)));
            match cb {
                Some(obj) => {
                    let msg = format!("Warning: {}\n", long_message(cat, ufunc_name));
                    let _: Py<PyAny> = obj.call_method1(py, "write", (msg,))?;
                    Ok(())
                }
                None => Err(pyo3::exceptions::PyNameError::new_err(format!(
                    "log specified for {} (in {}) but no object with write method found.",
                    cat.short_phrase(),
                    ufunc_name
                ))),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `errstate`: context manager + decorator.
// ---------------------------------------------------------------------------

/// `numpy.errstate(**kwargs)`. Constructing an instance has zero effect on
/// its own (spec doc section 4, "merely constructing...has zero effect") --
/// only `__enter__`/`__exit__` (directly, or via `with`/the decorator
/// wrapper below) touch the thread-local state. `_saved` holds the state to
/// restore on `__exit__`, `None` when not currently entered (so double
/// `__exit__` or an `__exit__` with no matching `__enter__` is a no-op
/// rather than corrupting state).
#[pyclass(name = "errstate", module = "anionpy")]
pub struct ErrState_ {
    divide: Option<Mode>,
    over: Option<Mode>,
    under: Option<Mode>,
    invalid: Option<Mode>,
    // `std::sync::Mutex`, not `RefCell`: `#[pyclass]` requires `Send + Sync`
    // (a Python object can in principle be handed across threads even
    // though this crate's own use of it stays single-threaded under the
    // GIL), and `RefCell` is `!Sync`. A `Mutex` costs nothing extra here --
    // `__enter__`/`__exit__` never contend (they're always called from the
    // same thread that holds the GIL for this object at that moment).
    saved: std::sync::Mutex<Option<ErrState>>,
}

#[pymethods]
impl ErrState_ {
    #[new]
    #[pyo3(signature = (divide=None, over=None, under=None, invalid=None, all=None))]
    fn new(
        divide: Option<Bound<'_, PyAny>>,
        over: Option<Bound<'_, PyAny>>,
        under: Option<Bound<'_, PyAny>>,
        invalid: Option<Bound<'_, PyAny>>,
        all: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let all_mode = match &all {
            Some(a) => Some(parse_mode_kwarg("all", a)?),
            None => None,
        };
        let pick = |v: Option<Bound<'_, PyAny>>, name: &str| -> PyResult<Option<Mode>> {
            match v {
                Some(x) => Ok(Some(parse_mode_kwarg(name, &x)?)),
                None => Ok(all_mode),
            }
        };
        Ok(ErrState_ {
            divide: pick(divide, "divide")?,
            over: pick(over, "over")?,
            under: pick(under, "under")?,
            invalid: pick(invalid, "invalid")?,
            saved: std::sync::Mutex::new(None),
        })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        let previous = current_state();
        let mut new_state = previous;
        if let Some(m) = slf.divide {
            new_state.divide = m;
        }
        if let Some(m) = slf.over {
            new_state.over = m;
        }
        if let Some(m) = slf.under {
            new_state.under = m;
        }
        if let Some(m) = slf.invalid {
            new_state.invalid = m;
        }
        *slf.saved.lock().unwrap() = Some(previous);
        STATE.with(|s| *s.borrow_mut() = new_state);
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<bool> {
        if let Some(prev) = self.saved.lock().unwrap().take() {
            STATE.with(|s| *s.borrow_mut() = prev);
        }
        Ok(false)
    }

    /// Decorator form: `@errstate(divide='raise') def f(): ...`. Returns a
    /// plain Python closure (built once via `PyModule::from_code`, same
    /// idiom `errors.rs`'s `AxisError` fallback uses for hand-written
    /// Python glue) that wraps `func` with an `__enter__`/`__exit__` pair
    /// around each CALL -- not just once at decoration time -- matching
    /// numpy's measured behavior ("state is restored to whatever it was
    /// before the call once the decorated function returns, exception or
    /// not", spec doc section 4). `functools.wraps` preserves `func`'s
    /// `__name__`/`__doc__` on the returned wrapper.
    fn __call__<'py>(slf: Bound<'py, Self>, func: Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let helper = decorate_helper(py)?;
        Ok(helper.call1((slf, func))?.unbind())
    }
}

fn decorate_helper<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    static SRC: &str = "\
import functools
def _ionp_errstate_decorate(es, func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with es:
            return func(*args, **kwargs)
    return wrapper
";
    let code = CString::new(SRC).expect("no interior NUL");
    let module = pyo3::types::PyModule::from_code(
        py,
        code.as_c_str(),
        c"_ionp_errstate_decorate.py",
        c"_ionp_errstate_decorate",
    )?;
    module.getattr("_ionp_errstate_decorate")
}

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<ErrState_>()?;
    m.add_function(wrap_pyfunction!(seterr, m)?)?;
    m.add_function(wrap_pyfunction!(geterr, m)?)?;
    m.add_function(wrap_pyfunction!(seterrcall, m)?)?;
    m.add_function(wrap_pyfunction!(geterrcall, m)?)?;
    Ok(())
}
