//! Discrete Fourier Transform primitives (the `fft` namespace).
//!
//! Mirrors numpy 2.5.1's `numpy/fft/_pocketfft.py` and `numpy/fft/_helper.py`
//! semantics exactly: argument defaults, the `n`/`s`/`axes` crop-or-pad
//! rules, the per-direction/per-norm normalization factor, and (as far as
//! is practical for a hand-ported error path) numpy's own error text,
//! including the two textually-different "invalid norm" messages numpy
//! itself produces depending on whether the underlying transform is a
//! "forward" one (`fft`, `rfft`) or a "backward" one (`ifft`, `irfft`,
//! `hfft`, `ihfft`) -- read directly from the installed numpy source, not
//! guessed:
//!
//! - forward direction: `Invalid norm value {norm}; should be "backward","ortho" or "forward".`
//! - backward direction: `Invalid norm value {norm}; should be "backward", "ortho" or "forward".`
//!
//! (Note the extra space after the first comma in the backward-direction
//! message -- this is numpy's own inconsistency between `_raw_fft`'s
//! direct `else` branch and `_swap_direction`'s `except KeyError`, not a
//! typo introduced here.)
//!
//! ## The underlying 1-D complex transform: `rustfft`
//!
//! `rustfft` (v6.4.1) was already a pinned-but-unused workspace dependency
//! (declared in `ionp-ion`'s own `Cargo.toml` before this task,
//! zero call sites anywhere in the repo) -- see `ionp-core/Cargo.toml` for
//! the full disclosure of why it was chosen over hand-rolling a from-
//! scratch mixed-radix/Bluestein engine: correctness on arbitrary `n` (not
//! just powers of two) out of the box, pure Rust, zero Python/numpy
//! involvement, and it is not a genuinely new addition to the workspace's
//! dependency graph.
//!
//! ## Precision policy
//!
//! Every transform here is computed internally in `Complex<f64>`
//! regardless of the input dtype, then the final result is cast down to
//! numpy's own output dtype for that input dtype (see
//! `complex_result_dtype`/`real_result_dtype` below, verified empirically
//! against live numpy). This can NEVER be bit-identical to pocketfft's own
//! output in general: pocketfft is a different algorithm (specialized
//! real/complex butterworth passes, mixed float32/float64 internal
//! arithmetic for complex64 vs complex128 input) than rustfft's Bluestein/
//! mixed-radix implementation, and two different correctly-rounded-per-step
//! FFT algorithms do not, in general, accumulate rounding error identically.
//! Measured max-ULP divergence per item/dtype is reported and, where
//! justified, declared with `ulp_tolerance` in `ionp/_state/fft.py` --
//! never silently assumed.

use rustfft::FftPlanner;

use crate::array::{NdArray, Order};
use crate::buffer::{Buffer, C128};
use crate::dtype::DType;
use crate::error::IonpError;
use crate::manip::roll;

// ---------------------------------------------------------------------
// Small local helpers (axis normalization kept as a private copy here,
// per this codebase's established convention -- see `manip.rs`'s own doc
// comment on `normalize_axis` -- of small per-module copies rather than a
// shared axis-normalization module; `manip::roll` itself, a `pub fn` on a
// file owned by another task, IS reused directly below for
// `fftshift`/`ifftshift`, which is a plain call into someone else's public
// API, not a duplication of ownership).
// ---------------------------------------------------------------------

fn normalize_axis_local(axis: isize, ndim: usize) -> Result<usize, IonpError> {
    let nd = ndim as isize;
    let a = if axis < 0 { axis + nd } else { axis };
    if a < 0 || a >= nd.max(0) {
        return Err(IonpError::AxisError { axis, ndim: Some(ndim) });
    }
    Ok(a as usize)
}

fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && (r < 0) != (b < 0) {
        q - 1
    } else {
        q
    }
}

fn invalid_n(n: i64) -> IonpError {
    IonpError::Value(format!("Invalid number of FFT data points ({n}) specified."))
}

fn invalid_norm_forward(norm: &str) -> IonpError {
    IonpError::Value(format!(
        "Invalid norm value {norm}; should be \"backward\",\"ortho\" or \"forward\"."
    ))
}

fn invalid_norm_backward(norm: &str) -> IonpError {
    IonpError::Value(format!(
        "Invalid norm value {norm}; should be \"backward\", \"ortho\" or \"forward\"."
    ))
}

/// numpy's `_swap_direction`: swaps which end of a forward/backward pair
/// carries the default (unnormalized) factor. An involution on the three
/// valid tokens (`ortho` maps to itself); raises the "backward-direction"
/// (space-after-comma) invalid-norm message on anything else, matching
/// numpy raising this from `_swap_direction`'s own `except KeyError`.
fn swap_direction(norm: Option<&str>) -> Result<&'static str, IonpError> {
    match norm {
        None | Some("backward") => Ok("forward"),
        Some("ortho") => Ok("ortho"),
        Some("forward") => Ok("backward"),
        Some(other) => Err(invalid_norm_backward(other)),
    }
}

/// The multiplicative normalization factor `_raw_fft` applies after an
/// unnormalized transform (`rustfft`'s `process`, like pocketfft's raw
/// ufuncs, computes an unnormalized DFT in both directions). `is_forward`
/// selects which of numpy's two error-message spellings applies on an
/// invalid `norm`.
fn norm_factor(n: usize, norm: Option<&str>, is_forward: bool) -> Result<f64, IonpError> {
    let effective = if is_forward {
        match norm {
            None | Some("backward") | Some("ortho") | Some("forward") => norm,
            Some(other) => return Err(invalid_norm_forward(other)),
        }
    } else {
        Some(swap_direction(norm)?)
    };
    match effective {
        None | Some("backward") => Ok(1.0),
        Some("ortho") => Ok(1.0 / (n as f64).sqrt()),
        Some("forward") => Ok(1.0 / n as f64),
        Some(_) => unreachable!("validated above"),
    }
}

fn complex_result_dtype(input: DType) -> DType {
    match input {
        DType::F16 | DType::F32 | DType::C64 => DType::C64,
        _ => DType::C128,
    }
}

fn real_result_dtype(input: DType) -> DType {
    match input {
        DType::F16 => DType::F16,
        DType::F32 => DType::F32,
        DType::C64 => DType::F32,
        _ => DType::F64,
    }
}

fn crop_or_pad_c128(row: &[C128], n: usize) -> Vec<C128> {
    let take = row.len().min(n);
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&row[..take]);
    out.resize(n, C128::new(0.0, 0.0));
    out
}

fn crop_or_pad_f64(row: &[f64], n: usize) -> Vec<f64> {
    let take = row.len().min(n);
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&row[..take]);
    out.resize(n, 0.0);
    out
}

/// Permutation (numpy `transpose(axes)` convention: `axes[i]` names the
/// SOURCE axis that lands at output position `i`) that moves `axis` to
/// the end, keeping every other axis in its original relative order.
fn axes_moving_to_last(ndim: usize, axis: usize) -> Vec<isize> {
    let mut perm: Vec<isize> = (0..ndim).filter(|&i| i != axis).map(|i| i as isize).collect();
    perm.push(axis as isize);
    perm
}

fn inverse_perm(perm: &[isize], ndim: usize) -> Vec<isize> {
    let mut inv = vec![0isize; ndim];
    for (new_pos, &old_pos) in perm.iter().enumerate() {
        inv[old_pos as usize] = new_pos as isize;
    }
    inv
}

// ---------------------------------------------------------------------
// Axis-level engines
// ---------------------------------------------------------------------

/// `fft`/`ifft` on arbitrary input: cast to complex128, run the 1-D
/// complex transform along `axis` (already-normalized, `0..ndim`) for
/// every line, crop/pad each line to `n`, apply `norm_factor`.
fn complex_fft_axis(
    a: &NdArray,
    n: i64,
    axis: usize,
    norm: Option<&str>,
    is_forward: bool,
) -> Result<NdArray, IonpError> {
    if n < 1 {
        return Err(invalid_n(n));
    }
    let n = n as usize;
    let fct = norm_factor(n, norm, is_forward)?;
    let out_dtype = complex_result_dtype(a.dtype());
    let ndim = a.ndim();
    let perm = axes_moving_to_last(ndim, axis);
    let moved = a.transpose_axes(&perm)?;
    let contig = moved.cast_to(DType::C128);
    let shape = contig.shape().to_vec();
    let n_in = *shape.last().unwrap_or(&1);
    let outer: usize = shape[..shape.len().saturating_sub(1)].iter().product();
    let data: &[C128] = match contig.buffer() {
        Buffer::C128(v) => v,
        _ => unreachable!("cast_to(C128) always yields a C128 buffer"),
    };

    let mut planner = FftPlanner::<f64>::new();
    let plan = if is_forward {
        planner.plan_fft_forward(n)
    } else {
        planner.plan_fft_inverse(n)
    };
    let mut out_data: Vec<C128> = Vec::with_capacity(outer * n);
    for r in 0..outer {
        let row = &data[r * n_in..(r + 1) * n_in];
        let mut buf = crop_or_pad_c128(row, n);
        plan.process(&mut buf);
        for c in buf.iter_mut() {
            *c *= fct;
        }
        out_data.extend_from_slice(&buf);
    }

    let mut out_shape = shape.clone();
    if let Some(last) = out_shape.last_mut() {
        *last = n;
    } else {
        out_shape.push(n);
    }
    let out_arr = NdArray::from_buffer(Buffer::C128(out_data), out_shape, Order::C)?;
    let out_arr = out_arr.cast_to(out_dtype);
    let inv_perm = inverse_perm(&perm, ndim);
    out_arr.transpose_axes(&inv_perm)
}

/// `rfft`: real input, forward transform, keeps only the `n/2+1`
/// non-negative-frequency terms. `conjugate_output` implements `ihfft`
/// (see module docs / `ihfft` below) without a separate code path.
fn real_forward_fft_axis(
    a: &NdArray,
    n: i64,
    axis: usize,
    norm: Option<&str>,
    conjugate_output: bool,
) -> Result<NdArray, IonpError> {
    if n < 1 {
        return Err(invalid_n(n));
    }
    let n = n as usize;
    let fct = norm_factor(n, norm, true)?;
    let out_dtype = complex_result_dtype(a.dtype());
    let ndim = a.ndim();
    let perm = axes_moving_to_last(ndim, axis);
    let moved = a.transpose_axes(&perm)?;
    let contig = moved.cast_to(DType::F64);
    let shape = contig.shape().to_vec();
    let n_in = *shape.last().unwrap_or(&1);
    let outer: usize = shape[..shape.len().saturating_sub(1)].iter().product();
    let data: &[f64] = match contig.buffer() {
        Buffer::F64(v) => v,
        _ => unreachable!("cast_to(F64) always yields an F64 buffer"),
    };

    let n_out = n / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(n);
    let mut out_data: Vec<C128> = Vec::with_capacity(outer * n_out);
    for r in 0..outer {
        let row = &data[r * n_in..(r + 1) * n_in];
        let real_row = crop_or_pad_f64(row, n);
        let mut buf: Vec<C128> = real_row.iter().map(|&x| C128::new(x, 0.0)).collect();
        plan.process(&mut buf);
        for c in buf.iter_mut() {
            *c *= fct;
            if conjugate_output {
                *c = c.conj();
            }
        }
        out_data.extend_from_slice(&buf[..n_out]);
    }

    let mut out_shape = shape.clone();
    if let Some(last) = out_shape.last_mut() {
        *last = n_out;
    } else {
        out_shape.push(n_out);
    }
    let out_arr = NdArray::from_buffer(Buffer::C128(out_data), out_shape, Order::C)?;
    let out_arr = out_arr.cast_to(out_dtype);
    let inv_perm = inverse_perm(&perm, ndim);
    out_arr.transpose_axes(&inv_perm)
}

/// `irfft`: complex Hermitian-half input (`m = n/2+1` points), full
/// `n`-length inverse transform, real output. `conjugate_input`
/// implements `hfft` (see module docs / `hfft` below) without a separate
/// code path.
fn real_inverse_fft_axis(
    a: &NdArray,
    n: i64,
    axis: usize,
    norm: Option<&str>,
    conjugate_input: bool,
) -> Result<NdArray, IonpError> {
    if n < 1 {
        return Err(invalid_n(n));
    }
    let n = n as usize;
    let fct = norm_factor(n, norm, false)?;
    let out_dtype = real_result_dtype(a.dtype());
    let ndim = a.ndim();
    let perm = axes_moving_to_last(ndim, axis);
    let moved = a.transpose_axes(&perm)?;
    let contig = moved.cast_to(DType::C128);
    let shape = contig.shape().to_vec();
    let n_in = *shape.last().unwrap_or(&1);
    let outer: usize = shape[..shape.len().saturating_sub(1)].iter().product();
    let data: &[C128] = match contig.buffer() {
        Buffer::C128(v) => v,
        _ => unreachable!("cast_to(C128) always yields a C128 buffer"),
    };

    let m = n / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_inverse(n);
    let mut out_data: Vec<f64> = Vec::with_capacity(outer * n);
    for r in 0..outer {
        let row = &data[r * n_in..(r + 1) * n_in];
        let mut half = crop_or_pad_c128(row, m);
        if conjugate_input {
            for c in half.iter_mut() {
                *c = c.conj();
            }
        }
        let mut full = vec![C128::new(0.0, 0.0); n];
        full[..m].copy_from_slice(&half[..m]);
        for k in 1..=(n - m) {
            full[n - k] = half[k].conj();
        }
        plan.process(&mut full);
        for c in full.iter_mut() {
            *c *= fct;
        }
        out_data.extend(full.iter().map(|c| c.re));
    }

    let mut out_shape = shape.clone();
    if let Some(last) = out_shape.last_mut() {
        *last = n;
    } else {
        out_shape.push(n);
    }
    let out_arr = NdArray::from_buffer(Buffer::F64(out_data), out_shape, Order::C)?;
    let out_arr = out_arr.cast_to(out_dtype);
    let inv_perm = inverse_perm(&perm, ndim);
    out_arr.transpose_axes(&inv_perm)
}

// ---------------------------------------------------------------------
// Public 1-D API (mirrors numpy.fft.{fft,ifft,rfft,irfft,hfft,ihfft})
// ---------------------------------------------------------------------

pub fn fft(a: &NdArray, n: Option<i64>, axis: isize, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axis_n = normalize_axis_local(axis, a.ndim())?;
    let n = n.unwrap_or(a.shape()[axis_n] as i64);
    complex_fft_axis(a, n, axis_n, norm, true)
}

pub fn ifft(a: &NdArray, n: Option<i64>, axis: isize, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axis_n = normalize_axis_local(axis, a.ndim())?;
    let n = n.unwrap_or(a.shape()[axis_n] as i64);
    complex_fft_axis(a, n, axis_n, norm, false)
}

pub fn rfft(a: &NdArray, n: Option<i64>, axis: isize, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axis_n = normalize_axis_local(axis, a.ndim())?;
    let n = n.unwrap_or(a.shape()[axis_n] as i64);
    real_forward_fft_axis(a, n, axis_n, norm, false)
}

pub fn irfft(a: &NdArray, n: Option<i64>, axis: isize, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axis_n = normalize_axis_local(axis, a.ndim())?;
    let n = n.unwrap_or((a.shape()[axis_n] as i64 - 1) * 2);
    real_inverse_fft_axis(a, n, axis_n, norm, false)
}

pub fn hfft(a: &NdArray, n: Option<i64>, axis: isize, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axis_n = normalize_axis_local(axis, a.ndim())?;
    let n = n.unwrap_or((a.shape()[axis_n] as i64 - 1) * 2);
    let new_norm = swap_direction(norm)?;
    real_inverse_fft_axis(a, n, axis_n, Some(new_norm), true)
}

pub fn ihfft(a: &NdArray, n: Option<i64>, axis: isize, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axis_n = normalize_axis_local(axis, a.ndim())?;
    let n = n.unwrap_or(a.shape()[axis_n] as i64);
    let new_norm = swap_direction(norm)?;
    real_forward_fft_axis(a, n, axis_n, Some(new_norm), true)
}

// ---------------------------------------------------------------------
// N-D cook-args (`_cook_nd_args` port) and the N-D/2-D API
// ---------------------------------------------------------------------

/// Port of numpy's `_cook_nd_args`. Returns `(s, axes)` where `s[i]` is
/// the resolved length to use for `axes[i]` (still possibly out of range
/// -- validated lazily by each per-axis 1-D call, exactly like numpy).
/// The deprecated "`s` may contain `None` entries" numpy 2.0 compatibility
/// shim is deliberately NOT implemented (see module docs: an explicit,
/// documented non-goal, not a silent gap) -- callers passing a `None`
/// inside `s` get a `NotImplementedError` from the `ionp-py` binding layer
/// before this function is ever reached.
fn cook_nd_args(
    a: &NdArray,
    s: Option<&[i64]>,
    axes: Option<&[isize]>,
    invreal: bool,
) -> Result<(Vec<i64>, Vec<isize>), IonpError> {
    let shapeless = s.is_none();
    let mut s_vec: Vec<i64> = match s {
        Some(sv) => sv.to_vec(),
        None => match axes {
            None => a.shape().iter().map(|&d| d as i64).collect(),
            Some(ax) => {
                let mut v = Vec::with_capacity(ax.len());
                for &axi in ax {
                    let normed = normalize_axis_local(axi, a.ndim())?;
                    v.push(a.shape()[normed] as i64);
                }
                v
            }
        },
    };

    let axes_vec: Vec<isize> = match axes {
        Some(ax) => ax.to_vec(),
        None => {
            let len = s_vec.len() as isize;
            (0..s_vec.len()).map(|i| -len + i as isize).collect()
        }
    };

    if s_vec.len() != axes_vec.len() {
        return Err(IonpError::Value("Shape and axes have different lengths.".to_string()));
    }

    if invreal && shapeless {
        if let Some(&last_axis) = axes_vec.last() {
            let normed = normalize_axis_local(last_axis, a.ndim())?;
            let last_len = a.shape()[normed] as i64;
            if let Some(last_s) = s_vec.last_mut() {
                *last_s = (last_len - 1) * 2;
            }
        }
    }

    for (i, &axi) in axes_vec.iter().enumerate() {
        if s_vec[i] == -1 {
            let normed = normalize_axis_local(axi, a.ndim())?;
            s_vec[i] = a.shape()[normed] as i64;
        }
    }

    Ok((s_vec, axes_vec))
}

fn raw_fftnd_complex(
    a: &NdArray,
    s: Option<&[i64]>,
    axes: Option<&[isize]>,
    is_forward: bool,
    norm: Option<&str>,
) -> Result<NdArray, IonpError> {
    let (s, axes) = cook_nd_args(a, s, axes, false)?;
    let mut cur = a.clone();
    for i in (0..axes.len()).rev() {
        cur = complex_fft_axis(&cur, s[i], normalize_axis_local(axes[i], cur.ndim())?, norm, is_forward)?;
    }
    Ok(cur)
}

pub fn fftn(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    raw_fftnd_complex(a, s, axes, true, norm)
}

pub fn ifftn(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    raw_fftnd_complex(a, s, axes, false, norm)
}

const DEFAULT_2D_AXES: [isize; 2] = [-2, -1];

pub fn fft2(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axes = axes.unwrap_or(&DEFAULT_2D_AXES);
    raw_fftnd_complex(a, s, Some(axes), true, norm)
}

pub fn ifft2(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axes = axes.unwrap_or(&DEFAULT_2D_AXES);
    raw_fftnd_complex(a, s, Some(axes), false, norm)
}

fn rfftn_core(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let (s, axes) = cook_nd_args(a, s, axes, false)?;
    let last = axes.len() - 1;
    let mut cur = rfft(a, Some(s[last]), axes[last], norm)?;
    for i in (0..last).rev() {
        cur = fft(&cur, Some(s[i]), axes[i], norm)?;
    }
    Ok(cur)
}

pub fn rfftn(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    rfftn_core(a, s, axes, norm)
}

pub fn rfft2(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axes = axes.unwrap_or(&DEFAULT_2D_AXES);
    rfftn_core(a, s, Some(axes), norm)
}

fn irfftn_core(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let (s, axes) = cook_nd_args(a, s, axes, true)?;
    let last = axes.len() - 1;
    let mut cur = a.clone();
    for i in 0..last {
        cur = ifft(&cur, Some(s[i]), axes[i], norm)?;
    }
    irfft(&cur, Some(s[last]), axes[last], norm)
}

pub fn irfftn(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    irfftn_core(a, s, axes, norm)
}

pub fn irfft2(a: &NdArray, s: Option<&[i64]>, axes: Option<&[isize]>, norm: Option<&str>) -> Result<NdArray, IonpError> {
    let axes = axes.unwrap_or(&DEFAULT_2D_AXES);
    irfftn_core(a, s, Some(axes), norm)
}

// ---------------------------------------------------------------------
// fftshift / ifftshift / fftfreq / rfftfreq
// ---------------------------------------------------------------------

pub fn fftshift(a: &NdArray, axes: Option<&[isize]>) -> Result<NdArray, IonpError> {
    let (axes_vec, shifts): (Vec<isize>, Vec<isize>) = match axes {
        None => {
            let ax: Vec<isize> = (0..a.ndim() as isize).collect();
            let sh: Vec<isize> = a.shape().iter().map(|&d| (d / 2) as isize).collect();
            (ax, sh)
        }
        Some(axs) => {
            let mut sh = Vec::with_capacity(axs.len());
            for &axi in axs {
                let normed = normalize_axis_local(axi, a.ndim())?;
                sh.push((a.shape()[normed] / 2) as isize);
            }
            (axs.to_vec(), sh)
        }
    };
    roll(a, &shifts, Some(&axes_vec))
}

pub fn ifftshift(a: &NdArray, axes: Option<&[isize]>) -> Result<NdArray, IonpError> {
    let (axes_vec, shifts): (Vec<isize>, Vec<isize>) = match axes {
        None => {
            let ax: Vec<isize> = (0..a.ndim() as isize).collect();
            let sh: Vec<isize> = a.shape().iter().map(|&d| -((d / 2) as isize)).collect();
            (ax, sh)
        }
        Some(axs) => {
            let mut sh = Vec::with_capacity(axs.len());
            for &axi in axs {
                let normed = normalize_axis_local(axi, a.ndim())?;
                sh.push(-((a.shape()[normed] / 2) as isize));
            }
            (axs.to_vec(), sh)
        }
    };
    roll(a, &shifts, Some(&axes_vec))
}

pub fn fftfreq(n: i64, d: f64) -> Result<NdArray, IonpError> {
    if n < 0 {
        return Err(IonpError::Value("negative dimensions are not allowed".to_string()));
    }
    let val = 1.0 / (n as f64 * d);
    let big_n = floor_div(n - 1, 2) + 1;
    let neg_start = -floor_div(n, 2);
    let mut vals: Vec<f64> = Vec::with_capacity(n as usize);
    for k in 0..big_n {
        vals.push(k as f64 * val);
    }
    for k in neg_start..0 {
        vals.push(k as f64 * val);
    }
    NdArray::from_buffer(Buffer::F64(vals), vec![n as usize], Order::C)
}

/// Unlike `fftfreq` (which allocates via a Python-equivalent of
/// `numpy.empty(n, int)` and therefore genuinely raises for negative `n`),
/// numpy's `rfftfreq` builds its result via `numpy.arange(0, n // 2 + 1)`
/// with no allocation-by-size step at all -- `arange(0, N)` for a
/// non-positive `N` is simply an empty range, not an error, in real numpy.
/// Verified directly against live numpy 2.5.1 and by reading
/// `inspect.getsource(numpy.fft.rfftfreq)`: `rfftfreq(-3, 1.0)`,
/// `rfftfreq(-1, 1.0)`, and `rfftfreq(-8, 1.0)` all return `array([],
/// dtype=float64)`, never raise. Reproduced here by clamping `big_n` to
/// `>= 0` before building the output, rather than rejecting negative `n`
/// the way `fftfreq` correctly does above.
pub fn rfftfreq(n: i64, d: f64) -> Result<NdArray, IonpError> {
    let val = 1.0 / (n as f64 * d);
    let big_n = (floor_div(n, 2) + 1).max(0);
    let vals: Vec<f64> = (0..big_n).map(|k| k as f64 * val).collect();
    NdArray::from_buffer(Buffer::F64(vals), vec![big_n as usize], Order::C)
}
