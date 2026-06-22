//! NVRTC kernel source + the per-gate uniform structs the host passes to the
//! kernels by value. Each struct's byte layout MUST match the matching CUDA
//! `struct` in `kernels.cu`; `const _: () = assert!(size_of == …)` pins it.

use cudarc::driver::DeviceRepr;

/// CUDA C++ source for the FP64 state-vector kernels, compiled once at
/// backend construction via NVRTC.
pub(crate) const SV_KERNELS_SRC: &str = include_str!("kernels.cu");

/// Entry-point names (declared `extern "C"`, so NVRTC keeps them unmangled).
pub(crate) const APPLY_1Q: &str = "apply_1q";
pub(crate) const APPLY_1Q_MULTI: &str = "apply_1q_multi";
pub(crate) const APPLY_KQ: &str = "apply_kq";

/// Hard ceiling on the single-qubit-gate batch [`apply_1q_multi`] applies in one
/// sweep — the `Multi1q` 32-entry local block (2^5). Sizes the struct slots and
/// the `with_layer_batch` clamp; not the production default (see
/// [`DEFAULT_LAYER_BATCH`]).
pub(crate) const MAX_LAYER_BATCH: usize = 5;

/// Production batch width for [`crate::CudaSvBackend::run_layered`]. **3** is the
/// measured sweet spot of `apply_1q_multi`: the P5.9-03 A/B bench (n=28) found
/// b=2/3/4 tie at the top (≤1.52×) but b=5 collapses to ~1.0× — its strided
/// 2^5 gather + register spill erase the fewer-passes win, the same wall the
/// P5.9-02b k-sweep hit at k=5.
pub(crate) const DEFAULT_LAYER_BATCH: usize = 3;

/// Per-gate uniform for `apply_1q`. Matches the CUDA `Gate1q` struct:
/// `cplx m[4]` (row-major 2×2: m00 m01 m10 m11) then four `u32`. `m` is stored
/// as 8 interleaved `f64` (re, im, re, im, …) so the struct is plain-old-data
/// with no `Complex` field (keeps the `DeviceRepr` impl trivially sound).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Gate1qParams {
    pub m: [f64; 8],
    pub target: u32,
    pub t_bit: u32,
    pub ctrl_mask: u32,
    pub _pad: u32,
}

// SAFETY: `Gate1qParams` is `#[repr(C)]`, contains only POD scalar fields, has
// no padding bytes (8×f64 = 64, then 4×u32 = 16; 80 total, 8-byte-aligned), and
// every bit pattern is a valid value. That is exactly cudarc's `DeviceRepr`
// contract for a by-value kernel argument.
unsafe impl DeviceRepr for Gate1qParams {}

// cplx m[4] = 64 bytes + 4×u32 = 16 ⇒ 80, matching CUDA's `Gate1q`.
const _: () = assert!(core::mem::size_of::<Gate1qParams>() == 80);

/// Per-gate uniform for `apply_kq`. Matches the CUDA `GateKq` struct (12×`u32`,
/// 48 bytes, no padding). `qbit[j]` is the global state-index bit for
/// matrix-index bit `j` (the host fills it MSB-first — see `gate_kq_params`);
/// `sorted` = target positions ascending for zero-bit insertion; slots `j >= k`
/// are zero and never read by the kernel.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GateKqParams {
    pub k: u32,
    pub qbit: [u32; 5],
    pub sorted: [u32; 5],
    pub ctrl_mask: u32,
}

// SAFETY: same contract as `Gate1qParams` — `#[repr(C)]`, all-`u32` POD, no
// padding (12×4 = 48), every bit pattern valid.
unsafe impl DeviceRepr for GateKqParams {}

// 1 + 5 + 5 + 1 = 12 u32 = 48 bytes, matching CUDA's `GateKq`.
const _: () = assert!(core::mem::size_of::<GateKqParams>() == 48);

/// Per-batch uniform for `apply_1q_multi` (P5.9-03). Matches the CUDA `Multi1q`
/// struct: `cplx mats[20]` (5 gates × 2×2 interleaved `f64`), then `m`, the
/// ascending `sorted[5]` target positions, and two padding `u32`. `mats[j*8..]`
/// holds the 2×2 of the gate on qubit `sorted[j]`; slots `j >= m` are unread.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Multi1qParams {
    pub mats: [f64; 40],
    pub m: u32,
    pub sorted: [u32; 5],
    pub _pad: [u32; 2],
}

// SAFETY: `#[repr(C)]`, POD scalar fields, no padding bytes (40×f64 = 320 at
// offset 0, then 8×u32 = 32 contiguous ⇒ 352 total, 8-byte-aligned), every bit
// pattern valid — cudarc's `DeviceRepr` contract for a by-value kernel arg.
unsafe impl DeviceRepr for Multi1qParams {}

// 40×f64 (320) + 8×u32 (32) = 352 bytes, matching CUDA's `Multi1q`.
const _: () = assert!(core::mem::size_of::<Multi1qParams>() == 352);
