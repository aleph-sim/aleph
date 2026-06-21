//! NVRTC kernel source + the per-gate uniform structs the host passes to the
//! kernels by value. Each struct's byte layout MUST match the matching CUDA
//! `struct` in `kernels.cu`; `const _: () = assert!(size_of == …)` pins it.

use cudarc::driver::DeviceRepr;

/// CUDA C++ source for the FP64 state-vector kernels, compiled once at
/// backend construction via NVRTC.
pub(crate) const SV_KERNELS_SRC: &str = include_str!("kernels.cu");

/// Entry-point names (declared `extern "C"`, so NVRTC keeps them unmangled).
pub(crate) const APPLY_1Q: &str = "apply_1q";
pub(crate) const APPLY_KQ: &str = "apply_kq";

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
