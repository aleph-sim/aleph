//! The single-qubit kernel: MSL source and the `Gate1q` uniform block that
//! the host uploads per gate via `set_bytes`.

use aleph_core::Complex;

/// MSL source for the 1q kernel; compiled at runtime by `MetalSvBackend::new`.
pub(crate) const SV_1Q_SRC: &str = include_str!("../shaders/sv_1q.metal");

/// Entry-point name inside `SV_1Q_SRC`.
pub(crate) const SV_1Q_ENTRY: &str = "apply_1q";

/// Per-gate uniform block. **Layout MUST match the MSL `Gate1q` struct**:
/// 4×`float2` (row-major 2×2), then `target`, `t_bit`, `ctrl_mask`, and a `u32`
/// pad so the size is 48 bytes (Metal rounds the struct up to its 8-byte
/// `float2` alignment). `Complex<f32>: Pod` comes from num-complex's `bytemuck`
/// feature; the derive requires every field be `Pod` and the struct be
/// padding-free (it is: 32 + 4×4 = 48 bytes, all 4-byte-aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Gate1q {
    pub m: [Complex<f32>; 4],
    pub target: u32,
    pub t_bit: u32,
    pub ctrl_mask: u32,
    pub _pad: u32,
}

// Compile-time guarantee the Rust struct matches the 48-byte MSL layout.
// Rust align is 4 (max field align); MSL's is 8 (float2). Mismatch is safe
// because every upload path (`set_bytes`, Metal page-aligned buffers) provides
// ≥8-byte alignment, and `set_bytes` copies by value so the GPU reads a fresh
// aligned copy — the host struct's own alignment never reaches the kernel.
const _: () = assert!(core::mem::size_of::<Gate1q>() == 48);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;

    /// Smoke test only: verifies the MSL compiles and a valid pipeline object
    /// is built on-device. Amplitude-level correctness after dispatch is tested
    /// in the `MetalSvBackend` unit/oracle tests (later P5.5-02 tasks).
    #[test]
    fn sv_1q_kernel_compiles_into_a_pipeline() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping kernel compile test: no Metal device");
                return;
            }
        };
        let pipeline = ctx.make_compute_pipeline(SV_1Q_SRC, SV_1Q_ENTRY);
        assert!(pipeline.is_ok(), "apply_1q must compile: {pipeline:?}");
    }
}
