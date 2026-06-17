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

/// MSL source for the generic dense k-qubit kernel.
pub(crate) const SV_KQ_SRC: &str = include_str!("../shaders/sv_kq.metal");

/// Entry-point name inside `SV_KQ_SRC`.
pub(crate) const SV_KQ_ENTRY: &str = "apply_kq";

/// Per-gate uniform for [`SV_KQ_SRC`]. **Layout MUST match the MSL `GateKqMeta`
/// struct** (all `uint`, 48 bytes, no padding). `sorted` = target bit positions
/// ascending (zero-bit insertion); `tbit` = `1 << q[j]` in logical/MSB order
/// (matrix-index bit assignment); slots `j >= k` are zero and never read.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GateKqMeta {
    pub k: u32,
    pub sorted: [u32; 5],
    pub tbit: [u32; 5],
    pub ctrl_mask: u32,
}

// 1 + 5 + 5 + 1 = 12 u32 = 48 bytes, all 4-byte-aligned (matches MSL).
const _: () = assert!(core::mem::size_of::<GateKqMeta>() == 48);

/// MSL source for the diagonal-phase kernel.
pub(crate) const SV_DIAG_SRC: &str = include_str!("../shaders/sv_diag.metal");

/// Entry-point name inside [`SV_DIAG_SRC`].
pub(crate) const SV_DIAG_ENTRY: &str = "apply_diagonal_phase";

/// Per-term descriptor for [`SV_DIAG_SRC`]. **Layout MUST match the MSL
/// `DiagTermDesc` struct** (16 bytes, no padding).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DiagTermDesc {
    pub cond_offset: u32,
    pub n_conds: u32,
    pub angle: f32,
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<DiagTermDesc>() == 16);

/// Scalar uniform for [`SV_DIAG_SRC`]: the term count.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DiagMeta {
    pub n_terms: u32,
}

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

    #[test]
    fn sv_kq_kernel_compiles_into_a_pipeline() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping kernel compile test: no Metal device");
                return;
            }
        };
        let pipeline = ctx.make_compute_pipeline(SV_KQ_SRC, SV_KQ_ENTRY);
        assert!(pipeline.is_ok(), "apply_kq must compile: {pipeline:?}");
    }

    #[test]
    fn sv_diag_kernel_compiles_into_a_pipeline() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping kernel compile test: no Metal device");
                return;
            }
        };
        let pipeline = ctx.make_compute_pipeline(SV_DIAG_SRC, SV_DIAG_ENTRY);
        assert!(
            pipeline.is_ok(),
            "apply_diagonal_phase must compile: {pipeline:?}"
        );
    }
}
