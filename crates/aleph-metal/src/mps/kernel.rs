//! MSL sources and the per-gate uniform blocks for the MPS-on-Metal kernels
//! (P5.5-06). Each uniform's `#[repr(C)]` layout must match the MSL `struct` it
//! is uploaded into via `set_bytes`.

use aleph_core::Complex;

/// MSL source for the 1q site kernel; compiled at runtime by the backend.
pub(crate) const MPS_1Q_SRC: &str = include_str!("../shaders/mps_1q.metal");
/// Entry-point name inside [`MPS_1Q_SRC`].
pub(crate) const MPS_1Q_ENTRY: &str = "apply_1q_site";

/// MSL source for the two-site contraction kernel.
pub(crate) const MPS_CONTRACT_SRC: &str = include_str!("../shaders/mps_contract.metal");
/// Entry-point name inside [`MPS_CONTRACT_SRC`].
pub(crate) const MPS_CONTRACT_ENTRY: &str = "contract_2site";

/// MSL source for the 2q gate-apply-on-Θ kernel.
pub(crate) const MPS_APPLY2Q_SRC: &str = include_str!("../shaders/mps_apply2q.metal");
/// Entry-point name inside [`MPS_APPLY2Q_SRC`].
pub(crate) const MPS_APPLY2Q_ENTRY: &str = "apply_2q_theta";

/// MSL source for the GPU-resident one-sided Jacobi thin-SVD kernel (P5.7-02).
/// Holds both the single-block (`jacobi_svd`) and batched (`jacobi_svd_batched`)
/// entry points; one source file, two pipelines.
pub(crate) const MPS_JACOBI_SRC: &str = include_str!("../shaders/mps_jacobi.metal");
/// Entry-point name inside [`MPS_JACOBI_SRC`].
pub(crate) const MPS_JACOBI_ENTRY: &str = "jacobi_svd";
/// Batched entry-point name inside [`MPS_JACOBI_SRC`] (P5.7-04): one threadgroup
/// per block, `num_blocks` threadgroups dispatched in a single launch.
pub(crate) const MPS_JACOBI_BATCHED_ENTRY: &str = "jacobi_svd_batched";

/// MSL source for the GPU column-major pack (P5.8-03): Θ′ (row-major) → the
/// column-major `A` the Jacobi kernel consumes, so Θ′ never leaves the GPU between
/// the contraction and the SVD.
pub(crate) const MPS_PACK_SRC: &str = include_str!("../shaders/mps_pack.metal");
/// Entry-point name inside [`MPS_PACK_SRC`].
pub(crate) const MPS_PACK_ENTRY: &str = "pack_theta";

/// MSL source for the GPU-resident split finalize (P5.8-03): sort σ, pick χ, and
/// assemble the two new site tensors on-device, so U/V/σ are never read back.
pub(crate) const MPS_FINALIZE_SRC: &str = include_str!("../shaders/mps_finalize.metal");
/// Entry-point name inside [`MPS_FINALIZE_SRC`].
pub(crate) const MPS_FINALIZE_ENTRY: &str = "finalize_split";

/// Per-gate uniform for [`MPS_1Q_SRC`]. **Layout MUST match the MSL `Mps1q`
/// struct**: 4×`float2` (row-major 2×2) then `right` and one u32 pad → 40 bytes,
/// no internal padding (32 + 4 + 4, all 4-byte-aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Mps1q {
    pub m: [Complex<f32>; 4],
    pub right: u32,
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<Mps1q>() == 40);

/// Per-gate uniform for [`MPS_CONTRACT_SRC`]. **Layout MUST match the MSL
/// `ContractMeta` struct** (16 bytes): `c` (shared bond), `ri` (site-j right
/// bond), two pads.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ContractMeta {
    pub c: u32,
    pub ri: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(core::mem::size_of::<ContractMeta>() == 16);

/// Per-gate uniform for [`MPS_APPLY2Q_SRC`]. **Layout MUST match the MSL
/// `Apply2qMeta` struct** (16 bytes): `ri`, `i_is_msb` (1 if the left site holds
/// the matrix MSB qubit), two pads.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Apply2qMeta {
    pub ri: u32,
    pub i_is_msb: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(core::mem::size_of::<Apply2qMeta>() == 16);

/// Per-call uniform for [`MPS_JACOBI_SRC`]. **Layout MUST match the MSL
/// `JacobiMeta` struct** (16 bytes): `m` (rows, host-guaranteed ≥ `n`), `n`
/// (cols = number of singular values), two pads.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct JacobiMeta {
    pub m: u32,
    pub n: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(core::mem::size_of::<JacobiMeta>() == 16);

/// Per-gate uniform for [`MPS_PACK_SRC`] (P5.8-03). **Layout MUST match the MSL
/// `PackMeta` struct** (16 bytes): `rows`, `cols` (Θ′ shape), `wide` (1 when
/// `rows < cols`, so Θ′ is packed as its adjoint), one pad.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PackMeta {
    pub rows: u32,
    pub cols: u32,
    pub wide: u32,
    pub _pad0: u32,
}

const _: () = assert!(core::mem::size_of::<PackMeta>() == 16);

/// Per-gate uniform for [`MPS_FINALIZE_SRC`] (P5.8-03). **Layout MUST match the MSL
/// `FinalizeMeta` struct** (32 bytes): `rows`, `cols`, `wide`, `max_bond`,
/// `renormalize`, `k` (= `min(rows, cols)`), two pads.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FinalizeMeta {
    pub rows: u32,
    pub cols: u32,
    pub wide: u32,
    pub max_bond: u32,
    pub renormalize: u32,
    pub k: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(core::mem::size_of::<FinalizeMeta>() == 32);

/// Output scalars from [`MPS_FINALIZE_SRC`] (P5.8-03). **Layout MUST match the MSL
/// `FinalizeOut` struct** (16 bytes): kept bond `chi`, `accept` (0 ⇒ host f64
/// fallback), relative discarded weight `trunc_rel`, one pad.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FinalizeOut {
    pub chi: u32,
    pub accept: u32,
    pub trunc_rel: f32,
    pub _pad: f32,
}

const _: () = assert!(core::mem::size_of::<FinalizeOut>() == 16);

/// Per-block descriptor for the batched Jacobi kernel (P5.7-04). **Layout MUST
/// match the MSL `JacobiBlockMeta` struct** (32 bytes): `m`, `n` (block dims, as
/// in [`JacobiMeta`]), then the float2 offsets of this block's `A`/`V` and the
/// float offset of its `sig` inside the packed buffers, then three pads. Packed
/// as an array into the kernel's `buffer(3)`, so the 32-byte stride must match.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct JacobiBlockMeta {
    pub m: u32,
    pub n: u32,
    pub a_off: u32,
    pub v_off: u32,
    pub sig_off: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(core::mem::size_of::<JacobiBlockMeta>() == 32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;

    /// Smoke test: each MPS kernel compiles into a pipeline on-device. Amplitude
    /// correctness is covered by the backend unit tests + the oracle.
    #[test]
    fn mps_kernels_compile_into_pipelines() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping MPS kernel compile test: no Metal device");
                return;
            }
        };
        for (src, entry) in [
            (MPS_1Q_SRC, MPS_1Q_ENTRY),
            (MPS_CONTRACT_SRC, MPS_CONTRACT_ENTRY),
            (MPS_APPLY2Q_SRC, MPS_APPLY2Q_ENTRY),
            (MPS_JACOBI_SRC, MPS_JACOBI_ENTRY),
            (MPS_JACOBI_SRC, MPS_JACOBI_BATCHED_ENTRY),
            (MPS_PACK_SRC, MPS_PACK_ENTRY),
            (MPS_FINALIZE_SRC, MPS_FINALIZE_ENTRY),
        ] {
            let p = ctx.make_compute_pipeline(src, entry);
            assert!(p.is_ok(), "{entry} must compile: {p:?}");
        }
    }
}
