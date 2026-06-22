//! GPU-safe gate fusion (P5.9-01).
//!
//! The CUDA state-vector backends are memory-bandwidth bound: each gate streams
//! the whole `2^n` state, so fewer gates ⇒ fewer passes ⇒ proportionally less
//! wall-clock. aleph already has IR fusion passes; [`fuse_for_gpu`] applies the
//! subset whose output the GPU backends can apply *today* — `Fuse1qRuns` and
//! `Fuse2q`, which emit `Unitary1q` / `Unitary2q` blocks (those carry a
//! `GateMatrix`; `UnitaryKq` and `DiagonalPhase` do not and need kernels still to
//! come, P5.9-02/03). It deliberately omits the CPU-tiling `RelabelQubits` /
//! `TileBlock` passes, which permute qubit labels.
//!
//! Measured at n=28 (RTX 4000 Ada): 1.64× (random), 2.05× (VQE), 2.48× (QAOA);
//! QFT is unchanged because its diagonal-phase ladder needs `FuseDiagonalRuns`
//! (P5.9-02). See `docs/perf/p5-08-gpu-report.md` for the Aer-GPU target.

use aleph_ir::passes::{CancelInversePairs, DeadCodeElim, Fuse1qRuns, Fuse2q, PassPipeline};
use aleph_ir::Circuit;

/// Return a fused copy of `circuit` using only the GPU-applicable passes.
///
/// The result is computationally equivalent (oracle-pinned at 1e-10 in
/// `tests/gpu_fusion_bench.rs`) and applies in fewer full-state passes.
pub fn fuse_for_gpu(circuit: &Circuit) -> Circuit {
    let mut c = circuit.clone();
    let pipe = PassPipeline::new(vec![
        Box::new(CancelInversePairs),
        Box::new(DeadCodeElim),
        Box::new(Fuse1qRuns),
        Box::new(Fuse2q),
    ]);
    // These passes never error (no transpile / relabel); a failure here is a
    // genuine internal-invariant bug, so surface it.
    let _ = pipe.run(&mut c).expect("GPU fusion pipeline");
    c
}
