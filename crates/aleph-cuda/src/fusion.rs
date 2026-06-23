//! GPU-safe gate fusion (P5.9-01 / P5.9-02b).
//!
//! The CUDA state-vector backends are memory-bandwidth bound: each gate streams
//! the whole `2^n` state, so fewer gates ⇒ fewer passes ⇒ proportionally less
//! wall-clock. aleph already has IR fusion passes; [`fuse_for_gpu`] applies the
//! subset whose output the GPU backends can apply:
//!
//! - `Fuse1qRuns` + `Fuse2q` collapse adjacent gates into `Unitary1q` /
//!   `Unitary2q` blocks (P5.9-01).
//! - `FuseKq` then merges *runs* of those into dense 3q `UnitaryKq` blocks,
//!   which the `apply_kq` kernel now applies directly (P5.9-02a). `FuseKq`
//!   only emits a block at span ≥ 3 with ≥ 2 members, so isolated 1q/2q gates
//!   keep their specialised kernels — it is purely additive over P5.9-01.
//!
//! `FuseDiagonalRuns` runs first (P5.9-06): it collapses a controlled-phase
//! ladder (QFT/QPE) into one `DiagonalPhase`, which the GPU `apply_phase_poly`
//! kernel applies in a single coalesced sweep. It deliberately omits the
//! CPU-tiling `RelabelQubits` / `TileBlock` passes, which permute qubit labels.
//!
//! **Why width 3, not Aer's 5 — even with the register-tiled kernel.** The
//! P5.9-02b A/B bench (n=28, RTX 4000 Ada; `docs/perf/p5.9-gpu-fusion.md`)
//! measured `max_qubits ∈ {3,4,5}` on the generic `apply_kq`: k=3 wins
//! **1.08–1.14×** on every dense workload, but k=4/5 *regress* (down to 0.68×).
//! P5.10-01 then built the register-tiled `apply_kq_tiled` kernel (one amplitude
//! per warp lane, matvec = intra-warp shuffle reduction, matrix in shared
//! memory) so neither `v[32]`/`gidx[32]` spills. It **strictly beats** generic
//! `apply_kq` at every width (1.07–1.18×, the margin growing with k —
//! `docs/perf/p5.10-01-tiled-fused-block.md`), confirming the spill was real.
//! But k=4/5 fusion *still* loses to k=3 (tiled k=5 is 0.73–0.80× the generic-k3
//! baseline): the dominant wall past k=3 is the **O(4^k) matvec compute** itself
//! (a 32×32 dense matvec per group), not the spill — fewer passes (n/5 vs n/3)
//! can't pay for 4× the arithmetic. So the sweet spot stays **3**. The tiled
//! kernel is still the production default at k≤3 (`tiled_min_k=2`), worth ~1.07×
//! on the dense cells; the k=4,5 path is correct and kept for callers that raise
//! the width. See `docs/perf/p5-08-gpu-report.md` for the Aer-GPU target.

use aleph_ir::passes::{
    CancelInversePairs, DeadCodeElim, Fuse1qRuns, Fuse2q, FuseDiagonalRuns, FuseKq, Pass,
    PassPipeline,
};
use aleph_ir::Circuit;

/// Largest dense block `FuseKq` may emit for the GPU path. **3** is the
/// measured sweet spot of the generic `apply_kq` kernel — k=4,5 regress because
/// the per-group `2^k × 2^k` matvec outgrows the pass-count savings (see the
/// module docs and the P5.9-02b A/B bench). The kernel still supports k≤5
/// (P5.9-02a); this is purely the throughput-optimal fusion width today.
pub const MAX_FUSE_QUBITS: usize = 3;

/// Return a fused copy of `circuit` using the GPU-applicable passes, including
/// dense 3q `UnitaryKq` fusion ([`MAX_FUSE_QUBITS`]).
///
/// The result is computationally equivalent (oracle-pinned at 1e-10 in
/// `tests/gpu_fusion_bench.rs` and `tests/gpu_unitary_kq.rs`) and applies in
/// fewer full-state passes.
pub fn fuse_for_gpu(circuit: &Circuit) -> Circuit {
    fuse_for_gpu_with(circuit, Some(MAX_FUSE_QUBITS))
}

/// A/B-tunable fusion: `kq_max = None` reproduces the P5.9-01 pipeline
/// (`Fuse1qRuns` + `Fuse2q`, no dense ≥3q blocks); `Some(k)` appends
/// `FuseKq { max_qubits: k }` so adjacent 1q/2q blocks collapse into dense
/// k-qubit `UnitaryKq`. Exposed for the P5.9-02b benchmark; production callers
/// use [`fuse_for_gpu`].
pub fn fuse_for_gpu_with(circuit: &Circuit, kq_max: Option<usize>) -> Circuit {
    let mut c = circuit.clone();
    let mut passes: Vec<Box<dyn Pass>> = vec![
        Box::new(CancelInversePairs),
        Box::new(DeadCodeElim),
        // P5.9-06: collapse controlled-phase ladders (QFT/QPE) into one
        // DiagonalPhase, applied by the GPU `apply_phase_poly` kernel in a single
        // coalesced sweep. Runs before Fuse1q/Fuse2q (canonical pipeline order).
        Box::new(FuseDiagonalRuns),
        Box::new(Fuse1qRuns),
        Box::new(Fuse2q),
    ];
    if let Some(k) = kq_max {
        passes.push(Box::new(FuseKq { max_qubits: k }));
    }
    let pipe = PassPipeline::new(passes);
    // These passes never error (no transpile / relabel); a failure here is a
    // genuine internal-invariant bug, so surface it.
    let _ = pipe.run(&mut c).expect("GPU fusion pipeline");
    c
}
