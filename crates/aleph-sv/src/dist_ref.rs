//! CPU reference executor for `aleph_ir::dist` plans (P6-02).
//!
//! Simulates `2^g` ranks as separate `2^m` CPU slices, runs each `Local`
//! step through `specialize` + `NaiveSvBackend`, and performs `Exchange`
//! with the same contiguous-chunk algorithm the GPU transports implement —
//! so it is both the correctness oracle for the planner and the reference
//! the multi-GPU exchange is diffed against.

use aleph_backend::{Backend, BackendError};
use aleph_core::{AlignedBuf, Complex};
use aleph_ir::dist::{specialize, DistError, DistLayout, DistPlan, DistStep};
use aleph_ir::Instruction;

use crate::{CpuState, NaiveSvBackend};

/// Failure of [`run_dist`]: a planning/specialisation error or a backend error.
#[derive(Debug, thiserror::Error)]
pub enum DistRefError {
    #[error(transparent)]
    Dist(#[from] DistError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}

/// Execute `plan` from `|0…0⟩` and return the `2^n` state in **logical**
/// qubit order (directly comparable to a single-node run).
pub fn run_dist(plan: &DistPlan) -> Result<Vec<Complex>, DistRefError> {
    let l = plan.layout;
    let m = l.m();
    let size = 1usize << m;
    let zero = Complex::new(0.0, 0.0);
    let mut ranks: Vec<Vec<Complex>> = (0..l.ranks()).map(|_| vec![zero; size]).collect();
    ranks[0][0] = Complex::new(1.0, 0.0);
    let mut be = NaiveSvBackend::new();

    for step in &plan.steps {
        match step {
            DistStep::Local(instrs) => {
                for (r, slice) in ranks.iter_mut().enumerate() {
                    let mut st = CpuState {
                        num_qubits: m,
                        amps: AlignedBuf::from_slice(slice),
                    };
                    for i in instrs {
                        match specialize(i, l, r as u32)? {
                            None => {}
                            Some(Instruction::Gate(g)) => be.apply_gate(&mut st, &g)?,
                            Some(Instruction::DiagonalPhase(dp)) => {
                                be.apply_diagonal_phase(&mut st, &dp)?
                            }
                            Some(_) => {
                                return Err(DistError::Unsupported {
                                    kind: "specialize output",
                                }
                                .into())
                            }
                        }
                    }
                    slice.copy_from_slice(&st.amps);
                }
            }
            DistStep::Exchange { global_bits } => exchange_cpu(&mut ranks, l, global_bits),
        }
    }

    // Gather physical, then permute to logical via final_map.
    let mut out = vec![zero; 1usize << l.n];
    for (x, slot) in out.iter_mut().enumerate() {
        let mut p = 0usize;
        for (lq, &pq) in plan.final_map.iter().enumerate() {
            p |= ((x >> lq) & 1) << pq;
        }
        *slot = ranks[p >> m][p & (size - 1)];
    }
    Ok(out)
}

/// Swap global physical bits `global_bits[j]` with local bits `m-k+j`.
///
/// Rank `r`'s slice is `2^k` contiguous chunks `c` (value of the top `k`
/// local bits). Chunk `c` of rank `r` lands on rank `r'` = `r` with the
/// chosen global bits set to `c`, at chunk index = `r`'s old values of those
/// bits. Out-of-place here (reference only); the GPU path streams it.
pub fn exchange_cpu(ranks: &mut [Vec<Complex>], l: DistLayout, global_bits: &[u32]) {
    let m = l.m();
    let k = global_bits.len() as u32;
    let chunk = 1usize << (m - k);
    let old: Vec<Vec<Complex>> = ranks.to_vec();
    for (r, src) in old.iter().enumerate() {
        let r = r as u32;
        for c in 0..(1u32 << k) {
            let mut dst_rank = r;
            let mut dst_chunk = 0u32;
            for (j, &gb) in global_bits.iter().enumerate() {
                let rb = gb - m; // rank-bit index of this global qubit
                dst_chunk |= ((r >> rb) & 1) << j;
                dst_rank = (dst_rank & !(1 << rb)) | (((c >> j) & 1) << rb);
            }
            let s = c as usize * chunk;
            let d = dst_chunk as usize * chunk;
            ranks[dst_rank as usize][d..d + chunk].copy_from_slice(&src[s..s + chunk]);
        }
    }
}
