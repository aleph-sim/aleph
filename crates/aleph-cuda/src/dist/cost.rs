//! P6-05 GPU cost model (spec §6.2). `classify` mirrors the decision order of
//! `CudaSvBackend::apply_gate` (sv/backend.rs) and `apply_diagonal_phase`, so
//! each instruction is priced as the kernel it actually launches.

use aleph_core::Gate;
use aleph_ir::dist::{CostModel, DistError, DistLayout};
use aleph_ir::Instruction;

use crate::common::diagonal_of;

/// The CUDA kernel an instruction of a fused rank program launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// `apply_1q` (dense, non-diagonal 1-qubit).
    Dense1,
    /// `apply_kq_tiled`, k = 2 (dense 2q incl. `Swap`, controlled `Cnot`).
    Dense2,
    /// `apply_kq_tiled`, k = 3 (`UnitaryKq` k=3, `Toffoli`, dense 3q).
    Dense3,
    /// `apply_diag_1q` (numerically diagonal 1-qubit).
    Diag1,
    /// `apply_diag`, k = 2 or 3 (numerically diagonal 2q/3q).
    DiagK,
    /// `apply_cnot` (bare `Cnot`, no external controls).
    Cnot,
    /// `apply_phase_poly`, its terms split by shape: `single` terms have ≤ 1
    /// cond, `multi` terms are an AND of ≥ 2 conds. A term's cost tracks how
    /// often it fires (1/2 of amplitudes for one 1-bit cond, 1/4 for an AND of
    /// two), so the two shapes are priced separately.
    PhasePoly {
        /// Terms with at most one cond.
        single: usize,
        /// Terms with two or more conds.
        multi: usize,
    },
    /// No kernel (barrier, empty `DiagonalPhase`).
    Free,
}

/// Kernel kind `instr` launches on `CudaSvBackend` / `CudaSvBackendF32`.
///
/// The classification assumes the backend's defaults (`custom_2q` and
/// `custom_diag` on), as `DistSvBackend` runs them.
pub fn classify(instr: &Instruction) -> Result<KernelKind, DistError> {
    let g = match instr {
        Instruction::Gate(g) => g,
        Instruction::DiagonalPhase(dp) => {
            let multi = dp.terms.iter().filter(|t| t.conds.len() >= 2).count();
            let single = dp.terms.len() - multi;
            return Ok(if single + multi == 0 {
                KernelKind::Free
            } else {
                KernelKind::PhasePoly { single, multi }
            });
        }
        Instruction::Barrier(_) => return Ok(KernelKind::Free),
        _ => {
            return Err(DistError::Unsupported {
                kind: "cost: non-unitary instruction",
            })
        }
    };
    // 1. bare Cnot → apply_cnot (backend.rs:665-687)
    if matches!(g.gate, Gate::Cnot) && g.controls.is_empty() {
        return Ok(KernelKind::Cnot);
    }
    // 2. UnitaryKq → apply_kq_tiled before any diagonal test (backend.rs:693-705)
    if let Gate::UnitaryKq { k, .. } = &g.gate {
        return match k {
            1 => Ok(KernelKind::Dense1),
            2 => Ok(KernelKind::Dense2),
            3 => Ok(KernelKind::Dense3),
            _ => Err(DistError::Unsupported {
                kind: "cost: UnitaryKq k > 3 (unreachable from fuse_for_gpu)",
            }),
        };
    }
    // 3–5. matrix: numerically diagonal → apply_diag(_1q), else dense by arity
    let m = g.gate.matrix()?;
    let diag = diagonal_of(&m).is_some();
    Ok(match (g.qubits.len(), diag) {
        (1, true) => KernelKind::Diag1,
        (_, true) => KernelKind::DiagK,
        (1, false) => KernelKind::Dense1,
        (2, false) => KernelKind::Dense2,
        _ => KernelKind::Dense3,
    })
}

/// Calibrated seconds per kernel launch at a `2^m_ref` slice (the RTX 4000
/// presets come from `tests/dist_cost_calibrate.rs`). Every kind is a full pass
/// over the slice, so time scales by `2^(m − m_ref)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KindTimes {
    /// Slice size (qubits) the per-kind times were measured at.
    pub m_ref: u32,
    /// `Dense1` seconds at `m_ref`.
    pub dense1: f64,
    /// `Dense2` seconds at `m_ref`.
    pub dense2: f64,
    /// `Dense3` seconds at `m_ref`.
    pub dense3: f64,
    /// `Diag1` seconds at `m_ref`.
    pub diag1: f64,
    /// `DiagK` seconds at `m_ref`.
    pub diag_k: f64,
    /// `Cnot` seconds at `m_ref`.
    pub cnot: f64,
    /// `PhasePoly` fixed cost at `m_ref`.
    pub phase_base: f64,
    /// `PhasePoly` per-term cost at `m_ref`, terms with ≤ 1 cond.
    pub phase_term: f64,
    /// `PhasePoly` per-term cost at `m_ref`, terms with ≥ 2 conds (AND).
    pub phase_term_multi: f64,
}

impl KindTimes {
    /// Seconds for one launch of kind `k` on a `2^m` slice.
    pub fn seconds(&self, k: KernelKind, m: u32) -> f64 {
        let at_ref = match k {
            KernelKind::Dense1 => self.dense1,
            KernelKind::Dense2 => self.dense2,
            KernelKind::Dense3 => self.dense3,
            KernelKind::Diag1 => self.diag1,
            KernelKind::DiagK => self.diag_k,
            KernelKind::Cnot => self.cnot,
            KernelKind::PhasePoly { single, multi } => {
                self.phase_base
                    + self.phase_term * single as f64
                    + self.phase_term_multi * multi as f64
            }
            KernelKind::Free => 0.0,
        };
        at_ref * 2f64.powi(m as i32 - self.m_ref as i32)
    }
}

/// Per-exchange link bandwidth by exchange width (bytes/s, index `k − 1`).
/// Widths past the table use its last entry (spec §6.2's documented
/// extrapolation of the two-bit value).
#[derive(Debug, Clone, PartialEq)]
pub struct LinkModel {
    /// Bandwidth in bytes/s for exchange width `k = index + 1`.
    pub bw_by_k: Vec<f64>,
}

impl LinkModel {
    /// AWS g6.12xlarge, 4× L4, no P2P, NCCL memcpy-SHM, FP64
    /// (docs/perf/p6-multi-gpu.md §4.1: 7.16 GB/s single-bit, 4.35 two-bit).
    pub fn aws_g6_fp64() -> Self {
        Self {
            bw_by_k: vec![7.16e9, 4.35e9],
        }
    }

    /// FP32: 7.09 GB/s single-bit measured; two-bit scaled by the FP64 ratio
    /// (no FP32 two-bit measurement exists — spec §6.2).
    pub fn aws_g6_fp32() -> Self {
        Self {
            bw_by_k: vec![7.09e9, 7.09e9 * 4.35 / 7.16],
        }
    }

    /// Seconds for one `k`-bit exchange of a `2^m` slice: `(1 − 2^−k)` of it moves.
    pub fn seconds(&self, k: u32, m: u32, amp_bytes: f64) -> f64 {
        if k == 0 || self.bw_by_k.is_empty() {
            return 0.0;
        }
        let idx = (k as usize - 1).min(self.bw_by_k.len() - 1);
        let frac = 1.0 - 2f64.powi(-(k as i32));
        frac * 2f64.powi(m as i32) * amp_bytes / self.bw_by_k[idx]
    }
}

/// Calibrated GPU cost model (spec §6.2): prices a rank's specialised, fused
/// program by kernel kind, and exchanges by `LinkModel`.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuCostModel {
    /// Per-kernel-kind launch times.
    pub kinds: KindTimes,
    /// Exchange bandwidth model.
    pub link: LinkModel,
    /// 16 (FP64) or 8 (FP32).
    pub amp_bytes: f64,
    /// Must match the executing `DistSvBackend`'s fusion setting.
    pub fuse: bool,
}

impl GpuCostModel {
    /// Seconds rank `rank` spends on one `Local` step.
    pub fn rank_segment(
        &self,
        instrs: &[Instruction],
        layout: DistLayout,
        rank: u32,
    ) -> Result<f64, DistError> {
        let c = super::rank_circuit(instrs, layout, rank, self.fuse)?;
        let m = layout.m();
        let mut t = 0.0;
        for i in c.instructions() {
            t += self.kinds.seconds(classify(i)?, m);
        }
        Ok(t)
    }

    /// RTX 4000 SFF Ada, FP64, fused like `DistSvBackend::new` — constants
    /// from `tests/dist_cost_calibrate.rs` (Task 4 documents the command there).
    pub fn rtx4000_fp64() -> Self {
        Self {
            kinds: RTX4000_FP64,
            link: LinkModel::aws_g6_fp64(),
            amp_bytes: 16.0,
            fuse: true,
        }
    }

    /// FP32 counterpart.
    pub fn rtx4000_fp32() -> Self {
        Self {
            kinds: RTX4000_FP32,
            link: LinkModel::aws_g6_fp32(),
            amp_bytes: 8.0,
            fuse: true,
        }
    }
}

impl CostModel for GpuCostModel {
    /// The busiest rank, R − 1 (all global bits 1: every global-controlled
    /// gate is live) — an upper bound; ranks sync at each exchange anyway.
    fn local_segment(&self, instrs: &[Instruction], layout: DistLayout) -> Result<f64, DistError> {
        self.rank_segment(instrs, layout, layout.ranks() - 1)
    }

    fn exchange(&self, k: u32, m: u32) -> f64 {
        self.link.seconds(k, m, self.amp_bytes)
    }
}

/// Measured per-launch seconds per kernel kind on the RTX 4000 SFF Ada (20 GiB).
///
/// Calibrated 2026-10-04 with
/// `cargo test --release -p aleph-cuda --features cuda --test dist_cost_calibrate -- --ignored --nocapture`.
/// Method: each launch interleaved with an `H`, best of 5 × 32 launches, the
/// interleaved-H baseline subtracted. Each constant is the **mean of 2 runs**
/// (largest run-to-run spread: FP64 dense3 +2.4 %, phase_base +2.0 %).
/// `dense2` alone is timed on a scrambled state (H layer + one Rx and one Rz
/// per qubit, in payload and baseline) and was re-measured later the same day
/// (mean of 2 runs, FP64 spread +3.3 %); the other constants are from the
/// earlier pair. At the card's 70 W cap, FP64 kernel time depends on the
/// amplitude data: dense2 costs ~1.2x on a generic complex state, and dense3
/// (still uniform-state, to keep GHZ-like low-entropy states right) carries a
/// known state-dependent error (~1.12x on generic states).
/// Most kinds sit on the ~17.6 ms single-pass bandwidth floor (dense1/diag1/
/// diag_k are within ~1 % of each other, so their ordering is noise); cnot
/// touches half the state, dense3 is compute-bound at FP64, and phase_poly pays
/// a per-term cost that depends on the term shape (single cond vs AND of two).
/// `phase_base` includes the per-launch device allocation and upload of the terms.
const RTX4000_FP64: KindTimes = KindTimes {
    m_ref: 27,
    dense1: 1.755645e-2,
    dense2: 2.168771e-2,
    dense3: 3.436177e-2,
    diag1: 1.774244e-2,
    diag_k: 1.759981e-2,
    cnot: 1.080041e-2,
    phase_base: 2.147593e-2,
    phase_term: 6.712988e-4,
    phase_term_multi: 4.776938e-4,
};
/// FP32 counterpart (same provenance as [`RTX4000_FP64`], state size 2^28).
const RTX4000_FP32: KindTimes = KindTimes {
    m_ref: 28,
    dense1: 1.763479e-2,
    dense2: 1.773724e-2,
    dense3: 1.797886e-2,
    diag1: 1.769642e-2,
    diag_k: 1.771855e-2,
    cnot: 1.295351e-2,
    phase_base: 4.623386e-2,
    phase_term: 1.285620e-3,
    phase_term_multi: 8.999014e-4,
};

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Complex, GateInstance, Param};
    use aleph_ir::{DiagonalPhase, PhaseTerm};
    use smallvec::smallvec;

    fn g(gate: Gate, q: &[u32]) -> Instruction {
        Instruction::Gate(GateInstance::new(gate, q.to_vec()))
    }

    #[test]
    fn classify_mirrors_apply_gate() {
        use KernelKind::*;
        let p = Param::Concrete(0.3);
        assert_eq!(classify(&g(Gate::H, &[0])).unwrap(), Dense1);
        assert_eq!(classify(&g(Gate::Rx(p), &[0])).unwrap(), Dense1);
        assert_eq!(classify(&g(Gate::Rz(p), &[0])).unwrap(), Diag1);
        assert_eq!(classify(&g(Gate::T, &[0])).unwrap(), Diag1);
        assert_eq!(classify(&g(Gate::Cnot, &[0, 1])).unwrap(), Cnot);
        let ccx = Instruction::Gate(GateInstance::controlled(
            Gate::Cnot,
            vec![0u32, 1],
            vec![2u32],
        ));
        assert_eq!(classify(&ccx).unwrap(), Dense2);
        assert_eq!(classify(&g(Gate::Swap, &[0, 1])).unwrap(), Dense2);
        assert_eq!(classify(&g(Gate::Iswap, &[0, 1])).unwrap(), Dense2);
        assert_eq!(classify(&g(Gate::Cz, &[0, 1])).unwrap(), DiagK);
        assert_eq!(classify(&g(Gate::CRz(p), &[0, 1])).unwrap(), DiagK);
        let d2 = Gate::Unitary2qDiag(Box::new([Complex::new(1.0, 0.0); 4]));
        assert_eq!(classify(&g(d2, &[0, 1])).unwrap(), DiagK);
        // Numerically diagonal dense Unitary2q → apply_diag (diagonal_of is numeric).
        let mut m = [[Complex::new(0.0, 0.0); 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = Complex::new(1.0, 0.0);
        }
        assert_eq!(
            classify(&g(Gate::Unitary2q(Box::new(m)), &[0, 1])).unwrap(),
            DiagK
        );
        assert_eq!(classify(&g(Gate::Toffoli, &[0, 1, 2])).unwrap(), Dense3);
        assert_eq!(classify(&g(Gate::Ccz, &[0, 1, 2])).unwrap(), DiagK);
        // UnitaryKq is dispatched before diagonal_of: dense even if diagonal.
        let mut id8 = vec![Complex::new(0.0, 0.0); 64];
        for i in 0..8 {
            id8[i * 8 + i] = Complex::new(1.0, 0.0);
        }
        let kq = Gate::UnitaryKq {
            k: 3,
            data: id8.into_boxed_slice(),
        };
        assert_eq!(classify(&g(kq, &[0, 1, 2])).unwrap(), Dense3);
    }

    #[test]
    fn phase_poly_and_free() {
        let dp = |terms: Vec<PhaseTerm>| {
            Instruction::DiagonalPhase(Box::new(DiagonalPhase { n_qubits: 3, terms }))
        };
        assert_eq!(classify(&dp(vec![])).unwrap(), KernelKind::Free);
        let t = PhaseTerm {
            conds: smallvec![0b011],
            angle: 0.2,
        };
        assert_eq!(
            classify(&dp(vec![t.clone(), t])).unwrap(),
            KernelKind::PhasePoly {
                single: 2,
                multi: 0
            }
        );
        assert_eq!(
            classify(&Instruction::Barrier(smallvec![0, 1])).unwrap(),
            KernelKind::Free
        );
    }

    #[test]
    fn phase_poly_splits_terms_by_cond_count() {
        let dp = Instruction::DiagonalPhase(Box::new(DiagonalPhase {
            n_qubits: 3,
            terms: vec![
                PhaseTerm {
                    conds: smallvec![],
                    angle: 0.1,
                },
                PhaseTerm {
                    conds: smallvec![0b001],
                    angle: 0.2,
                },
                PhaseTerm {
                    conds: smallvec![0b001, 0b100],
                    angle: 0.3,
                },
                PhaseTerm {
                    conds: smallvec![0b001, 0b010, 0b100],
                    angle: 0.4,
                },
            ],
        }));
        assert_eq!(
            classify(&dp).unwrap(),
            KernelKind::PhasePoly {
                single: 2,
                multi: 2
            }
        );
    }

    #[test]
    fn measure_is_rejected() {
        assert!(classify(&Instruction::Measure { qubit: 0, clbit: 0 }).is_err());
    }

    fn unit_times() -> KindTimes {
        KindTimes {
            m_ref: 20,
            dense1: 1.0,
            dense2: 2.0,
            dense3: 3.0,
            diag1: 0.5,
            diag_k: 0.75,
            cnot: 0.25,
            phase_base: 1.0,
            phase_term: 0.1,
            phase_term_multi: 0.05,
        }
    }

    #[test]
    fn kind_seconds_scale_with_slice() {
        let t = unit_times();
        assert_eq!(t.seconds(KernelKind::Dense2, 20), 2.0);
        assert_eq!(t.seconds(KernelKind::Dense2, 22), 8.0);
        let ph = KernelKind::PhasePoly {
            single: 10,
            multi: 0,
        };
        assert!((t.seconds(ph, 20) - 2.0).abs() < 1e-12);
        let ph = KernelKind::PhasePoly {
            single: 10,
            multi: 4,
        };
        assert!((t.seconds(ph, 21) - 2.0 * (2.0 + 4.0 * 0.05)).abs() < 1e-12);
        assert_eq!(t.seconds(KernelKind::Free, 25), 0.0);
    }

    #[test]
    fn scaling_handles_m_below_ref() {
        let t = unit_times();
        assert_eq!(t.seconds(KernelKind::Dense1, 18), 0.25);
        assert!(t.seconds(KernelKind::Dense1, 0) > 0.0);
    }

    #[test]
    fn link_extrapolates_last_entry() {
        let l = LinkModel {
            bw_by_k: vec![8.0, 4.0],
        };
        // k=1, m=3, 16 B/amp: (1 - 1/2) * 8 * 16 / 8 = 8 s
        assert!((l.seconds(1, 3, 16.0) - 8.0).abs() < 1e-12);
        // k=2: (3/4) * 8 * 16 / 4 = 24 s ; k=3 uses the k=2 bandwidth: (7/8)*8*16/4 = 28 s
        assert!((l.seconds(2, 3, 16.0) - 24.0).abs() < 1e-12);
        assert!((l.seconds(3, 3, 16.0) - 28.0).abs() < 1e-12);
    }

    #[test]
    fn rank_segment_prices_the_fused_program() {
        use aleph_ir::dist::DistLayout;
        let model = GpuCostModel {
            kinds: unit_times(),
            link: LinkModel::aws_g6_fp64(),
            amp_bytes: 16.0,
            fuse: false,
        };
        // n=21, g=1 -> m=20 = m_ref. H(0) dense1 + Cnot(0,1) + Rz(1) diag1.
        let l = DistLayout::new(21, 1).unwrap();
        let instrs = vec![
            Instruction::Gate(GateInstance::new(Gate::H, vec![0])),
            Instruction::Gate(GateInstance::new(Gate::Cnot, vec![0, 1])),
            Instruction::Gate(GateInstance::new(Gate::Rz(Param::Concrete(0.3)), vec![1])),
        ];
        let t = model.rank_segment(&instrs, l, 1).unwrap();
        assert!((t - (1.0 + 0.25 + 0.5)).abs() < 1e-12);
    }
}
