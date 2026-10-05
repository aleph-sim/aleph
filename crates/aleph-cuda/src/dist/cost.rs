//! P6-05 GPU cost model (spec §6.2). `classify` mirrors the decision order of
//! `CudaSvBackend::apply_gate` (sv/backend.rs) and `apply_diagonal_phase`, so
//! each instruction is priced as the kernel it actually launches.

use aleph_core::{Complex, Gate, GateMatrix};
use aleph_ir::dist::{CostModel, DistError, DistLayout};
use aleph_ir::Instruction;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

use crate::common::diagonal_of;

/// The CUDA kernel an instruction of a fused rank program launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
/// External controls do not change the kind: a gate with local external
/// controls is priced as a full pass of its target kind, although the kernels
/// skip amplitudes whose control bits are clear. This is a conservative
/// over-estimate; the §6.3 gate's Grover cell (controlled-Z oracle/diffusion)
/// bounds it at model/measured 1.034–1.042 (docs/perf/p6-05-compiler.md §2).
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
                kind: "cost: unsupported instruction",
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
            // The backend runs generic `apply_kq` for k=1; unreachable from
            // `fuse_for_gpu` (FuseKq emits k >= 2), so priced as Dense1.
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

/// Candidate state-class transition rules (#538, spec §2 rule 3). An
/// instruction that "makes the state generic" moves the amplitudes off the
/// small value set where FP64 kernels run fastest at the card's power cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateRule {
    /// Non-diagonal, and some matrix entry has magnitude outside {0, 1, 1/√2}.
    R1,
    /// R1, or some entry (diagonal or not) or `DiagonalPhase` term angle has a
    /// phase that is not a multiple of π/4.
    R2,
}

/// The rule Stage A chose (docs/perf/p6-05-compiler.md §4.1).
pub const STATE_RULE: StateRule = StateRule::R1;

/// Tolerance on magnitudes and phases (spec §2 rule 3).
const CLASS_TOL: f64 = 1e-9;

const NON_FINITE: DistError = DistError::Unsupported {
    kind: "cost: non-finite matrix entry or phase angle",
};

fn special_magnitude(r: f64) -> bool {
    [0.0, 1.0, FRAC_1_SQRT_2]
        .iter()
        .any(|v| (r - v).abs() <= CLASS_TOL)
}

fn quarter_pi_phase(a: f64) -> bool {
    (a - (a / FRAC_PI_4).round() * FRAC_PI_4).abs() <= CLASS_TOL
}

/// Dimension and row-major entries of `gate`'s target matrix. `UnitaryKq` is
/// read from its data, as the backend does (`Gate::matrix` rejects it).
fn matrix_entries(gate: &Gate) -> Result<(usize, Vec<Complex>), DistError> {
    if let Gate::UnitaryKq { k, data } = gate {
        return Ok((1usize << k, data.to_vec()));
    }
    Ok(match gate.matrix()? {
        GateMatrix::M2x2(m) => (2, m.iter().flatten().copied().collect()),
        GateMatrix::M4x4(m) => (4, m.iter().flatten().copied().collect()),
        GateMatrix::M8x8(m) => (8, m.iter().flatten().copied().collect()),
    })
}

/// Whether `instr` makes the state generic under `rule`.
///
/// Judged on the logical, pre-specialise instruction; external controls are
/// ignored (the target matrix decides, as in [`classify`]). `Barrier` is never
/// generic. Errors on a non-finite entry or angle (ADR 0006) and on
/// instructions `classify` rejects.
pub fn makes_generic_under(instr: &Instruction, rule: StateRule) -> Result<bool, DistError> {
    let g = match instr {
        Instruction::Gate(g) => g,
        Instruction::DiagonalPhase(dp) => {
            let mut generic = false;
            for t in &dp.terms {
                if !t.angle.is_finite() {
                    return Err(NON_FINITE);
                }
                generic |= rule == StateRule::R2 && !quarter_pi_phase(t.angle);
            }
            return Ok(generic);
        }
        Instruction::Barrier(_) => return Ok(false),
        _ => {
            return Err(DistError::Unsupported {
                kind: "cost: unsupported instruction",
            })
        }
    };
    let (dim, entries) = matrix_entries(&g.gate)?;
    let (mut non_diagonal, mut odd_magnitude, mut odd_phase) = (false, false, false);
    for (idx, z) in entries.iter().enumerate() {
        if !z.re.is_finite() || !z.im.is_finite() {
            return Err(NON_FINITE);
        }
        let r = z.norm();
        if idx / dim != idx % dim && r > CLASS_TOL {
            non_diagonal = true;
        }
        odd_magnitude |= !special_magnitude(r);
        odd_phase |= r > CLASS_TOL && !quarter_pi_phase(z.arg());
    }
    let r1 = non_diagonal && odd_magnitude;
    Ok(match rule {
        StateRule::R1 => r1,
        StateRule::R2 => r1 || odd_phase,
    })
}

/// [`makes_generic_under`] the chosen [`STATE_RULE`].
pub fn makes_generic(instr: &Instruction) -> Result<bool, DistError> {
    makes_generic_under(instr, STATE_RULE)
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
    ///
    /// Valid near `m_ref`: the `2^(m − m_ref)` scaling has no launch-latency
    /// floor, so costs at small `m` are not meaningful.
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
    /// `k = 0` costs 0. An empty table with `k > 0` returns `f64::INFINITY`
    /// (no bandwidth known), which `plan_cost` rejects.
    pub fn seconds(&self, k: u32, m: u32, amp_bytes: f64) -> f64 {
        if k == 0 {
            return 0.0;
        }
        if self.bw_by_k.is_empty() {
            return f64::INFINITY;
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
    /// Must match the executing `DistSvBackend`'s fusion setting
    /// (`DistSvBackend::fusion`).
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
    /// from `tests/dist_cost_calibrate.rs`, run with
    /// `cargo test --release -p aleph-cuda --features cuda --test dist_cost_calibrate -- --ignored --nocapture`.
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
    /// Rank R − 1 (all global bits 1: every global-controlled gate is live) —
    /// representative, typically the busiest; not a strict bound (a parity
    /// cond can expand into more terms on another rank). Ranks sync at each
    /// exchange anyway.
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
        // Controlled 1q gates keep their 1q kind (external controls priced as a full pass).
        let crz = Instruction::Gate(GateInstance::controlled(
            Gate::Rz(p),
            vec![0u32],
            vec![1u32],
        ));
        assert_eq!(classify(&crz).unwrap(), Diag1);
        let ch = Instruction::Gate(GateInstance::controlled(Gate::H, vec![0u32], vec![1u32]));
        assert_eq!(classify(&ch).unwrap(), Dense1);
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

    /// (instruction, generic under R1, generic under R2) — spec §3.3's list.
    fn rule_table() -> Vec<(&'static str, Instruction, bool, bool)> {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        let p = Param::Concrete;
        let ctl = |gate: Gate, t: u32, c: u32| {
            Instruction::Gate(GateInstance::controlled(gate, vec![t], vec![c]))
        };
        let dp = |angle: f64| {
            Instruction::DiagonalPhase(Box::new(DiagonalPhase {
                n_qubits: 3,
                terms: vec![PhaseTerm {
                    conds: smallvec![0b011],
                    angle,
                }],
            }))
        };
        // Rx(0.3) ⊗ I as a fused 2q block: non-diagonal, |cos 0.15|, |sin 0.15|.
        let (a, b) = ((0.15f64).cos(), (0.15f64).sin());
        let z = Complex::new(0.0, 0.0);
        let (ca, mib) = (Complex::new(a, 0.0), Complex::new(0.0, -b));
        let rx_i = [
            [ca, z, mib, z],
            [z, ca, z, mib],
            [mib, z, ca, z],
            [z, mib, z, ca],
        ];
        // 3q permutation block (simple) read from UnitaryKq data.
        let mut perm = vec![z; 64];
        for r in 0..8 {
            perm[r * 8 + (r ^ 1)] = Complex::new(1.0, 0.0);
        }
        let kq = Gate::UnitaryKq {
            k: 3,
            data: perm.into_boxed_slice(),
        };
        vec![
            ("H", g(Gate::H, &[0]), false, false),
            ("S", g(Gate::S, &[0]), false, false),
            ("T", g(Gate::T, &[0]), false, false),
            ("X", g(Gate::X, &[0]), false, false),
            ("Y", g(Gate::Y, &[0]), false, false),
            ("CNOT", g(Gate::Cnot, &[0, 1]), false, false),
            ("Toffoli", g(Gate::Toffoli, &[0, 1, 2]), false, false),
            ("CZ", g(Gate::Cz, &[0, 1]), false, false),
            ("Iswap", g(Gate::Iswap, &[0, 1]), false, false),
            ("MCZ", ctl(Gate::Z, 3, 0), false, false),
            ("Rx(0.3)", g(Gate::Rx(p(0.3)), &[0]), true, true),
            ("Ry(0.3)", g(Gate::Ry(p(0.3)), &[0]), true, true),
            ("Rz(0.3)", g(Gate::Rz(p(0.3)), &[0]), false, true),
            ("Rz(pi/2)", g(Gate::Rz(p(FRAC_PI_2)), &[0]), false, false),
            (
                "Phase(pi/4)",
                g(Gate::Phase(p(FRAC_PI_4)), &[0]),
                false,
                false,
            ),
            ("Phase(0.3)", g(Gate::Phase(p(0.3)), &[0]), false, true),
            ("CRz(0.3)", g(Gate::CRz(p(0.3)), &[0, 1]), false, true),
            (
                "Unitary2q Rx⊗I",
                g(Gate::Unitary2q(Box::new(rx_i)), &[0, 1]),
                true,
                true,
            ),
            ("UnitaryKq perm", g(kq, &[0, 1, 2]), false, false),
            ("DiagonalPhase pi/4", dp(FRAC_PI_4), false, false),
            ("DiagonalPhase 0.3", dp(0.3), false, true),
            ("ext-ctl Rx(0.3)", ctl(Gate::Rx(p(0.3)), 0, 1), true, true),
            ("ext-ctl Rz(0.3)", ctl(Gate::Rz(p(0.3)), 0, 1), false, true),
            (
                "Barrier",
                Instruction::Barrier(smallvec![0, 1]),
                false,
                false,
            ),
        ]
    }

    #[test]
    fn makes_generic_rule_tables() {
        for (name, instr, r1, r2) in rule_table() {
            assert_eq!(
                makes_generic_under(&instr, StateRule::R1).unwrap(),
                r1,
                "R1 {name}"
            );
            assert_eq!(
                makes_generic_under(&instr, StateRule::R2).unwrap(),
                r2,
                "R2 {name}"
            );
            assert_eq!(
                makes_generic(&instr).unwrap(),
                makes_generic_under(&instr, STATE_RULE).unwrap(),
                "STATE_RULE {name}"
            );
        }
    }

    #[test]
    fn makes_generic_rejects_bad_input() {
        let nan_dp = Instruction::DiagonalPhase(Box::new(DiagonalPhase {
            n_qubits: 2,
            terms: vec![PhaseTerm {
                conds: smallvec![0b01],
                angle: f64::NAN,
            }],
        }));
        let nan_rx = g(Gate::Rx(Param::Concrete(f64::NAN)), &[0]);
        let measure = Instruction::Measure { qubit: 0, clbit: 0 };
        for rule in [StateRule::R1, StateRule::R2] {
            assert!(
                makes_generic_under(&nan_dp, rule).is_err(),
                "{rule:?} NaN angle"
            );
            assert!(
                makes_generic_under(&nan_rx, rule).is_err(),
                "{rule:?} Rx(NaN)"
            );
            assert!(
                makes_generic_under(&measure, rule).is_err(),
                "{rule:?} Measure"
            );
        }
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
        let empty = LinkModel { bw_by_k: vec![] };
        assert_eq!(empty.seconds(0, 3, 16.0), 0.0);
        assert_eq!(empty.seconds(1, 3, 16.0), f64::INFINITY);
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
