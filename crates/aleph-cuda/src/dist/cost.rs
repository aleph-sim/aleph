//! P6-05 GPU cost model (spec §6.2). `classify` mirrors the decision order of
//! `CudaSvBackend::apply_gate` (sv/backend.rs) and `apply_diagonal_phase`, so
//! each instruction is priced as the kernel it actually launches.

use aleph_core::{Complex, Gate, GateMatrix};
use aleph_ir::dist::{CostModel, DistError, DistLayout, DistPlan, DistStep};
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
        // Guard before the shift: k >= 64 would overflow it, and a wrong data
        // length would misjudge the diagonal (never panic on input).
        let k = u32::from(*k);
        if !(1..=5).contains(&k) || data.len() != 1usize << (2 * k) {
            return Err(DistError::Unsupported {
                kind: "cost: malformed UnitaryKq",
            });
        }
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
    /// Generic-state values; the fields above are the simple-state (uniform) values.
    pub generic: GenericTimes,
}

/// Per-launch seconds at `m_ref` on a **generic** state, for the kinds whose
/// time depends on the state class (#538, spec §2 rule 2). `None`: the kind
/// keeps its one constant, the simple value in [`KindTimes`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GenericTimes {
    /// Generic-state `Dense1` seconds.
    pub dense1: Option<f64>,
    /// Generic-state `Dense2` seconds.
    pub dense2: Option<f64>,
    /// Generic-state `Dense3` seconds.
    pub dense3: Option<f64>,
    /// Generic-state `Diag1` seconds.
    pub diag1: Option<f64>,
    /// Generic-state `DiagK` seconds.
    pub diag_k: Option<f64>,
    /// Generic-state `Cnot` seconds.
    pub cnot: Option<f64>,
    /// Generic-state `PhasePoly` fixed cost.
    pub phase_base: Option<f64>,
    /// Generic-state `PhasePoly` per-term cost, terms with ≤ 1 cond.
    pub phase_term: Option<f64>,
    /// Generic-state `PhasePoly` per-term cost, terms with ≥ 2 conds (AND).
    pub phase_term_multi: Option<f64>,
}

impl GenericTimes {
    /// No kind is state-dependent: prices exactly as one constant per kind.
    pub const NONE: Self = Self {
        dense1: None,
        dense2: None,
        dense3: None,
        diag1: None,
        diag_k: None,
        cnot: None,
        phase_base: None,
        phase_term: None,
        phase_term_multi: None,
    };
}

impl KindTimes {
    /// Seconds for one launch of kind `k` on a `2^m` slice, on a generic state
    /// if `generic` (a kind without a generic value uses its simple one).
    ///
    /// Valid near `m_ref`: the `2^(m − m_ref)` scaling has no launch-latency
    /// floor, so costs at small `m` are not meaningful.
    pub fn seconds(&self, k: KernelKind, m: u32, generic: bool) -> f64 {
        let pick = |simple: f64, gen: Option<f64>| {
            if generic {
                gen.unwrap_or(simple)
            } else {
                simple
            }
        };
        let g = &self.generic;
        let at_ref = match k {
            KernelKind::Dense1 => pick(self.dense1, g.dense1),
            KernelKind::Dense2 => pick(self.dense2, g.dense2),
            KernelKind::Dense3 => pick(self.dense3, g.dense3),
            KernelKind::Diag1 => pick(self.diag1, g.diag1),
            KernelKind::DiagK => pick(self.diag_k, g.diag_k),
            KernelKind::Cnot => pick(self.cnot, g.cnot),
            KernelKind::PhasePoly { single, multi } => {
                pick(self.phase_base, g.phase_base)
                    + pick(self.phase_term, g.phase_term) * single as f64
                    + pick(self.phase_term_multi, g.phase_term_multi) * multi as f64
            }
            KernelKind::Free => 0.0,
        };
        at_ref * 2f64.powi(m as i32 - self.m_ref as i32)
    }
}

/// Whether each step of `plan` is priced on a generic state (spec §3.2).
///
/// The walk starts simple. A `Local` step that holds any instruction for
/// which [`makes_generic`] is true is priced generic in full, and so is every
/// later step. Exchanges permute amplitudes, so they keep the class.
pub fn state_classes(plan: &DistPlan) -> Result<Vec<bool>, DistError> {
    let mut generic = false;
    let mut out = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        if let DistStep::Local(instrs) = step {
            if !generic {
                for i in instrs {
                    if makes_generic(i)? {
                        generic = true;
                        break;
                    }
                }
            }
        }
        out.push(generic);
    }
    Ok(out)
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
    /// `generic`: price on a generic state (see [`state_classes`]).
    pub fn rank_segment(
        &self,
        instrs: &[Instruction],
        layout: DistLayout,
        rank: u32,
        generic: bool,
    ) -> Result<f64, DistError> {
        let c = super::rank_circuit(instrs, layout, rank, self.fuse)?;
        let m = layout.m();
        let mut t = 0.0;
        for i in c.instructions() {
            t += self.kinds.seconds(classify(i)?, m, generic);
        }
        Ok(t)
    }

    /// All-ranks compute of `plan` (Σ over `Local` steps and ranks), with the
    /// state-class walk: the quantity the §6.3 gate compares to one card
    /// running every rank.
    pub fn all_ranks(&self, plan: &DistPlan) -> Result<f64, DistError> {
        let l = plan.layout;
        let mut all = 0.0;
        for (step, generic) in plan.steps.iter().zip(state_classes(plan)?) {
            if let DistStep::Local(instrs) = step {
                for r in 0..l.ranks() {
                    all += self.rank_segment(instrs, l, r, generic)?;
                }
            }
        }
        Ok(all)
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
    /// Without plan context this prices a **simple** state; `plan_cost` uses `step_costs`, which carries the class.
    ///
    /// Rank R − 1 (all global bits 1: every global-controlled gate is live) —
    /// representative, typically the busiest; not a strict bound (a parity
    /// cond can expand into more terms on another rank). Ranks sync at each
    /// exchange anyway.
    fn local_segment(&self, instrs: &[Instruction], layout: DistLayout) -> Result<f64, DistError> {
        self.rank_segment(instrs, layout, layout.ranks() - 1, false)
    }

    /// Per-step costs with the state-class walk ([`state_classes`]); `Local`
    /// steps priced at rank R − 1 as in [`CostModel::local_segment`].
    fn step_costs(&self, plan: &DistPlan) -> Result<Vec<f64>, DistError> {
        let l = plan.layout;
        plan.steps
            .iter()
            .zip(state_classes(plan)?)
            .map(|(step, generic)| match step {
                DistStep::Local(instrs) => self.rank_segment(instrs, l, l.ranks() - 1, generic),
                DistStep::Exchange { global_bits } => {
                    Ok(self.exchange(global_bits.len() as u32, l.m()))
                }
            })
            .collect()
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
    generic: GenericTimes::NONE,
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
    generic: GenericTimes::NONE,
};

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Complex, GateInstance, Param};
    use aleph_ir::dist::{DistLayout, DistPlan, DistStep};
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
            generic: GenericTimes::NONE,
        }
    }

    fn rx(q: u32) -> Instruction {
        g(Gate::Rx(Param::Concrete(0.3)), &[q])
    }

    /// unit_times with every kind's generic value = 10 × simple.
    fn unit_times_generic() -> KindTimes {
        let s = unit_times();
        KindTimes {
            generic: GenericTimes {
                dense1: Some(10.0 * s.dense1),
                dense2: Some(10.0 * s.dense2),
                dense3: Some(10.0 * s.dense3),
                diag1: Some(10.0 * s.diag1),
                diag_k: Some(10.0 * s.diag_k),
                cnot: Some(10.0 * s.cnot),
                phase_base: Some(10.0 * s.phase_base),
                phase_term: Some(10.0 * s.phase_term),
                phase_term_multi: Some(10.0 * s.phase_term_multi),
            },
            ..s
        }
    }

    fn model(kinds: KindTimes) -> GpuCostModel {
        GpuCostModel {
            kinds,
            link: LinkModel::aws_g6_fp64(),
            amp_bytes: 16.0,
            fuse: false,
        }
    }

    /// n=21, g=1 (m=20=m_ref): steps = [Local(a), Exchange(top), Local(b)].
    fn two_step_plan(a: Vec<Instruction>, b: Vec<Instruction>) -> DistPlan {
        let l = DistLayout::new(21, 1).unwrap();
        DistPlan {
            layout: l,
            steps: vec![
                DistStep::Local(a),
                DistStep::Exchange {
                    global_bits: smallvec![20],
                },
                DistStep::Local(b),
            ],
            ..aleph_ir::dist::plan(
                &aleph_ir::Circuit::new(21, 0),
                l,
                aleph_ir::dist::Router::Naive,
            )
            .unwrap()
        }
    }

    #[test]
    fn none_generic_prices_like_simple() {
        let t = unit_times();
        for k in [KernelKind::Dense1, KernelKind::Dense3, KernelKind::Cnot] {
            assert_eq!(
                t.seconds(k, 21, true).to_bits(),
                t.seconds(k, 21, false).to_bits()
            );
        }
        let tg = unit_times_generic();
        assert_eq!(tg.seconds(KernelKind::Dense2, 20, true), 20.0);
        assert_eq!(tg.seconds(KernelKind::Dense2, 20, false), 2.0);
    }

    #[test]
    fn class_walk_survives_exchanges() {
        // Step 0 Clifford (simple), step 2 has Rx(0.3): step 0 simple, 2 generic.
        let p = two_step_plan(vec![g(Gate::H, &[0])], vec![g(Gate::H, &[0]), rx(1)]);
        assert_eq!(state_classes(&p).unwrap(), vec![false, false, true]);
        let costs = model(unit_times_generic()).step_costs(&p).unwrap();
        // H = Dense1: simple 1.0, generic 10.0; Rx = Dense1 too.
        assert_eq!(costs[0], 1.0);
        assert_eq!(costs[2], 20.0);
        // Transition first: everything after it stays generic across the exchange.
        let q = two_step_plan(vec![rx(0)], vec![g(Gate::H, &[0])]);
        assert_eq!(state_classes(&q).unwrap(), vec![true, true, true]);
        assert_eq!(model(unit_times_generic()).step_costs(&q).unwrap()[2], 10.0);
    }

    #[test]
    fn no_generic_gate_prices_like_pr3() {
        let p = two_step_plan(
            vec![g(Gate::H, &[0]), g(Gate::Cnot, &[0, 1])],
            vec![g(Gate::T, &[1]), g(Gate::Toffoli, &[0, 1, 2])],
        );
        let pr3 = model(unit_times());
        let new = model(unit_times_generic());
        let a = aleph_ir::dist::plan_cost(&p, &new).unwrap();
        let b = aleph_ir::dist::plan_cost(&p, &pr3).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
        assert_eq!(
            new.all_ranks(&p).unwrap().to_bits(),
            pr3.all_ranks(&p).unwrap().to_bits()
        );
    }

    #[test]
    fn empty_locals_cost_nothing() {
        let p = two_step_plan(vec![], vec![]);
        let m = model(unit_times_generic());
        assert_eq!(state_classes(&p).unwrap(), vec![false, false, false]);
        assert_eq!(m.all_ranks(&p).unwrap(), 0.0);
        let costs = m.step_costs(&p).unwrap();
        assert_eq!(costs[0], 0.0);
        assert_eq!(costs[1], m.exchange(1, 20));
    }

    #[test]
    fn globally_controlled_generic_gate_flips_all_ranks() {
        // Rx(0.3) on local q0 controlled by global q20: rank 0 drops it, rank 1
        // keeps it. The class is plan-wide, so both ranks price H(1) generic.
        let ctl_rx = Instruction::Gate(GateInstance::controlled(
            Gate::Rx(Param::Concrete(0.3)),
            vec![0u32],
            vec![20u32],
        ));
        let p = two_step_plan(vec![ctl_rx, g(Gate::H, &[1])], vec![]);
        assert!(state_classes(&p).unwrap()[0]);
        let m = model(unit_times_generic());
        let l = p.layout;
        let DistStep::Local(instrs) = &p.steps[0] else {
            unreachable!()
        };
        let r0 = m.rank_segment(instrs, l, 0, true).unwrap();
        let r1 = m.rank_segment(instrs, l, 1, true).unwrap();
        assert_eq!(r0, 10.0); // H only, generic
        assert_eq!(r1, 20.0); // Rx + H, generic
        assert_eq!(m.all_ranks(&p).unwrap(), 30.0);
    }

    #[test]
    fn local_segment_alone_prices_simple() {
        let m = model(unit_times_generic());
        let l = DistLayout::new(21, 1).unwrap();
        assert_eq!(m.local_segment(&[rx(0)], l).unwrap(), 1.0);
    }

    #[test]
    fn makes_generic_rejects_malformed_unitary_kq() {
        let z = Complex::new(0.0, 0.0);
        let short = g(
            Gate::UnitaryKq {
                k: 3,
                data: vec![z; 10].into_boxed_slice(),
            },
            &[0, 1, 2],
        );
        let wide = g(
            Gate::UnitaryKq {
                k: 6,
                data: vec![z; 16].into_boxed_slice(),
            },
            &[0, 1, 2, 3, 4, 5],
        );
        for rule in [StateRule::R1, StateRule::R2] {
            assert!(
                makes_generic_under(&short, rule).is_err(),
                "{rule:?} k=3 len 10"
            );
            assert!(makes_generic_under(&wide, rule).is_err(), "{rule:?} k=6");
        }
    }

    #[test]
    fn kind_seconds_scale_with_slice() {
        let t = unit_times();
        assert_eq!(t.seconds(KernelKind::Dense2, 20, false), 2.0);
        assert_eq!(t.seconds(KernelKind::Dense2, 22, false), 8.0);
        let ph = KernelKind::PhasePoly {
            single: 10,
            multi: 0,
        };
        assert!((t.seconds(ph, 20, false) - 2.0).abs() < 1e-12);
        let ph = KernelKind::PhasePoly {
            single: 10,
            multi: 4,
        };
        assert!((t.seconds(ph, 21, false) - 2.0 * (2.0 + 4.0 * 0.05)).abs() < 1e-12);
        assert_eq!(t.seconds(KernelKind::Free, 25, false), 0.0);
    }

    #[test]
    fn scaling_handles_m_below_ref() {
        let t = unit_times();
        assert_eq!(t.seconds(KernelKind::Dense1, 18, false), 0.25);
        assert!(t.seconds(KernelKind::Dense1, 0, false) > 0.0);
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
        let t = model.rank_segment(&instrs, l, 1, false).unwrap();
        assert!((t - (1.0 + 0.25 + 0.5)).abs() < 1e-12);
    }
}
