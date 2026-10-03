//! Per-rank rewriting of physical instructions.
//!
//! Each rank knows its own global bits, so any gate whose dependence on a
//! global qubit is diagonal (a phase) or a control can be rewritten into an
//! `m`-qubit local instruction. Non-diagonal *targets* on global qubits are
//! the planner's job (it inserts an `Exchange` first) and are rejected here.

use aleph_core::{Complex, Gate, GateInstance, GateMatrix};
use smallvec::{smallvec, SmallVec};

use super::{DistError, DistLayout};
use crate::{DiagonalPhase, Instruction, PhaseTerm};

/// Rewrite physical `instr` for `rank`; `Ok(None)` = no-op on this rank.
///
/// Output qubit indices are all `< layout.m()`; `DiagonalPhase` outputs have
/// `n_qubits = layout.m()`.
pub fn specialize(
    instr: &Instruction,
    layout: DistLayout,
    rank: u32,
) -> Result<Option<Instruction>, DistError> {
    match instr {
        Instruction::Gate(g) => specialize_gate(g, layout, rank),
        Instruction::DiagonalPhase(dp) => Ok(specialize_dp(dp, layout, rank)),
        Instruction::Barrier(_) => Ok(None),
        Instruction::Measure { .. } => Err(DistError::Unsupported { kind: "measure" }),
        Instruction::Reset(_) => Err(DistError::Unsupported { kind: "reset" }),
        Instruction::TiledBlock(_) => Err(DistError::Unsupported {
            kind: "tiled_block",
        }),
    }
}

fn specialize_gate(
    g: &GateInstance,
    l: DistLayout,
    rank: u32,
) -> Result<Option<Instruction>, DistError> {
    // 1. External controls: a global control selects whole ranks.
    let mut controls: SmallVec<[u32; 2]> = SmallVec::new();
    for &c in &g.controls {
        if l.is_global(c) {
            if l.rank_bit(rank, c) == 0 {
                return Ok(None);
            }
        } else {
            controls.push(c);
        }
    }

    // 2. In-qubit controls of non-diagonal controlled gates.
    let q = &g.qubits;
    let (gate, qubits): (Gate, SmallVec<[u32; 4]>) = match &g.gate {
        Gate::Cnot | Gate::CRx(_) | Gate::CRy(_) if l.is_global(q[0]) => {
            if l.rank_bit(rank, q[0]) == 0 {
                return Ok(None);
            }
            let reduced = match &g.gate {
                Gate::Cnot => Gate::X,
                Gate::CRx(p) => Gate::Rx(*p),
                Gate::CRy(p) => Gate::Ry(*p),
                _ => return unreachable_shape(),
            };
            (reduced, smallvec![q[1]])
        }
        Gate::Toffoli if l.is_global(q[0]) || l.is_global(q[1]) => {
            let mut kept: SmallVec<[u32; 2]> = SmallVec::new();
            for &c in &q[0..2] {
                if l.is_global(c) {
                    if l.rank_bit(rank, c) == 0 {
                        return Ok(None);
                    }
                } else {
                    kept.push(c);
                }
            }
            match kept.as_slice() {
                [] => (Gate::X, smallvec![q[2]]),
                [c] => (Gate::Cnot, smallvec![*c, q[2]]),
                _ => return unreachable_shape(),
            }
        }
        other => (other.clone(), q.clone()),
    };

    // 3. Diagonal gates touching a global qubit fold to local diagonals.
    if gate.is_diagonal() && qubits.iter().any(|&p| l.is_global(p)) {
        return diag_reduce(&gate, &qubits, &controls, l, rank).map(Some);
    }

    // 4. Anything still global is a planner bug.
    if let Some(&p) = qubits.iter().find(|&&p| l.is_global(p)) {
        return Err(DistError::GlobalTarget { qubit: p });
    }
    Ok(Some(Instruction::Gate(GateInstance {
        gate,
        qubits,
        controls,
    })))
}

/// Guard for match arms the enclosing pattern already excludes; keeps library
/// code panic-free.
fn unreachable_shape<T>() -> Result<T, DistError> {
    Err(DistError::Unsupported {
        kind: "internal: unreachable gate shape",
    })
}

fn diagonal_of(gate: &Gate) -> Result<SmallVec<[Complex; 8]>, DistError> {
    Ok(match gate.matrix()? {
        GateMatrix::M2x2(m) => (0..2).map(|i| m[i][i]).collect(),
        GateMatrix::M4x4(m) => (0..4).map(|i| m[i][i]).collect(),
        GateMatrix::M8x8(m) => (0..8).map(|i| m[i][i]).collect(),
    })
}

/// Restrict a diagonal gate to this rank's values of its global qubits.
fn diag_reduce(
    gate: &Gate,
    qubits: &[u32],
    controls: &[u32],
    l: DistLayout,
    rank: u32,
) -> Result<Instruction, DistError> {
    let d = diagonal_of(gate)?;
    let a = qubits.len();
    let local: SmallVec<[u32; 4]> = qubits
        .iter()
        .copied()
        .filter(|&p| !l.is_global(p))
        .collect();
    let nl = local.len();
    // d'[i'] over the local qubits (MSB = local[0]); matrix index bit for
    // qubits[j] is (a-1-j) — `qubits[0]` is the MSB (aleph-core kinds.rs).
    let mut dl: SmallVec<[Complex; 4]> = SmallVec::new();
    for ip in 0..(1usize << nl) {
        let mut i = 0usize;
        let mut t = 0usize;
        for (j, &p) in qubits.iter().enumerate() {
            let bit = if l.is_global(p) {
                l.rank_bit(rank, p) as usize
            } else {
                let b = (ip >> (nl - 1 - t)) & 1;
                t += 1;
                b
            };
            i |= bit << (a - 1 - j);
        }
        dl.push(d[i]);
    }
    let ctrl: SmallVec<[u32; 2]> = controls.iter().copied().collect();
    Ok(match nl {
        0 => {
            // A rank-dependent phase, gated on any remaining local controls.
            let conds: SmallVec<[u64; 2]> = ctrl.iter().map(|&c| 1u64 << c).collect();
            Instruction::DiagonalPhase(Box::new(DiagonalPhase {
                n_qubits: l.m(),
                terms: vec![PhaseTerm {
                    conds,
                    angle: dl[0].arg(),
                }],
            }))
        }
        1 => Instruction::Gate(GateInstance {
            gate: Gate::Unitary1qDiag(Box::new([dl[0], dl[1]])),
            qubits: local,
            controls: ctrl,
        }),
        2 => {
            let z = Complex::new(0.0, 0.0);
            let mut m = [[z; 4]; 4];
            for (k, row) in m.iter_mut().enumerate() {
                row[k] = dl[k];
            }
            Instruction::Gate(GateInstance {
                gate: Gate::Unitary2q(Box::new(m)),
                qubits: local,
                controls: ctrl,
            })
        }
        _ => {
            return Err(DistError::Unsupported {
                kind: "diagonal gate on >2 local qubits with a global qubit",
            })
        }
    })
}

/// Fold this rank's global parities into a `DiagonalPhase`.
///
/// A cond `popcount(mask & x)` odd splits into a local mask `lo` and a known
/// global parity `hi`; it fires iff `popcount(lo & x) ⊕ hi` is odd. `hi = 1`
/// with non-empty `lo` is a *negated* local cond, expanded by
/// inclusion–exclusion: `[A ∧ ¬B] = [A] − [A ∧ B]`.
fn specialize_dp(dp: &DiagonalPhase, l: DistLayout, rank: u32) -> Option<Instruction> {
    let m = l.m();
    let lo_mask: u64 = if m >= 64 { u64::MAX } else { (1u64 << m) - 1 };
    let rank_mask = if m >= 64 { 0 } else { u64::from(rank) << m };
    let mut out: Vec<PhaseTerm> = Vec::new();
    'term: for t in &dp.terms {
        let mut pos: SmallVec<[u64; 2]> = SmallVec::new();
        let mut neg: SmallVec<[u64; 2]> = SmallVec::new();
        for &mask in &t.conds {
            let lo = mask & lo_mask;
            let hi = (mask & rank_mask).count_ones() & 1;
            match (hi, lo) {
                (0, 0) => continue 'term, // never fires on this rank
                (0, lo) => pos.push(lo),
                (_, 0) => {} // always true on this rank
                (_, lo) => neg.push(lo),
            }
        }
        // [A ∧ ¬B1 ∧ … ∧ ¬Bk] = Σ_{S ⊆ B} (-1)^|S| [A ∧ S]
        for s in 0u32..(1 << neg.len()) {
            let mut conds = pos.clone();
            for (j, &b) in neg.iter().enumerate() {
                if (s >> j) & 1 == 1 {
                    conds.push(b);
                }
            }
            let sign = if s.count_ones() % 2 == 0 { 1.0 } else { -1.0 };
            out.push(PhaseTerm {
                conds,
                angle: sign * t.angle,
            });
        }
    }
    if out.is_empty() {
        return None;
    }
    Some(Instruction::DiagonalPhase(Box::new(DiagonalPhase {
        n_qubits: m,
        terms: out,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DiagonalPhase, PhaseTerm};
    use aleph_core::{Complex, Gate, GateInstance, Param};
    use smallvec::smallvec;

    fn lay() -> DistLayout {
        DistLayout::new(4, 2).unwrap() // m = 2; physical 2,3 are global
    }

    fn gate(g: Gate, q: &[u32]) -> Instruction {
        Instruction::Gate(GateInstance::new(g, q.to_vec()))
    }

    #[test]
    fn local_gate_passes_through() {
        let out = specialize(&gate(Gate::H, &[1]), lay(), 3).unwrap().unwrap();
        match out {
            Instruction::Gate(g) => {
                assert!(matches!(g.gate, Gate::H));
                assert_eq!(g.qubits.as_slice(), &[1]);
            }
            _ => panic!("expected gate"),
        }
    }

    #[test]
    fn cnot_global_control() {
        // control phys 2 (rank bit 0), target local 0
        let i = gate(Gate::Cnot, &[2, 0]);
        assert!(specialize(&i, lay(), 0b00).unwrap().is_none());
        match specialize(&i, lay(), 0b01).unwrap().unwrap() {
            Instruction::Gate(g) => {
                assert!(matches!(g.gate, Gate::X));
                assert_eq!(g.qubits.as_slice(), &[0]);
                assert!(g.controls.is_empty());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn external_global_control_skips_or_drops() {
        let gi = GateInstance::controlled(Gate::H, vec![0u32], vec![3u32, 1u32]);
        let i = Instruction::Gate(gi);
        assert!(specialize(&i, lay(), 0b01).unwrap().is_none()); // phys 3 = rank bit 1 = 0
        match specialize(&i, lay(), 0b10).unwrap().unwrap() {
            Instruction::Gate(g) => assert_eq!(g.controls.as_slice(), &[1]),
            _ => panic!(),
        }
    }

    #[test]
    fn toffoli_one_and_two_global_controls() {
        let i = gate(Gate::Toffoli, &[3, 1, 0]);
        assert!(specialize(&i, lay(), 0b00).unwrap().is_none());
        match specialize(&i, lay(), 0b10).unwrap().unwrap() {
            Instruction::Gate(g) => {
                assert!(matches!(g.gate, Gate::Cnot));
                assert_eq!(g.qubits.as_slice(), &[1, 0]);
            }
            _ => panic!(),
        }
        let both = gate(Gate::Toffoli, &[2, 3, 0]);
        match specialize(&both, lay(), 0b11).unwrap().unwrap() {
            Instruction::Gate(g) => assert!(matches!(g.gate, Gate::X)),
            _ => panic!(),
        }
        assert!(specialize(&both, lay(), 0b01).unwrap().is_none());
    }

    #[test]
    fn cz_one_global_becomes_local_diag() {
        // rank bit for phys 2 is 1 on rank 0b01 -> Z on local q0
        let i = gate(Gate::Cz, &[0, 2]);
        match specialize(&i, lay(), 0b01).unwrap().unwrap() {
            Instruction::Gate(g) => {
                let Gate::Unitary1qDiag(d) = &g.gate else {
                    panic!("{:?}", g.gate)
                };
                assert!((d[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
                assert!((d[1] - Complex::new(-1.0, 0.0)).norm() < 1e-12);
                assert_eq!(g.qubits.as_slice(), &[0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn cz_both_global_is_rank_phase() {
        let i = gate(Gate::Cz, &[2, 3]);
        let Instruction::DiagonalPhase(dp) = specialize(&i, lay(), 0b11).unwrap().unwrap() else {
            panic!()
        };
        assert_eq!(dp.n_qubits, 2);
        assert!((dp.phase_at(0).abs() - std::f64::consts::PI).abs() < 1e-12);
        if let Some(Instruction::DiagonalPhase(dp)) = specialize(&i, lay(), 0b01).unwrap() {
            assert!(dp.phase_at(0).abs() < 1e-12);
        }
    }

    #[test]
    fn rz_on_global_is_rank_dependent_phase() {
        let i = gate(Gate::Rz(Param::Concrete(0.7)), &[3]);
        let phase = |rank| match specialize(&i, lay(), rank).unwrap().unwrap() {
            Instruction::DiagonalPhase(dp) => dp.phase_at(0),
            _ => panic!(),
        };
        assert!((phase(0b00) + 0.35).abs() < 1e-12);
        assert!((phase(0b10) - 0.35).abs() < 1e-12);
    }

    #[test]
    fn non_diagonal_global_target_errors() {
        let i = gate(Gate::H, &[2]);
        assert!(matches!(
            specialize(&i, lay(), 0),
            Err(DistError::GlobalTarget { qubit: 2 })
        ));
    }

    #[test]
    fn diagonal_phase_matches_full_index_semantics() {
        // terms over physical masks mixing local (0,1) and global (2,3) bits
        let dp = DiagonalPhase {
            n_qubits: 4,
            terms: vec![
                PhaseTerm {
                    conds: smallvec![0b0101],
                    angle: 0.3,
                },
                PhaseTerm {
                    conds: smallvec![0b1000, 0b0010],
                    angle: -1.1,
                },
                PhaseTerm {
                    conds: smallvec![0b1100, 0b0011],
                    angle: 0.9,
                },
                PhaseTerm {
                    conds: smallvec![0b0100],
                    angle: 0.45,
                },
                PhaseTerm {
                    conds: smallvec![],
                    angle: 0.2,
                },
            ],
        };
        let full = Instruction::DiagonalPhase(Box::new(dp.clone()));
        for rank in 0..4u32 {
            let out = specialize(&full, lay(), rank).unwrap();
            for local in 0..4u64 {
                let x = ((rank as u64) << 2) | local;
                let want = dp.phase_at(x);
                let got = match &out {
                    Some(Instruction::DiagonalPhase(d)) => d.phase_at(local),
                    None => 0.0,
                    _ => panic!(),
                };
                assert!(
                    (want - got).abs() < 1e-12,
                    "rank {rank} local {local}: {want} vs {got}"
                );
            }
        }
    }

    #[test]
    fn barrier_dropped_measure_rejected() {
        assert!(specialize(&Instruction::Barrier(smallvec![0, 2]), lay(), 1)
            .unwrap()
            .is_none());
        assert!(matches!(
            specialize(&Instruction::Measure { qubit: 0, clbit: 0 }, lay(), 0),
            Err(DistError::Unsupported { kind: "measure" })
        ));
    }
}
