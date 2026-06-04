//! `RelabelQubits` — permute qubit indices so high-traffic qubits occupy
//! low (cache-local) bit positions, maximizing the gates `TileBlock` can
//! confine to a tile. Records the permutation `π[logical] = physical` on
//! the circuit (`set_qubit_permutation`); the run driver un-permutes
//! results. Runs FIRST in the pipeline (before fusion). Conservative: only
//! commits a non-identity permutation when it strictly increases the number
//! of tile-confinable gate targets — correctness never depends on the
//! heuristic, only the speedup. See the P2-09 design spec.

use crate::diagonal_phase::DiagonalPhase;
use crate::passes::tile_block::DEFAULT_TILE_BITS;
use crate::passes::{Pass, PassError, PassStats};
use crate::{Circuit, Instruction};

/// Pass that relabels qubit indices to push high-traffic qubits to low bit
/// positions, recording the resulting permutation on the circuit so the run
/// driver can un-permute the final state. See the module doc for the
/// net-win guard that keeps this conservative.
pub struct RelabelQubits {
    /// Number of low bit positions a tile spans. Mirrors
    /// [`TileBlock::tile_bits`](crate::passes::TileBlock); the net-win
    /// guard uses it to predict confinability.
    pub tile_bits: u8,
}

impl Default for RelabelQubits {
    fn default() -> Self {
        Self {
            tile_bits: DEFAULT_TILE_BITS,
        }
    }
}

impl RelabelQubits {
    /// Construct with an explicit tile width (used by tests / a tuned backend).
    pub fn new(tile_bits: u8) -> Self {
        Self { tile_bits }
    }
}

impl Pass for RelabelQubits {
    fn name(&self) -> &'static str {
        "RelabelQubits"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let n = circuit.num_qubits as usize;
        let len = circuit.instructions.len();
        let noop = PassStats {
            gates_before: len,
            gates_after: len,
            transformations: 0,
        };
        // Nothing to gain for ≤1 qubit, and never relabel twice.
        if n <= 1 || circuit.qubit_permutation().is_some() {
            return Ok(noop);
        }
        // 1. Traffic score per logical qubit (gate appearances).
        let mut score = vec![0u64; n];
        for inst in &circuit.instructions {
            for q in inst.used_qubits() {
                score[q as usize] += 1;
            }
        }
        // 2. physical_of[logical] = physical bit. Highest score → lowest bit.
        //    Stable tie-break by qubit index keeps it deterministic.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| score[b].cmp(&score[a]).then(a.cmp(&b)));
        let mut physical_of = vec![0u32; n]; // π[logical] = physical
        for (phys_bit, &logical) in order.iter().enumerate() {
            physical_of[logical] = phys_bit as u32;
        }
        // 3. Net-win guard: count tile-confinable gates before vs after.
        let before = count_confinable(circuit, self.tile_bits, None);
        let after = count_confinable(circuit, self.tile_bits, Some(&physical_of));
        if after <= before {
            return Ok(noop); // identity — no benefit
        }
        // 4. Rewrite all instruction qubit indices through physical_of.
        for inst in &mut circuit.instructions {
            remap_instruction(inst, &physical_of);
        }
        circuit.set_qubit_permutation(physical_of.into_boxed_slice());
        Ok(PassStats {
            gates_before: len,
            gates_after: len,
            transformations: 1,
        })
    }
}

/// Count gates whose targets are all `< tile_bits`, optionally after
/// remapping each qubit `q -> map[q]`. Mirrors `TileBlock::confinable`'s
/// target rule (targets only; arity ≤ 2; representable matrix).
fn count_confinable(circuit: &Circuit, tile_bits: u8, map: Option<&[u32]>) -> usize {
    let tb = tile_bits as u32;
    let remap = |q: u32| map.map_or(q, |m| m[q as usize]);
    circuit
        .instructions
        .iter()
        .filter(|inst| match inst {
            Instruction::Gate(g) => {
                g.qubits.len() <= 2
                    && g.gate.matrix().is_ok()
                    && g.qubits.iter().all(|&q| remap(q) < tb)
            }
            _ => false,
        })
        .count()
}

/// Rewrite every qubit index in `inst` through `map` (`q -> map[q]`).
fn remap_instruction(inst: &mut Instruction, map: &[u32]) {
    match inst {
        Instruction::Gate(g) => {
            for q in g.qubits.iter_mut() {
                *q = map[*q as usize];
            }
            for c in g.controls.iter_mut() {
                *c = map[*c as usize];
            }
        }
        Instruction::Measure { qubit, .. } => {
            *qubit = map[*qubit as usize];
        }
        Instruction::Reset(q) => {
            *q = map[*q as usize];
        }
        Instruction::Barrier(qs) => {
            for q in qs.iter_mut() {
                *q = map[*q as usize];
            }
        }
        Instruction::DiagonalPhase(dp) => {
            remap_diagonal_phase(dp, map);
        }
        Instruction::TiledBlock(_) => {
            // RelabelQubits runs FIRST (before TileBlock), so a TiledBlock
            // never exists at relabel time. Unreachable in the pipeline; if a
            // caller relabels an already-tiled circuit, that is unsupported.
            unreachable!("RelabelQubits runs before TileBlock; no TiledBlock present");
        }
    }
}

/// Rebuild every condition bitmask of a [`DiagonalPhase`] through `map`.
///
/// Each `conds` entry is a `u64` bitmask over qubit indices; remapping
/// `q -> map[q]` rebuilds the mask bit-by-bit: for every set bit `q` in the
/// old mask, set bit `map[q]` in the new mask. The angle and the AND-of-
/// parities semantics are unchanged — only the bit positions move.
fn remap_diagonal_phase(dp: &mut DiagonalPhase, map: &[u32]) {
    for term in dp.terms.iter_mut() {
        for cond in term.conds.iter_mut() {
            let mut new_mask = 0u64;
            let mut m = *cond;
            while m != 0 {
                let q = m.trailing_zeros() as usize;
                new_mask |= 1u64 << map[q];
                m &= m - 1; // clear lowest set bit
            }
            *cond = new_mask;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagonal_phase::{DiagonalPhase, PhaseTerm};
    use crate::{Circuit, Instruction};
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    // (a) A 6-qubit circuit whose gates all act on qubits 4,5 relabels them
    //     to low bit positions, sets the permutation, and increases the
    //     tile-confinable count.
    #[test]
    fn high_qubit_circuit_relabels_to_low_bits() {
        let mut c = Circuit::new(6, 0);
        c.h(4).unwrap();
        c.h(5).unwrap();
        c.cnot(4, 5).unwrap();
        c.h(4).unwrap();
        c.cnot(5, 4).unwrap();

        // With tile_bits=2, NOTHING is confinable before (all targets ≥ 4).
        let before = count_confinable(&c, 2, None);
        assert_eq!(before, 0, "no gate confinable with targets at 4,5");

        let stats = RelabelQubits::new(2).run(&mut c).unwrap();
        assert_eq!(stats.transformations, 1, "should have relabeled");

        let perm = c.qubit_permutation().expect("permutation recorded");
        // Qubits 4 and 5 are the only traffic → they map to bits {0,1}.
        let p4 = perm[4];
        let p5 = perm[5];
        assert!(p4 < 2 && p5 < 2, "4,5 must map to low bits, got {p4},{p5}");
        assert_ne!(p4, p5, "permutation must be a bijection");

        // After relabel, every gate target is now < 2 → all confinable.
        let after = count_confinable(&c, 2, None);
        assert!(after > before, "confinable count must strictly increase");

        // Every instruction's targets are rewritten to low bits.
        for inst in c.instructions() {
            if let Instruction::Gate(g) = inst {
                for &q in g.qubits.iter() {
                    assert!(q < 2, "gate target {q} not rewritten to low bit");
                }
            }
        }
    }

    // (b) A circuit already on qubits 0,1 (n=6) yields no strict increase, so
    //     the guard leaves the permutation None and the instructions intact.
    #[test]
    fn already_low_circuit_is_noop() {
        let mut c = Circuit::new(6, 0);
        c.h(0).unwrap();
        c.h(1).unwrap();
        c.cnot(0, 1).unwrap();
        c.h(0).unwrap();
        let snapshot = c.clone();

        let stats = RelabelQubits::new(2).run(&mut c).unwrap();
        assert_eq!(stats.transformations, 0, "no relabel expected");
        assert!(
            c.qubit_permutation().is_none(),
            "guard must leave permutation None"
        );
        // Instructions unchanged.
        assert_eq!(c.instructions().len(), snapshot.instructions().len());
        for (a, b) in c.instructions().iter().zip(snapshot.instructions().iter()) {
            match (a, b) {
                (Instruction::Gate(ga), Instruction::Gate(gb)) => {
                    assert_eq!(ga.qubits.as_slice(), gb.qubits.as_slice());
                    assert_eq!(ga.controls.as_slice(), gb.controls.as_slice());
                }
                _ => panic!("unexpected instruction kind"),
            }
        }
    }

    // (c) remap_instruction rewrites Measure and a controlled gate's targets
    //     AND controls through a known map.
    #[test]
    fn remap_instruction_rewrites_indices() {
        // map: 4->0, 5->1, identity elsewhere (size 6).
        let map = [2u32, 3, 4, 5, 0, 1];

        let mut m = Instruction::Measure { qubit: 5, clbit: 7 };
        remap_instruction(&mut m, &map);
        match m {
            Instruction::Measure { qubit, clbit } => {
                assert_eq!(qubit, 1, "measure qubit 5 -> 1");
                assert_eq!(clbit, 7, "clbit untouched");
            }
            _ => panic!("kind changed"),
        }

        // controlled X with target 4, control 5.
        let mut g = Instruction::Gate(GateInstance::controlled(
            Gate::X,
            smallvec![4u32],
            smallvec![5u32],
        ));
        remap_instruction(&mut g, &map);
        match g {
            Instruction::Gate(gi) => {
                assert_eq!(gi.qubits.as_slice(), &[0u32], "target 4 -> 0");
                assert_eq!(gi.controls.as_slice(), &[1u32], "control 5 -> 1");
            }
            _ => panic!("kind changed"),
        }
    }

    // (d) DiagonalPhase conds bitmasks are rebuilt under the map.
    #[test]
    fn diagonal_phase_conds_remap() {
        // map: q0->0, q1->3, q2->2, q3->1 (a bijection over 4 qubits).
        let map = [0u32, 3, 2, 1];

        let mut dp = DiagonalPhase {
            n_qubits: 4,
            terms: vec![
                // cond on bits {0,1} -> {0,3}
                PhaseTerm {
                    conds: smallvec![0b0011u64],
                    angle: 0.5,
                },
                // controlled term: bits {1} and {3} -> {3} and {1}
                PhaseTerm {
                    conds: smallvec![0b0010u64, 0b1000u64],
                    angle: 0.7,
                },
            ],
        };
        remap_diagonal_phase(&mut dp, &map);
        // term0: 0b0011 (bits 0,1) -> bits {0,3} = 0b1001
        assert_eq!(dp.terms[0].conds[0], 0b1001u64);
        // term1: 0b0010 (bit 1) -> bit 3 = 0b1000
        assert_eq!(dp.terms[1].conds[0], 0b1000u64);
        // term1: 0b1000 (bit 3) -> bit 1 = 0b0010
        assert_eq!(dp.terms[1].conds[1], 0b0010u64);
        // angles untouched
        assert_eq!(dp.terms[0].angle, 0.5);
        assert_eq!(dp.terms[1].angle, 0.7);
    }

    // (e) ≤1 qubit circuit is a no-op.
    #[test]
    fn single_qubit_is_noop() {
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        let stats = RelabelQubits::default().run(&mut c).unwrap();
        assert_eq!(stats.transformations, 0);
        assert!(c.qubit_permutation().is_none());
    }

    // Guard: never relabel a circuit that already carries a permutation.
    #[test]
    fn already_permuted_is_noop() {
        let mut c = Circuit::new(6, 0);
        c.h(4).unwrap();
        c.h(5).unwrap();
        c.cnot(4, 5).unwrap();
        c.set_qubit_permutation(vec![0u32, 1, 2, 3, 4, 5].into_boxed_slice());
        let stats = RelabelQubits::new(2).run(&mut c).unwrap();
        assert_eq!(stats.transformations, 0, "must not relabel twice");
    }
}
