//! `TileBlock` — groups maximal runs of consecutive gates whose targets are
//! all `< tile_bits` (and arity ≤ 2 with a representable matrix) into one
//! `Instruction::TiledBlock`, so the backend applies them tile-major (one
//! DRAM pass per run instead of one per gate). Runs LAST in the pipeline,
//! after all fusion. See `docs/superpowers/specs/2026-06-04-p2-09-cache-blocking-design.md`.

use crate::passes::{Pass, PassError, PassStats};
use crate::{Circuit, Instruction, TiledBlock};
use aleph_core::GateInstance;

/// Default tile width (log2 amplitudes). EPYC 8124P L2 = 1 MiB/core
/// = 2^16 Complex<f64>; a 2^15-amp tile (512 KiB) leaves working-set
/// headroom. Conservative default; the backend may rebuild the pipeline
/// with a tuned value.
pub const DEFAULT_TILE_BITS: u8 = 15;

/// Pass that groups maximal runs of tile-confinable gates into
/// [`Instruction::TiledBlock`] so the backend can apply them tile-major
/// (one DRAM pass per run).
///
/// A gate is tile-confinable iff:
/// - all its *target* qubits are `< tile_bits` (controls may be higher —
///   the executor masks them per tile), AND
/// - `gate.matrix()` succeeds (arity ≤ 2, no symbolic/non-finite params).
///
/// Runs of length 1 stay as a plain `Instruction::Gate` (no benefit).
/// Any other instruction (Measure, Reset, Barrier, DiagonalPhase, …)
/// is a hard run-breaker and is emitted verbatim.
pub struct TileBlock {
    pub tile_bits: u8,
}

impl Default for TileBlock {
    fn default() -> Self {
        Self {
            tile_bits: DEFAULT_TILE_BITS,
        }
    }
}

impl TileBlock {
    /// Construct with an explicit tile width (used by tests / a tuned backend).
    pub fn new(tile_bits: u8) -> Self {
        Self { tile_bits }
    }

    /// A gate is tile-confinable iff all its TARGETS are `< tile_bits`
    /// (controls may be higher — masked per tile by the executor) AND it
    /// has a fixed-size matrix of arity ≤ 2 (the tile kernels handle 1q/2q).
    fn confinable(&self, g: &GateInstance) -> bool {
        let tb = self.tile_bits as u32;
        g.qubits.len() <= 2 && g.qubits.iter().all(|&q| q < tb) && g.gate.matrix().is_ok()
    }
}

/// Flush the current run of confinable gates into the output vector.
/// A length-1 run is emitted as a plain `Instruction::Gate` (no benefit from
/// wrapping). A length-≥2 run becomes an `Instruction::TiledBlock`.
fn flush(run: &mut Vec<GateInstance>, out: &mut Vec<Instruction>, blocks: &mut u64, tile_bits: u8) {
    match run.len() {
        0 => {}
        1 => {
            // SAFETY (clippy): `run.len() == 1` so `pop()` is infallible.
            // Using `if let` avoids an unwrap() call in library code.
            if let Some(g) = run.pop() {
                out.push(Instruction::Gate(g));
            }
        }
        _ => {
            *blocks += 1;
            out.push(Instruction::TiledBlock(Box::new(TiledBlock {
                gates: std::mem::take(run),
                tile_bits,
            })));
        }
    }
}

impl Pass for TileBlock {
    fn name(&self) -> &'static str {
        "TileBlock"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let input = std::mem::take(&mut circuit.instructions);
        let gates_before = input.len();
        let mut out: Vec<Instruction> = Vec::with_capacity(input.len());
        let mut run: Vec<GateInstance> = Vec::new();
        let mut blocks = 0u64;
        let tile_bits = self.tile_bits;

        for inst in input {
            match inst {
                Instruction::Gate(g) if self.confinable(&g) => run.push(g),
                other => {
                    flush(&mut run, &mut out, &mut blocks, tile_bits);
                    out.push(other);
                }
            }
        }
        flush(&mut run, &mut out, &mut blocks, tile_bits);

        circuit.instructions = out;
        Ok(PassStats {
            gates_before,
            gates_after: circuit.instructions.len(),
            transformations: blocks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Circuit, Instruction};
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    // Helper: build a 1q gate instance on the given qubit.
    fn h_gate(q: u32) -> GateInstance {
        GateInstance::new(Gate::H, smallvec![q])
    }

    // Helper: build a CNOT (2q) instance.
    fn cnot_gate(ctrl: u32, tgt: u32) -> GateInstance {
        GateInstance::new(Gate::Cnot, smallvec![ctrl, tgt])
    }

    // (a) A run of 3 low-target 1q gates → exactly one TiledBlock with 3 gates.
    #[test]
    fn run_of_three_low_target_gates_fuses_into_block() {
        let mut c = Circuit::new(4, 0);
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(1))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        TileBlock::new(4).run(&mut c).unwrap();
        let insts = c.instructions();
        assert_eq!(insts.len(), 1, "expected one TiledBlock");
        match &insts[0] {
            Instruction::TiledBlock(tb) => {
                assert_eq!(tb.gates.len(), 3, "block should hold all 3 gates");
                assert_eq!(tb.tile_bits, 4);
            }
            other => panic!("expected TiledBlock, got {other:?}"),
        }
    }

    // (b) A high-target gate splits two runs: [block, Gate(high), block].
    #[test]
    fn high_target_gate_splits_runs() {
        let mut c = Circuit::new(8, 0);
        // low run: q0, q1
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(1))).unwrap();
        // high (target q5 >= tile_bits=4) — not confinable
        c.add_instruction(Instruction::Gate(h_gate(5))).unwrap();
        // low run: q0, q1
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(1))).unwrap();

        TileBlock::new(4).run(&mut c).unwrap();
        let insts = c.instructions();
        assert_eq!(insts.len(), 3, "expected [block, Gate(high), block]");
        assert!(
            matches!(&insts[0], Instruction::TiledBlock(tb) if tb.gates.len() == 2),
            "first block should have 2 gates"
        );
        assert!(
            matches!(&insts[1], Instruction::Gate(g) if g.qubits[0] == 5),
            "middle should be the high-target gate"
        );
        assert!(
            matches!(&insts[2], Instruction::TiledBlock(tb) if tb.gates.len() == 2),
            "last block should have 2 gates"
        );
    }

    // (c) A Barrier between gates flushes the run and is preserved verbatim.
    #[test]
    fn barrier_flushes_run_and_is_preserved() {
        let mut c = Circuit::new(4, 0);
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(1))).unwrap();
        c.barrier([0u32, 1u32]).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(1))).unwrap();

        TileBlock::new(4).run(&mut c).unwrap();
        let insts = c.instructions();
        assert_eq!(insts.len(), 3, "expected [block, Barrier, block]");
        assert!(matches!(&insts[0], Instruction::TiledBlock(_)));
        assert!(matches!(&insts[1], Instruction::Barrier(_)));
        assert!(matches!(&insts[2], Instruction::TiledBlock(_)));
    }

    // (c continued) A Measure between gates flushes the run and is preserved.
    #[test]
    fn measure_flushes_run_and_is_preserved() {
        let mut c = Circuit::new(4, 1);
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(1))).unwrap();
        c.measure(2, 0).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(1))).unwrap();

        TileBlock::new(4).run(&mut c).unwrap();
        let insts = c.instructions();
        assert_eq!(insts.len(), 3, "expected [block, Measure, block]");
        assert!(matches!(&insts[0], Instruction::TiledBlock(_)));
        assert!(matches!(&insts[1], Instruction::Measure { .. }));
        assert!(matches!(&insts[2], Instruction::TiledBlock(_)));
    }

    // (d) A single confinable gate stays as a plain Instruction::Gate.
    #[test]
    fn single_confinable_gate_stays_plain() {
        let mut c = Circuit::new(4, 0);
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();

        TileBlock::new(4).run(&mut c).unwrap();
        let insts = c.instructions();
        assert_eq!(insts.len(), 1);
        assert!(
            matches!(&insts[0], Instruction::Gate(_)),
            "single gate must not be wrapped in a TiledBlock"
        );
    }

    // (e) A gate with LOW target but HIGH control is still confinable.
    // CNOT with target=1 (low) but the control placed externally at q5 (high).
    // `confinable` checks only `qubits` (targets), not `controls`.
    #[test]
    fn low_target_high_control_is_confinable() {
        // Use GateInstance::controlled so q5 is in `controls`, not `qubits`.
        let ctrl_x = GateInstance::controlled(Gate::X, smallvec![1u32], smallvec![5u32]);
        let mut c = Circuit::new(8, 0);
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(ctrl_x)).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(2))).unwrap();

        TileBlock::new(4).run(&mut c).unwrap();
        let insts = c.instructions();
        // All 3 gates are confinable (targets 0, 1, 2 all < 4), so one block.
        assert_eq!(insts.len(), 1, "expected one TiledBlock");
        match &insts[0] {
            Instruction::TiledBlock(tb) => {
                assert_eq!(tb.gates.len(), 3);
                // The high-control gate is present in the block.
                assert!(tb.gates.iter().any(|g| g.controls.contains(&5u32)));
            }
            other => panic!("expected TiledBlock, got {other:?}"),
        }
    }

    // stats.transformations counts the number of TiledBlock emissions.
    #[test]
    fn transformations_counts_blocks_emitted() {
        let mut c = Circuit::new(4, 1);
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(1))).unwrap();
        c.measure(0, 0).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();
        c.add_instruction(Instruction::Gate(h_gate(1))).unwrap();

        let stats = TileBlock::new(4).run(&mut c).unwrap();
        assert_eq!(
            stats.transformations, 2,
            "two blocks should have been emitted"
        );
    }

    // A 2q CNOT with both qubits below tile_bits is confinable.
    #[test]
    fn two_qubit_gate_low_targets_confinable() {
        let mut c = Circuit::new(8, 0);
        // q0 and q1 are both < 4
        c.add_instruction(Instruction::Gate(cnot_gate(0, 1)))
            .unwrap();
        c.add_instruction(Instruction::Gate(h_gate(0))).unwrap();

        TileBlock::new(4).run(&mut c).unwrap();
        let insts = c.instructions();
        assert_eq!(insts.len(), 1, "both gates should fuse into one TiledBlock");
        assert!(matches!(&insts[0], Instruction::TiledBlock(tb) if tb.gates.len() == 2));
    }
}
