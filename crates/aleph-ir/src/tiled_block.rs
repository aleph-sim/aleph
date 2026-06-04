//! `TiledBlock` — a maximal run of consecutive gates whose targets are all
//! below a cache-tile bit width, grouped so a backend can apply them
//! **tile-major** (one DRAM pass over the state instead of one per gate).
//! Produced only by `passes::TileBlock`; never by the parser. See
//! `docs/superpowers/specs/2026-06-04-p2-09-cache-blocking-design.md`.

use aleph_core::GateInstance;

/// A run of gates confined to the low `tile_bits` qubits (targets only;
/// controls may be higher and are masked per-tile by the executor).
#[derive(Debug, Clone)]
pub struct TiledBlock {
    /// Gates in original application order. Each gate's `qubits` (targets)
    /// are all `< tile_bits`; `controls` may be any qubit.
    pub gates: Vec<GateInstance>,
    /// log2 of the tile size in amplitudes. A tile is `2^tile_bits`
    /// contiguous amplitudes; targets `< tile_bits` pair within a tile.
    pub tile_bits: u8,
}

impl TiledBlock {
    /// All qubits touched by any gate in the block (targets ∪ controls).
    pub fn used_qubits(&self) -> smallvec::SmallVec<[u32; 6]> {
        let mut out: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
        for g in &self.gates {
            for &q in g.qubits.iter().chain(g.controls.iter()) {
                if !out.contains(&q) {
                    out.push(q);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Gate;
    use smallvec::smallvec;

    #[test]
    fn used_qubits_unions_targets_and_controls() {
        let tb = TiledBlock {
            gates: vec![
                GateInstance::new(Gate::H, smallvec![0u32]),
                GateInstance::controlled(Gate::X, smallvec![1u32], smallvec![5u32]),
            ],
            tile_bits: 4,
        };
        let mut q = tb.used_qubits().to_vec();
        q.sort();
        assert_eq!(q, vec![0, 1, 5]);
    }
}
