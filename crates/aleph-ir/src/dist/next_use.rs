//! Next-use index for the lookahead router: for every logical qubit, the
//! ascending instruction indices at which it must be local
//! (`required_local`). Belady-style eviction asks "when is `q` needed next?".

use super::plan::required_local;
use crate::{Circuit, Instruction};
use aleph_core::Gate;

pub(crate) struct NextUse {
    pos: Vec<Vec<usize>>,
    cursor: Vec<usize>,
}

impl NextUse {
    pub(crate) fn build(circuit: &Circuit) -> Self {
        let n = circuit.num_qubits() as usize;
        let mut pos = vec![Vec::new(); n];
        for (i, instr) in circuit.instructions().iter().enumerate() {
            if let Instruction::Gate(g) = instr {
                if matches!(g.gate, Gate::Swap) && g.controls.is_empty() {
                    continue; // relabel: needs nothing local
                }
                for q in required_local(g) {
                    pos[q as usize].push(i);
                }
            }
        }
        Self {
            pos,
            cursor: vec![0; n],
        }
    }

    /// First index `>= from` at which logical `q` must be local; `usize::MAX`
    /// if never. `from` must be non-decreasing per `q` across calls.
    pub(crate) fn next(&mut self, q: u32, from: usize) -> usize {
        let (p, c) = (&self.pos[q as usize], &mut self.cursor[q as usize]);
        while *c < p.len() && p[*c] < from {
            *c += 1;
        }
        p.get(*c).copied().unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Circuit;

    #[test]
    fn next_use_tracks_required_positions() {
        let mut c = Circuit::new(3, 0);
        c.h(0).unwrap(); // 0: q0 required
        c.cnot(0, 1).unwrap(); // 1: q1 required (target)
        c.rz(0.2, 0).unwrap(); // 2: diagonal, nothing required
        c.swap(0, 2).unwrap(); // 3: relabel, nothing required
        c.h(0).unwrap(); // 4: q0 required
        let mut nu = NextUse::build(&c);
        assert_eq!(nu.next(0, 0), 0);
        assert_eq!(nu.next(0, 1), 4);
        assert_eq!(nu.next(1, 0), 1);
        assert_eq!(nu.next(1, 2), usize::MAX);
        assert_eq!(nu.next(2, 0), usize::MAX);
        assert_eq!(nu.next(0, 5), usize::MAX);
    }
}
