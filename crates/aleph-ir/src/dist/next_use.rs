//! Next-use index for the lookahead router: for every qubit's *data*, the
//! ascending instruction indices at which it must be local
//! (`required_local`). Belady-style eviction asks "when is `q` needed next?".
//!
//! A relabel `Swap(a, b)` hands label `a`'s data to label `b`, so uses are
//! recorded per *track* (data identity), and labels are mapped to tracks —
//! at build time by replaying the swaps, at plan time via [`NextUse::relabel`].

use super::plan::required_local;
use crate::{Circuit, Instruction};
use aleph_core::Gate;

pub(crate) struct NextUse {
    /// Per track: ascending indices where that data must be local.
    pos: Vec<Vec<usize>>,
    cursor: Vec<usize>,
    /// `track[label]`: which data the circuit label currently holds.
    track: Vec<u32>,
}

impl NextUse {
    pub(crate) fn build(circuit: &Circuit) -> Self {
        let n = circuit.num_qubits() as usize;
        let mut pos = vec![Vec::new(); n];
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for (i, instr) in circuit.instructions().iter().enumerate() {
            if let Instruction::Gate(g) = instr {
                if matches!(g.gate, Gate::Swap) && g.controls.is_empty() {
                    // relabel: needs nothing local, but the labels trade data
                    perm.swap(g.qubits[0] as usize, g.qubits[1] as usize);
                    continue;
                }
                for q in required_local(g) {
                    pos[perm[q as usize] as usize].push(i);
                }
            }
        }
        Self {
            pos,
            cursor: vec![0; n],
            track: (0..n as u32).collect(),
        }
    }

    /// Mirror a relabel `Swap(a, b)` the planner just absorbed.
    pub(crate) fn relabel(&mut self, a: u32, b: u32) {
        self.track.swap(a as usize, b as usize);
    }

    /// First index `>= from` at which the data now under label `q` must be
    /// local; `usize::MAX` if never. `from` must be non-decreasing per track.
    pub(crate) fn next(&mut self, q: u32, from: usize) -> usize {
        let t = self.track[q as usize] as usize;
        let (p, c) = (&self.pos[t], &mut self.cursor[t]);
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
        c.h(2).unwrap(); // 3: q2 required
        c.h(0).unwrap(); // 4: q0 required
        let mut nu = NextUse::build(&c);
        assert_eq!(nu.next(0, 0), 0);
        assert_eq!(nu.next(0, 1), 4);
        assert_eq!(nu.next(1, 0), 1);
        assert_eq!(nu.next(1, 2), usize::MAX);
        assert_eq!(nu.next(2, 0), 3);
        assert_eq!(nu.next(0, 5), usize::MAX);
    }

    #[test]
    fn next_use_follows_data_across_relabel() {
        // Swap(0, 2) at index 2 hands label 0's data to label 2 and vice versa.
        let mut c = Circuit::new(3, 0);
        c.h(0).unwrap(); // 0: data D0 (label 0)
        c.h(2).unwrap(); // 1: data D2 (label 2)
        c.swap(0, 2).unwrap(); // 2: relabel
        c.h(2).unwrap(); // 3: label 2 = D0
        c.h(0).unwrap(); // 4: label 0 = D2
        c.h(0).unwrap(); // 5: label 0 = D2
        let mut nu = NextUse::build(&c);
        // Before the relabel, label 0 is D0, next needed at 3 (as label 2).
        assert_eq!(nu.next(0, 1), 3);
        // Label 2 is D2, next needed at 4 (as label 0).
        assert_eq!(nu.next(2, 2), 4);
        nu.relabel(0, 2);
        assert_eq!(nu.next(2, 3), 3); // D0
        assert_eq!(nu.next(0, 3), 4); // D2
        assert_eq!(nu.next(0, 5), 5);
        assert_eq!(nu.next(2, 4), usize::MAX);
    }
}
