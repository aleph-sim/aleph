//! P6-05 initial placement: start the `g` qubits that are needed local *last*
//! as the global ones (ties: fewest local needs, then highest index). Free,
//! because plans start from |0…0⟩.

use aleph_core::Gate;

use super::plan::required_local;
use super::DistLayout;
use crate::{Circuit, Instruction};

/// `l2p[logical] = physical`: chosen globals at `m..n` (ascending logical),
/// the rest at `0..m` in logical order.
pub fn initial_placement(circuit: &Circuit, layout: DistLayout) -> Vec<u32> {
    let n = layout.n as usize;
    let mut first = vec![usize::MAX; n];
    let mut uses = vec![0u32; n];
    // track[label] = which t=0 data the label holds (relabel swaps move data).
    let mut track: Vec<usize> = (0..n).collect();
    for (i, instr) in circuit.instructions().iter().enumerate() {
        let Instruction::Gate(g) = instr else {
            continue;
        };
        if matches!(g.gate, Gate::Swap) && g.controls.is_empty() {
            track.swap(g.qubits[0] as usize, g.qubits[1] as usize);
            continue;
        }
        for q in required_local(g) {
            let t = track[q as usize];
            first[t] = first[t].min(i);
            uses[t] += 1;
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        first[b]
            .cmp(&first[a])
            .then(uses[a].cmp(&uses[b]))
            .then(b.cmp(&a))
    });
    let mut global: Vec<usize> = order[..layout.g as usize].to_vec();
    global.sort_unstable();
    let mut l2p = vec![0u32; n];
    let mut next_local = 0u32;
    for (q, slot) in l2p.iter_mut().enumerate() {
        if !global.contains(&q) {
            *slot = next_local;
            next_local += 1;
        }
    }
    for (j, &q) in global.iter().enumerate() {
        l2p[q] = layout.m() + j as u32;
    }
    l2p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{plan, plan_from, Router};

    #[test]
    fn unused_qubits_go_global() {
        let mut c = Circuit::new(6, 0);
        for q in 0..4 {
            c.h(q).unwrap();
        }
        let l = DistLayout::new(6, 2).unwrap();
        assert_eq!(initial_placement(&c, l), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn latest_first_need_goes_global() {
        let mut c = Circuit::new(6, 0);
        for q in [5u32, 4, 3, 0, 1, 2] {
            c.h(q).unwrap();
        }
        let l = DistLayout::new(6, 2).unwrap();
        // First needs: 5@0 4@1 3@2 0@3 1@4 2@5 → globals {1, 2}.
        assert_eq!(initial_placement(&c, l), vec![0, 4, 5, 1, 2, 3]);
        let id = plan(&c, l, Router::Naive).unwrap().stats.exchanges;
        let placed = plan_from(&c, l, Router::Naive, &initial_placement(&c, l))
            .unwrap()
            .stats
            .exchanges;
        assert!(placed <= id);
    }

    #[test]
    fn diagonal_and_control_uses_do_not_count() {
        let mut c = Circuit::new(4, 0);
        c.rz(0.1, 3).unwrap(); // diagonal: never needs local
        c.cnot(3, 0).unwrap(); // 3 is only a control
        c.h(1).unwrap();
        c.h(2).unwrap();
        let l = DistLayout::new(4, 1).unwrap();
        assert_eq!(initial_placement(&c, l), vec![0, 1, 2, 3]);
    }

    #[test]
    fn g0_is_identity() {
        let mut c = Circuit::new(3, 0);
        c.h(2).unwrap();
        assert_eq!(
            initial_placement(&c, DistLayout::new(3, 0).unwrap()),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn relabel_swaps_are_followed() {
        // swap(0, 3) then H on label 3: the data needed is track 0's.
        let mut c = Circuit::new(4, 0);
        c.swap(0, 3).unwrap();
        c.h(3).unwrap();
        c.h(1).unwrap();
        c.h(2).unwrap();
        let l = DistLayout::new(4, 1).unwrap();
        // Track 3 (label 3 at t=0) is never needed: it goes global.
        assert_eq!(initial_placement(&c, l), vec![0, 1, 2, 3]);
    }
}
