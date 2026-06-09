//! P4-07 acceptance test: surface-code logical/physical error detection.
//!
//! Deterministic, no Stim. From logical |0⟩_L (data |0…0⟩) the Z-stabilizers
//! are deterministic: a single physical X error fires exactly its adjacent
//! Z-ancillas, while a *logical* X̄ (a full data column) fires none — the
//! defining "undetectable" property of a logical operator. The X-basis mirror
//! (|+…+⟩, X-stabilizers, physical/logical Z) is symmetric.

use aleph_backend::Backend;
use aleph_benches::{Ancilla, SurfaceCode};
use aleph_core::{Gate, GateInstance};
use aleph_stab::StabilizerBackend;

/// Measure every ancilla after applying `pre` (errors/logicals) then one cycle,
/// from the all-|0⟩ start. Returns outcomes keyed by ancilla index.
fn syndrome(
    sc: &SurfaceCode,
    seed: u64,
    pre: &[GateInstance],
) -> std::collections::HashMap<u32, bool> {
    let mut be = StabilizerBackend::with_seed(seed);
    let mut t = be.allocate(sc.num_qubits as u32).unwrap();
    for g in pre {
        be.apply_gate(&mut t, g).unwrap();
    }
    for g in sc.cycle_gates() {
        be.apply_gate(&mut t, &g).unwrap();
    }
    sc.ancilla_order()
        .iter()
        .map(|&a| (a, be.measure(&mut t, a).unwrap()))
        .collect()
}

fn z_ancillas(sc: &SurfaceCode) -> Vec<&Ancilla> {
    sc.ancillas.iter().filter(|a| !a.is_x).collect()
}
fn x_ancillas(sc: &SurfaceCode) -> Vec<&Ancilla> {
    sc.ancillas.iter().filter(|a| a.is_x).collect()
}

#[test]
fn physical_x_error_fires_adjacent_z_ancillas() {
    for d in [3usize, 5] {
        let sc = SurfaceCode::new(d);
        // Pick an interior data qubit so it has >=1 Z-neighbour.
        let q = (d / 2 * d + d / 2) as u32;
        let pre = vec![GateInstance::new(Gate::X, vec![q])];
        let s = syndrome(&sc, 0, &pre);
        let expected_fired: std::collections::HashSet<u32> = z_ancillas(&sc)
            .iter()
            .filter(|a| a.data_neighbours.contains(&q))
            .map(|a| a.index)
            .collect();
        assert!(
            !expected_fired.is_empty(),
            "d={d}: chosen qubit has no Z-neighbour"
        );
        for a in z_ancillas(&sc) {
            let want = expected_fired.contains(&a.index);
            assert_eq!(
                s[&a.index], want,
                "d={d}: Z-ancilla {} fired={:?}, want {want}",
                a.index, s[&a.index]
            );
        }
    }
}

#[test]
fn logical_x_is_undetectable() {
    for d in [3usize, 5] {
        let sc = SurfaceCode::new(d);
        let pre: Vec<GateInstance> = sc
            .logical_x
            .iter()
            .map(|&q| GateInstance::new(Gate::X, vec![q]))
            .collect();
        let s = syndrome(&sc, 0, &pre);
        for a in z_ancillas(&sc) {
            assert!(
                !s[&a.index],
                "d={d}: logical X-bar fired Z-ancilla {}",
                a.index
            );
        }
    }
}

#[test]
fn physical_z_error_fires_adjacent_x_ancillas() {
    for d in [3usize, 5] {
        let sc = SurfaceCode::new(d);
        let q = (d / 2 * d + d / 2) as u32;
        // Prepare |+…+⟩ via H on all data, then inject Z, then cycle.
        let mut pre: Vec<GateInstance> = sc
            .data
            .iter()
            .map(|&dq| GateInstance::new(Gate::H, vec![dq]))
            .collect();
        pre.push(GateInstance::new(Gate::Z, vec![q]));
        let s = syndrome(&sc, 0, &pre);
        let expected: std::collections::HashSet<u32> = x_ancillas(&sc)
            .iter()
            .filter(|a| a.data_neighbours.contains(&q))
            .map(|a| a.index)
            .collect();
        assert!(
            !expected.is_empty(),
            "d={d}: chosen qubit has no X-neighbour"
        );
        for a in x_ancillas(&sc) {
            let want = expected.contains(&a.index);
            assert_eq!(
                s[&a.index], want,
                "d={d}: X-ancilla {} fired={:?}, want {want}",
                a.index, s[&a.index]
            );
        }
    }
}

#[test]
fn logical_z_is_undetectable() {
    for d in [3usize, 5] {
        let sc = SurfaceCode::new(d);
        let mut pre: Vec<GateInstance> = sc
            .data
            .iter()
            .map(|&dq| GateInstance::new(Gate::H, vec![dq]))
            .collect();
        for &q in &sc.logical_z {
            pre.push(GateInstance::new(Gate::Z, vec![q]));
        }
        let s = syndrome(&sc, 0, &pre);
        for a in x_ancillas(&sc) {
            assert!(
                !s[&a.index],
                "d={d}: logical Z-bar fired X-ancilla {}",
                a.index
            );
        }
    }
}
