//! End-to-end Tier-1 algorithm circuits — Bell pair, GHZ-3.

use aleph_core::Gate;
use aleph_ir::{Circuit, CircuitError, Instruction};

fn bell_pair() -> Result<Circuit, CircuitError> {
    let mut c = Circuit::new(2, 2).with_name("bell");
    c.h(0)?.cnot(0, 1)?.measure(0, 0)?.measure(1, 1)?;
    Ok(c)
}

fn ghz_3() -> Result<Circuit, CircuitError> {
    let mut c = Circuit::new(3, 0).with_name("ghz-3");
    c.h(0)?.cnot(0, 1)?.cnot(1, 2)?;
    Ok(c)
}

#[test]
fn bell_pair_has_4_instructions_and_3_layers() {
    let c = bell_pair().unwrap();
    assert_eq!(c.len(), 4);
    assert_eq!(c.layers(), vec![vec![0], vec![1], vec![2, 3]]);
}

#[test]
fn bell_pair_metadata_carries_name() {
    let c = bell_pair().unwrap();
    assert_eq!(c.metadata().name.as_deref(), Some("bell"));
}

#[test]
fn ghz_3_has_3_instructions_and_3_layers() {
    let c = ghz_3().unwrap();
    assert_eq!(c.len(), 3);
    assert_eq!(c.layers(), vec![vec![0], vec![1], vec![2]]);
}

#[test]
fn ghz_3_instruction_sequence_matches_recipe() {
    let c = ghz_3().unwrap();
    let kinds: Vec<&Gate> = c
        .instructions()
        .iter()
        .map(|i| match i {
            Instruction::Gate(g) => &g.gate,
            other => panic!("expected gate, got {other:?}"),
        })
        .collect();
    assert_eq!(kinds, vec![&Gate::H, &Gate::Cnot, &Gate::Cnot]);
}
