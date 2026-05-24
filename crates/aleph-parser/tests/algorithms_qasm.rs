//! Parse each Tier-1 fixture and assert key IR-level facts (instruction
//! count, sample gate variants). Round-trip checks live in
//! `round_trip.rs`.

use aleph_core::Gate;
use aleph_ir::Instruction;
use aleph_parser::parse;

fn fixture(name: &str) -> String {
    let path = format!("tests/fixtures/{name}.qasm");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

#[test]
fn ghz_parses() {
    let c = parse(&fixture("ghz")).unwrap();
    assert_eq!(c.num_qubits(), 3);
    assert_eq!(c.num_clbits(), 3);
    // 1 H + 2 CNOT + 3 Measure (whole-register measure fans out) = 6.
    assert_eq!(c.len(), 6);
    assert!(matches!(
        &c.instructions()[0],
        Instruction::Gate(g) if g.gate == Gate::H
    ));
    assert!(matches!(
        &c.instructions()[1],
        Instruction::Gate(g) if g.gate == Gate::Cnot
    ));
}

#[test]
fn qft_parses() {
    let c = parse(&fixture("qft")).unwrap();
    assert_eq!(c.num_qubits(), 3);
    // 3 H + 3 Cz + 3 Rz = 9 instructions.
    assert_eq!(c.len(), 9);
}

#[test]
fn grover_parses() {
    let c = parse(&fixture("grover")).unwrap();
    assert_eq!(c.num_qubits(), 2);
    assert_eq!(c.num_clbits(), 2);
    // 2H (init) + Cz (oracle) + 9 (diffusion: 4H + 4X + Cz) + 2 measure = 14.
    assert_eq!(c.len(), 14);
}

#[test]
fn random_parses() {
    let c = parse(&fixture("random")).unwrap();
    assert_eq!(c.num_qubits(), 4);
    assert_eq!(c.num_clbits(), 4);
    let kinds: Vec<&Gate> = c
        .instructions()
        .iter()
        .filter_map(|i| match i {
            Instruction::Gate(g) => Some(&g.gate),
            _ => None,
        })
        .collect();
    assert!(kinds.contains(&&Gate::H));
    assert!(kinds.contains(&&Gate::Toffoli));
    assert!(kinds.contains(&&Gate::Swap));
}

#[test]
fn random_sets_generated_from_metadata() {
    let c = parse(&fixture("random")).unwrap();
    assert_eq!(c.metadata().generated_from.as_deref(), Some("openqasm:3.0"));
}
