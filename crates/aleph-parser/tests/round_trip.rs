//! Fixture-driven round-trip: parse → emit → parse, assert IR equality.

use aleph_parser::{emit, parse};

fn fixture(name: &str) -> String {
    let path = format!("tests/fixtures/{name}.qasm");
    std::fs::read_to_string(&path).expect("read fixture")
}

fn assert_round_trip(name: &str) {
    let src = fixture(name);
    let c1 = parse(&src).unwrap_or_else(|e| panic!("{name}: parse: {e}"));
    let out = emit(&c1).unwrap_or_else(|e| panic!("{name}: emit: {e}"));
    let c2 = parse(&out).unwrap_or_else(|e| {
        panic!(
            "{name}: re-parse failed.\nemitted source:\n{out}\nerror:\n{}",
            e.render()
        )
    });
    assert_eq!(c1.len(), c2.len(), "{name}: instruction count differs");
    assert_eq!(
        c1.num_qubits(),
        c2.num_qubits(),
        "{name}: num_qubits differs"
    );
    assert_eq!(
        c1.num_clbits(),
        c2.num_clbits(),
        "{name}: num_clbits differs"
    );
    for (i, (a, b)) in c1
        .instructions()
        .iter()
        .zip(c2.instructions().iter())
        .enumerate()
    {
        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "{name}: instruction[{i}] differs"
        );
    }
}

#[test]
fn ghz_round_trip() {
    assert_round_trip("ghz");
}

#[test]
fn qft_round_trip() {
    assert_round_trip("qft");
}

#[test]
fn grover_round_trip() {
    assert_round_trip("grover");
}

#[test]
fn random_round_trip() {
    assert_round_trip("random");
}
