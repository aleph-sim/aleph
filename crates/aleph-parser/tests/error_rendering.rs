//! Snapshot-style tests that the `ParseError::render()` output looks
//! sensible for the error paths users will actually hit.

use aleph_parser::{parse, ParseError, ParseErrorKind};

fn err(src: &str) -> ParseError {
    parse(src).unwrap_err()
}

#[test]
fn unknown_gate_renders_three_lines_with_caret() {
    let e = err("qubit[1] q;\nhadamard q[0];\n");
    let out = e.render();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("unknown gate"));
    assert_eq!(lines[1], "hadamard q[0];");
    assert!(lines[2].starts_with('^') || lines[2].starts_with(' '));
}

#[test]
fn unknown_register_renders() {
    let e = err("qubit[1] q;\nh r[0];\n");
    let out = e.render();
    let lines: Vec<&str> = out.lines().collect();
    assert!(lines[0].contains("undeclared register"));
    assert_eq!(lines[1], "h r[0];");
}

#[test]
fn index_out_of_bounds_renders() {
    let e = err("qubit[2] q; h q[9];");
    let out = e.render();
    assert!(out.contains("out of bounds"));
    assert!(out.contains("size 2"));
}

#[test]
fn too_many_qubits_renders() {
    let big = aleph_ir::MAX_QUBITS + 1;
    let src = format!("qubit[{big}] q;");
    let e = err(&src);
    assert!(matches!(e.kind, ParseErrorKind::TooManyQubits { .. }));
    let out = e.render();
    assert!(out.contains("too many qubits"));
}

#[test]
fn size_mismatch_renders() {
    let e = err("qubit[2] q; bit[3] c; measure q -> c;");
    let out = e.render();
    assert!(out.contains("size mismatch"));
}

#[test]
fn display_one_liner_includes_line_col() {
    let e = err("qubit[1] q;\nhadamard q[0];\n");
    let one = format!("{e}");
    assert!(one.starts_with("error at 2:"));
    assert!(one.contains("unknown gate"));
}
