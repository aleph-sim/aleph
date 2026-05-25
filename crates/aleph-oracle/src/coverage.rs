//! Coverage check: every gate name accepted by the OpenQASM parser
//! must appear at least once as a leading token in some
//! `oracle/circuits/*.qasm` file. Fails the build if a contributor
//! adds a new gate to the parser without also adding a corpus
//! circuit that exercises it.
//!
//! The expected gate set is hand-maintained here. It mirrors
//! `crates/aleph-parser/src/lower.rs` and must be updated whenever
//! the parser learns a new keyword.

#[cfg(test)]
mod tests {
    use crate::fixture::workspace_path;

    /// Gate keywords currently lowered by `aleph-parser`. Keep in
    /// sync with `crates/aleph-parser/src/lower.rs`.
    const SUPPORTED_GATES: &[&str] = &[
        "h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cz", "swap", "ccx",
    ];

    #[test]
    fn every_supported_gate_appears_in_corpus() {
        let circuits_dir = workspace_path("oracle/circuits");
        let mut all_qasm = String::new();
        for entry in std::fs::read_dir(&circuits_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|x| x == "qasm") {
                all_qasm.push_str(&std::fs::read_to_string(&path).unwrap());
                all_qasm.push('\n');
            }
        }
        let mut missing: Vec<&str> = Vec::new();
        for &gate in SUPPORTED_GATES {
            // Match the keyword followed by whitespace or `(`, on a
            // word boundary, so `s` doesn't match `swap` and `z`
            // doesn't match `cz`.
            let token_space = format!("\n{gate} ");
            let token_paren = format!("\n{gate}(");
            let token_start = format!("\n{gate}\n");
            if !all_qasm.contains(&token_space)
                && !all_qasm.contains(&token_paren)
                && !all_qasm.contains(&token_start)
            {
                missing.push(gate);
            }
        }
        assert!(
            missing.is_empty(),
            "gates not exercised by any fixture: {missing:?}"
        );
    }
}
