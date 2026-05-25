//! Property test: random `Circuit` (restricted to emitter-supported
//! variants) round-trips through `emit → parse → compare`.

use aleph_parser::{emit, parse};
use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

proptest! {
    #[test]
    fn parse_emit_roundtrip(c in arb_circuit_emittable(4, 2, 12)) {
        // `arb_circuit_emittable` is specifically curated to only
        // generate variants the emitter supports — any `emit()` Err
        // is a regression worth surfacing, not silently skipping.
        let out = emit(&c).map_err(|e| TestCaseError::fail(format!(
            "emit() failed on a circuit drawn from arb_circuit_emittable: {e:?}\n\
             circuit: {c:?}"
        )))?;
        let c2 = parse(&out).map_err(|e| TestCaseError::fail(format!(
            "re-parse failed.\nemitted:\n{out}\nerror:\n{}",
            e.render()
        )))?;
        prop_assert_eq!(c.len(), c2.len(), "instruction count mismatch");
        prop_assert_eq!(c.num_qubits(), c2.num_qubits());
        prop_assert_eq!(c.num_clbits(), c2.num_clbits());
        // Spec § 10 invariant: `generated_from` is set by the parser
        // to "openqasm:3.0" on every successfully-parsed circuit.
        prop_assert_eq!(c2.metadata().generated_from.as_deref(), Some("openqasm:3.0"));
        for (i, (a, b)) in c.instructions().iter().zip(c2.instructions().iter()).enumerate() {
            prop_assert_eq!(format!("{a:?}"), format!("{b:?}"), "instr {} differs", i);
        }
    }
}
