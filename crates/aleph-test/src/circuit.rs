//! `OpKind` union enum + circuit strategies.  See spec §4.3 and
//! the plan's §"Spec amendment" — the parser and IR tests
//! intentionally curate divergent vocabularies; this module
//! exports the union plus two `arb_op_*` / `arb_circuit_*`
//! strategies so neither test loses coverage.
