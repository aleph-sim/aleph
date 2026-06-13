//! Oracle: `run_noisy` @100k shots must match Qiskit Aer's *exact* noisy
//! distribution (density-matrix + analytic readout) within the calibrated 5σ
//! band. The aleph NoiseModel here mirrors the byte-identical model built in
//! oracle/noise/gen_noise.py — keep the two in sync.
//!
//! ## Gate-name attachment keys
//!
//! gen_noise.py attaches errors to Qiskit's gate names ("h", "cx", "id", "x").
//! aleph's `NoiseModel::errors_for` keys on `Gate::name()`, which returns
//! aleph's *own* spelling, NOT the QASM mnemonic: `h` → `Gate::H` (name "H"),
//! `x` → `Gate::X` (name "X"), `cx` → `Gate::Cnot` (name "Cnot"). So the
//! attachment keys below are "H"/"X"/"Cnot". The fixtures don't care about
//! names — they encode the resulting distribution; only the aleph-side
//! attachment key must match aleph's own `Gate::name()` for the channel to
//! fire. (Verified: `aleph_parser::lower` maps the QASM mnemonics to those
//! variants; `crates/aleph-core/src/gate/kinds.rs::name`.)
//!
//! ## The `id` carrier (amp/phase damping)
//!
//! gen_noise.py attaches the damping channel to the `id` instruction. aleph's
//! parser has no `id` lowering (`lower.rs` returns `None` → the QASM would fail
//! to parse) and the `Gate` enum has no identity variant. So the two damping
//! circuits are built via the IR API with an explicit identity `Gate::Unitary1q`
//! (a genuine no-op on the state, `Gate::name() == "Unitary1q"`), and the
//! channel is attached to "Unitary1q". The resulting state evolution — H then a
//! no-op then the damping channel — is identical to Aer's H-then-id-with-damping.

use std::path::Path;

use aleph_core::{Complex, Gate, GateInstance};
use aleph_ir::Circuit;
use aleph_oracle::assert_distribution_close;
use aleph_parser::parse;
use aleph_sv::noise::{
    amplitude_damping_error, depolarizing_error, phase_damping_error, run_noisy, NoiseModel,
    ReadoutError,
};

const SHOTS: u32 = 100_000;
const SEED: u64 = 0;

#[derive(serde::Deserialize)]
struct NoiseFixture {
    name: String,
    num_qubits: u32,
    exact_probs: Vec<f64>,
}

fn load(name: &str) -> NoiseFixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../oracle/noise")
        .join(format!("{name}.json"));
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn check(name: &str, circ: &Circuit, nm: &NoiseModel) {
    let fx = load(name);
    assert_eq!(
        fx.num_qubits,
        circ.num_qubits(),
        "{name}: fixture num_qubits {} != circuit {}",
        fx.num_qubits,
        circ.num_qubits()
    );
    assert_eq!(
        fx.exact_probs.len(),
        1usize << fx.num_qubits,
        "{name}: fixture exact_probs len {} != 2^{}",
        fx.exact_probs.len(),
        fx.num_qubits
    );
    let counts = run_noisy(circ, nm, SHOTS, SEED).unwrap();
    assert_distribution_close(&fx.name, fx.num_qubits, &counts, &fx.exact_probs, SHOTS);
}

fn from_qasm(qasm: &str) -> Circuit {
    parse(qasm).unwrap()
}

/// Build `H q[0]; <identity carrier> q[0];` on a 1-qubit circuit. The carrier
/// is `Gate::Unitary1q(I)` — a no-op whose `Gate::name()` is "Unitary1q", the
/// key the damping channels attach to here (see module docs for why we can't
/// use the QASM `id`).
fn h_then_id_carrier() -> Circuit {
    let zero = Complex::new(0.0, 0.0);
    let one = Complex::new(1.0, 0.0);
    let identity = Box::new([[one, zero], [zero, one]]);
    let mut c = Circuit::new(1, 0);
    c.add_gate(GateInstance::new(Gate::H, &[0u32][..])).unwrap();
    c.add_gate(GateInstance::new(Gate::Unitary1q(identity), &[0u32][..]))
        .unwrap();
    c
}

const H1: &str = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; h q[0];";
const BELL: &str = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; h q[0]; cx q[0], q[1];";
const X1: &str = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; x q[0];";
const GHZ3: &str =
    "OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; h q[0]; cx q[0], q[1]; cx q[1], q[2];";

#[test]
fn oracle_depol_h() {
    let mut nm = NoiseModel::new();
    nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), &["H"]);
    check("depol_h", &from_qasm(H1), &nm);
}

#[test]
fn oracle_depol_cx() {
    let mut nm = NoiseModel::new();
    nm.add_quantum_error(depolarizing_error(0.1, 2), &["Cnot"], &[0, 1]);
    check("depol_cx", &from_qasm(BELL), &nm);
}

#[test]
fn oracle_amp_damp() {
    let mut nm = NoiseModel::new();
    nm.add_quantum_error(amplitude_damping_error(0.2), &["Unitary1q"], &[0]);
    check("amp_damp_h", &h_then_id_carrier(), &nm);
}

#[test]
fn oracle_phase_damp() {
    let mut nm = NoiseModel::new();
    nm.add_quantum_error(phase_damping_error(0.3), &["Unitary1q"], &[0]);
    check("phase_damp_h", &h_then_id_carrier(), &nm);
}

#[test]
fn oracle_readout() {
    let mut nm = NoiseModel::new();
    nm.add_readout_error(ReadoutError::new([[0.98, 0.02], [0.05, 0.95]]), 0);
    check("readout_x", &from_qasm(X1), &nm);
}

#[test]
fn oracle_combined_ghz3() {
    let mut nm = NoiseModel::new();
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), &["Cnot"]);
    for q in 0..3 {
        nm.add_readout_error(ReadoutError::new([[0.97, 0.03], [0.04, 0.96]]), q);
    }
    check("combined_ghz3", &from_qasm(GHZ3), &nm);
}
