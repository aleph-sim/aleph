//! Oracle tests for P1-08 multi-controlled gate kernels.
//!
//! Verifies CCX (permuted qubit order), Grover-CCZ, CCCX (Toffoli with
//! 1 external control), and MCX k=7 (X with 7 controls) against
//! analytically derived expected amplitudes to within `STATE_TOLERANCE`
//! (1e-10 per amplitude, per `docs/testing.md`).
//!
//! - CCX permuted and Grover-CCZ go through the QASM parser + oracle harness
//!   (exercising the full dispatch chain from parse → IR → backend).
//! - CCCX and MCX k=7 use the public Backend API directly because the QASM
//!   parser does not support external controls or MCX mnemonics. Both still
//!   compare against analytically computed amplitudes at 1e-10 tolerance.
//!
//! The MCX k=7 test is the verification anchor for the BACKLOG acceptance
//! criterion "generic MCX with up to 8 controls" — it routes through P1-05's
//! anti-diagonal kernel with 7 external controls.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance};
use aleph_oracle::{
    load_fixture, load_qasm, run_state_oracle, workspace_path, Fixture, StateVectorFixture,
    STATE_TOLERANCE,
};
use aleph_sv::NaiveSvBackend;
use smallvec::smallvec;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a synthetic `Fixture` from analytically known amplitudes.
/// Mirrors the helper used in `harness.rs` unit tests.
fn synth(name: &str, n: u32, amps: Vec<(f64, f64)>) -> Fixture {
    Fixture {
        schema_version: 1,
        name: name.into(),
        num_qubits: n,
        qasm_path: format!("circuits/{name}.qasm"),
        qiskit_version: "hand-crafted".into(),
        aer_version: "hand-crafted".into(),
        generated_at: "2026-05-28T00:00:00Z".into(),
        shots: 100_000,
        rng_seed: 0,
        statevector: StateVectorFixture {
            endianness: "little".into(),
            amplitudes: amps,
        },
        counts: BTreeMap::new(),
    }
}

/// Assert every amplitude in `actual` is within `STATE_TOLERANCE` of the
/// corresponding (re, im) pair in `expected`. Panics with a structured
/// message on the first violation.
fn assert_amps_close(name: &str, actual: &[aleph_core::Complex], expected: &[(f64, f64)]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: amplitude vector length mismatch"
    );
    for (i, (a, &(er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
        if !a.re.is_finite() || !a.im.is_finite() || !er.is_finite() || !ei.is_finite() {
            panic!(
                "oracle: {name} non-finite amplitude at index {i}: ours=({}, {}) expected=({}, {})",
                a.re, a.im, er, ei
            );
        }
        let dr = a.re - er;
        let di = a.im - ei;
        let delta = (dr * dr + di * di).sqrt();
        assert!(
            delta <= STATE_TOLERANCE,
            "oracle: {name} amplitude mismatch at index {i}\n  ours     ({:.16e}, {:.16e})\n  expected ({:.16e}, {:.16e})\n  |Δ| {:.3e} > tol {:.3e}",
            a.re, a.im, er, ei, delta, STATE_TOLERANCE,
        );
    }
}

// ---------------------------------------------------------------------------
// Test 1: CCX with permuted qubit order (via QASM + oracle harness)
// ---------------------------------------------------------------------------

/// `ccx q[1], q[2], q[0]` applied to |011⟩ (q[1]=q[2]=1, q[0]=0).
///
/// CCX gate with controls q[1] and q[2], target q[0].
/// Initial state index: q[0]*1 + q[1]*2 + q[2]*4 = 0+2+4 = 6.
/// After CCX (both controls set): q[0] flips → index 1+2+4 = 7.
///
/// This exercises the Toffoli dispatch with a different qubit ordering than
/// `kernel_ccx` (which uses ccx q[0],q[1],q[2]) — ensuring the kernel is
/// not sensitive to the specific c0/c1/t assignment.
#[test]
fn multi_ctrl_ccx_permuted_naive() {
    let json_path = workspace_path("oracle/fixtures/multi_ctrl_ccx_permuted.json");
    let qasm_path = workspace_path("oracle/circuits/multi_ctrl_ccx_permuted.qasm");
    let fx = load_fixture(&json_path).expect("load fixture multi_ctrl_ccx_permuted");
    let qasm = load_qasm(&qasm_path).expect("load qasm multi_ctrl_ccx_permuted");
    let mut backend = NaiveSvBackend::with_seed(0);
    run_state_oracle(&mut backend, &fx, &qasm).expect("oracle multi_ctrl_ccx_permuted");
}

#[test]
fn multi_ctrl_ccx_permuted_soa() {
    use aleph_sv::SoaSvBackend;
    let json_path = workspace_path("oracle/fixtures/multi_ctrl_ccx_permuted.json");
    let qasm_path = workspace_path("oracle/circuits/multi_ctrl_ccx_permuted.qasm");
    let fx = load_fixture(&json_path).expect("load fixture multi_ctrl_ccx_permuted");
    let qasm = load_qasm(&qasm_path).expect("load qasm multi_ctrl_ccx_permuted");
    let mut backend = SoaSvBackend::with_seed(0);
    run_state_oracle(&mut backend, &fx, &qasm).expect("oracle multi_ctrl_ccx_permuted soa");
}

// ---------------------------------------------------------------------------
// Test 2: Grover-CCZ oracle (via QASM + oracle harness)
// ---------------------------------------------------------------------------

/// H⊗3 then CCZ(q0,q1,q2) on n=3 qubits.
///
/// CCZ sign-flips the amplitude at |111⟩ (index 7 in little-endian). The
/// initial uniform superposition from H⊗3 means all 8 amplitudes start at
/// 1/√8 ≈ 0.35355... After CCZ, amplitude at index 7 becomes -1/√8.
///
/// Analytically: state = (1/√8) * [1, 1, 1, 1, 1, 1, 1, -1].
///
/// This exercises:
/// - `Gate::Ccz` parsed from the new `ccz` QASM gate name (added for P1-08).
/// - The CCZ dispatch in `apply_3q` (via the `is_ccz` shape detector).
/// - P1-08's `dispatch_ccz` Tier A/C paths.
#[test]
fn multi_ctrl_grover_ccz_3q_naive() {
    let json_path = workspace_path("oracle/fixtures/multi_ctrl_grover_ccz_3q.json");
    let qasm_path = workspace_path("oracle/circuits/multi_ctrl_grover_ccz_3q.qasm");
    let fx = load_fixture(&json_path).expect("load fixture multi_ctrl_grover_ccz_3q");
    let qasm = load_qasm(&qasm_path).expect("load qasm multi_ctrl_grover_ccz_3q");
    let mut backend = NaiveSvBackend::with_seed(0);
    run_state_oracle(&mut backend, &fx, &qasm).expect("oracle multi_ctrl_grover_ccz_3q");
}

#[test]
fn multi_ctrl_grover_ccz_3q_soa() {
    use aleph_sv::SoaSvBackend;
    let json_path = workspace_path("oracle/fixtures/multi_ctrl_grover_ccz_3q.json");
    let qasm_path = workspace_path("oracle/circuits/multi_ctrl_grover_ccz_3q.qasm");
    let fx = load_fixture(&json_path).expect("load fixture multi_ctrl_grover_ccz_3q");
    let qasm = load_qasm(&qasm_path).expect("load qasm multi_ctrl_grover_ccz_3q");
    let mut backend = SoaSvBackend::with_seed(0);
    run_state_oracle(&mut backend, &fx, &qasm).expect("oracle multi_ctrl_grover_ccz_3q soa");
}

// ---------------------------------------------------------------------------
// Test 3: CCCX — Toffoli with 1 external control (via Backend API)
// ---------------------------------------------------------------------------

/// CCCX (4-qubit Toffoli): Gate::Toffoli on qubits=[0,1,2], external
/// controls=[3] applied to n=4, starting from |1011⟩.
///
/// State indexing (little-endian): q[0]*1 + q[1]*2 + q[2]*4 + q[3]*8.
/// Initial state |1011⟩: q[0]=1, q[1]=1, q[2]=0, q[3]=1 → index 1+2+0+8 = 11.
///
/// CCCX fires when q[0]=q[1]=q[3]=1 (the two Toffoli controls + 1 external
/// control). q[2] (target) gets flipped: 0 → 1 → index 1+2+4+8 = 15.
///
/// Expected: amplitude 1.0 at index 15, zero elsewhere.
///
/// The QASM parser does not support external controls or `cccx`; this test
/// exercises the path via the public `Backend::apply_gate` API. It validates
/// the P1-08 scalar Tier-C path for Toffoli with external controls (and the
/// SIMD Tier-A path on x86_64 with AVX-512F, if available).
#[test]
fn multi_ctrl_cccx_4q_oracle() {
    // Expected: basis state at index 15 = 0b1111.
    let expected: Vec<(f64, f64)> = (0..16)
        .map(|i| if i == 15 { (1.0, 0.0) } else { (0.0, 0.0) })
        .collect();
    let fx = synth("multi_ctrl_cccx_4q", 4, expected.clone());

    let mut backend = NaiveSvBackend::with_seed(0);
    let mut state = backend.allocate(4).unwrap();

    // Build initial state |1011⟩: flip q[0], q[1], q[3]; leave q[2]=0.
    let x0 = GateInstance::new(Gate::X, smallvec![0u32]);
    let x1 = GateInstance::new(Gate::X, smallvec![1u32]);
    let x3 = GateInstance::new(Gate::X, smallvec![3u32]);
    backend.apply_gate(&mut state, &x0).unwrap();
    backend.apply_gate(&mut state, &x1).unwrap();
    backend.apply_gate(&mut state, &x3).unwrap();

    // CCCX: Gate::Toffoli (c0=0, c1=1, t=2) + external control = 3.
    let cccx = GateInstance::controlled(Gate::Toffoli, smallvec![0u32, 1u32, 2u32], smallvec![3u32]);
    backend.apply_gate(&mut state, &cccx).unwrap();

    let actual = state.amplitudes();
    assert_amps_close(&fx.name, actual, &expected);
}

// ---------------------------------------------------------------------------
// Test 4: MCX k=7 — X with 7 controls (via Backend API)
// ---------------------------------------------------------------------------

/// MCX with k=7 controls: Gate::X on target=7 with controls=[0,1,2,3,4,5,6].
///
/// Initial state |01111111⟩: q[0..6]=1, q[7]=0.
/// Little-endian index: 1+2+4+8+16+32+64 = 127.
///
/// When all 7 control bits q[0..6] are 1, q[7] (target) flips: 0 → 1.
/// Final state |11111111⟩ = index 127+128 = 255.
///
/// Expected: amplitude 1.0 at index 255, zero elsewhere (256 entries).
///
/// This is the **MCX verification anchor** for the BACKLOG acceptance
/// criterion "generic MCX with up to 8 controls". It routes through:
/// - `Backend::apply_gate` with a GateMatrix::M2x2 (Gate::X is 1-qubit).
/// - The `apply_1q` dispatch in `kernels/aos.rs`.
/// - P1-05's anti-diagonal kernel with k=7 external controls.
///
/// The QASM parser cannot express this gate; it is tested via the backend API.
#[test]
fn multi_ctrl_mcx_k7_8q_oracle() {
    // Expected: basis state at index 255 = 0b11111111.
    let expected: Vec<(f64, f64)> = (0..256usize)
        .map(|i| if i == 255 { (1.0, 0.0) } else { (0.0, 0.0) })
        .collect();
    let fx = synth("multi_ctrl_mcx_k7_8q", 8, expected.clone());

    let mut backend = NaiveSvBackend::with_seed(0);
    let mut state = backend.allocate(8).unwrap();

    // Build initial state: flip q[0] through q[6] (leave q[7]=0).
    // Index = sum(2^i for i in 0..7) = 127.
    for q in 0u32..7 {
        backend
            .apply_gate(&mut state, &GateInstance::new(Gate::X, smallvec![q]))
            .unwrap();
    }

    // MCX: X on q[7], controlled by q[0..6] (7 controls).
    let mcx = GateInstance::controlled(
        Gate::X,
        smallvec![7u32],
        smallvec![0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32],
    );
    backend.apply_gate(&mut state, &mcx).unwrap();

    let actual = state.amplitudes();
    assert_amps_close(&fx.name, actual, &expected);
}
