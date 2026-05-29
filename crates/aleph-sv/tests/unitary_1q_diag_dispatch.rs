//! Confirms `Gate::Unitary1qDiag` routes through the diagonal-1q
//! kernel in `NaiveSvBackend`. Backends classify on matrix shape
//! (`is_diagonal_2x2` in `crates/aleph-sv/src/kernels/mod.rs:113`),
//! so the variant requires no backend code change — this test simply
//! pins the behaviour against a known basis-state expectation.
//!
//! Math: starting from `|1⟩` (prepared via `X` on `|0⟩`), applying
//! `diag(1, e^{iπ/2}) = diag(1, i)` yields `i·|1⟩`, i.e.
//! `amps[0] = 0`, `amps[1] = i`. If a future refactor changes the
//! classifier so that `Unitary1qDiag` is no longer recognised as a
//! diagonal matrix shape, the dispatch will silently shift; this
//! oracle pins the post-dispatch numerical result.

use aleph_backend::run;
use aleph_core::gate::{Gate, GateInstance};
use aleph_core::Complex;
use aleph_ir::{Circuit, Instruction};
use aleph_sv::NaiveSvBackend;
use smallvec::smallvec;

const TOL: f64 = 1e-14;

#[test]
fn naive_backend_executes_unitary_1q_diag() {
    // Prepare |1⟩, then apply diag(1, i).
    let mut c = Circuit::new(1, 0);
    c.x(0).expect("X on q0 of a 1-qubit circuit is in range");
    let gate = Gate::Unitary1qDiag(Box::new([
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0), // e^{iπ/2}
    ]));
    c.add_instruction(Instruction::Gate(GateInstance::new(gate, smallvec![0u32])))
        .expect("Unitary1qDiag on q0 is a valid 1q gate");

    let mut backend = NaiveSvBackend::with_seed(0);
    let state = run(&mut backend, &c).expect("naive backend must execute diagonal 1q gate");
    let amps = state.amplitudes();

    assert_eq!(amps.len(), 2);
    assert!(
        (amps[0] - Complex::new(0.0, 0.0)).norm() < TOL,
        "amps[0] expected 0, got {:?}",
        amps[0]
    );
    assert!(
        (amps[1] - Complex::new(0.0, 1.0)).norm() < TOL,
        "amps[1] expected i, got {:?}",
        amps[1]
    );
}
