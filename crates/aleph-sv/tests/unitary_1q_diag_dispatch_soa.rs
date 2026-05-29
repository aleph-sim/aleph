//! SoA mirror of `unitary_1q_diag_dispatch.rs` — confirms
//! `Gate::Unitary1qDiag` routes through the SoA diagonal-1q kernel.
//!
//! Gated to `x86_64` builds and requires `avx512f` at runtime — the
//! SoA kernels exercised by P1-05/06/07/08 are AVX-512 specialised
//! (see ADR 0008/0011). On aarch64 hosts (e.g. local M-series dev
//! machines) the `is_x86_feature_detected!` runtime check returns
//! `false` and the test is a no-op; on EPYC CI it executes.
//!
//! Same math as the Naive test: `X|0⟩ = |1⟩`, then
//! `diag(1, i)|1⟩ = i·|1⟩` ⇒ `amps == [0, i]`.

use aleph_backend::run;
use aleph_core::gate::{Gate, GateInstance};
use aleph_core::Complex;
use aleph_ir::{Circuit, Instruction};
use aleph_sv::SoaSvBackend;
use smallvec::smallvec;

const TOL: f64 = 1e-14;

#[test]
fn soa_backend_executes_unitary_1q_diag() {
    // SoA AVX-512 kernels only fire when the host actually has AVX-512.
    // On aarch64 or older x86_64 hosts, exit early so the test passes
    // trivially rather than failing on a missing-feature dispatch.
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        return;
    }

    #[allow(unreachable_code)]
    {
        let mut c = Circuit::new(1, 0);
        c.x(0).expect("X on q0 of a 1-qubit circuit is in range");
        let gate = Gate::Unitary1qDiag(Box::new([
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0), // e^{iπ/2}
        ]));
        c.add_instruction(Instruction::Gate(GateInstance::new(gate, smallvec![0u32])))
            .expect("Unitary1qDiag on q0 is a valid 1q gate");

        let mut backend = SoaSvBackend::with_seed(0);
        let state = run(&mut backend, &c).expect("SoA backend must execute diagonal 1q gate");
        // SoA stores (re, im) split; materialise into AoS for comparison.
        let amps = state.to_aos();

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
}
