//! SoA mirror of `unitary_1q_diag_dispatch.rs` — confirms
//! `Gate::Unitary1qDiag` routes through the SoA diagonal-1q kernel.
//!
//! Gated to `x86_64` builds and requires `avx512f` at runtime — the
//! SoA kernels exercised by P1-05/06/07/08 are AVX-512 specialised
//! (see ADR 0008/0011). On aarch64 hosts (e.g. local M-series dev
//! machines) the `is_x86_feature_detected!` runtime check returns
//! `false` and the test is a no-op; on EPYC CI it executes.
//!
//! **Why n=5, target=q3 instead of n=1, target=q0?** Round-2 code
//! review caught that the original n=1 circuit had state size 2,
//! below `LANES_SOA = 8`. Several SoA kernels (e.g. Tier-A diagonal
//! variants) only fire for `target >= LANES_SOA_BITS = 3` —
//! see `crates/aleph-sv/src/kernels/soa.rs` lines 172, 192. Even
//! though `apply_1q_diagonal_soa` itself is a scalar loop today
//! (relying on LLVM auto-vectorisation), the n=5 / q3 configuration
//! exercises a state size of 32 amplitudes with the target in the
//! "high-bit" regime so the test stays meaningful if a future
//! refactor specialises the dispatch on target position.
//!
//! Math: starting from `|00000⟩`, `X` on q3 produces `|01000⟩`
//! (amp index 8, per ADR 0004 — qubit `q` is bit `q` of the index).
//! Then `diag(1, i)` on q3 multiplies amplitudes whose bit-3 is set
//! by `i`. So amp[8] = `i`; all other amps = 0.

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
        // n=5 → state size 32; target=q3 → target_bit = 8 ≥ LANES_SOA.
        let mut c = Circuit::new(5, 0);
        c.x(3).expect("X on q3 of a 5-qubit circuit is in range");
        let gate = Gate::Unitary1qDiag(Box::new([
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0), // e^{iπ/2}
        ]));
        c.add_instruction(Instruction::Gate(GateInstance::new(gate, smallvec![3u32])))
            .expect("Unitary1qDiag on q3 is a valid 1q gate");

        let mut backend = SoaSvBackend::with_seed(0);
        let state = run(&mut backend, &c).expect("SoA backend must execute diagonal 1q gate");
        // SoA stores (re, im) split; materialise into AoS for comparison.
        let amps = state.to_aos();

        // Per ADR 0004: amp index 8 = binary 01000 ↔ q3=1, others 0.
        // After X on q3 and diag(1, i) on q3: amp[8] = i, all others = 0.
        const ONE_AMP_INDEX: usize = 8;

        assert_eq!(amps.len(), 32);
        for (i, a) in amps.iter().enumerate() {
            let expected = if i == ONE_AMP_INDEX {
                Complex::new(0.0, 1.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            assert!(
                (*a - expected).norm() < TOL,
                "amps[{}] expected {:?}, got {:?}",
                i,
                expected,
                a
            );
        }
    }
}
