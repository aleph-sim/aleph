//! P2-08 correctness gate for the scalar single-precision path.
//!
//! This is THE end-to-end validation for the FP32 backend (Tasks 3–9). It
//! runs a spread of Tier-1 circuits on BOTH the trusted f64 reference
//! (`NaiveSvBackend`) and the new `Fp32SvBackend`, widens the f32 result via
//! `to_aos_f64()`, and asserts the amplitudes agree elementwise within an
//! FP32 tolerance. Each fixture is exercised twice — via `run` (raw,
//! gate-by-gate dispatch) AND via `run_optimized` (fusion pipeline → exercises
//! the fused hot types: Unitary1q/Diag, Unitary2q, UnitaryKq, DiagonalPhase).
//! If a Task 3–9 kernel miscomputes, one of these comparisons catches it.
//!
//! Part 2 adds proptest cases that hammer the f32 backend on random circuits
//! and assert the squared-amplitude norm stays ≈ 1.

use aleph_backend::{run, run_optimized};
use aleph_core::Complex;
use aleph_ir::Circuit;
use aleph_sv::{Fp32SvBackend, NaiveSvBackend};

/// Elementwise absolute tolerance for f64-vs-widened-f32 amplitude compare.
///
/// Single-precision carries ~7 decimal digits (machine ε ≈ 1.2e-7). Over the
/// depths and qubit counts here (n ≤ 12, depth ≤ ~5n) accumulated drift stays
/// comfortably under 1e-4 — the measured worst case across all fixtures is
/// reported in the task notes. We do NOT relax this to mask a kernel bug.
const FP32_TOL: f64 = 1e-4;

// ---------------------------------------------------------------------------
// Tier-1 circuit builders (parameterized by n, via the public Circuit API).
// ---------------------------------------------------------------------------

/// GHZ-n: `H q0; CX q0,q1; CX q1,q2; …`. Final state is
/// (|0…0⟩ + |1…1⟩)/√2. Exercises H + a CNOT chain.
fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for i in 0..n - 1 {
        c.cnot(i, i + 1).unwrap();
    }
    c
}

/// Controlled-phase `cp(λ)` on (control, target) via the textbook
/// CNOT+Rz decomposition (same idiom as `aleph-sv/tests/tier1.rs`). This
/// keeps us on well-trodden Cnot + Rz kernels rather than relying on a
/// controlled-phase qubit-order convention. `cp(λ) = diag(1,1,1,e^{iλ})`.
fn cphase(c: &mut Circuit, lambda: f64, control: u32, target: u32) {
    c.rz(lambda / 2.0, target).unwrap();
    c.cnot(control, target).unwrap();
    c.rz(-lambda / 2.0, target).unwrap();
    c.cnot(control, target).unwrap();
    c.rz(lambda / 2.0, control).unwrap();
}

/// QFT-n on the all-|0…0⟩-after-a-seed input. We first seed a non-trivial
/// computational basis state (X on the even qubits) so the controlled
/// phases actually fire, then apply the standard QFT (H + cascade of
/// controlled phases + final qubit-reversal swaps). Exercises H, Rz, CNOT,
/// and Swap — and, after fusion, DiagonalPhase / Unitary2q / UnitaryKq.
fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    // Seed |x⟩ with a non-uniform x so phases are visible.
    for q in (0..n).step_by(2) {
        c.x(q).unwrap();
    }
    for j in 0..n {
        c.h(j).unwrap();
        for k in (j + 1)..n {
            let lambda = std::f64::consts::PI / (1u64 << (k - j)) as f64;
            cphase(&mut c, lambda, k, j);
        }
    }
    // Bit-reversal swaps.
    let mut a = 0;
    let mut b = n - 1;
    while a < b {
        c.swap(a, b).unwrap();
        a += 1;
        b -= 1;
    }
    c
}

/// One Grover iteration over n qubits, marking |1…1⟩. Layout:
/// uniform superposition (H^⊗n), oracle (phase-flip on |1…1⟩ via a CCZ
/// expansion through Toffoli), then the diffusion operator. Uses
/// H, X, CZ, Toffoli — covering 1q, diagonal-2q, and 3q kernels.
fn grover_iter(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    let iters = 1; // one iteration is enough to exercise every kernel.
    for _ in 0..iters {
        // Oracle: phase-flip |1…1⟩  ==  multi-controlled Z on all qubits.
        mcz(&mut c, n);
        // Diffusion: H^⊗n, X^⊗n, multi-controlled-Z, X^⊗n, H^⊗n.
        for q in 0..n {
            c.h(q).unwrap();
        }
        for q in 0..n {
            c.x(q).unwrap();
        }
        mcz(&mut c, n);
        for q in 0..n {
            c.x(q).unwrap();
        }
        for q in 0..n {
            c.h(q).unwrap();
        }
    }
    c
}

/// A diagonal phase-marking block on qubits `0..n`. For n==1 it's a plain Z;
/// n==2 a CZ; n>=3 a CCZ on the top three qubits (via H·CCX·H on the last
/// qubit) followed by a CZ ladder so every remaining qubit participates. This
/// is NOT a true multi-controlled Z — it's a representative diagonal/3q
/// workload for the Grover-style fixture, and equivalence is checked against
/// the f64 backend (the oracle), so the exact unitary need not be the textbook
/// MCZ. Exercises the diagonal-2q (CZ) and 3q (Toffoli) kernels.
fn mcz(c: &mut Circuit, n: u32) {
    match n {
        1 => {
            c.z(0).unwrap();
        }
        2 => {
            c.cz(0, 1).unwrap();
        }
        _ => {
            // CCZ on the top 3 qubits as a representative diagonal-3q flip,
            // plus a CZ ladder over the rest so every qubit participates.
            let t = n - 1;
            c.h(t).unwrap();
            c.ccx(0, 1, t).unwrap();
            c.h(t).unwrap();
            for q in 2..(n - 1) {
                c.cz(q, t).unwrap();
            }
        }
    }
}

/// Deterministic-but-varied "random-brickwall" circuit: alternating layers of
/// single-qubit rotations (Rx/Ry/Rz/Phase, angle from a cheap LCG) and a
/// brickwall of CNOT/CZ entanglers. Exercises generic 1q dense kernels, the
/// diagonal-1q (Phase) path, and both 2q kernels. Depth ~ `depth` layers.
fn random_brickwall(n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    // Tiny deterministic LCG → reproducible "random" angles/choices.
    let mut state: u64 = 0x9E3779B97F4A7C15u64
        .wrapping_add(n as u64)
        .wrapping_mul(depth as u64 + 1);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for layer in 0..depth {
        // Single-qubit layer.
        for q in 0..n {
            let r = next();
            let angle =
                ((next() % 1000) as f64 / 1000.0) * std::f64::consts::TAU - std::f64::consts::PI;
            match r % 4 {
                0 => c.rx(angle, q).unwrap(),
                1 => c.ry(angle, q).unwrap(),
                2 => c.rz(angle, q).unwrap(),
                _ => c.phase(angle, q).unwrap(),
            };
        }
        // Entangling brickwall: even pairs on even layers, odd pairs on odd.
        let start = layer % 2;
        let mut q = start;
        while q + 1 < n {
            if next() % 2 == 0 {
                c.cnot(q, q + 1).unwrap();
            } else {
                c.cz(q, q + 1).unwrap();
            }
            q += 2;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Part 1 — f32 vs f64 equivalence.
// ---------------------------------------------------------------------------

/// f64 reference amplitudes (raw run).
fn f64_raw(c: &Circuit) -> Vec<Complex> {
    run(&mut NaiveSvBackend::with_seed(0), c)
        .expect("f64 raw run")
        .amplitudes()
        .to_vec()
}

/// f64 reference amplitudes (optimized run).
fn f64_opt(c: &Circuit) -> Vec<Complex> {
    run_optimized(&mut NaiveSvBackend::with_seed(0), c)
        .expect("f64 optimized run")
        .amplitudes()
        .to_vec()
}

/// f32 amplitudes widened to f64 (raw run).
fn f32_raw(c: &Circuit) -> Vec<Complex> {
    run(&mut Fp32SvBackend::with_seed(0), c)
        .expect("f32 raw run")
        .to_aos_f64()
}

/// f32 amplitudes widened to f64 (optimized run).
fn f32_opt(c: &Circuit) -> Vec<Complex> {
    run_optimized(&mut Fp32SvBackend::with_seed(0), c)
        .expect("f32 optimized run")
        .to_aos_f64()
}

/// Compare two amplitude vectors elementwise; returns the max abs error
/// (over re and im of every index). Asserts within `FP32_TOL`.
fn assert_close(reference: &[Complex], actual: &[Complex], label: &str) -> f64 {
    assert_eq!(
        reference.len(),
        actual.len(),
        "{label}: length mismatch {} vs {}",
        reference.len(),
        actual.len()
    );
    let mut max_err = 0.0f64;
    let mut worst_i = 0usize;
    for (i, (r, a)) in reference.iter().zip(actual.iter()).enumerate() {
        let dre = (r.re - a.re).abs();
        let dim = (r.im - a.im).abs();
        let e = dre.max(dim);
        if e > max_err {
            max_err = e;
            worst_i = i;
        }
    }
    assert!(
        max_err < FP32_TOL,
        "{label}: max abs err {max_err:e} >= {FP32_TOL:e} at amp[{worst_i}]"
    );
    max_err
}

/// Run one fixture both ways (raw + optimized) on both precisions and assert
/// the f32 result tracks the f64 result. Returns the max abs error seen.
fn check_fixture(c: &Circuit, label: &str) -> f64 {
    let ref_raw = f64_raw(c);
    let got_raw = f32_raw(c);
    let e_raw = assert_close(&ref_raw, &got_raw, &format!("{label} [run]"));

    let ref_opt = f64_opt(c);
    let got_opt = f32_opt(c);
    let e_opt = assert_close(&ref_opt, &got_opt, &format!("{label} [run_optimized]"));

    e_raw.max(e_opt)
}

#[test]
fn tier1_f32_matches_f64() {
    let ns = [6u32, 8, 10, 12];
    let mut global_max = 0.0f64;
    let mut worst_label = String::new();

    for &n in &ns {
        let cases: Vec<(String, Circuit)> = vec![
            (format!("ghz_n{n}"), ghz(n)),
            (format!("qft_n{n}"), qft(n)),
            (format!("grover_n{n}"), grover_iter(n)),
            (
                format!("random_brickwall_n{n}_d{}", 2 * n),
                random_brickwall(n, 2 * n),
            ),
        ];
        for (label, circ) in cases {
            let e = check_fixture(&circ, &label);
            println!("{label}: max abs err = {e:e}");
            if e > global_max {
                global_max = e;
                worst_label = label;
            }
        }
    }
    println!("\nGLOBAL MAX f32-vs-f64 abs err = {global_max:e} (worst fixture: {worst_label})");
}

// ---------------------------------------------------------------------------
// Part 2 — property tests (proptest).
// ---------------------------------------------------------------------------

mod property {
    use super::*;
    use aleph_test::circuit::arb_circuit_emittable;
    use proptest::prelude::*;

    /// Filter a generated circuit down to gate-only (drop Measure/Reset),
    /// mirroring the `run_optimized_oracle.rs` helper. Barriers are kept so
    /// the fencing path is exercised on the optimized run.
    fn gate_only(src: &Circuit) -> Circuit {
        use aleph_ir::Instruction;
        let mut out = Circuit::new(src.num_qubits(), src.num_clbits());
        for inst in src.instructions() {
            match inst {
                Instruction::Gate(g) => {
                    out.add_gate(g.clone()).unwrap();
                }
                Instruction::Barrier(_) => {
                    out.add_instruction(inst.clone()).unwrap();
                }
                _ => {}
            }
        }
        out
    }

    /// Squared-amplitude norm of an f64 amplitude vector.
    fn norm_sq(amps: &[Complex]) -> f64 {
        amps.iter().map(|a| a.norm_sqr()).sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// The f32 backend must keep Σ|amp|² ≈ 1 over random gate circuits,
        /// on BOTH the raw and the optimized run path. Drift bounded by the
        /// FP32 tolerance.
        #[test]
        fn fp32_preserves_norm(raw in arb_circuit_emittable(8, 2, 30)) {
            // nq is fixed at 8 by the strategy; it emits a mix of 1q and 2q
            // gates (and barriers), which is exactly the vocabulary we want
            // to stress for norm preservation.
            let c = gate_only(&raw);

            let raw_amps = f32_raw(&c);
            let s_raw = norm_sq(&raw_amps);
            prop_assert!(
                (s_raw - 1.0).abs() < FP32_TOL,
                "f32 [run] norm² = {} (drift {:e})", s_raw, (s_raw - 1.0).abs()
            );

            let opt_amps = f32_opt(&c);
            let s_opt = norm_sq(&opt_amps);
            prop_assert!(
                (s_opt - 1.0).abs() < FP32_TOL,
                "f32 [run_optimized] norm² = {} (drift {:e})", s_opt, (s_opt - 1.0).abs()
            );
        }

        /// Cross-check: on the same random circuit, the widened f32 amplitudes
        /// must track the f64 reference within FP32 tolerance (raw path). This
        /// is the proptest analogue of the Part-1 fixture compare, covering the
        /// generic-dispatch kernels on arbitrary gate sequences.
        #[test]
        fn fp32_tracks_f64_random(raw in arb_circuit_emittable(8, 2, 24)) {
            let c = gate_only(&raw);
            let reference = f64_raw(&c);
            let actual = f32_raw(&c);
            prop_assert_eq!(reference.len(), actual.len());
            for (i, (r, a)) in reference.iter().zip(actual.iter()).enumerate() {
                let e = (r.re - a.re).abs().max((r.im - a.im).abs());
                prop_assert!(
                    e < FP32_TOL,
                    "amp[{}] abs err {:e} >= {:e} (f64={:?}, f32={:?})",
                    i, e, FP32_TOL, r, a
                );
            }
        }
    }
}
