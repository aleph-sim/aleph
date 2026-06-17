//! FP32 invariants for `MetalSvBackend`: a random 1q/2q/3q circuit must keep
//! the state normalized (||psi||^2 ~ 1) at FP32 tolerance. Unitarity of the
//! evolution is implied — a non-unitary gate would break normalization — and
//! is also gated per-gate by the backend's `unitarity_deviation` check.
//! Skips (passes) when no Metal device is present so headless CI stays green.

#![cfg(all(target_os = "macos", feature = "metal"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::{Circuit, Instruction};
use aleph_metal::MetalSvBackend;
use aleph_oracle::HasAmplitudes;
use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Build a random circuit mixing 1q, 2q, and 3q gates over `n` qubits.
fn random_mixed_circuit(rng: &mut StdRng, n: u32, gates: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..gates {
        let theta = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
        match rng.gen_range(0..9u32) {
            0 => {
                c.add_instruction(Instruction::Gate(GateInstance::new(
                    Gate::H,
                    vec![rng.gen_range(0..n)],
                )))
                .unwrap();
            }
            1 => {
                c.add_instruction(Instruction::Gate(GateInstance::new(
                    Gate::X,
                    vec![rng.gen_range(0..n)],
                )))
                .unwrap();
            }
            2 => {
                c.add_instruction(Instruction::Gate(GateInstance::new(
                    Gate::Rz(Param::Concrete(theta)),
                    vec![rng.gen_range(0..n)],
                )))
                .unwrap();
            }
            3 => {
                c.add_instruction(Instruction::Gate(GateInstance::new(
                    Gate::Ry(Param::Concrete(theta)),
                    vec![rng.gen_range(0..n)],
                )))
                .unwrap();
            }
            4 if n >= 2 => {
                let (a, b) = distinct_pair(rng, n);
                c.add_instruction(Instruction::Gate(GateInstance::new(Gate::Cnot, vec![a, b])))
                    .unwrap();
            }
            5 if n >= 2 => {
                let (a, b) = distinct_pair(rng, n);
                c.add_instruction(Instruction::Gate(GateInstance::new(Gate::Cz, vec![a, b])))
                    .unwrap();
            }
            6 if n >= 2 => {
                let (a, b) = distinct_pair(rng, n);
                c.add_instruction(Instruction::Gate(GateInstance::new(Gate::Swap, vec![a, b])))
                    .unwrap();
            }
            7 if n >= 3 => {
                let (a, b, d) = distinct_triple(rng, n);
                c.add_instruction(Instruction::Gate(GateInstance::new(
                    Gate::Toffoli,
                    vec![a, b, d],
                )))
                .unwrap();
            }
            // Reached for rng value 8 and as the fallback when arm 7's `n >= 3`
            // Toffoli guard fails (n == 2): substitute a 1q H.
            _ => {
                c.add_instruction(Instruction::Gate(GateInstance::new(
                    Gate::H,
                    vec![rng.gen_range(0..n)],
                )))
                .unwrap();
            }
        }
    }
    c
}

fn distinct_pair(rng: &mut StdRng, n: u32) -> (u32, u32) {
    let a = rng.gen_range(0..n);
    let mut b = rng.gen_range(0..n);
    while b == a {
        b = rng.gen_range(0..n);
    }
    (a, b)
}

fn distinct_triple(rng: &mut StdRng, n: u32) -> (u32, u32, u32) {
    let (a, b) = distinct_pair(rng, n);
    let mut d = rng.gen_range(0..n);
    while d == a || d == b {
        d = rng.gen_range(0..n);
    }
    (a, b, d)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn fp32_state_stays_normalized(seed in any::<u64>(), n in 2u32..=8, gates in 8usize..=40) {
        let mut gpu = match MetalSvBackend::with_seed(seed) {
            Ok(b) => b,
            Err(_) => return Ok(()), // headless: skip
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let circuit = random_mixed_circuit(&mut rng, n, gates);
        let state = run(&mut gpu, &circuit).expect("gpu run");
        let amps = HasAmplitudes::amplitudes(&state);
        let norm_sq: f64 = amps.iter().map(|z| z.norm_sqr()).sum();
        prop_assert!(
            (norm_sq - 1.0).abs() < 1e-4,
            "||psi||^2 = {norm_sq} (n={n}, gates={gates}, seed={seed})"
        );
        // Every amplitude finite (no NaN/Inf leaked through the f32 kernels).
        prop_assert!(amps.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
    }
}
