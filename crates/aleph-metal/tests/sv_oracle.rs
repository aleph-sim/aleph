//! FP32 state oracle: random single-qubit circuits must match the f64
//! `NaiveSvBackend` reference within 1e-5. Skips (passes) when no Metal device
//! is available so headless CI stays green; runs for real on Apple Silicon.

#![cfg(all(target_os = "macos", feature = "metal"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::{Circuit, Instruction};
use aleph_metal::MetalSvBackend;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Build a random 1q circuit: `gates` single-qubit gates over `n` qubits,
/// drawn deterministically from `rng`.
fn random_1q_circuit(rng: &mut StdRng, n: u32, gates: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..gates {
        let q = rng.gen_range(0..n);
        let theta = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
        let phi = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
        let lam = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
        let gate = match rng.gen_range(0..11u32) {
            0 => Gate::H,
            1 => Gate::X,
            2 => Gate::Y,
            3 => Gate::Z,
            4 => Gate::S,
            5 => Gate::T,
            6 => Gate::Rx(Param::Concrete(theta)),
            7 => Gate::Ry(Param::Concrete(theta)),
            8 => Gate::Rz(Param::Concrete(theta)),
            9 => Gate::Phase(Param::Concrete(theta)),
            _ => Gate::U3(
                Param::Concrete(theta),
                Param::Concrete(phi),
                Param::Concrete(lam),
            ),
        };
        c.add_instruction(Instruction::Gate(GateInstance::new(gate, vec![q])))
            .expect("valid 1q gate");
    }
    c
}

fn assert_close(name: &str, gpu: &[aleph_core::Complex<f64>], cpu: &[aleph_core::Complex<f64>]) {
    assert_eq!(gpu.len(), cpu.len(), "{name}: dim mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            g.re.is_finite() && g.im.is_finite(),
            "{name}: non-finite GPU amplitude at {i}: {g:?}"
        );
        let d = ((g.re - c.re).powi(2) + (g.im - c.im).powi(2)).sqrt();
        assert!(
            d <= 1e-5,
            "{name}: amplitude {i} |Δ|={d:.3e} > 1e-5\n  gpu {g:?}\n  cpu {c:?}"
        );
    }
}

#[test]
fn fp32_1q_oracle_matches_naive_sv() {
    let mut gpu = match MetalSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping FP32 oracle: no Metal device available");
            return;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);

    let mut rng = StdRng::seed_from_u64(0xA1E0);
    for n in 1..=12u32 {
        for trial in 0..4 {
            let circuit = random_1q_circuit(&mut rng, n, 24);
            let gpu_state = run(&mut gpu, &circuit).expect("gpu run");
            let cpu_state = run(&mut cpu, &circuit).expect("cpu run");
            let gpu_amps = gpu_state.amplitudes();
            let cpu_amps = HasAmplitudes::amplitudes(&cpu_state);
            assert_close(&format!("n={n} trial={trial}"), &gpu_amps, &cpu_amps);
        }
    }
}
