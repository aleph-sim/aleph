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

// ---- Tier-1 fixtures (mirror crates/aleph-sv/tests/fp32_equiv.rs) ----

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 1..n {
        c.cnot(0, q).unwrap();
    }
    c
}

fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in 0..n {
        c.h(j).unwrap();
        for (offset, k) in ((j + 1)..n).enumerate() {
            let theta = std::f64::consts::PI / (1u64 << (offset + 1)) as f64;
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(theta)),
                vec![k],
                vec![j],
            ))
            .unwrap();
        }
    }
    c
}

/// Representative diagonal/3q phase-marking block (matches fp32_equiv::mcz).
fn mcz(c: &mut Circuit, n: u32) {
    match n {
        1 => {
            c.z(0).unwrap();
        }
        2 => {
            c.cz(0, 1).unwrap();
        }
        _ => {
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

fn grover_iter(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    mcz(&mut c, n);
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
    c
}

fn random_brickwall(rng: &mut StdRng, n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..n {
            let theta = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
            match rng.gen_range(0..3u32) {
                0 => c.rx(theta, q).unwrap(),
                1 => c.ry(theta, q).unwrap(),
                _ => c.rz(theta, q).unwrap(),
            };
        }
        let mut q = 0;
        while q + 1 < n {
            if rng.gen_bool(0.5) {
                c.cnot(q, q + 1).unwrap();
            } else {
                c.cz(q, q + 1).unwrap();
            }
            q += 2;
        }
    }
    c
}

fn run_oracle(name: &str, circuit: &Circuit) {
    let mut gpu = match MetalSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping {name}: no Metal device");
            return;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let gpu_state = run(&mut gpu, circuit).expect("gpu run");
    let cpu_state = run(&mut cpu, circuit).expect("cpu run");
    assert_close(
        name,
        &HasAmplitudes::amplitudes(&gpu_state),
        &HasAmplitudes::amplitudes(&cpu_state),
    );
}

#[test]
fn tier1_raw_oracle_matches_naive_sv() {
    let mut rng = StdRng::seed_from_u64(0x7173);
    for n in 2..=10u32 {
        run_oracle(&format!("ghz n={n}"), &ghz(n));
        run_oracle(&format!("qft n={n}"), &qft(n));
        run_oracle(&format!("grover n={n}"), &grover_iter(n));
        run_oracle(&format!("random n={n}"), &random_brickwall(&mut rng, n, 6));
    }
}
