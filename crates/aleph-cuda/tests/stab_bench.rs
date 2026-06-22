//! P5-07 benchmark: GPU stabilizer Clifford throughput vs the CPU `aleph-stab`
//! backend and vs Stim's `TableauSimulator`, on the *same* random Clifford
//! circuit. All `#[ignore]` — needs a GPU and runs for seconds.
//!
//! ```bash
//! ALEPH_STAB_NS="1000,4000,16000,65000" ALEPH_STAB_DEPTH=200 \
//!   ALEPH_STIM_PY=/root/stimvenv/bin/python \
//!   cargo test -p aleph-cuda --features cuda --release \
//!   -- --ignored --nocapture stab_bench
//! ```
//!
//! Stim is driven over the exact same gate sequence (serialised to a Stim
//! program and piped to `ALEPH_STIM_PY`); if that env var is unset the Stim arm
//! is skipped and only GPU-vs-CPU is reported.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use aleph_backend::run;
use aleph_core::{Gate, GateInstance};
use aleph_cuda::{stab_op, CudaStab, StabOp};
use aleph_ir::Circuit;
use aleph_stab::StabilizerBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Random disjoint-qubit Clifford layers + the matching `Circuit` (for the CPU
/// arm). Same generator as the oracle, so throughput is measured on realistic
/// mixed H/S/CNOT/Pauli traffic.
fn random_clifford(rng: &mut StdRng, n: u32, depth: u32) -> (Vec<Vec<StabOp>>, Circuit, usize) {
    let mut layers = Vec::with_capacity(depth as usize);
    let mut circuit = Circuit::new(n, 0);
    let mut gates = 0usize;
    for _ in 0..depth {
        let mut perm: Vec<u32> = (0..n).collect();
        for i in (1..perm.len()).rev() {
            perm.swap(i, rng.gen_range(0..=i));
        }
        let mut layer = Vec::new();
        let mut i = 0usize;
        while i < perm.len() {
            let q = perm[i];
            if i + 1 < perm.len() && rng.gen_bool(0.5) {
                let t = perm[i + 1];
                layer.push(StabOp::cnot(q, t));
                circuit
                    .add_gate(GateInstance::new(Gate::Cnot, vec![q, t]))
                    .unwrap();
                i += 2;
            } else {
                let (sop, g) = match rng.gen_range(0..5) {
                    0 => (StabOp::h(q), Gate::H),
                    1 => (StabOp::s(q), Gate::S),
                    2 => (StabOp::x(q), Gate::X),
                    3 => (StabOp::y(q), Gate::Y),
                    _ => (StabOp::z(q), Gate::Z),
                };
                layer.push(sop);
                circuit.add_gate(GateInstance::new(g, vec![q])).unwrap();
                i += 1;
            }
        }
        gates += layer.len();
        layers.push(layer);
    }
    (layers, circuit, gates)
}

/// Serialise the layers to a Stim program (`H 0 1`, `S 2`, `CX 3 4`, …).
fn to_stim(layers: &[Vec<StabOp>]) -> String {
    let mut s = String::new();
    for layer in layers {
        for g in layer {
            let line = match g.op {
                stab_op::H => format!("H {}\n", g.a),
                stab_op::S => format!("S {}\n", g.a),
                stab_op::CNOT => format!("CX {} {}\n", g.a, g.b),
                stab_op::X => format!("X {}\n", g.a),
                stab_op::Y => format!("Y {}\n", g.a),
                _ => format!("Z {}\n", g.a),
            };
            s.push_str(&line);
        }
    }
    s
}

/// Drive Stim over the exact circuit; returns best-of-3 `do_circuit` seconds, or
/// `None` if `ALEPH_STIM_PY` is unset or the call fails.
fn stim_seconds(layers: &[Vec<StabOp>]) -> Option<f64> {
    let py = std::env::var("ALEPH_STIM_PY").ok()?;
    let prog = to_stim(layers);
    let script = r#"
import sys, time, stim
circ = stim.Circuit(sys.stdin.read())
sim = stim.TableauSimulator(); sim.do_circuit(circ)  # warmup
best = float('inf')
for _ in range(3):
    sim = stim.TableauSimulator()
    t = time.perf_counter(); sim.do_circuit(circ); best = min(best, time.perf_counter()-t)
print(best)
"#;
    let mut child = Command::new(py)
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(prog.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        eprintln!("stim failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

#[test]
#[ignore]
fn stab_bench() {
    let depth = env_usize("ALEPH_STAB_DEPTH", 200) as u32;
    let reps = env_usize("ALEPH_STAB_REPS", 3);
    let ns: Vec<u32> = std::env::var("ALEPH_STAB_NS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1000, 4000, 16000, 65000]);

    let driver = match CudaStab::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping stab_bench: {e}");
            return;
        }
    };

    println!("== stabilizer Clifford throughput (depth={depth}, reps={reps}) ==");
    println!("    n     gates |  GPU batched   CPU aleph     Stim    | GPU/CPU  GPU/Stim");
    for n in ns {
        let mut rng = StdRng::seed_from_u64(0x57AB_0000 ^ n as u64);
        let (layers, circuit, gates) = random_clifford(&mut rng, n, depth);

        // GPU batched: fresh alloc + all layers + sync, best of `reps`.
        let mut gpu = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let mut st = driver.allocate(n).unwrap();
            for layer in &layers {
                driver.apply_layer(&mut st, layer).unwrap();
            }
            driver.synchronize().unwrap();
            gpu = gpu.min(t.elapsed().as_secs_f64());
        }

        // CPU aleph-stab.
        let mut cpu = f64::INFINITY;
        for _ in 0..reps {
            let mut be = StabilizerBackend::with_seed(0);
            let t = Instant::now();
            let _ = run(&mut be, &circuit).unwrap();
            cpu = cpu.min(t.elapsed().as_secs_f64());
        }

        let stim = stim_seconds(&layers);
        let gps = |s: f64| gates as f64 / s / 1e6; // M gates/s
        let stim_str = stim
            .map(|s| format!("{:7.1}", gps(s)))
            .unwrap_or_else(|| "    n/a".into());
        let gpu_stim = stim
            .map(|s| format!("{:6.2}x", s / gpu))
            .unwrap_or_else(|| "   n/a".into());
        println!(
            "{n:6} {gates:9} | {:8.1}    {:8.1}  {} | {:6.2}x  {}",
            gps(gpu),
            gps(cpu),
            stim_str,
            cpu / gpu,
            gpu_stim,
        );
    }
    println!("(throughput in M gates/s; ratios >1 mean GPU faster)");
}
