//! Lazy-transfer verification (P5-05): only the initial state and the final
//! results cross PCIe. The state vector stays device-resident across all gates,
//! and GPU-resident readout copies back only a scalar / `2^k` marginal / `shots`
//! indices — never the full `2^n` amplitudes.
//!
//! We assert this with an in-process device→host byte counter
//! (`aleph_cuda::device_dtoh_bytes`), which is more deterministic than an Nsight
//! trace and catches any regression that reintroduces a full-state download.
//! Skips cleanly without a CUDA device.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::{run, Backend};
use aleph_core::{Pauli, PauliString};
use aleph_cuda::{device_dtoh_bytes, CudaSvBackend};
use aleph_ir::Circuit;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn random_brickwall(rng: &mut StdRng, n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..n {
            c.ry(rng.gen::<f64>() * std::f64::consts::TAU, q).unwrap();
        }
        for q in 0..n.saturating_sub(1) {
            c.cnot(q, q + 1).unwrap();
        }
    }
    c
}

#[test]
fn readout_does_not_download_full_state() {
    let mut be = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping transfer test: {e}");
            return;
        }
    };
    let n = 20u32; // full state = 2^20 complex f64 = 16 MiB
    let state_bytes = (1u64 << n) * 16;
    let mut rng = StdRng::seed_from_u64(1);
    let circuit = random_brickwall(&mut rng, n, 12);

    // Applying many gates must not download the state at all.
    let before_gates = device_dtoh_bytes();
    let state = run(&mut be, &circuit).expect("run");
    let gate_dtoh = device_dtoh_bytes() - before_gates;
    assert_eq!(
        gate_dtoh, 0,
        "gate application crossed PCIe device→host ({gate_dtoh} bytes); state should stay resident"
    );

    // Each readout copies back only a small result, never ~state_bytes.
    let small = state_bytes / 64; // generous ceiling; real results are far smaller

    let b0 = device_dtoh_bytes();
    let _ = be
        .expectation_value(&state, &PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap())
        .expect("expectation");
    let exp_dtoh = device_dtoh_bytes() - b0;
    assert!(
        exp_dtoh < small,
        "expectation downloaded {exp_dtoh} bytes (state is {state_bytes})"
    );

    let b1 = device_dtoh_bytes();
    let _ = be.probabilities(&state, &[0, 1, 2]).expect("probabilities");
    let prob_dtoh = device_dtoh_bytes() - b1;
    assert!(
        prob_dtoh < small,
        "probabilities downloaded {prob_dtoh} bytes (state is {state_bytes})"
    );

    let b2 = device_dtoh_bytes();
    let _ = be.sample(&state, 4096).expect("sample");
    let sample_dtoh = device_dtoh_bytes() - b2;
    // Sampling copies back only the `shots` u64 indices (+ the tiny norm check),
    // independent of the 2^n state size.
    assert!(
        sample_dtoh < small,
        "sample downloaded {sample_dtoh} bytes (state is {state_bytes})"
    );

    let mut st = state;
    let b3 = device_dtoh_bytes();
    let _ = be.measure(&mut st, 0).expect("measure");
    let meas_dtoh = device_dtoh_bytes() - b3;
    assert!(
        meas_dtoh < small,
        "measure downloaded {meas_dtoh} bytes (state is {state_bytes}); collapse must stay on device"
    );
}

/// Wall-clock of GPU-resident readout + bytes-saved vs a full-state download.
/// `#[ignore]`; run with
/// `cargo test -p aleph-cuda --features cuda --release -- --ignored --nocapture
/// readout_throughput`.
#[test]
#[ignore]
fn readout_throughput() {
    use std::time::Instant;

    let mut be = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let n: u32 = std::env::var("ALEPH_READOUT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let state_bytes = (1u64 << n) * 16;
    let mut rng = StdRng::seed_from_u64(1);
    let circuit = random_brickwall(&mut rng, n, 16);
    let state = run(&mut be, &circuit).expect("run");
    let z0 = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();

    let reps = 200;
    let b0 = device_dtoh_bytes();
    let t = Instant::now();
    for _ in 0..reps {
        let _ = be.expectation_value(&state, &z0).expect("expect");
    }
    let us = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
    let dtoh = (device_dtoh_bytes() - b0) / reps;

    println!(
        "expectation n={n} ({} MiB state): {us:.1} µs/op, {dtoh} B/op downloaded \
         (vs {} MiB for a full-state download — {}× less PCIe)",
        state_bytes / (1 << 20),
        state_bytes / (1 << 20),
        state_bytes / dtoh.max(1),
    );
}
