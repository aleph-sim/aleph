//! P5.11-04: FP32 GPU-resident readout correctness.
//!
//! `CudaSvBackendF32` is now a first-class [`Backend`] with GPU-resident
//! `measure` / `sample` / `expectation_value` / `probabilities` (reductions read
//! the f32 amplitudes but accumulate in f64). This pins those against the exact
//! FP64 CPU `NaiveSvBackend`:
//! - **expectation / marginal probabilities** within **1e-5** (FP32 accuracy),
//! - **sampling** statistically against the exact distribution (100k+ shots),
//! - **measurement** via GHZ correlation + a valid collapsed basis state (FP32
//!   coin-flip probabilities can straddle the FP64 ones, so we don't pin the
//!   per-qubit outcome against the CPU coin — we pin GHZ's all-equal invariant).
//!
//! Exercises the [`Backend`] trait path (`aleph_backend::run` → trait
//! `apply_gate`). Gated on `cfg(all(target_os = "linux", feature = "cuda"))`;
//! skips cleanly with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::{run, Backend};
use aleph_core::{Pauli, PauliString};
use aleph_cuda::CudaSvBackendF32;
use aleph_ir::Circuit;
use aleph_sv::NaiveSvBackend;

/// FP32 accuracy tolerance for expectation / marginal comparisons.
const TOL: f64 = 1e-5;

fn mixed_state(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.ry(0.3 + 0.37 * q as f64, q).unwrap();
        c.rz(0.17 * (q as f64 + 1.0), q).unwrap();
    }
    for q in 0..n.saturating_sub(1) {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 1..n {
        c.cnot(0, q).unwrap();
    }
    c
}

fn test_paulis(n: u32) -> Vec<PauliString> {
    let mut v = vec![
        PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap(),
        PauliString::new(1.0, vec![(0, Pauli::X)]).unwrap(),
        PauliString::new(1.0, vec![(0, Pauli::Y)]).unwrap(),
        PauliString::identity(2.0),
    ];
    if n >= 2 {
        v.push(PauliString::new(0.7, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap());
        v.push(PauliString::new(1.0, vec![(0, Pauli::X), (1, Pauli::Y)]).unwrap());
    }
    if n >= 3 {
        v.push(PauliString::new(1.5, vec![(0, Pauli::X), (1, Pauli::Y), (2, Pauli::Z)]).unwrap());
    }
    v
}

#[test]
fn fp32_expectation_and_marginals_match_cpu() {
    let mut gpu = match CudaSvBackendF32::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 readout oracle: {e}");
            return;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let mut worst = 0.0f64;
    for n in 2..=8u32 {
        let c = mixed_state(n);
        let gs = run(&mut gpu, &c).expect("gpu run");
        let cs = run(&mut cpu, &c).expect("cpu run");

        for p in test_paulis(n) {
            let g = gpu.expectation_value(&gs, &p).expect("gpu expect");
            let r = cpu.expectation_value(&cs, &p).expect("cpu expect");
            worst = worst.max((g - r).abs());
            assert!(
                (g - r).abs() <= TOL,
                "expectation n={n} {p:?}: gpu={g} cpu={r}"
            );
        }

        let subsets: Vec<Vec<u32>> = vec![vec![0], vec![n - 1], vec![0, 1], vec![1, 0, n - 1]];
        for qs in subsets {
            let mut uniq = qs.clone();
            uniq.sort_unstable();
            uniq.dedup();
            if qs.iter().any(|&q| q >= n) || uniq.len() != qs.len() {
                continue;
            }
            let g = gpu.probabilities(&gs, &qs).expect("gpu probs");
            let r = cpu.probabilities(&cs, &qs).expect("cpu probs");
            assert_eq!(g.len(), r.len(), "probs len n={n} {qs:?}");
            for (i, (a, b)) in g.iter().zip(r.iter()).enumerate() {
                worst = worst.max((a - b).abs());
                assert!(
                    (a - b).abs() <= TOL,
                    "probs n={n} {qs:?} bin {i}: gpu={a} cpu={b}"
                );
            }
        }
    }
    println!("fp32 readout worst |Δ| = {worst:.2e} (tol {TOL:.0e})");
}

/// GHZ: measuring any qubit forces all others equal. The FP32 collapse must yield
/// a consistent basis state `|bb…b⟩` (amplitude ~1), independent of the FP64 coin.
#[test]
fn fp32_measure_ghz_correlation() {
    let mut gpu = match CudaSvBackendF32::with_seed(7) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 measure: {e}");
            return;
        }
    };
    let n = 6;
    let mut gs = run(&mut gpu, &ghz(n)).expect("gpu run");
    let b0 = gpu.measure(&mut gs, 0).expect("measure q0");
    for q in 1..n {
        let bq = gpu.measure(&mut gs, q).expect("measure");
        assert_eq!(bq, b0, "GHZ correlation broken at q={q}");
    }
    // The collapsed state is the single basis vector |b0…b0⟩ with |amp|²≈1.
    let amps = gs.amplitudes_vec();
    let idx = if b0 { (1usize << n) - 1 } else { 0 };
    let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
    assert!(
        (amps[idx].norm_sqr() - 1.0).abs() < 1e-4,
        "collapsed mass off: {}",
        amps[idx].norm_sqr()
    );
    assert!((norm - 1.0).abs() < 1e-4, "post-measure norm off: {norm}");
}

/// Sampling reproduces the exact distribution (statistical) and is reproducible
/// for a fixed seed.
#[test]
fn fp32_sample_distribution() {
    let mut gpu = match CudaSvBackendF32::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fp32 sample: {e}");
            return;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let n = 4;
    let c = mixed_state(n);
    let gs = run(&mut gpu, &c).expect("gpu run");
    let cs = run(&mut cpu, &c).expect("cpu run");
    let exact = cpu
        .probabilities(&cs, &(0..n).collect::<Vec<_>>())
        .expect("cpu probs");

    let shots = 400_000u32;
    let samples = gpu.sample(&gs, shots).expect("gpu sample");
    assert_eq!(samples.len(), shots as usize);
    let dim = 1usize << n;
    let mut hist = vec![0u64; dim];
    for &s in &samples {
        assert!((s as usize) < dim, "sample {s} out of range");
        hist[s as usize] += 1;
    }
    for (i, &p) in exact.iter().enumerate() {
        let emp = hist[i] as f64 / shots as f64;
        assert!(
            (emp - p).abs() <= 0.01,
            "sample dist bin {i}: emp={emp:.4} exact={p:.4}"
        );
    }

    // Determinism: same seed ⇒ same samples.
    let mut gpu2 = CudaSvBackendF32::with_seed(0).unwrap();
    let gs2 = run(&mut gpu2, &c).expect("gpu run 2");
    let samples2 = gpu2.sample(&gs2, 1000).expect("gpu sample 2");
    assert_eq!(&samples[..1000], &samples2[..], "sampling not reproducible");
}
