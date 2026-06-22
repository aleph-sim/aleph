//! Readout oracle tests (P5-05): the GPU-resident `measure` / `sample` /
//! `expectation_value` / `probabilities` must match the CPU `NaiveSvBackend`.
//! Both legs are FP64, so expectation/probability tolerances are tight (1e-9);
//! sampling is checked statistically against the exact distribution.
//!
//! Generic over the backend so the same assertions pin **both** GPU backends
//! (hand-written `CudaSvBackend` and, under `cuquantum`, `CuStateVecBackend`),
//! which share the readout kernels. Skips cleanly without a CUDA device.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::{run, Backend};
use aleph_core::{Pauli, PauliString};
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;

const TOL: f64 = 1e-9;

/// A non-trivial state with X/Y/Z structure (distinct per-qubit rotations +
/// entanglement) so expectation/marginals exercise real amplitudes.
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

fn expectation_and_marginals<B, F>(mut mk: F)
where
    B: Backend,
    B::State: HasAmplitudes,
    F: FnMut() -> B,
{
    let mut gpu = mk();
    let mut cpu = NaiveSvBackend::with_seed(0);
    for n in 2..=8u32 {
        let c = mixed_state(n);
        let gs = run(&mut gpu, &c).expect("gpu run");
        let cs = run(&mut cpu, &c).expect("cpu run");

        for p in test_paulis(n) {
            let g = gpu.expectation_value(&gs, &p).expect("gpu expect");
            let r = cpu.expectation_value(&cs, &p).expect("cpu expect");
            assert!(
                (g - r).abs() <= TOL,
                "expectation n={n} {p:?}: gpu={g} cpu={r}"
            );
        }

        // Marginals over a few qubit subsets (order matters for the bin layout).
        let subsets: Vec<Vec<u32>> = vec![vec![0], vec![n - 1], vec![0, 1], vec![1, 0, n - 1]];
        for qs in subsets {
            // Skip subsets that are out of range or have duplicate qubits for
            // this n (e.g. [1,0,n-1] collapses when n=2).
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
                assert!(
                    (a - b).abs() <= TOL,
                    "probs n={n} {qs:?} bin {i}: gpu={a} cpu={b}"
                );
            }
        }
    }
}

fn measure_matches<B, F>(mut mk: F)
where
    B: Backend,
    B::State: HasAmplitudes,
    F: FnMut() -> B,
{
    // Same seed ⇒ identical branch decision (both do one `rng.gen::<f64>() < p1`
    // over matching probabilities), and the GHZ chain makes q1.. forced
    // (degenerate) after q0 — exercising the no-RNG path too.
    let mut gpu = mk();
    let mut cpu = NaiveSvBackend::with_seed(0);
    let n = 6;
    let c = ghz(n);
    let mut gs = run(&mut gpu, &c).expect("gpu run");
    let mut cs = run(&mut cpu, &c).expect("cpu run");

    for q in 0..n {
        let og = gpu.measure(&mut gs, q).expect("gpu measure");
        let oc = cpu.measure(&mut cs, q).expect("cpu measure");
        assert_eq!(og, oc, "measure outcome q={q}");
    }
    // Collapsed states must agree amplitude-for-amplitude.
    let g = HasAmplitudes::amplitudes(&gs);
    let r = HasAmplitudes::amplitudes(&cs);
    for (i, (a, b)) in g.iter().zip(r.iter()).enumerate() {
        assert!((a - b).norm() <= TOL, "collapsed amp {i}: gpu={a} cpu={b}");
    }
}

fn sample_distribution<B, F>(mut mk: F)
where
    B: Backend,
    B::State: HasAmplitudes,
    F: FnMut() -> B,
{
    let mut gpu = mk();
    let mut cpu = NaiveSvBackend::with_seed(0);
    let n = 4;
    let c = mixed_state(n);
    let gs = run(&mut gpu, &c).expect("gpu run");
    let cs = run(&mut cpu, &c).expect("cpu run");

    // Exact full-register distribution from the CPU oracle.
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
            "sample dist bin {i}: empirical={emp:.4} exact={p:.4}"
        );
    }

    // Determinism: same seed ⇒ same samples.
    let mut gpu2 = mk();
    let gs2 = run(&mut gpu2, &c).expect("gpu run 2");
    let samples2 = gpu2.sample(&gs2, 1000).expect("gpu sample 2");
    assert_eq!(&samples[..1000], &samples2[..], "sampling not reproducible");
}

#[test]
fn cuda_sv_readout_matches_cpu() {
    if CudaSvBackend::with_seed(0).is_err() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    expectation_and_marginals(|| CudaSvBackend::with_seed(0).unwrap());
    measure_matches(|| CudaSvBackend::with_seed(0).unwrap());
    sample_distribution(|| CudaSvBackend::with_seed(0).unwrap());
}

#[cfg(feature = "cuquantum")]
#[test]
fn cuquantum_readout_matches_cpu() {
    use aleph_cuda::CuStateVecBackend;
    if CuStateVecBackend::with_seed(0).is_err() {
        eprintln!("skipping: no CUDA device / cuQuantum");
        return;
    }
    expectation_and_marginals(|| CuStateVecBackend::with_seed(0).unwrap());
    measure_matches(|| CuStateVecBackend::with_seed(0).unwrap());
    sample_distribution(|| CuStateVecBackend::with_seed(0).unwrap());
}
