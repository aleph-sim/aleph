//! Public fused-IR boundary tests against a dense basis-index oracle.
use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, Gate, GateInstance};
use aleph_sv::{Fp32SvBackend, NaiveSvBackend, SoaSvBackend};

fn matrix(k: u8) -> Vec<Complex> {
    let d = 1usize << k;
    (0..d * d)
        .map(|i| {
            let (r, c) = (i / d, i % d);
            Complex::from_polar(
                1.0 / (d as f64).sqrt(),
                std::f64::consts::TAU * (r * c) as f64 / d as f64 + 0.137 * r as f64,
            )
        })
        .collect()
}
fn prepare<B: Backend>(backend: &mut B, n: u32) -> B::State {
    let mut state = backend.allocate(n).unwrap();
    for q in 0..n {
        backend
            .apply_gate(
                &mut state,
                &GateInstance::new(Gate::Ry((0.173 * (q + 1) as f64).into()), &[q][..]),
            )
            .unwrap();
        backend
            .apply_gate(
                &mut state,
                &GateInstance::new(Gate::Rz((0.319 * (q + 1) as f64).into()), &[q][..]),
            )
            .unwrap();
    }
    for q in 1..n {
        backend
            .apply_gate(&mut state, &GateInstance::new(Gate::Cnot, &[q - 1, q][..]))
            .unwrap();
    }
    state
}
// Scatter each input basis amplitude through the matrix. No fixed-bit expansion,
// sorted offsets, backend application or kernel helpers are used by the oracle.
fn oracle(input: &[Complex], targets: &[u32], data: &[Complex]) -> Vec<Complex> {
    let d = 1usize << targets.len();
    let mask = targets.iter().fold(0usize, |mask, &q| mask | (1usize << q));
    let mut output = vec![Complex::new(0.0, 0.0); input.len()];
    for (basis, &amplitude) in input.iter().enumerate() {
        let column = targets
            .iter()
            .fold(0usize, |index, &q| (index << 1) | ((basis >> q) & 1));
        for row in 0..d {
            let mut destination = basis & !mask;
            for (operand, &q) in targets.iter().enumerate() {
                destination |= ((row >> (targets.len() - 1 - operand)) & 1) << q;
            }
            output[destination] += data[row * d + column] * amplitude;
        }
    }
    output
}
fn check<B: Backend>(mut backend: B, read: impl Fn(&B::State) -> Vec<Complex>, tolerance: f64) {
    for k in 2..=5u8 {
        for n in [k as u32, 9] {
            // scalar-sized states and real SIMD-sized states
            let targets: Vec<u32> = (0..k as u32)
                .map(|i| if i % 2 == 0 { n - 1 - i / 2 } else { i / 2 })
                .collect();
            let mut sorted = targets.clone();
            sorted.sort_unstable();
            let mut reverse = sorted.clone();
            reverse.reverse();
            for qubits in [sorted, reverse, targets] {
                let data = matrix(k);
                let mut state = prepare(&mut backend, n);
                let before = read(&state);
                let expected = oracle(&before, &qubits, &data);
                let gate = GateInstance::new(
                    Gate::UnitaryKq {
                        k,
                        data: data.into_boxed_slice(),
                    },
                    &qubits[..],
                );
                backend.apply_gate(&mut state, &gate).unwrap();
                let actual = read(&state);
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (*a - *b).norm())
                    .fold(0.0, f64::max);
                assert!(
                    error < tolerance,
                    "k={k}, n={n}, targets={qubits:?}, error={error}"
                );
                let norm: f64 = actual.iter().map(|a| a.norm_sqr()).sum();
                assert!((norm - 1.0).abs() < tolerance * 10.0);
            }
        }
    }
}
fn reject<B: Backend>(mut backend: B, read: impl Fn(&B::State) -> Vec<Complex>) {
    let mut valid = vec![Complex::new(0.0, 0.0); 16];
    for i in 0..4 {
        valid[i * 4 + i] = Complex::new(1.0, 0.0);
    }
    let mut nan = valid.clone();
    nan[0].re = f64::NAN;
    let mut infinite = valid.clone();
    infinite[0].im = f64::INFINITY;
    let invalid = vec![
        (0, vec![]),
        (1, vec![Complex::new(1.0, 0.0); 4]),
        (6, vec![]),
        (2, vec![]),
        (2, vec![Complex::new(0.0, 0.0); 17]),
        (2, vec![Complex::new(0.0, 0.0); 16]),
        (2, nan),
        (2, infinite),
        (2, vec![Complex::new(1e308, 0.0); 16]),
    ];
    for (k, data) in invalid {
        let mut state = prepare(&mut backend, 9);
        let before = read(&state);
        let qubits: Vec<u32> = (0..k as u32).collect();
        let gate = GateInstance::new(
            Gate::UnitaryKq {
                k,
                data: data.into_boxed_slice(),
            },
            &qubits[..],
        );
        assert!(backend.apply_gate(&mut state, &gate).is_err());
        assert_eq!(read(&state), before, "error must leave state unchanged");
    }
    for controls in [vec![2], vec![2, 3]] {
        let mut state = prepare(&mut backend, 4);
        let before = read(&state);
        let gate = GateInstance::controlled(
            Gate::UnitaryKq {
                k: 2,
                data: valid.clone().into_boxed_slice(),
            },
            &[0, 1][..],
            &controls[..],
        );
        assert!(matches!(
            backend.apply_gate(&mut state, &gate),
            Err(BackendError::UnsupportedGate { .. })
        ));
        assert_eq!(read(&state), before);
    }
    // Same allocation/address, mutated after a successful call: pointer-based
    // validation caches would incorrectly trust the now-invalid matrix.
    let mut state = prepare(&mut backend, 3);
    let mut gate = GateInstance::new(
        Gate::UnitaryKq {
            k: 2,
            data: valid.into_boxed_slice(),
        },
        &[0, 1][..],
    );
    backend.apply_gate(&mut state, &gate).unwrap();
    let before = read(&state);
    gate.controls.push(2);
    assert!(matches!(
        backend.apply_gate(&mut state, &gate),
        Err(BackendError::UnsupportedGate { .. })
    ));
    assert_eq!(read(&state), before);
    gate.controls.clear();
    if let Gate::UnitaryKq { data, .. } = &mut gate.gate {
        data[0] = Complex::new(2.0, 0.0);
    }
    assert!(matches!(
        backend.apply_gate(&mut state, &gate),
        Err(BackendError::NonUnitaryMatrix { .. })
    ));
    assert_eq!(read(&state), before);
}
#[test]
fn aos_matches_dense_oracle() {
    check(
        NaiveSvBackend::with_seed(573),
        |s| s.amplitudes().to_vec(),
        1e-10,
    );
}
#[test]
fn soa_matches_dense_oracle() {
    check(SoaSvBackend::with_seed(573), |s| s.to_aos(), 1e-10);
}
#[test]
fn fp32_matches_dense_oracle() {
    check(Fp32SvBackend::with_seed(573), |s| s.to_aos_f64(), 2e-6);
}
#[test]
fn aos_rejects_invalid_inputs_without_mutation() {
    reject(NaiveSvBackend::with_seed(573), |s| s.amplitudes().to_vec());
}
#[test]
fn soa_rejects_invalid_inputs_without_mutation() {
    reject(SoaSvBackend::with_seed(573), |s| s.to_aos());
}
#[test]
fn fp32_rejects_invalid_inputs_without_mutation() {
    reject(Fp32SvBackend::with_seed(573), |s| s.to_aos_f64());
}

fn generated<B: Backend>(mut backend: B, read: impl Fn(&B::State) -> Vec<Complex>, tolerance: f64) {
    use aleph_ir::{
        passes::{FuseKq, Pass},
        Circuit, Instruction,
    };
    for k in [3, 5] {
        let targets: Vec<u32> = (0..k)
            .map(|i| if i % 2 == 0 { 8 - i / 2 } else { i / 2 })
            .collect();
        let mut circuit = Circuit::new(9, 0);
        for (i, &q) in targets.iter().enumerate() {
            circuit
                .add_gate(GateInstance::new(
                    Gate::Ry((0.271 * (i + 1) as f64).into()),
                    &[q][..],
                ))
                .unwrap();
            circuit
                .add_gate(GateInstance::new(
                    Gate::Rz((0.137 * (i + 1) as f64).into()),
                    &[q][..],
                ))
                .unwrap();
        }
        for pair in targets.windows(2) {
            circuit
                .add_gate(GateInstance::new(Gate::Cnot, pair))
                .unwrap();
        }
        let reference = read(&aleph_backend::run(&mut backend, &circuit).unwrap());
        FuseKq {
            max_qubits: k as usize,
        }
        .run(&mut circuit)
        .unwrap();
        assert!(circuit
            .instructions()
            .iter()
            .any(|i| matches!(i,Instruction::Gate(g) if matches!(g.gate,Gate::UnitaryKq { .. }))));
        let actual = read(&aleph_backend::run(&mut backend, &circuit).unwrap());
        let error = actual
            .iter()
            .zip(&reference)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0, f64::max);
        assert!(error < tolerance, "generated k={k}, error={error}");
    }
}
#[test]
fn generated_fused_aos_matches_raw() {
    generated(
        NaiveSvBackend::with_seed(573),
        |s| s.amplitudes().to_vec(),
        1e-10,
    );
}
#[test]
fn generated_fused_soa_matches_raw() {
    generated(SoaSvBackend::with_seed(573), |s| s.to_aos(), 1e-10);
}
#[test]
fn generated_fused_fp32_matches_raw() {
    generated(Fp32SvBackend::with_seed(573), |s| s.to_aos_f64(), 2e-6);
}
