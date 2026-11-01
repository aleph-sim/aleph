//! Public single-shot sampling distributions and noisy worker-count invariance.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance};
use aleph_oracle::assert_distribution_close;
use aleph_parser::parse;
use aleph_sv::noise::{
    amplitude_damping_error, depolarizing_error, run_noisy, NoiseModel, ReadoutError,
};
use aleph_sv::{Fp32SvBackend, NaiveSvBackend, SoaSvBackend};

fn check<B: Backend>(mut backend: B) {
    let mut state = backend.allocate(3).unwrap();
    // Nonuniform analytic product distribution, including an exactly zero qubit.
    for (q, angle) in [(0, 0.73_f64), (1, 1.19)] {
        backend
            .apply_gate(
                &mut state,
                &GateInstance::new(Gate::Ry(aleph_core::Param::Concrete(angle)), &[q][..]),
            )
            .unwrap();
    }
    let (a, b) = (
        (0.73_f64 / 2.0).sin().powi(2),
        (1.19_f64 / 2.0).sin().powi(2),
    );
    let exact = vec![
        (1.0 - a) * (1.0 - b),
        a * (1.0 - b),
        (1.0 - a) * b,
        a * b,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    const SHOTS: u32 = 100_000;
    let mut counts = vec![0_u64; 8];
    for _ in 0..SHOTS {
        counts[backend.sample(&state, 1).unwrap()[0] as usize] += 1;
    }
    assert_distribution_close("single_shot_product", 3, &counts, &exact, SHOTS);
    assert!(backend.sample(&state, 0).unwrap().is_empty());
    let mut alias_counts = vec![0_u64; 8];
    for i in backend.sample(&state, SHOTS).unwrap() {
        alias_counts[i as usize] += 1;
    }
    assert_distribution_close("alias_product", 3, &alias_counts, &exact, SHOTS);
}

#[test]
fn single_and_many_draws_match_analytic_distribution() {
    check(NaiveSvBackend::with_seed(556));
    check(SoaSvBackend::with_seed(556));
    check(Fp32SvBackend::with_seed(556));
}

#[test]
fn noise_with_pauli_kraus_and_readout_is_worker_invariant() {
    let circuit = parse(
        "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; h q[0]; cx q[0],q[1]; ry(0.37) q[1];",
    )
    .unwrap();
    let mut model = NoiseModel::new();
    model.add_all_qubit_quantum_error(depolarizing_error(0.09, 1), &["H"]);
    model.add_all_qubit_quantum_error(amplitude_damping_error(0.13), &["Ry"]);
    model.add_readout_error(ReadoutError::new([[0.97, 0.03], [0.11, 0.89]]), 0);
    let mut reference = None;
    for workers in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let counts = pool.install(|| run_noisy(&circuit, &model, 4096, 556).unwrap());
        assert_eq!(counts.iter().sum::<u64>(), 4096);
        if let Some(ref previous) = reference {
            assert_eq!(previous, &counts);
        }
        reference = Some(counts);
    }
}
