//! Final-state optimization must retain unmeasured preparation and restore order.
use aleph_backend::{
    run, run_optimized_cpu_final_state_with_stats, run_optimized_final_state_with_stats,
};
use aleph_ir::Circuit;
use aleph_sv::{Fp32SvBackend, NaiveSvBackend};

fn compare(c: &Circuit, seed: u64) {
    let raw = run(&mut NaiveSvBackend::with_seed(seed), c).unwrap();
    let (opt, _) =
        run_optimized_final_state_with_stats(&mut NaiveSvBackend::with_seed(seed), c).unwrap();
    let (f32, _) =
        run_optimized_cpu_final_state_with_stats(&mut Fp32SvBackend::with_seed(seed), c).unwrap();
    let (cpu, _) =
        run_optimized_cpu_final_state_with_stats(&mut NaiveSvBackend::with_seed(seed), c).unwrap();
    for (r, o) in raw.amplitudes().iter().zip(cpu.amplitudes()) {
        assert!((*r - *o).norm() < 1e-11);
    }
    for ((r, o), f) in raw
        .amplitudes()
        .iter()
        .zip(opt.amplitudes())
        .zip(f32.to_aos_f64())
    {
        assert!((*r - *o).norm() < 1e-11);
        assert!((*r - f).norm() < 2e-5);
    }
}

#[test]
fn partial_measurement_retains_unmeasured_and_post_measurement_gates() {
    let c = aleph_parser::parse("OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; bit[1] c; x q[1]; h q[0]; s q[0]; barrier q; measure q[0] -> c[0]; h q[2]; s q[2];").unwrap();
    for seed in 0..16 {
        compare(&c, seed);
    }
}

#[test]
fn relabelling_and_fp32_tile_replay_preserve_complete_amplitudes() {
    let mut c = Circuit::new(16, 1);
    c.x(0).unwrap();
    for _ in 0..8 {
        c.h(15).unwrap();
        c.t(15).unwrap();
        c.cnot(15, 14).unwrap();
        c.s(14).unwrap();
        c.h(14).unwrap();
    }
    let mut planned = c.clone();
    aleph_ir::passes::PassPipeline::final_state_pipeline()
        .run(&mut planned)
        .unwrap();
    assert!(
        planned.qubit_permutation().is_some(),
        "fixture must exercise restoration"
    );
    compare(&c, 3);
}

#[test]
fn diagonal_and_dense_fusion_are_executed() {
    let mut c = Circuit::new(4, 0);
    c.h(0).unwrap();
    c.s(0).unwrap();
    c.t(0).unwrap();
    c.s(0).unwrap();
    c.cnot(0, 1).unwrap();
    c.h(1).unwrap();
    c.cnot(1, 2).unwrap();
    let (_, stats) =
        run_optimized_final_state_with_stats(&mut NaiveSvBackend::with_seed(1), &c).unwrap();
    assert!(stats.planned_instructions < stats.input_instructions);
    assert!(stats.diagonal_phases + stats.fused_gates + stats.tiled_blocks > 0);
    compare(&c, 1);
}

#[test]
fn external_controls_remain_fences_and_tile_replay_preserves_them() {
    let mut c = Circuit::new(6, 0);
    c.h(5).unwrap();
    c.t(5).unwrap();
    c.h(0).unwrap();
    c.add_gate(aleph_core::GateInstance::controlled(
        aleph_core::Gate::X,
        smallvec::smallvec![2],
        smallvec::smallvec![5],
    ))
    .unwrap();
    c.add_gate(aleph_core::GateInstance::controlled(
        aleph_core::Gate::S,
        smallvec::smallvec![0],
        smallvec::smallvec![5],
    ))
    .unwrap();
    c.h(1).unwrap();
    c.cnot(0, 1).unwrap();
    compare(&c, 4);
}

mod property {
    use super::*;
    use aleph_test::circuit::arb_circuit_emittable;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn random_final_states_match(raw in arb_circuit_emittable(5, 2, 30)) {
            let mut c = Circuit::new(raw.num_qubits(), raw.num_clbits());
            for inst in raw.instructions() {
                if !matches!(inst, aleph_ir::Instruction::Reset(_)) {
                    c.add_instruction(inst.clone()).unwrap();
                }
            }
            compare(&c, 7);
        }
    }
}

#[test]
#[ignore = "offline n=20 full-state FP32 rounding check"]
fn brickwall_fp32_distribution_matches_fp64_reference() {
    let n = 20;
    let mut source = format!("OPENQASM 3.0; include \"stdgates.inc\"; qubit[{n}] q;\n");
    for layer in 0..12 {
        for q in 0..n {
            let angle = (((layer + 1) * (q + 1)) as f64).sin();
            source.push_str(&format!(
                "rz({angle}) q[{q}]; rx({}) q[{q}];\n",
                angle * 0.7 + 0.13
            ));
        }
        for q in (layer % 2..n - 1).step_by(2) {
            source.push_str(&format!("cx q[{q}],q[{}];\n", q + 1));
        }
    }
    let c = aleph_parser::parse(&source).unwrap();
    let exact = run(&mut NaiveSvBackend::with_seed(7), &c).unwrap();
    let raw = run(&mut Fp32SvBackend::with_seed(7), &c).unwrap();
    let (opt, _) =
        run_optimized_cpu_final_state_with_stats(&mut Fp32SvBackend::with_seed(7), &c).unwrap();
    let mut raw_l1 = 0.0;
    let mut opt_l1 = 0.0;
    let mut max_error = 0.0_f64;
    for ((r, f), o) in exact
        .amplitudes()
        .iter()
        .zip(raw.amplitudes())
        .zip(opt.amplitudes())
    {
        let f = aleph_core::Complex::new(f.re as f64, f.im as f64);
        let o = aleph_core::Complex::new(o.re as f64, o.im as f64);
        raw_l1 += (r.norm_sqr() - f.norm_sqr()).abs();
        opt_l1 += (r.norm_sqr() - o.norm_sqr()).abs();
        max_error = max_error.max((*r - o).norm());
    }
    println!("brickwall n=20: raw_fp32_probability_l1={raw_l1:.9e}, optimized_fp32_probability_l1={opt_l1:.9e}, optimized_max_amplitude_error={max_error:.9e}");
    assert!(opt_l1 < 2e-4 && max_error < 2e-5);
}
