//! Full-final-state CUDA plan oracles, including DCE, precision and batching guards.
#![cfg(all(target_os = "linux", feature = "cuda"))]
use aleph_backend::{run, Backend};
use aleph_core::{Complex, Gate, GateInstance};
use aleph_cuda::{
    CudaExecutionOptions as Options, CudaExecutionPlan as Plan, CudaSvBackend, CudaSvBackendF32,
};
use aleph_ir::{Circuit, Instruction};

#[test]
fn pure_planning_contracts() {
    for n in [33, 64] {
        assert!(Plan::compile(&Circuit::new(n, 0), Options::default()).is_err());
    }
    assert!(Plan::compile(&Circuit::new(32, 0), Options::default()).is_ok());
    let mut c = Circuit::new(4, 1);
    c.h(0).unwrap();
    c.measure(0, 0).unwrap();
    c.x(3).unwrap();
    let plan = Plan::compile(&c, Options::default()).unwrap();
    assert!(
        plan.circuit()
            .instructions()
            .iter()
            .any(|i| i.used_qubits().contains(&3)),
        "measurement DCE changed final state"
    );
    assert_eq!(
        plan.circuit()
            .instructions()
            .iter()
            .filter(|i| matches!(i, Instruction::Measure { .. }))
            .count(),
        1
    );
    assert_eq!(plan.fusion_width, 3);
    let raw = Plan::compile(&c, Options { optimize: false }).unwrap();
    assert_eq!(raw.circuit().len(), c.len());
    c.reset(0).unwrap();
    assert!(Plan::compile(&c, Options::default()).is_err());
    let mut bad = Circuit::new(3, 0);
    bad.add_gate(GateInstance::new(
        Gate::UnitaryKq {
            k: 3,
            data: vec![Complex::new(1., 0.)].into_boxed_slice(),
        },
        vec![0, 1, 2],
    ))
    .unwrap();
    assert!(Plan::compile(&bad, Options::default()).is_err());
    let mut dense = Circuit::new(8, 0);
    for _ in 0..3 {
        for q in 0..8 {
            dense.ry(0.3, q).unwrap();
        }
        for q in 0..7 {
            dense.cnot(q, q + 1).unwrap();
        }
    }
    let fused = Plan::compile(&dense, Options::default()).unwrap();
    assert!(fused.instruction_kinds().2 > 0);
    assert!(fused.circuit().len() < dense.len());
    let mut relabeled = Circuit::new(4, 0);
    relabeled.h(2).unwrap();
    aleph_ir::passes::Pass::run(&aleph_ir::passes::RelabelQubits::new(1), &mut relabeled).unwrap();
    assert!(Plan::compile(&relabeled, Options::default()).is_err());
}

#[test]
fn plans_match_full_cpu_state_and_preserve_collapse() {
    if !aleph_cuda::device_available() {
        return;
    }
    for seed in 0..12 {
        let mut c = Circuit::new(7, 1);
        for q in 0..7 {
            c.ry(0.13 * (q + seed + 1) as f64, q).unwrap();
            c.rz(0.27 * (q + 1) as f64, q).unwrap();
        }
        for q in 0..6 {
            c.cnot(q, q + 1).unwrap();
        }
        c.add_instruction(Instruction::Barrier(vec![0, 1, 2].into()))
            .unwrap();
        if seed % 2 == 0 {
            c.x(0).unwrap();
            c.x(0).unwrap();
        }
        c.add_gate(GateInstance::controlled(Gate::X, vec![5], vec![0, 2]))
            .unwrap();
        for q in 0..7 {
            c.rx(0.73, q).unwrap();
        }
        let mut cpu = aleph_sv::NaiveSvBackend::with_seed(42);
        let expected = run(&mut cpu, &c).unwrap();
        let plan = Plan::compile(&c, Options::default()).unwrap();
        macro_rules! check {
            ($backend:ty,$tol:expr) => {{
                let mut be = <$backend>::with_seed(42).unwrap();
                for _ in 0..2 {
                    let actual = plan.execute(&mut be).unwrap().amplitudes_vec();
                    for (a, b) in actual.iter().zip(expected.amplitudes()) {
                        assert!((*a - *b).norm() < $tol, "seed={seed}");
                    }
                }
            }};
        }
        check!(CudaSvBackend, 1e-10);
        check!(CudaSvBackendF32, 1e-5);
    }
    // Definite measured qubit plus unrelated final X: DCE would erase X.
    let mut c = Circuit::new(3, 1);
    c.h(1).unwrap();
    c.measure(0, 0).unwrap();
    c.x(2).unwrap();
    let plan = Plan::compile(&c, Options::default()).unwrap();
    let mut be = CudaSvBackend::with_seed(0).unwrap();
    let state = plan.execute(&mut be).unwrap();
    assert!(be
        .sample(&state, 100)
        .unwrap()
        .iter()
        .all(|&i| i == 4 || i == 6));
}

#[test]
fn plan_executes_fusion_layers_without_inheriting_tf32() {
    if !aleph_cuda::device_available() {
        return;
    }
    let mut c = Circuit::new(8, 0);
    for q in 0..8 {
        c.h(q).unwrap();
    }
    c.add_instruction(Instruction::Barrier((0..8).collect()))
        .unwrap();
    for _ in 0..4 {
        for q in 0..7 {
            c.ry(0.13, q).unwrap();
            c.cnot(q, q + 1).unwrap();
        }
    }
    let plan = Plan::compile(&c, Options::default()).unwrap();
    assert!(plan.instruction_kinds().2 > 0);
    let mut be = CudaSvBackendF32::with_seed(1).unwrap();
    let before = aleph_cuda::execution_counters();
    plan.execute(&mut be).unwrap();
    let after = aleph_cuda::execution_counters();
    assert!(after.0 > before.0);
    assert!(after.1 > before.1);
    assert_eq!(after.2, before.2);
    // Repeat strict planning on a backend whose optional TF32 module may exist.
    let strict = Plan::compile(&c, Options::default()).unwrap();
    assert_eq!(strict.fusion_width, 3);
    let before = aleph_cuda::execution_counters();
    let actual = strict.execute(&mut be).unwrap().amplitudes_vec();
    let after = aleph_cuda::execution_counters();
    assert_eq!(after.2, before.2);
    // A caller can supply legacy width-4 IR. Strict policy must disable TF32
    // even for raw execution, rather than merely relying on a width-3 optimizer.
    let manual = aleph_cuda::fuse_for_gpu_tf32(&c);
    let raw = Plan::compile(&manual, Options { optimize: false }).unwrap();
    let before = aleph_cuda::execution_counters();
    let manual_actual = raw.execute(&mut be).unwrap().amplitudes_vec();
    let after = aleph_cuda::execution_counters();
    assert_eq!(after.2, before.2);
    let expected = run(&mut aleph_sv::NaiveSvBackend::new(), &c).unwrap();
    for (a, b) in actual.iter().zip(expected.amplitudes()) {
        assert!((*a - *b).norm() < 1e-5);
    }
    for (a, b) in manual_actual.iter().zip(expected.amplitudes()) {
        assert!((*a - *b).norm() < 1e-5);
    }
}
