use aleph_backend::{run, Backend};
use aleph_core::{Complex, Gate, GateInstance, Param, Pauli, PauliString, PauliSum};
use aleph_ir::passes::{FuseDiagonalRuns, FuseKq, Pass, RelabelQubits, TileBlock};
use aleph_ir::{Circuit, DiagonalPhase, Instruction, PhaseTerm, TiledBlock};
use aleph_sv::{
    expectation_query::{ExactExpectationQuery, ExactQueryError},
    NaiveSvBackend,
};
use proptest::prelude::*;
use smallvec::smallvec;
const BUDGET: usize = 32 * 1024 * 1024;
fn term(c: f64, qs: &[(u32, Pauli)]) -> PauliString {
    PauliString::new(c, qs.to_vec()).unwrap()
}
fn sum(terms: Vec<PauliString>) -> PauliSum {
    PauliSum { terms }
}
fn full(c: &Circuit, p: &PauliSum) -> f64 {
    let mut backend = NaiveSvBackend::with_seed(0);
    let state = run(&mut backend, c).unwrap();
    p.terms
        .iter()
        .map(|p| backend.expectation_value(&state, p).unwrap())
        .sum()
}
fn query(c: &Circuit, p: &PauliSum) -> f64 {
    ExactExpectationQuery::compile(c, p, BUDGET)
        .unwrap()
        .execute()
        .unwrap()
        .value
}
fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() < 1e-10,
        "{a} != {b}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn random_controls_fusion_tiling_and_permutation_match_full_sv(
        n in 2u32..7, ops in prop::collection::vec((0u8..8,any::<u8>(),any::<u8>(),-300i16..300),0..45),
        factors in prop::collection::vec(0u8..4,6..7)) {
        let mut c=Circuit::new(n,0);
        for (kind,a,b,angle) in ops {
            let q=u32::from(a)%n;let r=(q+1+u32::from(b)%(n-1))%n;
            let theta=f64::from(angle)/100.;
            let g=match kind {
                0=>GateInstance::new(Gate::H,smallvec![q]),
                1=>GateInstance::new(Gate::Rx(Param::Concrete(theta)),smallvec![q]),
                2=>GateInstance::new(Gate::Rz(Param::Concrete(theta)),smallvec![q]),
                3=>GateInstance::new(Gate::S,smallvec![q]),
                4=>GateInstance::new(Gate::Cnot,smallvec![q,r]),
                5=>GateInstance::new(Gate::Swap,smallvec![q,r]),
                6=>GateInstance::controlled(Gate::Y,smallvec![q],smallvec![r]),
                _=>GateInstance::controlled(Gate::Ry(Param::Concrete(theta)),smallvec![q],smallvec![r]),
            }; c.add_gate(g).unwrap();
        }
        let ps=term(-0.7,&(0..n).map(|q|(q,match factors[q as usize]{0=>Pauli::I,1=>Pauli::X,2=>Pauli::Y,_=>Pauli::Z})).collect::<Vec<_>>());
        let observable=sum(vec![ps,term(0.31,&[(n-1,Pauli::Y)]),PauliString::identity(0.2)]);
        let expected=full(&c,&observable);
        close(query(&c,&observable),expected);
        let mut optimized=c.clone();
        RelabelQubits::new(1).run(&mut optimized).unwrap();
        FuseDiagonalRuns.run(&mut optimized).unwrap();
        FuseKq{max_qubits:3}.run(&mut optimized).unwrap();
        TileBlock::new(2).run(&mut optimized).unwrap();
        close(query(&optimized,&observable),expected);
    }
}

#[test]
fn identity_zero_coefficients_and_shared_states() {
    let mut c = Circuit::new(128, 0);
    c.h(127).unwrap();
    c.s(127).unwrap();
    let p = sum(vec![
        term(2., &[(127, Pauli::Y)]),
        term(-3., &[(127, Pauli::X)]),
        PauliString::identity(4.),
        term(0., &[(0, Pauli::Z)]),
    ]);
    let q = ExactExpectationQuery::compile(&c, &p, 8192).unwrap();
    assert_eq!(q.plan().cones().len(), 1);
    assert_eq!(q.plan().cones()[0].compact_to_original(), &[127]);
    assert_eq!(q.plan().cones()[0].original_to_compact()[127], Some(0));
    assert_eq!(q.plan().cones()[0].observables().len(), 2);
    close(q.execute().unwrap().value, 6.);
    close(q.execute().unwrap().value, 6.);
    let identity =
        ExactExpectationQuery::compile(&c, &sum(vec![PauliString::identity(-2.)]), 0).unwrap();
    assert_eq!(identity.peak_dense_workspace_bytes(), 0);
    close(identity.execute().unwrap().value, -2.);
    close(
        ExactExpectationQuery::compile(&Circuit::new(0, 0), &sum(vec![]), 0)
            .unwrap()
            .execute()
            .unwrap()
            .value,
        0.,
    );
}

#[test]
fn shallow_128q_analytic_pairs_and_disjoint_cones_do_not_union() {
    let mut c = Circuit::new(128, 0);
    for q in (0..128).step_by(2) {
        c.ry(0.73, q).unwrap();
        c.cnot(q, q + 1).unwrap();
    }
    let p = sum(vec![
        term(1., &[(125, Pauli::Z)]),
        term(0.4, &[(124, Pauli::X), (125, Pauli::X)]),
        term(-0.2, &[(2, Pauli::Y), (3, Pauli::Y)]),
    ]);
    let q = ExactExpectationQuery::compile(&c, &p, 8192).unwrap();
    assert_eq!(q.plan().cones().len(), 2);
    assert!(q
        .plan()
        .cones()
        .iter()
        .all(|x| x.circuit().num_qubits() == 2));
    assert_eq!(q.peak_dense_workspace_bytes(), 4256);
    close(
        q.execute().unwrap().value,
        0.73f64.cos() + 0.6 * 0.73f64.sin(),
    );
}

#[test]
fn barrier_does_not_entangle_support_and_order_is_kept() {
    let mut c = Circuit::new(100, 0);
    c.h(99).unwrap();
    c.add_instruction(Instruction::Barrier((0..100).collect()))
        .unwrap();
    c.s(99).unwrap();
    c.x(0).unwrap();
    let p = sum(vec![term(1., &[(99, Pauli::Y)])]);
    let q = ExactExpectationQuery::compile(&c, &p, 8192).unwrap();
    let cone = &q.plan().cones()[0];
    assert_eq!(cone.circuit().num_qubits(), 1);
    assert!(
        matches!(&cone.circuit().instructions()[1],Instruction::Barrier(qs) if qs.as_slice()==[0])
    );
    assert_eq!(cone.retained_source_indices(), &[0, 1, 2]);
    close(q.execute().unwrap().value, 1.);
}

#[test]
fn reordered_controlled_manual_kq_preserves_msb_operand_contract() {
    let mut reference = Circuit::new(5, 0);
    reference.h(4).unwrap();
    reference.ry(0.41, 1).unwrap();
    reference.ry(0.8, 3).unwrap();
    let mut candidate = reference.clone();
    let matrix = match Gate::Cnot.matrix().unwrap() {
        aleph_core::GateMatrix::M4x4(m) => m,
        _ => unreachable!(),
    };
    reference
        .add_gate(GateInstance::controlled(
            Gate::Unitary2q(Box::new(matrix)),
            smallvec![3, 1],
            smallvec![4],
        ))
        .unwrap();
    candidate
        .add_gate(GateInstance::controlled(
            Gate::UnitaryKq {
                k: 2,
                data: matrix.into_iter().flatten().collect(),
            },
            smallvec![3, 1],
            smallvec![4],
        ))
        .unwrap();
    for a in [Pauli::X, Pauli::Y, Pauli::Z] {
        for b in [Pauli::X, Pauli::Y, Pauli::Z] {
            let p = sum(vec![term(1., &[(1, a), (3, b), (4, Pauli::X)])]);
            close(query(&candidate, &p), full(&reference, &p));
        }
    }
}

#[test]
fn manual_diagonal_high_bits_and_tiled_controls() {
    let mut c = Circuit::new(100, 0);
    c.h(63).unwrap();
    c.add_instruction(Instruction::DiagonalPhase(Box::new(DiagonalPhase {
        n_qubits: 64,
        terms: vec![PhaseTerm {
            conds: smallvec![1u64 << 63],
            angle: 0.39,
        }],
    })))
    .unwrap();
    close(
        query(&c, &sum(vec![term(1., &[(63, Pauli::X)])])),
        0.39f64.cos(),
    );
    let mut c = Circuit::new(6, 0);
    c.h(5).unwrap();
    let mut flat = c.clone();
    let gates = vec![
        GateInstance::controlled(Gate::X, smallvec![0], smallvec![5]),
        GateInstance::new(Gate::S, smallvec![0]),
    ];
    for g in &gates {
        flat.add_gate(g.clone()).unwrap();
    }
    c.add_instruction(Instruction::TiledBlock(Box::new(TiledBlock {
        tile_bits: 1,
        gates,
    })))
    .unwrap();
    let p = sum(vec![term(1., &[(0, Pauli::Y), (5, Pauli::X)])]);
    close(query(&c, &p), full(&flat, &p));
}

#[test]
fn nontrivial_relabeling_exposes_logical_maps() {
    let mut c = Circuit::new(12, 0);
    for _ in 0..4 {
        c.h(11).unwrap();
        c.cnot(11, 9).unwrap();
        c.rz(0.4, 9).unwrap();
    }
    let original = c.clone();
    RelabelQubits::new(2).run(&mut c).unwrap();
    assert!(c.qubit_permutation().is_some());
    let p = sum(vec![term(1., &[(9, Pauli::Y), (11, Pauli::X)])]);
    let q = ExactExpectationQuery::compile(&c, &p, 8192).unwrap();
    let cone = &q.plan().cones()[0];
    for (compact, &original) in cone.compact_to_original().iter().enumerate() {
        assert_eq!(
            cone.original_to_compact()[original as usize],
            Some(compact as u32)
        );
    }
    close(q.execute().unwrap().value, full(&original, &p));
}

#[test]
fn rejects_all_nonunitary_input_even_when_disjoint_or_analytic() {
    for instruction in [
        Instruction::Reset(0),
        Instruction::Measure { qubit: 0, clbit: 0 },
    ] {
        let mut c = Circuit::new(3, 1);
        c.add_instruction(instruction).unwrap();
        for p in [
            sum(vec![term(1., &[(2, Pauli::Z)])]),
            sum(vec![PauliString::identity(1.)]),
            sum(vec![]),
        ] {
            assert!(ExactExpectationQuery::compile(&c, &p, BUDGET).is_err());
        }
    }
    let mut c = Circuit::new(3, 0);
    c.add_gate(GateInstance::new(
        Gate::Unitary1q(Box::new([
            [Complex::new(2., 0.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::new(2., 0.)],
        ])),
        smallvec![0],
    ))
    .unwrap();
    assert!(
        ExactExpectationQuery::compile(&c, &sum(vec![term(1., &[(2, Pauli::Z)])]), BUDGET).is_err()
    );
}

#[test]
fn rejects_invalid_manual_ir_and_mutated_paulis() {
    let c = Circuit::new(4, 0);
    for p in [
        PauliString {
            coefficient: f64::NAN,
            terms: vec![],
        },
        PauliString {
            coefficient: 1.,
            terms: vec![(0, Pauli::X), (0, Pauli::Y)],
        },
        term(1., &[(4, Pauli::Z)]),
    ] {
        assert!(ExactExpectationQuery::compile(&c, &sum(vec![p]), BUDGET).is_err());
    }
    for data in [
        vec![Complex::new(1., 0.); 15],
        vec![Complex::new(f64::NAN, 0.); 16],
        vec![Complex::new(2., 0.); 16],
    ] {
        let mut bad = c.clone();
        bad.add_gate(GateInstance::new(
            Gate::UnitaryKq {
                k: 2,
                data: data.into(),
            },
            smallvec![0, 1],
        ))
        .unwrap();
        assert!(ExactExpectationQuery::compile(
            &bad,
            &sum(vec![PauliString::identity(1.)]),
            BUDGET
        )
        .is_err());
    }
    let mut bad = c.clone();
    bad.add_instruction(Instruction::DiagonalPhase(Box::new(DiagonalPhase {
        n_qubits: 4,
        terms: vec![PhaseTerm {
            conds: smallvec![1 << 5],
            angle: 0.2,
        }],
    })))
    .unwrap();
    assert!(ExactExpectationQuery::compile(&bad, &sum(vec![]), BUDGET).is_err());
}

#[test]
fn global_and_overflow_cones_reject_before_state_allocation() {
    for n in [20, 100] {
        let mut c = Circuit::new(n, 0);
        c.h(0).unwrap();
        for q in 0..n - 1 {
            c.cnot(q, q + 1).unwrap();
        }
        let error =
            ExactExpectationQuery::compile(&c, &sum(vec![term(1., &[(n - 1, Pauli::Z)])]), 8192)
                .unwrap_err();
        assert!(matches!(error,ExactQueryError::Memory{active_qubits,..} if active_qubits==n));
    }
}

#[path = "../examples/support/cone_workloads.rs"]
mod workloads;

#[test]
fn p3_10_shallow_128q_matches_independent_window_reference() {
    let c = workloads::brickwork(128, 3, 0);
    let p = workloads::observable(64);
    let q = ExactExpectationQuery::compile(&c, &p, 1 << 20).unwrap();
    assert!(q.plan().cones()[0].circuit().num_qubits() <= 8);
    // Independent spatial crop, built directly (no query planner): six sites
    // on either side of the central site exceed a depth-three NN light cone.
    let reference = workloads::brickwork(13, 3, 58);
    let expected = full(&reference, &workloads::observable(6));
    assert!(expected.abs() > 0.01);
    close(q.execute().unwrap().value, expected);
    let global = workloads::global(8);
    let global_query =
        ExactExpectationQuery::compile(&global, &workloads::observable(7), BUDGET).unwrap();
    assert_eq!(global_query.plan().cones()[0].circuit().num_qubits(), 8);
    close(global_query.execute().unwrap().value, 0.);
}

#[test]
fn aer_fixture_amplitudes_independently_contract_pauli_expectations() {
    for name in [
        "qft_5",
        "random_nonclifford_n4_d20",
        "multi_ctrl_ccx_permuted",
        "awkward_angles",
        "bell_phi_plus",
    ] {
        let fixture = aleph_oracle::load_fixture(&aleph_oracle::workspace_path(&format!(
            "oracle/fixtures/{name}.json"
        )))
        .unwrap();
        let source = aleph_oracle::load_qasm(&aleph_oracle::workspace_path(&format!(
            "oracle/{}",
            fixture.qasm_path
        )))
        .unwrap();
        let circuit = aleph_parser::parse(&source).unwrap();
        let amplitudes: Vec<Complex> = fixture
            .statevector
            .amplitudes
            .iter()
            .map(|&(re, im)| Complex::new(re, im))
            .collect();
        for axis in [Pauli::X, Pauli::Y, Pauli::Z] {
            let p = term(-0.71, &[(fixture.num_qubits - 1, axis)]);
            let mut expected = Complex::new(0., 0.);
            for (i, &a) in amplitudes.iter().enumerate() {
                for (j, &b) in amplitudes.iter().enumerate() {
                    let mut element = Complex::new(1., 0.);
                    for q in 0..fixture.num_qubits {
                        let factor = if q == fixture.num_qubits - 1 {
                            axis
                        } else {
                            Pauli::I
                        };
                        element *= factor.matrix()[(i >> q) & 1][(j >> q) & 1];
                    }
                    expected += a.conj() * element * b;
                }
            }
            close(query(&circuit, &sum(vec![p])), -0.71 * expected.re);
        }
    }
}

#[test]
fn finite_parameters_diagonal_scope_and_sum_overflow_are_guarded() {
    for (scope, angle) in [(65, 0.1), (4, f64::NAN), (4, f64::INFINITY)] {
        let mut c = Circuit::new(4, 0);
        c.add_instruction(Instruction::DiagonalPhase(Box::new(DiagonalPhase {
            n_qubits: scope,
            terms: vec![PhaseTerm {
                conds: smallvec![1],
                angle,
            }],
        })))
        .unwrap();
        assert!(
            ExactExpectationQuery::compile(&c, &sum(vec![PauliString::identity(1.)]), 0).is_err()
        );
    }
    let mut c = Circuit::new(4, 0);
    c.rx(f64::NAN, 0).unwrap();
    assert!(
        ExactExpectationQuery::compile(&c, &sum(vec![term(1., &[(3, Pauli::Z)])]), BUDGET).is_err()
    );
    let q = ExactExpectationQuery::compile(
        &Circuit::new(0, 0),
        &sum(vec![
            PauliString::identity(f64::MAX),
            PauliString::identity(f64::MAX),
        ]),
        0,
    )
    .unwrap();
    assert!(matches!(q.execute(), Err(ExactQueryError::NonFiniteResult)));
}
