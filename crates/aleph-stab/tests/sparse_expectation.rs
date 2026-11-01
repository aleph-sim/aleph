use aleph_backend::{run, Backend};
use aleph_core::{Gate, GateInstance, Param, Pauli, PauliString, PauliSum};
use aleph_ir::{
    passes::{Pass, RelabelQubits, TileBlock},
    Circuit,
};
use aleph_stab::{SparsePauliError, SparsePauliOptions, SparsePauliQuery, StabilizerBackend};
use aleph_sv::NaiveSvBackend;
use proptest::prelude::*;
use smallvec::smallvec;
fn observable(ps: &[(u32, Pauli)]) -> PauliSum {
    PauliSum {
        terms: vec![PauliString::new(1., ps.to_vec()).unwrap()],
    }
}
fn sparse(c: &Circuit, p: &PauliSum, cones: bool) -> aleph_stab::SparsePauliResult {
    SparsePauliQuery::compile(
        c,
        p,
        SparsePauliOptions {
            use_cones: cones,
            ..Default::default()
        },
    )
    .unwrap()
    .execute()
    .unwrap()
}
fn dense(c: &Circuit, p: &PauliSum) -> f64 {
    let mut b = NaiveSvBackend::with_seed(0);
    let s = run(&mut b, c).unwrap();
    p.terms
        .iter()
        .map(|t| b.expectation_value(&s, t).unwrap())
        .sum()
}
fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() < 1e-10,
        "{a} != {b}"
    );
}
fn p(i: u8) -> Pauli {
    match i % 4 {
        0 => Pauli::I,
        1 => Pauli::X,
        2 => Pauli::Y,
        _ => Pauli::Z,
    }
}
proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn random_supported_circuits_phases_terms_cones_and_permutations_match_fp64(
        n in 2u32..7, ops in prop::collection::vec((0u8..19,any::<u8>(),any::<u8>(),-400i16..400),0..40),
        factors in prop::collection::vec(any::<u8>(),12..13)) {
        let mut c=Circuit::new(n,0);
        for (kind,a,b,theta) in ops {
            let q=u32::from(a)%n;let r=(q+1+u32::from(b)%(n-1))%n;let theta=Param::Concrete(f64::from(theta)/100.);
            let gate=match kind {
                0=>Gate::H,1=>Gate::X,2=>Gate::Y,3=>Gate::Z,4=>Gate::S,5=>Gate::Sdg,
                6=>Gate::T,7=>Gate::Tdg,8=>Gate::Rx(theta),9=>Gate::Ry(theta),10=>Gate::Rz(theta),11=>Gate::Phase(theta),
                12=>Gate::Cnot,13=>Gate::Cz,14=>Gate::Swap,15=>Gate::Iswap,16=>Gate::IswapDg,17=>Gate::X,_=>Gate::Y,
            };
            let g=if kind>=17 {GateInstance::controlled(gate,smallvec![q],smallvec![r])}
            else if gate.arity()==2{GateInstance::new(gate,smallvec![q,r])}else{GateInstance::new(gate,smallvec![q])};
            c.add_gate(g).unwrap();
        }
        let mut sum=PauliSum{terms:vec![PauliString::identity(-0.17)]};
        for i in 0..2 {sum.terms.push(PauliString::new(if i==0{0.8}else{-0.3},(0..n).map(|q|(q,p(factors[(q as usize)+6*i]))).collect()).unwrap());}
        let expected=dense(&c,&sum);
        close(sparse(&c,&sum,false).value,expected);close(sparse(&c,&sum,true).value,expected);
        RelabelQubits::new(1).run(&mut c).unwrap();
        close(sparse(&c,&sum,false).value,expected);close(sparse(&c,&sum,true).value,expected);
    }
}
#[test]
fn exhaustive_two_qubit_conjugations_on_product_eigenstates() {
    let gates = vec![
        GateInstance::new(Gate::H, smallvec![0]),
        GateInstance::new(Gate::S, smallvec![0]),
        GateInstance::new(Gate::Sdg, smallvec![1]),
        GateInstance::new(Gate::Cnot, smallvec![0, 1]),
        GateInstance::new(Gate::Cnot, smallvec![1, 0]),
        GateInstance::new(Gate::Cz, smallvec![0, 1]),
        GateInstance::new(Gate::Iswap, smallvec![0, 1]),
        GateInstance::new(Gate::IswapDg, smallvec![0, 1]),
        GateInstance::controlled(Gate::Y, smallvec![1], smallvec![0]),
        GateInstance::new(Gate::Rx((-0.7).into()), smallvec![0]),
        GateInstance::new(Gate::Ry(0.4.into()), smallvec![1]),
        GateInstance::new(Gate::Rz((-0.2).into()), smallvec![0]),
    ];
    // Six eigenstates of each Pauli span the operator space, including negative signs/Y phases.
    for a in 0..6 {
        for b in 0..6 {
            for g in &gates {
                let mut c = Circuit::new(2, 0);
                for (q, state) in [(0, a), (1, b)] {
                    if state % 2 == 1 {
                        c.x(q).unwrap();
                    }
                    if state >= 2 {
                        c.h(q).unwrap();
                    }
                    if state >= 4 {
                        c.s(q).unwrap();
                    }
                }
                c.add_gate(g.clone()).unwrap();
                for x in 0..4 {
                    for y in 0..4 {
                        let ps = observable(&[(0, p(x)), (1, p(y))]);
                        close(sparse(&c, &ps, false).value, dense(&c, &ps));
                    }
                }
            }
        }
    }
}
#[test]
fn clifford_only_matches_tableau_without_branching() {
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.h(q).unwrap();
        c.s(q).unwrap();
    }
    for q in 0..5 {
        c.cnot(q, q + 1).unwrap();
    }
    c.swap(0, 5).unwrap();
    let ps = observable(&[(0, Pauli::Y), (2, Pauli::X), (5, Pauli::Z)]);
    let mut b = StabilizerBackend::with_seed(0);
    let state = run(&mut b, &c).unwrap();
    let r = sparse(&c, &ps, false);
    close(r.value, b.expectation_value(&state, &ps.terms[0]).unwrap());
    assert_eq!(r.stats.branching_rotations, 0);
    assert_eq!(r.stats.peak_terms, 1);
}
#[test]
fn exact_quarter_rotations_do_not_branch_but_nearby_angles_do() {
    for k in -4..5 {
        for axis in 0..3 {
            let mut c = Circuit::new(1, 0);
            c.h(0).unwrap();
            c.s(0).unwrap();
            let theta = (k as f64) * std::f64::consts::FRAC_PI_2;
            let g = match axis {
                0 => Gate::Rx(theta.into()),
                1 => Gate::Ry(theta.into()),
                _ => Gate::Rz(theta.into()),
            };
            c.add_gate(GateInstance::new(g, smallvec![0])).unwrap();
            let ps = observable(&[(0, Pauli::Y)]);
            let r = sparse(&c, &ps, false);
            close(r.value, dense(&c, &ps));
            assert_eq!(r.stats.branching_rotations, 0);
        }
    }
    let mut c = Circuit::new(1, 0);
    c.ry(std::f64::consts::FRAC_PI_2 + 1e-12, 0).unwrap();
    let r = sparse(&c, &observable(&[(0, Pauli::Z)]), false);
    assert_eq!(r.stats.peak_terms, 2);
    assert_eq!(r.stats.branching_rotations, 1);
}
#[test]
fn inverse_t_gates_merging_and_exact_cancellation() {
    let mut c = Circuit::new(2, 0);
    c.h(0).unwrap();
    c.t(0).unwrap();
    c.tdg(0).unwrap();
    c.cnot(0, 1).unwrap();
    c.cnot(0, 1).unwrap();
    let mut ps = observable(&[(0, Pauli::X)]);
    ps.terms
        .push(PauliString::new(-1., vec![(0, Pauli::X)]).unwrap());
    ps.terms.push(PauliString::identity(0.25));
    let r = sparse(&c, &ps, false);
    close(r.value, 0.25);
    assert!(r.stats.merges > 0);
    close(sparse(&c, &observable(&[(0, Pauli::X)]), true).value, 1.);
}
#[test]
fn global_128q_ghz_rotation_matches_independent_analytic_reference() {
    for n in [6, 128] {
        let mut c = Circuit::new(n, 0);
        c.h(0).unwrap();
        for q in 1..n {
            c.cnot(0, q).unwrap();
        }
        c.rz(-0.37, 0).unwrap();
        let ps = PauliSum {
            terms: vec![PauliString::new(1., (0..n).map(|q| (q, Pauli::X)).collect()).unwrap()],
        };
        for cones in [false, true] {
            let r = sparse(&c, &ps, cones);
            close(r.value, (-0.37f64).cos());
            assert_eq!(r.stats.max_active_qubits, n);
            assert_eq!(r.stats.peak_terms, 2);
        }
        if n == 6 {
            close(dense(&c, &ps), (-0.37f64).cos());
        }
    }
}
#[test]
fn unsupported_or_invalid_disjoint_operations_rejected_even_for_identity() {
    let ps = PauliSum {
        terms: vec![PauliString::identity(1.)],
    };
    for kind in 0..5 {
        let mut c = Circuit::new(2, 1);
        match kind {
            0 => {
                c.measure(1, 0).unwrap();
            }
            1 => {
                c.reset(1).unwrap();
            }
            2 => {
                c.u3(0.2, 0.3, 0.4, 1).unwrap();
            }
            3 => {
                c.rx(f64::NAN, 1).unwrap();
            }
            _ => {
                c.add_gate(GateInstance::controlled(
                    Gate::H,
                    smallvec![0],
                    smallvec![1],
                ))
                .unwrap();
            }
        }
        assert!(matches!(
            SparsePauliQuery::compile(&c, &ps, Default::default()),
            Err(SparsePauliError::Invalid(_))
        ));
    }
    let c = Circuit::new(2, 0);
    let bad = PauliSum {
        terms: vec![PauliString {
            coefficient: f64::NAN,
            terms: vec![],
        }],
    };
    assert!(SparsePauliQuery::compile(&c, &bad, Default::default()).is_err());
    let bad = PauliSum {
        terms: vec![PauliString {
            coefficient: 1.,
            terms: vec![(0, Pauli::X), (0, Pauli::Z)],
        }],
    };
    assert!(SparsePauliQuery::compile(&c, &bad, Default::default()).is_err());
}
#[test]
fn resource_limits_reject_before_expansion_without_partial_value() {
    let mut c = Circuit::new(8, 0);
    for q in 0..8 {
        c.ry(0.3, q).unwrap();
    }
    let ps = PauliSum {
        terms: vec![PauliString::new(1., (0..8).map(|q| (q, Pauli::Z)).collect()).unwrap()],
    };
    let q = SparsePauliQuery::compile(
        &c,
        &ps,
        SparsePauliOptions {
            max_terms: 8,
            use_cones: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        q.execute(),
        Err(SparsePauliError::Terms {
            required: 16,
            limit: 8
        })
    ));
    assert!(matches!(
        SparsePauliQuery::compile(
            &c,
            &ps,
            SparsePauliOptions {
                max_workspace_bytes: 0,
                ..Default::default()
            }
        ),
        Err(SparsePauliError::Memory { .. })
    ));
    let q = SparsePauliQuery::compile(
        &c,
        &ps,
        SparsePauliOptions {
            max_workspace_bytes: 300,
            use_cones: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(q.execute(), Err(SparsePauliError::Memory { .. })));
}
#[test]
fn tiles_empty_queries_zero_qubits_and_overflow() {
    let mut c = Circuit::new(2, 0);
    c.h(0).unwrap();
    c.t(0).unwrap();
    c.cnot(0, 1).unwrap();
    let ps = observable(&[(0, Pauli::Y), (1, Pauli::X)]);
    let e = dense(&c, &ps);
    TileBlock::new(2).run(&mut c).unwrap();
    close(sparse(&c, &ps, true).value, e);
    let c = Circuit::new(0, 0);
    close(sparse(&c, &PauliSum { terms: vec![] }, false).value, 0.);
    close(
        sparse(
            &c,
            &PauliSum {
                terms: vec![PauliString::identity(-2.)],
            },
            true,
        )
        .value,
        -2.,
    );
    let ps = PauliSum {
        terms: vec![
            PauliString::identity(f64::MAX),
            PauliString::identity(f64::MAX),
        ],
    };
    assert!(SparsePauliQuery::compile(&c, &ps, Default::default())
        .unwrap()
        .execute()
        .is_err());
}

#[test]
fn tiny_coefficients_are_not_pruned_and_malformed_empty_gates_do_not_panic() {
    let c = Circuit::new(1, 0);
    let ps = PauliSum {
        terms: vec![PauliString::new(1e-200, vec![(0, Pauli::Z)]).unwrap()],
    };
    for cones in [false, true] {
        let r = sparse(&c, &ps, cones);
        assert_eq!(r.value, 1e-200);
        assert_eq!(r.stats.peak_terms, 1);
    }
    // Public fused IR can contain an arity-zero malformed gate. Avoid constructor debug asserts.
    let mut c = Circuit::new(1, 0);
    c.add_instruction(aleph_ir::Instruction::TiledBlock(Box::new(
        aleph_ir::TiledBlock {
            tile_bits: 1,
            gates: vec![GateInstance {
                gate: Gate::UnitaryKq {
                    k: 0,
                    data: vec![aleph_core::Complex::new(1., 0.)].into_boxed_slice(),
                },
                qubits: smallvec![],
                controls: smallvec![],
            }],
        },
    )))
    .unwrap();
    assert!(SparsePauliQuery::compile(&c, &ps, Default::default()).is_err());
}
