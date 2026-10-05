//! P6-02 oracle: distributed plan + CPU reference executor vs NaiveSvBackend.

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance, Param};
use aleph_ir::dist::{
    compile_detailed, initial_placement, plan, plan_from, CostModel, Dag, DistError, DistLayout,
    DistStep, Router,
};
use aleph_ir::{Circuit, DiagonalPhase, Instruction, PhaseTerm};
use aleph_sv::dist_ref::run_dist;
use aleph_sv::NaiveSvBackend;
use proptest::prelude::*;

const TOL: f64 = 1e-10;

fn reference(c: &Circuit) -> Vec<Complex> {
    let mut b = NaiveSvBackend::new();
    run(&mut b, c).unwrap().amplitudes().to_vec()
}

fn assert_close(a: &[Complex], b: &[Complex], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).norm() < TOL, "{what}: amp {i}: {x} vs {y}");
    }
}

fn check(c: &Circuit, g: u32, what: &str) {
    for router in [
        Router::Naive,
        Router::Lookahead,
        Router::Reorder { max_k: 1 },
        Router::Reorder { max_k: 3 },
    ] {
        let p = plan(c, DistLayout::new(c.num_qubits(), g).unwrap(), router).unwrap();
        assert_close(
            &run_dist(&p).unwrap(),
            &reference(c),
            &format!("{what} g={g} {router:?}"),
        );
    }
}

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in (0..n).rev() {
        c.h(j).unwrap();
        for k in (0..j).rev() {
            let theta = std::f64::consts::PI / f64::from(1u32 << (j - k));
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(theta)),
                vec![j],
                vec![k],
            ))
            .unwrap();
        }
    }
    for q in 0..n / 2 {
        c.swap(q, n - 1 - q).unwrap();
    }
    c
}

fn brickwall(n: u32, depth: usize, seed: u64) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let mut s = seed;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64) * std::f64::consts::TAU
    };
    for d in 0..depth {
        for q in 0..n {
            c.rx(rnd(), q).unwrap();
            c.rz(rnd(), q).unwrap();
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    c
}

#[test]
fn ghz_all_layouts() {
    for g in 0..=3 {
        check(&ghz(8), g, "ghz8");
    }
}

#[test]
fn qft_all_layouts() {
    for g in 0..=3 {
        check(&qft(8), g, "qft8");
    }
}

#[test]
fn brickwall_all_layouts() {
    for g in 0..=3 {
        check(&brickwall(9, 6, 7), g, "brick9");
    }
}

#[test]
fn rz_on_top_qubit_is_rank_phase_not_global() {
    let mut c = Circuit::new(5, 0);
    for q in 0..5 {
        c.h(q).unwrap();
    }
    c.rz(0.9, 4).unwrap();
    c.cz(3, 4).unwrap();
    check(&c, 2, "rz-top");
}

#[test]
fn trailing_relabel_swap() {
    let mut c = ghz(6);
    c.swap(0, 5).unwrap();
    let p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Naive).unwrap();
    assert_eq!(p.stats.relabels, 1);
    assert_close(&run_dist(&p).unwrap(), &reference(&c), "trailing swap");
}

#[test]
fn toffoli_and_controlled_with_global_controls() {
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.h(q).unwrap();
    }
    c.add_gate(GateInstance::new(Gate::Toffoli, vec![5, 4, 0]))
        .unwrap();
    c.add_gate(GateInstance::new(Gate::Toffoli, vec![5, 1, 2]))
        .unwrap();
    c.add_gate(GateInstance::controlled(
        Gate::H,
        vec![1u32],
        vec![5u32, 0u32],
    ))
    .unwrap();
    c.add_gate(GateInstance::new(Gate::Ccz, vec![4, 5, 0]))
        .unwrap();
    check(&c, 2, "toffoli");
}

#[test]
fn mutation_dropping_an_exchange_breaks_oracle() {
    let c = qft(6);
    let mut p = plan(&c, DistLayout::new(6, 2).unwrap(), Router::Naive).unwrap();
    // The *last* exchange: the first one acts on |0…0⟩, where swapping two
    // |0⟩ qubits is the identity, so dropping it would be a no-op mutation.
    let idx = p
        .steps
        .iter()
        .rposition(|s| matches!(s, DistStep::Exchange { .. }))
        .expect("qft6 on g=2 needs an exchange");
    p.steps.remove(idx);
    let bad = match run_dist(&p) {
        Err(_) => true,
        Ok(v) => v
            .iter()
            .zip(reference(&c))
            .any(|(a, b)| (a - b).norm() > 1e-6),
    };
    assert!(bad, "removing an exchange must be detected");
}

/// Strip non-unitary instructions so arbitrary generated circuits are plannable.
fn unitary_only(c: &Circuit) -> Circuit {
    let mut out = Circuit::new(c.num_qubits(), 0);
    for i in c.instructions() {
        match i {
            Instruction::Gate(g) => {
                out.add_gate(g.clone()).unwrap();
            }
            Instruction::Barrier(_) => {
                out.add_instruction(i.clone()).unwrap();
            }
            _ => {}
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn prop_random_circuits_match(
        c in aleph_test::circuit::arb_circuit_full(6, 2, 40),
        g in 0u32..=3,
        router in prop_oneof![
            Just(Router::Naive),
            Just(Router::Lookahead),
            (1u32..=3).prop_map(|max_k| Router::Reorder { max_k }),
        ],
    ) {
        let c = unitary_only(&c);
        let p = plan(&c, DistLayout::new(6, g).unwrap(), router).unwrap();
        let got = run_dist(&p).unwrap();
        let want = reference(&c);
        for (x, y) in got.iter().zip(&want) {
            prop_assert!((x - y).norm() < TOL);
        }
    }
}

fn h_layer(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
        c.t(q).unwrap();
    }
    c
}

/// Review focus 2: a two-target gate on {top local slot, global} (and both
/// global, and second exchange hitting the first-brought qubit), checked on
/// amplitudes rather than stats.
#[test]
fn iswap_and_controlled_iswap_on_top_slot_and_globals() {
    for &(a, b) in &[(3u32, 5u32), (5, 3), (4, 5), (5, 4), (3, 4)] {
        let mut c = h_layer(6);
        c.add_gate(GateInstance::new(Gate::Iswap, vec![a, b]))
            .unwrap();
        c.add_gate(GateInstance::controlled(
            Gate::Iswap,
            vec![b, a],
            vec![0u32],
        ))
        .unwrap();
        let u2 = match Gate::CRx(Param::Concrete(0.8)).matrix().unwrap() {
            aleph_core::GateMatrix::M4x4(m) => m,
            _ => unreachable!(),
        };
        c.add_gate(GateInstance::new(Gate::Unitary2q(Box::new(u2)), vec![a, b]))
            .unwrap();
        check(&c, 2, &format!("iswap({a},{b})"));
    }
}

/// Every diagonal gate kind with a global qubit in every operand position.
#[test]
fn every_diagonal_gate_on_global_positions() {
    let p = |x| Param::Concrete(x);
    let d1 = Gate::Unitary1qDiag(Box::new([
        Complex::from_polar(1.0, 0.4),
        Complex::from_polar(1.0, -1.3),
    ]));
    let one_q = [
        Gate::Z,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::Rz(p(0.9)),
        Gate::Phase(p(-0.6)),
        d1,
    ];
    for g1 in one_q {
        for q in [4u32, 5] {
            let mut c = h_layer(6);
            c.add_gate(GateInstance::new(g1.clone(), vec![q])).unwrap();
            c.add_gate(GateInstance::controlled(g1.clone(), vec![q], vec![1u32]))
                .unwrap();
            c.add_gate(GateInstance::controlled(g1.clone(), vec![1], vec![q]))
                .unwrap();
            check(&c, 2, &format!("{g1:?} on {q}"));
        }
    }
    for g2 in [Gate::Cz, Gate::CRz(p(1.1))] {
        for qs in [[0u32, 5], [5, 0], [4, 5], [5, 4]] {
            let mut c = h_layer(6);
            c.add_gate(GateInstance::new(g2.clone(), qs.to_vec()))
                .unwrap();
            c.add_gate(GateInstance::controlled(
                g2.clone(),
                qs.to_vec(),
                vec![2u32],
            ))
            .unwrap();
            check(&c, 2, &format!("{g2:?} on {qs:?}"));
        }
    }
    for qs in [[0u32, 1, 5], [5, 0, 1], [4, 5, 0], [4, 1, 5], [3, 4, 5]] {
        let mut c = h_layer(6);
        c.add_gate(GateInstance::new(Gate::Ccz, qs.to_vec()))
            .unwrap();
        check(&c, 3, &format!("Ccz on {qs:?}"));
        check(&c, 2, &format!("Ccz on {qs:?}"));
    }
}

#[test]
fn grover_n8_matches() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/qiskit-baseline/circuits/grover_n8_iters13.qasm"
    ))
    .unwrap();
    let c = unitary_only(&aleph_parser::parse(&src).unwrap());
    for g in 0..=3 {
        check(&c, g, "grover8");
    }
}

/// One random gate from every family the planner/specialize distinguish,
/// with 0–2 external controls.
fn arb_any_gate(n: u32) -> impl Strategy<Value = GateInstance> {
    (
        0usize..19,
        Just((0..n).collect::<Vec<u32>>()).prop_shuffle(),
        0usize..=2,
        -3.0f64..3.0,
        -3.0f64..3.0,
        -3.0f64..3.0,
    )
        .prop_map(|(kind, perm, nctrl, a, b, t)| {
            let p = Param::Concrete;
            let (gate, arity) = match kind {
                0 => (Gate::H, 1),
                1 => (Gate::Y, 1),
                2 => (Gate::Rx(p(a)), 1),
                3 => (Gate::U3(p(a), p(b), p(t)), 1),
                4 => (Gate::Rz(p(a)), 1),
                5 => {
                    let m = match Gate::U3(p(a), p(b), p(t)).matrix().unwrap() {
                        aleph_core::GateMatrix::M2x2(m) => m,
                        _ => unreachable!(),
                    };
                    (Gate::Unitary1q(Box::new(m)), 1)
                }
                6 => (
                    Gate::Unitary1qDiag(Box::new([
                        Complex::from_polar(1.0, a),
                        Complex::from_polar(1.0, b),
                    ])),
                    1,
                ),
                7 => (Gate::Cnot, 2),
                8 => (Gate::CRx(p(a)), 2),
                9 => (Gate::CRy(p(b)), 2),
                10 => (Gate::CRz(p(t)), 2),
                11 => (Gate::Iswap, 2),
                12 => (Gate::IswapDg, 2),
                13 => (Gate::Swap, 2),
                14 => {
                    let m = match Gate::CRy(p(a)).matrix().unwrap() {
                        aleph_core::GateMatrix::M4x4(m) => m,
                        _ => unreachable!(),
                    };
                    (Gate::Unitary2q(Box::new(m)), 2)
                }
                15 => (Gate::Toffoli, 3),
                16 => (Gate::X, 1),
                17 => (Gate::Cz, 2),
                _ => (Gate::Ccz, 3),
            };
            let nctrl = nctrl.min(perm.len() - arity);
            GateInstance::controlled(
                gate,
                perm[..arity].to_vec(),
                perm[arity..arity + nctrl].to_vec(),
            )
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]
    #[test]
    fn prop_all_gate_families_with_controls(
        gates in prop::collection::vec(arb_any_gate(6), 1..30),
        g in 0u32..=2,
        router in prop_oneof![
            Just(Router::Naive),
            Just(Router::Lookahead),
            (1u32..=3).prop_map(|max_k| Router::Reorder { max_k }),
        ],
    ) {
        let mut c = h_layer(6);
        for gi in gates {
            c.add_gate(gi).unwrap();
        }
        let p = plan(&c, DistLayout::new(6, g).unwrap(), router).unwrap();
        let got = run_dist(&p).unwrap();
        let want = reference(&c);
        for (x, y) in got.iter().zip(&want) {
            prop_assert!((x - y).norm() < TOL);
        }
    }
}

#[test]
fn exchange_cpu_multi_bit_is_product_of_swaps() {
    use aleph_sv::dist_ref::exchange_cpu;
    for (n, g, bits) in [
        (5u32, 2u32, vec![3u32, 4]),
        (5, 2, vec![4, 3]),
        (6, 3, vec![5, 3, 4]),
    ] {
        let l = DistLayout::new(n, g).unwrap();
        let m = l.m();
        let size = 1usize << m;
        let mut ranks: Vec<Vec<Complex>> = (0..l.ranks() as usize)
            .map(|r| {
                (0..size)
                    .map(|i| Complex::new((r * size + i) as f64, 0.0))
                    .collect()
            })
            .collect();
        let full: Vec<Complex> = ranks.concat();
        exchange_cpu(&mut ranks, l, &bits);
        let got: Vec<Complex> = ranks.concat();
        let k = bits.len() as u32;
        for (x, want) in full.iter().enumerate() {
            // destination of index x: swap bit bits[j] with bit m-k+j for all j
            let mut y = x;
            for (j, &gb) in bits.iter().enumerate() {
                let lb = m - k + j as u32;
                let (a, b) = ((x >> gb) & 1, (x >> lb) & 1);
                y = (y & !(1 << gb) & !(1 << lb)) | (b << gb) | (a << lb);
            }
            assert_eq!(got[y], *want, "n={n} bits={bits:?} x={x}");
        }
    }
}

#[test]
fn mid_circuit_relabel_then_lookahead() {
    let mut c = brickwall(8, 3, 11);
    c.swap(1, 7).unwrap();
    c.swap(6, 2).unwrap();
    let tail = brickwall(8, 3, 12);
    for i in tail.instructions() {
        if let Instruction::Gate(g) = i {
            c.add_gate(g.clone()).unwrap();
        }
    }
    for g in 1..=3 {
        check(&c, g, "relabel-mid");
    }
}

#[test]
fn lookahead_moves_less_on_brickwall() {
    let c = brickwall(12, 10, 3);
    let l = DistLayout::new(12, 2).unwrap();
    let n = plan(&c, l, Router::Naive).unwrap().stats;
    let a = plan(&c, l, Router::Lookahead).unwrap().stats;
    assert!(
        a.amps_moved_per_rank < n.amps_moved_per_rank,
        "{a:?} vs {n:?}"
    );
}

#[test]
fn tight_m_prefetch_cap_oracle() {
    for (n, g) in [(4u32, 2u32), (5, 3), (3, 1)] {
        let mut c = h_layer(n);
        c.add_gate(GateInstance::new(Gate::Iswap, vec![0, n - 2]))
            .unwrap();
        c.h(n - 1).unwrap();
        c.add_gate(GateInstance::new(Gate::Iswap, vec![n - 1, 1]))
            .unwrap();
        c.rx(0.4, n - 2).unwrap();
        c.h(0).unwrap();
        check(&c, g, "tight-m");
    }
}

#[test]
fn required_qubit_in_top_slot_k2_oracle() {
    let mut c = h_layer(6);
    c.add_gate(GateInstance::new(Gate::Iswap, vec![3, 5]))
        .unwrap();
    c.h(4).unwrap();
    c.rx(0.2, 3).unwrap();
    check(&c, 2, "req-top-k2");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn prop_any_initial_map_matches(
        gates in prop::collection::vec(arb_any_gate(6), 1..30),
        g in 0u32..=2,
        init in Just((0..6u32).collect::<Vec<u32>>()).prop_shuffle(),
        router in prop_oneof![
            Just(Router::Naive),
            Just(Router::Lookahead),
            (1u32..=3).prop_map(|max_k| Router::Reorder { max_k }),
        ],
    ) {
        let mut c = h_layer(6);
        for gi in gates {
            c.add_gate(gi).unwrap();
        }
        let p = plan_from(&c, DistLayout::new(6, g).unwrap(), router, &init).unwrap();
        let got = run_dist(&p).unwrap();
        let want = reference(&c);
        for (x, y) in got.iter().zip(&want) {
            prop_assert!((x - y).norm() < TOL);
        }
    }
}

/// `c` re-emitted in a pseudo-random topological order of its DAG.
fn random_topo(c: &Circuit, seed: u64) -> Circuit {
    let mut dag = Dag::build(c).unwrap();
    let mut ready = dag.initial_ready();
    let mut out = Circuit::new(c.num_qubits(), 0);
    let mut s = seed | 1;
    while !ready.is_empty() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let i = ready.swap_remove((s >> 33) as usize % ready.len());
        out.add_instruction(c.instructions()[i].clone()).unwrap();
        dag.complete(i, &mut ready).unwrap();
    }
    assert_eq!(out.len(), c.len(), "DAG must schedule every instruction");
    out
}

fn arb_dp(n: u32) -> impl Strategy<Value = Instruction> {
    prop::collection::vec(
        (prop::collection::vec(1u64..(1u64 << n), 1..3), -3.0f64..3.0),
        1..4,
    )
    .prop_map(move |terms| {
        Instruction::DiagonalPhase(Box::new(DiagonalPhase {
            n_qubits: n,
            terms: terms
                .into_iter()
                .map(|(conds, angle)| PhaseTerm {
                    conds: conds.into_iter().collect(),
                    angle,
                })
                .collect(),
        }))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    /// Spec §7: any order the DAG allows is the same operator; random orders
    /// test the DAG itself, not one scheduler's choice.
    #[test]
    fn prop_every_dag_order_is_equivalent(
        gates in prop::collection::vec(arb_any_gate(6), 1..40),
        dp in arb_dp(6),
        dp_at in 0usize..1000,
        seed in any::<u64>(),
    ) {
        let mut c = h_layer(6);
        let at = dp_at % (gates.len() + 1);
        let n_gates = gates.len();
        for (k, gi) in gates.into_iter().enumerate() {
            if k == at {
                c.add_instruction(dp.clone()).unwrap();
            }
            c.add_gate(gi).unwrap();
        }
        if at == n_gates {
            c.add_instruction(dp.clone()).unwrap();
        }
        let want = reference(&c);
        for k in 0..8u64 {
            let got = reference(&random_topo(&c, seed ^ (k.wrapping_mul(0x9E37_79B9))));
            for (x, y) in got.iter().zip(&want) {
                prop_assert!((x - y).norm() < TOL);
            }
        }
    }
}

/// Mutation check: an unsound DAG (here, treating CNOT target as Z) would be
/// caught. Swapping the two CNOTs of `cx(0,1); cx(1,0)` changes the state.
#[test]
fn reordering_non_commuting_cnots_is_detectable() {
    let mut c = h_layer(2);
    c.rx(0.7, 0).unwrap();
    c.cnot(0, 1).unwrap();
    c.cnot(1, 0).unwrap();
    let mut swapped = h_layer(2);
    swapped.rx(0.7, 0).unwrap();
    swapped.cnot(1, 0).unwrap();
    swapped.cnot(0, 1).unwrap();
    let (a, b) = (reference(&c), reference(&swapped));
    assert!(a.iter().zip(&b).any(|(x, y)| (x - y).norm() > 1e-6));
}

/// Guards against an over-serialising DAG: commuting neighbours must be
/// allowed to reorder, and every sampled order must stay state-equivalent.
#[test]
fn dag_allows_commuting_reorders() {
    let mut c = Circuit::new(3, 0);
    c.rz(0.1, 0).unwrap();
    c.cz(0, 1).unwrap();
    c.rz(0.2, 1).unwrap();
    c.cnot(0, 2).unwrap();
    c.cnot(1, 2).unwrap();
    c.h(2).unwrap();
    let dag = Dag::build(&c).unwrap();
    assert!(
        dag.initial_ready().len() > 1,
        "commuting heads must be ready together"
    );
    let orig: Vec<String> = c.instructions().iter().map(|i| format!("{i:?}")).collect();
    let want = reference(&c);
    let mut reordered = false;
    for seed in 0..32u64 {
        let o = random_topo(&c, seed);
        let seq: Vec<String> = o.instructions().iter().map(|i| format!("{i:?}")).collect();
        reordered |= seq != orig;
        for (x, y) in reference(&o).iter().zip(&want) {
            assert!((x - y).norm() < TOL);
        }
    }
    assert!(reordered, "DAG never reordered anything");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn prop_initial_placement_matches(
        gates in prop::collection::vec(arb_any_gate(6), 1..30),
        dp in arb_dp(6),
        dp_at in 0usize..1000,
        g in 0u32..=3,
        router in prop_oneof![
            Just(Router::Naive),
            Just(Router::Lookahead),
            (1u32..=3).prop_map(|max_k| Router::Reorder { max_k }),
        ],
    ) {
        let mut c = h_layer(6);
        let at = dp_at % (gates.len() + 1);
        let n_gates = gates.len();
        for (k, gi) in gates.into_iter().enumerate() {
            if k == at {
                c.add_instruction(dp.clone()).unwrap();
            }
            c.add_gate(gi).unwrap();
        }
        if at == n_gates {
            c.add_instruction(dp.clone()).unwrap();
        }
        let l = DistLayout::new(6, g).unwrap();
        let p = plan_from(&c, l, router, &initial_placement(&c, l).unwrap()).unwrap();
        let got = run_dist(&p).unwrap();
        let want = reference(&c);
        for (x, y) in got.iter().zip(&want) {
            prop_assert!((x - y).norm() < TOL);
        }
    }
}

/// Stub models for `compile`: `w` per local instruction, `(1 - 2^-k)*x` per
/// exchange. Different weights steer `compile` to different candidates.
struct StubModel {
    w: f64,
    x: f64,
}
impl CostModel for StubModel {
    fn local_segment(&self, instrs: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
        Ok(self.w * instrs.len() as f64)
    }
    fn exchange(&self, k: u32, _m: u32) -> f64 {
        self.x * (1.0 - 0.5f64.powi(k as i32))
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn prop_compile_matches(
        gates in prop::collection::vec(arb_any_gate(6), 1..30),
        dp in arb_dp(6),
        dp_at in 0usize..1000,
        g in 0u32..=3,
        w in prop_oneof![Just(0.0), Just(0.01), Just(1.0)],
        x in prop_oneof![Just(0.0), Just(1.0), Just(100.0)],
    ) {
        let mut c = h_layer(6);
        let at = dp_at % (gates.len() + 1);
        let n_gates = gates.len();
        for (k, gi) in gates.into_iter().enumerate() {
            if k == at {
                c.add_instruction(dp.clone()).unwrap();
            }
            c.add_gate(gi).unwrap();
        }
        if at == n_gates {
            c.add_instruction(dp.clone()).unwrap();
        }
        let l = DistLayout::new(6, g).unwrap();
        let out = compile_detailed(&c, l, &StubModel { w, x }).unwrap();
        let got = run_dist(&out.plan).unwrap();
        let want = reference(&c);
        for (x, y) in got.iter().zip(&want) {
            prop_assert!((x - y).norm() < TOL);
        }
    }
}
