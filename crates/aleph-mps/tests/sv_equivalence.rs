//! Oracle equivalence vs NaiveSvBackend + MPS invariants.
//! MPS dense reconstruction must match the SV amplitude vector (ADR-0004).

use aleph_backend::{run, Backend};
use aleph_benches::g;
use aleph_core::{Complex, Gate, Param, Pauli, PauliString};
use aleph_mps::{MpsBackend, MpsState, TruncationPolicy};
use aleph_sv::NaiveSvBackend;

fn mps_dense(circuit: &aleph_ir::Circuit, chi: usize) -> Vec<Complex> {
    let mut be = MpsBackend::with_seed(0).with_max_bond(chi);
    let st: MpsState = run(&mut be, circuit).unwrap();
    st.dense_statevector()
}

fn sv_dense(circuit: &aleph_ir::Circuit) -> Vec<Complex> {
    let mut be = NaiveSvBackend::with_seed(0);
    let st = run(&mut be, circuit).unwrap();
    st.amplitudes().to_vec()
}

#[test]
fn bell_matches_sv() {
    let mut c = aleph_ir::Circuit::new(2, 0);
    c.add_gate(g(Gate::H, &[0])).unwrap();
    c.add_gate(g(Gate::Cnot, &[0, 1])).unwrap();
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10);
    }
}

#[test]
fn nn_chain_matches_sv() {
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    for q in 0..n {
        c.add_gate(g(Gate::Rz(Param::Concrete(0.3 + q as f64 * 0.1)), &[q]))
            .unwrap();
    }
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10);
    }
}

#[test]
fn expectation_matches_sv() {
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    let mut mps = MpsBackend::with_seed(0).with_max_bond(64);
    let ms = run(&mut mps, &c).unwrap();
    let mut sv = NaiveSvBackend::with_seed(0);
    let svs = run(&mut sv, &c).unwrap();
    let observables = [
        vec![(0u32, Pauli::Z), (1, Pauli::Z)],
        vec![(0, Pauli::X), (2, Pauli::X)],
        vec![(1, Pauli::Z)],
    ];
    for terms in observables {
        let p = PauliString::new(1.0, terms).unwrap();
        let em = mps.expectation_value(&ms, &p).unwrap();
        let es = sv.expectation_value(&svs, &p).unwrap();
        assert!(
            (em - es).abs() < 1e-10,
            "expectation mismatch: {em} vs {es}"
        );
    }
}

#[test]
fn probabilities_matches_sv() {
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    let mut mps = MpsBackend::with_seed(0).with_max_bond(64);
    let ms = run(&mut mps, &c).unwrap();
    let mut sv = NaiveSvBackend::with_seed(0);
    let svs = run(&mut sv, &c).unwrap();
    for subset in [vec![0u32], vec![0, 2], vec![1, 3, 0]] {
        let pm = mps.probabilities(&ms, &subset).unwrap();
        let ps = sv.probabilities(&svs, &subset).unwrap();
        assert_eq!(pm.len(), ps.len());
        for (x, y) in pm.iter().zip(ps.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }
}

#[test]
fn small_chi_weak_entanglement_near_exact() {
    // Shallow nearest-neighbor circuit keeps Schmidt rank low; χ=4 near-exact.
    let n = 6u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    let a = mps_dense(&c, 4);
    let b = sv_dense(&c);
    let mut err = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        err += (x - y).norm_sqr();
    }
    assert!(err.sqrt() < 1e-6, "L2 error {} too large", err.sqrt());
}

#[test]
fn vqe_h2_matches_sv_machine_precision() {
    // 4-qubit hardware-efficient ansatz: Ry layers + nearest-neighbor CNOT ladder.
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    let thetas = [0.31, 0.59, 0.27, 0.18, 0.44, 0.62, 0.11, 0.53];
    let mut ti = 0usize;
    for _layer in 0..2 {
        for q in 0..n {
            c.add_gate(g(
                Gate::Ry(Param::Concrete(thetas[ti % thetas.len()])),
                &[q],
            ))
            .unwrap();
            ti += 1;
        }
        for q in 0..n - 1 {
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
        }
    }
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10, "VQE-H2 MPS vs SV mismatch");
    }
}

#[test]
#[ignore = "50-qubit MPS run; minutes-scale, runs on CI nightly"]
fn qaoa50_nn_ring_runs_reasonably() {
    let n = 50u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for _p in 0..3 {
        // Cost layer: nearest-neighbor ZZ via CNOT–RZ–CNOT on (q, q+1).
        for q in 0..n - 1 {
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            c.add_gate(g(Gate::Rz(Param::Concrete(0.7)), &[q + 1]))
                .unwrap();
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
        }
        // Mixer: RX on every qubit.
        for q in 0..n {
            c.add_gate(g(Gate::Rx(Param::Concrete(0.5)), &[q])).unwrap();
        }
    }
    let mut be = MpsBackend::with_seed(1).with_max_bond(64);
    let st = run(&mut be, &c).unwrap();
    // "Reasonable": bounded truncation, non-degenerate sampling.
    assert!(
        st.truncation_error() < 1e-1,
        "trunc_error {}",
        st.truncation_error()
    );
    let shots = be.sample(&st, 1000).unwrap();
    let distinct: std::collections::HashSet<u64> = shots.iter().copied().collect();
    assert!(distinct.len() > 1, "sampling produced a single bitstring");
}

fn mps_dense_policy(circuit: &aleph_ir::Circuit, policy: TruncationPolicy) -> (Vec<Complex>, f64) {
    let mut be = MpsBackend::with_seed(0).with_truncation(policy);
    let st: MpsState = run(&mut be, circuit).unwrap();
    (st.dense_statevector(), st.truncation_error())
}

#[test]
fn error_bounded_eps0_is_exact() {
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for q in 0..n - 1 {
        c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
    }
    for q in 0..n {
        c.add_gate(g(Gate::Rz(Param::Concrete(0.2 + q as f64 * 0.1)), &[q]))
            .unwrap();
    }
    let (a, err) = mps_dense_policy(
        &c,
        TruncationPolicy::ErrorBounded {
            epsilon: 0.0,
            max_bond: 64,
        },
    );
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10);
    }
    assert!(err < 1e-12, "ε=0 should discard nothing, got {err}");
}

#[test]
fn error_bounded_deviation_within_budget() {
    let n = 6u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for layer in 0..3 {
        for q in 0..n - 1 {
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            c.add_gate(g(
                Gate::Rz(Param::Concrete(0.3 + layer as f64 * 0.1)),
                &[q + 1],
            ))
            .unwrap();
        }
    }
    let (a, err) = mps_dense_policy(
        &c,
        TruncationPolicy::ErrorBounded {
            epsilon: 1e-4,
            max_bond: 64,
        },
    );
    let b = sv_dense(&c);
    let l2: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm_sqr())
        .sum::<f64>()
        .sqrt();
    // Per-truncation 2-norm error is √discardedᵢ; by Cauchy–Schwarz the total
    // satisfies ‖Δψ‖ ≤ √(K · Σ discardedᵢ) over K truncations. This circuit has
    // K = 3 layers × 5 CNOTs = 15 ≤ 16, so the 4× factor is a sound bound. If
    // you enlarge the circuit (K > 16) raise the factor to √K accordingly —
    // do NOT just bump it to make a real regression pass.
    assert!(
        l2 <= 4.0 * err.sqrt() + 1e-9,
        "L2 {l2} vs √err {}",
        err.sqrt()
    );
}

/// Regression: this exact nearest-neighbor circuit drove the MPS state norm to
/// 0.5 under the old nalgebra-based truncated SVD (orthonormal-but-wrong
/// vectors on a degenerate complex two-site block). faer fixes it. Deterministic
/// guard so the bug cannot silently return.
#[test]
fn regression_svd_norm_loss_seq() {
    let seq = [0u8, 1, 2, 2, 1, 4, 3, 1, 1, 1, 2, 4, 4, 4, 0, 4, 4];
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    let mut q = 0u32;
    for op in seq {
        q = (q + 1) % n;
        match op {
            0 => {
                c.add_gate(g(Gate::H, &[q])).unwrap();
            }
            1 => {
                c.add_gate(g(Gate::X, &[q])).unwrap();
            }
            2 => {
                c.add_gate(g(Gate::S, &[q])).unwrap();
            }
            3 => {
                c.add_gate(g(Gate::Y, &[q])).unwrap();
            }
            _ => {
                let lo = q.min(n - 2);
                c.add_gate(g(Gate::Cnot, &[lo, lo + 1])).unwrap();
            }
        }
    }
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    let norm: f64 = a.iter().map(|z| z.norm_sqr()).sum();
    assert!(
        (norm - 1.0).abs() < 1e-9,
        "MPS norm^2 = {norm} (was 0.5 with the bug)"
    );
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10, "MPS dense diverged from SV");
    }
}

#[test]
fn nonadjacent_matches_sv() {
    // Asymmetric control/target + various distances; χ large = exact.
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    c.add_gate(g(Gate::Cnot, &[0, 3])).unwrap(); // distance 3
    c.add_gate(g(Gate::Cnot, &[4, 1])).unwrap(); // reversed, distance 3
    c.add_gate(g(Gate::Cz, &[0, 4])).unwrap(); // distance 4 (symmetric)
    c.add_gate(g(Gate::Cnot, &[2, 0])).unwrap(); // reversed, distance 2
    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10);
    }
}

#[test]
fn lazy_perm_reads_match_sv() {
    // Long-range gates leave a non-identity permutation; every read API
    // must still report in logical-qubit order (P3-09).
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    c.add_gate(g(Gate::Cnot, &[0, 4])).unwrap(); // distance 4
    c.add_gate(g(Gate::Rz(Param::Concrete(0.4)), &[2])).unwrap();
    c.add_gate(g(Gate::Cnot, &[3, 1])).unwrap(); // reversed, distance 2
    c.add_gate(g(Gate::Cz, &[4, 2])).unwrap(); // distance 2 after permutation drift

    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10, "dense mismatch under permutation");
    }

    let mut mps = MpsBackend::with_seed(0).with_max_bond(64);
    let ms = run(&mut mps, &c).unwrap();
    assert!(
        ms.swaps_applied() > 0,
        "circuit must exercise the lazy router"
    );
    let mut sv = NaiveSvBackend::with_seed(0);
    let svs = run(&mut sv, &c).unwrap();

    for subset in [vec![0u32], vec![4, 0], vec![1, 3, 2]] {
        let pm = mps.probabilities(&ms, &subset).unwrap();
        let ps = sv.probabilities(&svs, &subset).unwrap();
        for (x, y) in pm.iter().zip(ps.iter()) {
            assert!(
                (x - y).abs() < 1e-10,
                "probabilities mismatch under permutation"
            );
        }
    }
    for terms in [
        vec![(0u32, Pauli::Z), (4, Pauli::Z)],
        vec![(2, Pauli::X)],
        vec![(1, Pauli::Z), (3, Pauli::Z)],
    ] {
        let p = PauliString::new(1.0, terms).unwrap();
        let em = mps.expectation_value(&ms, &p).unwrap();
        let es = sv.expectation_value(&svs, &p).unwrap();
        assert!(
            (em - es).abs() < 1e-10,
            "expectation mismatch: {em} vs {es}"
        );
    }
}

#[test]
fn swap_dense_matches_sv() {
    // P3-12: explicit user SWAPs (relabeled, never physical) interleaved with
    // CNOTs. χ large = exact; the dense vector must match SV through every
    // relabel, including a read after the final relabel.
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    c.add_gate(g(Gate::Cnot, &[0, 1])).unwrap();
    c.add_gate(g(Gate::Swap, &[0, 4])).unwrap(); // long-range relabel
    c.add_gate(g(Gate::Cnot, &[0, 1])).unwrap(); // qubit 0 now routed elsewhere
    c.add_gate(g(Gate::Swap, &[2, 3])).unwrap(); // adjacent relabel
    c.add_gate(g(Gate::Cnot, &[3, 4])).unwrap();
    c.add_gate(g(Gate::Rz(Param::Concrete(0.37)), &[0]))
        .unwrap();
    c.add_gate(g(Gate::Swap, &[1, 4])).unwrap();
    c.add_gate(g(Gate::Cz, &[0, 2])).unwrap();

    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10, "SWAP-dense dense mismatch");
    }

    // The three user SWAPs are discharged as relabels. (Physical router SWAPs
    // may still occur from the long-range CNOTs that follow — that is the lazy
    // router doing its job, not the SWAP gates going physical.)
    let mut be = MpsBackend::with_seed(0).with_max_bond(64);
    let st: MpsState = run(&mut be, &c).unwrap();
    assert_eq!(st.relabels(), 3, "each user SWAP must be one relabel");
}

#[test]
fn swap_relabel_adds_no_truncation_error() {
    // AC#3: at a saturated χ, a circuit whose SWAP is discharged as a relabel
    // has BIT-IDENTICAL trunc_error and final state to the SWAP-free logically
    // equivalent. Construction: conjugating a circuit by SWAP(1,3) and applying
    // the (1 3) transposition to every interior gate's qubit args yields the
    // same logical unitary (SWAP·∏Gτ·SWAP = ∏G). The relabel only renames
    // sites, so every truncated SVD sees bit-identical operands.
    let n = 6u32;
    let chi = 2usize; // saturated: forces truncation
    let tau = |q: u32| -> u32 {
        match q {
            1 => 3,
            3 => 1,
            other => other,
        }
    };

    // Interior (logical) circuit, exercising long-range routing under truncation.
    let interior: &[(Gate, Vec<u32>)] = &[
        (Gate::H, vec![0]),
        (Gate::H, vec![1]),
        (Gate::H, vec![2]),
        (Gate::H, vec![3]),
        (Gate::H, vec![4]),
        (Gate::H, vec![5]),
        (Gate::Cnot, vec![0, 1]),
        (Gate::Cnot, vec![1, 2]),
        (Gate::Cnot, vec![2, 3]),
        (Gate::Cnot, vec![3, 4]),
        (Gate::Cnot, vec![4, 5]),
        (Gate::Cnot, vec![0, 5]), // long range
        (Gate::Rz(Param::Concrete(0.41)), vec![2]),
        (Gate::Cnot, vec![1, 4]),
    ];

    // Baseline: interior circuit as written.
    let mut base = aleph_ir::Circuit::new(n, 0);
    for (gate, qs) in interior {
        base.add_gate(g(gate.clone(), qs)).unwrap();
    }

    // Conjugated: SWAP(1,3); interior with τ applied to every qubit arg; SWAP(1,3).
    let mut conj = aleph_ir::Circuit::new(n, 0);
    conj.add_gate(g(Gate::Swap, &[1, 3])).unwrap();
    for (gate, qs) in interior {
        let mapped: Vec<u32> = qs.iter().map(|&q| tau(q)).collect();
        conj.add_gate(g(gate.clone(), &mapped)).unwrap();
    }
    conj.add_gate(g(Gate::Swap, &[1, 3])).unwrap();

    let policy = TruncationPolicy::FixedBond(chi);
    let mut be_base = MpsBackend::with_seed(0).with_truncation(policy);
    let sb: MpsState = run(&mut be_base, &base).unwrap();
    let mut be_conj = MpsBackend::with_seed(0).with_truncation(policy);
    let sc: MpsState = run(&mut be_conj, &conj).unwrap();

    // Truncation error is bit-identical (the relabel does no SVD).
    assert_eq!(
        sb.truncation_error().to_bits(),
        sc.truncation_error().to_bits(),
        "SWAP must not perturb trunc_error: {} vs {}",
        sb.truncation_error(),
        sc.truncation_error()
    );
    // The two SWAPs were relabels, not physical router SWAPs.
    assert_eq!(sc.relabels(), 2);
    assert_eq!(
        sc.swaps_applied(),
        sb.swaps_applied(),
        "routing must be identical between the two circuits"
    );
    // Logically identical → bit-identical final state.
    let db = sb.dense_statevector();
    let dc = sc.dense_statevector();
    for (x, y) in db.iter().zip(dc.iter()) {
        assert!((x - y).norm() < 1e-15, "conjugated SWAP circuit diverged");
    }
}

#[test]
fn lazy_perm_sample_matches_probabilities() {
    // Sampling under a non-identity permutation: empirical distribution over
    // all qubits must match the exact marginals.
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    c.add_gate(g(Gate::Cnot, &[0, 3])).unwrap();
    c.add_gate(g(Gate::Cnot, &[2, 0])).unwrap();
    let mut be = MpsBackend::with_seed(7).with_max_bond(64);
    let st = run(&mut be, &c).unwrap();
    assert!(st.swaps_applied() > 0);
    let shots = be.sample(&st, 20000).unwrap();
    let mut counts = vec![0u64; 16];
    for sh in &shots {
        counts[*sh as usize] += 1;
    }
    let probs = be.probabilities(&st, &[0, 1, 2, 3]).unwrap();
    // Calibrated 5σ band instead of an ad-hoc ±0.02 (P3-16).
    aleph_oracle::assert_distribution_close("lazy_perm_sample", 4, &counts, &probs, 20000);
}

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Random nearest-neighbor 1q+2q circuit on 4 qubits, χ=64 (no truncation):
    /// MPS dense must equal SV dense to 1e-9, norm ≈ 1.
    #[test]
    fn random_nn_circuit_matches_sv(seq in prop::collection::vec(0u8..6, 0..30)) {
        let n = 4u32;
        let mut c = aleph_ir::Circuit::new(n, 0);
        let mut q = 0u32;
        for op in seq {
            q = (q + 1) % n;
            match op {
                0 => { c.add_gate(g(Gate::H, &[q])).unwrap(); }
                1 => { c.add_gate(g(Gate::X, &[q])).unwrap(); }
                2 => { c.add_gate(g(Gate::S, &[q])).unwrap(); }
                3 => { c.add_gate(g(Gate::Y, &[q])).unwrap(); }
                _ => {
                    let lo = q.min(n - 2);
                    c.add_gate(g(Gate::Cnot, &[lo, lo + 1])).unwrap();
                }
            }
        }
        if c.is_empty() { return Ok(()); }
        let a = mps_dense(&c, 64);
        let b = sv_dense(&c);
        let mut norm = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            prop_assert!((x - y).norm() < 1e-9);
            norm += x.norm_sqr();
        }
        prop_assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn error_bounded_eps0_matches_sv_random(seq in prop::collection::vec(0u8..6, 0..24)) {
        let n = 4u32;
        let mut c = aleph_ir::Circuit::new(n, 0);
        let mut q = 0u32;
        for op in seq {
            q = (q + 1) % n;
            match op {
                0 => { c.add_gate(g(Gate::H, &[q])).unwrap(); }
                1 => { c.add_gate(g(Gate::X, &[q])).unwrap(); }
                2 => { c.add_gate(g(Gate::S, &[q])).unwrap(); }
                3 => { c.add_gate(g(Gate::Y, &[q])).unwrap(); }
                _ => { let lo = q.min(n - 2); c.add_gate(g(Gate::Cnot, &[lo, lo + 1])).unwrap(); }
            }
        }
        if c.is_empty() { return Ok(()); }
        let (a, _) = mps_dense_policy(&c, TruncationPolicy::ErrorBounded { epsilon: 0.0, max_bond: 64 });
        let b = sv_dense(&c);
        for (x, y) in a.iter().zip(b.iter()) { prop_assert!((x - y).norm() < 1e-9); }
    }

    /// Random circuit with SWAP injection (op 5/6): SWAPs are discharged as
    /// relabels, yet the dense vector must still equal SV to 1e-9 (P3-12).
    #[test]
    fn random_swap_injection_matches_sv(seq in prop::collection::vec((0u8..7, 0u8..5, 0u8..5), 0..30)) {
        let n = 5u32;
        let mut c = aleph_ir::Circuit::new(n, 0);
        for (op, x, y) in seq {
            let a = (x as u32) % n;
            match op {
                0 => { c.add_gate(g(Gate::H, &[a])).unwrap(); }
                1 => { c.add_gate(g(Gate::S, &[a])).unwrap(); }
                2 => { c.add_gate(g(Gate::X, &[a])).unwrap(); }
                3 | 4 => {
                    let b = (y as u32) % n;
                    if a != b { c.add_gate(g(Gate::Cnot, &[a, b])).unwrap(); }
                }
                _ => {
                    let b = (y as u32) % n;
                    if a != b { c.add_gate(g(Gate::Swap, &[a, b])).unwrap(); }
                }
            }
        }
        if c.is_empty() { return Ok(()); }
        let am = mps_dense(&c, 64);
        let bm = sv_dense(&c);
        for (x, y) in am.iter().zip(bm.iter()) { prop_assert!((x - y).norm() < 1e-9); }
    }

    #[test]
    fn random_long_range_matches_sv(seq in prop::collection::vec((0u8..5, 0u8..5, 0u8..5), 0..20)) {
        let n = 5u32;
        let mut c = aleph_ir::Circuit::new(n, 0);
        for (op, x, y) in seq {
            let a = (x as u32) % n;
            match op {
                0 => { c.add_gate(g(Gate::H, &[a])).unwrap(); }
                1 => { c.add_gate(g(Gate::S, &[a])).unwrap(); }
                _ => {
                    let b = (y as u32) % n;
                    if a != b { c.add_gate(g(Gate::Cnot, &[a, b])).unwrap(); }
                }
            }
        }
        if c.is_empty() { return Ok(()); }
        let am = mps_dense(&c, 64);
        let bm = sv_dense(&c);
        for (x, y) in am.iter().zip(bm.iter()) { prop_assert!((x - y).norm() < 1e-9); }
    }
}
