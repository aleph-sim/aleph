//! P4.6-02 batched Pauli-frame sampler oracles.
//!
//! A stabilizer state's Z-basis measurement distribution is *exactly uniform*
//! over its support (a coset of {0,1}^n), so `1/|support|` on the support is an
//! EXACT reference — not an empirical one. We find the support with the trusted
//! per-shot CHP `measure` loop, then check the batched sampler against the exact
//! uniform distribution with the calibrated 5σ band (P3-16).

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance};
use aleph_oracle::assert_distribution_close;
use aleph_stab::{StabilizerBackend, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn gi(g: Gate, qs: &[u32]) -> GateInstance {
    GateInstance::new(g, qs.to_vec())
}

/// Build a final stabilizer state by applying `gates` to |0…0⟩.
fn prepare(n: u32, gates: &[GateInstance]) -> Tableau {
    let mut be = StabilizerBackend::with_seed(0);
    let mut t = be.allocate(n).unwrap();
    for g in gates {
        be.apply_gate(&mut t, g).unwrap();
    }
    t
}

/// Per-shot CHP reference: measure all qubits `shots` times, return the
/// histogram over `2^n` basis states (the trusted oracle the AC names).
fn per_shot_hist(state: &Tableau, n: usize, shots: u32, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut counts = vec![0u64; 1usize << n];
    for _ in 0..shots {
        let mut t = state.clone();
        let mut bits = 0u64;
        for q in 0..n {
            if t.measure(q, &mut rng).unwrap() {
                bits |= 1u64 << q;
            }
        }
        counts[bits as usize] += 1;
    }
    counts
}

fn batched_hist(state: &Tableau, n: usize, shots: u32, seed: u64) -> Vec<u64> {
    let mut be = StabilizerBackend::with_seed(seed);
    let mut counts = vec![0u64; 1usize << n];
    for s in be.sample(state, shots).unwrap() {
        counts[s as usize] += 1;
    }
    counts
}

/// Exact uniform-on-support distribution, with `support` taken from a large
/// per-shot reference run (every support point has prob ≥ 2^-n, so a big run
/// sees all of them).
fn exact_uniform(state: &Tableau, n: usize, ref_shots: u32) -> Vec<f64> {
    let hist = per_shot_hist(state, n, ref_shots, 777);
    let k = hist.iter().filter(|&&c| c > 0).count();
    hist.iter()
        .map(|&c| if c > 0 { 1.0 / k as f64 } else { 0.0 })
        .collect()
}

#[test]
fn ghz_batched_only_correlated_outcomes() {
    // GHZ-4: support is exactly {0000, 1111}, ~50/50.
    let mut gates = vec![gi(Gate::H, &[0])];
    for q in 0..3 {
        gates.push(gi(Gate::Cnot, &[q, q + 1]));
    }
    let state = prepare(4, &gates);
    let batched = batched_hist(&state, 4, 100_000, 9);
    assert!(
        batched[0b0000] > 0 && batched[0b1111] > 0,
        "GHZ peaks missing"
    );
    for (x, &c) in batched.iter().enumerate() {
        if x != 0b0000 && x != 0b1111 {
            assert_eq!(c, 0, "GHZ produced impossible outcome {x:04b}");
        }
    }
    let exact = vec![
        0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,
    ];
    assert_distribution_close("ghz4_frame", 4, &batched, &exact, 100_000);
}

#[test]
fn batched_matches_exact_uniform_on_support() {
    // Mixed Clifford on 5 qubits: H/S/CNOT scramble → non-trivial support.
    let gates = [
        gi(Gate::H, &[0]),
        gi(Gate::H, &[2]),
        gi(Gate::S, &[2]),
        gi(Gate::Cnot, &[0, 1]),
        gi(Gate::Cnot, &[2, 3]),
        gi(Gate::Cnot, &[1, 4]),
        gi(Gate::H, &[3]),
        gi(Gate::Cnot, &[3, 4]),
    ];
    let state = prepare(5, &gates);
    let exact = exact_uniform(&state, 5, 200_000);
    let batched = batched_hist(&state, 5, 200_000, 42);
    assert_distribution_close("frame_clifford5", 5, &batched, &exact, 200_000);
    // Sanity: the per-shot reference also matches the same exact distribution
    // (confirms the uniform-on-support assumption, independent of the batched path).
    let per_shot = per_shot_hist(&state, 5, 200_000, 4242);
    assert_distribution_close("pershot_clifford5", 5, &per_shot, &exact, 200_000);
}

#[test]
fn batched_partial_final_batch() {
    // shots not a multiple of 64 must still be correct (last batch is partial).
    let gates = [gi(Gate::H, &[0]), gi(Gate::Cnot, &[0, 1])];
    let state = prepare(2, &gates);
    let shots = 100_000 + 37; // 37 in the final partial batch
    let batched = batched_hist(&state, 2, shots, 5);
    // Bell pair: support {00, 11} uniform.
    assert_eq!(batched[0b01], 0);
    assert_eq!(batched[0b10], 0);
    let exact = vec![0.5, 0.0, 0.0, 0.5];
    assert_distribution_close("bell_partial", 2, &batched, &exact, shots);
}

#[test]
fn batched_deterministic_same_seed() {
    let gates = [
        gi(Gate::H, &[0]),
        gi(Gate::Cnot, &[0, 1]),
        gi(Gate::H, &[2]),
        gi(Gate::Cnot, &[2, 1]),
    ];
    let state = prepare(3, &gates);
    let mut be_a = StabilizerBackend::with_seed(2026);
    let mut be_b = StabilizerBackend::with_seed(2026);
    let a = be_a.sample(&state, 5000).unwrap();
    let b = be_b.sample(&state, 5000).unwrap();
    assert_eq!(a, b, "same seed must yield the same shot table");
    let mut be_c = StabilizerBackend::with_seed(2027);
    let c = be_c.sample(&state, 5000).unwrap();
    assert_ne!(a, c, "different seed should (almost surely) differ");
}

#[test]
fn subset_batched_matches_per_shot_subset() {
    // The public subset API: sample only qubits [0, 2] of a 4-qubit state in a
    // chosen order; batched must match the per-shot subset measurement.
    let gates = [
        gi(Gate::H, &[0]),
        gi(Gate::Cnot, &[0, 1]),
        gi(Gate::H, &[2]),
        gi(Gate::Cnot, &[2, 3]),
        gi(Gate::Cnot, &[1, 2]),
    ];
    let state = prepare(4, &gates);
    let subset = [0usize, 2];
    let shots = 200_000u32;

    let mut rng = StdRng::seed_from_u64(11);
    let words = state.sample_qubits_batched(&subset, shots, &mut rng);
    let mut bcounts = vec![0u64; 1 << subset.len()];
    for w in &words {
        bcounts[*w as usize] += 1;
    }
    // Per-shot subset reference.
    let mut rng2 = StdRng::seed_from_u64(99);
    let mut pcounts = vec![0u64; 1 << subset.len()];
    for _ in 0..shots {
        let mut t = state.clone();
        let mut bits = 0u64;
        for (i, &q) in subset.iter().enumerate() {
            if t.measure(q, &mut rng2).unwrap() {
                bits |= 1u64 << i;
            }
        }
        pcounts[bits as usize] += 1;
    }
    let k = pcounts.iter().filter(|&&c| c > 0).count();
    let exact: Vec<f64> = pcounts
        .iter()
        .map(|&c| if c > 0 { 1.0 / k as f64 } else { 0.0 })
        .collect();
    assert_distribution_close("subset_frame", subset.len() as u32, &bcounts, &exact, shots);
}

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Random Clifford circuit on 4 qubits: the batched sampler's distribution
    /// must equal the exact uniform-on-support reference (which the trusted
    /// per-shot path defines). Op codes: 0=H, 1=S, 2=X, 3+=CNOT(a, a+1 mod n).
    #[test]
    fn batched_matches_exact_over_random_cliffords(
        ops in prop::collection::vec((0u8..6, 0u8..4), 0..24)
    ) {
        let n = 4u32;
        let mut gates = Vec::new();
        for (op, q) in ops {
            let a = (q as u32) % n;
            match op {
                0 => gates.push(gi(Gate::H, &[a])),
                1 => gates.push(gi(Gate::S, &[a])),
                2 => gates.push(gi(Gate::X, &[a])),
                _ => {
                    let b = (a + 1) % n;
                    gates.push(gi(Gate::Cnot, &[a, b]));
                }
            }
        }
        let state = prepare(n, &gates);
        let exact = exact_uniform(&state, n as usize, 80_000);
        let batched = batched_hist(&state, n as usize, 60_000, 7);
        assert_distribution_close("prop_frame", n, &batched, &exact, 60_000);
    }
}
