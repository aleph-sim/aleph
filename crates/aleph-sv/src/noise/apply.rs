//! Channel application (quantum-jump) and per-shot RNG seeding.

use aleph_core::{Complex, Pauli};
use rand::{rngs::StdRng, Rng};

use std::collections::HashMap;

use super::error::{KrausChannel, PauliChannel, QuantumError, ReadoutError};
use crate::measure::DEGENERATE_BRANCH_THRESHOLD;

/// Apply one channel to `amps` by quantum-jump. `qubits` maps the channel's
/// local qubit indices to global qubit indices. Pauli channels take the
/// state-independent fast path (sample a Pauli, apply via unitary kernels, no
/// renormalization); general Kraus channels compute `pᵢ=‖Kᵢ|ψ〉‖²`, sample,
/// apply, and renormalize. Spec §3.
// used by run_noisy in Task 7
#[allow(dead_code)]
pub(super) fn apply_channel(
    amps: &mut [Complex],
    _num_qubits: u32,
    err: &QuantumError,
    qubits: &[u32],
    rng: &mut StdRng,
) {
    match err {
        QuantumError::Pauli(c) => apply_pauli_channel(amps, c, qubits, rng),
        QuantumError::Kraus(c) => apply_kraus_1q(amps, c, qubits[0], rng),
    }
}

/// Fast path: sample a Pauli term from the fixed weights and apply each
/// single-qubit Pauli factor via the existing `apply_1q` kernel. No norm pass,
/// no renormalization — the weights are state-independent.
fn apply_pauli_channel(amps: &mut [Complex], c: &PauliChannel, qubits: &[u32], rng: &mut StdRng) {
    let r: f64 = rng.gen::<f64>();
    let mut acc = 0.0;
    let mut chosen = c.terms.len() - 1; // last term absorbs FP residue
    for (i, (p, _)) in c.terms.iter().enumerate() {
        acc += *p;
        if r < acc {
            chosen = i;
            break;
        }
    }
    let (_, paulis) = &c.terms[chosen];
    for (local, pl) in paulis.iter().enumerate() {
        if *pl == Pauli::I {
            continue;
        }
        let m = pl.matrix();
        crate::kernels::aos::apply_1q(amps, qubits[local], &[], &m);
    }
}

/// General 1q quantum-jump. For Kraus set {Kᵢ} on qubit `q`:
///   1. pᵢ = ‖Kᵢ|ψ〉‖² (Σpᵢ = 1 by CPTP);
///   2. sample branch i with probability pᵢ;
///   3. apply Kᵢ to |ψ〉 and renormalize by 1/√pᵢ.
///
/// Works pairwise over the (qubit `q` = 0, qubit `q` = 1) amplitude pairs.
fn apply_kraus_1q(amps: &mut [Complex], c: &KrausChannel, q: u32, rng: &mut StdRng) {
    let qbit = 1usize << q;
    // Step 1: branch probabilities. For each pair (a0, a1) and each Kraus op,
    // the local image is (K[0][0]a0 + K[0][1]a1, K[1][0]a0 + K[1][1]a1).
    let mut probs = vec![0.0_f64; c.kraus.len()];
    // Paired index access: amps[i] is qubit-q=0, amps[i|qbit] is qubit-q=1.
    // A plain iterator cannot express this pairwise pattern.
    #[allow(clippy::needless_range_loop)]
    for i in 0..amps.len() {
        if i & qbit != 0 {
            continue; // visit each pair once, from its qbit-clear index
        }
        let a0 = amps[i];
        let a1 = amps[i | qbit];
        for (ki, k) in c.kraus.iter().enumerate() {
            let o0 = k[0][0] * a0 + k[0][1] * a1;
            let o1 = k[1][0] * a0 + k[1][1] * a1;
            probs[ki] += o0.norm_sqr() + o1.norm_sqr();
        }
    }
    // Step 2: sample a branch (last branch absorbs FP residue).
    let r = rng.gen::<f64>();
    let mut acc = 0.0;
    let mut chosen = c.kraus.len() - 1;
    for (ki, p) in probs.iter().enumerate() {
        acc += *p;
        if r < acc {
            chosen = ki;
            break;
        }
    }
    // Step 3: apply the chosen Kraus op and renormalize by 1/√p_chosen.
    let pc = probs[chosen];
    if pc < DEGENERATE_BRANCH_THRESHOLD {
        // Degenerate branch (sampled only via FP residue): nothing meaningful
        // to project onto. Leave the state as-is — it is still the normalized
        // pre-channel state — rather than scaling by ~1e150 (mirrors measure.rs).
        return;
    }
    let inv = 1.0 / pc.sqrt();
    let k = &c.kraus[chosen];
    // Same pairwise (amps[i], amps[i|qbit]) access as Step 1 — needs the index.
    #[allow(clippy::needless_range_loop)]
    for i in 0..amps.len() {
        if i & qbit != 0 {
            continue;
        }
        let a0 = amps[i];
        let a1 = amps[i | qbit];
        amps[i] = (k[0][0] * a0 + k[0][1] * a1) * inv;
        amps[i | qbit] = (k[1][0] * a0 + k[1][1] * a1) * inv;
    }
}

/// Apply per-qubit readout error to a sampled basis-state index. For each
/// measured qubit with a `ReadoutError`, the recorded outcome bit is the true
/// bit `t` flipped to `1-t` with probability `m[t][1-t]`. Qubits without an
/// entry are read out perfectly.
// used by run_noisy in Task 7
#[allow(dead_code)]
pub(super) fn apply_readout(
    index: u64,
    num_qubits: u32,
    readout: &HashMap<u32, ReadoutError>,
    rng: &mut StdRng,
) -> u64 {
    if readout.is_empty() {
        return index;
    }
    let mut out = index;
    for q in 0..num_qubits {
        let Some(ro) = readout.get(&q) else { continue };
        let bit = ((index >> q) & 1) as usize; // true value t
        let p_flip = ro.m[bit][1 - bit]; // P(measure 1-t | true t)
        if rng.gen::<f64>() < p_flip {
            out ^= 1u64 << q; // record the flipped bit
        }
    }
    out
}

/// Deterministic per-shot seed: a splitmix64 mix of `(seed, shot)` so shot
/// outcomes are reproducible regardless of rayon scheduling (spec §1).
///
/// # Why splitmix64?
/// Plain `seed + shot` leaves strong linear correlation between adjacent shots
/// that can skew Monte-Carlo estimators. The splitmix64 finalizer provides full
/// 64-bit avalanche with no correlation at a cost of ~4 instructions per shot.
// used by run_noisy in Task 7
#[allow(dead_code)]
pub(super) fn shot_seed(seed: u64, shot: u64) -> u64 {
    // splitmix64 finalizer applied to (seed·INC + shot + INC).
    // Adjacent shots differ by 1 pre-finalize; the three multiply-xorshift
    // rounds provide full 64-bit avalanche so post-finalize outputs are
    // statistically independent despite the linear pre-image spacing.
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(shot)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shot_seed_is_deterministic_and_distinct() {
        assert_eq!(shot_seed(7, 3), shot_seed(7, 3));
        assert_ne!(shot_seed(7, 3), shot_seed(7, 4));
        assert_ne!(shot_seed(7, 3), shot_seed(8, 3));
    }
}

#[cfg(test)]
mod apply_tests {
    use super::*;
    use crate::noise::error::{
        amplitude_damping_error, depolarizing_error, pauli_error, phase_damping_error, ReadoutError,
    };
    use aleph_core::{AlignedBuf, Complex};
    use rand::{rngs::StdRng, SeedableRng};

    /// A Pauli channel that applies X with probability 1 must turn |0⟩ into |1⟩
    /// (deterministic — no dependence on the RNG branch).
    #[test]
    fn certain_x_flips_basis_state() {
        let mut amps = AlignedBuf::from_slice(&[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        let err = pauli_error(&[("X", 1.0)]);
        let mut rng = StdRng::seed_from_u64(0);
        apply_channel(&mut amps, 1, &err, &[0], &mut rng);
        assert!((amps[0].norm() - 0.0).abs() < 1e-12);
        assert!((amps[1].norm() - 1.0).abs() < 1e-12);
    }

    /// Identity-weight-1 Pauli channel leaves the state untouched and consumes
    /// no normalization (norm stays exactly 1).
    #[test]
    fn certain_identity_is_noop() {
        let mut amps = AlignedBuf::from_slice(&[Complex::new(0.6, 0.0), Complex::new(0.8, 0.0)]);
        let err = pauli_error(&[("I", 1.0)]);
        let mut rng = StdRng::seed_from_u64(0);
        apply_channel(&mut amps, 1, &err, &[0], &mut rng);
        assert!((amps[0] - Complex::new(0.6, 0.0)).norm() < 1e-12);
        assert!((amps[1] - Complex::new(0.8, 0.0)).norm() < 1e-12);
    }

    /// 2q depolarizing applies a 2-qubit Pauli (tensor product) via two 1q
    /// kernels; the state must stay normalized.
    #[test]
    fn depolarizing_2q_stays_normalized() {
        let mut amps = AlignedBuf::from_slice(&[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ]);
        let err = depolarizing_error(0.5, 2);
        let mut rng = StdRng::seed_from_u64(42);
        apply_channel(&mut amps, 2, &err, &[0, 1], &mut rng);
        let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-12, "norm {norm}");
    }

    /// Amplitude damping with γ=1 sends |1⟩ → |0⟩ deterministically (the only
    /// branch with nonzero probability is K₁).
    #[test]
    fn amplitude_damping_gamma1_resets_excited_state() {
        let mut amps = AlignedBuf::from_slice(&[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]);
        let err = amplitude_damping_error(1.0);
        let mut rng = StdRng::seed_from_u64(1);
        apply_channel(&mut amps, 1, &err, &[0], &mut rng);
        assert!(
            (amps[0].norm() - 1.0).abs() < 1e-12,
            "|0⟩ amp {}",
            amps[0].norm()
        );
        assert!(amps[1].norm() < 1e-12);
    }

    /// Quantum-jump must preserve normalization: after applying a general
    /// channel and renormalizing, ‖state‖ = 1 for any seed and any γ.
    #[test]
    fn general_channel_preserves_norm() {
        for (seed, gamma) in [(0u64, 0.2), (1, 0.5), (2, 0.8), (3, 0.99)] {
            let mut amps =
                AlignedBuf::from_slice(&[Complex::new(0.6, 0.0), Complex::new(0.0, 0.8)]);
            let err = amplitude_damping_error(gamma);
            let mut rng = StdRng::seed_from_u64(seed);
            apply_channel(&mut amps, 1, &err, &[0], &mut rng);
            let n: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            assert!((n - 1.0).abs() < 1e-10, "seed {seed} γ {gamma}: norm {n}");
        }
    }

    #[test]
    fn phase_damping_preserves_norm() {
        let mut amps = AlignedBuf::from_slice(&[Complex::new(0.5, 0.5), Complex::new(0.5, -0.5)]);
        let err = phase_damping_error(0.6);
        let mut rng = StdRng::seed_from_u64(7);
        apply_channel(&mut amps, 1, &err, &[0], &mut rng);
        let n: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((n - 1.0).abs() < 1e-10, "norm {n}");
    }

    /// Amplitude damping acts as identity (up to the K₀ scale that is 1 on the
    /// |0⟩ component) when the state is exactly |0⟩: only K₀ has nonzero
    /// probability and K₀|0⟩ = |0⟩. The state must remain |0⟩, not permute.
    #[test]
    fn amplitude_damping_fixes_ground_state() {
        let mut amps = AlignedBuf::from_slice(&[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        let err = amplitude_damping_error(0.5);
        let mut rng = StdRng::seed_from_u64(3);
        apply_channel(&mut amps, 1, &err, &[0], &mut rng);
        assert!(
            (amps[0] - Complex::new(1.0, 0.0)).norm() < 1e-12,
            "amp0 {:?}",
            amps[0]
        );
        assert!(amps[1].norm() < 1e-12, "amp1 {:?}", amps[1]);
    }

    /// Readout error with P(1|0)=1 and P(0|1)=1 flips every measured bit.
    #[test]
    fn readout_flips_all_bits_when_certain() {
        let ro = ReadoutError::new([[0.0, 1.0], [1.0, 0.0]]);
        let map: std::collections::HashMap<u32, ReadoutError> =
            [(0u32, ro), (1u32, ro)].into_iter().collect();
        let mut rng = StdRng::seed_from_u64(0);
        // basis state |01⟩ = index 0b01 = 1 over 2 qubits → both bits flip → |10⟩ = 2
        let out = apply_readout(1, 2, &map, &mut rng);
        assert_eq!(out, 0b10);
    }

    /// Identity readout (perfect measurement) is the identity on the index.
    #[test]
    fn readout_identity_is_noop() {
        let ro = ReadoutError::new([[1.0, 0.0], [0.0, 1.0]]);
        let map: std::collections::HashMap<u32, ReadoutError> =
            [(0u32, ro), (1u32, ro), (2u32, ro)].into_iter().collect();
        let mut rng = StdRng::seed_from_u64(123);
        assert_eq!(apply_readout(0b101, 3, &map, &mut rng), 0b101);
    }
}
