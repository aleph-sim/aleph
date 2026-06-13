//! Channel application (quantum-jump) and per-shot RNG seeding.

use aleph_core::{Complex, Pauli};
use rand::{rngs::StdRng, Rng};

use super::error::{KrausChannel, PauliChannel, QuantumError};

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

fn apply_kraus_1q(_amps: &mut [Complex], _c: &KrausChannel, _q: u32, _rng: &mut StdRng) {
    unimplemented!("general Kraus path — Task 4")
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
    use crate::noise::error::{depolarizing_error, pauli_error};
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
}
