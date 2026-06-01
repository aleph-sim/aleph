//! Measurement, sampling, expectation, marginals over `SoaState`.
//!
//! Mirrors `crate::measure` 1-to-1 except for index arithmetic:
//! probabilities are `re[i]² + im[i]²`, collapse zeros both `re[i]`
//! and `im[i]`, etc. ADR 0006 NaN discipline applies — every FP
//! comparison is guarded by an explicit `.is_finite()` check.

use aleph_backend::BackendError;
use rand::{rngs::StdRng, Rng};

use crate::soa_state::SoaState;

/// Same degeneracy floor as `measure::DEGENERATE_BRANCH_THRESHOLD`.
const DEGENERATE_BRANCH_THRESHOLD: f64 = 1e-300;

/// SoA analogue of `measure::validate_state`. Walks both `re` and
/// `im`, rejects empty / mismatched-length / non-finite / un-normalised
/// states. Returns the per-amp `re² + im²` vector so the four query
/// methods share a single pass — same pattern as the AoS path.
pub(crate) fn validate_state_soa(state: &SoaState) -> Result<Vec<f64>, BackendError> {
    let n = state.re.len();
    if n == 0 {
        return Err(BackendError::InvalidState {
            reason: "empty state vector",
        });
    }
    if state.im.len() != n {
        return Err(BackendError::InvalidState {
            reason: "re.len() != im.len()",
        });
    }
    let expected = 1usize
        .checked_shl(state.num_qubits)
        .ok_or(BackendError::InvalidState {
            reason: "num_qubits exceeds platform usize::BITS",
        })?;
    if n != expected {
        return Err(BackendError::InvalidState {
            reason: "re.len() != 2^num_qubits",
        });
    }
    let mut probs = Vec::with_capacity(n);
    let mut total = 0.0_f64;
    for (r, i) in state.re.iter().zip(state.im.iter()) {
        let p = r * r + i * i;
        if !p.is_finite() {
            return Err(BackendError::InvalidState {
                reason: "non-finite amplitude norm²",
            });
        }
        total += p;
        probs.push(p);
    }
    let drift_budget = (n as f64).sqrt() * aleph_core::AMPLITUDE_TOL;
    if (total - 1.0).abs() > drift_budget {
        return Err(BackendError::InvalidState {
            reason: "state norm² deviates from 1 beyond drift budget",
        });
    }
    Ok(probs)
}

pub(crate) fn measure_impl_soa(
    rng: &mut StdRng,
    state: &mut SoaState,
    qubit: u32,
) -> Result<bool, BackendError> {
    let n = state.num_qubits;
    if qubit >= n {
        return Err(BackendError::QubitOutOfRange {
            qubit,
            num_qubits: n,
        });
    }
    let probs = validate_state_soa(state)?;
    let q_bit = 1usize << qubit;
    let mut p0 = 0.0_f64;
    let mut p1 = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() {
        if i & q_bit != 0 {
            p1 += p;
        } else {
            p0 += p;
        }
    }
    let p0 = p0.clamp(0.0, 1.0);
    let p1 = p1.clamp(0.0, 1.0);
    let one_degen = p1 < DEGENERATE_BRANCH_THRESHOLD;
    let zero_degen = p0 < DEGENERATE_BRANCH_THRESHOLD;
    let (outcome, p) = match (zero_degen, one_degen) {
        (true, true) => {
            return Err(BackendError::DegenerateMeasurement {
                qubit,
                probability: p1.max(p0),
            });
        }
        (true, false) => (true, p1),
        (false, true) => (false, p0),
        (false, false) => {
            let outcome = rng.gen::<f64>() < p1;
            let p = if outcome { p1 } else { p0 };
            (outcome, p)
        }
    };
    let norm = p.sqrt();
    for i in 0..state.re.len() {
        let bit_set = (i & q_bit) != 0;
        if bit_set == outcome {
            state.re[i] /= norm;
            state.im[i] /= norm;
        } else {
            state.re[i] = 0.0;
            state.im[i] = 0.0;
        }
    }
    Ok(outcome)
}

pub(crate) fn sample_impl_soa(
    rng: &mut StdRng,
    state: &SoaState,
    shots: u32,
) -> Result<Vec<u64>, BackendError> {
    let probs = validate_state_soa(state)?;
    let table = crate::sampling::AliasTable::build(&probs);
    let mut out = Vec::with_capacity(shots as usize);
    for _ in 0..shots {
        out.push(table.draw(rng) as u64);
    }
    Ok(out)
}

pub(crate) fn expectation_value_impl_soa(
    state: &SoaState,
    pauli: &aleph_core::PauliString,
) -> Result<f64, BackendError> {
    let n = state.num_qubits;
    let probs = validate_state_soa(state)?;
    if !pauli.coefficient.is_finite() {
        return Err(BackendError::InvalidPauliString {
            reason: "non-finite coefficient",
        });
    }
    let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
    for (q, _) in &pauli.terms {
        if *q >= n {
            return Err(BackendError::QubitOutOfRange {
                qubit: *q,
                num_qubits: n,
            });
        }
        if seen.contains(q) {
            return Err(BackendError::DuplicateQubit { qubit: *q });
        }
        seen.push(*q);
    }
    // Z-fast-path (same algorithm as AoS: diagonal Pauli string evaluated
    // from `probs` alone, no state clone, no kernel apply).
    let mut z_mask = 0u64;
    let mut all_z_or_i = true;
    for (q, p) in &pauli.terms {
        if *q >= 64 {
            all_z_or_i = false;
            break;
        }
        match p {
            aleph_core::Pauli::I => {}
            aleph_core::Pauli::Z => z_mask |= 1u64 << q,
            aleph_core::Pauli::X | aleph_core::Pauli::Y => {
                all_z_or_i = false;
                break;
            }
        }
    }
    if all_z_or_i {
        let mut acc = 0.0_f64;
        for (i, &p) in probs.iter().enumerate() {
            let sign = if (i as u64 & z_mask).count_ones() & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            acc += sign * p;
        }
        return Ok(pauli.coefficient * acc);
    }
    // Slow path: clone both (re, im) buffers, apply each non-identity
    // Pauli as a 1q gate over the SoA kernel, accumulate Re(⟨ψ|φ⟩) =
    // Σ (re[i]·new_re[i] + im[i]·new_im[i]).
    let mut tmp_re = state.re.clone();
    let mut tmp_im = state.im.clone();
    for (q, p) in &pauli.terms {
        if *p == aleph_core::Pauli::I {
            continue;
        }
        let m = p.matrix();
        crate::kernels::soa::apply_1q(&mut tmp_re, &mut tmp_im, *q, &[], &m);
    }
    let mut acc = 0.0_f64;
    for i in 0..state.re.len() {
        // Re(conj(a) * b) = a.re*b.re + a.im*b.im
        acc += state.re[i] * tmp_re[i] + state.im[i] * tmp_im[i];
    }
    Ok(pauli.coefficient * acc)
}

pub(crate) fn probabilities_impl_soa(
    state: &SoaState,
    qubits: &[u32],
) -> Result<Vec<f64>, BackendError> {
    let n = state.num_qubits;
    let probs = validate_state_soa(state)?;
    let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
    for &q in qubits {
        if q >= n {
            return Err(BackendError::QubitOutOfRange {
                qubit: q,
                num_qubits: n,
            });
        }
        if seen.contains(&q) {
            return Err(BackendError::DuplicateQubit { qubit: q });
        }
        seen.push(q);
    }
    if qubits.is_empty() {
        return Ok(vec![1.0]);
    }
    let out_dim = 1usize << qubits.len();
    let mut out = vec![0.0_f64; out_dim];
    for (i, p) in probs.iter().enumerate() {
        let mut k = 0usize;
        for (pos, &q) in qubits.iter().enumerate() {
            if (i >> q) & 1 == 1 {
                k |= 1usize << pos;
            }
        }
        out[k] += *p;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Pauli, PauliString};
    use rand::SeedableRng;

    fn zero_ket(n: u32) -> SoaState {
        let dim = 1usize << n;
        let mut re = aleph_core::AlignedBuf::<f64>::zeroed(dim);
        re[0] = 1.0;
        SoaState {
            num_qubits: n,
            re,
            im: aleph_core::AlignedBuf::<f64>::zeroed(dim),
        }
    }

    #[test]
    fn validate_rejects_empty_state() {
        let s = SoaState {
            num_qubits: 0,
            re: aleph_core::AlignedBuf::from_slice(&[]),
            im: aleph_core::AlignedBuf::from_slice(&[]),
        };
        assert!(matches!(
            validate_state_soa(&s),
            Err(BackendError::InvalidState { .. })
        ));
    }

    #[test]
    fn validate_rejects_mismatched_lens() {
        let s = SoaState {
            num_qubits: 1,
            re: aleph_core::AlignedBuf::from_slice(&[1.0, 0.0]),
            im: aleph_core::AlignedBuf::from_slice(&[0.0]),
        };
        assert!(matches!(
            validate_state_soa(&s),
            Err(BackendError::InvalidState { .. })
        ));
    }

    #[test]
    fn validate_rejects_nan_amplitude() {
        let mut s = zero_ket(1);
        s.re[1] = f64::NAN;
        assert!(matches!(
            validate_state_soa(&s),
            Err(BackendError::InvalidState { .. })
        ));
    }

    #[test]
    fn validate_rejects_unnormalised_state() {
        let s = SoaState {
            num_qubits: 1,
            re: aleph_core::AlignedBuf::from_slice(&[0.9, 0.9]),
            im: aleph_core::AlignedBuf::from_slice(&[0.0, 0.0]),
        };
        assert!(matches!(
            validate_state_soa(&s),
            Err(BackendError::InvalidState { .. })
        ));
    }

    #[test]
    fn measure_zero_state_returns_false() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut s = zero_ket(2);
        let outcome = measure_impl_soa(&mut rng, &mut s, 0).unwrap();
        assert!(!outcome);
        assert_eq!(s.re[0], 1.0);
    }

    #[test]
    fn sample_zero_state_only_returns_zero() {
        let mut rng = StdRng::seed_from_u64(0);
        let s = zero_ket(3);
        let shots = sample_impl_soa(&mut rng, &s, 100).unwrap();
        assert!(shots.iter().all(|&v| v == 0));
        assert_eq!(shots.len(), 100);
    }

    #[test]
    fn expectation_z_on_zero_is_plus_one() {
        let s = zero_ket(1);
        let z = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        let ev = expectation_value_impl_soa(&s, &z).unwrap();
        assert!((ev - 1.0).abs() < 1e-12);
    }

    #[test]
    fn expectation_x_on_zero_is_zero_slow_path() {
        let s = zero_ket(1);
        let x = PauliString::new(1.0, vec![(0, Pauli::X)]).unwrap();
        let ev = expectation_value_impl_soa(&s, &x).unwrap();
        assert!(ev.abs() < 1e-12);
    }

    #[test]
    fn probabilities_zero_state_full_basis() {
        let s = zero_ket(2);
        let p = probabilities_impl_soa(&s, &[0, 1]).unwrap();
        assert_eq!(p, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn probabilities_empty_subset_is_one() {
        let s = zero_ket(2);
        assert_eq!(probabilities_impl_soa(&s, &[]).unwrap(), vec![1.0]);
    }

    #[test]
    fn probabilities_out_of_range_rejected() {
        let s = zero_ket(2);
        let err = probabilities_impl_soa(&s, &[5]).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 2,
            }
        );
    }
}
