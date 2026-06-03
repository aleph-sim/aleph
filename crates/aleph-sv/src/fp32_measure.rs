//! Measurement, sampling, expectation, marginals for the f32 backend.
//!
//! Mirror of [`crate::measure`] (the f64 `CpuState` path) over
//! [`crate::Fp32CpuState`]. **Accuracy rule:** every probability / norm /
//! expectation accumulation widens each `Complex<f32>` amplitude to f64
//! (`as f64`) before squaring or multiplying. Summing 2^n single-precision
//! terms in f32 invites catastrophic cancellation; the f64 accumulator keeps
//! the norm / marginal sums honest even though the state itself is f32.

use aleph_backend::BackendError;
use aleph_core::Complex;
use rand::{rngs::StdRng, Rng};

use crate::Fp32CpuState;

/// Threshold under which we refuse to collapse the state. Mirrors the f64
/// path: collapsing on a `< 1e-300` branch scales amplitudes by `≈ 1e150`.
const DEGENERATE_BRANCH_THRESHOLD: f64 = 1e-300;

/// Relaxed normalization tolerance for the f32 path. The f64 path uses
/// `AMPLITUDE_TOL = 1e-10` with a `√n` drift budget; f32-accumulated states
/// drift far more, so we accept `|Σp − 1| < 1e-4` here (per the P2-08 plan).
const FP32_NORM_TOL: f64 = 1e-4;

/// Widen one f32 amplitude's `|a|²` to f64. Squaring in f64 (after widening
/// both components) avoids the f32 cancellation that the norm sum is prone to.
#[inline]
fn norm_sqr_f64(a: Complex<f32>) -> f64 {
    let re = a.re as f64;
    let im = a.im as f64;
    re * re + im * im
}

/// Walk every amplitude once: reject empty/non-finite/non-normalised states,
/// returning the per-amp `|a|²` (computed in f64) so callers avoid a second
/// pass. Same shared-preamble role as [`crate::measure::validate_state`].
pub(crate) fn validate_state_f32(state: &Fp32CpuState) -> Result<Vec<f64>, BackendError> {
    let n = state.amps.len();
    if n == 0 {
        return Err(BackendError::InvalidState {
            reason: "empty state vector",
        });
    }
    let expected = 1usize
        .checked_shl(state.num_qubits)
        .ok_or(BackendError::InvalidState {
            reason: "num_qubits exceeds platform usize::BITS",
        })?;
    if n != expected {
        return Err(BackendError::InvalidState {
            reason: "amps.len() != 2^num_qubits",
        });
    }
    let mut probs = Vec::with_capacity(n);
    let mut total = 0.0_f64;
    for a in state.amps.iter() {
        let p = norm_sqr_f64(*a);
        if !p.is_finite() {
            return Err(BackendError::InvalidState {
                reason: "non-finite amplitude norm²",
            });
        }
        total += p;
        probs.push(p);
    }
    // f32 states drift more than f64 — use the relaxed absolute tolerance.
    if (total - 1.0).abs() > FP32_NORM_TOL {
        return Err(BackendError::InvalidState {
            reason: "state norm² deviates from 1 beyond f32 tolerance",
        });
    }
    Ok(probs)
}

pub(crate) fn measure_impl_f32(
    rng: &mut StdRng,
    state: &mut Fp32CpuState,
    qubit: u32,
) -> Result<bool, BackendError> {
    let n = state.num_qubits;
    if qubit >= n {
        return Err(BackendError::QubitOutOfRange {
            qubit,
            num_qubits: n,
        });
    }
    let probs = validate_state_f32(state)?;
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
    // Compute the renormalisation in f64, then cast `1/√p` to f32 for the
    // in-place scale (the state is single-precision).
    let scale = (1.0 / p.sqrt()) as f32;
    for (i, a) in state.amps.iter_mut().enumerate() {
        let bit_set = (i & q_bit) != 0;
        if bit_set == outcome {
            a.re *= scale;
            a.im *= scale;
        } else {
            *a = Complex::<f32>::new(0.0, 0.0);
        }
    }
    Ok(outcome)
}

/// Sample basis-state indices from `|amps[i]|²` via the alias table.
///
/// Mirrors [`crate::measure::sample_impl`]; the alias table is built from
/// the f64-widened probability vector for numerical fidelity.
pub(crate) fn sample_impl_f32(
    rng: &mut StdRng,
    state: &Fp32CpuState,
    shots: u32,
) -> Result<Vec<u64>, BackendError> {
    let probs = validate_state_f32(state)?;
    let table = crate::sampling::AliasTable::build(&probs);
    let mut out = Vec::with_capacity(shots as usize);
    for _ in 0..shots {
        out.push(table.draw(rng) as u64);
    }
    Ok(out)
}

/// Naive expectation value over the f32 state. Z-only Pauli strings take the
/// diagonal fast path; mixed strings fall back to a direct-index slow path.
/// All inner products accumulate in f64 (rule 9).
pub(crate) fn expectation_value_impl_f32(
    state: &Fp32CpuState,
    pauli: &aleph_core::PauliString,
) -> Result<f64, BackendError> {
    let n = state.num_qubits;
    let probs = validate_state_f32(state)?;
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
    // Pauli-Z fast path: ⟨ψ| ⊗ᵢ Zᵢ |ψ⟩ = Σᵢ (-1)^popcount(i & z_mask) · |aᵢ|².
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
        return Ok(pauli.coefficient * expectation_z_diag(&probs, z_mask));
    }
    // Slow path: build the transformed state by direct amplitude inspection
    // (Z = diag(+1,-1), X swaps i↔i^bit, Y = X⊗diag(i,-i)), then take
    // Re(⟨ψ|φ⟩). Widen each f32 product to f64 before accumulating.
    let mut tmp: Vec<Complex<f32>> = state.amps.to_vec();
    for (q, p) in &pauli.terms {
        let qbit = 1usize << q;
        match p {
            aleph_core::Pauli::I => {}
            aleph_core::Pauli::Z => {
                for (i, a) in tmp.iter_mut().enumerate() {
                    if i & qbit != 0 {
                        *a = -*a;
                    }
                }
            }
            aleph_core::Pauli::X => {
                for i in 0..tmp.len() {
                    if i & qbit == 0 {
                        tmp.swap(i, i | qbit);
                    }
                }
            }
            aleph_core::Pauli::Y => {
                for i in 0..tmp.len() {
                    if i & qbit == 0 {
                        let j = i | qbit;
                        let a = tmp[i];
                        let b = tmp[j];
                        tmp[i] = Complex::<f32>::new(b.im, -b.re); // -i * b
                        tmp[j] = Complex::<f32>::new(-a.im, a.re); // +i * a
                    }
                }
            }
        }
    }
    let mut acc_re = 0.0_f64;
    for (lhs, rhs) in state.amps.iter().zip(tmp.iter()) {
        // Re(conj(lhs) * rhs) = lhs.re*rhs.re + lhs.im*rhs.im, accumulated in f64.
        acc_re += (lhs.re as f64) * (rhs.re as f64) + (lhs.im as f64) * (rhs.im as f64);
    }
    Ok(pauli.coefficient * acc_re)
}

/// Marginal probabilities over the named qubit subset. Output length
/// `2^qubits.len()`, indexed with `qubits[0]` as LSB (global convention).
pub(crate) fn probabilities_impl_f32(
    state: &Fp32CpuState,
    qubits: &[u32],
) -> Result<Vec<f64>, BackendError> {
    let n = state.num_qubits;
    let probs = validate_state_f32(state)?;
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

/// Diagonal `⟨ψ| ⊗ᵢ Zᵢ |ψ⟩` over precomputed f64 `probs`. Identity terms
/// contribute nothing to `z_mask`.
fn expectation_z_diag(probs: &[f64], z_mask: u64) -> f64 {
    let mut acc = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() {
        let sign = if (i as u64 & z_mask).count_ones() & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        acc += sign * p;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fp32SvBackend;
    use aleph_backend::Backend;
    use aleph_core::{Gate, GateInstance, Pauli, PauliString};
    use smallvec::smallvec;

    #[test]
    fn measure_zero_state_returns_false() {
        let mut b = Fp32SvBackend::with_seed(42);
        let mut s = b.allocate(2).unwrap();
        let outcome = b.measure(&mut s, 0).unwrap();
        assert!(!outcome);
        assert_eq!(s.amplitudes()[0], Complex::<f32>::new(1.0, 0.0));
    }

    #[test]
    fn measure_plus_state_collapses_to_basis() {
        let mut b = Fp32SvBackend::with_seed(123);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let outcome = b.measure(&mut s, 0).unwrap();
        let a = s.amplitudes();
        if outcome {
            assert!((a[1].norm() - 1.0).abs() < 1e-5);
            assert!(a[0].norm() < 1e-6);
        } else {
            assert!((a[0].norm() - 1.0).abs() < 1e-5);
            assert!(a[1].norm() < 1e-6);
        }
    }

    #[test]
    fn measure_qubit_out_of_range() {
        let mut b = Fp32SvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        let err = b.measure(&mut s, 5).unwrap_err();
        assert_eq!(
            err,
            BackendError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 1,
            }
        );
    }

    #[test]
    fn probabilities_plus_state_uniform() {
        let mut b = Fp32SvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let p = b.probabilities(&s, &[0]).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sample_bell_state_only_returns_00_or_11() {
        let mut b = Fp32SvBackend::with_seed(7);
        let mut s = b.allocate(2).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        b.apply_gate(
            &mut s,
            &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
        )
        .unwrap();
        let shots = b.sample(&s, 1000).unwrap();
        assert!(shots.iter().all(|&v| v == 0 || v == 3));
    }

    #[test]
    fn expectation_z_on_zero_is_plus_one() {
        let mut b = Fp32SvBackend::with_seed(0);
        let s = b.allocate(1).unwrap();
        let z = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        let ev = b.expectation_value(&s, &z).unwrap();
        assert!((ev - 1.0).abs() < 1e-6);
    }

    #[test]
    fn expectation_x_on_plus_is_plus_one() {
        let mut b = Fp32SvBackend::with_seed(0);
        let mut s = b.allocate(1).unwrap();
        b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
            .unwrap();
        let x = PauliString::new(1.0, vec![(0, Pauli::X)]).unwrap();
        let ev = b.expectation_value(&s, &x).unwrap();
        assert!((ev - 1.0).abs() < 1e-5);
    }

    #[test]
    fn expectation_zz_sign_table() {
        let cases: &[(u32, f64)] = &[(0b00, 1.0), (0b01, -1.0), (0b10, -1.0), (0b11, 1.0)];
        for &(basis, expected) in cases {
            let mut b = Fp32SvBackend::with_seed(0);
            let mut s = b.allocate(2).unwrap();
            if basis & 0b01 != 0 {
                b.apply_gate(&mut s, &GateInstance::new(Gate::X, smallvec![0u32]))
                    .unwrap();
            }
            if basis & 0b10 != 0 {
                b.apply_gate(&mut s, &GateInstance::new(Gate::X, smallvec![1u32]))
                    .unwrap();
            }
            let zz = PauliString::new(1.0, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
            let ev = b.expectation_value(&s, &zz).unwrap();
            assert!(
                (ev - expected).abs() < 1e-6,
                "basis {basis:02b}: got {ev}, want {expected}"
            );
        }
    }
}
