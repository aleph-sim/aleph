//! Host-side measurement / sampling / expectation / marginals for the CUDA SV
//! backend. The first version downloads the amplitudes to the host and reduces
//! on the CPU — simplest correct path; a GPU reduction is a later optimization
//! (P5-04/05). The algorithms mirror `aleph-sv`'s `measure.rs` exactly (same
//! degenerate-branch threshold and norm drift budget) so the distribution
//! oracle agrees with the CPU backend.

use aleph_backend::BackendError;
use aleph_core::{Complex, Pauli, PauliString};
use rand::{rngs::StdRng, Rng};

/// See `aleph_sv::measure::DEGENERATE_BRANCH_THRESHOLD`.
const DEGENERATE_BRANCH_THRESHOLD: f64 = 1e-300;

/// Validate finiteness + normalization and return the per-amplitude `|aᵢ|²`.
fn validate(amps: &[Complex], num_qubits: u32) -> Result<Vec<f64>, BackendError> {
    let n = amps.len();
    if n == 0 {
        return Err(BackendError::InvalidState {
            reason: "empty state vector",
        });
    }
    let expected = 1usize
        .checked_shl(num_qubits)
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
    for a in amps {
        let p = a.norm_sqr();
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

/// Collapse `amps` onto the measured branch of `qubit`, returning the outcome.
/// Mutates the host amplitude buffer in place; the caller re-uploads it.
pub(crate) fn measure(
    rng: &mut StdRng,
    amps: &mut [Complex],
    num_qubits: u32,
    qubit: u32,
) -> Result<bool, BackendError> {
    if qubit >= num_qubits {
        return Err(BackendError::QubitOutOfRange { qubit, num_qubits });
    }
    let probs = validate(amps, num_qubits)?;
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
    let scale = 1.0 / p.sqrt();
    for (i, a) in amps.iter_mut().enumerate() {
        if (i & q_bit != 0) == outcome {
            a.re *= scale;
            a.im *= scale;
        } else {
            *a = Complex::new(0.0, 0.0);
        }
    }
    Ok(outcome)
}

/// Draw `shots` basis-state samples via inverse-CDF over `|aᵢ|²`.
pub(crate) fn sample(
    rng: &mut StdRng,
    amps: &[Complex],
    num_qubits: u32,
    shots: u32,
) -> Result<Vec<u64>, BackendError> {
    let probs = validate(amps, num_qubits)?;
    let mut cdf = Vec::with_capacity(probs.len());
    let mut acc = 0.0_f64;
    for &p in &probs {
        acc += p;
        cdf.push(acc);
    }
    let total = acc;
    let mut out = Vec::with_capacity(shots as usize);
    for _ in 0..shots {
        let r = rng.gen::<f64>() * total;
        let idx = cdf.partition_point(|&c| c <= r).min(probs.len() - 1);
        out.push(idx as u64);
    }
    Ok(out)
}

/// `⟨ψ| c·P |ψ⟩`. Diagonal (Z/I-only) strings use the no-copy fast path;
/// mixed X/Y strings apply each Pauli to a host copy and take `Re⟨ψ|φ⟩`.
pub(crate) fn expectation_value(
    amps: &[Complex],
    num_qubits: u32,
    pauli: &PauliString,
) -> Result<f64, BackendError> {
    let probs = validate(amps, num_qubits)?;
    if !pauli.coefficient.is_finite() {
        return Err(BackendError::InvalidPauliString {
            reason: "non-finite coefficient",
        });
    }
    let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
    for (q, _) in &pauli.terms {
        if *q >= num_qubits {
            return Err(BackendError::QubitOutOfRange {
                qubit: *q,
                num_qubits,
            });
        }
        if seen.contains(q) {
            return Err(BackendError::DuplicateQubit { qubit: *q });
        }
        seen.push(*q);
    }
    let mut z_mask = 0u64;
    let mut all_z_or_i = true;
    for (q, p) in &pauli.terms {
        if *q >= 64 {
            all_z_or_i = false;
            break;
        }
        match p {
            Pauli::I => {}
            Pauli::Z => z_mask |= 1u64 << q,
            Pauli::X | Pauli::Y => {
                all_z_or_i = false;
                break;
            }
        }
    }
    if all_z_or_i {
        let mut acc = 0.0_f64;
        for (i, &p) in probs.iter().enumerate() {
            if (i as u64 & z_mask).count_ones() & 1 == 0 {
                acc += p;
            } else {
                acc -= p;
            }
        }
        return Ok(pauli.coefficient * acc);
    }
    let mut tmp = amps.to_vec();
    for (q, p) in &pauli.terms {
        if *p == Pauli::I {
            continue;
        }
        host_apply_1q(&mut tmp, *q, &p.matrix());
    }
    let mut acc = Complex::new(0.0, 0.0);
    for (lhs, rhs) in amps.iter().zip(tmp.iter()) {
        acc += lhs.conj() * *rhs;
    }
    Ok(pauli.coefficient * acc.re)
}

/// Marginal probabilities over `qubits`, length `2^qubits.len()`, indexed with
/// `qubits[0]` as LSB (global convention).
pub(crate) fn probabilities(
    amps: &[Complex],
    num_qubits: u32,
    qubits: &[u32],
) -> Result<Vec<f64>, BackendError> {
    let probs = validate(amps, num_qubits)?;
    let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
    for &q in qubits {
        if q >= num_qubits {
            return Err(BackendError::QubitOutOfRange {
                qubit: q,
                num_qubits,
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
        let mut bin = 0usize;
        for (pos, &q) in qubits.iter().enumerate() {
            if (i >> q) & 1 == 1 {
                bin |= 1usize << pos;
            }
        }
        out[bin] += p;
    }
    Ok(out)
}

/// Standard 1q butterfly on a host amplitude buffer (slow expectation path).
fn host_apply_1q(amps: &mut [Complex], q: u32, m: &[[Complex; 2]; 2]) {
    let t_bit = 1usize << q;
    let mut i = 0usize;
    while i < amps.len() {
        if i & t_bit == 0 {
            let j = i | t_bit;
            let a = amps[i];
            let b = amps[j];
            amps[i] = m[0][0] * a + m[0][1] * b;
            amps[j] = m[1][0] * a + m[1][1] * b;
        }
        i += 1;
    }
}
