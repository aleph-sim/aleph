//! Host-side readout over the unified-memory FP32 statevector. On Apple
//! Silicon the buffer is zero-copy CPU-visible after the GPU completes, so
//! probabilities / sampling / measurement / expectation run on the host with
//! no GPU reduction. Accumulations widen each `Complex<f32>` to f64 to avoid
//! catastrophic cancellation when summing 2^n single-precision terms (mirrors
//! `aleph-sv::fp32_measure`).

use aleph_backend::BackendError;
use aleph_core::{Complex, Pauli, PauliString};
use rand::{rngs::StdRng, Rng};

use super::state::MetalSvState;

/// Branch probability below which collapse is refused (scaling by ~1/√p would
/// explode). Mirrors the CPU path.
const DEGENERATE_BRANCH_THRESHOLD: f64 = 1e-300;
/// Relaxed normalization tolerance for an f32-accumulated state.
const FP32_NORM_TOL: f64 = 1e-4;

/// `|a|²` of one f32 amplitude, computed in f64. Widening both components
/// before squaring avoids the f32 cancellation a naive `a.norm_sqr()` would
/// suffer when these terms are summed over 2^n amplitudes (mirrors aleph-sv).
#[inline]
fn norm_sqr_f64(a: Complex<f32>) -> f64 {
    let re = a.re as f64;
    let im = a.im as f64;
    re * re + im * im
}

/// Validate dimensions/finiteness/normalization; return per-amplitude `|a|²`
/// (computed in f64) so callers avoid a second pass.
fn validate(state: &MetalSvState) -> Result<Vec<f64>, BackendError> {
    let amps = state.amplitudes_f32();
    let len = amps.len();
    if len == 0 {
        return Err(BackendError::InvalidState {
            reason: "empty state vector",
        });
    }
    let expected = 1usize
        .checked_shl(state.num_qubits)
        .ok_or(BackendError::InvalidState {
            reason: "num_qubits exceeds platform usize::BITS",
        })?;
    if len != expected {
        return Err(BackendError::InvalidState {
            reason: "amps.len() != 2^num_qubits",
        });
    }
    let mut probs = Vec::with_capacity(len);
    let mut total = 0.0_f64;
    for &a in amps {
        let p = norm_sqr_f64(a);
        if !p.is_finite() {
            return Err(BackendError::InvalidState {
                reason: "non-finite amplitude norm²",
            });
        }
        total += p;
        probs.push(p);
    }
    if (total - 1.0).abs() > FP32_NORM_TOL {
        return Err(BackendError::InvalidState {
            reason: "state norm² deviates from 1 beyond f32 tolerance",
        });
    }
    Ok(probs)
}

/// Marginal probabilities over `qubits`. Output length `2^qubits.len()`,
/// indexed with `qubits[0]` as the LSB (global convention).
pub(crate) fn probabilities(
    state: &MetalSvState,
    qubits: &[u32],
) -> Result<Vec<f64>, BackendError> {
    let n = state.num_qubits;
    let probs = validate(state)?;
    let mut seen: Vec<u32> = Vec::new();
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

/// Sample basis-state indices from `|amps|²` via an inverse-CDF scan. Simple
/// (no alias table) — adequate at this scale and deterministic for a seed.
pub(crate) fn sample(
    rng: &mut StdRng,
    state: &MetalSvState,
    shots: u32,
) -> Result<Vec<u64>, BackendError> {
    let probs = validate(state)?;
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

/// Measure `qubit`, collapse the buffer in place, renormalize. Returns the
/// outcome bit.
pub(crate) fn measure(
    rng: &mut StdRng,
    state: &mut MetalSvState,
    qubit: u32,
) -> Result<bool, BackendError> {
    let n = state.num_qubits;
    if qubit >= n {
        return Err(BackendError::QubitOutOfRange {
            qubit,
            num_qubits: n,
        });
    }
    let probs = validate(state)?;
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
            })
        }
        (true, false) => (true, p1),
        (false, true) => (false, p0),
        (false, false) => {
            let outcome = rng.gen::<f64>() < p1;
            let p = if outcome { p1 } else { p0 };
            (outcome, p)
        }
    };
    let scale = (1.0 / p.sqrt()) as f32;
    for (i, a) in state.amps.as_mut_slice().iter_mut().enumerate() {
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

/// `⟨ψ|cP|ψ⟩` over the f32 state, accumulated in f64. Z-only strings take the
/// diagonal fast path; mixed strings build the transformed state directly.
pub(crate) fn expectation_value(
    state: &MetalSvState,
    pauli: &PauliString,
) -> Result<f64, BackendError> {
    let n = state.num_qubits;
    let probs = validate(state)?;
    if !pauli.coefficient.is_finite() {
        return Err(BackendError::InvalidPauliString {
            reason: "non-finite coefficient",
        });
    }
    let mut seen: Vec<u32> = Vec::new();
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
    // Z/I-only fast path: ⟨⊗ Zᵢ⟩ = Σ (-1)^popcount(i & z_mask) |aᵢ|².
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
            let sign = if (i as u64 & z_mask).count_ones() & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            acc += sign * p;
        }
        return Ok(pauli.coefficient * acc);
    }
    // Slow path: φ = P|ψ⟩ by direct amplitude inspection, then Re(⟨ψ|φ⟩).
    let mut tmp: Vec<Complex<f32>> = state.amplitudes_f32().to_vec();
    for (q, p) in &pauli.terms {
        let qbit = 1usize << q;
        match p {
            Pauli::I => {}
            Pauli::Z => {
                for (i, a) in tmp.iter_mut().enumerate() {
                    if i & qbit != 0 {
                        *a = -*a;
                    }
                }
            }
            Pauli::X =>
            {
                #[allow(clippy::needless_range_loop)]
                for i in 0..tmp.len() {
                    if i & qbit == 0 {
                        tmp.swap(i, i | qbit);
                    }
                }
            }
            Pauli::Y => {
                #[allow(clippy::needless_range_loop)]
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
    for (lhs, rhs) in state.amplitudes_f32().iter().zip(tmp.iter()) {
        acc_re += (lhs.re as f64) * (rhs.re as f64) + (lhs.im as f64) * (rhs.im as f64);
    }
    Ok(pauli.coefficient * acc_re)
}
