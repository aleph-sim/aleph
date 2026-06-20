//! Host-side readout over the unified-memory FP32 statevector. On Apple
//! Silicon the buffer is zero-copy CPU-visible after the GPU completes, so
//! probabilities / sampling / measurement / expectation run on the host with
//! no GPU reduction. Accumulations widen each `Complex<f32>` to f64 to avoid
//! catastrophic cancellation when summing 2^n single-precision terms (mirrors
//! `aleph-sv::fp32_measure`).

use aleph_backend::BackendError;
use aleph_core::{Complex, Pauli, PauliString};
use rand::{rngs::StdRng, Rng};
use rayon::prelude::*;

use super::state::MetalSvState;

/// Branch probability below which collapse is refused (scaling by ~1/√p would
/// explode). Mirrors the CPU path.
const DEGENERATE_BRANCH_THRESHOLD: f64 = 1e-300;
/// Relaxed normalization tolerance for an f32-accumulated state.
const FP32_NORM_TOL: f64 = 1e-4;
/// Amplitude count (≈ n=14) below which the host readout stays serial: rayon's
/// fork/join overhead exceeds the gain on small states, and the unit tests
/// (n ≤ ~10) keep their tight timing. At/above it the 2^n reductions and the
/// measure collapse fan out across cores (P5.6-06).
const PAR_THRESHOLD: usize = 1 << 14;

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
    // Per-amplitude |a|² (the array every readout op reduces over). Parallel map
    // for large states; the finiteness check and norm sum fold in afterwards.
    let probs: Vec<f64> = if len >= PAR_THRESHOLD {
        amps.par_iter().map(|&a| norm_sqr_f64(a)).collect()
    } else {
        amps.iter().map(|&a| norm_sqr_f64(a)).collect()
    };
    let (total, all_finite) = if len >= PAR_THRESHOLD {
        probs
            .par_iter()
            .map(|&p| (p, p.is_finite()))
            .reduce(|| (0.0, true), |(s1, f1), (s2, f2)| (s1 + s2, f1 && f2))
    } else {
        probs
            .iter()
            .fold((0.0, true), |(s, f), &p| (s + p, f && p.is_finite()))
    };
    if !all_finite {
        return Err(BackendError::InvalidState {
            reason: "non-finite amplitude norm²",
        });
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
    // Marginal bin for amplitude index `i`: gather the queried qubit bits.
    let bin = |i: usize| -> usize {
        let mut k = 0usize;
        for (pos, &q) in qubits.iter().enumerate() {
            if (i >> q) & 1 == 1 {
                k |= 1usize << pos;
            }
        }
        k
    };
    let out = if probs.len() >= PAR_THRESHOLD {
        // Per-thread local bins, summed at the join — out_dim is small (2^|qubits|).
        probs
            .par_iter()
            .enumerate()
            .fold(
                || vec![0.0_f64; out_dim],
                |mut acc, (i, &p)| {
                    acc[bin(i)] += p;
                    acc
                },
            )
            .reduce(
                || vec![0.0_f64; out_dim],
                |mut a, b| {
                    for (slot, v) in a.iter_mut().zip(b) {
                        *slot += v;
                    }
                    a
                },
            )
    } else {
        let mut out = vec![0.0_f64; out_dim];
        for (i, p) in probs.iter().enumerate() {
            out[bin(i)] += *p;
        }
        out
    };
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
    // Split weight into the outcome-0 and outcome-1 branches.
    let (p0, p1) = if probs.len() >= PAR_THRESHOLD {
        probs
            .par_iter()
            .enumerate()
            .map(|(i, &p)| if i & q_bit != 0 { (0.0, p) } else { (p, 0.0) })
            .reduce(|| (0.0, 0.0), |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1))
    } else {
        probs
            .iter()
            .enumerate()
            .fold((0.0_f64, 0.0_f64), |(a0, a1), (i, &p)| {
                if i & q_bit != 0 {
                    (a0, a1 + p)
                } else {
                    (a0 + p, a1)
                }
            })
    };
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
    // Collapse: zero the eliminated branch, renormalize the kept one. Each index
    // is independent, so the rewrite fans out over disjoint amplitudes.
    let collapse = |i: usize, a: &mut Complex<f32>| {
        if (i & q_bit != 0) == outcome {
            a.re *= scale;
            a.im *= scale;
        } else {
            *a = Complex::<f32>::new(0.0, 0.0);
        }
    };
    let amps = state.amps.as_mut_slice();
    if amps.len() >= PAR_THRESHOLD {
        amps.par_iter_mut()
            .enumerate()
            .for_each(|(i, a)| collapse(i, a));
    } else {
        amps.iter_mut()
            .enumerate()
            .for_each(|(i, a)| collapse(i, a));
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
        // ⟨⊗Zᵢ⟩ = Σ (-1)^popcount(i & z_mask) |aᵢ|² — a sign-weighted parallel sum.
        let signed = |i: usize, p: f64| {
            if (i as u64 & z_mask).count_ones() & 1 == 0 {
                p
            } else {
                -p
            }
        };
        let acc: f64 = if probs.len() >= PAR_THRESHOLD {
            probs
                .par_iter()
                .enumerate()
                .map(|(i, &p)| signed(i, p))
                .sum()
        } else {
            probs.iter().enumerate().map(|(i, &p)| signed(i, p)).sum()
        };
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
    // Re⟨ψ|φ⟩ with φ = P|ψ⟩ — a parallel dot product over the 2^n amplitudes.
    let lhs = state.amplitudes_f32();
    let dot = |l: &Complex<f32>, r: &Complex<f32>| {
        (l.re as f64) * (r.re as f64) + (l.im as f64) * (r.im as f64)
    };
    let acc_re: f64 = if tmp.len() >= PAR_THRESHOLD {
        lhs.par_iter()
            .zip(tmp.par_iter())
            .map(|(l, r)| dot(l, r))
            .sum()
    } else {
        lhs.iter().zip(tmp.iter()).map(|(l, r)| dot(l, r)).sum()
    };
    Ok(pauli.coefficient * acc_re)
}
