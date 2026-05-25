//! Measurement, sampling, expectation, marginals.

use aleph_backend::BackendError;
use aleph_core::Complex;
use rand::{rngs::StdRng, Rng};

use crate::state::CpuState;

/// Threshold under which we refuse to collapse the state — collapsing
/// on a branch of probability `< 1e-300` would scale amplitudes by
/// `≈ 1e150` and destroy any meaningful state.
const DEGENERATE_BRANCH_THRESHOLD: f64 = 1e-300;

/// Walk every amplitude once: reject empty/non-finite/non-normalised
/// states. Returns the per-amp `norm_sqr` vector so callers don't pay
/// for a second pass.
///
/// All four query methods (`measure`, `sample`, `expectation_value`,
/// `probabilities`) share this preamble so a corrupted state surfaces
/// the same `BackendError::InvalidState` no matter which entry point
/// the caller used. Without this shared guard the methods drifted into
/// asymmetric checks (round-2 hardened only `measure` and `sample`).
pub(crate) fn validate_state(state: &CpuState) -> Result<Vec<f64>, BackendError> {
    let n = state.amps.len();
    if n == 0 {
        return Err(BackendError::InvalidState {
            reason: "empty state vector",
        });
    }
    let mut probs = Vec::with_capacity(n);
    let mut total = 0.0_f64;
    for a in &state.amps {
        let p = a.norm_sqr();
        if !p.is_finite() {
            return Err(BackendError::InvalidState {
                reason: "non-finite amplitude norm²",
            });
        }
        total += p;
        probs.push(p);
    }
    // Drift budget: √n · AMPLITUDE_TOL absorbs the worst-case sum-of-
    // independent-errors growth over an n-element accumulation while
    // still rejecting genuinely un-normalised inputs (e.g. norm² = 0.5).
    let drift_budget = (n as f64).sqrt() * aleph_core::AMPLITUDE_TOL;
    if (total - 1.0).abs() > drift_budget {
        return Err(BackendError::InvalidState {
            reason: "state norm² deviates from 1 beyond drift budget",
        });
    }
    Ok(probs)
}

pub(crate) fn measure_impl(
    rng: &mut StdRng,
    state: &mut CpuState,
    qubit: u32,
) -> Result<bool, BackendError> {
    let n = state.num_qubits;
    if qubit >= n {
        return Err(BackendError::QubitOutOfRange {
            qubit,
            num_qubits: n,
        });
    }
    // Single source of truth for finiteness and total-norm validation;
    // returns the per-amp probabilities so we don't pay for two passes.
    let probs = validate_state(state)?;
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
    // Clamp into [0, 1] to absorb the residual FP drift validate_state
    // already bounded — `sqrt` on a slightly-negative p would produce
    // NaN amplitudes during renormalisation.
    let p0 = p0.clamp(0.0, 1.0);
    let p1 = p1.clamp(0.0, 1.0);
    // Decide outcome WITHOUT consuming RNG when the answer is forced
    // by a degenerate branch. This keeps `with_seed(N)` reproducibility
    // intact for the highly-polarized legal cases and only consumes RNG
    // on genuine superpositions.
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
    for (i, a) in state.amps.iter_mut().enumerate() {
        let bit_set = (i & q_bit) != 0;
        if bit_set == outcome {
            *a /= Complex::new(norm, 0.0);
        } else {
            *a = Complex::new(0.0, 0.0);
        }
    }
    Ok(outcome)
}

/// Sample basis-state indices from `|amps[i]|²` via Vose's alias method.
///
/// `O(n)` build (`n = 2^num_qubits`) + `O(1)` per shot. Replaces the
/// `O(log n)`-per-shot inverse-CDF path used in P0-09; see
/// `crates/aleph-sv/src/sampling.rs` for the build algorithm and
/// `benches/sample.rs` for the measured speedup.
pub(crate) fn sample_impl(
    rng: &mut StdRng,
    state: &CpuState,
    shots: u32,
) -> Result<Vec<u64>, BackendError> {
    let probs = validate_state(state)?;
    let table = crate::sampling::AliasTable::build(&probs);
    let mut out = Vec::with_capacity(shots as usize);
    for _ in 0..shots {
        out.push(table.draw(rng) as u64);
    }
    Ok(out)
}

/// Naive expectation value: copy state, apply each non-identity Pauli
/// as a 1q gate to the copy, then take `Re(⟨ψ|φ⟩)`.
///
/// O(N · k) where N = 2^n and k = `pauli.terms.len()`. P0-11 will add
/// the Pauli-Z fast path that doesn't need a copy.
pub(crate) fn expectation_value_impl(
    state: &CpuState,
    pauli: &aleph_core::PauliString,
) -> Result<f64, BackendError> {
    let n = state.num_qubits;
    let _ = validate_state(state)?; // surface state corruption symmetrically
                                    // Revalidate the public invariants on `PauliString`. The fields are
                                    // `pub` so a caller can bypass `PauliString::new`'s sort/dedup/finite
                                    // checks by direct struct-literal construction. Trusting those
                                    // invariants here produced silently-wrong expectation values for
                                    // duplicate-qubit terms.
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
    // Pauli-Z fast path: diagonal Pauli strings need no state clone
    // and no kernel apply. ⟨ψ| ⊗ᵢ Zᵢ |ψ⟩ = Σᵢ (-1)^popcount(i & z_mask) · |aᵢ|².
    let mut z_mask = 0u64;
    let mut all_z_or_i = true;
    for (q, p) in &pauli.terms {
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
        return Ok(pauli.coefficient * expectation_z_diag(&state.amps, z_mask));
    }
    let mut tmp = state.amps.clone();
    for (q, p) in &pauli.terms {
        if *p == aleph_core::Pauli::I {
            continue;
        }
        let m = p.matrix();
        crate::kernels::apply_1q(&mut tmp, *q, &[], &m);
    }
    let mut acc = Complex::new(0.0, 0.0);
    for (lhs, rhs) in state.amps.iter().zip(tmp.iter()) {
        acc += lhs.conj() * (*rhs);
    }
    Ok(pauli.coefficient * acc.re)
}

/// Marginal probabilities over the named qubit subset.
///
/// Returns a vector of length `2^qubits.len()`. The output is indexed
/// with `qubits[0]` as LSB to match the global gate-ordering convention.
pub(crate) fn probabilities_impl(
    state: &CpuState,
    qubits: &[u32],
) -> Result<Vec<f64>, BackendError> {
    let n = state.num_qubits;
    let probs = validate_state(state)?;
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

/// Diagonal `⟨ψ| ⊗ᵢ Zᵢ |ψ⟩` evaluation for a Z-only Pauli string.
///
/// `z_mask` has bit `q` set iff the input Pauli string carries
/// `(q, Pauli::Z)`. Identity terms contribute nothing to the mask
/// (their sign is always +1).
///
/// `i` is at most `2^28` (`MAX_NAIVE_QUBITS`); `i as u64` is exact.
/// `count_ones` lowers to `popcnt` on x86-64 and to a single
/// instruction on aarch64.
fn expectation_z_diag(amps: &[Complex], z_mask: u64) -> f64 {
    let mut acc = 0.0_f64;
    for (i, a) in amps.iter().enumerate() {
        let sign = if (i as u64 & z_mask).count_ones() & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        acc += sign * a.norm_sqr();
    }
    acc
}
