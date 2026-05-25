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
    // Structural invariant: `amps.len() == 2^num_qubits`. P0-11's alias
    // sampler enforces a power-of-two table only via `debug_assert!`;
    // release builds would otherwise silently bias `bits & (n-1)` for a
    // mismatched length. Catch it here at the validation boundary so the
    // bug surfaces as `InvalidState`, not as biased samples.
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

#[cfg(test)]
mod tests {
    // Both `super::*` and `proptest::prelude::*` would glob-import a
    // `Rng` trait, which trips the `ambiguous_glob_imported_traits`
    // future-incompat warning. Name parent-module imports explicitly
    // and only glob the proptest prelude.
    use super::{expectation_value_impl, CpuState};
    use aleph_core::{Complex, Pauli, PauliString};
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Reference implementation: always-clone, kernel-apply path.
    /// Mirrors what `expectation_value_impl` did before the Z fast
    /// path landed; used by the proptest below to assert the fast
    /// path agrees on Z-only Pauli strings.
    fn reference_expectation(state: &CpuState, pauli: &PauliString) -> f64 {
        let mut tmp = state.amps.clone();
        for (q, p) in &pauli.terms {
            if *p == Pauli::I {
                continue;
            }
            let m = p.matrix();
            crate::kernels::apply_1q(&mut tmp, *q, &[], &m);
        }
        let mut acc = Complex::new(0.0, 0.0);
        for (lhs, rhs) in state.amps.iter().zip(tmp.iter()) {
            acc += lhs.conj() * (*rhs);
        }
        pauli.coefficient * acc.re
    }

    fn random_normalised_state(n: u32, seed: u64) -> CpuState {
        let mut rng = StdRng::seed_from_u64(seed);
        let dim = 1usize << n;
        let mut amps = Vec::with_capacity(dim);
        let mut norm2 = 0.0_f64;
        for _ in 0..dim {
            let re: f64 = rng.gen_range(-1.0..1.0);
            let im: f64 = rng.gen_range(-1.0..1.0);
            norm2 += re * re + im * im;
            amps.push(Complex::new(re, im));
        }
        let inv = norm2.sqrt().recip();
        for a in &mut amps {
            *a *= Complex::new(inv, 0.0);
        }
        CpuState {
            num_qubits: n,
            amps,
        }
    }

    /// Minimal op-vocabulary for the `∑ P = 1` proptest. We sample
    /// from this small subset so the proptest stays under a second.
    #[derive(Debug, Clone)]
    enum RandomOp {
        H(u32),
        X(u32),
        Y(u32),
        Z(u32),
        S(u32),
        T(u32),
        Cnot(u32, u32),
    }

    impl RandomOp {
        fn realize(&self, n: u32) -> Option<aleph_core::GateInstance> {
            use aleph_core::{Gate, GateInstance};
            use smallvec::smallvec;
            match *self {
                RandomOp::H(q) if q < n => Some(GateInstance::new(Gate::H, smallvec![q])),
                RandomOp::X(q) if q < n => Some(GateInstance::new(Gate::X, smallvec![q])),
                RandomOp::Y(q) if q < n => Some(GateInstance::new(Gate::Y, smallvec![q])),
                RandomOp::Z(q) if q < n => Some(GateInstance::new(Gate::Z, smallvec![q])),
                RandomOp::S(q) if q < n => Some(GateInstance::new(Gate::S, smallvec![q])),
                RandomOp::T(q) if q < n => Some(GateInstance::new(Gate::T, smallvec![q])),
                RandomOp::Cnot(c, t) if c < n && t < n && c != t => {
                    Some(GateInstance::new(Gate::Cnot, smallvec![c, t]))
                }
                _ => None,
            }
        }
    }

    fn any_random_op() -> impl Strategy<Value = RandomOp> {
        let q = 0u32..6;
        prop_oneof![
            q.clone().prop_map(RandomOp::H),
            q.clone().prop_map(RandomOp::X),
            q.clone().prop_map(RandomOp::Y),
            q.clone().prop_map(RandomOp::Z),
            q.clone().prop_map(RandomOp::S),
            q.clone().prop_map(RandomOp::T),
            (0u32..6, 0u32..6).prop_map(|(c, t)| RandomOp::Cnot(c, t)),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

        /// Fast-path equivalence: for any Z-only PauliString on any
        /// concrete normalised state, the diagonal fast path and the
        /// copy-and-rotate slow path must agree to 1e-12.
        #[test]
        fn z_fast_path_matches_slow_path(
            n in 1u32..=5,
            seed in any::<u64>(),
            mask in any::<u32>(),
            coeff in -2.0_f64..=2.0,
        ) {
            let state = random_normalised_state(n, seed);
            // Build a Z-only PauliString from the low n bits of `mask`.
            let mut terms = Vec::new();
            for q in 0..n {
                if (mask >> q) & 1 == 1 {
                    terms.push((q, Pauli::Z));
                }
            }
            let ps = PauliString::new(coeff, terms).unwrap();
            let fast = expectation_value_impl(&state, &ps).unwrap();
            let slow = reference_expectation(&state, &ps);
            prop_assert!(
                (fast - slow).abs() < 1e-12,
                "n={n} mask={mask:0width$b} coeff={coeff}: fast={fast}, slow={slow}",
                width = n as usize,
            );
        }

        /// Sum of marginal probabilities over the full qubit subset
        /// must equal 1 within `√n · AMPLITUDE_TOL`. BACKLOG testing
        /// requirement; see spec §9.
        #[test]
        fn probabilities_full_basis_sums_to_one(
            n in 1u32..=6,
            ops in proptest::collection::vec(any_random_op(), 0..30),
        ) {
            use aleph_backend::Backend;
            let mut b = crate::NaiveSvBackend::with_seed(0);
            let mut s = b.allocate(n).unwrap();
            for op in &ops {
                if let Some(gi) = op.realize(n) {
                    b.apply_gate(&mut s, &gi).unwrap();
                }
            }
            let qubits: Vec<u32> = (0..n).collect();
            let p = b.probabilities(&s, &qubits).unwrap();
            let sum: f64 = p.iter().sum();
            let drift = (p.len() as f64).sqrt() * aleph_core::AMPLITUDE_TOL;
            prop_assert!((sum - 1.0).abs() <= drift, "sum = {sum}");
        }
    }
}
