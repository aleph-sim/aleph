//! Measurement, sampling, expectation, marginals.

use aleph_backend::BackendError;
use aleph_core::Complex;
use rand::{rngs::StdRng, Rng};

use crate::state::CpuState;

/// Threshold under which we refuse to collapse the state — collapsing
/// on a branch of probability `< 1e-300` would scale amplitudes by
/// `≈ 1e150` and destroy any meaningful state.
const DEGENERATE_BRANCH_THRESHOLD: f64 = 1e-300;

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
    let q_bit = 1usize << qubit;
    let mut p1 = 0.0_f64;
    for (i, a) in state.amps.iter().enumerate() {
        if i & q_bit != 0 {
            p1 += a.norm_sqr();
        }
    }
    let outcome: bool = rng.gen::<f64>() < p1;
    let p = if outcome { p1 } else { 1.0 - p1 };
    if p < DEGENERATE_BRANCH_THRESHOLD {
        return Err(BackendError::DegenerateMeasurement {
            qubit,
            probability: p,
        });
    }
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

/// Sample basis-state indices from `|amps[i]|²` via inverse-CDF.
///
/// Builds the CDF once, then binary-searches per shot. CDF is clamped
/// at 1.0 at the last index to absorb floating-point drift; a shot
/// with `u == 1.0` (rare but possible) maps to the last basis index.
pub(crate) fn sample_impl(
    rng: &mut StdRng,
    state: &CpuState,
    shots: u32,
) -> Result<Vec<u64>, BackendError> {
    let n = state.amps.len();
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0_f64;
    for a in &state.amps {
        acc += a.norm_sqr();
        cdf.push(acc);
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    let mut out = Vec::with_capacity(shots as usize);
    for _ in 0..shots {
        let u: f64 = rng.gen();
        let idx = cdf.partition_point(|&c| c < u);
        let idx = idx.min(n.saturating_sub(1));
        out.push(idx as u64);
    }
    Ok(out)
}
