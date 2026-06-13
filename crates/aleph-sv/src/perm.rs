//! Final physical→logical state reorder for the P2-09 relabelling pass
//! (`passes::RelabelQubits`). The pass permutes qubit indices for cache
//! locality, leaving the simulated state in PHYSICAL-bit order; these helpers
//! produce the LOGICAL-order amplitude vector with a single gather.
//!
//! The index arithmetic is shared with every other backend via
//! [`aleph_core::bit_permute_buf`] (P3-15); the wrappers here only pin the
//! element type for each of this crate's amplitude representations.

use aleph_core::{bit_permute_buf, AlignedBuf, Complex};

/// Reorder `phys` (physical-bit order) into logical order per `perm`.
/// `perm.len() == num_qubits` and `perm` is a permutation of
/// `0..num_qubits`. `phys.len() == 2^num_qubits`.
// Called by `NaiveSvBackend::unpermute_state` (the P2-09 driver tail).
pub(crate) fn bit_permute_state(phys: &[Complex], perm: &[u32]) -> AlignedBuf<Complex> {
    bit_permute_buf(phys, perm)
}

/// f32 AoS analogue of [`bit_permute_state`] for the FP32 backend.
pub(crate) fn bit_permute_state_f32(
    phys: &[Complex<f32>],
    perm: &[u32],
) -> AlignedBuf<Complex<f32>> {
    bit_permute_buf(phys, perm)
}

/// Split-buffer (SoA) analogue: permutes a single `f64` plane (re or im).
pub(crate) fn bit_permute_plane(plane: &[f64], perm: &[u32]) -> AlignedBuf<f64> {
    bit_permute_buf(plane, perm)
}
