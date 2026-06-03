//! Dense k-qubit gate kernel: one pass, a 2^k×2^k matvec per 2^k-block.
//! Real bodies land in P2-07 Task 7; this module is wired by Task 6.
use aleph_core::Complex;

/// Apply a dense `2^k × 2^k` unitary (`data`, row-major, MSB-first operand
/// order `qubits[0]`=MSB) to an AoS state in one pass.
pub(crate) fn apply_kq_aos(amps: &mut [Complex], qubits: &[u32], k: u8, data: &[Complex]) {
    apply_kq_scalar_aos(amps, qubits, k, data);
}

/// SoA variant (split real/imag arrays).
pub(crate) fn apply_kq_soa(
    re: &mut [f64],
    im: &mut [f64],
    qubits: &[u32],
    k: u8,
    data: &[Complex],
) {
    apply_kq_scalar_soa(re, im, qubits, k, data);
}

// --- implemented in Task 7 ---
pub(crate) fn apply_kq_scalar_aos(
    _amps: &mut [Complex],
    _qubits: &[u32],
    _k: u8,
    _data: &[Complex],
) {
    unimplemented!("apply_kq_scalar_aos: P2-07 Task 7")
}

pub(crate) fn apply_kq_scalar_soa(
    _re: &mut [f64],
    _im: &mut [f64],
    _qubits: &[u32],
    _k: u8,
    _data: &[Complex],
) {
    unimplemented!("apply_kq_scalar_soa: P2-07 Task 7")
}
