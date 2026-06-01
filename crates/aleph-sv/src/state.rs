//! `CpuState` — the dense `AlignedBuf<Complex>` of size 2^n used by the naive
//! CPU state-vector backend. Fields are private; consumers go through
//! the read-only getters.

use aleph_core::{AlignedBuf, Complex};

/// State vector held by [`crate::NaiveSvBackend`].
///
/// Layout is array-of-structs (`AlignedBuf<Complex>`); `amps[i]` is the
/// amplitude of basis state `|i⟩` where bit `q` of `i` is the value of
/// qubit `q`. This is the textbook layout from Nielsen & Chuang §4.
/// The buffer is 64-byte (cache-line) aligned so SIMD units sit on the
/// cache-line grid and parallel tasks never share a boundary line.
#[derive(Debug, Clone)]
pub struct CpuState {
    pub(crate) num_qubits: u32,
    pub(crate) amps: AlignedBuf<Complex>,
}

impl CpuState {
    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Read-only view of the underlying amplitude buffer.
    pub fn amplitudes(&self) -> &[Complex] {
        &self.amps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn getters_match_construction() {
        let s = CpuState {
            num_qubits: 3,
            amps: AlignedBuf::from_slice(&[Complex::new(0.0, 0.0); 8]),
        };
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.amplitudes().len(), 8);
    }
}
