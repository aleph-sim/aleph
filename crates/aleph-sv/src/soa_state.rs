//! `SoaState` — struct-of-arrays state vector: two parallel `Vec<f64>`
//! for the real and imaginary parts, indexed by basis state.

use aleph_core::Complex;

#[derive(Debug, Clone)]
pub struct SoaState {
    pub(crate) num_qubits: u32,
    pub(crate) re: Vec<f64>,
    pub(crate) im: Vec<f64>,
}

impl SoaState {
    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Read-only view of the real-part buffer.
    pub fn re(&self) -> &[f64] {
        &self.re
    }

    /// Read-only view of the imaginary-part buffer.
    pub fn im(&self) -> &[f64] {
        &self.im
    }

    /// Materialise as `Vec<Complex>` for oracle / interop paths.
    /// Allocates `2^num_qubits` Complexes. NOT for hot paths —
    /// the SoA backend's primitives operate on `re` / `im` directly.
    pub fn to_aos(&self) -> Vec<Complex> {
        self.re
            .iter()
            .zip(self.im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn getters_match_construction() {
        let s = SoaState {
            num_qubits: 3,
            re: vec![0.0; 8],
            im: vec![0.0; 8],
        };
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.re().len(), 8);
        assert_eq!(s.im().len(), 8);
    }

    #[test]
    fn to_aos_roundtrips_paired_values() {
        let s = SoaState {
            num_qubits: 1,
            re: vec![0.5, -0.25],
            im: vec![0.0, 0.75],
        };
        let aos = s.to_aos();
        assert_eq!(aos.len(), 2);
        assert_eq!(aos[0], Complex::new(0.5, 0.0));
        assert_eq!(aos[1], Complex::new(-0.25, 0.75));
    }
}
