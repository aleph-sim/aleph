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
    ///
    /// `re.len() == im.len()` is a structural invariant enforced by
    /// every measure-path entry via `measure_soa::validate_state_soa`;
    /// the assertion here mirrors that discipline so a divergent-
    /// length state surfaces as a clear panic on this path too,
    /// rather than silently truncating via `zip`'s `min(a, b)`
    /// semantics. P0-09 round-3 lesson: every state-consuming entry
    /// point surfaces the same `InvalidState`-style failure.
    pub fn to_aos(&self) -> Vec<Complex> {
        assert_eq!(
            self.re.len(),
            self.im.len(),
            "SoaState::to_aos: re.len()={} != im.len()={}",
            self.re.len(),
            self.im.len(),
        );
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

    #[test]
    #[should_panic(expected = "re.len()=2 != im.len()=1")]
    fn to_aos_rejects_mismatched_lens() {
        // Direct field-literal construction bypasses the backend's
        // allocate path; `to_aos` must surface the inconsistency
        // rather than silently truncate via `zip`'s min semantics.
        let s = SoaState {
            num_qubits: 1,
            re: vec![0.5, -0.25],
            im: vec![0.0],
        };
        let _ = s.to_aos();
    }
}
