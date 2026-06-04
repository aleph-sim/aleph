//! Mixed-canonical MPS state: init, dense reconstruction, canonicalization,
//! gate application, expectation, measurement, sampling, probabilities.

use crate::tensor::Site;
use aleph_core::Complex;

/// Mixed-canonical MPS. Sites left of `center` are left-canonical, sites right
/// are right-canonical; the center site carries the norm.
#[derive(Debug, Clone)]
pub struct MpsState {
    pub(crate) sites: Vec<Site>,
    pub(crate) center: usize,
    pub(crate) max_bond: usize,
    pub(crate) trunc_error: f64,
}

impl MpsState {
    /// Allocate |0…0⟩ on `n` qubits with bond cap `max_bond`.
    pub fn new(n: usize, max_bond: usize) -> Self {
        let sites = (0..n).map(|_| Site::ket0()).collect();
        MpsState {
            sites,
            center: 0,
            max_bond: max_bond.max(1),
            trunc_error: 0.0,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.sites.len()
    }

    /// Accumulated discarded Schmidt weight from all SVD truncations so far.
    pub fn truncation_error(&self) -> f64 {
        self.trunc_error
    }

    /// Contract the whole chain into a dense `2^n` amplitude vector.
    /// TEST/SMALL-n ONLY (allocates 2^n). Amplitude index uses the ADR-0004
    /// convention: qubit `q` (== site `q`) occupies bit `q`.
    pub fn dense_statevector(&self) -> Vec<Complex> {
        let n = self.sites.len();
        // amps is laid out as [basis_prefix * left_dim + l]:
        //   basis_prefix is the partial basis index accumulated so far (bits 0..q-1),
        //   l is the left-bond index of the current site.
        //
        // We start with a single virtual "left bond = 1" amplitude of value 1.
        let mut amps: Vec<Complex> = vec![Complex::new(1.0, 0.0)]; // left bond of site 0 = 1
        let mut left_dim = 1usize;

        for (q, site) in self.sites.iter().enumerate() {
            debug_assert_eq!(site.left, left_dim);
            let prefix_count = amps.len() / left_dim;
            // next is laid out as [new_prefix * site.right + r],
            // where new_prefix = old_prefix | (p << q).
            let mut next = vec![Complex::new(0.0, 0.0); prefix_count * 2 * site.right];

            // Allow explicit index arithmetic — the multi-index contraction is
            // clearer expressed as nested loops than as iterator gymnastics.
            #[allow(clippy::needless_range_loop)]
            for prefix in 0..prefix_count {
                for p in 0..2usize {
                    // Bit q of the basis index is the physical index p of site q.
                    let new_prefix = prefix | (p << q);
                    for r in 0..site.right {
                        let mut acc = Complex::new(0.0, 0.0);
                        for l in 0..left_dim {
                            acc += amps[prefix * left_dim + l] * site.get(l, p, r);
                        }
                        next[new_prefix * site.right + r] += acc;
                    }
                }
            }
            amps = next;
            left_dim = site.right;
        }

        debug_assert_eq!(left_dim, 1, "right bond of the last site must be 1");
        // At this point amps[i] == amplitude of basis state i (all n bits set).
        let _ = n; // used only for the initial capacity reasoning; length is 2^n
        amps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm_sq(v: &[Complex]) -> f64 {
        v.iter().map(|c| c.norm_sqr()).sum()
    }

    #[test]
    fn ket0_dense_is_e0() {
        let s = MpsState::new(3, 64);
        let v = s.dense_statevector();
        assert_eq!(v.len(), 8);
        assert!((v[0].re - 1.0).abs() < 1e-12);
        assert!((norm_sq(&v) - 1.0).abs() < 1e-12);
        for amp in &v[1..] {
            assert!(amp.norm() < 1e-12);
        }
    }

    #[test]
    fn single_qubit_dense() {
        let s = MpsState::new(1, 64);
        let v = s.dense_statevector();
        assert_eq!(v.len(), 2);
        assert!((v[0].re - 1.0).abs() < 1e-12);
        assert!(v[1].norm() < 1e-12);
    }
}
