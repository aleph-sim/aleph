//! Mixed-canonical MPS state: init, dense reconstruction, canonicalization,
//! gate application, expectation, measurement, sampling, probabilities.

use crate::tensor::{thin_qr, Site};
use aleph_core::Complex;
use nalgebra::DMatrix;

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

    /// Apply a 1q unitary to site `i` (qubit `i`). Preserves canonical form,
    /// so neither the center nor any SVD is touched.
    pub(crate) fn apply_1q(&mut self, i: usize, u: &[[Complex; 2]; 2]) {
        let site = &mut self.sites[i];
        for l in 0..site.left {
            for r in 0..site.right {
                let a0 = site.get(l, 0, r);
                let a1 = site.get(l, 1, r);
                *site.get_mut(l, 0, r) = u[0][0] * a0 + u[0][1] * a1;
                *site.get_mut(l, 1, r) = u[1][0] * a0 + u[1][1] * a1;
            }
        }
    }

    /// Multiply matrix `r` into site `i`'s LEFT bond:
    /// A'[l',p,r2] = Σ_l r[l',l] · A[l,p,r2].
    fn absorb_into_left(&mut self, i: usize, r: &DMatrix<Complex>) {
        let site = &self.sites[i];
        let new_left = r.nrows();
        let mut out = Site::zeros(new_left, site.right);
        // Explicit index arithmetic for clarity of the bond contraction.
        #[allow(clippy::needless_range_loop)]
        for lp in 0..new_left {
            for p in 0..2 {
                for r2 in 0..site.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for l in 0..site.left {
                        acc += r[(lp, l)] * site.get(l, p, r2);
                    }
                    *out.get_mut(lp, p, r2) = acc;
                }
            }
        }
        self.sites[i] = out;
    }

    /// Multiply matrix `l` into site `i`'s RIGHT bond:
    /// A'[l2,p,r'] = Σ_r A[l2,p,r] · l[r,r'].
    fn absorb_into_right(&mut self, i: usize, l: &DMatrix<Complex>) {
        let site = &self.sites[i];
        let new_right = l.ncols();
        let mut out = Site::zeros(site.left, new_right);
        // Explicit index arithmetic for clarity of the bond contraction.
        #[allow(clippy::needless_range_loop)]
        for l2 in 0..site.left {
            for p in 0..2 {
                for rp in 0..new_right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for r in 0..site.right {
                        acc += site.get(l2, p, r) * l[(r, rp)];
                    }
                    *out.get_mut(l2, p, rp) = acc;
                }
            }
        }
        self.sites[i] = out;
    }

    /// Shift center right from `i` to `i+1` using thin QR on the grouped-left
    /// matrix. Site `i` becomes left-canonical; the R factor is absorbed into
    /// site `i+1`'s left bond.
    fn move_center_right(&mut self) {
        let i = self.center;
        let m = self.sites[i].to_group_left(); // (left*2) × right
        let (q, r) = thin_qr(&m); // q:(left*2)×k, r:k×right
        let k = q.ncols();
        let left = self.sites[i].left;
        self.sites[i] = Site::from_group_left(&q, left, k);
        self.absorb_into_left(i + 1, &r);
        self.center += 1;
    }

    /// Shift center left from `i` to `i-1` using thin QR on the adjoint of the
    /// grouped-right matrix (LQ decomposition). Site `i` becomes right-canonical;
    /// the Rᴴ factor is absorbed into site `i-1`'s right bond.
    fn move_center_left(&mut self) {
        let i = self.center;
        let m = self.sites[i].to_group_right(); // left × (2*right)
        let mh = m.adjoint(); // (2*right) × left
        let (q, r) = thin_qr(&mh); // q:(2*right)×k, r:k×left
        let k = q.ncols();
        let right = self.sites[i].right;
        let site_mat = q.adjoint(); // k × (2*right) — right-canonical
        self.sites[i] = Site::from_group_right(&site_mat, k, right);
        let r_into = r.adjoint(); // left × k — absorbed into left neighbor's right bond
        self.absorb_into_right(i - 1, &r_into);
        self.center -= 1;
    }

    /// Move the orthogonality center to `target` by stepping one site at a time.
    pub(crate) fn move_center_to(&mut self, target: usize) {
        while self.center < target {
            self.move_center_right();
        }
        while self.center > target {
            self.move_center_left();
        }
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
    use crate::tensor::Site;
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    fn norm_sq(v: &[Complex]) -> f64 {
        v.iter().map(|c| c.norm_sqr()).sum()
    }

    /// Left-canonical check: Σ_{l,p} conj(A[l,p,r1]) A[l,p,r2] == δ(r1,r2).
    fn is_left_canonical(site: &Site) -> bool {
        for r1 in 0..site.right {
            for r2 in 0..site.right {
                let mut acc = aleph_core::Complex::new(0.0, 0.0);
                for l in 0..site.left {
                    for p in 0..2 {
                        acc += site.get(l, p, r1).conj() * site.get(l, p, r2);
                    }
                }
                let expect = if r1 == r2 { 1.0 } else { 0.0 };
                if (acc.re - expect).abs() > 1e-9 || acc.im.abs() > 1e-9 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn move_center_right_makes_left_canonical_and_preserves_state() {
        let mut s = MpsState::new(3, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        s.apply_1q(1, &h);
        let before = s.dense_statevector();
        s.move_center_to(2);
        assert_eq!(s.center, 2);
        assert!(is_left_canonical(&s.sites[0]));
        assert!(is_left_canonical(&s.sites[1]));
        let after = s.dense_statevector();
        for (a, b) in before.iter().zip(after.iter()) {
            assert!(
                (a - b).norm() < 1e-9,
                "state changed under canonicalization"
            );
        }
    }

    #[test]
    fn move_center_left_preserves_state() {
        let mut s = MpsState::new(3, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        s.apply_1q(1, &h);
        s.apply_1q(2, &h);
        s.move_center_to(2);
        let before = s.dense_statevector();
        s.move_center_to(0);
        assert_eq!(s.center, 0);
        let after = s.dense_statevector();
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).norm() < 1e-9, "state changed moving center left");
        }
    }

    #[test]
    fn x_on_zero_is_one() {
        let mut s = MpsState::new(1, 64);
        let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![0u32])).unwrap();
        s.apply_1q(0, &x);
        let v = s.dense_statevector();
        assert!(v[0].norm() < 1e-12);
        assert!((v[1].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn h_on_zero_is_plus() {
        let mut s = MpsState::new(2, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let v = s.dense_statevector();
        let inv = 1.0 / 2f64.sqrt();
        assert!((v[0].re - inv).abs() < 1e-12); // |00>
        assert!((v[1].re - inv).abs() < 1e-12); // |01> (q0=1)
        assert!(v[2].norm() < 1e-12);
        assert!(v[3].norm() < 1e-12);
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
