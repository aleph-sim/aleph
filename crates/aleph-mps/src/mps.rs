//! Mixed-canonical MPS state: init, dense reconstruction, canonicalization,
//! gate application, expectation, measurement, sampling, probabilities.

use crate::tensor::{thin_qr, truncated_svd, Site};
use crate::MpsError;
use aleph_core::{Complex, GateInstance};
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

    /// Apply a 2q unitary `u` (4×4 matrix) to nearest-neighbor qubits.
    ///
    /// The gate's qubit ordering follows the ADR-0004 / P0-06 MSB convention:
    /// `g.qubits[0]` is the **most-significant** bit of the matrix row/column
    /// index. This matches every other backend in the codebase.
    ///
    /// Only nearest-neighbor pairs (`|qa − qb| == 1`) are supported; the basic
    /// MPS chain cannot apply long-range gates without SWAP networks (P3-06).
    pub(crate) fn apply_2q(
        &mut self,
        g: &GateInstance,
        u: &[[Complex; 4]; 4],
    ) -> Result<(), MpsError> {
        let qa = g.qubits[0];
        let qb = g.qubits[1];
        if qa.abs_diff(qb) != 1 {
            return Err(MpsError::NonNearestNeighbor { a: qa, b: qb });
        }

        let i = qa.min(qb) as usize;
        let j = i + 1;

        // Move the orthogonality center to site i so that the two-site
        // contraction preserves normalization after re-factorization.
        self.move_center_to(i);

        let li = self.sites[i].left;
        let mi = self.sites[i].right; // shared bond between site i and j
        let ri = self.sites[j].right;

        // Build the two-site tensor Θ[l, a, b, r] = Σ_m sites[i][l,a,m] · sites[j][m,b,r]
        // Flat index: ((l * 2 + a) * 2 + b) * ri + r
        let theta_len = li * 2 * 2 * ri;
        let mut theta = vec![Complex::new(0.0, 0.0); theta_len];

        // Explicit loops — nested-tensor contractions are clearer than iterators here.
        #[allow(clippy::needless_range_loop)]
        for l in 0..li {
            for a in 0..2usize {
                for b in 0..2usize {
                    for r in 0..ri {
                        let mut acc = Complex::new(0.0, 0.0);
                        for m in 0..mi {
                            acc += self.sites[i].get(l, a, m) * self.sites[j].get(m, b, r);
                        }
                        theta[((l * 2 + a) * 2 + b) * ri + r] = acc;
                    }
                }
            }
        }

        // Helper: given the physical indices of site i (phys_i) and site j (phys_j),
        // return the 2q matrix row/column index following the MSB convention:
        // g.qubits[0] is the MSB, g.qubits[1] is the LSB.
        let out = |phys_i: usize, phys_j: usize| -> usize {
            // Identify which physical index maps to qubits[0] (MSB) and qubits[1] (LSB).
            let bit_q0 = if g.qubits[0] as usize == i {
                phys_i
            } else {
                phys_j
            };
            let bit_q1 = if g.qubits[1] as usize == i {
                phys_i
            } else {
                phys_j
            };
            // qubits[0] is MSB, qubits[1] is LSB (ADR-0004 / P0-06 convention).
            (bit_q0 << 1) | bit_q1
        };

        // Apply the gate: Θ'[l,a',b',r] = Σ_{a,b} U[out(a',b')][out(a,b)] · Θ[l,a,b,r]
        let mut theta2 = vec![Complex::new(0.0, 0.0); theta_len];

        #[allow(clippy::needless_range_loop)]
        for l in 0..li {
            for ap in 0..2usize {
                for bp in 0..2usize {
                    let row = out(ap, bp);
                    for a in 0..2usize {
                        for b in 0..2usize {
                            let col = out(a, b);
                            let u_entry = u[row][col];
                            if u_entry == Complex::new(0.0, 0.0) {
                                continue;
                            }
                            for r in 0..ri {
                                theta2[((l * 2 + ap) * 2 + bp) * ri + r] +=
                                    u_entry * theta[((l * 2 + a) * 2 + b) * ri + r];
                            }
                        }
                    }
                }
            }
        }

        // Reshape Θ' to matrix M of shape (li*2) × (2*ri):
        //   row = l*2 + a'  (group left bond and physical of site i)
        //   col = b'*ri + r (group physical of site j and right bond)
        // This matches from_group_left / from_group_right conventions.
        let m = DMatrix::from_fn(li * 2, 2 * ri, |row, col| {
            let l = row / 2;
            let ap = row % 2;
            let bp = col / ri;
            let r = col % ri;
            theta2[((l * 2 + ap) * 2 + bp) * ri + r]
        });

        // Truncated SVD: M = U · diag(s) · Vt, keeping at most max_bond components.
        let (u_s, s_kept, vt_s, discarded) = truncated_svd(&m, self.max_bond);
        self.trunc_error += discarded;
        let chi = s_kept.len();

        // New site i: left-canonical from the U factor, shape (li, chi).
        self.sites[i] = Site::from_group_left(&u_s, li, chi);

        // New site j: multiply singular values into Vt rows, shape (chi, ri) physical-grouped.
        // sv[(row, col)] = s_kept[row] * vt_s[(row, col)]
        let mut sv = vt_s;
        for row in 0..chi {
            for col in 0..sv.ncols() {
                sv[(row, col)] *= s_kept[row];
            }
        }
        // sv has shape chi × (2*ri) — matches from_group_right(left=chi, right=ri).
        self.sites[j] = Site::from_group_right(&sv, chi, ri);
        self.center = j;

        Ok(())
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
    use crate::MpsError;
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

    #[test]
    fn bell_via_h_cnot() {
        let mut s = MpsState::new(2, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let g = GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]);
        let cnot = crate::gate::matrix_4x4(&g).unwrap();
        s.apply_2q(&g, &cnot).unwrap();
        let v = s.dense_statevector();
        let inv = 1.0 / 2f64.sqrt();
        assert!((v[0].re - inv).abs() < 1e-10); // |00>
        assert!(v[1].norm() < 1e-10);
        assert!(v[2].norm() < 1e-10);
        assert!((v[3].re - inv).abs() < 1e-10); // |11>
        assert!(s.truncation_error() < 1e-12);
    }

    #[test]
    fn rejects_non_adjacent() {
        let mut s = MpsState::new(3, 64);
        let g = GateInstance::new(Gate::Cnot, smallvec![0u32, 2u32]);
        let cnot = crate::gate::matrix_4x4(&g).unwrap();
        let err = s.apply_2q(&g, &cnot).unwrap_err();
        assert!(matches!(err, MpsError::NonNearestNeighbor { a: 0, b: 2 }));
    }

    #[test]
    fn cnot_reversed_qubit_order() {
        // CNOT qubits [1,0]: control=q1, target=q0. Prep q1=|1>, then CNOT → |11>.
        let mut s = MpsState::new(2, 64);
        let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![1u32])).unwrap();
        s.apply_1q(1, &x);
        let g = GateInstance::new(Gate::Cnot, smallvec![1u32, 0u32]);
        let cnot = crate::gate::matrix_4x4(&g).unwrap();
        s.apply_2q(&g, &cnot).unwrap();
        let v = s.dense_statevector();
        assert!((v[3].re - 1.0).abs() < 1e-10); // |11> index 3
    }
}
