//! Rank-3 MPS site tensor `(left, 2, right)` and its reshape ↔ nalgebra views.

use aleph_core::Complex;
use nalgebra::DMatrix;

/// A single MPS site tensor of shape `(left, 2, right)`.
/// Row-major flat storage: `data[(l*2 + p)*right + r]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub left: usize,
    pub right: usize,
    pub data: Vec<Complex>,
}

impl Site {
    /// Returns a `(left, 2, right)` tensor filled with zeros.
    pub fn zeros(left: usize, right: usize) -> Self {
        Site {
            left,
            right,
            data: vec![Complex::new(0.0, 0.0); left * 2 * right],
        }
    }

    /// Returns the `(1, 2, 1)` tensor `[1, 0]` — a qubit in state |0⟩.
    pub fn ket0() -> Self {
        let mut s = Site::zeros(1, 1);
        s.data[0] = Complex::new(1.0, 0.0);
        s
    }

    /// Flat row-major index for `(l, p, r)`.
    #[inline]
    pub fn idx(&self, l: usize, p: usize, r: usize) -> usize {
        (l * 2 + p) * self.right + r
    }

    /// Read element `(l, p, r)`.
    #[inline]
    pub fn get(&self, l: usize, p: usize, r: usize) -> Complex {
        self.data[self.idx(l, p, r)]
    }

    /// Mutable reference to element `(l, p, r)`.
    #[inline]
    pub fn get_mut(&mut self, l: usize, p: usize, r: usize) -> &mut Complex {
        let i = self.idx(l, p, r);
        &mut self.data[i]
    }

    /// Reshape to a `(left*2) × right` matrix, grouping the left bond and
    /// physical index into rows — the form needed to move the orthogonality
    /// center rightward via QR/SVD.
    pub fn to_group_left(&self) -> DMatrix<Complex> {
        DMatrix::from_fn(self.left * 2, self.right, |row, r| {
            let l = row / 2;
            let p = row % 2;
            self.get(l, p, r)
        })
    }

    /// Reconstruct a `Site` from a `(left*2) × right` matrix produced by
    /// [`to_group_left`].
    pub fn from_group_left(m: &DMatrix<Complex>, left: usize, right: usize) -> Site {
        let mut s = Site::zeros(left, right);
        // Allow explicit index arithmetic — clearer than iterator gymnastics
        // for the (row → (l, p)) split.
        #[allow(clippy::needless_range_loop)]
        for row in 0..left * 2 {
            for r in 0..right {
                let l = row / 2;
                let p = row % 2;
                *s.get_mut(l, p, r) = m[(row, r)];
            }
        }
        s
    }

    /// Reshape to a `left × (2*right)` matrix, grouping the physical index and
    /// right bond into columns — the form needed to move the orthogonality
    /// center leftward via QR/SVD.
    pub fn to_group_right(&self) -> DMatrix<Complex> {
        DMatrix::from_fn(self.left, 2 * self.right, |l, col| {
            let p = col / self.right;
            let r = col % self.right;
            self.get(l, p, r)
        })
    }

    /// Reconstruct a `Site` from a `left × (2*right)` matrix produced by
    /// [`to_group_right`].
    pub fn from_group_right(m: &DMatrix<Complex>, left: usize, right: usize) -> Site {
        let mut s = Site::zeros(left, right);
        // Allow explicit index arithmetic — clearer than iterator gymnastics
        // for the (col → (p, r)) split.
        #[allow(clippy::needless_range_loop)]
        for l in 0..left {
            for col in 0..2 * right {
                let p = col / right;
                let r = col % right;
                *s.get_mut(l, p, r) = m[(l, col)];
            }
        }
        s
    }
}

/// SVD of `m` truncated to at most `max_bond` singular values, renormalized to
/// preserve unit weight (input must come from a normalized state). Returns
/// `(u_kept, s_kept, vt_kept, discarded_weight)` where:
/// - `u_kept` has shape `rows × χ` with orthonormal columns (left isometry)
/// - `s_kept` is the χ kept (renormalized) singular values, descending
/// - `vt_kept` has shape `χ × cols` with orthonormal rows
/// - `discarded_weight` is the sum of squares of discarded singular values
///
/// # Why not `nalgebra`'s SVD
///
/// `nalgebra`'s complex `svd()` is unreliable for rank-deficient matrices: it
/// returns orthonormal but *incorrect* singular vectors, so `U·Σ·Vᴴ` no longer
/// reconstructs `M` (verified: `recompose()` errs by ~1e-1 on a rank-1 complex
/// 4×4 — see the P3-04 debugging notes). Two-site MPS blocks are routinely
/// rank-deficient (e.g. after a CNOT collapses Schmidt rank), so we instead
/// diagonalise the Hermitian Gram matrix `G = Mᴴ M` with `SymmetricEigen`
/// (robust for complex Hermitian inputs): its eigenpairs give `σ² = λ` and the
/// right singular vectors `V`, from which `U = M·V·Σ⁻¹`. Numerically-zero
/// singular values are dropped, yielding proper isometries and exact
/// reconstruction.
pub fn truncated_svd(
    m: &DMatrix<Complex>,
    max_bond: usize,
) -> (DMatrix<Complex>, Vec<f64>, DMatrix<Complex>, f64) {
    let rows = m.nrows();
    let cols = m.ncols();

    // Right Gram matrix G = Mᴴ M (cols × cols), Hermitian positive-semidefinite.
    let g = m.adjoint() * m;
    let eig = nalgebra::linalg::SymmetricEigen::new(g);

    // (singular value, eigenvector column index), sorted descending.
    let mut pairs: Vec<(f64, usize)> = (0..cols)
        .map(|k| (eig.eigenvalues[k].max(0.0).sqrt(), k))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Drop singular values that are numerically zero relative to the largest
    // (these correspond to null directions; their "singular vectors" are
    // arbitrary and `M·v = 0`). Keep at most `max_bond`, at least one.
    let s_max = pairs.first().map(|p| p.0).unwrap_or(0.0);
    let eps = 1e-12 * s_max.max(f64::MIN_POSITIVE);
    let significant = pairs.iter().filter(|p| p.0 > eps).count().max(1);
    let chi = significant.min(max_bond.max(1));

    let discarded: f64 = pairs[chi..].iter().map(|p| p.0 * p.0).sum();
    let kept_weight: f64 = pairs[..chi].iter().map(|p| p.0 * p.0).sum();
    let scale = if kept_weight > 0.0 {
        (1.0 / kept_weight).sqrt()
    } else {
        1.0
    };

    let mut u_kept = DMatrix::<Complex>::zeros(rows, chi);
    let mut vt_kept = DMatrix::<Complex>::zeros(chi, cols);
    let mut s_kept = vec![0.0_f64; chi];
    for (new_k, &(sigma, eig_k)) in pairs[..chi].iter().enumerate() {
        let vk = eig.eigenvectors.column(eig_k);
        // vt row = vᴴ
        for c in 0..cols {
            vt_kept[(new_k, c)] = vk[c].conj();
        }
        // u column = M·v / σ (a unit, mutually-orthonormal vector for σ > 0;
        // left as zero for a numerically-zero σ, which carries no weight).
        let mvk = m * vk;
        let inv = if sigma > eps { 1.0 / sigma } else { 0.0 };
        for r in 0..rows {
            u_kept[(r, new_k)] = mvk[r] * Complex::new(inv, 0.0);
        }
        s_kept[new_k] = sigma * scale;
    }
    (u_kept, s_kept, vt_kept, discarded)
}

/// Thin QR decomposition: returns `(Q, R)` with `Q` of shape `rows × k` and
/// `R` of shape `k × cols`, where `k = min(rows, cols)`.
///
/// Used to move the orthogonality center one site at a time without introducing
/// truncation (no singular-value threshold applied here; that comes in SVD-based
/// truncation in Task 5+).
pub fn thin_qr(m: &DMatrix<Complex>) -> (DMatrix<Complex>, DMatrix<Complex>) {
    let qr = m.clone().qr();
    let q_full = qr.q(); // rows × rows unitary
    let r_full = qr.r(); // rows × cols upper-triangular
    let k = m.nrows().min(m.ncols());
    let q = q_full.columns(0, k).into_owned(); // rows × k
    let r = r_full.rows(0, k).into_owned(); // k × cols
    (q, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Complex;

    fn c(re: f64) -> Complex {
        Complex::new(re, 0.0)
    }

    #[test]
    fn ket0_site_shape() {
        let s = Site::ket0();
        assert_eq!((s.left, s.right), (1, 1));
        assert_eq!(s.get(0, 0, 0), c(1.0));
        assert_eq!(s.get(0, 1, 0), c(0.0));
    }

    #[test]
    fn group_left_roundtrip() {
        let mut s = Site::zeros(2, 3);
        for l in 0..2 {
            for p in 0..2 {
                for r in 0..3 {
                    *s.get_mut(l, p, r) = c((l * 100 + p * 10 + r) as f64);
                }
            }
        }
        let m = s.to_group_left(); // (left*2) rows × right cols
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.ncols(), 3);
        let back = Site::from_group_left(&m, 2, 3);
        assert_eq!(back, s);
    }

    #[test]
    fn group_right_roundtrip() {
        let mut s = Site::zeros(2, 3);
        for l in 0..2 {
            for p in 0..2 {
                for r in 0..3 {
                    *s.get_mut(l, p, r) = c((l * 100 + p * 10 + r) as f64);
                }
            }
        }
        let m = s.to_group_right(); // left rows × (2*right) cols
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 6);
        let back = Site::from_group_right(&m, 2, 3);
        assert_eq!(back, s);
    }
}
