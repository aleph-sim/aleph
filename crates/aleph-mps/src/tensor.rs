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

/// How `truncated_svd` chooses how many singular values to keep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationPolicy {
    /// Keep at most `χ` singular values (the largest).
    FixedBond(usize),
    /// Keep the fewest singular values whose discarded squared weight is `≤ ε`,
    /// never exceeding `max_bond`.
    ErrorBounded { epsilon: f64, max_bond: usize },
}

/// SVD of `m` truncated according to `policy` (fixed-χ or error-bounded), renormalized to
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
    policy: &TruncationPolicy,
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

    // Drop singular values that are numerically zero relative to the largest.
    // The Gram matrix `MᴴM` SQUARES the condition number, so the resolvable
    // floor for a singular value is ~√(machine-ε)·σ_max ≈ 1e-8·σ_max — null
    // directions surface as spurious σ at that level. A floor of `1e-7·σ_max`
    // prunes them (keeping true Schmidt values, which are far larger) so a
    // rank-r block collapses to bond r instead of inflating toward `max_bond`
    // with noise. (A finer error-bounded threshold is P3-05.)
    let s_max = pairs.first().map(|p| p.0).unwrap_or(0.0);
    let eps = 1e-7 * s_max.max(f64::MIN_POSITIVE);
    let significant = pairs.iter().filter(|p| p.0 > eps).count().max(1);
    // Suffix sums of σ²: suffix_sq[k] = Σ_{j≥k} σ_j² (non-increasing in k).
    let mut suffix_sq = vec![0.0_f64; pairs.len() + 1];
    for k in (0..pairs.len()).rev() {
        suffix_sq[k] = suffix_sq[k + 1] + pairs[k].0 * pairs[k].0;
    }
    let chi = match *policy {
        TruncationPolicy::FixedBond(max_bond) => significant.min(max_bond.max(1)),
        TruncationPolicy::ErrorBounded { epsilon, max_bond } => {
            let cap = significant.min(max_bond.max(1));
            // Smallest keep ∈ [1, cap] with discarded tail Σ_{j≥keep} σ_j² ≤ ε.
            // `keep` is used both as an index into `suffix_sq` and as the returned
            // count — the index and the value are coupled, so a range loop is the
            // clearest expression here.
            #[allow(clippy::needless_range_loop)]
            let mut chosen = cap;
            #[allow(clippy::needless_range_loop)]
            for keep in 1..=cap {
                if suffix_sq[keep] <= epsilon {
                    chosen = keep;
                    break;
                }
            }
            chosen
        }
    };
    let discarded: f64 = suffix_sq[chi];
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

    #[test]
    fn truncated_svd_reconstructs_complex_full_rank() {
        // Generic complex 4×4: U·diag(s)·Vt must reconstruct M (renorm scale=1
        // only if ‖M‖=1; here ‖M‖≠1, so compare to scale·M via re-deriving).
        let m = DMatrix::from_fn(4, 4, |i, j| {
            Complex::new(
                (i as f64 - j as f64) * 0.3 + 1.0,
                (i * 2 + j) as f64 * 0.17 - 0.5,
            )
        });
        let fro: f64 = m.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let (u, s, vt, _disc) = truncated_svd(&m, &TruncationPolicy::FixedBond(64));
        // reconstruction = U·diag(s)·Vt = (1/fro)·M  (renormalized to unit weight)
        let mut maxd = 0.0_f64;
        for r in 0..4 {
            for col in 0..4 {
                let mut acc = Complex::new(0.0, 0.0);
                for k in 0..s.len() {
                    acc += u[(r, k)] * Complex::new(s[k], 0.0) * vt[(k, col)];
                }
                maxd = maxd.max((acc - m[(r, col)] / Complex::new(fro, 0.0)).norm());
            }
        }
        assert!(maxd < 1e-10, "complex SVD reconstruction err {maxd:e}");
    }

    #[test]
    fn truncated_svd_rank1_complex_collapses_to_chi1() {
        // Rank-1 complex matrix (outer product a·bᴴ) must yield χ=1, not an
        // inflated bond padded with Gram-noise singular values.
        let a = [
            Complex::new(0.5, 0.3),
            Complex::new(-0.2, 0.7),
            Complex::new(0.1, -0.4),
            Complex::new(0.6, 0.0),
        ];
        let b = [
            Complex::new(0.4, -0.1),
            Complex::new(0.2, 0.5),
            Complex::new(-0.3, 0.2),
            Complex::new(0.1, 0.1),
        ];
        let m = DMatrix::from_fn(4, 4, |i, j| a[i] * b[j].conj());
        let (u, s, vt, _disc) = truncated_svd(&m, &TruncationPolicy::FixedBond(64));
        assert_eq!(
            s.len(),
            1,
            "rank-1 block must collapse to χ=1, got χ={}",
            s.len()
        );
        // And it must still reconstruct (1/‖M‖)·M.
        let fro: f64 = m.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let mut maxd = 0.0_f64;
        for r in 0..4 {
            for col in 0..4 {
                let acc = u[(r, 0)] * Complex::new(s[0], 0.0) * vt[(0, col)];
                maxd = maxd.max((acc - m[(r, col)] / Complex::new(fro, 0.0)).norm());
            }
        }
        assert!(maxd < 1e-10, "rank-1 reconstruction err {maxd:e}");
    }

    fn diag_sigma() -> DMatrix<Complex> {
        let s = [1.0, 0.1, 0.01, 0.001];
        DMatrix::from_fn(4, 4, |i, j| {
            if i == j {
                Complex::new(s[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
    }

    #[test]
    fn error_bounded_keeps_minimal_chi() {
        let m = diag_sigma();
        let (_, s, _, disc) = truncated_svd(
            &m,
            &TruncationPolicy::ErrorBounded {
                epsilon: 1e-3,
                max_bond: 64,
            },
        );
        assert_eq!(s.len(), 2, "expected χ=2");
        assert!(disc <= 1e-3 + 1e-15, "discarded {disc} exceeds ε");
    }

    #[test]
    fn error_bounded_tiny_eps_keeps_all() {
        let m = diag_sigma();
        let (_, s, _, disc) = truncated_svd(
            &m,
            &TruncationPolicy::ErrorBounded {
                epsilon: 0.0,
                max_bond: 64,
            },
        );
        assert_eq!(s.len(), 4, "ε=0 must keep full rank");
        assert!(disc < 1e-12);
    }

    #[test]
    fn error_bounded_cap_overrides_eps() {
        let m = diag_sigma();
        let (_, s, _, _) = truncated_svd(
            &m,
            &TruncationPolicy::ErrorBounded {
                epsilon: 10.0,
                max_bond: 1,
            },
        );
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn fixed_bond_matches_legacy() {
        let m = diag_sigma();
        let (_, s, _, _) = truncated_svd(&m, &TruncationPolicy::FixedBond(2));
        assert_eq!(s.len(), 2);
    }
}
